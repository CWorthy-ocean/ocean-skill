"""Run a suite: ``ocean-skill-run suite.yaml`` (see ``docs/suites.md`` for the grammar).

A suite (``ocean_skill.config.SuiteConfig``) is a YAML file listing pages -- each
one a single ``osk.field``, ``osk.compare``, or ``osk.summary`` call (a ``field:``
page may chain a few more methods after it -- see ``then:`` in ``docs/suites.md``), an
``osk.XY``/``osk.TS`` property-property plot (``XY:``/``TS:``), or a ``section:``
divider page that is only text -- plus shared defaults and output settings. Running it
draws every page, writes a PNG per figure, collects them into one PDF (unless
``pdf: false``), and writes a metrics CSV and a ``manifest.json`` recording exactly
what was drawn. It also writes ``run.log`` -- everything printed to the terminal over
the course of the run, plus full tracebacks for skipped pages and for a fatal crash,
which the terminal itself never shows. See ``docs/suites.md``.

By default the report goes in one stable directory, ``<output_dir>/<name>/``, refreshed
in place each run: only the previous run's own artifacts (:data:`_REPORT_ARTIFACTS`) are
cleared first, and anything else a user keeps there is left alone. ``timestamped_dir:
true`` writes a new directory per run instead (see :func:`_report_dir_name`).

``report.pdf``'s first page is a cover describing the simulation
(:mod:`ocean_skill.suites.cover`), also recorded as the manifest's ``run_info`` with
copies of the C-Star blueprint and workplan under ``run_info/``. Like a ``section:``
page it is not a figure: PDF-only, unnumbered, and dropped if nothing draws.

A ``section:`` page is not a figure and so is not counted as one: it appears only in
``report.pdf`` (and only if some page after it drew -- see
:meth:`~ocean_skill.suites.report.PdfReport.section`), and
:attr:`SuiteResult.exit_code` and ``main``'s "N page(s) drawn" count both ignore it.

A suite whose ``refresh:`` block rebuilds a live run's kerchunk reference gets a copy
of that reference under the report directory's own ``refs/``, so the report stays
tied to the exact data it was drawn from even after the shared reference is
overwritten by a later run.

Guardrails against finding a bug only after hours of compute (see "Guardrails" in
``docs/suites.md``): ``smoke=N`` ("test mode") reads every model source of the
``refresh:`` block with only its first and last N steps, on every ``smoke_stride``-th
grid cell, with each page's date list cut to its first and last entry and figures drawn
at a lower dpi (own output directory and results cache),
``self_check`` redraws every ``compare:`` page against itself and flags any difference,
``only=GLOB`` draws just the matching pages or whole ``section:`` parts (each in its own
report folder), and ``strict`` counts a layout warning as a problem. Whatever goes wrong
-- a skipped or failed page, a self-check mismatch, a strict warning -- is collected in
:attr:`SuiteResult.problems`, listed at the end of the
run with its time and peak memory, and makes :attr:`SuiteResult.exit_code` nonzero.

Optionally the suite can refresh a model's kerchunk reference before running, which
is what makes it usable against a run that is still writing output. The same block's
``forcing:`` list catalogs the run's roms-tools forcing files (boundary and surface)
into that catalog too, so forcing pages need no separate catalog build.
"""

from __future__ import annotations

import argparse
import contextlib
import fnmatch
import io
import json
import os
import re
import shutil
import sys
import threading
import time
import traceback
import warnings
from dataclasses import asdict, dataclass
from dataclasses import field as _dc_field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

__all__ = ["PageProblem", "SuiteResult", "main", "run_suite"]

#: What a run writes into its report directory, so what a fixed-mode run clears first
#: (see :func:`_clear_previous_report`). Nothing else in the directory is ever touched.
_REPORT_ARTIFACTS = (
    "report.pdf",
    "figures",
    "metrics",
    "refs",
    "run_info",
    "manifest.json",
    "run.log",
    "suite.yaml",
)


class _Tee:
    """A stream that mirrors every write to the original stream and to a file.

    Before :meth:`attach` is called there is no file yet -- writes are buffered
    in memory -- because a suite's ``refresh:`` step can print before the report
    directory (and so ``run.log``'s path) exists. ``attach`` flushes that buffer
    into the file and every write after that goes straight through.
    """

    def __init__(self, original: TextIO) -> None:
        self.original = original
        self._buffer = io.StringIO()
        self._file: TextIO | None = None

    def write(self, s: str) -> int:
        n = self.original.write(s)
        (self._file or self._buffer).write(s)
        if self._file is not None:
            self._file.flush()
        return n

    def flush(self) -> None:
        self.original.flush()
        if self._file is not None:
            self._file.flush()

    def isatty(self) -> bool:
        return self.original.isatty()

    def fileno(self) -> int:
        return self.original.fileno()

    @property
    def encoding(self) -> str:
        return self.original.encoding

    def attach(self, path: Path) -> None:
        self._file = path.open("a", encoding="utf-8")
        self._file.write(self._buffer.getvalue())
        self._file.flush()
        self._buffer = io.StringIO()

    def log_only(self, s: str) -> None:
        """Write to the log file (or the pre-attach buffer) but not the terminal."""
        (self._file or self._buffer).write(s)
        if self._file is not None:
            self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


class _Capture:
    """Handle yielded by :func:`_capture_terminal`: ``attach`` and ``log_only``."""

    def __init__(self, out: _Tee, err: _Tee) -> None:
        self._out = out
        self._err = err

    def attach(self, path: Path) -> None:
        self._out.attach(path)
        self._err.attach(path)

    def log_only(self, s: str) -> None:
        self._err.log_only(s)


@contextlib.contextmanager
def _capture_terminal():
    """Mirror stdout/stderr (and therefore ``warnings.warn``) to a file.

    The file is only attached partway through the run, once the report
    directory exists (see :class:`_Tee`); anything printed before that is
    buffered and flushed in once ``attach`` is called. A fatal exception still
    gets its traceback written to the log before propagating, even though the
    terminal itself only ever sees ``main``'s one-line ``error: ...``.
    """
    out, err = _Tee(sys.stdout), _Tee(sys.stderr)
    sys.stdout, sys.stderr = out, err
    cap = _Capture(out, err)
    try:
        yield cap
    except BaseException:
        cap.log_only(traceback.format_exc())
        raise
    finally:
        sys.stdout, sys.stderr = out.original, err.original
        out.close()
        err.close()


def _reopen_catalog(catalog_path: Path):
    """Reopen ``catalog_path`` to update in place, or start a fresh catalog."""
    import intake

    from ocean_skill.build import new_catalog

    if catalog_path.exists():
        try:
            return intake.from_yaml_file(str(catalog_path))
        except Exception as exc:
            print(
                f"  refresh: could not reopen existing {catalog_path} ({exc}); "
                "starting a fresh catalog"
            )
    return new_catalog(title="refreshed by ocean_skill.suites.run")


def _refresh_sources(spec: list[dict[str, Any]], catalog_path: str | Path) -> None:
    """Rebuild kerchunk references and their catalog entries from the current files.

    Reopens the existing catalog (if any) and updates it in place, rather than
    building a fresh one from scratch: a run between output steps — or one that was
    restarted and has temporarily produced no files — must not wipe out every other
    entry the catalog already had. A source whose glob matches nothing simply keeps
    whatever entry it already had; only sources that actually rebuilt get touched.

    An entry's ``subchunk`` key is forwarded to ``make_kerchunk`` as it is. Its
    ``keep`` key, when given, is forwarded to
    :func:`ocean_skill.build.make_kerchunk` — a restart stream that is still being
    refreshed against a live run declares ``keep: latest-per-file`` to drop each
    file's earlier, superseded record. Left unset, ``make_kerchunk``'s own default
    (``"last"``) applies, so an ordinary stream's restart-boundary overlaps are
    collapsed on every refresh without the suite having to say so -- newer segment
    wins, with a loud warning (not a raise) if the overlap actually disagrees.

    Files that match the glob but look unfinished (still being written, or never
    completed -- see :func:`ocean_skill.build._partition_unfinished`) are filtered out
    here, before the rebuild: a stream whose only matched file is the one currently
    being written is treated the same as a stream that matched nothing, keeping its
    existing entry rather than raising, since that is an ordinary state for a suite
    refreshed against a live run rather than a failure. An entry's ``min_age`` key,
    when given, is forwarded to both the filter and ``make_kerchunk``.

    ``files`` is one glob, or a list of globs for a run whose output is spread over
    several task folders (``tasks/first_wk/output/output_rst.*.nc``,
    ``tasks/second_wk/...``, then a second run's ``tasks/*/output/...``). A list is
    joined into one reference in *time* order, not path order (a folder named
    ``first_half`` sorts before ``first_wk``), by
    :func:`ocean_skill.build.order_segments`. Where two files give the same time step
    -- after ``keep: latest-per-file`` has cut each file to its newest record -- the
    later one wins: a file of a later glob beats one of an earlier glob, and within one
    glob the later path in sorted order beats the earlier. Each file dropped that way
    is logged (``refresh: <name>: <time> from <path> replaces <path>``). A single glob
    given as a string is read as it always was: every match in path order, repeats
    collapsed by ``make_kerchunk``'s own ``keep`` rule.
    """
    import glob as _glob

    from ocean_skill.build import (
        _partition_unfinished,
        add_source,
        make_kerchunk,
        order_segments,
        save,
    )

    catalog_path = Path(catalog_path).expanduser()
    cat = _reopen_catalog(catalog_path)

    rebuilt, skipped = [], []
    for entry in spec:
        # glob.glob (not Path.glob) so absolute patterns work
        patterns = entry["files"]
        joined = not isinstance(patterns, str)
        groups = [
            sorted(_glob.glob(str(Path(pattern).expanduser())))
            for pattern in (patterns if joined else [patterns])
        ]
        files = [f for group in groups for f in group]
        if not files:
            skipped.append(entry["name"])
            continue
        min_age = entry.get("min_age", 0.0)
        ready, unfinished = _partition_unfinished(
            [Path(f) for f in dict.fromkeys(files)], min_age=min_age
        )
        for p, why in unfinished.items():
            print(f"  refresh: {entry['name']}: skipping {p} ({why})")
        if not ready:
            skipped.append(entry["name"])
            continue
        replaced = []
        if joined:
            ready_set = set(ready)
            ready, replaced = order_segments(
                [[Path(f) for f in group if Path(f) in ready_set] for group in groups],
                keep=entry.get("keep", "last"),
            )
            for when, winner, loser in replaced:
                print(
                    f"  refresh: {entry['name']}: {when} from {winner} "
                    f"replaces {loser}"
                )
        # Each left out entirely when the suite doesn't name it, so make_kerchunk's
        # own defaults (keep, automatic subchunking) take effect here too.
        kwargs = {k: entry[k] for k in ("keep", "subchunk") if k in entry}
        ref = make_kerchunk(
            ready, entry["ref"], grid=entry.get("grid"), min_age=min_age, **kwargs
        )
        # ``label:`` is how figures name this source (entry metadata, read by
        # ``ocean_skill.catalog.source_label``); left out when the suite has none
        meta = {"label": entry["label"]} if entry.get("label") else {}
        add_source(cat, entry["name"], ref, **meta)
        rebuilt.append(entry["name"])
        note = f" ({len(replaced)} replaced by a later segment)" if replaced else ""
        print(f"  refresh: {entry['name']} <- {len(ready)} files{note}")

    if skipped:
        print(f"  refresh: no files matched for {skipped}; kept their existing entries")
    if not rebuilt and not skipped:
        return  # nothing declared to refresh; leave the file untouched
    save(cat, catalog_path)


#: The ``refresh.forcing`` keys each ``kind`` forwards to its builder (the config
#: model's own defaults, ``climatology: false``, must not reach a builder that has no
#: such key).
_FORCING_KEYS = {
    "boundary": (
        "variables", "climatology", "sides", "strip", "ref", "subchunk", "label"
    ),
    "surface": (
        "variables", "climatology", "time_coord", "ref", "subchunk", "label"
    ),
    "initial_conditions": ("variables", "label"),
    "river": ("ref", "label"),
}


def _refresh_forcing(spec: list[dict[str, Any]], catalog_path: str | Path) -> None:
    """Catalog a suite's roms-tools forcing files into the ``refresh:`` catalog.

    Each ``refresh.forcing`` entry becomes the catalog entries
    :func:`ocean_skill.build.add_boundary_sources` (``kind: boundary``, one per side,
    ``<name>_<side>``, plus ``<name>_strip`` when ``strip:`` names sides),
    :func:`ocean_skill.build.add_forcing_source` (``kind: surface``) or
    :func:`ocean_skill.build.add_initial_conditions` (``kind: initial_conditions``, one
    entry from one file) or :func:`ocean_skill.build.add_river_source` (``kind:
    river``, one entry) write, alongside the model output ``refresh.sources``
    rebuilt. The
    entries are rebuilt on every run. An entry with a ``ref`` reads a kerchunk
    reference kept there, which the builder rebuilds only when the matched files
    change, so a run with unchanged forcing reuses it and opens nothing else; an entry
    without one reads the files themselves. An entry's ``subchunk`` key (only with a
    ``ref``) is forwarded to the reference build, to join files whose time chunks
    differ in length.

    ``files`` is a glob or a list of globs, each expanded and sorted, then joined in
    the order given (so one run's forcing spread over several folders reads as one
    time series). An entry whose globs match nothing keeps whatever entries it had, as
    for ``refresh.sources``. An entry that matches files but cannot be built is
    reported (``FAILED``, naming it and the error) and left as it was, and the rest
    still build: a page reading an entry that was never built is skipped like any page
    whose source is missing, so one bad forcing set costs only its own pages.
    """
    import glob as _glob

    from ocean_skill.build import (
        _local_path,
        add_boundary_sources,
        add_forcing_source,
        add_initial_conditions,
        add_river_source,
        save,
    )

    adders = {
        "boundary": add_boundary_sources,
        "surface": add_forcing_source,
        "initial_conditions": add_initial_conditions,
        "river": add_river_source,
    }

    catalog_path = Path(catalog_path).expanduser()
    cat = _reopen_catalog(catalog_path)

    matched = {}
    for entry in spec:
        globs = [entry["files"]] if isinstance(entry["files"], str) else entry["files"]
        matched[entry["name"]] = [
            f
            for pattern in globs
            for f in sorted(_glob.glob(str(Path(pattern).expanduser())))
        ]
    # Entries may share a ref only over the very same files: otherwise each would
    # rebuild it over its own on every run, and the first one's catalog entry would
    # read the last one's data.
    by_ref: dict[Path, list[str]] = {}
    for entry in spec:
        if "ref" in entry and matched[entry["name"]]:
            by_ref.setdefault(_local_path(entry["ref"]), []).append(entry["name"])
    conflicted = {
        name: names
        for names in by_ref.values()
        if len({tuple(matched[n]) for n in names}) > 1
        for name in names
    }

    rebuilt, skipped, failed = [], [], []
    for entry in spec:
        files = matched[entry["name"]]
        if entry["name"] in conflicted:
            failed.append(entry["name"])
            print(
                f"  refresh: FAILED {entry['name']}: its ref {entry['ref']} is shared "
                f"with {conflicted[entry['name']]} over different files; give each "
                "file set its own ref"
            )
            continue
        if not files:
            skipped.append(entry["name"])
            continue
        kwargs = {k: entry[k] for k in _FORCING_KEYS[entry["kind"]] if k in entry}
        if "hc" in entry:  # a grid (and data) file that states no critical depth
            kwargs["vertical"] = {"hc": float(entry["hc"])}
        add = adders[entry["kind"]]
        # the builder writes this record of the files next to a reference it
        # (re)builds, so an unchanged one means the reference was reused
        sidecar = (
            Path(str(_local_path(entry["ref"])) + ".sources.json")
            if "ref" in entry
            else None
        )
        before = _mtime(sidecar)
        try:
            added = add(cat, entry["name"], files, entry["grid"], **kwargs)
        except Exception as exc:
            failed.append(entry["name"])
            print(
                f"  refresh: FAILED {entry['name']} ({len(files)} file(s) starting "
                f"{files[0]}): {type(exc).__name__}: {exc}"
            )
            continue
        added = [added] if isinstance(added, str) else added
        rebuilt.append(entry["name"])
        how = ""
        if sidecar is not None:
            how = " (reference reused)" if _mtime(sidecar) == before else (
                " (reference rebuilt)"
            )
        print(
            f"  refresh: {entry['name']} <- {len(files)} files{how} -> "
            f"{', '.join(added)}"
        )

    if skipped:
        print(f"  refresh: no forcing files matched for {skipped}; kept their entries")
    if failed:
        print(f"  refresh: could not build {failed}; their pages will be skipped")
    if rebuilt:
        save(cat, catalog_path)


def _mtime(path: Path | None) -> int | None:
    """``path``'s modification time in ns, or None when there is no such file."""
    try:
        return path.stat().st_mtime_ns if path is not None else None
    except FileNotFoundError:
        return None


def _snapshot_refs(refresh: Any, report_dir: Path) -> list[dict[str, Any]]:
    """Copy each ``refresh:`` source's kerchunk reference into ``report_dir/refs/``.

    The shared reference at ``source.ref`` is rewritten in place by every future
    refresh (:func:`_refresh_sources`), so without this a report directory stops
    matching the data it was drawn from the moment the model output changes again.
    Copied rather than linked -- a parquet reference is a directory that
    ``to_kerchunk`` overwrites wholesale, so a symlink or hard link would not
    survive the next refresh.

    The copy pins file paths and byte ranges, not file contents: a restart that
    rewrites an output file in place still changes what the snapshot reads. A
    second suite run started concurrently, between this run's own refresh and this
    copy, could in principle overwrite the shared reference first -- out of scope
    here.

    Returns one record per source: ``{"name", "ref": absolute path str, "snapshot":
    absolute path str or None if no reference exists yet to copy}``.
    """
    records = []
    for source in refresh.sources:
        src = Path(source.ref).expanduser().resolve()
        record: dict[str, Any] = {
            "name": source.name,
            "ref": str(src),
            "snapshot": None,
        }
        if not src.exists():
            print(f"  refresh: {source.name}: no reference at {src} to snapshot")
            records.append(record)
            continue
        dst = report_dir / "refs" / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        record["snapshot"] = str(dst)
        records.append(record)
    return records


@dataclass
class PageProblem:
    """One thing a run found wrong with a page, for the end-of-run list.

    ``kind`` is ``"skipped"`` (the page, or part of it, was not drawn for a reason the
    code knew: its source could not be read, its data was all NaN), ``"failed"`` (the
    draw raised), ``"self-check"`` (``run_suite(self_check=True)`` found the two
    identical lanes disagreeing) or ``"strict"`` (a layout warning under ``strict``).
    ``seconds`` and ``peak_mb`` are what drawing the page took and the process's peak
    memory while it did (``None`` if it was never tried).
    """

    title: str
    kind: str
    reason: str
    seconds: float | None = None
    peak_mb: float | None = None


@dataclass
class SuiteResult:
    """What :func:`run_suite` produced."""

    pages: list[Any] = _dc_field(default_factory=list)
    report_dir: Path | None = None
    pdf: Path | None = None
    figures: list[Path] = _dc_field(default_factory=list)
    metrics: Path | None = None
    manifest: Path | None = None
    log: Path | None = None
    refs: list[Path] = _dc_field(default_factory=list)
    #: Everything wrong with the pages that were attempted (see :class:`PageProblem`).
    problems: list[PageProblem] = _dc_field(default_factory=list)

    @property
    def exit_code(self) -> int:
        """Return the process exit code.

        ``0`` clean, ``1`` no page drew, ``3`` pages skipped or flagged, ``4`` a page
        failed or a self-check found a lane bug (so not a benign skip).
        """
        # A section: divider is not a figure: whether it was written to the PDF says
        # nothing about whether the report drew, so it takes no part in the code.
        statuses = {p.status for p in self.pages if p.kind != "section"}
        if "ok" not in statuses:
            return 1
        if any(p.kind in ("failed", "self-check") for p in self.problems):
            return 4
        if statuses == {"ok"} and not self.problems:
            return 0
        # some page skipped, or every page drew but something is still wrong with one
        return 3


#: Warnings that mean a page drew but badly (``strict`` turns them into problems): the
#: layout ones. Ordinary data warnings -- thin samples, partial bins -- are not here.
_STRICT_PATTERNS = (
    r".*the maps are only [\d.]+in wide",
    r".*on a canvas capped at [\d.]+in leaves .* less than the labeling needs",
    r".*the maps may be squeezed",
    r".*[Tt]ight layout not applied",
    r".*constrained_layout not applied",
)


@contextlib.contextmanager
def _watch_warnings(patterns):
    """Collect the text of every warning matching ``patterns`` raised in the block.

    The warning still goes wherever it was going (the terminal and ``run.log``); this
    only listens. A matching warning is made to fire every time (``"always"``), since
    Python otherwise reports one location once and a second page's identical problem
    would never be seen.
    """
    hits: list[str] = []
    compiled = [re.compile(p, re.DOTALL) for p in patterns]
    with warnings.catch_warnings():
        for p in patterns:
            warnings.filterwarnings("always", message=p)
        previous = warnings.showwarning

        def show(message, category, filename, lineno, file=None, line=None):
            text = str(message)
            if any(c.match(text) for c in compiled):
                hits.append(text)
            previous(message, category, filename, lineno, file, line)

        warnings.showwarning = show
        yield hits


def _rss_mb() -> float | None:
    """Return this process's resident memory in MB, or ``None`` without psutil.

    ``resource.ru_maxrss`` would do for a whole run but is a process-lifetime peak, so
    it says nothing about one page; better no number than a misleading one.
    """
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        return None


class _PeakMemory:
    """Context manager recording the peak resident memory (MB) reached in its block.

    A background thread samples the process every 0.1 s, so a short spike between
    samples can be missed; it is for spotting the page that needs gigabytes, not for
    accounting. ``peak_mb`` stays ``None`` when memory cannot be read (no psutil).
    """

    interval = 0.1

    def __enter__(self) -> _PeakMemory:
        self.peak_mb = _rss_mb()
        self._thread = None
        if self.peak_mb is None:
            return self
        self._stop = threading.Event()

        def sample() -> None:
            while not self._stop.wait(self.interval):
                now = _rss_mb()
                if now is not None:
                    self.peak_mb = max(self.peak_mb, now)

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join()
        now = _rss_mb()
        if now is not None:
            self.peak_mb = max(self.peak_mb, now)


def _only_globs(only: str | list[str] | None) -> list[str]:
    """Return ``only`` as a list of globs: ``[]`` for ``None``, ``[only]`` for a str."""
    if only is None:
        return []
    return [only] if isinstance(only, str) else list(only)


def _only_tag(globs: list[str]) -> str:
    """Return the report-folder tag of an ``only`` run: ``-only-<slug of the globs>``.

    The globs are joined with ``-``, lowercased, and every run of characters other than
    letters and digits becomes one ``-``: ``"River inputs"`` -> ``-only-river-inputs``.
    Only one glob of plain words (letters and digits, single spaces) is spelled out
    alone; anything else -- several globs, a wildcard or other punctuation, a slug
    past 40 characters (cut to 33) -- also gets the first six hex characters of the
    sha1 of the lowercased globs, and so does a glob with no letters or digits at all
    (its slug would be empty). Two different selections then never share a folder,
    where the second run would clear the first one's report; case aside, since
    matching ignores it and the selection is the same.
    """
    import hashlib

    joined = "-".join(g.lower() for g in globs)
    # hashed one glob per line, so ["a-b"] and ["a", "b"] differ
    digest = hashlib.sha1("\n".join(g.lower() for g in globs).encode()).hexdigest()[:6]
    slug = re.sub(r"[^a-z0-9]+", "-", joined).strip("-")
    lossless = len(globs) == 1 and re.fullmatch(r"[a-z0-9]+( [a-z0-9]+)*", joined)
    if len(slug) > 40:
        slug, lossless = slug[:33].rstrip("-"), False
    if not slug:
        return f"-only-{digest}"
    return f"-only-{slug}" if lossless else f"-only-{slug}-{digest}"


def _select_pages(expanded: list[Any], globs: str | list[str]) -> list[Any]:
    """Keep the parts of the suite that ``globs`` name (case-insensitive fnmatch).

    A glob matching a ``section:`` divider's title keeps that whole section: the divider
    and every page after it up to the next divider. A glob matching a figure page's
    title keeps that page, and the divider over it stays only because of it, so the
    report has no heading over nothing. A page is kept if any glob keeps it, and a glob
    that matches nothing is an error.
    """
    given = _only_globs(globs)
    patterns = [g.lower() for g in given]
    matched: set[int] = set()
    selected: list[Any] = []
    pending = None
    in_kept_section = False
    for page in expanded:
        got = {
            i
            for i, g in enumerate(patterns)
            if fnmatch.fnmatchcase(page.title.lower(), g)
        }
        matched |= got
        if page.kind == "section":
            in_kept_section = bool(got)
            pending = None if got else page
            if got:
                selected.append(page)
        elif in_kept_section or got:
            if pending is not None:
                selected.append(pending)
                pending = None
            selected.append(page)
    unmatched = [g for i, g in enumerate(given) if i not in matched]
    if unmatched:
        sections = [p.title for p in expanded if p.kind == "section"]
        titles = [p.title for p in expanded if p.kind != "section"]
        raise ValueError(
            f"--only {unmatched[0]!r} matches no section or page; the sections are: "
            + "; ".join(repr(t) for t in sections[:12])
            + (" ..." if len(sections) > 12 else "")
            + "; the pages are: "
            + "; ".join(repr(t) for t in titles[:12])
            + (" ..." if len(titles) > 12 else "")
        )
    return selected


def _short(text: str, limit: int = 200) -> str:
    """Return ``text`` on one line, cut to ``limit`` characters."""
    one = " ".join(str(text).split())
    return one if len(one) <= limit else one[: limit - 3] + "..."


def _problem_lines(problems: list[PageProblem]) -> list[str]:
    """Format the end-of-run problem list, one line per problem."""
    if not problems:
        return []
    lines = [f"{len(problems)} problem(s):"]
    for p in problems:
        cost = []
        if p.seconds is not None:
            cost.append(f"{p.seconds:.1f}s")
        if p.peak_mb is not None:
            cost.append(f"peak {p.peak_mb:,.0f} MB")
        tail = f" ({', '.join(cost)})" if cost else ""
        lines.append(f"  [{p.kind}] {p.title}: {_short(p.reason)}{tail}")
    return lines


def _report_dir_name(name: str, test_source: str, index: Any) -> str:
    import uuid

    from ocean_skill import outputs

    t0 = str(index[0].date()) if hasattr(index[0], "date") else str(index[0])
    t1 = str(index[-1].date()) if hasattr(index[-1], "date") else str(index[-1])
    now = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    # A short random suffix guarantees two runs invoked within the same second --
    # a test, or a scheduler re-firing quickly -- still land in different
    # directories; the timestamp above stays the human-readable part of the name.
    suffix = uuid.uuid4().hex[:6]
    return outputs.slug(f"{name}_{test_source}_{t0}_to_{t1}_{now}_{suffix}")


def _clear_previous_report(report_dir: Path) -> None:
    """Remove the previous run's own artifacts from ``report_dir``.

    Only the names in :data:`_REPORT_ARTIFACTS` go; ``report_dir`` itself and everything
    else in it is left alone.

    The suite's ``cache_dir`` may be this very directory, and a user may keep notes or
    extra exports beside the report, so this names what to remove rather than emptying
    the directory. It also keeps :func:`_snapshot_refs` from colliding with last run's
    ``refs/`` and stale PNGs of since-removed pages from lingering in ``figures/``.
    """
    for name in _REPORT_ARTIFACTS:
        target = report_dir / name
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        elif target.exists() or target.is_symlink():
            target.unlink()


def _write_manifest(
    path: Path,
    *,
    suite: Any,
    pages: list[Any],
    report_dir: Path,
    catalog_dirs: list[Path] | None = None,
    refresh: dict[str, Any] | None = None,
    pdf_images_applied: bool = False,
    run_info: dict[str, Any] | None = None,
    modes: dict[str, Any] | None = None,
    problems: list[PageProblem] | None = None,
) -> None:
    import ocean_skill
    from ocean_skill import cache as _cache

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "ocean_skill_version": getattr(ocean_skill, "__version__", None),
        "name": suite.name,
        "output_dir": str(report_dir),
        "catalog_search_paths": [str(d) for d in (catalog_dirs or [])],
        "cache_dir": str(_cache.base_dir()),
        "refresh": refresh,
        # smoke / self_check / only / strict as the run was invoked, ``None`` for an
        # ordinary full run
        "modes": modes,
        "problems": [asdict(p) for p in problems or []],
        # The cover page's content (see ocean_skill.suites.cover.RunInfo.as_dict).
        "run_info": run_info,
        # What the suite asked of report.pdf's rasters, and whether the Ghostscript
        # pass actually ran (False for "lossless", pdf: false, no PDF, or a fallback).
        "pdf_images": suite.pdf_images,
        "pdf_images_applied": pdf_images_applied,
        "pages": [
            {
                **p.as_dict(),
                "status": p.status,
                "reason": p.reason,
                "seconds": p.elapsed,
                "peak_mb": p.peak_mb,
                # Data-dependent (a then: step's own extremum(), say) and so left
                # out of as_dict()/the determinism check -- see ExpandedPage.
                "results": p.results,
            }
            for p in pages
        ],
    }
    path.write_text(json.dumps(payload, indent=2, default=str))


def _resolve_catalog_dirs(entries: list[str], suite_path: Path) -> list[Path]:
    """Resolve a suite's ``catalog_search_paths:`` entries to absolute directories, in
    YAML order.

    A relative entry resolves against ``suite_path``'s own directory -- not the
    working directory -- so the suite means the same thing from cron or from any
    launch directory. This deliberately differs from ``refresh:``/``output_dir``,
    which are working-directory-relative today.
    """
    resolved = []
    for entry in entries:
        d = Path(entry).expanduser()
        if not d.is_absolute():
            d = suite_path.parent / d
        resolved.append(d.resolve())
    return resolved


def _resolve_cache_dir(entry: str | None, suite_path: Path) -> Path | None:
    """Resolve a suite's ``cache_dir:`` entry to an absolute directory, or ``None``.

    Same rule as :func:`_resolve_catalog_dirs`: a relative entry resolves against
    ``suite_path``'s own directory, not the working directory, so the suite means
    the same thing run from cron, a notebook, or any shell.
    """
    if entry is None:
        return None
    d = Path(entry).expanduser()
    if not d.is_absolute():
        d = suite_path.parent / d
    return d.resolve()


def _self_check_pages(expanded: list[Any]) -> tuple[list[Any], set[int]]:
    """Swap every ``compare:`` page for its self-check copy; also return their ids."""
    from ocean_skill.suites import pages as _pages

    out, checked = [], set()
    for page in expanded:
        copy = _pages.self_check_page(page) if not page.problem else None
        if copy is None:
            if page.kind == "compare":
                print(f"self-check: {page.title!r} cannot be checked; drawn as written")
            out.append(page)
            continue
        out.append(copy)
        checked.add(id(copy))
    return out, checked


def _page_problems(
    page: Any, strict_hits: list[str], self_checked: bool
) -> list[tuple[str, str]]:
    """Return the ``(kind, reason)`` problems of a page that drew."""
    from ocean_skill.suites import pages as _pages

    found = []
    if page.dropped:
        found.append(
            (
                "skipped",
                f"{len(page.dropped)} comparison(s) left out: "
                + "; ".join(page.dropped[:3]),
            )
        )
    if self_checked:
        reason = _pages.self_check_problem(page)
        if reason:
            found.append(("self-check", reason))
    if strict_hits:
        more = f" (+{len(strict_hits) - 1} more)" if len(strict_hits) > 1 else ""
        found.append(("strict", strict_hits[0] + more))
    return found


#: Test mode's default horizontal stride: every 4th cell of each model source, ~1/16 of
#: the cells to read and draw (``run_suite(smoke_stride=...)``, ``--smoke-stride``).
SMOKE_STRIDE = 4

#: How many ``<base_dir>/smoke-cache/<hash>/`` folders a smoke run leaves behind (the
#: current one plus the newest others, by last use); older ones are deleted.
SMOKE_CACHE_KEEP = 3

#: A smoke-cache folder used within this long is never deleted by another run's
#: pruning, so two runs going at once cannot pull the folder from under each other.
SMOKE_CACHE_GRACE_S = 3600.0


def _code_identity() -> str | None:
    """Name the exact code a run executes, or ``None`` when that cannot be known.

    The installed package version, plus -- from the git checkout holding the package
    -- ``HEAD`` and, when the tree has uncommitted changes under the package, a hash
    of those changes (``git diff HEAD`` and the contents of untracked files), so an
    edit not yet committed also changes the identity. ``None`` (no git, or not a git
    checkout) makes the smoke cache always cold: nothing then vouches that the code
    is the one that wrote an earlier result.
    """
    import hashlib
    import subprocess
    from importlib import metadata

    import ocean_skill

    pkg = Path(ocean_skill.__file__).resolve().parent

    def git(*args: str) -> bytes:
        return subprocess.run(
            ["git", "-C", str(pkg), *args],
            check=True,
            capture_output=True,
            timeout=60,
        ).stdout

    try:
        version = metadata.version("ocean-skill")
    except metadata.PackageNotFoundError:
        version = "unknown"
    try:
        parts = [version, git("rev-parse", "HEAD").decode().strip()]
        if git("status", "--porcelain", "--", str(pkg)).strip():
            digest = hashlib.sha256(git("diff", "HEAD", "--", str(pkg)))
            untracked = git(
                "ls-files", "--others", "--exclude-standard", "-z", "--", str(pkg)
            )
            for rel in sorted(r for r in untracked.decode().split("\0") if r):
                digest.update(rel.encode())
                digest.update((pkg / rel).read_bytes())  # paths are relative to pkg
            parts.append("dirty-" + digest.hexdigest())
    except (OSError, subprocess.SubprocessError):
        return None
    return "|".join(parts)


def _smoke_expand(suite: Any, only: list[str]) -> list[Any] | None:
    """Expand the suite once, as test mode will, *before* the truncation is set.

    Month windows and other placeholders then resolve against the whole run (expansion
    itself reads the time index, so expanding from inside the truncation would see only
    the steps it is about to choose). The date lists are already thinned
    (``thin_dates``), so the steps pinned from these pages are only the ones the real
    expansion will ask for. Output and warnings of this throwaway pass are dropped --
    the real expansion repeats them. ``None`` for a suite that will not expand here.
    """
    import io

    from ocean_skill.suites import pages as _pages

    try:
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            expanded = _pages.expand(suite, thin_dates=True)
    except Exception as exc:  # noqa: BLE001 -- any failure only costs the pins
        warnings.warn(
            f"smoke: could not collect the time steps the pages name "
            f"({type(exc).__name__}: {exc}); keeping only the first and last steps",
            stacklevel=2,
        )
        return None
    return _select_pages(expanded, only) if only else expanded


def _smoke_pins(suite: Any, only: list[str]) -> Any:
    """Return the time steps the suite's pages name, for ``--smoke`` to keep.

    Taken from the thinned throwaway expansion of :func:`_smoke_expand`; a suite that
    will not expand keeps only its first and last steps.
    """
    from ocean_skill import sources as _sources
    from ocean_skill.suites import pages as _pages

    expanded = _smoke_expand(suite, only)
    return _sources.TimePins() if expanded is None else _pages.time_pins(expanded)


def _smoke_stamps(
    names: list[str], steps: int
) -> list[tuple[str, int, str, str, int, str]]:
    """Return ``(name, steps, first, last, kept, digest)`` for each truncated source.

    Called inside :func:`ocean_skill.sources.truncate_time`, so these describe the time
    stamps the run will actually see (``-`` and 0 for a source with no time axis):
    ``kept`` is how many, ``digest`` a hash of all of them, so a change to the steps
    pinned in between (not just the ends) starts a new cache.
    A source that cannot be read gets a stamp no other run shares, so its cache is cold.
    """
    import hashlib

    from ocean_skill import operators
    from ocean_skill import sources as _sources

    out = []
    for name in names:
        try:
            obj = _sources.read(name)
            dim = operators.resolve_dim(obj, "time") if hasattr(obj, "dims") else None
            index = obj.indexes.get(dim) if dim is not None else None
            if index is not None and len(index):
                first, last, kept = str(index[0]), str(index[-1]), len(index)
                digest = hashlib.sha256(
                    "\n".join(str(t) for t in index).encode()
                ).hexdigest()[:16]
            else:
                first, last, kept, digest = "-", "-", 0, "-"
        except Exception as exc:
            first = last = f"unreadable-{type(exc).__name__}-{time.time_ns()}"
            kept, digest = 0, "-"
        out.append((name, steps, first, last, kept, digest))
    return out


def _stat_tree(path: Path) -> list[tuple[str, int, int]]:
    """Return ``(name, mtime_ns, size)`` of ``path``, or of every file under a folder.

    Stat only -- nothing is opened. A parquet kerchunk reference is a folder, whose own
    mtime does not move when one of its files is rewritten, so each file counts. A
    path that does not exist gives ``[]``.
    """
    try:
        if not path.is_dir():
            st = path.stat()
            return [(path.name, st.st_mtime_ns, st.st_size)]
        out = []
        for root, _dirs, names in os.walk(path):
            for name in sorted(names):
                f = Path(root) / name
                st = f.stat()
                out.append((str(f.relative_to(path)), st.st_mtime_ns, st.st_size))
        return sorted(out)
    except OSError:
        return []


def _smoke_data_stamps(refresh: Any) -> list[Any]:
    """Return what the data behind a suite's ``refresh:`` block looks like on disk.

    The truncation stamps (:func:`_smoke_stamps`) name which time steps are kept, not
    what is in them: a model rerun or a rebuilt reference that rewrites the same steps
    in place would otherwise be served the earlier run's cached results. So the key
    also covers, for each ``refresh.sources`` and ``refresh.forcing`` entry, the
    ``(mtime_ns, size)`` of its kerchunk ``ref`` (a rebuild rewrites it) -- or, with no
    ref, of the files its globs match -- and of the refresh catalog file. Cheap stats
    only. ``[]`` for a suite with no ``refresh:`` block.
    """
    import glob as _glob

    if refresh is None:
        return []
    out: list[Any] = [
        ["catalog", _stat_tree(Path(refresh.catalog).expanduser().resolve())]
    ]
    for group, entries in (("source", refresh.sources), ("forcing", refresh.forcing)):
        for entry in entries:
            if entry.ref:
                stats = _stat_tree(Path(entry.ref).expanduser().resolve())
            else:
                patterns = entry.files
                if isinstance(patterns, str):
                    patterns = [patterns]
                stats = [
                    (f, *st[1:])
                    for pattern in patterns
                    for f in sorted(_glob.glob(str(Path(pattern).expanduser())))
                    for st in _stat_tree(Path(f))
                ]
            out.append([group, entry.name, stats])
    return out


def _smoke_cache_key(
    stamps: list[tuple[str, int, str, str, int, str]],
    data: list[Any] | None = None,
    stride: int = 1,
) -> tuple[str, str | None]:
    """Return the smoke-cache folder name and the code identity it was built from.

    A short hash of the truncation (``stamps``), the data on disk behind it (``data``,
    :func:`_smoke_data_stamps`), the horizontal ``stride`` (the stamps name only the
    time steps, and a result on every 4th cell is not one on every 2nd) and the code
    (:func:`_code_identity`). A stride of 1 adds nothing to the hash, so a run that does
    not stride finds the folder it always did. With no code identity the name carries a
    fresh nonce, so it is never found again.
    """
    import hashlib
    import uuid

    identity = _code_identity()
    nonce = None if identity else uuid.uuid4().hex
    parts = [stamps, data or [], identity or nonce]
    if stride != 1:
        parts.append({"stride": stride})
    payload = json.dumps(parts)
    return hashlib.sha256(payload.encode()).hexdigest()[:12], identity


def _smoke_last_used(folder: Path) -> float:
    """Return when ``folder`` (or one of its kind subfolders) last changed."""
    stamps = [folder.stat().st_mtime]
    for sub in folder.iterdir():
        try:
            stamps.append(sub.stat().st_mtime)
        except OSError:
            pass
    return max(stamps)


def _prune_smoke_cache(root: Path, current: Path) -> list[Path]:
    """Delete old smoke-cache folders under ``root``; return those removed.

    Keeps ``current`` and the newest others, :data:`SMOKE_CACHE_KEEP` in all, and never
    removes ``current`` or anything used within :data:`SMOKE_CACHE_GRACE_S`. A folder
    named for a cache kind is a leftover of the older one-folder layout: it is not
    counted towards the keep, and goes once it is past the grace period.
    """
    from ocean_skill import cache as _cache

    try:
        siblings = [d for d in root.iterdir() if d.is_dir() and d != current]
    except OSError:
        return []
    used = {}
    for d in siblings:
        try:
            used[d] = _smoke_last_used(d)
        except OSError:
            pass
    legacy = [d for d in used if d.name in _cache.KINDS]
    hashed = sorted(
        (d for d in used if d not in legacy), key=used.__getitem__, reverse=True
    )
    doomed = hashed[max(SMOKE_CACHE_KEEP - 1, 0) :] + legacy
    now = time.time()
    removed = []
    for d in doomed:
        if now - used[d] > SMOKE_CACHE_GRACE_S:
            shutil.rmtree(d, ignore_errors=True)
            removed.append(d)
    return removed


def run_suite(
    path: str | Path,
    *,
    list_only: bool = False,
    refresh: bool = True,
    smoke: int | None = None,
    smoke_stride: int = SMOKE_STRIDE,
    self_check: bool = False,
    only: str | list[str] | None = None,
    strict: bool = False,
) -> SuiteResult:
    """Load, expand, and draw a suite YAML; return a :class:`SuiteResult`.

    ``list_only=True`` validates the schema, resolves every ``latest``/``month:
    run``/``for_each`` and the literal time windows, and prints the resolved page
    list without drawing any page or writing a report. It is not read-only: a
    ``refresh:`` block still runs first, rebuilding the kerchunk references and
    catalog entries it names -- unless ``refresh=False``, which reads the catalog
    and references as they already are. That is what lets several suites sharing
    one ``refresh:`` target run at once: one refreshes, the rest only read.

    The remaining options are guardrails that make a bug show up in minutes rather than
    after the full compute; all combine with each other and with ``list_only``/
    ``refresh``:

    ``smoke=N``
        Read every model source named under ``refresh.sources`` (or, with no
        ``refresh:`` block, ``defaults.test``) with only its first and last ``N`` time
        steps, plus the steps the pages pin (the nearest step to each date a page names,
        and the first and last ``N`` inside each window under 62 days; see
        :func:`_smoke_pins`) --
        through :func:`ocean_skill.sources.truncate_time`, so the ``latest``/``month:
        run`` lookups at expansion see the same shortened record. The cache key does not
        know about the truncation, so the run's results go in
        ``<cache>/smoke-cache/<hash>/`` (:func:`ocean_skill.cache.scoped`), never the
        real cache, and the report in ``<output_dir>/<name>-smoke/``. The hash names the
        truncation (sources, ``N``, the time stamps kept, pinned ones included) and the
        code version
        (:func:`_code_identity`), so an identical run reuses it and any change starts
        cold; old folders are pruned (:data:`SMOKE_CACHE_KEEP`).

        This is "test mode", and it cuts three more things so a suite's bugs show up in
        minutes, not as long as the real run: each compare page's ``times:`` list and
        each ``for_each: {month: ...}`` fan keep only their first and last entry
        (``thin_dates``; the pins are collected after that, so fewer steps are kept);
        every truncated model source is read on every ``smoke_stride``-th horizontal
        cell (:func:`ocean_skill.sources.truncate_time`); and figures are saved at no
        more than :data:`~ocean_skill.suites.report.SMOKE_DPI` dpi. The strided
        values are physically inconsistent (staggering, cell areas) -- it is a test of
        the code paths, not of the numbers. The stride is part of the smoke-cache hash.
    ``smoke_stride``
        The horizontal stride of test mode (default :data:`SMOKE_STRIDE`; 1 leaves the
        grid whole). Ignored without ``smoke``.
    ``self_check``
        Every ``compare:`` page is redrawn with the reference replaced by the test
        source (:func:`ocean_skill.suites.pages.self_check_page`) and any bias, error
        or correlation shortfall between those identical lanes is a problem. Report in
        ``<name>-selfcheck/``.
    ``only``
        Draw only one part of the suite: a glob (or a list of globs) matched
        case-insensitively against the titles. A glob that matches a ``section:``
        divider keeps that whole section (the divider and every page up to the next
        divider); one that matches a figure page keeps that page. A page is kept if any
        glob keeps it, and a glob that matches nothing is an error. Each selection
        reports in its own ``<name>-only-<slug of the globs>/`` (``--only "River
        inputs"`` -> ``<name>-only-river-inputs/``), so rerunning one part never
        overwrites another's. The results cache is the full run's own: a part reads
        what a full run stored and fills it for the full run.
    ``strict``
        A layout warning raised while drawing a page (:data:`_STRICT_PATTERNS`) is a
        problem; the page still draws.

    Whatever is wrong with the pages is returned in :attr:`SuiteResult.problems`.
    """
    import yaml

    from ocean_skill import catalog, extrema, outputs
    from ocean_skill.config import SuiteConfig
    from ocean_skill.suites import pages as _pages

    path = Path(path).expanduser()
    raw = yaml.safe_load(path.read_text())
    suite = SuiteConfig.model_validate(raw)

    if smoke is not None and smoke < 1:
        raise ValueError(f"smoke: need at least 1 time step, got {smoke}")
    if smoke is not None and smoke_stride < 1:
        raise ValueError(f"smoke: stride must be at least 1, got {smoke_stride}")

    only_globs = _only_globs(only)
    # Only the options that are set. A run that is not the whole suite as written must
    # not replace the real report, so these also name its directory (``mode_tag``).
    modes: dict[str, Any] = {
        k: v
        for k, v in (
            ("smoke", smoke),
            ("self_check", True if self_check else None),
            ("only", only_globs or None),
            ("strict", True if strict else None),
        )
        if v is not None
    }
    mode_tag = "".join(
        tag
        for key, tag in (
            ("smoke", "-smoke"),
            ("self_check", "-selfcheck"),
            ("only", _only_tag(only_globs)),
        )
        if key in modes
    )
    smoke_sources: list[str] | None = None
    smoke_dir: Path | None = None

    with _capture_terminal() as cap, contextlib.ExitStack() as stack:
        catalog_dirs = _resolve_catalog_dirs(suite.catalog_search_paths, path)
        # The folder this suite's own refresh: writes its catalog into may not exist
        # before the first refresh; it is created rather than refused.
        refresh_dir = (
            Path(suite.refresh.catalog).expanduser().resolve().parent
            if suite.refresh is not None and refresh
            else None
        )
        for entry, resolved in zip(suite.catalog_search_paths, catalog_dirs):
            if resolved == refresh_dir:
                resolved.mkdir(parents=True, exist_ok=True)
            if not resolved.is_dir():
                raise FileNotFoundError(
                    f"catalog_search_paths: {entry!r} resolved to {resolved}, which "
                    "is not a directory"
                )
            if resolved not in catalog._added_dirs:
                catalog.add_search_path(resolved)

        cache_dir = _resolve_cache_dir(suite.cache_dir, path)
        if cache_dir is not None:
            from ocean_skill import cache

            cache.enable(cache_dir)

        if suite.refresh is not None and not refresh:
            print(
                f"refresh: skipped (--no-refresh); reading {suite.refresh.catalog} "
                "as it is"
            )
        elif suite.refresh is not None:
            _refresh_sources(
                [s.model_dump(exclude_none=True) for s in suite.refresh.sources],
                suite.refresh.catalog,
            )
            if suite.refresh.forcing:
                _refresh_forcing(
                    [f.model_dump(exclude_none=True) for f in suite.refresh.forcing],
                    suite.refresh.catalog,
                )

        if smoke is not None:
            from ocean_skill import cache as _smoke_cache
            from ocean_skill import sources as _sources
            from ocean_skill.suites.report import SMOKE_DPI

            test = suite.defaults.get("test")
            smoke_sources = (
                [src.name for src in suite.refresh.sources] if suite.refresh else []
            )
            if not smoke_sources and isinstance(test, str):
                smoke_sources = [test]
            if not smoke_sources:
                raise ValueError(
                    "smoke: no model source to shorten -- name them under "
                    "refresh.sources or set defaults.test"
                )
            modes["smoke_sources"] = smoke_sources
            modes["smoke_stride"] = smoke_stride
            # the other sources are cut to end where the test source does, so a
            # reference run that continues past the suite's window stays inside it
            anchor = test if isinstance(test, str) else smoke_sources[0]
            # plus the steps the pages name, once their date lists are thinned
            # (collected before the cut, see _smoke_expand)
            throwaway = _smoke_expand(suite, only_globs)
            pins = (
                _sources.TimePins()
                if throwaway is None
                else _pages.time_pins(throwaway)
            )
            n_thinned = sum(1 for p in throwaway or () if p.thinned)
            modes["smoke_pins"] = {
                "instants": len(pins.instants),
                "windows": len(pins.windows),
            }
            modes["smoke_thinned_pages"] = n_thinned
            stack.enter_context(
                _sources.truncate_time(
                    smoke_sources,
                    smoke,
                    anchor=anchor,
                    pins=pins,
                    stride=smoke_stride,
                )
            )
            # Its keys know neither N nor how far the model has grown, so the cache is
            # named for the truncation (what it keeps) and the code, and a change to
            # either starts a new, cold one.
            stamps = _smoke_stamps(smoke_sources, smoke)
            key, identity = _smoke_cache_key(
                stamps, _smoke_data_stamps(suite.refresh), smoke_stride
            )
            smoke_dir = stack.enter_context(_smoke_cache.scoped(f"smoke-cache/{key}"))
            reused = smoke_dir.is_dir()
            if not list_only:
                smoke_dir.mkdir(parents=True, exist_ok=True)
                os.utime(smoke_dir)  # marks it recently used for other runs' pruning
                _prune_smoke_cache(smoke_dir.parent, smoke_dir)
            modes["smoke_cache"] = str(smoke_dir)
            modes["smoke_cache_key"] = key
            modes["smoke_cache_reused"] = reused
            kept = ", ".join(f"{st[0]} {st[4]} step(s)" for st in stamps)
            grid = "whole grid" if smoke_stride == 1 else f"every {smoke_stride}th cell"
            print(
                f"smoke: reading the first and last {smoke} time step(s) of "
                f"{smoke_sources}, plus the steps the pages pin "
                f"({len(pins.instants)} instant(s), {len(pins.windows)} short "
                f"window(s)); kept {kept}; test mode: model grid {grid} "
                f"(stride {smoke_stride}), date lists cut to first and last on "
                f"{n_thinned} page(s), figures at most {SMOKE_DPI} dpi; "
                f"results cached in {smoke_dir}"
            )
            print(f"smoke cache: {smoke_dir} ({'reused' if reused else 'new'})")
            if identity is None:
                print(
                    "smoke cache: no git checkout to name the code version, so every "
                    "run starts cold"
                )

        expanded = _pages.expand(suite, thin_dates=smoke is not None)
        if only_globs:
            expanded = _select_pages(expanded, only_globs)
        if self_check:
            expanded, self_checked = _self_check_pages(expanded)
        else:
            self_checked = set()

        test_source = suite.defaults.get("test")
        index = None
        if test_source:
            # Not fatal: expand() has already flagged the pages this source spoils
            # (a refresh that failed to build it, or a climatology), and the rest
            # still draw. The report then just has no time range in its name.
            try:
                index = extrema._native_time_index(test_source)
            except Exception as exc:
                print(
                    f"latest step of {test_source}: unknown "
                    f"({type(exc).__name__}: {exc})"
                )
        # Printed once, unconditionally, so it lands in run.log even for an
        # ordinary (non-list_only) run: whether "latest" moved since the last run
        # is exactly what decides whether the cache-eligible pages below actually
        # hit or recompute -- see "Caching" in docs/suites.md.
        if index is not None:
            print(f"latest step of {test_source}: {index[-1]}")

        if list_only:
            from ocean_skill import cache as _cache

            print(f"cache: {_cache.base_dir()}")
            if smoke_dir is not None:
                print(f"smoke cache: {smoke_dir} ({'reused' if reused else 'new'})")
            for i, p in enumerate(expanded, 1):
                if p.kind == "section":
                    print(f"{i:2d}. [section] {p.title}")
                    continue
                if p.cache:
                    cache_note = "cache"
                elif not suite.cache:
                    cache_note = "no-cache (cache: false)"
                else:
                    cache_note = "no-cache (may change as the run grows)"
                chain = "".join(
                    " -> {}({})".format(
                        s["name"],
                        ", ".join(f"{k}={v!r}" for k, v in s["kwargs"].items()),
                    )
                    for s in p.steps
                )
                if p.contours:
                    chain += " + contours"
                if p.draw not in (None, "plot"):
                    chain += f" + draw: {p.draw}"
                thinned = f"  ({p.thinned})" if p.thinned else ""
                print(
                    f"{i:2d}. [{p.kind:7s}] {p.title}  ({cache_note}){thinned}{chain}"
                )
                if p.problem:
                    print(f"      will be skipped: {p.problem}")
            return SuiteResult(pages=expanded)

        output_dir = (
            Path(suite.output_dir).expanduser()
            if suite.output_dir
            else outputs.base_dir()
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        if suite.timestamped_dir:
            report_dir = output_dir / _report_dir_name(
                suite.name + mode_tag, test_source or "model", index
            )
        else:
            report_dir = output_dir / outputs.slug(suite.name + mode_tag)
        report_dir.mkdir(parents=True, exist_ok=True)
        if not suite.timestamped_dir:
            _clear_previous_report(report_dir)
        cap.attach(report_dir / "run.log")
        (report_dir / "suite.yaml").write_text(path.read_text())

        from ocean_skill.suites import cover

        # gather() only warns over missing or garbled C-Star files; this guard is for a
        # bug in it, which must not cost a long suite run its report.
        try:
            info = cover.gather(suite, path, test_source=test_source, index=index)
        except Exception as exc:
            warnings.warn(f"cover: no cover page ({exc})", stacklevel=2)
            cap.log_only(traceback.format_exc())
            info = None
        for src in (info.blueprint, info.workplan) if info else ():
            if src is not None and src.exists():
                (report_dir / "run_info").mkdir(exist_ok=True)
                shutil.copy2(src, report_dir / "run_info" / src.name)

        refresh_info = None
        if suite.refresh is not None:
            ref_records = _snapshot_refs(suite.refresh, report_dir)
            refresh_info = {
                "catalog": str(Path(suite.refresh.catalog).expanduser().resolve()),
                "rebuilt": refresh,
                "sources": ref_records,
            }

        problems: list[PageProblem] = []

        manifest_path = report_dir / "manifest.json"
        _write_manifest(
            manifest_path,
            suite=suite,
            pages=expanded,
            report_dir=report_dir,
            catalog_dirs=catalog_dirs,
            refresh=refresh_info,
            run_info=info.as_dict() if info else None,
            modes=modes or None,
        )

        pdf_path = (report_dir / "report.pdf") if suite.pdf else None
        pooled_records: list[dict[str, Any]] = []
        # where the records since the last ``section:`` divider begin
        section_start = 0
        # what ``health:``/``rivers:``/``inputs:`` pages share: the diagnostic objects,
        # computed by the first page that needs each, and the job-segment start dates,
        # read from the workplan only when a health page builds (a failure there is that
        # page's, not the report's)
        run_context: dict[str, Any] = {
            "joins": lambda: cover.segment_starts(suite, path),
            "objects": {},
        }

        from ocean_skill.suites.report import PDF_DPI, PNG_DPI, SMOKE_DPI, PdfReport

        # test mode caps the figures' resolution: drawing, saving and the PDF's
        # rasterized maps all scale with it
        dpi = {"png_dpi": PNG_DPI, "pdf_dpi": PDF_DPI}
        if smoke is not None:
            dpi = {k: min(v, SMOKE_DPI) for k, v in dpi.items()}
        with PdfReport(pdf_path, report_dir / "figures", **dpi) as report:
            if suite.cover and pdf_path is not None and info is not None:
                try:
                    report.cover(cover.cover_figures(info))
                except Exception as exc:
                    warnings.warn(f"cover: no cover page ({exc})", stacklevel=2)
                    cap.log_only(traceback.format_exc())
            n = len(expanded)
            for i, page in enumerate(expanded, 1):
                if page.kind == "section":
                    # Not drawn by build(): a divider is queued on the report and is
                    # only written if a figure follows it (see PdfReport.section).
                    # Its status is settled after the loop, once it is known whether
                    # one did.
                    print(f"== section: {page.title} ==")
                    report.section(i, page.title, page.kwargs["text"])
                    section_start = len(pooled_records)
                    continue
                note = f" ({page.thinned})" if page.thinned else ""
                print(f"== page {i}/{n}: {page.title}{note} ==")
                if page.problem:
                    # found while expanding (its source could not be read), so not
                    # tried again here: that read can itself be the slow part
                    page.elapsed = 0.0
                    page.status = "skipped"
                    page.reason = page.problem
                    print(f"   SKIPPED: {page.reason}")
                    problems.append(
                        PageProblem(page.title, "skipped", page.problem, 0.0)
                    )
                    continue
                t0 = time.perf_counter()
                watch = (
                    _watch_warnings(_STRICT_PATTERNS)
                    if strict
                    else contextlib.nullcontext([])
                )
                with _PeakMemory() as mem, watch as strict_hits:
                    try:
                        results = _pages.build(
                            page,
                            pooled_records=(
                                pooled_records[section_start:]
                                if page.kind == "summary"
                                and page.kwargs.get("scope") == "section"
                                else pooled_records
                            ),
                            **(
                                {"run_context": run_context}
                                if page.kind in ("health", "rivers", "inputs")
                                else {}
                            ),
                        )
                        error = None
                    except Exception as exc:
                        error = exc
                        cap.log_only(traceback.format_exc())
                    if error is None:
                        # inside the watch: layout warnings (tight/constrained layout)
                        # fire when the figure is rendered, which is here, not in build
                        for k, (suffix, fig) in enumerate(results):
                            # the caption goes under the page's first figure only
                            report.emit(
                                fig,
                                outputs.slug(page.title) + suffix,
                                caption=page.caption if k == 0 else None,
                            )
                page.elapsed = time.perf_counter() - t0
                page.peak_mb = mem.peak_mb
                if error is not None:
                    page.status = "skipped"
                    page.reason = f"{type(error).__name__}: {error}"
                    warnings.warn(
                        f"skipping page {page.title!r}: {error}", stacklevel=2
                    )
                    print(f"   SKIPPED after {page.elapsed:.1f}s: {page.reason}")
                    kind = (
                        "skipped" if isinstance(error, _pages.PageSkipped) else "failed"
                    )
                    problems.append(
                        PageProblem(
                            page.title, kind, page.reason, page.elapsed, page.peak_mb
                        )
                    )
                    continue
                page.status = "ok"
                print(f"   done in {page.elapsed:.1f}s")
                found = _page_problems(page, strict_hits, id(page) in self_checked)
                for kind, reason in found:
                    print(f"   PROBLEM [{kind}]: {_short(reason)}")
                    problems.append(
                        PageProblem(
                            page.title, kind, reason, page.elapsed, page.peak_mb
                        )
                    )
                pooled_records.extend(page.metrics_records)
                if page.summary_table is not None:
                    table_dir = report_dir / "metrics"
                    table_dir.mkdir(exist_ok=True)
                    page.summary_table.to_csv(
                        table_dir / f"{suite.name}_{outputs.slug(page.title)}.csv"
                    )

            metrics_csv = None
            if pooled_records:
                from ocean_skill import metrics as _metrics

                metrics_csv = _metrics.write(
                    pooled_records, report_dir, stem=suite.name
                )

        # A divider is "ok" only if PdfReport actually wrote it, i.e. some page after
        # it drew before the next section (or the end of the run) came up.
        for i, page in enumerate(expanded, 1):
            if page.kind != "section":
                continue
            if i in report.written_sections:
                page.status = "ok"
            else:
                page.status = "skipped"
                page.reason = (
                    "pdf: false -- section pages appear only in report.pdf"
                    if pdf_path is None
                    else "no page after this section drew"
                )

        pdf_images_applied = False
        if suite.pdf_images == "jpeg" and pdf_path is not None and pdf_path.exists():
            from ocean_skill.suites.report import compress_pdf_images

            sizes = compress_pdf_images(pdf_path)
            if sizes is not None:
                pdf_images_applied = True
                print(
                    f"report.pdf: {sizes[0] / 1e6:.1f} MB -> {sizes[1] / 1e6:.1f} MB "
                    "(JPEG via Ghostscript)"
                )

        _write_manifest(
            manifest_path,
            suite=suite,
            pages=expanded,
            report_dir=report_dir,
            catalog_dirs=catalog_dirs,
            refresh=refresh_info,
            pdf_images_applied=pdf_images_applied,
            run_info=info.as_dict() if info else None,
            modes=modes or None,
            problems=problems,
        )

        if not mode_tag:  # a smoke/self-check/partial report is not "the latest report"
            latest_path = output_dir / "latest.txt"
            latest_path.write_text(str(report_dir))
            latest_link = output_dir / "latest"
            try:
                if latest_link.is_symlink() or latest_link.exists():
                    latest_link.unlink()
                latest_link.symlink_to(report_dir.name)
            except OSError:
                pass  # best-effort only; latest.txt is the real contract

        result = SuiteResult(
            pages=expanded,
            report_dir=report_dir,
            pdf=pdf_path if pdf_path and pdf_path.exists() else None,
            figures=report.png_paths,
            metrics=metrics_csv,
            manifest=manifest_path,
            log=report_dir / "run.log",
            refs=[
                Path(r["snapshot"])
                for r in (refresh_info["sources"] if refresh_info else [])
                if r["snapshot"]
            ],
            problems=problems,
        )
    return result


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    ``ocean-skill-run suite.yaml [--list] [--no-refresh] [--smoke [N]]
    [--smoke-stride K] [--self-check] [--only GLOB]... [--strict]``. Put ``--smoke``'s
    count right after it (``--smoke 3``, or ``--smoke=3``), and the suite path before
    it, or the path is taken as the count.
    """
    import matplotlib

    matplotlib.use("Agg")

    parser = argparse.ArgumentParser(
        prog="ocean-skill-run",
        description="Run a suite YAML: draw every page, write PNGs and a PDF.",
    )
    parser.add_argument("suite", help="path to the suite YAML")
    parser.add_argument(
        "--list",
        action="store_true",
        help="print the expanded page list and exit without drawing anything",
    )
    parser.add_argument(
        "--no-refresh",
        action="store_true",
        help="skip the suite's refresh: block and read its catalog as it is",
    )
    parser.add_argument(
        "--smoke",
        nargs="?",
        const=2,
        default=None,
        type=int,
        metavar="N",
        help="test mode: read only the first and last N (default 2) time steps of the "
        "suite's model sources, on every Kth cell (--smoke-stride), with date lists "
        "cut to first and last and figures at low dpi; report in <name>-smoke/, "
        "results in a separate smoke-cache",
    )
    parser.add_argument(
        "--smoke-stride",
        type=int,
        default=SMOKE_STRIDE,
        metavar="K",
        help=f"with --smoke, read every Kth cell of the model grid (default "
        f"{SMOKE_STRIDE}; 1 keeps the whole grid): values are then physically "
        "inconsistent, which is fine for a test of the code paths",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="draw every compare page against itself (reference = test source) and "
        "report any bias/error/correlation shortfall between the identical lanes",
    )
    parser.add_argument(
        "--only",
        metavar="GLOB",
        action="append",
        help="draw only one part: a section title (that whole section) or a page "
        "title, as a case-insensitive glob; repeat it for several parts; report in "
        "<name>-only-<glob>/",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="count a layout warning (cramped maps...) raised while drawing a page "
        "as a problem",
    )
    argv = argv if argv is not None else sys.argv[1:]
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 2

    try:
        result = run_suite(
            args.suite,
            list_only=args.list,
            refresh=not args.no_refresh,
            smoke=args.smoke,
            smoke_stride=args.smoke_stride,
            self_check=args.self_check,
            only=args.only,
            strict=args.strict,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.list:
        return 0

    figure_pages = [p for p in result.pages if p.kind != "section"]
    n_ok = sum(1 for p in figure_pages if p.status == "ok")
    n_skipped = len(figure_pages) - n_ok
    summary_lines = [
        f"{n_ok} page(s) drawn, {n_skipped} skipped -> {result.report_dir}"
    ]
    if result.pdf:
        summary_lines.append(f"  report: {result.pdf}")
    if result.metrics:
        summary_lines.append(f"  metrics: {result.metrics}")
    if result.refs:
        summary_lines.append(
            f"  refs: {len(result.refs)} reference snapshot(s) -> "
            f"{result.report_dir / 'refs'}"
        )
    summary_lines.extend(_problem_lines(result.problems))
    for line in summary_lines:
        print(line)
    # run_suite's own tee is closed by the time these lines print, so append them
    # to run.log directly -- this is the only bit of terminal output not already
    # captured by run_suite itself.
    if result.log is not None:
        with result.log.open("a", encoding="utf-8") as f:
            f.write("\n".join(summary_lines) + "\n")
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
