"""Run a suite: ``ocean-skill-run suite.yaml`` (see ``docs/suites.md`` for the grammar).

A suite (``ocean_skill.config.SuiteConfig``) is a YAML file listing pages -- each
one a single ``osk.field``, ``osk.compare``, or ``osk.summary`` call -- plus shared
defaults and output settings. Running it draws every page, writes a PNG per figure,
collects them into one PDF (unless ``pdf: false``), and writes a metrics CSV and a
``manifest.json`` recording exactly what was drawn. It also writes ``run.log`` --
everything printed to the terminal over the course of the run, plus full tracebacks
for skipped pages and for a fatal crash, which the terminal itself never shows. See
``docs/suites.md``.

Optionally the suite can refresh a model's kerchunk reference before running, which
is what makes it usable against a run that is still writing output.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
import traceback
import warnings
from dataclasses import dataclass
from dataclasses import field as _dc_field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

__all__ = ["main", "run_suite"]


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


def _refresh_sources(spec: list[dict[str, Any]], catalog_path: str | Path) -> None:
    """Rebuild kerchunk references and their catalog entries from the current files.

    Reopens the existing catalog (if any) and updates it in place, rather than
    building a fresh one from scratch: a run between output steps — or one that was
    restarted and has temporarily produced no files — must not wipe out every other
    entry the catalog already had. A source whose glob matches nothing simply keeps
    whatever entry it already had; only sources that actually rebuilt get touched.

    An entry's ``keep`` key, when given, is forwarded to
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
    """
    import glob as _glob

    import intake

    from ocean_skill.build import (
        _partition_unfinished,
        add_source,
        make_kerchunk,
        new_catalog,
        save,
    )

    catalog_path = Path(catalog_path).expanduser()
    cat = None
    if catalog_path.exists():
        try:
            cat = intake.from_yaml_file(str(catalog_path))
        except Exception as exc:
            print(
                f"  refresh: could not reopen existing {catalog_path} ({exc}); "
                "starting a fresh catalog"
            )
    if cat is None:
        cat = new_catalog(title="refreshed by ocean_skill.workflows.run")

    rebuilt, skipped = [], []
    for entry in spec:
        # glob.glob (not Path.glob) so absolute patterns work
        files = sorted(_glob.glob(str(Path(entry["files"]).expanduser())))
        if not files:
            skipped.append(entry["name"])
            continue
        min_age = entry.get("min_age", 0.0)
        ready, unfinished = _partition_unfinished(
            [Path(f) for f in files], min_age=min_age
        )
        for p, why in unfinished.items():
            print(f"  refresh: {entry['name']}: skipping {p} ({why})")
        if not ready:
            skipped.append(entry["name"])
            continue
        # Left out entirely (not defaulted to "all") when the suite doesn't name one,
        # so make_kerchunk's own default -- "unique" -- takes effect here too.
        kwargs = {"keep": entry["keep"]} if "keep" in entry else {}
        ref = make_kerchunk(
            ready, entry["ref"], grid=entry.get("grid"), min_age=min_age, **kwargs
        )
        add_source(cat, entry["name"], ref)
        rebuilt.append(entry["name"])
        print(f"  refresh: {entry['name']} <- {len(ready)} files")

    if skipped:
        print(f"  refresh: no files matched for {skipped}; kept their existing entries")
    if not rebuilt and not skipped:
        return  # nothing declared to refresh; leave the file untouched
    save(cat, catalog_path)


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

    @property
    def exit_code(self) -> int:
        if not self.pages:
            return 1
        statuses = {p.status for p in self.pages}
        if statuses == {"ok"}:
            return 0
        if "ok" not in statuses:
            return 1
        return 3


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


def _write_manifest(
    path: Path,
    *,
    suite: Any,
    pages: list[Any],
    report_dir: Path,
    catalog_dirs: list[Path] | None = None,
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
        "pages": [
            {
                **p.as_dict(),
                "status": p.status,
                "reason": p.reason,
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


def _title_text(suite: Any, pages: list[Any], *, test_source: str, index: Any) -> str:
    lines = [
        f"suite: {suite.name}",
        f"test source: {test_source}",
        f"run coverage: {index[0]} .. {index[-1]}  ({len(index)} steps)",
        f"generated: {datetime.now(UTC).isoformat()}",
        f"pages planned: {len(pages)}",
        "",
    ]
    for p in pages:
        lines.append(f"  - {p.title}")
    return "\n".join(lines)


def _log_text(pages: list[Any], metrics_csv: Path | None) -> str:
    lines = ["run log", ""]
    for p in pages:
        mark = "ok" if p.status == "ok" else "SKIPPED"
        timing = f"  ({p.elapsed:.1f}s)" if p.elapsed is not None else ""
        lines.append(f"  [{mark}] {p.title}{timing}")
        if p.reason:
            lines.append(f"          {p.reason}")
    lines.append("")
    lines.append(
        f"metrics: {metrics_csv}" if metrics_csv else "metrics: (none written)"
    )
    return "\n".join(lines)


def run_suite(path: str | Path, *, list_only: bool = False) -> SuiteResult:
    """Load, expand, and draw a suite YAML; return a :class:`SuiteResult`.

    ``list_only=True`` validates the schema, resolves every ``latest``/``month:
    run``/``for_each`` and the literal time windows, and prints the resolved page
    list without drawing or writing anything.
    """
    import yaml

    from ocean_skill import catalog, extrema, outputs
    from ocean_skill.config import SuiteConfig
    from ocean_skill.workflows import pages as _pages

    path = Path(path).expanduser()
    raw = yaml.safe_load(path.read_text())
    suite = SuiteConfig.model_validate(raw)

    with _capture_terminal() as cap:
        catalog_dirs = _resolve_catalog_dirs(suite.catalog_search_paths, path)
        for entry, resolved in zip(suite.catalog_search_paths, catalog_dirs):
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

        if suite.refresh is not None:
            _refresh_sources(
                [s.model_dump(exclude_none=True) for s in suite.refresh.sources],
                suite.refresh.catalog,
            )

        expanded = _pages.expand(suite)

        test_source = suite.defaults.get("test")
        index = extrema._native_time_index(test_source) if test_source else None

        if list_only:
            from ocean_skill import cache as _cache

            print(f"cache: {_cache.base_dir()}")
            for i, p in enumerate(expanded, 1):
                cache_note = "cache" if p.cache else "no-cache (open window)"
                print(f"{i:2d}. [{p.kind:7s}] {p.title}  ({cache_note})")
            return SuiteResult(pages=expanded)

        output_dir = (
            Path(suite.output_dir).expanduser()
            if suite.output_dir
            else outputs.base_dir()
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        report_dir = output_dir / _report_dir_name(
            suite.name, test_source or "model", index
        )
        report_dir.mkdir(parents=True, exist_ok=True)
        cap.attach(report_dir / "run.log")
        (report_dir / "suite.yaml").write_text(path.read_text())

        manifest_path = report_dir / "manifest.json"
        _write_manifest(
            manifest_path,
            suite=suite,
            pages=expanded,
            report_dir=report_dir,
            catalog_dirs=catalog_dirs,
        )

        pdf_path = (report_dir / "report.pdf") if suite.pdf else None
        pooled_records: list[dict[str, Any]] = []

        from ocean_skill.workflows.report import PdfReport

        with PdfReport(pdf_path, report_dir / "figures") as report:
            report.title_page(
                _title_text(suite, expanded, test_source=test_source, index=index)
            )

            n = len(expanded)
            for i, page in enumerate(expanded, 1):
                print(f"== page {i}/{n}: {page.title} ==")
                t0 = time.perf_counter()
                try:
                    results = _pages.build(page, pooled_records=pooled_records)
                except Exception as exc:
                    page.elapsed = time.perf_counter() - t0
                    page.status = "skipped"
                    page.reason = f"{type(exc).__name__}: {exc}"
                    warnings.warn(
                        f"skipping page {page.title!r}: {exc}", stacklevel=2
                    )
                    cap.log_only(traceback.format_exc())
                    print(f"   SKIPPED after {page.elapsed:.1f}s: {page.reason}")
                    continue
                page.elapsed = time.perf_counter() - t0
                page.status = "ok"
                print(f"   done in {page.elapsed:.1f}s")
                for suffix, fig in results:
                    report.emit(fig, outputs.slug(page.title) + suffix)
                pooled_records.extend(page.metrics_records)

            metrics_csv = None
            if pooled_records:
                from ocean_skill import metrics as _metrics

                metrics_csv = _metrics.write(
                    pooled_records, report_dir, stem=suite.name
                )

            report.log_page(_log_text(expanded, metrics_csv))

        _write_manifest(
            manifest_path,
            suite=suite,
            pages=expanded,
            report_dir=report_dir,
            catalog_dirs=catalog_dirs,
        )

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
        )
    return result


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: ``ocean-skill-run suite.yaml [--list]``."""
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
    argv = argv if argv is not None else sys.argv[1:]
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 2

    try:
        result = run_suite(args.suite, list_only=args.list)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.list:
        return 0

    n_ok = sum(1 for p in result.pages if p.status == "ok")
    n_skipped = len(result.pages) - n_ok
    summary_lines = [
        f"{n_ok} page(s) drawn, {n_skipped} skipped -> {result.report_dir}"
    ]
    if result.pdf:
        summary_lines.append(f"  report: {result.pdf}")
    if result.metrics:
        summary_lines.append(f"  metrics: {result.metrics}")
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
