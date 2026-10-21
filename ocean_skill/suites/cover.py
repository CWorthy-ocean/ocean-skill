"""The cover page of a suite's ``report.pdf``: which simulation the report is for.

:func:`gather` reads the C-Star files behind a suite's model -- the resolved blueprint
(what was actually run) and the workplan that produced it -- and collects what a reader
of the PDF wants on page 1 into a :class:`RunInfo`; :func:`cover_figures` draws it as
one or more letter-size figures. The files are parsed as plain YAML with every field
read through ``.get``, so there is no ``cstar`` dependency and a different schema
version degrades to fewer lines rather than an error. A missing or garbled file only
warns: a cover must never fail a run.
"""

from __future__ import annotations

import re
import textwrap
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from ocean_skill.config import SuiteConfig
from ocean_skill.plot.typography import PAGE_H, PAGE_W, STEP, reference_scale
from ocean_skill.suites.report import _wrap_lines

__all__ = [
    "RunInfo",
    "cover_figures",
    "find_workplan",
    "gather",
    "newest_blueprint",
    "segment_starts",
]

#: Page geometry, in figure fractions: text starts at ``MARGIN_X`` and a value column
#: begins at ``VALUE_X`` (labels sit between), and a page ends at ``BOTTOM`` -- the next
#: line that would pass it starts a new page.
MARGIN_X = 0.08
VALUE_X = 0.29
TOP = 0.93
BOTTOM = 0.06

#: Type: the title is the report's suptitle role raised this many modular-scale steps
#: (as for a section divider); body text is 8.4pt monospace with ``LINE_SPACING`` times
#: its size between baselines. Values wrap at ``VALUE_WRAP`` characters (what fits
#: between ``VALUE_X`` and the right margin at that size), labels at ``LABEL_WRAP``.
TITLE_STEPS = 4
LINE_SPACING = 1.45
VALUE_WRAP = 74
LABEL_WRAP = 24
TITLE_WRAP = 34
SUBTITLE_WRAP = 80

_NO_BRANCH = {"", "na", "n/a", "none", "null"}


@dataclass
class RunInfo:
    """What the cover page says: a title, a subtitle, and headed sections of rows."""

    name: str
    subtitle: str
    sections: list[tuple[str, list[tuple[str, str]]]]
    blueprint: Path | None
    workplan: Path | None

    def as_dict(self) -> dict:
        """JSON-ready form, for ``manifest.json``."""
        return {
            "name": self.name,
            "subtitle": self.subtitle,
            "blueprint": None if self.blueprint is None else str(self.blueprint),
            "workplan": None if self.workplan is None else str(self.workplan),
            "sections": [
                {"heading": h, "items": [[k, v] for k, v in items]}
                for h, items in self.sections
            ],
        }


# --- reading C-Star files --------------------------------------------------------


def _load(path: Path) -> dict | None:
    """Parse ``path`` as a YAML mapping, or warn and return None."""
    try:
        data = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError, UnicodeDecodeError) as exc:
        warnings.warn(f"cover: cannot read {path}: {exc}", stacklevel=3)
        return None
    if not isinstance(data, dict):
        warnings.warn(f"cover: {path} is not a YAML mapping", stacklevel=3)
        return None
    return data


def _resolve(given: str, base: Path) -> Path | None:
    """``given`` made absolute against ``base`` (the suite's folder), if it exists."""
    path = Path(given).expanduser()
    path = path if path.is_absolute() else base / path
    if path.is_file():
        return path
    warnings.warn(f"cover: run_info path {path} does not exist", stacklevel=3)
    return None


def _globs(files: str | list[str]) -> list[str]:
    """Return a ``refresh.sources[].files`` value (glob or list of globs) as a list."""
    return [files] if isinstance(files, str) else list(files)


def _run_dir(suite: SuiteConfig) -> Path | None:
    """Find the C-Star run folder behind ``refresh.sources[].files``, if any."""
    for source in suite.refresh.sources if suite.refresh else []:
        first = source.files if isinstance(source.files, str) else source.files[0]
        parent = Path(first).expanduser().parent
        for cand in (parent, parent.parent):
            if (
                not re.search(r"[*?\[]", str(cand))
                and (cand / "work" / "directives.yaml").is_file()
            ):
                return cand
    return None


def newest_blueprint(run_dir: Path) -> Path | None:
    """Find the newest ``work/*.yaml`` that looks like a (resolved) blueprint."""
    files = [
        p for p in (run_dir / "work").glob("*.yaml") if p.name != "directives.yaml"
    ]
    for path in sorted(files, key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = yaml.safe_load(path.read_text())
        except (OSError, yaml.YAMLError, UnicodeDecodeError):
            continue
        if isinstance(data, dict) and (
            "forcing" in data or "initial_conditions" in data
        ):
            return path
    return None


def _same_dir(a: Any, b: Path) -> bool:
    return isinstance(a, str) and Path(a).expanduser().resolve() == b.resolve()


def _step_for(workplan: dict, run_dir: Path | None) -> dict:
    """Pick the workplan step running in ``run_dir`` (or the only step)."""
    steps = [s for s in workplan.get("steps") or [] if isinstance(s, dict)]
    if run_dir is not None:
        for step in steps:
            if _same_dir(step.get("working_dir"), run_dir):
                return step
    return steps[0] if len(steps) == 1 else {}


def find_workplan(run_dir: Path) -> Path | None:
    """Find the one workplan beside ``tasks/`` with a step running in ``run_dir``."""
    hits = []
    for path in sorted(run_dir.parent.parent.glob("*.yaml")):
        try:
            data = yaml.safe_load(path.read_text())
        except (OSError, yaml.YAMLError, UnicodeDecodeError):
            continue
        steps = data.get("steps") if isinstance(data, dict) else None
        if any(
            isinstance(s, dict) and _same_dir(s.get("working_dir"), run_dir)
            for s in steps or []
        ):
            hits.append(path)
    if len(hits) > 1:
        names = ", ".join(p.name for p in hits)
        warnings.warn(
            f"cover: {len(hits)} workplans have a step in {run_dir} ({names}); "
            "set run_info.workplan to choose one",
            stacklevel=3,
        )
    return hits[0] if len(hits) == 1 else None


def _workplan_dates(wp: dict, wp_path: Path) -> tuple[list[datetime], list[datetime]]:
    """Return every step's ``start_date`` and ``end_date`` in workplan ``wp`` (naive).

    A step's dates are its ``runtime_params`` ``start_date``/``end_date``: from its
    blueprint, then the resolved blueprint in its working folder (``work/*.yaml``), then
    the workplan's own ``apply-overrides`` for that step, later ones winning -- steps
    often share one base blueprint and differ only in the latter two. A blueprint path
    that is relative is read relative to the workplan's folder (``wp_path``).

    A run continued under a new workplan keeps its earlier jobs' folders beside the
    current ones (``tasks/jan_1wk`` next to ``tasks/july_ccs``), so every other folder
    next to a step's working folder is read too: its resolved blueprint, then its
    ``work/directives.yaml`` overrides. A folder with neither adds nothing.
    """
    import pandas as pd

    def stamp(value):
        t = pd.Timestamp(value)
        t = t.tz_convert("UTC").tz_localize(None) if t.tzinfo is not None else t
        return t.to_pydatetime()

    starts, ends = [], []

    def add(runtime: dict) -> None:
        if runtime.get("start_date") is not None:
            starts.append(stamp(runtime["start_date"]))
        if runtime.get("end_date") is not None:
            ends.append(stamp(runtime["end_date"]))

    for step in wp.get("steps") or []:
        if not isinstance(step, dict):
            continue
        runtime: dict = {}  # later sources win: base blueprint, task's, overrides
        blueprint = step.get("blueprint")
        if isinstance(blueprint, str):
            bp_path = Path(blueprint).expanduser()
            bp_path = bp_path if bp_path.is_absolute() else wp_path.parent / bp_path
            bp = _load(bp_path) if bp_path.is_file() else None
            runtime |= (bp or {}).get("runtime_params") or {}
        work = step.get("working_dir")
        task_bp = newest_blueprint(Path(work)) if isinstance(work, str) else None
        if task_bp is not None:
            runtime |= (_load(task_bp) or {}).get("runtime_params") or {}
        overrides = (
            ((step.get("directives") or {}).get("apply-overrides") or {})
            .get("overrides", {})
            .get("runtime_params")
        )
        runtime |= overrides or {}
        add(runtime)
    # earlier jobs of the same run, from workplans since replaced
    seen = {
        Path(s["working_dir"]).expanduser().resolve()
        for s in wp.get("steps") or []
        if isinstance(s, dict) and isinstance(s.get("working_dir"), str)
    }
    for parent in sorted({d.parent for d in seen}):
        for task in sorted(p for p in parent.iterdir() if p.is_dir()):
            if task.resolve() in seen:
                continue
            runtime = {}
            task_bp = newest_blueprint(task)
            if task_bp is not None:
                runtime |= (_load(task_bp) or {}).get("runtime_params") or {}
            path = task / "work" / "directives.yaml"
            directives = (_load(path) if path.is_file() else None) or {}
            runtime |= (
                ((directives.get("directives") or {}).get("apply-overrides") or {})
                .get("overrides", {})
                .get("runtime_params")
            ) or {}
            add(runtime)
    return starts, ends


def segment_starts(
    suite: SuiteConfig, suite_path: Path | None = None
) -> list[datetime]:
    """Return when the run's job segments after the first began (its restart "joins").

    A run restarted in pieces is one workplan with a step per piece. A step's dates are
    its ``runtime_params`` ``start_date``/``end_date``: from its blueprint, then the
    resolved blueprint in its working folder (``work/*.yaml``), then the workplan's own
    ``apply-overrides`` for that step, later ones winning -- steps often share one base
    blueprint and differ only in the latter two. Every start and end is a join except
    the run's first start and last end. The workplan is found
    as :func:`gather` finds it (``run_info.workplan``, relative to the suite file's
    folder, else the one with a step in the ``refresh:`` run folder); a step's
    blueprint, when its path is relative, is read relative to the workplan's folder.
    The result is sorted and free of repeats, without the first start (the run's own,
    not a join), and naive datetimes. Anything unreadable or missing yields ``[]``
    rather than an error: these dates only mark where a health figure should look for a
    restart.
    """
    try:
        info = suite.run_info
        base = (
            Path(suite_path).expanduser().resolve().parent
            if suite_path is not None
            else Path.cwd()
        )
        wp_path = None
        if info and info.workplan:
            given = Path(info.workplan).expanduser()
            given = given if given.is_absolute() else base / given
            wp_path = given if given.is_file() else None
        else:
            run_dir = _run_dir(suite)
            wp_path = find_workplan(run_dir) if run_dir is not None else None
        wp = None if wp_path is None else _load(wp_path)
        if wp is None:
            return []
        starts, ends = _workplan_dates(wp, wp_path)
        if not starts and not ends:
            return []
        # every step's start and end is a join, except where the whole run starts and
        # ends (steps that share one base blueprint carry their own dates in the
        # task's blueprint or the workplan's overrides, so ends are read as well)
        first = min(starts) if starts else None
        last = max(ends) if ends else None
        joins: list[datetime] = []
        for t in sorted({t for t in (*starts, *ends) if t != first and t != last}):
            # one segment's end and the next one's start (often a day apart) are one
            # join: keep the later, where the new segment begins
            if joins and t - joins[-1] < timedelta(days=2):
                joins[-1] = t
            else:
                joins.append(t)
        return joins
    except Exception as exc:
        warnings.warn(f"cover: no segment start dates ({exc})", stacklevel=2)
        return []


# --- condensing file lists ---------------------------------------------------------


def _condense(paths: list[str]) -> list[str]:
    """Lines listing ``paths``: each directory once, numbered series collapsed.

    ``/d/x_200912.nc ... /d/x_201101.nc`` becomes ``/d/`` and an indented
    ``x_{200912..201101}.nc (14 files)``: names that differ only in one run of digits
    form a series; singletons and names differing elsewhere stay as written.
    """
    by_dir: dict[str, list[str]] = {}
    for p in paths:
        parent, _, name = str(p).rpartition("/")
        by_dir.setdefault(parent + "/" if parent else "", []).append(name)
    lines: list[str] = []
    for directory, names in by_dir.items():
        if directory:
            lines.append(directory)
        indent = "  " if directory else ""
        lines.extend(indent + n for n in _series(sorted(names)))
    return lines


def _series(names: list[str]) -> list[str]:
    groups: dict[tuple, list[list[str]]] = {}
    for name in names:
        tokens = re.split(r"(\d+)", name)  # odd indexes are the digit runs
        groups.setdefault((len(tokens), *tokens[0::2]), []).append(tokens)
    out = []
    for members in groups.values():
        varying = [
            i for i in range(1, len(members[0]), 2) if len({m[i] for m in members}) > 1
        ]
        if len(members) == 1 or len(varying) != 1:
            out.extend("".join(m) for m in members)
            continue
        i = varying[0]
        runs = sorted((m[i] for m in members), key=lambda s: (len(s), s))
        pattern = [*members[0][:i], f"{{{runs[0]}..{runs[-1]}}}", *members[0][i + 1 :]]
        out.append(f"{''.join(pattern)} ({len(members)} files)")
    return out


# --- gathering ---------------------------------------------------------------------


def _locations(entry: Any) -> list[str]:
    """Paths of a blueprint ``{data: [{location: ...}, ...]}`` entry (lenient)."""
    data = entry.get("data") if isinstance(entry, dict) else None
    if isinstance(data, (str, dict)):
        data = [data]
    out = []
    for item in data if isinstance(data, list) else []:
        loc = item.get("location") if isinstance(item, dict) else item
        if isinstance(loc, str):
            out.append(loc)
    return out


def _kind(path: str) -> str:
    tokens = set(re.split(r"[^a-z0-9]+", path.rpartition("/")[2].lower()))
    return "physics" if "physics" in tokens else "BGC" if "bgc" in tokens else "other"


def _flatten(d: dict, prefix: str = "") -> list[str]:
    lines = []
    for key, value in d.items():
        if isinstance(value, dict):
            lines.extend(_flatten(value, f"{prefix}{key}."))
        else:
            lines.append(f"{prefix}{key} = {value}")
    return lines


def _text(value: Any) -> str:
    return " ".join(str(value).split())


def _date(value: Any) -> str:
    return str(value)[:10]


def _code(code: Any) -> list[tuple[str, str]]:
    rows = []
    for name, spec in code.items() if isinstance(code, dict) else []:
        spec = spec if isinstance(spec, dict) else {}
        loc, commit, branch = (spec.get(k) for k in ("location", "commit", "branch"))
        ref = commit or (None if str(branch).lower() in _NO_BRANCH else branch)
        rows.append((str(name), f"{loc or '?'}" + (f" @ {ref}" if ref else "")))
    return rows


def _forcing(forcing: dict) -> list[tuple[str, str]]:
    rows = []
    for key, title in (("boundary", "Boundary"), ("surface", "Surface")):
        paths = _locations(forcing.get(key))
        for kind in ("physics", "BGC", "other"):
            chosen = [p for p in paths if _kind(p) == kind]
            if chosen:
                rows.append((f"{title} ({kind})", "\n".join(_condense(chosen))))
    for key, title in (("river", "Rivers"), ("tidal", "Tides")):
        paths = _locations(forcing.get(key))
        if paths:
            rows.append((title, "\n".join(_condense(paths))))
    return rows


def _rows(*pairs: tuple[str, Any]) -> list[tuple[str, str]]:
    """``(label, value)`` pairs, without the ones whose value is missing or empty."""
    return [(k, str(v)) for k, v in pairs if v not in (None, "", [])]


def _run_span(wp: dict, wp_path: Path | None, runtime: Any) -> str | None:
    """Return the whole run's dates as ``"<first start> to <last end>"`` (dates only).

    A restarted run is one workplan with a task per job, and each task's blueprint has
    only its own piece's dates, so the span is read from every step of the workplan
    (:func:`_workplan_dates`): the earliest start to the latest end. With no workplan
    steps the blueprint's own ``runtime_params`` give it, else ``None``.
    """
    starts: list[Any] = []
    ends: list[Any] = []
    if wp_path is not None and wp:
        try:
            starts, ends = _workplan_dates(wp, wp_path)
        except Exception as exc:  # noqa: BLE001 -- a cover must never fail a run
            warnings.warn(f"cover: no workplan run dates ({exc})", stacklevel=2)
    rt = runtime if isinstance(runtime, dict) else {}
    start = min(starts) if starts else rt.get("start_date")
    end = max(ends) if ends else rt.get("end_date")
    if start is None or end is None:
        return None
    return f"{_date(start)} to {_date(end)}"


def gather(
    suite: SuiteConfig,
    suite_path: Path,
    *,
    test_source: str | None,
    index: Any | None,
) -> RunInfo:
    """Collect the cover page's content for ``suite``; never raises over a bad file.

    ``run_info.blueprint`` / ``run_info.workplan`` win when given (relative to the
    suite file). Otherwise they are found from ``refresh:`` -- the run folder is the
    one holding ``work/directives.yaml`` next to the refreshed ``files:``; its blueprint
    is the newest ``work/*.yaml``, its workplan the one YAML two levels up with a step
    running in it. ``index`` is the model source's time index (for its span).
    """
    info = suite.run_info
    base = Path(suite_path).expanduser().resolve().parent
    bp_path = _resolve(info.blueprint, base) if info and info.blueprint else None
    wp_path = _resolve(info.workplan, base) if info and info.workplan else None
    bp = _load(bp_path) if bp_path else None
    wp = _load(wp_path) if wp_path else None
    bp_path, wp_path = (
        (bp_path if bp is not None else None),
        (wp_path if wp is not None else None),
    )
    run_dir = None
    if not (bp_path and wp_path):
        run_dir = _run_dir(suite)
        if run_dir is not None:
            if not bp_path and (found := newest_blueprint(run_dir)):
                bp_path, bp = found, _load(found)
            if not wp_path and (found := find_workplan(run_dir)):
                wp_path, wp = found, _load(found)
    bp = bp or {}
    wp = wp or {}
    if run_dir is None and isinstance(bp.get("working_dir"), str):
        run_dir = Path(bp["working_dir"]).expanduser()
    step = _step_for(wp, run_dir)
    run_span = _run_span(wp, wp_path, bp.get("runtime_params"))

    span = (
        f"{_date(index[0])} - {_date(index[-1])}"
        if index is not None and len(index)
        else None
    )
    source = test_source or suite.defaults.get("test")
    refresh = suite.refresh
    import ocean_skill

    sections: list[tuple[str, list[tuple[str, str]]]] = [
        (
            "Report",
            _rows(
                ("Suite", suite.name),
                ("Suite file", Path(suite_path)),
                ("Generated", datetime.now(UTC).isoformat(timespec="seconds")),
                ("ocean-skill", getattr(ocean_skill, "__version__", None)),
                ("Model source", source),
                ("Model span", span),
                ("Refresh catalog", refresh.catalog if refresh else None),
                (
                    "Refreshed files",
                    "\n".join(
                        f for s in refresh.sources for f in _globs(s.files)
                    )
                    if refresh
                    else None,
                ),
                (
                    "Grid",
                    ", ".join(dict.fromkeys(s.grid for s in refresh.sources if s.grid))
                    if refresh
                    else None,
                ),
            ),
        ),
        (
            "Workplan",
            _rows(
                ("File", wp_path),
                ("Name", wp.get("name")),
                (
                    "Description",
                    _text(wp["description"]) if wp.get("description") else None,
                ),
                ("Step", step.get("name")),
                ("Step blueprint", step.get("blueprint")),
            ),
        ),
    ]
    nesting = _locations(bp.get("nesting_info"))
    part = bp.get("partitioning") if isinstance(bp.get("partitioning"), dict) else {}
    forcing = bp.get("forcing") if isinstance(bp.get("forcing"), dict) else {}
    overrides = bp.get("namelist_overrides")
    valid = (bp.get("valid_start_date"), bp.get("valid_end_date"))
    sections += [
        (
            "Simulation",
            _rows(
                ("Blueprint", bp_path),
                ("Name", bp.get("name")),
                (
                    "Description",
                    _text(bp["description"]) if bp.get("description") else None,
                ),
                ("Run folder", bp.get("working_dir")),
                ("Application", bp.get("application")),
                ("Schema version", bp.get("schema_version")),
                ("Run dates", run_span.replace(" to ", " -> ") if run_span else None),
                ("Valid dates", f"{valid[0]} -> {valid[1]}" if all(valid) else None),
                (
                    "Initial conditions",
                    "\n".join(_condense(_locations(bp.get("initial_conditions")))),
                ),
                ("Grid", "\n".join(_condense(_locations(bp.get("grid"))))),
                ("Nesting", "\n".join(_condense(nesting))),
            ),
        ),
        ("Forcing", _forcing(forcing)),
        ("Code", _code(bp.get("code"))),
        (
            "Parameters",
            _rows(
                (
                    "Namelist overrides",
                    "\n".join(_flatten(overrides))
                    if isinstance(overrides, dict)
                    else None,
                ),
                (
                    "Partitioning",
                    f"{part.get('n_procs_x')} x {part.get('n_procs_y')}"
                    + (f" (use_pio: {part['use_pio']})" if "use_pio" in part else "")
                    if part
                    else None,
                ),
            ),
        ),
        ("Notes", [(str(k), str(v)) for k, v in (info.notes if info else {}).items()]),
    ]
    if not (bp_path or wp_path):
        sections[0][1].append(
            (
                "Simulation",
                "no C-Star files found; set run_info.blueprint in the suite",
            )
        )

    bp_name, bp_desc = bp.get("name"), bp.get("description")
    if info and info.name:
        title = info.name
    elif wp.get("name"):
        title = (
            f"{wp['name']} · {step['name']}" if step.get("name") else str(wp["name"])
        )
    else:
        title = str(bp_name or source or suite.name)
    period = run_span or span
    sub = [
        str(bp_name) if bp_name and str(bp_name) != title else None,
        # an auto-generated description ("Generated blueprint; overridden keys ...")
        # is noise under the title; the Simulation table still lists it
        _text(bp_desc)
        if bp_desc and not _text(bp_desc).lower().startswith("generated blueprint")
        else None,
        f"Simulated {period}" if period else None,
        f"ocean-skill report: {suite.name}",
    ]
    return RunInfo(
        name=title,
        subtitle="\n".join(s for s in sub if s),
        sections=[(h, r) for h, r in sections if r],
        blueprint=bp_path if bp else None,
        workplan=wp_path if wp else None,
    )


# --- drawing -----------------------------------------------------------------------


def _wrap_value(text: str) -> list[str]:
    """Wrap ``text`` line by line; an indented line's continuations stay indented."""
    out: list[str] = []
    for line in text.splitlines() or [""]:
        indent = line[: len(line) - len(line.lstrip())]
        out.extend(
            textwrap.wrap(
                line,
                width=VALUE_WRAP,
                subsequent_indent=indent + "  ",
                break_on_hyphens=False,
            )
            or [""]
        )
    return out


def _records(info: RunInfo) -> list[tuple[str, str, str]]:
    """Every body line as ``(kind, label_part, value_part)``; kind is head/row/gap."""
    recs: list[tuple[str, str, str]] = []
    for heading, items in info.sections:
        recs += [("gap", "", ""), ("head", heading, "")]
        for label, value in items:
            lab, val = _wrap_lines(label, LABEL_WRAP), _wrap_value(value)
            for k in range(max(len(lab), len(val))):
                recs.append(
                    (
                        "row",
                        lab[k] if k < len(lab) else "",
                        val[k] if k < len(val) else "",
                    )
                )
    return recs


def cover_figures(info: RunInfo) -> list:
    """Draw ``info`` as letter-size figures: page 1 titled, later pages continued.

    Built directly (not through pyplot), exactly ``PAGE_W`` x ``PAGE_H``, to be saved
    untrimmed like a section divider. Lines are laid out top to bottom at a fixed
    spacing; the next one that would pass the bottom margin starts a new page.
    """
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    sizes = reference_scale()
    title_pt = sizes["suptitle"] * STEP**TITLE_STEPS
    body_pt = sizes["metrics"]
    line_h = body_pt * LINE_SPACING / 72 / PAGE_H

    def rule(fig: Any, y: float) -> None:
        fig.add_artist(
            Line2D(
                [0.38, 0.62], [y, y], color="0.55", lw=0.8, transform=fig.transFigure
            )
        )

    figs: list[Any] = []

    def new_page() -> tuple[Any, float]:
        fig = Figure(figsize=(PAGE_W, PAGE_H))
        figs.append(fig)
        if len(figs) > 1:
            fig.text(
                MARGIN_X,
                0.95,
                f"{info.name} (continued)",
                fontsize=body_pt,
                color="0.4",
                va="top",
            )
            return fig, TOP
        title = _wrap_lines(info.name, TITLE_WRAP)
        fig.text(
            0.5,
            0.95,
            "\n".join(title),
            ha="center",
            va="top",
            fontsize=title_pt,
            fontweight="bold",
            multialignment="center",
        )
        y = 0.95 - len(title) * title_pt * 1.3 / 72 / PAGE_H - 0.01
        sub = [
            ln
            for s in info.subtitle.splitlines()
            for ln in _wrap_lines(s, SUBTITLE_WRAP)
        ]
        if sub:
            sub_pt = sizes["title"]
            fig.text(
                0.5,
                y,
                "\n".join(sub),
                ha="center",
                va="top",
                fontsize=sub_pt,
                color="0.4",
                multialignment="center",
                linespacing=1.4,
            )
            y -= len(sub) * sub_pt * 1.4 / 72 / PAGE_H + 0.012
        rule(fig, y)
        return fig, y - 0.012

    fig, y = new_page()
    recs = _records(info)
    for kind, label, value in recs:
        # A heading needs room for itself and the row under it; a gap at the top of a
        # fresh page is dropped.
        need = line_h * (3 if kind == "head" else 1)
        if y - need < BOTTOM:
            if kind == "gap":
                continue
            fig, y = new_page()
        elif kind == "gap":
            y -= line_h * 0.7
            continue
        y -= line_h  # text is placed by baseline, so lines keep an even pitch
        if kind == "head":
            fig.text(MARGIN_X, y, label, fontsize=body_pt * STEP, fontweight="bold")
        else:
            fig.text(MARGIN_X, y, label, fontsize=body_pt, color="0.3")
            fig.text(VALUE_X, y, value, fontsize=body_pt, family="DejaVu Sans Mono")
    return figs
