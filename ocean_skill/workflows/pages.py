r"""Expand a :class:`~ocean_skill.config.SuiteConfig` into pages, then draw one.

Two halves, deliberately separate:

:func:`expand` turns the suite's ``pages:`` list into a flat list of plain
``dict``\ s -- one per drawn figure -- doing every bit of resolution that does
*not* need to read real data: ``for_each`` fanned out into its Cartesian product,
``{placeholder}`` strings filled in, the bare word ``latest`` and ``month: run``
resolved against the test source's own time axis (one cheap, coordinate-only read),
a literal ``{"min", "max"}`` time window injected wherever a page would otherwise
mean "whatever the run happens to cover right now", a field page's own ``then:``
chain (see :data:`STEP_REGISTRY`) normalized and type-checked, and each page's
``cache=`` decided against that same index (see :func:`_is_closed`) -- so the
expanded list, written verbatim into a report's ``manifest.json``, is what actually
reproduces the figures. The result is JSON-serializable and takes no arguments a
Phase 2 consumer (a notebook, a dashboard) could not also supply.

:func:`build` draws one already-expanded page and returns a
:class:`matplotlib.figure.Figure` (plus the metric records a ``compare`` page
produced, for the summary page and the metrics CSV). This is the only place that
touches ``osk.field``/``osk.compare``/``osk.summary``, so the suite YAML's own
grammar can change without changing what any of those calls do. A ``field:`` page's
``then:`` steps are applied here too, on the object ``osk.field()`` returned,
before ``.plot()`` -- so a suite page's ``then: [{extremum: min}, {series:
...}]`` runs exactly the Python chain ``osk.field(...).extremum("min").series(...)``
would.

A ``section:`` page is the one kind :func:`build` never draws: it has no figure, just
a title and notes text for a divider page in ``report.pdf``. :func:`expand` templates
both against ``defaults`` (there is no ``for_each``), and the runner hands them
straight to :meth:`~ocean_skill.workflows.report.PdfReport.section`; :func:`build`
refuses one outright.
"""

from __future__ import annotations

import calendar
import copy
import re
import warnings
from dataclasses import dataclass
from dataclasses import field as _dc_field
from itertools import product
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

__all__ = ["STEP_REGISTRY", "MetricRecord", "build", "expand"]

_EXACT_PLACEHOLDER_RE = re.compile(
    r"^\{([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\}$"
)


class MetricRecord:
    """A pooled comparison's metric row, with the aligned data already dropped.

    Stands in for a :class:`~ocean_skill.comparison.Comparison` wherever
    :func:`ocean_skill.comparison.summary` or
    :meth:`~ocean_skill.comparison.ComparisonSet.metrics` need one: both read only
    ``.metrics()`` and ``.label`` (see ``ocean_skill/plot/summary.py::_records``),
    never ``.aligned`` unless it is already present -- so this satisfies them
    without holding the regridded arrays a whole report's worth of comparisons
    would otherwise keep alive at once.
    """

    def __init__(self, record: dict[str, Any], *, label: str, units: str | None = None):
        self._record = dict(record)
        self.label = label
        self.units = units

    def metrics(self, **extra: Any) -> dict[str, Any]:
        return {**self._record, **extra}

    def as_item(
        self,
    ) -> dict[str, Any]:  # pragma: no cover - never called for summary/metrics
        raise NotImplementedError(
            "a pooled metric record has no field data left to draw -- it is only "
            "usable for osk.summary()/ComparisonSet.metrics(), which read "
            ".metrics() and .label, never .as_item()"
        )


# -- then: steps -----------------------------------------------------------------
#
# A field page's ``then:`` runs a small, fixed chain of methods on the object
# ``osk.field()`` built -- the suite-YAML form of a Python chain like
# ``osk.field(...).extremum("min").series(variables=[...])``. Each step is
# registered here with the object type it needs (``accepts``), the type it
# produces (``returns``), a small pydantic model for its keyword arguments (with
# ``extra="forbid"``, so a typo'd kwarg is a schema error, not a silently ignored
# one), and which argument a bare scalar shorthand fills in (``extremum: min`` ->
# ``{"kind": "min"}``). ``expand()`` walks the whole chain against these types
# before any data is read, so a step in the wrong order or a bad argument fails
# ``--list`` rather than showing up as a page silently skipped every run.
#
# Deliberately only ``extremum``/``series`` for now -- see the module docstring on
# :func:`build` for how a step is actually applied. A step that returns a figure
# or writes a file (a future ``movie``, or ``Comparison``'s ``taylor``/``target``/
# ``map_locations``) does not belong here: the page's final output stays a
# page-level key (``plot:`` today), never a step, so ``defaults.plot``/PDF page
# pinning/the comparison per-family split all still have exactly one place to
# apply.

_TYPE_FIELD = "field"  # a single Field (never a multi-member FieldSet)
_TYPE_EXTREMUM = "extremum"  # an Extremum
_TYPE_SERIES = "series"  # a FieldSet drawn as a point series


class _ExtremumArgs(BaseModel):
    """``then: [{extremum: min}]`` / ``then: [{extremum: {kind: min}}]``."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["min", "max"] = "max"


class _SeriesArgs(BaseModel):
    """``then: [..., {series: {variables: [...], time: ..., pad: ..., label: ...}}]``.

    Loosely typed (``Any``) the same way ``field:``/``compare:`` are -- validated
    for real by :meth:`~ocean_skill.extrema.Extremum.series` itself, which already
    accepts everything here. ``pad`` is left ``None`` (rather than defaulting to
    :data:`~ocean_skill.extrema.DEFAULT_PAD_STEPS` here) so :func:`expand` can tell
    "not given" apart from an explicit value when it resolves the window to a
    literal (see ``_resolve_series_window``).
    """

    model_config = ConfigDict(extra="forbid")

    variables: Any = None
    time: Any = None
    pad: int | None = None
    label: str | None = None

    @field_validator("time")
    @classmethod
    def _no_bare_latest(cls, v: Any) -> Any:
        # "latest" only means anything as a *page's* select.time -- expand()
        # resolves it there against the test source's own time index before this
        # step ever sees it (see the field branch of expand()). Left here, it
        # would reach operators.select as a literal, unrecognized string.
        if isinstance(v, str) and v == "latest":
            raise ValueError(
                "time: latest is not supported inside a then: step -- give the "
                "page's own select.time: latest instead (then: series with no "
                "time: already follows it, padded by pad:)"
            )
        return v


@dataclass(frozen=True)
class StepSpec:
    """One entry in :data:`STEP_REGISTRY`."""

    name: str
    accepts: str
    returns: str
    args_model: type[BaseModel]
    #: The keyword argument a bare scalar shorthand (``extremum: min``) fills.
    #: ``None`` means this step takes keyword arguments only.
    shorthand: str | None


STEP_REGISTRY: dict[str, StepSpec] = {
    "extremum": StepSpec(
        "extremum", _TYPE_FIELD, _TYPE_EXTREMUM, _ExtremumArgs, "kind"
    ),
    "series": StepSpec(
        "series", _TYPE_EXTREMUM, _TYPE_SERIES, _SeriesArgs, "variables"
    ),
}


def _normalize_step(raw: Any, *, title: str) -> dict[str, Any]:
    """One ``then:`` list entry -> ``{"name": str, "kwargs": dict}``.

    Accepts a bare name (``"extremum"``), a null-valued single-key mapping (YAML's
    own reading of a bare ``- extremum:`` list item), or ``{name: <scalar or
    dict>}``. Raises :class:`ValueError` naming the page for anything else, an
    unknown step name, or arguments that fail the step's own model.
    """
    if isinstance(raw, str):
        name, value = raw, None
    elif isinstance(raw, dict):
        if len(raw) != 1:
            raise ValueError(
                f"page {title!r}: then: {raw!r} must name exactly one step per "
                "list entry"
            )
        (name, value) = next(iter(raw.items()))
    else:
        raise ValueError(
            f"page {title!r}: then: {raw!r} is not a step name or {{name: args}}"
        )

    spec = STEP_REGISTRY.get(name)
    if spec is None:
        raise ValueError(
            f"page {title!r}: then: {name!r} is not a known step -- choose one "
            f"of {sorted(STEP_REGISTRY)}"
        )

    if value is None:
        kwargs: dict[str, Any] = {}
    elif isinstance(value, dict):
        kwargs = dict(value)
    else:
        if spec.shorthand is None:
            raise ValueError(
                f"page {title!r}: then: {name}: {value!r} -- this step takes "
                f"keyword arguments only, e.g. {name}: {{...}}"
            )
        kwargs = {spec.shorthand: value}

    try:
        validated = spec.args_model.model_validate(kwargs)
    except ValidationError as exc:
        raise ValueError(f"page {title!r}: then: {name}: {exc}") from exc
    return {"name": name, "kwargs": validated.model_dump(exclude_none=True)}


def _check_step_chain(
    steps: list[dict[str, Any]], *, title: str, n_members: int
) -> None:
    r"""Walk ``steps``' declared types; raise if the chain does not fit together.

    ``n_members`` is how many ``Field``\ s this page's ``field:`` would build --
    ``osk.field()`` builds a :class:`~ocean_skill.field.FieldSet` whenever
    ``source``/``variable`` is a list, even a one-element one (see :func:`build`,
    which unwraps that one-element case before applying steps). A chain is
    refused up front for anything wider, since ``extremum`` has no single field to
    start from.
    """
    if not steps:
        return
    if n_members != 1:
        raise ValueError(
            f"page {title!r}: then: needs a single source/variable to start "
            f"from -- this page's field: builds {n_members} members. Narrow "
            "variables:/source: to one value (a one-element list is fine), or "
            "give each its own page."
        )
    current = _TYPE_FIELD
    for step in steps:
        spec = STEP_REGISTRY[step["name"]]
        if spec.accepts != current:
            raise ValueError(
                f"page {title!r}: then: {step['name']!r} needs a {spec.accepts}, "
                f"but the chain has a {current} at that point -- check the step "
                "order"
            )
        current = spec.returns


# -- placeholder namespaces --------------------------------------------------------


class _DepthNS:
    """Wraps one ``for_each: {depth: ...}`` value for ``{depth}``/``{depth.label}``."""

    __slots__ = ("raw",)

    def __init__(self, raw: Any):
        self.raw = raw

    @property
    def label(self) -> str:
        from ocean_skill.comparison import _depth_label

        return _depth_label(self.raw)

    def __str__(self) -> str:
        return str(self.raw)

    def __format__(self, spec: str) -> str:
        return format(str(self.raw), spec)


class _MonthNS:
    """One ``(year, month)`` from ``for_each: {month: run}``."""

    __slots__ = ("month", "year")

    def __init__(self, year: int, month: int):
        self.year = year
        self.month = month

    @property
    def mm(self) -> str:
        return f"{self.month:02d}"

    @property
    def name(self) -> str:
        return calendar.month_name[self.month]

    @property
    def window(self) -> dict[str, str]:
        last = calendar.monthrange(self.year, self.month)[1]
        return {
            "min": f"{self.year:04d}-{self.month:02d}-01T00:00:00",
            "max": f"{self.year:04d}-{self.month:02d}-{last:02d}T23:59:59",
        }

    @property
    def raw(self) -> tuple[int, int]:
        return (self.year, self.month)

    def __str__(self) -> str:
        return f"{self.year:04d}-{self.month:02d}"

    def __format__(self, spec: str) -> str:
        return format(str(self), spec)


def _resolve_path(namespace: dict[str, Any], path: str) -> Any:
    segments = path.split(".")
    obj = namespace[segments[0]]  # KeyError -> caller wraps with the page title
    for seg in segments[1:]:
        obj = getattr(obj, seg)
    if len(segments) == 1 and hasattr(obj, "raw"):
        obj = obj.raw
    return obj


def _template_value(value: Any, namespace: dict[str, Any], *, title: str) -> Any:
    r"""Fill ``{placeholder}``\ s in ``value`` (recursively through dict/list).

    A string that is *exactly* one placeholder (``"{depths}"``, ``"{month.window}"``)
    is replaced by the referenced object itself -- a list or dict keeps its type,
    which ``str.format`` alone cannot do. Anything else with a ``{`` in it goes
    through ordinary ``str.format_map``, which already resolves dotted attribute
    access (``{month.name}``) on its own; a literal brace needs the usual ``{{``.
    """
    if isinstance(value, str):
        m = _EXACT_PLACEHOLDER_RE.match(value)
        if m:
            try:
                return _resolve_path(namespace, m.group(1))
            except (KeyError, AttributeError) as exc:
                raise ValueError(
                    f"page {title!r}: unknown placeholder {value!r} ({exc})"
                ) from exc
        if "{" in value:
            try:
                return value.format_map(namespace)
            except (KeyError, IndexError, AttributeError) as exc:
                raise ValueError(
                    f"page {title!r}: unknown placeholder in {value!r} ({exc})"
                ) from exc
        return value
    if isinstance(value, dict):
        return {k: _template_value(v, namespace, title=title) for k, v in value.items()}
    if isinstance(value, list):
        return [_template_value(v, namespace, title=title) for v in value]
    return value


# -- for_each expansion -------------------------------------------------------------


def _months_in(index: Any, spec: Any) -> list[tuple[int, int]]:
    """Every ``(year, month)`` on ``index``, or a suffix of it.

    ``spec="run"`` -> every calendar month the axis touches, oldest first.
    ``spec={"run": {"last": N}}`` -> just the last ``N`` of those.
    """
    months = sorted({(int(t.year), int(t.month)) for t in index})
    if spec == "run":
        return months
    if (
        isinstance(spec, dict)
        and set(spec) == {"run"}
        and isinstance(spec["run"], dict)
    ):
        last = spec["run"].get("last")
        if last is not None:
            return months[-int(last) :] if last > 0 else []
        return months
    raise ValueError(
        f"for_each: month: {spec!r} is not supported -- use 'run' or "
        "{'run': {'last': N}}"
    )


def _for_each_values(
    name: str, spec: Any, *, defaults: dict[str, Any], test_index: Any
) -> list[Any]:
    """Return the concrete list of raw values one ``for_each`` name iterates over."""
    if name == "month":
        return [_MonthNS(y, m) for y, m in _months_in(test_index, spec)]
    if isinstance(spec, str):
        m = _EXACT_PLACEHOLDER_RE.match(spec)
        if m:
            resolved = _resolve_path(defaults, m.group(1))
            if not isinstance(resolved, list):
                raise ValueError(
                    f"for_each: {name}: {spec!r} must resolve to a list "
                    f"(defaults.{m.group(1)} is {type(resolved).__name__})"
                )
            return resolved
        raise ValueError(f"for_each: {name}: {spec!r} is not a list or a placeholder")
    if isinstance(spec, list):
        return spec
    raise ValueError(
        f"for_each: {name}: {spec!r} must be a list, 'run', or a placeholder"
    )


def _expand_for_each(
    page_title: str,
    for_each: dict[str, Any] | None,
    *,
    defaults: dict[str, Any],
    test_index: Any,
) -> list[dict[str, Any]]:
    """One namespace dict per combination, in YAML key order then product order."""
    if not for_each:
        return [{}]
    collisions = set(for_each) & set(defaults)
    if collisions:
        raise ValueError(
            f"page {page_title!r}: for_each name(s) {sorted(collisions)} collide "
            "with defaults: -- rename one"
        )
    names = list(for_each)
    value_lists = [
        _for_each_values(name, for_each[name], defaults=defaults, test_index=test_index)
        for name in names
    ]
    combos = []
    for combo in product(*value_lists):
        ns: dict[str, Any] = {}
        for name, value in zip(names, combo, strict=True):
            ns[name] = _DepthNS(value) if name == "depth" else value
        combos.append(ns)
    return combos


# -- window injection ------------------------------------------------------------------


def _inject_field_window(kwargs: dict[str, Any], t0: str, t1: str) -> None:
    """Give a ``field`` page's select an explicit time window if it has none.

    Must run *before* the caller resolves a literal ``"latest"`` to an ISO string,
    or that resolved value looks like an ordinary, already-explicit time key here.
    """
    select = kwargs.setdefault("select", {})
    if "time" not in select:
        select["time"] = {"min": t0, "max": t1}


def _inject_compare_window(kwargs: dict[str, Any], t0: str, t1: str) -> None:
    """Do the same for a ``compare`` page's test lane.

    Handles the two shapes the shipped pages actually use: no ``select`` at all
    (a whole-run mean, e.g. the GLODAP page), and a ``{"test": ..., "reference":
    ...}`` pair-spec whose test lane has no time key of its own (everything else
    -- a flat, non-paired select with no time key -- is left alone rather than
    guessed at; see ``docs/suites.md``). Must run *before* the caller resolves a
    literal ``"latest"`` in the test lane, for the same reason as
    :func:`_inject_field_window`.
    """
    select = kwargs.get("select")
    if select is None:
        # A pair-spec select needs both keys even when only one has anything to
        # say -- compare() refuses a select naming just "test" or just "reference".
        kwargs["select"] = {"test": {"time": {"min": t0, "max": t1}}, "reference": {}}
        return
    if "test" in select or "reference" in select:
        # ``select.get(...) or {}`` rather than ``setdefault`` -- a pair-spec
        # naming "test" explicitly as null (``select: {test: null, reference:
        # [...]}}``) leaves the key present with value ``None``, which
        # ``setdefault`` would hand back unchanged, and ``"time" not in None``
        # raises. Written back either way, so a bare ``None`` becomes the ``{}``
        # every other caller of this select already expects.
        test_sel = select.get("test") or {}
        select["test"] = test_sel
        if "time" not in test_sel:
            test_sel["time"] = {"min": t0, "max": t1}


# -- the cache flag: whether a lane's time selection tracks the run's own growth -----
#
# A page's ``cache=`` follows one rule: cache whenever the test lane's time selection
# is either *pinned* to the run's own last step (so the cache key changes the moment
# that step does -- see ``time: latest`` and the whole-run window injected just above)
# or *closed* (see :func:`_is_closed`) -- provably unaffected by the run growing or its
# last step being replaced. Anything else keeps ``cache=False``, exactly as every page
# does today. See "Caching" in ``docs/suites.md``.


def _steps_selected(lane: Any, index: Any) -> Any:
    """Return the timestamps ``lane``'s time key(s) pick out of ``index``, or ``None``.

    Builds a bare 1-D stand-in on ``index`` (named ``"time"``, the spelling every
    shipped select uses) and runs the real :func:`ocean_skill.operators.select`
    against it -- so a period string, a nearest-matched instant, a ``{"min",
    "max"}`` window, and a positional ``{"index": ...}`` are all read exactly as
    they will be when the page is actually drawn, with no separate
    reimplementation of what a time key means. Any other key in ``lane``
    (``depth``, a lon/lat box, ...) silently no-ops on this time-only stand-in,
    the same way :func:`~ocean_skill.operators.select` already treats a key
    naming an axis a given source lacks.

    ``None`` means the selection could not be read at all here (an empty axis, a
    key that raises rather than no-ops) -- the caller treats that as "cannot
    vouch for this lane", not as "it selects nothing".
    """
    import numpy as np
    import xarray as xr

    from ocean_skill import operators

    standin = xr.DataArray(
        np.arange(len(index)), dims=("time",), coords={"time": index}
    )
    try:
        out = operators.select(standin, lane if isinstance(lane, dict) else {})
    except Exception:
        return None
    return np.atleast_1d(out["time"].values)


def _appended_index(index: Any) -> Any:
    """``index`` plus one more step at the end, or ``None`` if it can't be grown.

    A :class:`pandas.DatetimeIndex` grows by one unit of its own ``.resolution``
    word -- the same mapping :func:`ocean_skill.operators._string_instant` uses to
    decide whether a date string names an instant or a period -- so "one more
    step" can never flip that word to something finer than the real data ever
    produces (a plain "smallest gap in the index" step could: a restart's
    ``23:59:59`` stamp already reads as second-resolution, but a day-aligned axis
    appended by one arbitrary sub-day gap elsewhere in the run would flip from
    "day" to whatever that gap was).

    A :class:`~xarray.CFTimeIndex` has no ``.resolution`` to preserve, so it grows
    by its own last gap instead (one day if it has only one step).
    :meth:`~xarray.CFTimeIndex.append` also hands back a plain
    :class:`pandas.Index` for this type, which :func:`~ocean_skill.operators.select`
    cannot use as a datetime axis, so it is rewrapped.
    """
    import pandas as pd
    import xarray as xr

    if isinstance(index, pd.DatetimeIndex):
        step = {
            "day": pd.Timedelta(days=1),
            "hour": pd.Timedelta(hours=1),
            "minute": pd.Timedelta(minutes=1),
            "second": pd.Timedelta(seconds=1),
            "millisecond": pd.Timedelta(milliseconds=1),
            "microsecond": pd.Timedelta(microseconds=1),
            "nanosecond": pd.Timedelta(nanoseconds=1),
        }.get(index.resolution, pd.Timedelta(days=1))
        return index.append(pd.DatetimeIndex([index[-1] + step]))
    if isinstance(index, xr.CFTimeIndex):
        step = index[-1] - index[-2] if len(index) > 1 else pd.Timedelta(days=1)
        return xr.CFTimeIndex([*index, index[-1] + step])
    return None


def _is_closed(lane: Any, index: Any, *, margin: Any = None) -> bool:
    """Whether ``lane``'s time selection is safe to cache against ``index``.

    Closed means: it selects at least one step, none of them the run's current
    last step (``index[-1]``, or within ``margin`` of it -- see below), and it
    would select exactly the same steps if the run had one more (appended past
    the end) or one fewer (its own last step taken away, standing in for that
    step being *replaced*, which is what a restart file still being written does
    under ``keep: latest-per-file``). Checked, not assumed -- a lopsided
    ``{"min", "max"}`` window (or a bug in how one is read) could otherwise pick a
    different step than intended without ever touching the run's actual last
    step, and that would slip past a check that only looked at the current index.

    ``margin`` (a duration, e.g. :class:`pandas.Timedelta`), when given, pushes
    the cutoff back by that much before the last step -- for ``detide=``, whose
    PL33 filter leaves roughly ``margin`` worth of edge NaNs near either end of
    the *unselected* lane that keep changing shape as the run grows, even though
    the filtered value at a step already well clear of the edge cannot change
    once more data lands past it (see :func:`ocean_skill.detide.detide`).
    """
    current = _steps_selected(lane, index)
    if current is None or len(current) == 0:
        return False
    cutoff = index[-1] if margin is None else index[-1] - margin
    if any(t >= cutoff for t in current):
        return False
    grown = _appended_index(index)
    if grown is None:
        return False
    after_grow = _steps_selected(lane, grown)
    after_shrink = _steps_selected(lane, index[:-1])
    return (
        after_grow is not None
        and list(current) == list(after_grow)
        and after_shrink is not None
        and list(current) == list(after_shrink)
    )


def _detide_margin(detide: Any, *, lane: str | None) -> Any:
    """Return one lane's ``detide=`` cutoff, as a ``Timedelta``, or ``None``.

    ``lane`` is ``"test"``/``"reference"`` for a two-lane ``compare:`` page
    (normalized via :func:`ocean_skill.comparison._normalize_detide`), or
    ``None`` for a single-lane ``field:`` page (normalized via
    :func:`ocean_skill.comparison._normalize_detide_side`, the same helper
    :class:`~ocean_skill.field.Field` itself uses for its own ``detide=``).

    ``None`` when the page has no ``detide=`` at all, when the named lane isn't
    detided, and (rather than raising) for a ``detide=`` shape that doesn't
    normalize -- an invalid spec fails the page itself once it is actually drawn;
    this only ever makes the cache flag more conservative, never less.
    """
    if not detide:
        return None
    import pandas as pd

    try:
        if lane is None:
            from ocean_skill.comparison import _normalize_detide_side

            spec = _normalize_detide_side(detide)
        else:
            from ocean_skill.comparison import _normalize_detide

            spec = _normalize_detide(detide).get(lane)
    except Exception:
        return None
    return pd.Timedelta(hours=spec["T"]) if spec else None


# -- semantic checks ------------------------------------------------------------------

#: Variable nicknames that are 2-D (no vertical axis) -- a page selecting a real
#: depth beside one of these is a contradiction ``Field``/``Comparison`` raise on.
SURFACE_ONLY_MARKERS = ("ssh", "co2_flux", "pco2", "ph")


def _is_calculate_spec(spec: Any) -> bool:
    return isinstance(spec, dict) and "calculate" in spec


def _has_real_depth(select: Any) -> bool:
    if not isinstance(select, dict):
        return False
    depth = select.get("depth")
    if depth is None:
        return False
    if isinstance(depth, str) and depth.lower() == "surface":
        return False
    return True


def _check_field_depth_contradiction(title: str, kwargs: dict[str, Any]) -> None:
    if not _has_real_depth(kwargs.get("select")):
        return
    for v in kwargs.get("variable") or []:
        if _is_calculate_spec(v) or (
            isinstance(v, str) and v.lower() in SURFACE_ONLY_MARKERS
        ):
            raise ValueError(
                f"page {title!r}: {v!r} has no vertical axis -- select.depth "
                "must stay surface (or be left out) alongside it"
            )


# -- the public entry points -----------------------------------------------------------


@dataclass
class ExpandedPage:
    """One fully-resolved page: everything ``build`` needs to draw it."""

    title: str
    kind: str  # "field" | "compare" | "summary" | "section"
    kwargs: dict[str, Any]
    plot: dict[str, Any]
    cache: bool
    #: A field page's normalized ``then:`` chain (``[]`` for every other page):
    #: ``[{"name": "extremum", "kwargs": {"kind": "min"}}, ...]``. Fully resolved
    #: by :func:`expand` (placeholders filled, a fixed-snapshot ``series`` window
    #: turned into a literal) -- part of ``as_dict()`` since none of it needs data.
    steps: list[dict[str, Any]] = _dc_field(default_factory=list)
    status: str = "pending"
    reason: str | None = None
    elapsed: float | None = None
    metrics_records: list[dict[str, Any]] = _dc_field(default_factory=list)
    #: What a ``then:`` step actually found once the page was drawn -- e.g. one
    #: record per ``extremum`` step (value, position, snapshot). Data-dependent,
    #: so filled in by :func:`build`, not :func:`expand`; deliberately left out of
    #: ``as_dict()`` (see ``test_expand_is_json_serializable_and_deterministic``)
    #: and added to the manifest separately, the same way ``status``/``reason``
    #: already are.
    results: list[dict[str, Any]] = _dc_field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "kind": self.kind,
            "kwargs": self.kwargs,
            "plot": self.plot,
            "cache": self.cache,
            "steps": self.steps,
        }


def _pin_to_page(kwargs: dict[str, Any], *, seen: set[str]) -> dict[str, Any]:
    """Drop ``zoom``/``size``/``figsize`` and pin the figure to the ``"page"`` canvas.

    A suite's ``report.pdf`` pages are a fixed 8.5x11in (see
    :mod:`ocean_skill.workflows.report`), so a page or its ``defaults.plot`` asking to
    be drawn a different size no longer means anything once it lands there -- the
    figure is pinned regardless, with a warning naming the ignored kwarg. Warned once
    per key across the whole suite (via ``seen``), not once per expanded page, since
    ``defaults.plot: {zoom: ...}`` would otherwise repeat the same warning on every
    page. Only called when ``suite.pdf`` is true; a PNG-only suite keeps its own sizing.
    """
    pinned = dict(kwargs)
    for key in ("zoom", "size", "figsize"):
        if key in pinned:
            value = pinned.pop(key)
            if key not in seen:
                seen.add(key)
                warnings.warn(
                    f"{key}={value!r} ignored: report.pdf pages are a fixed "
                    "8.5x11in; set pdf: false to size figures freely",
                    stacklevel=2,
                )
    pinned["size"] = "page"
    return pinned


def _field_member_count(kwargs: dict[str, Any]) -> int:
    r"""How many ``Field``\ s this page's ``source``/``variable`` would build.

    Mirrors :func:`ocean_skill.field.field`'s own fan-out rule (a list on either
    side builds one member per (source, variable) pair) without importing it --
    ``expand()`` stays a data-free, dependency-light pass. Variable-name dedup
    (aliases of the same canonical name collapse to one member) can only make the
    real count *smaller* than this, never larger, so a chain this refuses would
    never have silently produced the wrong thing -- it may in rare alias cases
    refuse a chain that would have worked; narrowing ``variables:`` to one
    resolves it either way.
    """
    source = kwargs.get("source")
    variable = kwargs.get("variable")
    n_sources = len(source) if isinstance(source, list) else 1
    n_variables = len(variable) if isinstance(variable, list) else 1
    return n_sources * n_variables


def _has_any_key(select: Any, keys: Any) -> bool:
    return isinstance(select, dict) and any(k in select for k in keys)


def _resolve_series_window(
    steps: list[dict[str, Any]],
    *,
    select: dict[str, Any],
    index: Any,
    suite_cache: bool,
    extrema: Any,
) -> None:
    """Turn a ``series`` step's implicit pad window (and cache flag) into a literal.

    Done in place, whenever the page's own snapshot is a single fixed instant --
    so the manifest reproduces the same figure regardless of how much the run has
    grown by the time it is replayed.

    Left alone (:meth:`~ocean_skill.extrema.Extremum.series`'s own runtime
    default, still read lazily at build time) when the page's ``time`` is
    instead a range: the extremum's own landing time is then only known once the
    data is actually read, so there is nothing to resolve here without one.
    Likewise left alone when the step itself already names ``time=`` -- an
    explicit window is used verbatim, never padded.
    """
    time_value = select.get("time")
    if not isinstance(time_value, str):
        return
    for step in steps:
        if step["name"] != "series":
            continue
        kwargs = step["kwargs"]
        if "time" in kwargs:
            continue
        pad = kwargs.get("pad", extrema.DEFAULT_PAD_STEPS)
        window = extrema._window_select(index, time_value, pad)
        kwargs["time"] = window
        reaches_latest = window["max"] == str(index[-1])
        kwargs["cache"] = suite_cache and not reaches_latest


def _check_extremum_guardrails(
    steps: list[dict[str, Any]],
    *,
    title: str,
    select: dict[str, Any],
    aggregate: Any,
    had_explicit_time: bool,
    time_keys: Any,
    vertical_keys: Any,
) -> None:
    """Refuse an ``extremum`` step on a page that would read too much, too vaguely.

    That means either the whole (and ever-growing) run, with no explicit time
    named, or a column ``.plot()`` would have shown as just the surface. Both
    checks run on the page's own select *before* :func:`_inject_field_window`
    -- a page this refuses would otherwise have had a whole-run window injected
    for it, since that injection only fires when the page names no time of its
    own, exactly the case this guards against.
    """
    if not any(s["name"] == "extremum" for s in steps):
        return
    aggregate_collapses_time = isinstance(aggregate, dict) and any(
        k in aggregate for k in time_keys
    )
    if not had_explicit_time and not aggregate_collapses_time:
        raise ValueError(
            f"page {title!r}: then: extremum needs an explicit select.time (a "
            "literal instant, a range, or 'latest') or an aggregate.time that "
            "collapses it -- without one the search would load the whole, "
            "ever-growing run into memory. Add one to select= (or aggregate=) "
            "first."
        )
    if not _has_any_key(select, vertical_keys):
        raise ValueError(
            f"page {title!r}: then: extremum needs an explicit vertical key "
            "(select.depth/Z/z/vertical/sigma0) -- .plot() defaults a bare "
            "column to the surface, but extremum() searches whatever vertical "
            "axis is left standing, which can report a different level than "
            "the map you would otherwise see."
        )


def expand(suite: Any) -> list[ExpandedPage]:
    """Turn ``suite.pages`` into a flat, fully-resolved list of :class:`ExpandedPage`.

    ``suite`` is a :class:`~ocean_skill.config.SuiteConfig`. Reads the test
    source's native time index at most once (coordinate-only; memoized here), used
    to resolve ``latest``/``month: run`` and to bound the literal windows injected
    into any page that would otherwise mean "however much of the run exists right
    now". That same index also decides each page's ``cache=`` (see
    :func:`_is_closed`), against whatever it resolved to on *this* call --
    a page's cache entry is only ever as fresh as the last time the suite ran.
    """
    from ocean_skill import extrema
    from ocean_skill.comparison import _ANY_VERTICAL_KEYS
    from ocean_skill.sources import _TIME_KEYS

    defaults = dict(suite.defaults)
    test_source = defaults.get("test")

    index_cache: dict[str, Any] = {}
    pin_seen: set[str] = set()

    def get_index(source: str) -> Any:
        if source not in index_cache:
            index_cache[source] = extrema._native_time_index(source)
        return index_cache[source]

    out: list[ExpandedPage] = []
    for page in suite.pages:
        if page.kind == "section":
            # Handled before everything the other kinds share: a divider has no
            # source (so no time index to read), no plot (so ``defaults.plot`` must
            # not be merged in, and ``_pin_to_page`` must not warn about a
            # ``defaults.plot: {zoom: ...}`` it will never apply to), and no
            # ``for_each`` (the schema already refused one). Title and notes are
            # templated against ``defaults`` alone.
            title = _template_value(page.title, defaults, title=page.title)
            text = _template_value(page.section, defaults, title=title)
            if not isinstance(text, str):
                raise ValueError(
                    f"page {title!r}: section: must resolve to text, got "
                    f"{type(text).__name__} ({page.section!r})"
                )
            out.append(
                ExpandedPage(
                    title=title,
                    kind="section",
                    kwargs={"text": text},
                    plot={},
                    cache=False,
                )
            )
            continue

        page_source = None
        if page.kind == "field":
            page_source = (page.field or {}).get("source", test_source)
        elif page.kind == "compare":
            page_source = (page.compare or {}).get("test", test_source)

        test_index = get_index(page_source) if page.for_each and page_source else None
        combos = _expand_for_each(
            page.title, page.for_each, defaults=defaults, test_index=test_index
        )

        plot_defaults = defaults.get("plot", {})

        for combo in combos:
            namespace = {**defaults, **combo}
            title = _template_value(page.title, namespace, title=page.title)
            plot = {
                **plot_defaults,
                **_template_value(page.plot, namespace, title=title),
            }
            if suite.pdf:
                plot = _pin_to_page(plot, seen=pin_seen)

            if page.kind == "field":
                # Deep-copied so an exact-placeholder select (``select: "{sel}"``,
                # resolved by ``_template_value`` to the referenced object itself
                # -- see its own docstring) is never the *same* dict as whatever
                # ``defaults``/another page's combo holds; the "latest" resolution
                # and the window injection below both write into ``select`` in
                # place, and without this a shared placeholder would leak one
                # page's resolved time into every other page that names it.
                kwargs = copy.deepcopy(
                    _template_value(dict(page.field), namespace, title=title)
                )
                kwargs.setdefault("source", test_source)
                source = kwargs["source"]
                if source is None:
                    raise ValueError(f"page {title!r}: no source (set defaults.test)")
                if "variables" in kwargs:
                    kwargs["variable"] = kwargs.pop("variables")
                _check_field_depth_contradiction(title, kwargs)
                index = get_index(source)
                t0, t1 = index[0].isoformat(), index[-1].isoformat()
                select = kwargs.setdefault("select", {})
                had_explicit_time = "time" in select
                was_latest = select.get("time") == "latest"
                pinned = was_latest or not had_explicit_time
                _inject_field_window(kwargs, t0, t1)
                if was_latest:
                    select["time"] = t1
                cacheable = pinned or _is_closed(
                    select,
                    index,
                    margin=_detide_margin(kwargs.get("detide"), lane=None),
                )

                steps = [
                    _normalize_step(raw, title=title)
                    for raw in _template_value(page.then or [], namespace, title=title)
                ]
                _check_step_chain(
                    steps, title=title, n_members=_field_member_count(kwargs)
                )
                _check_extremum_guardrails(
                    steps,
                    title=title,
                    select=select,
                    aggregate=kwargs.get("aggregate"),
                    had_explicit_time=had_explicit_time,
                    time_keys=_TIME_KEYS,
                    vertical_keys=_ANY_VERTICAL_KEYS,
                )
                _resolve_series_window(
                    steps,
                    select=select,
                    index=index,
                    suite_cache=suite.cache,
                    extrema=extrema,
                )

                out.append(
                    ExpandedPage(
                        title=title,
                        kind="field",
                        kwargs=kwargs,
                        plot=plot,
                        cache=suite.cache and cacheable,
                        steps=steps,
                    )
                )

            elif page.kind == "compare":
                # See the field branch above for why this is deep-copied.
                kwargs = copy.deepcopy(
                    _template_value(dict(page.compare), namespace, title=title)
                )
                kwargs.setdefault("test", test_source)
                source = kwargs["test"]
                if source is None:
                    raise ValueError(
                        f"page {title!r}: no test source (set defaults.test)"
                    )
                index = get_index(source)
                t0, t1 = index[0].isoformat(), index[-1].isoformat()
                select = kwargs.get("select")
                is_pair_spec = isinstance(select, dict) and (
                    "test" in select or "reference" in select
                )
                detide_margin = _detide_margin(kwargs.get("detide"), lane="test")
                if select is None or is_pair_spec:
                    was_latest = is_pair_spec and (
                        isinstance(select.get("test"), dict)
                        and select["test"].get("time") == "latest"
                    )
                    pinned = (
                        select is None
                        or was_latest
                        or "time" not in (select.get("test") or {})
                    )
                    _inject_compare_window(kwargs, t0, t1)
                    if was_latest:
                        kwargs["select"]["test"]["time"] = t1
                    test_lane = kwargs["select"]["test"]
                    cacheable = pinned or _is_closed(
                        test_lane, index, margin=detide_margin
                    )
                else:
                    # A flat, non-paired select applies to both lanes at once --
                    # compare()'s own contract -- so rewriting only the "test"
                    # side here would silently change what the reference reads
                    # too. Never rewritten, only checked.
                    cacheable = _is_closed(select, index, margin=detide_margin)
                if kwargs.get("times") is not None:
                    # times= fans this one page into several per-bin comparisons,
                    # each replacing whatever time entry select carried with its
                    # own bin value at draw time (comparison._fanned_time_select)
                    # -- what actually gets keyed is not what was just resolved
                    # above, so nothing here can vouch for it.
                    cacheable = False
                out.append(
                    ExpandedPage(
                        title=title,
                        kind="compare",
                        kwargs=kwargs,
                        plot=plot,
                        cache=suite.cache and cacheable,
                    )
                )

            else:  # summary
                kwargs = _template_value(dict(page.summary), namespace, title=title)
                if suite.pdf:
                    kwargs = _pin_to_page(kwargs, seen=pin_seen)
                out.append(
                    ExpandedPage(
                        title=title,
                        kind="summary",
                        kwargs=kwargs,
                        plot=plot,
                        cache=suite.cache,
                    )
                )
    return out


def _extremum_record(ext: Any) -> dict[str, Any]:
    """Build a small, JSON-safe record of what an ``extremum`` step found.

    Written into the manifest as :attr:`ExpandedPage.results` (see :func:`build`).
    """
    return {
        "kind": ext.kind,
        "variable": ext.variable,
        "standard_name": ext.standard_name,
        "value": ext.value,
        "units": ext.units,
        "lon": ext.lon,
        "lat": ext.lat,
        "lon_convention": ext.lon_convention,
        "indices": ext.indices,
        "coords": ext.coords,
        "time": str(ext.time) if ext.time is not None else None,
        "time_reason": ext.time_reason,
    }


def build(page: ExpandedPage, *, pooled_records: list[MetricRecord] | None = None):
    """Draw one expanded page. Returns a list of ``(suffix, Figure)`` pairs.

    A ``section`` page has nothing to draw and raises :class:`ValueError` -- see the
    module docstring.

    Almost always one pair (``suffix=""``); a ``compare`` page whose comparisons
    span more than one plot family draws one figure per family instead (see
    :meth:`ocean_skill.comparison.ComparisonSet.plot`), each suffixed by its
    family name so no PNG is overwritten. A ``field`` page's ``select=``/
    ``aggregate=`` (and a ``compare`` page's own kwargs) are used exactly as
    :func:`ocean_skill.field.field`/:func:`ocean_skill.comparison.compare` already
    validate them -- this function adds nothing to that grammar. A ``field:``
    page's own ``qc``/``detide``/``label`` (or anything else :func:`~ocean_skill
    .field.field` accepts) pass straight through the same way; only ``cache`` is
    reserved, since that is the suite's own to set (``cache:``/``cache_dir:``, or
    a page's own open-window rule -- see ``docs/suites.md``).

    A field page's ``then:`` chain (already normalized and type-checked by
    :func:`expand`) is applied here, in order, via plain ``getattr`` -- one
    ``osk.field(...).extremum(...).series(...)`` for each step. ``source``/
    ``variable`` builds a one-member :class:`~ocean_skill.field.FieldSet`
    whenever either is a list, even a one-element one (see :func:`~ocean_skill
    .field.field`'s own docstring); a page with steps unwraps that single member
    first, since a step like ``extremum`` runs on one :class:`~ocean_skill.field
    .Field`, not a set. Whichever step returns an
    :class:`~ocean_skill.extrema.Extremum` is also printed (so it lands in
    ``run.log``) and recorded onto :attr:`ExpandedPage.results` (so it lands in
    ``manifest.json`` too) -- the one place in this chain that is inherently
    data-dependent and so cannot have been resolved by :func:`expand`.
    """
    if page.kind == "section":
        raise ValueError(
            f"page {page.title!r}: a section: page is a divider in report.pdf, not "
            "a figure -- the runner hands it to PdfReport.section(), never build()"
        )

    import ocean_skill as osk

    if page.kind == "field":
        if "cache" in page.kwargs:
            raise ValueError(
                f"page {page.title!r}: field: cache: is not supported -- "
                "caching is controlled by the suite's own cache:/cache_dir: "
                "settings, not a per-page kwarg"
            )
        field_kwargs = {
            k: v for k, v in page.kwargs.items() if k not in ("source", "variable")
        }
        obj = osk.field(
            page.kwargs["source"],
            page.kwargs["variable"],
            cache=page.cache,
            **field_kwargs,
        )
        if page.steps:
            if isinstance(obj, osk.FieldSet):
                # expand()'s _check_step_chain already refused anything wider
                # than one member -- this is that one member, unwrapped.
                obj = obj[0]
            for step in page.steps:
                method = getattr(obj, step["name"], None)
                if method is None:
                    raise ValueError(
                        f"page {page.title!r}: then: {step['name']} step "
                        f"produced a {type(obj).__name__}, which has no "
                        f".{step['name']}() -- check the step order"
                    )
                obj = method(**step["kwargs"])
                if isinstance(obj, osk.Extremum):
                    print(repr(obj))
                    page.results.append(_extremum_record(obj))
        fig = obj.plot(**page.plot)
        return [("", fig)]

    if page.kind == "compare":
        compare_kwargs = {k: v for k, v in page.kwargs.items() if k not in ("test",)}
        cs = osk.compare(
            test=page.kwargs["test"],
            skip_missing=True,
            cache=page.cache,
            **compare_kwargs,
        )
        if len(cs) == 0:
            raise ValueError("every comparison in this page was skipped (skip_missing)")
        page.metrics_records = [c.metrics() for c in cs]
        families = sorted({c.family for c in cs})
        results = []
        for fam in families:
            bucket = osk.ComparisonSet([c for c in cs if c.family == fam])
            fig = bucket.plot(**page.plot)
            suffix = "" if len(families) == 1 else f"_{fam}"
            results.append((suffix, fig))
        return results

    # summary
    pooled = pooled_records or []
    if not pooled:
        raise ValueError("no compare page produced results to summarize")
    comparisons = [
        MetricRecord(
            r, label=r.get("label") or r.get("variable", ""), units=r.get("units")
        )
        for r in pooled
    ]
    fig = osk.summary(comparisons, **page.kwargs)
    return [("", fig)]
