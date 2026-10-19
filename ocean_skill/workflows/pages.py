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

An ``XY:``/``TS:`` page is a ``members:`` mapping of label -> one ``osk.field()``
call each, drawn together by :class:`~ocean_skill.xy.XY` (``TS:`` is the salinity-
temperature preset). :func:`expand` gives the member that reads the test source the
same pinned run window a ``field:`` page gets (:func:`_pin_test_lane`); every other
member -- a climatology, say -- is passed through as written, since its time axis is
not the run's.

A ``section:`` page is the one kind :func:`build` never draws: it has no figure, just
a title and notes text for a divider page in ``report.pdf``. :func:`expand` templates
both against ``defaults`` (there is no ``for_each``), and the runner hands them
straight to :meth:`~ocean_skill.workflows.report.PdfReport.section`; :func:`build`
refuses one outright.

A ``health:``/``rivers:`` page draws the figures of a run-diagnostic object
(:func:`ocean_skill.run_health.health`, :func:`ocean_skill.river_inputs.rivers`).
:func:`expand` fills in its ``source`` and pins its time window like a ``field:``
page's; :func:`build` computes the object once per (source, time window) and hands
every later page with the same key the same one (see :func:`_diagnostic_object`).

An ``inputs:`` page draws the integrity table of :func:`ocean_skill.input_health` over
one or more input-file sources (``sources:``, a list of catalog names or ``{name, label,
vars}`` mappings; placeholders expand to nested lists, which :func:`expand` flattens).
Its ``time`` is passed as written: a forcing file's time axis is not the run's.
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

from ocean_skill.comparison import (
    _depth_label,
    _merged_select,
    _time_label,
    aggregate_for,
    is_pair_spec,
    names_transect,
    qc_for,
    select_for,
    selects_entry_parts,
    transect_pair_problem,
    variable_for,
)
from ocean_skill.config import (
    COMPARE_DRAWS,
    check_diagnostic_draw,
    check_draw,
    summary_kinds,
)
from ocean_skill.plot.spec import UnusableOptions
from ocean_skill.table import _collapse_range

__all__ = [
    "COMPARE_DRAWS",
    "STEP_REGISTRY",
    "MetricRecord",
    "PageSkipped",
    "build",
    "expand",
    "self_check_page",
    "self_check_problem",
]

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
        from ocean_skill.comparison import _canonical

        self._record = dict(record)
        self.label = label
        self.units = units
        # What pooling de-duplicates on: the whole row. Two rows are the same
        # comparison only if every field -- pair, variable, depth, time and every
        # metric -- matches, which is what drawing one page twice would produce.
        self.identity = ("metric-record", _canonical(self._record))
        # What the summary's point labels are read from (comparison._LABEL_DIMS), off
        # the row's own columns.
        self.variable = record.get("variable")
        self.test_name = record.get("test")
        self.reference_name = record.get("reference")
        self.select = {
            k: record[k] for k in ("depth", "time") if record.get(k) is not None
        }

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


class _Item(dict):
    """A ``for_each`` item that is a mapping, its keys also read as attributes.

    So ``{group.name}`` fills a title (through ``str.format_map``) and
    ``"{group.variables}"`` a value, the same way a depth's or a month's attributes do.
    """

    def __getattribute__(self, name: str) -> Any:
        # A key wins over a dict method of the same name (``values``, ``items``, ``get``
        # ...): ``{group.values}`` must read the mapping's own entry, never the bound
        # method. Dunder lookups stay ordinary, and the dict's own C-level methods never
        # go through here, so dict behaviour is unchanged.
        if not name.startswith("__") and dict.__contains__(self, name):
            return dict.__getitem__(self, name)
        return super().__getattribute__(name)


def _resolve_path(namespace: dict[str, Any], path: str) -> Any:
    segments = path.split(".")
    obj = namespace[segments[0]]  # KeyError -> caller wraps with the page title
    for seg in segments[1:]:
        obj = getattr(obj, seg)
    if isinstance(obj, _Item):
        # the whole mapping as a value: a plain dict for pydantic/compare/json, which
        # may call ``.items()`` on it even when the mapping has a key of that name
        return dict(obj)
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


# -- sources whose time axis cannot pin a window ---------------------------------------
#
# ``expand()`` reads each test source's native time index to inject a run window. Two
# things can go wrong there without the suite being wrong, and neither may abort the
# whole expansion: a source with *no time axis* (a monthly climatology on a ``month``
# dimension) is static data -- its pages get no window and are cacheable; a source that
# *cannot be read at all* (its catalog entry is missing, opening it raises) leaves its
# pages unwindowed, uncacheable and flagged with ``ExpandedPage.problem`` so the run
# can list them and skip just those. The "no time axis" case is told apart from a real
# read failure by looking at the opened source (``_lacks_time_axis``), never by parsing
# the error text.


class _SourceUnreadable(Exception):
    """A page's source could not be read at expand time (the message says why)."""

    def __init__(self, source: str, cause: BaseException) -> None:
        super().__init__(
            f"source {source!r} could not be read at expand time: "
            f"{type(cause).__name__}: {cause}"
        )
        self.source = source


def _lacks_time_axis(source: str) -> bool:
    """Whether ``source`` opens fine but has no time axis at all (a climatology).

    Checked on the opened object itself so any other failure -- a missing catalog
    entry, an open error -- stays a failure. A time axis that exists but carries no
    values is a failure too, not "timeless": that is a run whose time did not decode,
    and treating it as static would cache pages that go stale as it grows.
    """
    try:
        from ocean_skill import operators
        from ocean_skill.sources import read

        obj = read(source)
        dim = operators.resolve_dim(obj, "time")
        return dim is None
    except Exception:
        return False


# -- for_each expansion -------------------------------------------------------------


def _thin(values: list[Any]) -> list[Any]:
    """Return the first and last of ``values`` (all of them when two or fewer).

    Test mode (``ocean-skill-run --smoke``) keeps the ends of a date list: the first and
    last dates exercise every code path the middle ones do, at a fraction of the reads.
    """
    return [values[0], values[-1]] if len(values) > 2 else list(values)


#: ``min_pairs`` a compare page gets in test mode unless it sets its own. A test run
#: keeps only the first and last few steps, so a window that holds fewer than the real
#: default's valid pairs (``comparison.DEFAULT_MIN_PAIRS``) would mask every cell of a
#: per-cell skill map and read as NaN lanes; the threshold is what the data can reach,
#: not part of what the page exercises.
TEST_MIN_PAIRS = 1


def _thin_times(kwargs: dict[str, Any]) -> str | None:
    """Cut a compare page's ``times:`` list to its first and last date, in place.

    Returns the note for :attr:`ExpandedPage.thinned` (``"2 of 16 dates"``), or ``None``
    when nothing was cut: no ``times:``, a resample/bins mapping (it bins whatever is
    kept) or a list of two or fewer. The list is replaced, never edited -- it may be the
    suite's own ``defaults`` list, shared by other pages.
    """
    times = kwargs.get("times")
    if not isinstance(times, list) or len(times) <= 2:
        return None
    kwargs["times"] = _thin(times)
    return f"{len(kwargs['times'])} of {len(times)} dates"


def _months_in(index: Any, spec: Any, *, thin: bool = False) -> list[tuple[int, int]]:
    """Every ``(year, month)`` on ``index``, or a suffix of it.

    ``spec="run"`` -> every calendar month the axis touches, oldest first.
    ``spec={"run": {"last": N}}`` -> just the last ``N`` of those.
    ``thin`` keeps only the first and last of whichever months that gives (test mode).
    """
    months = _months_in_all(index, spec)
    return _thin(months) if thin else months


def _months_in_all(index: Any, spec: Any) -> list[tuple[int, int]]:
    """:func:`_months_in` without thinning."""
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
    name: str,
    spec: Any,
    *,
    defaults: dict[str, Any],
    test_index: Any,
    index_error: Exception | None = None,
    thin_months: bool = False,
) -> list[Any]:
    """Return the concrete list of raw values one ``for_each`` name iterates over.

    ``index_error`` is why ``test_index`` is missing (a source with no time axis, or
    one that could not be read): raised only when ``month: run`` needs the index, so
    every other ``for_each`` name still expands. ``thin_months`` keeps only the first
    and last month of a ``month`` fan (test mode); every other name stays complete.
    """
    if name == "month":
        if index_error is not None:
            raise index_error
        return [
            _MonthNS(y, m) for y, m in _months_in(test_index, spec, thin=thin_months)
        ]
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
    index_error: Exception | None = None,
    thin_months: bool = False,
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
        _for_each_values(
            name,
            for_each[name],
            defaults=defaults,
            test_index=test_index,
            index_error=index_error,
            thin_months=thin_months,
        )
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
    if is_pair_spec(select):
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


def _pin_test_lane(
    kwargs: dict[str, Any], index: Any
) -> tuple[dict[str, Any], bool, bool]:
    """Pin one ``field()``-style kwargs dict's time selection to the run, in place.

    Shared by a ``field:`` page and an ``XY:``/``TS:`` page's test-source member.
    Gives ``kwargs["select"]`` the whole-run window if it names no ``time`` (before
    resolving a literal ``"latest"`` to the run's last step, or that value would look
    like an explicit time key here) and decides whether the result is safe to cache
    (see the section header above). Returns ``(select, had_explicit_time,
    cacheable)``: ``select`` is the (now windowed) dict, ``had_explicit_time`` whether
    the page named a time at all -- ``"latest"`` counts -- and ``cacheable`` that
    verdict, before the suite-wide ``cache:`` switch is applied.

    ``index`` is ``None`` for a source with no time axis (static data): nothing is
    injected and the result is cacheable.
    """
    if index is None:
        select = kwargs.setdefault("select", {})
        return select, "time" in select, True
    t0, t1 = index[0].isoformat(), index[-1].isoformat()
    select = kwargs.setdefault("select", {})
    had_explicit_time = "time" in select
    was_latest = select.get("time") == "latest"
    pinned = was_latest or not had_explicit_time
    _inject_field_window(kwargs, t0, t1)
    if was_latest:
        select["time"] = t1
    cacheable = pinned or _is_closed(
        select, index, margin=_detide_margin(kwargs.get("detide"), lane=None)
    )
    return select, had_explicit_time, cacheable


# -- XY: / TS: pages ------------------------------------------------------------------
#
# ``members`` maps a label (the legend entry) to one ``field()`` call's kwargs, with
# ``source`` defaulting to ``defaults.test`` and ``variables`` to ``[y, x]``. Only the
# page's own shape is checked here; whether the members really hold x and y, and
# whether each region box lands on data, is XY's to say once it reads anything. A
# member may also carry ``window: run`` (a suite key, popped before ``field()`` sees
# the spec) to take the test run's span as its own ``select.time``.

#: The salinity-temperature preset ``TS:`` stands for.
_TS_X, _TS_Y = "salinity", "temperature"

#: ``field()`` keywords a member may not set: the suite owns ``cache`` (as on a
#: ``field:`` page) and the member's key is its label.
_XY_RESERVED_MEMBER_KEYS = ("cache", "label")


class _XYArgs(BaseModel):
    """An ``XY:``/``TS:`` page's mapping: ``members``, ``x``/``y``, ``regions``..."""

    model_config = ConfigDict(extra="forbid")

    members: dict[str, dict[str, Any] | None]
    x: Any = None
    y: Any = None
    regions: dict[str, dict[str, Any]] | None = None
    at_center: list[str] = []

    @field_validator("members")
    @classmethod
    def _members_not_empty(cls, v: dict[str, Any]) -> dict[str, Any]:
        if not v:
            raise ValueError("members: needs at least one entry")
        return v


def _check_xy_args(raw: dict[str, Any], *, kind: str, title: str) -> _XYArgs:
    """Validate one ``XY:``/``TS:`` page's mapping; ``x``/``y`` per ``kind``."""
    try:
        args = _XYArgs.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"page {title!r}: {kind}: {exc}") from exc
    given = args.model_fields_set
    if kind == "TS":
        if given & {"x", "y"}:
            raise ValueError(
                f"page {title!r}: TS: sets x=salinity and y=temperature itself -- "
                "use XY: for any other pair"
            )
    else:
        missing = [k for k in ("x", "y") if getattr(args, k) is None]
        if missing:
            raise ValueError(
                f"page {title!r}: XY: needs x: and y: (the variables to plot "
                f"against each other) -- missing {missing}; TS: is the "
                "salinity-temperature shorthand"
            )
    return args


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
    #: "field" | "compare" | "summary" | "section" | "XY" | "TS" | "health" | "rivers"
    #: | "inputs"
    kind: str
    kwargs: dict[str, Any]
    plot: dict[str, Any]
    cache: bool
    #: A field page's normalized ``then:`` chain (``[]`` for every other page):
    #: ``[{"name": "extremum", "kwargs": {"kind": "min"}}, ...]``. Fully resolved
    #: by :func:`expand` (placeholders filled, a fixed-snapshot ``series`` window
    #: turned into a literal) -- part of ``as_dict()`` since none of it needs data.
    steps: list[dict[str, Any]] = _dc_field(default_factory=list)
    #: A field/compare page's contour overlay (``False`` for every other page, or
    #: none asked for): ``True`` contours the plotted object itself; a mapping is
    #: the *fully resolved* kwargs of a second ``field:``/``compare:`` call (the
    #: page's own kwargs with the page's ``contours:`` overrides merged over them,
    #: run through the same window injection) that :func:`build` draws the lines
    #: from. Part of ``as_dict()`` since none of it needs data.
    contours: bool | dict[str, Any] = False
    #: A compare page's drawing method (one of :data:`COMPARE_DRAWS`; ``"plot"``
    #: unless the page says otherwise), ``None`` for every other kind of page.
    draw: str | None = None
    #: Text drawn under the page's (first) figure in ``report.pdf`` and recorded in the
    #: manifest (``None`` for none). Templated like ``title:``; part of ``as_dict()``
    #: only when set, so every other page's dict is unchanged.
    caption: str | None = None
    #: The ``plot:`` keys that came from ``defaults.plot`` and not from the page itself.
    #: A page whose draw method does not accept one of them drops it with a warning
    #: instead of failing (:func:`_draw`), so a suite-wide ``defaults.plot: {robust:
    #: true}`` need not suit every kind of page. Not part of ``as_dict()``.
    default_plot: tuple[str, ...] = ()
    #: Why this page cannot be drawn, found at expand time -- a source that could not
    #: be read (``"source 'x' could not be read at expand time: KeyError: ..."``).
    #: ``None`` for a healthy page. Such a page is still emitted (unwindowed, never
    #: cached) so the run can list it and skip it with this reason. Part of
    #: ``as_dict()`` only when set, so every other page's dict is unchanged.
    problem: str | None = None
    #: What test mode (``ocean-skill-run --smoke``) cut from this page's dates, as text
    #: for the page list (``"test: 2 of 16 dates"``); ``None`` for a page left whole.
    #: Not part of ``as_dict()``.
    thinned: str | None = None
    status: str = "pending"
    reason: str | None = None
    elapsed: float | None = None
    #: Peak resident memory of the process while the page was drawn, in MB (``None``
    #: until drawn). Written by the runner into the manifest, not part of ``as_dict()``.
    peak_mb: float | None = None
    #: Comparisons :func:`build` left out of a compare page because their test or
    #: reference data was entirely NaN, one reason each (the page itself still drew
    #: from the rest). Empty for every other page.
    dropped: list[str] = _dc_field(default_factory=list)
    metrics_records: list[dict[str, Any]] = _dc_field(default_factory=list)
    #: A summary page's table of pooled metrics, as the frame written beside
    #: ``metrics.csv`` (``None`` unless the page draws ``kind: table``).
    summary_table: Any = None
    #: What a ``then:`` step actually found once the page was drawn -- e.g. one
    #: record per ``extremum`` step (value, position, snapshot). Data-dependent,
    #: so filled in by :func:`build`, not :func:`expand`; deliberately left out of
    #: ``as_dict()`` (see ``test_expand_is_json_serializable_and_deterministic``)
    #: and added to the manifest separately, the same way ``status``/``reason``
    #: already are.
    results: list[dict[str, Any]] = _dc_field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        out = {
            "title": self.title,
            "kind": self.kind,
            "kwargs": self.kwargs,
            "plot": self.plot,
            "cache": self.cache,
            "steps": self.steps,
            "contours": self.contours,
            "draw": self.draw,
        }
        if self.problem is not None:
            out["problem"] = self.problem
        if self.caption is not None:
            out["caption"] = self.caption
        return out


#: Plot families that stack one row per comparison, so a long set can be cut into
#: figures of ``max_rows`` rows each (:func:`_row_chunks`). The others -- series,
#: profiles -- draw every comparison on one panel, which cutting would only scatter.
ROW_FAMILIES = ("field_row", "time_depth", "section_row", "skill_map")

#: A ``compare:`` page's ``plot: {max_rows: ...}`` when it names none.
DEFAULT_MAX_ROWS = 4

#: How many comparisons a summary page's metrics table draws per figure (``max_rows:``
#: on the ``summary:``): about what one Letter page holds at a readable type size.
DEFAULT_TABLE_ROWS = 25


def _max_rows_key(
    plot: dict[str, Any],
    own_plot: dict[str, Any],
    *,
    applies: bool,
    title: str,
) -> dict[str, Any]:
    """Validate ``plot``'s ``max_rows`` (kept in it; :func:`build` pops it).

    ``max_rows`` is the page-level cap of :func:`build`, not a ``.plot()`` option, so it
    must not reach a draw method that would refuse it. A page whose draw it does not
    steer (``applies`` false: not a ``compare:`` page, or one with a ``draw:`` other
    than ``plot``) refuses it when the page wrote it and drops it silently when it
    only came from ``defaults.plot`` -- a suite-wide cap suits the pages it suits.
    """
    if "max_rows" not in plot:
        return plot
    if not applies:
        if "max_rows" in own_plot:
            raise ValueError(
                f"page {title!r}: plot: max_rows: only applies to a compare: page "
                "drawn with draw: plot (it caps the rows of a stacked figure)"
            )
        return {k: v for k, v in plot.items() if k != "max_rows"}
    value = plot["max_rows"]
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, int) or value < 0
    ):
        raise ValueError(
            f"page {title!r}: plot: max_rows: must be a whole number of rows (0 or "
            f"null for no limit), got {value!r}"
        )
    return plot


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
    side builds one member per (source, variable) pair, and a top-level
    ``aggregate`` list one more factor per spec) without importing it --
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
    aggregate = kwargs.get("aggregate")
    n_aggregates = len(aggregate) if isinstance(aggregate, list) else 1
    return n_sources * n_variables * n_aggregates


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
    if index is None or not isinstance(time_value, str):
        return  # no time axis to pad around (or no instant to pad)
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


def _expand_xy_page(
    page: Any,
    namespace: dict[str, Any],
    *,
    title: str,
    test_source: str | None,
    get_index: Any,
    suite_cache: bool,
) -> tuple[dict[str, Any], str | None]:
    """Resolve one ``XY:``/``TS:`` page into the kwargs :func:`build` draws from.

    Each member becomes a ``field()`` kwargs dict (``source``, ``variable``, its own
    ``select``/``aggregate``/..., and the ``cache`` flag decided here). The member that
    reads ``defaults.test`` is pinned to the run exactly like a ``field:`` page
    (:func:`_pin_test_lane`) and caches by the same rule; every other member is left
    as written, with no run window -- its time axis is not the run's -- and caches with
    the suite, unless it says ``window: run``: then it gets the test run's whole span
    as a literal ``select.time`` (a reanalysis that must cover the run's dates, which
    change per run; refused if it names a ``time`` of its own, and a no-op on the test
    member). ``regions`` and ``at_center`` are checked here (so a typo fails at
    ``--list``) but stored as written: XY expands the boxes itself.

    Returns ``(kwargs, problem)``: ``problem`` is the :class:`_SourceUnreadable`
    message when a member's source could not be read (that member is then left
    unwindowed and uncached), else ``None``. A test source with no time axis pins
    nothing and is cacheable.
    """
    kind = page.kind
    # Deep-copied for the same reason as a field page's kwargs (see expand()): a
    # ``regions: "{pacific_regions}"`` placeholder resolves to the defaults' own dict.
    raw = copy.deepcopy(
        _template_value(dict(getattr(page, kind)), namespace, title=title)
    )
    args = _check_xy_args(raw, kind=kind, title=title)
    x, y = (_TS_X, _TS_Y) if kind == "TS" else (args.x, args.y)

    problem: str | None = None
    members: dict[str, dict[str, Any]] = {}
    for label, spec in args.members.items():
        m = dict(spec or {})
        window = m.pop("window", None)
        if window is not None and window != "run":
            raise ValueError(
                f"page {title!r}: {kind}: member {label!r}: window: {window!r} is "
                "not supported -- the only value is 'run' (the test run's span)"
            )
        for key in _XY_RESERVED_MEMBER_KEYS:
            if key in m:
                why = (
                    "caching is controlled by the suite's own cache:/cache_dir: "
                    "settings, not a per-member kwarg"
                    if key == "cache"
                    else f"the member's key ({label!r}) is its label"
                )
                raise ValueError(
                    f"page {title!r}: {kind}: member {label!r}: {key}: is not "
                    f"supported -- {why}"
                )
        m.setdefault("source", test_source)
        source = m["source"]
        if source is None:
            raise ValueError(
                f"page {title!r}: {kind}: member {label!r} has no source (give it "
                "a source: or set defaults.test)"
            )
        if "variables" in m:
            m["variable"] = m.pop("variables")
        m.setdefault("variable", [y, x])
        if test_source is not None and source == test_source:
            if m.get("select") is None:
                m["select"] = {}
            try:
                _, _, cacheable = _pin_test_lane(m, get_index(source))
            except _SourceUnreadable as exc:
                problem = problem or str(exc)
                cacheable = False
            m["cache"] = suite_cache and cacheable
        else:
            if window == "run":
                if test_source is None:
                    raise ValueError(
                        f"page {title!r}: {kind}: member {label!r}: window: run "
                        "needs defaults.test (the run whose span it takes)"
                    )
                if "time" in (m.get("select") or {}):
                    raise ValueError(
                        f"page {title!r}: {kind}: member {label!r}: window: run "
                        "and select.time both set the time window -- give one or "
                        "the other"
                    )
                # A literal window, so the cache key stays right when the run grows.
                try:
                    index = get_index(test_source)
                except _SourceUnreadable as exc:
                    problem = problem or str(exc)
                else:
                    if index is None:
                        raise ValueError(
                            f"page {title!r}: {kind}: member {label!r}: window: run "
                            f"needs the test source {test_source!r} to have a time "
                            "axis, but it has none (a climatology has no run span)"
                        )
                    _inject_field_window(m, index[0].isoformat(), index[-1].isoformat())
            m["cache"] = suite_cache
        members[label] = m

    unknown = [a for a in args.at_center if a not in members]
    if unknown:
        raise ValueError(
            f"page {title!r}: {kind}: at_center: {unknown} not among the members "
            f"{list(members)}"
        )
    if args.at_center and not args.regions:
        raise ValueError(
            f"page {title!r}: {kind}: at_center: needs regions: -- it samples a "
            "member at each region's center"
        )
    if args.regions is not None:
        from ocean_skill.xy import normalize_regions

        try:
            normalize_regions(args.regions)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"page {title!r}: {kind}: regions: {exc}") from exc

    return {
        "members": members,
        "x": x,
        "y": y,
        "regions": args.regions,
        "at_center": list(args.at_center),
    }, problem


# -- health: / rivers: / inputs: pages -------------------------------------------------
#
# ``health: {source, time, draw}`` and ``rivers: {source, time, draw}``: the run
# diagnostics of :mod:`ocean_skill.run_health` and :mod:`ocean_skill.river_inputs`. One
# page is one ``draw`` (one figure group) of an object that several pages share, so
# the expensive read happens once per (source, window) however many pages draw from it.
# ``inputs: {sources, time, max_steps, draw}`` is the same for
# :mod:`ocean_skill.input_integrity`, over a list of input-file sources.

#: The keys of a ``health:``/``rivers:`` mapping.
_DIAGNOSTIC_KEYS = ("source", "time", "draw")

#: The keys of an ``inputs:`` mapping.
_INPUTS_KEYS = ("sources", "time", "max_steps", "draw")


def _expand_inputs_page(
    page: Any, namespace: dict[str, Any], *, title: str
) -> dict[str, Any]:
    """Resolve one ``inputs:`` page into its kwargs.

    ``sources`` is filled in and flattened to a plain list of ``{name, label?, vars?}``
    mappings (:func:`ocean_skill.input_integrity.normalize_sources`); ``draw`` defaults
    to the first of :data:`ocean_skill.config.INPUT_DRAWS`. ``time`` and ``max_steps``
    are passed as written.
    """
    from ocean_skill.config import INPUT_DRAWS
    from ocean_skill.input_integrity import normalize_sources

    raw = _template_value(dict(page.inputs), namespace, title=title)
    unknown = sorted(set(raw) - set(_INPUTS_KEYS))
    if unknown:
        raise ValueError(
            f"page {title!r}: inputs: has no key(s) {unknown} -- it takes "
            f"{list(_INPUTS_KEYS)}"
        )
    if "sources" not in raw:
        raise ValueError(f"page {title!r}: inputs: needs sources: (catalog names)")
    kwargs = copy.deepcopy(raw)
    try:
        kwargs["sources"] = normalize_sources(kwargs["sources"])
    except ValueError as exc:
        raise ValueError(f"page {title!r}: {exc}") from exc
    kwargs.setdefault("draw", INPUT_DRAWS[0])
    check_diagnostic_draw(title, "inputs", kwargs["draw"])
    return kwargs


def _expand_diagnostic_page(
    page: Any,
    namespace: dict[str, Any],
    *,
    title: str,
    test_source: str | None,
    get_index: Any,
) -> tuple[dict[str, Any], bool, str | None]:
    """Resolve one ``health:``/``rivers:`` page into ``(kwargs, cacheable, problem)``.

    ``source`` defaults to ``defaults.test`` and ``draw`` to the kind's first figure.
    A ``health:`` page's time window is pinned to the source's time axis like a
    ``field:`` page's (:func:`_pin_test_lane`): no ``time`` becomes the whole run, a
    window with only a ``min`` or a ``max`` takes the run's own end for the other, and
    ``latest`` the last step, so the cache rule is the field pages' and the kwargs
    reproduce the figure. A ``rivers:`` page's ``time`` is passed as written -- a river
    file's time axis is not the run's. A source that cannot be read leaves the page
    unwindowed, uncached, with ``problem`` set for the run to list and skip.
    """
    from ocean_skill.config import HEALTH_DRAWS, RIVER_DRAWS

    kind = page.kind
    raw = _template_value(dict(getattr(page, kind)), namespace, title=title)
    unknown = sorted(set(raw) - set(_DIAGNOSTIC_KEYS))
    if unknown:
        raise ValueError(
            f"page {title!r}: {kind}: has no key(s) {unknown} -- it takes "
            f"{list(_DIAGNOSTIC_KEYS)}"
        )
    kwargs = copy.deepcopy(raw)
    kwargs.setdefault("source", test_source)
    if kwargs["source"] is None:
        raise ValueError(f"page {title!r}: no source (set defaults.test)")
    if not isinstance(kwargs["source"], str):
        raise ValueError(
            f"page {title!r}: {kind}: source: must be one source name, got "
            f"{kwargs['source']!r}"
        )
    kwargs.setdefault("draw", (HEALTH_DRAWS if kind == "health" else RIVER_DRAWS)[0])
    check_diagnostic_draw(title, kind, kwargs["draw"])
    if kind == "rivers":
        return kwargs, False, None
    try:
        index = get_index(kwargs["source"])
    except _SourceUnreadable as exc:
        return kwargs, False, str(exc)
    if index is None:  # no time axis: nothing to window
        kwargs.pop("time", None)
        return kwargs, True, None
    time = kwargs.pop("time", None)
    open_ended = False
    if isinstance(time, dict) and ("min" in time) != ("max" in time):
        time = {
            "min": time.get("min", index[0].isoformat()),
            "max": time.get("max", index[-1].isoformat()),
        }
        open_ended = True
    scratch = {"select": {} if time is None else {"time": time}}
    select, _, cacheable = _pin_test_lane(scratch, index)
    kwargs["time"] = select["time"]
    return kwargs, cacheable or open_ended, None


def _window_source(source: Any, test_source: Any, *, title: str) -> str:
    """Pick the one source whose time axis pins a ``field:`` page's window and cache.

    A plain ``source: name`` is its own. A list (``source: [a, b]`` -- ``osk.field``
    builds one member per entry, overlaid in a :class:`~ocean_skill.field.FieldSet`)
    has several time axes but only one window, shared by every member: it comes from
    ``defaults.test`` when that is one of the entries, else from the first entry. The
    list's order is therefore meaningful when no entry is the test source -- put the
    run whose "latest"/whole-run span the page should follow first.
    """
    if isinstance(source, (list, tuple)):
        if not source:
            raise ValueError(f"page {title!r}: field: source: is an empty list")
        source = test_source if test_source in source else source[0]
    if not isinstance(source, str):
        raise ValueError(
            f"page {title!r}: field: source: must be a source name or a list of "
            f"them, got {source!r}"
        )
    return source


def _resolve_field_kwargs(
    kwargs: dict[str, Any], *, title: str, test_source: Any, get_index: Any
) -> tuple[dict[str, Any], dict[str, Any], bool, bool, Any, str | None]:
    """Resolve one ``field:`` page's templated kwargs in place.

    Fills the ``source`` default, renames ``variables`` to ``variable``, checks the
    depth contradiction and pins the time selection to the run (:func:`_pin_test_lane`)
    -- against the time axis of :func:`_window_source`'s pick when ``source`` is a
    list. Returns ``(kwargs, select, had_explicit_time, cacheable, index, problem)``.
    ``index`` is ``None`` (and the page cacheable, nothing injected) for a source with
    no time axis; when the source cannot be read at all, the page is left unwindowed
    and uncacheable and ``problem`` says why (else ``None``). A page's ``contours:``
    override mapping goes through here too, so the lines cover the same dates as the
    fill.
    """
    kwargs.setdefault("source", test_source)
    source = kwargs["source"]
    if source is None:
        raise ValueError(f"page {title!r}: no source (set defaults.test)")
    if "variables" in kwargs:
        kwargs["variable"] = kwargs.pop("variables")
    _check_field_depth_contradiction(title, kwargs)
    try:
        index = get_index(_window_source(source, test_source, title=title))
    except _SourceUnreadable as exc:
        select = kwargs.setdefault("select", {})
        return kwargs, select, "time" in select, False, None, str(exc)
    select, had_explicit_time, cacheable = _pin_test_lane(kwargs, index)
    return kwargs, select, had_explicit_time, cacheable, index, None


def _lane_of(select: Any, role: str) -> dict[str, Any]:
    """One lane's side of a select as a dict: ``{}`` for a null side or a non-dict."""
    if not isinstance(select, dict):
        return {}
    return dict(select_for(select, role) or {})


def _has_time_key(lane: dict[str, Any]) -> bool:
    from ocean_skill.sources import _TIME_KEYS

    return any(k in lane for k in _TIME_KEYS)


def _check_selects_transects(
    title: str, select: Any, selects: dict[str, Any]
) -> None:
    """Refuse a ``selects:`` entry naming a transect that compare() would refuse.

    Each entry is merged over the page's select the way compare() merges it
    (:func:`_merged_select`; a missing select stands for an empty pair-spec, the run
    window being a time key only) and the result handed to
    :func:`transect_pair_problem`, so the page fails at expand time with compare()'s
    own reason -- e.g. the page's pair-spec carries a ``depth`` on one lane only.
    """
    base = {role: _lane_of(select, role) for role in ("test", "reference")}
    for label, entry in selects.items():
        raw, _ = selects_entry_parts(entry)
        if not names_transect(raw):
            continue
        entry_select = (
            {role: _lane_of(raw, role) for role in ("test", "reference")}
            if is_pair_spec(raw)
            else _lane_of(raw, "test")
        )
        problem = transect_pair_problem(_merged_select(base, entry_select))
        if problem is not None:
            raise ValueError(
                f"page {title!r}: compare: selects: names a transect ({label!r}), "
                f"which compare() refuses here: {problem}"
            )


def _resolve_compare_kwargs(
    kwargs: dict[str, Any], *, title: str, test_source: Any, get_index: Any
) -> tuple[dict[str, Any], bool, str | None]:
    """Resolve one ``compare:`` page's templated kwargs in place.

    Fills the ``test`` default, injects the run window into the test lane and decides
    whether the result is cacheable (before the suite-wide ``cache:`` switch). Returns
    ``(kwargs, cacheable, problem)``. A test source with no time axis gets no window
    and is cacheable (static data); one that cannot be read leaves the page
    unwindowed and uncacheable, ``problem`` saying why (else ``None``). A page's
    ``contours:`` override mapping goes through here too, so the lines cover the same
    dates as the fill.
    """
    kwargs.setdefault("test", test_source)
    source = kwargs["test"]
    if source is None:
        raise ValueError(f"page {title!r}: no test source (set defaults.test)")
    # ``notes:`` is display text for osk.table(); it passes straight to compare(notes=).
    if "notes" in kwargs and not isinstance(kwargs["notes"], str):
        raise ValueError(
            f"page {title!r}: compare: notes must be a string, got "
            f"{type(kwargs['notes']).__name__} ({kwargs['notes']!r})"
        )
    select = kwargs.get("select")
    paired = is_pair_spec(select)
    selects = kwargs.get("selects")
    if selects is not None and not isinstance(selects, dict):
        raise ValueError(
            f"page {title!r}: compare: selects: must be a mapping of label -> select, "
            f"got {type(selects).__name__} ({selects!r})"
        )
    if selects is not None and not selects:
        raise ValueError(
            f"page {title!r}: compare: selects: is empty -- if it is a {{placeholder}} "
            "(the templates' {transects}), fill in that entry under defaults:"
        )
    # The run window of a page with no flat select is a pair-spec (test lane only: the
    # reference may be a climatology with a time axis of its own). compare() takes a
    # transect inside a pair-spec only when both sides name the same transect and
    # depths, which a plain labelled entry merged over the page's select does -- so
    # only the merges compare() would still refuse are refused here, whatever the
    # page's own select (a pair-spec entry over a flat select can be refused too).
    if selects:
        _check_selects_transects(title, select, selects)
    try:
        index = get_index(source)
    except _SourceUnreadable as exc:
        return kwargs, False, str(exc)
    if index is None:
        # A static source: nothing to window. Only ``times:`` can still make it
        # uncacheable (see below).
        return kwargs, kwargs.get("times") is None, None
    t0, t1 = index[0].isoformat(), index[-1].isoformat()
    detide_margin = _detide_margin(kwargs.get("detide"), lane="test")
    if select is None or paired:
        was_latest = paired and (
            isinstance(select.get("test"), dict)
            and select["test"].get("time") == "latest"
        )
        pinned = (
            select is None or was_latest or "time" not in (select.get("test") or {})
        )
        _inject_compare_window(kwargs, t0, t1)
        if was_latest:
            kwargs["select"]["test"]["time"] = t1
        test_lane = kwargs["select"]["test"]
        cacheable = pinned or _is_closed(test_lane, index, margin=detide_margin)
        # each labelled select is merged over this lane: one that names a time of
        # its own replaces the window, so it has to be closed like any explicit one
        for entry in (selects or {}).values():
            entry_lane = _lane_of(selects_entry_parts(entry)[0], "test")
            if _has_time_key(entry_lane):
                cacheable = cacheable and _is_closed(
                    {**test_lane, **entry_lane}, index, margin=detide_margin
                )
    else:
        # A flat, non-paired select applies to both lanes at once --
        # compare()'s own contract -- so rewriting only the "test"
        # side here would silently change what the reference reads
        # too. Never rewritten, only checked.
        cacheable = _is_closed(select, index, margin=detide_margin)
        for entry in (selects or {}).values():
            entry_lane = _lane_of(selects_entry_parts(entry)[0], "test")
            if _has_time_key(entry_lane):
                cacheable = cacheable and _is_closed(
                    {**select, **entry_lane}, index, margin=detide_margin
                )
    if kwargs.get("times") is not None:
        # times= fans this one page into several per-bin comparisons,
        # each replacing whatever time entry select carried with its
        # own bin value at draw time (comparison._fanned_time_select)
        # -- what actually gets keyed is not what was just resolved
        # above, so nothing here can vouch for it.
        cacheable = False
    return kwargs, cacheable, None


def _check_contours_spec(spec: Any, *, kind: str, title: str) -> bool | dict[str, Any]:
    """Validate a page's (templated) ``contours:`` value: a bool or a mapping."""
    if spec is None:
        return False
    if isinstance(spec, (bool, dict)):
        return spec
    raise ValueError(
        f"page {title!r}: contours: must be true (contour the plotted {kind} "
        f"itself) or a mapping of {kind}: overrides for a second one, got "
        f"{spec!r}"
    )


def _contour_overrides(
    spec: dict[str, Any], raw: dict[str, Any], *, kind: str, title: str
) -> dict[str, Any]:
    """Shallow-merge a ``contours:`` mapping over the page's own (templated) kwargs.

    ``raw`` is the page's own ``field:``/``compare:`` kwargs before resolution, so the
    merge happens in the keys the YAML used -- and a ``field:`` page's ``variables`` and
    ``variable`` spellings are treated as the one key they are, so an override in
    either spelling replaces the page's own rather than sitting beside it.
    """
    if not spec:
        raise ValueError(
            f"page {title!r}: contours: {{}} overrides nothing -- use contours: true "
            f"to contour the plotted {kind} itself"
        )
    merged = dict(raw)
    if kind == "field" and ("variables" in spec or "variable" in spec):
        merged.pop("variables", None)
        merged.pop("variable", None)
    merged.update(spec)
    return merged


def expand(suite: Any, *, thin_dates: bool = False) -> list[ExpandedPage]:
    """Turn ``suite.pages`` into a flat, fully-resolved list of :class:`ExpandedPage`.

    ``suite`` is a :class:`~ocean_skill.config.SuiteConfig`. Reads the test
    source's native time index at most once (coordinate-only; memoized here), used
    to resolve ``latest``/``month: run`` and to bound the literal windows injected
    into any page that would otherwise mean "however much of the run exists right
    now". That same index also decides each page's ``cache=`` (see
    :func:`_is_closed`), against whatever it resolved to on *this* call --
    a page's cache entry is only ever as fresh as the last time the suite ran.

    A source that cannot supply that index never aborts the expansion: one with no
    time axis (a monthly climatology) is static -- its pages get no window and are
    cacheable -- and one that cannot be read at all leaves its pages unwindowed with
    :attr:`ExpandedPage.problem` set, for the run to list and skip.

    ``thin_dates=True`` is test mode (``ocean-skill-run --smoke``): a compare page's
    ``times:`` list and a ``for_each: {month: ...}`` fan keep only their first and last
    entry, and each page so cut says so in :attr:`ExpandedPage.thinned`. Variables,
    depths and every other fan stay complete -- they exercise different code. A compare
    page that sets no ``min_pairs`` gets :data:`TEST_MIN_PAIRS`, since the few steps
    kept cannot reach the real default. Left off (the default), the pages are exactly
    what the suite says.
    """
    from ocean_skill import extrema
    from ocean_skill.comparison import _ANY_VERTICAL_KEYS
    from ocean_skill.sources import _TIME_KEYS

    defaults = dict(suite.defaults)
    test_source = defaults.get("test")

    index_cache: dict[str, Any] = {}
    pin_seen: set[str] = set()

    def get_index(source: str) -> Any:
        """Return the source's native time index, or ``None`` if it has no time axis.

        Raises :class:`_SourceUnreadable` (memoized like a success) when it cannot be
        read at all -- callers turn that into a page's ``problem``.
        """
        if source not in index_cache:
            try:
                index_cache[source] = extrema._native_time_index(source)
            except Exception as exc:
                timeless = isinstance(exc, ValueError) and _lacks_time_axis(source)
                index_cache[source] = (
                    None if timeless else _SourceUnreadable(source, exc)
                )
        result = index_cache[source]
        if isinstance(result, _SourceUnreadable):
            raise result
        return result

    out: list[ExpandedPage] = []
    for page in suite.pages:
        if page.kind == "section":
            # Handled before everything the other kinds share: a divider has no
            # source (so no time index to read), no plot (so ``defaults.plot`` must
            # not be merged in, and ``_pin_to_page`` must not warn about a
            # ``defaults.plot: {zoom: ...}`` it will never apply to), and no
            # ``for_each`` (the schema already refused one). Title and notes are
            # templated against ``defaults`` alone.
            # a mapping default reads by attribute, as on the other pages
            # ("{period.min}", "{period.max:.10}" for the date alone)
            namespace = {
                k: _Item(v) if isinstance(v, dict) else v for k, v in defaults.items()
            }
            title = _template_value(page.title, namespace, title=page.title)
            text = _template_value(page.section, namespace, title=title)
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
            # only a ``for_each:`` reads this (to size its time axis); the page's own
            # window is picked, once per combo, by ``_resolve_field_kwargs`` below
            if page.for_each and isinstance(page_source, (list, tuple)):
                page_source = _window_source(page_source, test_source, title=page.title)
        elif page.kind == "compare":
            page_source = (page.compare or {}).get("test", test_source)
        elif page.kind in ("XY", "TS"):
            page_source = test_source
        elif page.kind in ("health", "rivers"):
            page_source = (getattr(page, page.kind) or {}).get("source", test_source)
        # an inputs: page names its own sources and has no run to take months from

        test_index = None
        index_error: Exception | None = None
        if page.for_each and page_source:
            try:
                test_index = get_index(page_source)
            except _SourceUnreadable as exc:
                index_error = exc
            else:
                if test_index is None:
                    index_error = ValueError(
                        f"page {page.title!r}: for_each: month: needs the time axis "
                        f"of {page_source!r}, but it has none (a climatology has no "
                        "run to take months from) -- list the months explicitly"
                    )
        month_note = None
        try:
            combos = _expand_for_each(
                page.title,
                page.for_each,
                defaults=defaults,
                test_index=test_index,
                index_error=index_error,
                thin_months=thin_dates,
            )
            if thin_dates and "month" in (page.for_each or {}):
                n_months = len(_months_in_all(test_index, page.for_each["month"]))
                kept = len({str(c["month"]) for c in combos})
                if kept < n_months:
                    month_note = f"{kept} of {n_months} months"
        except _SourceUnreadable as exc:
            # ``month: run`` is the only thing that reads the index, and without it
            # there are no combinations to name: one placeholder carries the reason.
            out.append(
                ExpandedPage(
                    title=page.title,
                    kind=page.kind,
                    kwargs={},
                    plot={},
                    cache=False,
                    problem=str(exc),
                )
            )
            continue

        plot_defaults = defaults.get("plot", {})

        for combo in combos:
            date_note = None
            namespace = {
                **defaults,
                **{k: _Item(v) if isinstance(v, dict) else v for k, v in combo.items()},
            }
            title = _template_value(page.title, namespace, title=page.title)
            own_plot = _template_value(page.plot, namespace, title=title)
            plot = {**plot_defaults, **own_plot}
            plot = _page_title_option(plot, title)
            if suite.pdf:
                plot = _pin_to_page(plot, seen=pin_seen)
            default_plot = tuple(
                k for k in plot_defaults if k in plot and k not in own_plot
            )
            caption = (
                _template_value(page.caption, namespace, title=title)
                if page.caption is not None
                else None
            )
            if caption is not None and not isinstance(caption, str):
                raise ValueError(
                    f"page {title!r}: caption: must resolve to text, got "
                    f"{type(caption).__name__} ({page.caption!r})"
                )
            if page.kind != "compare":
                plot = _max_rows_key(plot, own_plot, applies=False, title=title)

            if page.kind == "field":
                # Deep-copied so an exact-placeholder select (``select: "{sel}"``,
                # resolved by ``_template_value`` to the referenced object itself
                # -- see its own docstring) is never the *same* dict as whatever
                # ``defaults``/another page's combo holds; the "latest" resolution
                # and the window injection below both write into ``select`` in
                # place, and without this a shared placeholder would leak one
                # page's resolved time into every other page that names it.
                raw = _template_value(dict(page.field), namespace, title=title)
                kwargs, select, had_explicit_time, cacheable, index, problem = (
                    _resolve_field_kwargs(
                        copy.deepcopy(raw),
                        title=title,
                        test_source=test_source,
                        get_index=get_index,
                    )
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

                contours = _check_contours_spec(
                    _template_value(page.contours, namespace, title=title),
                    kind="field",
                    title=title,
                )
                if contours and steps:
                    raise ValueError(
                        f"page {title!r}: contours: is not supported with a then: "
                        "chain -- the steps return something other than the field "
                        "the lines would be drawn from"
                    )
                if isinstance(contours, dict):
                    # Resolved like the page itself (source default, run window,
                    # latest) so the lines cover the same dates as the fill; the
                    # page is only cacheable when both reads are.
                    contours, _, _, contour_cacheable, _, contour_problem = (
                        _resolve_field_kwargs(
                            copy.deepcopy(
                                _contour_overrides(
                                    contours, raw, kind="field", title=title
                                )
                            ),
                            title=title,
                            test_source=test_source,
                            get_index=get_index,
                        )
                    )
                    cacheable = cacheable and contour_cacheable
                    problem = problem or contour_problem
                out.append(
                    ExpandedPage(
                        title=title,
                        kind="field",
                        kwargs=kwargs,
                        plot=plot,
                        cache=suite.cache and cacheable,
                        steps=steps,
                        contours=contours,
                        problem=problem,
                    )
                )

            elif page.kind == "compare":
                # See the field branch above for why this is deep-copied.
                raw = _template_value(dict(page.compare), namespace, title=title)
                if thin_dates:
                    date_note = _thin_times(raw)
                    raw.setdefault("min_pairs", TEST_MIN_PAIRS)
                kwargs, cacheable, problem = _resolve_compare_kwargs(
                    copy.deepcopy(raw),
                    title=title,
                    test_source=test_source,
                    get_index=get_index,
                )
                contours = _check_contours_spec(
                    _template_value(page.contours, namespace, title=title),
                    kind="compare",
                    title=title,
                )
                draw = _template_value(page.draw, namespace, title=title) or "plot"
                check_draw(title, draw, contours)
                plot = _max_rows_key(
                    plot, own_plot, applies=draw == "plot", title=title
                )
                if isinstance(contours, dict):
                    # Same treatment as the page (see the field branch above).
                    overrides = _contour_overrides(
                        contours, raw, kind="compare", title=title
                    )
                    if thin_dates:
                        _thin_times(overrides)  # the lines pair with the fill's bins
                    contours, contour_cacheable, contour_problem = (
                        _resolve_compare_kwargs(
                            copy.deepcopy(overrides),
                            title=title,
                            test_source=test_source,
                            get_index=get_index,
                        )
                    )
                    cacheable = cacheable and contour_cacheable
                    problem = problem or contour_problem
                out.append(
                    ExpandedPage(
                        title=title,
                        kind="compare",
                        kwargs=kwargs,
                        plot=plot,
                        cache=suite.cache and cacheable,
                        contours=contours,
                        draw=draw,
                        problem=problem,
                    )
                )

            elif page.kind in ("XY", "TS"):
                kwargs, problem = _expand_xy_page(
                    page,
                    namespace,
                    title=title,
                    test_source=test_source,
                    get_index=get_index,
                    suite_cache=suite.cache,
                )
                out.append(
                    ExpandedPage(
                        title=title,
                        kind=page.kind,
                        kwargs=kwargs,
                        plot=plot,
                        cache=all(m["cache"] for m in kwargs["members"].values()),
                        problem=problem,
                    )
                )

            elif page.kind in ("health", "rivers"):
                kwargs, cacheable, problem = _expand_diagnostic_page(
                    page,
                    namespace,
                    title=title,
                    test_source=test_source,
                    get_index=get_index,
                )
                out.append(
                    ExpandedPage(
                        title=title,
                        kind=page.kind,
                        kwargs=kwargs,
                        plot=plot,
                        cache=page.kind == "health" and suite.cache and cacheable,
                        problem=problem,
                    )
                )

            elif page.kind == "inputs":
                out.append(
                    ExpandedPage(
                        title=title,
                        kind="inputs",
                        kwargs=_expand_inputs_page(page, namespace, title=title),
                        plot=plot,
                        cache=suite.cache,
                    )
                )

            else:  # summary
                kwargs = _template_value(dict(page.summary), namespace, title=title)
                kwargs["kind"] = summary_kinds(title, kwargs)
                kwargs["scope"] = kwargs.get("scope", "all")
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
            out[-1].caption = caption
            out[-1].default_plot = default_plot
            out[-1].thinned = _thinned_note(date_note, month_note)
    return out


def _thinned_note(*notes: str | None) -> str | None:
    """``"test: ..."`` text for the notes that apply to a page, or ``None`` for none."""
    parts = [n for n in notes if n]
    return f"test: {', '.join(parts)}" if parts else None


def time_pins(pages: list[ExpandedPage]) -> Any:
    """Return the times ``pages`` name, as :class:`ocean_skill.sources.TimePins`.

    Read-only.

    Walks every resolved page's kwargs, ``then:`` steps and contour overrides for a
    ``time`` entry (any spelling in :data:`ocean_skill.sources._TIME_KEYS`) or a
    ``times:`` list: a date is an instant, a short ``{min, max}`` window or a period
    string (``"2010-01"``) a window. Longer windows, ``times: {resample: ...}``, and
    ``"latest"`` add nothing (:meth:`ocean_skill.sources.TimePins.from_values`). Used by
    ``ocean-skill-run --smoke`` so the truncated read keeps the steps the pages ask for.
    """
    from ocean_skill.sources import _TIME_KEYS, TimePins

    values: list[Any] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in _TIME_KEYS or key == "times":
                    values.append(value)
                walk(value)
        elif isinstance(node, (list, tuple)):
            for item in node:
                walk(item)

    for page in pages:
        if page.kind == "section":
            continue
        walk(page.kwargs)
        walk(page.steps)
        walk(page.contours)
    return TimePins.from_values(values)


def _check_contour_families(
    title: str, families: list[str], plotted: list[str], contour_cs: Any
) -> None:
    """Refuse a ``contours:`` comparison set that does not pair with the plotted one.

    Each plotted family bucket is paired with the contour comparisons of the same
    family, by position (see ``ComparisonSet.plot(contours=...)``), so a family with
    no counterpart -- or a different count -- would mis-pair rather than overlay.
    """
    contour_families = [c.family for c in contour_cs]
    for fam in families:
        n_plot, n_contour = plotted.count(fam), contour_families.count(fam)
        if n_contour == 0:
            raise ValueError(
                f"page {title!r}: contours: built no {fam!r} comparison to pair "
                f"with the plotted one (it made {sorted(set(contour_families))} -- "
                "check the contour variables, or that none was skipped for missing "
                "data)"
            )
        if n_contour != n_plot:
            raise ValueError(
                f"page {title!r}: contours: built {n_contour} {fam!r} comparison(s) "
                f"for {n_plot} plotted -- the lines are paired with the fill by "
                "position, so the counts must match"
            )


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


class PageSkipped(Exception):
    """A page that was deliberately not drawn, as opposed to one that crashed.

    :func:`build` raises it when the page's data is not there to draw (every
    comparison entirely NaN, say). The runner treats it like any other skip but files it
    under ``skipped`` rather than ``failed`` in its problems list.
    """


def _test_side_entry(entry: Any) -> Any:
    """Return a ``selects:`` entry cut to its test side (select and own aggregate)."""
    select, _ = selects_entry_parts(entry)
    if select is entry:  # a plain select, not the {select:, aggregate:} form
        return select_for(entry, "test")
    out = dict(entry)
    if "select" in out:
        out["select"] = select_for(out["select"], "test")
    aggregate = out.get("aggregate")
    if isinstance(aggregate, list):
        out["aggregate"] = [aggregate_for(a, "test") for a in aggregate]
    elif aggregate is not None:
        out["aggregate"] = aggregate_for(aggregate, "test")
    return out


def _self_check_kwargs(kwargs: dict[str, Any]) -> dict[str, Any] | None:
    """One compare call's kwargs with every lane-specific setting set to the test's.

    ``None`` when the call cannot be turned into a test-against-itself (several test
    sources: the reference list would pair each with all the others).
    """
    test = kwargs.get("test")
    if not isinstance(test, str):
        return None
    out = copy.deepcopy(kwargs)
    out["reference"] = test
    variables = out.get("variables")
    if isinstance(variables, list):
        out["variables"] = [variable_for(v, "test") for v in variables]
    elif variables is not None:
        out["variables"] = variable_for(variables, "test")
    select = out.get("select")
    if select is not None:
        out["select"] = select_for(select, "test")
    if isinstance(out.get("selects"), dict):
        out["selects"] = {k: _test_side_entry(v) for k, v in out["selects"].items()}
    aggregate = out.get("aggregate")
    if isinstance(aggregate, list):
        out["aggregate"] = [aggregate_for(a, "test") for a in aggregate]
    elif aggregate is not None:
        out["aggregate"] = aggregate_for(aggregate, "test")
    if "qc" in out:
        out["qc"] = qc_for(out["qc"], "test")
    # Per-lane switches that would otherwise treat the two copies of one source
    # differently (detide the reference only, demean one side): both lanes get the
    # test's choice.
    sm = out.get("subtract_mean")
    if isinstance(sm, str):
        out["subtract_mean"] = sm == "test"
    elif isinstance(sm, dict):
        out["subtract_mean"] = bool(sm.get("test", False))
    dt = out.get("detide")
    if dt == "reference":
        out["detide"] = False
    elif dt == "test":
        out["detide"] = True
    elif isinstance(dt, dict) and dt.keys() & {"test", "reference"}:
        out["detide"] = dt.get("test", False)
    return out


def self_check_page(page: ExpandedPage) -> ExpandedPage | None:
    """Return a ``compare`` page redrawn against itself, or ``None`` if not possible.

    The reference is replaced by the test source, and every ``{test, reference}``
    pair-spec (``variables``, ``select``, ``aggregate``, ``qc``) is collapsed to its
    test side, so both lanes are the very same recipe: any difference the comparison
    then reports comes from the pipeline treating the two lanes differently, not from
    the data. The copy is titled ``"[self-check] <title>"`` and never cached (a stored
    result would hide whatever the code does now). ``None`` for any other kind of page,
    and for a page whose ``test`` is not one source name.
    """
    if page.kind != "compare":
        return None
    kwargs = _self_check_kwargs(page.kwargs)
    if kwargs is None:
        return None
    contours = page.contours
    if isinstance(contours, dict):
        contours = _self_check_kwargs(contours)
        if contours is None:
            return None
    from dataclasses import replace

    return replace(
        page,
        title=f"[self-check] {page.title}",
        kwargs=kwargs,
        contours=contours,
        cache=False,
        metrics_records=[],
        results=[],
        dropped=[],
    )


def self_check_problem(page: ExpandedPage) -> str | None:
    """Why a drawn self-check page's metrics are not those of identical lanes, or None.

    Two lanes carrying the same data must show zero bias, zero error and perfect
    correlation. Each comparison's record is checked against a tolerance scaled to the
    data (``1e-6`` of the larger of its mean, spread and 1), so float noise passes and a
    lane that was cropped, shifted or masked differently does not.
    """
    bad = []
    for rec in page.metrics_records:
        scale = max(
            [abs(rec[k]) for k in ("mean_test", "std_test") if _finite(rec.get(k))]
            + [1.0]
        )
        tol = 1e-6 * scale
        found = []
        # A NaN here is itself the finding: identical lanes always have a bias and an
        # error. (corr alone may be NaN -- a constant field has no correlation.)
        nan = [k for k in ("bias", "rmse") if k in rec and _non_finite(rec[k])]
        if nan:
            found.append(f"{'/'.join(nan)} NaN although the lanes should be identical")
        if _finite(rec.get("bias")) and abs(rec["bias"]) > tol:
            found.append(f"bias {rec['bias']:.4g}")
        if _finite(rec.get("rmse")) and rec["rmse"] > tol:
            found.append(f"rmse {rec['rmse']:.4g}")
        if _finite(rec.get("corr")) and rec["corr"] < 1 - 1e-6:
            found.append(f"corr {rec['corr']:.6g}")
        if found:
            label = rec.get("label") or rec.get("variable") or "comparison"
            where = ", ".join(
                f"{k}={rec[k]}" for k in ("depth", "time") if rec.get(k) is not None
            )
            bad.append(f"{label}{f' ({where})' if where else ''}: {', '.join(found)}")
    if not bad:
        return None
    more = f" (+{len(bad) - 3} more)" if len(bad) > 3 else ""
    return (
        f"identical lanes disagree -- {'; '.join(bad[:3])}{more} -- a lane is being "
        "treated differently from the other"
    )


def _non_finite(x: Any) -> bool:
    """Report whether ``x`` is a number that is NaN or infinite (not ``None``)."""
    import math

    try:
        return not math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _finite(x: Any) -> bool:
    import math

    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _all_nan_reason(c: Any) -> str | None:
    """``"all-NaN: <lane> has no data at ..."`` if a comparison's lane is all NaN."""
    import numpy as np

    try:
        aligned = c.aligned
        lane = next(
            (
                name
                for name in ("test", "reference")
                if name in aligned and not bool(np.isfinite(aligned[name]).any())
            ),
            None,
        )
    except Exception as exc:
        warnings.warn(
            f"could not check {getattr(c, 'variable', '?')!r} "
            f"({getattr(c, 'test_name', '?')} vs {getattr(c, 'reference_name', '?')}) "
            f"for all-NaN data, so it is drawn as it is: {type(exc).__name__}: {exc}",
            stacklevel=2,
        )
        return None
    if lane is None:
        return None
    try:
        depth, time = c._record_depth_time()
    except Exception:
        depth = time = None
    where = ", ".join(
        f"{k} {v}" for k, v in (("depth", depth), ("time", time)) if v is not None
    )
    what = (
        f"{getattr(c, 'variable', None)}: {getattr(c, 'test_name', '?')} vs "
        f"{getattr(c, 'reference_name', '?')}"
    )
    return f"all-NaN: {lane} has no data at {where or 'the selected cells'} ({what})"


def _drop_all_nan(page: ExpandedPage, cs: Any, *, osk: Any) -> tuple[Any, list[bool]]:
    """``cs`` without its all-NaN comparisons, and which of its members were kept.

    Raises :class:`PageSkipped` if none is left; otherwise the reasons for those that
    went are warned and kept on ``page.dropped``.
    """
    reasons = [_all_nan_reason(c) for c in cs]
    keep = [r is None for r in reasons]
    if all(keep):
        return cs, keep
    dropped = [r for r in reasons if r is not None]
    if not any(keep):
        more = f" (+{len(dropped) - 3} more)" if len(dropped) > 3 else ""
        raise PageSkipped("; ".join(dropped[:3]) + more)
    page.dropped = dropped
    warnings.warn(
        f"page {page.title!r}: dropped {len(dropped)} of {len(reasons)} comparison(s): "
        + "; ".join(dropped[:3]),
        stacklevel=3,
    )
    return osk.ComparisonSet([c for c, k in zip(cs, keep, strict=True) if k]), keep


def _page_title_option(plot: dict[str, Any], page_title: str) -> dict[str, Any]:
    """Resolve ``plot: {title: page}``: the figure's title is the page's own.

    ``title:`` replaces the figure's own (generic) suptitle, so a page can say what its
    figure is -- which boundary side, which month -- and the literal ``page`` is the
    shorthand for the page's title, placeholders already filled. Anything else must be
    text (``""`` draws no title); a number or a list would reach the renderer as a
    mangled one.
    """
    if "title" not in plot:
        return plot
    given = plot["title"]
    if given == "page":
        return {**plot, "title": page_title}
    if given is not None and not isinstance(given, str):
        raise ValueError(
            f"page {page_title!r}: plot: title: must be text (or `page` for the "
            f"page's own title), got {type(given).__name__} ({given!r})"
        )
    return plot


def _draw(page: ExpandedPage, method: Any, kwargs: dict[str, Any]) -> Any:
    """Call ``method(**kwargs)``, dropping refused options that ``defaults.plot`` set.

    A suite-wide ``defaults.plot: {robust: true}`` suits the maps and not every other
    page (a ``draw: metrics_series`` page has no such option). A draw method names the
    options it cannot use before it draws anything, so when every one it refuses came
    from ``defaults.plot`` (:attr:`ExpandedPage.default_plot`), the page is drawn
    without them, silently (a suite-wide default is not an error on a page it does not
    suit, and warning on every such page buried the ones that matter). An option the
    page wrote itself is still an error, as is anything else the method refuses.
    """
    try:
        return method(**kwargs)
    except UnusableOptions as exc:
        refused = [k for k in page.default_plot if k in kwargs and k in exc.options]
        if not refused:
            raise
    return method(**{k: v for k, v in kwargs.items() if k not in refused})


def _pools_rows(plot_opts: dict[str, Any]) -> bool:
    """Whether these plot options put a row family's rows on a scale they all share."""
    pooled = bool(plot_opts.get("shared_limits"))
    return pooled or plot_opts.get("panels") == "difference"


def _draw_continued(page: ExpandedPage, bucket: Any, kwargs: dict[str, Any]) -> Any:
    """Draw a chunk after the first of a split page, its title ending in ``(cont.)``.

    A title the page gave (``plot: {title: ...}``) is extended before drawing. Otherwise
    the renderer's own suptitle is extended after it. A chunk the renderer gave no
    suptitle keeps none: it is drawn once, never again with a title of its own (a
    suptitle added to a finished figure would land on top of its panels), and the
    ``_partN`` file suffix and the report's page order carry the continuity.
    """
    if "title" in kwargs:
        title = kwargs["title"]
        if title:  # an explicit "" asks for no title at all
            kwargs = {**kwargs, "title": f"{title} (cont.)"}
        return _draw(page, bucket.plot, kwargs)
    fig = _draw(page, bucket.plot, kwargs)
    sup = getattr(fig, "_suptitle", None)
    if sup is not None and sup.get_text():
        sup.set_text(f"{sup.get_text()} (cont.)")
    return fig


def _labelled_record(comparison: Any) -> dict[str, Any]:
    """Return a comparison's metric record carrying the comparison's own label."""
    record = comparison.metrics()
    label = getattr(comparison, "label", None)
    return {**record, "label": label} if label else record


#: Record fields that tell two pooled comparisons with the same label apart, in the
#: order a disambiguated label lists them.
_DISTINGUISHING_FIELDS = (
    "variable", "statistic", "aggregate", "depth", "time", "reference", "test"
)


#: A mixed-layer-depth variable as the labels spell it, with or without the method
#: (``mld``, ``mld (density_threshold)``): named in full in a table.
_MLD_LABEL_RE = re.compile(r"mld(?: \([^)]*\))?", re.IGNORECASE)


def _short_label(text: str) -> str:
    """Shorten a label made of CF standard names the way the figures do.

    Every figure titles a variable by :func:`ocean_skill.vars.short_name`
    (``"temperature"``, ``"nitrate"``); a summary table that named its rows by the
    record's raw variable instead read ``mole_concentration_of_nitrate_in_sea_water``
    in one table and ``nitrate`` in the plots next to it. Applied to each `` · ``-joined
    part on its own, and only to a single token that *is* a standard name -- a name the
    vocabulary knows, or one the short-name rules would actually shorten -- so a
    source name (``mld_bry_phys_strip``) or a phrase is left exactly as written.
    """
    from ocean_skill.vars import short_name
    from ocean_skill.vocabulary import is_known

    def one(part: str) -> str:
        if _MLD_LABEL_RE.fullmatch(part):  # "mld", "mld (density_threshold)"
            return "mixed layer depth"
        if " " in part or "_" not in part:
            return part
        short = short_name(part)
        return short if is_known(part) or short != part.replace("_", " ") else part

    return " · ".join(one(part) for part in text.split(" · "))


def _spell_field(name: str, value: Any) -> str:
    """Spell one record field for a label: a depth band or time window as its label.

    A band or window is a dict in the record (``{"min": 0, "max": 500}``); ``str()`` of
    it would put the dict's repr into a summary table's row name, where the Notes
    column already spells it (``"0-500 m"``, ``"2010-01-01–2013-12-31"``).
    """
    if name == "depth" and (
        isinstance(value, dict | list | tuple)
        or (isinstance(value, int | float) and not isinstance(value, bool))
    ):
        return _depth_label(value)  # a bare 200 reads "200 m"; "surface" stays as is
    if name == "time" and isinstance(value, dict | list | tuple):
        return _collapse_range(_time_label(value))
    if name == "variable":
        return _short_label(str(value))
    if name in ("test", "reference"):
        from ocean_skill.catalog import source_label

        return source_label(str(value))
    return str(value)


def _name_differences(names: dict[int, str]) -> dict[int, str]:
    """Return each source name cut to the ``_``-separated words the others don't share.

    ``pacmed_bry_phys_south`` and ``pacmed_bry_phys_west`` become ``south`` and
    ``west``: the words every name starts and ends with are dropped. A name left with
    nothing (one is the others' prefix) is kept whole.
    """
    words = {i: n.split("_") for i, n in names.items()}
    lists = list(words.values())
    head = 0
    while all(len(w) > head for w in lists) and len({w[head] for w in lists}) == 1:
        head += 1
    tail = 0
    while (
        all(len(w) > head + tail for w in lists)
        and len({w[-1 - tail] for w in lists}) == 1
    ):
        tail += 1
    out = {}
    for i, w in words.items():
        core = w[head : len(w) - tail]
        out[i] = " ".join(core) if core else names[i]
    return out


def _distinct_labels(records: list[dict[str, Any]]) -> list[str]:
    """Return one label per pooled record, unique across the pool.

    A record's own label (else its variable) is kept when no other record shares it.
    Records that share one -- the run mean and the seasonal spread of one variable, or
    two pages of the same quantity -- get the fields that differ between them appended
    (``"temperature · std · 200"``), and a running number as the last resort, so the
    summary table has one row per comparison instead of refusing the pool.
    """
    base = [
        _short_label(str(r.get("label") or r.get("variable") or "")) for r in records
    ]
    groups: dict[str, list[int]] = {}
    for i, label in enumerate(base):
        groups.setdefault(label, []).append(i)
    out = list(base)
    for label, idx in groups.items():
        if len(idx) < 2:
            continue
        differ = [
            f
            for f in _DISTINGUISHING_FIELDS
            if len({str(records[i].get(f)) for i in idx}) > 1
        ]
        spelled = {
            f: {
                i: _spell_field(f, records[i][f])
                for i in idx
                if records[i].get(f) is not None
            }
            for f in differ
        }
        for f in ("test", "reference"):
            # sources that share a catalog label (one boundary file's south and west
            # sides, both "boundary file") are named by what tells their names apart
            names = {i: str(records[i][f]) for i in spelled.get(f, {})}
            if len(set(spelled.get(f, {}).values())) < len(set(names.values())):
                spelled[f] = _name_differences(names)
        for i in idx:
            parts = [spelled[f][i] for f in differ if i in spelled[f]]
            # the variable reads first ("temperature surface mean"), the rest after
            named = "variable" in differ and records[i].get("variable") is not None
            lead = parts.pop(0) + " " if named else ""
            out[i] = lead + (" · ".join([label, *parts]) if parts else label)
        seen: dict[str, int] = {}
        for i in idx:
            seen[out[i]] = seen.get(out[i], 0) + 1
        if any(n > 1 for n in seen.values()):
            for n, i in enumerate(idx, start=1):
                out[i] = f"{out[i]} ({n})"
    return out


def _diagnostic_object(page: ExpandedPage, run_context: dict[str, Any] | None):
    """Return the diagnostic object a ``health:``/``rivers:``/``inputs:`` page uses.

    Computed on the first page that needs it and shared by every later page with the
    same kind, source, time window and cache flag -- the object holds the expensive
    read, a page only picks which figures of it to draw. The shared state is
    ``run_context`` (``{"joins": ..., "objects": {...}}``, owned by the runner):
    ``joins`` is the list of job-segment start dates handed to
    :func:`ocean_skill.run_health.health`, or a no-argument callable returning it,
    called at most once and only when a health page is built. A failed computation is
    remembered too, so each page of a broken source fails at once instead of repeating
    the read.
    """
    import json

    ctx = run_context if run_context is not None else {"joins": None, "objects": {}}
    objects = ctx.setdefault("objects", {})
    kwargs = page.kwargs
    key = (
        page.kind,
        json.dumps(kwargs.get("sources", kwargs.get("source")), default=str),
        json.dumps([kwargs.get("time"), kwargs.get("max_steps")], default=str),
        page.cache,
    )
    if key not in objects:
        try:
            if page.kind == "inputs":
                from ocean_skill.input_integrity import input_health

                objects[key] = input_health(
                    kwargs["sources"],
                    time=kwargs.get("time"),
                    max_steps=kwargs.get("max_steps"),
                    cache=page.cache,
                )
            elif page.kind == "health":
                from ocean_skill.run_health import health

                joins = ctx.get("joins")
                if callable(joins):
                    joins = ctx["joins"] = joins()
                objects[key] = health(
                    kwargs["source"],
                    time=kwargs.get("time"),
                    joins=joins,
                    cache=page.cache,
                )
            else:
                from ocean_skill.river_inputs import rivers

                objects[key] = rivers(kwargs["source"], time=kwargs.get("time"))
        except Exception as exc:
            objects[key] = exc
    obj = objects[key]
    if isinstance(obj, Exception):
        raise obj
    return obj


def _build_diagnostic(page: ExpandedPage, run_context: dict[str, Any] | None):
    """Draw a ``health:``/``rivers:``/``inputs:`` page; see :func:`_diagnostic_object`.

    The page's ``plot:`` is not used: the diagnostic figures take no options.
    """
    obj = _diagnostic_object(page, run_context)
    draw = page.kwargs["draw"]
    results = list(obj.figures(draw))
    if not results:
        raise PageSkipped(
            f"{page.kind}: draw: {draw} has nothing to draw for "
            f"{page.kwargs.get('source', page.kwargs.get('sources'))!r} (it has no "
            "such variables or data)"
        )
    if draw in ("integrity", "extremes"):
        table = obj.table()
        if table is not None and len(table):
            page.summary_table = table
    page.results = list(obj.problems())
    return results


def build(
    page: ExpandedPage,
    *,
    pooled_records: list[MetricRecord] | None = None,
    run_context: dict[str, Any] | None = None,
):
    """Draw one expanded page. Returns a list of ``(suffix, Figure)`` pairs.

    A ``section`` page has nothing to draw and raises :class:`ValueError` -- see the
    module docstring. An ``XY``/``TS`` page builds each member with
    :func:`ocean_skill.field.field` (its own ``cache`` flag, decided by
    :func:`expand`), then draws them together with :class:`~ocean_skill.xy.XY`/
    :class:`~ocean_skill.xy.TS` -- one figure.

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

    A ``compare`` page with ``draw:`` other than ``plot`` (see
    :data:`COMPARE_DRAWS`) is drawn by that :class:`~ocean_skill.comparison
    .ComparisonSet` method instead -- ``plot:`` forwarded as its keyword
    arguments -- as one figure of the whole set, so there is no per-family split.

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

    A ``health``/``rivers``/``inputs`` page draws one figure group of a diagnostic
    object that ``run_context`` shares between pages (see
    :func:`_diagnostic_object`); it records the object's problems on
    :attr:`ExpandedPage.results` and, for the tables it
    draws, :attr:`ExpandedPage.summary_table`, but no metric records, so the final
    ``summary:`` page stays about the comparisons. One with nothing to draw raises
    :class:`PageSkipped`.
    """
    if page.kind == "section":
        raise ValueError(
            f"page {page.title!r}: a section: page is a divider in report.pdf, not "
            "a figure -- the runner hands it to PdfReport.section(), never build()"
        )

    import ocean_skill as osk

    if page.contours and "contours" in page.plot:
        raise ValueError(
            f"page {page.title!r}: plot: contours: clashes with the page's own "
            "contours: key -- set one of them"
        )

    if page.kind == "field":
        if "cache" in page.kwargs:
            raise ValueError(
                f"page {page.title!r}: field: cache: is not supported -- "
                "caching is controlled by the suite's own cache:/cache_dir: "
                "settings, not a per-page kwarg"
            )

        def build_field(kwargs: dict[str, Any]):
            field_kwargs = {
                k: v for k, v in kwargs.items() if k not in ("source", "variable")
            }
            return osk.field(
                kwargs["source"],
                kwargs["variable"],
                cache=page.cache,
                **field_kwargs,
            )

        obj = build_field(page.kwargs)
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
        plot_kwargs = dict(page.plot)
        if page.contours is True:
            plot_kwargs["contours"] = obj
        elif page.contours:
            plot_kwargs["contours"] = build_field(page.contours)
        fig = _draw(page, obj.plot, plot_kwargs)
        return [("", fig)]

    if page.kind in ("XY", "TS"):
        members = {}
        for label, spec in page.kwargs["members"].items():
            m = dict(spec)
            members[label] = osk.field(
                m.pop("source"), m.pop("variable"), cache=m.pop("cache"), **m
            )
        if page.kind == "XY":
            obj = osk.XY(
                members,
                x=page.kwargs["x"],
                y=page.kwargs["y"],
                regions=page.kwargs["regions"],
                at_center=page.kwargs["at_center"],
            )
        else:
            obj = osk.TS(
                members,
                regions=page.kwargs["regions"],
                at_center=page.kwargs["at_center"],
            )
        return [("", _draw(page, obj.plot, dict(page.plot)))]

    if page.kind == "compare":

        def build_compare(kwargs: dict[str, Any]):
            compare_kwargs = {k: v for k, v in kwargs.items() if k not in ("test",)}
            return osk.compare(
                test=kwargs["test"],
                skip_missing=True,
                cache=page.cache,
                **compare_kwargs,
            )

        cs = build_compare(page.kwargs)
        if len(cs) == 0:
            raise ValueError("every comparison in this page was skipped (skip_missing)")
        n_built = len(cs)
        cs, kept = _drop_all_nan(page, cs, osk=osk)
        # Each record carries its comparison's own label ("temperature mean", a
        # selects: entry's name), so a summary table names rows the way the figure does.
        page.metrics_records = [_labelled_record(c) for c in cs]
        families = sorted({c.family for c in cs})
        contour_cs = None
        if isinstance(page.contours, dict):
            contour_cs = build_compare(page.contours)
            if len(cs) != n_built:
                if len(contour_cs) != n_built:
                    # the lines are paired with the fill by position; with members
                    # gone and no one-to-one match left, any pairing would be a guess
                    raise ValueError(
                        f"page {page.title!r}: {n_built - len(cs)} of {n_built} "
                        "comparison(s) were left out as all-NaN, and the contours: "
                        f"overlay has {len(contour_cs)} -- they can no longer be "
                        "paired with the fill. Narrow the page so every comparison "
                        "has data."
                    )
                # they lose the same members
                contour_cs = osk.ComparisonSet(
                    [c for c, k in zip(contour_cs, kept, strict=True) if k]
                )
            _check_contour_families(
                page.title, families, [c.family for c in cs], contour_cs
            )
        if page.draw not in (None, "plot"):
            # The whole set as one figure -- metric diagrams and series pool every
            # family, so there is no per-family split (and no suffix) here.
            return [("", _draw(page, getattr(cs, page.draw), dict(page.plot)))]
        plot_opts = dict(page.plot)
        max_rows = plot_opts.pop("max_rows", DEFAULT_MAX_ROWS)
        results = []
        for fam in families:
            members = [c for c in cs if c.family == fam]
            overlay = (
                [c for c in contour_cs if c.family == fam]
                if contour_cs is not None
                else None
            )
            # Several rows in one figure get too small to read: a row family is cut into
            # figures of at most max_rows rows, in set order (a variable's statistics
            # stay together when max_rows is a multiple of how many there are).
            step = max_rows if max_rows and fam in ROW_FAMILIES else len(members)
            if step < len(members) and _pools_rows(plot_opts):
                # One colour scale over every row (shared_limits=True) or one
                # difference scale over every panel (panels="difference") is the whole
                # set's to compute, and the renderers take no limits from outside:
                # each figure would find its own, so the same variable would change
                # scale at the cut. The group is the whole set, so the cut is not made.
                warnings.warn(
                    f"page {page.title!r}: max_rows={max_rows} was not applied to "
                    f"{len(members)} {fam} rows because shared_limits/panels="
                    '"difference" put them on one colour scale, which separate '
                    "figures cannot share; they are drawn as one figure (set "
                    "max_rows: 0 to say so, or drop the shared scale to cut them)",
                    stacklevel=2,
                )
                step = len(members)
            for part, start in enumerate(range(0, len(members), step)):
                rows = slice(start, start + step)
                bucket = osk.ComparisonSet(members[rows])
                plot_kwargs = dict(plot_opts)
                if page.contours is True:
                    plot_kwargs["contours"] = bucket
                elif overlay is not None:
                    plot_kwargs["contours"] = osk.ComparisonSet(overlay[rows])
                if part == 0:
                    fig = _draw(page, bucket.plot, plot_kwargs)
                else:
                    fig = _draw_continued(page, bucket, plot_kwargs)
                suffix = ("" if len(families) == 1 else f"_{fam}") + (
                    f"_part{part + 1}" if part else ""
                )
                results.append((suffix, fig))
        return results

    if page.kind in ("health", "rivers", "inputs"):
        return _build_diagnostic(page, run_context)

    # summary
    pooled = pooled_records or []
    if not pooled:
        raise ValueError("no compare page produced results to summarize")
    labels = _distinct_labels(pooled)
    comparisons = [
        MetricRecord(r, label=label, units=r.get("units"))
        for r, label in zip(pooled, labels, strict=True)
    ]
    kinds = page.kwargs["kind"]
    max_rows = page.kwargs.get("max_rows", DEFAULT_TABLE_ROWS)
    diagram_kwargs = {
        k: v for k, v in page.kwargs.items() if k not in ("kind", "scope", "max_rows")
    }
    results = []
    for kind in kinds:
        suffix = "" if len(kinds) == 1 else f"_{kind}"
        if kind != "table":
            fig = osk.summary(comparisons, kind=kind, **diagram_kwargs)
            results.append((suffix, fig))
            continue
        # Keyed by the (distinct) labels, so rows are named as the figures name them
        # rather than by the table's own variable/depth rule.
        table = osk.table({c.label: c for c in comparisons})
        page.summary_table = table.to_dataframe(units=True)
        figs = table.to_figures(
            max_rows=max_rows,
            title=diagram_kwargs.get("title", page.title),
            common_notes=True,
        )
        results.extend(
            (suffix + (f"_part{i + 1}" if i else ""), fig) for i, fig in enumerate(figs)
        )
    return results
