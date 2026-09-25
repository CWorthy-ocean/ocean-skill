r"""Expand a :class:`~ocean_skill.config.SuiteConfig` into pages, then draw one.

Two halves, deliberately separate:

:func:`expand` turns the suite's ``pages:`` list into a flat list of plain
``dict``\ s -- one per drawn figure -- doing every bit of resolution that does
*not* need to read real data: ``for_each`` fanned out into its Cartesian product,
``{placeholder}`` strings filled in, the bare word ``latest`` and ``month: run``
resolved against the test source's own time axis (one cheap, coordinate-only read),
and a literal ``{"min", "max"}`` time window injected wherever a page would
otherwise mean "whatever the run happens to cover right now" -- so the expanded
list, written verbatim into a report's ``manifest.json``, is what actually
reproduces the figures. The result is JSON-serializable and takes no arguments a
Phase 2 consumer (a notebook, a dashboard) could not also supply.

:func:`build` draws one already-expanded page and returns a
:class:`matplotlib.figure.Figure` (plus the metric records a ``compare`` page
produced, for the summary page and the metrics CSV). This is the only place that
touches ``osk.field``/``osk.compare``/``osk.summary``, so the suite YAML's own
grammar can change without changing what any of those calls do.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from dataclasses import field as _dc_field
from itertools import product
from typing import Any

__all__ = ["MetricRecord", "build", "expand"]

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


# -- window injection & the cache flag -----------------------------------------------


def _inject_field_window(kwargs: dict[str, Any], t0: str, t1: str) -> bool:
    """Give a ``field`` page's select an explicit time window if it has none.

    Returns whether this page's own time now reaches the run's latest step --
    the "open window" pages a stale disk-cache entry could otherwise serve. Must
    run *before* the caller resolves a literal ``"latest"`` to an ISO string, or
    that resolved value looks like an ordinary, already-explicit time key here.
    """
    select = kwargs.setdefault("select", {})
    if "time" not in select:
        select["time"] = {"min": t0, "max": t1}
        return True
    return False


def _inject_compare_window(kwargs: dict[str, Any], t0: str, t1: str) -> bool:
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
        return True
    if "test" in select or "reference" in select:
        test_sel = select.setdefault("test", {})
        if "time" not in test_sel:
            test_sel["time"] = {"min": t0, "max": t1}
            return True
        return False
    return False


def _is_open_month(namespace: dict[str, Any], latest: Any) -> bool:
    month_ns = namespace.get("month")
    if month_ns is None:
        return False
    return (month_ns.year, month_ns.month) == (latest.year, latest.month)


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
    kind: str  # "field" | "compare" | "summary"
    kwargs: dict[str, Any]
    plot: dict[str, Any]
    cache: bool
    status: str = "pending"
    reason: str | None = None
    elapsed: float | None = None
    metrics_records: list[dict[str, Any]] = _dc_field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "kind": self.kind,
            "kwargs": self.kwargs,
            "plot": self.plot,
            "cache": self.cache,
        }


def expand(suite: Any) -> list[ExpandedPage]:
    """Turn ``suite.pages`` into a flat, fully-resolved list of :class:`ExpandedPage`.

    ``suite`` is a :class:`~ocean_skill.config.SuiteConfig`. Reads the test
    source's native time index at most once (coordinate-only; memoized here), used
    to resolve ``latest``/``month: run`` and to bound the literal windows injected
    into any page that would otherwise mean "however much of the run exists right
    now".
    """
    from ocean_skill import extrema

    defaults = dict(suite.defaults)
    test_source = defaults.get("test")

    index_cache: dict[str, Any] = {}

    def get_index(source: str) -> Any:
        if source not in index_cache:
            index_cache[source] = extrema._native_time_index(source)
        return index_cache[source]

    out: list[ExpandedPage] = []
    for page in suite.pages:
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

            if page.kind == "field":
                kwargs = _template_value(dict(page.field), namespace, title=title)
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
                was_latest = select.get("time") == "latest"
                open_window = was_latest or _inject_field_window(kwargs, t0, t1)
                if was_latest:
                    select["time"] = t1
                out.append(
                    ExpandedPage(
                        title=title,
                        kind="field",
                        kwargs=kwargs,
                        plot=plot,
                        cache=suite.cache and not open_window,
                    )
                )

            elif page.kind == "compare":
                kwargs = _template_value(dict(page.compare), namespace, title=title)
                kwargs.setdefault("test", test_source)
                source = kwargs["test"]
                if source is None:
                    raise ValueError(
                        f"page {title!r}: no test source (set defaults.test)"
                    )
                index = get_index(source)
                t0, t1 = index[0].isoformat(), index[-1].isoformat()
                select = kwargs.get("select")
                was_latest = (
                    isinstance(select, dict)
                    and "test" in select
                    and select["test"].get("time") == "latest"
                )
                open_window = was_latest or _inject_compare_window(kwargs, t0, t1)
                if was_latest:
                    kwargs["select"]["test"]["time"] = t1
                if not open_window and "month" in namespace:
                    open_window = _is_open_month(namespace, index[-1])
                out.append(
                    ExpandedPage(
                        title=title,
                        kind="compare",
                        kwargs=kwargs,
                        plot=plot,
                        cache=suite.cache and not open_window,
                    )
                )

            else:  # summary
                kwargs = _template_value(dict(page.summary), namespace, title=title)
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


def build(page: ExpandedPage, *, pooled_records: list[MetricRecord] | None = None):
    """Draw one expanded page. Returns a list of ``(suffix, Figure)`` pairs.

    Almost always one pair (``suffix=""``); a ``compare`` page whose comparisons
    span more than one plot family draws one figure per family instead (see
    :meth:`ocean_skill.comparison.ComparisonSet.plot`), each suffixed by its
    family name so no PNG is overwritten. A ``field`` page's ``variable=``/
    ``select=``/``aggregate=`` (and a ``compare`` page's own kwargs) are used
    exactly as :func:`ocean_skill.field.field`/:func:`ocean_skill.comparison.compare`
    already validate them -- this function adds nothing to that grammar.
    """
    import ocean_skill as osk

    if page.kind == "field":
        obj = osk.field(
            page.kwargs["source"],
            page.kwargs["variable"],
            select=page.kwargs.get("select"),
            aggregate=page.kwargs.get("aggregate"),
            cache=page.cache,
        )
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
