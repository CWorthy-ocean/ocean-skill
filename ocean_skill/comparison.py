"""The compare layer: ``Comparison`` (one pair) and ``compare`` (fan-out).

A :class:`Comparison` holds a reference and a test source for one variable plus a
selection; it reads both, reduces them to a comparable 2-D field, aligns them
(test → reference), and exposes the difference, metrics and a plot. Roles are assigned
here, not in the catalog: ``diff = test − reference``, alignment brings test onto
reference. :func:`compare` fans over the reference × test × variable × depth
cross-product and collects the results into a :class:`ComparisonSet` that can write one
tidy metrics table and one stacked figure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "SURFACE",
    "Comparison",
    "ComparisonSet",
    "as_select",
    "compare",
    "is_surface_request",
    "prepare_source",
    "summary",
]


#: Sentinel meaning "the model's own top level" as distinct from an explicit request
#: for the field interpolated to literal 0 m — see :func:`_prepare`.
SURFACE = "surface"


#: CF featureTypes whose data is a *place*, not a field: one position, a time axis, and
#: nothing to draw a map of. A comparison against one of these keeps its time axis and
#: draws as lines (the ``series`` family) — which is what "featureType drives the plot
#: recipe" means in practice for the non-gridded half of the catalog.
#:
#: ``profile``, ``timeSeriesProfile`` and ``trajectory`` are deliberately absent: each
#: needs its own recipe (a profile's axis is depth and its x is the value; a trajectory
#: is a position that moves, so one sampling point is wrong for it), and inferring
#: "series" for them would draw something plausible and wrong.
POINT_FEATURE_TYPES = frozenset({"timeSeries", "point", "station"})


def _feature_type(source: str) -> str | None:
    """Return a source's catalog featureType, or ``None`` if it is unresolvable."""
    from ocean_skill.catalog import resolve

    try:
        return resolve(source).metadata.get("featureType")
    except KeyError:
        return None


def _implied_over(reference: str) -> tuple[str | None, str]:
    """Return ``(over, why)`` — the axis a reference's featureType asks to keep.

    The featureType decides the recipe, as the README has promised since the beginning:
    a ``timeSeries`` reference has one position and a time axis, so time is the axis
    that survives and the comparison is a line plot rather than a map. Returning the
    *reason* alongside is the point — the choice is then recorded on the aligned result
    (``family_reason``), so a family that surprises someone can be traced to what
    decided it rather than guessed at.
    """
    feature = _feature_type(reference)
    if feature in POINT_FEATURE_TYPES:
        return "time", f"the reference's featureType is {feature!r}"
    return None, "the reference is gridded"


def _domain_of(source: str) -> tuple[float, float, float, float] | None:
    """Return ``(lon_min, lat_min, lon_max, lat_max)`` for a source's catalog extent.

    Used to draw the model-domain outline on a map (as in Abigale Wyatt's
    ``Obs_comparisons.ipynb``) without the caller having to know or repeat the model's
    bounding box. Returns ``None`` if the source is unresolvable or the catalog entry
    doesn't declare a geospatial extent, in which case no box is drawn.
    """
    from ocean_skill.catalog import resolve

    try:
        meta = resolve(source).metadata
    except KeyError:
        return None
    keys = (
        "geospatial_lon_min",
        "geospatial_lat_min",
        "geospatial_lon_max",
        "geospatial_lat_max",
    )
    lon_min, lat_min, lon_max, lat_max = (meta.get(k) for k in keys)
    if None in (lon_min, lat_min, lon_max, lat_max):
        return None
    # Catalogs may declare 0-360 (ROMS' native convention); maps are drawn in ±180, so
    # normalize. This assumes the domain doesn't itself straddle the anti-meridian.
    lon_min, lon_max = (((lo + 180) % 360) - 180 for lo in (lon_min, lon_max))
    return lon_min, lat_min, lon_max, lat_max


def as_select(select: Any) -> dict[str, Any]:
    """Return ``select`` as a dict, refusing anything else with a usable message.

    ``select="surface"`` is the natural slip, because that is very nearly how
    :func:`compare` spells it (``depths=("surface",)``) and because "surface" reads
    like a complete thought on its own. Passing it to ``dict()`` raises "dictionary
    update sequence element #0 has length 1; 2 is required", which names neither the
    parameter at fault nor the spelling that works — and it is raised from the
    constructor, so there is no traceback line pointing at ``select`` either.
    """
    if select is None:
        return {}
    if isinstance(select, dict):
        return dict(select)
    raise TypeError(
        f"select must be a dict of axis -> selection, got {select!r}. "
        f'Did you mean select={{"depth": {select!r}}}? '
        'Other axes take the same form: {"time": "2012-01"}, '
        '{"lat": {"min": 20, "max": 30}}.'
    )


def is_depth_band(depth: Any) -> bool:
    """Report whether ``depth`` asks for an average over a band rather than a level.

    ``{"min": 0, "max": 10}`` — the YAML-friendly spelling of a range, since YAML has
    no slice literal. A band is averaged, not interpolated: see
    :func:`ocean_skill.roms.depth_average`.
    """
    return isinstance(depth, dict) and {"min", "max"} <= set(depth)


def is_surface_request(depth: Any) -> bool:
    """Report whether ``depth`` means "surface" rather than a literal depth in metres.

    Unset (``None``) and the ``"surface"`` sentinel both mean "the model's own top
    level" (:func:`ocean_skill.roms.surface`). A literal ``0``/``0.0`` is a real depth
    request instead — interpolated like any other depth via
    :func:`ocean_skill.roms.to_depth`, which may legitimately come back all-NaN (with a
    warning) if the topmost cell centre already sits below 0 m. Conflating the two
    silently hid that distinction; now only the explicit sentinel gets the shortcut.
    """
    return depth is None or (isinstance(depth, str) and depth.lower() == SURFACE)


def _depth_label(depth: Any) -> str:
    """Format a depth for labels/repr: ``"surface"``, ``"0-10 m"`` or ``"<n> m"``."""
    if is_surface_request(depth):
        return SURFACE
    if is_depth_band(depth):
        return f"{float(depth['min']):g}-{float(depth['max']):g} m"
    return f"{float(depth):g} m"


def _variable_label(spec: Any) -> str:
    """Short display name for a variable spec, whether a name or a combination."""
    if isinstance(spec, str):
        return spec.split("_of_")[-1]
    if name := spec.get("standard_name"):
        return str(name).split("_of_")[-1]
    how, names = next(iter((k, v) for k, v in spec.items() if k != "standard_name"))
    return f"{how}({'+'.join(map(str, names))})"


def _short_variable_label(spec: Any) -> str:
    """Return a variable's name as a point label, short where the vocabulary knows one.

    :func:`_variable_label` is the fallback rather than the rule because it only knows
    how to strip a CF name apart, while :func:`ocean_skill.vars.short_name` knows what
    the package calls things — and a combination spec has no short name to look up.
    Shared by :func:`compare` and :func:`_pooled_labels` so a set's own labels and a
    pooled set's relabelling cannot drift apart.
    """
    from ocean_skill.vars import short_name

    return short_name(spec) if isinstance(spec, str) else _variable_label(spec)


#: Keys naming the vertical axis in a `select`, in any accepted spelling.
_VERTICAL_KEYS = frozenset({"depth", "Z", "vertical", "z"})


def _vertical_only(agg: dict[str, Any] | None) -> dict[str, Any]:
    """Return the part of an aggregation spec addressing the vertical axis."""
    return {k: v for k, v in (agg or {}).items() if k in _VERTICAL_KEYS}


def _without_vertical(agg: dict[str, Any] | None) -> dict[str, Any]:
    """Return the part of an aggregation spec for every axis but the vertical."""
    return {k: v for k, v in (agg or {}).items() if k not in _VERTICAL_KEYS}


def _selected_depth(select: dict[str, Any]) -> Any:
    """Return what a ``select`` asks for vertically, whichever spelling it used.

    ``compare`` writes ``"depth"``, but a ``Comparison`` built directly keeps the
    ``select`` it was given, and ``{"Z": 100}`` is as valid there as anywhere else. For
    labels the difference is not cosmetic: reading only ``"depth"`` reports a comparison
    at 100 m as ``surface``, and two comparisons at different depths as the same point.
    """
    for key in ("depth", "Z", "z", "vertical"):
        if key in select:
            return select[key]
    return SURFACE


#: What ``aggregate=None`` means: **reduce nothing**. There is deliberately no default
#: reduction any more.
#:
#: This used to be ``{"time": "mean"}``, which made ``select={"time": "2010-01"}``
#: silently return the January *mean* — a reasonable thing to want and a terrible thing
#: to assume, since the caller who wanted the 31 days had no way to tell it had happened
#: except by looking at the shape. It also worked directly against
#: :class:`~ocean_skill.field.Field`, whose reason to exist is the axis that survives.
#:
#: Nothing is assumed in the other direction either: a reduction is *required* the
#: moment an axis survives that the consumer cannot draw, and each consumer says what it
#: can take. A :class:`Comparison` must reduce to one map (:func:`_require_reduced`
#: below); a ``Field`` can draw one leftover axis as panels or frames and two as a grid,
#: refusing three (:func:`ocean_skill.field._facet_dims`). So you are asked to choose
#: exactly when the data forces a choice, and never averaged behind your back.
NO_AGGREGATION: dict[str, Any] = {}


def _require_reduced(da, role: str, source: str) -> None:
    """Raise unless ``da`` is a single map, naming what is left and how to collapse it.

    A comparison regrids one field onto another and differences them, so it has to be
    2-D; :func:`ocean_skill.align._require_2d` enforces the same thing one step later,
    where it protects xesmf. This exists in front of it for two reasons: it can name the
    *source* whose lane is at fault, which align cannot (it sees two anonymous arrays),
    and it runs before :func:`prepare_source` computes anything, so being told to choose
    costs nothing rather than costing the whole vertical transform first.
    """
    from ocean_skill.align import _lat_name, _lon_name

    spatial: set[str] = set()
    for name in (_lon_name(da), _lat_name(da)):
        if name is not None:
            spatial |= {str(d) for d in da[name].dims}
    extra = [str(d) for d in da.dims if d not in spatial]
    if not extra:
        return
    sizes = ", ".join(f"{d}={da.sizes[d]}" for d in extra)
    raise ValueError(
        f"the {role} lane ({source!r}) still has {sizes} beyond its horizontal axes, "
        "so it is not a single map and cannot be compared. A comparison differences "
        "two fields on one grid, so it needs the other axes collapsed — say how:\n"
        f'  aggregate={{"{extra[0]}": "mean"}}          one map, the mean over '
        f"{extra[0]}\n"
        f'  aggregate={{"{extra[0]}": {{"reduce": "quantile", "q": 0.9}}}}   or any '
        "other reduction\n"
        f'  select={{"{extra[0]}": <one value>}}    or narrow it to a single value '
        "instead\n"
        f'  over="{extra[0]}"                     or keep it and score against it, '
        "cell by cell\n"
        "There is no default reduction: a comparison will not average an axis you did "
        "not ask it to. The third line gives a map per metric instead of one map per "
        "field; to keep the axis and look at one source over it, use osk.field() "
        "rather than a comparison."
    )


def _prepare(
    obj,
    meta: dict[str, Any],
    variable: Any,
    select: dict[str, Any],
    aggregate: dict[str, Any] | None = None,
):
    """Reduce a source to one comparable 2-D field (variable, aggregation, depth).

    ``variable`` is anything :func:`ocean_skill.operators.resolve_variable` accepts:
    a name, or a combination such as ``{"sum": ["spChl", "diatChl", "diazChl"]}``.
    ``aggregate`` is a :func:`ocean_skill.operators.aggregate` spec; ``None`` reduces
    **nothing** (see :data:`NO_AGGREGATION`), leaving whatever ``select`` left standing
    for the caller's consumer to draw or refuse.

    Resolving the variable *first*, and bailing out when it is absent, is
    deliberate: falling through to the whole dataset is both wasteful and unsafe —
    a bare ``.mean("time")`` chokes on whatever non-numeric fields ride along (ROMS'
    ``spherical`` flag), so "variable not found" must fail closed.
    """
    from ocean_skill import operators, roms, tabular, units

    # A point source arrives as a DataFrame (osk.read's contract for a station), and
    # everything from resolve_variable down speaks xarray. Converting here rather than
    # in read() keeps that contract, and rather than in align() because a Field over one
    # station wants the same lane and never reaches align.
    if tabular.is_frame(obj):
        obj = tabular.to_dataset(obj, meta)

    depth = next((select[k] for k in _VERTICAL_KEYS if k in select), None)
    surface = is_surface_request(depth)
    band = is_depth_band(depth)
    agg = NO_AGGREGATION if aggregate is None else aggregate

    da = operators.resolve_variable(obj, variable)
    if da is None:
        return None, None

    # Order matters twice over. Selection precedes reduction, or "the mean of
    # January" would average the whole record. And the *non-vertical* reduction runs
    # before the vertical step, so an expensive s-coordinate transform sees as few
    # fields as the reduction leaves it -- one, for a time mean; every step, now that
    # no reduction is the default, which is the cost of asking for every step. The
    # vertical part of the aggregation then collapses whatever the vertical selection
    # left standing.
    horizontal = {k: v for k, v in select.items() if k not in _VERTICAL_KEYS}
    da = operators.select(da, horizontal)
    da = operators.aggregate(da, _without_vertical(agg))

    if meta.get("model") == "roms":
        # The vertical transform needs a Dataset carrying the grid; a DataArray
        # brings its coordinates (h, mask, Cs_r, ...) along, so this round trip
        # keeps everything roms.surface/to_depth reads.
        name = da.name or "field"
        sub = da.to_dataset(name=name)
        # A DataArray only carries coordinates sharing its dimensions, so the
        # interface-grid variables (on s_w, which a tracer has no part of) are
        # dropped by to_dataset. They are exactly what depth_average needs.
        #
        # Static grid fields only -- deliberately not `zeta`, which still carries the
        # time dimension this field has already been averaged over; re-attaching it
        # would make z_rho time-varying against a time-less field and break the xgcm
        # transform. Both depth routines fall back to zeta=0, which is the
        # approximation already in force here and is small against metre-scale cells.
        for grid_var in ("sigma_w", "Cs_w", "sigma_r", "Cs_r", "h"):
            if grid_var in obj.variables and grid_var not in sub.variables:
                sub = sub.assign({grid_var: obj[grid_var]})
        if surface:
            sub = roms.surface(sub, meta)
        elif band:
            # A band is averaged over native cells with thickness weights, not
            # interpolated: above the shallowest cell *centre* -- 7 m down in deep
            # water on this grid -- there is nothing to interpolate from, so a
            # target grid over 0-10 m would be mostly NaN offshore.
            # A *selection*: keeps the cells and their thickness weights, so the
            # vertical aggregation below decides how to collapse them.
            sub = roms.depth_band(sub, meta, depth["min"], depth["max"])
        else:
            # A list interpolates to several levels in one field, which the vertical
            # aggregation then collapses; a scalar gives one level and no axis.
            targets = (
                [float(d) for d in depth]
                if isinstance(depth, list | tuple)
                else float(depth)
            )
            sub = roms.to_depth(sub, meta, targets)
        da = sub[name]
        # Squeeze only a single interpolated level: a scalar depth request collapses
        # the axis by itself (as `.sel` does everywhere), while a list or band leaves
        # several levels for the vertical aggregation to reduce. Squeezing
        # unconditionally used to discard every level but the first, silently.
        if "z" in da.dims and da.sizes["z"] == 1:
            da = da.isel(z=0)
    else:
        # observational depth axes vary: real metres, or an index with depths alongside
        zname = next(
            (n for n in ("depth", "depth_surface", "lev") if n in da.dims), None
        )
        if zname is not None:
            levels = (
                np.asarray(obj["Depth"])
                if "Depth" in obj.variables
                else np.asarray(da[zname])
            )
            if band:
                # Observational products report at standard levels, so a band is just
                # the levels inside it -- no cell thicknesses to weight by, and
                # inventing some would imply structure the product does not claim.
                # Left standing for the vertical aggregation, like the model side.
                inside = np.where((levels >= depth["min"]) & (levels <= depth["max"]))[
                    0
                ]
                if inside.size == 0:  # band falls between levels; take the nearest
                    inside = [int(np.abs(levels - depth["min"]).argmin())]
                attrs = dict(da.attrs)
                da = da.isel({zname: list(inside)})
                da.attrs = attrs
                da.attrs["actual_depth"] = float(np.mean(levels[list(inside)]))
            elif isinstance(depth, list | tuple):
                keep = [int(np.abs(levels - float(d)).argmin()) for d in depth]
                attrs = dict(da.attrs)
                da = da.isel({zname: keep})
                da.attrs = attrs
                da.attrs["actual_depth"] = float(np.mean(levels[keep]))
            else:
                target = 0.0 if surface else float(depth)
                k = int(np.abs(levels - target).argmin())
                da = da.isel({zname: k})
                da.attrs["actual_depth"] = float(levels[k])

    if da is None:
        return None, None
    # Now the vertical reduction, on whatever the vertical selection left: one level
    # (already collapsed), several interpolated levels, or a weighted band.
    da = operators.aggregate(da, _vertical_only(agg))
    # A station carries its instrument depth as a scalar coordinate rather than an axis
    # to select from (see ocean_skill.tabular.depth_of), so the depth actually compared
    # is read off the lane itself. Reported the same way as an observational level's:
    # through prepare_source, the lane cache, and the metrics row's `obs_depth`.
    if "actual_depth" not in da.attrs:
        if "depth" in da.coords and not da["depth"].dims:
            da.attrs["actual_depth"] = float(da["depth"])
        elif da.attrs.get("depth_m") is not None:
            # A station's depth also rides on the variable's attrs, which is what is
            # left once a reduction has dropped a coordinate along time -- so the
            # metrics row still reports the depth the comparison was made at.
            da.attrs["actual_depth"] = float(da.attrs["depth_m"])
    return units.convert_units(da), da.attrs.get("actual_depth")


#: Size (bytes) past which :func:`prepare_source` says so before loading a lane. Not a
#: cap: a year of daily output really is that large and refusing to read it would be
#: worse than taking a while over it. But the load is eager (see below), so an extra
#: axis nobody meant to keep is worth catching before the memory is spent, not after.
LOAD_WARN_BYTES = 2 * 1024**3


def prepare_source(
    source: str,
    variable: Any,
    select: dict[str, Any] | None,
    aggregate: dict[str, Any] | None,
    *,
    use_cache: bool = True,
    refresh: bool = False,
    require_reduced: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    time_window: tuple[Any, Any] | None = None,
):
    """Reduce one source to its prepared field, via the lane cache.

    Keyed on this source alone, so the same model/variable/depth is prepared once
    however many references it is compared against — the vertical transform is the
    pipeline's most expensive step and does not depend on the other side of the
    comparison at all. That independence is also why this is a module-level function
    rather than a :class:`Comparison` method: a model-only
    :class:`~ocean_skill.field.Field` wants exactly this lane and exactly this cache
    entry, and a second copy of the keying rules would be a second chance to key them
    differently.

    ``require_reduced`` names this lane's role (``"test"``/``"reference"``) for a
    :func:`_require_reduced` check, which a comparison wants and a
    :class:`~ocean_skill.field.Field` must not have: the axis a comparison cannot
    tolerate is the one a field exists to draw. Checked here rather than by the caller
    because here is the only place it is free — dimensions are known from the lazy
    graph, so an unreduced lane is refused before the ``.load()`` below spends the
    vertical transform on every step of it. It is checked on a cache *hit* too: the key
    below does not include it, deliberately (the same field is the same field), so an
    entry written by a caller that tolerates a surviving axis would otherwise let a
    caller that does not sail straight past its own gate.

    ``bbox`` crops the lane to ``(lon_min, lat_min, lon_max, lat_max)`` plus
    :data:`ocean_skill.align.DEFAULT_PAD` *before* the load below, and joins the cache
    key. :func:`ocean_skill.align.align` does this crop anyway, but only once both lanes
    are in memory — fine for a single global map, the dominant cost for a global product
    that kept a long time axis. Passing the test lane's own extent here changes nothing
    about the result and a great deal about the peak memory. It does cost this
    lane its cross-model reuse (a reference cropped to one model's domain is not the one
    another model wants), which is why the caller passes it only when it is worth that.

    Returns ``(DataArray, actual_depth)``, or ``(None, None)`` if the source does not
    carry the variable.
    """
    import ocean_skill as osk
    from ocean_skill import cache as _cache
    from ocean_skill.catalog import resolve

    key_select: dict[str, Any] = {**(select or {}), "_aggregate": aggregate}
    if bbox is not None:
        # rounded so that float noise in an extent does not fragment the cache
        key_select["_bbox"] = [round(float(b), 4) for b in bbox]
    key = _cache.key_for_prepared(
        source=source,
        variable=variable,
        select=key_select,
    )
    if use_cache and not refresh:
        hit = _cache.load_field(key)
        if hit is not None:
            if hit[0] is not None and require_reduced:
                _require_reduced(hit[0], require_reduced, source)
            return hit

    da, depth = _prepare(
        osk.read(source),
        resolve(source).metadata,
        variable,
        dict(select or {}),
        aggregate,
    )
    if da is not None and require_reduced:
        # before .load(), while it is still free -- see the docstring
        _require_reduced(da, require_reduced, source)
    if da is not None and bbox is not None:
        from ocean_skill.align import subset_to_bbox

        da = subset_to_bbox(da, bbox)
    if da is not None and time_window is not None:
        from ocean_skill.align import subset_to_time

        da = subset_to_time(da, time_window)
    if da is not None and da.nbytes > LOAD_WARN_BYTES:
        import warnings

        from ocean_skill import _stacklevel

        warnings.warn(
            f"reading {source!r} into memory as {da.nbytes / 1024**3:.1f} GB "
            f"({dict(da.sizes)}). Narrow it with select= (e.g. "
            '{"time": "2012"}), coarsen it with aggregate= (e.g. '
            '{"time": {"resample": "1MS", "reduce": "mean"}}), or accept the wait — '
            "this happens once per lane per process, and the result is cached.",
            stacklevel=_stacklevel.find(),
        )
    # Computed here rather than left lazy: the vertical transform is the most
    # expensive step in the pipeline, and a dask-backed lane re-runs it on every
    # consumer that touches values -- the metrics, then each panel of each plot.
    # to_zarr below computes the same graph anyway and discards the result, so
    # this costs nothing when caching is on and saves a full recompute when a
    # save fails or caching is off. See align() for the same move on the pair.
    if da is not None:
        da = da.load()
    # A miss is cached; "this source doesn't carry this variable" is not, since
    # that is cheap to rediscover and would otherwise persist past a fixed catalog.
    if use_cache and da is not None:
        _cache.save_field(key, da, depth)
    return da, depth


#: Valid pairs a cell needs before its metrics are reported. Named apart from
#: ``metrics.compute``'s ``min_samples`` on purpose: that one counts *cells over space*
#: and wants a few dozen, this counts *time steps at one cell* and wants a handful. One
#: word for two quantities in one call signature is how the wrong one gets passed.
DEFAULT_MIN_PAIRS = 5


class Comparison:
    """One reference↔test comparison for a single variable at a single depth.

    Ordinarily both sources are reduced to a single map and the comparison is that pair
    plus their difference. Naming an axis in ``over`` instead keeps that axis: the lanes
    are matched along it (:func:`ocean_skill.align.match_axis`), and every metric is
    then computed *cell by cell* along it, giving a map of bias, a map of correlation.
    So the axis a comparison normally refuses becomes the one it scores against, and the
    figure stops being test-vs-reference and becomes a panel per metric (:meth:`maps`).
    """

    def __init__(
        self,
        *,
        reference: str,
        test: str,
        variable: Any,
        select: dict[str, Any] | None = None,
        aggregate: dict[str, Any] | None = None,
        method: str = "conservative_normed",
        over: str | None = None,
        time_method: str = "auto",
        tolerance: float | None = None,
        bin_anchor: str = "auto",
        min_pairs: int = DEFAULT_MIN_PAIRS,
        metrics: tuple[str, ...] | None = None,
        label: str | None = None,
        cache: bool | None = None,
    ):
        from ocean_skill.vocabulary import resolve_and_report

        self.reference_name = reference
        self.test_name = test
        # A plain name resolves through the vocabulary (short name, canonical
        # standard_name, or any alias, in any case) so everything downstream sees
        # one consistent name. A dict is a combination spec for
        # ocean_skill.operators and is carried through as given.
        self.variable = (
            resolve_and_report(variable, context="Comparison variable=")
            if isinstance(variable, str)
            else variable
        )
        self.select = as_select(select)
        self.aggregate = aggregate
        self.method = method
        # An explicit over= always wins; otherwise the reference's featureType decides,
        # since a station reference has a time axis and no map to draw. The reason is
        # kept so the family this ends up choosing can be traced to what chose it.
        if over is None:
            over, self.over_reason = _implied_over(reference)
        else:
            self.over_reason = "over= as asked"
        self.over = over
        self.time_method = time_method
        self.tolerance = tolerance
        self.bin_anchor = bin_anchor
        self.min_pairs = min_pairs
        self.metric_names = tuple(metrics) if metrics else None
        self.label = label
        # None = follow the global setting (on unless osk.cache.disable()); an
        # explicit True/False overrides it for this comparison only.
        self.cache = cache
        self._aligned = None
        self._metrics = None
        self._maps = None
        self._actual_depth = None

    @property
    def standard_name(self) -> str | None:
        """The CF name this comparison represents, for colormaps/labels/metrics.

        A combination carries it explicitly (``{"sum": [...], "standard_name": ...}``);
        without one there is no single CF name, and downstream falls back to defaults.
        """
        from ocean_skill.operators import DERIVED

        spec = self.variable
        if isinstance(spec, str):
            # A DERIVED key is a name for a spec, not a CF name -- expand it, or
            # colormaps, plot labels and the metrics table all see the key.
            spec = DERIVED.get(spec, spec)
        return spec if isinstance(spec, str) else spec.get("standard_name")

    @property
    def _cache_key(self) -> str:
        from ocean_skill import cache as _cache

        # The matching knobs join the key the way `_aggregate` does -- smuggled into
        # `select` rather than added to key_for's signature, since they narrow *which*
        # aligned pair this is by exactly the same logic. `min_pairs` deliberately stays
        # out: it masks the metric maps, which are cheap and not cached, and leaves the
        # aligned pair itself untouched.
        extra = (
            {}
            if self.over is None
            else {
                "_over": self.over,
                "_time_method": self.time_method,
                "_tolerance": self.tolerance,
                "_bin_anchor": self.bin_anchor,
            }
        )
        return _cache.key_for(
            test=self.test_name,
            reference=self.reference_name,
            variable=self.variable,
            select={**self.select, "_aggregate": self.aggregate, **extra},
            method=self.method,
        )

    def _use_cache(self) -> bool:
        """Whether this comparison caches: its own setting, else the global one."""
        from ocean_skill import cache as _cache

        return _cache.enabled() if self.cache is None else self.cache

    def _prepare_lane(
        self,
        source: str,
        use_cache: bool,
        refresh: bool,
        role: str = "test",
        bbox: tuple[float, float, float, float] | None = None,
        time_window: tuple[Any, Any] | None = None,
    ):
        """Reduce one source to its comparable field, via the lane cache.

        ``role`` names the lane in the error raised if the reduction left it more than a
        map — the one thing a comparison's use of :func:`prepare_source` needs that a
        field's does not. With ``over`` set that check is *not* wanted: the axis it
        would refuse is the one this comparison exists to score against, and
        :func:`ocean_skill.align.align` still refuses any further one.

        ``bbox`` crops the lane before it is read into memory; see
        :func:`prepare_source`.
        """
        return prepare_source(
            source,
            self.variable,
            self.select,
            self.aggregate,
            use_cache=use_cache,
            refresh=refresh,
            require_reduced=None if self.over else role,
            bbox=bbox,
            time_window=time_window,
        )

    # -- pipeline ---------------------------------------------------------------
    def align(self, *, refresh: bool = False):
        """Read both sources, reduce them, and regrid test onto reference.

        The result is always computed, never lazy — see the ``.load()`` note below —
        so repeat consumers (metrics, a redrawn figure) read values rather than
        re-running the pipeline that produced them.

        The result is cached to disk (see :mod:`ocean_skill.cache`) and reused on a
        later call with the same sources, variable, selection and method — including
        from a fresh process, which is the case that matters, since an in-process
        repeat is already served by :attr:`aligned`'s own memo. Pass
        ``refresh=True`` to recompute and overwrite a stale entry, or construct the
        comparison with ``cache=False`` to bypass disk entirely.
        """
        from ocean_skill import align as _align
        from ocean_skill import cache as _cache

        use_cache = self._use_cache()
        if use_cache and not refresh:
            hit = _cache.load(self._cache_key)
            if hit is not None:
                self._aligned = hit
                # actual_depth rides along in attrs precisely so a cached result
                # restores the same state a freshly computed one would have.
                self._actual_depth = hit.attrs.get("actual_depth")
                return self._aligned

        # The test lane goes first when an axis is being kept, so the reference can be
        # cropped to its extent *before* being read (see prepare_source's bbox=).
        # align() crops it anyway, but only once both lanes are in memory, and a product
        # that kept a year of daily maps is the wrong thing to hold whole. Exact, not an
        # approximation: the bbox and the pad are the ones align() would have used.
        t, _ = self._prepare_lane(self.test_name, use_cache, refresh, role="test")
        bbox = None
        window = None
        if self.over is not None and t is not None:
            from ocean_skill.align import bbox_of, time_span_of

            bbox = bbox_of(t)
            # ...and the same crop along time. Cropping the region but not the window
            # still reads the whole record: MUR over a regional model's footprint is a
            # workable map per step and 2.2 TB across its 8838 daily ones.
            window = time_span_of(t)
        r, r_depth = self._prepare_lane(
            self.reference_name,
            use_cache,
            refresh,
            role="reference",
            bbox=bbox,
            time_window=window,
        )
        if r is None or t is None:
            missing = self.reference_name if r is None else self.test_name
            raise KeyError(f"{self.variable!r} not available in {missing!r}")
        self._actual_depth = r_depth
        # .load() so a computed result is a computed result: a cache hit hands back
        # eager arrays (open_zarr(...).load()), and a miss must leave the comparison
        # in the same state or the session that *fills* the cache is the slowest one
        # -- every plot re-reading the sources through a graph that was already
        # evaluated once to write the entry.
        self._aligned = _align.align(
            t,
            r,
            method=self.method,
            test_name="test",
            reference_name="reference",
            over=self.over,
            time_method=self.time_method,
            tolerance=self.tolerance,
            bin_anchor=self.bin_anchor,
            metadata=self._reference_metadata(),
        ).load()
        if r_depth is not None:
            self._aligned.attrs["actual_depth"] = r_depth
        if use_cache:
            _cache.save(self._cache_key, self._aligned)
        return self._aligned

    @property
    def aligned(self):
        """The aligned pair (test, reference, difference); computed on first access."""
        if self._aligned is None:
            self.align()
        return self._aligned

    data = aligned  # alias: prepared arrays for bespoke plotting

    def _reference_metadata(self) -> dict[str, Any] | None:
        """Return the reference's catalog metadata, or ``None`` if it will not resolve.

        Only the axis matching wants it — a ``period`` there says whether the
        reference's steps are averages or instants — and a stubbed or unregistered
        source should not be a reason a comparison cannot run.
        """
        from ocean_skill.catalog import resolve

        try:
            return resolve(self.reference_name).metadata
        except Exception:
            # Any resolution failure is the same failure here: an unknown source, an
            # unreadable catalog, a stub in a test. None is a reason a comparison
            # cannot run — the metadata only refines a default the matching can reach
            # on its own, and which it says out loud when it has to.
            return None

    @property
    def _scored_axis(self) -> str | None:
        """The axis name the aligned pair kept, as the reference names it."""
        return self.aligned.attrs.get("scored_over") if self.over else None

    @property
    def is_series(self) -> bool:
        """Whether this comparison is a place through time rather than a field.

        Read off the *aligned pair*, not off the featureType that led to it: the
        featureType chooses the recipe, but what a renderer can draw is decided by what
        the alignment actually produced. A station reference leaves no horizontal
        dimension to map, whatever the catalog said — and a catalog that said the wrong
        thing shows up here as a family that surprises someone, which is why
        :attr:`family_reason` records both halves.
        """
        from ocean_skill.align import point_of

        return point_of(self.aligned["reference"]) is not None

    @property
    def family(self) -> str:
        """The plot family this comparison's own shape admits.

        Three cases, in the order the data decides them: a place through time draws as
        ``series``, a pair scored over an axis draws as ``skill_map``, and a pair of
        single maps draws as ``field_row``. No argument selects it — the same rule
        :meth:`ocean_skill.field.Field.plot` follows between a map and a facet grid.
        """
        if self.is_series:
            return "series"
        return "field_row" if self.over is None else "skill_map"

    @property
    def family_reason(self) -> str:
        """Why :attr:`family` is what it is, in one sentence.

        Worth carrying because two things decide it and they can disagree: the catalog's
        featureType picks the recipe, and the aligned pair's own shape is what a
        renderer can honour. When a comparison draws as something unexpected — a
        ``timeSeries`` entry that is really gridded, a scored comparison that collapsed
        to a point — this says which of the two was responsible.
        """
        if self.is_series:
            return f"drawn as lines: {self.over_reason}, so the time axis is kept"
        if self.over is not None:
            return f"drawn as metric maps: {self.over_reason}"
        return "drawn as test | reference | difference: no axis is being scored"

    def maps(self, *names: str):
        """One 2-D map per metric, each computed cell by cell along ``over``.

        The pointwise counterpart of :meth:`metrics`: the same registry entries
        (:data:`ocean_skill.metrics.REGISTRY`), reduced over the scored axis alone
        instead of over everything, so ``bias`` becomes *where* the model is biased and
        ``corr`` becomes *where* it tracks the observations. Defaults to
        :data:`ocean_skill.metrics.DEFAULT_MAP_METRICS`; any registered name works.

        Not area-weighted, and that is not an omission: one cell's series sits at one
        latitude, so cos(lat) is a constant that cancels out of every metric here. The
        weighting belongs to the space-and-time scalar in :meth:`metrics`, which is
        exactly where it is applied.

        Cells with fewer than ``min_pairs`` valid pairs are masked in *every* map (the
        ``n`` map excepted, being the count itself): a correlation from three cloud-free
        days is noise, and a figure where ``bias`` and ``corr`` cover different cells
        cannot be read. How many cells that removed is a warning, not something written
        on the figure.
        """
        import warnings

        import numpy as np
        import xarray as xr

        from ocean_skill import _stacklevel
        from ocean_skill import metrics as _metrics

        if self.is_series:
            raise ValueError(
                "this comparison is at one place, so there is no map to make: a "
                "per-cell metric over one cell is the number metrics() already gives. "
                "Use .metrics() for the numbers and .plot() for the series."
            )

        requested = tuple(names) or self.metric_names or _metrics.DEFAULT_MAP_METRICS
        if self.over is None:
            raise ValueError(
                "metric maps need an axis to score along, and this comparison reduced "
                'them all. Build it with over= (e.g. compare(..., over="time")) to '
                "keep the time axis and score against it cell by cell."
            )
        if self._maps is not None and set(requested) <= set(self._maps.data_vars):
            return self._maps[list(requested)]

        axis = self._scored_axis
        wanted = tuple(dict.fromkeys((*requested, "n")))
        evaluated = _metrics.evaluate(
            self.aligned,
            wanted,
            dim=axis,
            test_name="test",
            reference_name="reference",
            weighted=False,
        )
        counts = evaluated["n"]
        enough = counts >= self.min_pairs
        maps = xr.Dataset(
            {
                name: (value if name == "n" else value.where(enough))
                for name, value in evaluated.items()
            }
        )
        covered = int((counts > 0).sum())
        dropped = covered - int(enough.sum())
        if covered and dropped:
            median = (
                float(np.median(counts.values[counts.values >= self.min_pairs]))
                if enough.any()
                else 0.0
            )
            warnings.warn(
                f"{dropped} of {covered} cells with any data had fewer than "
                f"{self.min_pairs} valid pairs along {axis!r} and are masked in every "
                f"metric map ({dropped / covered:.0%} of the covered domain; the cells "
                f"that remain have a median of {median:g} pairs). Cloud-gapped "
                "references do this; lower min_pairs to keep them, at the cost of "
                "metrics computed from very short series.",
                stacklevel=_stacklevel.find(),
            )
        self._maps = maps
        return maps[list(requested)]

    def metrics(self, **extra: Any) -> dict[str, Any]:
        """Compute (and cache) the metric record for this comparison.

        Every dimension is reduced, so this is one number per metric for the whole
        comparison — and when an axis is being scored ``over``, that means *space and
        that axis together*: the overall value the maps in :meth:`maps` decompose. It is
        computed on the same cells those maps show (the ``min_pairs`` mask applies here
        too), or the number printed beside a figure would describe a different domain
        from the figure.
        """
        from ocean_skill import metrics as _metrics

        if self._metrics is None:
            aligned = self.aligned
            # The mask is per *cell*: it exists so the scalar describes the same domain
            # the maps show. A station has one cell and no maps, so there is nothing to
            # mask and asking for them would raise.
            if self.over is not None and not self.is_series:
                enough = self.maps("n")["n"] >= self.min_pairs
                aligned = aligned.where(enough)
            self._metrics = _metrics.compute(
                aligned,
                test_name="test",
                reference_name="reference",
                # A station has one latitude, so cos(lat) is a constant: the arithmetic
                # is identical either way, but the row would claim an area weighting
                # that never happened. What `n` counts is steps here, not cells.
                **(
                    {"weighted": False, "sample_noun": "time steps"}
                    if self.is_series
                    else {}
                ),
                variable=self.standard_name or str(self.variable),
                test=self.test_name,
                reference=self.reference_name,
                depth=self.select.get("depth", SURFACE),
                obs_depth=self._actual_depth,
                regrid=self.method,
                **(
                    {"over": self.over, "min_pairs": self.min_pairs}
                    if self.over
                    else {}
                ),
                **extra,
            )
        return self._metrics

    def difference(self):
        """Return the ``test − reference`` field on the reference grid."""
        return self.aligned["difference"]

    def as_item(self) -> dict[str, Any]:
        """Return this comparison as a spec item.

        Two shapes, because a scored comparison is a different figure: with ``over`` set
        the item carries the metric maps and the overall record that annotates them, and
        with it unset the aligned trio the ``test | reference | difference`` row draws.
        """
        common = {
            "metrics": self.metrics(),
            "units": self.aligned["reference"].attrs.get("units"),
            "standard_name": self.standard_name,
            "label": self.label,
            # this comparison's own source names, for its row's column titles —
            # not necessarily the same pair as other rows in the same set (a
            # compare() fan-out commonly pairs one variable per reference source).
            "labels": (self.test_name, self.reference_name),
        }
        # A scored comparison carries metric maps -- unless it is a place through time,
        # where there is no map to carry and the series family draws the pair itself.
        if self.over is None or self.is_series:
            return {"aligned": self.aligned, **common}
        from ocean_skill.metrics import DEFAULT_MAP_METRICS

        names = tuple(self.metric_names or DEFAULT_MAP_METRICS)
        return {"skill": self.maps(*names), "metric_names": names, **common}

    def plot(self, *, renderer: str = "matplotlib", **kwargs: Any):
        """Render as a ``test | reference | difference`` row, or as metric maps.

        Which of the two follows from the comparison, not from an argument: a pair
        reduced to single maps has a test, a reference and a difference to show, while
        one scored ``over`` an axis has a map per metric and nothing to set beside it.
        That is the same choice :meth:`ocean_skill.field.Field.plot` makes between a
        single map and a facet grid — the data decides the family.

        Goes through the renderer registry, so ``renderer="holoviews"`` gives the
        interactive version of the same plot with no other change.
        """
        from ocean_skill.plot.registry import render
        from ocean_skill.plot.spec import PlotSpec

        family = self.family
        if family != "skill_map":
            kwargs.setdefault("labels", (self.test_name, self.reference_name))
        if family != "series":
            # Outline the test (model) source's own declared extent, matching Abigale
            # Wyatt's side-by-side plots — pass domain=None to suppress it. A line plot
            # has no map to outline, and series() would refuse the option outright.
            kwargs.setdefault("domain", _domain_of(self.test_name))
        spec = PlotSpec(family=family, items=[self.as_item()], options=kwargs)
        return render(spec, renderer=renderer)

    def save(
        self,
        project: str | None = None,
        *,
        stem: str | None = None,
        figure: bool = True,
        metrics: bool = True,
        renderer: str = "matplotlib",
        **plot_kwargs: Any,
    ) -> dict[str, Path]:
        """Write this comparison's figure and metrics row under ``output/<project>/``.

        The single-row counterpart of :meth:`ComparisonSet.save`; see there and
        :mod:`ocean_skill.outputs` for the layout and why deliverables are kept out
        of the cache.
        """
        stem = stem or _variable_label(self.variable)[:24]
        return ComparisonSet([self]).save(
            project or f"{self.test_name}_vs_{self.reference_name}",
            stem=stem,
            figure=figure,
            metrics=metrics,
            renderer=renderer,
            **plot_kwargs,
        )

    def __repr__(self) -> str:
        scored = f" over {self.over}" if self.over else ""
        return (
            f"<Comparison {_variable_label(self.variable)[:24]} "
            f"{self.test_name} vs {self.reference_name} "
            f"@ {_depth_label(_selected_depth(self.select))}{scored}>"
        )


def _identity(c) -> tuple:
    """Return what makes two comparisons the same one: their whole specification.

    Built from the object's own attributes rather than from ``c._cache_key``, which
    hashes very nearly this but exists to name a zarr store — pooling has no business
    depending on the cache's format version or on what it deliberately leaves out.
    """
    return (
        getattr(c, "test_name", None),
        getattr(c, "reference_name", None),
        repr(getattr(c, "variable", None)),
        repr(getattr(c, "select", None)),
        repr(getattr(c, "aggregate", None)),
        getattr(c, "method", None),
        getattr(c, "over", None),
        getattr(c, "time_method", None),
        getattr(c, "tolerance", None),
        getattr(c, "bin_anchor", None),
    )


def _flatten(objs: Any) -> list[Comparison]:
    """Collect comparisons out of whatever shape they were handed in, dropping repeats.

    Accepts a :class:`Comparison`, a :class:`ComparisonSet`, a mapping of either, or any
    nesting of those, because "the comparisons I already have" is rarely one tidy list —
    it is a couple of ``compare()`` results and a one-off or two.

    Exact repeats are dropped rather than drawn twice: two sets built for different
    figures commonly share a pair (a nutrients fan-out and a depth fan-out both hold
    ``no3 @ surface``), and pooling them would put one marker exactly on top of another,
    add a duplicate key entry, and duplicate a row in :meth:`ComparisonSet.metrics`.
    """
    out: list[Comparison] = []
    seen: set[tuple] = set()
    dropped = 0

    def add(obj: Any) -> None:
        nonlocal dropped
        if isinstance(obj, ComparisonSet):
            for c in obj:
                add(c)
            return
        if isinstance(obj, dict):
            for v in obj.values():
                add(v)
            return
        if hasattr(obj, "metrics") and hasattr(obj, "as_item"):
            key = _identity(obj)
            if key in seen:
                dropped += 1
                return
            seen.add(key)
            out.append(obj)
            return
        if isinstance(obj, str) or not hasattr(obj, "__iter__"):
            raise TypeError(
                f"expected comparisons, got {obj!r}. Pass a Comparison, a "
                "ComparisonSet (what compare() returns), a list of either, or a "
                "{name: comparisons} dict."
            )
        for item in obj:
            add(item)

    add(objs)
    if dropped:
        print(f"  pooled: dropped {dropped} duplicate comparison(s)")
    return out


#: Dimensions a pooled label can be built from, in the order they are spelled, paired
#: with how to read each one off a comparison. The same four a metrics record carries,
#: so ``color_by``/``marker_by`` can group by anything a label can name.
_LABEL_DIMS: tuple[tuple[str, Any], ...] = (
    ("variable", lambda c: _short_variable_label(c.variable)),
    ("depth", lambda c: _depth_label(_selected_depth(c.select))),
    ("test", lambda c: c.test_name),
    ("reference", lambda c: c.reference_name),
)


def _pooled_labels(comparisons: list[Comparison]) -> list[str]:
    """Name each point by what distinguishes it *within this pool*.

    :func:`compare` labels only what varies across its own fan-out, which is right for
    the set it built and wrong the moment that set is pooled with another: two calls
    that each fanned over depth both produce a point labelled ``surface``. So the same
    rule is applied again over the pooled comparisons, where the varying dimension may
    now be the model or the reference rather than the depth.

    Dimensions are added only while they are still doing work — in priority order, each
    kept only if it tells more points apart than the label already did, and stopped as
    soon as every point is distinct. Naming every varying dimension instead produces
    labels like ``nitrate 0 m woa23_nitrate_month07 woa23_nitrate_month01``, where the
    sources vary in lockstep with the variable and so say nothing the first word did not
    — and a legend of those is unreadable at any type size.

    With nothing varying — the same pair at the same depth, differing only in how it was
    aggregated — there is no dimension to name, and each comparison keeps the label it
    came with. Anything still colliding after that is suffixed rather than left
    ambiguous; a diagram with two points called the same thing is unreadable.
    """
    if not comparisons:
        return []

    def spell(dims) -> list[str]:
        return [" ".join(str(read(c)) for _, read in dims) for c in comparisons]

    chosen: list[tuple[str, Any]] = []
    apart = 1  # points the label currently tells apart; one label, one group
    for dim in _LABEL_DIMS:
        if apart == len(comparisons):
            break
        gain = len(set(spell([*chosen, dim])))
        if gain > apart:
            chosen.append(dim)
            apart = gain
    if not chosen:
        return [c.label or _short_variable_label(c.variable) for c in comparisons]
    labels = spell(chosen)

    counts: dict[str, int] = {}
    collided = False
    for i, lab in enumerate(labels):
        counts[lab] = counts.get(lab, 0) + 1
        if counts[lab] > 1:
            labels[i] = f"{lab} ({counts[lab]})"
            collided = True
    if collided:
        print(
            "  pooled: some comparisons differ only in something a label cannot name "
            "(aggregation, region); suffixed them to keep the points apart"
        )
    return labels


def _named_labels(mapping: dict[str, Any]) -> tuple[list[Comparison], list[str]]:
    """Label a ``{name: comparisons}`` pool by its keys rather than by a rule.

    Your key is what disambiguates across groups, so a group's members keep their own
    labels and get the key in front of them. A group of one is labelled with the key
    alone — ``{"run A": c}`` means the point is called ``run A``, not ``run A: no3``.
    """
    comparisons: list[Comparison] = []
    labels: list[str] = []
    for name, obj in mapping.items():
        members = _flatten(obj)
        for c in members:
            own = c.label or _short_variable_label(c.variable)
            labels.append(str(name) if len(members) == 1 else f"{name}: {own}")
        comparisons += members
    return comparisons, labels


class ComparisonSet:
    """A set of comparisons: stacked rows in one figure, one tidy metrics table.

    Also how comparisons you already have are pooled onto one summary diagram — the
    constructor takes any nesting of comparisons and sets, and ``+`` joins two sets:

        pooled = nutrients + depths        # relabelled by what varies across the pool
        osk.ComparisonSet({"hindcast": a, "forecast": b})   # or name the groups
    """

    def __init__(
        self,
        comparisons: Any,
        *,
        labels: list[str] | None = None,
    ):
        if isinstance(comparisons, dict):
            if labels is not None:
                raise TypeError(
                    "pass either a {name: comparisons} dict or labels=, not both — "
                    "the dict's keys are already the labels"
                )
            self.comparisons, labels = _named_labels(comparisons)
        else:
            self.comparisons = _flatten(comparisons)
        if labels is not None and len(labels) != len(self.comparisons):
            raise ValueError(
                f"labels has {len(labels)} entries for {len(self.comparisons)} "
                "comparisons — there must be one per comparison"
            )
        #: Per-comparison label overrides, or None to use each comparison's own.
        self.labels = list(labels) if labels is not None else None

    def __len__(self) -> int:
        return len(self.comparisons)

    def __iter__(self):
        return iter(self.comparisons)

    def __getitem__(self, i):
        return self.comparisons[i]

    def __add__(self, other: Any) -> ComparisonSet:
        """Pool two sets into one, relabelled by what varies across the pool.

        List-like ``+``, because a set is a container (it has ``__len__``, ``__iter__``
        and ``__getitem__``) — nothing here is added to anything numerically. Combining
        *variables* is a different operation with its own spelling; see
        :mod:`ocean_skill.operators`.
        """
        pooled = _flatten([self.comparisons, other])
        return ComparisonSet(pooled, labels=_pooled_labels(pooled))

    def _label_for(self, i: int) -> Any:
        """Return this set's label for comparison ``i``: its override, else the own one.

        Overriding here rather than writing onto the comparison is the whole reason
        pooling is safe to do with objects you already have: a comparison pooled into a
        summary keeps the label its own set draws as a row label and its own movie draws
        as a frame label.
        """
        if self.labels is not None:
            return self.labels[i]
        return self.comparisons[i].label

    def metrics(self):
        """Return every comparison's metrics as a tidy DataFrame (one row each)."""
        import pandas as pd

        return pd.DataFrame([c.metrics() for c in self.comparisons])

    def write_metrics(self, out_dir: str | Path, stem: str = "metrics") -> Path:
        """Write the tidy metrics table to ``<out_dir>/metrics/<stem>.csv``."""
        from ocean_skill import metrics as _metrics

        return _metrics.write(
            [c.metrics() for c in self.comparisons], out_dir, stem=stem
        )

    def save(
        self,
        project: str | None = None,
        *,
        stem: str = "comparison",
        figure: bool = True,
        metrics: bool = True,
        renderer: str = "matplotlib",
        **plot_kwargs: Any,
    ) -> dict[str, Path]:
        """Write this set's figure and metrics table under ``output/<project>/``.

        Deliverables, not cache — see :mod:`ocean_skill.outputs` for why they are
        kept apart. Returns the paths written, keyed ``"figure"``/``"metrics"``.

        ``project`` defaults to a slug of the first comparison's ``test vs
        reference``. Extra keyword arguments are forwarded to the renderer, so the
        styling knobs in ``docs/plot_styling_reference.md`` all work here too.
        """
        from ocean_skill import outputs

        if not self.comparisons:
            raise ValueError("nothing to save: this set has no comparisons")
        first = self.comparisons[0]
        project = project or f"{first.test_name}_vs_{first.reference_name}"

        written: dict[str, Path] = {}
        if metrics:
            written["metrics"] = self.write_metrics(
                outputs.project_dir(project), stem=stem
            )
        if figure:
            fig_path = outputs.figures_dir(project) / f"{stem}.png"
            self.plot(renderer=renderer, save=fig_path, **plot_kwargs)
            written["figure"] = fig_path
        return written

    def _items(self) -> list[dict[str, Any]]:
        """Spec items for every comparison in the set."""
        items = []
        for i, c in enumerate(self.comparisons):
            label = self._label_for(i)
            items.append({**c.as_item(), "label": label, "row_label": label})
        return items

    def _metric_items(self) -> list[dict[str, Any]]:
        """Spec items carrying only what a summary diagram reads: metrics and a label.

        The summary families are the one place :meth:`_items` is more than is needed and
        the extra costs real work: for a set scored ``over`` an axis, ``as_item`` builds
        a metric map per comparison (:meth:`Comparison.maps`) that a Taylor or target
        point never looks at. Both renderers take exactly ``metrics`` and ``label`` off
        these items — see ``matplotlib_renderer._Record`` and the interactive target —
        so this is the whole contract, not a subset of it.
        """
        return [
            {"metrics": c.metrics(), "label": self._label_for(i)}
            for i, c in enumerate(self.comparisons)
        ]

    def plot(self, *, renderer: str = "matplotlib", **kwargs: Any):
        """Render all comparisons as stacked rows in one figure.

        A set whose comparisons were scored ``over`` an axis has metric maps rather than
        an aligned trio per row, so it draws the ``skill_map`` family instead: metrics
        across the columns, comparisons down the rows, exactly as ``field_grid`` stacks
        rows for the unscored case.
        """
        from ocean_skill.plot.registry import render
        from ocean_skill.plot.spec import PlotSpec

        if not self.comparisons:
            raise ValueError(
                "no comparisons to plot: every pair was skipped. Check that the "
                "reference actually offers the requested variables (its catalog "
                "metadata lists them under 'variables')."
            )
        items = self._items()
        first = self.comparisons[0]
        families = {c.family for c in self.comparisons}
        if len(families) > 1:
            # Each family is a different figure -- a metric per panel, lines on one time
            # axis, or test | reference | difference -- so there is nothing to draw that
            # is all of them. Naming which comparison went which way is the point: the
            # usual cause is one reference in the set being a station, the rest grids.
            detail = ", ".join(
                f"{c.label or c.test_name + ' vs ' + c.reference_name}:"
                f" {c.family_reason}"
                for c in self.comparisons
            )
            raise ValueError(
                f"this set mixes {sorted(families)}, which are different figures — "
                f"{detail}. Plot them separately."
            )
        family = families.pop()
        if family == "field_grid" or family == "series":
            kwargs.setdefault("labels", (first.test_name, first.reference_name))
        if family != "series":
            # Outlines the first row's test (model) extent; rows sharing one test source
            # (the common case) all get the same box. Pass domain=None to suppress, or
            # your own bbox if rows mix test sources with different domains.
            kwargs.setdefault("domain", _domain_of(first.test_name))
        # field_row is one comparison's family; a set of them stacks as a grid.
        family = "field_grid" if family == "field_row" else family
        return render(
            PlotSpec(family=family, items=items, options=kwargs),
            renderer=renderer,
        )

    def movie(self, *, renderer: str = "matplotlib", **kwargs: Any):
        """Play the set's comparisons as movie frames rather than stacking them as rows.

        The same items :meth:`plot` lays out down the page, animated instead: one
        ``test | reference | difference`` row, redrawn per comparison, with each
        comparison's own label as the frame label and its own metrics in the corner box.
        So a set that reads as a small multiple statically reads as a movie here, and
        both are the same set — nothing is re-prepared.

        ``save`` names the file and its extension picks the format: ``.mp4`` (needs
        ffmpeg) or ``.gif`` (needs nothing extra) statically, ``.html`` with
        ``renderer="holoviews"``, where the frames go on a slider instead. See
        :func:`ocean_skill.plot.matplotlib_renderer.field_movie`.

        Ordering is the set's own order, which for a :func:`compare` fan-out is the
        order the fan produced. That is what you want when one thing varies across the
        set (a depth, a time, a run) and is meaningless when several do — a movie whose
        frames step through both variable *and* depth is not a movie of either.
        """
        from ocean_skill.plot.registry import render
        from ocean_skill.plot.spec import PlotSpec

        if not self.comparisons:
            raise ValueError("no comparisons to animate: every pair was skipped")
        series = [c for c in self.comparisons if c.is_series]
        if series:
            raise ValueError(
                f"{len(series)} of these comparisons are time series, which have no "
                "frames to play: their time axis is already the x axis of the figure. "
                "Use .plot() — a movie of a line plot would be one line drawn over and "
                "over."
            )
        first = self.comparisons[0]
        # a row label (drawn rotated at a grid row's left edge) and a frame label (drawn
        # in the panel, changing as the movie plays) are the same identity in two
        # places, so the movie reads it from the same c.label the grid does
        frames = [
            {**item, "frame_label": item.get("row_label")} for item in self._items()
        ]
        kwargs.setdefault("labels", (first.test_name, first.reference_name))
        kwargs.setdefault("domain", _domain_of(first.test_name))
        return render(
            PlotSpec(family="field_movie", items=frames, options=kwargs),
            renderer=renderer,
        )

    def taylor(self, *, renderer: str = "matplotlib", **kwargs: Any):
        """Taylor diagram of the set (correlation + variability; blind to bias)."""
        from ocean_skill.plot.registry import render
        from ocean_skill.plot.spec import PlotSpec

        return render(
            PlotSpec(family="taylor", items=self._metric_items(), options=kwargs),
            renderer=renderer,
        )

    def target(self, *, renderer: str = "matplotlib", **kwargs: Any):
        """Target diagram of the set (bias vs signed centred RMSD)."""
        from ocean_skill.plot.registry import render
        from ocean_skill.plot.spec import PlotSpec

        return render(
            PlotSpec(family="target", items=self._metric_items(), options=kwargs),
            renderer=renderer,
        )

    def summary(self, *, renderer: str = "matplotlib", **kwargs: Any):
        """Taylor and Target side by side for the whole set."""
        from ocean_skill.plot.registry import render
        from ocean_skill.plot.spec import PlotSpec

        return render(
            PlotSpec(family="paired", items=self._metric_items(), options=kwargs),
            renderer=renderer,
        )

    def __repr__(self) -> str:
        return f"<ComparisonSet: {len(self)} comparisons>"


#: What ``summary(kind=...)`` names, and the set method each one is.
_SUMMARY_KINDS = {"both": "summary", "taylor": "taylor", "target": "target"}


def summary(
    comparisons: Any,
    *,
    kind: str = "both",
    renderer: str = "matplotlib",
    **kwargs: Any,
):
    """Summarize comparisons you already have on one diagram.

    The counterpart to :func:`compare`, for the case where the comparisons exist: a
    nutrients fan-out, a depth fan-out, a one-off pair, pooled onto a single Taylor
    and/or target diagram without re-expressing them as one ``compare()`` call — which
    is often impossible anyway, since a pool may mix references, aggregations, or a
    station with a grid::

        osk.summary([nutrients, depths, c])
        osk.summary({"hindcast": nutrients, "forecast": other}, kind="taylor")

    Pooling is safe here in a way it is not for :meth:`ComparisonSet.plot`, which
    refuses a set mixing plot families: a metrics record is a handful of scalars whether
    the comparison was a map, a scored map or a station series, and both diagrams
    normalize by the reference's standard deviation, so unlike a figure of fields these
    points are comparable across variables and units.

    ``kind`` picks the diagram — ``"both"`` (Taylor and target side by side),
    ``"taylor"`` or ``"target"``. It is not ``renderer``, which sits beside it and picks
    static, interactive, or ``"both"`` of *those*; the two words mean different things
    and both take that value.

    Points are named by what varies across the pool, or by your own names if
    ``comparisons`` is a ``{name: comparisons}`` dict. Either way the comparisons
    themselves are untouched — see :meth:`ComparisonSet._label_for`. Remaining keyword
    arguments go to the diagram (``color_by``, ``marker_by``, ``labels``, ``title``,
    ``save``, ...); see :mod:`ocean_skill.plot.summary`.
    """
    if kind not in _SUMMARY_KINDS:
        raise ValueError(
            f"kind={kind!r} is not one of {tuple(_SUMMARY_KINDS)} — 'both' draws "
            "Taylor and target side by side, 'taylor' or 'target' just the one. "
            "(To choose static vs interactive, use renderer=.)"
        )
    if isinstance(comparisons, dict):
        pooled = ComparisonSet(comparisons)  # keys are the labels
    else:
        members = _flatten(comparisons)
        pooled = ComparisonSet(members, labels=_pooled_labels(members))
    return getattr(pooled, _SUMMARY_KINDS[kind])(renderer=renderer, **kwargs)


def compare(
    *,
    reference,
    test,
    variables: list[Any],
    depths=None,
    select: dict[str, Any] | None = None,
    aggregate: dict[str, Any] | None = None,
    method: str = "conservative_normed",
    over: str | None = None,
    time_method: str = "auto",
    tolerance: float | None = None,
    bin_anchor: str = "auto",
    min_pairs: int = DEFAULT_MIN_PAIRS,
    metrics: tuple[str, ...] | None = None,
    skip_missing: bool = True,
    cache: bool | None = None,
    refresh: bool = False,
) -> ComparisonSet:
    """Fan over reference × test × variable × depth and return a :class:`ComparisonSet`.

    ``reference`` and ``test`` each take a source name or a list; ``variables`` is a
    list of anything :mod:`ocean_skill.vocabulary` recognizes — a short vocabulary key
    (``"oxygen"``), a canonical CF standard_name, or any alias (in any case) —
    resolved to its canonical standard_name once here (warning once per variable if
    the name given wasn't already the canonical form, so it's never unclear which
    variable was actually used). Pairs where the variable is absent are skipped with a
    message rather than failing the whole run, unless ``skip_missing=False``.

    A variable may also be a *combination* — ``{"sum": ["spChl", "diatChl",
    "diazChl"], "standard_name": CHL}`` — see :mod:`ocean_skill.operators`.
    ``aggregate`` names how dimensions collapse, and **there is no default**: a
    comparison has to be a single map, so if the sources carry a time axis you have to
    say what happens to it — ``{"time": "mean"}`` for the whole selection's mean,
    ``select={"time": <one step>}`` to compare one instant. Omitting it no longer means
    "take the mean"; it means "reduce nothing", which then fails with the axis named
    rather than averaging something you did not ask for (see :data:`NO_AGGREGATION`).

    ``{"time": {"groupby": "month", "reduce": "mean"}}`` gives a climatology and
    ``{"time": {"resample": "1MS", "reduce": "mean"}}`` consecutive months. Both keep an
    axis standing, which a comparison cannot use unless it is told to *score against* it
    — see :func:`ocean_skill.field.field` for the model-only path that plots those as
    panels.

    ``over`` is the third answer to "what happens to the time axis", and the one for a
    reference that varies in time as well as space. ``over="time"`` keeps the axis,
    matches the two lanes along it (:func:`ocean_skill.align.match_axis` — a finer test
    is averaged into the reference's bins) and computes every metric *cell by cell*
    along it, so the figure becomes one map per metric with the overall value beside it.
    ``over="Z"`` does the same down each water column.
    ``time_method``/``tolerance``/``bin_anchor`` tune the matching, ``min_pairs`` how
    many pairs a cell needs before it is reported, and ``metrics`` which maps are
    computed (default :data:`ocean_skill.metrics.DEFAULT_MAP_METRICS`).

    ``variables`` is required, deliberately not inferred from the sources' catalog
    metadata: a reference or test with several declared variables gives no way to
    tell *which* of them the caller actually wants compared, so guessing would either
    have to pick arbitrarily or silently expand to "compare everything in common" —
    neither is a default worth having quietly happen. Naming the variable(s) is the
    one thing this call cannot infer on your behalf.

    ``depths`` defaults to ``("surface",)`` — the model's own top level, via
    :func:`ocean_skill.roms.surface`. A literal ``0`` is a *different*, real request:
    the field interpolated to exactly 0 m, which for a model whose topmost cell centre
    sits a few metres down legitimately comes back all-NaN (with a warning) rather than
    silently reusing the surface field.

    Each pair's aligned result is cached to disk and reused on a later run with the
    same arguments (see :mod:`ocean_skill.cache`, which prints where once per
    process). ``cache=False`` bypasses it; ``refresh=True`` recomputes and overwrites
    — what to use after rerunning a model, since entries are keyed on source and
    selection rather than on file contents.
    """
    from ocean_skill.catalog import resolve
    from ocean_skill.vocabulary import equivalent_names, resolve_and_report

    # depths defaults to the vertical entry already in `select`, if any, so the two
    # spellings agree instead of one clobbering the other.
    if depths is None:
        vertical = next(
            (select[k] for k in _VERTICAL_KEYS if select and k in select), SURFACE
        )
        depths = (vertical,)

    refs = [reference] if isinstance(reference, str) else list(reference)
    tests = [test] if isinstance(test, str) else list(test)
    # Resolve each requested variable to its canonical standard_name once, up front
    # -- both so _offers() below matches against catalog metadata correctly (which
    # declares the canonical name), and so the one "name resolved to..." warning
    # fires once per variable rather than once per (ref, test, depth) combination
    # this fans out to.
    variables = [
        resolve_and_report(v, context="compare variables=") if isinstance(v, str) else v
        for v in variables
    ]

    def _offers(source: str, variable: Any) -> bool:
        """Report whether the source advertises this variable (or an equivalent).

        A combination has more than one way to be satisfied — every component, or
        the ``standard_name`` it falls back to for sources using the other
        convention (MODIS ships total chlorophyll; MARBL ships the components).
        Any one complete option is enough.
        """
        from ocean_skill.operators import spec_names
        from ocean_skill.vocabulary import is_known

        try:
            declared = resolve(source).metadata.get("variables")
        except KeyError:
            return True
        if not declared:
            return True  # no metadata to filter on; let the read decide
        declared = set(declared)
        options = spec_names(variable)
        if any(all(equivalent_names(n) & declared for n in opt) for opt in options):
            return True  # positively offered

        # Not positively offered -- but "absent" and "unknowable" are different.
        # A catalog's `variables` list holds CF standard_names, so it can never
        # confirm or deny a raw model variable like `spChl`; concluding "absent"
        # from that silently drops a comparison that would have worked. Only
        # exclude when every name involved is one the catalog could have listed.
        return not all(is_known(n) for opt in options for n in opt)

    out: list[Comparison] = []
    for var in variables:
        # Pair each variable with the sources that actually carry it, rather than
        # forming a blind cross-product. Observational catalogs are usually one
        # entry per variable (WOA ships nitrate and phosphate separately), and so
        # is model output: a ROMS run writes physics and BGC to different streams,
        # on different time axes, so they cannot be one source. Filtering both
        # sides lets you pass every stream as `test=` and have each variable find
        # the one that has it, instead of failing half the cross-product.
        matching = [r for r in refs if _offers(r, var)]
        matching_tests = [t for t in tests if _offers(t, var)]
        if not matching:
            print(f"  no reference offers {var!r}; skipped")
            continue
        if not matching_tests:
            print(f"  no test offers {var!r}; skipped")
            continue
        for ref in matching:
            for tst in matching_tests:
                for d in depths:
                    # `depths` is sugar for fanning select's vertical entry over a
                    # list -- one comparison per value. Writing it into the same key
                    # the caller may have used means a select={"Z": ...} is honoured
                    # rather than silently overridden by the default of "surface".
                    sel = {**(select or {})}
                    for key in _VERTICAL_KEYS:
                        sel.pop(key, None)
                    sel["depth"] = d
                    # Label only what varies across the set: repeating the variable
                    # name on every point of a single-variable fan-out just collides.
                    short = _short_variable_label(var)
                    many_vars, many_depths = len(variables) > 1, len(depths) > 1
                    if many_vars and many_depths:
                        label = f"{short} {_depth_label(d)}"
                    elif many_depths:
                        label = _depth_label(d)
                    else:
                        label = short
                    c = Comparison(
                        reference=ref,
                        test=tst,
                        variable=var,
                        select=sel,
                        aggregate=aggregate,
                        method=method,
                        over=over,
                        time_method=time_method,
                        tolerance=tolerance,
                        bin_anchor=bin_anchor,
                        min_pairs=min_pairs,
                        metrics=metrics,
                        label=label,
                        cache=cache,
                    )
                    try:
                        c.align(refresh=refresh)
                    except KeyError as exc:
                        if not skip_missing:
                            raise
                        print(f"  skipped: {exc}")
                        continue
                    out.append(c)
    return ComparisonSet(out)
