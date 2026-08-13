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
]


#: Sentinel meaning "the model's own top level" as distinct from an explicit request
#: for the field interpolated to literal 0 m — see :func:`_prepare`.
SURFACE = "surface"


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


#: Keys naming the vertical axis in a `select`, in any accepted spelling.
_VERTICAL_KEYS = frozenset({"depth", "Z", "vertical", "z"})


def _vertical_only(agg: dict[str, Any] | None) -> dict[str, Any]:
    """Return the part of an aggregation spec addressing the vertical axis."""
    return {k: v for k, v in (agg or {}).items() if k in _VERTICAL_KEYS}


def _without_vertical(agg: dict[str, Any] | None) -> dict[str, Any]:
    """Return the part of an aggregation spec for every axis but the vertical."""
    return {k: v for k, v in (agg or {}).items() if k not in _VERTICAL_KEYS}


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
    if "actual_depth" not in da.attrs and "depth" in da.coords and not da["depth"].dims:
        da.attrs["actual_depth"] = float(da["depth"])
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
        if self.over is not None and t is not None:
            from ocean_skill.align import bbox_of

            bbox = bbox_of(t)
        r, r_depth = self._prepare_lane(
            self.reference_name, use_cache, refresh, role="reference", bbox=bbox
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
            if self.over is not None:
                enough = self.maps("n")["n"] >= self.min_pairs
                aligned = aligned.where(enough)
            self._metrics = _metrics.compute(
                aligned,
                test_name="test",
                reference_name="reference",
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
        if self.over is None:
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

        if self.over is None:
            kwargs.setdefault("labels", (self.test_name, self.reference_name))
        # Outline the test (model) source's own declared extent, matching Abigale
        # Wyatt's side-by-side plots — pass domain=None to suppress it.
        kwargs.setdefault("domain", _domain_of(self.test_name))
        family = "field_row" if self.over is None else "skill_map"
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
            f"@ {_depth_label(self.select.get('depth', SURFACE))}{scored}>"
        )


class ComparisonSet:
    """A set of comparisons: stacked rows in one figure, one tidy metrics table."""

    def __init__(self, comparisons: list[Comparison]):
        self.comparisons = comparisons

    def __len__(self) -> int:
        return len(self.comparisons)

    def __iter__(self):
        return iter(self.comparisons)

    def __getitem__(self, i):
        return self.comparisons[i]

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
        return [{**c.as_item(), "row_label": c.label} for c in self.comparisons]

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
        scored = [c.over is not None for c in self.comparisons]
        if any(scored) and not all(scored):
            raise ValueError(
                "this set mixes comparisons scored over an axis with comparisons "
                "reduced to single maps, which are two different figures: one draws a "
                "metric per panel, the other test | reference | difference. Plot them "
                "separately."
            )
        if not any(scored):
            kwargs.setdefault("labels", (first.test_name, first.reference_name))
        # Outlines the first row's test (model) extent; rows sharing one test source
        # (the common case) all get the same box. Pass domain=None to suppress it, or
        # your own bbox if rows mix test sources with different domains.
        kwargs.setdefault("domain", _domain_of(first.test_name))
        family = "skill_map" if any(scored) else "field_grid"
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
            PlotSpec(family="taylor", items=self._items(), options=kwargs),
            renderer=renderer,
        )

    def target(self, *, renderer: str = "matplotlib", **kwargs: Any):
        """Target diagram of the set (bias vs signed centred RMSD)."""
        from ocean_skill.plot.registry import render
        from ocean_skill.plot.spec import PlotSpec

        return render(
            PlotSpec(family="target", items=self._items(), options=kwargs),
            renderer=renderer,
        )

    def summary(self, *, renderer: str = "matplotlib", **kwargs: Any):
        """Taylor and Target side by side for the whole set."""
        from ocean_skill.plot.registry import render
        from ocean_skill.plot.spec import PlotSpec

        return render(
            PlotSpec(family="paired", items=self._items(), options=kwargs),
            renderer=renderer,
        )

    def __repr__(self) -> str:
        return f"<ComparisonSet: {len(self)} comparisons>"


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
    from ocean_skill.vars import short_name
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
                    short = (
                        short_name(var)
                        if isinstance(var, str)
                        else (_variable_label(var))
                    )
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
