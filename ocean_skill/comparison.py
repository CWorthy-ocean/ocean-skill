"""The compare layer: ``Comparison`` (one pair) and ``compare`` (fan-out).

A :class:`Comparison` holds a reference and a test source for one variable plus a
selection; it reads both, reduces them to a comparable 2-D field, aligns them onto
whichever lane is coarser, and exposes the difference, metrics and a plot. Roles are
assigned here, not in the catalog: ``diff = test − reference`` always, regardless of
which lane's axis the alignment lands on. :func:`compare` fans over the reference ×
test × variable × depth cross-product and collects the results into a
:class:`ComparisonSet` that can write one tidy metrics table and one stacked figure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "SURFACE",
    "Comparison",
    "ComparisonSet",
    "aggregate_for",
    "as_select",
    "compare",
    "is_pair_spec",
    "is_surface_request",
    "prepare_source",
    "select_for",
    "summary",
    "variable_for",
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


#: Spec key spellings :func:`_collapses_time` recognizes as naming the time axis --
#: matches how a ``select``/``aggregate`` spec is actually written (see
#: :func:`ocean_skill.operators.resolve_dim`'s ``_CF_AXES``), not every fallback name
#: :func:`ocean_skill.cf.find_coord` would also accept, since this check runs before
#: any data is read and has no dataset to resolve a raw dim name like ``ocean_time``
#: against.
_TIME_KEYS = frozenset({"time", "T"})


def _collapses_time(agg: dict[str, Any] | None) -> bool:
    """Whether ``agg`` reduces time to a single value rather than keeping an axis.

    A plain reducer (``"mean"``, or ``{"reduce": "mean"}`` with neither ``groupby``
    nor ``resample``) collapses time to one number per position, leaving nothing
    for a line to be drawn along -- a select that narrows the reference to a place
    should not imply ``over="time"`` on top of that (see
    :meth:`Comparison._point_select_implies_time`). ``groupby``/``resample`` both
    *keep* an axis instead (a climatology, or consecutive periods -- see
    :func:`ocean_skill.operators.aggregate`), so neither disables the inference the
    mooring recipe already depends on.
    """
    agg = agg or {}
    for key in _TIME_KEYS:
        if key not in agg:
            continue
        spec = agg[key]
        if isinstance(spec, dict) and ("groupby" in spec or "resample" in spec):
            return False
        return True
    return False


def _outline_of(source: str, convention: str | None = None) -> np.ndarray | None:
    """Return the source's true grid-edge outline as an ``(N, 2)`` ``[lon, lat]`` ring.

    Reads the ``domain_outline`` a curvilinear source's catalog entry carries (written
    by :mod:`ocean_skill.build` from the actual grid, native lon values — see
    ``ocean_skill.align.perimeter_of``), and re-expresses it in ``convention``
    (``"0-360"``/``"-180-180"``) so it lands on the same longitude axis the comparison
    is plotted on. ``None`` (the default) picks whichever convention keeps the ring
    contiguous, the same rule :func:`_domain_of` uses for a bbox. Returns ``None`` when
    the source is unresolvable, declares no outline (a rectilinear grid, or a catalog
    built before this key existed — :func:`_domain_of`'s bbox is the fallback then), or
    the value is malformed.
    """
    from ocean_skill.catalog import resolve

    try:
        meta = resolve(source).metadata
    except KeyError:
        return None
    raw = meta.get("domain_outline")
    if not raw:
        return None
    try:
        ring = np.asarray(raw, dtype="float64")
    except (TypeError, ValueError):
        return None
    if ring.ndim != 2 or ring.shape[1] != 2 or ring.shape[0] < 3:
        return None
    ring = ring.copy()
    # Re-unwrap defensively (the stored ring was already unwrapped when built, but a
    # round trip through YAML/user-editing offers no guarantee) before re-grounding it
    # in the requested convention.
    ring[:, 0] = np.unwrap(ring[:, 0], period=360.0)
    lon_min, lon_max = float(ring[:, 0].min()), float(ring[:, 0].max())
    wrapped_min = ((lon_min + 180) % 360) - 180
    wrapped_max = ((lon_max + 180) % 360) - 180
    # ±180 wraps each end independently; if that keeps them ordered, both ends fell in
    # the same 360-cycle and one shift re-grounds the whole (already contiguous) ring.
    # If it doesn't, ±180 would split the ring at its own seam — the case _domain_of
    # keeps in 0-360 for the same reason (see its docstring) — so ground there instead.
    splits_at_180 = wrapped_min > wrapped_max
    if convention is None:
        convention = "0-360" if splits_at_180 else "-180-180"
    if convention == "-180-180" and not splits_at_180:
        shift = wrapped_min - lon_min
    else:
        shift = -360.0 * np.floor(lon_min / 360.0)
    ring[:, 0] += shift
    return ring


def _domain_of(source: str) -> tuple[float, float, float, float] | None:
    """Return ``(lon_min, lat_min, lon_max, lat_max)`` for a source's catalog extent.

    Used to draw the model-domain outline on a map (as in Abigale Wyatt's
    ``Obs_comparisons.ipynb``) without the caller having to know or repeat the model's
    bounding box — the fallback when the source has no :func:`_outline_of` ring (a
    rectilinear grid, whose bbox already *is* its perimeter, or an entry built before
    ``domain_outline`` existed). Returns ``None`` if the source is unresolvable or the
    catalog entry doesn't declare a geospatial extent, in which case no box is drawn.
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
    # Catalogs may declare 0-360 (ROMS' native convention); maps are usually drawn in
    # ±180, so normalize — unless the domain straddles the antimeridian, where ±180
    # endpoints read backwards (lon_min > lon_max) and would draw as two stray
    # verticals. Such a box stays in 0-360, matching the convention align() resolves
    # for the same domain's data (see ocean_skill.align.natural_convention).
    wrapped = tuple(((lo + 180) % 360) - 180 for lo in (lon_min, lon_max))
    if wrapped[0] <= wrapped[1]:
        lon_min, lon_max = wrapped
    else:
        lon_min, lon_max = (lo % 360 for lo in (lon_min, lon_max))
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


#: Reported (in labels, repr, and the metrics table) for a comparison whose variable
#: has no vertical axis at all -- a calculated diagnostic like mixed layer depth.
#: Distinct from :data:`SURFACE`, which means "the model's own top level": a real
#: vertical position a calculated field does not have, and reporting one anyway
#: (the previous default) claimed something specific and wrong about where in the
#: column the number came from.
NO_VERTICAL_AXIS = "n/a"


def _sigma_label(value: Any) -> str:
    """Format a sigma0 request for labels/repr: ``"σ₀ = 26.5 kg/m³"``.

    A list — several isopycnals kept as facet rows — spells each element the same
    way, comma-joined, matching how :func:`_depth_label` spells a list of depths.
    """
    if isinstance(value, list | tuple):
        return ", ".join(_sigma_label(v) for v in value)
    return f"σ₀ = {float(value):g} kg/m³"


def _depth_label(depth: Any) -> str:
    """Format a depth for labels/repr: ``"surface"``, ``"0-10 m"`` or ``"<n> m"``.

    A list — several levels kept as facet rows — spells each element by the same
    rules: ``["surface", 50, 100]`` reads ``"surface, 50 m, 100 m"``.

    ``{"sigma0": ...}`` is the marker :func:`_selected_depth` returns for an
    isopycnal request — distinct from a depth band's ``{"min", "max"}`` — and is
    spelled through :func:`_sigma_label` instead of as a depth.
    """
    if depth == NO_VERTICAL_AXIS:
        return NO_VERTICAL_AXIS
    if isinstance(depth, dict) and "sigma0" in depth:
        return _sigma_label(depth["sigma0"])
    if is_surface_request(depth):
        return SURFACE
    if is_depth_band(depth):
        return f"{float(depth['min']):g}-{float(depth['max']):g} m"
    if isinstance(depth, list | tuple):
        return ", ".join(_depth_label(d) for d in depth)
    return f"{float(depth):g} m"


def is_pair_spec(spec: Any) -> bool:
    """Report whether ``spec`` names a *different* variable per lane.

    ``{"test": <spec>, "reference": <spec>, "standard_name": ...}`` — for the case a
    plain or combination spec cannot cover: the two lanes carry the same physical
    quantity under genuinely different recipes (a model computes mixed layer depth
    from temperature and salinity; an observational climatology already ships it as
    a plain field). Requiring *both* keys, not just one, is what tells a one-sided
    typo (``{"test": ...}`` alone) apart from an ordinary combination spec, whose keys
    are combiner names (``sum``/``product``/``difference``/``ratio``) or ``calculate``
    — never ``test``/``reference`` — so there is no ambiguity between the two shapes.
    """
    return isinstance(spec, dict) and {"test", "reference"} <= spec.keys()


def _require_pair_spec(spec: dict[str, Any], kind: str = "variable") -> None:
    """Raise if ``spec`` has exactly one of ``test``/``reference`` — a likely typo.

    A dict with neither key is an ordinary combination spec and never reaches this;
    one with both is a valid pair. One alone is almost certainly a slip (typing
    ``"tests"``, or meaning to write a plain combination) and is worth naming rather
    than silently treating as a combiner key that will fail some other way downstream.

    ``kind`` names what's being validated (``"variable"``, ``"select"``,
    ``"aggregate"``) so the same check reads correctly for all three pair-specs.
    """
    have = {"test", "reference"} & spec.keys()
    if have and have != {"test", "reference"}:
        missing = ({"test", "reference"} - have).pop()
        article = "an" if kind[0] in "aeiou" else "a"
        raise ValueError(
            f"{article} {kind} pair-spec needs both 'test' and 'reference'; got "
            f"{sorted(have)} but not {missing!r}. {spec!r}"
        )


def variable_for(spec: Any, role: str) -> Any:
    """Return the spec one lane (``"test"``/``"reference"``) should resolve.

    A pair-spec's own side; any other spec, unchanged -- so every caller can go
    through this once rather than repeating the ``is_pair_spec`` check.
    """
    return spec[role] if is_pair_spec(spec) else spec


def select_for(spec: Any, role: str) -> dict[str, Any]:
    """Return the select dict one lane (``"test"``/``"reference"``) should use.

    A pair-spec's own side; any other select, unchanged. Mirrors :func:`variable_for`
    exactly, and is meant to be called on an already-normalized select (see
    :func:`_normalize_pair`, used by :class:`Comparison` and :func:`compare`) — by the
    time this runs, either side of a pair is already a plain dict, not something that
    still needs :func:`as_select`.

    The pair spelling exists for a reference whose axis cannot take the same value the
    test lane needs — a WOA climatology's undecoded, numeric ``time`` cannot be asked
    for ``"2010-01"`` the way the model's calendar time can — so ``select={"test":
    {"time": "2010-01"}, "reference": {}}`` gives each lane its own selection instead
    of the one shared dict every other axis still uses.
    """
    return spec[role] if is_pair_spec(spec) else spec


def aggregate_for(spec: Any, role: str) -> dict[str, Any] | None:
    """Return the aggregate spec one lane should use. Mirrors :func:`select_for`."""
    return spec[role] if is_pair_spec(spec) else spec


def _normalize_pair(
    spec: Any, kind: str, *, normalize_side=lambda x: x
) -> Any:
    """Validate ``spec`` as a possible select/aggregate pair-spec, and normalize it.

    Any top-level ``test``/``reference`` key signals pair intent: both are then
    required (:func:`_require_pair_spec`) and, unlike a variable pair-spec, no other
    key is allowed alongside them -- a select/aggregate pair has nothing else to
    carry (no ``standard_name``), and there is no precedence rule in this package for
    a key floating outside the two sides, so one is refused rather than silently
    applied to one side, the other, or neither. Anything without pair intent passes
    through ``normalize_side`` as a whole, unchanged in shape.
    """
    if isinstance(spec, dict):
        _require_pair_spec(spec, kind=kind)
    if is_pair_spec(spec):
        extra = spec.keys() - {"test", "reference"}
        if extra:
            raise ValueError(
                f"a {kind} pair-spec takes only 'test' and 'reference', got extra "
                f"key(s) {sorted(extra)} -- put any axis shared by both lanes inside "
                "both sides instead."
            )
        return {
            "test": normalize_side(spec["test"]),
            "reference": normalize_side(spec["reference"]),
        }
    return normalize_side(spec)


def _expand_derived(spec: Any) -> Any:
    """Follow a :data:`ocean_skill.operators.DERIVED` string to what it names.

    Mirrors :func:`ocean_skill.operators.resolve_variable`'s own expansion, so a
    check against the raw spec (e.g. whether it is a calculate-spec) sees what
    ``resolve_variable`` will actually use rather than the alias pointing at it --
    ``register_derived("mld_dt", {"calculate": "mld", ...})`` makes ``"mld_dt"`` a
    calculate-spec in every way that matters, even though it is spelled as a string.
    The ``seen`` guard is only for a pathological cyclic registration; real chains
    are at most one hop.
    """
    from ocean_skill.operators import DERIVED

    seen: set[str] = set()
    while isinstance(spec, str) and spec in DERIVED and spec not in seen:
        seen.add(spec)
        spec = DERIVED[spec]
    return spec


def _is_calculated(spec: Any) -> bool:
    """Report whether ``spec`` (or either side of a pair-spec) is a ``calculate`` spec.

    A calculated diagnostic collapses the vertical axis itself -- see the ``ValueError``
    in :func:`_prepare` -- so :func:`compare`'s depth fan-out, which otherwise defaults
    every variable to ``depths=("surface",)``, has nothing to fan over for it.
    """
    spec = _expand_derived(spec)
    if isinstance(spec, dict) and "calculate" in spec:
        return True
    if is_pair_spec(spec):
        return _is_calculated(spec["test"]) or _is_calculated(spec["reference"])
    return False


def _calculate_method(spec: Any) -> str | None:
    """The ``method`` a calculate-spec (or a pair-spec's test side) declares, if any.

    Two calculate-spec methods computing the same quantity (density-threshold vs
    temperature-threshold MLD) are different recipes, not the same one -- and since
    they legitimately share one explicit ``standard_name``, that alone is not enough
    to tell their labels apart (two pair-specs both named ``standard_name="ocean_
    mixed_layer_thickness"`` used to draw as two identically-labelled, unreadable
    rows in the same figure). Folded into every label exactly once, in
    :func:`_variable_label`/:func:`_short_variable_label`, regardless of whether the
    spec carries its own standard_name or falls back to the calculator's plain name.
    """
    target = spec["test"] if is_pair_spec(spec) else spec
    return target.get("method") if isinstance(target, dict) and "calculate" in target else None


def _variable_label_base(spec: Any) -> str:
    """The name half of :func:`_variable_label`, before any calculate-method suffix."""
    if isinstance(spec, str):
        return spec.split("_of_")[-1]
    if is_pair_spec(spec):
        if name := spec.get("standard_name"):
            return str(name).split("_of_")[-1]
        return _variable_label_base(spec["test"])  # the test side names the figure
    if name := spec.get("standard_name"):
        return str(name).split("_of_")[-1]
    if "calculate" in spec:
        # The calculator's own name, not a components list to join -- {"calculate":
        # "mld", "method": ...} has no name-list to iterate the way a combination's
        # ("sum": [...]) does, and treating the calculator name as one silently
        # joined its characters ("mld" -> "m+l+d") until this was noticed here.
        return str(spec["calculate"])
    how, names = next(iter((k, v) for k, v in spec.items() if k != "standard_name"))
    return f"{how}({'+'.join(map(str, names))})"


def _variable_label(spec: Any) -> str:
    """Short display name for a variable spec, whether a name or a combination."""
    base = _variable_label_base(spec)
    method = _calculate_method(spec)
    return f"{base} ({method})" if method else base


def _short_variable_label_base(spec: Any) -> str:
    """The name half of :func:`_short_variable_label`, before any method suffix."""
    from ocean_skill.vars import short_name

    if is_pair_spec(spec):
        if name := spec.get("standard_name"):
            return short_name(name)
        return _short_variable_label_base(spec["test"])  # it may be a plain name
    return short_name(spec) if isinstance(spec, str) else _variable_label_base(spec)


def _short_variable_label(spec: Any) -> str:
    """Return a variable's name as a point label, short where the vocabulary knows one.

    :func:`_variable_label` is the fallback rather than the rule because it only knows
    how to strip a CF name apart, while :func:`ocean_skill.vars.short_name` knows what
    the package calls things — and a combination spec has no short name to look up.
    Shared by :func:`compare` and :func:`_pooled_labels` so a set's own labels and a
    pooled set's relabelling cannot drift apart.
    """
    base = _short_variable_label_base(spec)
    method = _calculate_method(spec)
    return f"{base} ({method})" if method else base


#: Keys naming the vertical axis in a `select`, in any accepted spelling.
_VERTICAL_KEYS = frozenset({"depth", "Z", "vertical", "z"})

#: The key naming an isopycnal (constant potential density) request. Kept separate
#: from :data:`_VERTICAL_KEYS` rather than folded into it: the two spellings name the
#: *same axis* but ask for different operations (:func:`ocean_skill.roms.to_depth`
#: vs. :func:`ocean_skill.roms.to_sigma0`), and a `select` naming both is a
#: contradiction :func:`_prepare` refuses rather than picking one silently.
_ISOPYCNAL_KEYS = frozenset({"sigma0"})


#: Every key naming a vertical request, depth or density alike.
_ANY_VERTICAL_KEYS = _VERTICAL_KEYS | _ISOPYCNAL_KEYS


def _vertical_only(agg: dict[str, Any] | None) -> dict[str, Any]:
    """Return the part of an aggregation spec addressing the vertical axis."""
    return {k: v for k, v in (agg or {}).items() if k in _ANY_VERTICAL_KEYS}


def _without_vertical(agg: dict[str, Any] | None) -> dict[str, Any]:
    """Return the part of an aggregation spec for every axis but the vertical."""
    return {k: v for k, v in (agg or {}).items() if k not in _ANY_VERTICAL_KEYS}


def _selected_depth(select: dict[str, Any]) -> Any:
    """Return what a ``select`` asks for vertically, whichever spelling it used.

    ``compare`` writes ``"depth"``, but a ``Comparison`` built directly keeps the
    ``select`` it was given, and ``{"Z": 100}`` is as valid there as anywhere else. For
    labels the difference is not cosmetic: reading only ``"depth"`` reports a comparison
    at 100 m as ``surface``, and two comparisons at different depths as the same point.

    An isopycnal request (``{"sigma0": ...}``) comes back wrapped in a one-key dict
    rather than as the bare value, so :func:`_depth_label` can tell "26.5" the depth
    apart from "26.5" the density anomaly — the two mean wildly different things and
    a caller reading only the number back would not be able to tell which this was.
    """
    for key in ("depth", "Z", "z", "vertical"):
        if key in select:
            return select[key]
    if "sigma0" in select:
        return {"sigma0": select["sigma0"]}
    return SURFACE


def _display_depth(variable: Any, select: dict[str, Any]) -> Any:
    """The depth a comparison should report in labels/repr/metrics.

    A calculated diagnostic (mixed layer depth, ...) has no vertical axis at all --
    see :data:`NO_VERTICAL_AXIS` -- so :func:`_selected_depth`'s own default
    (:data:`SURFACE`, meant for an ordinary field nothing narrowed vertically) would
    misreport where in the column the number actually came from.

    A pair-spec select reports the **test** side's depth -- an arbitrary but
    necessary choice, matching :attr:`Comparison.standard_name`'s own precedent of
    letting the test side name the figure when the two lanes could disagree.
    """
    if _is_calculated(variable):
        return NO_VERTICAL_AXIS
    return _selected_depth(select_for(select, "test"))


def _surface_and_levels(sub, meta, name: str, depths) -> Any:
    """Assemble a ``z`` axis mixing the model's own surface with interpolated levels.

    ``select={"depth": ["surface", 50, 100]}`` asks for levels no single vertical
    operation can produce: ``"surface"`` is the native top cell
    (:func:`ocean_skill.roms.surface` — interpolating to 0 m is NaN wherever the top
    cell centre sits deeper), while the numbers are fixed levels via
    :func:`ocean_skill.roms.to_depth`. So the two are computed separately and
    concatenated along ``z`` in the order asked for.

    The surface layer sits at ``z=0.0`` — a coordinate value, not a claim it was
    interpolated there. The coordinate has to stay numeric (the lane cache is zarr,
    which cannot hold a mixed-type axis), so the honest spelling rides in a
    ``level_labels`` attr instead, which row labels read
    (:func:`ocean_skill.plot.matplotlib_renderer.facet_labels`).
    """
    import xarray as xr

    from ocean_skill import roms

    numeric = [float(d) for d in depths if not is_surface_request(d)]
    levels = roms.to_depth(sub, meta, numeric)[name] if numeric else None
    top = roms.surface(sub, meta)[name]
    if levels is not None:
        # expand_dims puts z first where the transform put it last; concat needs one
        # order, and the transform's is the one the numeric-only path already has.
        top = top.expand_dims("z").transpose(*levels.dims)
    else:
        top = top.expand_dims("z")
    top = top.assign_coords(z=[0.0])

    pieces, i = [], 0
    for d in depths:
        if is_surface_request(d):
            pieces.append(top)
        else:
            pieces.append(levels.isel(z=[i]))
            i += 1
    # drop_conflicts rather than the default first-wins: the transform can shed
    # attrs the surface layer kept (units), and which piece comes first is the
    # caller's ordering, not a fact about the field.
    da = xr.concat(pieces, dim="z", combine_attrs="drop_conflicts")
    da["z"].attrs["level_labels"] = [_depth_label(d) for d in depths]
    return da.to_dataset(name=name)


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
#:
#: A surviving axis of length 1 is the one exception, and not really an exception at
#: all: squeezing it changes no number (the one value already *is* the mean, the max,
#: the member), so there is no choice being made on the caller's behalf for
#: :func:`_require_reduced` to ask about. A WOA climatology's ``time=1`` therefore
#: needs no ``aggregate={"time": "mean"}`` boilerplate; a real, multi-step axis still
#: does.
NO_AGGREGATION: dict[str, Any] = {}


def _require_reduced(da, role: str, source: str):
    """Squeeze away singleton axes, then raise unless what's left is a single map.

    A comparison regrids one field onto another and differences them, so it has to be
    2-D; :func:`ocean_skill.align._require_2d` enforces the same thing one step later,
    where it protects xesmf. This exists in front of it for two reasons: it can name the
    *source* whose lane is at fault, which align cannot (it sees two anonymous arrays),
    and it runs before :func:`prepare_source` computes anything, so being told to choose
    costs nothing rather than costing the whole vertical transform first.

    A non-spatial axis of size 1 is squeezed away first, with ``drop=True`` — its
    coordinate is discarded, not kept as a scalar. Keeping it would seem the more
    generous move (provenance for a title), but neither title nor depth actually reads
    it: the displayed time comes from the caller's own ``select`` (:func:`_display_time`),
    and depth travels as ``attrs["actual_depth"]``, captured before this ever runs. What
    a kept scalar coord *would* do is collide: a test lane squeezed from a real date and
    a WOA reference squeezed from its undecoded numeric time disagree on ``time``'s
    dtype, and :func:`ocean_skill.align.align`'s ``xr.Dataset({test, reference,
    difference})`` refuses to merge two conflicting scalar coordinates under one name.
    Dropping it sidesteps that for free. Only genuinely ambiguous axes — size greater
    than 1 — are named in the error below; a caller's ``aggregate={"time": "mean"}``
    is worth asking for only where it would change a number.

    Returns the (possibly squeezed) ``da`` so a caller can use the reduced result
    rather than the one it checked.
    """
    from ocean_skill.align import _lat_name, _lon_name

    spatial: set[str] = set()
    for name in (_lon_name(da), _lat_name(da)):
        if name is not None:
            spatial |= {str(d) for d in da[name].dims}
    singles = [
        str(d) for d in da.dims if d not in spatial and da.sizes[d] == 1
    ]
    if singles:
        da = da.squeeze(dim=singles, drop=True)
    extra = [str(d) for d in da.dims if d not in spatial]
    if not extra:
        return da
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


def _select_horizontal_then_aggregate(
    da, horizontal: dict[str, Any], agg: dict[str, Any] | None, source: str
):
    """Apply the horizontal ``select`` and non-vertical ``aggregate``, in that order.

    A ``select`` key ordinarily has to name an axis already standing on ``da``
    (selection precedes reduction — see the ordering note in :func:`_prepare`), but
    it can also name an axis the aggregation itself *creates*:
    ``{"time": {"groupby": "month", "reduce": "mean"}}`` renames ``time`` to
    ``month`` (:func:`ocean_skill.operators.aggregate`), so a caller who wants one
    month of that climatology writes ``select={"month": 1}`` — a key that matches
    nothing until *after* the aggregate has run. Such a key is given a second try
    once the aggregate has run, rather than being silently skipped the way an axis
    a variable genuinely lacks is (:func:`ocean_skill.operators.select`'s own
    contract, needed so one shared spec can cover several variables that don't all
    carry the same axes).

    A key still unmatched after *both* tries draws a warning naming it, rather
    than vanishing without a trace — it is usually a typo, or a misunderstanding
    of what the aggregate produced (a groupby renames its dim to the grouping
    key), but the warning's own wording also covers the ordinary case of a select
    shared across variables that don't all carry the same axis, so it can be
    ignored there.
    """
    import warnings

    from ocean_skill import _stacklevel, operators

    applied = {
        name
        for name in horizontal
        if (dim := operators.resolve_dim(da, name)) is not None and dim in da.dims
    }
    # A scalar lon+lat pair is a point request even when neither key names a
    # dimension on its own -- ROMS's curvilinear lon_rho/lat_rho are 2-D, so
    # resolve_dim above finds nothing for them, yet operators.select's point
    # branch (operators.point_in_spec) can still sample the source there. Without
    # this, a curvilinear point select would fall through to "matched no axis"
    # (see the warning below) instead of narrowing to a place.
    if hit := operators._point_selectable(da, horizontal):
        applied |= {hit[0], hit[1]}
    da = operators.select(
        da, {k: v for k, v in horizontal.items() if k in applied}, subject=source
    )
    da = operators.aggregate(da, agg)
    deferred = {k: v for k, v in horizontal.items() if k not in applied}
    if not deferred:
        return da
    still_unmatched = {
        k
        for k in deferred
        if (dim := operators.resolve_dim(da, k)) is None or dim not in da.dims
    }
    if to_apply := {k: v for k, v in deferred.items() if k not in still_unmatched}:
        da = operators.select(da, to_apply, subject=source)
    if still_unmatched:
        warnings.warn(
            f"{source!r}: select key(s) {sorted(still_unmatched)} matched no axis "
            f"before or after aggregate={agg!r} ran; the standing axes are "
            f"{sorted(da.dims)}. If one of these names an axis the aggregate "
            "creates -- a groupby renames its dim to the grouping key, e.g. time -> "
            "month -- check the spelling against what the aggregate actually "
            "produced. Otherwise this is the ordinary case of a select shared "
            "across variables that don't all carry the same axis, and can be "
            "ignored.",
            stacklevel=_stacklevel.find(),
        )
    return da


def _prepare(
    obj,
    meta: dict[str, Any],
    variable: Any,
    select: dict[str, Any],
    aggregate: dict[str, Any] | None = None,
    *,
    source: str = "<unnamed>",
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
    sigma = select.get("sigma0")
    if depth is not None and sigma is not None:
        raise ValueError(
            f"select cannot ask for both a depth ({depth!r}) and a density surface "
            f"(sigma0={sigma!r}) -- pick one vertical request."
        )
    surface = is_surface_request(depth)
    band = is_depth_band(depth)
    agg = NO_AGGREGATION if aggregate is None else aggregate
    # A registered calculator (mixed layer depth, ...) reads the whole water column
    # itself and returns a field with no vertical axis at all -- there is nothing
    # left for the surface/to_depth/depth_band machinery below to do, and a depth
    # selection alongside one is a contradiction worth saying so about rather than
    # silently ignoring (see the ValueError below). Expanded through DERIVED first:
    # a registered name pointing at a calculate-spec is one in every way that
    # matters here, even spelled as a plain string (see _expand_derived).
    expanded_variable = _expand_derived(variable)
    calculated = isinstance(expanded_variable, dict) and "calculate" in expanded_variable

    da = operators.resolve_variable(obj, variable)
    if da is None:
        return None, None

    # A plain surface request is a free isel -- unlike to_depth/depth_band/to_sigma0,
    # it needs no grid attached and no water column, just the top s-level -- so it is
    # worth taking *before* the non-vertical reduction below, not after. Left in the
    # usual place (inside the elif meta.get("model") == "roms" ladder further down),
    # the reduction would run over every s_rho level a source happens to store, only
    # for the ladder to throw all but the top one away; the isel there becomes a
    # no-op once it has already happened here. Hoisted only for the plain scalar
    # case -- a band, a level list, or an isopycnal slice all still need the full
    # column, so they are left for the ladder.
    if not calculated and sigma is None and surface and meta.get("model") == "roms":
        name = da.name or "field"
        da = roms.surface(da.to_dataset(name=name), meta)[name]

    # Order matters twice over. Selection precedes reduction, or "the mean of
    # January" would average the whole record. And the *non-vertical* reduction runs
    # before the vertical step, so an expensive s-coordinate transform sees as few
    # fields as the reduction leaves it -- one, for a time mean; every step, now that
    # no reduction is the default, which is the cost of asking for every step. The
    # vertical part of the aggregation then collapses whatever the vertical selection
    # left standing. A select key can also name an axis the aggregation *creates*
    # (groupby renames its dim to the grouping key) rather than one already
    # standing -- see :func:`_select_horizontal_then_aggregate`, which gives such a
    # key a second try once the aggregate has run.
    horizontal = {k: v for k, v in select.items() if k not in _ANY_VERTICAL_KEYS}
    da = _select_horizontal_then_aggregate(
        da, horizontal, _without_vertical(agg), source
    )

    if calculated:
        bad = "sigma0" if sigma is not None else "depth" if depth is not None else None
        if bad is not None:
            raise ValueError(
                f"{variable!r} is a registered calculator, which already reduces "
                "the vertical axis itself (mixed layer depth is a single number per "
                "water column, not a level of one) -- "
                f"select={{{bad!r}: ...}} does not apply to it. Drop the {bad} key, "
                "or select on one of the plain variables it is computed from instead."
            )
    elif sigma is not None and meta.get("model") != "roms":
        raise ValueError(
            f"select={{'sigma0': {sigma!r}}} needs a ROMS source: an isopycnal is "
            "read off the model's own temperature/salinity field, and there is no "
            "way to compute one for an observational or already-gridded product "
            "here. Use select={'depth': ...} instead, or apply this to the model "
            "source directly."
        )
    elif meta.get("model") == "roms":
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
        if sigma is not None:
            # An isopycnal slice needs the full water column of temperature and
            # salinity, not just the one variable this lane resolved -- reduced by
            # the *same* horizontal select and non-vertical aggregate the sliced
            # variable already went through (not the raw column), or the target
            # density would still vary along an axis (e.g. time) the field being
            # sliced no longer has, which xgcm's transform cannot reconcile.
            for standard_name in (
                "sea_water_potential_temperature",
                "sea_water_practical_salinity",
            ):
                column = units.find_variable(obj, standard_name)
                if column is None:
                    raise ValueError(
                        f"an isopycnal slice needs {standard_name!r}, which is not "
                        "in this dataset (or not standardized to that name -- "
                        "check the catalog entry's standard_names map, or that the "
                        "source actually carries it)."
                    )
                column = _select_horizontal_then_aggregate(
                    column, horizontal, _without_vertical(agg), source
                )
                sub = sub.assign({standard_name: column})
            targets = (
                [float(v) for v in sigma]
                if isinstance(sigma, list | tuple)
                else float(sigma)
            )
            sub = roms.to_sigma0(sub, meta, targets)
        elif surface:
            # A no-op when the hoist above already ran (s_dim is gone from sub's dims,
            # so roms.surface's own guard skips the isel) -- kept unconditional rather
            # than tracked with a flag, since re-entering an already-surfaced dataset
            # costs nothing and one fewer branch is one fewer thing to keep in sync.
            sub = roms.surface(sub, meta)
        elif band:
            # A band is averaged over native cells with thickness weights, not
            # interpolated: above the shallowest cell *centre* -- 7 m down in deep
            # water on this grid -- there is nothing to interpolate from, so a
            # target grid over 0-10 m would be mostly NaN offshore.
            # A *selection*: keeps the cells and their thickness weights, so the
            # vertical aggregation below decides how to collapse them.
            sub = roms.depth_band(sub, meta, depth["min"], depth["max"])
        elif isinstance(depth, list | tuple) and any(
            is_surface_request(d) for d in depth
        ):
            # "surface" beside numbers, e.g. ["surface", 50, 100]: no single
            # vertical operation produces that, so the levels are assembled.
            sub = _surface_and_levels(sub, meta, name, depth)
        else:
            # A list interpolates to several levels in one field, which the vertical
            # aggregation then collapses; a scalar gives one level and no axis.
            try:
                targets = (
                    [float(d) for d in depth]
                    if isinstance(depth, list | tuple)
                    else float(depth)
                )
            except (TypeError, ValueError):
                raise ValueError(
                    f"cannot read {depth!r} as a depth selection: use metres (50), "
                    '"surface", a band ({"min": 0, "max": 10}), or a list mixing '
                    'metres and "surface" (["surface", 50, 100]).'
                ) from None
            sub = roms.to_depth(sub, meta, targets)
        da = sub[name]
        # Squeeze only a single interpolated level: a scalar depth request collapses
        # the axis by itself (as `.sel` does everywhere), while a list or band leaves
        # several levels for the vertical aggregation to reduce. Squeezing
        # unconditionally used to discard every level but the first, silently.
        if "z" in da.dims and da.sizes["z"] == 1:
            da = da.isel(z=0)
        if "sigma0" in da.dims and da.sizes["sigma0"] == 1:
            da = da.isel(sigma0=0)
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
                # "surface" in a list means what it means as a scalar here: the
                # nearest standard level to 0 m (products report at real levels, so
                # the row honestly labels itself with the level it is).
                targets = [
                    0.0 if is_surface_request(d) else float(d) for d in depth
                ]
                keep = [int(np.abs(levels - t).argmin()) for t in targets]
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


#: Size (bytes) past which a source's largest *storage* chunk is worth a warning, before
#: :func:`_prepare` ever runs. Unlike :data:`LOAD_WARN_BYTES` below, which inspects the
#: field :func:`_prepare` hands back -- already reduced to whatever ``select``/
#: ``aggregate`` left standing, routinely megabytes -- this looks at the chunk dask
#: actually has to read: for a kerchunk reference over per-timestep ROMS output, one
#: whole time record's full water column. A reduction that runs before a selection
#: narrows it (see the ordering note in :func:`_prepare`) pulls every one of those into
#: memory at once, on however many threads dask has -- exactly the failure
#: ``LOAD_WARN_BYTES`` cannot see coming, since by the time it looks the damage (or the
#: kernel) is already done.
CHUNK_WARN_BYTES = 512 * 1024**2


def _warn_if_chunk_is_large(obj: Any, source: str) -> None:
    """Warn once if a variable in ``obj`` stores a chunk over :data:`CHUNK_WARN_BYTES`.

    Runs on whatever :func:`ocean_skill.read` returned, before :func:`_prepare`
    reduces anything. Silently does nothing for a source with no ``data_vars`` at
    all -- a DataFrame station, a read that returned ``None`` -- and for one with
    no dask-backed variables (already loaded into memory, nothing left for a chunk
    to hide behind).
    """
    data_vars = getattr(obj, "data_vars", None)
    if data_vars is None:
        return
    worst_bytes = 0
    worst_name = None
    for name, da in data_vars.items():
        if da.chunks is None:
            continue
        max_elems = 1
        for sizes in da.chunks:
            max_elems *= max(sizes)
        nbytes = max_elems * da.dtype.itemsize
        if nbytes > worst_bytes:
            worst_bytes, worst_name = nbytes, name
    if worst_bytes > CHUNK_WARN_BYTES:
        import warnings

        from ocean_skill import _stacklevel

        warnings.warn(
            f"{source!r}'s {worst_name!r} is stored in chunks up to "
            f"{worst_bytes / 1024**2:.0f} MB. Reading it -- and any reduction that "
            "runs before a selection narrows it, such as a time mean ahead of a "
            "depth pick -- pulls that much into memory per chunk, times however "
            "many chunks the reduction touches at once. If the store is "
            "uncompressed, rebuild its kerchunk reference with a smaller chunk "
            "grid (ocean_skill.build.make_kerchunk(..., target_chunk_mb=...) or "
            "subchunk={...}). A compressed store can't be split after the fact -- "
            "repack the source files with smaller chunks (nccopy/h5repack) "
            "instead, or limit concurrency in the meantime with "
            "dask.config.set(num_workers=...).",
            stacklevel=_stacklevel.find(),
        )


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

    That check also squeezes away any singleton axis it finds (see
    :func:`_require_reduced`), and the squeeze is applied only to what this function
    *returns*, never to what it caches: the cache key has no role in it, so the stored
    field must not have one baked into its shape either, or whichever caller happens to
    fill the entry first decides what a later, differently-tolerant caller receives.

    ``bbox`` crops the lane to ``(lon_min, lat_min, lon_max, lat_max)`` plus
    :data:`ocean_skill.align.DEFAULT_PAD` *before* the load below, and joins the cache
    key. :func:`ocean_skill.align.align` does this crop anyway, but only once both lanes
    are in memory — fine for a single global map, the dominant cost for a global product
    that kept a long time axis. Passing the test lane's own extent here changes nothing
    about the result and a great deal about the peak memory. It does cost this
    lane its cross-model reuse (a reference cropped to one model's domain is not the one
    another model wants), which is why the caller passes it only when it is worth that.

    ``time_window`` is the same idea along time: ``(start, stop)`` the lane is cropped
    to, and part of the cache key for the same reason ``bbox`` is. A skill map derives
    it from its test lane (see :meth:`Comparison.align`), so the caller knows the window
    before this reads anything — which is what lets it reach an ERDDAP table as a
    server-side constraint rather than as a crop applied to a record already downloaded.

    Returns ``(DataArray, actual_depth)``, or ``(None, None)`` if the source does not
    carry the variable.
    """
    import ocean_skill as osk
    from ocean_skill import cache as _cache
    from ocean_skill.catalog import resolve
    from ocean_skill.sources import erddap_constraints

    key_select: dict[str, Any] = {**(select or {}), "_aggregate": aggregate}
    if bbox is not None:
        # rounded so that float noise in an extent does not fragment the cache
        key_select["_bbox"] = [round(float(b), 4) for b in bbox]
    if time_window is not None:
        # A cropped lane is not the uncropped one, and two test lanes over the same
        # region but different years share a `_bbox`. Without this they would share an
        # entry as well, and the second would be served the first one's window.
        key_select["_time_window"] = [str(w) for w in time_window]
    key = _cache.key_for_prepared(
        source=source,
        variable=variable,
        select=key_select,
    )
    if use_cache and not refresh:
        hit = _cache.load_field(key)
        if hit is not None:
            da_hit, depth_hit = hit
            if da_hit is not None and require_reduced:
                da_hit = _require_reduced(da_hit, require_reduced, source)
            return da_hit, depth_hit

    # An ERDDAP table is fetched whole in one request, so the time narrowing below has
    # to travel with the request rather than follow it -- see erddap_constraints, which
    # returns nothing for every other kind of source. The same `select` is applied in
    # memory regardless, so this changes the size of the download and nothing else.
    meta = resolve(source).metadata
    constraints = erddap_constraints(meta, select, time_window)
    obj = osk.read(source, constraints=constraints) if constraints else osk.read(source)
    _warn_if_chunk_is_large(obj, source)
    da, depth = _prepare(
        obj, meta, variable, dict(select or {}), aggregate, source=source
    )
    if da is not None and require_reduced:
        # A fail-fast check only -- before .load(), while it is still free -- see the
        # docstring. Its return (squeezed or not) is deliberately dropped: what gets
        # cached and crop-processed below must stay unsqueezed, and the squeeze that
        # matters happens again, on the way out, right before this function returns.
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
    # Squeezed on the way out, never into the cache -- see the docstring above.
    if da is not None and require_reduced:
        da = _require_reduced(da, require_reduced, source)
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
        # one consistent name. A dict is a combination spec (or a {"test",
        # "reference"} pair-spec, resolved side by side below) for
        # ocean_skill.operators and is carried through as given.
        if isinstance(variable, dict):
            _require_pair_spec(variable)
        if is_pair_spec(variable):
            self.variable = {
                **variable,
                "test": (
                    resolve_and_report(variable["test"], context="Comparison variable=")
                    if isinstance(variable["test"], str)
                    else variable["test"]
                ),
                "reference": (
                    resolve_and_report(
                        variable["reference"], context="Comparison variable="
                    )
                    if isinstance(variable["reference"], str)
                    else variable["reference"]
                ),
            }
        else:
            self.variable = (
                resolve_and_report(variable, context="Comparison variable=")
                if isinstance(variable, str)
                else variable
            )
        # Either may also be a {"test": ..., "reference": ...} pair-spec, for a
        # reference whose axis cannot take the same value the test lane needs (a WOA
        # climatology's undecoded, numeric time cannot be asked for "2010-01" the way
        # the model's calendar time can) -- see select_for/aggregate_for, which each
        # lane's own prepare() call resolves this through, mirroring variable_for.
        self.select = _normalize_pair(select, "select", normalize_side=as_select)
        self.aggregate = _normalize_pair(aggregate, "aggregate")
        if is_pair_spec(self.aggregate):
            for role, side in self.aggregate.items():
                if side is not None and not isinstance(side, dict):
                    raise TypeError(
                        f"aggregate={{'test': ..., 'reference': ...}}'s {role!r} "
                        f"side must be a dict or None, matching aggregate= itself; "
                        f"got {side!r}."
                    )
        self.method = method
        # An explicit over= always wins; otherwise the reference's featureType decides,
        # since a station reference has a time axis and no map to draw -- and failing
        # that, a select that already narrows the reference to one position implies the
        # same thing, since there is no map left to draw either way (a gridded model
        # asked for one lon/lat is exactly as reduced as a mooring is by nature). The
        # reason is kept so the family this ends up choosing can be traced to what
        # chose it.
        if over is None:
            over, self.over_reason = _implied_over(reference)
            if over is None and self._point_select_implies_time():
                over = "time"
                self.over_reason = "the select narrows the reference to one position"
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

    def _point_select_implies_time(self) -> bool:
        """Whether the select alone narrows the reference to a place worth a line.

        Called only when neither an explicit ``over=`` nor the reference's own
        featureType (:func:`_implied_over`) already settled it. A select that pins
        the reference's lon *and* lat to scalars (:func:`ocean_skill.operators
        .point_in_spec`) leaves it exactly as reduced as a real mooring is by
        nature -- there is no map left to draw either way -- unless the
        reference's own ``aggregate`` has also collapsed time itself
        (:func:`_collapses_time`), in which case there is no axis left for a line
        to run along and inferring one would be wrong rather than merely unasked.
        """
        from ocean_skill import operators

        ref_select = select_for(self.select, "reference")
        if operators.point_in_spec(ref_select) is None:
            return False
        return not _collapses_time(aggregate_for(self.aggregate, "reference"))

    def _point_route(self) -> tuple[str, str, float, float] | None:
        """The point key/lon/lat a shared select narrows both lanes to, if routing.

        ``None`` unless ``over`` is set (nothing to route otherwise -- a map has no
        line to sample a point onto) and the select is shared rather than a
        pair-spec: a deliberate ``select={"test": ..., "reference": ...}`` split
        names two positions on purpose (two moorings, or a hand-narrowed test), and
        :func:`ocean_skill.align.align`'s two-point branch -- which compares them
        as given and warns if they are far apart -- has to see both lanes
        unrouted, or a real mismatch would go unreported.

        Used by :meth:`align` to keep the test lane gridded rather than also
        narrowing it to the same place, and by :attr:`_cache_key` so a routed
        comparison and a pair-spec one never share a cache entry despite an
        otherwise-identical ``select``.
        """
        if self.over is None or is_pair_spec(self.select):
            return None
        from ocean_skill import operators

        return operators.point_in_spec(self.select)

    @property
    def standard_name(self) -> str | None:
        """The CF name this comparison represents, for colormaps/labels/metrics.

        A combination carries it explicitly (``{"sum": [...], "standard_name": ...}``);
        without one there is no single CF name, and downstream falls back to defaults.

        A pair-spec's explicit ``standard_name`` wins the same way; absent that, the
        **test** side names the figure -- an arbitrary but necessary choice, since the
        two sides may resolve to different CF names (that mismatch itself is reported
        once, in :meth:`align`, rather than silently picked between here).
        """
        from ocean_skill.operators import DERIVED

        spec = self.variable
        if is_pair_spec(spec):
            if name := spec.get("standard_name"):
                return name
            spec = spec["test"]
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
        if self._point_route() is not None:
            # Without this, a routed comparison (test kept gridded, sampled at the
            # reference's snapped cell) and the same select= run before routing
            # existed (both lanes independently narrowed to a point, compared as
            # given) would be byte-identical keys for two different results -- a
            # warm cache from the old behavior would silently serve it forever.
            extra["_point_sample"] = True
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
        drop_keys: tuple[str, ...] = (),
    ):
        """Reduce one source to its comparable field, via the lane cache.

        ``role`` names the lane in the error raised if the reduction left it more than a
        map — the one thing a comparison's use of :func:`prepare_source` needs that a
        field's does not. With ``over`` set that check is *not* wanted: the axis it
        would refuse is the one this comparison exists to score against, and
        :func:`ocean_skill.align.align` still refuses any further one.

        ``bbox`` crops the lane before it is read into memory; see
        :func:`prepare_source`.

        ``drop_keys`` removes keys from this lane's own select before it is prepared
        — used by :meth:`align` to keep the test lane gridded when a shared point
        select is routed through :func:`ocean_skill.align.sample_at` instead of also
        narrowing the test independently (see :meth:`_point_route`).
        """
        select = select_for(self.select, role)
        if drop_keys:
            select = {k: v for k, v in select.items() if k not in drop_keys}
        return prepare_source(
            source,
            variable_for(self.variable, role),
            select,
            aggregate_for(self.aggregate, role),
            use_cache=use_cache,
            refresh=refresh,
            require_reduced=None if self.over else role,
            bbox=bbox,
            time_window=time_window,
        )

    def _warn_on_pair_spec_mismatch(
        self, test_da, reference_da, *, trust_name_fallback: bool = True
    ) -> None:
        """Warn once when a pair-spec's two sides resolve to different standard_names.

        An explicit ``standard_name`` on the pair-spec settles the question, so this
        never fires (the caller has already said the two recipes describe the same
        quantity). Without one, a mismatch means the comparison may not be scoring
        one quantity against itself -- the metrics have no way to say that on their
        own (see the standing "warn, don't annotate" rule), so this is the one place
        it gets said, once, before anything downstream treats the pair as settled --
        including on a *cached* result, or every process after the one that first
        computed the pair would silently miss the one safety net an unlabelled
        pair-spec has.

        Checked against the *resolved* fields rather than the specs themselves: a
        ``{"calculate": ...}`` spec's output name is not statically knowable (only its
        *inputs* are, via :data:`ocean_skill.operators.CALCULATOR_INPUTS`), so this has
        to wait until both lanes have actually been read. ``attrs["standard_name"] or
        .name`` is the same fallback :func:`ocean_skill.units.find_variable` itself
        uses -- a properly standardized field is keyed *by* its standard_name even
        when nothing stamped the attribute explicitly.

        ``trust_name_fallback=False`` for the two arrays :meth:`align` pulls out of a
        *cached, aligned* result: :func:`ocean_skill.align.align` names its output
        variables literally ``"test"``/``"reference"`` (see its ``test_name``/
        ``reference_name`` parameters), so ``.name`` there is always ``"test"`` and
        always ``"reference"`` -- always unequal, regardless of what the original
        fields were -- and falling back to it would warn on *every* cache hit rather
        than only a genuine mismatch. Only ``attrs["standard_name"]`` (which the
        regrid step preserves, ``keep_attrs=True``) is trustworthy there; if it is
        missing on either side, there is nothing left to check and this stays silent
        rather than manufacture a comparison against a name that was never real.
        """
        import warnings

        from ocean_skill import _stacklevel

        spec = self.variable
        if not is_pair_spec(spec) or spec.get("standard_name"):
            return
        test_name = test_da.attrs.get("standard_name")
        reference_name = reference_da.attrs.get("standard_name")
        if trust_name_fallback:
            test_name = test_name or test_da.name
            reference_name = reference_name or reference_da.name
        if test_name and reference_name and test_name != reference_name:
            warnings.warn(
                f"this comparison's test side resolves to {test_name!r} but its "
                f"reference side resolves to {reference_name!r} -- a pair-spec with "
                "no explicit 'standard_name' assumes the two recipes describe the "
                "same quantity, and here they may not. The metrics will not say so "
                "on their own; pass standard_name= on the pair-spec if this is "
                "intentional, or check the two method/variable choices if it is not.",
                stacklevel=_stacklevel.find(),
            )

    # -- pipeline ---------------------------------------------------------------
    def align(self, *, refresh: bool = False):
        """Read both sources, reduce them, and regrid onto the coarser lane's grid.

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
                # restores the same state a freshly computed one would have -- the
                # pair-spec mismatch check is the same restoration, off the cached
                # fields' own attrs rather than the (always "test"/"reference",
                # unusable here) array names.
                self._actual_depth = hit.attrs.get("actual_depth")
                self._warn_on_pair_spec_mismatch(
                    hit["test"], hit["reference"], trust_name_fallback=False
                )
                return self._aligned

        # The test lane goes first when an axis is being kept, so the reference can be
        # cropped to its extent *before* being read (see prepare_source's bbox=).
        # align() crops it anyway, but only once both lanes are in memory, and a product
        # that kept a year of daily maps is the wrong thing to hold whole. Exact, not an
        # approximation: the bbox and the pad are the ones align() would have used.
        #
        # A select shared by both lanes that narrows to one position is routed
        # differently: the point names where to *sample* the test, not a second place
        # to also narrow it to on its own -- narrowing both independently would compare
        # two different grids' nearest cells to each other, which is the "km apart"
        # mismatch align()'s two-point branch exists to warn about, not the co-located
        # sample this is meant to be. So the point is dropped from the test lane's own
        # select and used as a small bbox instead, leaving the test gridded for align()
        # to sample properly, at the reference's own (possibly curvilinear-snapped)
        # position.
        route = self._point_route()
        drop_keys = route[:2] if route is not None else ()
        point_bbox = (
            (route[2], route[3], route[2], route[3]) if route is not None else None
        )
        try:
            t, _ = self._prepare_lane(
                self.test_name,
                use_cache,
                refresh,
                role="test",
                bbox=point_bbox,
                drop_keys=drop_keys,
            )
        except ValueError as err:
            # A coarse test grid, or a point near the edge of its extent: the pad
            # subset_to_bbox adds around one location can still miss the nearest cell.
            # Retrying without the bbox reads the whole lane instead of failing the
            # comparison outright, and lets align()'s own sample_at report the
            # (possibly large) offset the way it already does for a real station.
            if point_bbox is None or "no overlap" not in str(err):
                raise
            t, _ = self._prepare_lane(
                self.test_name, use_cache, refresh, role="test", drop_keys=drop_keys
            )
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
            missing_role = "reference" if r is None else "test"
            missing = self.reference_name if r is None else self.test_name
            raise KeyError(
                f"{variable_for(self.variable, missing_role)!r} not available in "
                f"{missing!r}"
            )
        self._warn_on_pair_spec_mismatch(t, r)
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
            test_metadata=self._test_metadata(),
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

    def _test_metadata(self) -> dict[str, Any] | None:
        """Return the test's catalog metadata, or ``None`` if it will not resolve.

        The mirror of :meth:`_reference_metadata`: only wanted when a finer reference
        forces the axis matching to ask the test lane whether *its* steps are
        averages or instants (see :func:`ocean_skill.align.resolve_match_method`).
        """
        from ocean_skill.catalog import resolve

        try:
            return resolve(self.test_name).metadata
        except Exception:
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
                # The position itself rides along too — it lives on `aligned.attrs`
                # (written by align._align_at_point) and would otherwise never reach
                # the metrics record, which is the one thing a spatial map of many
                # stations' metrics (osk.map_metrics) needs from each of them.
                **(
                    {
                        "weighted": False,
                        "sample_noun": "time steps",
                        "station_lon": self.aligned.attrs.get("station_lon"),
                        "station_lat": self.aligned.attrs.get("station_lat"),
                    }
                    if self.is_series
                    else {}
                ),
                variable=self.standard_name or str(self.variable),
                test=self.test_name,
                reference=self.reference_name,
                depth=_display_depth(self.variable, self.select),
                time=_display_time(self.select),
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
        """Return the ``test − reference`` field on the aligned (coarser) grid."""
        return self.aligned["difference"]

    def as_item(self) -> dict[str, Any]:
        """Return this comparison as a spec item.

        Two shapes, because a scored comparison is a different figure: with ``over`` set
        the item carries the metric maps and the overall record that annotates them, and
        with it unset the aligned trio the ``test | reference | difference`` row draws.
        """
        # The depth and time a select= has collapsed to one map, spelled for a title —
        # what a single field_row has no row label to say. Uses the same test-side
        # precedent as standard_name (a calculated diagnostic reports NO_VERTICAL_AXIS,
        # which is not a depth to name, so it is dropped rather than shown as "n/a").
        depth = _depth_label(_display_depth(self.variable, self.select))
        selected_time = _display_time(self.select)
        common = {
            "metrics": self.metrics(),
            "units": self.aligned["reference"].attrs.get("units"),
            "standard_name": self.standard_name,
            "depth": None if depth == NO_VERTICAL_AXIS else depth,
            "time": None if selected_time is None else _time_label(selected_time),
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
        if family != "series" and "domain" not in kwargs:
            # Outline the test (model) source's own true grid shape when the catalog
            # declares one, falling back to its bbox otherwise — matching Abigale
            # Wyatt's side-by-side plots. Pass domain=None to suppress it, or your own
            # bbox/ring to override; checking "not in kwargs" (rather than
            # kwargs.setdefault) keeps that override working once the default value
            # is an ndarray, whose truthiness setdefault can't rely on. A line plot has
            # no map to outline, and series() would refuse the option outright.
            convention = self.aligned.attrs.get("lon_convention")
            outline = _outline_of(self.test_name, convention)
            kwargs["domain"] = (
                outline if outline is not None else _domain_of(self.test_name)
            )
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
        time = _display_time(self.select)
        at_time = f" @ {_time_label(time)}" if time is not None else ""
        return (
            f"<Comparison {_variable_label(self.variable)[:24]} "
            f"{self.test_name} vs {self.reference_name} "
            f"@ {_depth_label(_display_depth(self.variable, self.select))}"
            f"{at_time}{scored}>"
        )


def _canonical(obj: Any) -> str:
    """A dict-key-order-insensitive representation of ``obj``.

    Plain ``repr()`` treats ``{"test": "a", "reference": "b"}`` and ``{"reference":
    "b", "test": "a"}`` as different strings even though they are the same spec --
    which would let two logically-identical pair-spec comparisons escape
    :func:`_flatten`'s dedup (drawn twice, one atop the other) while
    :func:`ocean_skill.cache.key_for`'s own ``json.dumps(..., sort_keys=True)``
    treats them as the same cache entry. Matching that canonicalization here keeps
    pooling's notion of "the same comparison" from disagreeing with the disk
    cache's. ``default=str`` is the same fallback ``key_for`` uses, for values
    ``json`` cannot serialize on its own (a ``slice`` in a ``select``, a numpy
    scalar); anything neither can handle falls back to plain ``repr``.
    """
    import json

    try:
        return json.dumps(obj, sort_keys=True, default=str)
    except TypeError:
        return repr(obj)


def _identity(c) -> tuple:
    """Return what makes two comparisons the same one: their whole specification.

    Built from the object's own attributes rather than from ``c._cache_key``, which
    hashes very nearly this but exists to name a zarr store — pooling has no business
    depending on the cache's format version or on what it deliberately leaves out.
    """
    return (
        getattr(c, "test_name", None),
        getattr(c, "reference_name", None),
        _canonical(getattr(c, "variable", None)),
        _canonical(getattr(c, "select", None)),
        _canonical(getattr(c, "aggregate", None)),
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
#: with how to read each one off a comparison. The same five a metrics record
#: carries, so ``color_by``/``marker_by`` can group by anything a label can name.
_LABEL_DIMS: tuple[tuple[str, Any], ...] = (
    ("variable", lambda c: _short_variable_label(c.variable)),
    ("depth", lambda c: _depth_label(_display_depth(c.variable, c.select))),
    (
        "time",
        lambda c: (
            _time_label(t)
            if (t := _display_time(c.select)) is not None
            else "full record"
        ),
    ),
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
        if family != "series" and "domain" not in kwargs:
            # Outlines the first row's test (model) true grid shape (or its bbox,
            # lacking one); rows sharing one test source (the common case) all get the
            # same outline. Pass domain=None to suppress, or your own bbox/ring if rows
            # mix test sources with different domains.
            convention = first.aligned.attrs.get("lon_convention")
            outline = _outline_of(first.test_name, convention)
            kwargs["domain"] = (
                outline if outline is not None else _domain_of(first.test_name)
            )
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
        if "domain" not in kwargs:
            convention = first.aligned.attrs.get("lon_convention")
            outline = _outline_of(first.test_name, convention)
            kwargs["domain"] = (
                outline if outline is not None else _domain_of(first.test_name)
            )
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

    def map_metrics(self, *, renderer: str = "matplotlib", **kwargs: Any):
        """Interpolate this set's per-station metrics onto a map, one panel each.

        Every comparison in the set should be a station (a place through time) —
        anything else is skipped with a warning, since it has no single position to
        plot. See :func:`ocean_skill.plot.map_metrics.map_metrics`, which this
        delegates to, for what gets interpolated and how, and what it cannot do
        (route an interpolated surface around land).
        """
        from ocean_skill.plot.map_metrics import map_metrics as _map_metrics

        return _map_metrics(self, renderer=renderer, **kwargs)

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


def _fan_vertical_entries(
    select: dict[str, Any],
    keys: frozenset[str],
    *,
    sugar_label: str = "compare()'s depths=/select vertical",
    example: str = "depths=/select={'sigma0': ...}",
    axis_noun: str = "vertical",
) -> dict[str, Any]:
    """Return the entries (of ``keys``) a compare() fan-out sugar sees.

    ``select`` here is always a dict (never ``None``) by the time :func:`compare`
    calls this -- normalized up front through :func:`_normalize_pair`. For a plain
    select, the entries it carries under ``keys`` are returned as-is. For a
    pair-spec select, both sides must *agree* on any key of ``keys`` present in
    both -- the sugar fans one set of values over both lanes, which only makes
    sense if both lanes are being asked the same question along that axis. A
    genuine conflict raises, naming both sides, rather than silently preferring
    one: there is no depths=/sigma0/times= spelling for a different value per
    lane -- construct :class:`Comparison` directly (with its own per-role select)
    for that.

    Shared between the vertical fan (``depths=``/``select={'sigma0': ...}``) and
    the time fan (``times=``); ``sugar_label``/``example``/``axis_noun`` only
    change the error's wording so each reads as its own sugar rather than the
    other's.
    """
    if not is_pair_spec(select):
        return {k: select[k] for k in keys if k in select}
    merged: dict[str, Any] = {}
    for role in ("test", "reference"):
        side = select[role] or {}
        for k in keys:
            if k not in side:
                continue
            if k in merged and merged[k] != side[k]:
                raise ValueError(
                    f"{sugar_label} sugar fans one value over both lanes, but this "
                    f"pair-spec select disagrees on {k!r}: "
                    f"test={select['test'].get(k)!r}, "
                    f"reference={select['reference'].get(k)!r}. Leave {axis_noun} "
                    f"keys out of a pair-spec select and pass {example} normally "
                    "(it is applied to both sides), or construct Comparison(...) "
                    f"directly for genuinely different per-lane {axis_noun} "
                    "requests."
                )
            merged[k] = side[k]
    return merged


def _fanned_select(
    select: dict[str, Any], fan_key: str, value: Any, calculated: bool
) -> dict[str, Any]:
    """Build the per-comparison select compare()'s depths=/sigma0 fan-out uses.

    Strips any existing vertical key first -- ``depths=``/``select={'sigma0': ...}``
    is sugar that *replaces* whatever vertical entry ``select`` already carried,
    honouring the key the caller used rather than silently overriding it with a
    different spelling -- then writes the fanned value back under ``fan_key``. For a
    pair-spec select this happens on **both** sides, since the sugar asks the same
    vertical question of both lanes (:func:`_fan_vertical_entries` already refused a
    pair select that disagreed with itself on that axis).
    """

    def _strip(side: dict[str, Any] | None) -> dict[str, Any]:
        return {k: v for k, v in (side or {}).items() if k not in _ANY_VERTICAL_KEYS}

    if is_pair_spec(select):
        sel = {"test": _strip(select["test"]), "reference": _strip(select["reference"])}
        if not calculated:
            sel["test"][fan_key] = value
            sel["reference"][fan_key] = value
        return sel
    sel = _strip(select)
    if not calculated:
        sel[fan_key] = value
    return sel


def _selected_time(select: dict[str, Any]) -> Any:
    """Return what a ``select`` asks for along time, or ``None`` if it names none.

    Mirrors :func:`_selected_depth`: ``compare()`` writes plain ``"time"``, but a
    ``Comparison`` built directly (or a pair-spec's own side) may use any spelling
    in :data:`ocean_skill.sources._TIME_KEYS`, and reading only one back would
    silently under-report a caller's own request.
    """
    from ocean_skill.sources import _TIME_KEYS

    for key in _TIME_KEYS:
        if key in select:
            return select[key]
    return None


def _display_time(select: dict[str, Any]) -> Any:
    """The time value a comparison should report in labels/repr/metrics.

    A pair-spec select reports the **test** side, matching :func:`_display_depth`'s
    own precedent: the test side names the figure when the two lanes could
    disagree, which is exactly why a pair-spec select exists.
    """
    return _selected_time(select_for(select, "test"))


def _time_label(value: Any) -> str:
    """Format one fanned time value for labels/repr.

    A ``{"min", "max"}`` window (:func:`_time_select_value`'s fallback for a
    frequency with no single partial-date spelling) reads as ``"start–stop"``,
    each end sliced to ten characters; anything else — the common case, a
    partial-date string like ``"2010-01"`` — is already its own label.
    """
    if isinstance(value, dict) and {"min", "max"} <= set(value):
        return f"{str(value['min'])[:10]}–{str(value['max'])[:10]}"
    return str(value)


def _normalize_time_value(value: Any) -> Any:
    """Return one ``times=`` list entry in its canonical, cache-stable spelling.

    A :class:`~pandas.Timestamp`/``datetime`` and the string naming the same
    instant must hash to the same cache key
    (:func:`ocean_skill.cache.key_for`'s ``default=str`` covers either alone, but
    not consistently against each other), so anything that is not already a
    string or the YAML-friendly ``{"min", "max"}`` window is turned into one here,
    once, rather than trusting every downstream reader to agree.
    """
    if isinstance(value, dict) and {"min", "max"} <= set(value):
        return {"min": str(value["min"]), "max": str(value["max"])}
    return value if isinstance(value, str) else str(value)


def _normalize_times(times: Any) -> tuple[str, Any] | None:
    """Return ``("bins", spec)``, ``("list", tuple)``, or ``None`` for ``times=None``.

    A dict names how to *derive* bins from the test source's own time axis
    (:func:`_time_bins`) — it must carry ``"resample"`` and ``"reduce"``, the same
    vocabulary ``aggregate={"time": ...}`` already uses. Anything else names an
    explicit set of time values, one comparison per entry, mirroring how a bare
    ``depths=50`` is sugar for ``depths=(50,)``.
    """
    if times is None:
        return None
    if isinstance(times, dict):
        if is_pair_spec(times):
            raise ValueError(
                "times= fans one question over both lanes, like depths=; it "
                "takes no {'test': ..., 'reference': ...} pair-spec. Use select= "
                "for a per-lane time entry instead."
            )
        if "groupby" in times:
            raise ValueError(
                f"times={times!r} names a climatology (every January of the "
                "record folded into one field), which is not a set of time "
                "steps to fan into separate comparisons. Build the climatology "
                "with aggregate={'time': {'groupby': ...}} instead, then "
                "osk.field() to plot its months as panels, or select= to pick "
                "one month out of it."
            )
        if "resample" not in times or "reduce" not in times:
            raise ValueError(
                f"times={times!r} needs both 'resample' (a bin width, e.g. "
                "'1MS') and 'reduce' (how each bin collapses, e.g. 'mean') -- "
                "the same vocabulary as aggregate={'time': ...}. Or pass an "
                "explicit list of time values instead, one comparison per entry."
            )
        return "bins", dict(times)
    values = (
        (times,)
        if isinstance(times, str) or not hasattr(times, "__iter__")
        else tuple(times)
    )
    return "list", tuple(_normalize_time_value(v) for v in values)


def _time_bins(source: str, freq: str, window: Any) -> list[tuple[Any, Any]]:
    """Return ``(bin_start, bin_last_value)`` for every resample bin with data.

    Coordinate-only — like :func:`ocean_skill.operators._bin_counts`, which finds
    the bin edges here, reads: "a time coordinate is a small in-memory index and
    already carries everything the count needs" — so resolving ``times=``'s dict
    form against a lazily-opened, multi-file model run costs nothing like opening
    the data itself would. ``window`` narrows which part of the axis counts,
    mirroring how ``depths=`` can default from a select entry already present.

    ``bin_last_value`` is each bin's own *realized* last timestamp, found by
    comparison (``searchsorted``) against the real coordinate values — never by
    adding an offset to a bin's start, which means something different for a
    360-day ROMS calendar than for real dates and has to work identically for
    both (see :func:`_time_select_value`, the only reader of this).
    """
    import pandas as pd
    import xarray as xr

    from ocean_skill import operators
    from ocean_skill.sources import read

    obj = read(source)
    dim = operators.resolve_dim(obj, "time")
    if dim is None:
        raise ValueError(
            f"{source!r} has no time axis, so times= has nothing to fan over."
        )
    index = obj.indexes.get(dim)
    if not isinstance(index, pd.DatetimeIndex | xr.CFTimeIndex):
        raise ValueError(
            f"{source!r}'s time axis is not a decoded calendar axis"
            f"{f' ({type(index).__name__})' if index is not None else ''}, so "
            "times= cannot bin it into calendar periods. This is usually a "
            "climatology read with decode_times=False -- give it a fixed "
            "select= instead of times=, or aggregate={'time': 'mean'}."
        )
    if window is not None:
        # A slice-shaped window that matches nothing comes back empty, size 0,
        # below; a bare period string ("2099-01") that matches nothing raises its
        # own KeyError instead (operators.select's empty-period restatement).
        # Both mean the same thing here -- normalize to one ValueError so a
        # caller (and compare()'s skip_missing) has one exception type to catch.
        try:
            obj = operators.select(obj, {dim: window})
        except KeyError as err:
            raise ValueError(
                f"times=...'s window {window!r} matched no data in {source!r} "
                f"along its time axis: {err}"
            ) from err
        if obj.sizes.get(dim, 0) == 0:
            raise ValueError(
                f"times=...'s window {window!r} matched no data in {source!r} "
                "along its time axis."
            )
    if obj.sizes.get(dim, 0) == 0:
        # Not a windowing outcome (that case is caught above with a message
        # naming the window) -- the source's time axis is empty outright, which
        # xarray's own .resample() refuses with a much less legible error.
        raise ValueError(f"{source!r} has no data along its time axis.")
    counts = operators._bin_counts(obj[dim], freq)
    edges = counts[dim].values
    values = np.sort(np.asarray(obj[dim].values))
    bin_of = np.searchsorted(edges, values, side="right") - 1
    bins = []
    for i, (edge, n) in enumerate(zip(edges, counts.values, strict=True)):
        if n <= 0:
            continue
        bins.append((edge, values[bin_of == i].max()))
    if not bins:
        window_note = f" within {window!r}" if window is not None else ""
        raise ValueError(
            f"times={{'resample': {freq!r}, ...}} found no bins with any data "
            f"in {source!r}{window_note}."
        )
    return bins


def _time_select_value(bin_start: Any, bin_last: Any, freq: str) -> Any:
    """The per-bin select value for one resample bin.

    A whole calendar unit (year/month/day, unit multiplier 1) already names the
    whole bin as a partial-date string, sliced to that resolution — ``"2010-01"``
    is all of January, xarray's own indexing, and the same idiom
    :func:`ocean_skill.operators._bin_label` uses (numpy or cftime alike, since
    both spell their first few characters the same way). Anything coarser or
    irregular (``"3MS"``, ``"10D"``, hourly) has no single partial-date spelling,
    so the bin's own realized span is written out as the ``{"min", "max"}``
    window :func:`ocean_skill.operators.select` already reads.
    """
    import pandas as pd

    try:
        offset = pd.tseries.frequencies.to_offset(freq)
        unit = offset.rule_code.split("-")[0]
        n = offset.n
    except ValueError:
        unit, n = None, None
    width = {"YS": 4, "YE": 4, "MS": 7, "ME": 7, "D": 10}.get(unit) if n == 1 else None
    if width is not None:
        return str(bin_start)[:width]
    return {"min": str(bin_start), "max": str(bin_last)}


def _fanned_time_select(sel: dict[str, Any], value: Any) -> dict[str, Any]:
    """Write ``value`` into ``sel``'s time entry on both sides, replacing any there.

    The time analog of :func:`_fanned_select`'s strip-then-write-back: ``times=``
    is sugar that *replaces* whatever time entry ``select`` already carried (its
    only role there was narrowing which bins to fan over — see ``compare()``'s
    window handling), and writes the fanned value back under plain ``"time"``,
    the spelling every one of ``compare()``'s own selections uses.
    """
    from ocean_skill.sources import _TIME_KEYS

    def _strip(side: dict[str, Any] | None) -> dict[str, Any]:
        return {k: v for k, v in (side or {}).items() if k not in _TIME_KEYS}

    if is_pair_spec(sel):
        return {
            "test": {**_strip(sel["test"]), "time": value},
            "reference": {**_strip(sel["reference"]), "time": value},
        }
    return {**_strip(sel), "time": value}


def _merged_time_aggregate(aggregate: Any, time_entry: dict[str, Any]) -> Any:
    """Fold ``times=``'s reduction into ``aggregate``, on both sides of a pair-spec.

    Only for ``times=``'s dict form. The bin has *already* been isolated by the
    per-bin ``select`` (:func:`_fanned_time_select` writes the period value under
    ``"time"``), so all the aggregate has to do is *collapse* that one bin to the
    single map a ``Comparison`` requires (:func:`_require_reduced`). That is a
    plain reduction over the time axis — not another ``resample``: resample keeps
    the axis standing (one period start per bin), which for a single selected bin
    would now merely be squeezed away by :func:`_require_reduced` rather than
    reduced. ``'resample'`` is dropped anyway, keeping only ``'reduce'`` and any
    reduction kwargs (e.g. ``q``): saying "reduce this one bin" is the honest
    spelling of what's happening, truer than resampling a single bin and leaning
    on the squeeze to clean up after it.
    """
    reduction = {k: v for k, v in time_entry.items() if k != "resample"}
    if is_pair_spec(aggregate):
        return {
            "test": {**(aggregate.get("test") or {}), "time": reduction},
            "reference": {**(aggregate.get("reference") or {}), "time": reduction},
        }
    return {**(aggregate or {}), "time": reduction}


def _has_time_entry(spec: Any) -> bool:
    """Whether a normalized select/aggregate names a time entry, either side."""
    from ocean_skill.sources import _TIME_KEYS

    sides = (
        (spec.get("test") or {}, spec.get("reference") or {})
        if is_pair_spec(spec)
        else (spec or {},)
    )
    return any(k in side for side in sides for k in _TIME_KEYS)


def compare(
    *,
    reference,
    test,
    variables: list[Any],
    depths=None,
    times: dict[str, Any] | list | tuple | str | None = None,
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
    """Fan over reference × test × variable × depth × time into a ComparisonSet.

    ``reference`` and ``test`` each take a source name or a list; ``variables`` is a
    list of anything :mod:`ocean_skill.vocabulary` recognizes — a short vocabulary key
    (``"oxygen"``), a canonical CF standard_name, or any alias (in any case) —
    resolved to its canonical standard_name once here (warning once per variable if
    the name given wasn't already the canonical form, so it's never unclear which
    variable was actually used). Pairs where the variable is absent are skipped with a
    message rather than failing the whole run, unless ``skip_missing=False``.

    A variable may also be a *combination* — ``{"sum": ["spChl", "diatChl",
    "diazChl"], "standard_name": CHL}`` — see :mod:`ocean_skill.operators`.

    A variable may also be a **pair-spec** — ``{"test": <spec>, "reference": <spec>}``
    — when the two sides need genuinely different recipes for the same quantity, e.g.
    a model computing mixed layer depth (``{"calculate": "mld", "method":
    "density_threshold"}``) against an observational climatology that already ships it
    as a plain field (``"mld_dt_mean"``). Add ``"standard_name"`` when you know it, both
    to name the figure precisely and to settle whether the two sides describe the same
    quantity; without one, a mismatch between what the two sides actually resolve to is
    reported once as a warning rather than assumed away.
    ``aggregate`` names how dimensions collapse, and **there is no default**: a
    comparison has to be a single map, so if the sources carry a time axis you have to
    say what happens to it — ``{"time": "mean"}`` for the whole selection's mean,
    ``select={"time": <one step>}`` to compare one instant. Omitting it no longer means
    "take the mean"; it means "reduce nothing", which then fails with the axis named
    rather than averaging something you did not ask for (see :data:`NO_AGGREGATION`).

    ``{"time": {"groupby": "month", "reduce": "mean"}}`` gives a climatology and
    ``{"time": {"resample": "1MS", "reduce": "mean"}}`` consecutive months. Both keep an
    axis standing, which a comparison cannot use directly: score against it cell by
    cell with ``over="time"``, fan it into one comparison per month with ``times=``
    (below) rather than ``aggregate``, or see :func:`ocean_skill.field.field` for the
    model-only path that plots those as panels.

    ``select`` and ``aggregate`` may also each be a **pair-spec** — ``{"test": ...,
    "reference": ...}`` — for when the two lanes' axes are not asking the same
    question, not just carrying different values. A model spanning several years
    compared against a WOA monthly climatology (read with ``decode_times=False``,
    since a climatology has no calendar year) is the motivating case: the model needs
    ``{"time": {"groupby": "month", "reduce": "mean"}}`` then a single month picked
    out of the result, while the climatology just needs its lone time step meaned
    away — a shared groupby would crash on it (no calendar to group by), and a shared
    ``select={"time": "2010-01"}`` fails the same way trying to match a date string
    against its undecoded, numeric time coordinate::

        osk.compare(
            ..., variables=["nitrate"], reference="woa23_nitrate_month01",
            aggregate={"test": {"time": {"groupby": "month", "reduce": "mean"}},
                       "reference": {"time": "mean"}},
            select={"test": {"month": 1}, "reference": {}},
        )

    Note the ``select={"month": 1}`` above: it names an axis the *aggregation itself
    creates* (a groupby renames its dim to the grouping key), which is why it can't
    run before the aggregate the way ``select`` ordinarily does — it is deferred and
    applied a second time once the aggregate has run, and only warns (rather than
    silently doing nothing) if it still matches no axis afterward. ``depths=``/
    ``select={"depth": ...}`` sugar still applies to both lanes of a pair-spec select
    at once — write the vertical key into only one side, or into both agreeing on the
    same value, and let the sugar fan it out.

    ``over`` is the third answer to "what happens to the time axis", and the one for a
    reference that varies in time as well as space. ``over="time"`` keeps the axis,
    matches the two lanes along it (:func:`ocean_skill.align.match_axis` — the finer
    lane is averaged into the coarser one's bins, whichever lane that is; a finer
    reference is coarsened this way too, with a warning) and computes every metric
    *cell by cell* along it, so the figure becomes one map per metric with the overall
    value beside it. ``over="Z"`` does the same down each water column.
    ``time_method``/``tolerance``/``bin_anchor`` tune the matching, ``min_pairs`` how
    many pairs a cell needs before it is reported, and ``metrics`` which maps are
    computed (default :data:`ocean_skill.metrics.DEFAULT_MAP_METRICS`).

    ``over="time"`` is also implied, without being asked, whenever there is no map
    left to draw: a reference whose catalog ``featureType`` is a mooring/station/point
    (a place through time by nature), or — the two-*model* case — a ``select`` that
    pins both horizontal axes to one lon/lat, since a gridded run asked for one
    position is exactly as reduced as a mooring already is. Either way the comparison
    draws as lines rather than maps (the ``series`` family; see
    :attr:`Comparison.family_reason` for which of the two decided it), the reference's
    own grid decides the exact position, and the *test* is sampled there too — so the
    pair is genuinely co-located rather than each lane separately narrowing to its own
    nearest cell of the raw request. A pair-spec ``select={"test": ..., "reference":
    ...}`` naming two *different* positions on purpose (two moorings, or a
    hand-narrowed test) opts out of that and is compared as given, with a warning past
    1 km apart. The one thing this inference cannot see: a select that also pins time
    to a single instant leaves no axis for a line either, and there ``over`` stays
    unset rather than guessing.

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

    A surface of constant depth is not always the most meaningful slice through a
    stratified column — ``select={"sigma0": 26.5}`` (or a list, faceted the same way
    ``depths`` is) asks for an isopycnal instead, via
    :func:`ocean_skill.roms.to_sigma0`; ROMS sources only, and not alongside
    ``depths=`` or another vertical ``select`` key.

    ``times=`` is the month-by-month analogue of ``depths=``: instead of one map,
    fan out one comparison per time bin, each reduced on its own. A dict names how
    to *derive* the bins from the **test** source's own time axis, in the same
    vocabulary ``aggregate={"time": ...}`` uses —
    ``times={"resample": "1MS", "reduce": "mean"}`` gives one comparison per month
    actually present in the record, monthly-meaned, without reading the data
    yourself first to find out which months those are (this is the one case where
    ``compare()`` itself opens a source, cheaply — the time coordinate alone, not
    the data). A list instead names an explicit set of values —
    ``times=["2010-01", "2010-02"]`` — one comparison per entry, with no bin
    derivation and no aggregate merged in for you: pair it with your own
    ``aggregate={"time": "mean"}`` the way you would without ``times=`` at all.

    A ``select`` time entry alongside the dict form narrows *which* bins are
    enumerated — the window, not a conflict, mirroring how ``depths=`` can default
    from a select entry already present; alongside the list form it is dropped
    with a warning, since the list already says exactly which comparisons to
    build. ``times=`` and ``over="time"`` answer the same question two ways — fan
    into separate comparisons, or keep the axis and score against it — and
    ``compare()`` refuses both at once. The resulting :class:`ComparisonSet` plots
    as monthly rows (``.plot()``), animates as monthly frames (``.movie()``), or
    scores as skill through time (``.taylor()``/``.target()``); its ``.metrics()``
    table carries a ``time`` column. Against a reference whose time axis is not a
    decoded calendar (a climatology read with ``decode_times=False``) ``times=``
    has no axis to fan against — use the pair-spec ``select``/``aggregate``
    pattern above instead.

    Each pair's aligned result is cached to disk and reused on a later run with the
    same arguments (see :mod:`ocean_skill.cache`, which prints where once per
    process). ``cache=False`` bypasses it; ``refresh=True`` recomputes and overwrites
    — what to use after rerunning a model, since entries are keyed on source and
    selection rather than on file contents.
    """
    import warnings

    from ocean_skill import _stacklevel
    from ocean_skill.catalog import resolve
    from ocean_skill.vocabulary import equivalent_names, resolve_and_report

    # Validated (and, for a pair, normalized to plain per-side dicts) once up front,
    # like `variables` below -- otherwise a one-sided {"test": ...} select/aggregate
    # would surface only later, mid-fan-out, in the first Comparison() this builds.
    # From here on `select` is always a dict (never None) and, if a pair, exactly
    # {"test": ..., "reference": ...}.
    select = _normalize_pair(select, "select", normalize_side=as_select)
    aggregate = _normalize_pair(aggregate, "aggregate")

    # depths defaults to the vertical entry already in `select`, if any, so the two
    # spellings agree instead of one clobbering the other. Recorded *before*
    # defaulting -- see the calculated-variable check below, which needs to tell a
    # caller who explicitly asked for a real depth apart from one who never asked
    # at all and is only seeing the ("surface",) sentinel. For a pair-spec select,
    # both sides must agree on any vertical key present in both -- see
    # :func:`_fan_vertical_entries`.
    depths_was_explicit = depths is not None
    explicit_vertical_select = _fan_vertical_entries(select, _VERTICAL_KEYS)
    sigma_request = _fan_vertical_entries(select, _ISOPYCNAL_KEYS).get("sigma0")
    if depths_was_explicit and sigma_request is not None:
        raise ValueError(
            "compare() got both depths= and select={'sigma0': ...} -- pick one "
            "vertical request: a set of fixed depths, or a set of density "
            "surfaces, not both."
        )
    # `fan_key`/`fan_values` generalize `depths` to whichever vertical axis is
    # actually being asked for -- a set of depths (the default) or, when `select`
    # carries `sigma0`, a set of isopycnals instead. Kept as one pair of names
    # through the rest of this function rather than branching every step on which
    # one was asked for.
    if sigma_request is not None:
        fan_key = "sigma0"
        fan_values = (
            tuple(sigma_request)
            if isinstance(sigma_request, list | tuple)
            else (sigma_request,)
        )
    else:
        fan_key = "depth"
        if depths is None:
            vertical = next(iter(explicit_vertical_select.values()), SURFACE)
            depths = (vertical,)
        fan_values = depths

    # times= is the same kind of fan-out along the time axis instead of the
    # vertical one -- see the compare() docstring's own `times=` paragraph. Only a
    # dict form opens anything (the test source's time coordinate, to derive its
    # bins); a list form is pure sugar over an explicit set of values, like
    # `reference=` accepting a list. The pair-spec agreement check below only runs
    # when times= is actually asked for -- unlike the vertical fan, a pair-spec
    # select is *allowed* to carry genuinely different time values per lane
    # without times= in the picture at all (the WOA-climatology pattern this
    # module's own docstring documents), and times=None must leave that untouched.
    from ocean_skill.sources import _TIME_KEYS

    times_fan = _normalize_times(times)
    time_freq: str | None = None
    time_window: Any = None
    if times_fan is not None:
        if over is not None and over in _TIME_KEYS:
            raise ValueError(
                f"compare() got both times= and over={over!r}: times= fans the "
                "time axis into one comparison per bin, over='time' keeps it "
                "standing and scores against it, cell by cell. Pick one."
            )
        if times_fan[0] == "bins":
            if _has_time_entry(aggregate):
                raise ValueError(
                    "compare() got both times={'resample': ..., 'reduce': ...} "
                    "and an aggregate= time entry: times= already says how each "
                    "bin reduces (its own 'reduce'), so a separate "
                    "aggregate={'time': ...} would conflict with it. Drop the "
                    "aggregate time entry, or drop times= and use "
                    "aggregate={'time': {'resample': ..., 'reduce': ...}} "
                    "yourself with osk.field() to keep the axis standing as "
                    "panels instead."
                )
            # The window is actually *used* as one value here (unlike the list
            # form below, which only checks whether a select time entry exists at
            # all before dropping it), so a pair-spec select disagreeing on time
            # is refused up front, exactly like depths=/select={'sigma0': ...}.
            explicit_time_select = _fan_vertical_entries(
                select,
                _TIME_KEYS,
                sugar_label="compare()'s times=",
                example="times=",
                axis_noun="time",
            )
            time_window = next(iter(explicit_time_select.values()), None)
            time_entry = dict(times_fan[1])
            time_freq = time_entry["resample"]
            aggregate = _merged_time_aggregate(aggregate, time_entry)
        elif _has_time_entry(select):
            warnings.warn(
                f"compare() got both times={times!r} and a select= time entry; "
                "times= already says exactly which comparisons to build, so the "
                "select= entry is dropped rather than narrowing each of them "
                "further.",
                stacklevel=_stacklevel.find(),
            )

    refs = [reference] if isinstance(reference, str) else list(reference)
    tests = [test] if isinstance(test, str) else list(test)
    # Resolve each requested variable to its canonical standard_name once, up front
    # -- both so _offers() below matches against catalog metadata correctly (which
    # declares the canonical name), and so the one "name resolved to..." warning
    # fires once per variable rather than once per (ref, test, depth) combination
    # this fans out to. A pair-spec resolves each side the same way -- see
    # Comparison.__init__, which does the identical per-side resolution when the
    # spec reaches it directly rather than through this fan-out.
    def _resolve_one(v):
        if isinstance(v, dict):
            # Validated for every dict up front, not only ones that already pass
            # is_pair_spec (which requires *both* keys and so can never itself
            # observe a one-sided pair) -- otherwise a one-sided {"test": ...} in a
            # multi-variable variables=[...] surfaces only later, mid-fan-out, in
            # Comparison.__init__, after earlier variables have already aligned.
            _require_pair_spec(v)
        if is_pair_spec(v):
            return {
                **v,
                "test": (
                    resolve_and_report(v["test"], context="compare variables=")
                    if isinstance(v["test"], str)
                    else v["test"]
                ),
                "reference": (
                    resolve_and_report(v["reference"], context="compare variables=")
                    if isinstance(v["reference"], str)
                    else v["reference"]
                ),
            }
        return resolve_and_report(v, context="compare variables=") if isinstance(v, str) else v

    variables = [_resolve_one(v) for v in variables]

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
        if not options:
            # A calculator that registered no `inputs=` (ocean_skill.operators
            # .CALCULATOR_INPUTS) reports nothing to check -- there is no more
            # metadata to filter on here than there is for a source declaring no
            # `variables` at all (the branch above), and the same "let the read
            # decide" rule has to apply, or every calculator without a registered
            # `inputs=` is invisible not just to catalog *search* but to compare()
            # itself, which would silently skip the documented no-`inputs=` example
            # in register_calculator's own docstring.
            return True
        if any(all(equivalent_names(n) & declared for n in opt) for opt in options):
            return True  # positively offered

        # Not positively offered -- but "absent" and "unknowable" are different.
        # A catalog's `variables` list holds CF standard_names, so it can never
        # confirm or deny a raw model variable like `spChl`; concluding "absent"
        # from that silently drops a comparison that would have worked. Only
        # exclude when every name involved is one the catalog could have listed.
        return not all(is_known(n) for opt in options for n in opt)

    # Resolved once per test source (dict form of times= only) and shared across
    # every variable/reference/depth that source appears under -- the whole point
    # of memoizing here rather than inside a single (var, ref, tst) iteration, the
    # way `depths=`'s own fan_values needs no such cache (it costs nothing to
    # repeat, being pure sugar over a value the caller already gave).
    _time_bins_cache: dict[str, tuple[Any, ...]] = {}

    def _times_for(tst: str) -> tuple[Any, ...]:
        if times_fan is None:
            return (None,)
        if times_fan[0] == "list":
            return times_fan[1]
        if tst not in _time_bins_cache:
            bins = _time_bins(tst, time_freq, time_window)
            _time_bins_cache[tst] = tuple(
                _time_select_value(start, last, time_freq) for start, last in bins
            )
        return _time_bins_cache[tst]

    out: list[Comparison] = []
    for var in variables:
        # Pair each variable with the sources that actually carry it, rather than
        # forming a blind cross-product. Observational catalogs are usually one
        # entry per variable (WOA ships nitrate and phosphate separately), and so
        # is model output: a ROMS run writes physics and BGC to different streams,
        # on different time axes, so they cannot be one source. Filtering both
        # sides lets you pass every stream as `test=` and have each variable find
        # the one that has it, instead of failing half the cross-product.
        # A pair-spec asks a different question of each side -- e.g. mixed layer
        # depth computed on the model but already a plain field on the reference --
        # so each side is filtered against its *own* spec, not the pair as a whole.
        matching = [r for r in refs if _offers(r, variable_for(var, "reference"))]
        matching_tests = [t for t in tests if _offers(t, variable_for(var, "test"))]
        if not matching:
            print(f"  no reference offers {var!r}; skipped")
            continue
        if not matching_tests:
            print(f"  no test offers {var!r}; skipped")
            continue
        # A calculated diagnostic (mixed layer depth, ...) already collapses the
        # vertical axis, so there is nothing for the depth fan-out to iterate --
        # one comparison, no depth key at all, rather than repeating the same
        # collapsed field once per requested depth (or worse, injecting the
        # "surface" default into a select={"depth": ...} that _prepare refuses).
        calculated = _is_calculated(var)
        if calculated:
            # The bare default -- neither depths= nor a vertical select= key was
            # actually asked for, only the ("surface",) sentinel this function
            # invents when nothing else says otherwise -- is silently skipped: it
            # carries no real request to discard. Anything the caller *did* ask
            # for explicitly is a genuine contradiction with this variable and is
            # worth saying so about, rather than vanishing without a trace the way
            # it used to.
            asked = []
            if fan_key == "sigma0":
                asked.append(f"select={{'sigma0': {sigma_request!r}}}")
            else:
                if depths_was_explicit and any(
                    not is_surface_request(d) for d in depths
                ):
                    asked.append(f"depths={depths!r}")
                asked += [
                    f"select={{{k!r}: {v!r}}}"
                    for k, v in explicit_vertical_select.items()
                    if not is_surface_request(v)
                ]
            if asked:
                warnings.warn(
                    f"{var!r} is a calculated diagnostic, which already reduces the "
                    f"vertical axis itself, so {' and '.join(asked)} does not apply "
                    "to it and is being dropped for this variable rather than "
                    "honoured. Compare it on its own without a depth request, or "
                    "leave depths=/select= to the variables that do carry a "
                    "vertical axis.",
                    stacklevel=_stacklevel.find(),
                )
        these_values = (None,) if calculated else fan_values
        label_fn = _sigma_label if fan_key == "sigma0" else _depth_label
        many_vars = len(variables) > 1
        many_values = not calculated and len(fan_values) > 1
        for ref in matching:
            for tst in matching_tests:
                try:
                    these_times = _times_for(tst)
                except ValueError as exc:
                    if not skip_missing:
                        raise
                    print(f"  skipped {tst!r}: {exc}")
                    continue
                many_times = times_fan is not None and len(these_times) > 1
                for d in these_values:
                    # `depths`/`sigma0` is sugar for fanning select's vertical entry
                    # over a list -- one comparison per value. Writing it into the
                    # same key the caller may have used means a select={"Z": ...} is
                    # honoured rather than silently overridden by the default. Into
                    # both sides of a pair-spec select -- see :func:`_fanned_select`.
                    sel = _fanned_select(select, fan_key, d, calculated)
                    for t in these_times:
                        # `times=` is the same sugar along time, nested inside the
                        # depth fan -- see :func:`_fanned_time_select`. Skipped
                        # entirely when times= was never asked for, so `sel` (and
                        # every label below) is untouched and this loop iterates
                        # exactly once, matching today's behaviour byte for byte.
                        sel_t = (
                            sel if times_fan is None else _fanned_time_select(sel, t)
                        )
                        # Label only what varies across the set: repeating the
                        # variable name on every point of a single-variable
                        # fan-out just collides.
                        short = _short_variable_label(var)
                        parts = []
                        if many_vars:
                            parts.append(short)
                        if many_values:
                            parts.append(label_fn(d))
                        if many_times:
                            parts.append(_time_label(t))
                        label = " ".join(parts) if parts else short
                        c = Comparison(
                            reference=ref,
                            test=tst,
                            variable=var,
                            select=sel_t,
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
                            print(f"  skipped {label}: {exc}")
                            continue
                        out.append(c)
    return ComparisonSet(out)
