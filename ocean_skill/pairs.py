"""Answer a point comparison's variants from one saved set of matched pairs.

A comparison against a cast or a mooring pairs the model with the observations at the
observations' own times and levels, and that pairing -- the model read and sampled at
every obs point -- is the expensive part. Almost everything a caller then asks of it is
arithmetic on those pairs: a time slice keeps some of them, a depth list keeps some
levels, a depth band or a monthly mean averages them. So the *basic* comparison (no
aggregate, the obs's own levels) is computed and cached once as the **base** -- the
ordinary aligned Dataset -- and every variant is **derived** from it here instead of
re-reading the model.

"Surface" is not one of those variants for a base with levels: a cast or a repeat-visit
station has no surface measurement, so there is nothing to pair the model's top cell
with, and the caller refuses the request before it gets here
(:func:`ocean_skill.comparison._depth_request_problem`). It is meaningful for a station
that is itself at the surface (a surface buoy): its base has no depth axis, *is* the
surface comparison, and a "surface" request of it is the base unchanged.

Two functions: :func:`derivable` says whether a request can be answered this way, and
:func:`derive` does it. They are deliberately conservative -- an unrecognized key, a
depth the observations never sampled (unless ``depth_method`` says how to answer
between their levels, below), a ``"column"`` request, anything horizontal, a
``test``/``reference`` pair-spec -- is *not* derivable, and the caller falls back to
the full pipeline. Deriving a wrong answer quickly is worse than computing the right
one slowly.

``depth_method`` opens one door: a single depth between the observations' levels (an
ADCP asked at 10 m when its bins are at 8 and 12). ``"nearest"`` answers a scalar depth
with the closest level (``actual_depth`` says which, ``depth_requested`` what was
asked); ``"interp"`` (or ``"linear"``) answers a scalar or a list by interpolating the
*pairs* linearly in depth between the two levels either side, test and reference each
on their own, and does not extrapolate past the shallowest or deepest level. Neither
applies to a station with no depth axis, to a vertical mean, or to bands. With
``depth_method=None`` only the observations' own levels are derivable.

Derived answers differ from the full pipeline in one honest way: a depth band is the
plain mean of the *pairs* at the obs levels inside it, not the model's
thickness-weighted band average matched against the obs. Both lanes are masked to the
points where both are finite first, so the two means describe the same samples.

A band with ``"average": False`` (``{"min": 0, "max": 5, "average": False}``) is the
other reading of a band, and only pairs can give it: the obs levels inside
``[min, max]`` are kept standing, unaveraged, as the pairs at those levels (jointly
masked the same way) -- every level its own sample, for pooling. It takes no vertical
aggregate (the two ask for opposite things), one band at a time, and a layer holding
none of the obs levels is not derivable; ``depth_band`` records the layer. The lane
pipeline cannot answer it, so a request that is not derivable here is refused there
(:func:`ocean_skill.comparison._require_averaged_bands`).

The order is fixed -- time select, detide, depth, time aggregate, then ``difference``
recomputed from the reduced ``test`` and ``reference`` (never reduced itself: a standard
deviation of differences is not the difference of standard deviations).
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from ocean_skill import _stacklevel

__all__ = ["derivable", "derive"]

#: Spellings a ``select`` may use for time (:data:`ocean_skill.sources._TIME_KEYS`) and
#: an ``aggregate`` may (:func:`ocean_skill.operators.aggregate` resolves ``"time"`` and
#: ``"T"`` only, so a lowercase ``"t"`` there would be skipped silently -- not
#: derivable).
_SELECT_TIME_KEYS = frozenset({"time", "T", "t"})
#: ``season`` picks one group off the axis a ``{"groupby": "season"}`` aggregate
#: creates, so it is derivable only alongside one (see :func:`_check`); it never narrows
#: the raw time axis.
_SEASON_KEY = "season"
_AGGREGATE_TIME_KEYS = frozenset({"time", "T"})

#: How close two depths must be (m) to count as the same level.
_LEVEL_TOLERANCE = 1e-6

#: The finest regular cadence PL33 is meaningful at: coarser than this and a 33-hour
#: filter has too few samples per period to separate the tide from the signal.
_MAX_DETIDE_STEP_HOURS = 6.0


class _NotDerivable(Exception):
    """A request the saved pairs cannot answer; the message is the reason."""


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float | np.number) and not isinstance(value, bool)


def _vocabulary():
    """Return the comparison module's vertical-key set and depth-request predicates.

    Imported on use: :mod:`ocean_skill.comparison` imports this module, and the
    vocabulary lives there so it is spelled once.
    """
    from ocean_skill import comparison

    return comparison


def _classify_depth(depth: Any) -> tuple[str, Any]:
    """Name what ``select``'s depth asks for: ``(kind, payload)``, or raise.

    ``kind`` is ``"all"`` (nothing named: the base's levels as they are), ``"surface"``
    (``"surface"`` by name, alone or as a one-item list), ``"scalar"`` (one level),
    ``"levels"`` (a list of levels), ``"bands"`` (a list of ``(min, max)`` pairs, each
    averaged; one band is a list of one), or ``"band_levels"`` (exactly one
    ``(min, max)`` band with ``"average": False``: the observed levels inside it kept
    standing, unaveraged). Several kept-levels bands, or one mixed with averaged ones,
    are not derivable -- :func:`ocean_skill.comparison.compare` sends one band per
    comparison, and overlapping layers have no single set of levels. Anything else --
    ``"column"``, a string, a slice -- is not derivable.
    """
    c = _vocabulary()
    if c.is_column_request(depth):
        raise _NotDerivable(
            '"column" asks for the model\'s native levels, which the saved pairs '
            "(at the observations' own levels) do not hold"
        )
    if depth is None:
        return "all", None
    if c.is_surface_request(depth) or (
        isinstance(depth, list | tuple)
        and len(depth) == 1
        and isinstance(depth[0], str)
        and c.is_surface_request(depth[0])
    ):
        return "surface", None
    if c.is_depth_band(depth):
        return _classify_bands([depth])
    if _is_number(depth):
        return "scalar", float(depth)
    if isinstance(depth, list | tuple) and depth:
        if all(isinstance(d, dict) for d in depth):
            return _classify_bands(list(depth))
        if all(_is_number(d) for d in depth):
            return "levels", [float(d) for d in depth]
    raise _NotDerivable(
        f"the depth request {depth!r} is not one the saved pairs answer"
    )


def _classify_bands(specs: list[Any]) -> tuple[str, Any]:
    """Name a list of band specs: ``("bands", [...])`` or ``("band_levels", [one])``."""
    c = _vocabulary()
    bands = [_band(spec) for spec in specs]
    keeps = [c.band_keeps_levels(spec) for spec in specs]
    if not any(keeps):
        return "bands", bands
    if len(specs) != 1:
        raise _NotDerivable(
            'a depth band with "average": False keeps the observed levels, which only '
            "one band at a time can ask for (several, or one mixed with averaged "
            "bands, have no single set of levels to keep)"
        )
    return "band_levels", bands


def _band(spec: Any) -> tuple[float, float]:
    """Return ``(min, max)`` of one ``{"min", "max"[, "average"]}`` band, or raise."""
    if not (
        _vocabulary().is_depth_band(spec)
        and set(spec) <= {"min", "max", "average"}
        and _is_number(spec["min"])
        and _is_number(spec["max"])
    ):
        raise _NotDerivable(
            f"{spec!r} is not a plain {{'min', 'max'[, 'average']}} depth band"
        )
    return float(spec["min"]), float(spec["max"])


#: ``depth_method`` spellings that interpolate between levels; ``"nearest"`` snaps.
_INTERP_METHODS = frozenset({"interp", "linear"})
_DEPTH_METHODS = _INTERP_METHODS | {"nearest"}


def _check(select, aggregate, *, feature_type, obs_levels, depth_method=None) -> None:
    """Raise :class:`_NotDerivable` naming the first thing the saved pairs cannot do."""
    from ocean_skill import operators

    c = _vocabulary()
    if depth_method is not None and depth_method not in _DEPTH_METHODS:
        raise _NotDerivable(
            f"depth_method {depth_method!r} is not one of "
            f"{sorted(_DEPTH_METHODS)} or None"
        )
    if c.is_pair_spec(select) or c.is_pair_spec(aggregate):
        raise _NotDerivable(
            "a per-lane (test/reference) select or aggregate asks the two sides "
            "different questions, which one saved set of pairs cannot answer"
        )
    if feature_type not in c._SINGLE_POSITION_FEATURE_TYPES:
        raise _NotDerivable(
            f"featureType {feature_type!r} is not a point-like reference"
        )

    vertical_keys = c._VERTICAL_KEYS
    horizontal = operators._POINT_LON_KEYS | operators._POINT_LAT_KEYS
    for what, spec, time_keys in (
        ("select", select, _SELECT_TIME_KEYS | {_SEASON_KEY}),
        ("aggregate", aggregate, _AGGREGATE_TIME_KEYS),
    ):
        for key in spec:
            if key in time_keys or key in vertical_keys:
                continue
            kind = "horizontal" if key in horizontal else "unrecognized"
            raise _NotDerivable(
                f"the {kind} {what} key {key!r} cannot be applied to saved pairs"
            )
        if sum(k in vertical_keys for k in spec) > 1:
            raise _NotDerivable(f"the {what} names the vertical axis more than once")
        if feature_type == "profile" and any(k in time_keys for k in spec):
            raise _NotDerivable(
                f"a time {what} on a profile chooses or combines casts, and the "
                "saved pair is one cast"
            )

    if _SEASON_KEY in select:
        t_agg = next(
            (aggregate[k] for k in _AGGREGATE_TIME_KEYS if k in aggregate), None
        )
        if not (
            isinstance(t_agg, dict)
            and t_agg.get("groupby") == "season"
            and isinstance(select[_SEASON_KEY], str)
        ):
            raise _NotDerivable(
                "a season select needs a seasonal time aggregate "
                '({"groupby": "season", ...}) to pick the season from'
            )

    depth = next((select[k] for k in vertical_keys if k in select), None)
    kind, payload = _classify_depth(depth)
    if kind == "surface" and obs_levels is not None and len(obs_levels) > 0:
        raise _NotDerivable(
            "a base with levels has no surface measurement to answer "
            '"surface" with (only a station at the surface does)'
        )
    z_agg = next((aggregate[k] for k in vertical_keys if k in aggregate), None)
    if kind == "band_levels" and z_agg is not None:
        raise _NotDerivable(
            'a depth band with "average": False keeps the observed levels, so it '
            f"cannot also be reduced by a vertical aggregate ({z_agg!r}): drop one"
        )
    if z_agg is not None and not operators._is_plain_mean(z_agg):
        raise _NotDerivable(
            f"only a plain mean over depth can be taken of saved pairs (got {z_agg!r})"
        )
    if kind == "bands" and z_agg is None:
        raise _NotDerivable(
            "a depth band without a vertical mean asks for the model averaged over "
            "the band, which the saved pairs at the obs levels cannot give"
        )
    if (
        kind in ("bands", "band_levels")
        and obs_levels is not None
        and len(obs_levels) > 0
    ):
        # A layer none of the observations' levels falls in has no data: the nearest
        # level would silently pool an out-of-range depth into its average. Several
        # layers may have an empty one among them (derive leaves it NaN); asking for
        # nothing but empty layers is no answer at all.
        empty = c._empty_layers(payload, obs_levels)
        if len(empty) == len(payload):
            raise _NotDerivable(
                f"none of the observations' levels falls in {c._layer_label(payload)}"
            )
    if kind in ("scalar", "levels"):
        wanted = [payload] if kind == "scalar" else payload
        have = None if obs_levels is None else np.asarray(obs_levels, dtype=float)
        # off-bin answers need a depth axis to answer from: two or more levels (one is
        # a mooring's single instrument depth -- the base has no axis to pick along)
        # and no vertical mean (a mean over interpolated levels is not the full
        # pipeline's band average either)
        between = (
            depth_method is not None
            and have is not None
            and have.size >= 2
            and z_agg is None
        )
        for depth_m in wanted:
            if have is not None and np.any(np.abs(have - depth_m) <= _LEVEL_TOLERANCE):
                continue
            if between and depth_method == "nearest" and kind == "scalar":
                continue  # derive snaps to the closest level
            if between and depth_method in _INTERP_METHODS:
                lo, hi = float(have.min()), float(have.max())
                if lo <= depth_m <= hi:
                    continue
                raise _NotDerivable(
                    f"depth {depth_m:g} m is outside the observations' levels "
                    f"({lo:g}–{hi:g} m), and {depth_method} does not extrapolate"
                )
            raise _NotDerivable(
                f"depth {depth_m:g} m is not sampled by the observations"
            )


def derivable(
    select, aggregate, *, feature_type: str, obs_levels, detide, depth_method=None
) -> tuple[bool, str]:
    """Say whether a saved base of matched pairs can answer a request: ``(ok, reason)``.

    ``select``/``aggregate`` are the request's plain dicts (a ``{"test", "reference"}``
    pair-spec is not derivable). ``detide`` is the normalized form
    :class:`~ocean_skill.comparison.Comparison` stores, or ``False``/``None`` --
    accepted here without being inspected, since filtering a series is always
    derivable (whether it is regular enough is a data question :func:`derive` answers,
    with a warning). ``obs_levels`` is the observations' own depths (m, positive down)
    or ``None``/empty when it has none to name (a station at the surface); a literal
    depth is derivable only if the observations sampled it, since anywhere else the
    model would have to be read there, and ``"surface"`` only when there are no levels.

    ``depth_method`` (``None``, ``"nearest"``, ``"interp"``/``"linear"``) lets a depth
    between the levels be answered from the pairs anyway, and only when there are two
    or more levels and no vertical mean: ``"nearest"`` for one scalar depth (a list is
    a literal request for those depths and stays with the full pipeline), ``"interp"``
    for a scalar or list inside the levels' range -- outside it nothing is derivable,
    since the pairs do not extrapolate. ``None`` keeps to the observations' own levels.
    ``reason`` says why not, or for ``True`` what was checked.

    A depth band is derivable only with a plain vertical mean (the pairs' levels in it
    averaged) -- or, with ``"average": False``, only *without* any vertical aggregate
    (the levels in it kept standing); the two together, a layer holding none of the
    levels, and several kept-levels bands at once are not derivable.
    """
    select, aggregate = dict(select or {}), dict(aggregate or {})
    try:
        _check(
            select,
            aggregate,
            feature_type=feature_type,
            obs_levels=obs_levels,
            depth_method=depth_method,
        )
    except _NotDerivable as err:
        return False, str(err)
    return True, "the request is a slice or average of the saved pairs"


def _depth_dim(ds: xr.Dataset) -> str | None:
    """Return the depth dimension of a base (looked up, never assumed), or ``None``."""
    from ocean_skill import operators

    dim = operators.resolve_dim(ds["reference"], "Z")
    return dim if dim is not None and dim in ds["reference"].dims else None


def _time_dim(ds) -> str | None:
    """Return the time dimension of ``ds`` if it still stands, else ``None``."""
    from ocean_skill import operators

    dim = operators.resolve_dim(ds, "T")
    return dim if dim is not None and dim in ds.dims else None


def _levels_of(ds: xr.Dataset, zdim: str | None) -> np.ndarray | None:
    """Return the base's levels as metres positive down, whatever the source stored."""
    if zdim is None:
        return None
    from ocean_skill import depth_convention

    coord = ds[zdim]
    return depth_convention.positive_down_values(coord.values, coord.attrs)


def _keep_attrs(new: xr.DataArray, old: xr.DataArray) -> xr.DataArray:
    new.attrs = dict(old.attrs)
    return new


def _joint(test: xr.DataArray, reference: xr.DataArray):
    """Mask both lanes to the points where both are finite.

    So a mean of each describes the same samples -- the full pipeline's two
    independent means would not.
    """
    both = test.notnull() & reference.notnull()
    return _keep_attrs(test.where(both), test), _keep_attrs(
        reference.where(both), reference
    )


def _regular_step_hours(ds: xr.Dataset, tdim: str) -> float | None:
    """Return the sampling step (h) if the time axis is regular, else ``None``."""
    try:
        stamps = pd.to_datetime(np.asarray(ds[tdim].values)).values
    except (TypeError, ValueError):
        return None
    if stamps.size < 3:
        return None
    steps = np.diff(stamps).astype("timedelta64[s]").astype(float) / 3600.0
    step = float(steps[0])
    return step if step > 0 and np.allclose(steps, step, rtol=0.01) else None


def _detide(ds: xr.Dataset, detide: Any) -> xr.Dataset:
    """Low-pass the lanes that ask for it, on the paired series; warn and skip if unfit.

    PL33 reads its sampling step from the first two stamps, so it is only right on a
    regular series, and only worth running where the cadence resolves the tide.
    """
    from ocean_skill.comparison import _normalize_detide
    from ocean_skill.detide import _max_gap_of
    from ocean_skill.detide import detide as run_detide

    spec = _normalize_detide(detide)
    ds = ds.copy()
    tdim = _time_dim(ds)
    for role in ("test", "reference"):
        side = spec.get(role)
        if not side:
            continue
        if tdim is None:
            why = "the paired data has no time dimension to filter along"
        else:
            step = _regular_step_hours(ds, tdim)
            why = (
                None
                if step is not None and step <= _MAX_DETIDE_STEP_HOURS
                else "the paired series is not regularly spaced at "
                f"{_MAX_DETIDE_STEP_HOURS:g} h or finer"
            )
        if why is not None:
            warnings.warn(
                f"detide={{'T': {side['T']!r}}} was asked for the {role} lane, but "
                f"{why} -- nothing was detided there.",
                stacklevel=_stacklevel.find(),
            )
            continue
        ds[role] = run_detide(
            ds[role],
            T=side["T"],
            component="subtidal",
            max_gap=_max_gap_of(side),
        )
    return ds


def _from_positive_down(values: np.ndarray, coord: xr.DataArray) -> np.ndarray:
    """Express metres positive down in the convention ``coord`` was stored in.

    The inverse of :func:`ocean_skill.depth_convention.positive_down_values`, found
    from the coordinate itself rather than by re-deciding its convention: that mapping
    is affine (a sign and a unit factor), so two of the coordinate's own values and
    their positive-down metres fix it. A coordinate already normalized (or already
    positive-down metres) comes back unchanged.
    """
    from ocean_skill import depth_convention

    raw = np.asarray(coord.values, dtype=float)
    down = depth_convention.positive_down_values(raw, coord.attrs)
    ok = np.flatnonzero(np.isfinite(raw) & np.isfinite(down))
    first = ok[0]
    other = next((i for i in ok if raw[i] != raw[first]), None)
    if other is None:
        return np.asarray(values, dtype=float)
    slope = (down[other] - down[first]) / (raw[other] - raw[first])
    return raw[first] + (np.asarray(values, dtype=float) - down[first]) / slope


def _interpolate_depth(
    test: xr.DataArray,
    reference: xr.DataArray,
    coord: xr.DataArray,
    levels: np.ndarray,
    targets: list[float],
    zdim: str,
) -> tuple[xr.DataArray, xr.DataArray, dict[str, Any]]:
    """Interpolate both lanes linearly in depth to each target; return lanes and attrs.

    ``levels`` are the base's depths (m, positive down) in the depth axis's own index
    order, which need not be sorted, so the bracketing is found in sorted order and the
    lanes are read back by position. A target on a level takes that level as it is; one
    between two takes ``w * upper + (1 - w) * lower`` with ``w`` the weight on the
    shallower level, per lane (NaN in either bracketing level stays NaN: no gap is
    filled). Targets must lie inside the levels' range (:func:`_check` ensures it).

    A scalar request drops the depth axis and gets ``actual_depth`` (the depth asked)
    and, when interpolated, ``depth_interpolated_between``; a list keeps a depth axis
    at the targets, in the coordinate's own convention (sign and units), with the
    coordinate's attributes.
    """
    order = np.argsort(levels, kind="stable")
    sorted_levels = levels[order]

    def one(lane: xr.DataArray, d: float):
        hit = np.flatnonzero(np.abs(levels - d) <= _LEVEL_TOLERANCE)
        if hit.size:
            return _keep_attrs(lane.isel({zdim: int(hit[0])}, drop=True), lane), None
        j = int(np.searchsorted(sorted_levels, d))
        i = j - 1
        z_i, z_j = float(sorted_levels[i]), float(sorted_levels[j])
        w = (z_j - d) / (z_j - z_i)  # weight on the shallower (z_i) level
        shallow = lane.isel({zdim: int(order[i])}, drop=True)
        deep = lane.isel({zdim: int(order[j])}, drop=True)
        return _keep_attrs(w * shallow + (1.0 - w) * deep, lane), [z_i, z_j]

    parts_t, parts_r, between = [], [], []
    for d in targets:
        t_d, bracket = one(test, d)
        r_d, _ = one(reference, d)
        parts_t.append(t_d)
        parts_r.append(r_d)
        between.append(bracket)
    new_attrs: dict[str, Any] = {}
    if len(targets) == 1:
        new_attrs["actual_depth"] = float(targets[0])
        if between[0] is not None:
            new_attrs["depth_interpolated_between"] = between[0]
        return parts_t[0], parts_r[0], new_attrs
    index = pd.Index(_from_positive_down(np.asarray(targets), coord), name=zdim)
    out = []
    for parts, lane in ((parts_t, test), (parts_r, reference)):
        joined = _keep_attrs(xr.concat(parts, dim=index), lane)
        joined[zdim].attrs = dict(coord.attrs)
        out.append(joined)
    return out[0], out[1], new_attrs


def _prune_scored_over(attrs: dict[str, Any], out: xr.Dataset) -> None:
    """Drop from ``scored_over`` every axis the derivation reduced away."""
    from ocean_skill import operators

    over = attrs.get("scored_over")
    if over is None:
        return
    names = [over] if isinstance(over, str) else list(over)
    kept = [
        n
        for n in names
        if n in out["test"].dims
        or (operators.resolve_dim(out["test"], n) in out["test"].dims)
    ]
    if kept:
        attrs["scored_over"] = kept[0] if isinstance(over, str) else kept
    else:
        del attrs["scored_over"]


def derive(
    base: xr.Dataset,
    *,
    select,
    aggregate,
    detide,
    feature_type: str,
    obs_levels=None,
    subject: str = "the observations",
    depth_method=None,
) -> xr.Dataset:
    """Apply the request's slicing and averaging to the base, as today's aligned pair.

    ``base`` is the basic aligned Dataset. The result carries
    ``test``, ``reference`` and a recomputed ``difference`` (plus ``coverage`` and any
    ``test_spread``/``reference_spread`` when the base or the aggregate has them), the
    base's attrs, and ``actual_depth`` for a depth collapsed to one level or one band.
    Raises ``ValueError`` for a request :func:`derivable` refuses.

    Edge cases: a depth band holding no level has no data (a cast or a station has
    nothing in a layer none of its levels falls in, and the nearest level would pool an
    out-of-range depth into the average): one band alone is not derivable, and among
    two or more bands an empty one is NaN, with one warning naming it (``subject`` is
    what that warning calls the observations); "surface" (or no depth) of a base with
    no depth axis -- a station at the surface, whose base is the surface comparison --
    leaves it as it is; two or more bands give a depth axis of band midpoints with a
    ``<depth>_bounds`` coordinate and the ``bounds`` attribute naming it; a time select
    that leaves nothing is an error, not an empty pair.

    A ``"average": False`` band keeps the base's levels inside ``[min, max]``
    (inclusive) with no mean: the lanes are masked jointly, ``depth_band`` is the layer,
    and ``actual_depth`` names the level only when exactly one is kept.

    ``obs_levels`` names the levels the pairs sit at when the base has no depth axis to
    read them from -- a mooring's one instrument depth -- so a request naming that depth
    is recognised as the base itself rather than refused as unsampled.

    ``depth_method`` answers a single depth the observations never sampled (see
    :func:`derivable`; it is ignored for a base with no depth axis, which has nothing
    to pick or interpolate along). ``"nearest"`` takes the closest level for a scalar
    depth and records the asked depth as ``depth_requested`` beside ``actual_depth``;
    ``"interp"``/``"linear"`` interpolates test and reference separately between the
    two levels either side (NaN in either bracketing level gives NaN), for a scalar
    sets ``actual_depth`` to the depth asked and ``depth_interpolated_between`` to the
    two levels used, and for a list returns a depth axis at the targets.
    """
    from ocean_skill import align, operators

    select, aggregate = dict(select or {}), dict(aggregate or {})
    zdim = _depth_dim(base)
    ok, why = derivable(
        select,
        aggregate,
        feature_type=feature_type,
        obs_levels=_levels_of(base, zdim) if zdim is not None else obs_levels,
        detide=detide,
        depth_method=depth_method if zdim is not None else None,
    )
    if not ok:
        raise ValueError(f"cannot derive this comparison from saved pairs: {why}")
    c = _vocabulary()

    # 1. time select -- the package's own select, so every spelling it takes works here
    # A month mask ({"season": "MAM"} / {"months": [3, 4, 5]}) is split off the entry:
    # its min/max window applies here, but the mask itself waits until after the
    # detide, which needs the regular series the mask would punch gaps in.
    ds = base
    window: dict[str, Any] = {}
    month_masks: dict[str, dict[str, Any]] = {}
    for k, v in select.items():
        if k not in _SELECT_TIME_KEYS:
            continue
        key = "T" if k == "t" else k
        entry, months = operators._split_time_mask(v)
        if months is not None:
            month_masks[key] = {
                name: x for name, x in v.items() if name in operators._TIME_MASK_KEYS
            }
        if entry is not None:
            window[key] = entry
    if window:
        ds = operators.select(ds, window, subject="the saved pairs")
        tdim = _time_dim(ds)
        if tdim is not None and ds.sizes[tdim] == 0:
            raise ValueError("the time select leaves none of the saved pairs")

    # 2. detide
    from ocean_skill.comparison import _normalize_detide

    if any(_normalize_detide(detide).values()):
        ds = _detide(ds, detide)

    # 2b. month mask -- the samples of the named months, after the window and the detide
    # (the entry's own, via select, so the empty-mask error and the cftime reading are
    # the lane pipeline's)
    if month_masks:
        ds = operators.select(ds, month_masks, subject="the saved pairs")

    # 3. depth
    test, reference = ds["test"], ds["reference"]
    attrs = dict(base.attrs)
    depth = next((select[k] for k in c._VERTICAL_KEYS if k in select), None)
    kind, payload = _classify_depth(depth)
    z_mean = any(k in aggregate for k in c._VERTICAL_KEYS)
    levels = _levels_of(ds, zdim)
    extra: dict[str, Any] = {}
    band_bounds = None
    if kind == "all" and z_mean and zdim is not None:
        # a vertical mean with no depth named: every level the base has
        kind, payload = "levels", [float(v) for v in levels]
    if kind in ("all", "surface"):
        pass  # no depth operation: the base is already what was asked for
    elif zdim is not None and kind in ("scalar", "levels"):
        wanted = [payload] if kind == "scalar" else payload
        idx = [int(np.abs(levels - d).argmin()) for d in wanted]
        off_bin = [
            d
            for d, i in zip(wanted, idx, strict=True)
            if abs(levels[i] - d) > _LEVEL_TOLERANCE
        ]
        if off_bin and depth_method in _INTERP_METHODS and not z_mean:
            # derivable() only lets these through inside the levels' range
            test, reference, new_attrs = _interpolate_depth(
                test, reference, base[zdim], levels, wanted, zdim
            )
            attrs.pop("actual_depth", None)
            attrs.update(new_attrs)
        elif kind == "scalar" and not z_mean:
            test, reference = test.isel({zdim: idx[0]}), reference.isel({zdim: idx[0]})
            attrs["actual_depth"] = float(levels[idx[0]])
            if off_bin:
                attrs["depth_requested"] = float(payload)
        elif z_mean:
            test, reference = _joint(
                test.isel({zdim: idx}), reference.isel({zdim: idx})
            )
            test = _keep_attrs(test.mean(zdim, skipna=True), test)
            reference = _keep_attrs(reference.mean(zdim, skipna=True), reference)
            attrs["actual_depth"] = float(np.mean(levels[idx]))
        else:
            test, reference = test.isel({zdim: idx}), reference.isel({zdim: idx})
            attrs.pop("actual_depth", None)
            if len(idx) == 1:
                attrs["actual_depth"] = float(levels[idx[0]])
    elif zdim is not None and kind == "band_levels":
        # the observed levels inside the layer, kept standing: no mean, so the layer
        # is just a depth subset -- jointly masked so both lanes hold the same samples
        ((lo, hi),) = payload
        inside = np.flatnonzero((levels >= lo) & (levels <= hi))
        if inside.size == 0:
            raise ValueError(
                f"{subject} has no level in {c._layer_label(payload)} to keep"
            )
        test, reference = _joint(
            test.isel({zdim: inside}), reference.isel({zdim: inside})
        )
        attrs.pop("actual_depth", None)
        if inside.size == 1:
            attrs["actual_depth"] = float(levels[inside[0]])
        attrs["depth_band"] = [lo, hi]
    elif zdim is not None and kind == "bands":
        test, reference = _joint(test, reference)
        parts_t, parts_r, used, empty = [], [], [], []
        for lo, hi in payload:
            inside = np.flatnonzero((levels >= lo) & (levels <= hi))
            used.append(inside)
            if inside.size == 0:
                # no level in the layer: no data there -- NaN for both lanes, on the
                # axis the layer was asked for (derivable ruled out all of them)
                empty.append((lo, hi))
                parts_t.append(xr.full_like(test.isel({zdim: 0}, drop=True), np.nan))
                parts_r.append(
                    xr.full_like(reference.isel({zdim: 0}, drop=True), np.nan)
                )
                continue
            parts_t.append(test.isel({zdim: inside}).mean(zdim, skipna=True))
            parts_r.append(reference.isel({zdim: inside}).mean(zdim, skipna=True))
        if empty:
            warnings.warn(
                f"{subject} has no data in {c._layer_label(empty)} -- none of its "
                f"levels ({', '.join(f'{v:g}' for v in np.unique(levels))} m) falls "
                "in it, so it is left empty (NaN), not filled from the nearest level.",
                stacklevel=_stacklevel.find(),
            )
        if len(payload) == 1:
            test = _keep_attrs(parts_t[0], test)
            reference = _keep_attrs(parts_r[0], reference)
            attrs["actual_depth"] = float(np.mean(levels[used[0]]))
            attrs["depth_band"] = list(payload[0])
        else:
            mids = pd.Index([(lo + hi) / 2 for lo, hi in payload], name=zdim)
            test = _keep_attrs(xr.concat(parts_t, dim=mids), test)
            reference = _keep_attrs(xr.concat(parts_r, dim=mids), reference)
            bounds = f"{zdim}_bounds"
            band_bounds = (bounds, np.array(payload, dtype=float))
            for da in (test, reference):
                da[zdim].attrs = {"units": "m", "positive": "down", "bounds": bounds}
            attrs.pop("actual_depth", None)
    if "coverage" in ds:
        extra["coverage"] = ds["coverage"]

    # 4. time aggregate -- after masking jointly, so both means cover the same samples
    time_spec = {k: v for k, v in aggregate.items() if k in _AGGREGATE_TIME_KEYS}
    reduced_dims: set[str] = set()
    if time_spec and _time_dim(test) is not None:
        reduced_dims.add(_time_dim(test))
        test, reference = _joint(test, reference)
        test = operators.aggregate(test, time_spec)
        reference = operators.aggregate(reference, time_spec)
        if _SEASON_KEY in select:
            # the season select narrows the axis the aggregate just created
            pick = {_SEASON_KEY: select[_SEASON_KEY]}
            test = operators.select(test, pick, subject="the saved pairs")
            reference = operators.select(reference, pick, subject="the saved pairs")

    # 5. difference (and coverage) from the reduced lanes
    test, reference, attach_spread = align._split_spread(test, reference)
    out = xr.Dataset(
        {"test": test, "reference": reference, "difference": test - reference}
    )
    out["difference"].attrs = dict(base["difference"].attrs) or {
        "long_name": "test − reference",
        "units": reference.attrs.get("units", ""),
    }
    for name, cov in extra.items():
        # a coverage over an axis that was just reduced would need re-deriving from
        # the regrid, which point pairs never had; keep it only where it still means
        # what it said
        if set(cov.dims) <= set(out["test"].dims) and not (
            set(cov.dims) & reduced_dims
        ):
            out[name] = cov
    out = attach_spread(out)
    if band_bounds is not None:
        name, values = band_bounds
        out = out.assign_coords({name: ((zdim, "bound"), values)})
    out.attrs = attrs
    _prune_scored_over(out.attrs, out)
    return out
