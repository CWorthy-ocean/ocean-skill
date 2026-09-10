"""ROMS model adapter (reader-independent).

The catalog says *how to read* a ROMS file (driver/args → intake); this module turns
that raw output into a CF-standardized dataset ocean-skill can compare: attach the grid
(lon/lat/h/mask), decode ``ocean_time``, rename variables to CF standard_names, mask
land, and reconstruct depth. The s-coordinate → z transform is xgcm-based (Vtransform 2,
using ``Cs_r``/``sigma_r`` from the grid); it stays lazy (dask) — no unchunk needed.
Lateral regridding lives in :mod:`ocean_skill.align` (xesmf), not here.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import xarray as xr

__all__ = [
    "AREA_COORD",
    "GEOGRAPHIC_VELOCITY_NAMES",
    "GRID_RELATIVE_VELOCITY_NAMES",
    "WEIGHT_COORD",
    "add_depth_coord",
    "add_interface_coord",
    "depth_average",
    "depth_band",
    "derived_geographic_velocities",
    "nearest_depth_levels",
    "standardize",
    "surface",
    "to_depth",
    "to_sigma0",
]

#: ROMS' own grid-relative velocity standard_names (build.py's ROMS_STANDARD_NAMES
#: maps `u`/`v` here) -- staggered, not directly comparable to an in-situ instrument
#: on a rotated grid. See :data:`GEOGRAPHIC_VELOCITY_NAMES` and
#: :func:`derived_geographic_velocities`.
GRID_RELATIVE_VELOCITY_NAMES = ("sea_water_x_velocity", "sea_water_y_velocity")

#: The true geographic velocity :func:`_add_geographic_velocity` derives from
#: :data:`GRID_RELATIVE_VELOCITY_NAMES` + the grid ``angle``, whenever a ROMS source
#: has both. This pair (and the fact that *both* grid-relative names are the
#: trigger) is also what :func:`derived_geographic_velocities` reports to callers
#: outside this module -- kept as one set of names so ocean_skill.build (catalog-time
#: advertisement) and ocean_skill.comparison (runtime catalog pre-filter) cannot
#: silently drift from what this module actually produces.
GEOGRAPHIC_VELOCITY_NAMES = (
    "eastward_sea_water_velocity",
    "northward_sea_water_velocity",
)


def derived_geographic_velocities(present) -> list[str]:
    """Return the geographic velocity names :func:`standardize` would derive.

    ``present`` is any iterable of standard_names a ROMS source declares or carries
    (a catalog entry's ``variables`` list, or a Dataset's own data_var names).
    Returns :data:`GEOGRAPHIC_VELOCITY_NAMES` as a list when ``present`` has BOTH of
    :data:`GRID_RELATIVE_VELOCITY_NAMES` -- exactly the condition
    :func:`_add_geographic_velocity` checks before rotating -- else an empty list.

    The single source of truth for "what standardize adds" so a catalog-time
    advertisement (:func:`ocean_skill.build._probe`) and a runtime catalog
    pre-filter (:func:`ocean_skill.comparison.compare`'s ``_offers``,
    :func:`ocean_skill.catalog.find`/``search``) agree with this module and with
    each other, rather than each hardcoding its own copy of these four names.
    """
    if set(GRID_RELATIVE_VELOCITY_NAMES) <= set(present):
        return list(GEOGRAPHIC_VELOCITY_NAMES)
    return []

#: Coordinate name carrying per-cell horizontal area, so a spatial mean can honour
#: it — mirrors :data:`WEIGHT_COORD`'s "weights ride on the data" pattern:
#: :func:`ocean_skill.operators.aggregate` needs no special case for a box mean,
#: ``{"lat": "mean", "lon": "mean"}`` finds it and uses it, exactly as ``{"Z":
#: "mean"}`` needs nothing special for depth. Attached in :func:`standardize` when
#: the grid carries ``pm``/``pn`` (ROMS's inverse grid spacings, 1/m); absent from
#: a grid that does not, in which case a spatial mean falls back to cos(latitude).
AREA_COORD = "cell_area"


def _open_grid(meta: dict[str, Any]) -> xr.Dataset:
    grid_path = meta.get("grid")
    if not grid_path or str(grid_path).startswith("TODO"):
        raise ValueError(
            "ROMS entry needs a real 'grid' file in metadata (lon/lat/h + s-coord); "
            f"got {grid_path!r}."
        )
    return xr.open_dataset(grid_path)


def _decode_time(ds: xr.Dataset, meta: dict[str, Any]) -> xr.Dataset:
    """Decode the ROMS time coordinate (non-CF 'seconds since <reference_date>')."""
    tcoord = meta.get("time_coord", "ocean_time")
    tdim = meta.get("time_dim", "time")
    if tcoord not in ds.variables:
        return ds
    ref = np.datetime64(meta.get("reference_date", "2000-01-01"))
    units = meta.get("time_units", "seconds")
    if units not in ("seconds", "second", "s"):
        raise ValueError(f"Unsupported time_units {units!r} (expected seconds).")
    times = ref + ds[tcoord].values.astype("timedelta64[s]")
    return ds.assign_coords(time=(tdim, times))


def _average_to_rho(da: xr.DataArray, stagger_dim: str, rho_dim: str) -> xr.Variable:
    """Average one ROMS velocity component from its staggered dim onto rho points.

    ``da`` sits on ``stagger_dim`` (``xi_u`` or ``eta_v``): each of its N-1 values
    lies between two neighbouring rho points, so the result on ``rho_dim``
    (``xi_rho``/``eta_rho``) has N points -- an interior one is the plain 2-point
    average of its two bracketing staggered values, and the two edges take the
    nearest staggered value outright. That edge rule is exactly xgcm's own
    ``boundary="extend"`` convention for an outer-to-center interpolation
    (duplicating the edge value before averaging reduces to taking it directly);
    reproduced here in plain xarray/dask rather than by standing up an xgcm X/Y
    ``Grid`` for it, since roms-tools output does not reliably carry the
    ``axis``/``c_grid_axis_shift`` metadata that grid needs. Lazy: ``.isel``,
    arithmetic and ``concat`` only, so a dask-backed ``da`` stays dask-backed.

    Returns a bare :class:`~xarray.Variable` (not a `DataArray`) on ``rho_dim`` --
    the caller assigns it into a Dataset that already carries the rho-point
    coordinates (``lon``/``lat``/``mask_rho``/``angle``/...), which pick it up by
    dimension name; building a `DataArray` here would only have to carry no
    coordinates of its own, since the staggered input's own index (if any) does
    not describe rho positions.
    """
    var = da.variable
    # Collapse the staggered dim to a single chunk before the misaligned
    # ``[:-1]``/``[1:]`` slices below, then merge the ``[edge, interior, edge]``
    # concat back into one chunk after. Without this, ``0.5*(left+right)`` adds two
    # slices whose chunk boundaries are offset by one, so dask unifies them into a
    # (1, N, 1, N, 1, ...) chunking -- a size-1 chunk at every boundary -- and the
    # concat's size-1 edges compound it; the later rotation multiply
    # (:func:`_add_geographic_velocity`) then has to unify *that* against the grid
    # ``angle``, building a task graph that scales into the millions over a long,
    # one-step-per-chunk history file (thousands of hourly steps x s_rho x the
    # fragments) -- the confirmed cause of a multi-minute hang, mostly in memory
    # pressure, before either lane is even cropped. Only one spatial dim is made a
    # single chunk, so the transient stays modest; ``.rechunk`` never moves values,
    # so the averaged result is byte-identical.
    if var.chunks is not None:
        axis = var.dims.index(stagger_dim)
        if len(var.chunks[axis]) > 1:
            var = xr.Variable(var.dims, var.data.rechunk({axis: -1}))
    left = var.isel({stagger_dim: slice(None, -1)})
    right = var.isel({stagger_dim: slice(1, None)})
    interior = 0.5 * (left + right)
    edge_lo = var.isel({stagger_dim: slice(0, 1)})
    edge_hi = var.isel({stagger_dim: slice(-1, None)})
    full = xr.Variable.concat([edge_lo, interior, edge_hi], dim=stagger_dim)
    if full.chunks is not None:
        axis = full.dims.index(stagger_dim)
        full = xr.Variable(full.dims, full.data.rechunk({axis: -1}))
    rho_dims = tuple(rho_dim if d == stagger_dim else d for d in full.dims)
    return xr.Variable(rho_dims, full.data)


def _add_geographic_velocity(ds: xr.Dataset) -> xr.Dataset:
    """Attach lazy true eastward/northward velocity, derived from staggered u/v.

    ROMS' own ``u``/``v`` (renamed to ``sea_water_x_velocity``/``sea_water_y_velocity``
    by the caller) are grid-relative and staggered -- on a rotated grid neither is
    the same quantity as geographic east/north, and neither reaches rho points (see
    :func:`to_depth`'s deferral of them). This averages each to rho points with
    :func:`_average_to_rho` and rotates by the grid's own ``angle`` (radians,
    ROMS/roms-tools convention: the angle from true east to the grid's local xi
    direction, CCW positive)::

        east  = u_rho*cos(angle) - v_rho*sin(angle)
        north = u_rho*sin(angle) + v_rho*cos(angle)

    so the result is directly comparable to an in-situ instrument's own eastward/
    northward reading (an ADCP mooring, say) -- see the "east_velocity"/
    "north_velocity" vocabulary entries. Lazy throughout (plain ``xr.Variable``
    arithmetic on the dask-backed components), and lands on
    ``sea_water_x_velocity``'s own non-staggered dims (typically
    ``(time, s_rho, eta_rho, xi_rho)``), which is exactly what :func:`to_depth`'s
    rho-dims guard needs to pick the result up -- unlike the staggered components
    themselves, which it still skips.

    A no-op (returning ``ds`` unchanged) when ``u``, ``v`` or ``angle`` is missing;
    warns first if the grid has velocity but no ``angle`` to rotate it by, since that
    silently leaves only the grid-relative components for a caller who asked for
    geographic east/north.
    """
    x_name, y_name = GRID_RELATIVE_VELOCITY_NAMES
    east_name, north_name = GEOGRAPHIC_VELOCITY_NAMES
    u = ds.get(x_name)
    v = ds.get(y_name)
    have_angle = "angle" in ds.coords
    if u is None or v is None or not have_angle:
        if (u is not None or v is not None) and not have_angle:
            warnings.warn(
                f"ROMS grid-relative velocity ({x_name}/{y_name}) is present but "
                "the grid `angle` is not, so true geographic eastward/northward "
                "velocity cannot be derived (the rotation needs it) -- only the "
                "grid-relative components are available.",
                stacklevel=2,
            )
        return ds

    # ROMS/roms-tools' own fixed staggered-dim names (as used throughout this module,
    # e.g. to_depth's deferral comment) -- NOT "whichever dim isn't a rho dim", which
    # would wrongly pick "s_rho"/"time" themselves when u/v still carry those.
    u_rho = _average_to_rho(u, "xi_u", "xi_rho") if "xi_u" in u.dims else u.variable
    v_rho = _average_to_rho(v, "eta_v", "eta_rho") if "eta_v" in v.dims else v.variable
    return _rotate_and_assign(ds, u_rho, v_rho, u.attrs, v.attrs)


def _rotate_and_assign(
    ds: xr.Dataset,
    u_rho: xr.Variable,
    v_rho: xr.Variable,
    u_attrs: dict[str, Any],
    v_attrs: dict[str, Any],
) -> xr.Dataset:
    """Rotate rho-point ``u_rho``/``v_rho`` by ``ds["angle"]`` and assign east/north.

    The shared tail of :func:`_add_geographic_velocity` (full-domain) and
    :func:`add_geographic_velocity_windowed` (a point-cropped column) -- both
    average their own staggered input to rho points first, by different means, then
    reach here with the same rotation/transpose/assign. See
    :func:`_add_geographic_velocity` for the rotation convention.
    """
    east_name, north_name = GEOGRAPHIC_VELOCITY_NAMES
    angle = ds["angle"].variable
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    east = u_rho * cos_a - v_rho * sin_a
    north = u_rho * sin_a + v_rho * cos_a

    # Canonical dim order, matching add_depth_coord's own transpose below -- u_rho/
    # v_rho's arithmetic can otherwise reorder dims depending on which operand's
    # layout xarray's broadcasting happens to keep.
    order = tuple(d for d in ("time", "s_rho", "eta_rho", "xi_rho") if d in east.dims)
    east, north = east.transpose(*order), north.transpose(*order)

    return ds.assign(
        {
            east_name: (
                east.dims,
                east.data,
                {
                    "standard_name": east_name,
                    "long_name": "eastward (true geographic) sea water velocity",
                    "units": u_attrs.get("units", "m s-1"),
                },
            ),
            north_name: (
                north.dims,
                north.data,
                {
                    "standard_name": north_name,
                    "long_name": "northward (true geographic) sea water velocity",
                    "units": v_attrs.get("units", "m s-1"),
                },
            ),
        }
    )


def add_geographic_velocity_windowed(ds: xr.Dataset, meta: dict[str, Any]) -> xr.Dataset:
    """Re-derive east/north on a point-cropped window, byte-identical to a full-domain
    derive-then-crop.

    A full-domain :func:`_add_geographic_velocity` builds a task graph over the whole
    grid and every time step; for a ROMS history file chunked one step per chunk, that
    graph scales into the millions and stays that large even after cropping to a
    single water column (culling it does not remove the per-step task overhead the
    concat/rechunk inside :func:`_average_to_rho` creates) -- the confirmed cause of a
    multi-minute hang on a point/station comparison. This instead re-derives on a
    lane whose staggered ``sea_water_x_velocity``/``sea_water_y_velocity`` were
    cropped to the point window *plus a one-cell halo* by
    :func:`ocean_skill.align._point_window`, which records how many extra halo rho
    points that produced, per rho dim, in ``ds.attrs["_roms_stagger_trim"]`` --
    ``{dim: (low, high)}``, each ``1`` when that side of the window is interior (an
    extra halo rho point was produced there) or ``0`` when it already sits at the
    domain edge (the ordinary edge rule already lands on the right value, nothing to
    trim). Popped and consumed here; a caller that never set it (no halo crop
    happened) gets an untrimmed, already-exact result.

    Requires ``ds`` to already have any pre-derived ``east/north`` dropped (a stale
    full-width pair here would silently prefer the wrong one) -- see the point-lane
    gate in :func:`ocean_skill.comparison.prepare_source`. A no-op, mirroring
    :func:`_add_geographic_velocity`, when ``u``, ``v`` or ``angle`` is missing.
    Re-applies the identical ``mask_rho == 1`` land mask :func:`standardize` uses,
    since this runs after that mask already ran once (on the stale, now-dropped
    pair).
    """
    trim = dict(ds.attrs.pop("_roms_stagger_trim", {}) or {})
    x_name, y_name = GRID_RELATIVE_VELOCITY_NAMES
    u = ds.get(x_name)
    v = ds.get(y_name)
    if u is None or v is None or "angle" not in ds.coords:
        return ds

    def _trimmed(component, stagger_dim, rho_dim):
        rho = (
            _average_to_rho(component, stagger_dim, rho_dim)
            if stagger_dim in component.dims
            else component.variable
        )
        low, high = trim.get(rho_dim, (0, 0))
        if not low and not high:
            return rho
        axis = rho.dims.index(rho_dim)
        n = rho.shape[axis]
        return rho.isel({rho_dim: slice(low, n - high)})

    u_rho = _trimmed(u, "xi_u", "xi_rho")
    v_rho = _trimmed(v, "eta_v", "eta_rho")
    ds = _rotate_and_assign(ds, u_rho, v_rho, u.attrs, v.attrs)

    # Same rule as standardize's own land-mask loop, scoped to the pair just
    # derived -- everything else on this lane was already masked once, before the
    # stale full-width pair this replaces was dropped.
    if "mask_rho" in ds.variables:
        mask = ds["mask_rho"] == 1
        for name in GEOGRAPHIC_VELOCITY_NAMES:
            if {"eta_rho", "xi_rho"} <= set(ds[name].dims):
                ds[name] = ds[name].where(mask)
    return ds


def standardize(ds: xr.Dataset, meta: dict[str, Any]) -> xr.Dataset:
    """Return a CF-standardized ROMS Dataset (grid attached, renamed, masked, depth).

    Parameters
    ----------
    ds
        Raw ROMS output opened per the catalog entry (rho-point fields).
    meta
        The catalog entry ``metadata`` (``grid``, ``vertical``, ``standard_names``,
        ``reference_date``/``time_*``).
    """
    # the grid may be a separate file, or already merged into the output
    # (self_contained_grid, e.g. a combined ROMS file)
    grid = ds if meta.get("self_contained_grid") else _open_grid(meta)
    ds = _decode_time(ds, meta)

    # attach horizontal coords + grid fields AS COORDS (so they are not treated as
    # comparable data variables downstream); moving them to coords also drops h/mask/
    # Cs_r/sigma_r out of data_vars for a self-contained file.
    ds = ds.assign_coords(lon=grid["lon_rho"], lat=grid["lat_rho"])
    for name in ("h", "Cs_r", "sigma_r", "mask_rho", "angle"):
        if name in grid.variables:
            ds = ds.assign_coords({name: grid[name]})

    # true cell area, for an area-weighted spatial mean (ocean_skill.operators
    # aggregate); absent from a grid that ships no pm/pn, in which case a spatial
    # mean falls back to cos(latitude) -- see AREA_COORD.
    if "pm" in grid.variables and "pn" in grid.variables:
        ds = ds.assign_coords(
            {
                AREA_COORD: (
                    1.0 / (grid["pm"] * grid["pn"])
                ).assign_attrs(units="m2", long_name="grid cell area (1/(pm*pn))")
            }
        )

    # rename model variable names -> CF standard_names
    rename = {
        k: v for k, v in (meta.get("standard_names") or {}).items() if k in ds.variables
    }
    ds = ds.rename(rename)

    # derive TRUE geographic east/north velocity from the staggered grid-relative
    # components + the grid angle, before the mask loop below so the new rho-dim
    # vars get land-masked with everything else. Lazy; a no-op when the source has
    # no velocity (or no angle to rotate it by) -- see _add_geographic_velocity.
    ds = _add_geographic_velocity(ds)

    # mask land on rho-point data variables (mask_rho: 1 ocean, 0 land)
    if "mask_rho" in ds.variables:
        mask = ds["mask_rho"] == 1
        for var in list(ds.data_vars):
            da = ds[var]
            if {"eta_rho", "xi_rho"} <= set(da.dims) and var != "mask_rho":
                ds[var] = da.where(mask)

    ds = add_depth_coord(ds, meta)
    ds.attrs["featureType"] = meta.get("featureType", "grid")
    ds.attrs["ocean_skill_model"] = meta.get("model", "roms")
    return ds


def add_depth_coord(
    ds: xr.Dataset, meta: dict[str, Any], *, zero_zeta: bool = False
) -> xr.Dataset:
    """Attach the (lazy) ROMS z_rho depth coordinate via Vtransform 2.

    ``z_rho = zeta + (zeta + h) * (hc*sigma_r + h*Cs_r) / (hc + h)`` (metres, negative
    down). Requires ``h``/``Cs_r``/``sigma_r`` (from the grid) and ``zeta``.

    ``zero_zeta=True`` ignores any ``zeta``/``sea_surface_height_above_geoid`` on
    ``ds`` and uses ``zeta = 0`` instead, the same fallback already used when neither
    is present -- collapsing the formula to ``z_rho = h * (hc*sigma_r + h*Cs_r) /
    (hc + h)``, a pure function of the (always-finite) bathymetry. This is for a
    section's depth *mesh*, not the water column itself: ``zeta`` is a land-masked
    rho-point field (see :func:`standardize`), so the ordinary path leaves ``z_rho``
    NaN over land -- fine for interpolating data onto depths, fatal for a plot's
    depth-mesh coordinate (pcolormesh refuses to draw with a non-finite coordinate).
    ``zeta`` is metres against an ``h`` of hundreds to thousands, so dropping it here
    costs nothing a vertical section could show.
    """
    vert = meta.get("vertical", {})
    hc = float(vert.get("hc"))
    if zero_zeta:
        zeta = xr.zeros_like(ds["h"])
    elif "zeta" in ds.variables:
        zeta = ds["zeta"]
    elif "sea_surface_height_above_geoid" in ds.variables:
        zeta = ds["sea_surface_height_above_geoid"]
    else:  # no free-surface field: use zeta = 0
        zeta = xr.zeros_like(ds["h"])
    s = (hc * ds["sigma_r"] + ds["h"] * ds["Cs_r"]) / (hc + ds["h"])
    z_rho = zeta + (zeta + ds["h"]) * s
    # The preferred order for the full field; `...` carries forward anything not
    # named here (`along`, for a vertical section already sliced to one grid
    # column -- see ocean_skill.transect.grid_slice) in whatever order it already
    # has, rather than requiring every possible dim be spelled out.
    dims = ("time", "s_rho", "eta_rho", "xi_rho")
    z_rho = z_rho.transpose(*[d for d in dims if d in z_rho.dims], ...)
    return ds.assign_coords(z_rho=z_rho)


def add_interface_coord(
    ds: xr.Dataset, meta: dict[str, Any], *, zero_zeta: bool = False
) -> xr.Dataset:
    """Attach the (lazy) ``z_w`` cell-*interface* depths, the companion to ``z_rho``.

    Same Vtransform-2 formula as :func:`add_depth_coord`, evaluated on ``sigma_w``/
    ``Cs_w`` (N+1 interfaces) instead of ``sigma_r``/``Cs_r`` (N centres).

    The two serve different operations and neither replaces the other. Data lives at
    *centres*, so interpolating to a depth (:func:`to_depth`) must use ``z_rho``.
    Cell *thicknesses* only exist between interfaces, so a depth-band average
    (:func:`depth_average`) must use ``z_w``. Interfaces also start exactly at the
    free surface, which is why a band average has no NaN problem where interpolation
    does: the shallowest ``z_rho`` can be 7 m down in deep water, but the shallowest
    ``z_w`` is always 0.

    ``zero_zeta`` -- see :func:`add_depth_coord`, the same zeta-free mesh for a
    section built on ``s_w`` instead of ``s_rho``.
    """
    vert = meta.get("vertical", {})
    hc = float(vert.get("hc"))
    if zero_zeta:
        zeta = xr.zeros_like(ds["h"])
    elif "zeta" in ds.variables:
        zeta = ds["zeta"]
    elif "sea_surface_height_above_geoid" in ds.variables:
        zeta = ds["sea_surface_height_above_geoid"]
    else:
        zeta = xr.zeros_like(ds["h"])
    s = (hc * ds["sigma_w"] + ds["h"] * ds["Cs_w"]) / (hc + ds["h"])
    z_w = zeta + (zeta + ds["h"]) * s
    dims = ("time", "s_w", "eta_rho", "xi_rho")  # see add_depth_coord's note on `...`
    z_w = z_w.transpose(*[d for d in dims if d in z_w.dims], ...)
    return ds.assign_coords(z_w=z_w)


#: Coordinate name carrying per-cell weights, so a later reduction can honour them.
#: Riding on the data means :func:`ocean_skill.operators.aggregate` needs no special
#: case for depth -- ``{"Z": "mean"}`` finds the weights and uses them, exactly as
#: ``{"T": "mean"}`` needs nothing special for time.
WEIGHT_COORD = "dz"


def depth_band(
    ds: xr.Dataset, meta: dict[str, Any], low: float, high: float
) -> xr.Dataset:
    """Return the cells overlapping ``low``-``high`` m, with overlap as weights.

    A *selection*, not a reduction: the vertical dimension survives, narrowed to the
    cells the band touches, with :data:`WEIGHT_COORD` giving how much of each lies
    inside it (partial at the boundary). Collapsing it is
    :func:`ocean_skill.operators.aggregate`'s job, which keeps "select narrows,
    aggregate collapses" true for depth exactly as it is for time — and makes
    ``{"Z": "max"}`` or ``{"Z": "std"}`` over a band meaningful rather than
    impossible.
    """
    if "z_w" not in ds.coords:
        ds = add_interface_coord(ds, meta)
    s_dim = meta.get("vertical", {}).get("s_dim", "s_rho")
    w_dim = next((d for d in ds["z_w"].dims if d not in ds[s_dim].dims), "s_w")

    # z_w is negative-down; work in positive-down metres to match the request.
    depth_w = -ds["z_w"]
    shallower = depth_w.isel({w_dim: slice(1, None)}).rename({w_dim: s_dim})
    deeper = depth_w.isel({w_dim: slice(None, -1)}).rename({w_dim: s_dim})
    overlap = (deeper.clip(max=float(high)) - shallower.clip(min=float(low))).clip(
        min=0.0
    )

    # Keep only cells the band actually touches, so a reduction that ignores weights
    # (max, std) still operates on the right set rather than the whole column.
    touched = (overlap > 0).any([d for d in overlap.dims if d != s_dim])
    out = ds.isel({s_dim: touched.values.nonzero()[0]})
    out = out.assign_coords(
        {WEIGHT_COORD: overlap.isel({s_dim: touched.values.nonzero()[0]})}
    )
    out.attrs = dict(ds.attrs)
    out.attrs["depth_band"] = f"{low}-{high} m"
    return out


def depth_average(
    ds: xr.Dataset, meta: dict[str, Any], low: float, high: float
) -> xr.Dataset:
    """Thickness-weighted average of ``ds`` over the depth band ``low``-``high`` (m).

    Each native cell contributes its own value weighted by how much of it lies inside
    the band, with partial weight for the cell the boundary cuts through::

        mean = sum(value_k * overlap_k) / sum(overlap_k)

    No interpolation, deliberately. A ROMS ``s_rho`` value *is* the cell's value, so
    ``value x overlap`` is exact under that reading; reconstructing a profile between
    centres instead would assume sub-cell structure the model does not have. Measured
    against a smooth analytic profile this matches or beats interpolation everywhere
    (-0.2% vs -0.2% on the shelf, -1.8% vs -3.3% on the slope, -9.0% vs -10.0% in
    deep water), and it needs no arbitrary target spacing.

    That deep-water residual is a resolution limit, not a method error: the model
    describes the top 17 m with one number, and no averaging scheme recovers what was
    never resolved. It is still the right comparison for satellite chlorophyll,
    because the band is the *same depth everywhere* — unlike :func:`surface`, whose
    effective depth ranges from 0.2 m on the shelf to 17 m offshore on this grid.
    """
    band = depth_band(ds, meta, low, high)
    s_dim = meta.get("vertical", {}).get("s_dim", "s_rho")
    weights = band[WEIGHT_COORD]
    total = weights.sum(s_dim)

    out = {}
    for name, var in band.data_vars.items():
        if s_dim not in var.dims:
            continue
        averaged = (var * weights).sum(s_dim) / total.where(total > 0)
        averaged.attrs = dict(var.attrs)
        averaged.attrs["depth_average"] = f"thickness-weighted mean over {low}-{high} m"
        out[name] = averaged
    result = xr.Dataset(out, attrs=dict(ds.attrs))
    for coord in ("lon", "lat", "lon_rho", "lat_rho", "mask_rho", "h", AREA_COORD):
        if coord in ds.coords and coord not in result.coords:
            result = result.assign_coords({coord: ds[coord]})
    return result


def surface(ds: xr.Dataset, meta: dict[str, Any] | None = None) -> xr.Dataset:
    """Return the surface field: the topmost s-coordinate level (``s_rho=-1``).

    This is the right operation for surface comparisons — unlike interpolating to a
    fixed shallow depth, which yields NaN wherever the top model cell-center is deeper
    than the target (common over deep water). Drops the vertical dimension.
    """
    s_dim = (meta or {}).get("vertical", {}).get("s_dim", "s_rho")
    top = ds.isel({s_dim: -1}) if s_dim in ds.dims else ds
    return top.drop_vars([s_dim, "z_rho"], errors="ignore")


def _contiguous_column(da: xr.DataArray, s_dim: str) -> xr.DataArray:
    """Return ``da`` with its vertical axis in a single dask chunk.

    Interpolating to a depth reads the whole water column at once — xgcm passes the
    vertical to ``apply_ufunc`` as a *core* dimension — so a source chunked along
    ``s_rho`` fails outright with "consists of multiple chunks, but is also a core
    dimension". Whether that happens is a property of how the store was written, which
    is why it can lie unnoticed until a particular dataset is used.

    Rechunking here rather than passing xgcm ``allow_rechunk=True``: the two do the
    same work, but allow_rechunk lets dask decide, and its warning that this "may
    significantly increase memory usage" is well earned on a full model run. One
    column is the smallest unit the interpolation can act on, and doing it explicitly
    leaves the *horizontal* chunking — which is what bounds memory here — untouched.

    A numpy-backed array is returned unchanged; ``.chunk()`` on one would make it lazy,
    which is a surprising thing for a rechunk helper to do.
    """
    if da.chunks is None or s_dim not in da.dims:
        return da
    return da.chunk({s_dim: -1})


def _z_grid(ds: xr.Dataset, s_dim: str):
    """Build the xgcm ``Grid`` :func:`to_depth` and :func:`to_sigma0` both transform on.

    Split out because the two share every step of the vertical transform except the
    target coordinate itself — one is against ``z_rho``, the other against sigma0.
    """
    import xgcm

    try:
        # >=1.0 dropped `periodic` in favour of `padding` (default None, i.e. not
        # periodic) -- passing periodic=False here raises ValueError rather than
        # the TypeError this except clause catches, so the modern branch omits it.
        return xgcm.Grid(
            ds,
            coords={"Z": {"center": s_dim}},
            autoparse_metadata=False,
        )
    except TypeError:  # older xgcm: no autoparse_metadata, defaults to periodic=True,
        # and transform() refuses a periodic axis -- must be explicit here.
        return xgcm.Grid(ds, coords={"Z": {"center": s_dim}}, periodic=False)


def _transform_spread(grid, ds: xr.Dataset, s_dim: str, targets, target_data, h_dims):
    """Interpolate a riding ``spread`` coordinate onto the transform's target levels.

    Both :func:`to_depth` and :func:`to_sigma0` rebuild their result with a fixed
    coordinate whitelist (lon/lat/z or sigma0/cell_area); left alone, that silently
    drops ``operators.aggregate``'s ``spread`` coordinate (a mean+std envelope) on a
    model lane, while an observational lane -- which never goes through this transform
    -- keeps its own. Transformed the same way each data variable is (same grid, same
    target_data, same linear method), so the band lands on the requested
    depths/isopycnals instead of disappearing. Returns ``None`` when there is no spread
    to carry, or its dims do not match this transform (nothing to interpolate against).
    """
    from ocean_skill.operators import SPREAD_COORD

    if SPREAD_COORD not in ds.coords:
        return None
    spread = ds[SPREAD_COORD]
    if s_dim not in spread.dims or not (h_dims <= set(spread.dims)):
        return None
    transformed = grid.transform(
        _contiguous_column(spread, s_dim),
        "Z",
        targets,
        target_data=target_data,
        method="linear",
    )
    transformed.attrs = dict(spread.attrs)
    return transformed


def _nearest_depth_spread(ds, s_dim: str, idx, h_dims):
    """Snap a riding ``spread`` coordinate onto :func:`nearest_depth_levels`' own index.

    The nearest counterpart of :func:`_transform_spread`: the same "a mean+std
    envelope silently vanishes through a fixed coordinate whitelist" problem,
    solved the same way :func:`nearest_depth_levels` reads every ordinary data
    variable -- ``isel`` at the already-computed nearest-level ``idx``, not a
    fresh interpolation. Returns ``None`` under the same conditions
    :func:`_transform_spread` does: no spread riding at all, or its dims do not
    match this transform (nothing to select against). Reachability (NaN below/
    above the reference column's range) is left to the caller, exactly as the
    ordinary data variables get ``.where(reachable)`` applied after this returns.
    """
    from ocean_skill.operators import SPREAD_COORD

    if SPREAD_COORD not in ds.coords:
        return None
    spread = ds[SPREAD_COORD]
    if s_dim not in spread.dims or not (h_dims <= set(spread.dims)):
        return None
    selected = spread.isel({s_dim: idx}).reset_coords(drop=True)
    selected.attrs = dict(spread.attrs)
    return selected


def to_depth(
    ds: xr.Dataset, meta: dict[str, Any], d: float | list[float]
) -> xr.Dataset:
    """Interpolate s-coordinate fields to fixed depth(s) ``d`` (metres, positive down).

    Uses xgcm's vertical transform against ``z_rho`` (linear; NaN outside the water
    column — no extrapolation). Keeps the result lazy. ``d`` may be a scalar or a list.
    For a true surface field use :func:`surface` instead; for a surface of constant
    potential density rather than constant depth, see :func:`to_sigma0`.
    Non-``s_rho`` variables drop.
    """
    if "z_rho" not in ds.coords:
        ds = add_depth_coord(ds, meta)
    s_dim = meta.get("vertical", {}).get("s_dim", "s_rho")
    depths = np.atleast_1d(np.asarray(d, dtype=float))
    targets = xr.DataArray(-depths, dims="z", coords={"z": -depths})

    grid = _z_grid(ds, s_dim)
    z_rho = _contiguous_column(ds["z_rho"], s_dim)
    # Rho-point fields share z_rho's own horizontal dims; staggered u/v (xi_u/eta_v)
    # need interpolation to rho first (deferred), so are skipped here. Read off
    # z_rho's own dims rather than the hardcoded ("eta_rho", "xi_rho") pair, so a
    # variable already sliced to one grid column or transect (see
    # ocean_skill.transect.grid_slice, whose renamed "along" dim replaces one of the
    # two) still matches -- it shares z_rho's dims exactly, since z_rho went through
    # the same slice. `time` is deliberately excluded: z_rho can carry it (when zeta
    # is present) while a time-invariant variable legitimately does not, and that
    # variable must not be skipped just because it lacks an axis z_rho happens to have.
    h_dims = set(ds["lon"].dims) if "lon" in ds.coords else {"eta_rho", "xi_rho"}
    out = {}
    for var in ds.data_vars:
        da = ds[var]
        if s_dim in da.dims and h_dims <= set(da.dims):
            transformed = grid.transform(
                _contiguous_column(da, s_dim),
                "Z",
                targets,
                target_data=z_rho,
                method="linear",
            )
            # the transform sheds attrs on some xarray versions (see to_sigma0's
            # identical note); carry the source variable's forward explicitly
            # rather than depend on apply_ufunc's keep_attrs default.
            transformed.attrs = {**da.attrs, **transformed.attrs}
            out[var] = transformed
    coords = {"lon": ds["lon"], "lat": ds["lat"], "z": -depths}
    if AREA_COORD in ds.coords:
        coords[AREA_COORD] = ds[AREA_COORD]
    spread = _transform_spread(grid, ds, s_dim, targets, z_rho, h_dims)
    if spread is not None:
        from ocean_skill.operators import SPREAD_COORD

        coords[SPREAD_COORD] = spread
    result = xr.Dataset(out, coords=coords)
    result.attrs.update(ds.attrs)

    # A target shallower than the topmost cell centre (or deeper than the bottom one)
    # interpolates to nothing and silently yields an all-NaN level — most often when
    # asking for exactly 0 m. Say so, and point at surface() for the surface case.
    #
    # Reachability is a property of the grid alone — whether *any* column, at any
    # time, brackets the target between its bottom and top cell centres — not of any
    # particular variable's data, so it is checked once against z_rho directly
    # rather than once per variable against the (lazy) transformed result. Checking
    # the latter used to force the whole transform eagerly here, only to compute it
    # again for real once the caller loads the result — doubling the cost of every
    # interpolation just to phrase this warning. `z_rho.min`/`.max` reduce a single
    # small (zeta-sized) coordinate, not the data, so this keeps the promise this
    # function's docstring already makes: the result stays lazy.
    #
    # The trade-off: a variable that is NaN everywhere *within* a reachable depth
    # (masked out for a reason other than geometry) no longer warns here — only a
    # target the grid itself cannot reach does. A land column is the only routine
    # case that used to trigger the old check outside of unreachable geometry, and a
    # land column's cell centres are themselves NaN, so it still counts as
    # unreachable below.
    #
    # One warning for the whole call, not one per variable: a profile's own depths
    # can drive dozens of below-the-bottom targets (a CTD cast reaching past the
    # model's deepest cell centre is routine), and one line each buries the signal.
    # Collapse the run of NaN depths into a single line naming their span.
    col_min = z_rho.min(s_dim)
    col_max = z_rho.max(s_dim)
    other = [dim for dim in col_min.dims if dim != "z"]
    reachable = (col_min <= targets) & (targets <= col_max)
    reachable = reachable.any(dim=other) if other else reachable
    reachable = np.asarray(reachable)
    nan_depths = [float(d) for i, d in enumerate(depths) if not bool(reachable[i])]
    if nan_depths:
        hint = " use surface() for the surface field" if min(nan_depths) < 5 else ""
        if len(nan_depths) == 1:
            where = f"target depth {nan_depths[0]:g} m is"
        else:
            where = (
                f"{len(nan_depths)} target depths "
                f"({min(nan_depths):g}-{max(nan_depths):g} m) are"
            )
        warnings.warn(
            f"{where} entirely NaN: the target lies outside the model's "
            f"cell-centre range, so nothing can be interpolated;{hint}",
            stacklevel=2,
        )
    return result


def nearest_depth_levels(
    ds: xr.Dataset, meta: dict[str, Any], d: float | list[float], *, ref_time: Any = None
) -> xr.Dataset:
    """Snap fixed target depth(s) ``d`` to the nearest native model level -- no interpolation.

    The nearest-level counterpart of :func:`to_depth`: rather than linearly blending
    two levels together, this looks up the closest ``s_rho`` cell centre (per column)
    and reads its value directly, so what comes back is real model output, never an
    average of two. Keeps the result lazy exactly as :func:`to_depth` does, except for
    the tiny lookup itself (below).

    A level's true depth still moves with the free surface, so "nearest" needs a
    single reference profile to measure against -- built once, from ``ref_time`` (the
    model's own time nearest it) or, absent that, simply the first step of ``ds``'s own
    (already time-cropped) record. That lookup is then applied across **every** time
    step as one static index, deliberately not re-matched per step: recomputing
    ``z_rho`` (a function of the moving ``zeta``) at every one of a mooring's hourly
    steps is exactly the per-timestep cost this function exists to avoid, and the
    reference profile it uses instead is small enough to evaluate eagerly regardless
    of how lazily the rest of ``ds`` is chunked.

    A target outside the *reference* column's own [shallowest, deepest] cell-centre
    range comes back NaN, the same no-extrapolation convention :func:`to_depth` uses
    and for the same reason -- there is nothing there to snap to. ``d`` may be a
    scalar or a list, exactly as in :func:`to_depth`; the result matches its shape and
    coordinate conventions (a ``z`` axis in metres, stored negative-down to match
    ``z_rho``) so the two are interchangeable to a caller. Non-``s_rho`` variables
    drop, exactly as in :func:`to_depth`.
    """
    if "z_rho" not in ds.coords:
        ds = add_depth_coord(ds, meta)
    s_dim = meta.get("vertical", {}).get("s_dim", "s_rho")
    depths = np.atleast_1d(np.asarray(d, dtype=float))
    targets = xr.DataArray(-depths, dims="z", coords={"z": -depths})

    z_rho = _contiguous_column(ds["z_rho"], s_dim)
    if "time" in z_rho.dims:
        z_profile = (
            z_rho.sel(time=ref_time, method="nearest")
            if ref_time is not None
            else z_rho.isel(time=0)
        )
    else:
        z_profile = z_rho
    # Small (one time, one water column or a point-cropped window) and read once --
    # loaded eagerly so the index built from it below is a plain array, never a lazy
    # graph the per-time fields would otherwise be forced through to resolve it.
    # The scalar `time` this leaves behind (a leftover coordinate, not a dimension
    # any more) would otherwise conflict with the real, multi-step `time` on every
    # field the index below is applied to -- dropped for exactly that reason.
    z_profile = z_profile.load()
    if "time" in z_profile.coords:
        z_profile = z_profile.drop_vars("time")

    diff = np.abs(z_profile - targets)
    # fillna guards a masked (land) column: without it, argmin's tie-breaking on a
    # NaN cell centre is undefined rather than simply "never nearest".
    idx = diff.fillna(np.inf).argmin(s_dim)

    h_dims = set(ds["lon"].dims) if "lon" in ds.coords else {"eta_rho", "xi_rho"}
    out = {}
    for var in ds.data_vars:
        da = ds[var]
        if s_dim in da.dims and h_dims <= set(da.dims):
            selected = da.isel({s_dim: idx})
            # Plain isel (unlike to_depth's xgcm transform) drags every coordinate
            # sharing the indexed dim along for the ride -- z_rho chief among them,
            # now itself indexed onto the picked levels. Dropped so this result
            # carries exactly the coordinate set to_depth's does (lon/lat/z, no
            # more), or a mixed ["surface", ...] request downstream (which
            # concatenates this against roms.surface's own z_rho-free result) sees
            # a coordinate mismatch between the two pieces.
            selected = selected.reset_coords(drop=True)
            selected.attrs = dict(da.attrs)
            out[var] = selected
    coords = {"lon": ds["lon"], "lat": ds["lat"], "z": -depths}
    if AREA_COORD in ds.coords:
        coords[AREA_COORD] = ds[AREA_COORD]
    spread = _nearest_depth_spread(ds, s_dim, idx, h_dims)

    # Reachability, exactly as to_depth checks it: a property of the reference
    # column's geometry alone, not of any one variable's data.
    col_min = z_profile.min(s_dim)
    col_max = z_profile.max(s_dim)
    other = [dim for dim in col_min.dims if dim != "z"]
    reachable = (col_min <= targets) & (targets <= col_max)
    for name in out:
        out[name] = out[name].where(reachable)
    if spread is not None:
        from ocean_skill.operators import SPREAD_COORD

        coords[SPREAD_COORD] = spread.where(reachable)
    result = xr.Dataset(out, coords=coords)
    result.attrs.update(ds.attrs)
    reachable_any = reachable.any(dim=other) if other else reachable
    reachable_any = np.asarray(reachable_any)
    nan_depths = [float(dd) for i, dd in enumerate(depths) if not bool(reachable_any[i])]
    if nan_depths:
        hint = " use surface() for the surface field" if min(nan_depths) < 5 else ""
        if len(nan_depths) == 1:
            where = f"target depth {nan_depths[0]:g} m is"
        else:
            where = (
                f"{len(nan_depths)} target depths "
                f"({min(nan_depths):g}-{max(nan_depths):g} m) are"
            )
        warnings.warn(
            f"{where} entirely NaN: the target lies outside the reference column's "
            f"cell-centre range, so nothing can be snapped to;{hint}",
            stacklevel=2,
        )
    return result


def to_sigma0(
    ds: xr.Dataset, meta: dict[str, Any], s: float | list[float]
) -> xr.Dataset:
    """Interpolate s-coordinate fields onto surface(s) of constant potential density.

    An isopycnal slice: the same xgcm vertical transform :func:`to_depth` uses, but
    against potential density anomaly (sigma0, TEOS-10 via
    :func:`ocean_skill.mld.potential_density`) instead of ``z_rho``. Water masses
    move along density surfaces, not depth surfaces, so this is often the more
    physically meaningful slice through a stratified column. ``s`` may be a scalar
    or a list, exactly as ``d`` is for :func:`to_depth`.

    ``s`` is read as sigma0 in kg/m3 -- *anomaly* form (roughly 20-28 through most of
    the ocean), not full in-situ density (roughly 1020-1028): a value at or above
    1000 almost certainly means a full density was quoted by mistake, and is refused
    rather than silently naming a surface far from the one meant.

    There is deliberately no ``"rho"``/``"density"`` alias for this request. ROMS's
    own ``rho`` output is *in-situ* density -- pressure/compressibility included --
    which at depth names a materially different surface from sigma0 (density
    referenced to the sea surface); reading a value off one and asking for the
    other under the same name would be a silent, and depth-growing, mismatch. sigma0
    carries its own reference pressure in its name, leaving room for ``sigma2``/
    ``sigma4`` (referenced to 2000/4000 dbar, the usual choice below where a
    surface-referenced potential density becomes thermobarically unreliable) as
    later siblings rather than a redefinition of what "density" means here.

    sigma0 is computed from whatever ``ds`` carries for
    ``sea_water_potential_temperature``/``sea_water_practical_salinity`` *at the
    time this is called* -- if those have already been reduced (a time mean, say),
    the slice is onto the density surface of that mean, not the mean of
    instantaneously sliced surfaces. Two water masses of different
    temperature/salinity can share a sigma0 value (density compensation), and a
    column where sigma0 is not monotonic with depth (a density inversion) gives
    xgcm's linear transform more than one crossing to choose from; this does not
    detect or resolve that, it interpolates whatever profile it is given.

    NaN outside the column's own sigma0 range (no extrapolation), with a warning
    naming the target -- the same shape :func:`to_depth` uses for a target beyond
    the water column.
    """
    from ocean_skill.mld import potential_density
    from ocean_skill.units import find_variable

    if "z_rho" not in ds.coords:
        ds = add_depth_coord(ds, meta)
    s_dim = meta.get("vertical", {}).get("s_dim", "s_rho")

    values = np.atleast_1d(np.asarray(s, dtype=float))
    over_1000 = values[values >= 1000]
    if over_1000.size:
        raise ValueError(
            f"sigma0={s!r} looks like a full density (roughly 1020-1028 kg/m3), not "
            "a potential density *anomaly* -- sigma0 is density minus 1000 kg/m3, "
            f"typically 20-28 for seawater. Did you mean {list(over_1000 - 1000)!r}?"
        )

    temp = find_variable(ds, "sea_water_potential_temperature")
    salt = find_variable(ds, "sea_water_practical_salinity")
    if temp is None or salt is None:
        missing = (
            "sea_water_potential_temperature" if temp is None else
            "sea_water_practical_salinity"
        )
        raise ValueError(
            f"an isopycnal slice needs {missing!r}, which is not in this dataset "
            "(or not standardized to that name -- check the catalog entry's "
            "standard_names map, or that the source actually carries it)."
        )
    try:
        sigma0 = potential_density(temp, salt, ds["z_rho"], ds["lon"], ds["lat"])
    except ImportError as exc:
        raise ImportError(
            "isopycnal slicing needs gsw (TEOS-10); it is listed in "
            "environment.yml but not installed -- `conda install -c conda-forge "
            "gsw` or `pip install gsw`."
        ) from exc

    grid = _z_grid(ds, s_dim)
    sigma0 = _contiguous_column(sigma0, s_dim)
    targets = xr.DataArray(values, dims="sigma0", coords={"sigma0": values})

    # See the identical note in to_depth: read off lon's own dims rather than the
    # hardcoded rho pair, so a variable already sliced to a transect matches too.
    h_dims = set(ds["lon"].dims) if "lon" in ds.coords else {"eta_rho", "xi_rho"}
    out = {}
    for var in ds.data_vars:
        da = ds[var]
        # only rho-point 3-D fields share sigma0's grid; staggered u/v need
        # interpolation to rho first (deferred, as in to_depth), so skip them here.
        if s_dim in da.dims and h_dims <= set(da.dims):
            transformed = grid.transform(
                _contiguous_column(da, s_dim),
                "Z",
                targets,
                target_data=sigma0,
                method="linear",
            )
            # the transform sheds attrs; carry the source variable's forward, plus a
            # note of how this level came to be, since "sliced onto a density
            # surface" is not otherwise recoverable from the result alone.
            transformed.attrs = {
                **da.attrs,
                "isopycnal_slice": (
                    "linear interpolation onto sigma0 (potential density anomaly, "
                    "TEOS-10 via gsw) surfaces"
                ),
            }
            out[var] = transformed
    coords = {"lon": ds["lon"], "lat": ds["lat"], "sigma0": values}
    if AREA_COORD in ds.coords:
        coords[AREA_COORD] = ds[AREA_COORD]
    spread = _transform_spread(grid, ds, s_dim, targets, sigma0, h_dims)
    if spread is not None:
        from ocean_skill.operators import SPREAD_COORD

        coords[SPREAD_COORD] = spread
    result = xr.Dataset(out, coords=coords)
    result.attrs.update(ds.attrs)
    result["sigma0"].attrs = {
        "units": "kg m-3",
        "long_name": "potential density anomaly (sigma0, TEOS-10)",
        "standard_name": "sea_water_sigma_theta",
    }

    # A target denser or lighter than the column holds anywhere interpolates to
    # nothing and silently yields an all-NaN level. One warning per variable naming
    # the span of dead surfaces, not one line each (see to_depth's identical note).
    for var in result.data_vars:
        nan_targets = [
            float(target)
            for i, target in enumerate(values)
            if not bool(np.isfinite(result[var].isel(sigma0=i)).any())
        ]
        if not nan_targets:
            continue
        if len(nan_targets) == 1:
            where = f"at sigma0={nan_targets[0]:g} kg/m3 is"
        else:
            where = (
                f"at {len(nan_targets)} target surfaces "
                f"(sigma0 {min(nan_targets):g}-{max(nan_targets):g} kg/m3) are"
            )
        warnings.warn(
            f"{var!r} {where} entirely NaN: the target density lies outside this "
            "water column's sigma0 range everywhere, so nothing can be interpolated.",
            stacklevel=2,
        )
    return result
