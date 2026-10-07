"""ROMS model adapter (reader-independent).

The catalog says *how to read* a ROMS file (driver/args → intake); this module turns
that raw output into a CF-standardized dataset ocean-skill can compare: attach the grid
(lon/lat/h/mask), decode ``ocean_time``, rename variables to CF standard_names, mask
land, and reconstruct depth. The s-coordinate → z reconstruction (Vtransform 1 or 2,
using ``Cs_r``/``sigma_r`` from the grid) stays lazy (dask) — no unchunk needed.
Lateral regridding lives in :mod:`ocean_skill.align` (xesmf), not here.

Matching the model to a *depth* is the other half, and the one place the free surface
matters. ROMS' ``z_rho`` is a height relative to mean sea level, so it rides up and down
with ``zeta`` -- but most in-situ depths (a CTD cast, a pressure sensor, a profiler) are
measured *below the instantaneous surface*, while a pier sonde or a bottom-mounted
instrument sits at a position fixed in space. Which one a source is decides the right
target (in a macrotidal estuary, up to a tidal range of difference), so
:func:`to_depth`, :func:`nearest_depth_levels` and :func:`depth_band` work in a *frame*
(:func:`frame_coordinate`): ``z - zeta`` for ``origin: surface``, ``z - datum_z_m`` for
``origin: fixed``. In either, a depth ``d`` is simply the coordinate ``-d``. The caller
says which origin applies (``convention=``, a plain mapping -- this module never reads
catalog metadata for it); this module only applies it.

The two vertical matches share one numpy kernel (:func:`_match_columns`) that matches
**per time step** against the frame of that step, and that treats the top and bottom
half-cells -- inside the water column but beyond the outermost cell centres --
explicitly (see :func:`to_depth` for the edge policy) instead of leaving a NaN band
there.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np
import xarray as xr

__all__ = [
    "AREA_COORD",
    "FREE_SURFACE_NAMES",
    "GEOGRAPHIC_VELOCITY_NAMES",
    "GRID_RELATIVE_VELOCITY_NAMES",
    "GRID_VARIABLE_NAMES",
    "WEIGHT_COORD",
    "add_depth_coord",
    "add_interface_coord",
    "bottom",
    "depth_average",
    "depth_band",
    "derived_geographic_velocities",
    "frame_coordinate",
    "nearest_depth_levels",
    "standardize",
    "surface",
    "to_depth",
    "to_sigma0",
    "water_column_bounds",
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


#: The ROMS grid and s-coordinate variables the loader looks up by their *own* names:
#: ``h`` (depth coordinate, water-column bounds), ``mask_rho`` (land mask), ``angle``
#: (velocity rotation), ``pm``/``pn`` (cell area), ``Cs_r``/``sigma_r``/``hc``/
#: ``Vtransform`` (the s-coordinate -> ``z_rho`` transform, ``Cs_w``/``sigma_w`` for
#: ``z_w``), ``lon_rho``/``lat_rho`` (positions), and the staggered/auxiliary members
#: of the same families. A file may well give one of them a ``standard_name`` of its
#: own (``h``: ``sea_floor_depth``, ``mask_rho``: ``land_binary_mask``), and the build
#: probe records attributes ahead of its fallback table -- but a renamed ``h`` is no
#: longer found, so no depth coordinate, no land mask. Hence this one list, read by both
#: sides: :func:`ocean_skill.build._probe` never records these names in a source's
#: ``standard_names``, and :func:`standardize` never renames them, which also protects
#: catalogs written before the probe knew better.
GRID_VARIABLE_NAMES = (
    # horizontal grid: positions, land masks, metrics, rotation, Coriolis
    "lon_rho",
    "lat_rho",
    "lon_u",
    "lat_u",
    "lon_v",
    "lat_v",
    "lon_psi",
    "lat_psi",
    "mask_rho",
    "mask_u",
    "mask_v",
    "mask_psi",
    "wetdry_mask_rho",
    "wetdry_mask_u",
    "wetdry_mask_v",
    "wetdry_mask_psi",
    "h",
    "angle",
    "pm",
    "pn",
    "f",
    # vertical s-coordinate: the values, the stretching and the transform's parameters
    "s_rho",
    "s_w",
    "sigma_r",
    "sigma_w",
    "Cs_r",
    "Cs_w",
    "hc",
    "Vtransform",
    "Vstretching",
    "theta_s",
    "theta_b",
    "Tcline",
)

#: The names a ROMS source's free surface goes by: ``zeta`` as the model writes it, and
#: the standard name the catalog's ``standard_names`` renames it to (build.py's
#: ``ROMS_STANDARD_NAMES``). The only two :func:`_zeta_of` looks for, so the only two
#: :func:`standardize` will leave the free surface under -- anything else a catalog (or
#: a file's own ``standard_name``) calls it, it is not found and the depth coordinate
#: silently rides a flat ``zeta = 0``. The last is the standard name the probe records.
FREE_SURFACE_NAMES = ("zeta", "sea_surface_height_above_geoid")

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
    if tcoord in ds.dims and tcoord != tdim:
        # Classic Rutgers ROMS makes ``ocean_time`` itself the dimension, where UCLA
        # puts the ``ocean_time`` variable on a dim called ``time``. Left alone, the
        # assign_coords below would mint a brand-new, unrelated ``time`` dim while the
        # data stayed on ``ocean_time``. Renaming the dim first (dropping its index so
        # the variable rides along as an ordinary non-index coord, like UCLA's) puts
        # the data on ``tdim`` and lets the decode below attach to it.
        if tcoord in ds.indexes:
            ds = ds.drop_indexes(tcoord)
        ds = ds.rename_dims({tcoord: tdim})
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


def _normalize_classic_layout(ds: xr.Dataset) -> xr.Dataset:
    """Give a classic-Rutgers s-coordinate the UCLA layout (``sigma_*`` + bare dim).

    Classic ROMS output stores its sigma values as the 1-D dimension coordinates
    ``s_rho``/``s_w`` and ships no ``sigma_r``/``sigma_w``; UCLA-ROMS has the opposite
    layout, ``sigma_r``/``sigma_w`` data variables over *bare* ``s_rho``/``s_w`` dims.
    Everything downstream is written against the UCLA one -- :func:`add_depth_coord`
    reads ``sigma_r``, :mod:`ocean_skill.comparison` re-attaches ``sigma_r``/
    ``sigma_w`` after slicing, and plotting's vertical-coordinate lookup
    (``operators.vertical_coord_on``) expects ``s_rho`` to be a bare dim so it falls
    through to ``z_rho``. A *valued* ``s_rho`` would make plots and profiles use the
    sigma values as the vertical axis instead of depth in metres.

    So when ``sigma_r``/``sigma_w`` is missing and the matching ``s_rho``/``s_w``
    variable exists, move its values (and attrs) onto the ``sigma_*`` data variable
    and drop the label, leaving the bare dimension. A no-op on UCLA data, which
    already has ``sigma_*``.
    """
    for label, sigma in (("s_rho", "sigma_r"), ("s_w", "sigma_w")):
        if sigma not in ds.variables and label in ds.variables:
            ds = ds.assign({sigma: (label, ds[label].values, ds[label].attrs)})
            ds = ds.drop_vars(label)
    return ds


def _standard_name_renames(ds: xr.Dataset, meta: dict[str, Any]) -> dict[str, str]:
    """Return the ``{variable: standard_name}`` renames :func:`standardize` applies.

    The catalog entry's ``standard_names`` map, restricted to what ``ds`` carries and
    less what the loader could not do without. Two rules, both there for catalogs
    written before :func:`ocean_skill.build._probe` stopped recording the offending
    entries (a store whose own ``standard_name`` attributes beat the probe's fallback
    table put them there):

    * a name in :data:`GRID_VARIABLE_NAMES` is never renamed. The depth coordinate, the
      land mask and the velocity rotation look ``h``/``mask_rho``/``angle``/... up by
      those names, and a renamed one raised ``KeyError: "No variable named 'h'"`` or --
      for the mask -- silently left land unmasked;
    * ``zeta`` is renamed only to a name :func:`_zeta_of` finds
      (:data:`FREE_SURFACE_NAMES`). Any other target (``sea_surface_elevation_anomaly``,
      ...) is replaced by ``sea_surface_height_above_geoid`` -- still the CF name a
      catalog would give a free surface, but one the depth coordinate can see, where a
      stranger would fall back to a flat ``zeta = 0`` with no error.

    Every other mapping is applied as written.
    """
    rename: dict[str, str] = {}
    for name, target in (meta.get("standard_names") or {}).items():
        if name not in ds.variables or name in GRID_VARIABLE_NAMES:
            continue
        if name == "zeta" and target not in FREE_SURFACE_NAMES:
            target = FREE_SURFACE_NAMES[-1]
        if target != name:
            rename[name] = target
    return rename


def standardize(
    ds: xr.Dataset, meta: dict[str, Any], *, derive_velocity: bool = False
) -> xr.Dataset:
    """Return a CF-standardized ROMS Dataset (grid attached, renamed, masked, depth).

    Parameters
    ----------
    ds
        Raw ROMS output opened per the catalog entry (rho-point fields).
    meta
        The catalog entry ``metadata`` (``grid``, ``vertical``, ``standard_names``,
        ``reference_date``/``time_*``). ``standard_names`` renames variables to their
        CF names, except the grid/vertical variables (:data:`GRID_VARIABLE_NAMES`),
        which keep their own, and ``zeta``, which only ever lands on a name the depth
        coordinate can find (:data:`FREE_SURFACE_NAMES`) -- see
        :func:`_standard_name_renames`.
    derive_velocity
        Whether to derive true geographic east/north velocity from the staggered
        grid-relative components (see :func:`_add_geographic_velocity`) here, up
        front. Defaults to ``False``: the derivation is a dask task graph that
        scales with the whole file's chunk count and can cost tens of seconds to
        *build*, well before anything is computed or even the requested variable
        is known -- doing it unconditionally, for every read of a ROMS source
        regardless of what a caller actually wants, is the exact cost this default
        avoids. :func:`ocean_skill.sources.read` (the ordinary read path) always
        uses this default; a real caller derives velocity itself, on demand, only
        once a request actually names it -- see :func:`ocean_skill.comparison
        .prepare_source` (full-domain or windowed, whichever its crop resolved to)
        and :func:`ocean_skill.comparison._variable_available` (a cheap check,
        with no graph built at all). Pass ``True`` only for a caller -- direct or
        test-only -- that wants the older, simpler all-in-one shape.
    """
    # first, so a self-contained grid (grid is ds, below) sees ``sigma_r`` too
    ds = _normalize_classic_layout(ds)
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

    # rename model variable names -> CF standard_names (see _standard_name_renames:
    # never the grid/vertical variables, and ``zeta`` only to a name _zeta_of knows)
    ds = ds.rename(_standard_name_renames(ds, meta))

    # derive TRUE geographic east/north velocity from the staggered grid-relative
    # components + the grid angle, before the mask loop below so the new rho-dim
    # vars get land-masked with everything else. Lazy; a no-op when the source has
    # no velocity (or no angle to rotate it by) -- see _add_geographic_velocity.
    # Off by default (derive_velocity=False, see this function's own docstring): a
    # real caller derives on demand instead, once it actually knows a request
    # names velocity, rather than paying for this on every read regardless.
    if derive_velocity:
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


def _vertical_params(ds: xr.Dataset, meta: dict[str, Any]) -> tuple[float, int]:
    """Return ``(hc, Vtransform)`` for the s-coordinate -> depth transform.

    Each comes from the catalog entry's ``vertical`` block when it names it, else from
    the file's own 0-d ``hc``/``Vtransform`` variable (classic Rutgers output carries
    both). ``hc`` has no default -- it is a property of the run -- so a missing one
    raises a clear error rather than the opaque ``float(None)`` it used to. An absent
    ``Vtransform`` means 2, the UCLA-ROMS/roms-tools convention this module grew up on.
    """
    vert = meta.get("vertical", {})
    hc = vert.get("hc")
    if hc is None and "hc" in ds.variables:
        hc = ds["hc"].values
    if hc is None:
        raise ValueError(
            "ROMS depth needs 'hc' -- in the catalog entry's metadata 'vertical' block "
            "or as a variable on the dataset."
        )
    vt = vert.get("Vtransform")
    if vt is None and "Vtransform" in ds.variables:
        vt = ds["Vtransform"].values
    if vt is None:
        vt = 2
    vt = int(vt)
    if vt not in (1, 2):
        raise ValueError(f"Unsupported ROMS Vtransform {vt!r}; expected 1 or 2.")
    return float(hc), vt


def _s_to_z(sigma, Cs, h, zeta, hc: float, vtransform: int):
    """Evaluate the ROMS s-coordinate -> depth transform (metres, negative down).

    Vtransform 2 (UCLA-ROMS, roms-tools) is ``zeta + (zeta + h) * s`` with
    ``s = (hc*sigma + h*Cs) / (hc + h)``; the expression order is kept exactly so
    results are bit-identical to the inline form it replaced. Vtransform 1 (classic
    Rutgers, and what xroms applies) is ``z0 + zeta * (1 + z0/h)`` with
    ``z0 = hc*(sigma - Cs) + Cs*h``. ``vtransform`` is validated by
    :func:`_vertical_params`, so it is 1 or 2 here.
    """
    if vtransform == 1:
        z0 = hc * (sigma - Cs) + Cs * h
        return z0 + zeta * (1 + z0 / h)
    s = (hc * sigma + h * Cs) / (hc + h)
    return zeta + (zeta + h) * s


def _zeta_of(ds: xr.Dataset) -> xr.DataArray:
    """Return the free-surface height ``zeta`` (metres above mean sea level) of ``ds``.

    ``zeta`` when ``ds`` carries it, else ``sea_surface_height_above_geoid`` (what the
    catalog's ``standard_names`` rename it to -- a data variable or a coordinate alike),
    else zeros shaped like ``h``: a flat free surface, so a dataset with no free-surface
    field still has a (static) depth coordinate and frame. The one lookup the depth
    coordinates, the matching frame and the water-column bounds all share, so none of
    them can quietly use a different surface from the others.
    """
    for name in FREE_SURFACE_NAMES:
        if name in ds.variables:
            return ds[name]
    return xr.zeros_like(ds["h"])  # no free-surface field: use zeta = 0


def add_depth_coord(
    ds: xr.Dataset, meta: dict[str, Any], *, zero_zeta: bool = False
) -> xr.Dataset:
    """Attach the (lazy) ROMS z_rho depth coordinate via Vtransform 1 or 2.

    Vtransform 2: ``z_rho = zeta + (zeta + h) * (hc*sigma_r + h*Cs_r) / (hc + h)``.
    Vtransform 1: ``z0 = hc*(sigma_r - Cs_r) + Cs_r*h`` and
    ``z_rho = z0 + zeta * (1 + z0/h)`` (metres, negative down). Requires
    ``h``/``Cs_r``/``sigma_r`` (from the grid) and ``zeta``. ``hc`` and the Vtransform
    come from the catalog's ``vertical`` block, or from the file's own ``hc``/
    ``Vtransform`` variables when it carries them (see :func:`_vertical_params`).

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
    hc, vtransform = _vertical_params(ds, meta)
    zeta = xr.zeros_like(ds["h"]) if zero_zeta else _zeta_of(ds)
    z_rho = _s_to_z(ds["sigma_r"], ds["Cs_r"], ds["h"], zeta, hc, vtransform)
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

    Same Vtransform-1-or-2 formula as :func:`add_depth_coord`, evaluated on
    ``sigma_w``/``Cs_w`` (N+1 interfaces) instead of ``sigma_r``/``Cs_r`` (N centres).

    The two serve different operations and neither replaces the other. Data lives at
    *centres*, so interpolating to a depth (:func:`to_depth`) must use ``z_rho``.
    Cell *thicknesses* only exist between interfaces, so a depth-band average
    (:func:`depth_average`) must use ``z_w``. Interfaces also start exactly at the
    free surface, which is why a band average needs no edge rule where a point-depth
    match does: the shallowest ``z_rho`` can be 7 m down in deep water (a target above
    it is in the top half-cell, which :func:`to_depth` fills with that cell's value),
    but the shallowest ``z_w`` is always the surface itself.

    ``zero_zeta`` -- see :func:`add_depth_coord`, the same zeta-free mesh for a
    section built on ``s_w`` instead of ``s_rho``.
    """
    hc, vtransform = _vertical_params(ds, meta)
    zeta = xr.zeros_like(ds["h"]) if zero_zeta else _zeta_of(ds)
    z_w = _s_to_z(ds["sigma_w"], ds["Cs_w"], ds["h"], zeta, hc, vtransform)
    dims = ("time", "s_w", "eta_rho", "xi_rho")  # see add_depth_coord's note on `...`
    z_w = z_w.transpose(*[d for d in dims if d in z_w.dims], ...)
    return ds.assign_coords(z_w=z_w)


#: The two things a depth can be measured from: ``"surface"`` -- below the moving free
#: surface (a CTD cast, a pressure-derived depth) -- or ``"fixed"`` -- a position fixed
#: in space relative to a datum (a pier sonde, a bottom-mounted instrument). Spelled out
#: here rather than imported: this module is handed the answer as a plain mapping (see
#: :func:`_frame_spec`) and never reads catalog metadata to decide it.
_ORIGINS = ("surface", "fixed")


def _check_origin(origin: str) -> str:
    """Return ``origin`` if it is one of :data:`_ORIGINS`, else raise a clear error."""
    if origin not in _ORIGINS:
        raise ValueError(
            f"depth origin must be one of {_ORIGINS}, got {origin!r}: 'surface' "
            "measures depth below the moving free surface, 'fixed' at a position "
            "fixed in space."
        )
    return origin


def _check_datum(datum_z_m: float) -> float:
    """Return ``datum_z_m`` as a float if it is finite, else raise a clear error."""
    datum = float(datum_z_m or 0.0)
    if not np.isfinite(datum):
        raise ValueError(f"datum_z_m must be a finite number, got {datum_z_m!r}.")
    return datum


def _frame_spec(convention: Mapping[str, Any] | None) -> tuple[str, float, str]:
    """Read ``(origin, datum_z_m, source)`` from a plain ``convention`` mapping.

    Only these three keys are read; anything else the mapping carries (``support``,
    ``positive``, ...) is the caller's business, not a matter for the vertical match.
    Missing keys mean the historical meaning: ``origin="fixed"``, ``datum_z_m=0.0`` and
    ``source="default"`` -- "nobody declared this" (``"declared"``/``"inferred"``/
    ``"data"`` say someone did, and are what silences the large-tide warning in
    :func:`to_depth`). ``datum_z_m`` only exists for a fixed origin: a surface origin
    has no datum to offset from, so it is 0 there whatever was passed.
    """
    if convention is None:
        convention = {}
    elif not isinstance(convention, Mapping):
        raise TypeError(
            "convention must be a mapping with 'origin' / 'datum_z_m' / 'source' keys "
            f"(or None), got {type(convention).__name__}."
        )
    origin = _check_origin(convention.get("origin") or "fixed")
    datum = 0.0 if origin == "surface" else _check_datum(convention.get("datum_z_m"))
    return origin, datum, str(convention.get("source") or "default")


def frame_coordinate(
    ds: xr.Dataset,
    meta: dict[str, Any],
    origin: str = "fixed",
    *,
    datum_z_m: float = 0.0,
    z: str = "z_rho",
) -> xr.DataArray:
    """Return the vertical coordinate in the matching *frame*: a depth is ``-d`` in it.

    ``z_rho`` (or, with ``z="z_w"``, the cell interfaces) is a height above mean sea
    level, and it moves: ``z_rho = zeta + (zeta + h) * s``. A depth quoted *below the
    free surface* therefore sits at ``z = zeta - d``, and one quoted at a position
    *fixed in space* at ``z = datum_z_m - d`` (``datum_z_m`` is the height of the
    observation datum in the model's frame, mean sea level = 0). Subtracting that
    reference leaves a coordinate in which the target is ``-d`` either way::

        origin="surface":  z - zeta         (= (zeta + h) * s for Vtransform 2)
        origin="fixed":    z - datum_z_m    (z itself for the default datum 0)

    which is why :func:`to_depth` and :func:`nearest_depth_levels` need only one kernel
    for both origins. Only ``z`` and ``zeta`` enter, so this holds for Vtransform 1 and
    2 alike (Vtransform 1: ``z - zeta = z0 * (1 + zeta/h)``, the same stretching of the
    unperturbed column that ``(zeta + h)/h`` gives Vtransform 2).

    ``z_rho``/``z_w`` is attached first (:func:`add_depth_coord`/
    :func:`add_interface_coord`) when ``ds`` lacks it. An existing one is used as
    found, so it must have been built from the same ``zeta`` that ``ds`` carries -- a
    ``zeta`` that was reduced *after* ``z_rho`` was built would put the frame out of
    step with it. ``datum_z_m`` is ignored for ``origin="surface"``. Lazy.
    """
    origin = _check_origin(origin)
    if z not in ("z_rho", "z_w"):
        raise ValueError(
            f"z must be 'z_rho' (cell centres) or 'z_w' (interfaces), got {z!r}."
        )
    if z not in ds.coords:
        attach = add_depth_coord if z == "z_rho" else add_interface_coord
        ds = attach(ds, meta)
    height = ds[z]
    datum = _check_datum(datum_z_m)
    if origin == "surface":
        frame = height - _zeta_of(ds)
    elif datum:
        frame = height - datum
    else:
        frame = height  # datum 0: the frame *is* z (no arithmetic, no graph layer)
    # ``ds[z]`` lists itself among its own coordinates; the frame is not ``z`` any more
    # (a surface frame is shifted by zeta), so the stale copy under that name goes.
    return frame.drop_vars(z, errors="ignore")


def water_column_bounds(
    ds: xr.Dataset,
    meta: dict[str, Any],
    origin: str = "fixed",
    *,
    datum_z_m: float = 0.0,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Return ``(top, bottom)``: the free surface and seafloor in the matching frame.

    The two bounds a target depth has to lie between to be *in the water*, in the same
    frame as :func:`frame_coordinate` (so directly comparable with its values and with
    a target's ``-d``)::

        origin="surface":  top = 0,              bottom = -(h + zeta)
        origin="fixed":    top = zeta - datum,   bottom = -h - datum

    The outermost cell *centres* stop half a cell short of both, so the bounds are what
    tell "in the top (bottom) half-cell" -- edge-filled with that cell's value -- from
    "above the free surface" (below the seafloor), which is NaN; see :func:`to_depth`.
    ``zeta`` is land-masked (see :func:`standardize`), so over land the bounds go NaN
    like the frame does (a surface origin's ``top = 0`` is made NaN there too). Lazy.
    """
    origin = _check_origin(origin)
    zeta = _zeta_of(ds)
    h = ds["h"]
    if origin == "surface":
        # ``h + zeta`` puts h's dims first; zeta's own order (time first) is the one
        # ``top`` has and ``z_rho`` follows, so keep the pair consistent.
        bottom = (-(h + zeta)).transpose(*zeta.dims, ...)
        return xr.zeros_like(zeta).where(zeta.notnull()), bottom
    datum = _check_datum(datum_z_m)
    return zeta - datum, -h - datum


#: Coordinate name carrying per-cell weights, so a later reduction can honour them.
#: Riding on the data means :func:`ocean_skill.operators.aggregate` needs no special
#: case for depth -- ``{"Z": "mean"}`` finds the weights and uses them, exactly as
#: ``{"T": "mean"}`` needs nothing special for time.
WEIGHT_COORD = "dz"


def _interface_dim(ds: xr.Dataset, s_dim: str) -> str:
    """Return the name of the cell-interface dimension (``s_w``) ``z_w`` is built on.

    Read off ``sigma_w``'s own (one) dimension when ``ds`` carries it -- ``z_w`` is
    built from it, so it is the interface axis by construction. The old rule ("the
    first dim of ``z_w`` that is not the centre dim") only held while ``z_w`` had no
    time axis: with a free surface that moves, ``z_w`` is ``(time, s_w, ...)`` and that
    rule picks ``time``.
    """
    if "sigma_w" in ds.variables and ds["sigma_w"].ndim == 1:
        return str(ds["sigma_w"].dims[0])
    z_w_dims = ds["z_w"].dims
    if "s_w" in z_w_dims:
        return "s_w"
    return next((d for d in z_w_dims if d not in ds[s_dim].dims), "s_w")


def depth_band(
    ds: xr.Dataset,
    meta: dict[str, Any],
    low: float,
    high: float,
    *,
    convention: Mapping[str, Any] | None = None,
) -> xr.Dataset:
    """Return the cells overlapping ``low``-``high`` m, with overlap as weights.

    A *selection*, not a reduction: the vertical dimension survives, narrowed to the
    cells the band touches, with :data:`WEIGHT_COORD` giving how much of each lies
    inside it (partial at the boundary). Collapsing it is
    :func:`ocean_skill.operators.aggregate`'s job, which keeps "select narrows,
    aggregate collapses" true for depth exactly as it is for time — and makes
    ``{"Z": "max"}`` or ``{"Z": "std"}`` over a band meaningful rather than
    impossible.

    ``low``/``high`` are depths in the ``convention``'s frame (see
    :func:`frame_coordinate`): metres below the moving free surface for ``origin:
    surface`` -- the band then follows the tide, its top edge always the surface -- or
    below the fixed datum (mean sea level by default) for ``origin: fixed``. With a
    free surface that varies in time, the interfaces and so the weights vary per time
    step (``dz`` keeps its time axis) and a cell is kept if the band touches it at *any*
    step, so a later reduction sees one consistent set of cells. ``convention=None``
    is the fixed origin at datum 0, ``-z_w`` exactly as before the frame existed.
    """
    origin, datum_z_m, _ = _frame_spec(convention)
    if "z_w" not in ds.coords:
        ds = add_interface_coord(ds, meta)
    s_dim = meta.get("vertical", {}).get("s_dim", "s_rho")
    w_dim = _interface_dim(ds, s_dim)

    # the frame is negative-down; work in positive-down metres to match the request.
    depth_w = -frame_coordinate(ds, meta, origin, datum_z_m=datum_z_m, z="z_w")
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
    ds: xr.Dataset,
    meta: dict[str, Any],
    low: float,
    high: float,
    *,
    convention: Mapping[str, Any] | None = None,
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

    ``convention`` says what the band is measured from (see :func:`depth_band`):
    below the moving free surface (``origin: surface``) or at depths fixed in space.
    """
    band = depth_band(ds, meta, low, high, convention=convention)
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

    This is the right operation for surface comparisons — unlike matching a fixed
    shallow depth, which asserts a depth the model's top cell (0.2 m thick on a shelf,
    17 m offshore) may not resolve, and which is NaN wherever the free surface has
    dropped below that depth (see :func:`to_depth`'s edge policy). The native top cell
    is the model's own surface layer whatever its thickness and wherever the tide is.
    Drops the vertical dimension.
    """
    s_dim = (meta or {}).get("vertical", {}).get("s_dim", "s_rho")
    top = ds.isel({s_dim: -1}) if s_dim in ds.dims else ds
    return top.drop_vars([s_dim, "z_rho"], errors="ignore")


def bottom(ds: xr.Dataset, meta: dict[str, Any] | None = None) -> xr.Dataset:
    """Return the bottom field: the lowest s-coordinate level (``s_rho`` index 0).

    The twin of :func:`surface`, for a bottom-mounted instrument -- a seabed ADCP or
    pressure sensor measures the near-bed water, whatever depth that is -- or any
    comparison that means "the model's bottom cell" rather than a depth. Like
    :func:`surface` it takes the native cell (ROMS orders ``s_rho`` bottom -> top)
    instead of matching a depth, so it is never NaN for want of a cell centre at the
    target and does not depend on where the free surface is. Drops the vertical
    dimension.
    """
    s_dim = (meta or {}).get("vertical", {}).get("s_dim", "s_rho")
    low = ds.isel({s_dim: 0}) if s_dim in ds.dims else ds
    return low.drop_vars([s_dim, "z_rho"], errors="ignore")


def _contiguous_column(da: xr.DataArray, s_dim: str) -> xr.DataArray:
    """Return ``da`` with its vertical axis in a single dask chunk.

    Matching to a depth reads the whole water column at once — the vertical is an
    ``apply_ufunc`` *core* dimension, as it is for the depth kernel here and for xgcm's
    transform — so a source chunked along ``s_rho`` fails outright with "consists of
    multiple chunks, but is also a core dimension". Whether that happens is a property
    of how the store was written, which is why it can lie unnoticed until a particular
    dataset is used.

    Rechunking here rather than passing ``allow_rechunk=True``: the two do the
    same work, but allow_rechunk lets dask decide, and its warning that this "may
    significantly increase memory usage" is well earned on a full model run. One
    column is the smallest unit the match can act on, and doing it explicitly
    leaves the *horizontal* chunking — which is what bounds memory here — untouched.

    A numpy-backed array is returned unchanged; ``.chunk()`` on one would make it lazy,
    which is a surprising thing for a rechunk helper to do.
    """
    if da.chunks is None or s_dim not in da.dims:
        return da
    return da.chunk({s_dim: -1})


def _z_grid(ds: xr.Dataset, s_dim: str):
    """Build the xgcm ``Grid`` :func:`to_sigma0` transforms on.

    Kept apart from the transform itself so the grid construction (and its xgcm-version
    fallbacks) lives in one place. :func:`to_depth` used to share it; it now matches
    depths with its own kernel (:func:`_match_columns`), which needs no grid.
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

    :func:`to_sigma0` rebuilds its result with a fixed coordinate whitelist
    (lon/lat/sigma0/cell_area); left alone, that silently drops
    ``operators.aggregate``'s ``spread`` coordinate (a mean+std envelope) on a model
    lane, while an observational lane -- which never goes through this transform --
    keeps its own. Transformed the same way each data variable is (same grid, same
    target_data, same linear method), so the band lands on the requested isopycnals
    instead of disappearing. Returns ``None`` when there is no spread to carry, or its
    dims do not match this transform (nothing to interpolate against). The depth matches
    do the same for their own kernel: :func:`_match_spread`.
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


# -- matching to depths: one kernel, per time step, with an explicit edge policy ------

#: What became of one (sample, target) pair -- see :func:`_locate`. 0-2 carry a value
#: (an interior match, or an edge-fill from the top / bottom half-cell); 3-5 are NaN.
_INTERIOR, _TOP_CELL, _BOTTOM_CELL, _ABOVE_SURFACE, _BELOW_BOTTOM, _MASKED = range(6)


def _gatherer(a: np.ndarray, lead: tuple[int, ...]):
    """Return ``take(idx)``: ``a[..., idx]`` for one index per column (``idx`` is lead).

    A flat ``take`` at row offsets, several times faster than ``np.take_along_axis``'s
    multi-dimensional fancy indexing (the gather is what dominates the kernel). ``a``
    is ``(..., N)`` whose leading axes are each ``lead``'s size or 1 -- they broadcast
    by offsetting the *rows* against ``lead``, so ``a`` itself is never copied up to
    the broadcast shape (a static field under a time-varying frame stays its own size).
    """
    n = a.shape[-1]
    flat = np.ascontiguousarray(a).reshape(-1)
    rows = (np.arange(flat.size // n) * n).reshape(a.shape[:-1])
    base = np.broadcast_to(rows, lead)
    return lambda idx: flat.take(base + idx)


def _locate(frame, top, bottom, masked, t):
    """Say where the frame-coordinate target ``t`` falls in every column.

    ``frame`` is ``(..., N)`` and ascending (``s_rho`` runs bottom -> top), ``top`` and
    ``bottom`` the water column's bounds in the same frame, ``(...)``, and ``masked``
    which columns are all-NaN (land) -- the same for every target, so the caller finds
    it once rather than once per target. Returns ``(idx, code)``: ``idx`` is how many
    cell centres lie at or below ``t`` (0..N), which brackets it between ``idx - 1``
    and ``idx``; ``code`` is the edge-policy outcome (the ``_INTERIOR`` .. ``_MASKED``
    constants, see :func:`to_depth`).

    Later assignments win, and that order *is* the priority the policy needs: a masked
    column is masked whatever else is true; above the surface and below the seafloor
    are NaN even where ``t`` is also past the outermost *centre*; only a target still
    inside the water is a half-cell edge-fill. A target exactly on the top (bottom)
    centre is interior, not an edge.
    """
    n = frame.shape[-1]
    idx = np.asarray((frame <= t).sum(axis=-1))
    code = np.zeros(idx.shape, dtype=np.int8)
    code[(idx == n) & (t > frame[..., -1])] = _TOP_CELL
    code[idx == 0] = _BOTTOM_CELL
    code[t < bottom] = _BELOW_BOTTOM
    code[t > top] = _ABOVE_SURFACE
    code[masked] = _MASKED
    return idx, code


def _match_columns(values, frame, top, bottom, *, targets, mode, dtype):
    """Match every column of ``values`` to every target, per sample -- the one kernel.

    ``values`` and ``frame`` are ``(..., N)`` (vertical last, bottom -> top, so
    ``frame`` ascends), ``top`` and ``bottom`` ``(...)``. The leading dimensions only
    need to *broadcast* against each other: a field with no time axis against a frame
    that has one, say, or a static frame under a time series. Returns
    ``(..., len(targets))``.

    Each target is its own pass over the columns: the same work as one big comparison,
    but the temporaries stay the size of one field instead of ``(..., Z, N)`` -- for a
    profile of a few hundred depths over a model year, the difference between fitting
    in memory and not. ``idx`` (:func:`_locate`) brackets the target between two
    centres; clipping those two to the column is what makes the edges work -- beyond
    the top centre both are the top cell, beyond the bottom one both are the bottom
    cell, so the "interpolation" is that cell's own value -- and the code from
    :func:`_locate` then decides whether such an edge-fill stands or the sample is NaN
    (above the surface, below the seafloor, over land). ``mode="interp"`` is linear
    between the bracketing centres; ``"nearest"`` takes the closer of the two (a tie
    goes to the deeper, as ``argmin`` does).
    """
    n = frame.shape[-1]
    # Everything about *where* a target falls depends on the frame alone, so it is
    # worked out at the frame's own shape -- a static frame under a long time series is
    # located once, not once per step -- and only the data gathers run at the full one.
    where_lead = np.broadcast_shapes(frame.shape[:-1], top.shape, bottom.shape)
    lead = np.broadcast_shapes(where_lead, values.shape[:-1])
    take_frame, take_value = _gatherer(frame, where_lead), _gatherer(values, lead)
    frame = np.broadcast_to(frame, (*where_lead, n))
    top = np.broadcast_to(top, where_lead)
    bottom = np.broadcast_to(bottom, where_lead)
    masked = np.isnan(frame).all(axis=-1)

    out = np.empty((*lead, len(targets)), dtype=dtype)
    for j, t in enumerate(targets):
        idx, code = _locate(frame, top, bottom, masked, t)
        lo = np.clip(idx - 1, 0, n - 1)
        hi = np.minimum(idx, n - 1)
        f_lo = take_frame(lo)
        f_hi = take_frame(hi)
        if mode == "nearest":
            pick = np.where(np.abs(f_hi - t) < np.abs(f_lo - t), hi, lo)
            column = take_value(np.broadcast_to(pick, lead))
        else:
            v_lo = take_value(np.broadcast_to(lo, lead))
            v_hi = take_value(np.broadcast_to(hi, lead))
            span = f_hi - f_lo
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = np.where(span != 0, (t - f_lo) / span, 0.0)
            # ``weight == 0`` is an exact hit on a centre (or an edge-fill): that
            # centre's own value, even where its neighbour is NaN (``0 * NaN`` is NaN).
            column = np.where(weight == 0, v_lo, v_lo + weight * (v_hi - v_lo))
        out[..., j] = np.where(code >= _ABOVE_SURFACE, np.nan, column)
    return out


def _classify_columns(frame, top, bottom, *, targets):
    """Return the edge-policy code of every (sample, target): the kernel minus the data.

    The same :func:`_locate` the data path runs, on the frame alone, so what the counts
    (and the unreachable-target warning) report is exactly what happened to the data --
    without ever reading any. ``(..., N)`` frame in, ``(..., len(targets))`` int8 out.
    """
    lead = np.broadcast_shapes(frame.shape[:-1], top.shape, bottom.shape)
    frame = np.broadcast_to(frame, (*lead, frame.shape[-1]))
    top = np.broadcast_to(top, lead)
    bottom = np.broadcast_to(bottom, lead)
    masked = np.isnan(frame).all(axis=-1)
    codes = np.empty((*lead, len(targets)), dtype=np.int8)
    for j, t in enumerate(targets):
        codes[..., j] = _locate(frame, top, bottom, masked, t)[1]
    return codes


def _match_dtype(values_dtype, frame_dtype, mode: str):
    """Return the dtype a match produces: the data's own, or float where it must be.

    A nearest match returns real model values untouched, so a float32 field stays
    float32 (a NaN needs a float, so anything else becomes float64). An interpolation
    blends in the frame's precision, so it is at least that.
    """
    if not np.issubdtype(values_dtype, np.floating):
        return np.dtype(np.float64)
    if mode == "nearest":
        return np.dtype(values_dtype)
    return np.result_type(values_dtype, frame_dtype)


def _match_variable(da, frame, top, bottom, s_dim: str, targets, mode: str):
    """Run :func:`_match_columns` over one variable, lazily (dask in -> dask out).

    The vertical is an ``apply_ufunc`` *core* dimension, so it is made one dask chunk
    first (:func:`_contiguous_column`); the horizontal and time chunking, which is what
    bounds memory, is left alone. The result has ``z`` last; any dimension the frame
    has and the variable lacks (a time axis, for a field that does not vary in time) is
    put first, ahead of the variable's own. Coordinates other than the dimension ones
    are left off -- the caller attaches the ones it wants.
    """
    values = _contiguous_column(da.reset_coords(drop=True), s_dim)
    dtype = _match_dtype(da.dtype, frame.dtype, mode)
    out = xr.apply_ufunc(
        _match_columns,
        values,
        frame,
        top,
        bottom,
        input_core_dims=[[s_dim], [s_dim], [], []],
        output_core_dims=[["z"]],
        dask="parallelized",
        output_dtypes=[dtype],
        dask_gufunc_kwargs={"output_sizes": {"z": len(targets)}},
        kwargs={"targets": targets, "mode": mode, "dtype": dtype},
    )
    extra = [d for d in out.dims if d != "z" and d not in da.dims]
    own = [d for d in da.dims if d != s_dim]
    return out.transpose(*extra, *own, "z")


def _match_spread(ds, s_dim: str, h_dims, frame, top, bottom, targets, mode: str):
    """Carry a riding ``spread`` coordinate onto the match's target depths.

    Both depth matches rebuild their result with a fixed coordinate whitelist
    (lon/lat/z/cell_area); left alone, that silently drops
    :func:`ocean_skill.operators.aggregate`'s ``spread`` coordinate (a mean+std
    envelope) on a model lane, while an observational lane -- which never goes through
    this match -- keeps its own. Matched the way each data variable is (same frame,
    same kernel, same ``mode``), so the band lands on the requested depths and, for a
    nearest match, on the very level the data was read from. Returns ``None`` when
    there is no spread to carry, or its dims do not match this match (nothing to match
    it against).
    """
    from ocean_skill.operators import SPREAD_COORD

    if SPREAD_COORD not in ds.coords:
        return None
    spread = ds[SPREAD_COORD]
    if s_dim not in spread.dims or not (h_dims <= set(spread.dims)):
        return None
    matched = _match_variable(spread, frame, top, bottom, s_dim, targets, mode)
    matched.attrs = dict(spread.attrs)
    return matched


def _at_ref_time(da: xr.DataArray, meta: dict[str, Any], ref_time: Any) -> xr.DataArray:
    """Return ``da`` at the model time nearest ``ref_time``, loaded (one small instant).

    Eager on purpose: it is one time step of the frame (a single water column or a
    point-cropped window), read once, so the match built from it is a plain array
    rather than a lazy graph every field would be pushed through. The scalar ``time``
    coordinate it leaves behind would conflict with the real, multi-step ``time`` of
    the data, so it is dropped.
    """
    tdim = meta.get("time_dim", "time")
    if tdim not in da.dims:
        return da
    at = da.sel({tdim: ref_time}, method="nearest").load()
    return at.drop_vars(tdim, errors="ignore")


def _frame_codes(frame, top, bottom, s_dim: str, targets) -> xr.DataArray:
    """Return the edge-policy code of every (sample, target), lazy when dask-backed."""
    return xr.apply_ufunc(
        _classify_columns,
        frame,
        top,
        bottom,
        input_core_dims=[[s_dim], [], []],
        output_core_dims=[["z"]],
        dask="parallelized",
        output_dtypes=[np.int8],
        dask_gufunc_kwargs={"output_sizes": {"z": len(targets)}},
        kwargs={"targets": targets},
    )


def _tally_codes(codes: xr.DataArray) -> tuple[list[int], np.ndarray]:
    """Count each outcome over every (sample, target); say which targets get a value.

    Returns ``(counts, reached)``: ``counts[code]`` for each of the six codes, and a
    boolean per target -- ``True`` if any sample got a value for it (interior or
    edge-fill). One ``compute`` for all seven reductions, so a dask-backed frame is
    evaluated once, chunk by chunk, not held whole in memory.
    """
    other = [dim for dim in codes.dims if dim != "z"]
    got_value = codes <= _BOTTOM_CELL
    tally = xr.Dataset(
        {
            **{f"n{code}": (codes == code).sum() for code in range(6)},
            "reached": got_value.any(dim=other) if other else got_value,
        }
    ).compute()
    return [int(tally[f"n{code}"]) for code in range(6)], np.asarray(tally["reached"])


def _surface_range(ds: xr.Dataset, h_dims) -> float:
    """Return the largest per-column range of the free surface over time, in metres.

    ``max - min`` of ``zeta`` over every non-horizontal dimension (the time axis), per
    water column, then the widest column. *Per column*: a full-domain lane's spatial
    differences in mean sea level are not "the surface moves", and must not count. A
    ``zeta`` with no time axis (absent, or already reduced) does not move: 0. Eager --
    ``zeta`` is one field, a fraction of the ``(time, s_rho, ...)`` frame.
    """
    zeta = _zeta_of(ds)
    over = [dim for dim in zeta.dims if dim not in h_dims]
    if not over:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # an all-land column is all-NaN
        per_column = zeta.max(over, skipna=True) - zeta.min(over, skipna=True)
        widest = float(per_column.max(skipna=True))
    return widest if np.isfinite(widest) else 0.0


def _warn_depth_match(
    depths,
    counts,
    reached,
    *,
    mode: str,
    origin: str,
    datum_z_m: float,
    source: str,
    tide_range,
) -> None:
    """Emit the (up to three) warnings a depth match owes its caller, once each.

    1. *No sample reaches a target at all*: it is entirely NaN -- outside the water at
       every time step and column (above the free surface, below the seafloor, or only
       land). One line for the whole call, naming the span, not one per target: a
       profile's own depths can drive dozens of below-the-bottom targets (a cast
       reaching past the model's deepest cell is routine) and one line each buries the
       signal.
    2. *Fixed origin, and some samples are above the free surface*: how many, and the
       likeliest cause -- a depth measured below the surface (a CTD, a profiler) being
       matched as if fixed in space, which the surface drops out from under at low
       tide.
    3. *Fixed origin by default, over a surface that moves a lot*: nothing declared how
       this depth is measured, so it was matched fixed in space, which is wrong by up
       to the tidal range if it was in fact measured below the surface. Silenced the
       moment the source declares (or infers) its convention -- a declared fixed origin
       is a decision, not a default. ``tide_range`` is a callable, evaluated only here
       (it reads ``zeta``), so a declared lane never pays for it.

    Edge-fills do not warn; the counts on the result carry them. ``stacklevel`` reaches
    the caller of :func:`to_depth`/:func:`nearest_depth_levels`.
    """
    stacklevel = 4  # here -> _match_depths -> to_depth / nearest_depth_levels -> caller
    verb = "interpolated" if mode == "interp" else "snapped to"
    nan_depths = [float(dd) for i, dd in enumerate(depths) if not bool(reached[i])]
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
            f"{where} entirely NaN: the target lies outside the water column (above "
            "the free surface, below the seafloor, or over land) at every sample, so "
            f"nothing can be {verb};{hint}",
            stacklevel=stacklevel,
        )
    if origin != "fixed":
        return
    above = counts[_ABOVE_SURFACE]
    if above:
        live = sum(counts) - counts[_MASKED]
        warnings.warn(
            f"{above} of {live} (sample, target) matches lie above the free surface "
            "and are NaN: depth was matched at positions fixed in space "
            f"(z = {datum_z_m:g} - depth), and the surface drops below the target at "
            "those samples. If the observation measures depth below the instantaneous "
            "surface (a CTD, a profiler, a pressure sensor), declare "
            "`depth_convention: {origin: surface}` for it (or pass "
            '`depth_origin="surface"`) so the target follows the surface.',
            stacklevel=stacklevel,
        )
    if source == "default" and len(depths):
        threshold = max(0.5, 0.1 * float(np.min(depths)))
        moves = tide_range()
        if moves > threshold:
            warnings.warn(
                f"the free surface moves {moves:.2g} m over time at some water column "
                f"(more than {threshold:g} m), but the observation declares no "
                "`depth_convention`, so its depth is matched at positions fixed in "
                "space (origin: fixed), not below the moving surface. If it is "
                "measured below the surface, declare `depth_convention: {origin: "
                'surface}` (or pass `depth_origin="surface"`); if it really is fixed '
                "in space, declare `depth_convention: {origin: fixed}` to say so.",
                stacklevel=stacklevel,
            )


def _match_depths(ds, meta, d, *, mode: str, convention, ref_time=None) -> xr.Dataset:
    """Match ``ds`` to depths ``d``: the shared body of the two public depth matches.

    :func:`to_depth` is ``mode="interp"`` and :func:`nearest_depth_levels`
    ``"nearest"``; everything else -- the frame, the kernel, the coordinates, the
    counts and the warnings -- is the same, which is what keeps the two
    interchangeable.
    """
    origin, datum_z_m, source = _frame_spec(convention)
    if "z_rho" not in ds.coords:
        ds = add_depth_coord(ds, meta)
    s_dim = meta.get("vertical", {}).get("s_dim", "s_rho")
    depths = np.atleast_1d(np.asarray(d, dtype=float))
    targets = -depths  # in the frame a depth d is the coordinate -d, either origin

    frame = frame_coordinate(ds, meta, origin, datum_z_m=datum_z_m)
    top, bottom = water_column_bounds(ds, meta, origin, datum_z_m=datum_z_m)
    frame, top, bottom = (a.reset_coords(drop=True) for a in (frame, top, bottom))
    if ref_time is not None:
        # The old static lookup: the frame as it was at one reference time, applied to
        # every step. Because the frame then has no time axis, the very same kernel
        # picks one level per column and the data follows it through time.
        frame, top, bottom = (
            _at_ref_time(a, meta, ref_time) for a in (frame, top, bottom)
        )
    frame = _contiguous_column(frame, s_dim)

    # Rho-point fields share z_rho's own horizontal dims; staggered u/v (xi_u/eta_v)
    # need interpolation to rho first (deferred), so are skipped here. Read off lon's
    # own dims rather than the hardcoded ("eta_rho", "xi_rho") pair, so a variable
    # already sliced to one grid column or transect (see
    # ocean_skill.transect.grid_slice, whose renamed "along" dim replaces one of the
    # two) still matches -- it shares lon's dims exactly, since lon went through the
    # same slice. `time` is deliberately excluded: z_rho can carry it (when zeta is
    # present) while a time-invariant variable legitimately does not, and that
    # variable must not be skipped just because it lacks an axis z_rho happens to
    # have.
    h_dims = set(ds["lon"].dims) if "lon" in ds.coords else {"eta_rho", "xi_rho"}
    out = {}
    for var in ds.data_vars:
        da = ds[var]
        if s_dim in da.dims and h_dims <= set(da.dims):
            matched = _match_variable(da, frame, top, bottom, s_dim, targets, mode)
            # the match sheds attrs; carry the source variable's forward explicitly
            matched.attrs = dict(da.attrs)
            out[var] = matched

    z_attrs = {"positive": "up", "units": "m", "depth_origin": origin}
    if origin == "fixed" and datum_z_m:
        z_attrs["depth_datum_z_m"] = datum_z_m
    coords = {"lon": ds["lon"], "lat": ds["lat"], "z": ("z", targets, z_attrs)}
    if AREA_COORD in ds.coords:
        coords[AREA_COORD] = ds[AREA_COORD]
    spread = _match_spread(ds, s_dim, h_dims, frame, top, bottom, targets, mode)
    if spread is not None:
        from ocean_skill.operators import SPREAD_COORD

        coords[SPREAD_COORD] = spread
    result = xr.Dataset(out, coords=coords)
    result.attrs.update(ds.attrs)

    # What the edge policy did, counted from the frame and the water-column bounds
    # alone -- the same cost as asking "is any column deep enough?" (the frame is a
    # function of the small zeta field), and never a pass over the data, so loading
    # the result later does not change them. Reachability is a property of the
    # geometry, not of any one variable's data: a variable that is NaN everywhere
    # within a reachable depth (masked out for a reason other than geometry) does not
    # warn here.
    counts, reached = _tally_codes(_frame_codes(frame, top, bottom, s_dim, targets))
    stamp = {
        "depth_edge_top": counts[_TOP_CELL],
        "depth_edge_bottom": counts[_BOTTOM_CELL],
        "depth_above_surface": counts[_ABOVE_SURFACE],
        "depth_below_bottom": counts[_BELOW_BOTTOM],
        "depth_origin": origin,
    }
    result.attrs.update(stamp)
    for name in result.data_vars:
        result[name].attrs.update(stamp)

    _warn_depth_match(
        depths,
        counts,
        reached,
        mode=mode,
        origin=origin,
        datum_z_m=datum_z_m,
        source=source,
        tide_range=lambda: _surface_range(ds, h_dims),
    )
    return result


def to_depth(
    ds: xr.Dataset,
    meta: dict[str, Any],
    d: float | list[float],
    *,
    convention: Mapping[str, Any] | None = None,
) -> xr.Dataset:
    """Interpolate s-coordinate fields to depth(s) ``d`` (metres, positive down).

    Linear between the two ``s_rho`` cell centres that bracket the target, **per time
    step, in the** ``convention``'s frame (:func:`frame_coordinate`): ``origin: fixed``
    (the default) reads ``d`` as the height ``z = datum_z_m - d``, a position fixed in
    space; ``origin: surface`` as ``z = zeta - d``, ``d`` metres below the free surface
    *as it is at that step*. ``convention`` is a plain mapping -- ``origin``,
    ``datum_z_m`` and ``source`` are the keys read -- or ``None`` for fixed at datum 0
    (what a caller that never heard of the frame meant). Keeps the result lazy: a dask
    input gives dask output. ``d`` may be a scalar or a list. For a true surface field
    use :func:`surface`; for a surface of constant potential density rather than
    constant depth, see :func:`to_sigma0`. Non-``s_rho`` variables drop.

    **Edge policy** -- per column, per time step, per target ``t = -d``, with ``f`` the
    cell-centre frame values (bottom -> top) and ``top``/``bottom`` the water column's
    bounds (:func:`water_column_bounds`):

    ==============================  ==================================================
    where ``t`` is                  result
    ==============================  ==================================================
    ``f[0] <= t <= f[-1]``          interpolated between the bracketing centres
    ``f[-1] < t <= top``            the top cell's value (top half-cell)
    ``bottom <= t < f[0]``          the bottom cell's value (bottom half-cell)
    ``t > top``                     NaN: above the free surface
    ``t < bottom``                  NaN: below the seafloor
    column all-NaN (land, masked)   NaN
    ==============================  ==================================================

    The half-cells are filled, not left NaN, because the water *is* there: the
    outermost centres sit half a cell inside the column (7 m down in deep water on a
    stretched grid), and a target in that half-cell is in the water -- the model's one
    statement about it is that cell's value, which is also what :func:`surface`
    returns for the surface. (Before the frame existed this was NaN, which made
    ``d=0`` and every shallow target in deep water an empty field.) Only a target
    outside the water itself is NaN.

    The result keeps ``(..., z)`` dims with ``z = -d`` (metres, negative up, as
    ``z_rho``), ``lon``/``lat`` and any ``cell_area``/``spread`` coordinates. ``z``
    carries ``positive="up"``, ``units="m"``, ``depth_origin`` (and
    ``depth_datum_z_m`` for a fixed origin with a non-zero datum). The result and each
    variable carry integer counts of what the edge policy did, over every (sample,
    target) pair (a sample is a column at a time step -- the frame's own, so a frame
    with no time axis counts each column once): ``depth_edge_top`` /
    ``depth_edge_bottom`` (edge-filled) and ``depth_above_surface`` /
    ``depth_below_bottom`` (NaN for lying outside the water), plus ``depth_origin``.
    Counted from the frame and bounds alone, never from the data.

    Warns (once each per call): a target that no sample reaches at all is *entirely
    NaN*; for a fixed origin, how many samples sit above the free surface (and that a
    depth measured below the surface should declare ``origin: surface``); and, when
    nothing declared the origin (``convention`` source ``"default"``), a free surface
    that moves more than ``max(0.5, 0.1 * min(d))`` m at some column over time -- so a
    depth quietly matched fixed in space under a big tide does not go unremarked.
    """
    return _match_depths(ds, meta, d, mode="interp", convention=convention)


def nearest_depth_levels(
    ds: xr.Dataset,
    meta: dict[str, Any],
    d: float | list[float],
    *,
    ref_time: Any = None,
    convention: Mapping[str, Any] | None = None,
) -> xr.Dataset:
    """Snap target depth(s) ``d`` to the nearest native model level -- no interpolation.

    The nearest-level counterpart of :func:`to_depth`: rather than linearly blending
    two levels together, this looks up the closest ``s_rho`` cell centre (per column)
    and reads its value directly, so what comes back is real model output, never an
    average of two. Keeps the result lazy exactly as :func:`to_depth` does; ``d`` may
    be a scalar or a list; the frame (``convention``), the edge policy, the counts, the
    warnings and the shape and coordinate conventions of the result (a ``z`` axis in
    metres, stored negative-up to match ``z_rho``) are all :func:`to_depth`'s, so the
    two are interchangeable to a caller. Non-``s_rho`` variables drop, exactly as there.

    The closest level is chosen **at every time step**, against the frame of that
    step: a level's true depth moves with the free surface, so the level nearest 1 m
    down at high tide is not the one nearest it at low tide, and a static lookup would
    hand the same cell to every step (and report a value where the water is no longer
    deep enough). A tie between two levels goes to the deeper one. The frame is a
    function of the small ``zeta`` field, so matching it per step costs little next to
    reading data.

    ``ref_time`` restores the old, static behaviour for a caller that wants it: the
    frame -- and the water-column bounds -- is taken at the model's own time nearest
    ``ref_time`` and the level chosen there is applied to **every** step (a field that
    stays on one cell through the record). The edge policy still applies, against that
    one reference frame.
    """
    return _match_depths(
        ds, meta, d, mode="nearest", convention=convention, ref_time=ref_time
    )


def to_sigma0(
    ds: xr.Dataset, meta: dict[str, Any], s: float | list[float]
) -> xr.Dataset:
    """Interpolate s-coordinate fields onto surface(s) of constant potential density.

    An isopycnal slice: the xgcm vertical transform :func:`to_depth` itself used before
    it got its own kernel, but against potential density anomaly (sigma0, TEOS-10 via
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
    naming the target -- the same shape :func:`to_depth` uses for a target that no
    sample reaches. (A density has no half-cell to fall back on: beyond the column's
    own sigma0 range there is no level to read, so there it stays NaN.)
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
