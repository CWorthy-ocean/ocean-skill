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
    "WEIGHT_COORD",
    "add_depth_coord",
    "add_interface_coord",
    "depth_average",
    "depth_band",
    "standardize",
    "surface",
    "to_depth",
]


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

    # rename model variable names -> CF standard_names
    rename = {
        k: v for k, v in (meta.get("standard_names") or {}).items() if k in ds.variables
    }
    ds = ds.rename(rename)

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


def add_depth_coord(ds: xr.Dataset, meta: dict[str, Any]) -> xr.Dataset:
    """Attach the (lazy) ROMS z_rho depth coordinate via Vtransform 2.

    ``z_rho = zeta + (zeta + h) * (hc*sigma_r + h*Cs_r) / (hc + h)`` (metres, negative
    down). Requires ``h``/``Cs_r``/``sigma_r`` (from the grid) and ``zeta``.
    """
    vert = meta.get("vertical", {})
    hc = float(vert.get("hc"))
    if "zeta" in ds.variables:
        zeta = ds["zeta"]
    elif "sea_surface_height_above_geoid" in ds.variables:
        zeta = ds["sea_surface_height_above_geoid"]
    else:  # no free-surface field: use zeta = 0
        zeta = xr.zeros_like(ds["h"])
    s = (hc * ds["sigma_r"] + ds["h"] * ds["Cs_r"]) / (hc + ds["h"])
    z_rho = zeta + (zeta + ds["h"]) * s
    dims = ("time", "s_rho", "eta_rho", "xi_rho")
    z_rho = z_rho.transpose(*[d for d in dims if d in z_rho.dims])
    return ds.assign_coords(z_rho=z_rho)


def add_interface_coord(ds: xr.Dataset, meta: dict[str, Any]) -> xr.Dataset:
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
    """
    vert = meta.get("vertical", {})
    hc = float(vert.get("hc"))
    if "zeta" in ds.variables:
        zeta = ds["zeta"]
    elif "sea_surface_height_above_geoid" in ds.variables:
        zeta = ds["sea_surface_height_above_geoid"]
    else:
        zeta = xr.zeros_like(ds["h"])
    s = (hc * ds["sigma_w"] + ds["h"] * ds["Cs_w"]) / (hc + ds["h"])
    z_w = zeta + (zeta + ds["h"]) * s
    dims = ("time", "s_w", "eta_rho", "xi_rho")
    z_w = z_w.transpose(*[d for d in dims if d in z_w.dims])
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
    for coord in ("lon", "lat", "lon_rho", "lat_rho", "mask_rho", "h"):
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


def to_depth(
    ds: xr.Dataset, meta: dict[str, Any], d: float | list[float]
) -> xr.Dataset:
    """Interpolate s-coordinate fields to fixed depth(s) ``d`` (metres, positive down).

    Uses xgcm's vertical transform against ``z_rho`` (linear; NaN outside the water
    column — no extrapolation). Keeps the result lazy. ``d`` may be a scalar or a list.
    For a true surface field use :func:`surface` instead. Non-``s_rho`` variables drop.
    """
    import xgcm

    if "z_rho" not in ds.coords:
        ds = add_depth_coord(ds, meta)
    s_dim = meta.get("vertical", {}).get("s_dim", "s_rho")
    depths = np.atleast_1d(np.asarray(d, dtype=float))
    targets = xr.DataArray(-depths, dims="z", coords={"z": -depths})

    try:
        grid = xgcm.Grid(
            ds,
            coords={"Z": {"center": s_dim}},
            periodic=False,
            autoparse_metadata=False,
        )
    except TypeError:  # older xgcm without autoparse_metadata kwarg
        grid = xgcm.Grid(ds, coords={"Z": {"center": s_dim}}, periodic=False)

    out = {}
    for var in ds.data_vars:
        da = ds[var]
        # only rho-point 3-D fields share z_rho's grid; staggered u/v (xi_u/eta_v) need
        # interpolation to rho first (deferred), so skip them here.
        if s_dim in da.dims and {"eta_rho", "xi_rho"} <= set(da.dims):
            out[var] = grid.transform(
                da, "Z", targets, target_data=ds["z_rho"], method="linear"
            )
    result = xr.Dataset(out, coords={"lon": ds["lon"], "lat": ds["lat"], "z": -depths})
    result.attrs.update(ds.attrs)

    # A target shallower than the topmost cell centre (or deeper than the bottom one)
    # interpolates to nothing and silently yields an all-NaN level — most often when
    # asking for exactly 0 m. Say so, and point at surface() for the surface case.
    for var in result.data_vars:
        for i, d in enumerate(depths):
            if not bool(np.isfinite(result[var].isel(z=i)).any()):
                hint = " use surface() for the surface field" if d < 5 else ""
                warnings.warn(
                    f"{var!r} at {d:g} m is entirely NaN: the target lies outside the "
                    f"model's cell-centre range, so nothing can be interpolated;{hint}",
                    stacklevel=2,
                )
    return result
