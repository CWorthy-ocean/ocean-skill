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
    "to_sigma0",
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
        return xgcm.Grid(
            ds,
            coords={"Z": {"center": s_dim}},
            periodic=False,
            autoparse_metadata=False,
        )
    except TypeError:  # older xgcm without autoparse_metadata kwarg
        return xgcm.Grid(ds, coords={"Z": {"center": s_dim}}, periodic=False)


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
    out = {}
    for var in ds.data_vars:
        da = ds[var]
        # only rho-point 3-D fields share z_rho's grid; staggered u/v (xi_u/eta_v) need
        # interpolation to rho first (deferred), so skip them here.
        if s_dim in da.dims and {"eta_rho", "xi_rho"} <= set(da.dims):
            out[var] = grid.transform(
                _contiguous_column(da, s_dim),
                "Z",
                targets,
                target_data=z_rho,
                method="linear",
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

    out = {}
    for var in ds.data_vars:
        da = ds[var]
        # only rho-point 3-D fields share sigma0's grid; staggered u/v need
        # interpolation to rho first (deferred, as in to_depth), so skip them here.
        if s_dim in da.dims and {"eta_rho", "xi_rho"} <= set(da.dims):
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
    result = xr.Dataset(
        out, coords={"lon": ds["lon"], "lat": ds["lat"], "sigma0": values}
    )
    result.attrs.update(ds.attrs)
    result["sigma0"].attrs = {
        "units": "kg m-3",
        "long_name": "potential density anomaly (sigma0, TEOS-10)",
        "standard_name": "sea_water_sigma_theta",
    }

    # A target denser or lighter than the column holds anywhere interpolates to
    # nothing and silently yields an all-NaN level. Say so, per level.
    for var in result.data_vars:
        for i, target in enumerate(values):
            if not bool(np.isfinite(result[var].isel(sigma0=i)).any()):
                warnings.warn(
                    f"{var!r} at sigma0={target:g} kg/m3 is entirely NaN: the "
                    "target density lies outside this water column's sigma0 range "
                    "everywhere, so nothing can be interpolated.",
                    stacklevel=2,
                )
    return result
