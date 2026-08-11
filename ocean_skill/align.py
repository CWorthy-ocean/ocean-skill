"""Alignment: bring the ``test`` entity onto the ``reference`` grid/coordinates.

Direction is set by role (not argument order): test → reference (⇒ model→data by
default), with an ``onto`` override. Longitude conventions are harmonized first —
a 0-360 model vs a ±180 observational grid otherwise produces silently empty overlap.
The reference is subset to the test's bounding box so we regrid onto the overlap
rather than a whole globe. The result is **always xarray**, so one metrics engine
serves both gridded and point comparisons.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import xarray as xr

from ocean_skill import _stacklevel

__all__ = ["align", "grid_of", "harmonize_longitude", "subset_to_bbox"]


def _lon_name(obj) -> str | None:
    """Name of the longitude coordinate, preferring canonical then ROMS names."""
    for nm in ("lon", "longitude", "lon_rho"):
        if nm in obj.coords or nm in getattr(obj, "variables", {}):
            return nm
    return None


def _lat_name(obj) -> str | None:
    """Name of the latitude coordinate, preferring canonical then ROMS names."""
    for nm in ("lat", "latitude", "lat_rho"):
        if nm in obj.coords or nm in getattr(obj, "variables", {}):
            return nm
    return None


def harmonize_longitude(obj, convention: Literal["0-360", "-180-180"] = "-180-180"):
    """Return ``obj`` with longitudes in ``convention``.

    Mismatched conventions (a 0-360 model against a ±180 climatology) are a silent
    correctness hazard: regridding "succeeds" and yields all-NaN or garbage. A
    longitude *dimension* coordinate is re-sorted so it stays monotonic.
    """
    lon = _lon_name(obj)
    if lon is None:
        return obj
    vals = obj[lon]
    if convention == "-180-180":
        new = ((vals + 180) % 360) - 180
    else:
        new = vals % 360
    out = obj.assign_coords({lon: new})
    if lon in out.dims:  # 1-D dimension coordinate must stay sorted
        out = out.sortby(lon)
    return out


def _oriented_slice(obj, dim: str, low: float, high: float) -> slice:
    """Return a slice ordered to match ``dim``'s own direction.

    ``.sel`` with a slice follows the coordinate's stored order, so ``slice(16, 32)``
    against a **descending** axis selects nothing at all — silently, returning an
    empty array rather than raising. Satellite L3 products are stored north-to-south
    (MODIS runs 89.979 to -89.979), so this is the common case, not an exotic one:
    it turned every model-vs-MODIS comparison into an ``IndexError`` deep in the
    corner-derivation code, with nothing pointing at latitude ordering.
    """
    values = np.asarray(obj[dim])
    descending = values.size > 1 and values[0] > values[-1]
    return slice(high, low) if descending else slice(low, high)


def subset_to_bbox(obj, bbox, pad: float = 1.0):
    """Subset ``obj`` to ``bbox`` (lon_min, lat_min, lon_max, lat_max) plus ``pad``.

    Honours each axis's stored direction (see :func:`_oriented_slice`), and refuses
    to return an empty result: no overlap at all means the two sources do not cover
    the same region, which is worth saying plainly rather than failing later.
    """
    lon, lat = _lon_name(obj), _lat_name(obj)
    if lon is None or lat is None:
        return obj
    lon_min, lat_min, lon_max, lat_max = bbox
    sel = {}
    if lon in obj.dims:
        sel[lon] = _oriented_slice(obj, lon, lon_min - pad, lon_max + pad)
    if lat in obj.dims:
        sel[lat] = _oriented_slice(obj, lat, lat_min - pad, lat_max + pad)
    if not sel:
        return obj
    out = obj.sel(**sel)
    empty = [d for d in sel if out.sizes.get(d, 1) == 0]
    if empty:
        raise ValueError(
            f"no overlap along {empty}: the reference does not cover the test's "
            f"extent (lon {lon_min:g} to {lon_max:g}, lat {lat_min:g} to "
            f"{lat_max:g}). Check the two sources really overlap in space."
        )
    return out


def bbox_of(obj) -> tuple[float, float, float, float]:
    """Return ``(lon_min, lat_min, lon_max, lat_max)`` of ``obj``."""
    lon, lat = _lon_name(obj), _lat_name(obj)
    lo, la = np.asarray(obj[lon]), np.asarray(obj[lat])
    return (
        float(np.nanmin(lo)),
        float(np.nanmin(la)),
        float(np.nanmax(lo)),
        float(np.nanmax(la)),
    )


def _as_xesmf(obj):
    """Return ``obj`` with coords named ``lon``/``lat`` as xesmf expects."""
    lon, lat = _lon_name(obj), _lat_name(obj)
    ren = {}
    if lon and lon != "lon":
        ren[lon] = "lon"
    if lat and lat != "lat":
        ren[lat] = "lat"
    return obj.rename(ren) if ren else obj


def _corners_1d(c: np.ndarray) -> np.ndarray:
    """Cell edges for 1-D centres: midpoints, with the two ends extrapolated."""
    mid = 0.5 * (c[:-1] + c[1:])
    first = c[0] - (mid[0] - c[0])
    last = c[-1] + (c[-1] - mid[-1])
    return np.concatenate([[first], mid, [last]])


def _corners_2d(a: np.ndarray) -> np.ndarray:
    """Cell corners ``(ny+1, nx+1)`` for 2-D curvilinear centres ``(ny, nx)``.

    Pads the centres by one ring using linear extrapolation, then averages each 2x2
    block. ROMS grids ship rho/u/v points but no corner (psi) array covering the full
    boundary, so corners have to be derived like this for conservative regridding.
    """
    ny, nx = a.shape
    p = np.empty((ny + 2, nx + 2), dtype=float)
    p[1:-1, 1:-1] = a
    p[0, 1:-1] = 2 * a[0, :] - a[1, :]
    p[-1, 1:-1] = 2 * a[-1, :] - a[-2, :]
    p[:, 0] = 2 * p[:, 1] - p[:, 2]
    p[:, -1] = 2 * p[:, -2] - p[:, -3]
    return 0.25 * (p[:-1, :-1] + p[:-1, 1:] + p[1:, :-1] + p[1:, 1:])


def grid_of(obj, bounds: bool = False) -> xr.Dataset:
    """Return the grid xesmf needs: ``lon``/``lat`` (+ corners when ``bounds``).

    A separate Dataset is required because cell corners live on their own dimensions,
    which a DataArray cannot carry. Handles 1-D (regular) and 2-D (curvilinear)
    coordinates; ROMS ships no full corner array, so corners are derived.
    """
    lon_da, lat_da = obj["lon"], obj["lat"]
    grid = xr.Dataset({"lon": lon_da, "lat": lat_da})
    if not bounds:
        return grid
    lon, lat = np.asarray(lon_da), np.asarray(lat_da)
    if lon.ndim == 1 and lat.ndim == 1:
        grid = grid.assign_coords(
            lon_b=("lon_b", _corners_1d(lon)), lat_b=("lat_b", _corners_1d(lat))
        )
    elif lon.ndim == 2 and lat.ndim == 2:
        grid = grid.assign_coords(
            lon_b=(("y_b", "x_b"), _corners_2d(lon)),
            lat_b=(("y_b", "x_b"), _corners_2d(lat)),
        )
    else:
        raise ValueError(f"cannot derive bounds for lon/lat with ndim {lon.ndim}")
    return grid


def _require_2d(da, role: str) -> None:
    """Raise a useful error if ``da`` still carries a dimension beyond lat/lon.

    A leftover axis means the selection or aggregation did not collapse it — most
    often ``aggregate={"time": {"groupby": "month", ...}}``, which produces twelve
    fields where a single reference field is expected. Without this, xesmf fails
    much later with a shape mismatch that says nothing about the cause.
    """
    # The horizontal dims are whatever lon/lat are defined *on* — not necessarily
    # named lat/lon: a curvilinear ROMS field is (eta_rho, xi_rho) with 2-D lon/lat
    # coordinates riding on those dims.
    lon, lat = _lon_name(da), _lat_name(da)
    if lon is None or lat is None:
        return  # nothing to measure against; let the regridder complain instead
    spatial = set(da[lon].dims) | set(da[lat].dims)
    extra = [str(d) for d in da.dims if d not in spatial]
    if extra:
        raise ValueError(
            f"the {role} field still has {extra} beyond its horizontal axes, so it "
            "is not a single map. Collapse it with aggregate= (e.g. "
            '{"time": "mean"}) or narrow it with select= (e.g. {"time": "2012-01"}); '
            'a groupby such as {"groupby": "month"} deliberately keeps a dimension '
            "and cannot be compared against a single field."
        )


def align(
    test,
    reference,
    *,
    method: str = "bilinear",
    convention: Literal["0-360", "-180-180"] = "-180-180",
    pad: float = 1.0,
    min_coverage: float = 0.5,
    test_name: str = "test",
    reference_name: str = "reference",
) -> xr.Dataset:
    """Regrid ``test`` onto ``reference``'s grid; return both plus their difference.

    Both inputs should be 2-D (lat/lon) DataArrays — select time/depth beforehand.
    Returns a Dataset with ``test``, ``reference`` and ``difference`` (test − reference)
    on the reference grid.

    ``method="conservative_normed"`` (or ``"conservative"``) **area-averages** the test
    onto the reference cells, which is the right operator when the test is much finer
    than the reference (a km-scale model against a 1-degree climatology): bilinear would
    *sample* the fine field and discard subgrid structure. Cell corners are derived by
    :func:`grid_of` when absent.

    ``min_coverage`` drops reference cells that the test only partly covers. Plain
    ``"conservative"`` divides by the *whole* destination cell area, so a half-covered
    coastal or edge cell reads about half its true value — a large, purely artificial
    difference. ``"conservative_normed"`` renormalizes by the covered fraction, and the
    coverage mask then removes cells too sparsely covered to be meaningful.
    """
    import xesmf as xe

    from ocean_skill import units as _units

    _require_2d(test, "test")
    _require_2d(reference, "reference")

    # Subtracting umol/kg from mmol/m3 used to yield a difference of 0.0, labelled
    # with the reference's units and no warning at all — plausible, and wrong by the
    # density factor. Harmonize first, and refuse outright when the two are not the
    # same physical quantity, since no conversion can rescue that.
    same = _units.compatible(test.attrs.get("units"), reference.attrs.get("units"))
    if same is False:
        raise ValueError(
            f"cannot difference {test.attrs.get('units')!r} against "
            f"{reference.attrs.get('units')!r}: not the same physical quantity. "
            "Convert one first, or check the variables really do match."
        )
    if same:
        test = _units.to_units(test, reference.attrs.get("units"))
    elif same is None:
        warnings.warn(
            f"cannot verify units {test.attrs.get('units')!r} vs "
            f"{reference.attrs.get('units')!r}; differencing them unchecked.",
            stacklevel=_stacklevel.find(),
        )

    test = harmonize_longitude(test, convention)
    reference = harmonize_longitude(reference, convention)

    # regrid onto the overlap, not the reference's full (often global) grid
    reference = subset_to_bbox(reference, bbox_of(test), pad=pad)

    src, tgt = _as_xesmf(test), _as_xesmf(reference)
    need_bounds = method.startswith("conservative")
    regridder = xe.Regridder(
        grid_of(src, need_bounds),
        grid_of(tgt, need_bounds),
        method,
        unmapped_to_nan=True,
    )
    regridded = regridder(src, keep_attrs=True)

    # Coverage = the same regrid applied to a field of ones over the valid test cells.
    coverage = None
    if min_coverage:
        ones = xr.ones_like(src).where(np.isfinite(src))
        coverage = regridder(ones)
        regridded = regridded.where(coverage >= min_coverage)

    out = xr.Dataset(
        {
            test_name: regridded,
            reference_name: tgt,
            "difference": regridded - tgt,
        }
    )
    out["difference"].attrs = {
        "long_name": f"{test_name} − {reference_name}",
        "units": tgt.attrs.get("units", ""),
    }
    if coverage is not None:
        out["coverage"] = coverage
        out["coverage"].attrs = {
            "long_name": "fraction of the reference cell covered by valid test data"
        }
    out.attrs["regrid_method"] = method
    out.attrs["lon_convention"] = convention
    out.attrs["min_coverage"] = min_coverage
    return out
