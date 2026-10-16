"""Water-column calculators: where a profile's extremum sits, and how thick a layer is.

Two registered calculators, each reading a variable's *whole* water column and returning
one number per column (a 2-D map, per time step)::

    {"calculate": "depth_of", "variable": "oxygen", "extremum": "min"}
    {"calculate": "thickness", "variable": "oxygen", "below": 50}

- ``depth_of`` is the depth (m, positive down) of the column's minimum or maximum -- the
  depth of the oxygen minimum, of the chlorophyll maximum. Where the extremum is reached
  at several levels, the shallowest is reported.
- ``thickness`` is the summed thickness (m) of the layer where the variable is strictly
  below (``below=``) or strictly above (``above=``) a threshold, *in the variable's own
  units* -- ``below: 50`` on oxygen in mmol m-3 is the 50 mmol m-3 oxygen minimum zone
  (OMZ). It is the total across the column, not the span between the first and last
  qualifying level: two separate thin layers add up.

Both work on the two shapes of vertical axis this package meets, which is the point of
having one calculator for both rather than ``{"Z": "min"}`` spelled once per product:

- **Native ROMS output** (``z_rho`` attached by :func:`ocean_skill.roms.standardize`):
  the depth of each level differs per column and per time step, so the actual ``z_rho``
  is used, and a layer's thickness is the distance between its own two interfaces
  (``z_w``, built from ``sigma_w``/``Cs_w`` when the run carries them -- exactly what
  :func:`ocean_skill.roms.depth_band` uses). If the run carries no interface variables,
  the interfaces are estimated from the level centres (halfway between neighbours,
  extrapolated at the ends) with a warning -- near-exact on a smoothly stretched grid,
  but an estimate.
- **Depth-level data** (WOA23, GLODAP, a gridded product on standard levels): the
  level depths are the vertical coordinate. A level's thickness is the distance between
  the midpoints to its neighbours; the shallowest level starts at the surface (0 m) and
  the deepest extends half a spacing below its depth, so a level at 1000 m of a 500 m
  spacing stands for 750-1250 m. Because the interfaces come from the *levels*, not from
  where the data happen to stop, a WOA column whose valid levels end at 400 m is
  bounded at 400 m plus half a spacing, not at the bottom of the standard-level list.

A column with no valid data (land) is NaN for both; a column with data in which nothing
qualifies is 0 for ``thickness``.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import xarray as xr

__all__ = ["calculate_depth_of", "calculate_thickness"]

_EXTREMA = {"min": ("minimum", "min"), "max": ("maximum", "max")}


def _resolve(ds, variable: Any, calculator: str):
    """Return the field ``variable`` names in ``ds``, or raise a clear KeyError."""
    from ocean_skill.operators import resolve_variable

    if variable is None:
        raise KeyError(
            f"{calculator} needs variable= (a name or a variable spec), e.g. "
            f"{{'calculate': {calculator!r}, 'variable': 'oxygen', ...}}."
        )
    da = resolve_variable(ds, variable)
    if da is None:
        raise KeyError(
            f"{calculator} needs {variable!r}, which is not in this dataset."
        )
    return da


def _vertical_of(ds, da, s_dim: str):
    """Return ``(zdim, depth, is_roms)`` for ``da``.

    ``zdim`` is its vertical dimension and ``depth`` the depth of every level,
    positive down (m), broadcastable against ``da``.
    """
    from ocean_skill.operators import resolve_dim, vertical_coord_on

    if "z_rho" in ds.coords and s_dim in da.dims:
        return s_dim, -ds.coords["z_rho"], True

    zdim = resolve_dim(da, "Z")
    if zdim is None or zdim not in da.dims:
        raise ValueError(
            f"{da.name!r} has no vertical axis (dimensions {list(da.dims)}), so there "
            "is no water column to reduce."
        )
    coord = vertical_coord_on(da, zdim)
    if coord is None:
        raise ValueError(
            f"the vertical dimension {zdim!r} of {da.name!r} carries no depth values "
            "(it is a bare dimension), so depths cannot be read off it."
        )
    depth = coord.astype(float)
    if str(coord.attrs.get("positive", "down")).lower() == "up":
        depth = -depth
    return zdim, depth, False


def _drop_vertical_coords(obj, zdim: str):
    """Return ``obj`` without coordinates on ``zdim``, so it combines by position."""
    return obj.drop_vars([c for c in obj.coords if zdim in obj[c].dims])


def _thickness_from_centres(depth, zdim: str, *, clip_surface: bool):
    """Return each level's layer thickness from its centre depths (positive down).

    Interfaces sit halfway between neighbouring centres, and the two outermost are the
    end centres pushed out by half their own spacing. ``clip_surface`` pins the
    shallowest interface at 0 m (obs levels start at the sea surface); ``depth`` must
    then be ascending along ``zdim``. A ROMS run's centres can lie above 0 where the
    free surface is up, so it does not clip.
    """
    n = depth.sizes[zdim]
    if n < 2:
        raise ValueError(
            f"a layer thickness needs at least two levels along {zdim!r}, got {n}."
        )
    d = _drop_vertical_coords(depth, zdim)
    shallow, deep = d.isel({zdim: slice(None, -1)}), d.isel({zdim: slice(1, None)})
    mid = (shallow + deep) / 2.0
    first = 2.0 * d.isel({zdim: [0]}) - mid.isel({zdim: [0]})
    last = 2.0 * d.isel({zdim: [-1]}) - mid.isel({zdim: [-1]})
    if clip_surface:
        first = first.clip(min=0.0)
    edges = xr.concat([first, mid, last], zdim)
    out = abs(edges.isel({zdim: slice(1, None)}) - edges.isel({zdim: slice(None, -1)}))
    return out


def _roms_thickness(ds, depth, s_dim: str):
    """Return per-level cell thickness (m) for a ROMS ``ds``, on ``s_dim``.

    From ``z_w`` interface differences when the interfaces exist or can be built
    (``sigma_w``/``Cs_w``/``h`` plus ``hc`` on the dataset); otherwise estimated from
    the level centres, with a warning.
    """
    from ocean_skill import roms

    z_w = ds.coords.get("z_w")
    if z_w is None and all(n in ds.variables for n in ("sigma_w", "Cs_w", "h")):
        try:
            z_w = roms.add_interface_coord(ds, {})["z_w"]
        except ValueError:
            z_w = None  # no hc / Vtransform to build the transform from
    if z_w is not None:
        w_dim = roms._interface_dim(ds, s_dim)
        z_w = _drop_vertical_coords(z_w, w_dim)
        upper = z_w.isel({w_dim: slice(1, None)}).rename({w_dim: s_dim})
        lower = z_w.isel({w_dim: slice(None, -1)}).rename({w_dim: s_dim})
        return upper - lower
    warnings.warn(
        "this ROMS dataset carries no cell-interface depths (z_w, or sigma_w/Cs_w/h "
        "with hc) -- layer thicknesses are estimated from the level centres (halfway "
        "between neighbors), which is close on a smoothly stretched grid but not "
        "exact.",
        stacklevel=3,
    )
    return _thickness_from_centres(depth, s_dim, clip_surface=False)


def _label(variable: Any, da) -> str:
    """Return the short name a result is named after: the spec's, else the field's."""
    if isinstance(variable, str):
        return variable
    return str(da.name or "field")


def _describe(da) -> str:
    return str(da.attrs.get("long_name") or da.name or "field")


def calculate_depth_of(
    ds, *, variable=None, extremum: str | None = None, s_dim: str = "s_rho"
):
    """Depth (m, positive down) of a column's minimum or maximum of ``variable``.

    ``extremum`` is ``"min"`` or ``"max"`` and is required -- there is no neutral
    default between "the oxygen minimum" and "the oxygen maximum". The shallowest level
    wins a tie; an all-NaN column is NaN. The vertical axis is whatever the source has
    (native ROMS levels, or standard depths) -- see the module docstring.
    """
    if extremum not in _EXTREMA:
        raise KeyError(
            f"depth_of needs extremum= one of {sorted(_EXTREMA)}, got {extremum!r}."
        )
    word, method = _EXTREMA[extremum]
    da = _resolve(ds, variable, "depth_of")
    zdim, depth, _ = _vertical_of(ds, da, s_dim)

    best = getattr(da, method)(zdim)
    # Equal to the extremum -> candidate; shallowest of the candidates wins. NaN never
    # equals anything, so an all-NaN column has no candidate and comes out NaN.
    out = _drop_vertical_coords(depth, zdim).where(da == best).min(zdim)
    out = out.where(da.notnull().any(zdim)).transpose(
        *[d for d in da.dims if d != zdim], ...
    )
    label = _label(variable, da)
    out.name = f"depth_of_{word}_{label}"
    out.attrs = {
        "units": "m",
        "positive": "down",
        "long_name": f"depth of {word} {_describe(da)}",
        "depth_of_extremum": extremum,
        "depth_of_variable": label,
    }
    return out


def calculate_thickness(
    ds,
    *,
    variable=None,
    below: float | None = None,
    above: float | None = None,
    s_dim: str = "s_rho",
):
    """Return the summed thickness (m) where ``variable`` is below/above a threshold.

    Exactly one of ``below=`` (strictly less than) or ``above=`` (strictly greater
    than), in the variable's own units. A column with data but no qualifying layer is 0;
    a column with no data is NaN. Layer thicknesses are described in the module
    docstring.
    """
    if (below is None) == (above is None):
        raise ValueError(
            "thickness needs exactly one of below= or above= (a threshold in the "
            f"variable's own units), got below={below!r}, above={above!r}."
        )
    da = _resolve(ds, variable, "thickness")
    zdim, depth, is_roms = _vertical_of(ds, da, s_dim)

    if is_roms:
        dz = _roms_thickness(ds, depth, s_dim)
    else:
        # Ascending depth so "the first interface" is the surface-most one; sorting the
        # data with it keeps level and thickness paired.
        order = np.argsort(np.asarray(depth.values))
        da = da.isel({zdim: order})
        depth = depth.isel({zdim: order})
        dz = _thickness_from_centres(depth, zdim, clip_surface=True)

    word, threshold = ("below", below) if below is not None else ("above", above)
    inside = da < threshold if below is not None else da > threshold
    out = dz.where(inside).sum(zdim).where(da.notnull().any(zdim))
    out = out.transpose(*[d for d in da.dims if d != zdim], ...)
    label = _label(variable, da)
    units = str(da.attrs.get("units", "")).strip()
    shown = f"{float(threshold):g}" + (f" {units}" if units else "")
    out.name = f"thickness_of_{label}_{word}_{float(threshold):g}"
    out.attrs = {
        "units": "m",
        "long_name": f"thickness where {_describe(da)} is {word} {shown}",
        f"thickness_{word}": float(threshold),
        "thickness_variable": label,
    }
    return out


def _inputs(spec: dict[str, Any]) -> list[list[str]]:
    """Return what both calculators need, for :func:`ocean_skill.operators.spec_names`.

    That is the variable the spec names (itself possibly a combination or other spec).
    """
    from ocean_skill.operators import spec_names

    variable = spec.get("variable")
    return spec_names(variable) if variable is not None else []


def _register() -> None:
    from ocean_skill.operators import register_calculator

    register_calculator("depth_of", inputs=_inputs)(calculate_depth_of)
    register_calculator("thickness", inputs=_inputs)(calculate_thickness)


_register()
