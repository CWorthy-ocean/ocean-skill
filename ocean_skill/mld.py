"""Mixed layer depth (MLD): the genuinely custom diagnostic named in operators.py.

Threshold-criterion methods only, for now (``density_threshold``,
``temperature_threshold``), so a model run can be compared against an observational
MLD product with the *same* definition rather than an incidental one. The product this
is aimed at is the Holte & Talley Argo monthly climatology
(``Argo_mixedlayers_monthlyclim_04142022.nc``), and that file ships density-based MLD
fields only: ``mld_dt_*`` (density threshold — 0.03 kg/m3 from 10 dbar, which
``density_threshold`` matches) and ``mld_da_*`` (their hybrid density algorithm). It has
no temperature-criterion MLD field at all, so ``temperature_threshold`` (de Boyer
Montégut's 0.2 degC from 10 m) is the same kind of definition but has nothing in that
file to be scored against. The hybrid ``*_algorithm`` methods (``density_algorithm``,
``temperature_algorithm``) are Holte & Talley's 2009 hybrid method — a profile-shape
fit plus gradient/curvature feature detection plus a selection tree — and are a
separate, larger port (reference: the Climate Data Toolbox's ``mld.m``,
https://github.com/chadagreene/CDT/blob/master/cdt/mld.m); ``calculate_mld`` names
them explicitly so asking for one now fails with what to expect instead of a bare
KeyError. (``mld_da_*`` is what a ported ``density_algorithm`` would be scored
against.)

**Definition** (de Boyer Montégut 2004, the same criterion Holte & Talley's
density-threshold field uses): the shallowest depth at which a profile variable
differs from its value at ``ref_depth`` by more than ``threshold``, found by linear
interpolation between the two bracketing model levels — not snapped to the nearer
one. A column shallower than ``ref_depth``, or one with no crossing at all (fully
mixed to the deepest resolved level), returns NaN, mirroring how
:func:`ocean_skill.roms.to_depth` treats a target outside the water column: no
extrapolation, said with NaN rather than a guess.

**CF names** (:data:`_STANDARD_NAMES`) follow the method, because CF's
``ocean_mixed_layer_thickness_defined_by_<criterion>`` family encodes only the
criterion *variable*: the base of the layer is where that quantity (temperature,
sigma-t, or sigma-theta) differs from its surface value by a certain amount, and
neither the amount nor the depth standing in for "the surface" is part of the name.
So ``density_threshold`` produces
``ocean_mixed_layer_thickness_defined_by_sigma_theta`` (sigma0 via TEOS-10 is a
potential-density anomaly, i.e. CF's sigma-theta criterion) and
``temperature_threshold`` produces
``ocean_mixed_layer_thickness_defined_by_temperature``. Neither produces the generic
``ocean_mixed_layer_thickness``, which names no criterion at all. What the name
leaves out is recorded on the result instead, in its ``mld_method``,
``mld_threshold`` and ``mld_ref_depth`` attrs — the place to look for "which
threshold, measured from what depth?".

Only ROMS output is supported: the calculation needs the full water column *before*
any vertical selection/reduction has collapsed it, which is exactly what
:func:`ocean_skill.roms.standardize` already attaches as ``z_rho`` before this ever
runs (see :func:`ocean_skill.operators.resolve_variable`'s ``calculate`` dispatch).
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "calculate_mld",
    "mld_density_threshold",
    "mld_temperature_threshold",
    "mld_threshold",
    "potential_density",
]

#: Default reference depth (m) and thresholds, matching de Boyer Montégut (2004) /
#: Holte & Talley's threshold criteria, and CDT's ``mld.m`` defaults (``refpres``,
#: ``dthresh``, ``tthresh``) — named the same way now so the hybrid methods slot in
#: later without a signature break.
REF_DEPTH = 10.0
DENSITY_THRESHOLD = 0.03  # kg/m3
TEMPERATURE_THRESHOLD = 0.2  # degC

#: Holte & Talley's two hybrid ("algorithm") methods. Not implemented here — see the
#: module docstring — named explicitly so a request for one says what to expect
#: instead of "unknown method".
_HYBRID_METHODS = frozenset({"density_algorithm", "temperature_algorithm"})

#: The CF standard_name each method's output carries — one per *criterion variable*,
#: because that is all CF's ``ocean_mixed_layer_thickness_defined_by_<criterion>``
#: family encodes: the layer's base is where the criterion quantity differs from its
#: surface value by a certain amount, and neither the amount (``threshold``) nor what
#: stands in for "the surface" (``ref_depth``) is part of the name. Those ride on the
#: result's ``mld_threshold``/``mld_ref_depth`` attrs instead (see :func:`_mld_attrs`).
#:
#: ``density_threshold`` is a sigma-theta criterion: :func:`potential_density` is
#: TEOS-10 sigma0, a potential-density anomaly referenced to the sea surface, which is
#: what CF means by sigma-theta. Deliberately *not* the generic
#: ``ocean_mixed_layer_thickness``: that names no criterion, so it could not tell
#: these two definitions (or a mixing scheme's own boundary-layer depth, CF's
#: ``..._defined_by_mixing_scheme``) apart. A method added to :data:`_METHODS` needs
#: an entry here too — :func:`_mld_attrs` raises KeyError rather than quietly falling
#: back to the generic name.
_STANDARD_NAMES = {
    "density_threshold": "ocean_mixed_layer_thickness_defined_by_sigma_theta",
    "temperature_threshold": "ocean_mixed_layer_thickness_defined_by_temperature",
}


def _mld_threshold_1d(
    values: np.ndarray, depth: np.ndarray, *, threshold: float, ref_depth: float
) -> float:
    """Threshold-crossing MLD for one water column (1-D, any order, NaNs allowed).

    ``depth`` is positive-down. Returns NaN if the column is shallower than
    ``ref_depth``, if no data reach ``ref_depth``, or if the criterion is never
    exceeded (a fully mixed column, at least to the deepest resolved level).
    """
    order = np.argsort(depth)
    d, v = depth[order], values[order]
    valid = np.isfinite(d) & np.isfinite(v)
    d, v = d[valid], v[valid]
    if d.size < 2 or d[0] > ref_depth or d[-1] < ref_depth:
        return np.nan

    ref_value = np.interp(ref_depth, d, v)
    diff = v - ref_value
    below = d >= ref_depth
    exceed = np.flatnonzero((np.abs(diff) > threshold) & below)
    if exceed.size == 0:
        return np.nan
    first = int(exceed[0])
    if first == 0:
        # The crossing coincides with the reference level itself (threshold is ~0,
        # or ref_depth sits exactly on a model level already past it).
        return float(d[first])

    d0, d1 = d[first - 1], d[first]
    v0, v1 = diff[first - 1], diff[first]
    target = threshold if v1 > 0 else -threshold
    if v1 == v0:  # degenerate: two levels with an identical value (repeated depth)
        return float(d1)
    frac = (target - v0) / (v1 - v0)
    return float(d0 + frac * (d1 - d0))


def mld_threshold(var, z, *, threshold: float, ref_depth: float, s_dim: str = "s_rho"):
    """Vectorized threshold-crossing MLD along ``s_dim``, over every other dimension.

    ``var`` and ``z`` (negative-down, e.g. ``z_rho``) must share ``s_dim`` and
    broadcast against each other otherwise. Returns a DataArray without ``s_dim``,
    in metres, positive down.
    """
    from ocean_skill.roms import _contiguous_column

    var = _contiguous_column(var, s_dim)
    z = _contiguous_column(z, s_dim)
    depth = -z

    import xarray as xr

    return xr.apply_ufunc(
        _mld_threshold_1d,
        var,
        depth,
        input_core_dims=[[s_dim], [s_dim]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        kwargs={"threshold": float(threshold), "ref_depth": float(ref_depth)},
    )


def potential_density(temp, salt, z, lon, lat):
    """Potential density anomaly (sigma0, kg/m3) via TEOS-10, from model fields.

    ROMS carries potential temperature and practical salinity; gsw wants absolute
    salinity and conservative temperature, so this is ``SP -> SA -> (with pt) CT ->
    sigma0`` (``gsw.SA_from_SP``, ``gsw.CT_from_pt``, ``gsw.sigma0``), with pressure
    from ``z`` (negative-down, matching gsw's own convention) via ``gsw.p_from_z``.
    Imported inside the function, not at module level — gsw is a dependency scoped to
    this calculator, not the package core (see ``environment.yml``).
    """
    import gsw
    import xarray as xr

    pressure = xr.apply_ufunc(gsw.p_from_z, z, lat, dask="parallelized", output_dtypes=[float])
    sa = xr.apply_ufunc(
        gsw.SA_from_SP, salt, pressure, lon, lat, dask="parallelized", output_dtypes=[float]
    )
    ct = xr.apply_ufunc(gsw.CT_from_pt, sa, temp, dask="parallelized", output_dtypes=[float])
    sigma0 = xr.apply_ufunc(gsw.sigma0, sa, ct, dask="parallelized", output_dtypes=[float])
    sigma0.name = "sea_water_sigma_theta"
    sigma0.attrs = {
        "units": "kg m-3",
        "long_name": "potential density anomaly (sigma0, TEOS-10)",
    }
    return sigma0


def _require(ds, standard_name: str):
    from ocean_skill.units import find_variable

    da = find_variable(ds, standard_name)
    if da is None:
        raise KeyError(
            f"mld needs {standard_name!r}, which is not in this dataset (or has not "
            "been standardized -- ocean_skill.roms.standardize renames ROMS output to "
            "CF standard_names before a calculator ever sees it)."
        )
    return da


def _mld_attrs(method: str, threshold: float, ref_depth: float) -> dict[str, Any]:
    """Return the attrs of a computed MLD: its CF name from the method, plus its recipe.

    The ``standard_name`` comes from :data:`_STANDARD_NAMES` (the criterion variable,
    which is all a CF name can say); the ``mld_*`` attrs carry what the name leaves out
    — the method, the threshold and the reference depth — so a result stays
    self-describing after it has left this module.
    """
    return {
        "units": "m",
        "standard_name": _STANDARD_NAMES[method],
        "long_name": f"mixed layer depth ({method.replace('_', ' ')})",
        "mld_method": method,
        "mld_threshold": threshold,
        "mld_ref_depth": ref_depth,
    }


def mld_density_threshold(
    ds, *, threshold: float = DENSITY_THRESHOLD, ref_depth: float = REF_DEPTH, s_dim: str = "s_rho"
):
    """MLD as the shallowest depth where sigma0 exceeds sigma0(ref_depth) + threshold.

    de Boyer Montégut (2004); Holte & Talley's ``mld_dt_*`` monthly-climatology
    fields. Defaults (0.03 kg/m3 from 10 m) match both. The result is named
    ``ocean_mixed_layer_thickness_defined_by_sigma_theta`` (sigma0 is a
    potential-density anomaly, CF's sigma-theta criterion); the threshold and
    reference depth are on its ``mld_threshold``/``mld_ref_depth`` attrs, not in the
    name (see the module docstring).
    """
    temp = _require(ds, "sea_water_potential_temperature")
    salt = _require(ds, "sea_water_practical_salinity")
    if "z_rho" not in ds.coords:
        raise KeyError(
            "mld needs 'z_rho' (attached by ocean_skill.roms.standardize); this "
            "dataset has not been through the ROMS adapter."
        )
    z_rho = ds.coords["z_rho"]
    sigma0 = potential_density(temp, salt, z_rho, ds.coords["lon"], ds.coords["lat"])
    out = mld_threshold(sigma0, z_rho, threshold=threshold, ref_depth=ref_depth, s_dim=s_dim)
    # The variable's name *is* its standard_name; taking both from the one attrs dict
    # means they cannot drift apart.
    attrs = _mld_attrs("density_threshold", threshold, ref_depth)
    out = out.rename(attrs["standard_name"])
    out.attrs = attrs
    return out


def mld_temperature_threshold(
    ds, *, threshold: float = TEMPERATURE_THRESHOLD, ref_depth: float = REF_DEPTH, s_dim: str = "s_rho"
):
    """MLD as the shallowest depth where |T - T(ref_depth)| exceeds threshold.

    de Boyer Montégut (2004)'s temperature criterion; the default (0.2 degC from 10 m)
    is theirs. Holte & Talley's monthly climatology carries no temperature-criterion
    MLD field, so there is nothing in it to match (see the module docstring). The
    result is named ``ocean_mixed_layer_thickness_defined_by_temperature``; the
    threshold and reference depth are on its ``mld_threshold``/``mld_ref_depth``
    attrs, not in the name.
    """
    temp = _require(ds, "sea_water_potential_temperature")
    if "z_rho" not in ds.coords:
        raise KeyError(
            "mld needs 'z_rho' (attached by ocean_skill.roms.standardize); this "
            "dataset has not been through the ROMS adapter."
        )
    z_rho = ds.coords["z_rho"]
    out = mld_threshold(temp, z_rho, threshold=threshold, ref_depth=ref_depth, s_dim=s_dim)
    # Name and standard_name from one attrs dict, as in mld_density_threshold.
    attrs = _mld_attrs("temperature_threshold", threshold, ref_depth)
    out = out.rename(attrs["standard_name"])
    out.attrs = attrs
    return out


#: Registered under ``{"calculate": "mld", "method": ...}`` -- see
#: :data:`ocean_skill.operators.CALCULATORS`.
_METHODS = {
    "density_threshold": mld_density_threshold,
    "temperature_threshold": mld_temperature_threshold,
}


def calculate_mld(ds, *, method: str | None = None, **kwargs):
    """Dispatch to one of the threshold MLD methods by name.

    ``method`` is required rather than defaulted, on purpose: four methods are named
    here (two implemented, two hybrid ones not yet ported), and picking one silently
    would be the kind of "looks right" number this project otherwise refuses to
    produce (see :func:`ocean_skill.operators.combine`). The method also decides the
    result's CF name (:data:`_STANDARD_NAMES`), so it is never just a detail of how
    the number was made.
    """
    if method in _HYBRID_METHODS:
        raise NotImplementedError(
            f"{method!r} is Holte & Talley's (2009) hybrid algorithm -- a "
            "profile-shape fit plus gradient/curvature feature detection plus a "
            "selection tree, not yet ported (see the module docstring). Available "
            f"now: {sorted(_METHODS)}."
        )
    if method not in _METHODS:
        raise KeyError(
            f"mld needs method= one of {sorted(_METHODS)} (or {sorted(_HYBRID_METHODS)}, "
            f"not yet implemented), got {method!r}."
        )
    return _METHODS[method](ds, **kwargs)


#: What :func:`calculate_mld` needs, as a function of its spec -- for
#: :func:`ocean_skill.operators.spec_names`, which pre-filters catalog sources before
#: anything is read. A density criterion needs temperature *and* salinity; a
#: temperature criterion needs temperature alone. ``method`` missing or unrecognized
#: reports the broadest requirement rather than none, since a spec that will fail
#: loudly in :func:`calculate_mld` anyway should not be pre-filtered as if it needed
#: nothing.
def _mld_inputs(spec: dict[str, Any]) -> list[list[str]]:
    temp = "sea_water_potential_temperature"
    salt = "sea_water_practical_salinity"
    if spec.get("method") in ("temperature_threshold", "temperature_algorithm"):
        return [[temp]]
    return [[temp, salt]]


def _register() -> None:
    from ocean_skill.operators import register_calculator

    register_calculator("mld", inputs=_mld_inputs)(calculate_mld)


_register()
