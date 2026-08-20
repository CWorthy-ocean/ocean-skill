"""CF-standardization helpers (cf-xarray based).

Detect axes (T/Z/Y/X), rename coordinates to a canonical ``time``/``depth``/``lat``/
``lon``, map variables to CF standard_names (from the catalog entry's ``standard_names``
map or dataset attributes), and tag the object with its ``featureType``.
"""

from __future__ import annotations

__all__ = ["find_coord", "standardize", "tag_feature_type"]


def standardize(
    obj, standard_names: dict[str, str] | None = None, axes: dict | None = None
):
    """Return ``obj`` with canonical axis names and variables keyed by standard_name."""
    # TODO: use cf_xarray for axis/coord detection; apply the standard_names rename map;
    # normalize units where declared.
    raise NotImplementedError


def tag_feature_type(obj, feature_type: str):
    """Attach a ``featureType`` tag to a standardized object."""
    raise NotImplementedError


#: Name fallbacks used when cf-xarray can't identify an axis. ROMS is the motivating
#: case: it writes ``units="degrees East"`` (CF wants ``degrees_east``) and
#: ``ocean_time`` with ``units="second"``, so cf-xarray detects nothing. Order matters —
#: rho-points before the staggered/coarse variants. ``sigma0`` is the isopycnal axis
#: :func:`ocean_skill.roms.to_sigma0` produces -- a vertical axis in every way that
#: matters here (a facet row, what ``{"Z": "mean"}`` collapses), just not a depth.
_COORD_FALLBACKS: dict[str, tuple[str, ...]] = {
    "longitude": ("lon_rho", "lon", "longitude", "nav_lon", "x_rho", "lon_u", "lon_v"),
    "latitude": ("lat_rho", "lat", "latitude", "nav_lat", "y_rho", "lat_u", "lat_v"),
    "time": ("time", "ocean_time", "t", "T"),
    "vertical": ("depth", "z", "lev", "s_rho", "z_rho", "depth_surface", "sigma0"),
}


def find_coord(ds, kind: str):
    """Return the coordinate/variable for ``kind`` (cf-xarray first, then name match).

    ``kind`` is ``longitude``/``latitude``/``time``/``vertical``. Returns ``None`` if
    nothing plausible is present.
    """
    import cf_xarray  # noqa: F401  (registers the .cf accessor)

    axis = {"longitude": "X", "latitude": "Y", "time": "T", "vertical": "Z"}[kind]
    # A DataArray has no `.variables`; its coords are the equivalent namespace.
    known = getattr(ds, "variables", None)
    if known is None:
        known = ds.coords
    for key in (kind, axis):
        try:
            got = ds.cf[key]
        except (KeyError, ValueError):
            continue
        # Ignore bare dimension indices (e.g. a "time" dim with no coordinate variable,
        # which cf-xarray still reports as the T axis) — they carry no real values.
        if getattr(got, "name", None) in known:
            return got
    for nm in _COORD_FALLBACKS[kind]:
        if nm in known:
            return ds[nm]
    return None
