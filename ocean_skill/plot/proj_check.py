"""One tripwire for a specific, silent cartopy/PROJ mis-projection.

PROJ 9.8 made ``+proj=eqc`` ellipsoidal (`OSGeo/PROJ#4656
<https://github.com/OSGeo/PROJ/pull/4656>`_). Cartopy's
:class:`~cartopy.crs.PlateCarree` still assumes the old spherical ``eqc`` and
un-scales degrees by the sphere's own radius (`SciTools/cartopy#2645
<https://github.com/SciTools/cartopy/issues/2645>`_) -- fixed in cartopy by
`#2653 <https://github.com/SciTools/cartopy/pull/2653>`_, merged 2026-08-27
but not yet in a release (milestone 0.26). Any environment with cartopy <=
0.25 and PROJ >= 9.8 mis-transforms *every* point that goes through
:class:`~cartopy.crs.PlateCarree` -- roughly 0.2 degrees of latitude at
mid-latitudes -- with no error, no NaN, nothing to catch in a normal test:
the numbers just come out wrong. That is how a set of Hvalfjörður CTD casts
ended up plotted on the Icelandic mainland.

There is no version pair to special-case here: the safe way to detect a bad
transform is to run one and check the answer, once per process, and say so
loudly if it is wrong.
"""

from __future__ import annotations

import functools
import math
import warnings

#: Web Mercator (EPSG:3857) sphere radius, metres -- the same value
#: :func:`ocean_skill.plot.holoviews_renderer._to_mercator` projects onto.
_WEB_MERCATOR_R = 6378137.0

#: A point with no special-cased latitude (equator, pole, antimeridian) to
#: transform as the canary. 45°N sits well inside the skew's range (largest
#: at low-to-mid latitudes, shrinking towards the poles) and is unambiguous
#: in either hemisphere.
_PROBE_LON, _PROBE_LAT = 0.0, 45.0

#: Metres of disagreement below which the transform counts as correct --
#: generous next to a ~25,000 m (~0.23°) skew, tight next to ordinary
#: floating-point/PROJ-pipeline noise.
_TOLERANCE_M = 1.0


def _web_mercator_y(lat_deg: float) -> float:
    """Closed-form Web Mercator northing for ``lat_deg`` -- the expected answer."""
    return _WEB_MERCATOR_R * math.log(
        math.tan(math.pi / 4.0 + math.radians(lat_deg) / 2.0)
    )


@functools.cache
def projection_skew() -> str | None:
    """``None`` if cartopy's PlateCarree -> Web Mercator transform is correct here.

    Otherwise a message identifying the installed cartopy/PROJ versions, the
    measured offset, the cause, and both remedies. Runs one transform, cached
    for the life of the process -- cheap enough to call from every
    map-drawing call site, and Python's default warning de-duplication keeps
    a per-process cache from meaning a per-process warning storm.
    """
    import cartopy
    import cartopy.crs as ccrs
    import numpy as np
    import pyproj

    y_transformed = ccrs.GOOGLE_MERCATOR.transform_points(
        ccrs.PlateCarree(), np.array([_PROBE_LON]), np.array([_PROBE_LAT])
    )[0, 1]
    y_expected = _web_mercator_y(_PROBE_LAT)
    offset_m = float(y_transformed - y_expected)
    if abs(offset_m) <= _TOLERANCE_M:
        return None

    return (
        f"cartopy {cartopy.__version__} with PROJ {pyproj.proj_version_str} "
        f"mis-projects PlateCarree data by {offset_m:.0f} m "
        f"({abs(offset_m) / 111_320:.3f}° latitude) at this probe point, and "
        "every geographic map drawn from this environment is skewed the "
        "same way. Cause: PROJ >= 9.8 made 'eqc' ellipsoidal "
        "(OSGeo/PROJ#4656), which cartopy <= 0.25's PlateCarree does not "
        "account for (SciTools/cartopy#2645). Fix by running "
        '`conda install "proj<9.8"` (or, once released, upgrading to '
        "cartopy >= 0.26, which carries the fix in SciTools/cartopy#2653)."
    )


def warn_projection_skew() -> None:
    """Warn once per process if this environment's cartopy/PROJ pairing is broken."""
    message = projection_skew()
    if message is not None:
        warnings.warn(message, stacklevel=3)
