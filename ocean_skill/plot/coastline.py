"""Coastline resolution policy: one value ladder, shared by both renderers.

A map panel's coastline can come from two datasets. **Natural Earth** ships three
built-in scales (``"110m"``, ``"50m"``, ``"10m"`` — coarser numbers are literally
metres-per-pixel at 1:1,000,000, so bigger number = coarser line) and cartopy already
knows how to pick among them from the axes extent via its ``AdaptiveScaler``: a
near-global domain gets ``"110m"`` and a tight zoom gets ``"10m"``, with no user
input needed. **GSHHS** goes finer still (``"coarse"`` through ``"full"``), enough to
resolve a spit or a fjord head that even Natural Earth ``"10m"`` smears over — at the
cost of a one-time shapefile download (large at ``"full"``) and slower renders on a
big domain, so it is never chosen automatically.

``"auto"`` is the default everywhere: Natural Earth, scaled to the map's own extent,
exactly matching cartopy's builtin behaviour for undecorated ``ax.coastlines()`` /
``cfeature.LAND``. A user who wants more than Natural Earth's finest asks for GSHHS
by name (``coastline_resolution="full"``, say) — never silently, since a domain that
size on ``"full"`` GSHHS is not free.

Both renderers import from here so ``"auto"`` means the same map extent -> resolution
mapping whether the figure is a static PNG or a hvplot pane, and so a GSHHS request
that the interactive renderer can't draw falls back to the same Natural Earth scale
either renderer would.
"""

from __future__ import annotations

import warnings

__all__ = [
    "DEFAULT_COASTLINE_RESOLUTION",
    "GSHHS_SCALES",
    "NE_RESOLUTIONS",
    "auto_ne_resolution",
    "is_gshhs",
    "nearest_ne_resolution",
    "normalize_coastline_resolution",
]

#: Natural Earth's three built-in coastline/land scales, coarsest first.
NE_RESOLUTIONS = ("110m", "50m", "10m")

#: GSHHS scale names, coarsest to finest, and cartopy's single-letter aliases for them.
_GSHHS_NAMES = ("coarse", "low", "intermediate", "high", "full")
_GSHHS_ALIASES = {"c": "coarse", "l": "low", "i": "intermediate", "h": "high", "f": "full"}
GSHHS_SCALES = _GSHHS_NAMES

#: Natural Earth, scaled to the map's own extent -- matches plain ``ax.coastlines()``.
DEFAULT_COASTLINE_RESOLUTION = "auto"

#: GSHHS scale -> the Natural Earth resolution nearest it in detail, for the
#: interactive renderer (geoviews/hvplot cannot draw GSHHS at all).
_GSHHS_TO_NE = {
    "coarse": "110m",
    "low": "50m",
    "intermediate": "10m",
    "high": "10m",
    "full": "10m",
}


def normalize_coastline_resolution(value: str) -> str:
    """Validate a ``coastline_resolution`` value and expand GSHHS aliases.

    Returns ``"auto"``, one of :data:`NE_RESOLUTIONS`, or one of :data:`GSHHS_SCALES`
    (aliases like ``"f"`` are expanded to ``"full"``). Raises :class:`ValueError` on
    anything else, naming the accepted values so a typo doesn't surface as a cryptic
    cartopy error two calls deeper.
    """
    if value == "auto" or value in NE_RESOLUTIONS:
        return value
    if value in _GSHHS_NAMES:
        return value
    if value in _GSHHS_ALIASES:
        return _GSHHS_ALIASES[value]
    accepted = ("auto", *NE_RESOLUTIONS, *_GSHHS_NAMES)
    raise ValueError(
        f"Unknown coastline_resolution {value!r}; expected one of {accepted!r}."
    )


def is_gshhs(resolution: str) -> bool:
    """Whether a normalized ``coastline_resolution`` names a GSHHS scale."""
    return resolution in _GSHHS_NAMES


def auto_ne_resolution(extent: tuple[float, float, float, float] | None) -> str:
    """Natural Earth resolution cartopy's ``AdaptiveScaler`` would pick for ``extent``.

    ``extent`` is ``(lon0, lon1, lat0, lat1)`` in degrees, the same shape
    ``GeoAxes.get_extent()`` / ``ax.coastlines(resolution="auto")`` use internally.
    Delegates to cartopy's own scaler rather than re-encoding its thresholds here, so
    this can never drift from what plain ``ax.coastlines()`` already does.
    """
    from cartopy.feature import AdaptiveScaler

    # Same thresholds cartopy's cfeature.LAND / cfeature.COASTLINE ship with: 110m
    # above 50 degrees of extent, 50m above 15, 10m below that.
    scaler = AdaptiveScaler("110m", (("50m", 50), ("10m", 15)))
    return scaler.scale_from_extent(extent)


def nearest_ne_resolution(resolution: str, *, extent=None) -> str:
    """The Natural Earth resolution to draw for a normalized ``coastline_resolution``.

    ``"auto"`` resolves by ``extent`` (see :func:`auto_ne_resolution`); a Natural
    Earth value passes through unchanged; a GSHHS scale maps down to its nearest NE
    neighbour *and* warns, since the interactive renderer cannot draw GSHHS itself --
    a silent downgrade would look like a bug rather than a documented limitation.
    """
    if resolution == "auto":
        return auto_ne_resolution(extent)
    if resolution in NE_RESOLUTIONS:
        return resolution
    ne = _GSHHS_TO_NE[resolution]
    warnings.warn(
        f"coastline_resolution={resolution!r} (GSHHS) is not supported on the "
        f"interactive renderer; using Natural Earth {ne!r} instead. GSHHS detail is "
        "only available on static plots.",
        stacklevel=3,
    )
    return ne
