"""Shared geometry for contour lines drawn over a map panel.

``o2_omz.plot(contours=o2_mean)`` draws oxygen isolines over an OMZ-thickness map --
the map counterpart of a section's isotherms over phosphate (see
:mod:`ocean_skill.plot.section`). Both renderers read the lines' geometry from here,
so a static and an interactive figure cannot disagree about where a line runs:

* :func:`check_map_overlay` refuses an overlay that is not a map with the panel's own
  facet axes;
* :func:`map_contour_arrays` turns one overlay map into the 2-D ``x, y, z`` a contour
  engine takes, its longitudes in the panel's own convention;
* :func:`prepare_map_overlay` is the two in one, the call both renderers make.

Which values get a line (:func:`~ocean_skill.plot.section.contour_levels`), the lines
themselves for a renderer with no contour primitive
(:func:`~ocean_skill.plot.section.contour_paths`) and the labels' spelling
(:func:`~ocean_skill.plot.section.contour_label`) are the section helpers, unchanged:
none of them knows what its axes mean.

Unlike a section's overlay, a map's need not share the panel's grid. Map lines live in
absolute longitude and latitude, so a climatology's isolines over a model's fill land
where they belong without regridding; a section's x axis is a distance measured along
each panel's own path, which is why a section refuses a mismatched mesh instead.

The one thing that has to agree is the longitude convention. A domain straddling the
antimeridian (a Pacific model running 77°E to 316°E) stored in ±180 jumps from +179
to −179 inside its own array; a filled mesh shrugs that off, but a contour engine
joins the two sides and draws its lines across the whole globe. So the overlay is put
in the convention the *panel* is contiguous in
(:func:`~ocean_skill.align.natural_convention`) before anything is contoured -- the
same test the static renderer's frame (``_map_projection``) and the interactive
renderer's (``_output_projection``) already make.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

__all__ = [
    "check_map_overlay",
    "horizontal_dims",
    "map_contour_arrays",
    "prepare_map_overlay",
    "refuse_difference_contours",
]


def horizontal_dims(da: xr.DataArray) -> set[str]:
    """Return the dims ``da``'s longitude and latitude coordinates run along.

    ``{"lat", "lon"}`` on a rectilinear grid, ``{"eta_rho", "xi_rho"}`` on a
    curvilinear ROMS one -- whatever is *not* a facet axis (time, depth) of the map.
    """
    from ocean_skill.align import _lat_name, _lon_name

    dims: set[str] = set()
    for name in (_lon_name(da), _lat_name(da)):
        if name is not None:
            dims |= {str(d) for d in da[name].dims}
    return dims


def _facet_sizes(da: xr.DataArray) -> dict[str, int]:
    """Return ``{dim: size}`` for ``da``'s dims beyond its horizontal ones."""
    horizontal = horizontal_dims(da)
    return {str(d): int(da.sizes[d]) for d in da.dims if d not in horizontal}


def check_map_overlay(overlay: xr.DataArray, panel: xr.DataArray) -> None:
    """Refuse an overlay that cannot be drawn as lines over map ``panel``.

    The overlay has to be a map: longitude and latitude coordinates spanning at least
    two points along each of two horizontal dims. Beyond those it must stand on
    exactly the axes ``panel`` does (a map faceted by time is paired with an overlay
    faceted by the same number of times, panel by panel, by position), so that each
    panel draws its own overlay's lines rather than a guessed one. Its *grid* may
    differ from the panel's -- see the module docstring.

    Raises
    ------
    ValueError
        If ``overlay`` has no longitude/latitude, too few points to contour, or
        standing axes (names or sizes) that differ from ``panel``'s.
    """
    from ocean_skill.align import _lat_name, _lon_name

    if _lon_name(overlay) is None or _lat_name(overlay) is None:
        raise ValueError(
            "contours= over a map needs an overlay with longitude and latitude, but "
            f"it has only {sorted(map(str, overlay.coords))} -- build it the way the "
            "map was built (same source, select= and aggregate=) with the variable "
            "to draw as lines."
        )
    horizontal = horizontal_dims(overlay)
    if len(horizontal) < 2:
        # A 1-D lon/lat pair on one shared dim is a track or a list of points, and
        # scalar ones are a single point: neither is a surface a contour runs across.
        where = (
            f"both run along {sorted(horizontal)} only (a track or a set of points)"
            if horizontal
            else "are single values (one point)"
        )
        raise ValueError(
            "contours= over a map needs an overlay on a 2-D horizontal grid, but its "
            f"longitude and latitude {where}."
        )
    if any(overlay.sizes[d] < 2 for d in horizontal):
        raise ValueError(
            "contours= over a map needs at least two points along each horizontal "
            f"axis to draw a line, but the overlay has "
            f"{ {d: int(overlay.sizes[d]) for d in sorted(horizontal)} }."
        )
    want, got = _facet_sizes(panel), _facet_sizes(overlay)
    if got != want:
        raise ValueError(
            "contours= pairs each map panel with its overlay's own map, so the "
            f"overlay must stand on the same axes as the map: the map has "
            f"{want or 'none beyond lon/lat'}, the overlay "
            f"{got or 'none beyond lon/lat'}. Narrow or collapse the overlay with "
            "the same select=/aggregate= as the map."
        )


def refuse_difference_contours(has_contours: bool) -> None:
    """Refuse ``contours=`` on a grid of difference panels only -- both renderers.

    A comparison row draws its overlay's test lines over the test panel and its
    reference lines over the reference panel; the difference panel draws none, since a
    difference has no isolines of its own and either side's would be a guess at which
    one was meant. ``panels="difference"`` keeps only that panel, so the lines would
    have nowhere to go: refused by name rather than dropped.
    """
    if has_contours:
        raise ValueError(
            "contours= draws lines over the test and reference panels, but "
            'panels="difference" draws only the difference panel, which takes none. '
            'Drop panels="difference" to see the lines, or drop contours=.'
        )


#: Slack (degrees) around the panel's lon/lat box an overlay is kept within: a line
#: that runs along the panel's edge, on an overlay a little coarser than the fill, is
#: not cut short.
_PAD = 0.5


def _panel_box(
    panel: xr.DataArray, convention
) -> tuple[float, float, float, float] | None:
    """Return ``panel``'s padded ``(lon0, lon1, lat0, lat1)`` in ``convention``."""
    from ocean_skill.align import _lat_name, _lon_name, harmonize_longitude

    panel = harmonize_longitude(panel, convention)
    lon = np.asarray(panel[_lon_name(panel)].values, dtype="float64")
    lat = np.asarray(panel[_lat_name(panel)].values, dtype="float64")
    lon, lat = lon[np.isfinite(lon)], lat[np.isfinite(lat)]
    if not (lon.size and lat.size):
        return None
    return (
        float(lon.min()) - _PAD,
        float(lon.max()) + _PAD,
        float(lat.min()) - _PAD,
        float(lat.max()) + _PAD,
    )


def _window(da: xr.DataArray, lon_name: str, lat_name: str, box) -> xr.DataArray | None:
    """Return ``da`` cut to the index window whose lon/lat fall inside ``box``.

    Read from the coordinates alone, before any value is loaded: a global
    climatology's isolines over a regional model are then read and contoured over the
    panel's few cells, not the whole Earth for every panel. A rectilinear grid is cut
    along each 1-D axis; a curvilinear one to the bounding index box of the cells
    inside (its corners, still outside, are blanked by the caller). ``None`` when no
    point of the overlay is inside at all.
    """
    lo_x, hi_x, lo_y, hi_y = box
    lon, lat = da[lon_name], da[lat_name]
    if lon.ndim == 1 and lat.ndim == 1:
        keep = {
            lon.dims[0]: np.flatnonzero((lon.values >= lo_x) & (lon.values <= hi_x)),
            lat.dims[0]: np.flatnonzero((lat.values >= lo_y) & (lat.values <= hi_y)),
        }
    else:
        lon2, lat2 = xr.broadcast(lon, lat)
        dims = lon2.dims
        inside = (
            (lon2.values >= lo_x)
            & (lon2.values <= hi_x)
            & (lat2.transpose(*dims).values >= lo_y)
            & (lat2.transpose(*dims).values <= hi_y)
        )
        keep = dict(zip(dims, np.nonzero(inside), strict=True))
    if any(idx.size == 0 for idx in keep.values()):
        return None
    return da.isel(
        {d: slice(int(idx.min()), int(idx.max()) + 1) for d, idx in keep.items()}
    )


def prepare_map_overlay(
    overlay: xr.DataArray, panel: xr.DataArray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Check ``overlay`` against ``panel`` and return its contour arrays.

    :func:`check_map_overlay` then :func:`map_contour_arrays`: the one way a renderer
    prepares a map panel's overlay, so the static and interactive figures cannot come
    to differ on what is refused or how the lines are placed.
    """
    check_map_overlay(overlay, panel)
    return map_contour_arrays(overlay, panel=panel)


def map_contour_arrays(
    overlay: xr.DataArray, *, panel: xr.DataArray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``overlay`` as 2-D ``(lon, lat, values)`` float arrays, ready to contour.

    Parameters
    ----------
    overlay
        One map: the overlay's own slice for this panel, with no axes standing beyond
        its horizontal ones (select a facet panel's slice first). Rectilinear (1-D
        ``lon``/``lat``, broadcast here) or curvilinear (2-D, e.g. ROMS
        ``lon_rho``/``lat_rho``).
    panel
        The map this overlay is drawn over -- its own slice or the whole faceted
        field; only its longitude convention and lon/lat extent are read.

    Returns
    -------
    tuple of numpy.ndarray
        ``lon``, ``lat`` and ``values``, all of the overlay's horizontal shape.
        Longitudes are in the convention ``panel`` is contiguous in
        (:func:`~ocean_skill.align.natural_convention`), so a dateline-straddling map
        gets 0-360 and every other ±180, whatever either was stored in; a 1-D
        longitude axis is re-sorted to stay monotonic. The arrays cover only the
        overlay's cells over the panel's own lon/lat bounding box (half a degree of
        slack), cut from the coordinates before any value is read, and ``values`` is
        NaN at any cell of that window still outside the box, so neither lines nor
        their labels land off the panel. An overlay with no point over the panel
        comes back all NaN: no lines.

        The static renderer hands these to ``ax.contour(..., transform=
        PlateCarree())``, the transform its fill uses; the interactive one runs
        :func:`~ocean_skill.plot.section.contour_paths` on them and moves the
        vertices into its frame the way it moves every other plain overlay.

    Raises
    ------
    ValueError
        If ``overlay`` is not one map (see :func:`check_map_overlay`).
    """
    from ocean_skill.align import (
        _lat_name,
        _lon_name,
        harmonize_longitude,
        natural_convention,
    )

    extra = _facet_sizes(overlay)
    if extra:
        raise ValueError(
            "map_contour_arrays draws one map at a time, but the overlay still "
            f"stands on {extra} beyond lon/lat -- select this panel's slice first."
        )
    check_map_overlay(overlay, overlay)
    convention = natural_convention(panel)
    da = harmonize_longitude(overlay, convention)
    lon_name, lat_name = _lon_name(da), _lat_name(da)
    box = _panel_box(panel, convention)
    if box is not None:
        window = _window(da, lon_name, lat_name, box)
        if window is None:  # nothing of the overlay is over this panel: no lines
            tiny = da.isel({d: slice(0, 2) for d in horizontal_dims(da)})
            lon, lat = (
                tiny[n].broadcast_like(tiny).transpose(*tiny.dims)
                for n in (lon_name, lat_name)
            )
            nan = np.full(tiny.shape, np.nan)
            return (
                np.asarray(lon.values, "float64"),
                np.asarray(lat.values, "float64"),
                nan,
            )
        da = window
    dims = tuple(da.dims)
    lon_v = np.asarray(
        da[lon_name].broadcast_like(da).transpose(*dims).values, "float64"
    )
    lat_v = np.asarray(
        da[lat_name].broadcast_like(da).transpose(*dims).values, "float64"
    )
    z = np.asarray(da.values, dtype="float64")
    if box is not None:
        lo_x, hi_x, lo_y, hi_y = box
        inside = (lon_v >= lo_x) & (lon_v <= hi_x) & (lat_v >= lo_y) & (lat_v <= hi_y)
        z = np.where(inside, z, np.nan)
    return lon_v, lat_v, z
