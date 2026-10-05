"""Shared geometry for vertical-section plots.

One place both renderers read, so a section cannot look different statically than
interactively — the same reason :mod:`ocean_skill.plot.series` exists for the line
family. :func:`prepare_section` decides every axis convention a section needs once:
which coordinate is "depth" and which sign it reads positive, how the along-path
axis is labelled, and what a title calls the path itself. A renderer's own drawing
function calls this first and then only draws — it makes no convention decisions
of its own.

A section drawn as smooth filled bands with black contour lines of a second variable
on top (isotherms over phosphate) has more decisions of the same kind, and they live
here for the same reason -- a static and an interactive figure that each made them
would eventually disagree about where a line or a band edge sits:

* :func:`prepare_overlay` puts the line variable on the fill's own mesh, or refuses;
* :func:`contour_levels` picks which values get a line, once for the whole figure;
* :func:`contour_paths` computes the lines themselves, for a renderer with no
  contour primitive of its own (bokeh);
* :func:`fill_edges` places the filled bands' edges so every colour-bar tick sits on
  one.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import xarray as xr

from ocean_skill.plot._colorbar import _clean, tick_step

__all__ = [
    "CONTOUR_COLOR",
    "CONTOUR_WIDTH",
    "DEFAULT_CONTOUR_LEVELS",
    "DEFAULT_FILL_BANDS",
    "SECTION_MARKS",
    "SECTION_X",
    "X_DOMINANCE",
    "ContourLevel",
    "SectionGeometry",
    "check_section_options",
    "check_section_x",
    "contour_label",
    "contour_levels",
    "contour_paths",
    "difference_fill_levels",
    "fill_edges",
    "prepare_overlay",
    "prepare_section",
    "prepare_section_row",
]


@dataclass
class SectionGeometry:
    """What a section's axes mean, decided once and read by both renderers.

    ``x_name``/``y_name`` name the coordinates :func:`prepare_section` attaches to
    its returned field -- ``"distance"``/``"depth"`` -- both broadcast to the
    field's own 2-D shape regardless of whether the vertical axis is native
    s-levels (where depth genuinely varies along the path, so it is honestly 2-D)
    or a fixed-depth list (where it is 1-D values repeated across the other axis):
    one shape of coordinate for a renderer to read either way, no special case for
    which kind of vertical axis it is drawing.

    By default a path section's ``x_name`` holds the distance along it (km) and
    ``x_label`` says so -- but :func:`prepare_section` reads where the path runs, and
    one that is mostly east-west or north-south is drawn against its own longitude or
    latitude instead, exactly as a slab is. A slab -- a box averaged along one
    horizontal axis, whose along coordinate carries ``axis_coord`` (see
    :func:`ocean_skill.comparison._slab_to_section`) -- always has its *surviving*
    coordinate on ``x_name``, in degrees. Either way ``x_name`` still names the
    coordinate to draw against and ``x_label`` reads "latitude (°N)" /
    "longitude (°E)" / "distance along transect (km)". A renderer needs no branch for
    it: it draws ``x_name`` against ``y_name`` and labels the axes with
    ``x_label``/``y_label`` regardless. ``x_axis`` records which it is (``"distance"``,
    ``"lat"`` or ``"lon"``) for a caller that does care, such as one deciding whether
    two sections can share an x axis.
    """

    x_name: str
    y_name: str
    x_label: str
    y_label: str
    native_s: bool
    path_note: str
    x_axis: str = "distance"


def _band_note(axis: str, band) -> str:
    """``"mean over 180–200°E"`` / ``"mean over 5°S–5°N"``: what a slab averaged.

    ``axis`` is the averaged axis (``"lon"`` or ``"lat"``) and ``band`` its
    ``(lo, hi)`` range, both read off the along coordinate's attrs (see
    :func:`ocean_skill.comparison._slab_to_section`). A longitude band whose ends
    both sit in 0-360 reads as one run of east longitude ("180–200°E", even past
    180 -- the convention the box was asked for in); any other spells each end with
    its own hemisphere letter.
    """
    lo, hi = float(band[0]), float(band[1])
    if axis == "lat":
        return (
            f"mean over {abs(lo):g}°{'N' if lo >= 0 else 'S'}–"
            f"{abs(hi):g}°{'N' if hi >= 0 else 'S'}"
        )
    if lo >= 0 and hi >= 0:
        return f"mean over {lo:g}–{hi:g}°E"
    return (
        f"mean over {abs(lo):g}°{'E' if lo >= 0 else 'W'}–"
        f"{abs(hi):g}°{'E' if hi >= 0 else 'W'}"
    )


def _path_note(da, lon_name: str | None, lat_name: str | None) -> str:
    """``"29.0°N, 94.5°W → 27.5°N, 90.0°W"``, or ``"along 94.5°W"`` if lon is fixed.

    Read off the along-path lon/lat coordinates' own endpoints -- the same
    positive-down-style degree formatting :func:`ocean_skill.plot.series._place_of`
    uses for a station, extended to a pair of points rather than one. Empty when
    the field carries no lon/lat coordinates to read (should not happen for a
    real section, but a title with nothing to say is better than one that raises).

    A slab -- a box averaged along one axis, whose "path" is the band it was
    averaged over rather than a line anywhere -- says so instead
    (:func:`_band_note`), when the along coordinate's attrs record it.
    """
    from ocean_skill.align import ALONG_DIM

    along_attrs = da[ALONG_DIM].attrs if ALONG_DIM in da.coords else {}
    if along_attrs.get("band_axis") and along_attrs.get("band") is not None:
        return _band_note(along_attrs["band_axis"], along_attrs["band"])
    # The requested path (path_lon/path_lat) names the transect the caller asked
    # for -- "along 0.0°N" for an equator line -- where the snapped lon/lat would
    # name whichever row a coarse source happened to land on (0.5°S).
    lon, lat = _path_positions(da, lon_name, lat_name)
    if lon is None or lat is None:
        return ""
    if lon.size == 0:
        return ""

    def _fmt(lon_v: float, lat_v: float) -> str:
        return (
            f"{abs(lat_v):.1f}°{'N' if lat_v >= 0 else 'S'}, "
            f"{abs(lon_v):.1f}°{'E' if lon_v >= 0 else 'W'}"
        )

    lon_fixed = np.ptp(lon) < 1e-6
    lat_fixed = np.ptp(lat) < 1e-6
    if lon_fixed and not lat_fixed:
        return f"along {abs(lon[0]):.1f}°{'E' if lon[0] >= 0 else 'W'}"
    if lat_fixed and not lon_fixed:
        return f"along {abs(lat[0]):.1f}°{'N' if lat[0] >= 0 else 'S'}"
    return f"{_fmt(lon[0], lat[0])} → {_fmt(lon[-1], lat[-1])}"


#: What a section's x axis may run along: ``"auto"`` picks per path (see
#: :func:`_auto_x_axis`), the others force it. ``"distance"`` is kilometres along the
#: path, ``"lon"``/``"lat"`` the path's own longitude or latitude in degrees.
SECTION_X = ("auto", "distance", "lon", "lat")

#: How much longer a path's span must be in one direction than the other for ``"auto"``
#: to label the section by that coordinate. Both spans are in kilometres, so a path
#: that wanders 120° of longitude and 2° of latitude along the equator reads as
#: longitude, one that runs 45° across the grid does not, and a path in between
#: (neither direction at least twice the other) falls back to distance rather than
#: dressing a diagonal up as an east-west line. May be tuned.
X_DOMINANCE = 2.0

#: Kilometres per degree: of longitude at the equator (scaled by cos(latitude)), and of
#: latitude (averaged over the globe -- good to a fraction of a percent, which is all a
#: dominance test needs).
_KM_PER_DEG_LON = 111.32
_KM_PER_DEG_LAT = 110.57


def check_section_x(x) -> None:
    """Refuse an ``x`` / ``section_x`` that is not one of :data:`SECTION_X`."""
    if not isinstance(x, str) or x not in SECTION_X:
        raise ValueError(
            f"section_x={x!r} is not a section x axis; expected one of {SECTION_X}. "
            '"auto" picks longitude or latitude when the path runs mostly east-west or '
            'north-south, else distance along the path; "distance" always uses '
            'kilometres along the path; "lon"/"lat" force that coordinate.'
        )


def _path_positions(da, lon_name: str | None, lat_name: str | None):
    """Return ``(lon, lat)`` 1-D float arrays along the path, ``None`` where absent.

    The *requested* path positions (``path_lon``/``path_lat`` on the along dimension,
    see :func:`ocean_skill.transect.path_of`) win over the snapped ``lon``/``lat``:
    snapping a path to a model's grid jitters the coordinate that was meant to stay
    fixed (an equatorial line's latitude zigzagging by a cell), which would read as a
    path that doubles back. A coordinate that is not one-dimensional along the path
    -- a slab's averaged-out axis, say -- is reported as absent.
    """
    from ocean_skill.align import ALONG_DIM

    def _read(*names):
        for name in names:
            if name is None or name not in da.coords:
                continue
            coord = da[name]
            if coord.dims == (ALONG_DIM,):
                return np.asarray(coord, dtype="float64")
        return None

    return _read("path_lon", lon_name), _read("path_lat", lat_name)


def _strictly_monotonic(values: np.ndarray | None) -> bool:
    """Whether ``values`` has two or more finite entries that only ever rise or fall."""
    if values is None or values.size < 2 or not np.all(np.isfinite(values)):
        return False
    steps = np.diff(values)
    return bool(np.all(steps > 0) or np.all(steps < 0))


def _auto_x_axis(lon: np.ndarray | None, lat: np.ndarray | None) -> str:
    """Pick ``"lon"``, ``"lat"`` or ``"distance"`` for a path's x axis.

    ``lon`` is the path's longitude already unwrapped across the antimeridian. A
    coordinate is chosen when it runs strictly one way along the path *and* its span
    in kilometres is at least :data:`X_DOMINANCE` times the other direction's -- so
    degrees of longitude along a line that is mostly east-west, never along one that
    doubles back (the same longitude would label two places) or runs diagonally. Any
    shortfall -- too few points, NaNs, a missing coordinate -- is ``"distance"``.
    """
    if not (_strictly_monotonic(lon) or _strictly_monotonic(lat)):
        return "distance"
    finite_lat = lat[np.isfinite(lat)] if lat is not None else np.empty(0)
    mean_lat = float(finite_lat.mean()) if finite_lat.size else 0.0
    lon_km = (
        abs(float(lon[-1] - lon[0])) * _KM_PER_DEG_LON * np.cos(np.radians(mean_lat))
        if lon is not None and lon.size >= 2
        else 0.0
    )
    lat_km = (
        abs(float(lat[-1] - lat[0])) * _KM_PER_DEG_LAT
        if lat is not None and lat.size >= 2
        else 0.0
    )
    if _strictly_monotonic(lon) and lon_km >= X_DOMINANCE * lat_km:
        return "lon"
    if _strictly_monotonic(lat) and lat_km >= X_DOMINANCE * lon_km:
        return "lat"
    return "distance"


def prepare_section(
    da: xr.DataArray, x: str = "auto"
) -> tuple[xr.DataArray, SectionGeometry]:
    """Return ``(field, geometry)``: ``da`` with ``depth``/``distance`` coordinates.

    ``da`` must be exactly two-dimensional: :data:`ocean_skill.align.ALONG_DIM` and
    one vertical axis — either ``z`` (fixed depths, from
    :func:`ocean_skill.roms.to_depth`) or the model's native ``s_rho``/``s_w``,
    carrying a 2-D ``z_rho``/``z_w`` coordinate for its true depth. Both read as
    negative-down (the model's own convention); the returned ``depth`` coordinate
    is flipped to positive-down, which is what a reader — and every other depth
    label in this package (see ``facet_labels``' own ``abs()``) — already expects.
    The two renderers then each invert their y-axis once, so 0 m draws at the top
    and the seafloor at the bottom.

    The returned field carries two new coordinates, ``depth`` and ``distance``,
    both broadcast to its own two dimensions -- the along-path coordinate is
    renamed from :data:`~ocean_skill.align.ALONG_DIM` (rather than reused under
    that name) because a *dimension* coordinate must be one-dimensional, and this
    one is not once the vertical axis is native s-levels. A renderer draws
    ``x=geometry.x_name, y=geometry.y_name`` against the returned field and never
    needs to know which kind of vertical axis it got.

    ``x`` says what runs along the x axis (:data:`SECTION_X`), and the coordinate
    named ``distance`` holds it -- kilometres, or degrees when a longitude or latitude
    is drawn instead:

    * ``"auto"`` (default): a slab (a box averaged along one horizontal axis, whose
      along coordinate carries ``axis_coord``) draws its surviving coordinate. A path
      draws longitude when it runs mostly east-west, latitude when mostly
      north-south, and distance along the path otherwise (:func:`_auto_x_axis`), read
      off the *requested* path positions (``path_lon``/``path_lat``) when the field
      has them and its snapped ``lon``/``lat`` when not. Longitude is unwrapped across
      the antimeridian, so a path from 170°E to 170°W reads 170 to 190 rather than
      jumping.
    * ``"distance"``: kilometres along the path, a slab included.
    * ``"lon"`` / ``"lat"``: force that coordinate; raises if it does not run strictly
      one way along the path (a path that doubles back would label two places
      alike).
    """
    from ocean_skill.align import ALONG_DIM, _lat_name, _lon_name
    from ocean_skill.cf import find_coord

    check_section_x(x)
    if ALONG_DIM not in da.dims:
        raise ValueError(
            f"prepare_section expects a field with an {ALONG_DIM!r} dimension "
            f"(see ocean_skill.align.path_of) -- got dims {sorted(da.dims)}."
        )
    extra = [d for d in da.dims if d != ALONG_DIM]
    if len(extra) != 1:
        raise ValueError(
            "prepare_section expects exactly one vertical axis beside "
            f"{ALONG_DIM!r} -- got {sorted(da.dims)}."
        )
    vertical = extra[0]

    # The native-s aux depth (z_rho, or its w-level counterpart z_w) located via
    # find_coord rather than a hardcoded "z_rho" literal, so a differently-cased
    # spelling or z_w itself is recognized the same way every other vertical lookup
    # in this package is (see ocean_skill.vocabulary.COORD_VOCABULARY["Z"]). Scoped to
    # a coordinate that actually varies along `vertical` (not merely one that ranks
    # earlier in the fallback list) -- a dataset carrying both z_rho and z_w, with
    # only one of them riding on this field's own vertical dimension, must not pick
    # the wrong one.
    aux_depth = None if vertical == "z" else find_coord(da, "vertical")
    native_s = (
        aux_depth is not None
        and vertical in aux_depth.dims
        and str(aux_depth.name) != vertical
    )
    depth_source = aux_depth if native_s else da[vertical]
    if native_s or vertical == "z":
        positive_down = False
    else:
        # An observational axis (WOA's/GLORYS's own "depth", "lev", ...) arrives
        # as the product reports it, positive-down -- there is no model convention
        # to hold it to, and no earlier step that renamed it onto "z" and flipped
        # it (a *comparison* lane gets that treatment in _align_along_path; a
        # bare Field never passes through there). Read as positive-down when it
        # says so (CF's `positive: down`) or, with no word either way, when it
        # has no negative value; an axis that says `positive: up`, or runs
        # negative, is the model's own negative-down and is negated below.
        positive = str(depth_source.attrs.get("positive", "")).lower()
        values = np.asarray(depth_source, dtype="float64")
        positive_down = positive == "down" or (
            positive != "up" and bool(np.nanmin(values) >= 0)
        )
    if not native_s and not positive_down and float(
        np.nanmax(np.asarray(depth_source))
    ) > 0:
        # Fixed-z only: this coordinate is about to be negated below, on the
        # assumption that it already reads negative-down (the model's own
        # convention, from roms.to_depth). A positive-down "z" reaching here
        # instead -- a lane renamed onto "z" without being sign-flipped first --
        # would silently draw upside-down: negative tick values, the seafloor at
        # the top. (An observational axis under its own name is handled just
        # above; "z" is the one name that promises negative-down.) Native-s is
        # exempt, since z_rho under a positive free surface is legitimately
        # slightly positive right at the surface, not a sign-convention bug.
        raise ValueError(
            f"prepare_section expects {vertical!r} to be negative-down (the "
            "model's own convention), but its largest value is positive -- "
            "this coordinate needs to be sign-flipped to negative-down before "
            "reaching here, not drawn as given."
        )
    depth = (depth_source if positive_down else -depth_source).rename("depth")
    depth.attrs["units"] = "m"

    # Native-s only: a land column's z_rho/z_w would be NaN if built from the
    # ordinary (land-masked) zeta -- pcolormesh tolerates that in the *data* it
    # colours (that is how a below-bathymetry cell draws grey, see
    # ax.set_facecolor("0.85") in both renderers) but refuses it in the x/y
    # coordinate arrays it meshes against. There is nothing to fill here: the
    # section's z_rho/z_w already comes in finite everywhere, land included, from
    # ocean_skill.comparison._prepare calling roms.add_depth_coord/
    # add_interface_coord with zero_zeta=True for exactly this reason -- the
    # seafloor under land is still known (it's h, never masked), so the mesh has
    # no business going NaN there in the first place.

    lon_name, lat_name = _lon_name(da), _lat_name(da)
    # What the x axis shows. A slab (see ocean_skill.comparison._slab_to_section)
    # names the coordinate its x axis should be -- the one that survived the averaging
    # -- and draws it in degrees; a path section's x is chosen from where the path
    # runs (_auto_x_axis) or forced by the caller. Same coordinate name either way
    # ("distance"), so a renderer needs no branch for which it got.
    slab_axis = da[ALONG_DIM].attrs.get("axis_coord")
    slab_name = {"lat": lat_name, "lon": lon_name}.get(slab_axis)
    degrees = None
    if x in ("auto", slab_axis) and slab_name:
        x_axis = slab_axis
        degrees = np.asarray(da[slab_name], dtype="float64")
    elif x == "distance":
        x_axis = "distance"
    else:
        lon, lat = _path_positions(da, lon_name, lat_name)
        if lon is not None and np.all(np.isfinite(lon)):
            lon = np.unwrap(lon, period=360.0)
        x_axis = _auto_x_axis(lon, lat) if x == "auto" else x
        if x_axis in ("lon", "lat"):
            degrees = lon if x_axis == "lon" else lat
            if x != "auto" and not _strictly_monotonic(degrees):
                name = "longitude" if x_axis == "lon" else "latitude"
                why = (
                    "absent from this field"
                    if degrees is None
                    else "not monotonic (it doubles back, repeats or has gaps)"
                )
                raise ValueError(
                    f'section_x="{x_axis}" needs the path\'s {name} to run strictly '
                    f"one way along it, but it is {why}. Use section_x=\"distance\" "
                    "to draw kilometres along the path instead."
                )
    if x_axis in ("lat", "lon"):
        distance = xr.DataArray(
            degrees,
            dims=ALONG_DIM,
            coords={ALONG_DIM: da[ALONG_DIM]},
            name="distance",
        )
        distance.attrs["units"] = "degrees_north" if x_axis == "lat" else "degrees_east"
        x_label = "latitude (°N)" if x_axis == "lat" else "longitude (°E)"
    else:
        # rebuilt rather than renamed: a renamed coordinate shares its attrs with the
        # caller's own along coordinate, which the bookkeeping below (units, x_axis)
        # must not leak into
        distance = xr.DataArray(
            np.asarray(da[ALONG_DIM]),
            dims=ALONG_DIM,
            coords={ALONG_DIM: da[ALONG_DIM]},
            name="distance",
            attrs=dict(da[ALONG_DIM].attrs),
        )
        distance.attrs["units"] = da[ALONG_DIM].attrs.get("units", "km")
        x_label = "distance along transect (km)"
    depth2d, distance2d, values2d = xr.broadcast(depth, distance, da)
    order = tuple(values2d.dims)
    depth2d = depth2d.transpose(*order)
    distance2d = distance2d.transpose(*order)
    result = values2d.assign_coords(depth=depth2d, distance=distance2d)

    geometry = SectionGeometry(
        x_name="distance",
        y_name="depth",
        x_label=x_label,
        y_label="depth (m)",
        native_s=native_s,
        path_note=_path_note(da, lon_name, lat_name),
        x_axis=x_axis,
    )
    return result, geometry


def prepare_section_row(
    aligned: dict[str, xr.DataArray] | xr.Dataset, x: str = "auto"
) -> tuple[dict[str, xr.DataArray], SectionGeometry]:
    """Return ``(values, geometry)`` for a test | reference | difference row.

    ``aligned`` is a comparison's aligned trio — indexed by ``"test"``,
    ``"reference"``, ``"difference"`` — not necessarily an :class:`xr.Dataset`
    (test fixtures pass a plain ``dict`` with the same three keys, and this
    function only ever indexes it, never calls a Dataset-only method, so both
    work identically).

    Every lane must already carry a ``"z"`` dimension: a comparison lane is
    always fixed-depth (see :func:`ocean_skill.align._align_along_path`,
    which renames the reference's own vertical dim onto ``"z"`` and adopts the
    test's coordinate values) — native s-levels have no shared axis across two
    different sources to hold a ``test - reference`` difference on, so a
    section that still carries ``s_rho``/``s_w`` here is a caller error, not
    something this function can silently paper over.

    Each lane is run through :func:`prepare_section` independently, but the
    trio is aligned by construction (same ``z``, same ``along``/``distance``,
    same ``lon``/``lat`` -- see the alignment function's docstring), so their
    geometries are identical; only the test lane's is returned, matching
    :func:`ocean_skill.plot.matplotlib_renderer._field_row`'s single-geometry
    contract for a row of panels.

    ``x`` is :func:`prepare_section`'s. It is resolved on the test lane and the answer
    handed to the other two as a forced choice, so the three panels can never disagree
    about what runs along their shared x axis.
    """
    check_section_x(x)
    if "z" not in aligned["test"].dims:
        raise ValueError(
            "prepare_section_row expects a fixed-depth 'z' dimension on every "
            f"lane -- got dims {sorted(aligned['test'].dims)} on the test lane. "
            "A comparison section is always fixed-depth (see "
            "ocean_skill.align._align_along_path); pass select={'depth': [...]} "
            "rather than native s-levels."
        )

    values: dict[str, xr.DataArray] = {}
    geometry: SectionGeometry | None = None
    for lane in ("test", "reference", "difference"):
        values[lane], lane_geometry = prepare_section(aligned[lane], x)
        if lane == "test":
            geometry = lane_geometry
            x = lane_geometry.x_axis
    assert geometry is not None
    return values, geometry


# --- contour overlay: lines of a second variable on a filled section ------------------

#: About how many contour lines an overlay draws when it is just switched on. Six reads
#: as a handful of isotherms to follow rather than a hatching over the fill.
DEFAULT_CONTOUR_LEVELS = 6

#: The target number of filled bands :func:`fill_edges` cuts a section into when
#: ``fill_levels`` does not say. Chosen by eye against the paper's figure -- narrow
#: enough that the fill reads as a smooth field, wide enough that a band edge is
#: still a line the eye can follow -- and may be tuned.
DEFAULT_FILL_BANDS = 50

#: What a band's width may be, as ``1/k`` of the colour bar's tick step. Every ``k``
#: is a whole number, so a tick is always on a band edge.
_FILL_DIVISORS = (1, 2, 5, 10, 20)

#: How closely two section meshes must agree to count as the same one. Tight: one
#: select/aggregate gives bit-identical meshes, so this only forgives float32-versus-
#: float64 round-off, never a shifted level or a different position.
_MESH_TOL = 1e-6

_CONTOUR_ACCEPTED = (
    f"Accepted: True (about {DEFAULT_CONTOUR_LEVELS} round levels), an int n >= 1 "
    "(about n levels), a list/tuple/array of numbers (exactly those levels), or "
    "False/None (no lines)."
)
_FILL_ACCEPTED = (
    f"Accepted: None (about {DEFAULT_FILL_BANDS} round bands), an int n >= 1 (about n "
    "bands), or a list/tuple/array of at least two numbers (exactly those band edges)."
)

_UNIT_TEXT = {"km": " km", "m": " m", "degrees_north": "°N", "degrees_east": "°E"}


def _unit_text(units) -> str:
    """Return a ``units`` attribute as it reads after a number: ``" km"``, ``"°N"``."""
    units = str(units or "")
    return _UNIT_TEXT.get(units, f" {units}" if units else "")


def _span_text(values: np.ndarray) -> str:
    """``"0–500"``: the finite range of ``values``, or ``"no finite values"``."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return "no finite values"
    # "+ 0.0" turns a negated zero (-0.0, which "{:g}" spells "-0") back into 0
    return f"{float(finite.min()) + 0.0:g}–{float(finite.max()) + 0.0:g}"


def _same_mesh(a: np.ndarray, b: np.ndarray) -> bool:
    """Whether two coordinate arrays have one shape and agree to within round-off."""
    return a.shape == b.shape and bool(
        np.allclose(a, b, rtol=_MESH_TOL, atol=_MESH_TOL, equal_nan=True)
    )


def _along_positions(prepared: xr.DataArray, vertical: str) -> np.ndarray:
    """Return the position of every column as 1-D values (``distance`` repeats down)."""
    axis = prepared.dims.index(vertical)
    return np.take(np.asarray(prepared["distance"], dtype="float64"), 0, axis=axis)


def _mismatch_text(
    noun: str,
    where: str,
    overlay: tuple[int, str],
    panel: tuple[int, str],
) -> str:
    """``"overlay 37 levels 0–500 m, panel 20 levels 0–500 m"``, plus a hint if alike.

    ``overlay``/``panel`` are each ``(count, "range with units")``. When both read the
    same -- same count, same range -- the two still differ *inside* the range (one
    spaced evenly, the other not), which the bare numbers would not show, so the text
    says so rather than leaving a reader to look for a difference that is not printed.
    """
    text = (
        f"overlay {overlay[0]} {noun} {overlay[1]}, panel {panel[0]} {noun} {panel[1]}"
    )
    if overlay == panel:
        text += f" -- the same count and range, but not at the same {where}"
    return text


def prepare_overlay(
    overlay: xr.DataArray, panel: xr.DataArray, x: str = "auto"
) -> xr.DataArray:
    """Return ``overlay`` prepared on the same section grid as ``panel``, or raise.

    An overlay is the second variable of a section figure -- the isotherms drawn over
    the phosphate fill. Its lines are read against the fill's own axes, so it has to
    sit on the fill's mesh: the same positions along the section and the same depth
    levels. This runs :func:`prepare_section` on it (so it gets the same axis
    conventions as the panel, a native-s overlay's depth flipped to positive-down
    included), puts it in the panel's dimension order, and checks the two meshes
    agree. ``x`` is the panel's own (the ``section_x`` it was prepared with), so a
    forced ``"distance"`` or ``"lon"`` applies to the lines as it did to the fill; under
    ``"auto"`` the two choose alike for two variables cut from one path, and an overlay
    that is a different kind of section from its panel (a latitude slab against a
    transect) is reported as the mesh mismatch it is.

    It never regrids. Two variables of one source, cut with one ``select`` and
    ``aggregate``, always share a mesh, so a mismatch means the two were built
    differently -- a temperature on 37 levels under a phosphate on 20, say -- and
    resampling one onto the other would draw lines where the data never put them, with
    nothing on the figure to say so. A refusal that names the axis that differs, and
    how to fix it, is the honest answer. The two fields' NaN masks may differ (a
    variable can be missing where another is not); only the mesh must match.

    Parameters
    ----------
    overlay
        The raw section of the line variable: an ``along`` dimension plus one
        vertical dimension, ``z`` or the native ``s_rho`` with its 2-D ``z_rho`` --
        exactly what :func:`prepare_section` takes.
    panel
        The fill's already-prepared values: :func:`prepare_section`'s own return,
        carrying 2-D ``distance`` and ``depth`` coordinates.
    x
        What the panel's x axis was prepared with (:data:`SECTION_X`).

    Returns
    -------
    xarray.DataArray
        The prepared overlay, its dimensions in the panel's order. Its ``distance``
        and ``depth`` coordinates are the panel's own arrays (just checked equal to
        within round-off), so a renderer draws fill and lines on one mesh to the last
        bit.

    Raises
    ------
    ValueError
        If ``panel`` is not a prepared section, or the two do not share a vertical
        axis, depth levels, or positions along the section. The message says which
        axis differs, with the count and range on each side.
    """
    from ocean_skill.align import ALONG_DIM

    if ALONG_DIM not in panel.dims or not {"distance", "depth"} <= set(panel.coords):
        raise ValueError(
            "prepare_overlay expects the panel to be prepare_section's own return "
            f"(an {ALONG_DIM!r} dimension and 2-D 'distance' and 'depth' coordinates) "
            f"-- got dims {sorted(panel.dims)} and coordinates {sorted(panel.coords)}."
        )
    prepared, _ = prepare_section(overlay, x)
    vertical_o = next(d for d in prepared.dims if d != ALONG_DIM)
    vertical_p = next(d for d in panel.dims if d != ALONG_DIM)
    advice = (
        "Build both with the same select/aggregate so they share one section grid; "
        "nothing is regridded here, since a line drawn from resampled values would "
        "sit where the data never put it."
    )
    if vertical_o != vertical_p:
        raise ValueError(
            "The overlay and the panel are drawn on different meshes: vertical axes "
            f"differ (overlay on {vertical_o!r} with {prepared.sizes[vertical_o]} "
            f"levels, panel on {vertical_p!r} with {panel.sizes[vertical_p]} levels). "
            + advice
        )
    prepared = prepared.transpose(*panel.dims)

    n_levels = (prepared.sizes[vertical_o], panel.sizes[vertical_p])
    n_along = (prepared.sizes[ALONG_DIM], panel.sizes[ALONG_DIM])
    depth_o = np.asarray(prepared["depth"], dtype="float64")
    depth_p = np.asarray(panel["depth"].transpose(*panel.dims), dtype="float64")
    along_o = _along_positions(prepared, vertical_o)
    along_p = _along_positions(panel, vertical_p)
    units_o = _unit_text(prepared["distance"].attrs.get("units"))
    units_p = _unit_text(panel["distance"].attrs.get("units"))

    # Depth is compared value by value only when the two have as many columns: with
    # different numbers of them a native-s mesh has nothing to line up against, and
    # the position problem is the one to fix first.
    levels_differ = n_levels[0] != n_levels[1] or (
        n_along[0] == n_along[1] and not _same_mesh(depth_o, depth_p)
    )
    along_differs = (
        n_along[0] != n_along[1]
        or units_o != units_p
        or not _same_mesh(along_o, along_p)
    )
    problems = []
    if levels_differ:
        text = _mismatch_text(
            "levels",
            "depths",
            (n_levels[0], _span_text(depth_o) + " m"),
            (n_levels[1], _span_text(depth_p) + " m"),
        )
        problems.append(f"depth levels differ ({text})")
    if along_differs:
        text = _mismatch_text(
            "positions",
            "positions",
            (n_along[0], _span_text(along_o) + units_o),
            (n_along[1], _span_text(along_p) + units_p),
        )
        problems.append(f"positions along the section differ ({text})")
    if problems:
        raise ValueError(
            "The overlay and the panel are drawn on different meshes: "
            + "; ".join(problems)
            + ". "
            + advice
        )

    # The panel's own mesh arrays, not the overlay's near-identical ones.
    mesh = {
        name: (
            prepared.dims,
            panel[name].transpose(*prepared.dims).values,
            prepared[name].attrs,
        )
        for name in ("distance", "depth")
    }
    return prepared.assign_coords(mesh)


# --- which lines, and where they run --------------------------------------------------


def _as_float_array(values) -> np.ndarray:
    """Return ``values`` as float64 with masked entries as NaN.

    Takes a DataArray, an ndarray or a numpy masked array alike, so a mask a caller
    set is honoured the way a NaN is rather than read as the data under it.
    """
    if isinstance(values, np.ma.MaskedArray):
        return values.astype("float64").filled(np.nan)
    return np.asarray(values, dtype="float64")


def _finite_span(arrays) -> tuple[float, float] | None:
    """Return ``(min, max)`` over every finite value in ``arrays``, or ``None``."""
    if arrays is None:
        return None
    if isinstance(arrays, xr.DataArray | np.ndarray):
        arrays = (arrays,)
    lo, hi = np.inf, -np.inf
    for array in arrays:
        values = _as_float_array(array)
        values = values[np.isfinite(values)]
        if values.size:
            lo, hi = min(lo, float(values.min())), max(hi, float(values.max()))
    return (lo, hi) if lo <= hi else None


def _check_count(spec, accepted: str) -> int:
    """``spec`` as a count of lines or bands, or ``ValueError`` if it is below one."""
    if spec < 1:
        raise ValueError(f"{spec!r} is a count below 1. {accepted}")
    return int(spec)


def _explicit_levels(spec, accepted: str) -> np.ndarray:
    """``spec`` as sorted, unique, finite float levels -- or raise saying what works.

    A list, tuple or array of numbers is the only form that gets here; a string (itself
    a sequence), a dict, a float, a list of strings or booleans, or nested lists is a
    ``TypeError``, and a list holding NaN or infinity a ``ValueError``.
    """
    if isinstance(spec, str | bytes) or not isinstance(spec, Sequence | np.ndarray):
        raise TypeError(f"{spec!r} is not a valid levels spec. {accepted}")
    try:
        levels = np.asarray(spec)
    except ValueError:  # ragged nesting, e.g. [[1, 2], [3]]
        levels = np.empty((), dtype=object)
    if levels.ndim != 1 or levels.dtype.kind not in "iuf":
        raise TypeError(f"{spec!r} is not a flat list of numbers. {accepted}")
    levels = levels.astype("float64")
    if not np.isfinite(levels).all():
        raise ValueError(f"{spec!r} holds a level that is not finite. {accepted}")
    return np.unique(levels)


def _round_levels(lo: float, hi: float, count: int) -> tuple[float, ...]:
    """About ``count`` round levels strictly inside ``(lo, hi)``, noise-free."""
    from matplotlib.ticker import MaxNLocator

    ticks = MaxNLocator(nbins=count, steps=[1, 2, 5, 10]).tick_values(lo, hi)
    if len(ticks) < 2 or not ticks[1] > ticks[0]:
        return ()
    step = float(ticks[1] - ticks[0])
    tol = step * 1e-9
    levels = (_clean(float(tick), step) for tick in ticks)
    return tuple(v for v in levels if lo + tol < v < hi - tol)


def contour_levels(spec, arrays) -> tuple[float, ...]:
    """Return the contour-line levels for ``spec``, pooled over every overlay array.

    Levels are decided once for the whole figure, not per panel: a test | reference |
    difference row (or several rows) must show the same isotherms in each, or the eye
    is left comparing 10, 15 and 20 °C lines in one panel against 12.5, 15 and 17.5 in
    the next. So ``arrays`` is every overlay array the figure draws, and the automatic
    rule reads the range they cover together.

    Parameters
    ----------
    spec
        ``True``: about 6 round levels. An ``int`` ``n >= 1``: about ``n``. A list,
        tuple or array of numbers: exactly those, sorted and de-duplicated, whatever
        the data holds. ``False`` or ``None``: no lines.
    arrays
        The overlay arrays to pool -- an iterable of arrays or DataArrays, or just
        one. Only finite values count; NaN is ignored.

    Returns
    -------
    tuple of float
        Ascending levels. The automatic ones are 1, 2 or 5 times a power of ten -- the
        rule a colour bar's ticks follow, matplotlib's ``MaxNLocator`` over
        ``steps=[1, 2, 5, 10]`` -- free of float noise (``0.6``, not
        ``0.6000000000000001``), and strictly inside the pooled range: a line at the
        data's own minimum or maximum would be a scrap of the field's edge. "About
        ``n``" means at most ``n``, and sometimes fewer, since round values rarely
        divide a range into exactly ``n`` pieces. Empty when there is no finite data
        or the range is flat -- nothing to contour, and not an error.

    Raises
    ------
    TypeError
        If ``spec`` is a string, a float, a dict, or a sequence that is not a flat list
        of numbers.
    ValueError
        If ``spec`` is an int below 1 or a list holding NaN or infinity. Every message
        spells out the forms that are accepted.
    """
    if isinstance(spec, bool | np.bool_):
        if not spec:
            return ()
        count = DEFAULT_CONTOUR_LEVELS
    elif spec is None:
        return ()
    elif isinstance(spec, numbers.Integral):
        count = _check_count(spec, _CONTOUR_ACCEPTED)
    else:
        return tuple(float(v) for v in _explicit_levels(spec, _CONTOUR_ACCEPTED))
    span = _finite_span(arrays)
    return () if span is None else _round_levels(*span, count)


@dataclass(frozen=True)
class ContourLevel:
    """One contour level's lines: where they run, and where its label goes.

    ``lines`` is every separate piece at ``level`` -- an isotherm breaks at a seamount
    or the edge of the data, and can close on itself -- each ``(N, 2)`` of ``x, y``
    vertices (a closed one repeats its first vertex last). ``anchor`` is the one point
    a label for the level goes: a vertex of the longest piece, at the middle of its
    length, so the label sits on the line a reader is most likely to follow and not at
    the panel's edge. Both are empty (``lines=()``, ``anchor=None``) for a level that
    crosses nothing.
    """

    level: float
    lines: tuple[np.ndarray, ...]
    anchor: tuple[float, float] | None


def _extent(values: np.ndarray) -> float:
    """Return the peak-to-peak range of ``values``, or 1 if it has none to divide by."""
    extent = float(np.ptp(values)) if values.size else 0.0
    return extent if extent > 0 and math.isfinite(extent) else 1.0


def _label_anchor(
    lines: Sequence[np.ndarray], scale: tuple[float, float]
) -> tuple[float, float] | None:
    """Return the vertex at the arclength midpoint of the longest line, or ``None``.

    Length is measured with each axis divided by its extent (``scale``) -- as the
    panel is drawn, full width by full height -- rather than in raw data units, where
    a section's kilometres against its metres (or degrees against metres) would let
    whichever axis has the bigger numbers decide where the middle of a line is: an
    isotherm that runs level across the whole section and drops steeply at one end
    would put its label at the drop.
    """
    best, best_cum = None, None
    for line in lines:
        steps = np.hypot(np.diff(line[:, 0]) / scale[0], np.diff(line[:, 1]) / scale[1])
        cum = np.concatenate(([0.0], np.cumsum(steps)))
        if best_cum is None or cum[-1] > best_cum[-1]:
            best, best_cum = line, cum
    if best is None or best_cum is None:
        return None
    middle = int(np.argmin(np.abs(best_cum - best_cum[-1] / 2.0)))
    return float(best[middle, 0]), float(best[middle, 1])


def contour_paths(x, y, z, levels) -> list[ContourLevel]:
    """Return the contour lines of ``z`` at each of ``levels``, one entry per level.

    The interactive renderer draws its overlay from these: bokeh has no contour
    primitive, and re-contouring with some other library would let the two renderers
    disagree about where a line runs. This is contourpy, the engine behind matplotlib's
    own ``ax.contour``, run on the same arrays, so the lines are the ones the static
    figure draws, vertex for vertex.

    Parameters
    ----------
    x, y, z
        2-D arrays of one shape -- a prepared section's ``distance``, ``depth`` and
        values. ``y`` may vary along both axes (native s-levels, where depth really
        does change along the path). A point where ``z`` is NaN is masked, so lines
        stop at it as matplotlib's do; so is one where ``x`` or ``y`` is not finite,
        which would otherwise put a NaN vertex on a line.
    levels
        The values to contour, for instance from :func:`contour_levels`.

    Returns
    -------
    list of ContourLevel
        One entry per requested level, in the order given, even for a level that
        crosses nothing (empty ``lines``, ``anchor`` ``None``) -- a caller never has
        to ask which levels came back. A field with fewer than two points along either
        axis, or no finite point, has no lines at all.

    Raises
    ------
    ValueError
        If ``x``, ``y`` and ``z`` are not 2-D arrays of one shape.

    Notes
    -----
    The label ``anchor`` is placed by length measured in the panel's own proportions
    (each axis over its extent), not in raw data units; see :class:`ContourLevel`.
    """
    import contourpy

    x, y, z = _as_float_array(x), _as_float_array(y), _as_float_array(z)
    if not (x.ndim == y.ndim == z.ndim == 2 and x.shape == y.shape == z.shape):
        raise ValueError(
            "contour_paths needs x, y and z as 2-D arrays of one shape (a prepared "
            f"section's distance, depth and values) -- got shapes {x.shape}, "
            f"{y.shape} and {z.shape}."
        )
    wanted = [float(v) for v in np.atleast_1d(np.asarray(levels, dtype="float64"))]
    usable = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if min(z.shape) < 2 or not usable.any():
        return [ContourLevel(v, (), None) for v in wanted]

    scale = (_extent(x[usable]), _extent(y[usable]))
    generator = contourpy.contour_generator(
        np.where(usable, x, 0.0),
        np.where(usable, y, 0.0),
        np.ma.masked_array(np.where(usable, z, 0.0), mask=~usable),
        line_type=contourpy.LineType.Separate,
    )
    out = []
    for level in wanted:
        lines = tuple(
            np.asarray(line, dtype="float64") for line in generator.lines(level)
        )
        out.append(ContourLevel(level, lines, _label_anchor(lines, scale)))
    return out


# --- where the filled bands break -----------------------------------------------------


def _flat_edges(lo: float, hi: float) -> np.ndarray:
    """Two edges bracketing a flat or reversed range ``lo >= hi``: one band."""
    centre = 0.5 * (lo + hi)
    half = 0.05 * abs(centre) or 0.05  # as matplotlib's autoscaler widens a flat range
    return np.array([centre - half, centre + half])


def fill_edges(vmin, vmax, *, log: bool, fill_levels=None) -> np.ndarray:
    """Return the band edges for a filled-contour fill from ``vmin`` to ``vmax``.

    ``contourf`` cuts its colour scale into bands, and where it cuts decides how the
    figure reads. Edges at ``np.linspace(vmin, vmax, 21)`` fall at 0.15, 0.3, 0.45 --
    none of them where the colour bar has a tick, so the bar and the picture never
    quite line up and a band edge can only be read as "about 0.3". These edges are
    round, and every colour-bar tick (:func:`ocean_skill.plot._colorbar.colorbar_ticks`)
    sits on one.

    **Linear.** The band width is the bar's tick step divided by ``k`` in
    ``(1, 2, 5, 10, 20)`` -- a whole fraction, so the ticks are always band edges --
    with the ``k`` whose number of bands lands closest to the target
    (:data:`DEFAULT_FILL_BANDS`, or ``fill_levels`` when that is an int; a tie goes
    to the finer ``k``). The edges are the multiples of that width inside
    ``[vmin, vmax]``, float-noise free, plus ``vmin`` and ``vmax`` themselves when
    they are not multiples: a limit the user pinned may not be round, and the fill
    must still span the colour range exactly.

    **Log** (``log=True`` and ``vmin > 0``). ``np.geomspace(vmin, vmax, n + 1)`` for
    ``n`` bands, as before; log sections are rare and a log bar's ticks (1, 2, 5 times
    a power of ten) are not edges of this.

    Parameters
    ----------
    vmin, vmax
        The colour range the fill spans.
    log
        Whether the colour scale is logarithmic.
    fill_levels
        ``None`` (or ``True``): the default band count. An ``int`` ``n >= 1``: about
        ``n`` bands (the nearest count the round widths allow, so a small ``n`` gives
        the coarsest width and a large one the finest). A list, tuple or array of at
        least two numbers: exactly those edges, sorted and de-duplicated, whatever the
        range.

    Returns
    -------
    numpy.ndarray
        Ascending float edges, at least two. A range with nothing to band --
        non-finite, or ``vmin >= vmax`` -- still returns edges ``contourf`` accepts
        rather than the "levels must be increasing" error a constant field would give:
        a flat range ``c`` comes back as ``[c - h, c + h]`` (``h`` five percent of
        ``|c|``, as matplotlib's own autoscaler widens it) and a non-finite one as
        ``[0, 1]``, so the panel draws as one band, or as nothing.

    Raises
    ------
    TypeError, ValueError
        For a ``fill_levels`` that is not one of the forms above; the message says
        which are accepted.
    """
    if isinstance(fill_levels, bool | np.bool_):
        if not fill_levels:
            raise TypeError(
                f"fill_levels={fill_levels!r} is not valid. {_FILL_ACCEPTED}"
            )
        target = DEFAULT_FILL_BANDS
    elif fill_levels is None:
        target = DEFAULT_FILL_BANDS
    elif isinstance(fill_levels, numbers.Integral):
        target = _check_count(fill_levels, _FILL_ACCEPTED)
    else:
        edges = _explicit_levels(fill_levels, _FILL_ACCEPTED)
        if edges.size < 2:
            raise ValueError(
                f"fill_levels={fill_levels!r} gives fewer than two distinct band "
                f"edges. {_FILL_ACCEPTED}"
            )
        return edges

    try:
        lo, hi = float(vmin), float(vmax)
    except (TypeError, ValueError):
        lo = hi = math.nan
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return np.array([0.0, 1.0])
    if lo >= hi:
        return _flat_edges(lo, hi)
    if log and lo > 0:
        return np.geomspace(lo, hi, target + 1)

    step = tick_step(lo, hi)
    bands = {k: k * (hi - lo) / step for k in _FILL_DIVISORS}
    # round() so two counts equally far from the target compare as a tie, not by the
    # last bit of a float; -k then sends the tie to the finer width
    divisor = min(bands, key=lambda k: (round(abs(bands[k] - target), 9), -k))
    width = step / divisor
    tol = width * 1e-9
    first, last = math.ceil((lo - tol) / width), math.floor((hi + tol) / width)
    edges = [_clean(i * width, width) for i in range(first, last + 1)]
    if not edges or edges[0] - lo > tol:
        edges.insert(0, lo)
    if hi - edges[-1] > tol:
        edges.append(hi)
    return np.array(edges, dtype="float64")


#: The two ways a section panel is filled: cells as they are, or smooth filled bands.
SECTION_MARKS = ("pcolormesh", "contourf")

#: Overlay contour lines' default colour and width, read by both renderers: black, thin
#: enough to sit on a filled field without hiding it (Fig. 19's isotherms).
CONTOUR_COLOR = "black"
CONTOUR_WIDTH = 0.8


def contour_label(level: float, fmt: str = "%g") -> str:
    """Spell one contour line's label: ``fmt % level`` with a true minus sign.

    Shared so a static ``clabel`` and the interactive text label read the same --
    ``15``, ``34.5``, ``−1`` -- and a caller's ``contour_kwargs={"fmt": "%.1f"}``
    changes both.
    """
    return (fmt % level).replace("-", "\N{MINUS SIGN}")


def check_section_options(
    *,
    has_contours: bool,
    mark: str,
    contour_levels=None,
    contour_kwargs=None,
    fill_levels=None,
    section_x="auto",
) -> None:
    """Refuse a section option with nothing to act on -- one wording, both renderers.

    ``contour_levels=``/``contour_kwargs=`` style the lines ``contours=`` draws, so
    either one without an overlay on any panel would be silently ignored; so would
    ``fill_levels=`` -- the bands of a filled-contour fill -- on a ``pcolormesh``
    panel. Both are refused by name instead, as is a ``mark`` a section cannot draw
    (``None`` is the default, cells), and a ``section_x`` that names no x axis
    (:data:`SECTION_X`).
    """
    check_section_x(section_x)
    if mark is not None and mark not in SECTION_MARKS:
        raise ValueError(
            f"mark={mark!r} is not a section mark; expected one of {SECTION_MARKS}. "
            "(Line marks -- 'line', 'step' -- belong to a series, not a vertical "
            "section.)"
        )
    if not has_contours and (contour_levels is not None or contour_kwargs is not None):
        given = "contour_levels=" if contour_levels is not None else "contour_kwargs="
        raise ValueError(
            f"{given} styles the lines contours= draws, but no panel here has an "
            "overlay -- pass contours= as well (another field or comparison of the "
            "variable to draw as lines, built the same way)."
        )
    if fill_levels is not None and mark != "contourf":
        raise ValueError(
            f"fill_levels= sets the bands of a filled-contour fill, but mark={mark!r} "
            'draws cells -- pass mark="contourf" as well.'
        )


def difference_fill_levels(fill_levels):
    """Return the ``fill_levels`` a row's difference panel takes from its row's.

    A band *count* means the same on any range, so the difference panel shares it. A
    list of band *edges* is in the variable's own units -- phosphate's 0-3, say -- and
    on a difference spanning ±0.4 would fill nothing but its two open ends, so the
    difference panel keeps its own round default instead.
    """
    if fill_levels is None or isinstance(fill_levels, bool | int | np.integer):
        return fill_levels
    return None
