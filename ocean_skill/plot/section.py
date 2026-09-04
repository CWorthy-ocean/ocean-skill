"""Shared geometry for vertical-section plots.

One place both renderers read, so a section cannot look different statically than
interactively — the same reason :mod:`ocean_skill.plot.series` exists for the line
family. :func:`prepare_section` decides every axis convention a section needs once:
which coordinate is "depth" and which sign it reads positive, how the along-path
axis is labelled, and what a title calls the path itself. A renderer's own drawing
function calls this first and then only draws — it makes no convention decisions
of its own.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

__all__ = ["SectionGeometry", "prepare_section", "prepare_section_row"]


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
    """

    x_name: str
    y_name: str
    x_label: str
    y_label: str
    native_s: bool
    path_note: str


def _path_note(da, lon_name: str | None, lat_name: str | None) -> str:
    """``"29.0°N, 94.5°W → 27.5°N, 90.0°W"``, or ``"along 94.5°W"`` if lon is fixed.

    Read off the along-path lon/lat coordinates' own endpoints -- the same
    positive-down-style degree formatting :func:`ocean_skill.plot.series._place_of`
    uses for a station, extended to a pair of points rather than one. Empty when
    the field carries no lon/lat coordinates to read (should not happen for a
    real section, but a title with nothing to say is better than one that raises).
    """
    if lon_name is None or lat_name is None:
        return ""
    lon = np.asarray(da[lon_name], dtype="float64")
    lat = np.asarray(da[lat_name], dtype="float64")
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


def prepare_section(da: xr.DataArray) -> tuple[xr.DataArray, SectionGeometry]:
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
    """
    from ocean_skill.align import ALONG_DIM, _lat_name, _lon_name
    from ocean_skill.cf import find_coord

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
    if not native_s and float(np.nanmax(np.asarray(depth_source))) > 0:
        # Fixed-z only: this coordinate is about to be negated below, on the
        # assumption that it already reads negative-down (the model's own
        # convention, from roms.to_depth). A positive-down coordinate reaching
        # here instead -- an observational "depth"/"lev" axis that was renamed
        # onto "z" without being sign-flipped first -- would silently draw
        # upside-down: negative tick values, the seafloor at the top. Native-s
        # is exempt, since z_rho under a positive free surface is legitimately
        # slightly positive right at the surface, not a sign-convention bug.
        raise ValueError(
            f"prepare_section expects {vertical!r} to be negative-down (the "
            "model's own convention), but its largest value is positive -- "
            "this coordinate needs to be sign-flipped to negative-down before "
            "reaching here, not drawn as given."
        )
    depth = (-depth_source).rename("depth")
    depth.attrs["units"] = "m"

    distance = da[ALONG_DIM].rename("distance")
    distance.attrs["units"] = da[ALONG_DIM].attrs.get("units", "km")

    depth2d, distance2d, values2d = xr.broadcast(depth, distance, da)
    order = tuple(values2d.dims)
    depth2d = depth2d.transpose(*order)
    distance2d = distance2d.transpose(*order)
    result = values2d.assign_coords(depth=depth2d, distance=distance2d)

    lon_name, lat_name = _lon_name(da), _lat_name(da)
    geometry = SectionGeometry(
        x_name="distance",
        y_name="depth",
        x_label="distance along transect (km)",
        y_label="depth (m)",
        native_s=native_s,
        path_note=_path_note(da, lon_name, lat_name),
    )
    return result, geometry


def prepare_section_row(
    aligned: dict[str, xr.DataArray] | xr.Dataset,
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
    """
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
        values[lane], lane_geometry = prepare_section(aligned[lane])
        if lane == "test":
            geometry = lane_geometry
    assert geometry is not None
    return values, geometry
