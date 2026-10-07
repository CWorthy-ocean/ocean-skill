"""What runs along a vertical section's x axis: distance, longitude or latitude.

A section used to label x "distance along transect (km)" for any path, with only
box-averaged slabs reading degrees. :func:`ocean_skill.plot.section.prepare_section` now
looks at where the path runs (``x="auto"``): mostly east-west reads longitude (an
equatorial line), mostly north-south latitude, anything else kilometres along the path;
``section_x=`` on a plot call overrides it. These tests pin the rule on small synthetic
fields, then check that both renderers -- every section family -- act on it.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import ocean_skill as osk
from ocean_skill.align import ALONG_DIM
from ocean_skill.plot import section as sec
from ocean_skill.plot.registry import render
from ocean_skill.plot.section import (
    prepare_overlay,
    prepare_section,
    prepare_section_row,
)
from ocean_skill.plot.spec import PlotSpec
from tests.test_slab_section import VAR, patched_woa  # noqa: F401  (a fixture)

LON_LABEL = "longitude (°E)"
LAT_LABEL = "latitude (°N)"
KM_LABEL = "distance along transect (km)"


# -- fixtures -----------------------------------------------------------------------


def _path_field(lon, lat, *, path_lon=None, path_lat=None, attrs=None) -> xr.DataArray:
    """Build a fixed-depth section along a path of the given ``lon``/``lat`` positions.

    ``along`` is a made-up kilometre axis (the x rule reads positions from lon/lat,
    never from it); ``path_lon``/``path_lat`` ride on the along dimension as the
    requested positions when given.
    """
    lon, lat = np.asarray(lon, dtype="float64"), np.asarray(lat, dtype="float64")
    n = lon.size
    z = -np.array([0.0, 50.0, 100.0, 200.0])
    coords = {
        "z": z,
        ALONG_DIM: np.arange(n) * 25.0,
        "lon": (ALONG_DIM, lon),
        "lat": (ALONG_DIM, lat),
    }
    if path_lon is not None:
        coords["path_lon"] = (ALONG_DIM, np.asarray(path_lon, dtype="float64"))
    if path_lat is not None:
        coords["path_lat"] = (ALONG_DIM, np.asarray(path_lat, dtype="float64"))
    values = 20.0 - 0.05 * np.abs(z)[:, None] + 0.01 * np.arange(n)[None, :]
    da = xr.DataArray(values, dims=("z", ALONG_DIM), coords=coords)
    da[ALONG_DIM].attrs["units"] = "km"
    da[ALONG_DIM].attrs.update(attrs or {})
    return da


def _equatorial(n: int = 13, **extra) -> xr.DataArray:
    """Build an equatorial line from 143°E to 267°E (the Pacific, 0-360)."""
    return _path_field(np.linspace(143, 267, n), np.zeros(n), **extra)


def _x(da, x="auto"):
    field, geometry = prepare_section(da, x)
    return field, geometry


# -- the rule -----------------------------------------------------------------------


def test_an_equatorial_line_reads_longitude():
    field, geometry = _x(_equatorial())
    assert geometry.x_axis == "lon"
    assert geometry.x_label == LON_LABEL
    assert geometry.x_name == "distance"
    assert field["distance"].attrs["units"] == "degrees_east"
    np.testing.assert_allclose(
        np.unique(field["distance"].values), np.linspace(143, 267, 13)
    )


def test_a_meridional_line_reads_latitude():
    n = 11
    da = _path_field(np.full(n, -140.0), np.linspace(-10, 10, n))
    field, geometry = _x(da)
    assert geometry.x_axis == "lat"
    assert geometry.x_label == LAT_LABEL
    assert field["distance"].attrs["units"] == "degrees_north"
    np.testing.assert_allclose(
        np.unique(field["distance"].values), np.linspace(-10, 10, n)
    )


def test_a_gently_slanting_zonal_path_still_reads_longitude():
    n = 13
    da = _path_field(np.linspace(140, 260, n), np.linspace(0, 2, n))
    assert _x(da)[1].x_axis == "lon"


def test_a_diagonal_reads_distance():
    n = 13
    da = _path_field(np.linspace(-95, -85, n), np.linspace(24, 34, n))
    field, geometry = _x(da)
    assert geometry.x_axis == "distance"
    assert geometry.x_label == KM_LABEL
    assert field["distance"].attrs["units"] == "km"
    np.testing.assert_allclose(
        np.unique(field["distance"].values), da[ALONG_DIM].values
    )


def _ratio_path(ratio: float) -> xr.DataArray:
    """Build a monotonic path whose east-west km span is ``ratio`` times north-south.

    Latitude runs 0 to 10° (mean 5°), longitude over however many degrees make
    ``ratio`` times that distance at the path's mean latitude -- the same arithmetic
    the rule does.
    """
    n = 9
    lat_km = 10.0 * 110.57
    lon_span = ratio * lat_km / (111.32 * np.cos(np.radians(5.0)))
    return _path_field(np.linspace(0.0, lon_span, n), np.linspace(0.0, 10.0, n))


def test_the_dominance_ratio_is_the_documented_two():
    """A bit over twice the north-south span reads longitude; a bit under, distance."""
    assert sec.X_DOMINANCE == 2.0
    assert _x(_ratio_path(2.05))[1].x_axis == "lon"
    assert _x(_ratio_path(1.95))[1].x_axis == "distance"
    # and the mirror image
    assert _x(_ratio_path(1 / 2.05))[1].x_axis == "lat"
    assert _x(_ratio_path(1 / 1.95))[1].x_axis == "distance"


def test_a_path_that_doubles_back_in_longitude_reads_distance():
    lon = np.concatenate([np.linspace(150, 200, 8), np.linspace(200, 170, 6)[1:]])
    da = _path_field(lon, np.zeros(lon.size))
    assert _x(da)[1].x_axis == "distance"


def test_a_westward_path_reads_longitude_running_the_other_way():
    n = 9
    da = _path_field(np.linspace(267, 143, n), np.zeros(n))
    field, geometry = _x(da)
    assert geometry.x_axis == "lon"
    x = field["distance"].isel(z=0).values
    assert np.all(np.diff(x) < 0)


def test_a_dateline_crossing_stored_in_plus_minus_180_is_unwrapped():
    lon = np.concatenate([np.linspace(170, 180, 6)[:-1], np.linspace(-180, -170, 6)])
    da = _path_field(lon, np.zeros(lon.size))
    field, geometry = _x(da)
    assert geometry.x_axis == "lon"
    x = field["distance"].isel(z=0).values
    assert np.all(np.diff(x) > 0)
    assert x[0] == pytest.approx(170.0)
    assert x[-1] == pytest.approx(190.0)


def test_a_degenerate_path_reads_distance():
    one = _path_field([150.0], [0.0])
    assert _x(one)[1].x_axis == "distance"
    with_nan = _equatorial()
    lon = with_nan["lon"].values.copy()
    lon[4] = np.nan
    assert _x(with_nan.assign_coords(lon=(ALONG_DIM, lon)))[1].x_axis == "distance"


def test_a_field_with_no_lon_lat_reads_distance():
    da = _equatorial().drop_vars(["lon", "lat"])
    _, geometry = _x(da)
    assert geometry.x_axis == "distance"
    assert geometry.path_note == ""


def test_the_requested_path_wins_over_snapped_coordinates():
    """With ``path_lon``/``path_lat`` present, x is the requested longitude."""
    n = 13
    requested = np.linspace(143, 267, n)
    snapped = requested + np.where(np.arange(n) % 2 == 0, 0.0, 0.2)
    da = _path_field(snapped, np.zeros(n))
    plain, _ = _x(da)
    np.testing.assert_allclose(np.unique(plain["distance"].values), np.sort(snapped))
    with_path = da.assign_coords(
        path_lon=(ALONG_DIM, requested), path_lat=(ALONG_DIM, np.zeros(n))
    )
    field, geometry = _x(with_path)
    assert geometry.x_axis == "lon"
    np.testing.assert_allclose(np.unique(field["distance"].values), requested)


def test_a_jittery_snapped_longitude_is_rescued_by_the_requested_path():
    """A snapped longitude that repeats a cell reads distance; ``path_lon`` fixes it."""
    n = 9
    lon = np.linspace(143, 267, n)
    lon[3] = lon[2]  # two path points snapped to the same grid column
    da = _path_field(lon, np.zeros(n))
    assert _x(da)[1].x_axis == "distance"
    rescued = da.assign_coords(
        path_lon=(ALONG_DIM, np.linspace(143, 267, n)),
        path_lat=(ALONG_DIM, np.zeros(n)),
    )
    assert _x(rescued)[1].x_axis == "lon"


def test_a_zigzagging_snapped_latitude_does_not_hide_a_zonal_path():
    n = 13
    lat = np.where(np.arange(n) % 2 == 0, 0.0, 0.4)
    da = _path_field(np.linspace(143, 267, n), lat)
    # the snapped latitude wobbles 0.4° (44 km) against 124° (13 800 km) of longitude
    assert _x(da)[1].x_axis == "lon"
    # ... and path_lat == 0 is preferred when both are present
    both = da.assign_coords(path_lat=(ALONG_DIM, np.zeros(n)))
    assert _x(both)[1].x_axis == "lon"


# -- forcing it ---------------------------------------------------------------------


def test_forcing_distance_gives_kilometres_for_any_path():
    field, geometry = _x(_equatorial(), "distance")
    assert geometry.x_axis == "distance"
    assert geometry.x_label == KM_LABEL
    np.testing.assert_allclose(
        np.unique(field["distance"].values), _equatorial()[ALONG_DIM].values
    )


def test_forcing_lon_on_a_diagonal_draws_it_anyway():
    n = 9
    da = _path_field(np.linspace(-95, -85, n), np.linspace(24, 34, n))
    assert _x(da, "lon")[1].x_axis == "lon"
    assert _x(da, "lat")[1].x_axis == "lat"


def test_forcing_a_coordinate_that_does_not_run_one_way_raises():
    n = 11
    meridional = _path_field(np.full(n, -140.0), np.linspace(-10, 10, n))
    with pytest.raises(ValueError, match='section_x="distance"') as excinfo:
        _x(meridional, "lon")
    assert "longitude" in str(excinfo.value)
    lon = np.concatenate([np.linspace(150, 200, 8), np.linspace(200, 170, 6)[1:]])
    with pytest.raises(ValueError, match="not monotonic"):
        _x(_path_field(lon, np.zeros(lon.size)), "lon")


def test_forcing_a_coordinate_the_field_lacks_raises():
    da = _equatorial().drop_vars(["lon", "lat"])
    with pytest.raises(ValueError, match="absent from this field"):
        _x(da, "lon")


@pytest.mark.parametrize("bad", ["km", "longitude", "", None, 3, True])
def test_an_unknown_x_is_refused_by_name(bad):
    with pytest.raises(ValueError, match="section_x=") as excinfo:
        _x(_equatorial(), bad)
    for valid in ("auto", "distance", "lon", "lat"):
        assert valid in str(excinfo.value)
    with pytest.raises(ValueError, match="section_x="):
        sec.check_section_options(has_contours=False, mark=None, section_x=bad)


def test_preparing_a_section_leaves_the_input_alone():
    da = _equatorial()
    before = dict(da[ALONG_DIM].attrs)
    _x(da)
    _x(da, "distance")
    assert dict(da[ALONG_DIM].attrs) == before


# -- slabs --------------------------------------------------------------------------


def _slab(axis="lat") -> xr.DataArray:
    n = 7
    if axis == "lat":
        lon, lat = np.full(n, 190.0), np.linspace(-30, 30, n)
    else:
        lon, lat = np.linspace(150, 250, n), np.full(n, 0.0)
    return _path_field(
        lon,
        lat,
        attrs={
            "axis_coord": axis,
            "band_axis": "lon" if axis == "lat" else "lat",
            "band": (180.0, 200.0),
        },
    )


@pytest.mark.parametrize("axis", ["lat", "lon"])
def test_a_slabs_surviving_axis_wins_under_auto(axis):
    _, geometry = _x(_slab(axis))
    assert geometry.x_axis == axis
    assert geometry.x_label == (LAT_LABEL if axis == "lat" else LON_LABEL)
    assert "mean over" in geometry.path_note


@pytest.mark.parametrize("axis", ["lat", "lon"])
def test_forcing_distance_overrides_a_slab(axis):
    field, geometry = _x(_slab(axis), "distance")
    assert geometry.x_axis == "distance"
    assert geometry.x_label == KM_LABEL
    np.testing.assert_allclose(
        np.unique(field["distance"].values), _slab(axis)[ALONG_DIM].values
    )


# -- rows and overlays share one answer ----------------------------------------------


def _trio(**kwargs) -> dict:
    test = _equatorial(**kwargs)
    reference = test + 1.0
    return {"test": test, "reference": reference, "difference": test - reference}


def test_a_row_resolves_x_once_and_every_lane_follows():
    values, geometry = prepare_section_row(_trio())
    assert geometry.x_axis == "lon"
    for lane in values.values():
        assert lane["distance"].attrs["units"] == "degrees_east"
    values, geometry = prepare_section_row(_trio(), "distance")
    assert geometry.x_axis == "distance"
    for lane in values.values():
        assert lane["distance"].attrs["units"] == "km"
    with pytest.raises(ValueError, match="section_x="):
        prepare_section_row(_trio(), "up")


def test_an_overlay_is_prepared_with_its_panels_x():
    """The lines follow a forced x as the fill did, and ``auto`` alike for both."""
    panel, geometry = prepare_section(_equatorial(), "distance")
    assert geometry.x_axis == "distance"
    overlay = prepare_overlay(_equatorial() + 3.0, panel, "distance")
    np.testing.assert_array_equal(overlay["distance"].values, panel["distance"].values)
    assert overlay["distance"].attrs["units"] == "km"
    # forgetting to say so (auto) puts the lines in degrees against a km fill: refused
    with pytest.raises(ValueError, match="positions along the section differ"):
        prepare_overlay(_equatorial() + 3.0, panel)

    lon_panel, _ = prepare_section(_equatorial())
    lon_overlay = prepare_overlay(_equatorial() + 3.0, lon_panel)
    assert lon_overlay["distance"].attrs["units"] == "degrees_east"


@pytest.mark.parametrize("section_x", ["auto", "distance"])
@pytest.mark.parametrize("renderer", ["matplotlib", "holoviews"])
def test_contour_overlays_ride_the_same_section_x_in_both_renderers(
    section_x, renderer
):
    """An overlaid section draws (no mesh mismatch) whichever x the call forces."""
    item = _item(contour=_equatorial() + 3.0)
    render(
        PlotSpec(
            family="section",
            items=[item],
            options={"mark": "contourf", "section_x": section_x},
        ),
        renderer=renderer,
    )


def test_an_overlay_mismatch_reads_in_degrees():
    panel, _ = prepare_section(_equatorial(13))
    with pytest.raises(ValueError, match=r"positions along the section differ") as err:
        prepare_overlay(_equatorial(9), panel)
    assert "°E" in str(err.value)
    assert " km" not in str(err.value)


# -- both renderers -----------------------------------------------------------------


def _item(field=None, *, contour=None, label="roms") -> dict:
    item = {
        "field": _equatorial() if field is None else field,
        "units": "degC",
        "standard_name": None,
        "depth": None,
        "label": label,
    }
    if contour is not None:
        item["contour"] = contour
        item["contour_units"] = "degC"
    return item


def _row_item(*, contour: bool = False, **extra) -> dict:
    trio = _trio()
    item = {
        "aligned": trio,
        "units": "degC",
        "standard_name": None,
        "depth": None,
        "time": None,
        "metrics": {"bias": 0.1, "rmse": 0.5, "corr": 0.98},
        "labels": ("model", "obs"),
        **extra,
    }
    if contour:
        item["contour"] = {
            "test": _equatorial() + 3.0,
            "reference": _equatorial() + 4.0,
        }
    return item


def _static(family, items, **options):
    return render(PlotSpec(family=family, items=items, options=options))


def _static_x_labels(fig) -> set[str]:
    return {
        ax.get_xlabel()
        for ax in fig.axes
        if ax.get_label() != "<colorbar>" and ax.get_xlabel()
    }


_CASES = {
    "section": lambda **kw: ("section", [_item(**kw)]),
    "section[stacked]": lambda **kw: (
        "section",
        [_item(label="a", **kw), _item(label="b", **kw)],
    ),
    "section_row": lambda **kw: ("section_row", [_row_item(**kw)]),
    "section_row[stacked]": lambda **kw: (
        "section_row",
        [_row_item(row_label="A", **kw), _row_item(row_label="B", **kw)],
    ),
}


def _case(name, *, contour=False):
    if name.startswith("section_row"):
        return _CASES[name](contour=contour)
    return _CASES[name](contour=_equatorial() + 3.0 if contour else None)


@pytest.mark.parametrize("name", sorted(_CASES))
def test_static_equatorial_sections_are_labelled_by_longitude(name):
    family, items = _case(name)
    labels = _static_x_labels(_static(family, items))
    assert labels == {LON_LABEL}


@pytest.mark.parametrize("name", sorted(_CASES))
def test_static_section_x_distance_gives_kilometres(name):
    family, items = _case(name)
    assert _static_x_labels(_static(family, items, section_x="distance")) == {KM_LABEL}


@pytest.mark.parametrize("name", sorted(_CASES))
def test_static_contourf_with_contours_follows_section_x(name):
    family, items = _case(name, contour=True)
    fig = _static(family, items, mark="contourf")
    assert _static_x_labels(fig) == {LON_LABEL}
    fig = _static(family, items, mark="contourf", section_x="distance")
    assert _static_x_labels(fig) == {KM_LABEL}


def test_the_static_x_data_runs_over_the_longitudes():
    family, items = _case("section")
    fig = _static(family, items, mark="contourf")
    (ax,) = [a for a in fig.axes if a.get_label() != "<colorbar>"]
    lo, hi = ax.get_xlim()
    assert lo == pytest.approx(143, abs=1) and hi == pytest.approx(267, abs=1)


def test_a_westward_path_draws_on_both_renderers_and_both_marks():
    n = 13
    westward = _path_field(np.linspace(267, 143, n), np.zeros(n))
    for mark in ("pcolormesh", "contourf"):
        fig = _static("section", [_item(westward, contour=westward + 3.0)], mark=mark)
        assert _static_x_labels(fig) == {LON_LABEL}
        _holoviews("section", [_item(westward, contour=westward + 3.0)], mark=mark)


def test_static_cross_labels_each_panel_on_its_own():
    n = 13
    meridional = _path_field(np.full(n, -140.0), np.linspace(-6, 6, n))
    fig = _static("cross", [_item(label="xi"), _item(meridional, label="eta")])
    assert _static_x_labels(fig) == {LON_LABEL, LAT_LABEL}
    fig = _static(
        "cross",
        [_item(label="xi"), _item(meridional, label="eta")],
        section_x="distance",
    )
    assert _static_x_labels(fig) == {KM_LABEL}


@pytest.mark.parametrize("family", ["section", "section_row"])
@pytest.mark.parametrize("bad", ["km", "longitude", None])
def test_static_invalid_section_x_raises(family, bad):
    f, items = _case(family)
    with pytest.raises(ValueError, match="section_x="):
        _static(f, items, section_x=bad)


def test_static_forced_lon_that_doubles_back_raises():
    lon = np.concatenate([np.linspace(150, 200, 8), np.linspace(200, 170, 6)[1:]])
    items = [_item(_path_field(lon, np.zeros(lon.size)))]
    assert _static_x_labels(_static("section", items)) == {KM_LABEL}
    with pytest.raises(ValueError, match='section_x="distance"'):
        _static("section", items, section_x="lon")


# -- the interactive renderer --------------------------------------------------------

hv = pytest.importorskip("holoviews")
pytest.importorskip("hvplot")


def _holoviews(family, items, **options):
    return render(
        PlotSpec(family=family, items=items, options=options), renderer="holoviews"
    )


def _hv_x_labels(obj) -> set[str]:
    meshes = obj.traverse(lambda x: x, [hv.QuadMesh, hv.Polygons])
    assert meshes
    return {e.opts.get("plot").kwargs["xlabel"] for e in meshes}


@pytest.mark.parametrize("name", sorted(_CASES))
@pytest.mark.parametrize("mark", ["pcolormesh", "contourf"])
def test_holoviews_equatorial_sections_follow_section_x(name, mark):
    family, items = _case(name, contour=True)
    assert _hv_x_labels(_holoviews(family, items, mark=mark)) == {LON_LABEL}
    forced = _holoviews(family, items, mark=mark, section_x="distance")
    assert _hv_x_labels(forced) == {KM_LABEL}


def test_holoviews_x_data_runs_over_the_longitudes():
    family, items = _case("section")
    obj = _holoviews(family, items)
    (mesh,) = obj.traverse(lambda x: x, [hv.QuadMesh])
    x = np.asarray(mesh.dimension_values(0, expanded=False), dtype=float)
    assert x.min() == pytest.approx(143.0) and x.max() == pytest.approx(267.0)


def test_holoviews_cross_labels_each_panel_on_its_own():
    n = 13
    meridional = _path_field(np.full(n, -140.0), np.linspace(-6, 6, n))
    items = [_item(label="xi"), _item(meridional, label="eta")]
    assert _hv_x_labels(_holoviews("cross", items)) == {LON_LABEL, LAT_LABEL}
    assert _hv_x_labels(_holoviews("cross", items, section_x="distance")) == {KM_LABEL}


@pytest.mark.parametrize("family", ["section", "section_row"])
def test_holoviews_invalid_section_x_raises(family):
    f, items = _case(family)
    with pytest.raises(ValueError, match="section_x="):
        _holoviews(f, items, section_x="km")


# -- from a user's call --------------------------------------------------------------

EQUATOR = {"transect": {"lat": 0, "lon": {"min": 160, "max": 220}}}


def _equatorial_field(patched_woa):  # noqa: F811
    return osk.field(patched_woa(), VAR, select=EQUATOR, cache=False)


def test_field_plot_labels_an_equatorial_transect_and_forwards_section_x(
    patched_woa,  # noqa: F811
):
    f = _equatorial_field(patched_woa)
    assert _static_x_labels(f.plot(renderer="matplotlib")) == {LON_LABEL}
    forced = f.plot(renderer="matplotlib", section_x="distance")
    assert _static_x_labels(forced) == {KM_LABEL}
    assert _hv_x_labels(f.plot(renderer="holoviews")) == {LON_LABEL}
    assert _hv_x_labels(f.plot(renderer="holoviews", section_x="distance")) == {
        KM_LABEL
    }
    with pytest.raises(ValueError, match="section_x="):
        f.plot(renderer="matplotlib", section_x="east")


def test_a_set_of_sections_forwards_section_x(patched_woa):  # noqa: F811
    f = _equatorial_field(patched_woa)
    g = _equatorial_field(patched_woa)
    assert _static_x_labels(osk.plot([f, g], renderer="matplotlib")) == {LON_LABEL}
    fig = osk.plot({"a": f, "b": g}, renderer="matplotlib", section_x="distance")
    assert _static_x_labels(fig) == {KM_LABEL}
    assert _hv_x_labels(
        osk.plot([f, g], renderer="holoviews", section_x="distance")
    ) == {KM_LABEL}
