"""A map panel keeps its true shape: longitude shrinks with cos(lat).

Panels are drawn on plain PlateCarree, which spends the same length on a degree of
longitude as on a degree of latitude -- true only at the equator. At Cook Inlet (~60N)
a degree of longitude is half as long as one of latitude, so every map came out ~2x too
wide (Iceland, ~64N: ~2.3x). Both renderers now stretch by 1/cos(mid-latitude) and size
every panel to the corrected shape, so an equatorial domain is exactly what it was.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import xarray as xr

from ocean_skill.plot.registry import render
from ocean_skill.plot.spec import PlotSpec

#: a 4 deg x 2 deg window at 60N: truer shape is cos(60) * 4 / 2 = 1.0, not 2.0
LON60, LAT60 = (-154.0, -150.0), (59.0, 61.0)
#: the same 4 x 2 window on the equator, where nothing changes
LON0, LAT0 = (-154.0, -150.0), (-1.0, 1.0)
#: the 60N window as a ``(lon_min, lat_min, lon_max, lat_max)`` bbox
WINDOW60 = (-154.0, 59.0, -150.0, 61.0)


def _field(lon, lat, *, shape=(12, 16)) -> xr.DataArray:
    lats, lons = np.linspace(*lat, shape[0]), np.linspace(*lon, shape[1])
    z = np.add.outer(np.linspace(0.0, 1.0, shape[0]), np.linspace(0.0, 1.0, shape[1]))
    return xr.DataArray(
        z,
        dims=("lat", "lon"),
        coords={"lat": lats, "lon": lons},
        name="temp",
        attrs={"units": "degC"},
    )


def _item(field) -> dict:
    return {"field": field, "standard_name": "sea_water_temperature", "units": "degC"}


def _station() -> dict:
    """One catalog-style point, the way ``locations`` items are built."""
    from ocean_skill.plot.locations import HOVER_FIELDS

    return {
        **{f: ("stn" if f == "name" else "") for f in HOVER_FIELDS},
        "kind": "point",
        "featureType": "timeSeries",
        "lon": -152.0,
        "lat": 60.0,
    }


def _static_map(field):
    """Draw a one-panel static field map; return its figure and map axes."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig = render(PlotSpec("field_facet", [_item(field)], {"domain": None}))
        fig.canvas.draw()
    return fig, next(ax for ax in fig.axes if hasattr(ax, "projection"))


def _shape(ax) -> float:
    """Width / height of the axes as drawn."""
    box = ax.get_window_extent()
    return box.width / box.height


def _hv_maps(obj) -> list[dict]:
    """Every rendered bokeh figure's frame (px) and how it is scaled onto its ranges.

    ``lon_per_lat`` is pixels per degree of longitude over pixels per degree of
    latitude: ``cos(lat0)`` when the map is true to shape, ``1`` in plain degrees.
    """
    import holoviews as hv
    from bokeh.plotting import figure

    plot = hv.render(getattr(obj, "object", obj), backend="bokeh")
    out = []
    for f in plot.select({"type": figure}):
        x, y = f.x_range, f.y_range
        out.append(
            {
                "shape": f.frame_width / f.frame_height,
                "lon_per_lat": (f.frame_width / (x.end - x.start))
                / (f.frame_height / (y.end - y.start)),
            }
        )
    return out


# --- the shared measure ---------------------------------------------------------------


def test_aspect_is_the_true_shape_at_60n_and_unchanged_on_the_equator():
    from ocean_skill.plot.matplotlib_renderer import _aspect_of

    assert _aspect_of(_field(LON60, LAT60)) == pytest.approx(1.0, rel=0.01)
    assert _aspect_of(_field(LON0, LAT0)) == pytest.approx(2.0)


def test_a_polar_domain_does_not_blow_the_stretch_up():
    from ocean_skill.plot.matplotlib_renderer import _aspect_of

    # lat0 is capped at 80, so a window at the pole stays a finite, positive shape
    aspect = _aspect_of(_field((-10.0, 10.0), (88.0, 90.0)))
    assert np.isfinite(aspect) and aspect > 0


# --- static ---------------------------------------------------------------------------


def test_a_static_map_at_60n_is_drawn_in_true_shape():
    _, ax = _static_map(_field(LON60, LAT60))
    assert ax.get_aspect() == pytest.approx(2.0, rel=0.01)
    # 4 deg of longitude at 60N is as long as 2 deg of latitude: a square, not 2:1
    assert _shape(ax) == pytest.approx(1.0, rel=0.03)


def test_a_static_equatorial_map_is_unchanged():
    _, ax = _static_map(_field(LON0, LAT0))
    assert ax.get_aspect() == pytest.approx(1.0)
    assert _shape(ax) == pytest.approx(2.0, rel=0.03)


def test_the_figure_is_sized_to_the_true_shape_not_the_degree_shape():
    """A panel sized for 2:1 but drawn 1:1 would be a small square in a wide figure."""
    fig60, ax60 = _static_map(_field(LON60, LAT60))
    fig0, _ = _static_map(_field(LON0, LAT0))
    # the same window in degrees: the 60N one is a squarer panel, so its figure is
    # taller for the same width, and the panel is the bulk of it
    assert fig60.get_figheight() > 1.5 * fig0.get_figheight()
    assert ax60.get_position().height > 0.7


def test_a_static_field_row_is_drawn_in_true_shape():
    field = _field(LON60, LAT60)
    aligned = xr.Dataset(
        {"test": field, "reference": field + 0.1, "difference": field * 0 - 0.1}
    )
    item = {
        "aligned": aligned,
        "standard_name": "sea_water_temperature",
        "units": "degC",
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig = render(PlotSpec("field_row", [item], {}))
        fig.canvas.draw()
    panels = [ax for ax in fig.axes if hasattr(ax, "projection")]
    assert len(panels) == 3
    for ax in panels:
        assert ax.get_aspect() == pytest.approx(2.0, rel=0.01)
        assert _shape(ax) == pytest.approx(1.0, rel=0.05)


def test_a_static_skill_map_is_drawn_in_true_shape():
    skill = xr.Dataset({"bias": _field(LON60, LAT60)})
    item = {"skill": skill, "metric_names": ("bias",)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig = render(PlotSpec("skill_map", [item], {}))
        fig.canvas.draw()
    (ax,) = [a for a in fig.axes if hasattr(a, "projection")]
    assert ax.get_aspect() == pytest.approx(2.0, rel=0.01)
    assert _shape(ax) == pytest.approx(1.0, rel=0.05)


def test_an_extent_window_gets_the_aspect_of_the_window_not_the_grid():
    """``extent=`` re-frames the view, and the true shape goes with it."""
    skill = xr.Dataset({"bias": _field((-160.0, -140.0), (50.0, 70.0))})
    item = {"skill": skill, "metric_names": ("bias",)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig = render(
            PlotSpec("skill_map", [item], {"extent": (-154.0, -150.0, 59.0, 61.0)})
        )
        fig.canvas.draw()
    (ax,) = [a for a in fig.axes if hasattr(a, "projection")]
    assert ax.get_aspect() == pytest.approx(1 / np.cos(np.radians(60.0)), rel=0.01)


def test_a_dateline_straddling_map_keeps_its_180_centred_frame_and_true_shape():
    """The 180-centred frame (a Pacific domain) takes the same stretch."""
    lon = np.linspace(170.0, 190.0, 21)  # 0-360, crossing the dateline
    lat = np.linspace(50.0, 70.0, 21)  # lat0 = 60
    field = _field((lon[0], lon[-1]), (lat[0], lat[-1]), shape=(21, 21))
    _, ax = _static_map(field)
    assert ax.projection.proj4_params["lon_0"] == 180.0
    assert ax.get_aspect() == pytest.approx(2.0, rel=0.01)
    # 20 deg of longitude at 60N is 10 deg of latitude long, so half as wide as tall
    assert _shape(ax) == pytest.approx(0.5, rel=0.05)


def test_a_static_location_map_is_drawn_in_true_shape():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig = render(PlotSpec("locations", [_station()], {"extent": WINDOW60}))
        fig.canvas.draw()
    (ax,) = [a for a in fig.axes if hasattr(a, "projection")]
    assert ax.get_aspect() == pytest.approx(2.0, rel=0.01)
    assert _shape(ax) == pytest.approx(1.0, rel=0.05)


def test_a_static_movie_frame_is_drawn_in_true_shape():
    """Movies draw through the same panel code, so they take the stretch too."""
    time = np.array(["2012-01-01", "2012-02-01"], dtype="datetime64[ns]")
    field = xr.concat([_field(LON60, LAT60)] * 2, dim="time").assign_coords(time=time)
    item = {**_item(field), "facet_dim": "time", "row_dim": None}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ani = render(PlotSpec("facet_movie", [item], {"domain": None}))
        ani._fig.canvas.draw()
    (ax,) = [a for a in ani._fig.axes if hasattr(a, "projection")]
    assert ax.get_aspect() == pytest.approx(2.0, rel=0.01)
    assert _shape(ax) == pytest.approx(1.0, rel=0.05)


# --- interactive ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lon", "lat", "expected"), [(LON60, LAT60, 1.0), (LON0, LAT0, 2.0)]
)
def test_an_interactive_map_frame_has_the_true_shape(lon, lat, expected):
    pytest.importorskip("geoviews")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        obj = render(
            PlotSpec("field_facet", [_item(_field(lon, lat))], {"domain": None}),
            renderer="holoviews",
            tiles=False,
        )
    (m,) = _hv_maps(obj)
    assert m["shape"] == pytest.approx(expected, rel=0.03)
    assert m["lon_per_lat"] == pytest.approx(np.cos(np.radians(lat[0] + 1.0)), rel=0.03)


def test_an_interactive_skill_map_frame_has_the_true_shape():
    pytest.importorskip("geoviews")
    skill = xr.Dataset({"bias": _field(LON60, LAT60)})
    item = {"skill": skill, "metric_names": ("bias",)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        obj = render(
            PlotSpec("skill_map", [item], {}), renderer="holoviews", tiles=False
        )
    (m,) = _hv_maps(obj)
    assert m["shape"] == pytest.approx(1.0, rel=0.03)
    assert m["lon_per_lat"] == pytest.approx(0.5, rel=0.03)


def test_an_interactive_location_map_frame_has_the_true_shape():
    pytest.importorskip("geoviews")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        obj = render(
            PlotSpec("locations", [_station()], {"extent": WINDOW60}),
            renderer="holoviews",
            tiles=False,
        )
    (m,) = _hv_maps(obj)
    assert m["shape"] == pytest.approx(1.0, rel=0.03)
    assert m["lon_per_lat"] == pytest.approx(0.5, rel=0.03)


def test_an_interactive_dateline_straddling_map_keeps_its_frame():
    pytest.importorskip("geoviews")
    field = _field((170.0, 190.0), (50.0, 70.0), shape=(21, 21))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        obj = render(
            PlotSpec("field_facet", [_item(field)], {"domain": None}),
            renderer="holoviews",
            tiles=False,
        )
    (m,) = _hv_maps(obj)
    assert m["shape"] == pytest.approx(0.5, rel=0.03)
    assert m["lon_per_lat"] == pytest.approx(0.5, rel=0.03)
