"""Interactive vertical sections: filled-contour bands and contour-line overlays.

The holoviews renderer's twin of the static ``mark="contourf"`` section with black,
labelled contour lines of a second variable on top (isotherms over phosphate). Bokeh has
no contour primitive, so the bands are ``hv.Polygons`` (holoviews' ``contours``
operation) and the lines ``hv.Contours`` from :func:`ocean_skill.plot.section.
contour_paths`; what has to hold is that they are the *same drawing* as the static one:

* the band edges are :func:`~ocean_skill.plot.section.fill_edges` of the colour range,
  and each band is the colour matplotlib's ``contourf`` gives it (its midpoint through
  the norm, and the colormap's own end colours past a clipped end);
* the lines are at the figure's one set of :func:`~ocean_skill.plot.section.
  contour_levels`, labelled with :func:`~ocean_skill.plot.section.contour_label`, on the
  test and reference panels of a row and never on the difference;
* the colour bar keeps its round ticks, and the cell mesh stays on top, transparent, for
  an exact value on hover.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import xarray as xr

from ocean_skill.align import ALONG_DIM
from ocean_skill.plot._colorbar import colorbar_ticks, difference_limit
from ocean_skill.plot.registry import render
from ocean_skill.plot.section import (
    contour_label,
    contour_levels,
    fill_edges,
    prepare_overlay,
    prepare_section,
)
from ocean_skill.plot.spec import PlotSpec

hv = pytest.importorskip("holoviews")
pytest.importorskip("hvplot")

PHOSPHATE = "mole_concentration_of_phosphate_in_sea_water"
CHLOROPHYLL = "mass_concentration_of_chlorophyll_a_in_sea_water"

# -- fixtures -----------------------------------------------------------------------


def _section_field(
    *,
    offset: float = 0.0,
    scale: float = 1.0,
    n_z: int = 12,
    n_along: int = 20,
    length: float = 150.0,
) -> xr.DataArray:
    """Return a fixed-depth ``(z, along)`` ramp: 5 at one corner, 18 at the far one."""
    z = -np.linspace(0.0, 200.0, n_z)
    along = np.linspace(0.0, length, n_along)
    values = (5.0 - 0.05 * z[:, None] + 0.02 * along[None, :]) * scale + offset
    return xr.DataArray(
        values,
        dims=("z", ALONG_DIM),
        coords={
            "z": z,
            ALONG_DIM: along,
            "lon": (ALONG_DIM, np.linspace(-95.0, -93.0, n_along)),
            "lat": (ALONG_DIM, np.linspace(24.0, 26.0, n_along)),
        },
    )


def _section_item(
    field=None, *, contour=None, standard_name=PHOSPHATE, units="mmol m-3"
):
    item = {
        "field": _section_field() if field is None else field,
        "units": units,
        "standard_name": standard_name,
        "depth": None,
        "label": "roms_run",
    }
    if contour is not None:
        item |= {
            "contour": contour,
            "contour_standard_name": "sea_water_temperature",
            "contour_units": "degC",
        }
    return item


def _row_item(*, offset: float = 1.0, contour=None, length: float = 150.0):
    """Return a ``section_row`` item (an aligned trio); ``contour`` is a pair."""
    test = _section_field(length=length)
    reference = _section_field(offset=offset, length=length)
    # a difference that spans -1..1 exactly, so a symmetric colour range has every band
    difference = (test - 11.5) / 6.5
    item = {
        "aligned": {
            "test": test.rename("test"),
            "reference": reference.rename("reference"),
            "difference": difference.rename("difference"),
        },
        "units": "mmol m-3",
        "standard_name": PHOSPHATE,
        "depth": "0-200 m",
        "time": "2012-01",
        "metrics": {"bias": -1.0, "rmse": 1.0, "corr": 1.0},
        "labels": ("roms_run", "woa23"),
    }
    if contour is not None:
        item |= {
            "contour": {"test": contour[0], "reference": contour[1]},
            "contour_standard_name": "sea_water_temperature",
            "contour_units": "degC",
        }
    return item


def _temperature_pair(length: float = 150.0):
    """Two 'temperature' sections on the row's own mesh, spanning 11-50 and 14-53."""
    return (
        _section_field(scale=3.0, offset=-4.0, length=length),
        _section_field(scale=3.0, offset=-1.0, length=length),
    )


def _native_s_field(slope: float = 1.0) -> xr.DataArray:
    """Return an all-wet native-s section: its true depth varies along the path."""
    n_s, n_along = 8, 12
    sigma = np.linspace(-0.95, -0.05, n_s)
    z_rho = np.outer(sigma, np.linspace(30.0, 500.0, n_along))  # negative-down
    values = slope * (5.0 + np.linspace(0, 10, n_s * n_along).reshape(n_s, n_along))
    da = xr.DataArray(
        values,
        dims=("s_rho", ALONG_DIM),
        coords={
            "z_rho": (("s_rho", ALONG_DIM), z_rho),
            ALONG_DIM: np.linspace(0.0, 100.0, n_along),
            "lon": (ALONG_DIM, np.linspace(-95.0, -93.0, n_along)),
            "lat": (ALONG_DIM, np.linspace(24.0, 26.0, n_along)),
        },
    )
    da[ALONG_DIM].attrs["units"] = "km"
    return da


def _draw(family, items, **options):
    return render(
        PlotSpec(family=family, items=items, options=options), renderer="holoviews"
    )


def _exact(obj, kind) -> list:
    """Return every element of exactly ``kind`` (``Polygons`` is a ``Contours``)."""
    return [e for e in obj.traverse(lambda x: x, [kind]) if type(e) is kind]


def _panels(row) -> list:
    return list(row)


def _clim(polygons) -> tuple[float, float]:
    """Return the colour range a banded fill was drawn on (its value's range)."""
    low, high = polygons.vdims[0].range
    return float(low), float(high)


def _band_edges(polygons) -> np.ndarray:
    """Every finite band edge a ``Polygons`` element carries, ascending."""
    bounds = np.array(
        [[band["lower"], band["upper"]] for band in polygons.data], dtype="float64"
    )
    return np.unique(bounds[np.isfinite(bounds)])


def _drawn_levels(contours) -> list[float]:
    return sorted({line["level"] for line in contours.data})


def _label_texts(labels) -> list[str]:
    return [record["text"] for record in labels.data.to_dict("records")]


def _bokeh(obj):
    return hv.render(obj, backend="bokeh")


# -- mark: cells are the default and unchanged --------------------------------------


def test_the_default_mark_is_still_coloured_cells():
    obj = _draw("section", [_section_item()])
    assert type(obj) is hv.QuadMesh
    assert _exact(obj, hv.Polygons) == []


def test_mark_pcolormesh_is_the_default_spelled_out():
    default = _draw("section", [_section_item()])
    spelled = _draw("section", [_section_item()], mark="pcolormesh")
    assert type(spelled) is type(default) is hv.QuadMesh


def test_an_unknown_section_mark_is_refused_by_name():
    with pytest.raises(ValueError, match="not a section mark"):
        _draw("section", [_section_item()], mark="scatter")
    with pytest.raises(ValueError, match="not a section mark"):
        _draw("section_row", [_row_item()], mark="line")


# -- mark="contourf": the bands -----------------------------------------------------


def test_contourf_draws_polygons_with_the_cell_mesh_on_top_for_hover():
    obj = _draw("section", [_section_item()], mark="contourf")
    (polygons,) = _exact(obj, hv.Polygons)
    (mesh,) = _exact(obj, hv.QuadMesh)
    assert [type(e) for e in obj] == [hv.Polygons, hv.QuadMesh]  # cells above bands
    # the cells carry the hover but draw nothing and bring no colour bar of their own
    style, plot = mesh.opts.get("style").kwargs, mesh.opts.get("plot").kwargs
    assert style["fill_alpha"] == 0 and style["line_alpha"] == 0
    assert plot["colorbar"] is False
    assert "hover" in plot["tools"]
    assert polygons.opts.get("plot").kwargs["colorbar"] is True


def test_the_bands_break_at_fill_edges_of_the_colour_range():
    (polygons,) = _exact(
        _draw("section", [_section_item()], mark="contourf"), hv.Polygons
    )
    lo, hi = _clim(polygons)
    np.testing.assert_allclose(_band_edges(polygons), fill_edges(lo, hi, log=False))
    # one flat band is the value at its edges' midpoint
    for band in polygons.data:
        assert band["value"] == pytest.approx(0.5 * (band["lower"] + band["upper"]))


def test_fill_levels_as_an_int_sets_about_that_many_bands():
    (polygons,) = _exact(
        _draw("section", [_section_item()], mark="contourf", fill_levels=6), hv.Polygons
    )
    lo, hi = _clim(polygons)
    edges = fill_edges(lo, hi, log=False, fill_levels=6)
    np.testing.assert_allclose(_band_edges(polygons), edges)
    assert len(edges) < len(fill_edges(lo, hi, log=False))  # fewer than the default


def test_fill_levels_as_a_list_sets_exactly_those_edges():
    edges = [5.0, 8.0, 11.0, 14.0, 18.0]
    (polygons,) = _exact(
        _draw("section", [_section_item()], mark="contourf", fill_levels=edges),
        hv.Polygons,
    )
    np.testing.assert_allclose(_band_edges(polygons), edges)
    assert len(polygons.data) == len(edges) - 1


def test_the_difference_panel_is_filled_on_its_own_symmetric_range():
    row = _draw("section_row", [_row_item()], mark="contourf")
    polygons = [_exact(panel, hv.Polygons) for panel in _panels(row)]
    assert [len(p) for p in polygons] == [1, 1, 1]
    test, reference, difference = (p[0] for p in polygons)
    assert _clim(test) == _clim(reference)  # the two share one scale
    low, high = _clim(difference)
    assert low == -high == -difference_limit(_row_item()["aligned"]["difference"])
    np.testing.assert_allclose(
        _band_edges(difference), fill_edges(low, high, log=False)
    )
    # all three panels keep their cells on top, for the hover
    assert all(len(_exact(panel, hv.QuadMesh)) == 1 for panel in _panels(row))


def test_a_stacked_grid_of_sections_fills_every_panel():
    items = [_section_item(), _section_item(_section_field(offset=2.0))]
    grid = _draw("section", items, mark="contourf")
    assert len(_exact(grid, hv.Polygons)) == 2


def test_a_stacked_grid_of_rows_fills_every_panel():
    grid = _draw("section_row", [_row_item(), _row_item(offset=2.0)], mark="contourf")
    assert len(_exact(grid, hv.Polygons)) == 6


def test_hover_false_leaves_only_the_bands():
    obj = _draw("section", [_section_item()], mark="contourf", hover=False)
    assert type(obj) is hv.Polygons
    assert obj.opts.get("plot").kwargs["tools"] == []


def test_a_rasterized_mesh_still_gets_bands_and_a_transparent_image_for_hover():
    pytest.importorskip("datashader")
    obj = _draw("section", [_section_item()], mark="contourf", rasterize=True)
    assert (
        len(_exact(obj, hv.Polygons)) == 1
    )  # from the section's arrays, not the image
    (image,) = _exact(obj, hv.Image)
    assert image.opts.get("style").kwargs["alpha"] == 0
    assert _bokeh(obj) is not None


# -- mark="contourf": the colours are matplotlib's ----------------------------------


def _mapper_colour(value: float, mapper) -> str:
    """Return the colour bokeh's mapper gives ``value`` (its rule, in plain Python)."""
    from bokeh.models import LogColorMapper

    low, high, palette = mapper.low, mapper.high, mapper.palette
    if value < low:
        return mapper.low_color
    if value > high:
        return mapper.high_color
    if isinstance(mapper, LogColorMapper):
        fraction = np.log(value / low) / np.log(high / low)
    else:
        fraction = (value - low) / (high - low)
    return palette[min(max(int(fraction * len(palette)), 0), len(palette) - 1)]


def _drawn_band_colours(obj) -> dict[tuple[float, float], str]:
    """``{(lower, upper): colour}`` of the bands exactly as bokeh will colour them."""
    from bokeh.models import GlyphRenderer, MultiPolygons

    (polygons,) = _exact(obj, hv.Polygons)
    (renderer,) = [
        r
        for r in _bokeh(obj).select({"type": GlyphRenderer})
        if isinstance(r.glyph, MultiPolygons)
    ]
    mapper = renderer.glyph.fill_color.transform
    values = renderer.data_source.data["color"]
    assert len(values) == len(polygons.data)
    return {
        (float(band["lower"]), float(band["upper"])): _mapper_colour(float(v), mapper)
        for band, v in zip(polygons.data, values, strict=True)
    }


def _matplotlib_band_colours(field, polygons, *, log: bool, name: str):
    """Return ``{(lower, upper): colour}`` of the bands from matplotlib's contourf."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    from ocean_skill.colormaps import cmaps_for
    from ocean_skill.plot.matplotlib_renderer import _data_range, _extend

    values, _ = prepare_section(field)
    lo, hi = _clim(polygons)
    edges = fill_edges(lo, hi, log=log, fill_levels=_FILL_LEVELS.get(name))
    extend = _extend(lo, hi, _data_range(values, log=log))
    norm = mcolors.LogNorm(lo, hi) if log else mcolors.Normalize(lo, hi)
    fig, ax = plt.subplots()
    cs = ax.contourf(
        values["distance"].values,
        values["depth"].values,
        values.values,
        levels=edges,
        cmap=cmaps_for(PHOSPHATE if not log else CHLOROPHYLL)[0],
        norm=norm,
        extend=extend,
    )
    colours = [mcolors.to_hex(c) for c in cs.get_facecolor()]
    plt.close(fig)
    bounds = list(edges)
    if extend in ("min", "both"):
        bounds = [-np.inf, *bounds]
    if extend in ("max", "both"):
        bounds = [*bounds, np.inf]
    assert len(colours) == len(bounds) - 1
    return {(bounds[k], bounds[k + 1]): colours[k] for k in range(len(colours))}, extend


_FILL_LEVELS = {"int": 6, "list": [5.0, 7.0, 8.0, 12.0, 14.0]}


@pytest.mark.parametrize(
    ("name", "options"),
    [
        ("default", {}),
        ("clipped_low", {"vmin": 9.0}),
        ("clipped_high", {"vmax": 14.0}),
        ("clipped_both", {"vmin": 8.0, "vmax": 14.0}),
        ("int", {"vmin": 8.0, "vmax": 14.0, "fill_levels": 6}),
        (
            "list",
            {"vmin": 5.0, "vmax": 14.0, "fill_levels": [5.0, 7.0, 8.0, 12.0, 14.0]},
        ),
    ],
)
def test_every_band_is_the_colour_matplotlibs_contourf_gives_it(name, options):
    obj = _draw("section", [_section_item()], mark="contourf", **options)
    (polygons,) = _exact(obj, hv.Polygons)
    expected, extend = _matplotlib_band_colours(
        _section_field(), polygons, log=False, name=name
    )
    got = _drawn_band_colours(obj)
    assert got, "no band was drawn"
    for band, colour in got.items():
        assert colour == expected[band], (band, name)
    if name.startswith("clipped"):
        assert extend != "neither"  # these really do exercise a clipped end


def test_a_log_section_is_coloured_by_each_bands_geometric_midpoint():
    field = _section_field() * 0.0 + np.geomspace(0.02, 20.0, 12 * 20).reshape(12, 20)
    obj = _draw(
        "section",
        [_section_item(field, standard_name=CHLOROPHYLL, units="mg m-3")],
        mark="contourf",
    )
    (polygons,) = _exact(obj, hv.Polygons)
    expected, _ = _matplotlib_band_colours(field, polygons, log=True, name="log")
    got = _drawn_band_colours(obj)
    assert got
    for band, colour in got.items():
        assert colour == expected[band], band


def test_data_past_a_clipped_end_is_filled_in_the_colormaps_end_colours():
    obj = _draw("section", [_section_item()], mark="contourf", vmin=9.0, vmax=14.0)
    (polygons,) = _exact(obj, hv.Polygons)
    bands = {(float(b["lower"]), float(b["upper"])) for b in polygons.data}
    # one open-ended band at each end, rather than a hole where the data runs past
    assert any(np.isneginf(lower) for lower, _ in bands)
    assert any(np.isposinf(upper) for _, upper in bands)
    colours = _drawn_band_colours(obj)
    first = min(colours, key=lambda b: b[0])
    last = max(colours, key=lambda b: b[1])
    from bokeh.models import GlyphRenderer, MultiPolygons

    (renderer,) = [
        r
        for r in _bokeh(obj).select({"type": GlyphRenderer})
        if isinstance(r.glyph, MultiPolygons)
    ]
    mapper = renderer.glyph.fill_color.transform
    assert colours[first] == mapper.low_color
    assert colours[last] == mapper.high_color


# -- mark="contourf": the colour bar ------------------------------------------------


def test_the_banded_fills_colour_bar_keeps_its_round_fixed_ticks():
    from bokeh.models import ColorBar, FixedTicker

    obj = _draw("section", [_section_item()], mark="contourf")
    (polygons,) = _exact(obj, hv.Polygons)
    opts = polygons.opts.get("plot").kwargs["colorbar_opts"]
    lo, hi = _clim(polygons)
    ticks = colorbar_ticks(lo, hi, log=False)
    assert isinstance(opts["ticker"], FixedTicker)
    assert list(opts["ticker"].ticks) == [float(v) for v in ticks.values]
    assert opts["major_label_overrides"] == {
        float(v): text for v, text in zip(ticks.values, ticks.labels, strict=True)
    }
    # and on the bar bokeh actually draws: one bar (the cells bring none), still fixed
    (bar,) = _bokeh(obj).select({"type": ColorBar})
    assert isinstance(bar.ticker, FixedTicker)
    assert set(bar.major_label_overrides) == set(bar.ticker.ticks)


def test_a_clipped_end_of_a_banded_bar_keeps_its_end_label():
    obj = _draw("section", [_section_item()], mark="contourf", vmax=14.0)
    (polygons,) = _exact(obj, hv.Polygons)
    labels = polygons.opts.get("plot").kwargs["colorbar_opts"]["major_label_overrides"]
    assert labels[14.0].startswith("≥")


def test_each_row_panel_has_one_bar_with_round_ticks():
    from bokeh.models import ColorBar, FixedTicker

    bars = list(
        _bokeh(_draw("section_row", [_row_item()], mark="contourf")).select(
            {"type": ColorBar}
        )
    )
    assert len(bars) == 3
    assert all(isinstance(bar.ticker, FixedTicker) for bar in bars)


# -- the overlay: lines of a second variable ----------------------------------------


def _row_lines(row) -> list[list]:
    """For each of a row's three panels, its ``hv.Contours`` elements."""
    return [_exact(panel, hv.Contours) for panel in _panels(row)]


def _pooled_levels(pair, row_item=None) -> tuple[float, ...]:
    """Return the levels the shared helper picks for an overlay pair."""
    item = row_item or _row_item()
    values = prepare_section(item["aligned"]["test"])[0]
    arrays = [prepare_overlay(raw, values) for raw in pair]
    return contour_levels(True, arrays)


@pytest.mark.parametrize("mark", ["pcolormesh", "contourf"])
def test_lines_go_on_the_test_and_reference_panels_and_never_the_difference(mark):
    pair = _temperature_pair()
    row = _draw("section_row", [_row_item(contour=pair)], mark=mark)
    test, reference, difference = _row_lines(row)
    assert len(test) == len(reference) == 1 and difference == []
    for panel in _panels(row)[:2]:
        assert len(_exact(panel, hv.Labels)) == 1
    assert _exact(_panels(row)[2], hv.Labels) == []


def test_the_lines_levels_are_the_figures_pooled_contour_levels():
    pair = _temperature_pair()
    row = _draw("section_row", [_row_item(contour=pair)])
    (test,), (reference,), _ = _row_lines(row)
    expected = _pooled_levels(pair)
    assert expected == (20.0, 30.0, 40.0, 50.0)
    # a level the data does not cross on a panel simply has no line there (50 °C is the
    # test overlay's own maximum, not strictly inside it)
    assert _drawn_levels(test) == [20.0, 30.0, 40.0]
    assert _drawn_levels(reference) == list(expected)


def test_the_labels_read_contour_label_of_each_level_a_panel_draws():
    row = _draw("section_row", [_row_item(contour=_temperature_pair())])
    for panel in _panels(row)[:2]:
        (contours,), (labels,) = _exact(panel, hv.Contours), _exact(panel, hv.Labels)
        assert _label_texts(labels) == [
            contour_label(level) for level in _drawn_levels(contours)
        ]


def test_a_labels_anchor_lies_on_a_line_of_its_level():
    row = _draw("section_row", [_row_item(contour=_temperature_pair())])
    test = _panels(row)[0]
    (contours,), (labels,) = _exact(test, hv.Contours), _exact(test, hv.Labels)
    for record in labels.data.to_dict("records"):
        level = float(record["text"].replace("−", "-"))
        points = np.concatenate(
            [
                np.column_stack([line["distance"], line["depth"]])
                for line in contours.data
                if line["level"] == level
            ]
        )
        assert np.any(
            np.isclose(points, [record["distance"], record["depth"]]).all(axis=1)
        )


def test_the_hover_reads_the_overlays_name_level_and_units():
    pair = _temperature_pair()
    row = _draw("section_row", [_row_item(contour=pair)])
    (contours,), _, _ = _row_lines(row)
    assert {line["text"] for line in contours.data} == {"20 °C", "30 °C", "40 °C"}
    assert contours.opts.get("plot").kwargs["hover_tooltips"] == [
        ("temperature", "@text")
    ]
    assert contours.opts.get("plot").kwargs["show_legend"] is False


def test_the_lines_are_black_and_thin_and_the_labels_black_on_white():
    from ocean_skill.plot.section import CONTOUR_COLOR, CONTOUR_WIDTH
    from ocean_skill.plot.typography import PT_PER_CSS_PX

    row = _draw("section_row", [_row_item(contour=_temperature_pair())])
    test = _panels(row)[0]
    (contours,), (labels,) = _exact(test, hv.Contours), _exact(test, hv.Labels)
    style = contours.opts.get("style").kwargs
    assert style["color"] == CONTOUR_COLOR
    assert style["line_width"] == pytest.approx(CONTOUR_WIDTH * PT_PER_CSS_PX)
    assert style["line_dash"] == "solid"
    label_style = labels.opts.get("style").kwargs
    assert label_style["text_color"] == CONTOUR_COLOR
    assert label_style["background_fill_color"] == "white"
    assert label_style["text_font_size"].endswith("pt")


def test_lines_and_labels_sit_above_the_fill_and_the_hover_cells():
    row = _draw(
        "section_row", [_row_item(contour=_temperature_pair())], mark="contourf"
    )
    kinds = [type(e) for e in _panels(row)[0]]
    assert kinds == [hv.Polygons, hv.QuadMesh, hv.Contours, hv.Labels]
    kinds = [
        type(e)
        for e in _draw("section_row", [_row_item(contour=_temperature_pair())])[0]
    ]
    assert kinds == [hv.QuadMesh, hv.Contours, hv.Labels]


def test_a_single_section_draws_its_overlay_at_the_default_levels():
    overlay = _section_field(scale=3.0, offset=-4.0)
    obj = _draw("section", [_section_item(contour=overlay)])
    (contours,), (labels,) = _exact(obj, hv.Contours), _exact(obj, hv.Labels)
    values = prepare_section(_section_field())[0]
    expected = contour_levels(True, [prepare_overlay(overlay, values)])
    assert _drawn_levels(contours) == list(expected)
    assert _label_texts(labels) == [contour_label(v) for v in expected]


def test_explicit_contour_levels_are_exactly_those_that_cross():
    row = _draw(
        "section_row",
        [_row_item(contour=_temperature_pair())],
        contour_levels=[15.0, 25.0, 100.0],
    )
    test, reference, _ = _row_lines(row)
    assert _drawn_levels(test[0]) == [15.0, 25.0]  # 100 °C crosses nothing
    assert _drawn_levels(reference[0]) == [15.0, 25.0]


def test_an_int_contour_levels_picks_about_that_many():
    pair = _temperature_pair()
    row = _draw("section_row", [_row_item(contour=pair)], contour_levels=3)
    (test,), _, _ = _row_lines(row)
    values = prepare_section(_section_field())[0]
    expected = contour_levels(3, [prepare_overlay(raw, values) for raw in pair])
    assert set(_drawn_levels(test)) <= set(expected)
    assert 1 <= len(expected) <= 3


def test_stacked_rows_share_one_set_of_levels_decided_over_all_of_them():
    low = (_section_field(scale=2.0, offset=-10.0),) * 2  # 0-26
    high = (_section_field(scale=6.0, offset=0.0),) * 2  # 30-108
    items = [_row_item(contour=low), _row_item(offset=2.0, contour=high)]
    grid = _draw("section_row", items)
    panels = _panels(grid)
    assert len(panels) == 6
    values = prepare_section(_section_field())[0]
    arrays = [prepare_overlay(raw, values) for pair in (low, high) for raw in pair]
    pooled = contour_levels(True, arrays)
    drawn = [_drawn_levels(c) for panel in panels for c in _exact(panel, hv.Contours)]
    assert len(drawn) == 4  # two rows, test and reference each, none on a difference
    assert all(set(levels) <= set(pooled) for levels in drawn)
    # the low rows' own automatic levels would be finer than the shared ones
    own = contour_levels(True, [prepare_overlay(raw, values) for raw in low])
    assert set(drawn[0]) != set(own) and set(drawn[0]) <= set(pooled)
    # one set: wherever two panels both cross a level, it is the same number
    assert set().union(*map(set, drawn)) <= set(pooled)


def test_a_grid_of_sections_shares_one_set_of_levels_too():
    low = _section_field(scale=2.0, offset=-10.0)
    high = _section_field(scale=6.0, offset=0.0)
    items = [_section_item(contour=low), _section_item(contour=high)]
    grid = _draw("section", items)
    drawn = [_drawn_levels(c) for c in _exact(grid, hv.Contours)]
    assert len(drawn) == 2
    values = prepare_section(_section_field())[0]
    pooled = contour_levels(True, [prepare_overlay(o, values) for o in (low, high)])
    assert all(set(levels) <= set(pooled) for levels in drawn)
    assert set(drawn[0]) != set(contour_levels(True, [prepare_overlay(low, values)]))


def test_an_item_without_an_overlay_among_overlaid_ones_just_has_no_lines():
    overlay = _section_field(scale=3.0, offset=-4.0)
    items = [_section_item(contour=overlay), _section_item()]
    assert len(_exact(_draw("section", items), hv.Contours)) == 1


# -- contour_kwargs -----------------------------------------------------------------


def _first_lines(**contour_kwargs):
    row = _draw(
        "section_row",
        [_row_item(contour=_temperature_pair())],
        contour_kwargs=contour_kwargs,
    )
    test = _panels(row)[0]
    return _exact(test, hv.Contours)[0], _exact(test, hv.Labels)


def test_contour_kwargs_colors_linewidths_and_linestyles_restyle_the_lines():
    from ocean_skill.plot.typography import PT_PER_CSS_PX

    contours, (labels,) = _first_lines(colors="red", linewidths=2, linestyles="dashed")
    style = contours.opts.get("style").kwargs
    assert style["color"] == "#ff0000"
    assert style["line_width"] == pytest.approx(2 * PT_PER_CSS_PX)
    assert style["line_dash"] == "dashed"
    # the labels follow the lines' colour, as matplotlib's clabel does
    assert labels.opts.get("style").kwargs["text_color"] == "#ff0000"


@pytest.mark.parametrize(
    ("given", "dash"),
    [
        ("-", "solid"),
        ("--", "dashed"),
        (":", "dotted"),
        ("-.", "dashdot"),
        ("dotted", "dotted"),
    ],
)
def test_matplotlibs_linestyle_spellings_map_to_bokehs(given, dash):
    contours, _ = _first_lines(linestyles=given)
    assert contours.opts.get("style").kwargs["line_dash"] == dash


def test_a_one_item_list_is_taken_like_the_item():
    contours, _ = _first_lines(colors=["blue"], linewidths=[1.5], linestyles=["dotted"])
    style = contours.opts.get("style").kwargs
    assert (style["color"], style["line_dash"]) == ("#0000ff", "dotted")


def test_contour_kwargs_fmt_changes_every_label_and_the_hover():
    contours, (labels,) = _first_lines(fmt="%.1f")
    assert _label_texts(labels) == [f"{v:.1f}" for v in _drawn_levels(contours)]
    assert {line["text"] for line in contours.data} == {"20.0 °C", "30.0 °C", "40.0 °C"}


def test_contour_kwargs_labels_false_draws_the_lines_alone():
    contours, labels = _first_lines(labels=False)
    assert len(contours.data) and labels == []


def test_a_static_only_contour_kwarg_warns_without_raising():
    with pytest.warns(UserWarning, match="only affect the static") as record:
        contours, _ = _first_lines(colors="red", alpha=0.4, zorder=3)
    text = " ".join(str(w.message) for w in record)
    assert "'alpha'" in text and "'zorder'" in text and "'colors'" not in text
    assert contours.opts.get("style").kwargs["color"] == "#ff0000"  # the rest applied


def test_a_value_only_matplotlib_can_use_warns_too():
    with pytest.warns(UserWarning, match="only affect the static"):
        _first_lines(colors=["red", "blue"])


def test_a_stacked_grid_warns_about_a_static_only_key_once_not_per_panel():
    items = [_row_item(contour=_temperature_pair()) for _ in range(3)]
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _draw("section_row", items, contour_kwargs={"alpha": 0.4})
    assert sum("only affect the static" in str(w.message) for w in record) == 1


def test_contour_kwargs_levels_is_refused_as_it_is_statically():
    with pytest.raises(ValueError, match="contour_levels="):
        _first_lines(levels=[1.0, 2.0])


def test_contour_kwargs_must_be_a_mapping_and_fmt_a_percent_format():
    with pytest.raises(TypeError, match="contour_kwargs="):
        _draw(
            "section", [_section_item(contour=_section_field())], contour_kwargs="red"
        )
    with pytest.raises(ValueError, match="fmt="):
        _first_lines(fmt="{:.1f}")


# -- refusals -----------------------------------------------------------------------


def test_an_overlay_on_a_different_mesh_is_refused_never_regridded():
    coarse = _section_field(n_z=8)
    with pytest.raises(ValueError, match="different meshes"):
        _draw("section", [_section_item(contour=coarse)])
    pair = (coarse, _section_field())
    with pytest.raises(ValueError, match="different meshes"):
        _draw("section_row", [_row_item(contour=pair)])


@pytest.mark.parametrize("family", ["section", "section_row"])
@pytest.mark.parametrize("option", ["contour_levels", "contour_kwargs"])
def test_a_contour_option_without_an_overlay_is_refused(family, option):
    items = [_section_item()] if family == "section" else [_row_item()]
    value = [10.0] if option == "contour_levels" else {"colors": "red"}
    with pytest.raises(ValueError, match="no panel here has an overlay"):
        _draw(family, items, **{option: value})


@pytest.mark.parametrize("family", ["section", "section_row"])
def test_fill_levels_without_contourf_is_refused(family):
    items = [_section_item()] if family == "section" else [_row_item()]
    with pytest.raises(ValueError, match="draws cells"):
        _draw(family, items, fill_levels=6)
    with pytest.raises(ValueError, match="draws cells"):
        _draw(family, items, mark="pcolormesh", fill_levels=6)


def test_an_invalid_fill_levels_is_refused_before_any_panel_is_drawn():
    with pytest.raises((TypeError, ValueError), match=r"fill_levels|levels spec|count"):
        _draw("section", [_section_item()], mark="contourf", fill_levels="many")


def test_every_family_checks_its_options_before_drawing():
    items = [_row_item(), _row_item(offset=2.0)]
    with pytest.raises(ValueError, match="no panel here has an overlay"):
        _draw("section_row", items, contour_levels=3)
    items = [_section_item(), _section_item()]
    with pytest.raises(ValueError, match="no panel here has an overlay"):
        _draw("section", items, contour_levels=3)


# -- everything builds into a bokeh figure ------------------------------------------


@pytest.mark.parametrize(
    "build",
    [
        lambda: _draw(
            "section",
            [_section_item(contour=_section_field(scale=3.0, offset=-4.0))],
            mark="contourf",
        ),
        lambda: _draw(
            "section_row",
            [_row_item(contour=_temperature_pair())],
            mark="contourf",
            fill_levels=8,
            contour_levels=4,
            contour_kwargs={"colors": "navy", "linestyles": "dashed"},
        ),
        lambda: _draw(
            "section_row",
            [_row_item(contour=_temperature_pair()), _row_item(offset=2.0)],
            mark="contourf",
        ),
    ],
    ids=["section", "row", "stacked rows"],
)
def test_a_banded_figure_with_an_overlay_builds_into_a_bokeh_plot(build):
    from bokeh.models import GlyphRenderer, HoverTool, MultiLine, MultiPolygons

    plot = _bokeh(build())
    glyphs = [type(r.glyph) for r in plot.select({"type": GlyphRenderer})]
    assert MultiPolygons in glyphs and MultiLine in glyphs
    # one hover per line layer, reading the overlay's name and the level's text
    tooltips = [tool.tooltips for tool in plot.select({"type": HoverTool})]
    assert [("temperature", "@{text}")] in tooltips


def test_a_frame_with_lines_keeps_the_sections_axes():
    """The y axis stays inverted (shallow at the top) with the overlay on top."""
    from bokeh.models import GlyphRenderer, MultiLine

    obj = _draw(
        "section",
        [_section_item(contour=_section_field(scale=3.0, offset=-4.0))],
        mark="contourf",
    )
    plot = _bokeh(obj)
    assert plot.y_range.start > plot.y_range.end or getattr(
        plot.y_range, "flipped", False
    )
    assert any(
        isinstance(r.glyph, MultiLine) for r in plot.select({"type": GlyphRenderer})
    )


def test_rows_with_different_x_axes_keep_their_own_x_name_for_bands_and_lines():
    """Bokeh links axes by name, so rows of different extent get aliased x names."""
    items = [
        _row_item(contour=_temperature_pair()),
        _row_item(length=100.0, contour=_temperature_pair(length=100.0)),
    ]
    grid = _draw("section_row", items, mark="contourf")
    x_names = {e.kdims[0].name for e in _exact(grid, hv.Polygons)}
    assert x_names == {"distance_0", "distance_1"}
    line_x = {e.kdims[0].name for e in _exact(grid, hv.Contours)}
    assert line_x == {"distance_0", "distance_1"}
    assert _bokeh(grid) is not None


def test_a_native_s_section_is_banded_and_lined_on_its_own_curved_mesh():
    overlay = _native_s_field(slope=3.0)
    obj = _draw(
        "section",
        [_section_item(_native_s_field(), contour=overlay)],
        mark="contourf",
    )
    (polygons,) = _exact(obj, hv.Polygons)
    (contours,) = _exact(obj, hv.Contours)
    assert len(polygons.data) > 3 and len(contours.data) > 0
    # the depth really varies along a line of constant sigma: the mesh is not a grid
    depths = np.concatenate([np.ravel(band["depth"]) for band in polygons.data])
    assert np.isfinite(depths[~np.isnan(depths)]).all() and depths.max() > 300
    assert _bokeh(obj) is not None
