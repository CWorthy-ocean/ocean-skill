"""Interactive vertical sections built from casts: cast lines, names, the seafloor.

The holoviews renderer's twin of the static cast markers. What is drawn where is decided
by :func:`~ocean_skill.plot.section.cast_marks`, ``seafloor_line`` and ``depth_limit``
-- shared with the static renderer, tested with it -- so these tests hold the
*drawing* to account:

* every panel of a row (test, reference, difference) carries one dashed ``Segments``
  line per cast, from the surface to the cast's deepest observation, and its name as
  ``Labels``; a cast with no observation is named but draws no line;
* a seafloor is a filled ``Area`` *under* the data (a missing cell is transparent, so
  the rock shows through) and a thin ``Curve`` outline over it, under any contour
  lines, and the depth axis then runs from the surface to ``depth_limit``;
* the cell mesh stays the only element with a hover;
* ``cast_fill`` resamples the data (and any contour overlay) onto a fine x grid for
  drawing, so a deep cast between shallow ones keeps its colour out to halfway, while
  the cast marks stay at the casts;
* a panel with a seafloor has a white background (open water), one without keeps the
  grey of "no data";
* an item without the keys draws exactly what it always did.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import xarray as xr
from matplotlib.colors import to_hex

from ocean_skill.align import ALONG_DIM
from ocean_skill.plot.registry import render
from ocean_skill.plot.section import (
    CAST_COLOR,
    CAST_LABEL_COLOR,
    CAST_WIDTH,
    SEAFLOOR_COLOR,
    WATER_COLOR,
    depth_limit,
    prepare_section_row,
    seafloor_line,
)
from ocean_skill.plot.spec import PlotSpec
from ocean_skill.plot.typography import PT_PER_CSS_PX

hv = pytest.importorskip("holoviews")
pytest.importorskip("hvplot")

from test_section_contours_holoviews import (  # noqa: E402
    _exact,
    _row_item,
    _section_field,
)

N_CASTS = 5
LABELS = [f"HV{i}" for i in range(N_CASTS)]

# -- fixtures -----------------------------------------------------------------------


def _draw(items, **options):
    return render(
        PlotSpec(family="section_row", items=items, options=options),
        renderer="holoviews",
    )


def _cast_item(*, length: float = 150.0, depth_of_floor: float = 260.0, labels=LABELS):
    """Return a ``section_row`` item of five casts, with a seafloor under them.

    Cast 1 has nothing deeper than 100 m, and cast 3 has no observation at all -- the
    shapes a real station set has -- both in the reference lane (the one the marks are
    read from).
    """
    item = _row_item(length=length)
    item["aligned"] = dict(item["aligned"])
    reference = _section_field(offset=1.0, n_along=N_CASTS, length=length)
    reference = reference.where(
        ~((reference["z"] < -100.0) & (reference[ALONG_DIM] == reference[ALONG_DIM][1]))
    )
    reference[:, 3] = np.nan
    test = _section_field(n_along=N_CASTS, length=length)
    item["aligned"] = {
        "test": test.rename("test"),
        "reference": reference.rename("reference"),
        "difference": (test - reference).rename("difference"),
    }
    n = 40
    along = np.linspace(0.0, length, n)
    item["seafloor"] = xr.DataArray(
        np.linspace(80.0, depth_of_floor, n),
        dims=(ALONG_DIM,),
        coords={
            ALONG_DIM: along,
            "path_lon": (ALONG_DIM, np.linspace(-95.0, -93.0, n)),
            "path_lat": (ALONG_DIM, np.linspace(24.0, 26.0, n)),
        },
    )
    if labels is not None:
        item["cast_labels"] = list(labels)
    return item


def _panels(row) -> list:
    return list(row)


def _bokeh(obj):
    return hv.render(obj, backend="bokeh")


def _kinds(panel) -> list[str]:
    return [type(e).__name__ for e in panel]


def _reference_marks(item, section_x="distance"):
    """Return ``(xs, bottoms)`` the reference lane's casts should be drawn at."""
    values, _ = prepare_section_row(item["aligned"], section_x)
    reference = values["reference"]
    depth = np.asarray(reference["depth"].transpose(ALONG_DIM, ...), dtype="float64")
    data = np.asarray(reference.transpose(ALONG_DIM, ...), dtype="float64")
    bottoms = [
        np.max(depth[i][np.isfinite(data[i])]) if np.isfinite(data[i]).any() else np.nan
        for i in range(data.shape[0])
    ]
    return np.asarray(reference[ALONG_DIM], dtype="float64"), np.asarray(bottoms)


# -- the cast lines and their names -------------------------------------------------


def test_every_panel_carries_a_line_per_reaching_cast_and_every_name():
    item = _cast_item()
    row = _draw([item], section_x="distance")
    xs, bottoms = _reference_marks(item)
    reaching = np.isfinite(bottoms)
    assert reaching.sum() == N_CASTS - 1  # the fixture's cast 3 has nothing
    for panel in _panels(row):
        (segments,) = _exact(panel, hv.Segments)
        (labels,) = _exact(panel, hv.Labels)
        x0, y0, x1, y1 = (segments.dimension_values(i) for i in range(4))
        np.testing.assert_allclose(x0, xs[reaching])
        np.testing.assert_allclose(x1, xs[reaching])  # vertical
        np.testing.assert_allclose(y0, 0.0)  # from the surface
        np.testing.assert_allclose(y1, bottoms[reaching])
        # one name per cast, the unreaching one too, at its own x
        assert list(labels.dimension_values("text")) == LABELS
        np.testing.assert_allclose(labels.dimension_values(0).astype(float), xs)


def test_the_cast_that_stops_at_100_m_draws_a_shorter_line():
    item = _cast_item()
    _, bottoms = _reference_marks(item)
    assert bottoms[1] < 105.0 < bottoms[0]  # fixture sanity: cast 1 is the shallow one
    (segments,) = _exact(_panels(_draw([item], section_x="distance"))[0], hv.Segments)
    drawn = segments.dimension_values(3)
    assert drawn[1] == pytest.approx(bottoms[1])


def test_cast_lines_are_dashed_in_the_cast_colour_and_names_sit_inside_the_top():
    row = _draw([_cast_item()], section_x="distance")
    (segments,) = _exact(_panels(row)[0], hv.Segments)
    style = segments.opts.get("style").kwargs
    assert style["line_dash"] == "dashed"
    assert style["color"] == to_hex(CAST_COLOR)  # as bokeh spells it
    # thinner than a contour line, in CSS pixels
    assert style["line_width"] == pytest.approx(CAST_WIDTH * PT_PER_CSS_PX)
    (labels,) = _exact(_panels(row)[0], hv.Labels)
    assert labels.opts.get("style").kwargs["text_baseline"] == "top"
    # the surface is where the names sit, and a few pixels below it on screen
    assert set(labels.dimension_values(1)) == {0.0}
    glyph = next(
        r.glyph
        for r in _bokeh(_panels(row)[0]).renderers
        if type(r.glyph).__name__ == "Text"
    )
    assert glyph.y_offset < 0  # bokeh's offset is up-positive, so this is down


def test_cast_lines_and_names_are_the_topmost_layers():
    kinds = _kinds(_panels(_draw([_cast_item()], section_x="distance"))[0])
    assert kinds[-2:] == ["Segments", "Labels"]


def test_a_label_count_that_does_not_match_the_columns_raises():
    item = _cast_item(labels=LABELS[:-1])
    with pytest.raises(ValueError, match="cast label"):
        _draw([item])


def test_labels_only_draws_names_and_lines_without_a_seafloor():
    item = _cast_item()
    del item["seafloor"]
    panel = _panels(_draw([item], section_x="distance"))[0]
    assert _kinds(panel) == ["QuadMesh", "Segments", "Labels"]
    # no seafloor: the depth axis is the data's own, not forced to start at 0
    (labels,) = _exact(panel, hv.Labels)
    values, _ = prepare_section_row(item["aligned"], "distance")
    shallowest = min(float(np.nanmin(da["depth"])) for da in values.values())
    assert set(labels.dimension_values(1)) == {shallowest}


# -- the seafloor -------------------------------------------------------------------


def test_a_seafloor_is_a_fill_under_the_data_and_an_outline_over_it():
    item = _cast_item()
    row = _draw([item], section_x="distance")
    line_x, line_depth = seafloor_line(
        item["seafloor"], prepare_section_row(item["aligned"], "distance")[1]
    )
    for panel in _panels(row):
        kinds = _kinds(panel)
        assert kinds == ["Area", "QuadMesh", "Curve", "Segments", "Labels"]
        (area,) = _exact(panel, hv.Area)
        (outline,) = _exact(panel, hv.Curve)
        np.testing.assert_allclose(area.dimension_values(0), line_x)
        np.testing.assert_allclose(area.dimension_values(1), line_depth)
        np.testing.assert_allclose(outline.dimension_values(0), line_x)
        np.testing.assert_allclose(outline.dimension_values(1), line_depth)
        style = area.opts.get("style").kwargs
        assert style["fill_color"] == to_hex(SEAFLOOR_COLOR)
        assert style["line_alpha"] == 0
        assert outline.opts.get("style").kwargs["color"] == "#000000"


def test_the_fill_runs_from_the_line_down_to_the_axis_bottom():
    item = _cast_item()
    panel = _panels(_draw([item], section_x="distance"))[0]
    (area,) = _exact(panel, hv.Area)
    values, _ = prepare_section_row(item["aligned"], "distance")
    bottom = depth_limit(list(values.values()), np.asarray(item["seafloor"]))
    np.testing.assert_allclose(area.dimension_values(2), bottom)


def test_the_depth_axis_runs_from_the_surface_to_the_depth_limit():
    item = _cast_item(depth_of_floor=260.0)
    values, _ = prepare_section_row(item["aligned"], "distance")
    bottom = depth_limit(list(values.values()), np.asarray(item["seafloor"]))
    assert bottom == pytest.approx(260.0)
    for panel in _panels(_draw([item], section_x="distance")):
        fig = _bokeh(panel)
        # depth reads shallow at the top: the range runs from the bottom up to 0
        assert fig.y_range.start == pytest.approx(bottom)
        assert fig.y_range.end == pytest.approx(0.0)


def test_a_seafloor_shallower_than_the_observations_never_cuts_them_off():
    item = _cast_item(depth_of_floor=120.0)
    values, _ = prepare_section_row(item["aligned"], "distance")
    deepest = max(float(np.nanmax(da["depth"])) for da in values.values())
    assert deepest > 120.0
    fig = _bokeh(_panels(_draw([item], section_x="distance"))[0])
    assert max(fig.y_range.start, fig.y_range.end) == pytest.approx(deepest)


def test_the_seafloor_is_placed_by_longitude_when_the_axis_is_longitude():
    item = _cast_item()
    panel = _panels(_draw([item], section_x="lon"))[0]
    (outline,) = _exact(panel, hv.Curve)
    np.testing.assert_allclose(
        outline.dimension_values(0), np.asarray(item["seafloor"]["path_lon"])
    )


def test_the_outline_goes_under_any_contour_lines():
    item = _cast_item()
    # a 'temperature' pair on the casts' own mesh, spanning the isotherm levels
    ref = item["aligned"]["reference"]
    item |= {
        "contour": {
            lane: ref * 3.0 - 4.0 + k for k, lane in enumerate(("test", "reference"))
        },
        "contour_standard_name": "sea_water_temperature",
        "contour_units": "degC",
    }
    kinds = _kinds(_panels(_draw([item], section_x="distance"))[0])
    assert kinds.index("Curve") < kinds.index("Contours")
    assert kinds.index("Contours") < kinds.index("Segments")
    assert kinds[0] == "Area"


def test_the_mesh_stays_the_only_element_with_a_hover():
    from bokeh.models import HoverTool

    for mark in ("pcolormesh", "contourf"):
        for panel in _panels(_draw([_cast_item()], section_x="distance", mark=mark)):
            fig = _bokeh(panel)
            drawn = {
                type(r.glyph).__name__
                for h in fig.select({"type": HoverTool})
                for r in h.renderers
            }
            assert drawn
            assert not drawn & {"Segment", "Text", "Line", "Patch"}, drawn


def test_contourf_keeps_its_bands_between_the_seafloor_and_its_cells():
    kinds = _kinds(
        _panels(_draw([_cast_item()], section_x="distance", mark="contourf"))[0]
    )
    assert kinds == ["Area", "Polygons", "QuadMesh", "Curve", "Segments", "Labels"]


def test_each_panel_keeps_its_own_title_and_the_colour_bar():
    from bokeh.models import ColorBar

    row = _draw([_cast_item()], section_x="distance")
    titles = [_bokeh(p).title.text for p in _panels(row)]
    assert titles[:2] == ["test", "reference"]
    assert titles[2].startswith("difference")
    assert all(len(_bokeh(p).select({"type": ColorBar})) == 1 for p in _panels(row))


# -- styling ------------------------------------------------------------------------


def test_cast_kwargs_restyle_the_lines_not_the_names():
    row = _draw(
        [_cast_item()],
        section_x="distance",
        cast_kwargs={"colors": "tab:red", "linestyles": ":", "linewidths": 2.0},
    )
    panel = _panels(row)[0]
    (segments,) = _exact(panel, hv.Segments)
    style = segments.opts.get("style").kwargs
    assert style["color"] == "#d62728"
    assert style["line_dash"] == "dotted"
    assert style["line_width"] > 2.0  # points, drawn at 4/3 pixel each
    # the names stay black, like the static renderer's tick labels: colors= is the
    # lines' colour, not theirs
    (labels,) = _exact(panel, hv.Labels)
    assert labels.opts.get("style").kwargs["text_color"] == to_hex(CAST_LABEL_COLOR)


def test_cast_kwargs_labels_false_leaves_the_names_off():
    panel = _panels(
        _draw([_cast_item()], section_x="distance", cast_kwargs={"labels": False})
    )[0]
    assert _exact(panel, hv.Labels) == []
    assert len(_exact(panel, hv.Segments)) == 1


def test_seafloor_kwargs_style_the_fill_and_the_outline():
    row = _draw(
        [_cast_item()],
        section_x="distance",
        seafloor_kwargs={
            "color": "tab:brown",
            "alpha": 0.5,
            "edgecolor": "tab:red",
            "linewidth": 2.0,
        },
    )
    panel = _panels(row)[0]
    (area,) = _exact(panel, hv.Area)
    fill = area.opts.get("style").kwargs
    assert fill["fill_color"] == "#8c564b" and fill["fill_alpha"] == 0.5
    (outline,) = _exact(panel, hv.Curve)
    line = outline.opts.get("style").kwargs
    assert line["color"] == "#d62728" and line["line_width"] > 2.0


def test_edgecolor_none_draws_the_fill_with_no_outline():
    panel = _panels(
        _draw(
            [_cast_item()],
            section_x="distance",
            seafloor_kwargs={"edgecolor": "none"},
        )
    )[0]
    assert len(_exact(panel, hv.Area)) == 1
    assert _exact(panel, hv.Curve) == []


def test_keys_only_the_static_renderer_has_are_named_in_one_warning():
    with pytest.warns(UserWarning, match=r"seafloor_kwargs \['hatch'\]"):
        _draw([_cast_item()], section_x="distance", seafloor_kwargs={"hatch": "//"})
    with pytest.warns(UserWarning, match=r"cast_kwargs \['zorder'\]"):
        _draw([_cast_item()], section_x="distance", cast_kwargs={"zorder": 3})


def test_valid_style_keys_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _draw(
            [_cast_item()],
            section_x="distance",
            cast_kwargs={
                "color": "k",
                "linestyle": "--",
                "linewidth": 1,
                "rotation": 30,
            },
            seafloor_kwargs={"facecolor": "0.3", "alpha": 0.8},
        )


# -- nothing new without the keys ---------------------------------------------------


def test_an_item_without_the_keys_draws_exactly_the_old_panels():
    row = _draw([_row_item()])
    for panel in _panels(row):
        assert type(panel) is hv.QuadMesh
    row = _draw([_row_item()], mark="contourf")
    for panel in _panels(row):
        assert _kinds(panel) == ["Polygons", "QuadMesh"]
    fig = _bokeh(_panels(_draw([_row_item()]))[0])
    assert fig.y_range.start > fig.y_range.end  # the data's own, shallow at the top


def test_style_options_alone_draw_nothing():
    row = _draw(
        [_row_item()],
        cast_kwargs={"colors": "r"},
        seafloor_kwargs={"color": "r"},
    )
    assert all(type(p) is hv.QuadMesh for p in _panels(row))


# -- stacked grids ------------------------------------------------------------------


def test_a_grid_reads_the_keys_per_item():
    items = [_cast_item(), _row_item(length=150.0)]
    grid = _draw(items, section_x="distance")
    assert len(_exact(grid, hv.Segments)) == 3  # the first row's three panels only
    assert len(_exact(grid, hv.Area)) == 3
    assert len(_exact(grid, hv.Labels)) == 3
    assert len(_exact(grid, hv.QuadMesh)) == 6


def test_grid_rows_with_different_x_extents_keep_the_new_layers_on_their_own_x():
    items = [_cast_item(length=150.0), _cast_item(length=300.0)]
    grid = _draw(items, section_x="distance")
    panels = list(grid)
    assert len(panels) == 6
    names = []
    for panel in panels:
        (mesh,) = _exact(panel, hv.QuadMesh)
        (segments,) = _exact(panel, hv.Segments)
        (outline,) = _exact(panel, hv.Curve)
        mesh_x = mesh.kdims[0].name
        assert segments.kdims[0].name == outline.kdims[0].name == mesh_x
        names.append(mesh_x)
    # the two rows really are aliased apart, so linking by name keeps them unlinked
    assert names[0] == names[1] == names[2] != names[3]
    assert _bokeh(grid) is not None


def test_a_grid_styles_every_cast_row_the_same_and_warns_once():
    items = [_cast_item(), _cast_item()]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        grid = _draw(items, section_x="distance", seafloor_kwargs={"hatch": "//"})
    assert sum("seafloor_kwargs" in str(w.message) for w in caught) == 1
    assert len(_exact(grid, hv.Area)) == 6


def test_a_mismatched_label_count_in_a_grid_row_raises():
    with pytest.raises(ValueError, match="cast label"):
        _draw([_cast_item(), _cast_item(labels=["a"])], section_x="distance")


# -- filling between the casts ------------------------------------------------------


def _deep_cast_item(**extra):
    """Return a three-cast item whose middle reference cast is the only deep one.

    Casts sit at 0, 75 and 150 km; the outer two have nothing below 100 m, the middle
    one reaches 200 m. Without a fill the deepest level has one column of data; with it,
    the deep cast's colour should reach halfway to each neighbour.
    """
    item = _cast_item(labels=None)
    length = 150.0
    reference = _section_field(offset=1.0, n_along=3, length=length)
    deep = reference["z"] < -100.0
    reference = reference.where(~(deep & (reference[ALONG_DIM] != 75.0)))
    test = _section_field(n_along=3, length=length)
    item["aligned"] = {
        "test": test.rename("test"),
        "reference": reference.rename("reference"),
        "difference": (test - reference).rename("difference"),
    }
    item["cast_labels"] = ["A", "B", "C"]
    return item | extra


def _mesh(panel):
    (mesh,) = _exact(panel, hv.QuadMesh)
    return mesh


def _mesh_columns(panel) -> int:
    return _mesh(panel).data.sizes[ALONG_DIM]


def test_cast_fill_resamples_the_mesh_onto_many_more_columns_than_casts():
    for panel in _panels(
        _draw([_cast_item() | {"cast_fill": True}], section_x="distance")
    ):
        assert _mesh_columns(panel) > 100  # five casts, drawn on a fine grid
        assert _mesh(panel).data[_mesh(panel).vdims[0].name].dims == ("z", ALONG_DIM)


def test_without_cast_fill_the_mesh_has_one_column_per_cast():
    for item in (_cast_item(), _cast_item() | {"cast_fill": False}):
        for panel in _panels(_draw([item], section_x="distance")):
            assert _mesh_columns(panel) == N_CASTS


def test_cast_fill_carries_a_deep_cast_halfway_to_its_shallow_neighbours():
    # no seafloor: the fixture's own is shallower than the deep cast, and the fill
    # stops at the rock (see the test below)
    item = _deep_cast_item(cast_fill=True)
    item.pop("seafloor", None)
    panel = _panels(_draw([item], section_x="distance"))[1]  # the reference lane
    data = _mesh(panel).data
    depth = np.asarray(data["depth"])[:, 0]
    deepest = int(np.argmax(depth))
    x = np.asarray(data["distance"])[deepest]
    row = np.asarray(data["reference"])[deepest]
    finite = np.isfinite(row)
    # only the middle cast reaches the bottom: its colour covers the gap out to
    # halfway on both sides (37.5-112.5 km) and nothing beyond
    assert finite.any()
    assert x[finite].min() == pytest.approx(37.5, abs=1.0)
    assert x[finite].max() == pytest.approx(112.5, abs=1.0)
    assert (finite & (x < 75.0)).any() and (finite & (x > 75.0)).any()
    assert finite[np.abs(x - 75.0) <= 35.0].all()
    assert not finite[(x < 36.0) | (x > 114.0)].any()
    # a level every cast reaches is finite right across
    assert np.isfinite(np.asarray(data["reference"])[0]).all()


def test_cast_fill_stops_at_the_seafloor_but_keeps_the_cast_itself():
    item = _deep_cast_item(cast_fill=True)
    assert item.get("seafloor") is not None
    panel = _panels(_draw([item], section_x="distance"))[1]
    data = _mesh(panel).data
    depth = np.asarray(data["depth"])[:, 0]
    x = np.asarray(data["distance"])
    values = np.asarray(data["reference"])
    floor_x, floor_depth = seafloor_line(
        item["seafloor"], prepare_section_row(item["aligned"], "distance")[1]
    )
    floor = np.interp(x[0], floor_x, floor_depth)
    below = depth[:, None] > floor[None, :]
    # the three casts' own columns are kept as measured
    any_cast = np.isin(x[0], [0.0, 75.0, 150.0])
    # nothing between the casts is drawn below the rock...
    assert below[:, ~any_cast].any()
    assert not np.isfinite(values[below & ~any_cast[None, :]]).any()
    # ...but the deep cast's own column keeps its deepest value
    deepest = int(np.argmax(depth))
    assert np.isfinite(values[deepest, np.isclose(x[0], 75.0)]).all()


def test_without_cast_fill_the_deep_cast_is_a_single_column():
    panel = _panels(_draw([_deep_cast_item()], section_x="distance"))[1]
    data = _mesh(panel).data
    deepest = int(np.argmax(np.asarray(data["depth"])[:, 0]))
    assert np.isfinite(np.asarray(data["reference"])[deepest]).sum() == 1


def test_cast_fill_keeps_the_mesh_coordinates_the_hover_reads():
    panel = _panels(_draw([_cast_item() | {"cast_fill": True}], section_x="distance"))[
        0
    ]
    mesh = _mesh(panel)
    assert [d.name for d in mesh.kdims] == ["distance", "depth"]
    fig = _bokeh(panel)
    from bokeh.models import HoverTool

    hover = fig.select({"type": HoverTool})
    assert hover and all(
        type(r.glyph).__name__ not in {"Segment", "Text", "Line"}
        for h in hover
        for r in h.renderers
    )


@pytest.mark.parametrize("mark", ["pcolormesh", "contourf"])
def test_cast_fill_draws_both_marks_on_the_dense_mesh(mark):
    row = _draw([_cast_item() | {"cast_fill": True}], section_x="distance", mark=mark)
    for panel in _panels(row):
        assert _mesh_columns(panel) > 100
        if mark == "contourf":
            assert _exact(panel, hv.Polygons)  # the bands, over the dense mesh


def test_a_contour_overlay_is_drawn_on_the_dense_mesh_too():
    item = _deep_cast_item(cast_fill=True)
    ref = item["aligned"]["test"]
    item |= {
        "contour": {"test": ref * 3.0 - 4.0, "reference": ref * 3.0 - 3.0},
        "contour_standard_name": "sea_water_temperature",
        "contour_units": "degC",
    }
    plain = item | {"cast_fill": False}
    dense = _panels(_draw([item], section_x="distance"))[0]
    sparse = _panels(_draw([plain], section_x="distance"))[0]
    (lines,) = _exact(dense, hv.Contours)
    (sparse_lines,) = _exact(sparse, hv.Contours)
    assert len(lines.data) > 0
    # the same isotherms, but traced through ~400 columns, not 3
    assert {r["level"] for r in lines.data} == {r["level"] for r in sparse_lines.data}

    def vertices(contours):
        return sum(len(r["distance"]) for r in contours.data)

    assert vertices(lines) > 5 * vertices(sparse_lines)


def test_cast_marks_stay_at_the_casts_with_the_fill_on():
    item = _cast_item() | {"cast_fill": True}
    xs, bottoms = _reference_marks(item)
    reaching = np.isfinite(bottoms)
    for panel in _panels(_draw([item], section_x="distance")):
        (segments,) = _exact(panel, hv.Segments)
        (labels,) = _exact(panel, hv.Labels)
        np.testing.assert_allclose(segments.dimension_values(0), xs[reaching])
        np.testing.assert_allclose(segments.dimension_values(3), bottoms[reaching])
        np.testing.assert_allclose(labels.dimension_values(0).astype(float), xs)
        assert list(labels.dimension_values("text")) == LABELS


def test_the_cast_fill_keyword_overrides_the_item():
    item = _cast_item()
    on = _draw([item], section_x="distance", cast_fill=True)
    assert all(_mesh_columns(p) > 100 for p in _panels(on))


def test_the_colour_limits_are_those_of_the_casts_own_data():
    from bokeh.models import ColorBar

    def limits(row):
        bars = [_bokeh(p).select({"type": ColorBar})[0].color_mapper for p in row]
        return [(m.low, m.high) for m in bars]

    item = _cast_item()
    assert limits(_draw([item | {"cast_fill": True}], section_x="distance")) == limits(
        _draw([item], section_x="distance")
    )


# -- the background: open water over a seafloor, "no data" grey without ------------


def test_a_panel_with_a_seafloor_has_a_white_water_background():
    for panel in _panels(_draw([_cast_item()], section_x="distance")):
        assert _bokeh(panel).background_fill_color == to_hex(WATER_COLOR)
    for mark in ("pcolormesh", "contourf"):
        row = _draw(
            [_cast_item() | {"cast_fill": True}], section_x="distance", mark=mark
        )
        assert {_bokeh(p).background_fill_color for p in _panels(row)} == {
            to_hex(WATER_COLOR)
        }


def test_a_panel_without_a_seafloor_keeps_the_no_data_grey():
    item = _cast_item()
    del item["seafloor"]
    for panel in _panels(_draw([item], section_x="distance")):
        assert _bokeh(panel).background_fill_color == "#d9d9d9"
    for panel in _panels(_draw([_row_item()])):
        assert _bokeh(panel).background_fill_color == "#d9d9d9"


# -- the line width -----------------------------------------------------------------


def test_the_default_cast_line_width_is_the_shared_constant():
    (segments,) = _exact(
        _panels(_draw([_cast_item()], section_x="distance"))[0], hv.Segments
    )
    assert segments.opts.get("style").kwargs["line_width"] == pytest.approx(
        CAST_WIDTH * PT_PER_CSS_PX
    )


# -- stacked grids, mixed -----------------------------------------------------------


def test_a_grid_fills_between_casts_only_on_the_rows_that_ask():
    items = [
        _cast_item() | {"cast_fill": True},
        _cast_item(),
        _row_item(length=150.0),
    ]
    grid = _draw(items, section_x="distance")
    panels = list(grid)
    assert len(panels) == 9
    columns = [_mesh_columns(p) for p in panels]
    assert all(c > 100 for c in columns[:3])
    assert columns[3:6] == [N_CASTS] * 3
    assert columns[6:] == [20] * 3  # the gridded row, untouched
    # the seafloor row is white, the plain row grey, whatever its neighbours do
    colours = [_bokeh(p).background_fill_color for p in panels]
    assert colours[:6] == [to_hex(WATER_COLOR)] * 6
    assert colours[6:] == ["#d9d9d9"] * 3
    assert _bokeh(grid) is not None
