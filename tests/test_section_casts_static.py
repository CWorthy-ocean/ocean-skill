"""Static section figures built from casts: dashed cast lines, names, and the seafloor.

A comparison section cut along CTD casts
(``select={"transect": {"from": "reference"}}``) has one along-path column per cast.
The static renderer can then say where the data came from -- a dashed line down each
cast to its deepest value, the cast's name along the top of every panel -- and shade
the seafloor under the section, deepening the y axis to hold it. The geometry (where
each mark sits, how deep the axis reaches) is decided once in
:mod:`ocean_skill.plot.section`; these tests check that ``section_row`` and
``section_row_grid`` draw it on all three panels of a row, leave a section without
these inputs exactly as it was, and refuse what makes no sense. Everything runs on
small synthetic trios.
"""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from matplotlib.collections import LineCollection, PolyCollection, QuadMesh
from matplotlib.contour import ContourSet

from ocean_skill.align import ALONG_DIM
from ocean_skill.plot import matplotlib_renderer as mr
from ocean_skill.plot.registry import render
from ocean_skill.plot.section import (
    CAST_COLOR,
    CAST_WIDTH,
    SEAFLOOR_COLOR,
    WATER_COLOR,
    depth_limit,
    fill_between_casts,
    prepare_section_row,
)
from ocean_skill.plot.spec import PlotSpec

#: Where each cast sits along the path (km), and how deep its reference has values (m).
ALONG_KM = np.array([0.0, 20.0, 45.0, 80.0, 100.0, 130.0])
CAST_BOTTOMS = np.array([300.0, 450.0, 600.0, 200.0, 500.0, 350.0])
LABELS = ["A1", "A2", "A3", "A4", "A5", "A6"]
LON0, LON1 = -95.0, -93.0

#: The grey levels of the two style defaults, as matplotlib reads their string forms.
CAST_GREY = float(CAST_COLOR)
SEAFLOOR_GREY = float(SEAFLOOR_COLOR)

#: ``z`` runs 0 to -600 m in 50 m steps (negative-down, as a comparison lane has it).
Z = -np.arange(0.0, 650.0, 50.0)


def _lon_of(km):
    return LON0 + (LON1 - LON0) * np.asarray(km) / ALONG_KM[-1]


def _lane(values: np.ndarray) -> xr.DataArray:
    da = xr.DataArray(
        values,
        dims=("z", ALONG_DIM),
        coords={
            "z": Z,
            ALONG_DIM: ALONG_KM,
            "lon": (ALONG_DIM, _lon_of(ALONG_KM)),
            "lat": (ALONG_DIM, np.linspace(24.0, 26.0, ALONG_KM.size)),
        },
    )
    da[ALONG_DIM].attrs["units"] = "km"
    return da


def _aligned(tilt: float = 0.0) -> dict[str, xr.DataArray]:
    """Build a trio: the model everywhere, the reference only to each cast's reach."""
    depth = -Z[:, None] * np.ones((1, ALONG_KM.size))
    test = 5.0 + 0.01 * depth + tilt
    reference = np.where(depth <= CAST_BOTTOMS[None, :], test - 0.3, np.nan)
    ref = _lane(reference)
    tst = _lane(test)
    return {"test": tst, "reference": ref, "difference": tst - ref}


def _seafloor(n: int = 40, *, deepest: float = 520.0) -> xr.DataArray:
    """Build a bottom-depth line from the first cast to the last, finer than casts."""
    km = np.linspace(0.0, ALONG_KM[-1], n)
    depth = 150.0 + (deepest - 150.0) * np.sin(np.pi * km / ALONG_KM[-1]) ** 0.5
    return xr.DataArray(
        depth,
        dims=(ALONG_DIM,),
        coords={
            ALONG_DIM: km,
            "path_lon": (ALONG_DIM, _lon_of(km)),
            "path_lat": (ALONG_DIM, np.linspace(24.0, 26.0, n)),
        },
    )


def _item(**extra) -> dict:
    return {
        "aligned": _aligned(),
        "units": "degC",
        "standard_name": None,
        "depth": None,
        "time": None,
        "metrics": {"bias": 0.125, "rmse": 0.5, "corr": 0.98},
        "labels": ("model", "ctd"),
        **extra,
    }


def _cast_item(**extra) -> dict:
    return _item(cast_labels=list(LABELS), seafloor=_seafloor(), **extra)


def _panels(fig) -> list:
    return [ax for ax in fig.axes if ax.get_label() != "<colorbar>"]


def _cast_lines(ax) -> list[LineCollection]:
    return [
        c
        for c in ax.collections
        if isinstance(c, LineCollection)
        and np.allclose(c.get_edgecolor()[0][:3], CAST_GREY)
    ]


def _fills(ax) -> list[PolyCollection]:
    return [c for c in ax.collections if isinstance(c, PolyCollection)]


def _top_axes(ax) -> list:
    return [c for c in ax.child_axes if hasattr(c, "_functions")]


def _top_labels(ax) -> tuple[list[float], list[str]]:
    """Return the ``(positions, names)`` of a panel's secondary top axis."""
    (top,) = _top_axes(ax)
    ax.figure.canvas.draw()
    texts = [t.get_text() for t in top.xaxis.get_ticklabels()]
    return list(top.get_xticks()), texts


def _row(**options):
    return mr.section_row(_aligned(), **options)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# -- casts --------------------------------------------------------------------------


def test_each_cast_is_a_dashed_line_to_its_deepest_reference_value():
    fig = _row(cast_labels=LABELS, section_x="distance")
    for ax in _panels(fig):
        (casts,) = _cast_lines(ax)
        segments = casts.get_segments()
        assert len(segments) == len(LABELS)
        xs = [seg[0, 0] for seg in segments]
        np.testing.assert_allclose(xs, ALONG_KM)
        for seg in segments:  # from the surface down
            assert seg[0, 1] == 0.0
        np.testing.assert_allclose([seg[1, 1] for seg in segments], CAST_BOTTOMS)
        assert casts.get_linestyle()[0][1] is not None  # dashed, not solid
        assert casts.get_zorder() > 1  # over the data fill


def test_cast_names_ride_a_secondary_top_axis_on_all_three_panels():
    fig = _row(cast_labels=LABELS, section_x="distance")
    panels = _panels(fig)
    assert len(panels) == 3
    for ax in panels:
        positions, names = _top_labels(ax)
        np.testing.assert_allclose(positions, ALONG_KM)
        assert names == LABELS


def test_without_casts_there_are_no_lines_names_or_extra_axes():
    fig = _row(section_x="distance")
    for ax in _panels(fig):
        assert not _cast_lines(ax)
        assert not _top_axes(ax)


def test_cast_x_follows_the_sections_x_axis():
    fig = _row(cast_labels=LABELS, section_x="lon")
    for ax in _panels(fig):
        positions, names = _top_labels(ax)
        np.testing.assert_allclose(positions, _lon_of(ALONG_KM))
        (casts,) = _cast_lines(ax)
        np.testing.assert_allclose(
            [seg[0, 0] for seg in casts.get_segments()], _lon_of(ALONG_KM)
        )
        assert names == LABELS


def test_a_cast_with_no_finite_value_is_named_but_draws_no_line():
    aligned = _aligned()
    aligned["reference"][{ALONG_DIM: 2}] = np.nan
    fig = mr.section_row(aligned, cast_labels=LABELS, section_x="distance")
    ax = _panels(fig)[1]
    (casts,) = _cast_lines(ax)
    assert len(casts.get_segments()) == len(LABELS) - 1
    assert _top_labels(ax)[1] == LABELS


def test_a_label_count_that_does_not_match_the_casts_raises_before_any_figure():
    plt.close("all")
    with pytest.raises(ValueError, match="one column per cast"):
        _row(cast_labels=LABELS[:-1])
    assert plt.get_fignums() == []


def test_cast_kwargs_restyle_the_lines_and_the_names():
    fig = _row(
        cast_labels=LABELS,
        section_x="distance",
        cast_kwargs={"colors": "tab:red", "linestyles": ":", "labels": False},
    )
    for ax in _panels(fig):
        assert not _top_axes(ax)
        (lines,) = [c for c in ax.collections if isinstance(c, LineCollection)]
        assert not np.allclose(lines.get_edgecolor()[0][:3], CAST_GREY)
    fig = _row(cast_labels=LABELS, cast_kwargs={"rotation": 45})
    (top,) = _top_axes(_panels(fig)[0])
    assert all(t.get_rotation() == 45 for t in top.xaxis.get_ticklabels())


def test_the_cast_names_leave_room_above_the_panels():
    # the names are extra height over the panel titles: a figure with them is taller
    plain = _row()
    named = _row(cast_labels=LABELS)
    assert named.get_size_inches()[1] > plain.get_size_inches()[1]


def test_the_titles_sit_above_the_cast_names():
    fig = _row(cast_labels=LABELS, section_x="distance")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in _panels(fig):
        (top,) = _top_axes(ax)
        names_top = max(
            t.get_window_extent(renderer).y1
            for t in top.xaxis.get_ticklabels()
            if t.get_text()
        )
        assert ax.title.get_window_extent(renderer).y0 >= names_top - 1


# -- the seafloor -------------------------------------------------------------------


def test_the_seafloor_is_shaded_under_the_data_and_outlined_over_it():
    fig = _row(seafloor=_seafloor(), section_x="distance")
    for ax in _panels(fig):
        (fill,) = [c for c in _fills(ax) if c.get_zorder() < 1]
        np.testing.assert_allclose(fill.get_facecolor()[0][:3], SEAFLOOR_GREY)
        (outline,) = [ln for ln in ax.lines if ln.get_zorder() > 1]
        assert outline.get_color() == "k"
        xy = outline.get_xydata()
        np.testing.assert_allclose(xy[:, 0], _seafloor()[ALONG_DIM])
        np.testing.assert_allclose(xy[:, 1], _seafloor().values)
        # the data fill is a QuadMesh at the default zorder, between the two
        assert fill.get_zorder() < 1 < outline.get_zorder()


def test_the_y_axis_reaches_the_depth_limit_with_zero_on_top():
    seafloor = _seafloor()
    fig = _row(seafloor=seafloor, section_x="distance")
    values, _ = prepare_section_row(_aligned(), "distance")
    expected = depth_limit(
        [values[lane] for lane in ("test", "reference", "difference")],
        np.asarray(seafloor.values),
    )
    for ax in _panels(fig):
        bottom, top = ax.get_ylim()
        assert (bottom, top) == (expected, 0.0)
        assert bottom > top  # inverted -- once
        assert ax.yaxis_inverted()


def test_a_seafloor_deeper_than_every_value_sets_the_limit():
    seafloor = _seafloor(deepest=900.0)
    fig = _row(seafloor=seafloor, section_x="distance")
    for ax in _panels(fig):
        assert ax.get_ylim() == (pytest.approx(float(seafloor.max())), 0.0)
        assert ax.get_ylim()[0] > 800.0


def test_observations_below_the_seafloor_stay_inside_the_axes():
    # the cast reaches 600 m, the seafloor only 520: the axis still holds the cast
    fig = _row(
        seafloor=_seafloor(deepest=520.0), cast_labels=LABELS, section_x="distance"
    )
    for ax in _panels(fig):
        assert ax.get_ylim()[0] >= 600.0


def test_the_x_limits_stay_on_the_data_when_the_seafloor_runs_past_it():
    wide = _seafloor()
    wide = wide.assign_coords({ALONG_DIM: wide[ALONG_DIM] * 1.5 - 15.0})
    fig = _row(seafloor=wide, section_x="distance")
    plain = _row(section_x="distance")
    for ax, ref in zip(_panels(fig), _panels(plain), strict=True):
        np.testing.assert_allclose(ax.get_xlim(), ref.get_xlim())
        assert ax.get_xlim()[0] > wide[ALONG_DIM].min()  # not widened to the seafloor


def test_the_seafloor_is_placed_by_longitude_on_a_longitude_axis():
    fig = _row(seafloor=_seafloor(), section_x="lon")
    for ax in _panels(fig):
        (outline,) = [ln for ln in ax.lines if ln.get_zorder() > 1]
        np.testing.assert_allclose(
            outline.get_xydata()[:, 0], _lon_of(_seafloor()[ALONG_DIM])
        )


def test_seafloor_kwargs_split_the_fill_from_the_outline():
    fig = _row(
        seafloor=_seafloor(),
        section_x="distance",
        seafloor_kwargs={
            "color": "tan",
            "alpha": 0.5,
            "edgecolor": "tab:red",
            "linewidth": 2.0,
        },
    )
    ax = _panels(fig)[0]
    (fill,) = [c for c in _fills(ax) if c.get_zorder() < 1]
    assert fill.get_alpha() == 0.5
    assert not np.allclose(fill.get_facecolor()[0][:3], SEAFLOOR_GREY)
    (outline,) = [ln for ln in ax.lines if ln.get_zorder() > 1]
    assert outline.get_color() == "tab:red"
    assert outline.get_linewidth() == 2.0


def test_edgecolor_none_draws_no_outline():
    fig = _row(
        seafloor=_seafloor(),
        section_x="distance",
        seafloor_kwargs={"edgecolor": "none"},
    )
    assert not _panels(fig)[0].lines


def test_a_seafloor_does_not_mark_casts_and_casts_do_not_move_the_axis():
    only_floor = _row(seafloor=_seafloor(), section_x="distance")
    assert not _cast_lines(_panels(only_floor)[0])
    only_casts = _row(cast_labels=LABELS, section_x="distance")
    ax = _panels(only_casts)[0]
    assert ax.get_ylim()[0] > ax.get_ylim()[1]  # inverted, as without casts
    assert not [c for c in _fills(ax) if c.get_zorder() < 1]


# -- unchanged without them ---------------------------------------------------------


def test_without_casts_or_seafloor_the_axis_is_inverted_exactly_once():
    fig = _row(section_x="distance")
    for ax in _panels(fig):
        assert ax.yaxis_inverted()
        bottom, top = ax.get_ylim()
        assert bottom > top
        # the mesh's own extent, edges half a cell past the shallowest and deepest rows
        assert top == pytest.approx(-25.0)
        assert bottom == pytest.approx(625.0)
        assert not [c for c in _fills(ax) if c.get_zorder() < 1]
        assert not ax.lines


def test_a_section_with_casts_and_seafloor_is_inverted_once_not_twice():
    fig = _row(seafloor=_seafloor(), cast_labels=LABELS, section_x="distance")
    for ax in _panels(fig):
        assert ax.yaxis_inverted()
        assert ax.get_ylim()[1] == 0.0


# -- refusals -----------------------------------------------------------------------


def test_cast_kwargs_without_casts_is_refused():
    with pytest.raises(ValueError, match="cast_kwargs"):
        _row(cast_kwargs={"colors": "r"})


def test_seafloor_kwargs_without_a_seafloor_is_refused():
    with pytest.raises(ValueError, match="seafloor_kwargs"):
        _row(seafloor_kwargs={"color": "r"})


def test_a_kwargs_that_is_not_a_dict_is_refused():
    with pytest.raises(TypeError, match="cast_kwargs"):
        _row(cast_labels=LABELS, cast_kwargs="red")


# -- through the renderer's dispatch and the stacked grid ---------------------------


def test_a_single_item_reaches_section_row_through_render():
    fig = render(
        PlotSpec(
            family="section_row",
            items=[_cast_item()],
            options={"section_x": "distance", "cast_kwargs": {"rotation": 30}},
        )
    )
    for ax in _panels(fig):
        assert _top_labels(ax)[1] == LABELS
        assert ax.get_ylim()[1] == 0.0
        assert [ln for ln in ax.lines if ln.get_zorder() > 1]


def test_a_stacked_grid_draws_each_rows_own_casts_and_seafloor():
    items = [
        _cast_item(row_label="casts"),
        _item(row_label="plain"),
        _item(
            row_label="named",
            cast_labels=[f"B{i}" for i in range(6)],
            seafloor=_seafloor(deepest=800.0),
        ),
    ]
    fig = render(
        PlotSpec(family="section_row", items=items, options={"section_x": "distance"})
    )
    panels = _panels(fig)
    assert len(panels) == 9
    first, plain, third = panels[0:3], panels[3:6], panels[6:9]
    for ax in first:
        assert _top_labels(ax)[1] == LABELS
        assert ax.get_ylim()[1] == 0.0
    for ax in plain:
        assert not _top_axes(ax) and not _cast_lines(ax) and not ax.lines
        assert ax.yaxis_inverted()
    for ax in third:
        assert _top_labels(ax)[1] == [f"B{i}" for i in range(6)]
        assert ax.get_ylim()[0] > 790.0
    # rows keep their own y axes: the plain row did not inherit the deep one
    assert plain[0].get_ylim()[0] < 800.0


def test_a_grid_label_mismatch_raises_before_any_figure():
    plt.close("all")
    items = [_cast_item(), _item(cast_labels=["only", "two"])]
    with pytest.raises(ValueError, match="one column per cast"):
        render(PlotSpec(family="section_row", items=items, options={}))
    assert plt.get_fignums() == []


def test_a_grid_refuses_cast_kwargs_when_no_row_has_casts():
    with pytest.raises(ValueError, match="cast_kwargs"):
        render(
            PlotSpec(
                family="section_row",
                items=[_item(), _item()],
                options={"cast_kwargs": {"colors": "r"}},
            )
        )


# -- filling between the casts ------------------------------------------------------

#: A deep cast between two shallow ones: the case the fill exists for.
DEEP_KM = np.array([0.0, 40.0, 80.0])
DEEP_BOTTOMS = np.array([100.0, 500.0, 100.0])
DEEP_Z = -np.arange(0.0, 550.0, 50.0)


def _deep_aligned() -> dict[str, xr.DataArray]:
    """Build a three-cast trio whose middle cast reaches 500 m, the outer two 100 m."""

    def lane(values):
        da = xr.DataArray(
            values,
            dims=("z", ALONG_DIM),
            coords={
                "z": DEEP_Z,
                ALONG_DIM: DEEP_KM,
                "lon": (ALONG_DIM, np.linspace(-95.0, -94.0, 3)),
                "lat": (ALONG_DIM, np.linspace(24.0, 25.0, 3)),
            },
        )
        da[ALONG_DIM].attrs["units"] = "km"
        return da

    depth = -DEEP_Z[:, None] * np.ones((1, DEEP_KM.size))
    test = 5.0 + 0.01 * depth
    reference = np.where(depth <= DEEP_BOTTOMS[None, :], test - 0.3, np.nan)
    tst, ref = lane(test), lane(reference)
    return {"test": tst, "reference": ref, "difference": tst - ref}


def _mesh(ax):
    (mesh,) = [c for c in ax.collections if isinstance(c, QuadMesh)]
    return mesh


def _mesh_grid(ax):
    """Return ``(x_centres, y_centres, values)`` of the one QuadMesh drawn on ``ax``."""
    mesh = _mesh(ax)
    corners = mesh.get_coordinates()
    x = 0.5 * (corners[0, :-1, 0] + corners[0, 1:, 0])
    y = 0.5 * (corners[:-1, 0, 1] + corners[1:, 0, 1])
    data = np.ma.filled(np.ma.masked_invalid(mesh.get_array()), np.nan)
    return x, y, data.reshape(y.size, x.size)


def _contour_sets(ax) -> list[ContourSet]:
    return [c for c in ax.collections if isinstance(c, ContourSet)]


def test_cast_fill_draws_many_more_columns_than_there_are_casts():
    for mark in ("pcolormesh", "contourf"):
        plain = mr.section_row(_aligned(), cast_labels=LABELS, mark=mark)
        filled = mr.section_row(
            _aligned(), cast_labels=LABELS, cast_fill=True, mark=mark
        )
        for ax_plain, ax_fill in zip(_panels(plain), _panels(filled), strict=True):
            if mark == "pcolormesh":
                assert _mesh_grid(ax_plain)[0].size == len(LABELS)
                assert _mesh_grid(ax_fill)[0].size > 10 * len(LABELS)
            else:
                (cs_plain,) = _contour_sets(ax_plain)
                (cs_fill,) = _contour_sets(ax_fill)
                xs = lambda cs: {  # noqa: E731
                    round(v, 9) for path in cs.get_paths() for v in path.vertices[:, 0]
                }
                assert len(xs(cs_fill)) > len(xs(cs_plain))


def test_cast_fill_off_or_absent_leaves_the_mesh_alone():
    absent = mr.section_row(_aligned(), cast_labels=LABELS)
    off = mr.section_row(_aligned(), cast_labels=LABELS, cast_fill=False)
    for ax_a, ax_o in zip(_panels(absent), _panels(off), strict=True):
        xa, _, va = _mesh_grid(ax_a)
        xo, _, vo = _mesh_grid(ax_o)
        assert xa.size == len(LABELS)
        np.testing.assert_allclose(xa, xo)
        np.testing.assert_allclose(va, vo)


def test_a_deep_cast_between_shallow_ones_is_coloured_halfway_to_its_neighbours():
    fig = mr.section_row(
        _deep_aligned(), cast_labels=["a", "b", "c"], cast_fill=True,
        section_x="distance",
    )  # fmt: skip
    ax = _panels(fig)[1]  # the reference lane: nothing below 100 m at the outer casts
    x, y, data = _mesh_grid(ax)
    deepest = int(np.argmin(np.abs(y - 500.0)))
    finite = np.isfinite(data[deepest])
    # the cast at 40 km, and the stretch either side out to the midpoints (20, 60 km)
    assert finite[np.argmin(np.abs(x - 40.0))]
    assert finite[np.argmin(np.abs(x - 25.0))] and finite[np.argmin(np.abs(x - 55.0))]
    assert finite.sum() > 8
    inside = (x > 20.0 + 0.5) & (x < 60.0 - 0.5)
    assert finite[inside].all()
    # ... and nothing beyond halfway, where no cast reaches that deep
    assert not finite[(x < 20.0 - 0.5) | (x > 60.0 + 0.5)].any()
    # the shallow outer casts are still coloured down to their own 100 m
    shallow = int(np.argmin(np.abs(y - 100.0)))
    assert np.isfinite(data[shallow, np.argmin(np.abs(x - 0.0))])
    # the plain mesh has nothing at 500 m but the cast's own column
    plain = mr.section_row(_deep_aligned(), cast_labels=["a", "b", "c"])
    _, yp, dp = _mesh_grid(_panels(plain)[1])
    assert np.isfinite(dp[int(np.argmin(np.abs(yp - 500.0)))]).sum() == 1


def test_cast_fill_stops_at_the_seafloor_but_keeps_the_cast_itself():
    # a sill at 20 km rising to 50 m, and a model floor of only 300 m under the 500 m
    # cast at 40 km: the fill must not paint the sill, nor carry the cast into rock
    km = np.array([0.0, 20.0, 40.0, 80.0])
    floor = xr.DataArray(
        [120.0, 50.0, 300.0, 120.0],
        dims=(ALONG_DIM,),
        coords={
            ALONG_DIM: km,
            "path_lon": (ALONG_DIM, np.interp(km, DEEP_KM, [-95.0, -94.5, -94.0])),
            "path_lat": (ALONG_DIM, np.interp(km, DEEP_KM, [24.0, 24.5, 25.0])),
        },
    )
    fig = mr.section_row(
        _deep_aligned(), seafloor=floor, cast_fill=True, section_x="distance"
    )
    x, y, data = _mesh_grid(_panels(fig)[1])
    under = np.interp(x, km, floor.values)
    own = np.isclose(x[None, :], DEEP_KM[:, None], atol=1e-3).any(axis=0)
    below = y[:, None] > under[None, :]
    assert below[:, ~own].any()
    assert not np.isfinite(data[below & ~own[None, :]]).any()
    # the deep cast's own column still reaches 500 m
    deepest = int(np.argmin(np.abs(y - 500.0)))
    assert np.isfinite(data[deepest, np.argmin(np.abs(x - 40.0))])


def test_contour_overlay_still_draws_on_the_same_mesh_as_the_fill():
    aligned = _aligned()
    fig = mr.section_row(
        aligned,
        contour={"test": aligned["test"], "reference": aligned["reference"]},
        contour_levels=[5.25, 5.75, 6.25],  # between the rows' values, off the nodes
        contour_kwargs={"labels": False},  # inline labels would add their own vertices
        cast_labels=LABELS,
        cast_fill=True,
        section_x="distance",
    )
    test_ax, ref_ax, diff_ax = _panels(fig)
    values, geometry = prepare_section_row(aligned, "distance")
    dense = fill_between_casts(values["test"], geometry)
    x = np.unique(np.asarray(dense[geometry.x_name])[0])
    assert x.size > 10 * len(LABELS)
    for ax in (test_ax, ref_ax):
        assert _mesh_grid(ax)[0].size == x.size  # the fill is on that same mesh
        (lines,) = _contour_sets(ax)
        segments = [seg for level in lines.allsegs for seg in level]
        assert segments
        line_x = np.unique(np.round(np.concatenate([s[:, 0] for s in segments]), 9))
        assert line_x.size > len(LABELS)  # not on the casts' own columns any more
        assert x.min() - 1e-6 <= line_x.min() and line_x.max() <= x.max() + 1e-6
        if ax is test_ax:  # the reference's lines also end on its masked cells' edges
            assert (np.abs(line_x[:, None] - x[None, :]).min(axis=1) < 1e-3).all()
    assert not _contour_sets(diff_ax)  # the difference panel never gets lines


def test_cast_marks_stay_at_the_original_cast_positions_with_cast_fill():
    fig = _row(cast_labels=LABELS, cast_fill=True, section_x="distance")
    for ax in _panels(fig):
        (casts,) = _cast_lines(ax)
        np.testing.assert_allclose([s[0, 0] for s in casts.get_segments()], ALONG_KM)
        np.testing.assert_allclose(
            [s[1, 1] for s in casts.get_segments()], CAST_BOTTOMS
        )
        positions, names = _top_labels(ax)
        np.testing.assert_allclose(positions, ALONG_KM)
        assert names == LABELS


def test_open_water_is_white_with_a_seafloor_and_grey_without():
    with_floor = _row(seafloor=_seafloor(), cast_fill=True)
    without = _row(cast_labels=LABELS, cast_fill=True)
    for ax in _panels(with_floor):
        assert mcolors.same_color(ax.get_facecolor(), WATER_COLOR)
    for ax in _panels(without):
        np.testing.assert_allclose(ax.get_facecolor()[:3], 0.85)


def test_cast_lines_default_to_the_thin_cast_width():
    fig = _row(cast_labels=LABELS)
    for ax in _panels(fig):
        (casts,) = _cast_lines(ax)
        assert casts.get_linewidth()[0] == CAST_WIDTH
    fig = _row(cast_labels=LABELS, cast_kwargs={"linewidths": 2.0})
    assert _cast_lines(_panels(fig)[0])[0].get_linewidth()[0] == 2.0


def test_cast_fill_works_with_the_seafloor_and_a_contourf_mark():
    fig = _row(
        cast_labels=LABELS, seafloor=_seafloor(), cast_fill=True, mark="contourf"
    )
    for ax in _panels(fig):
        assert ax.get_ylim()[1] == 0.0
        assert _contour_sets(ax)
        assert [ln for ln in ax.lines if ln.get_zorder() > 1]


def test_a_single_item_carries_cast_fill_through_render():
    fig = render(
        PlotSpec(
            family="section_row",
            items=[_cast_item(cast_fill=True)],
            options={"section_x": "distance"},
        )
    )
    for ax in _panels(fig):
        assert _mesh_grid(ax)[0].size > 10 * len(LABELS)
        assert _top_labels(ax)[1] == LABELS


def test_a_stacked_grid_fills_only_the_rows_that_ask_for_it():
    items = [
        _cast_item(row_label="filled", cast_fill=True),
        _cast_item(row_label="plain"),
    ]
    fig = render(
        PlotSpec(family="section_row", items=items, options={"section_x": "distance"})
    )
    panels = _panels(fig)
    assert len(panels) == 6
    for ax in panels[0:3]:
        assert _mesh_grid(ax)[0].size > 10 * len(LABELS)
    for ax in panels[3:6]:
        assert _mesh_grid(ax)[0].size == len(LABELS)
    for ax in panels:  # the casts sit on the original columns either way
        (casts,) = _cast_lines(ax)
        np.testing.assert_allclose([s[0, 0] for s in casts.get_segments()], ALONG_KM)
