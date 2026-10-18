"""Labels around a map that fit what is left of the panel, not the page.

Cartopy spaces its meridians for a page-wide map and runs the longitude labels of a
narrow panel together (``170°W160°W150°W``); a colorbar's tick labels and axis label are
as wide as their text, and in a tight gutter reach onto the next map. Both are measured
after the layout settles, when each panel's width is known.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
import xarray as xr

from ocean_skill.plot.registry import render
from ocean_skill.plot.spec import PlotSpec

NITRATE = "mole_concentration_of_nitrate_in_sea_water"


def _field(lon0: float, lon1: float, offset: float = 0.0) -> xr.DataArray:
    rng = np.random.default_rng(3)
    return xr.DataArray(
        offset + rng.normal(size=(30, 30)),
        dims=("lat", "lon"),
        coords={"lat": np.linspace(12, 66, 30), "lon": np.linspace(lon0, lon1, 30)},
        attrs={"units": "mmol m-3"},
    )


def _row(lon0: float, lon1: float) -> PlotSpec:
    test, ref = _field(lon0, lon1, 1.0), _field(lon0, lon1)
    item = {
        "aligned": {"test": test, "reference": ref, "difference": test - ref},
        "units": "mmol m-3",
        "standard_name": NITRATE,
        "metrics": {"bias": 0.1, "rmse": 0.5, "corr": 0.9},
        "labels": ("model", "obs"),
        "row_label": "nitrate",
    }
    return PlotSpec(family="field_row", items=[item], options={})


def _gridliners(fig):
    return [
        artist
        for ax in fig.axes
        for artist in ax.artists
        if hasattr(artist, "xlocator")
    ]


def _bottom_labels(gl):
    return [t for t in gl.bottom_label_artists if t.get_visible() and t.get_text()]


def _crowded(fig, gl) -> bool:
    renderer = fig.canvas.get_renderer()
    boxes = sorted(
        (t.get_window_extent(renderer) for t in _bottom_labels(gl)), key=lambda b: b.x0
    )
    return any(a.x1 >= b.x0 for a, b in pairwise(boxes))


def test_a_narrow_panel_gets_fewer_meridians_so_its_labels_clear_each_other():
    fig = render(_row(-170.0, -110.0))
    fig.canvas.draw()
    labelled = [gl for gl in _gridliners(fig) if _bottom_labels(gl)]
    assert labelled
    for gl in labelled:
        texts = [t.get_text() for t in _bottom_labels(gl)]
        assert 2 <= len(texts) <= 4, texts
        assert not _crowded(fig, gl)
    # the thinned step is a round one, the same on every panel of the row
    def step(gl):
        values = [float(t.get_text().rstrip("°EW")) for t in _bottom_labels(gl)]
        return round(abs(float(np.diff(values)[0])))

    steps = {step(gl) for gl in labelled}
    assert len(steps) == 1
    assert steps <= {20, 30}


@pytest.mark.parametrize("width, thinned", [(12.0, False), (2.0, True)])
def test_only_a_panel_too_narrow_for_its_labels_is_re_spaced(width, thinned):
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    from cartopy.mpl.ticker import LongitudeLocator

    from ocean_skill.plot.matplotlib_renderer import _thin_longitude_gridlines

    fig = plt.figure(figsize=(width, 3))
    ax = fig.add_subplot(projection=ccrs.PlateCarree())
    ax.set_extent([-130, -110, 25, 27.5])
    gl = ax.gridlines(draw_labels=True)
    gl.xlabel_style = {"size": 8}
    fig.draw_without_rendering()
    assert _thin_longitude_gridlines(fig, fig.canvas.get_renderer()) is thinned
    # an untouched axis keeps cartopy's own locator
    assert isinstance(gl.xlocator, LongitudeLocator) is (not thinned)


def test_the_longitude_step_is_the_smallest_round_one_that_fits():
    from ocean_skill.plot.matplotlib_renderer import _longitude_step

    assert _longitude_step(60.0, 4) == 15.0
    assert _longitude_step(61.5, 4) == 20.0
    assert _longitude_step(360.0, 3) == 150.0
    assert _longitude_step(4.0, 4) == 1.0
    assert _longitude_step(2.0, 4) == 0.5


def _bar_beside(label: str):
    """Two panels with a colorbar for the left one jammed against the right one."""
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable

    from ocean_skill.plot.matplotlib_renderer import _draw_colorbar, _style_defaults
    from ocean_skill.plot.typography import type_scale

    fig, (left, right) = plt.subplots(1, 2, figsize=(5, 3))
    fig.set_layout_engine("none")
    left.set_position([0.05, 0.15, 0.30, 0.7])
    right.set_position([0.52, 0.15, 0.30, 0.7])
    defaults = _style_defaults(type_scale((5, 3), ncols=2), horizontal_colorbar=False)
    im = ScalarMappable(cmap="viridis")
    im.set_clim(32.5, 34.5)
    cbar = _draw_colorbar(fig, im, left, label, None, defaults["colorbar_kwargs"])
    cbar.ax.set_position([0.39, 0.15, 0.04, 0.7])
    return fig, left, right, cbar


def _label_edge(fig, cbar) -> float:
    fig.draw_without_rendering()
    renderer = fig.canvas.get_renderer()
    texts = [cbar.ax.yaxis.label, *cbar.ax.get_yticklabels()]
    return max(t.get_window_extent(renderer).x1 for t in texts if t.get_text())


def test_a_colorbars_label_is_pulled_back_off_the_next_panel():
    from ocean_skill.plot.matplotlib_renderer import _clear_colorbar_labels

    fig, _left, right, cbar = _bar_beside("[PSU]")
    gutter = right.get_window_extent(fig.canvas.get_renderer()).x0
    # moved right until its label overlaps the next map by a few pixels, as a tight
    # gutter leaves it
    fig.draw_without_rendering()
    shift = (gutter - _label_edge(fig, cbar) + 6) / fig.bbox.width
    pos = cbar.ax.get_position()
    cbar.ax.set_position([pos.x0 + shift, pos.y0, pos.width, pos.height])
    assert _label_edge(fig, cbar) > gutter  # the overlap this guards against

    assert _clear_colorbar_labels(fig, fig.canvas.get_renderer())
    assert _label_edge(fig, cbar) <= gutter


def test_a_colorbar_that_clears_its_neighbours_is_not_touched():
    from ocean_skill.plot.matplotlib_renderer import _clear_colorbar_labels

    fig, *_, cbar = _bar_beside("[PSU]")
    cbar.ax.set_position([0.37, 0.15, 0.03, 0.7])
    before = cbar.ax.get_position().bounds
    assert not _clear_colorbar_labels(fig, fig.canvas.get_renderer())
    assert cbar.ax.get_position().bounds == before


def test_a_live_layout_engine_is_left_to_itself():
    from ocean_skill.plot.matplotlib_renderer import _clear_colorbar_labels

    fig, *_ = _bar_beside("[PSU]")
    fig.set_layout_engine("constrained")
    assert not _clear_colorbar_labels(fig, fig.canvas.get_renderer())


def _bar_with_ticks(length: float, values):
    """Build a horizontal colorbar ``length`` wide, ticked at ``values``."""
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable

    from ocean_skill.plot.matplotlib_renderer import _draw_colorbar, _style_defaults
    from ocean_skill.plot.typography import type_scale

    fig, ax = plt.subplots(figsize=(5, 3))
    fig.set_layout_engine("none")
    ax.set_position([0.1, 0.4, 0.8, 0.5])
    defaults = _style_defaults(type_scale((5, 3), ncols=2), horizontal_colorbar=True)
    im = ScalarMappable(cmap="viridis")
    im.set_clim(min(values), max(values))
    cbar = _draw_colorbar(
        fig,
        im,
        ax,
        "label",
        {"orientation": "horizontal", "ticks": list(values)},
        defaults["colorbar_kwargs"],
    )
    cbar.ax.set_position([0.1, 0.2, length, 0.05])
    fig.draw_without_rendering()
    return fig, cbar


def _shown_ticks(cbar):
    return [t.get_text() for t in cbar.ax.get_xticklabels() if t.get_text()]


def test_a_colorbar_whose_tick_labels_collide_drops_every_second_tick():
    from ocean_skill.plot.matplotlib_renderer import _thin_colorbar_ticks

    values = [-15, -10, -5, 0, 5, 10, 15]
    fig, cbar = _bar_with_ticks(0.2, values)
    assert len(_shown_ticks(cbar)) == 7
    assert _thin_colorbar_ticks(fig, fig.canvas.get_renderer())
    fig.draw_without_rendering()
    # the survivors are round values, and zero (the difference bar's anchor) is one
    assert _shown_ticks(cbar) == ["\N{MINUS SIGN}10", "0", "10"]


def test_a_colorbar_whose_tick_labels_have_room_is_left_alone():
    from ocean_skill.plot.matplotlib_renderer import _thin_colorbar_ticks

    fig, cbar = _bar_with_ticks(0.8, [-15, -10, -5, 0, 5, 10, 15])
    assert not _thin_colorbar_ticks(fig, fig.canvas.get_renderer())
    assert len(_shown_ticks(cbar)) == 7


@pytest.mark.parametrize("dpi", [100, 300])
def test_a_nudged_bar_stays_in_the_figure_and_moves_only_a_little(dpi):
    from ocean_skill.plot.matplotlib_renderer import _clear_colorbar_labels

    fig, _left, right, cbar = _bar_beside("[PSU]")
    fig.set_dpi(dpi)
    gutter = right.get_window_extent(fig.canvas.get_renderer()).x0
    fig.draw_without_rendering()
    shift = (gutter - _label_edge(fig, cbar) + 6) / fig.bbox.width
    pos = cbar.ax.get_position()
    cbar.ax.set_position([pos.x0 + shift, pos.y0, pos.width, pos.height])
    before = cbar.ax.get_position().bounds

    assert _clear_colorbar_labels(fig, fig.canvas.get_renderer())
    after = cbar.ax.get_position().bounds
    assert 0 <= after[0] and after[0] + after[2] <= 1
    # a few pixels of overlap is a few percent of the figure at any resolution
    assert abs(after[0] - before[0]) < 0.05
    assert after[2] <= before[2] and after[2] >= 0.5 * before[2]
