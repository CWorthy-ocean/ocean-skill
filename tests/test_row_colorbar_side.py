"""A one-row map comparison puts its colour bars beside the panels, like a grid's row.

``field_row`` used to hang horizontal bars below the panels while a ``field_grid`` on
the same report page drew vertical ones to the right, which read as two different
figures and left white space under the single row. The default is now vertical
(``field_movie``'s frames follow it); ``colorbar_kwargs={"orientation": "horizontal"}``
still moves the bars below, and a very wide map still forces them there (the aspect
rule is unchanged). The interactive renderer's bars always sit at the right, which is
what is checked there.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

from ocean_skill.plot.registry import render
from ocean_skill.plot.spec import PlotSpec
from ocean_skill.plot.typography import colorbar_is_horizontal
from tests.test_colorbar_holoviews import _row_item
from tests.test_section_contours_static import _row_item as _section_row_item


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _bars(fig):
    return [ax._colorbar for ax in fig.axes if ax.get_label() == "<colorbar>"]


def _panels(fig):
    return [ax for ax in fig.axes if ax.get_label() != "<colorbar>"]


def _draw(family, item, **options):
    return render(PlotSpec(family=family, items=[item], options=options))


ROWS = [pytest.param("field_row", _row_item, id="field_row")]


@pytest.mark.parametrize(("family", "make_item"), ROWS)
def test_a_one_row_comparison_draws_vertical_bars_beside_its_panels(family, make_item):
    fig = _draw(family, make_item())
    bars = _bars(fig)
    assert len(bars) == 2  # the shared test/reference bar and the difference bar
    assert {bar.orientation for bar in bars} == {"vertical"}
    panel_bottom = min(ax.get_position().y0 for ax in _panels(fig))
    for bar in bars:
        box = bar.ax.get_position()
        assert box.height > box.width  # a bar, standing up
        assert box.y0 >= panel_bottom - 1e-6  # beside the maps, not under them


@pytest.mark.parametrize(("family", "make_item"), ROWS)
def test_a_one_row_comparison_can_still_ask_for_bars_below(family, make_item):
    fig = _draw(family, make_item(), colorbar_kwargs={"orientation": "horizontal"})
    bars = _bars(fig)
    assert len(bars) == 2
    assert {bar.orientation for bar in bars} == {"horizontal"}
    top_of_bars = max(bar.ax.get_position().y1 for bar in bars)
    bottom_of_panels = min(ax.get_position().y0 for ax in _panels(fig))
    assert top_of_bars <= bottom_of_panels + 1e-6  # below the maps
    for bar in bars:
        box = bar.ax.get_position()
        assert box.width > box.height


def test_a_vertical_row_is_shorter_than_a_horizontal_one():
    """The row height stops reserving room for bars below once they are beside."""
    vertical = _draw("field_row", _row_item())
    horizontal = _draw(
        "field_row", _row_item(), colorbar_kwargs={"orientation": "horizontal"}
    )
    assert vertical.get_size_inches()[1] < horizontal.get_size_inches()[1]


def test_flat_section_rows_keep_their_bars_below():
    """A section row's panels are too flat for a bar down the side: still horizontal."""
    fig = _draw("section_row", _section_row_item())
    assert {bar.orientation for bar in _bars(fig)} == {"horizontal"}


def test_the_aspect_overrides_are_unchanged():
    assert colorbar_is_horizontal(4.0, default_horizontal=False) is True
    assert colorbar_is_horizontal(0.5, default_horizontal=True) is False
    assert colorbar_is_horizontal(1.5, default_horizontal=False) is False
    assert (
        colorbar_is_horizontal(1.5, default_horizontal=False, requested="horizontal")
        is True
    )


def test_the_interactive_row_has_its_bars_at_the_right():
    import holoviews as hv
    from bokeh.models import ColorBar, Plot

    obj = render(
        PlotSpec(family="field_row", items=[_row_item()], options={}),
        renderer="holoviews",
    )
    layout = hv.render(obj)  # the bokeh model tree, which is where a bar's side is set
    figures = layout.select({"type": Plot})
    assert figures
    placed = {"right": 0, "below": 0, "other": 0}
    for fig in figures:
        for region in ("right", "left", "above", "below", "center"):
            count = sum(isinstance(m, ColorBar) for m in getattr(fig, region))
            placed["right" if region == "right" else "other"] += count
    assert placed["right"] >= 2
    assert placed["other"] == 0
