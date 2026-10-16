"""Layout passes measure a figure without painting its meshes.

A map page's cost is dominated by rasterizing its ``pcolormesh`` panels (1858x962 cells
each). The layout code has to *draw* to learn where things end up -- label extents, the
settled constrained layout, tight bounding boxes -- but none of those passes needs a
pixel, so they run ``fig.draw_without_rendering()``: the same draw (layout engine and
text measurement included) with the renderer's painting switched off. These tests count
the paints that actually reach the Agg renderer, so a ``fig.canvas.draw()`` that creeps
back into a measuring pass shows up as a count that grows with the number of passes.
"""

from __future__ import annotations

import io
from collections import Counter

import matplotlib.pyplot as plt
import pytest
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.collections import QuadMesh

from ocean_skill.plot.registry import render
from ocean_skill.workflows.report import PdfReport
from tests.test_grid_panels import _diff_items, _grid


@pytest.fixture
def mesh_paints(monkeypatch):
    """Count the quad meshes the Agg renderer actually paints.

    The renderer's draw methods are instance attributes (set by ``_update_methods``,
    and swapped for no-ops while ``draw_without_rendering`` runs), so wrapping them
    there sees real paints only.
    """
    counts: Counter = Counter()
    update = RendererAgg._update_methods

    def counting(self):
        update(self)
        paint = self.draw_quad_mesh

        def draw_quad_mesh(*args, **kwargs):
            counts["mesh"] += 1
            return paint(*args, **kwargs)

        self.draw_quad_mesh = draw_quad_mesh

    monkeypatch.setattr(RendererAgg, "_update_methods", counting)
    return counts


def _meshes(fig) -> int:
    return sum(isinstance(c, QuadMesh) for ax in fig.axes for c in ax.collections)


def test_building_a_map_grid_paints_no_mesh(mesh_paints):
    fig = render(
        _grid(_diff_items(4), panels="difference", ncols=2), renderer="matplotlib"
    )
    assert _meshes(fig) == 5  # four panels and the shared colorbar
    assert mesh_paints["mesh"] == 0  # the layout passes only measured


def test_saving_a_map_grid_paints_each_mesh_once_per_real_draw(mesh_paints):
    fig = render(
        _grid(_diff_items(4), panels="difference", ncols=2), renderer="matplotlib"
    )
    fig.savefig(io.BytesIO(), format="png", dpi=50, bbox_inches="tight")
    # one real draw for the file; a tight bbox is measured, not painted
    assert mesh_paints["mesh"] == _meshes(fig)


def test_a_report_page_measures_its_figure_without_painting_it(mesh_paints, tmp_path):
    fig, ax = plt.subplots()
    ax.pcolormesh([[1.0, 2.0], [3.0, 4.0]])
    with PdfReport(tmp_path / "r.pdf", tmp_path / "figures") as report:
        report.emit(fig, "one")
    # the PNG is one real draw; the page's tight bbox adds none
    assert mesh_paints["mesh"] <= 2


def test_a_measuring_pass_hides_clipped_data_artists_and_restores_them():
    from ocean_skill.plot.matplotlib_renderer import _meshes_hidden

    fig, ax = plt.subplots()
    mesh = ax.pcolormesh([[1.0, 2.0], [3.0, 4.0]])
    (line,) = ax.plot([0, 1], [0, 1])
    with _meshes_hidden(fig):
        assert not mesh.get_visible()
        assert line.get_visible()  # only data collections and images are hidden
    assert mesh.get_visible()
    with pytest.raises(RuntimeError), _meshes_hidden(fig):
        raise RuntimeError
    assert mesh.get_visible()  # restored even when the measurement fails
