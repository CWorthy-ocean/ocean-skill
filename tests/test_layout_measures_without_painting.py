"""Layout passes measure a figure without painting its meshes.

Counts the paints that reach the Agg renderer, so a ``fig.canvas.draw()`` creeping back
into a measuring pass shows up as a count that grows with the number of passes.
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
    """Count the quad meshes the Agg renderer actually paints (not measuring passes)."""
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


def test_a_map_grid_is_built_without_painting_and_saved_with_one_draw(mesh_paints):
    """Building paints no mesh; saving paints each once (a tight bbox only measures)."""
    fig = render(
        _grid(_diff_items(4), panels="difference", ncols=2), renderer="matplotlib"
    )
    n_meshes = sum(isinstance(c, QuadMesh) for ax in fig.axes for c in ax.collections)
    assert n_meshes == 5  # four panels and the shared colorbar
    assert mesh_paints["mesh"] == 0
    fig.savefig(io.BytesIO(), format="png", dpi=50, bbox_inches="tight")
    assert mesh_paints["mesh"] == n_meshes


def test_a_report_page_measures_its_figure_without_painting_it(mesh_paints, tmp_path):
    """The PNG is one real draw; the page's tight bbox adds none."""
    fig, ax = plt.subplots()
    ax.pcolormesh([[1.0, 2.0], [3.0, 4.0]])
    with PdfReport(tmp_path / "r.pdf", tmp_path / "figures") as report:
        report.emit(fig, "one")
    assert mesh_paints["mesh"] <= 2


def test_a_measuring_pass_hides_data_artists_and_restores_them():
    """Meshes (not lines) are hidden while measuring, and restored even on failure."""
    from ocean_skill.plot.matplotlib_renderer import _meshes_hidden

    fig, ax = plt.subplots()
    mesh = ax.pcolormesh([[1.0, 2.0], [3.0, 4.0]])
    (line,) = ax.plot([0, 1], [0, 1])
    with _meshes_hidden(fig):
        assert not mesh.get_visible() and line.get_visible()
    assert mesh.get_visible()
    with pytest.raises(RuntimeError), _meshes_hidden(fig):
        raise RuntimeError
    assert mesh.get_visible()
