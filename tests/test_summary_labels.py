"""How the summary diagrams identify their points.

Taylor and target show the *same* points, so a figure holding both must identify them
the same way — labelling one panel by legend and the other by annotation reads as two
unrelated plots. Both diagrams therefore support both modes, and :func:`paired` applies
one choice to both panels.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

from ocean_skill.plot.summary import paired, target, taylor


class _FakeComparison:
    """Minimal stand-in: the diagrams only call ``metrics()`` and read ``label``."""

    def __init__(self, label, corr, std_test, bias, crmsd, variable):
        self.label = label
        self._metrics = {
            "corr": corr,
            "std_test": std_test,
            "std_reference": 1.0,
            "bias": bias,
            "crmsd": crmsd,
            "variable": variable,
            "depth": 0,
        }

    def metrics(self):
        return self._metrics


@pytest.fixture
def comparisons():
    return [
        _FakeComparison("temp GOM", 0.95, 1.10, 0.30, 0.35, "sea_water_temperature"),
        _FakeComparison("salt GOM", 0.88, 0.82, -0.20, 0.50, "sea_water_salinity"),
        _FakeComparison("no3 GOM", 0.91, 0.95, -0.10, 0.40, "nitrate"),
    ]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def crossed_comparisons():
    """2 variables x 2 methods — one combination (alk/esper) deliberately missing.

    Exercises ``labels="grid"``: rows are ``variable`` levels, columns are ``method``
    levels, and the missing cell must draw blank rather than a glyph.
    """
    specs = [
        ("alk-depth", "alk", "depth_interp", 0.90, 1.00, 0.10, 0.35),
        ("alk-mld", "alk", "mld_interp", 0.80, 1.10, 0.20, 0.50),
        # alk/esper: no comparison — the grid's one deliberately blank cell
        ("dic-depth", "dic", "depth_interp", 0.95, 0.90, -0.10, 0.40),
        ("dic-mld", "dic", "mld_interp", 0.85, 1.00, 0.05, 0.30),
        ("dic-esper", "dic", "esper", 0.70, 1.20, 0.30, 0.70),
    ]
    comps = []
    for label, variable, method, corr, std_test, bias, crmsd in specs:
        c = _FakeComparison(label, corr, std_test, bias, crmsd, variable)
        c._metrics["method"] = method
        comps.append(c)
    return comps


def _annotated_texts(fig) -> list[str]:
    """Every annotation string in the figure, including Taylor's parasite axes.

    ``fig.findobj`` does *not* reach these. Taylor annotates the auxiliary polar axes
    from ``get_aux_axes``, which is a parasite of the host axes and appears neither in
    ``fig.axes`` nor anywhere in the figure's findobj tree — so the labels are drawn but
    invisible to the obvious search.
    """
    out = []
    for ax in fig.axes:
        for target_ax in (ax, *getattr(ax, "parasites", [])):
            out += [t.get_text() for t in target_ax.texts]
    return out


def _legend_texts(fig) -> set[str]:
    return {t.get_text() for lg in fig.legends for t in lg.get_texts()}


@pytest.mark.parametrize("diagram", [taylor, target])
def test_either_diagram_can_key_its_points_with_a_legend(diagram, comparisons):
    fig = diagram(comparisons, labels="legend")

    assert {"temp GOM", "salt GOM", "no3 GOM"} <= _legend_texts(fig)
    assert "reference" in _legend_texts(fig), "the star needs identifying too"
    assert not {"temp GOM", "salt GOM"} & set(_annotated_texts(fig))


@pytest.mark.parametrize("diagram", [taylor, target])
def test_either_diagram_can_annotate_its_markers(diagram, comparisons):
    fig = diagram(comparisons, labels="annotate")

    assert {"temp GOM", "salt GOM", "no3 GOM"} <= set(_annotated_texts(fig))
    assert not fig.legends


@pytest.mark.parametrize("diagram", [taylor, target])
def test_labels_none_gives_neither(diagram, comparisons):
    fig = diagram(comparisons, labels=None)

    assert not fig.legends
    assert not {"temp GOM", "salt GOM", "no3 GOM"} & set(_annotated_texts(fig))


def test_a_legend_needs_no_grouping_field(comparisons):
    """Legend handles used to exist only when ``color_by``/``marker_by`` was passed.

    Without this the common small-fan-out call produced a figure with no key at all.
    """
    fig = target(comparisons, labels="legend")
    assert {"temp GOM", "salt GOM"} <= _legend_texts(fig)


def test_paired_draws_one_shared_legend_not_two(comparisons):
    fig = paired(comparisons, labels="legend")

    assert len(fig.legends) == 1, "one key for both panels, not one per panel"
    assert {"temp GOM", "salt GOM", "no3 GOM"} <= _legend_texts(fig)
    # ...and neither panel falls back to annotating, which would double-label the points
    assert not {"temp GOM", "salt GOM"} & set(_annotated_texts(fig))


def test_paired_annotates_both_panels_or_neither(comparisons):
    """The whole point: the two panels must not disagree about how they label."""
    fig = paired(comparisons, labels="annotate")

    assert not fig.legends
    # Each label appears twice — once per panel — rather than once on the target only.
    texts = _annotated_texts(fig)
    for label in ("temp GOM", "salt GOM", "no3 GOM"):
        assert texts.count(label) == 2, f"{label!r} should be on both panels"


def test_an_unknown_label_mode_is_rejected(comparisons):
    with pytest.raises(ValueError, match="not one of"):
        taylor(comparisons, labels="below")


def test_grid_is_a_recognized_label_mode():
    from ocean_skill.plot.summary import LABEL_MODES

    assert "grid" in LABEL_MODES


# ---------------------------------------------------------------- labels="grid"


def _grid_cell_index(row, col, n_rows):
    """Handle index of cell ``(row, col)`` in a column-major grid of ``n_rows`` rows.

    Column ``col`` starts at ``(1 + col) * (n_rows + 1)`` (its header), so cell
    ``(row, col)`` sits one slot past that plus ``row``.
    """
    return (1 + col) * (n_rows + 1) + 1 + row


@pytest.mark.parametrize("diagram", [taylor, target])
def test_grid_renders_rows_by_colour_and_columns_by_marker(diagram, crossed_comparisons):
    fig = diagram(
        crossed_comparisons, color_by="variable", marker_by="method", labels="grid"
    )

    texts = _legend_texts(fig)
    assert {"alkalinity", "DIC"} <= texts, "row headers: variable levels"
    assert {"depth_interp", "mld_interp", "esper"} <= texts, "column headers: methods"
    assert "reference" in texts
    assert len(fig.legends) == 1
    point_labels = {c.label for c in crossed_comparisons}
    assert not point_labels & set(_annotated_texts(fig))


def test_grid_cell_glyphs_cross_colour_and_marker(crossed_comparisons):
    import matplotlib.colors as mcolors

    from ocean_skill.plot.style import COLOR_CYCLE

    fig = target(
        crossed_comparisons, color_by="variable", marker_by="method", labels="grid"
    )
    handles = fig.legends[0].legend_handles
    rows, cols = ["alk", "dic"], ["depth_interp", "mld_interp", "esper"]

    # Pins matplotlib's column-major fill: exactly (n_cols + 2) * (n_rows + 1) handles.
    assert len(handles) == (len(cols) + 2) * (len(rows) + 1)

    for j, _mlev in enumerate(cols):
        for i, _clev in enumerate(rows):
            if (rows[i], cols[j]) == ("alk", "esper"):
                continue  # the deliberately missing combination — see next test
            h = handles[_grid_cell_index(i, j, len(rows))]
            assert h.get_marker() == ("o", "^", "s")[j]
            assert mcolors.to_hex(h.get_mfc()) == mcolors.to_hex(COLOR_CYCLE[i])

    # Corner, row-label, and column-header handles carry no visible glyph — only the
    # cross-product cells do.
    n_rows = len(rows)
    assert handles[0].get_mfc() == "none", "corner"
    assert all(handles[1 + i].get_mfc() == "none" for i in range(n_rows)), "row labels"
    col0_header = (1 + 0) * (n_rows + 1)
    assert handles[col0_header].get_mfc() == "none", "column header"


def test_grid_leaves_missing_combinations_blank(crossed_comparisons):
    fig = target(
        crossed_comparisons, color_by="variable", marker_by="method", labels="grid"
    )
    handles = fig.legends[0].legend_handles
    rows, cols = ["alk", "dic"], ["depth_interp", "mld_interp", "esper"]
    missing = handles[_grid_cell_index(rows.index("alk"), cols.index("esper"), len(rows))]

    assert missing.get_mfc() == "none"
    assert missing.get_label() == ""


def test_grid_shows_the_reference_star_once(crossed_comparisons):
    fig = target(
        crossed_comparisons, color_by="variable", marker_by="method", labels="grid"
    )
    handles = fig.legends[0].legend_handles
    stars = [h for h in handles if h.get_marker() == "*"]

    assert len(stars) == 1
    assert stars[0].get_label() == "reference"


@pytest.mark.parametrize(
    "grouping", [{}, {"color_by": "variable"}, {"marker_by": "method"}]
)
def test_grid_without_both_channels_warns_and_falls_back(crossed_comparisons, grouping):
    with pytest.warns(UserWarning, match="grid"):
        fig = target(crossed_comparisons, labels="grid", **grouping)
    with_grid_texts = _legend_texts(fig)
    plt.close(fig)

    fig = target(crossed_comparisons, labels="legend", **grouping)
    assert with_grid_texts == _legend_texts(fig)


def test_paired_grid_draws_one_shared_grid_not_two(crossed_comparisons):
    fig = paired(
        crossed_comparisons, color_by="variable", marker_by="method", labels="grid"
    )

    assert len(fig.legends) == 1
    texts = _legend_texts(fig)
    assert {"alkalinity", "DIC", "depth_interp", "mld_interp", "esper"} <= texts
    point_labels = {c.label for c in crossed_comparisons}
    assert not point_labels & set(_annotated_texts(fig))


# ------------------------------------------------------- the interactive renderer

# `labels` has to reach the interactive target too, not just the static one: the
# holoviews `_target` took `**_`, so before this it accepted any label mode and
# silently dropped it — the "accepted and dropped" failure test_renderers.py guards.


@pytest.fixture
def items(comparisons):
    """Comparisons in the shape ``PlotSpec.items`` carries them."""
    return [
        {"label": c.label, "metrics": dict(c.metrics(), run="A" if i < 2 else "B")}
        for i, c in enumerate(comparisons)
    ]


def _interactive_target(items, **kwargs):
    from ocean_skill.plot.holoviews_renderer import _target

    return _target(items, **kwargs)


def _points(obj):
    import holoviews as hv

    return [e for e in obj if isinstance(e, hv.Points)]


def test_interactive_target_keys_its_points_with_a_legend(items):
    obj = _interactive_target(items, labels="legend")

    assert all(e.opts.get("plot").kwargs["show_legend"] for e in _points(obj))
    assert {e.label for e in _points(obj)} == {"temp GOM", "salt GOM", "no3 GOM"}


def test_interactive_target_annotates_its_markers(items):
    import holoviews as hv

    obj = _interactive_target(items, labels="annotate")

    text = [e for e in obj if isinstance(e, hv.Labels)]
    assert text, "annotate must add a Labels element"
    assert not any(e.opts.get("plot").kwargs["show_legend"] for e in _points(obj))


def test_interactive_target_labels_none_gives_neither(items):
    import holoviews as hv

    obj = _interactive_target(items, labels=None)

    assert not [e for e in obj if isinstance(e, hv.Labels)]
    assert not any(e.opts.get("plot").kwargs["show_legend"] for e in _points(obj))


def test_interactive_target_rejects_an_unknown_mode(items):
    with pytest.raises(ValueError, match="not one of"):
        _interactive_target(items, labels="below")


@pytest.fixture
def crossed_items(crossed_comparisons):
    """``crossed_comparisons`` in the shape ``PlotSpec.items`` carries them."""
    return [
        {"label": c.label, "metrics": c.metrics()} for c in crossed_comparisons
    ]


def test_interactive_target_grid_warns_and_falls_back_to_combined_legend(
    crossed_items,
):
    """Bokeh has no matrix legend — ``"grid"`` must fall back to ``"legend"``.

    Falling back loses only the tabular arrangement: the combined "colour · marker"
    entries ``"legend"`` already draws still spell out every present pair.
    """
    grouping = {"color_by": "variable", "marker_by": "method"}
    with pytest.warns(UserWarning, match="grid"):
        grid_obj = _interactive_target(crossed_items, labels="grid", **grouping)
    legend_obj = _interactive_target(crossed_items, labels="legend", **grouping)

    assert all(e.opts.get("plot").kwargs["show_legend"] for e in _points(grid_obj))
    assert {e.label for e in _points(grid_obj)} == {
        e.label for e in _points(legend_obj)
    }


@pytest.mark.parametrize(
    "grouping",
    [{}, {"color_by": "variable"}, {"marker_by": "run"}],
)
def test_both_renderers_agree_on_legend_entries(comparisons, items, grouping):
    """The same call must key the same groups whichever renderer draws it."""
    from ocean_skill.plot.summary import _group_styles

    recs = [
        dict(c.metrics(), label=c.label, run="A" if i < 2 else "B")
        for i, c in enumerate(comparisons)
    ]
    styles = _group_styles(recs, grouping.get("color_by"), grouping.get("marker_by"))
    static = [h.get_label() for h in styles.handles]

    obj = _interactive_target(items, labels="legend", **grouping)
    interactive = [e.label for e in _points(obj)]

    assert static == interactive


def test_both_renderers_use_the_same_colours(comparisons, items):
    """COLOR_CYCLE by level index in both, so a diagram keeps its colours across renderers."""
    import matplotlib.colors as mcolors

    from ocean_skill.plot.summary import _group_styles

    recs = [dict(c.metrics(), label=c.label) for c in comparisons]
    styles = _group_styles(recs, "variable", None)
    static = [mcolors.to_hex(c) for c in styles.colors]

    obj = _interactive_target(items, labels="legend", color_by="variable")
    interactive = [e.opts.get("style").kwargs["color"] for e in _points(obj)]

    assert static == interactive


def test_more_than_ten_comparisons_still_get_distinct_colours():
    """Regression: a tab10-only cycle silently repeats colour 1 on comparison 11.

    12 comparisons, one point each (no ``color_by``/``marker_by``), must produce 12
    distinct colours now that the cycle has 20 entries.
    """
    from ocean_skill.plot.summary import _group_styles

    recs = [{"label": f"c{i}"} for i in range(12)]
    styles = _group_styles(recs)

    assert len(set(styles.colors)) == 12
