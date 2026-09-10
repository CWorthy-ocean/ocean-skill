"""Marker sizing, transparency, and explicit colours on the summary diagrams.

``marker_scale``/``alpha`` are style knobs on every point drawn — sample points, the
reference star, and legend swatches all move together, so a figure at
``marker_scale=2`` doesn't have a key sized for the default.

``colors``/``alpha``/``marker_scale`` all key on the *same* field colour already
groups by — ``color_by`` if given, else ``marker_by``, else each comparison's own
label — so they compose with grouping instead of being silently discarded by it. That
composition is what makes one-call layering possible: style particular groups
(fainter, smaller) without touching the rest — see
``test_paired_one_call_layers_less_noticeable_groups_under_more_noticeable_ones``.
"""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pytest

from ocean_skill.plot.summary import (
    _group_styles,
    _signed_medabs,
    _summary_point_specs,
    paired,
    target,
    taylor,
)


class _FakeComparison:
    """Minimal stand-in: the diagrams only call ``metrics()`` and read ``label``."""

    def __init__(self, label, corr, std_test, bias, crmsd, variable, **extra):
        self.label = label
        self._metrics = {
            "corr": corr,
            "std_test": std_test,
            "std_reference": 1.0,
            "bias": bias,
            "crmsd": crmsd,
            "variable": variable,
            "depth": 0,
            **extra,
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


def _taylor_lines(fig):
    """Every Line2D drawn on a Taylor diagram, including its parasite (aux) axes.

    Mirrors ``test_summary_labels._annotated_texts``: Taylor's sample points and
    reference star live on ``get_aux_axes``'s parasite, which ``fig.findobj`` and
    ``fig.axes`` alike do not reach.
    """
    out = []
    for ax in fig.axes:
        for target_ax in (ax, *getattr(ax, "parasites", [])):
            out += list(target_ax.lines)
    return out


# ------------------------------------------------------------------ marker_scale


def test_taylor_marker_scale_scales_points_and_star(comparisons):
    fig = taylor(comparisons, marker_scale=2.0, labels=None)

    lines = _taylor_lines(fig)
    samples = [ln for ln in lines if ln.get_marker() == "o"]
    stars = [ln for ln in lines if ln.get_marker() == "*"]

    assert samples, "expected sample points on the diagram"
    assert all(ln.get_markersize() == pytest.approx(18.0) for ln in samples)
    assert stars, "expected the reference star"
    assert stars[0].get_markersize() == pytest.approx(20.0)


def test_target_marker_scale_scales_points_star_and_swatches(comparisons):
    fig = target(comparisons, marker_scale=2.0, labels="legend")
    ax = fig.axes[0]

    assert ax.collections[0].get_sizes() == pytest.approx([70 * 2.0**2])
    star = next(ln for ln in ax.lines if ln.get_marker() == "*")
    assert star.get_markersize() == pytest.approx(22.0)

    swatches = {
        h.get_label(): h.get_markersize() for h in fig.legends[0].legend_handles
    }
    assert swatches["temp GOM"] == pytest.approx(14.0)
    assert swatches["reference"] == pytest.approx(18.0)


def test_marker_scale_dict_styles_points_by_level_star_stays_default(comparisons):
    """A dict has no single level for the reference, so it stays at its default."""
    fig = target(
        comparisons,
        color_by="variable",
        marker_scale={"sea_water_temperature": 2.0},
        labels=None,
    )
    ax = fig.axes[0]
    sizes = [c.get_sizes()[0] for c in ax.collections]

    assert sizes[0] == pytest.approx(70 * 2.0**2), "temp: named in the dict"
    assert sizes[1:] == [pytest.approx(70.0)] * 2, "salt/no3: not named, stay default"
    star = next(ln for ln in ax.lines if ln.get_marker() == "*")
    assert star.get_markersize() == pytest.approx(11.0)


def test_paired_forwards_marker_scale_and_alpha(comparisons):
    fig = paired(comparisons, marker_scale=2.0, alpha=0.5, labels=None)
    target_ax = fig.axes[1]

    collection = target_ax.collections[0]
    assert collection.get_sizes() == pytest.approx([70 * 2.0**2])
    assert collection.get_alpha() == pytest.approx(0.5)


def test_paired_legend_swatches_honor_explicit_colors(comparisons):
    """Regression: the shared legend used to ignore ``colors`` and key off the cycle."""
    explicit = ["#1b9e77", "#d95f02", "#7570b3"]
    fig = paired(comparisons, colors=explicit, labels="legend")

    swatches = {h.get_label(): h.get_mfc() for h in fig.legends[0].legend_handles}
    for label, color in zip(("temp GOM", "salt GOM", "no3 GOM"), explicit, strict=True):
        assert mcolors.to_hex(swatches[label]) == color


def test_paired_one_call_layers_less_noticeable_groups_under_more_noticeable_ones(
    comparisons,
):
    """The motivating case: fade/shrink one group, leave the rest default, one call."""
    fig = paired(
        comparisons,
        color_by="variable",
        alpha={"sea_water_salinity": 0.15},
        marker_scale={"sea_water_temperature": 1.6},
        labels=None,
    )
    target_ax = fig.axes[1]
    alphas = [c.get_alpha() for c in target_ax.collections]
    sizes = [c.get_sizes()[0] for c in target_ax.collections]

    assert alphas[0] is None and alphas[1] == pytest.approx(0.15) and alphas[2] is None
    assert sizes[0] == pytest.approx(70 * 1.6**2)
    assert sizes[1:] == [pytest.approx(70.0)] * 2


# ------------------------------------------------------------------------- alpha


@pytest.mark.parametrize("diagram", [taylor, target])
def test_alpha_reaches_the_points_but_not_the_star(diagram, comparisons):
    fig = diagram(comparisons, alpha=0.5, labels=None)

    if diagram is taylor:
        lines = _taylor_lines(fig)
        samples = [ln for ln in lines if ln.get_marker() == "o"]
        stars = [ln for ln in lines if ln.get_marker() == "*"]
        assert all(ln.get_alpha() == pytest.approx(0.5) for ln in samples)
        assert stars[0].get_alpha() is None
    else:
        ax = fig.axes[0]
        assert ax.collections[0].get_alpha() == pytest.approx(0.5)
        star = next(ln for ln in ax.lines if ln.get_marker() == "*")
        assert star.get_alpha() is None


def test_alpha_dict_styles_one_group_leaves_the_rest_opaque(comparisons):
    fig = target(
        comparisons, color_by="variable", alpha={"sea_water_salinity": 0.2}, labels=None
    )
    ax = fig.axes[0]
    alphas = [c.get_alpha() for c in ax.collections]

    assert alphas == [None, pytest.approx(0.2), None]


# --------------------------------------------------------------- colors + grouping
#
# These are regressions for the reported failure: `colors` had no effect at all when
# combined with `color_by` or `groups` (both set color_by internally, and the old
# `elif colors:` branch never ran once color_by was set) or when given as a bare
# string or a list shorter than the number of points (an opaque zip() crash).


@pytest.mark.parametrize("diagram", [taylor, target])
def test_colors_composes_with_color_by_instead_of_being_ignored(diagram, comparisons):
    fig = diagram(
        comparisons,
        colors=["y", "r", "g"],
        color_by="variable",
        labels=None,
    )
    if diagram is taylor:
        cols = [ln.get_mfc() for ln in _taylor_lines(fig) if ln.get_marker() == "o"]
    else:
        cols = [c.get_facecolor()[0] for c in fig.axes[0].collections]
    assert [mcolors.to_hex(c) for c in cols] == [
        mcolors.to_hex(c) for c in ("y", "r", "g")
    ]


def test_colors_composes_with_groups_instead_of_being_ignored(comparisons):
    # groups is keyed by each comparison's `reference`, falling back to its `label`
    # when the comparison carries no `reference` — these fakes don't, so key by label.
    groups = {"temp GOM": "g1", "salt GOM": "g1", "no3 GOM": "g2"}
    fig = target(
        comparisons, groups=groups, colors=["orange", "purple"], labels="legend"
    )
    swatches = {h.get_label(): h.get_mfc() for h in fig.legends[0].legend_handles}
    assert mcolors.to_hex(swatches["g1"]) == mcolors.to_hex("orange")
    assert mcolors.to_hex(swatches["g2"]) == mcolors.to_hex("purple")


def test_bare_string_colors_broadcasts_to_every_point(comparisons):
    fig = target(comparisons, colors="y", labels=None)
    ax = fig.axes[0]
    cols = [mcolors.to_hex(c.get_facecolor()[0]) for c in ax.collections]
    assert cols == [mcolors.to_hex("y")] * 3


def test_short_colors_list_raises_a_clear_error_instead_of_a_zip_crash(comparisons):
    with pytest.raises(ValueError, match="colors has 1 entries but there are 3"):
        target(comparisons, colors=["y"], labels=None)


def test_colors_dict_rejects_an_unknown_level_and_lists_the_real_ones(comparisons):
    with pytest.raises(ValueError, match="sea_water_chlorophyll"):
        target(
            comparisons,
            color_by="variable",
            colors={"sea_water_chlorophyll": "r"},
            labels=None,
        )


def test_colors_dict_partial_override_falls_back_to_the_cycle(comparisons):
    """Naming one level doesn't force the caller to name every level."""
    from ocean_skill.plot.style import COLOR_CYCLE

    fig = target(
        comparisons,
        color_by="variable",
        colors={"sea_water_temperature": "r"},
        labels=None,
    )
    ax = fig.axes[0]
    cols = [c.get_facecolor()[0] for c in ax.collections]

    assert mcolors.to_hex(cols[0]) == mcolors.to_hex("r")
    assert mcolors.to_hex(cols[1]) == mcolors.to_hex(COLOR_CYCLE[1])
    assert mcolors.to_hex(cols[2]) == mcolors.to_hex(COLOR_CYCLE[2])


def test_marker_by_only_swatches_match_the_points_they_key(comparisons):
    """Regression: marker-by swatches used the cycle while points took `colors`."""
    fig = target(
        comparisons, marker_by="variable", colors=["y", "r", "g"], labels="legend"
    )
    ax = fig.axes[0]
    point_colors = [mcolors.to_hex(c.get_facecolor()[0]) for c in ax.collections]
    swatch_colors = {
        h.get_label(): mcolors.to_hex(h.get_mfc())
        for h in fig.legends[0].legend_handles
        if h.get_label() != "reference"
    }
    for pt_color, level in zip(
        point_colors,
        ("sea_water_temperature", "sea_water_salinity", "nitrate"),
        strict=True,
    ):
        from ocean_skill.plot.summary import pretty_level

        assert swatch_colors[pretty_level("variable", level)] == pt_color


def test_grid_swatches_honor_explicit_colors_and_marker_scale(comparisons):
    """``labels="grid"``'s cells key on the same ``colors``/``marker_scale`` dicts."""
    for c, method in zip(comparisons, ("depth_interp", "mld_interp", "esper"), strict=True):
        c._metrics["method"] = method

    fig = target(
        comparisons,
        color_by="variable",
        marker_by="method",
        colors={"sea_water_temperature": "r"},
        marker_scale={"sea_water_temperature": 2.0},
        labels="grid",
    )
    handles = fig.legends[0].legend_handles
    rows = ["sea_water_temperature", "sea_water_salinity", "nitrate"]
    n_rows = len(rows)
    # column 0 (depth_interp) starts at (1+0)*(n_rows+1); the temp cell sits one past
    # its header — see test_summary_labels._grid_cell_index for the general formula.
    temp_cell = (n_rows + 1) + 1

    assert mcolors.to_hex(handles[temp_cell].get_mfc()) == mcolors.to_hex("r")
    assert handles[temp_cell].get_markersize() == pytest.approx(14.0)


# --------------------------------------------------------- the interactive renderer


@pytest.fixture
def items(comparisons):
    """Comparisons in the shape ``PlotSpec.items`` carries them."""
    return [{"label": c.label, "metrics": c.metrics()} for c in comparisons]


def _interactive_target(items, **kwargs):
    from ocean_skill.plot.holoviews_renderer import _target

    return _target(items, **kwargs)


def _points(obj):
    import holoviews as hv

    return [e for e in obj if isinstance(e, hv.Points)]


def test_interactive_target_honors_explicit_colors(items):
    explicit = ["#1b9e77", "#d95f02", "#7570b3"]
    obj = _interactive_target(items, colors=explicit, labels="legend")

    colors = {e.label: e.opts.get("style").kwargs["color"] for e in _points(obj)}
    for label, color in zip(("temp GOM", "salt GOM", "no3 GOM"), explicit, strict=True):
        assert colors[label] == color


def test_interactive_colors_composes_with_color_by(comparisons, items):
    """Regression, interactive side: color_by used to discard `colors` entirely."""
    obj = _interactive_target(
        items, colors=["y", "r", "g"], color_by="variable", labels="legend"
    )
    colors = [e.opts.get("style").kwargs["color"] for e in _points(obj)]
    assert colors == ["y", "r", "g"]


def test_both_renderers_use_the_same_explicit_colors(comparisons, items):
    explicit = ["#1b9e77", "#d95f02", "#7570b3"]
    styles = _group_styles(
        [dict(c.metrics(), label=c.label) for c in comparisons], colors=explicit
    )
    static = [mcolors.to_hex(c) for c in styles.colors]

    obj = _interactive_target(items, colors=explicit, labels="legend")
    interactive = [e.opts.get("style").kwargs["color"] for e in _points(obj)]

    assert static == interactive


def test_both_renderers_agree_on_dict_styling(comparisons, items):
    """The layering call: same dicts, same per-group result, in either renderer."""
    kwargs = {
        "color_by": "variable",
        "colors": {"sea_water_temperature": "r"},
        "alpha": {"sea_water_salinity": 0.2},
        "marker_scale": {"sea_water_temperature": 1.5},
    }
    fig = target(comparisons, labels=None, **kwargs)
    static_colors = [
        mcolors.to_hex(c.get_facecolor()[0]) for c in fig.axes[0].collections
    ]
    static_alphas = [c.get_alpha() for c in fig.axes[0].collections]

    obj = _interactive_target(items, **kwargs)
    pts = _points(obj)
    interactive_colors = [
        mcolors.to_hex(e.opts.get("style").kwargs["color"]) for e in pts
    ]
    interactive_alphas = [e.opts.get("style").kwargs.get("fill_alpha") for e in pts]

    assert static_colors == interactive_colors
    assert static_alphas == interactive_alphas
    # Sizes aren't comparable in absolute terms (mpl scatter `s` is area in points²,
    # bokeh `size` is a pixel diameter) — but both renderers must agree on *which*
    # group got enlarged: only the temp point, named in marker_scale, above default.
    assert (
        fig.axes[0].collections[0].get_sizes()[0]
        > fig.axes[0].collections[1].get_sizes()[0]
        == fig.axes[0].collections[2].get_sizes()[0]
    )
    interactive_sizes = [e.opts.get("style").kwargs["size"] for e in pts]
    assert interactive_sizes[0] > interactive_sizes[1] == interactive_sizes[2]


def test_interactive_marker_scale(items):
    obj = _interactive_target(items, marker_scale=2.0)

    assert all(e.opts.get("style").kwargs["size"] == 22.0 for e in _points(obj))


def test_interactive_marker_scale_dict_styles_one_group(comparisons, items):
    obj = _interactive_target(
        items, color_by="variable", marker_scale={"sea_water_temperature": 2.0}
    )
    sizes = {e.label: e.opts.get("style").kwargs["size"] for e in _points(obj)}

    assert sizes["temperature"] == pytest.approx(22.0)
    assert sizes["salinity"] == pytest.approx(11.0)
    assert sizes["nitrate"] == pytest.approx(11.0)


def test_interactive_alpha(items):
    default = _interactive_target(items)
    faded = _interactive_target(items, alpha=0.5)

    default_style = _points(default)[0].opts.get("style").kwargs
    faded_style = _points(faded)[0].opts.get("style").kwargs

    assert "fill_alpha" not in default_style and "line_alpha" not in default_style
    assert faded_style["fill_alpha"] == pytest.approx(0.5)
    assert faded_style["line_alpha"] == pytest.approx(0.5)


def test_interactive_alpha_dict_styles_one_group(comparisons, items):
    obj = _interactive_target(
        items, color_by="variable", alpha={"sea_water_salinity": 0.2}
    )
    styles = {e.label: e.opts.get("style").kwargs for e in _points(obj)}

    assert "fill_alpha" not in styles["temperature"]
    assert styles["salinity"]["fill_alpha"] == pytest.approx(0.2)
    assert "fill_alpha" not in styles["nitrate"]


def test_interactive_colors_dict_rejects_an_unknown_level(items):
    with pytest.raises(ValueError, match="sea_water_chlorophyll"):
        _interactive_target(
            items, color_by="variable", colors={"sea_water_chlorophyll": "r"}
        )


# ------------------------------------------------------------------ overlay / summary_points
#
# A second, emphasized layer on top of the base cloud -- either a highlighted subset
# (overlay=) or a per-group centroid (summary_points=), or both. The one property that
# matters most: an overlay point's colour must match its own group's colour in the base
# cloud, never an independently re-cycled one -- a lone highlighted or summarized point
# has no encounter order of its own to cycle by. A centroid also always draws with its
# own group's individual marker shape (never a forced shape of its own), distinguished
# from the cloud only by the overlay layer's usual size bump and black edge. With no
# color_by/marker_by there is no group to centre -- one-per-comparison would be
# degenerate (a single point's median is itself) -- so a single centroid is drawn
# across every comparison instead.


def test_taylor_summary_points_draws_one_centroid_per_group_matching_its_colour_and_shape(
    comparisons,
):
    fig = taylor(comparisons, color_by="variable", summary_points=True, labels=None)
    lines = _taylor_lines(fig)
    # base sample points draw at Line2D's default zorder (2); every overlay/centroid
    # is 10 -- see test_taylor_summary_split_markers_matches_colour_and_marker_shape
    samples = [ln for ln in lines if ln.get_zorder() == 2]
    sample_colors = {ln.get_markerfacecolor() for ln in samples}
    # centroids share the sample marker ("o": no marker_by here) but are drawn
    # separately, at the overlay zorder, black-edged and larger
    centroids = [ln for ln in lines if ln.get_zorder() == 10]
    assert len(centroids) == 3, "one centroid per group (3 variables)"
    for c in centroids:
        assert c.get_marker() == "o", "centroid keeps the individual marker, never a hexagon"
        assert c.get_markeredgecolor() == "k", "centroid stays black-edged, its own emphasis"
    assert {c.get_markerfacecolor() for c in centroids} <= sample_colors, (
        "every centroid's colour must be one already used by the base cloud"
    )


def test_taylor_summary_points_with_no_grouping_draws_a_single_centroid(comparisons):
    """No color_by/marker_by: one centroid across ALL comparisons, not one each."""
    fig = taylor(comparisons, summary_points=True, labels=None)
    lines = _taylor_lines(fig)
    centroids = [ln for ln in lines if ln.get_zorder() == 10]
    assert len(centroids) == 1, "a single aggregate centroid, not one per comparison"
    (centroid,) = centroids
    assert centroid.get_marker() == "o"
    # zorder==2 alone also catches the diagram's own reference star and dashed
    # reference arc (both drawn at the same default zorder, before any sample) --
    # marker=="o" narrows to the actual data points.
    samples = [ln for ln in lines if ln.get_zorder() == 2 and ln.get_marker() == "o"]
    first_color = samples[0].get_markerfacecolor()
    assert centroid.get_markerfacecolor() == first_color, (
        "the aggregate centroid takes the cloud's first cycle colour"
    )


def test_target_summary_points_with_no_grouping_draws_a_single_centroid(comparisons):
    fig = target(comparisons, summary_points=True, labels=None)
    ax = fig.axes[0]
    centroids = [c for c in ax.collections if c.get_zorder() == 10]
    assert len(centroids) == 1, "a single aggregate centroid, not one per comparison"


def test_target_summary_points_median_matches_a_hand_computed_centroid():
    """Two points in one group: with only two, the median *is* their mean."""
    # x = crmsd/std_reference * sign(std_test - std_reference); std_reference=1.0 always
    pair = [
        _FakeComparison("a", 0.9, 1.2, 0.10, 0.20, "sea_water_temperature"),  # x=+0.20
        _FakeComparison("b", 0.8, 0.8, -0.30, 0.40, "sea_water_temperature"),  # x=-0.40
    ]
    fig = target(pair, color_by="variable", summary_points=True, labels=None)
    ax = fig.axes[0]
    # base points at zorder=4; the centroid is drawn separately at zorder=10
    centroid = next(c for c in ax.collections if c.get_zorder() == 10)
    (cx, cy) = centroid.get_offsets()[0]
    assert cx == pytest.approx((0.20 + -0.40) / 2)
    assert cy == pytest.approx((0.10 + -0.30) / 2)


def test_summary_points_rejects_an_unknown_reduce_value(comparisons):
    with pytest.raises(ValueError, match="summary_points="):
        taylor(comparisons, summary_points="bogus")


def test_overlay_highlights_a_point_without_recycling_its_colour(comparisons):
    """A one-point overlay must reuse its group's own colour, not the cycle's first.

    Each base point is its own ``ax.scatter`` call (one collection each, zorder=4);
    the overlay is a separate, later call (zorder=10) -- so the third base collection
    (comparisons[2], "no3 GOM") is the one the highlighted overlay must match.
    """
    fig = target(
        comparisons, color_by="variable", overlay=[comparisons[2]], labels=None
    )
    ax = fig.axes[0]
    base_collections = [c for c in ax.collections if c.get_zorder() == 4]
    nitrate_color = tuple(base_collections[2].get_facecolor()[0])
    overlay_collection = next(c for c in ax.collections if c.get_zorder() == 10)
    overlay_color = tuple(overlay_collection.get_facecolor()[0])
    assert overlay_color == nitrate_color, (
        "the highlighted nitrate point must match nitrate's own base colour"
    )


def test_overlay_and_summary_points_do_not_add_legend_entries(comparisons):
    """Neither highlighting a point nor a centroid introduces a new legend key."""
    fig_base = target(comparisons, color_by="variable", labels="legend")
    n_base = len(fig_base.legends[0].legend_handles)
    fig_more = target(
        comparisons,
        color_by="variable",
        overlay=[comparisons[0]],
        summary_points=True,
        labels="legend",
    )
    n_more = len(fig_more.legends[0].legend_handles)
    assert n_more == n_base


def test_overlay_marker_scale_and_alpha_default_to_emphasized(comparisons):
    fig = target(comparisons, color_by="variable", overlay=[comparisons[0]], labels=None)
    ax = fig.axes[0]
    overlay_collection = next(c for c in ax.collections if c.get_zorder() == 10)
    assert overlay_collection.get_sizes()[0] == pytest.approx(70 * 1.8**2)
    assert overlay_collection.get_alpha() == pytest.approx(1.0)


def test_interactive_target_summary_points_matches_static_colour(comparisons, items):
    """Both renderers must put the same centroid colour on the same group."""
    import holoviews as hv

    static_fig = target(comparisons, color_by="variable", summary_points=True)
    static_ax = static_fig.axes[0]
    static_colors = {
        tuple(c.get_facecolor()[0])
        for c in static_ax.collections
        if c.get_zorder() == 10
    }
    static_hex = {mcolors.to_hex(c) for c in static_colors}

    obj = _interactive_target(items, color_by="variable", summary_points=True)
    # the base cloud draws as hv.Points, the reference point and every centroid as
    # hv.Scatter -- isinstance + excluding the black reference dot isolates exactly
    # the centroids, regardless of which marker each one draws (it now matches its
    # own group's shape, same as the base cloud, rather than a forced "hex").
    centroids = [
        e
        for e in obj.traverse(lambda x: x)
        if isinstance(e, hv.Scatter)
        and e.opts.get(group="style").kwargs.get("color") != "black"
    ]
    interactive_colors = {e.opts.get(group="style").kwargs["color"] for e in centroids}
    assert interactive_colors == static_hex


# ------------------------------------------------ signed_medabs + summary_weights


def _star(fig):
    """The one non-reference star drawn on a target figure with one group."""
    ax = fig.axes[0]
    centroid = next(c for c in ax.collections if c.get_zorder() == 10)
    return tuple(centroid.get_offsets()[0])


def test_target_signed_medabs_keeps_magnitude_where_median_would_cancel():
    """Same pair as the median test, but signed_medabs must NOT land near zero."""
    pair = [
        _FakeComparison("a", 0.9, 1.2, 0.10, 0.20, "sea_water_temperature"),  # x=+0.20
        _FakeComparison("b", 0.8, 0.8, -0.30, 0.40, "sea_water_temperature"),  # x=-0.40
    ]
    fig = target(pair, color_by="variable", summary_points="signed_medabs", labels=None)
    cx, cy = _star(fig)
    # magnitude = median(|+0.20|, |-0.40|) = 0.30; sign = sign(median(+0.20, -0.40)) = -1
    assert cx == pytest.approx(-0.30)
    # magnitude = median(|+0.10|, |-0.30|) = 0.20; sign = sign(median(+0.10, -0.30)) = -1
    assert cy == pytest.approx(-0.20)


def test_signed_medabs_ties_fall_back_to_the_mean_then_positive():
    # median([-1, 1]) == 0 (falsy) and mean([-1, 1]) == 0 (falsy) too -> +1 fallback.
    assert _signed_medabs([-1, 1]) == pytest.approx(1.0)
    # median([-1, -0.5, 0.5, 100]) == 0 (falsy) but mean == 24.75 (sign +) breaks the tie.
    assert _signed_medabs([-1, -0.5, 0.5, 100]) == pytest.approx(0.75)


def test_target_summary_weights_puts_the_star_on_the_heavier_point():
    """Weight ratio 3:1 -> the weighted median lands exactly on the heavy point."""
    pair = [
        _FakeComparison(
            "a", 0.9, 1.2, 0.10, 0.20, "sea_water_temperature", n_eff=1
        ),  # x=+0.20, y=+0.10
        _FakeComparison(
            "b", 0.8, 0.8, -0.30, 0.40, "sea_water_temperature", n_eff=3
        ),  # x=-0.40, y=-0.30
    ]
    fig = target(
        pair,
        color_by="variable",
        summary_points="median",
        summary_weights="n_eff",
        labels=None,
    )
    cx, cy = _star(fig)
    assert cx == pytest.approx(-0.40)
    assert cy == pytest.approx(-0.30)


def test_target_summary_weights_missing_on_some_records_warns_and_defaults_to_one(
    comparisons,
):
    comparisons[0]._metrics["n_eff"] = 5.0  # the other two comparisons carry none
    with pytest.warns(UserWarning, match="n_eff"):
        target(comparisons, summary_points=True, summary_weights="n_eff", labels=None)


def test_target_summary_weights_missing_on_every_record_raises(comparisons):
    with pytest.raises(ValueError, match="no record carries"):
        target(comparisons, summary_points=True, summary_weights="bogus_field", labels=None)


def test_target_summary_weights_rejects_nonpositive_weights(comparisons):
    comparisons[0]._metrics["n_eff"] = -1.0
    comparisons[1]._metrics["n_eff"] = 1.0
    comparisons[2]._metrics["n_eff"] = 1.0
    with pytest.raises(ValueError, match="finite"):
        target(comparisons, summary_points=True, summary_weights="n_eff", labels=None)


def test_interactive_target_summary_weights_matches_static_star_position():
    """Both renderers must place the weighted centroid at the same coordinates."""
    import holoviews as hv

    from ocean_skill.plot.holoviews_renderer import _target as interactive_target

    pair = [
        _FakeComparison(
            "a", 0.9, 1.2, 0.10, 0.20, "sea_water_temperature", n_eff=1
        ),  # x=+0.20, y=+0.10
        _FakeComparison(
            "b", 0.8, 0.8, -0.30, 0.40, "sea_water_temperature", n_eff=3
        ),  # x=-0.40, y=-0.30
    ]
    items = [{"label": c.label, "metrics": c.metrics()} for c in pair]

    static_fig = target(
        pair, color_by="variable", summary_points=True, summary_weights="n_eff",
        labels=None,
    )
    static_xy = _star(static_fig)

    obj = interactive_target(
        items, color_by="variable", summary_points=True, summary_weights="n_eff"
    )
    # the base cloud draws as hv.Points, the reference point and the centroid as
    # hv.Scatter -- isinstance + excluding the black reference dot isolates the one
    # centroid here regardless of its marker.
    (star,) = [
        e
        for e in obj.traverse(lambda x: x)
        if isinstance(e, hv.Scatter)
        and e.opts.get(group="style").kwargs.get("color") != "black"
    ]
    interactive_xy = tuple(star.data.iloc[0][["x", "y"]])
    assert interactive_xy == pytest.approx(static_xy)


# ------------------------------------------------------------- summary_split_markers
#
# A centroid always keeps its own group's marker (never a forced shape of its own --
# the reference point alone owns "*"/"star"). summary_split_markers=True changes only
# the *grouping*: a cloud coloured by one field and marker-shaped by a second gets one
# centroid per (colour, marker) combination instead of one per colour group, so it
# reads as "the typical point of this exact colour+shape group."


def _split_recs():
    """8 records crossing 2 variables x 2 signals -- 4 (colour, marker) groups."""
    return [
        _FakeComparison("t-raw-1", 0.95, 1.10, 0.10, 0.20, "temp", signal="raw"),
        _FakeComparison("t-raw-2", 0.90, 1.20, 0.15, 0.22, "temp", signal="raw"),
        _FakeComparison("t-sub-1", 0.80, 0.90, -0.10, 0.25, "temp", signal="subtidal"),
        _FakeComparison("t-sub-2", 0.75, 0.95, -0.05, 0.28, "temp", signal="subtidal"),
        _FakeComparison("s-raw-1", 0.85, 1.05, 0.05, 0.18, "salt", signal="raw"),
        _FakeComparison("s-raw-2", 0.88, 1.02, 0.08, 0.19, "salt", signal="raw"),
        _FakeComparison("s-sub-1", 0.70, 0.85, -0.15, 0.30, "salt", signal="subtidal"),
        _FakeComparison("s-sub-2", 0.65, 0.88, -0.12, 0.31, "salt", signal="subtidal"),
    ]


def test_summary_point_specs_splits_by_colour_and_marker_when_asked():
    """Unit-level: the shared spec builder both renderers call, in isolation."""
    recs = [dict(c.metrics(), label=c.label) for c in _split_recs()]
    coord1 = [r["std_test"] for r in recs]
    coord2 = [r["corr"] for r in recs]

    unsplit = _summary_point_specs(recs, coord1, coord2, "variable", True)
    assert len(unsplit) == 2, "one centroid per variable, the pre-existing behaviour"
    assert {mk for *_, mk in unsplit} == {None}, (
        "a centroid always defers to its own group's marker, never a forced shape"
    )

    split = _summary_point_specs(
        recs, coord1, coord2, "variable", True, marker_field="signal"
    )
    assert len(split) == 4, "one centroid per (variable, signal) combination"
    assert {mk for *_, mk in split} == {None}, (
        "a split centroid defers to its group's own marker too"
    )
    groups = {(rec["variable"], rec["signal"]) for _, _, rec, _ in split}
    assert groups == {("temp", "raw"), ("temp", "subtidal"), ("salt", "raw"), ("salt", "subtidal")}


def test_summary_point_specs_marker_field_same_as_style_field_is_a_no_op():
    """Docstring contract: naming the same field twice doesn't double-split."""
    recs = [dict(c.metrics(), label=c.label) for c in _split_recs()]
    coord1, coord2 = [r["std_test"] for r in recs], [r["corr"] for r in recs]

    same_field = _summary_point_specs(
        recs, coord1, coord2, "variable", True, marker_field="variable"
    )
    unsplit = _summary_point_specs(recs, coord1, coord2, "variable", True)
    assert len(same_field) == len(unsplit) == 2
    assert {mk for *_, mk in same_field} == {None}


def test_taylor_summary_split_markers_matches_colour_and_marker_shape():
    fig = taylor(
        _split_recs(),
        color_by="variable",
        marker_by="signal",
        summary_points=True,
        summary_split_markers=True,
        labels=None,
    )
    lines = _taylor_lines(fig)
    # base sample points draw at Line2D's default zorder (2); the reference star is
    # 3 and every overlay/centroid is 10 -- so zorder==2 isolates the base cloud.
    base = [ln for ln in lines if ln.get_zorder() == 2]
    centroids = [ln for ln in lines if ln.get_zorder() == 10]

    assert base, "expected the base cloud to be found at Line2D's default zorder"
    assert len(centroids) == 4, "one centroid per (variable, signal) combination"
    base_pairs = {(ln.get_markerfacecolor(), ln.get_marker()) for ln in base}
    for c in centroids:
        assert c.get_marker() != "*", (
            "a split centroid keeps its group's own shape, never the reference's '*'"
        )
        assert (c.get_markerfacecolor(), c.get_marker()) in base_pairs, (
            "each centroid's colour+shape must match a real base-cloud group"
        )


def test_target_summary_split_markers_matches_colour_and_marker_shape():
    fig = target(
        _split_recs(),
        color_by="variable",
        marker_by="signal",
        summary_points=True,
        summary_split_markers=True,
        labels=None,
    )
    ax = fig.axes[0]
    base = [c for c in ax.collections if c.get_zorder() == 4]
    centroids = [c for c in ax.collections if c.get_zorder() == 10]

    assert len(centroids) == 4, "one centroid per (variable, signal) combination"
    base_pairs = {
        (mcolors.to_hex(c.get_facecolor()[0]), tuple(c.get_paths()[0].vertices.flat))
        for c in base
    }
    for c in centroids:
        pair = (
            mcolors.to_hex(c.get_facecolor()[0]),
            tuple(c.get_paths()[0].vertices.flat),
        )
        assert pair in base_pairs, (
            "each centroid's colour+marker-path must match a real base-cloud group "
            "(comparing raw marker path vertices, since a PathCollection has no "
            "single 'marker string' the way a Line2D does)"
        )


def test_summary_split_markers_ignored_without_marker_by(comparisons):
    """Docstring contract: no marker_by means the usual one-centroid-per-group."""
    with_flag = target(
        comparisons, color_by="variable", summary_points=True,
        summary_split_markers=True, labels=None,
    )
    without_flag = target(
        comparisons, color_by="variable", summary_points=True, labels=None,
    )
    n_with = sum(1 for c in with_flag.axes[0].collections if c.get_zorder() == 10)
    n_without = sum(1 for c in without_flag.axes[0].collections if c.get_zorder() == 10)
    assert n_with == n_without == 3, "one centroid per variable either way"


def test_interactive_target_summary_split_markers_matches_static_colours():
    """Both renderers must agree on which colour goes with which split centroid."""
    import holoviews as hv

    recs = _split_recs()
    items = [{"label": c.label, "metrics": c.metrics()} for c in recs]

    static_fig = target(
        recs, color_by="variable", marker_by="signal",
        summary_points=True, summary_split_markers=True,
    )
    static_colors = {
        mcolors.to_hex(c.get_facecolor()[0])
        for c in static_fig.axes[0].collections
        if c.get_zorder() == 10
    }

    obj = _interactive_target(
        items, color_by="variable", marker_by="signal",
        summary_points=True, summary_split_markers=True,
    )
    # every overlay/centroid layer draws as hv.Scatter (the base cloud is hv.Points),
    # so isinstance + excluding the black reference dot already isolates exactly the
    # 4 split centroids; the marker check also excludes "star", the reference point's
    # own marker, for a check that still means something regardless of which shape
    # each split group happens to draw.
    centroids = [
        e
        for e in obj.traverse(lambda x: x)
        if isinstance(e, hv.Scatter)
        and e.opts.get(group="style").kwargs.get("marker") != "star"
        and e.opts.get(group="style").kwargs.get("color") != "black"
    ]
    assert len(centroids) == 4
    interactive_colors = {e.opts.get(group="style").kwargs["color"] for e in centroids}
    assert interactive_colors == static_colors
