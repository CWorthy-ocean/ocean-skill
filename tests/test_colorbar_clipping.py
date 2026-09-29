"""A colour bar says when it is clipped: extension arrows, and ``≥``/``≤`` end labels.

``robust=``, a pinned ``vmin``/``vmax`` and a difference panel's 98th-percentile range
all draw values past an end of the bar in that end's colour, which reads as the top of
the scale when it is only where the scale gave up. The static renderer now draws
matplotlib's extension triangle on each end that has data beyond it; bokeh cannot grow
one, so the interactive renderer forces a tick on that end and labels it ``≥ 27`` or
``≤ 3``. ``colorbar_label_clipped=True`` adds the true extreme to either -- ``max 29.9``
at the arrow's tip, ``≥ 27 (max 29.9)`` on the tick.

What both renderers must agree on is *which* ends are marked, so that is tested as a
pair as well as on each side.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from ocean_skill.plot.matplotlib_renderer import (
    _clip_text,
    _data_range,
    _extend,
    _extend_of,
    _with_range,
)
from ocean_skill.plot.registry import render
from ocean_skill.plot.spec import PlotSpec

NITRATE = "mole_concentration_of_nitrate_in_sea_water"
TEMPERATURE = "sea_water_temperature"


# --- the pure helpers -----------------------------------------------------------------


def test_data_range_is_the_finite_extremes_across_every_array():
    a = np.array([1.0, np.nan, 3.0])
    b = np.array([-2.0, np.inf, 2.0])
    assert _data_range(a, b) == (-2.0, 3.0)


def test_data_range_of_nothing_finite_is_none():
    assert _data_range(np.array([np.nan, np.inf])) is None


def test_data_range_under_log_ignores_values_a_lognorm_would_mask():
    # a LogNorm masks <= 0 as *bad*; it is never clipped to the low end, so it cannot be
    # what an extension arrow points at
    assert _data_range(np.array([-5.0, 0.0, 0.5, 8.0]), log=True) == (0.5, 8.0)
    assert _data_range(np.array([-5.0, 0.0]), log=True) is None


@pytest.mark.parametrize(
    "data, expected",
    [
        ((2.0, 8.0), "neither"),  # inside
        ((0.0, 8.0), "min"),
        ((2.0, 12.0), "max"),
        ((0.0, 12.0), "both"),
        ((1.0, 10.0), "neither"),  # exactly on the limits is not beyond them
        (None, "neither"),  # unknown range: draw the bar as before
    ],
)
def test_extend_names_the_ends_with_data_beyond_them(data, expected):
    assert _extend(1.0, 10.0, data) == expected


def test_clip_text_reads_as_the_extreme_it_hides():
    assert _clip_text("max", 29.94) == "max 29.9"
    assert _clip_text("min", -1.83) == "min \N{MINUS SIGN}1.83"


def test_a_norm_carries_its_data_range_and_a_bare_one_draws_no_arrow():
    import matplotlib.colors as mcolors

    norm = _with_range(mcolors.Normalize(0.0, 10.0), np.array([-1.0, 5.0, 20.0]))
    assert _extend_of(norm) == "both"
    assert _extend_of(mcolors.Normalize(0.0, 10.0)) == "neither"


# --- static: field_facet ---------------------------------------------------------------


def _ramp_map(n_lat: int = 12, n_lon: int = 14, *, high=100.0, low=-40.0):
    """A 0-30 ramp with one high and one low outlier, so robust clips both ends."""
    vals = np.linspace(0.0, 30.0, n_lat * n_lon).reshape(n_lat, n_lon)
    if high is not None:
        vals[0, 0] = high
    if low is not None:
        vals[-1, -1] = low
    return xr.DataArray(
        vals,
        dims=("lat", "lon"),
        coords={
            "lat": np.linspace(20.0, 30.0, n_lat),
            "lon": np.linspace(-100.0, -90.0, n_lon),
        },
        attrs={"units": "mmol m-3"},
    )


def _facet_spec(field):
    item = {
        "field": field,
        "facet_dim": None,
        "row_dim": None,
        "units": "mmol m-3",
        "standard_name": NITRATE,
    }
    return PlotSpec(family="field_facet", items=[item])


def _mappables(fig):
    """Every drawn field (mesh or contour set), in axes order; each has ``.colorbar``."""
    from matplotlib.collections import QuadMesh
    from matplotlib.contour import ContourSet

    return [
        artist
        for ax in fig.axes
        if not getattr(ax, "_osk_cbar_parents", None)  # a bar's own `solids` mesh
        for artist in ax.collections
        if isinstance(artist, QuadMesh | ContourSet)
    ]


def _mappable(fig):
    return _mappables(fig)[0]


def _cbar_texts(fig):
    """Every annotation drawn on a colour bar's own axes."""
    return [
        t.get_text()
        for ax in fig.axes
        if getattr(ax, "_osk_cbar_parents", None)
        for t in ax.texts
    ]


def test_a_bar_showing_the_full_range_has_no_arrow():
    fig = render(_facet_spec(_ramp_map()))
    assert _mappable(fig).colorbar.extend == "neither"


def test_robust_puts_an_arrow_on_each_end_it_clips():
    fig = render(_facet_spec(_ramp_map()), robust=True)
    assert _mappable(fig).colorbar.extend == "both"


def test_an_end_with_nothing_beyond_it_gets_no_arrow_even_under_robust():
    # the ramp's own minimum is 0, so pinning vmin there leaves nothing below the bar
    field = _ramp_map(low=None)
    fig = render(_facet_spec(field), robust=True, vmin=0.0)
    assert _mappable(fig).colorbar.extend == "max"


def test_a_pinned_vmax_below_the_data_arrows_the_top_only():
    fig = render(_facet_spec(_ramp_map()), vmax=20.0)
    assert _mappable(fig).colorbar.extend == "max"


def test_pins_that_contain_the_data_draw_no_arrow():
    fig = render(_facet_spec(_ramp_map()), vmin=-100.0, vmax=200.0)
    assert _mappable(fig).colorbar.extend == "neither"


def test_an_explicit_extend_in_colorbar_kwargs_wins():
    fig = render(
        _facet_spec(_ramp_map()),
        robust=True,
        colorbar_kwargs={"extend": "neither"},
    )
    assert _mappable(fig).colorbar.extend == "neither"


def test_contourf_fills_the_clipped_regions_instead_of_leaving_holes():
    fig = render(_facet_spec(_ramp_map()), robust=True, mark="contourf")
    cs = _mappable(fig)
    # a contour set only fills between its levels unless told to extend past them, so
    # without this the outliers would be bare holes rather than the end colour
    assert cs.extend == "both"
    assert cs.colorbar.extend == "both"


# --- static: the true extreme at the arrow's tip --------------------------------------


def test_no_tip_label_unless_asked_for():
    fig = render(_facet_spec(_ramp_map()), robust=True)
    assert _cbar_texts(fig) == []


def test_label_clipped_writes_the_true_extreme_at_each_arrow():
    fig = render(_facet_spec(_ramp_map()), robust=True, colorbar_label_clipped=True)
    assert sorted(_cbar_texts(fig)) == ["max 100", "min \N{MINUS SIGN}40"]


def test_label_clipped_labels_only_the_ends_with_an_arrow():
    fig = render(_facet_spec(_ramp_map()), vmax=20.0, colorbar_label_clipped=True)
    assert _cbar_texts(fig) == ["max 100"]


def test_label_clipped_on_an_unclipped_bar_writes_nothing():
    fig = render(_facet_spec(_ramp_map()), colorbar_label_clipped=True)
    assert _cbar_texts(fig) == []


def test_label_clipped_also_works_on_a_contourf_bar():
    fig = render(
        _facet_spec(_ramp_map()),
        robust=True,
        mark="contourf",
        colorbar_label_clipped=True,
    )
    assert sorted(_cbar_texts(fig)) == ["max 100", "min \N{MINUS SIGN}40"]


def test_label_clipped_is_a_known_option_of_the_static_families():
    from ocean_skill.plot.matplotlib_renderer import _top_level_options

    assert "colorbar_label_clipped" in _top_level_options()


# --- static: a row's difference bar, and the bar's length -----------------------------


def _row_item(*, test_outlier=400.0):
    rng = np.random.default_rng(1)
    lat, lon = np.linspace(18, 26, 30), np.linspace(260, 270, 40)

    def field(offset):
        ramp = np.add.outer(np.linspace(0, 6, 30), np.linspace(0, 6, 40))
        vals = 15 + offset + ramp + 0.3 * rng.normal(size=(30, 40))
        return xr.DataArray(vals, dims=("lat", "lon"), coords={"lat": lat, "lon": lon})

    test, ref = field(1.0), field(0.0)
    if test_outlier is not None:
        test[0, 0] = test_outlier
    return {
        "aligned": {"test": test, "reference": ref, "difference": test - ref},
        "units": "degC",
        "standard_name": TEMPERATURE,
        "labels": ("model", "obs"),
    }


def test_the_difference_bar_is_arrowed_because_it_is_always_clipped():
    """Difference panels are cut at the 98th percentile of |d|, whatever robust says."""
    fig = render(PlotSpec(family="field_row", items=[_row_item()], options={}))
    _test_mesh, ref_mesh, diff_mesh = _mappables(fig)
    # the scale bar hangs off the reference mesh (it serves test and reference alike)
    assert ref_mesh.colorbar.extend == "neither"  # full range, nothing clipped
    assert diff_mesh.colorbar.extend in ("max", "both")


def test_the_scale_bar_answers_for_test_and_reference_together():
    """The one bar the test and reference share reports the reference's outlier.

    The test field peaks near 28, so a bar that only looked at it would say ``max 28``;
    the reference's 900 is what the shared scale actually cut off.
    """
    item = _row_item(test_outlier=None)
    item["aligned"]["reference"][0, 0] = 900.0
    fig = render(
        PlotSpec(family="field_row", items=[item], options={}),
        robust=True,
        colorbar_label_clipped=True,
    )
    assert "max 900" in _cbar_texts(fig)


def test_an_arrowed_bar_still_spans_the_panels_it_describes():
    """Matplotlib's extend locator re-imposes the bar's aspect on every draw.

    Left alone that undoes ``_align_colorbars`` and leaves the bar a fraction of the
    width of the two panels it belongs to; the arrow tips, not the body, must reach
    the panels' edges.
    """
    fig = render(
        PlotSpec(family="field_row", items=[_row_item()], options={}), robust=True
    )
    fig.canvas.draw()
    bars = [ax for ax in fig.axes if getattr(ax, "_osk_cbar_parents", None)]
    assert len(bars) == 2
    for cax in bars:
        boxes = [p.get_position() for p in cax._osk_cbar_parents]
        span = max(b.x1 for b in boxes) - min(b.x0 for b in boxes)
        body = cax.get_position()
        # the body is the span less room for two arrows of 5% each (a fifth at most)
        assert body.width >= 0.8 * span
        assert body.width <= span + 1e-6


# --- static: a movie answers for every frame ------------------------------------------


def _movie_frames():
    def frame(index, *, outlier=None):
        lat, lon = np.linspace(18, 26, 8), np.linspace(-100, -90, 10)
        ramp = np.linspace(0, 1, 80).reshape(8, 10)
        test = xr.DataArray(
            6.0 + ramp, dims=("lat", "lon"), coords={"lat": lat, "lon": lon}
        )
        ref = test - 1.0
        if outlier is not None:
            test[0, 0] = outlier
        return {
            "aligned": {"test": test, "reference": ref, "difference": test - ref},
            "units": "mmol m-3",
            "standard_name": NITRATE,
            "labels": ("a", "b"),
            "frame_label": f"frame {index}",
        }

    return [frame(0), frame(1), frame(2, outlier=50.0)]


@pytest.mark.slow
def test_a_movies_arrow_answers_for_a_later_frame_not_just_the_first():
    """With ``shared_limits=False`` the scale comes from frame 0 alone.

    Frame 2's outlier is then past the end of a bar that never moves, so the arrow has
    to be about every frame, not the one the scale was measured from.
    """
    spec = PlotSpec(
        family="field_movie", items=_movie_frames(), options={"shared_limits": False}
    )
    _test_mesh, scale, _diff_mesh = _mappables(render(spec)._fig)
    assert scale.colorbar.extend == "max"


@pytest.mark.slow
def test_a_movie_scaled_to_every_frame_has_nothing_to_flag():
    spec = PlotSpec(family="field_movie", items=_movie_frames(), options={})
    _test_mesh, scale, _diff_mesh = _mappables(render(spec)._fig)
    assert scale.colorbar.extend == "neither"


# --- metric maps: the percentile-scaled skill_map / portrait --------------------------


def test_metric_colors_records_the_extremes_of_the_values_it_scaled_to():
    from ocean_skill.colormaps import metric_colors

    values = np.r_[np.random.default_rng(0).normal(size=500), 9.0]
    colors = metric_colors("bias", values, standard_name=TEMPERATURE)
    low, high = colors.data_range()
    assert high == pytest.approx(9.0)
    assert low == pytest.approx(values.min())
    assert colors.vmax < 9.0  # the 98th-percentile spread leaves the outlier past it


def test_metric_colors_without_values_has_no_range_and_no_arrow():
    from ocean_skill.colormaps import metric_colors

    colors = metric_colors("bias", None)
    assert colors.data_range() is None
    assert _extend_of(colors.norm()) == "neither"


def test_covering_widens_the_recorded_extremes_but_never_the_limits():
    from ocean_skill.colormaps import metric_colors

    values = np.random.default_rng(0).normal(size=500)
    colors = metric_colors("bias", values, standard_name=TEMPERATURE)
    wider = colors.covering(np.array([25.0, np.nan]))
    assert (wider.vmin, wider.vmax) == (colors.vmin, colors.vmax)
    assert wider.data_range() == (colors.data_min, 25.0)
    assert colors.covering(np.array([np.nan])) is colors  # nothing to add


def test_a_metric_bar_arrows_the_ends_its_percentile_scale_cuts_off():
    import matplotlib.pyplot as plt

    from ocean_skill.colormaps import metric_colors
    from ocean_skill.plot.matplotlib_renderer import _draw_colorbar

    values = np.r_[np.random.default_rng(0).normal(size=500), 9.0]
    colors = metric_colors("bias", values, standard_name=TEMPERATURE)
    fig, ax = plt.subplots()
    im = ax.pcolormesh(
        values[:400].reshape(20, 20), cmap=colors.cmap, norm=colors.norm()
    )
    cbar = _draw_colorbar(fig, im, ax, "bias", None, {})
    plt.close(fig)
    # the scale is symmetric about zero at the 98th percentile of |bias|, so the tails
    # of a normal sample fall past it on both sides
    assert cbar.extend == "both"


# --- interactive: the ≥ / ≤ end labels -------------------------------------------------


def _color_bars(obj):
    import holoviews as hv
    from bokeh.models import ColorBar

    hv.extension("bokeh")
    return list(hv.render(obj, backend="bokeh").select({"type": ColorBar}))


def test_clip_opts_are_none_when_nothing_is_clipped():
    from ocean_skill.plot.holoviews_renderer import _clip_colorbar_opts

    assert _clip_colorbar_opts((0.0, 10.0), (0.0, 10.0)) is None
    assert _clip_colorbar_opts((0.0, 10.0), None) is None


def test_clip_opts_force_and_label_a_tick_on_each_clipped_end():
    from ocean_skill.plot.holoviews_renderer import _clip_colorbar_opts

    opts = _clip_colorbar_opts((3.0, 27.0), (-1.8, 29.9))
    assert opts["major_label_overrides"] == {
        3.0: "≤ 3",
        27.0: "≥ 27",
    }
    ticks = opts["ticker"].ticks
    assert 3.0 in ticks and 27.0 in ticks


def test_clip_opts_mark_only_the_clipped_end():
    from ocean_skill.plot.holoviews_renderer import _clip_colorbar_opts

    opts = _clip_colorbar_opts((3.0, 27.0), (3.0, 29.9))
    assert opts["major_label_overrides"] == {27.0: "≥ 27"}


def test_clip_opts_label_clipped_adds_the_true_extreme():
    from ocean_skill.plot.holoviews_renderer import _clip_colorbar_opts

    opts = _clip_colorbar_opts((3.0, 27.0), (-1.8, 29.9), label_clipped=True)
    assert opts["major_label_overrides"] == {
        3.0: "≤ 3 (min \N{MINUS SIGN}1.8)",
        27.0: "≥ 27 (max 29.9)",
    }


def test_clip_opts_keep_interior_ticks_off_the_end_labels():
    from ocean_skill.plot.holoviews_renderer import _clip_colorbar_opts

    opts = _clip_colorbar_opts((14.0, 25.0), (-10.0, 40.0))
    span = 25.0 - 14.0
    for tick in opts["ticker"].ticks:
        if tick not in (14.0, 25.0):
            # bokeh would print an interior label straight over the end label
            assert min(abs(tick - 14.0), abs(tick - 25.0)) / span > 0.1


def test_clip_opts_under_log_use_positive_ticks():
    from ocean_skill.plot.holoviews_renderer import _clip_colorbar_opts

    opts = _clip_colorbar_opts((0.01, 10.0), (0.001, 50.0), log=True)
    assert all(t > 0 for t in opts["ticker"].ticks)
    assert set(opts["major_label_overrides"]) == {0.01, 10.0}


def _interactive(field, **options):
    return render(_facet_spec(field), renderer="holoviews", **options)


def test_interactive_bar_marks_the_clipped_ends_under_robust():
    (bar,) = _color_bars(_interactive(_ramp_map(), robust=True))
    labels = sorted(bar.major_label_overrides.values())
    assert len(labels) == 2
    assert any(text.startswith("≤") for text in labels)
    assert any(text.startswith("≥") for text in labels)


def test_interactive_bar_of_the_full_range_keeps_bokehs_own_ticks():
    (bar,) = _color_bars(_interactive(_ramp_map()))
    assert bar.major_label_overrides == {}
    assert type(bar.ticker).__name__ == "BasicTicker"


def test_interactive_label_clipped_adds_the_true_extreme():
    (bar,) = _color_bars(
        _interactive(_ramp_map(), robust=True, colorbar_label_clipped=True)
    )
    text = " ".join(bar.major_label_overrides.values())
    assert "(max 100)" in text
    assert "(min \N{MINUS SIGN}40)" in text


def test_interactive_row_marks_test_and_reference_bars_alike():
    """One shared scale: an outlier in either panel is an outlier on both bars."""
    item = _row_item(test_outlier=None)
    item["aligned"]["reference"][0, 0] = 900.0
    obj = render(
        PlotSpec(family="field_row", items=[item], options={}),
        renderer="holoviews",
        robust=True,
        colorbar_label_clipped=True,
    )
    # the difference bar is titled differently; the other two share the scale's units
    test_bar, ref_bar = [b for b in _color_bars(obj) if b.title == "degC"]
    assert test_bar.major_label_overrides == ref_bar.major_label_overrides
    assert "(max 900)" in " ".join(test_bar.major_label_overrides.values())


# --- every family that draws a colour bar, in both renderers --------------------------


def _clipped_cases():
    """``(id, family, items, options)`` for each field family, each drawn with a top
    that a pinned ``vmax`` or ``robust`` cuts off, so its bar has to say so.
    """
    from tests.test_color_limits import (
        _outlier_time_depth,
        _section_item,
        _time_depth_item,
    )
    from tests.test_renderers import _item, _time_depth_row_item
    from tests.test_section import _section_row_item

    map_item = {"field": _ramp_map(), "units": "mmol m-3", "standard_name": NITRATE}
    pin = {"vmax": 20.0}
    return [
        ("section", "section", [_section_item()], pin),
        (
            "cross",
            "cross",
            [_section_item(label="lon fixed"), _section_item(label="lat fixed")],
            pin,
        ),
        ("time_depth", "time_depth", [_time_depth_item()], pin),
        (
            "time_depth-scatter",
            "time_depth",
            [_time_depth_item()],
            {**pin, "mark": "scatter"},
        ),
        (
            "time_depth_grid",
            "time_depth",
            [
                _time_depth_item(_outlier_time_depth(), label="a"),
                _time_depth_item(_outlier_time_depth(), label="b"),
            ],
            pin,
        ),
        (
            "field_map_grid",
            "field_map_grid",
            [map_item, dict(map_item)],
            {"robust": True},
        ),
        (
            "field_row",
            "field_row",
            [_item(NITRATE, "woa23_nitrate", "nitrate")],
            {"robust": True},
        ),
        (
            "field_grid",
            "field_row",
            [
                _item(NITRATE, "woa23_nitrate", "nitrate"),
                _item(NITRATE, "woa23_nitrate", "nitrate"),
            ],
            {"robust": True},
        ),
        ("section_row", "section_row", [_section_row_item()], {"robust": True}),
        (
            "time_depth_row",
            "time_depth_row",
            [_time_depth_row_item()],
            {"robust": True},
        ),
        (
            "time_depth_row_grid",
            "time_depth_row",
            [_time_depth_row_item("a", "A"), _time_depth_row_item("b", "B")],
            {"robust": True},
        ),
    ]


_CLIPPED = _clipped_cases()


@pytest.mark.parametrize(
    "family, items, options", [c[1:] for c in _CLIPPED], ids=[c[0] for c in _CLIPPED]
)
def test_every_static_family_arrows_and_labels_its_clipped_top(family, items, options):
    fig = render(
        PlotSpec(family=family, items=items),
        colorbar_label_clipped=True,
        **options,
    )
    assert any(text.startswith("max ") for text in _cbar_texts(fig))


@pytest.mark.parametrize(
    "family, items, options", [c[1:] for c in _CLIPPED], ids=[c[0] for c in _CLIPPED]
)
def test_every_interactive_family_marks_and_labels_its_clipped_top(
    family, items, options
):
    obj = render(
        PlotSpec(family=family, items=items),
        renderer="holoviews",
        colorbar_label_clipped=True,
        **options,
    )
    text = " ".join(
        t for bar in _color_bars(obj) for t in bar.major_label_overrides.values()
    )
    assert "≥" in text
    assert "(max " in text


def test_skill_map_bars_mark_the_tail_their_percentile_scale_cuts_off():
    """A skill_map bar is scaled to a percentile, so a lone bad station is past its end."""
    from ocean_skill.plot.map_metrics import build_items
    from tests.test_map_metrics import _records

    df = _records(n=24)
    df.loc[0, "bias"] = 40.0
    items = build_items(df, metrics=("bias",), grid="regular")
    spec = PlotSpec(family="skill_map", items=items, options={})

    fig = render(spec, colorbar_label_clipped=True)
    assert any(t.startswith("max ") for t in _cbar_texts(fig))

    text = " ".join(
        t
        for bar in _color_bars(
            render(spec, renderer="holoviews", colorbar_label_clipped=True)
        )
        for t in bar.major_label_overrides.values()
    )
    assert "≥" in text and "(max " in text


# --- the two renderers agree ----------------------------------------------------------


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"robust": True},
        {"vmax": 20.0},
        {"vmin": 5.0},
        {"vmin": -100.0, "vmax": 200.0},
    ],
    ids=["full-range", "robust", "vmax-pin", "vmin-pin", "roomy-pins"],
)
def test_both_renderers_mark_the_same_ends(options):
    field = _ramp_map()
    static = _mappable(render(_facet_spec(field), **options)).colorbar.extend
    (bar,) = _color_bars(_interactive(field, **options))
    text = list(bar.major_label_overrides.values())
    interactive = {
        (True, True): "both",
        (True, False): "min",
        (False, True): "max",
        (False, False): "neither",
    }[(any(t.startswith("≤") for t in text), any(t.startswith("≥") for t in text))]
    assert static == interactive
