"""Interactive colour bars tick at the shared round values, in the static text.

Left to itself bokeh picks a colour bar's ticks and formats them, which is how an
interactive bar came to read ``0.466, 0.911`` where the static one reads ``0.5, 1.0``.
Every bar the holoviews renderer builds is now a ``FixedTicker`` at the values
:func:`ocean_skill.plot._colorbar.colorbar_ticks` picks for its range, with a
``major_label_overrides`` entry for *every* tick spelling it as that function does (a
Unicode minus, a shared number of decimals, plain decimals on a log bar). An end with
data beyond it keeps its ``≥``/``≤`` label, spelled the same way (``≥ 3.0``).

The limits themselves come from the static renderer's ``_limits``, which snaps the ends
it chooses outward to round values, so a bar ends on a number too; a limit a caller pins
is never moved. Datetime bars (``color_by="time"``) are not touched by any of this.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
import xarray as xr

from ocean_skill.plot._colorbar import colorbar_ticks, difference_limit
from ocean_skill.plot.registry import render
from ocean_skill.plot.spec import PlotSpec

MINUS = "\N{MINUS SIGN}"
NITRATE = "mole_concentration_of_nitrate_in_sea_water"
CHLOROPHYLL = "mass_concentration_of_chlorophyll_a_in_sea_water"


# --- helpers --------------------------------------------------------------------------


def _ramp(low, high, *, n_lat: int = 12, n_lon: int = 14, geometric: bool = False):
    """Return a lat/lon map running evenly (or geometrically) from low to high."""
    space = np.geomspace if geometric else np.linspace
    return xr.DataArray(
        space(low, high, n_lat * n_lon).reshape(n_lat, n_lon),
        dims=("lat", "lon"),
        coords={
            "lat": np.linspace(20.0, 30.0, n_lat),
            "lon": np.linspace(-100.0, -90.0, n_lon),
        },
        attrs={"units": "mmol m-3"},
    )


def _map_item(field, standard_name=NITRATE):
    return {"field": field, "units": "mmol m-3", "standard_name": standard_name}


def _facet(field, standard_name=NITRATE, **options):
    """Render one map interactively (a single-panel ``field_facet``)."""
    item = {**_map_item(field, standard_name), "facet_dim": None, "row_dim": None}
    spec = PlotSpec(family="field_facet", items=[item])
    return render(spec, renderer="holoviews", **options)


def _row_item(*, difference_amplitude: float = 1.734):
    """Return a comparison whose difference spans +/- ``difference_amplitude``."""
    ramp = np.add.outer(np.linspace(0.0, 6.0, 30), np.linspace(0.0, 6.0, 40))
    spread = np.linspace(-1.0, 1.0, ramp.size).reshape(ramp.shape)
    wiggle = difference_amplitude * spread
    lat, lon = np.linspace(18.0, 26.0, 30), np.linspace(-100.0, -90.0, 40)
    coords = {"lat": lat, "lon": lon}
    reference = xr.DataArray(
        15.0 + ramp, dims=("lat", "lon"), coords=coords, name="reference"
    )
    test = (reference + wiggle).rename("test")
    # named as the aligned Dataset's variables are: holoviews merges the colour range of
    # same-named value dimensions across a layout, which a real comparison's never are
    difference = (test - reference).rename("difference")
    return {
        "aligned": {"test": test, "reference": reference, "difference": difference},
        "units": "degC",
        "standard_name": "sea_water_temperature",
        "labels": ("model", "obs"),
    }


def _meshes(obj):
    import holoviews as hv

    return obj.traverse(lambda x: x, [hv.QuadMesh])


def _clim(mesh) -> tuple[float, float]:
    """Return the colour range hvplot put on a mesh (its value dimension's range)."""
    return tuple(mesh.vdims[0].range)


def _bar_opts(mesh) -> dict:
    """Return the ``colorbar_opts`` the renderer put on a mesh."""
    return mesh.opts.get("plot").kwargs["colorbar_opts"]


def _color_bars(obj):
    """Every bokeh ``ColorBar`` of ``obj``, as bokeh would draw it."""
    import holoviews as hv
    from bokeh.models import ColorBar

    hv.extension("bokeh")
    return list(hv.render(obj, backend="bokeh").select({"type": ColorBar}))


def _expected(lo, hi, *, log=False) -> dict[float, str]:
    """``{tick: label}`` of the shared round ticks for a range."""
    ticks = colorbar_ticks(lo, hi, log=log)
    return dict(zip(ticks.values, ticks.labels, strict=True))


def _is_end_label(text: str) -> bool:
    return text.startswith(("≥", "≤"))


def _coloured_elements(obj) -> list:
    """Return every element of ``obj`` that draws a colour bar."""
    import holoviews as hv

    kinds = [hv.QuadMesh, hv.Image, hv.Points, hv.HeatMap]
    return [
        el
        for el in obj.traverse(lambda x: x, kinds)
        if el.opts.get("plot").kwargs.get("colorbar")
    ]


def _element_range(element) -> tuple[float, float]:
    """Return the colour range an element was drawn on, low end first."""
    low, high = element.vdims[0].range
    if low is None or high is None:  # a scatter / heat map carries it as an option
        low, high = element.opts.get("plot").kwargs["clim"]
    # a depth bar runs its clim deepest-first so the surface reads at the top
    return tuple(sorted((low, high)))


def _assert_round(element) -> None:
    """Check an element's bar is a ``FixedTicker`` on the shared round ticks.

    Its ticks are the shared ones for its own colour range except where a forced end
    label (an end with data beyond it) took the place of one.
    """
    from bokeh.models import FixedTicker

    opts = element.opts.get("plot").kwargs["colorbar_opts"]
    ticker, overrides = opts["ticker"], opts["major_label_overrides"]
    assert isinstance(ticker, FixedTicker), type(ticker).__name__
    ticks = list(ticker.ticks)
    assert ticks == sorted(set(ticks))
    assert set(overrides) == set(ticks), "every tick carries its own label"
    lo, hi = _element_range(element)
    expected = _expected(lo, hi, log=bool(element.opts.get("plot").kwargs.get("logz")))
    for tick in ticks:
        assert lo <= tick <= hi
        if _is_end_label(overrides[tick]):
            assert tick in (lo, hi)
        else:
            assert overrides[tick] == expected[tick]


def _assert_fixed(bar) -> None:
    """Check a bokeh bar kept the renderer's ticker and spells every tick."""
    from bokeh.models import FixedTicker

    assert isinstance(bar.ticker, FixedTicker), type(bar.ticker).__name__
    assert bar.ticker.ticks
    assert set(bar.major_label_overrides) == set(bar.ticker.ticks)


# --- the colorbar_opts the renderer builds, on their own ------------------------------


def test_an_unclipped_bar_ticks_at_the_shared_round_values_each_labelled():
    from ocean_skill.plot.holoviews_renderer import _clip_colorbar_opts

    opts = _clip_colorbar_opts((0.0, 3.0), (0.0, 3.0))
    ticks = colorbar_ticks(0.0, 3.0, log=False)
    assert type(opts["ticker"]).__name__ == "FixedTicker"
    assert opts["ticker"].ticks == list(ticks.values)
    assert opts["major_label_overrides"] == dict(zip(ticks.values, ticks.labels))
    # what the reader sees: shared decimals, nothing like 0.466 / 0.911
    assert list(opts["major_label_overrides"].values()) == [
        "0.0", "0.5", "1.0", "1.5", "2.0", "2.5", "3.0"
    ]


def test_a_bar_always_has_its_ticks_even_when_its_data_range_is_unknown():
    from ocean_skill.plot.holoviews_renderer import _clip_colorbar_opts

    opts = _clip_colorbar_opts((0.0, 3.0), None)
    assert opts is not None
    assert next(iter(opts["major_label_overrides"].values())) == "0.0"


@pytest.mark.parametrize(
    "clim", [(5.0, 5.0), (3.0, 1.0), (float("nan"), 1.0), (0.0, float("inf"))]
)
def test_a_range_with_nothing_to_tick_and_no_clipped_end_leaves_bokeh_alone(clim):
    from ocean_skill.plot.holoviews_renderer import _clip_colorbar_opts

    assert _clip_colorbar_opts(clim, None) is None
    assert _clip_colorbar_opts((5.0, 5.0), (5.0, 5.0)) is None  # nothing past an end


def test_a_clipped_end_is_labelled_in_the_bars_own_round_text():
    from ocean_skill.plot.holoviews_renderer import _clip_colorbar_opts

    top = _clip_colorbar_opts((0.0, 3.0), (0.0, 3.4))["major_label_overrides"]
    assert top[3.0] == "≥ 3.0"
    assert top[0.0] == "0.0"  # the other end is not clipped: an ordinary tick
    both = _clip_colorbar_opts((0.0, 3.0), (-0.4, 3.4))["major_label_overrides"]
    assert (both[0.0], both[3.0]) == ("≤ 0.0", "≥ 3.0")
    # the tick at 2.5 is further than a tenth of the bar from the forced end: it stays
    assert both[2.5] == "2.5"


def test_a_pinned_limit_is_spelled_exactly_not_rounded_into_a_number_it_is_not():
    from ocean_skill.plot.holoviews_renderer import _clip_colorbar_opts

    opts = _clip_colorbar_opts((0.37, 2.71), (0.0, 3.5))["major_label_overrides"]
    assert (opts[0.37], opts[2.71]) == ("≤ 0.37", "≥ 2.71")
    # ... while the ticks between them keep the bar's own one decimal
    assert opts[1.0] == "1.0"


def test_a_tick_within_a_tenth_of_a_forced_end_is_dropped_not_printed_over_it():
    from ocean_skill.plot.holoviews_renderer import _clip_colorbar_opts

    lo, hi = 0.3, 2.7
    opts = _clip_colorbar_opts((lo, hi), (0.0, 3.0))
    ticks = opts["ticker"].ticks
    assert 0.5 not in ticks and 2.5 not in ticks  # 0.2 from an end of a 2.4 bar
    assert set(opts["major_label_overrides"]) == set(ticks)
    for tick in ticks:
        if tick not in (lo, hi):
            assert min(abs(tick - lo), abs(tick - hi)) / (hi - lo) > 0.1


def test_a_clipped_end_can_carry_the_true_extreme():
    from ocean_skill.plot.holoviews_renderer import _clip_colorbar_opts

    opts = _clip_colorbar_opts((0.0, 3.0), (-0.4, 3.4), label_clipped=True)
    labels = opts["major_label_overrides"]
    assert labels[0.0] == f"≤ 0.0 (min {MINUS}0.4)"
    assert labels[3.0] == "≥ 3.0 (max 3.4)"
    assert labels[1.0] == "1.0"  # an ordinary tick says only its number


def test_negative_labels_use_a_true_minus_sign():
    from ocean_skill.plot.holoviews_renderer import _clip_colorbar_opts

    labels = _clip_colorbar_opts((-1.8, 1.8), (-2.5, 2.5))["major_label_overrides"]
    assert labels[-1.8] == f"≤ {MINUS}1.8"
    assert labels[1.8] == "≥ 1.8"
    assert labels[-0.5] == f"{MINUS}0.5"
    assert not any("-" in text for text in labels.values())


def test_a_log_bar_is_plain_decimals_and_a_clipped_end_reads_like_them():
    from ocean_skill.plot.holoviews_renderer import _clip_colorbar_opts

    whole = _clip_colorbar_opts((0.01, 10.0), (0.01, 10.0), log=True)
    assert list(whole["major_label_overrides"].values()) == ["0.01", "0.1", "1", "10"]
    clipped = _clip_colorbar_opts((0.01, 10.0), (0.001, 50.0), log=True)
    assert clipped["major_label_overrides"] == {
        0.01: "≤ 0.01",
        0.1: "0.1",
        1.0: "1",
        10.0: "≥ 10",
    }


# --- through the renderer: a field ----------------------------------------------------


def test_an_unclipped_field_gets_a_fixed_ticker_equal_to_the_shared_ticks():
    obj = _facet(_ramp(0.0213, 2.987))
    (mesh,) = _meshes(obj)
    clim = _clim(mesh)
    # the automatic limits snapped outward to round numbers: 0.0213-2.987 reads 0-3
    assert clim == (0.0, 3.0)
    opts = _bar_opts(mesh)
    ticks = colorbar_ticks(*clim, log=False)
    assert type(opts["ticker"]).__name__ == "FixedTicker"
    assert opts["ticker"].ticks == list(ticks.values)
    assert opts["major_label_overrides"] == dict(zip(ticks.values, ticks.labels))
    (bar,) = _color_bars(obj)  # and bokeh gets the same
    assert bar.ticker.ticks == list(ticks.values)
    spelled = [bar.major_label_overrides[t] for t in bar.ticker.ticks]
    assert spelled == list(ticks.labels)


def test_a_clipped_robust_field_labels_its_ends_with_round_text():
    obj = _facet(_ramp(0.0213, 2.987), robust=True)
    (mesh,) = _meshes(obj)
    lo, hi = _clim(mesh)
    ticks = colorbar_ticks(lo, hi, log=False)
    labels = _bar_opts(mesh)["major_label_overrides"]
    assert labels[lo] == f"≤ {ticks.text(lo)}"
    assert labels[hi] == f"≥ {ticks.text(hi)}"
    # round ends: robust's raw percentiles are 0.31... and 2.69..., not the snapped ones
    for text in (labels[lo], labels[hi]):
        assert re.fullmatch(r"[≤≥] \d+(\.\d)?", text), text
    _assert_round(mesh)
    (bar,) = _color_bars(obj)
    _assert_fixed(bar)


def test_automatic_limits_snap_outward_so_nothing_is_clipped_by_the_snap():
    field = _ramp(0.0213, 2.987)
    (mesh,) = _meshes(_facet(field))
    lo, hi = _clim(mesh)
    assert lo <= float(field.min()) and hi >= float(field.max())
    labels = _bar_opts(mesh)["major_label_overrides"].values()
    assert not any(_is_end_label(text) for text in labels)  # nothing marked clipped


def test_pinned_limits_never_move_and_are_labelled_exactly():
    field = _ramp(0.0213, 2.987)
    obj = _facet(field, vmin=0.37, vmax=2.71)
    (mesh,) = _meshes(obj)
    assert _clim(mesh) == (0.37, 2.71)
    labels = _bar_opts(mesh)["major_label_overrides"]
    assert labels[0.37] == "≤ 0.37" and labels[2.71] == "≥ 2.71"
    _assert_round(mesh)
    (bar,) = _color_bars(obj)
    _assert_fixed(bar)


def test_a_pin_on_one_end_leaves_the_other_to_snap():
    (mesh,) = _meshes(_facet(_ramp(0.0213, 2.987), vmax=2.71))
    lo, hi = _clim(mesh)
    assert hi == 2.71  # the caller's own number
    assert lo == 0.0  # the data's 0.0213, snapped outward


def test_a_log_field_gets_plain_decimal_labels():
    field = _ramp(0.0213, 7.9, geometric=True)
    obj = _facet(field, standard_name=CHLOROPHYLL)
    (mesh,) = _meshes(obj)
    lo, hi = _clim(mesh)
    assert (lo, hi) == pytest.approx((0.02, 8.0))
    ticks = colorbar_ticks(lo, hi, log=True)
    opts = _bar_opts(mesh)
    assert opts["ticker"].ticks == list(ticks.values)
    labels = list(opts["major_label_overrides"].values())
    assert labels == list(ticks.labels)
    assert all(v > 0 for v in ticks.values)
    # plain decimals, as a reader says them: never 1e-2, 10^-2 or 1.0e-02
    assert all(re.fullmatch(r"\d+(\.\d+)?", text) for text in labels), labels
    assert {"0.1", "1"} <= set(labels)
    _assert_round(mesh)
    (bar,) = _color_bars(obj)
    _assert_fixed(bar)


def test_a_log_field_clipped_at_the_top_labels_the_end_like_its_ticks():
    field = _ramp(0.0213, 7.9, geometric=True)
    obj = _facet(field, standard_name=CHLOROPHYLL, vmax=3.0)
    (mesh,) = _meshes(obj)
    labels = _bar_opts(mesh)["major_label_overrides"]
    assert labels[3.0] == "≥ 3"
    assert not any("e" in text for text in labels.values())


def test_a_rasterized_map_gets_the_same_ticks_on_its_image():
    pytest.importorskip("datashader")
    obj = _facet(_ramp(0.0213, 2.987), rasterize=True)
    (image,) = _coloured_elements(obj)
    assert type(image).__name__ == "Image"  # the mesh, once rasterized
    _assert_round(image)
    ticks = colorbar_ticks(0.0, 3.0, log=False)
    assert image.opts.get("plot").kwargs["colorbar_opts"]["ticker"].ticks == list(
        ticks.values
    )


# --- a difference panel ---------------------------------------------------------------


def test_a_difference_panel_gets_symmetric_round_ticks():
    item = _row_item()
    difference = item["aligned"]["difference"]
    obj = render(
        PlotSpec(family="field_row", items=[item], options={}), renderer="holoviews"
    )
    _test, _reference, diff = _meshes(obj)
    raw = float(np.percentile(np.abs(difference), 98))
    dmax = difference_limit(difference)
    assert _clim(diff) == (-dmax, dmax)
    assert raw <= dmax < raw + 0.1  # snapped up, by less than a fifth of a tick step
    opts = _bar_opts(diff)
    ticks = list(opts["ticker"].ticks)
    labels = opts["major_label_overrides"]
    assert ticks == [-t for t in reversed(ticks)]  # symmetric about zero
    assert labels[0.0] == "0.0"  # the middle of a diverging bar is a clean zero
    # the 98th-percentile range always has data beyond it, so both ends are labelled
    assert labels[-dmax] == f"≤ {MINUS}{dmax}" and labels[dmax] == f"≥ {dmax}"
    for value, text in labels.items():
        assert "-" not in text  # a true minus, never a hyphen
        if value < 0 and not _is_end_label(text):
            assert text.startswith(MINUS)
    _assert_round(diff)


def test_the_rows_scale_bars_and_difference_bar_each_get_their_ticks():
    obj = render(
        PlotSpec(family="field_row", items=[_row_item()], options={}),
        renderer="holoviews",
    )
    elements = _coloured_elements(obj)
    assert len(elements) == 3
    for element in elements:
        _assert_round(element)
    test_opts, reference_opts, _ = (e.opts.get("plot").kwargs for e in elements)
    # one shared scale: test and reference tick identically
    assert (
        test_opts["colorbar_opts"]["ticker"].ticks
        == reference_opts["colorbar_opts"]["ticker"].ticks
    )
    bars = _color_bars(obj)
    assert len(bars) == 3
    for bar in bars:
        _assert_fixed(bar)


# --- every colour bar, in every family -----------------------------------------------


def _render(family, items, **options):
    return render(
        PlotSpec(family=family, items=items, options=options), renderer="holoviews"
    )


def _section_items(n=1):
    from tests.test_color_limits import _section_item

    return [_section_item() for _ in range(n)]


def _time_depth_items(n=1):
    from tests.test_color_limits import _time_depth_item

    return [_time_depth_item() for _ in range(n)]


def _section_row_items(n=1):
    from tests.test_section import _section_row_item

    return [_section_row_item() for _ in range(n)]


def _time_depth_row_items(n=1):
    from tests.test_renderers import _time_depth_row_item

    return [_time_depth_row_item() for _ in range(n)]


def _map_items(*fields):
    return [_map_item(field) for field in fields]


#: ``(id, build)`` for every family that draws a colour bar, each building its figure
#: lazily (so collecting this file imports nothing heavy).
_FAMILIES = [
    ("field_facet", lambda: _facet(_ramp(0.0213, 2.987))),
    ("field_row", lambda: _render("field_row", [_row_item()])),
    ("field_grid", lambda: _render("field_row", [_row_item(), _row_item()])),
    (
        "field_map_grid",
        lambda: _render(
            "field_map_grid", _map_items(_ramp(0.0213, 2.987), _ramp(1.0, 4.0))
        ),
    ),
    (
        "field_map_grid-shared",
        lambda: _render(
            "field_map_grid",
            _map_items(_ramp(0.0213, 2.987), _ramp(1.0, 4.0)),
            shared_limits=True,
        ),
    ),
    ("section", lambda: _render("section", _section_items())),
    ("section_grid", lambda: _render("section", _section_items(2), shared_limits=True)),
    ("section_row", lambda: _render("section_row", _section_row_items())),
    (
        "section_row_grid",
        lambda: _render("section_row", _section_row_items(2), shared_limits=True),
    ),
    ("time_depth", lambda: _render("time_depth", _time_depth_items())),
    (
        "time_depth-scatter",
        lambda: _render("time_depth", _time_depth_items(), mark="scatter"),
    ),
    (
        "time_depth_grid",
        lambda: _render("time_depth", _time_depth_items(2), shared_limits=True),
    ),
    ("time_depth_row", lambda: _render("time_depth_row", _time_depth_row_items())),
    (
        "time_depth_row-scatter",
        lambda: _render("time_depth_row", _time_depth_row_items(), mark="scatter"),
    ),
    (
        "time_depth_row_grid",
        lambda: _render("time_depth_row", _time_depth_row_items(2), shared_limits=True),
    ),
]


@pytest.mark.parametrize(
    "build", [c[1] for c in _FAMILIES], ids=[c[0] for c in _FAMILIES]
)
def test_every_family_draws_its_bars_on_the_shared_round_ticks(build):
    obj = build()
    elements = _coloured_elements(obj)
    assert elements
    for element in elements:
        _assert_round(element)
    bars = _color_bars(obj)  # and bokeh's own bars kept them
    assert bars
    for bar in bars:
        _assert_fixed(bar)


def test_a_skill_map_bar_gets_the_shared_ticks():
    from ocean_skill.plot.map_metrics import build_items
    from tests.test_map_metrics import _records

    items = build_items(_records(n=24), metrics=("bias",), grid="regular")
    obj = _render("skill_map", items)
    elements = _coloured_elements(obj)
    assert elements
    for element in elements:
        _assert_round(element)
    for bar in _color_bars(obj):
        _assert_fixed(bar)


def test_a_portrait_bar_gets_the_shared_ticks():
    from tests.test_portrait import _FakeComparison, _items

    comparisons = [
        _FakeComparison("A", test="runA", variable="nitrate", bias=0.1),
        _FakeComparison("B", test="runB", variable="nitrate", bias=-0.35),
        _FakeComparison("C", test="runC", variable="nitrate", bias=0.8),
    ]
    obj = _render("portrait", _items(comparisons), metric_names="bias")
    (heat,) = _coloured_elements(obj)
    _assert_round(heat)
    (bar,) = _color_bars(obj)
    _assert_fixed(bar)


def test_a_movie_carries_the_shared_ticks_on_every_frame():
    from tests.test_colorbar_clipping import _movie_frames

    obj = render(
        PlotSpec(family="field_movie", items=_movie_frames(), options={}),
        renderer="holoviews",
        widget="dropdown",
    )
    elements = _coloured_elements(obj)
    assert len(elements) == 3 * len(_movie_frames())  # three panels, every frame
    for element in elements:
        _assert_round(element)
    for bar in _color_bars(obj):
        _assert_fixed(bar)


# --- XY: a depth bar is round, a datetime bar is left alone ---------------------------


def _xy(**options):
    from tests.test_xy_holoviews import _interactive, _ts_items

    return _interactive(_ts_items(), **options)


def test_an_xy_depth_bar_ticks_at_the_shared_round_values():
    from ocean_skill.plot import xy as xy_plot
    from tests.test_xy_holoviews import _ts_items

    scale = xy_plot.compose(_ts_items(), color_by="depth").colorbar
    obj = _xy(color_by="depth")
    (points,) = _coloured_elements(obj)
    _assert_round(points)  # the reversed (surface-at-top) clim does not matter
    ticks = colorbar_ticks(scale.vmin, scale.vmax, log=False)
    (bar,) = _color_bars(obj)
    assert bar.ticker.ticks == list(ticks.values)
    spelled = [bar.major_label_overrides[t] for t in bar.ticker.ticks]
    assert spelled == list(ticks.labels)


def test_an_xy_depth_pin_keeps_its_end_label_among_round_ticks():
    obj = _xy(color_by="depth", vmax=500)
    (points,) = _coloured_elements(obj)
    _assert_round(points)
    (bar,) = _color_bars(obj)
    assert bar.major_label_overrides[500.0] == "≥ 500"
    assert {"100", "200", "300", "400"} <= set(bar.major_label_overrides.values())


def test_a_datetime_bar_is_left_alone():
    from bokeh.models import DatetimeTickFormatter

    (bar,) = _color_bars(_xy(color_by="time"))
    assert type(bar.ticker).__name__ == "DatetimeTicker"
    assert not bar.major_label_overrides
    assert isinstance(bar.formatter, DatetimeTickFormatter)


def test_a_pinned_datetime_bar_keeps_its_date_labels():
    from bokeh.models import DatetimeTickFormatter

    (bar,) = _color_bars(_xy(color_by="time", vmax="2020-06-01"))
    assert list(bar.major_label_overrides.values()) == ["≥ 2020-06-01"]
    assert isinstance(bar.formatter, DatetimeTickFormatter)
    # no round numbers were put on a bar of dates
    assert len(bar.ticker.ticks) >= 3
