"""Static colour bars tick at round numbers, and an automatic range ends on one.

``Field.plot()``'s matplotlib bars used to read whatever the data and matplotlib's
locator happened to produce: ``0.021, 0.466, 0.911`` on a filled-contour bar (its ticks
sit on its band edges), ``10⁻²`` on a log bar, an offset such as ``+3.4e1`` on a narrow
one. :mod:`ocean_skill.plot._colorbar` now decides which values a bar ticks and where an
*automatic* range ends; the static renderer asks it for both (the interactive one asks
it too, so the two renderers read the same -- tested with the interactive renderer's own
files). Limits a caller pinned are never moved.

Everything here draws sections or plain figures rather than cartopy maps: a colour bar
is the same object either way, and a map costs a coastline.
"""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from matplotlib.ticker import FixedLocator, NullFormatter

from ocean_skill.align import ALONG_DIM
from ocean_skill.plot import matplotlib_renderer as mr
from ocean_skill.plot._colorbar import Ticks, difference_limit
from ocean_skill.plot.matplotlib_renderer import _draw_colorbar, _limits
from ocean_skill.plot.registry import render
from ocean_skill.plot.spec import PlotSpec

MINUS = "\N{MINUS SIGN}"
CHLOROPHYLL = "mass_concentration_of_chlorophyll_a_in_sea_water"

#: What 0.0213-2.987 reads as on a round bar: its limits snap to 0 and 3, and the ticks
#: are every half.
ROUND = ["0.0", "0.5", "1.0", "1.5", "2.0", "2.5", "3.0"]


# --- fixtures -------------------------------------------------------------------------


def _grid(values) -> xr.DataArray:
    """Return ``values`` as a (z, along) section: 6 depths by 8 stations."""
    values = np.asarray(values, dtype="float64").reshape(6, 8)
    return xr.DataArray(
        values,
        dims=("z", ALONG_DIM),
        coords={
            "z": -np.array([0.0, 10.0, 25.0, 50.0, 100.0, 200.0]),
            ALONG_DIM: np.linspace(0.0, 150.0, 8),
            "lon": (ALONG_DIM, np.linspace(-95.0, -93.0, 8)),
            "lat": (ALONG_DIM, np.linspace(24.0, 26.0, 8)),
        },
    )


def _ramp(lo: float = 0.0213, hi: float = 2.987) -> xr.DataArray:
    return _grid(np.linspace(lo, hi, 48))


def _section(field=None, **overrides) -> PlotSpec:
    item = {
        "field": _ramp() if field is None else field,
        "units": "mmol m-3",
        "standard_name": None,
        "depth": None,
        "label": "roms_run",
        **overrides,
    }
    return PlotSpec(family="section", items=[item])


def _bars(fig):
    """Every :class:`~matplotlib.colorbar.Colorbar` in ``fig``, in axes order."""
    return [ax._colorbar for ax in fig.axes if ax.get_label() == "<colorbar>"]


def _bar(fig):
    (bar,) = _bars(fig)
    return bar


def _axis(cbar):
    """Return the axis a bar's ticks are on: x for a horizontal bar, else y."""
    return cbar.ax.xaxis if cbar.orientation == "horizontal" else cbar.ax.yaxis


def _labels(cbar) -> list[str]:
    """Return the tick labels a bar draws (drawn first: they settle at draw time)."""
    cbar.ax.figure.canvas.draw()
    return [t.get_text() for t in _axis(cbar).get_ticklabels()]


def _minor_labels(cbar) -> list[str]:
    cbar.ax.figure.canvas.draw()
    return [t.get_text() for t in _axis(cbar).get_ticklabels(minor=True)]


def _texts(cbar) -> list[str]:
    """Annotations drawn on a bar's own axes (the ``max 100`` end labels)."""
    return [t.get_text() for t in cbar.ax.texts]


# --- the bars a figure draws ----------------------------------------------------------


@pytest.mark.parametrize("mark", ["contourf", "pcolormesh"])
def test_a_bar_ends_on_round_numbers_and_ticks_round_ones(mark):
    bar = _bar(render(_section(), mark=mark))
    # the data runs 0.0213 to 2.987: the limits snap outward, the labels read round
    assert (bar.norm.vmin, bar.norm.vmax) == (0.0, 3.0)
    assert _labels(bar) == ROUND


def test_a_contourf_bar_no_longer_ticks_its_band_edges():
    # 21 levels over 0.0213-2.987 put the ticks at 0.021, 0.466, 0.911, ...; the
    # snapped range is 0-3, so the edges are every 0.15 and the ticks are every 0.5
    bar = _bar(render(_section(), mark="contourf"))
    assert not any(label in _labels(bar) for label in ("0.466", "0.911", "0.021"))
    values = [float(label) for label in _labels(bar)]
    assert values == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]


def test_a_horizontal_bar_is_ticked_the_same_way():
    fig = render(_section(), colorbar_kwargs={"orientation": "horizontal"})
    bar = _bar(fig)
    assert bar.orientation == "horizontal"
    assert _labels(bar) == ROUND
    assert _minor_labels(bar) == [""] * len(_minor_labels(bar))


def test_minor_tick_labels_are_off():
    bar = _bar(render(_section(), mark="contourf"))
    assert isinstance(_axis(bar).get_minor_formatter(), NullFormatter)
    assert not any(_minor_labels(bar))


def test_a_negative_label_carries_a_unicode_minus_not_a_hyphen():
    field = _grid(np.linspace(-1.7, 1.7, 48))
    labels = _labels(_bar(render(_section(field))))
    assert labels[0].startswith(MINUS) and "-" not in "".join(labels)
    assert labels == [
        f"{MINUS}1.5",
        f"{MINUS}1.0",
        f"{MINUS}0.5",
        "0.0",
        "0.5",
        "1.0",
        "1.5",
    ]


def test_tick_kwargs_still_style_the_round_labels():
    fig = render(_section(), colorbar_kwargs={"tick_labelsize": 13})
    bar = _bar(fig)
    assert {t.get_fontsize() for t in _axis(bar).get_ticklabels()} == {13}
    assert _labels(bar) == ROUND


@pytest.mark.parametrize("mark", ["contourf", "pcolormesh"])
def test_a_log_bar_reads_in_plain_decimals(mark):
    rng = np.random.default_rng(0)
    field = _grid(10 ** rng.uniform(-1.7, 0.9, 48))  # 0.02 to 8 mg m-3
    fig = render(_section(field, standard_name=CHLOROPHYLL), mark=mark)
    bar = _bar(fig)
    assert isinstance(bar.norm, mcolors.LogNorm)
    # chlorophyll declares its own display range, which is never moved
    assert (bar.norm.vmin, bar.norm.vmax) == (0.01, 10.0)
    assert _labels(bar) == ["0.01", "0.1", "1", "10"]
    assert not any(_minor_labels(bar))  # marks may stay on a log bar, their numbers go


def test_a_narrow_log_bar_ticks_one_two_five_not_scientific_notation():
    field = _grid(np.geomspace(0.05, 0.5, 48))
    fig = render(_section(field, standard_name=CHLOROPHYLL), vmin=0.05, vmax=0.5)
    assert _labels(_bar(fig)) == ["0.05", "0.1", "0.2", "0.5"]


# --- a difference panel ---------------------------------------------------------------


def _row_item(difference) -> dict:
    reference = _ramp()
    return {
        "aligned": {
            "test": reference + difference,
            "reference": reference,
            "difference": difference,
        },
        "units": "degC",
        "standard_name": None,
        "depth": "0-200 m",
        "time": "2012-01",
        "metrics": {"bias": 0.125, "rmse": 0.5, "corr": 0.98},
        "labels": ("model", "obs"),
    }


def test_a_difference_bar_is_symmetric_and_round():
    diff = _grid(np.linspace(-1.7, 1.7, 48))
    fig = render(PlotSpec(family="section_row", items=[_row_item(diff)]))
    # the scale bar spans test and reference; the difference bar spans one panel
    (diff_bar,) = [b for b in _bars(fig) if len(b.ax._osk_cbar_parents) == 1]
    assert (diff_bar.norm.vmin, diff_bar.norm.vmax) == (-1.7, 1.7)
    assert diff_bar.norm.vmax == difference_limit(diff)
    assert _labels(diff_bar) == [
        f"{MINUS}1.5",
        f"{MINUS}1.0",
        f"{MINUS}0.5",
        "0.0",
        "0.5",
        "1.0",
        "1.5",
    ]


def test_the_shared_difference_range_is_the_same_rounded_one():
    diff = _grid(np.linspace(-1.7, 1.7, 48))
    item = _row_item(diff)
    seq, div = mr._shared_norms(
        [{"aligned": item["aligned"], "standard_name": None}] * 2, "test", "reference"
    )
    assert (div.vmin, div.vmax) == (-1.7, 1.7)
    # test and reference pooled, then snapped: the same rule as a single row's scale
    both = (item["aligned"]["test"], item["aligned"]["reference"])
    pooled = np.concatenate([np.ravel(a) for a in both])
    assert (seq.vmin, seq.vmax) == _limits(pooled, log=False)


def test_a_map_row_rounds_its_scale_bar_and_its_difference_bar():
    lat, lon = np.linspace(18.0, 26.0, 12), np.linspace(-100.0, -90.0, 14)

    def over_map(values):
        return xr.DataArray(
            np.reshape(values, (12, 14)),
            dims=("lat", "lon"),
            coords={"lat": lat, "lon": lon},
        )

    reference = over_map(np.linspace(0.0213, 2.987, 12 * 14))
    difference = over_map(np.linspace(-1.7, 1.7, 12 * 14))
    item = {
        "aligned": {
            "test": reference + difference,
            "reference": reference,
            "difference": difference,
        },
        "units": "degC",
        "standard_name": None,
        "labels": ("model", "obs"),
    }
    fig = render(PlotSpec(family="field_row", items=[item], options={}))
    scale_bar, diff_bar = _bars(fig)
    pooled = [item["aligned"]["test"], item["aligned"]["reference"]]
    assert (scale_bar.norm.vmin, scale_bar.norm.vmax) == _limits(*pooled, log=False)
    # snapped outward, so the data (-1.679 to 4.687) is all inside: nothing is clipped
    assert scale_bar.norm.vmin <= min(float(a.min()) for a in pooled)
    assert scale_bar.norm.vmax >= max(float(a.max()) for a in pooled)
    assert _labels(scale_bar) == [f"{MINUS}1", "0", "1", "2", "3", "4"]
    assert (diff_bar.norm.vmin, diff_bar.norm.vmax) == (-1.7, 1.7)
    assert _labels(diff_bar)[0] == f"{MINUS}1.5" and _labels(diff_bar)[-1] == "1.5"


@pytest.mark.parametrize("difference", [np.zeros(48), np.full(48, np.nan)])
def test_a_difference_with_nothing_to_span_still_gets_a_range(difference):
    diff = _grid(difference)
    fig = render(PlotSpec(family="section_row", items=[_row_item(diff)]))
    (diff_bar,) = [b for b in _bars(fig) if len(b.ax._osk_cbar_parents) == 1]
    assert (diff_bar.norm.vmin, diff_bar.norm.vmax) == (-1.0, 1.0)


# --- _limits --------------------------------------------------------------------------

DATA = np.array([0.0213, 1.0, 2.987])


def test_limits_snap_the_data_range_outward():
    lo, hi = _limits(DATA, log=False)
    assert (lo, hi) == (0.0, 3.0)
    assert lo <= DATA.min() and hi >= DATA.max()  # nothing is clipped by the snap


def test_limits_snap_a_robust_range_outward_too():
    vals = np.linspace(0.0213, 2.987, 1001)
    raw = (np.percentile(vals, 10), np.percentile(vals, 90))  # 0.318, 2.691
    lo, hi = _limits(vals, log=False, robust=True)
    assert (lo, hi) == pytest.approx((0.3, 2.7))
    assert lo <= raw[0] and hi >= raw[1]


def test_log_limits_round_to_one_significant_digit():
    assert _limits(np.array([0.0213, 2.987]), log=True) == (0.02, 3.0)


def test_a_pinned_vmin_is_kept_exactly_and_only_the_other_end_snaps():
    lo, hi = _limits(DATA, log=False, vmin=0.37)
    assert lo == 0.37 and hi == 3.0


def test_a_pinned_vmax_is_kept_exactly_and_only_the_other_end_snaps():
    lo, hi = _limits(DATA, log=False, vmax=2.71)
    assert lo == 0.0 and hi == 2.71


def test_two_pinned_ends_are_returned_untouched_whatever_the_data():
    assert _limits(DATA, log=False, vmin=0.0213, vmax=2.987) == (0.0213, 2.987)
    # the caller's own number, not a float() of it
    lo, hi = _limits(DATA, log=False, vmin=np.float32(0.1), vmax=3)
    assert type(lo) is np.float32 and type(hi) is int


def test_snap_false_returns_the_raw_range():
    assert _limits(DATA, log=False, snap=False) == (0.0213, 2.987)
    assert _limits(DATA, log=False, snap=False, vmin=0.37) == (0.37, 2.987)


def test_an_empty_range_is_not_snapped():
    assert _limits(np.array([]), log=False) == (0.0, 1.0)
    assert _limits(np.full(3, np.nan), log=True) == (0.0, 1.0)
    # a pinned end beside nothing to measure stays exactly where it was put
    assert _limits(np.array([np.nan]), log=False, vmin=0.37) == (0.37, 1.0)


def test_log_has_to_be_said():
    with pytest.raises(TypeError, match="log"):
        _limits(DATA)


def test_a_pinned_figure_keeps_its_pins_and_snaps_the_rest():
    fig = render(_section(), vmin=0.37)
    bar = _bar(fig)
    assert (bar.norm.vmin, bar.norm.vmax) == (0.37, 3.0)
    # the bar starts at the pin, so its first tick is the first round value above it
    assert _labels(bar) == ROUND[1:]


# --- what the caller chose wins -------------------------------------------------------


def test_a_users_own_ticks_are_respected():
    fig = render(_section(), colorbar_kwargs={"ticks": [0.25, 1.0, 2.75]})
    bar = _bar(fig)
    assert list(bar.get_ticks()) == [0.25, 1.0, 2.75]
    assert not any(label in _labels(bar) for label in ("0.5", "1.5"))


def test_a_users_own_format_is_respected():
    fig = render(_section(), colorbar_kwargs={"format": "%.2f"})
    labels = _labels(_bar(fig))
    assert labels and all(len(label.split(".")[1]) == 2 for label in labels)


def test_a_users_ticks_of_none_mean_no_choice():
    fig = render(_section(), colorbar_kwargs={"ticks": None})
    assert _labels(_bar(fig)) == ROUND


@pytest.mark.parametrize(
    "norm",
    [
        mcolors.BoundaryNorm([0.0, 1.0, 5.0, 10.0], 256),
        mcolors.NoNorm(),
        mcolors.PowerNorm(0.5, vmin=0.0, vmax=10.0),
        mcolors.SymLogNorm(1.0, vmin=-10.0, vmax=10.0),
    ],
    ids=lambda n: type(n).__name__,
)
def test_a_norm_that_is_not_a_plain_scale_keeps_its_own_ticks(norm, monkeypatch):
    def refused(*args, **kwargs):  # pragma: no cover - only reached on a regression
        raise AssertionError("round ticks were applied to a non-scale norm")

    monkeypatch.setattr(mr, "_round_ticks", refused)
    fig, ax = plt.subplots()
    im = ax.pcolormesh(np.arange(12.0).reshape(3, 4), cmap="viridis", norm=norm)
    cbar = _draw_colorbar(fig, im, ax, "x", None, {})
    assert cbar is not None
    plt.close(fig)


def test_a_boundary_bar_ticks_its_boundaries():
    fig, ax = plt.subplots()
    norm = mcolors.BoundaryNorm([0.0, 1.0, 5.0, 10.0], 256)
    im = ax.pcolormesh(np.arange(12.0).reshape(3, 4), cmap="viridis", norm=norm)
    cbar = _draw_colorbar(fig, im, ax, "x", None, {})
    fig.canvas.draw()
    assert list(cbar.get_ticks()) == [0.0, 1.0, 5.0, 10.0]
    plt.close(fig)


@pytest.mark.parametrize(
    ("norm", "expected"),
    [
        (mcolors.Normalize(0.0, 1.0), True),
        (mcolors.LogNorm(0.01, 10.0), True),
        (mcolors.TwoSlopeNorm(0.0, -1.0, 1.0), True),
        (mcolors.CenteredNorm(0.0, 1.0), True),
        (mcolors.BoundaryNorm([0.0, 1.0, 2.0], 4), False),
        (mcolors.NoNorm(), False),
        (mcolors.PowerNorm(0.5), False),
        (mcolors.SymLogNorm(1.0), False),
    ],
    ids=lambda v: type(v).__name__ if not isinstance(v, bool) else str(v),
)
def test_only_a_plain_scale_gets_round_ticks(norm, expected):
    assert mr._takes_round_ticks(norm) is expected


def test_a_range_with_nothing_to_tick_leaves_the_bars_own_ticks(monkeypatch):
    monkeypatch.setattr(mr, "colorbar_ticks", lambda *args, **kwargs: Ticks((), ()))
    fig, ax = plt.subplots()
    im = ax.pcolormesh(np.arange(12.0).reshape(3, 4), cmap="viridis")
    cbar = _draw_colorbar(fig, im, ax, "x", None, {})
    assert not isinstance(_axis(cbar).get_major_locator(), FixedLocator)
    plt.close(fig)


# --- the datetime bar -----------------------------------------------------------------


def test_round_ticks_can_be_declined_by_a_caller_that_ticks_the_bar_itself():
    def ticked(round_ticks):
        fig, ax = plt.subplots()
        im = ax.pcolormesh(
            np.arange(12.0).reshape(3, 4),
            cmap="viridis",
            norm=mcolors.Normalize(0.2, 9.7),
        )
        cbar = _draw_colorbar(fig, im, ax, "x", None, {}, round_ticks=round_ticks)
        out = isinstance(_axis(cbar).get_major_locator(), FixedLocator)
        plt.close(fig)
        return out

    assert ticked(True) and not ticked(False)


def test_a_time_bar_is_left_to_its_own_date_ticks(monkeypatch):
    import matplotlib.dates as mdates

    from tests.test_xy_renderers import _colorbars, _draw, _ts_items

    seen = []
    real = mr._round_ticks
    monkeypatch.setattr(
        mr, "_round_ticks", lambda cbar, norm: seen.append(norm) or real(cbar, norm)
    )

    _draw(_ts_items(), color_by="depth")
    assert len(seen) == 1  # the control: a depth bar is ticked by value ...
    seen.clear()

    fig = _draw(_ts_items(), color_by="time")
    (bar,) = _colorbars(fig)
    assert seen == []  # ... a time bar never is: it is days since 1970, read as dates
    assert isinstance(bar.yaxis.get_major_locator(), mdates.AutoDateLocator)
    assert isinstance(bar.yaxis.get_major_formatter(), mdates.ConciseDateFormatter)
    labels = [t.get_text() for t in bar.get_yticklabels()]
    assert any(labels)
    # not raw day counts such as 18262.5
    assert not any(t.replace(".", "").isdigit() and len(t) > 4 for t in labels)


def test_a_depth_bar_is_ticked_round_like_any_other():
    from tests.test_xy_renderers import _colorbars, _draw, _ts_items

    fig = _draw(_ts_items(), color_by="depth")
    (bar,) = _colorbars(fig)
    # ticked by value, not left to the default locator
    assert isinstance(_axis(bar._colorbar).get_major_locator(), FixedLocator)
    values = [float(label.replace(MINUS, "-")) for label in _labels(bar._colorbar)]
    assert values and all(v == round(v) for v in values)
    assert bar.yaxis_inverted()  # shallow water still at the top


# --- extension arrows and their labels ------------------------------------------------


@pytest.mark.parametrize("mark", ["contourf", "pcolormesh"])
def test_label_clipped_still_writes_the_true_extreme_at_the_arrow(mark):
    values = np.full(48, 5.0)
    values[0] = 100.0
    fig = render(
        _section(_grid(values)), vmax=20.0, mark=mark, colorbar_label_clipped=True
    )
    bar = _bar(fig)
    assert bar.extend == "max"
    assert _texts(bar) == ["max 100"]
    # ... and the bar still ticks round, from the pinned 20 down to the data's own end
    assert bar.norm.vmax == 20.0 and _labels(bar)[-1] == "20"


def test_no_end_label_unless_asked_for():
    values = np.full(48, 5.0)
    values[0] = 100.0
    bar = _bar(render(_section(_grid(values)), vmax=20.0))
    assert bar.extend == "max" and _texts(bar) == []
