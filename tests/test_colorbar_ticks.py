"""Round colour-bar ticks and limits (ocean_skill.plot._colorbar)."""

from __future__ import annotations

import numpy as np
import pytest

from ocean_skill.plot._colorbar import (
    _snap_step,
    colorbar_ticks,
    difference_limit,
    round_limits,
    tick_step,
)

MINUS = "\N{MINUS SIGN}"


@pytest.mark.parametrize(
    ("lo", "hi", "expected"),
    [
        (0.0213, 2.987, (0.0, 3.0)),  # phosphate: the paper's 0-3 bar
        (33.81, 36.42, (33.8, 36.5)),  # salinity
        (4.3, 29.87, (4.0, 30.0)),  # temperature
        (2201.3, 2398.7, (2200.0, 2400.0)),  # alkalinity
        (-1.734, 1.734, (-1.8, 1.8)),
        # ticked every 2: snapped to 0.2, a round step -- not to 0.4 (25.2, 4.8)
        (10.0, 24.87, (10.0, 25.0)),
        (5.0, 20.0, (5.0, 20.0)),
        (14.2, 28.9, (14.2, 29.0)),
    ],
)
def test_round_limits_snaps_outward_to_a_fifth_of_the_tick_step(lo, hi, expected):
    assert round_limits(lo, hi, log=False) == pytest.approx(expected)
    new_lo, new_hi = round_limits(lo, hi, log=False)
    assert new_lo <= lo and new_hi >= hi  # outward only: nothing is clipped


def test_round_limits_leaves_already_round_ends_alone():
    assert round_limits(0.0, 3.0, log=False) == (0.0, 3.0)
    assert round_limits(0.01, 10.0, log=True) == (0.01, 10.0)


def test_round_limits_keeps_pinned_ends_exactly():
    assert round_limits(0.37, 2.987, log=False, keep_lo=True) == (0.37, 3.0)
    assert round_limits(0.0213, 2.71, log=False, keep_hi=True) == (0.0, 2.71)
    both = round_limits(0.37, 2.71, log=False, keep_lo=True, keep_hi=True)
    assert both == (0.37, 2.71)


def test_round_limits_on_a_log_scale_never_reaches_zero():
    lo, hi = round_limits(0.0213, 2.987, log=True)
    assert (lo, hi) == pytest.approx((0.02, 3.0))
    assert round_limits(0.00731, 0.0491, log=True) == pytest.approx((0.007, 0.05))
    # a non-positive lower end cannot be snapped on a log scale -- left as given --
    # but the top still reads round
    assert round_limits(0.0, 5.0, log=True) == (0.0, 5.0)
    assert round_limits(0.0, 8.7, log=True) == (0.0, 9.0)


@pytest.mark.parametrize("bad", [(np.nan, 1.0), (1.0, np.inf), (2.0, 2.0), (3.0, 1.0)])
def test_round_limits_passes_through_a_range_with_nothing_to_snap(bad):
    out = round_limits(*bad, log=False)
    assert out == bad or (np.isnan(out[0]) and np.isnan(bad[0]))


@pytest.mark.parametrize(
    ("lo", "hi", "labels"),
    [
        (0.0, 3.0, ["0.0", "0.5", "1.0", "1.5", "2.0", "2.5", "3.0"]),
        (33.8, 36.5, ["34.0", "34.5", "35.0", "35.5", "36.0", "36.5"]),
        (4.0, 30.0, ["5", "10", "15", "20", "25", "30"]),
        (0.0, 52.0, ["0", "10", "20", "30", "40", "50"]),
        (2200.0, 2400.0, ["2200", "2250", "2300", "2350", "2400"]),
        (
            -1.8,
            1.8,
            [f"{MINUS}1.5", f"{MINUS}1.0", f"{MINUS}0.5", "0.0", "0.5", "1.0", "1.5"],
        ),
    ],
)
def test_linear_ticks_are_round_and_share_their_decimals(lo, hi, labels):
    ticks = colorbar_ticks(lo, hi, log=False)
    assert list(ticks.labels) == labels
    assert all(lo <= v <= hi for v in ticks.values)
    assert len(ticks.values) == len(ticks.labels)


def test_linear_ticks_never_spell_a_negative_zero_or_float_noise():
    ticks = colorbar_ticks(-0.3, 0.3, log=False)
    assert "0.0" in ticks.labels
    assert not any(label.startswith(MINUS + "0.0") for label in ticks.labels)
    assert all(len(label) <= 5 for label in colorbar_ticks(0.0, 0.7, log=False).labels)


@pytest.mark.parametrize(
    ("lo", "hi", "labels"),
    [
        (0.01, 10.0, ["0.01", "0.1", "1", "10"]),
        (0.05, 0.5, ["0.05", "0.1", "0.2", "0.5"]),
        (0.02, 3.0, ["0.02", "0.05", "0.1", "0.2", "0.5", "1", "2"]),
        (0.3, 0.5, ["0.3", "0.35", "0.4", "0.45", "0.5"]),
    ],
)
def test_log_ticks_are_plain_decimals(lo, hi, labels):
    assert list(colorbar_ticks(lo, hi, log=True).labels) == labels


def test_a_wide_log_bar_thins_its_decades():
    ticks = colorbar_ticks(1e-6, 1e6, log=True)
    assert 2 <= len(ticks.values) <= 7
    assert all(np.log10(v) == round(np.log10(v)) for v in ticks.values)


def test_ticks_are_empty_for_a_range_with_nothing_to_tick():
    assert colorbar_ticks(5.0, 5.0, log=False).values == ()
    assert colorbar_ticks(np.nan, 1.0, log=False).values == ()


def test_text_spells_an_end_like_the_ticks_but_keeps_digits_it_needs():
    ticks = colorbar_ticks(4.2, 30.0, log=False)
    assert ticks.text(30.0) == "30"
    assert ticks.text(4.2) == "4.2"  # not rounded to a number the bar does not end at
    assert colorbar_ticks(0.0, 3.0, log=False).text(3.0) == "3.0"
    assert colorbar_ticks(-1.8, 1.8, log=False).text(-1.8) == f"{MINUS}1.8"
    assert colorbar_ticks(0.02, 3.0, log=True).text(3.0) == "3"


def test_difference_limit_is_the_98th_percentile_snapped_up():
    d = np.linspace(-1.734, 1.734, 2001)
    raw = float(np.percentile(np.abs(d), 98))
    out = difference_limit(d)
    assert out >= raw
    assert out == pytest.approx(round_limits(-raw, raw, log=False)[1])


@pytest.mark.parametrize("d", [[np.nan, np.nan], [0.0, 0.0], []])
def test_difference_limit_falls_back_to_one(d):
    assert difference_limit(np.asarray(d, dtype=float)) == 1.0


def test_snapped_ends_are_round_whatever_the_tick_step():
    """Each end is a whole multiple of a 1-2-5 step -- never of 0.4 or 4."""
    for lo, hi in [(10.0, 24.87), (1.0, 9.93), (0.0213, 2.987), (113.0, 487.0)]:
        snap = _snap_step(tick_step(lo, hi))
        mantissa = snap / 10 ** np.floor(np.log10(snap))
        assert round(mantissa, 9) in (1.0, 2.0, 5.0)
        for end in round_limits(lo, hi, log=False):
            assert end / snap == pytest.approx(round(end / snap))


def test_tick_step_is_one_two_or_five_times_a_power_of_ten():
    for lo, hi in [(0, 3), (33.8, 36.5), (-1.8, 1.8), (0, 0.007), (0, 5000)]:
        step = tick_step(lo, hi)
        mantissa = step / 10 ** np.floor(np.log10(step))
        assert round(mantissa, 9) in (1.0, 2.0, 5.0)
