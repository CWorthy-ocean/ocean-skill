"""``ocean_skill.align.subset_to_time``/``subset_to_time_targets`` on an
irregular test time axis.

Both are crops along a test lane's time axis derived from a *different*
source's own record (see ``Comparison._reference_narrowing`` /
``Comparison._reference_time_targets``, which narrow the test lane to a
repeat-visit reference's catalog coverage or cast times), so both routinely
receive bounds/targets that land oddly relative to a test time axis that is
not a plain, strictly sorted ``datetime64`` index -- two source files
concatenated without a clean sort/dedupe is a plausible ROMS multi-file
artifact. ``subset_to_time``'s old label ``.sel`` slice ``KeyError``ed in that
shape instead of clipping to the overlap; ``subset_to_time_targets``'s
``pandas``-nearest lookup separately ``ValueError``ed, since it requires a
sorted index. These tests pin the crop's actual behavior in each shape, so a
regression here is caught directly rather than only through the much larger
``compare()`` integration path (see ``tests/test_station_record_past_model.py``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ocean_skill.align import subset_to_time, subset_to_time_targets


def _dataset(times) -> xr.Dataset:
    times = pd.to_datetime(times)
    return xr.Dataset({"x": ("time", np.arange(len(times)))}, coords={"time": times})


def test_a_window_inside_the_record_crops_normally():
    ds = _dataset(pd.date_range("2024-01-01", "2024-12-31", freq="7D"))
    out = subset_to_time(ds, (pd.Timestamp("2024-04-01"), pd.Timestamp("2024-06-01")))
    assert out.time.values.min() >= np.datetime64("2024-04-01")
    assert out.time.values.max() <= np.datetime64("2024-06-01")
    assert out.sizes["time"] < ds.sizes["time"]


def test_an_out_of_range_upper_bound_clips_to_the_records_own_end():
    """A reference's declared coverage can run a year past the test lane's own
    record -- see the real HV1 station this reproduces (2025-04-28 declared,
    against a model ending 2024-11-29)."""
    ds = _dataset(pd.date_range("2024-02-01", "2024-11-29", freq="7D"))
    out = subset_to_time(ds, (pd.Timestamp("2024-04-04"), pd.Timestamp("2025-04-29")))
    assert out.sizes["time"] > 0
    assert out.time.values.max() == ds.time.values.max()


def test_a_non_monotonic_axis_with_an_out_of_range_bound_does_not_raise():
    """The exact shape that used to ``KeyError`` under a label ``.sel`` slice:
    an out-of-order test time axis (a plausible multi-file-concat artifact)
    paired with a window bound past its end."""
    ds = _dataset(["2024-01-01", "2024-06-01", "2024-03-01", "2024-11-29", "2024-11-29"])
    # Confirm this fixture actually reproduces the old crash shape: a label
    # slice against it raises exactly the KeyError this feature exists to avoid.
    with pytest.raises(KeyError):
        ds.sel(time=slice(pd.Timestamp("2024-04-04"), pd.Timestamp("2025-04-29")))

    out = subset_to_time(ds, (pd.Timestamp("2024-04-04"), pd.Timestamp("2025-04-29")))
    assert sorted(pd.Timestamp(t) for t in out.time.values) == [
        pd.Timestamp("2024-06-01"),
        pd.Timestamp("2024-11-29"),
        pd.Timestamp("2024-11-29"),
    ]


def test_no_overlap_falls_back_to_the_unchanged_object():
    """A climatology or static field sharing no calendar span with the test is
    not an error here -- the comparison's own time handling reports that far
    more precisely than an empty crop could."""
    ds = _dataset(pd.date_range("2024-01-01", "2024-03-01", freq="7D"))
    out = subset_to_time(ds, (pd.Timestamp("2030-01-01"), pd.Timestamp("2030-02-01")))
    assert out.sizes["time"] == ds.sizes["time"]


def test_window_none_or_no_time_dim_is_a_no_op():
    ds = _dataset(pd.date_range("2024-01-01", "2024-03-01", freq="7D"))
    assert subset_to_time(ds, None) is ds
    static = xr.Dataset({"x": (("lat",), [1.0, 2.0])}, coords={"lat": [0.0, 1.0]})
    assert subset_to_time(static, (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01"))) is static


def test_a_descending_time_axis_crops_correctly():
    ds = _dataset(pd.date_range("2024-01-01", "2024-12-31", freq="7D")[::-1])
    out = subset_to_time(ds, (pd.Timestamp("2024-04-01"), pd.Timestamp("2024-06-01")))
    assert out.time.values.min() >= np.datetime64("2024-04-01")
    assert out.time.values.max() <= np.datetime64("2024-06-01")


def test_reversed_window_bounds_are_treated_as_a_range_either_way():
    ds = _dataset(pd.date_range("2024-01-01", "2024-12-31", freq="7D"))
    forwards = subset_to_time(ds, (pd.Timestamp("2024-04-01"), pd.Timestamp("2024-06-01")))
    backwards = subset_to_time(ds, (pd.Timestamp("2024-06-01"), pd.Timestamp("2024-04-01")))
    assert forwards.time.values.tolist() == backwards.time.values.tolist()


# -- subset_to_time_targets: the discrete, cast-nearest counterpart ------------


def test_targets_prune_to_the_nearest_step_on_a_monotonic_axis():
    ds = _dataset(pd.date_range("2024-01-01", "2024-12-31", freq="7D"))
    targets = np.array([np.datetime64("2024-04-03"), np.datetime64("2024-09-01")])
    out = subset_to_time_targets(ds, targets)
    assert out.sizes["time"] == 2


def test_targets_on_a_non_monotonic_axis_does_not_raise_and_matches_by_value():
    """The shape that used to raise ``ValueError: index must be monotonic
    increasing or decreasing`` out of pandas' own nearest-step lookup --
    exactly what stopped the fix to ``subset_to_time`` alone from being
    enough (see ``tests/test_station_record_past_model.py``)."""
    ds = _dataset(["2024-02-01", "2024-07-10", "2024-04-20", "2024-10-08", "2024-11-29"])
    with pytest.raises(ValueError, match="monotonic"):
        pd.Index(ds.time.values).get_indexer([np.datetime64("2024-04-20")], method="nearest")

    targets = np.array([np.datetime64("2024-04-21"), np.datetime64("2024-10-09")])
    out = subset_to_time_targets(ds, targets)
    assert sorted(pd.Timestamp(t) for t in out.time.values) == [
        pd.Timestamp("2024-04-20"),
        pd.Timestamp("2024-10-08"),
    ]


def test_targets_interp_on_a_non_monotonic_axis_brackets_by_value():
    """Interpolated onto the target instant itself, bracketed by the two
    nearest real steps *by value* (2024-04-20 and 2024-07-10) -- found
    correctly despite those two not being adjacent in the axis' stored,
    non-monotonic order."""
    ds = _dataset(["2024-02-01", "2024-07-10", "2024-04-20", "2024-10-08", "2024-11-29"])
    out = subset_to_time_targets(
        ds, np.array([np.datetime64("2024-05-01")]), method="interp"
    )
    assert out.sizes["time"] == 1
    assert pd.Timestamp(out.time.values[0]) == pd.Timestamp("2024-05-01")
    # Linear between x=0 (04-20) and x=4 (07-10) at 11 days in: 0 + 11/81*4.
    lo, hi = ds.sel(time="2024-04-20")["x"].item(), ds.sel(time="2024-07-10")["x"].item()
    frac = (pd.Timestamp("2024-05-01") - pd.Timestamp("2024-04-20")) / (
        pd.Timestamp("2024-07-10") - pd.Timestamp("2024-04-20")
    )
    assert out["x"].item() == pytest.approx(lo + frac * (hi - lo))


def test_targets_none_or_empty_or_no_time_dim_is_a_no_op():
    ds = _dataset(pd.date_range("2024-01-01", "2024-03-01", freq="7D"))
    assert subset_to_time_targets(ds, None) is ds
    assert subset_to_time_targets(ds, []) is ds
    static = xr.Dataset({"x": (("lat",), [1.0, 2.0])}, coords={"lat": [0.0, 1.0]})
    assert subset_to_time_targets(static, [np.datetime64("2024-01-01")]) is static
