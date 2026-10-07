"""``subset_to_time`` selects a contiguous run of steps with a slice, not a mask.

The crop is derived from a boolean value mask (see its docstring for why: it answers
"which steps fall in the window" for any axis shape, and never raises). But on a long
*lazy* dataset -- hundreds of thousands of single-step chunks, a year of hourly ROMS
history files opened as one dataset -- a fancy-index (mask / integer array) selection is
much more expensive to build than a slice: xarray turns the mask into an integer array
and dask builds a ``take`` over every chunk on the axis, where a slice only has to find
the few chunks it keeps. A sorted axis and a ``[lo, hi]`` window always keep one
contiguous run, so that run is selected with ``slice(i0, i1 + 1)``; any other kept set
(a gappy selection on an unsorted axis) keeps the mask. Results are identical either
way -- these tests pin both halves.
"""

from __future__ import annotations

import dask
import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ocean_skill.align import subset_to_time


def _dataset(times) -> xr.Dataset:
    times = pd.to_datetime(times)
    return xr.Dataset({"x": ("time", np.arange(len(times)))}, coords={"time": times})


def _by_mask(obj, window):
    """Select by the pre-slice route, spelled out: a boolean value mask via ``isel``."""
    values = obj["time"].values
    lo, hi = (np.datetime64(b) if b is not None else None for b in window)
    mask = np.ones(values.shape, dtype=bool)
    if lo is not None:
        mask &= values >= lo
    if hi is not None:
        mask &= values <= hi
    return obj.isel(time=mask)


@pytest.fixture
def time_indexers(monkeypatch):
    """Record the ``time`` indexer of every ``isel`` call that reaches xarray."""
    seen: list = []

    def wrap(cls):
        original = cls.isel

        def spy(self, indexers=None, *args, **kwargs):
            merged = dict(indexers or {}, **kwargs)
            if "time" in merged:
                seen.append(merged["time"])
            return original(self, indexers, *args, **kwargs)

        monkeypatch.setattr(cls, "isel", spy)

    wrap(xr.Dataset)
    wrap(xr.DataArray)
    return seen


# -- the selection is a slice -------------------------------------------------------


def test_a_window_on_a_sorted_axis_is_selected_with_a_slice(time_indexers):
    ds = _dataset(pd.date_range("2024-01-01", periods=100, freq="D"))
    out = subset_to_time(ds, (pd.Timestamp("2024-02-01"), pd.Timestamp("2024-02-10")))
    assert out.sizes["time"] == 10
    assert len(time_indexers) == 1
    assert isinstance(time_indexers[0], slice)
    assert (time_indexers[0].start, time_indexers[0].stop) == (31, 41)


def test_a_dataarray_is_sliced_too(time_indexers):
    da_ = _dataset(pd.date_range("2024-01-01", periods=100, freq="D"))["x"]
    out = subset_to_time(da_, (pd.Timestamp("2024-02-01"), pd.Timestamp("2024-02-10")))
    assert out.sizes["time"] == 10
    assert len(time_indexers) == 1 and isinstance(time_indexers[0], slice)


def test_a_contiguous_run_on_an_unsorted_axis_is_a_slice_as_well(time_indexers):
    """The rule is about the kept indices, not about the axis being sorted."""
    ds = _dataset(["2024-01-01", "2024-06-01", "2024-07-01", "2024-03-01"])
    out = subset_to_time(ds, (pd.Timestamp("2024-05-01"), pd.Timestamp("2024-08-01")))
    assert out["x"].values.tolist() == [1, 2]
    assert len(time_indexers) == 1 and isinstance(time_indexers[0], slice)


def test_a_dask_backed_dataset_with_thousands_of_one_step_chunks(time_indexers):
    n = 2000
    times = pd.date_range("2000-01-01", periods=n, freq="h")
    data = np.arange(n * 4, dtype="float64").reshape(n, 4)
    ds = xr.Dataset(
        {"x": (("time", "s"), da.from_array(data, chunks=(1, 4)))},
        coords={"time": times},
    )
    assert ds["x"].data.numblocks[0] == n

    out = subset_to_time(ds, (times[1000], times[1002]))
    used = list(time_indexers)  # before _by_mask below makes isel calls of its own

    assert len(used) == 1 and isinstance(used[0], slice)
    assert out["x"].data.numblocks == (3, 1)  # still lazy, still one chunk per step
    assert dask.is_dask_collection(out["x"])
    np.testing.assert_array_equal(out["x"].values, data[1000:1003])
    xr.testing.assert_identical(
        out.compute(), _by_mask(ds, (times[1000], times[1002])).compute()
    )


# -- identical to the mask ----------------------------------------------------------


_AXIS = pd.date_range("2024-01-01", periods=60, freq="D")


@pytest.mark.parametrize(
    "window",
    [
        pytest.param((_AXIS[10], _AXIS[20]), id="interior"),
        pytest.param((_AXIS[10], _AXIS[10]), id="one-step"),
        pytest.param((_AXIS[0], _AXIS[-1]), id="whole-axis"),
        pytest.param((pd.Timestamp("2023-06-01"), _AXIS[5]), id="overhangs-the-start"),
        pytest.param((_AXIS[50], pd.Timestamp("2025-06-01")), id="overhangs-the-end"),
        pytest.param(
            (pd.Timestamp("2023-01-01"), pd.Timestamp("2026-01-01")), id="swallows-it"
        ),
        pytest.param(
            (_AXIS[10] + pd.Timedelta("1h"), _AXIS[12] - pd.Timedelta("1h")),
            id="between-steps",
        ),
        pytest.param((_AXIS[20], _AXIS[10]), id="written-high-to-low"),
        pytest.param((None, _AXIS[10]), id="open-below"),
        pytest.param((_AXIS[50], None), id="open-above"),
    ],
)
@pytest.mark.parametrize("descending", [False, True], ids=["ascending", "descending"])
def test_the_slice_gives_exactly_the_mask_result(window, descending):
    ds = _dataset(_AXIS[::-1] if descending else _AXIS)
    ds = ds.assign(y=("time", np.linspace(0, 1, ds.sizes["time"])))
    lo, hi = window
    if lo is not None and hi is not None and lo > hi:
        lo, hi = hi, lo
    expected = _by_mask(ds, (lo, hi))
    if expected.sizes["time"] == 0:  # nothing kept -> subset_to_time hands obj back
        expected = ds
    xr.testing.assert_identical(subset_to_time(ds, window), expected)


def test_random_windows_on_sorted_axes_match_the_mask_and_are_slices(time_indexers):
    rng = np.random.default_rng(0)
    # Sorted, with repeats (a concatenation seam duplicated a step) and uneven gaps.
    steps = np.sort(rng.integers(0, 400, size=80))
    ds = _dataset(pd.Timestamp("2024-01-01") + pd.to_timedelta(steps, unit="D"))
    checked = 0
    for _ in range(200):
        lo, hi = np.sort(rng.integers(-50, 450, size=2))
        window = (
            pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(lo)),
            pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(hi)),
        )
        expected = _by_mask(ds, window)
        time_indexers.clear()
        got = subset_to_time(ds, window)
        if expected.sizes["time"] == 0:
            assert got.sizes["time"] == ds.sizes["time"]  # no overlap: unchanged
            assert time_indexers == []
            continue
        xr.testing.assert_identical(got, expected)
        assert len(time_indexers) == 1 and isinstance(time_indexers[0], slice)
        checked += 1
    assert checked > 100  # the windows really did land on the record


# -- anything that is not one contiguous run keeps the mask -------------------------


def test_a_gappy_selection_on_an_unsorted_axis_still_works(time_indexers):
    """Two overlapping files concatenated: the kept steps are not one run."""
    ds = _dataset(
        ["2024-01-01", "2024-06-01", "2024-03-01", "2024-11-29", "2024-11-29"]
    )
    window = (pd.Timestamp("2024-04-04"), pd.Timestamp("2025-04-29"))
    out = subset_to_time(ds, window)
    used = list(time_indexers)  # before _by_mask below makes isel calls of its own
    assert out["x"].values.tolist() == [1, 3, 4]  # kept in original order
    assert [pd.Timestamp(t) for t in out.time.values] == [
        pd.Timestamp("2024-06-01"),
        pd.Timestamp("2024-11-29"),
        pd.Timestamp("2024-11-29"),
    ]
    xr.testing.assert_identical(out, _by_mask(ds, window))
    assert len(used) == 1
    assert not isinstance(used[0], slice)  # the mask, as before


def test_a_selection_with_a_hole_in_the_middle_keeps_the_mask(time_indexers):
    # steps 0, 1 and 3 fall in the window; step 2 (an out-of-order stray) does not
    ds = _dataset(["2024-05-01", "2024-05-02", "2024-01-01", "2024-05-03"])
    window = (pd.Timestamp("2024-04-01"), pd.Timestamp("2024-06-01"))
    out = subset_to_time(ds, window)
    assert out["x"].values.tolist() == [0, 1, 3]
    assert not isinstance(time_indexers[0], slice)


def test_no_overlap_still_hands_the_object_back_without_selecting(time_indexers):
    ds = _dataset(pd.date_range("2024-01-01", periods=10, freq="D"))
    out = subset_to_time(ds, (pd.Timestamp("2030-01-01"), pd.Timestamp("2030-02-01")))
    assert out is ds
    assert time_indexers == []
