"""``osk.detide`` across gaps: no invented samples, no tide leaking out of a gap.

``oceans.filters.pl33tn`` fills every NaN in a DataArray with a straight line before
it filters, however long the gap -- so a multi-day outage came back as invented values,
and the ~``T`` hours either side of it carried a straight line's (tide-free) spectrum
into the subtidal series. ``detide(max_gap=...)`` bridges short gaps and treats a longer
one as the end of a stretch: the reference behaviour is filtering each stretch of record
on its own (the ``_pl33_stretches`` rule of the CIOFS3 paper figures), which the tests
below re-implement locally and compare against to float precision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ocean_skill.detide import _half_window, detide

pytest.importorskip("oceans", reason="detide needs the 'oceans' package (pl33tn)")

N = 1200  # hours
H = 32  # PL33's half-window at T=33, dt=1h: samples lost at each end of a stretch
GAP = slice(400, 472)  # a three-day outage
CLOSE = {"rtol": 1e-9, "atol": 1e-12, "equal_nan": True}


def _index(n=N):
    return pd.date_range("2020-01-01", periods=n, freq="h")


def _signal(n=N):
    t = np.arange(n, dtype=float)
    return (
        0.5 * np.sin(2 * np.pi * t / (24 * 10))
        + 1.0 * np.sin(2 * np.pi * t / 12.42)
        + 0.6 * np.sin(2 * np.pi * t / 24.0)
    )


def _da(values, index=None, name="zeta"):
    index = _index(len(values)) if index is None else index
    return xr.DataArray(values, dims=["time"], coords={"time": index}, name=name)


def _per_stretch(values, *, max_gap=6):
    """Filter by the paper's rule: stretches split at gaps over ``max_gap`` hours.

    Shorter gaps are bridged linearly, each stretch is filtered alone, edges NaN.
    """
    n = len(values)
    out = np.full(n, np.nan)
    good = np.flatnonzero(np.isfinite(values))
    breaks = np.flatnonzero(np.diff(good) > max_gap)
    starts = np.r_[good[0], good[breaks + 1]]
    stops = np.r_[good[breaks], good[-1]]
    for i0, i1 in zip(starts, stops, strict=True):
        stretch = pd.Series(values[i0 : i1 + 1]).interpolate(limit_area="inside")
        low = detide(_da(stretch.to_numpy()), max_gap=None).values
        out[i0 : i1 + 1] = low
    out[np.isnan(values)] = np.nan  # no invented samples
    return out


def _gappy(gap=GAP):
    v = _signal()
    v[gap] = np.nan
    return v


# -- the leak ------------------------------------------------------------------------


def test_long_gap_is_nan_inside_and_a_half_window_either_side():
    v = _gappy()
    out = detide(_da(v)).values

    assert np.isnan(out[GAP]).all()  # nothing invented inside the gap
    assert np.isnan(out[GAP.start - H : GAP.start]).all()  # contaminated margin
    assert np.isnan(out[GAP.stop : GAP.stop + H]).all()
    assert np.isfinite(out[H : GAP.start - H]).all()  # clean stretch is untouched
    assert np.isfinite(out[GAP.stop + H : N - H]).all()


def test_long_gap_matches_filtering_each_stretch_alone():
    v = _gappy()
    out = detide(_da(v)).values
    np.testing.assert_allclose(out, _per_stretch(v), **CLOSE)


def test_tide_does_not_leak_into_the_margin_it_masks():
    """The old filter left a tide-sized error beside a gap; what is kept is clean."""
    v = _signal()
    trend = 0.5 * np.sin(2 * np.pi * np.arange(N) / (24 * 10))
    v[GAP] = np.nan
    out = detide(_da(v)).values
    kept = np.isfinite(out)
    assert np.abs(out[kept] - trend[kept]).max() < 0.15  # tidal amplitude is 1.0


def test_short_and_long_gaps_in_one_record():
    v = _signal()
    v[GAP] = np.nan
    v[800:804] = np.nan  # four hours: bridged
    out = detide(_da(v)).values
    expected = _per_stretch(v)
    np.testing.assert_allclose(out, expected, **CLOSE)
    assert np.isnan(out[800:804]).all()
    assert np.isfinite(out[799]) and np.isfinite(out[804])


# -- short gaps are bridged -----------------------------------------------------------


def test_short_gap_is_bridged_nan_only_at_the_gap_samples():
    v = _signal()
    v[600:604] = np.nan
    out = detide(_da(v)).values
    assert np.isnan(out[600:604]).all()
    assert np.isfinite(out[599]) and np.isfinite(out[604])
    assert np.isfinite(out[H : 600]).all() and np.isfinite(out[604 : N - H]).all()
    bridged = pd.Series(v).interpolate(limit_area="inside").to_numpy()
    expected = detide(_da(bridged)).values
    expected[600:604] = np.nan
    np.testing.assert_allclose(out, expected, rtol=1e-12, equal_nan=True)


def test_max_gap_boundary_six_hours_between_samples_bridges_seven_does_not():
    five = _signal()
    five[600:605] = np.nan  # samples 599 and 605 are 6 h apart: bridged
    out5 = detide(_da(five), max_gap="6h").values
    assert np.isfinite(out5[599]) and np.isfinite(out5[605])

    six = _signal()
    six[600:606] = np.nan  # 599 and 606 are 7 h apart: ends the stretch
    out6 = detide(_da(six), max_gap="6h").values
    assert np.isnan(out6[600 - H : 606 + H]).all()


@pytest.mark.parametrize(
    "max_gap", [pd.Timedelta(hours=2), np.timedelta64(2, "h"), 2, 2.0, "2h", "120min"]
)
def test_max_gap_accepts_string_timedelta_and_hours(max_gap):
    v = _signal()
    v[600:603] = np.nan  # a 4 h hole between samples: longer than 2 h
    out = detide(_da(v), max_gap=max_gap).values
    assert np.isnan(out[600 - H : 603 + H]).all()


def test_max_gap_none_bridges_everything_as_before():
    v = _gappy()
    out = detide(_da(v), max_gap=None)
    old = detide(_da(v).interpolate_na("time"))  # what pl33tn did with any NaN run
    np.testing.assert_array_equal(out.values, old.values)
    assert np.isfinite(out.values[GAP]).all()  # the invented values, on request


@pytest.mark.parametrize("bad", [-1, "soon", "-3h", float("nan"), [6], True])
def test_max_gap_rejects_nonsense(bad):
    with pytest.raises(ValueError, match="max_gap"):
        detide(_da(_signal()), max_gap=bad)


# -- the fast path is unchanged -------------------------------------------------------


def test_gap_free_series_is_bit_identical_to_the_bare_filter():
    from oceans.filters import pl33tn

    v = _signal()
    da = _da(v)
    out = detide(da)
    da2 = da.assign_coords(time=da["time"].astype("datetime64[ns]"))
    da2["time"].attrs["standard_name"] = "time"
    np.testing.assert_array_equal(out.values, pl33tn(da2).values)
    np.testing.assert_array_equal(out.values, detide(da, max_gap=None).values)


# -- the axis -------------------------------------------------------------------------


def test_missing_stamps_behave_like_nan_gaps():
    v = _gappy()
    v[800:803] = np.nan  # a short hole too
    nan_version = detide(_da(v))

    keep = np.ones(N, bool)
    keep[GAP] = False
    keep[800:803] = False
    stamps = _index()[keep]
    sparse = detide(_da(_signal()[keep], index=stamps))

    assert sparse.sizes["time"] == keep.sum()  # returned on the caller's own stamps
    np.testing.assert_array_equal(sparse["time"].values, stamps.values)
    expected = nan_version.values.copy()
    holes = np.zeros(N, bool)
    holes[800:803] = True
    expected[holes] = np.nan
    np.testing.assert_allclose(
        sparse.values, expected[keep], rtol=1e-9, atol=1e-12, equal_nan=True
    )


def test_jittered_stamps_are_still_regular():
    rng = np.random.default_rng(3)
    jitter = pd.to_timedelta(rng.integers(-10, 10, N), unit="s")
    da = _da(_signal(), index=_index() + jitter)
    out = detide(da)
    clean = detide(_da(_signal()))
    np.testing.assert_allclose(out.values, clean.values, **CLOSE)


def test_irregular_axis_raises_with_a_resample_hint():
    rng = np.random.default_rng(0)
    steps = rng.choice([60, 90], size=N - 1)  # minutes: 1 h and 1.5 h
    stamps = pd.DatetimeIndex(
        pd.Timestamp("2020-01-01")
        + pd.to_timedelta(np.r_[0, np.cumsum(steps)], unit="min")
    )
    with pytest.raises(ValueError, match="resample"):
        detide(_da(_signal(), index=stamps))
    with pytest.raises(ValueError, match="resample"):
        detide(pd.Series(_signal(), index=stamps))
    with pytest.raises(ValueError, match="resample"):
        detide(_da(_signal(), index=stamps).to_dataset())


# -- shapes ---------------------------------------------------------------------------


def test_time_by_depth_gap_in_one_column_masks_only_that_column():
    base = _signal()
    data = np.stack([base, base + 0.1, base - 0.1], axis=1)
    data[GAP, 1] = np.nan
    da = xr.DataArray(
        data,
        dims=["time", "depth"],
        coords={"time": _index(), "depth": [5.0, 10.0, 20.0]},
    )
    out = detide(da)
    assert np.isnan(out.values[GAP.start - H : GAP.stop + H, 1]).all()
    np.testing.assert_allclose(out.values[:, 1], _per_stretch(data[:, 1]), **CLOSE)
    for j in (0, 2):  # untouched columns equal their own clean filter
        np.testing.assert_array_equal(out.values[:, j], detide(da.isel(depth=j)).values)
        assert np.isfinite(out.values[H : N - H, j]).all()


def test_dask_chunked_gappy_field_matches_numpy():
    pytest.importorskip("dask")
    base = _signal()
    data = np.stack([base, base + 0.1], axis=1)
    data[GAP, 0] = np.nan
    data[900:903, 1] = np.nan
    da = xr.DataArray(
        data, dims=["time", "depth"], coords={"time": _index(), "depth": [5.0, 10.0]}
    )
    lazy = detide(da.chunk({"time": 24, "depth": 1}))
    np.testing.assert_allclose(
        lazy.compute().values, detide(da).values, rtol=1e-12, equal_nan=True
    )


def test_dataset_tidal_and_both_follow():
    v = _gappy()
    ds = _da(v).to_dataset()
    both = detide(ds, component="both")
    low = detide(ds)["zeta"].values
    np.testing.assert_array_equal(both["zeta_subtidal"].values, low)
    tidal = both["zeta_tidal"].values
    np.testing.assert_array_equal(np.isnan(tidal), np.isnan(low))  # NaN where low is
    sub, tid = detide(_da(v), component="both")
    np.testing.assert_array_equal(sub.values, low)
    np.testing.assert_array_equal(np.isnan(tid.values), np.isnan(low))


def test_series_and_dataframe_follow_the_same_rule():
    v = _gappy()
    v[800:804] = np.nan
    s = pd.Series(v, index=_index(), name="zeta")
    out = detide(s)
    assert isinstance(out, pd.Series) and out.index.equals(s.index)
    expected = detide(_da(v)).values
    keep = slice(0, N - H)  # Series path leaves oceans' own one-sample tail quirk
    np.testing.assert_allclose(out.values[keep], expected[keep], **CLOSE)
    assert np.isnan(out.values[GAP]).all()
    assert np.isnan(out.values[800:804]).all() and np.isfinite(out.values[799])

    df = pd.DataFrame({"a": v, "b": _signal()}, index=_index())
    sub = detide(df)
    np.testing.assert_array_equal(sub["a"].values, out.values)
    np.testing.assert_array_equal(sub["b"].values, detide(df["b"]).values)
    assert np.isfinite(sub["b"].values[H : N - H]).all()


def test_series_with_missing_stamps_and_max_gap_none():
    v = _gappy()
    s = pd.Series(v, index=_index())
    old = detide(s.interpolate(limit_area="inside"))
    np.testing.assert_array_equal(detide(s, max_gap=None).values, old.values)
    keep = np.ones(N, bool)
    keep[GAP] = False
    sparse = detide(pd.Series(v[keep], index=_index()[keep]))
    assert len(sparse) == keep.sum()
    np.testing.assert_allclose(
        sparse.values, detide(s).values[keep], rtol=1e-9, atol=1e-12, equal_nan=True
    )


# -- the window width -----------------------------------------------------------------


@pytest.mark.parametrize("step_h", [1.0, 0.5, 0.25, 3.0, 1.5])
@pytest.mark.parametrize("T", [33.0, 72.0])
def test_half_window_is_the_edge_nan_count_pl33tn_leaves(step_h, T):
    n = 2500
    stamps = pd.date_range("2020-01-01", periods=n, freq=pd.Timedelta(hours=step_h))
    out = detide(_da(np.sin(np.arange(n) / 7.0), index=stamps), T=T).values
    lead = np.argmax(np.isfinite(out))
    trail = np.argmax(np.isfinite(out[::-1]))
    assert lead == trail == _half_window(step_h, T)


def test_long_gap_at_a_non_hourly_step():
    stamps = pd.date_range("2020-01-01", periods=2000, freq="30min")
    t = np.arange(2000) / 2.0
    v = np.sin(2 * np.pi * t / 12.42) + 0.5 * np.sin(2 * np.pi * t / 240.0)
    v[800:900] = np.nan  # 50 h
    out = detide(_da(v, index=stamps)).values
    h = _half_window(0.5, 33.0)
    assert np.isnan(out[800 - h : 900 + h]).all()
    assert np.isfinite(out[h : 800 - h]).all()
    assert np.isfinite(out[900 + h : 2000 - h]).all()


# -- the pair pipeline ----------------------------------------------------------------


def test_pairs_detide_picks_up_the_gap_handling():
    from ocean_skill import pairs

    v = _gappy()
    ds = xr.Dataset(
        {"test": ("time", _signal()), "reference": ("time", v)},
        coords={"time": _index()},
    )
    out = pairs._detide(ds, {"reference": True})
    assert np.isnan(out["reference"].values[GAP.start - H : GAP.stop + H]).all()
    assert np.isfinite(out["test"].values[H : N - H]).all()
