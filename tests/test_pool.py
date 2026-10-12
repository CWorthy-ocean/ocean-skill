"""``ComparisonSet.pool``: scoring a group's matched pairs as one pile of samples.

Where ``average()`` lines members up point by point and averages them (see
``tests/test_average.py``), ``pool()`` keeps every member's valid pairs as samples --
members of different shapes included -- and scores the pile. Built the same way as
test_average.py: aligned pairs are injected as ``Comparison._aligned``, so no catalog
entry needs to exist for any of the station or model names used here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ocean_skill import metrics as osk_metrics
from ocean_skill.comparison import Comparison, ComparisonSet

TEMPERATURE = "sea_water_potential_temperature"
SALINITY = "sea_water_practical_salinity"
TIMES = pd.date_range("2024-01-01", periods=6, freq="D")


def _pair(values_ref, values_test, *, dims=("time",), coords=None, lon=None, lat=None):
    """An aligned pair of any shape; ``lon``/``lat`` make it a place, else a bare array."""
    dims = () if np.ndim(values_ref) == 0 else dims
    ref = xr.DataArray(np.asarray(values_ref, float), dims=dims, coords=coords or {})
    tst = xr.DataArray(np.asarray(values_test, float), dims=dims, coords=coords or {})
    if lon is not None:
        ref = ref.assign_coords(lon=lon, lat=lat)
        tst = tst.assign_coords(lon=lon, lat=lat)
    ref.attrs["units"] = "degC"
    tst.attrs["units"] = "degC"
    diff = (tst - ref).rename("difference")
    diff.attrs["long_name"] = "test minus reference"
    return xr.Dataset(
        {"test": tst.rename("test"), "reference": ref.rename("reference"), "difference": diff}
    )


def _series(values_ref, values_test, *, times=None, lon=-150.0, lat=20.0):
    times = TIMES[: len(values_ref)] if times is None else times
    return _pair(values_ref, values_test, coords={"time": times}, lon=lon, lat=lat)


def _comparison(aligned, *, reference="HV1", test="his", variable=TEMPERATURE, select=None):
    c = Comparison(
        reference=reference, test=test, variable=variable, select=select, cache=False
    )
    c._aligned = aligned.copy(deep=True)
    return c


def _hand_metrics(test, reference) -> dict:
    """What compute() says about a plain 1-D pile -- the pool's metrics must match it."""
    ds = xr.Dataset(
        {
            "test": ("sample", np.asarray(test, float)),
            "reference": ("sample", np.asarray(reference, float)),
        }
    )
    return osk_metrics.compute(
        ds, test_name="test", reference_name="reference", weighted=False
    )


A_REF, A_TEST = [11.0, 12.0, 13.0], [10.0, 12.0, 14.0]
B_REF, B_TEST = [22.0, 21.0, 20.0, 19.0], [20.0, 21.0, 22.0, 24.0]


def _two_stations() -> ComparisonSet:
    return ComparisonSet(
        [
            _comparison(_series(A_REF, A_TEST), reference="A"),
            _comparison(_series(B_REF, B_TEST, lon=-152.0, lat=22.0), reference="B"),
        ]
    )


# -- the score is the pile's ----------------------------------------------------------


def test_pool_n_is_the_sum_of_the_members_valid_pairs():
    rec = _two_stations().pool(by="variable").comparisons[0].metrics()
    assert rec["n"] == len(A_REF) + len(B_REF)


def test_pool_metrics_equal_a_hand_concatenation():
    pooled = _two_stations().pool(by="variable").comparisons[0]
    expected = _hand_metrics(A_TEST + B_TEST, A_REF + B_REF)
    rec = pooled.metrics()
    for name in ("bias", "rmse", "corr", "n"):
        assert rec[name] == pytest.approx(expected[name]), name


def test_pool_is_not_the_average_of_the_stations():
    """Pooling scores the station-to-station spread that averaging smooths away."""
    comps = _two_stations()
    pooled = comps.pool(by="variable").comparisons[0].metrics()
    means_ref = np.mean([A_REF[:3], B_REF[:3]], axis=0)
    means_test = np.mean([A_TEST[:3], B_TEST[:3]], axis=0)
    averaged = _hand_metrics(means_test, means_ref)
    assert pooled["n"] == 7 and averaged["n"] == 3
    assert pooled["corr"] != pytest.approx(averaged["corr"])


def test_pooled_aligned_is_one_sample_axis_with_difference_recomputed():
    aligned = _two_stations().pool(by="variable").comparisons[0].aligned
    assert set(aligned.data_vars) == {"test", "reference", "difference"}
    assert aligned["test"].dims == ("sample",)
    assert not set(aligned.coords)
    np.testing.assert_allclose(
        aligned["difference"].values, aligned["test"].values - aligned["reference"].values
    )
    assert aligned["test"].attrs["units"] == "degC"
    assert aligned["reference"].attrs["units"] == "degC"
    assert aligned["difference"].attrs["long_name"] == "test minus reference"


# -- members of different shapes ------------------------------------------------------


def test_pool_members_of_different_lengths():
    comps = ComparisonSet(
        [
            _comparison(_series([1, 2], [1.5, 2.5]), reference="A"),
            _comparison(_series([3, 4, 5, 6], [3, 4, 5, 7]), reference="B"),
        ]
    )
    pooled = comps.pool().comparisons[0]
    assert pooled.metrics()["n"] == 6


def test_pool_a_zero_d_member_with_series_and_a_profile():
    zero_d = _pair(15.0, 14.0)  # one model value against one observation
    profile = _pair(
        [[1.0, 2.0], [3.0, 4.0]],
        [[1.5, 2.0], [3.0, 5.0]],
        dims=("time", "depth"),
        coords={"time": TIMES[:2], "depth": [0.0, 10.0]},
        lon=-150.0,
        lat=20.0,
    )
    comps = ComparisonSet(
        [
            _comparison(zero_d, reference="Z"),
            _comparison(_series(A_REF, A_TEST), reference="A"),
            _comparison(profile, reference="P"),
        ]
    )
    rec = comps.pool().comparisons[0].metrics()
    expected = _hand_metrics(
        [14.0, *A_TEST, 1.5, 2.0, 3.0, 5.0], [15.0, *A_REF, 1.0, 2.0, 3.0, 4.0]
    )
    assert rec["n"] == 8
    for name in ("bias", "rmse", "corr"):
        assert rec[name] == pytest.approx(expected[name]), name


def test_pool_members_spelling_the_time_axis_differently():
    upper = _series(B_REF, B_TEST).rename({"time": "TIME"})
    comps = ComparisonSet(
        [
            _comparison(_series(A_REF, A_TEST), reference="A"),
            _comparison(upper, reference="B"),
        ]
    )
    assert comps.pool().comparisons[0].metrics()["n"] == 7


def test_pool_drops_pairs_that_are_nan_in_either_lane():
    ref = [1.0, np.nan, 3.0, 4.0]
    tst = [1.0, 2.0, np.nan, 5.0]
    comps = ComparisonSet(
        [
            _comparison(_series(ref, tst), reference="A"),
            _comparison(_series(A_REF, A_TEST), reference="B"),
        ]
    )
    rec = comps.pool().comparisons[0].metrics()
    expected = _hand_metrics([1.0, 5.0, *A_TEST], [1.0, 4.0, *A_REF])
    assert rec["n"] == 5
    assert rec["bias"] == pytest.approx(expected["bias"])


def test_pool_refuses_a_group_with_no_valid_pair():
    comps = ComparisonSet([_comparison(_series([np.nan, 1.0], [1.0, np.nan]))])
    with pytest.raises(ValueError, match="nothing to score"):
        comps.pool()


# -- grouping -------------------------------------------------------------------------


def _fan() -> ComparisonSet:
    """2 stations x 2 variables x 2 depth layers x 2 seasons-by-time-window."""
    rng = np.random.default_rng(1)
    comps = []
    for variable in (TEMPERATURE, SALINITY):
        for depth in ({"min": 0, "max": 5}, {"min": 20, "max": 65}):
            for season in ("MAM", "JJA"):
                for station in ("A", "B"):
                    ref = rng.normal(10, 2, 5)
                    comps.append(
                        _comparison(
                            _series(ref, ref + rng.normal(0.5, 0.3, 5)),
                            reference=station,
                            variable=variable,
                            select={"depth": depth, "time": {"season": season}},
                        )
                    )
    return ComparisonSet(comps)


def test_pool_by_variable():
    pooled = _fan().pool(by="variable")
    assert len(pooled) == 2
    assert len({c.label for c in pooled}) == 2
    assert all(c.metrics()["n"] == 2 * 2 * 2 * 5 for c in pooled)  # depth x season x station


def test_pool_by_variable_depth_time_keeps_those_columns():
    pooled = _fan().pool(by=["variable", "depth", "time"])
    assert len(pooled) == 2 * 2 * 2
    table = pooled.metrics()
    assert set(table["time"]) == {"MAM", "JJA"}
    assert set(table["depth"]) == {"0-5 m", "20-65 m"}  # bands read as labels
    assert len(set(table["variable"])) == 2
    assert (table["n"] == 2 * 5).all()  # two stations x five samples each
    # one row per (variable, depth, time): the table pivots
    assert not table.duplicated(["variable", "depth", "time"]).any()
    assert table["weighted"].eq(False).all()
    assert "n_eff" not in table.columns or table["n_eff"].isna().all()
    assert "station_lon" not in table.columns


def test_pool_by_unknown_dimension_raises_naming_pool():
    with pytest.raises(ValueError, match=r"pool\(by=\.\.\.\) unknown"):
        _fan().pool(by="season")


def test_pool_warns_on_a_group_spanning_test_sources():
    comps = ComparisonSet(
        [
            _comparison(_series(A_REF, A_TEST), test="his"),
            _comparison(_series(B_REF, B_TEST), test="bgc"),
        ]
    )
    with pytest.warns(UserWarning, match=r"pool\(\): a group spans multiple test"):
        pooled = comps.pool(by="variable")
    assert pooled.comparisons[0].test_name == "his+bgc"


def test_pool_does_not_alter_the_members():
    comps = _two_stations()
    before = comps.comparisons[0].aligned.copy(deep=True)
    comps.pool()
    xr.testing.assert_identical(comps.comparisons[0].aligned, before)
    assert not comps.comparisons[0]._pooled


# -- what a pooled comparison is ------------------------------------------------------


def test_pooled_comparison_is_flagged_and_unscored_along_an_axis():
    pooled = _two_stations().pool().comparisons[0]
    assert pooled._pooled is True
    assert pooled.over is None
    assert _two_stations().comparisons[0]._pooled is False


def test_pooled_metrics_record_is_a_plain_pile_of_samples():
    rec = _two_stations().pool().comparisons[0].metrics()
    assert rec["weighted"] is False
    assert "station_lon" not in rec
    assert rec["variable"] == TEMPERATURE


# -- drawing --------------------------------------------------------------------------


@pytest.mark.parametrize("renderer", ["matplotlib", "holoviews"])
def test_plot_of_a_pooled_set_raises(renderer):
    pooled = _two_stations().pool()
    with pytest.raises(ValueError, match="pile of matched samples"):
        pooled.plot(renderer=renderer)
    with pytest.raises(ValueError, match="pile of matched samples"):
        pooled.comparisons[0].plot(renderer=renderer)


def test_as_item_of_a_pooled_comparison_raises():
    with pytest.raises(ValueError, match=r"\.metrics\(\)"):
        _two_stations().pool().comparisons[0].as_item()


@pytest.mark.parametrize("renderer", ["matplotlib", "holoviews"])
def test_plot_of_a_single_value_comparison_raises(renderer):
    """One model value against one observation draws as nothing -- say so."""
    c = _comparison(_pair(15.0, 14.0))
    assert c.family == "field_row"
    with pytest.raises(ValueError, match="one test value against one reference"):
        c.plot(renderer=renderer)
    with pytest.raises(ValueError, match=r"times=\{'seasons'"):
        ComparisonSet([c]).plot(renderer=renderer)


def test_target_and_taylor_work_on_a_pooled_set():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pooled = _fan().pool(by=["variable", "depth", "time"])
    pooled.target()
    plt.close("all")
    pooled.taylor()
    plt.close("all")
