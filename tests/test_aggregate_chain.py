"""An aggregate axis may take a chain of reductions as a list.

``{"time": [{"groupby": "month", "reduce": "mean"}, "var"]}`` is the variance of the
twelve monthly means -- the seasonal cycle's variance. Each step acts on the axis the
previous one left, so the tests pin both the arithmetic (analytic values on a field
whose value *is* its month) and the bookkeeping around it: what is refused before
anything is computed, which attrs the result carries, and the data caveats that are
warnings rather than annotations.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import xarray as xr

from ocean_skill import operators
from ocean_skill.operators import (
    REDUCERS,
    SPREAD_COORD,
    aggregate,
    final_step,
    spatial_mean_in_spec,
)

VAR_1_12 = 143.0 / 12.0  # population variance of 1..12


@pytest.fixture(autouse=True)
def _range_reducer_registered():
    """Keep the built-in ``range`` reducer present.

    ``test_operators`` registers and then pops its own ``range`` to exercise
    ``register_reducer``, which in one process would remove the built-in one for every
    test that follows.
    """
    REDUCERS.setdefault("range", operators._range_reducer)


def _field(time, values, *, units="mg/m^3"):
    """Build a (time, lat, lon) field holding ``values[i]`` everywhere at step i."""
    values = np.asarray(values, dtype=float)
    return xr.DataArray(
        values[:, None, None] * np.ones((1, 2, 3)),
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": [1.0, 2.0], "lon": [1.0, 2.0, 3.0]},
        attrs={"units": units},
    )


@pytest.fixture
def monthly():
    """Two years of monthly steps; value = the month number, so a groupby recovers k."""
    time = xr.date_range("2012-01-01", periods=24, freq="MS")
    return _field(time, [t.month for t in time])


@pytest.fixture
def three_years():
    """Three years of monthly steps: month number + 10 * year index."""
    time = xr.date_range("2012-01-01", periods=36, freq="MS")
    return _field(time, [t.month + 10 * (t.year - 2012) for t in time])


SEASONAL = {"groupby": "month", "reduce": "mean"}


def _everywhere(out, expected):
    np.testing.assert_allclose(out.values, expected)


# -- the arithmetic ---------------------------------------------------------------


def test_seasonal_cycle_variance(monthly):
    out = aggregate(monthly, {"time": [SEASONAL, "var"]})
    assert out.dims == ("lat", "lon")
    _everywhere(out, VAR_1_12)


def test_seasonal_cycle_std(monthly):
    _everywhere(aggregate(monthly, {"time": [SEASONAL, "std"]}), np.sqrt(VAR_1_12))


def test_kwargs_ride_through_a_chain_step(monthly):
    out = aggregate(monthly, {"time": [SEASONAL, {"reduce": "var", "ddof": 1}]})
    _everywhere(out, 143.0 / 11.0)


def test_seasonal_range(monthly):
    _everywhere(aggregate(monthly, {"time": [SEASONAL, "range"]}), 11.0)


def test_interannual_std_of_annual_means(three_years):
    annual = {"resample": "1YS", "reduce": "mean"}
    out = aggregate(three_years, {"time": [annual, "std"]})
    # annual means are 6.5, 16.5, 26.5
    _everywhere(out, np.std([6.5, 16.5, 26.5]))


def test_mean_annual_maximum(three_years):
    out = aggregate(
        three_years, {"time": [{"resample": "1YS", "reduce": "max"}, "mean"]}
    )
    _everywhere(out, np.mean([12.0, 22.0, 32.0]))


def test_mean_diurnal_range_on_hourly_data():
    time = xr.date_range("2012-01-01", periods=72, freq="h")
    hourly = _field(time, [t.hour + 0.5 * t.day for t in time])
    out = aggregate(hourly, {"time": [{"resample": "1D", "reduce": "range"}, "mean"]})
    _everywhere(out, 23.0)


def test_range_is_a_plain_single_step_reduction(monthly):
    _everywhere(aggregate(monthly, {"time": "range"}), 11.0)  # 1..12 over 24 steps


def test_month_balanced_annual_mean_carries_its_spread(monthly):
    out = aggregate(monthly, {"time": [SEASONAL, {"reduce": "mean", "spread": "std"}]})
    _everywhere(out, 6.5)
    assert SPREAD_COORD in out.coords
    _everywhere(out.coords[SPREAD_COORD], np.sqrt(VAR_1_12))


def test_a_chain_works_on_a_non_time_axis():
    """Chains are not time's alone: group depth into layers, then take the extreme."""
    da = xr.DataArray(
        np.arange(6.0),
        dims="depth",
        coords={
            "depth": [0, 10, 20, 30, 40, 50],
            "layer": ("depth", [0, 0, 0, 1, 1, 1]),
        },
        attrs={"units": "m"},
    )
    out = aggregate(da, {"depth": [{"groupby": "layer", "reduce": "mean"}, "max"]})
    assert float(out) == pytest.approx(4.0)  # layer means are 1 and 4


# -- single steps and the shape of a chain -------------------------------------


def test_a_one_element_list_is_its_single_step(monthly):
    for step in (SEASONAL, "mean", {"resample": "1YS", "reduce": "max"}):
        bare = aggregate(monthly, {"time": step})
        chained = aggregate(monthly, {"time": [step]})
        xr.testing.assert_identical(chained, bare)


def test_final_step_reads_the_last_element_of_a_chain():
    assert final_step([SEASONAL, "var"]) == "var"
    assert final_step("mean") == "mean"
    with pytest.raises(ValueError, match="at least one step"):
        final_step([])


def test_an_empty_chain_is_rejected(monthly):
    with pytest.raises(ValueError, match="empty"):
        aggregate(monthly, {"time": []})


def test_spread_is_only_allowed_on_the_last_step(monthly):
    chain = [{"groupby": "month", "reduce": "mean", "spread": "std"}, "mean"]
    with pytest.raises(ValueError, match="only allowed on the last step"):
        aggregate(monthly, {"time": chain})


def test_a_step_after_a_collapsing_step_is_rejected_naming_the_step(monthly):
    with pytest.raises(ValueError, match=r"step 1: .*already collapsed"):
        aggregate(monthly, {"time": ["mean", "var"]})


def test_a_groupby_after_a_groupby_is_rejected(monthly):
    with pytest.raises(ValueError, match="no time left to bin"):
        aggregate(monthly, {"time": [SEASONAL, {"groupby": "year", "reduce": "mean"}]})


def test_a_resample_after_a_groupby_is_rejected(monthly):
    with pytest.raises(ValueError, match="no time left to bin"):
        aggregate(monthly, {"time": [SEASONAL, {"resample": "1YS", "reduce": "mean"}]})


def test_a_resample_after_a_resample_is_fine(three_years):
    chain = [
        {"resample": "1MS", "reduce": "mean"},
        {"resample": "1YS", "reduce": "max"},
        "mean",
    ]
    _everywhere(aggregate(three_years, {"time": chain}), np.mean([12.0, 22.0, 32.0]))


def test_a_step_without_a_reduce_is_rejected(monthly):
    with pytest.raises(ValueError, match="needs a 'reduce'"):
        aggregate(monthly, {"time": [{"groupby": "month"}, "var"]})


def test_a_nested_list_step_is_rejected(monthly):
    with pytest.raises(ValueError, match="reduction name or a"):
        aggregate(monthly, {"time": [[SEASONAL], "var"]})


def test_a_bad_chain_is_rejected_before_anything_is_computed(monthly):
    """The bad chain is the *last* key; the good one before it must not have run."""

    class Boom(Exception):
        pass

    def explode(*args, **kwargs):
        raise Boom

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(operators, "_reduce_dim", explode)
        with pytest.raises(ValueError, match="empty"):
            aggregate(monthly, {"lat": "mean", "time": []})


# -- attrs ---------------------------------------------------------------------------


def test_chain_cell_methods_read_left_to_right(monthly):
    out = aggregate(monthly, {"time": [SEASONAL, "var"]})
    assert out.attrs["cell_methods"] == (
        "time: mean within months time: variance over months"
    )
    assert out.attrs["statistic"] == "var"


def test_cell_methods_and_statistic_on_single_steps(monthly):
    plain = aggregate(monthly, {"time": "mean"})
    assert plain.attrs["cell_methods"] == "time: mean"
    assert plain.attrs["statistic"] == "mean"
    assert aggregate(monthly, {"time": "max"}).attrs["cell_methods"] == "time: maximum"
    assert aggregate(monthly, {"time": "std"}).attrs["cell_methods"] == (
        "time: standard_deviation"
    )
    assert aggregate(monthly, {"time": "std"}).attrs["statistic"] == "std"
    # an unknown-to-the-table reduction is written under its own name
    q = aggregate(monthly, {"time": {"reduce": "quantile", "q": 0.5}})
    assert q.attrs["cell_methods"] == "time: quantile"


def test_cell_methods_append_to_an_existing_attribute(monthly):
    monthly.attrs["cell_methods"] = "area: mean"
    out = aggregate(monthly, {"time": "mean"})
    assert out.attrs["cell_methods"] == "area: mean time: mean"


def test_resample_chain_statistic_is_the_last_step(three_years):
    out = aggregate(
        three_years, {"time": [{"resample": "1YS", "reduce": "mean"}, "std"]}
    )
    assert out.attrs["statistic"] == "std"
    assert out.attrs["cell_methods"] == "time: mean time: standard_deviation"


def test_units_come_from_for_statistic_per_dimension_changing_step(
    monthly, monkeypatch
):
    from ocean_skill import units

    asked = []

    def fake(unit_string, statistic):
        asked.append(statistic)
        return f"({unit_string})^2" if statistic == "var" else unit_string

    monkeypatch.setattr(units, "for_statistic", fake)
    out = aggregate(monthly, {"time": [SEASONAL, "var"]})
    assert out.attrs["units"] == "(mg/m^3)^2"
    assert asked == ["var"], "the unit-preserving mean step is not even asked about"

    # a mean over a variance keeps variance units
    chained = aggregate(
        monthly,
        {"time": [{"resample": "1YS", "reduce": "var"}, "mean"]},
    )
    assert chained.attrs["units"] == "(mg/m^3)^2"


def test_spread_units_go_through_for_statistic(monthly, monkeypatch):
    from ocean_skill import units

    monkeypatch.setattr(
        units,
        "for_statistic",
        lambda u, s: f"{u}^2" if s == "var" else u,
    )
    out = aggregate(monthly, {"time": [SEASONAL, {"reduce": "mean", "spread": "var"}]})
    assert out.coords[SPREAD_COORD].attrs["units"] == "mg/m^3^2"
    assert out.attrs["units"] == "mg/m^3"


# -- the month-coverage warning --------------------------------------------------


def test_fewer_than_twelve_months_warns_once_the_months_are_folded():
    time = xr.date_range("2012-01-01", periods=6, freq="MS")
    half = _field(time, [t.month for t in time])
    with pytest.warns(UserWarning, match="only 6 of 12 months"):
        out = aggregate(half, {"time": [SEASONAL, "var"]})
    _everywhere(out, np.var(np.arange(1.0, 7.0)))


def test_a_full_cycle_does_not_warn(monthly):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        aggregate(monthly, {"time": [SEASONAL, "var"]})


def test_a_bare_partial_climatology_does_not_warn():
    time = xr.date_range("2012-01-01", periods=6, freq="MS")
    half = _field(time, [t.month for t in time])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = aggregate(half, {"time": SEASONAL})
    assert out.sizes["month"] == 6


# -- an undecoded climatology ---------------------------------------------------


def _undecoded(with_month: bool):
    """WOA's shape: the float positions 1..12 of a "months since" axis."""
    da = _field(np.arange(1.0, 13.0), np.arange(1.0, 13.0))
    da["time"].attrs["units"] = "months since 1955-01-01"
    if with_month:
        da = da.assign_coords(month=("time", np.arange(1, 13)))
    return da


def test_a_month_coordinate_stands_in_for_an_undecoded_time_axis():
    out = aggregate(_undecoded(True), {"time": [SEASONAL, "var"]})
    _everywhere(out, VAR_1_12)


def test_undecoded_time_without_a_month_coordinate_is_a_friendly_error():
    with pytest.raises(ValueError, match="not a decoded calendar axis"):
        aggregate(_undecoded(False), {"time": [SEASONAL, "var"]})


# -- the joint spatial mean --------------------------------------------------------


def test_spatial_mean_in_spec_reads_one_step_chains_only():
    both = {"lon": "mean", "lat": "mean"}
    assert spatial_mean_in_spec(both) == ("lon", "lat")
    assert spatial_mean_in_spec({"lon": ["mean"], "lat": [{"reduce": "mean"}]}) == (
        "lon",
        "lat",
    )
    assert spatial_mean_in_spec({"lon": ["mean", "max"], "lat": "mean"}) is None
    assert spatial_mean_in_spec({"lon": [], "lat": "mean"}) is None
    assert spatial_mean_in_spec({"lon": ["max"], "lat": ["mean"]}) is None


def test_one_step_chains_still_take_the_joint_area_weighted_path(monthly):
    out = aggregate(monthly, {"lon": ["mean"], "lat": ["mean"]})
    assert "spatial_mean" in out.attrs


def test_a_multi_step_chain_on_lon_goes_the_per_axis_way(monthly):
    monthly = monthly.assign_coords(cell=("lon", [0, 0, 1]))
    chain = [{"groupby": "cell", "reduce": "mean"}, "max"]
    out = aggregate(monthly, {"lon": chain, "lat": "mean"})
    assert "spatial_mean" not in out.attrs
    assert out.dims == ("time",)


def test_the_built_in_range_reducer_is_registered_and_reduces_a_groupby(monthly):
    assert "range" in REDUCERS
    out = aggregate(monthly, {"time": {"groupby": "month", "reduce": "range"}})
    _everywhere(out, 0.0)  # both years of a month share one value
