"""Tests for point sampling and time-series alignment (:mod:`ocean_skill.align`).

The synthetic pair mirrors the case this was built for: a 15-minute mooring at Station
Papa against a 1-degree monthly product **stamped mid-month**. That last detail is not
decoration — mid-month stamps against month-start bins share no timestamp at all, so a
join of the two is empty, and it is the failure most of these tests exist to pin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ocean_skill import align

STATION = (-144.245227, 49.977563)  # OOI Papa flanking mooring A


def station(
    *,
    start: str = "2015-01-01",
    end: str = "2015-07-01",
    freq: str = "15min",
    depth: float | None = 1.0,
    lon: float = STATION[0],
    period: float = 97.0,
):
    """Build a moored record: one dim (time), position and depth as scalar coords."""
    time = pd.date_range(start, end, freq=freq)
    values = 8.0 + np.sin(np.arange(time.size) / period)
    da = xr.DataArray(
        values, coords={"time": time}, dims="time", name="sea_water_temperature"
    )
    da = da.assign_coords(lon=lon, lat=STATION[1])
    if depth is not None:
        da = da.assign_coords(depth=depth)
    da.attrs["units"] = "degC"
    return da


def monthly_grid(
    *,
    start: str = "2014-06-01",
    end: str = "2016-06-01",
    mid_month: bool = True,
    values: np.ndarray | None = None,
):
    """Build a gridded monthly product, stamped mid-month as OceanSODA is."""
    time = pd.date_range(start, end, freq="MS")
    if mid_month:
        time = time + pd.Timedelta(days=14)
    lat = np.arange(45.5, 55.5)
    lon = np.arange(-150.5, -139.5)
    if values is None:
        values = 8.0 + np.random.RandomState(0).rand(time.size, lat.size, lon.size)
    da = xr.DataArray(
        values,
        coords={"time": time, "lat": lat, "lon": lon},
        dims=("time", "lat", "lon"),
        name="sea_surface_temperature",
    )
    da.attrs["units"] = "degC"
    return da


def curvilinear(nx: int = 11, ny: int = 10):
    """Build a ROMS-shaped field: 2-D lon/lat riding on (eta_rho, xi_rho)."""
    lon_1d = np.arange(-150.5, -150.5 + nx)
    lat_1d = np.arange(45.5, 45.5 + ny)
    lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
    da = xr.DataArray(
        np.arange(ny * nx, dtype=float).reshape(ny, nx),
        dims=("eta_rho", "xi_rho"),
        coords={
            "lon_rho": (("eta_rho", "xi_rho"), lon_2d),
            "lat_rho": (("eta_rho", "xi_rho"), lat_2d),
        },
        attrs={"units": "degC"},
    )
    return da


# -- point_of --------------------------------------------------------------------------


def test_point_of_recognizes_a_station_and_declines_a_grid():
    assert align.point_of(station()) == pytest.approx(STATION)
    assert align.point_of(monthly_grid()) is None


# -- sample_at -------------------------------------------------------------------------


def test_nearest_takes_the_cell_value_and_bilinear_interpolates_between_them():
    grid = monthly_grid().isel(time=0)
    nearest = float(align.sample_at(grid, *STATION))
    interpolated = float(align.sample_at(grid, *STATION, method="bilinear"))
    assert nearest in set(np.asarray(grid).ravel().tolist())
    assert interpolated not in set(np.asarray(grid).ravel().tolist())


def test_sampling_works_on_a_curvilinear_grid_both_ways():
    grid = curvilinear()
    nearest = float(align.sample_at(grid, *STATION))
    interpolated = float(align.sample_at(grid, *STATION, method="bilinear"))
    assert nearest in set(np.asarray(grid).ravel().tolist())
    assert interpolated != nearest


def test_the_nearest_cell_is_found_by_great_circle_not_by_degrees():
    """A degree of longitude is 64 km at 50 N and 111 km at the equator.

    Two candidate cells, one a degree east (71.5 km away) and one 0.9 degrees north
    (100 km away): in *degrees* the northern one looks closer, so a Euclidean search
    picks the cell 29 km further from the mooring.
    """
    lon0, lat0 = STATION
    candidates = xr.DataArray(
        [[1.0, 2.0]],
        dims=("point", "which"),
        coords={
            "lon_rho": (("point", "which"), np.array([[lon0 + 1.0, lon0]])),
            "lat_rho": (("point", "which"), np.array([[lat0, lat0 + 0.9]])),
        },
        attrs={"units": "degC"},
    )
    east = align._haversine_km(lon0 + 1.0, lat0, lon0, lat0)
    north = align._haversine_km(lon0, lat0 + 0.9, lon0, lat0)
    assert east < north, "the fixture no longer poses the question"
    assert float(align.sample_at(candidates, lon0, lat0)) == 1.0


def test_a_zero_to_360_grid_still_finds_a_negative_longitude_station():
    """The silent-empty-overlap case: a -144 station against a 0-360 product."""
    grid = monthly_grid().isel(time=0)
    shifted = grid.assign_coords(lon=(grid.lon % 360)).sortby("lon")
    assert float(align.sample_at(shifted, *STATION)) == float(
        align.sample_at(grid, *STATION)
    )


def test_the_grid_offset_is_reported_even_when_it_is_only_the_grid():
    """~55 km at Station Papa is just what a 1-degree cell is — and worth saying."""
    sampled = align.sample_at(monthly_grid().isel(time=0), *STATION)
    assert sampled.attrs["nearest_distance_km"] == pytest.approx(56.1, abs=1.0)
    assert sampled.attrs["cell_km"] > 0


def test_a_station_far_outside_the_grid_warns_about_the_distance():
    with pytest.warns(UserWarning, match="more than one cell away"):
        align.sample_at(monthly_grid().isel(time=0), -20.0, 10.0)


def test_a_masked_cell_raises_rather_than_moving_the_comparison():
    """Relocating to the nearest wet cell would compare a different body of water."""
    grid = monthly_grid().isel(time=0)
    masked = grid.where((grid.lon > -142) | (grid.lat > 54))
    with pytest.raises(ValueError, match="no valid data"):
        align.sample_at(masked, *STATION)


def test_a_masked_neighbour_under_interpolation_names_nearest_as_the_remedy():
    grid = monthly_grid().isel(time=0)
    with pytest.raises(ValueError, match='method="nearest"'):
        align.sample_at(grid.where(grid.lon > -142), *STATION, method="bilinear")


def test_a_conservative_method_cannot_sample_a_point():
    with pytest.raises(ValueError, match="has no area"):
        align.sample_at(
            monthly_grid().isel(time=0), *STATION, method="conservative_normed"
        )


# -- align_series ----------------------------------------------------------------------


def test_a_mid_month_product_and_a_fifteen_minute_mooring_actually_join():
    """The load-bearing case: month-start bins against mid-month stamps.

    Resampling only the fine lane leaves the two labelled differently and an inner join
    finds nothing at all, which is why *both* lanes are re-binned.
    """
    out = align.align_series(monthly_grid(), station())
    assert out.sizes["time"] == 7
    assert int(np.isfinite(out["reference"]).sum()) == 7
    assert int(np.isfinite(out["test"]).sum()) == 7


def test_the_finer_lane_is_aggregated_not_sampled():
    """The bin holds the mooring's monthly *mean*, not its value at the product's stamp.

    Pins the cadence decision itself: a sawtooth whose mean over March differs from its
    mid-March value distinguishes the two, where a smooth series would not.
    """
    obs = station(period=97.0)
    out = align.align_series(monthly_grid(), obs)
    march = float(out["reference"].sel(time="2015-03-01"))
    assert march == pytest.approx(float(obs.sel(time="2015-03").mean()))
    assert march != pytest.approx(
        float(obs.sel(time="2015-03-15T00:00", method="nearest")), abs=1e-4
    )


def test_the_bin_counts_are_of_original_samples():
    """A count taken after resampling is 1 everywhere and says nothing."""
    out = align.align_series(monthly_grid(), station())
    assert float(out["reference_count"].sel(time="2015-03-01")) == 31 * 96
    assert float(out["test_count"].sel(time="2015-03-01")) == 1


def test_the_join_is_not_widened_back_out_by_the_counts():
    """Building a Dataset aligns its members with an *outer* join.

    A count array still spanning the test lane's whole record would pad the pair back
    out with NaN, undoing the inner join and leaving metrics to reduce over mostly
    nothing.
    """
    out = align.align_series(monthly_grid(), station())
    assert out.sizes["time"] == 7
    assert not bool(np.isnan(out["reference"]).any())


def test_both_positions_survive_the_join():
    """The station's own position and the cell the test came from are both kept."""
    out = align.align_series(monthly_grid(), station())
    assert float(out["lon"]) == pytest.approx(STATION[0])
    assert float(out["test_lon"]) != pytest.approx(STATION[0])
    assert out.attrs["station_lat"] == pytest.approx(STATION[1])


def test_an_explicit_freq_overrides_the_inferred_cadence():
    out = align.align_series(monthly_grid(), station(), freq="YS")
    assert out.attrs["resample_freq"] == "YS"
    assert out.sizes["time"] == 1


def test_the_cadence_is_the_median_step_not_the_mean():
    """A mooring record spans deployment turnarounds; a mean cadence would smear.

    Six months of 15-minute data with a four-month gap has a mean step of hours and a
    median step of 15 minutes. Only the median leaves the monthly product as the coarser
    lane, which is what sets the bins.
    """
    obs = xr.concat(
        [station(start="2015-01-01", end="2015-02-01"), station(start="2015-06-01")],
        dim="time",
    )
    assert align._cadence_seconds(obs["time"]) == pytest.approx(900.0)
    assert align.align_series(monthly_grid(), obs).attrs["resample_freq"] == "MS"


def test_no_shared_period_says_what_each_lane_covers():
    with pytest.raises(ValueError, match="share no period"):
        align.align_series(
            monthly_grid(start="1990-01-01", end="1992-01-01"), station()
        )


def test_a_binning_mismatch_is_reported_as_our_bug_not_the_users():
    """Overlapping raw ranges plus an empty join can only be the binning."""
    message = align._no_overlap_message(monthly_grid(), station(), "time", "MS")
    assert "binning mismatch" in message


def test_a_subsurface_reference_against_a_surface_field_warns():
    with pytest.warns(UserWarning, match="compares a subsurface record"):
        align.align_series(monthly_grid(), station(depth=33.9))


def test_a_near_surface_reference_does_not_warn():
    """An instrument at 1 m against a surface field needs no caveat."""
    import warnings

    with warnings.catch_warnings(record=True) as log:
        warnings.simplefilter("always")
        align.align_series(monthly_grid(), station(depth=1.0))
    assert not [w for w in log if "subsurface" in str(w.message)]


def test_a_gridded_reference_is_refused_with_the_reason():
    with pytest.raises(ValueError, match="needs the reference to be one location"):
        align.align_series(monthly_grid(), monthly_grid())


def test_two_lanes_at_different_stations_are_reported():
    """Two point lanes need no sampling — but a position mismatch is not invisible."""
    with pytest.warns(UserWarning, match="km apart"):
        align.align_series(station(lon=STATION[0] + 1.0), station())
