"""Tests for point sampling and time-series alignment (:mod:`ocean_skill.align`).

The synthetic pair mirrors the case this was built for: a 15-minute mooring at Station
Papa against a 1-degree monthly product **stamped mid-month**. That last detail is not
decoration — mid-month stamps against month-start bins share no timestamp at all, so a
join of the two is empty, and it is the failure most of these tests exist to pin.
"""

from __future__ import annotations

import warnings

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


# -- a station comparison, through align() -------------------------------------------


def monthly_station(**kwargs):
    """Return a mooring pre-aggregated to monthly means.

    What a comparison against a monthly product requires: the alignment refuses to
    coarsen a reference on its own.
    """
    from ocean_skill import operators

    raw = station(**kwargs)
    out = operators.aggregate(raw, {"time": {"resample": "MS", "reduce": "mean"}})
    out.attrs.update(raw.attrs)
    return out


def _quiet_align(test, reference, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return align.align(test, reference, over="time", **kwargs)


def test_a_station_reference_is_sampled_rather_than_regridded():
    """The one thing that differs from a gridded comparison; the rest is shared."""
    out = _quiet_align(monthly_grid(), monthly_station())
    assert dict(out.sizes) == {"time": 7}
    assert set(out.data_vars) == {"test", "reference", "difference"}
    assert out.attrs["point_method"] in ("nearest", "bilinear")
    assert not bool(np.isnan(out["reference"]).any())


def test_a_reference_finer_than_the_test_is_refused_with_the_fix():
    """A 15-minute mooring against a monthly product, roles as they should be.

    The refusal is deliberate and not this family's: coarsening the *reference* would
    change the thing being scored against, so it has to be asked for. The message names
    the aggregate spec that does it, which is what ``monthly_station`` above applies.
    """
    with pytest.raises(ValueError, match="reference is the finer of the two"):
        align.align(monthly_grid(), station(), over="time")


def test_the_matching_and_the_sampling_both_go_on_the_record():
    """Two decisions were made for the caller, so both are written down."""
    out = _quiet_align(monthly_grid(), monthly_station())
    assert out.attrs["match_method"] in ("nearest", "mean", "exact")
    assert "match_reason" in out.attrs
    assert out.attrs["scored_over"] == "time"
    assert out.attrs["cell_km"] > 0


def test_both_positions_survive_the_join():
    """The station's own position and the cell the test came from are both kept."""
    out = _quiet_align(monthly_grid(), monthly_station())
    assert float(out["lon"]) == pytest.approx(STATION[0])
    assert "test_lon" in out.coords
    assert out.attrs["station_lat"] == pytest.approx(STATION[1])


def test_mid_month_stamps_still_pair_with_month_start_bins():
    """The trap this whole family kept tripping over.

    A monthly product stamped mid-month and a mooring binned to month starts share no
    timestamp, so an exact join is empty. ``match_axis`` pairs them by nearest step
    within a tolerance and says how far it had to shift them.
    """
    with warnings.catch_warnings(record=True) as log:
        warnings.simplefilter("always")
        out = align.align(monthly_grid(mid_month=True), monthly_station(), over="time")
    assert out.sizes["time"] == 7
    assert any("shifting each by" in str(w.message) for w in log)


def test_a_subsurface_reference_against_a_surface_field_warns():
    with pytest.warns(UserWarning, match="compares a subsurface record"):
        align.align(monthly_grid(), monthly_station(depth=33.9), over="time")


def test_a_near_surface_reference_does_not_warn():
    """An instrument at 1 m against a surface field needs no caveat."""
    with warnings.catch_warnings(record=True) as log:
        warnings.simplefilter("always")
        align.align(monthly_grid(), monthly_station(depth=1.0), over="time")
    assert not [w for w in log if "subsurface" in str(w.message)]


def test_two_lanes_at_different_stations_are_reported():
    """Two point lanes need no sampling — but a position mismatch is not invisible."""
    with pytest.warns(UserWarning, match="km apart"):
        align.align(
            monthly_station(lon=STATION[0] + 1.0), monthly_station(), over="time"
        )


def test_a_conservative_method_becomes_nearest_at_a_point():
    """``Comparison`` passes the package default on every call, so it cannot warn.

    An explicitly conservative method still raises (see ``sample_at``); this is only the
    default being read sensibly rather than refused.
    """
    out = _quiet_align(monthly_grid(), monthly_station(), method="conservative_normed")
    assert out.attrs["point_method"] == "nearest"


# -- the Comparison surface: featureType chooses, and says so --------------------------


@pytest.fixture
def station_lanes(monkeypatch):
    """Serve a station reference and a gridded test in place of real sources.

    Mirrors ``tests/test_skill_maps.py``'s ``stub``: a comparison is built and aligned
    without a catalog or a network, so the featureType is stated rather than read.
    """
    import ocean_skill.comparison as comparison

    lanes = {
        "papa": monthly_station(),
        "product": monthly_grid(),
        # a gridded reference, for the mixed-set case below
        "satellite": monthly_grid(start="2014-06-01"),
    }
    monkeypatch.setattr(
        comparison, "prepare_source", lambda source, *a, **k: (lanes[source], None)
    )
    monkeypatch.setattr(comparison, "_domain_of", lambda name: None)
    monkeypatch.setattr(
        comparison,
        "_feature_type",
        lambda source: "timeSeries" if source == "papa" else "grid",
    )
    return lanes


def _comparison(reference: str = "papa", **kwargs):
    from ocean_skill.comparison import Comparison

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = Comparison(
            reference=reference,
            test="product",
            variable="sea_water_temperature",
            cache=False,
            **kwargs,
        )
        out.align()
    return out


def test_a_timeseries_reference_keeps_its_time_axis_without_being_asked(station_lanes):
    """FeatureType drives the recipe — the promise README makes from the start."""
    comparison = _comparison()
    assert comparison.over == "time"
    assert comparison.aligned.sizes["time"] == 7


def test_a_station_comparison_draws_as_lines_and_says_why(station_lanes):
    comparison = _comparison()
    assert comparison.is_series
    assert comparison.family == "series"
    assert "featureType is 'timeSeries'" in comparison.family_reason


def test_the_shape_of_the_data_can_overrule_the_featuretype(station_lanes, monkeypatch):
    """A catalog can be wrong, and then the aligned pair is what a renderer can draw.

    ``featureType: timeSeries`` on an entry that is really gridded gives a scored
    comparison with maps, not lines — and ``family_reason`` says the scoring decided it,
    which is how someone traces the surprise back to the entry.
    """
    import ocean_skill.comparison as comparison_module

    lanes = {"papa": monthly_grid(start="2014-06-01"), "product": monthly_grid()}
    monkeypatch.setattr(
        comparison_module,
        "prepare_source",
        lambda source, *a, **k: (lanes[source], None),
    )
    comparison = _comparison()
    assert not comparison.is_series
    assert comparison.family == "skill_map"
    assert "metric maps" in comparison.family_reason


def test_an_explicit_over_wins_over_the_featuretype(station_lanes):
    assert _comparison(over="time").over_reason == "over= as asked"


def test_the_overall_metrics_of_a_station_are_not_area_weighted(station_lanes):
    """One position has one cos(lat); calling that an area weighting is a claim."""
    record = _comparison().metrics()
    assert record["weighted"] is False
    assert record["n"] == 7


def test_a_station_comparison_has_no_maps_to_draw(station_lanes):
    with pytest.raises(ValueError, match="one place"):
        _comparison().maps("bias")


def test_a_set_that_mixes_families_says_which_went_which_way(station_lanes):
    """Lines and metric maps are different figures, so a set of both is refused.

    Naming which comparison went which way matters more than the refusal: the usual
    cause is one reference in a fan-out being a mooring while the rest are gridded, and
    the reason each one gives is what points at it.
    """
    from ocean_skill.comparison import ComparisonSet

    series = _comparison()
    scored = _comparison(reference="satellite", over="time")
    assert (series.family, scored.family) == ("series", "skill_map")
    with pytest.raises(ValueError, match="different figures"):
        ComparisonSet([series, scored]).plot()


def test_the_depth_caveat_survives_the_reference_being_resampled():
    """The mooring is binned to monthly means, which drops its depth *coordinate*.

    Read from the variable's attrs instead, so the caveat still fires — and reports the
    spread, since a record whose instrument moved between 9 and 34 m is the one where
    "compared against a surface field" is most worth saying.
    """
    from ocean_skill import operators

    raw = station(depth=None)
    raw = raw.assign_attrs(depth_m=24.9, depth_range_m=(16.0, 39.5), units="degC")
    monthly = operators.aggregate(raw, {"time": {"resample": "MS", "reduce": "mean"}})
    monthly.attrs.update(raw.attrs)
    assert "depth" not in monthly.coords
    with pytest.warns(UserWarning, match=r"24.9 m \(varying 16-39.5 m\)"):
        align.align(monthly_grid(), monthly, over="time")
