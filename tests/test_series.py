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


def test_an_interpolating_method_at_a_station_warns_and_recommends_nearest():
    """``align()``'s station branch, not just :func:`sample_at` directly."""
    with pytest.warns(UserWarning, match='method="nearest" is recommended'):
        align.align(monthly_grid(), monthly_station(), over="time", method="bilinear")


def test_nearest_at_a_station_does_not_warn():
    with warnings.catch_warnings(record=True) as log:
        warnings.simplefilter("always")
        align.align(monthly_grid(), monthly_station(), over="time", method="nearest")
    assert not [w for w in log if "is recommended" in str(w.message)]


def test_the_translated_conservative_default_does_not_warn():
    """``conservative_normed`` silently becomes ``"nearest"`` at a station -- the
    warning is for an *interpolating* method reaching the station branch, not for
    the package's own default being used as given.
    """
    with warnings.catch_warnings(record=True) as log:
        warnings.simplefilter("always")
        align.align(
            monthly_grid(), monthly_station(), over="time",
            method="conservative_normed",
        )
    assert not [w for w in log if "is recommended" in str(w.message)]


def test_a_conservative_method_cannot_sample_a_point():
    with pytest.raises(ValueError, match="has no area"):
        align.sample_at(
            monthly_grid().isel(time=0), *STATION, method="conservative_normed"
        )


# -- a station comparison, through align() -------------------------------------------


def monthly_station(**kwargs):
    """Return a mooring pre-aggregated to monthly means.

    Left alone, aligning a 15-minute mooring against a monthly product coarsens the
    mooring itself (see :func:`test_a_finer_reference_is_averaged_with_a_warning`) and
    warns about it. Doing it here, deliberately, puts the choice on the record instead
    and runs quietly.
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


def test_a_finer_reference_is_averaged_with_a_warning():
    """A 15-minute mooring against a monthly product, with nothing pre-aggregated.

    The old behavior refused outright; now the mooring is coarsened into the
    product's own months automatically, the same way a fine satellite reference gets
    regridded onto a coarse model grid in space — but it warns, since coarsening the
    reference does change what is being scored against. ``monthly_station`` above is
    the quiet, deliberate way to reach the same place.
    """
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        out = align.align(monthly_grid(), station(), over="time")
    assert any("reference is the finer of the two" in str(w.message) for w in record)
    assert out.attrs["match_method"] == "mean"
    assert out.attrs["match_target"] == "test"
    assert set(out.data_vars) == {"test", "reference", "difference"}
    assert 6 <= out.sizes["time"] <= 7
    assert not bool(np.isnan(out["reference"]).any())


def test_the_matching_and_the_sampling_both_go_on_the_record():
    """Two decisions were made for the caller, so both are written down."""
    out = _quiet_align(monthly_grid(), monthly_station())
    assert out.attrs["match_method"] in ("nearest", "mean", "exact")
    assert out.attrs["match_target"] in ("test", "reference")
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

    The ``prepare_source`` stub records ``(source, kwargs)`` onto ``lanes.calls`` --
    a plain list attached to the returned dict -- so a test can assert what a
    metadata-derived ``bbox=``/``time_window=`` reached the test lane with, the way
    :func:`_model_comparison` below lets a routed one be checked through real
    ``align()`` machinery instead.
    """
    from ocean_skill import comparison

    lanes = {
        "papa": monthly_station(),
        "product": monthly_grid(),
        # a gridded reference, for the mixed-set case below
        "satellite": monthly_grid(start="2014-06-01"),
    }
    calls: list[tuple[str, dict]] = []
    lanes["calls"] = calls

    def _prepare_source(source, *a, **k):
        calls.append((source, k))
        return lanes[source], None

    monkeypatch.setattr(comparison, "prepare_source", _prepare_source)
    monkeypatch.setattr(comparison, "_domain_of", lambda name: None)
    monkeypatch.setattr(comparison, "_time_coverage_of", lambda name: None)
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


# -- reference-metadata narrowing of the test lane --------------------------------


def _lane_kwargs(lanes, source):
    """The last recorded ``prepare_source`` kwargs for ``source``, from station_lanes."""
    matches = [k for s, k in lanes["calls"] if s == source]
    assert matches, f"prepare_source was never called for {source!r}"
    return matches[-1]


def test_a_mooring_reference_narrows_the_test_lane_from_metadata(
    station_lanes, monkeypatch
):
    """The motivating case: a mooring's own catalog position/window crop the test."""
    import ocean_skill.comparison as comparison_module

    bbox = (-144.3, 49.9, -144.3, 49.9)
    window = (pd.Timestamp("2024-04-04"), pd.Timestamp("2024-08-14"))
    monkeypatch.setattr(comparison_module, "_domain_of", lambda name: bbox)
    monkeypatch.setattr(comparison_module, "_time_coverage_of", lambda name: window)

    _comparison()

    test_kwargs = _lane_kwargs(station_lanes, "product")
    assert test_kwargs["bbox"] == bbox
    assert test_kwargs["time_window"] == window
    # the reference is never narrowed *by its own* metadata -- it owns the metadata,
    # and the two-point-mismatch warning exists to catch exactly that mistake. It
    # still gets the pre-existing, unrelated crop derived from the (now-narrowed)
    # test lane's own domain (see align()'s bbox_of(t)/time_span_of(t)), which is
    # not this feature and is not the injected bbox/window themselves.
    reference_kwargs = _lane_kwargs(station_lanes, "papa")
    assert reference_kwargs["bbox"] != bbox
    assert reference_kwargs["time_window"] != window


def test_a_profile_reference_narrows_independently_of_over(monkeypatch):
    """Narrowing is decoupled from over= -- a profile's own over (``"Z"``, by its
    featureType's own definition — see :func:`ocean_skill.comparison._implied_over`)
    does not change whether its catalog position narrows the test lane.

    Checked directly against :meth:`Comparison._reference_narrowing` rather than
    through a full ``align()``: an unreduced gridded *test* lane still needs its own
    ``select=``/``aggregate=`` to become comparable over depth, same as any other
    profile comparison. That reduction is orthogonal to what this test checks.
    """
    import ocean_skill.comparison as comparison_module
    from ocean_skill.comparison import Comparison

    bbox = (-144.3, 49.9, -144.3, 49.9)
    monkeypatch.setattr(comparison_module, "_domain_of", lambda name: bbox)
    monkeypatch.setattr(comparison_module, "_time_coverage_of", lambda name: None)
    monkeypatch.setattr(comparison_module, "_feature_type", lambda name: "profile")

    c = Comparison(reference="papa", test="product", variable=MODEL_VAR)
    assert c.over == "Z"
    assert c._reference_narrowing() == (bbox, None)


def test_a_trajectory_reference_narrows_to_its_extent_box(monkeypatch):
    """A trajectory's declared extent is a real box, not a degenerate point."""
    import ocean_skill.comparison as comparison_module
    from ocean_skill.comparison import Comparison

    bbox = (-150.0, 40.0, -140.0, 55.0)
    monkeypatch.setattr(comparison_module, "_domain_of", lambda name: bbox)
    monkeypatch.setattr(comparison_module, "_time_coverage_of", lambda name: None)
    monkeypatch.setattr(comparison_module, "_feature_type", lambda name: "trajectory")

    c = Comparison(reference="papa", test="product", variable=MODEL_VAR)
    assert c._reference_narrowing() == (bbox, None)


def test_a_repeat_visit_stations_gps_scatter_collapses_to_its_centre(monkeypatch):
    """A station's geospatial_*_min/_max need not already be equal.

    A repeat-visit CTD station logs a slightly different GPS fix on each cast --
    real, un-degenerate min/max a few hundred metres apart -- and without the
    collapse below that bbox falls through to the *padded* region crop
    (:func:`ocean_skill.align.subset_to_bbox`) rather than the tight point one,
    which on a small regional model can keep nearly the whole grid. Only
    ``trajectory`` (a genuinely moving position) is exempt --
    :func:`test_a_trajectory_reference_narrows_to_its_extent_box` covers that.
    """
    import ocean_skill.comparison as comparison_module
    from ocean_skill.comparison import Comparison

    scattered = (-21.9895, 64.2615, -21.9867, 64.2648)  # ~137 m x 370 m of GPS noise
    monkeypatch.setattr(comparison_module, "_domain_of", lambda name: scattered)
    monkeypatch.setattr(comparison_module, "_time_coverage_of", lambda name: None)
    monkeypatch.setattr(
        comparison_module, "_feature_type", lambda name: "timeSeriesProfile"
    )

    c = Comparison(reference="papa", test="product", variable=MODEL_VAR)
    bbox, _ = c._reference_narrowing()
    lon = 0.5 * (scattered[0] + scattered[2])
    lat = 0.5 * (scattered[1] + scattered[3])
    assert bbox == (lon, lat, lon, lat)


def test_an_already_degenerate_bbox_is_unaffected_by_the_collapse(monkeypatch):
    """A genuinely fixed station is byte-identical to before this existed."""
    import ocean_skill.comparison as comparison_module
    from ocean_skill.comparison import Comparison

    point = (-144.3, 49.9, -144.3, 49.9)
    monkeypatch.setattr(comparison_module, "_domain_of", lambda name: point)
    monkeypatch.setattr(comparison_module, "_time_coverage_of", lambda name: None)
    monkeypatch.setattr(comparison_module, "_feature_type", lambda name: "station")

    c = Comparison(reference="papa", test="product", variable=MODEL_VAR)
    assert c._reference_narrowing() == (point, None)


def test_a_gridded_reference_is_never_narrowed(monkeypatch):
    """The policy table's other half: featureType outside NARROWING_FEATURE_TYPES."""
    import ocean_skill.comparison as comparison_module
    from ocean_skill.comparison import Comparison

    monkeypatch.setattr(
        comparison_module, "_domain_of", lambda name: (-150.0, 40.0, -140.0, 55.0)
    )
    monkeypatch.setattr(comparison_module, "_feature_type", lambda name: "grid")

    c = Comparison(reference="papa", test="product", variable=MODEL_VAR)
    assert c._reference_narrowing() == (None, None)


def test_missing_metadata_leaves_the_test_lane_unnarrowed(station_lanes):
    """The default station_lanes stubs return None -- byte-identical to no feature."""
    _comparison()
    test_kwargs = _lane_kwargs(station_lanes, "product")
    assert test_kwargs["bbox"] is None
    assert test_kwargs["time_window"] is None


def test_a_pair_spec_select_skips_the_derived_narrowing(station_lanes, monkeypatch):
    """A deliberate {"test":..., "reference":...} names two positions on purpose."""
    import ocean_skill.comparison as comparison_module

    monkeypatch.setattr(
        comparison_module, "_domain_of", lambda name: (-144.3, 49.9, -144.3, 49.9)
    )
    _comparison(select={"test": {"lon": -145.0}, "reference": {"lon": -144.3}})
    assert _lane_kwargs(station_lanes, "product")["bbox"] is None


# -- _time_coverage_of ------------------------------------------------------------


def _resolved(monkeypatch, metadata):
    """Patch ``catalog.resolve`` to hand back ``metadata`` for any source name."""
    from types import SimpleNamespace

    from ocean_skill import catalog

    monkeypatch.setattr(catalog, "resolve", lambda name: SimpleNamespace(metadata=metadata))


def test_time_coverage_prefers_minmax_time_over_the_truncated_pair(monkeypatch):
    """ERDDAP's own minTime/maxTime are full timestamps; time_coverage_* is a date."""
    from ocean_skill.comparison import _time_coverage_of

    _resolved(
        monkeypatch,
        {
            "minTime": "2024-04-04T06:00:00Z",
            "maxTime": "2024-08-14T18:00:00Z",
            "time_coverage_start": "2024-04-04",
            "time_coverage_end": "2024-08-14",
        },
    )
    start, stop = _time_coverage_of("mooring")
    assert start == pd.Timestamp("2024-04-04T06:00:00Z") - pd.Timedelta(days=1)
    assert stop == pd.Timestamp("2024-08-14T18:00:00Z") + pd.Timedelta(days=1)


def test_time_coverage_falls_back_to_the_truncated_pair_and_pads_a_day(monkeypatch):
    """No minTime/maxTime (a non-ERDDAP source): time_coverage_* still works, with
    the truncation covered by the same day of padding.
    """
    from ocean_skill.comparison import _time_coverage_of

    _resolved(
        monkeypatch,
        {"time_coverage_start": "2024-04-04", "time_coverage_end": "2024-08-14"},
    )
    start, stop = _time_coverage_of("mooring")
    assert start == pd.Timestamp("2024-04-03")
    assert stop == pd.Timestamp("2024-08-15")


def test_time_coverage_is_none_without_either_pair(monkeypatch):
    from ocean_skill.comparison import _time_coverage_of

    _resolved(monkeypatch, {"minTime": "2024-04-04T00:00:00Z"})  # maxTime missing
    assert _time_coverage_of("mooring") is None
    _resolved(monkeypatch, {})
    assert _time_coverage_of("mooring") is None


def test_time_coverage_is_none_on_garbage_rather_than_raising(monkeypatch):
    from ocean_skill.comparison import _time_coverage_of

    _resolved(monkeypatch, {"minTime": "not a date", "maxTime": "also not one"})
    assert _time_coverage_of("mooring") is None


# -- pre-selecting the test lane at the reference's own cast times --------------
#
# _reference_narrowing crops the test lane to the reference's declared *coverage*
# -- a contiguous (start, stop) span. A repeat-visit reference's actual casts are a
# sparse handful of instants inside that span, and alignment only ever pairs each
# one with its single nearest test step (ocean_skill.align._match_by_nearest) --
# so _reference_time_targets exists to prune the test lane to that same nearest-
# step set *before* the read/vertical-transform, not after.


def test_time_targets_reads_the_references_own_cast_times(monkeypatch):
    """The happy path: sorted, deduplicated cast times, for a kept time axis."""
    import ocean_skill as osk
    from ocean_skill.comparison import Comparison

    casts = pd.to_datetime(["2024-05-01", "2024-04-04", "2024-04-04"])  # dup, unsorted
    ds = xr.Dataset(coords={"time": ("time", casts.values)})
    monkeypatch.setattr(osk, "read", lambda name, **kw: ds)
    _resolved(monkeypatch, {"featureType": "timeSeriesProfile"})

    c = Comparison(reference="papa", test="product", variable=MODEL_VAR, over="time")
    targets = c._reference_time_targets()
    assert targets is not None
    assert list(targets) == sorted(pd.to_datetime(casts.unique()))


def _read_must_not_be_called(name, **kw):
    raise AssertionError("the reference must not be read for this comparison")


def test_time_targets_is_none_when_depth_is_kept_and_nothing_collapses_time(
    monkeypatch,
):
    """over='Z' keeps depth standing, and no aggregate touches time either -- there
    is no time axis a nearest-step prune would even apply to.
    """
    import ocean_skill as osk
    from ocean_skill.comparison import Comparison

    monkeypatch.setattr(osk, "read", _read_must_not_be_called)
    _resolved(monkeypatch, {"featureType": "timeSeriesProfile"})

    c = Comparison(reference="papa", test="product", variable=MODEL_VAR, over="Z")
    assert c._reference_time_targets() is None


def test_time_targets_also_fires_when_a_time_aggregate_collapses_it_under_over_z(
    monkeypatch,
):
    """over='Z' keeps *depth* standing, but a plain time reducer (mean/std, no
    groupby or resample) still collapses time to one number per depth level --
    every in-between model step feeds that number directly (nothing downstream
    discards them the way over='time' nearest-matching does), so matching the
    model to the reference's own cast times is not just faster here, it is the
    intended comparison: the model averaged over the times actually sampled,
    not over a full window mostly never visited.
    """
    import ocean_skill as osk
    from ocean_skill.comparison import Comparison

    casts = pd.to_datetime(["2024-05-01", "2024-04-04", "2024-04-04"])  # dup, unsorted
    ds = xr.Dataset(coords={"time": ("time", casts.values)})
    monkeypatch.setattr(osk, "read", lambda name, **kw: ds)
    _resolved(monkeypatch, {"featureType": "timeSeriesProfile"})

    c = Comparison(
        reference="papa",
        test="product",
        variable=MODEL_VAR,
        over="Z",
        aggregate={"time": {"reduce": "mean", "spread": "std"}},
    )
    targets = c._reference_time_targets()
    assert targets is not None
    assert list(targets) == sorted(pd.to_datetime(casts.unique()))


def test_time_targets_stays_none_for_a_climatology_under_over_z(monkeypatch):
    """A groupby/resample aggregate *keeps* a time axis (a climatology, or
    consecutive periods) rather than collapsing it -- it legitimately needs
    every step in the window, not just the ones nearest a cast -- so pruning
    must not fire here even though depth is kept and time is being aggregated.
    """
    import ocean_skill as osk
    from ocean_skill.comparison import Comparison

    monkeypatch.setattr(osk, "read", _read_must_not_be_called)
    _resolved(monkeypatch, {"featureType": "timeSeriesProfile"})

    c = Comparison(
        reference="papa",
        test="product",
        variable=MODEL_VAR,
        over="Z",
        aggregate={"time": {"groupby": "month", "reduce": "mean"}},
    )
    assert c._reference_time_targets() is None


def test_time_targets_is_memoized_across_repeated_calls(monkeypatch):
    """Both _cache_key and align() call this -- and it reads the reference to
    answer -- so a second call must not read it again.
    """
    import ocean_skill as osk
    from ocean_skill.comparison import Comparison

    casts = pd.to_datetime(["2024-05-01", "2024-04-04"])
    ds = xr.Dataset(coords={"time": ("time", casts.values)})
    calls = []

    def counted_read(name, **kw):
        calls.append(name)
        return ds

    monkeypatch.setattr(osk, "read", counted_read)
    _resolved(monkeypatch, {"featureType": "timeSeriesProfile"})

    c = Comparison(
        reference="papa",
        test="product",
        variable=MODEL_VAR,
        over="Z",
        aggregate={"time": {"reduce": "mean", "spread": "std"}},
    )
    first = c._reference_time_targets()
    second = c._reference_time_targets()
    assert len(calls) == 1
    assert list(first) == list(second)


def test_time_targets_none_is_itself_memoized(monkeypatch):
    """A None answer (no pruning applies here) is just as worth caching as an
    array -- a comparison whose over/aggregate settles the question up front
    must never read the reference at all, on any call.
    """
    import ocean_skill as osk
    from ocean_skill.comparison import Comparison

    monkeypatch.setattr(osk, "read", _read_must_not_be_called)
    _resolved(monkeypatch, {"featureType": "timeSeriesProfile"})

    c = Comparison(reference="papa", test="product", variable=MODEL_VAR, over="Z")
    assert c._reference_time_targets() is None
    assert c._reference_time_targets() is None  # a second call must not read either


def test_ref_time_targets_join_the_cache_key_when_present(monkeypatch):
    """A comparison whose test lane gets pruned to cast-nearest steps must not
    share a cache key with an otherwise-identical one that isn't pruned -- the
    pruned run's mean/spread is computed over a different (smaller) set of
    model steps, so the two are different results, not different routes to the
    same one. Isolated from _reference_time_targets' own gating logic by
    monkeypatching it directly on two instances built with identical
    reference/test/variable/over/aggregate, so the only thing that differs is
    the value _cache_key folds in.
    """
    import ocean_skill as osk
    from ocean_skill.comparison import Comparison

    monkeypatch.setattr(osk, "read", _read_must_not_be_called)
    _resolved(monkeypatch, {"featureType": "timeSeriesProfile"})
    kwargs = dict(
        reference="papa",
        test="product",
        variable=MODEL_VAR,
        over="Z",
        aggregate={"time": {"reduce": "mean", "spread": "std"}},
    )

    unpruned = Comparison(**kwargs)
    unpruned._reference_time_targets = lambda: None

    pruned = Comparison(**kwargs)
    pruned._reference_time_targets = lambda: pd.to_datetime(
        ["2024-04-04", "2024-05-01"]
    ).values

    assert unpruned._cache_key != pruned._cache_key


def test_time_targets_is_none_for_a_trajectory(monkeypatch):
    """A moving position pairs on space and time together; skip the prune (mirrors
    _reference_narrowing's own trajectory exception, one level over).
    """
    import ocean_skill as osk
    from ocean_skill.comparison import Comparison

    monkeypatch.setattr(osk, "read", _read_must_not_be_called)
    _resolved(monkeypatch, {"featureType": "trajectoryProfile"})

    c = Comparison(reference="papa", test="product", variable=MODEL_VAR, over="time")
    assert c._reference_time_targets() is None


def test_time_targets_is_none_for_a_pair_spec(monkeypatch):
    """A deliberate {"test":..., "reference":...} names two recipes on purpose."""
    import ocean_skill as osk
    from ocean_skill.comparison import Comparison

    monkeypatch.setattr(osk, "read", _read_must_not_be_called)
    _resolved(monkeypatch, {"featureType": "timeSeriesProfile"})

    c = Comparison(
        reference="papa",
        test="product",
        variable=MODEL_VAR,
        over="time",
        select={"test": {}, "reference": {}},
    )
    assert c._reference_time_targets() is None


def test_time_targets_fails_open_on_a_broken_reference_read(monkeypatch):
    """A savings only -- an unreadable reference must not break the comparison."""
    import ocean_skill as osk
    from ocean_skill.comparison import Comparison

    def broken_read(name, **kw):
        raise OSError("boom")

    monkeypatch.setattr(osk, "read", broken_read)
    _resolved(monkeypatch, {"featureType": "timeSeriesProfile"})

    c = Comparison(reference="papa", test="product", variable=MODEL_VAR, over="time")
    assert c._reference_time_targets() is None


def test_the_overall_metrics_of_a_station_are_not_area_weighted(station_lanes):
    """One position has one cos(lat); calling that an area weighting is a claim."""
    record = _comparison().metrics()
    assert record["weighted"] is False
    assert record["n"] == 7


def test_a_stations_metrics_record_carries_its_own_position(station_lanes):
    """osk.map_metrics needs a position on every station's record to plot it.

    The position lives on ``aligned.attrs`` (written by ``align._align_at_point``)
    and would otherwise never reach the metrics record — see
    ``ocean_skill.plot.map_metrics``.
    """
    record = _comparison().metrics()
    assert record["station_lon"] == pytest.approx(STATION[0])
    assert record["station_lat"] == pytest.approx(STATION[1])


def test_a_gridded_comparisons_record_carries_no_station_position(
    station_lanes, monkeypatch
):
    """A grid-vs-grid comparison has no single place to report."""
    import ocean_skill.comparison as comparison_module

    lanes = {"papa": monthly_grid(start="2014-06-01"), "product": monthly_grid()}
    monkeypatch.setattr(
        comparison_module,
        "prepare_source",
        lambda source, *a, **k: (lanes[source], None),
    )
    record = _comparison().metrics()
    assert "station_lon" not in record


def test_a_station_comparison_has_no_maps_to_draw(station_lanes):
    with pytest.raises(ValueError, match="one place"):
        _comparison().pointwise_metrics("bias")


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


# -- _warn_if_no_overlap -----------------------------------------------------------


def test_a_time_only_mismatch_names_the_offending_axis(monkeypatch):
    """The Anvil case: a profile's own catalog record predates the model's run."""
    import ocean_skill.comparison as comparison_module
    from ocean_skill.comparison import Comparison

    bbox = (-21.9877, 64.264, -21.9877, 64.264)
    monkeypatch.setattr(
        comparison_module,
        "_domain_of",
        lambda name: bbox if name == "reference" else (-22.5, 64.0, -21.0, 64.6),
    )
    monkeypatch.setattr(
        comparison_module,
        "_time_coverage_of",
        lambda name: {
            "reference": (pd.Timestamp("2025-01-08"), pd.Timestamp("2025-01-10")),
            "test": (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31")),
        }[name],
    )

    c = Comparison(reference="reference", test="test", variable=MODEL_VAR)
    with pytest.warns(UserWarning, match="do not overlap in time"):
        c._warn_if_no_overlap()


def test_a_space_only_mismatch_names_the_offending_axis(monkeypatch):
    import ocean_skill.comparison as comparison_module
    from ocean_skill.comparison import Comparison

    monkeypatch.setattr(
        comparison_module,
        "_domain_of",
        lambda name: {
            "reference": (170.0, 50.0, 170.0, 50.0),
            "test": (-10.0, -5.0, 10.0, 5.0),
        }[name],
    )
    monkeypatch.setattr(comparison_module, "_time_coverage_of", lambda name: None)

    c = Comparison(reference="reference", test="test", variable=MODEL_VAR)
    with pytest.warns(UserWarning, match="do not overlap in space"):
        c._warn_if_no_overlap()


def test_both_axes_mismatched_are_both_named(monkeypatch):
    import ocean_skill.comparison as comparison_module
    from ocean_skill.comparison import Comparison

    monkeypatch.setattr(
        comparison_module,
        "_domain_of",
        lambda name: {
            "reference": (170.0, 50.0, 170.0, 50.0),
            "test": (-10.0, -5.0, 10.0, 5.0),
        }[name],
    )
    monkeypatch.setattr(
        comparison_module,
        "_time_coverage_of",
        lambda name: {
            "reference": (pd.Timestamp("2025-01-08"), pd.Timestamp("2025-01-10")),
            "test": (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31")),
        }[name],
    )

    c = Comparison(reference="reference", test="test", variable=MODEL_VAR)
    with pytest.warns(UserWarning, match="do not overlap in space or time"):
        c._warn_if_no_overlap()


def test_overlapping_sources_say_nothing(monkeypatch, recwarn):
    import ocean_skill.comparison as comparison_module
    from ocean_skill.comparison import Comparison

    monkeypatch.setattr(
        comparison_module, "_domain_of", lambda name: (-22.5, 64.0, -21.0, 64.6)
    )
    monkeypatch.setattr(
        comparison_module,
        "_time_coverage_of",
        lambda name: (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31")),
    )

    c = Comparison(reference="reference", test="test", variable=MODEL_VAR)
    c._warn_if_no_overlap()
    assert not [w for w in recwarn.list if "overlap" in str(w.message)]


def test_unknown_extents_are_not_treated_as_a_mismatch(monkeypatch, recwarn):
    """Missing catalog metadata is 'unknown', not 'no' -- see Overlap's own contract."""
    import ocean_skill.comparison as comparison_module
    from ocean_skill.comparison import Comparison

    monkeypatch.setattr(comparison_module, "_domain_of", lambda name: None)
    monkeypatch.setattr(comparison_module, "_time_coverage_of", lambda name: None)

    c = Comparison(reference="reference", test="test", variable=MODEL_VAR)
    c._warn_if_no_overlap()
    assert not [w for w in recwarn.list if "overlap" in str(w.message)]


def test_a_climatology_time_mismatch_is_not_a_false_alarm(monkeypatch, recwarn):
    """A climatology's declared calendar span is a label, not a record (see
    ocean_skill.align.subset_to_time) -- a mismatch against one must not warn.
    """
    import ocean_skill.comparison as comparison_module
    from ocean_skill.comparison import Comparison

    monkeypatch.setattr(
        comparison_module, "_domain_of", lambda name: (-22.5, 64.0, -21.0, 64.6)
    )
    monkeypatch.setattr(
        comparison_module,
        "_time_coverage_of",
        lambda name: {
            "reference": (pd.Timestamp("1965-01-01"), pd.Timestamp("1965-01-31")),
            "test": (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31")),
        }[name],
    )
    monkeypatch.setattr(
        comparison_module, "_is_climatology", lambda name: name == "reference"
    )

    c = Comparison(reference="reference", test="test", variable=MODEL_VAR)
    c._warn_if_no_overlap()
    assert not [w for w in recwarn.list if "overlap" in str(w.message)]


# -- model-vs-model point series: over= inference and sample_at routing ---------------

#: Canonical name throughout this section so vocabulary resolution has nothing to
#: warn about -- these tests are about the routing, not the aliasing.
MODEL_VAR = "sea_water_potential_temperature"

#: A point request neither grid is centred on, so each lane's own nearest cell is a
#: distinct, non-trivial answer worth checking.
POINT_SELECT = {"lon": -144.3, "lat": 50.0, "time": slice("2014-06", "2014-08")}


def _offset_grid(*, lat_off=0.0, lon_off=0.0, seed=0):
    """A monthly rectilinear grid, offset and independently valued.

    Two calls with different offsets stand in for two models on their own native
    grids -- close enough to share a point select, not so close that their nearest
    cells to a shared request coincide (which would make "no km-apart warning" true
    trivially, even without the routing this section exists to test).
    """
    time = pd.date_range("2014-06-01", "2014-09-01", freq="MS") + pd.Timedelta(days=14)
    lat = np.arange(45.5, 55.5) + lat_off
    lon = np.arange(-150.5, -139.5) + lon_off
    values = 100 + np.random.RandomState(seed).rand(time.size, lat.size, lon.size)
    da = xr.DataArray(
        values,
        coords={"time": time, "lat": lat, "lon": lon},
        dims=("time", "lat", "lon"),
        name=MODEL_VAR,
    )
    da.attrs["units"] = "degC"
    return da.to_dataset()


def _coarse_grid(lon_pts, lat_pts, seed=1):
    """A widely-spaced rectilinear grid -- for the bbox-pad-misses-it case."""
    time = pd.date_range("2014-06-01", "2014-09-01", freq="MS") + pd.Timedelta(days=14)
    lon, lat = np.array(lon_pts, dtype=float), np.array(lat_pts, dtype=float)
    values = 100 + np.random.RandomState(seed).rand(time.size, lat.size, lon.size)
    da = xr.DataArray(
        values,
        coords={"time": time, "lat": lat, "lon": lon},
        dims=("time", "lat", "lon"),
        name=MODEL_VAR,
    )
    da.attrs["units"] = "degC"
    return da.to_dataset()


def _curvilinear_monthly(nx: int = 11, ny: int = 10, seed: int = 5):
    """A ROMS-shaped reference: 2-D lon_rho/lat_rho, with a time axis alongside."""
    time = pd.date_range("2014-06-01", "2014-09-01", freq="MS") + pd.Timedelta(days=14)
    lon_1d = np.arange(-150.5, -150.5 + nx)
    lat_1d = np.arange(45.5, 45.5 + ny)
    lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
    values = 200 + np.random.RandomState(seed).rand(time.size, ny, nx)
    da = xr.DataArray(
        values,
        dims=("time", "eta_rho", "xi_rho"),
        coords={
            "time": time,
            "lon_rho": (("eta_rho", "xi_rho"), lon_2d),
            "lat_rho": (("eta_rho", "xi_rho"), lat_2d),
        },
        name=MODEL_VAR,
        attrs={"units": "degC"},
    )
    return da.to_dataset()


def _model_comparison(monkeypatch, *, test, reference, metadata=None, **kwargs):
    """Build a ``Comparison`` between two gridded, featureType-less sources.

    Patches ``osk.read``/``catalog.resolve`` rather than ``comparison.prepare_source``
    (contrast ``station_lanes`` above) -- the point of this section is that a point
    select's routing through ``operators.select``/``align.sample_at`` actually runs,
    which a stubbed ``prepare_source`` would skip entirely.

    ``metadata`` optionally supplies ``{"run_baseline": {...}, "run_new": {...}}`` for
    the ``catalog.resolve(...).metadata`` a source reports -- read by
    :func:`ocean_skill.comparison._feature_type`/``_domain_of``/``_time_coverage_of``,
    so this is what lets a reference-metadata-derived narrowing run through the real
    ``align()`` machinery here, the same way an explicit point ``select`` already does.
    """
    from types import SimpleNamespace

    import ocean_skill as osk
    from ocean_skill import catalog
    from ocean_skill.comparison import Comparison

    grids = {"run_new": test, "run_baseline": reference}
    meta = metadata or {}
    monkeypatch.setattr(osk, "read", lambda name, **kw: grids[name])
    monkeypatch.setattr(
        catalog, "resolve", lambda name: SimpleNamespace(metadata=meta.get(name, {}))
    )
    return Comparison(
        reference="run_baseline",
        test="run_new",
        variable=MODEL_VAR,
        cache=False,
        **kwargs,
    )


def test_a_shared_point_select_implies_over_time(monkeypatch):
    """Two gridded runs, a point select, no over= -- inferred rather than refused."""
    c = _model_comparison(
        monkeypatch,
        test=_offset_grid(lat_off=0.4, lon_off=0.4, seed=1),
        reference=_offset_grid(seed=2),
        select=POINT_SELECT,
    )
    assert c.over == "time"
    assert c.over_reason == "the select narrows the reference to one position"
    assert c.family == "series"


def test_the_routed_point_select_samples_co_located_with_no_distance_warning(
    monkeypatch,
):
    """The primary case this session exists for: no spurious 'km apart' warning.

    Narrowing both lanes independently (the pre-existing two-point branch) would
    compare the test's nearest cell to *its own* request against the reference's
    nearest cell to the same request -- two different, unrelated positions. Routing
    instead samples the still-gridded test at the reference's own resolved cell, so
    the pair is genuinely co-located.
    """
    c = _model_comparison(
        monkeypatch,
        test=_offset_grid(lat_off=0.4, lon_off=0.4, seed=1),
        reference=_offset_grid(seed=2),
        select=POINT_SELECT,
    )
    with warnings.catch_warnings(record=True) as log:
        warnings.simplefilter("always")
        aligned = c.align()
    assert not any("km apart" in str(w.message) for w in log)
    assert aligned.attrs["point_method"] == "nearest"
    assert "nearest_distance_km" in aligned.attrs
    # the reference narrowed to its own snapped cell, not the raw request
    assert (float(aligned["lon"]), float(aligned["lat"])) == (-144.5, 50.5)
    # ...and the test was sampled there, on its own (different) grid
    assert (float(aligned["test_lon"]), float(aligned["test_lat"])) == (-144.1, 50.9)


def test_a_pair_spec_of_two_points_still_warns_about_the_distance(monkeypatch):
    """The deliberate escape hatch -- two real positions -- keeps its own warning."""
    c = _model_comparison(
        monkeypatch,
        test=_offset_grid(lat_off=0.4, lon_off=0.4, seed=1),
        reference=_offset_grid(seed=2),
        select={"test": POINT_SELECT, "reference": POINT_SELECT},
    )
    with pytest.warns(UserWarning, match="km apart"):
        c.align()


def test_explicit_over_time_routes_the_same_way_as_the_inference(monkeypatch):
    inferred = _model_comparison(
        monkeypatch,
        test=_offset_grid(lat_off=0.4, lon_off=0.4, seed=1),
        reference=_offset_grid(seed=2),
        select=POINT_SELECT,
    )
    explicit = _model_comparison(
        monkeypatch,
        test=_offset_grid(lat_off=0.4, lon_off=0.4, seed=1),
        reference=_offset_grid(seed=2),
        select=POINT_SELECT,
        over="time",
    )
    assert explicit.over_reason == "over= as asked"
    assert explicit._cache_key == inferred._cache_key


def test_a_point_select_with_a_time_collapsing_aggregate_does_not_imply_over(
    monkeypatch,
):
    """Nothing would be left for a line to run along, so inference stays off."""
    c = _model_comparison(
        monkeypatch,
        test=_offset_grid(lat_off=0.4, lon_off=0.4, seed=1),
        reference=_offset_grid(seed=2),
        select=POINT_SELECT,
        aggregate={"time": "mean"},
    )
    assert c.over is None
    assert c.over_reason == "the reference is gridded"


def test_a_curvilinear_reference_point_select_is_a_series(monkeypatch):
    c = _model_comparison(
        monkeypatch,
        test=_offset_grid(lat_off=0.4, lon_off=0.4, seed=1),
        reference=_curvilinear_monthly(),
        select={
            "lon_rho": -144.3,
            "lat_rho": 50.0,
            "time": slice("2014-06", "2014-08"),
        },
    )
    with warnings.catch_warnings(record=True) as log:
        warnings.simplefilter("always")
        c.align()
    assert c.is_series
    assert not any("matched no axis" in str(w.message) for w in log)


def test_a_coarse_test_grid_is_windowed_by_cells_not_missed_by_a_degree_pad(
    monkeypatch,
):
    """A degree-based pad around one point could miss a coarse grid's nearest cell
    entirely; the cell-based point window that replaced it centres on the true
    nearest cell directly, so there is no "no overlap" miss (and no unnarrowed
    re-read) to fall back from here -- one read of the (small) test grid.
    """
    c = _model_comparison(
        monkeypatch,
        test=_coarse_grid([-160.0, -150.0, -140.0, -130.0], [40.0, 50.0, 60.0]),
        reference=_offset_grid(seed=2),
        select=POINT_SELECT,
    )
    import ocean_skill as osk

    reads = {"run_new": 0}
    already_patched_read = osk.read

    def counting_read(name, **kw):
        if name in reads:
            reads[name] += 1
        return already_patched_read(name, **kw)

    monkeypatch.setattr(osk, "read", counting_read)
    with warnings.catch_warnings(record=True) as log:
        warnings.simplefilter("always")
        aligned = c.align()
    messages = [str(w.message) for w in log]
    assert not any("stale" in m or "no overlap" in m for m in messages)
    assert float(aligned["test_lon"]) == -140.0
    assert reads["run_new"] == 1


def _point_station(*, lon=STATION[0], lat=STATION[1], seed=3):
    """A single-position reference shaped like this section's other sources -- a
    ``Dataset``, ``MODEL_VAR``-named -- so the real, unstubbed ``align()`` pipeline's
    own ``point_of`` finds it, the way it would a real mooring.
    """
    time = pd.date_range("2014-06-01", "2014-09-01", freq="MS") + pd.Timedelta(days=14)
    values = 100 + np.random.RandomState(seed).rand(time.size)
    da = xr.DataArray(values, coords={"time": time}, dims="time", name=MODEL_VAR)
    da = da.assign_coords(lon=lon, lat=lat)
    da.attrs["units"] = "degC"
    return da.to_dataset()


def test_a_stale_metadata_position_is_corrected_by_the_post_read_verification(
    monkeypatch,
):
    """A wildly wrong catalog position must not leave the test lane windowed
    around the wrong place.

    A degenerate bbox derived from the reference's catalog metadata windows the
    test lane by cells, centred on *that* position -- so a stale metadata position
    no longer raises "no overlap" (:func:`ocean_skill.align.subset_to_bbox`'s point
    crop never does). :meth:`Comparison._verify_point_window` is what still
    guarantees the real, unstubbed station branch samples the reference's *actual*
    (not the metadata's claimed) position: it checks the window against
    ``align.point_of(reference)`` and re-reads the test lane around it when the
    two disagree by more than the window can absorb.

    The test grid here is deliberately *larger* than the point window
    (:data:`ocean_skill.align.POINT_WINDOW_CELLS`) on both axes -- a grid smaller
    than the window (as this test used before) is windowed to its whole self
    regardless of where the window is centred, which would let this test keep
    passing even with the verification step removed. Here the window built
    around the stale position (-131, 41) provably excludes the true nearest cell
    to the station's real position, so this fails without the fix: (-136, 46) is
    the nearest cell *inside* that stale window, not the true nearest cell
    (-144, 50) on the whole grid.
    """
    c = _model_comparison(
        monkeypatch,
        test=_coarse_grid(np.arange(-160.0, -129.0), np.arange(40.0, 61.0)),
        reference=_point_station(),
        metadata={
            "run_baseline": {
                "featureType": "timeSeries",
                "geospatial_lon_min": -131.0,
                "geospatial_lon_max": -131.0,
                "geospatial_lat_min": 41.0,
                "geospatial_lat_max": 41.0,
            }
        },
    )
    match = "catalog position is .* from its data's actual"
    with pytest.warns(UserWarning, match=match):
        aligned = c.align()  # must not raise "no overlap"
    assert (float(aligned["test_lon"]), float(aligned["test_lat"])) == (-144.0, 50.0)


def test_an_explicit_point_select_beats_the_metadata_position(monkeypatch):
    """Where to sample is what the caller asked for, not a guess from the catalog.

    A shared point ``select`` already routes the test lane's crop (see
    ``test_the_routed_point_select_samples_co_located_with_no_distance_warning``);
    a reference whose catalog metadata *also* claims a (here, deliberately different
    and otherwise plausible) position must not be allowed to override it -- the
    caller's explicit choice wins spatially, exactly as :meth:`Comparison.align`
    documents.
    """
    c = _model_comparison(
        monkeypatch,
        test=_offset_grid(lat_off=0.4, lon_off=0.4, seed=1),
        reference=_offset_grid(seed=2),
        select=POINT_SELECT,
        metadata={
            "run_baseline": {
                "featureType": "timeSeries",
                "geospatial_lon_min": -141.0,
                "geospatial_lon_max": -141.0,
                "geospatial_lat_min": 54.0,
                "geospatial_lat_max": 54.0,
            }
        },
    )
    with warnings.catch_warnings(record=True) as log:
        warnings.simplefilter("always")
        aligned = c.align()
    assert not any("km apart" in str(w.message) for w in log)
    # sampled at the caller's own point -- not the reference's stale metadata claim
    assert (float(aligned["test_lon"]), float(aligned["test_lat"])) == (-144.1, 50.9)


def test_a_routed_point_against_a_much_coarser_reference_is_still_verified(
    monkeypatch,
):
    """A routed point select shares the same verification a derived one gets.

    The reference is far coarser than the test grid, so its nearest cell to the
    routed point (-144.3, 50.0) snaps to (-145.8, 50.0) -- ~107 km away, more
    than the test lane's own point window can absorb (~79 km, three of the
    test grid's own fine 0.2-degree cells) but still inside the padded region
    the reference lane itself is cropped to (so the reference lane's own read
    still succeeds). Without :meth:`Comparison._verify_point_window` the test
    lane would stay windowed around the raw request and sample its own
    window's edge instead of the cell nearest the reference's actual position.
    """
    fine_lon = np.arange(-150.05, -138.05, 0.2)
    fine_lat = np.arange(45.05, 55.05, 0.2)
    c = _model_comparison(
        monkeypatch,
        test=_coarse_grid(fine_lon, fine_lat, seed=1),
        reference=_coarse_grid([-145.8, -100.0], [50.0], seed=6),
        select=POINT_SELECT,
    )
    with pytest.warns(UserWarning, match="snapped to a different cell"):
        aligned = c.align()
    assert (float(aligned["lon"]), float(aligned["lat"])) == (-145.8, 50.0)
    assert abs(float(aligned["test_lon"]) - (-145.8)) < 0.15
    assert abs(float(aligned["test_lat"]) - 50.0) < 0.15


def test_a_non_degenerate_derived_box_still_warns_before_the_unnarrowed_retry(
    monkeypatch,
):
    """A trajectory reference's declared *extent* (not a point) that misses the
    test grid entirely still falls back to reading the whole test lane -- but
    now says so, where before this fix it did silently.
    """
    c = _model_comparison(
        monkeypatch,
        # Same grid on both sides (no offset) so align() differences the two
        # fields directly rather than regridding -- this test is only about the
        # retry+warning during the test lane's *own* prep, not the regrid step.
        test=_offset_grid(seed=1),
        reference=_offset_grid(seed=2),
        metadata={
            "run_baseline": {
                "featureType": "trajectory",
                "geospatial_lon_min": -10.0,
                "geospatial_lon_max": -9.9,
                "geospatial_lat_min": 0.0,
                "geospatial_lat_max": 0.1,
            }
        },
        aggregate={"time": "mean"},
    )
    with pytest.warns(UserWarning, match="does not overlap .* grid"):
        c.align()


def test_the_cache_key_changes_when_the_reference_window_changes(monkeypatch):
    """A catalog rebuild that widens the reference's declared record must not serve
    a stale aligned pair back from a warm cache.
    """

    def key_for(max_time):
        meta = {
            "run_baseline": {
                "featureType": "timeSeries",
                "minTime": "2014-06-01T00:00:00Z",
                "maxTime": max_time,
            }
        }
        c = _model_comparison(
            monkeypatch,
            test=_offset_grid(lat_off=0.4, lon_off=0.4, seed=1),
            reference=_offset_grid(seed=2),
            select=POINT_SELECT,
            metadata=meta,
        )
        # Read the key *before* the next call's monkeypatch.setattr reassigns
        # catalog.resolve out from under it -- _cache_key resolves the reference's
        # metadata lazily, at access time, not at construction time.
        return c._cache_key

    a = key_for("2014-08-01T00:00:00Z")
    b = key_for("2014-08-01T00:00:00Z")
    widened = key_for("2015-01-01T00:00:00Z")
    assert a == b
    assert a != widened


def test_the_cache_key_is_stable_and_differs_from_a_pair_spec(monkeypatch):
    def make(select):
        return _model_comparison(
            monkeypatch,
            test=_offset_grid(lat_off=0.4, lon_off=0.4, seed=1),
            reference=_offset_grid(seed=2),
            select=select,
        )

    routed_a, routed_b = make(POINT_SELECT), make(POINT_SELECT)
    paired = make({"test": POINT_SELECT, "reference": POINT_SELECT})
    assert routed_a._cache_key == routed_b._cache_key
    assert routed_a._cache_key != paired._cache_key


# --------------------------------------------- a box mean implies over="time" too


#: A box neither grid is centred on, so each lane's own area mean is a non-trivial
#: reduction of a real subset, not the grid's own bounding box in disguise.
BOX_SELECT = {
    "lon": {"min": -149.0, "max": -142.0},
    "lat": {"min": 47.0, "max": 53.0},
    "time": slice("2014-06", "2014-08"),
}
BOX_MEAN = {"lat": "mean", "lon": "mean"}


def test_a_shared_box_mean_implies_over_time(monkeypatch):
    """An area-weighted spatial mean is exactly as reduced as a station -- there
    is no map left to draw either way -- so over="time" is inferred the same way
    a point select already implies it.
    """
    c = _model_comparison(
        monkeypatch,
        test=_offset_grid(lat_off=0.4, lon_off=0.4, seed=1),
        reference=_offset_grid(seed=2),
        select=BOX_SELECT,
        aggregate=BOX_MEAN,
    )
    assert c.over == "time"
    assert c.over_reason == "the aggregate collapses space to one box mean"
    assert c.family == "series"


def test_a_shared_box_mean_lands_both_lanes_on_the_same_position_with_no_distance_warning(
    monkeypatch,
):
    """Both lanes reduce the *same requested box* to its own midpoint, independent
    of their own grids -- unlike a point select's nearest-cell routing, there is no
    grid-dependent offset for the two to disagree about.
    """
    c = _model_comparison(
        monkeypatch,
        test=_offset_grid(lat_off=0.4, lon_off=0.4, seed=1),
        reference=_offset_grid(seed=2),
        select=BOX_SELECT,
        aggregate=BOX_MEAN,
    )
    with warnings.catch_warnings(record=True) as log:
        warnings.simplefilter("always")
        aligned = c.align()
    assert not any("km apart" in str(w.message) for w in log)
    assert float(aligned["lon"]) == pytest.approx(-145.5)
    assert float(aligned["lat"]) == pytest.approx(50.0)
    assert aligned["reference"].attrs["region"] == [-149.0, 47.0, -142.0, 53.0]


def test_a_time_collapsing_aggregate_alongside_a_box_mean_does_not_imply_over(
    monkeypatch,
):
    """Collapsing time too leaves nothing for a line to run along."""
    c = _model_comparison(
        monkeypatch,
        test=_offset_grid(lat_off=0.4, lon_off=0.4, seed=1),
        reference=_offset_grid(seed=2),
        select=BOX_SELECT,
        aggregate={**BOX_MEAN, "time": "mean"},
    )
    assert c.over is None
    assert c.over_reason == "the reference is gridded"


def test_a_pair_spec_of_two_different_boxes_still_warns_about_the_distance(
    monkeypatch,
):
    """The escape hatch mirrors the point-select one: two real, different boxes
    keep their own distance warning rather than being silently forced together.
    """
    other_box = {**BOX_SELECT, "lon": {"min": -148.0, "max": -145.0}}
    c = _model_comparison(
        monkeypatch,
        test=_offset_grid(lat_off=0.4, lon_off=0.4, seed=1),
        reference=_offset_grid(seed=2),
        select={"test": other_box, "reference": BOX_SELECT},
        aggregate=BOX_MEAN,
    )
    with pytest.warns(UserWarning, match="km apart"):
        c.align()
