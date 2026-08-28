"""Tests for :mod:`ocean_skill.plot.map_metrics`: interpolated maps for stations.

Three layers, tested separately: :func:`interpolate_records` (scattered values -> a
masked surface, via verde), :func:`build_items` (records -> the ``skill_map`` family's
item shape, from either a ``ComparisonSet`` or a plain table), and the station-dot
overlay both renderers draw on top of an interpolated (or any) skill map.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

pytest.importorskip("verde")
pytest.importorskip("pyproj")

from ocean_skill.plot.map_metrics import (
    _central_longitude,
    build_items,
    interpolate_records,
)

# --- scattered records used across several tests --------------------------------------


def _records(n: int = 24, seed: int = 0) -> pd.DataFrame:
    """A cluster of stations with a known, smooth ``bias`` gradient in longitude."""
    rng = np.random.default_rng(seed)
    lon = rng.uniform(-153.0, -151.0, n)
    lat = rng.uniform(59.0, 60.0, n)
    return pd.DataFrame(
        {
            "lon": lon,
            "lat": lat,
            "bias": lon - lon.mean(),  # smooth, mean-zero across the cluster
            "corr": rng.uniform(0.5, 0.99, n),
            "n": np.full(n, 50.0),
            "reference": [f"mooring_{i}" for i in range(n)],
        }
    )


# --- interpolate_records -------------------------------------------------------------


def test_recovers_a_smooth_gradient_near_the_data_centroid():
    df = _records()
    skill = interpolate_records(df, ("bias",))

    lon, lat = skill["lon"].to_numpy(), skill["lat"].to_numpy()
    d2 = (lon - df["lon"].mean()) ** 2 + (lat - df["lat"].mean()) ** 2
    iy, ix = np.unravel_index(np.argmin(d2), d2.shape)
    assert skill["bias"].to_numpy()[iy, ix] == pytest.approx(0.0, abs=0.3)


def test_output_carries_one_2d_variable_per_requested_metric():
    df = _records()
    skill = interpolate_records(df, ("bias", "corr"))

    assert list(skill.data_vars) == ["bias", "corr"]
    for name in ("bias", "corr"):
        assert skill[name].dims == ("y", "x")
        assert skill[name].attrs["long_name"]


def test_cells_far_from_every_station_are_masked():
    df = _records()
    skill = interpolate_records(df, ("bias",))

    assert not np.isfinite(skill["bias"].to_numpy()).all(), (
        "the fixture pads the grid well past the station cluster; some corner "
        "should be far enough to mask"
    )
    assert np.isfinite(skill["bias"].to_numpy()).any()


def test_a_caller_supplied_grid_is_used_and_masked_by_its_ocean_mask():
    df = _records()
    lon = df["lon"].to_numpy()
    lat = df["lat"].to_numpy()
    glon, glat = np.meshgrid(
        np.linspace(lon.min() - 0.2, lon.max() + 0.2, 6),
        np.linspace(lat.min() - 0.2, lat.max() + 0.2, 6),
    )
    ocean = np.ones_like(glon, dtype=bool)
    ocean[0, :] = False  # a "land" row

    skill = interpolate_records(df, ("bias",), grid=(glon, glat, ocean))

    assert skill["bias"].shape == glon.shape
    assert np.isnan(skill["bias"].to_numpy()[0, :]).all(), "masked by the ocean mask"


@pytest.mark.parametrize(
    "lon_key,lat_key", [("lon", "lat"), ("station_lon", "station_lat")]
)
def test_either_position_column_pair_is_accepted(lon_key, lat_key):
    df = _records().rename(columns={"lon": lon_key, "lat": lat_key})
    skill = interpolate_records(df, ("bias",))
    assert np.isfinite(skill["bias"].to_numpy()).any()


def test_duplicate_positions_are_pooled_to_their_median_with_a_warning():
    df = _records(n=8)
    dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # one exact repeat

    with pytest.warns(UserWarning, match="shared a position"):
        interpolate_records(dup, ("bias",))


def test_widely_uneven_record_lengths_warn():
    df = _records(n=10)
    df["n"] = [10.0] * 5 + [1000.0] * 5

    with pytest.warns(UserWarning, match="record lengths range"):
        interpolate_records(df, ("bias",))


def test_similar_record_lengths_do_not_warn():
    df = _records(n=10)
    df["n"] = np.linspace(45, 55, 10)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        interpolate_records(df, ("bias",))
    assert not any("record lengths" in str(w.message) for w in caught)


def test_a_missing_metric_column_names_what_is_available():
    df = _records()
    with pytest.raises(ValueError, match=r"no \['no_such_metric'\]"):
        interpolate_records(df, ("no_such_metric",))


def test_a_missing_position_raises():
    df = pd.DataFrame({"bias": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="no station position found"):
        interpolate_records(df, ("bias",))


def test_a_null_position_raises():
    df = _records(n=5)
    df.loc[2, "lon"] = np.nan
    with pytest.raises(ValueError, match="carry no position"):
        interpolate_records(df, ("bias",))


def test_too_few_stations_falls_back_to_a_fixed_damping_spline():
    df = _records(n=3)
    with pytest.warns(UserWarning, match="too few to"):
        skill = interpolate_records(df, ("bias",))
    assert np.isfinite(skill["bias"].to_numpy()).any()


def test_central_longitude_is_antimeridian_safe():
    """A plain mean of 178/179/-179/-178/179.5 would land on the wrong side of Earth."""
    lon = np.array([178.0, 179.0, -179.0, -178.0, 179.5])
    center = _central_longitude(lon)
    # every input point should be within a small arc of the computed centre --
    # a naive mean instead puts the centre roughly 180 degrees away from all of them
    diffs = np.abs(((lon - center + 180) % 360) - 180)
    assert (diffs < 5).all()


# --- build_items -------------------------------------------------------------------


def test_build_items_from_a_plain_dataframe():
    df = _records(n=12)
    items = build_items(df, metrics=("bias", "corr"), grid="regular")

    assert len(items) == 1
    item = items[0]
    assert list(item["skill"].data_vars) == ["bias", "corr"]
    assert "row_label" not in item
    assert item["stations"]["lon"].shape == (12,)
    assert item["stations"]["names"] == df["reference"].tolist()
    assert set(item["stations"]["values"]) == {"bias", "corr"}


def test_rows_builds_one_labelled_item_per_entry():
    winter, summer = _records(seed=1), _records(seed=2)
    items = build_items(
        rows={"DJF": winter, "JJA": summer}, metrics=("bias",), grid="regular"
    )

    assert [item["row_label"] for item in items] == ["DJF", "JJA"]


def test_neither_data_nor_rows_is_an_error():
    with pytest.raises(ValueError, match="nothing to map"):
        build_items()


def test_grid_option_rejects_an_unknown_value():
    with pytest.raises(ValueError, match="grid="):
        build_items(_records(), grid="curvilinear")


# --- the station-dot overlay, in both renderers -------------------------------------


def _scatter_collections(ax):
    """Every station-dot ``PathCollection`` drawn on one matplotlib axes."""
    import matplotlib.collections as mcoll

    return [c for c in ax.collections if isinstance(c, mcoll.PathCollection)]


def test_the_overlay_draws_one_dot_per_station_in_both_renderers():
    from ocean_skill.plot.registry import render
    from ocean_skill.plot.spec import PlotSpec

    df = _records(n=9)
    items = build_items(df, metrics=("bias", "corr"), grid="regular")
    spec = PlotSpec(family="skill_map", items=items, options={})

    fig = render(spec)
    scatters = {
        ax.get_title(): _scatter_collections(ax) for ax in fig.axes if ax.get_title()
    }
    assert set(scatters) == {"bias", "corr"}
    for title, collections in scatters.items():
        assert len(collections) == 1, title
        assert collections[0].get_offsets().shape == (9, 2), title

    import holoviews as hv

    obj = render(spec, renderer="holoviews")
    points = [e for e in obj.traverse() if isinstance(e, hv.Points)]
    assert len(points) == 2
    assert all(len(p.data) == 9 for p in points)


def test_an_item_without_stations_draws_exactly_as_before():
    """The overlay is additive: a plain skill_map item is unaffected."""
    from ocean_skill.plot.registry import render
    from ocean_skill.plot.spec import PlotSpec

    field = xr.DataArray(
        np.linspace(-1, 1, 80).reshape(8, 10),
        dims=("lat", "lon"),
        coords={"lat": np.linspace(58, 60, 8), "lon": np.linspace(-153, -151, 10)},
        attrs={"units": "degC"},
    )
    item = {"skill": xr.Dataset({"bias": field}), "metric_names": ("bias",)}
    fig = render(PlotSpec(family="skill_map", items=[item], options={}))

    for ax in fig.axes:
        assert not _scatter_collections(ax)


def test_a_big_skill_map_panel_is_rasterized_and_a_small_one_is_not():
    """Check that a big skill_map panel rasterizes like a big field_row does.

    A metric map is a curvilinear mesh too, and hits the same per-cell Python loop a
    big field_row does without rasterize="auto".
    """
    from ocean_skill.plot.holoviews_renderer import RASTERIZE_ABOVE_CELLS
    from ocean_skill.plot.registry import render
    from ocean_skill.plot.spec import PlotSpec

    def skill_map(shape):
        ny, nx = shape
        field = xr.DataArray(
            np.linspace(-1, 1, ny * nx).reshape(ny, nx),
            dims=("lat", "lon"),
            coords={"lat": np.linspace(58, 60, ny), "lon": np.linspace(-153, -151, nx)},
            attrs={"units": "degC"},
        )
        item = {"skill": xr.Dataset({"bias": field}), "metric_names": ("bias",)}
        return render(
            PlotSpec(family="skill_map", items=[item], options={}),
            renderer="holoviews",
        )

    def kinds(obj):
        return [type(n).__name__ for n in obj.traverse()]

    small, big = (8, 10), (400, 550)
    assert small[0] * small[1] < RASTERIZE_ABOVE_CELLS < big[0] * big[1]
    assert "QuadMesh" in kinds(skill_map(small))
    assert "Image" in kinds(skill_map(big))


# --- end to end: real Comparisons through ComparisonSet.map_metrics() ----------------


def _served_by(lanes: dict):
    """A ``prepare_source`` stand-in reading from an in-memory lane dict."""
    return lambda source, *a, **k: (lanes[source], None)


def _mooring(lon: float, lat: float, seed: int) -> xr.DataArray:
    time = pd.date_range("2015-01-01", periods=24 * 40, freq="h")
    rng = np.random.default_rng(seed)
    values = 8.0 + np.sin(np.arange(time.size) / 97.0) + rng.normal(0, 0.05, time.size)
    da = xr.DataArray(
        values, coords={"time": time}, dims="time", name="sea_water_temperature"
    )
    da = da.assign_coords(lon=lon, lat=lat, depth=1.0)
    da.attrs["units"] = "degC"
    return da


def _model(lon: float, lat: float, offset: float, seed: int) -> xr.DataArray:
    time = pd.date_range("2015-01-01", periods=24 * 40, freq="h")
    rng = np.random.default_rng(seed + 100)
    signal = np.sin(np.arange(time.size) / 97.0) + rng.normal(0, 0.05, time.size)
    da = xr.DataArray(
        8.0 + offset + signal,
        coords={"time": time},
        dims="time",
        name="sea_water_temperature",
    )
    da = da.assign_coords(lon=lon, lat=lat)
    da.attrs["units"] = "degC"
    return da


#: name, lon, lat, model bias offset
_STATIONS = (
    ("mooring_a", -153.0, 59.0, 0.2),
    ("mooring_b", -152.0, 59.5, -0.3),
    ("mooring_c", -151.5, 59.8, 0.05),
)


@pytest.fixture
def mooring_set(monkeypatch):
    """A real :class:`ComparisonSet` of three stubbed station comparisons.

    Mirrors ``tests/test_series.py``'s ``station_lanes``: a comparison is built and
    aligned without a catalog or a network, so each station's ``station_lon``/
    ``station_lat`` (the thing this whole feature reads) comes from the real
    alignment pipeline, not a hand-inserted attribute.
    """
    import ocean_skill.comparison as comparison_module
    from ocean_skill.comparison import Comparison, ComparisonSet

    lanes = {}
    for i, (name, lon, lat, offset) in enumerate(_STATIONS):
        lanes[name] = _mooring(lon, lat, seed=i)
        lanes[f"{name}_model"] = _model(lon, lat, offset, seed=i)
    monkeypatch.setattr(comparison_module, "prepare_source", _served_by(lanes))
    monkeypatch.setattr(comparison_module, "_domain_of", lambda name: None)
    monkeypatch.setattr(
        comparison_module,
        "_feature_type",
        lambda source: "grid" if source.endswith("_model") else "timeSeries",
    )

    comparisons = []
    for name, *_rest in _STATIONS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            c = Comparison(
                reference=name,
                test=f"{name}_model",
                variable="sea_water_temperature",
                cache=False,
            )
            c.align()
        comparisons.append(c)
    return ComparisonSet(comparisons)


def test_comparisonset_map_metrics_draws_every_station(mooring_set):
    import matplotlib.collections as mcoll

    fig = mooring_set.map_metrics(grid="regular", metrics=("bias", "corr"))

    scatters = [
        c
        for ax in fig.axes
        for c in ax.collections
        if isinstance(c, mcoll.PathCollection)
    ]
    assert scatters and all(c.get_offsets().shape == (3, 2) for c in scatters)


def _grid_field(offset: float = 0.0) -> xr.DataArray:
    return xr.DataArray(
        8.0 + offset + np.zeros((4, 5)),
        dims=("lat", "lon"),
        coords={
            "lat": np.linspace(58.5, 60.5, 4),
            "lon": np.linspace(-153.5, -150.5, 5),
        },
        name="sea_water_temperature",
        attrs={"units": "degC"},
    )


def test_comparisonset_map_metrics_skips_non_station_comparisons(
    mooring_set, monkeypatch
):
    """A set that mixes in a genuinely gridded comparison still maps the stations."""
    import ocean_skill.comparison as comparison_module

    lanes = {"grid_ref": _grid_field(), "grid_test": _grid_field(0.4)}
    monkeypatch.setattr(comparison_module, "prepare_source", _served_by(lanes))
    monkeypatch.setattr(comparison_module, "_domain_of", lambda name: None)
    monkeypatch.setattr(comparison_module, "_feature_type", lambda source: "grid")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid_comparison = comparison_module.Comparison(
            reference="grid_ref",
            test="grid_test",
            variable="sea_water_temperature",
            cache=False,
        )
        grid_comparison.align()
    assert not grid_comparison.is_series, "the fixture must actually be gridded"

    mixed = comparison_module.ComparisonSet([*mooring_set.comparisons, grid_comparison])
    with pytest.warns(UserWarning, match="not a place through time"):
        fig = mixed.map_metrics(grid="regular", metrics=("bias",))

    scatters = [c for ax in fig.axes for c in _scatter_collections(ax)]
    assert all(c.get_offsets().shape == (3, 2) for c in scatters), (
        "only the three stations should reach the surface"
    )


def test_rows_from_different_test_sources_falls_back_with_a_warning(mooring_set):
    """A facet built from two different test runs has no single grid to pick."""
    import ocean_skill.comparison as comparison_module

    a, b = mooring_set.comparisons[0], mooring_set.comparisons[1]
    rows = {
        "row1": comparison_module.ComparisonSet([a]),
        "row2": comparison_module.ComparisonSet([b]),
    }
    with pytest.warns(UserWarning, match="different test"):
        items = build_items(rows=rows, metrics=("bias",))

    assert len(items) == 2
