"""Comparisons scored *over* an axis: metric maps instead of a test/reference row.

``compare(..., over="time")`` keeps the axis the three "a comparison is one map" gates
normally refuse, matches the two lanes along it, and computes every metric cell by cell.
What is worth testing is the seam: that the axis really survives, that a *second* one is
still refused, that the overall number and the maps describe the same cells, and that
a plain comparison of the same pair is refused exactly as before.

The lanes are stubbed (as ``tests/test_facet.py`` does) so these exercise the pipeline
rather than the catalog. Anything that regrids needs xesmf.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

pytest.importorskip("xesmf")

NITRATE = "mole_concentration_of_nitrate_in_sea_water"


def _field(time, seed=0, lat=None, lon=None):
    lat = np.linspace(18, 30, 9) if lat is None else lat
    lon = np.linspace(-97, -82, 12) if lon is None else lon
    rng = np.random.default_rng(seed)
    swing = np.sin(np.linspace(0, 4, len(time)))[:, None, None]
    return xr.DataArray(
        swing + rng.normal(5.0, 0.5, (len(time), len(lat), len(lon))),
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": lat, "lon": lon},
        attrs={"units": "mmol m-3"},
    )


@pytest.fixture
def lanes():
    """Hourly model output and a daily reference on a coarser grid."""
    return {
        "model": _field(pd.date_range("2012-01-01", periods=24 * 30, freq="h")),
        "sat": _field(
            pd.date_range("2012-01-01", periods=30, freq="D"),
            7,
            lat=np.linspace(19, 29, 6),
            lon=np.linspace(-96, -83, 8),
        ),
    }


@pytest.fixture
def stub(monkeypatch, lanes):
    """Serve the two lanes in place of real sources, and draw no domain box."""
    import ocean_skill.comparison as comparison

    monkeypatch.setattr(
        comparison, "prepare_source", lambda source, *a, **k: (lanes[source], None)
    )
    monkeypatch.setattr(comparison, "_domain_of", lambda name: None)
    return lanes


def _scored(*, quiet: bool = True, **kwargs):
    """Build and align a scored comparison.

    ``quiet`` suppresses the warnings every one of these emits by construction — a
    reference with no declared period, a 30-step overlap — so a test asserting on a
    *particular* warning turns it off rather than fishing that one out of the noise.
    """
    from ocean_skill.comparison import Comparison

    with warnings.catch_warnings():
        if quiet:
            warnings.simplefilter("ignore")
        comparison = Comparison(
            reference="sat",
            test="model",
            variable=NITRATE,
            over="time",
            cache=False,
            **kwargs,
        )
        comparison.align()
    return comparison


# --- the pipeline --------------------------------------------------------------------


def test_the_scored_axis_survives_alignment_named_as_the_reference_names_it(stub):
    aligned = _scored().aligned
    assert aligned.sizes["time"] == 30, "the reference's 30 daily steps"
    assert aligned.sizes["lat"] == 6, "on the reference's grid"
    assert aligned.attrs["scored_over"] == "time"
    assert aligned.attrs["match_method"] == "mean"
    assert aligned.attrs["n_matched"] == 30


def test_the_matching_and_the_regrid_both_go_on_the_record(stub):
    """Two choices a reader of the numbers is entitled to see."""
    attrs = _scored().aligned.attrs
    assert attrs["regrid_method"] == "conservative_normed"
    assert "match_reason" in attrs and "averaged into" in attrs["match_reason"]
    assert attrs["steps_per_bin"] == 24


def test_coverage_is_computed_once_for_a_static_mask(stub):
    """A model land mask does not move, so 30 identical regrids buy nothing."""
    aligned = _scored().aligned
    assert aligned.attrs["coverage_time_invariant"] is True
    assert "time" not in aligned["coverage"].dims


def test_a_moving_mask_pays_for_itself_and_says_so(monkeypatch, lanes):
    import ocean_skill.comparison as comparison

    gappy = lanes["model"].copy()
    # a whole *bin* missing at one cell, not a few hours of it: the bin average skips
    # NaNs, so anything shorter than a day is filled in by the averaging itself
    gappy[:24, 4, 5] = np.nan
    monkeypatch.setattr(
        comparison,
        "prepare_source",
        lambda source, *a, **k: ({"model": gappy, "sat": lanes["sat"]}[source], None),
    )
    monkeypatch.setattr(comparison, "_domain_of", lambda name: None)
    with pytest.warns(UserWarning, match="valid cells change along"):
        aligned = _scored(quiet=False).aligned
    assert aligned.attrs["coverage_time_invariant"] is False


def test_a_second_surviving_axis_is_still_refused(stub, lanes):
    """One axis is scored over; a figure has nothing to do with a further one."""
    import ocean_skill.comparison as comparison

    with_depth = lanes["sat"].expand_dims(depth=[0.0, 50.0])
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        comparison,
        "prepare_source",
        lambda source, *a, **k: (
            {"model": lanes["model"], "sat": with_depth}[source],
            None,
        ),
    )
    try:
        with pytest.raises(ValueError, match="beyond its horizontal axes"):
            _scored()
    finally:
        monkeypatch.undo()


# --- the maps and the number beside them ---------------------------------------------


def test_the_default_maps_are_the_taylor_and_target_quantities(stub):
    maps = _scored().pointwise_metrics()
    assert list(maps.data_vars) == ["bias", "crmsd", "corr", "sigma_ratio"]
    for name, da in maps.items():
        assert da.dims == ("lat", "lon"), name
        assert np.isfinite(da).any(), name


def test_a_map_cell_is_that_cells_own_series_scored(stub):
    """The maps must be the metric, not something adjacent to it."""
    from ocean_skill import metrics as _metrics

    comparison = _scored()
    maps = comparison.pointwise_metrics("bias", "corr")
    cell = comparison.aligned.isel(lat=2, lon=3)
    scalar = _metrics.compute(cell, weighted=False, min_samples=0)
    assert float(maps["bias"].isel(lat=2, lon=3)) == pytest.approx(scalar["bias"])
    assert float(maps["corr"].isel(lat=2, lon=3)) == pytest.approx(scalar["corr"])


def test_asking_for_other_metrics_gets_them(stub):
    maps = _scored().pointwise_metrics("rmse", "mae", "n")
    assert list(maps.data_vars) == ["rmse", "mae", "n"]
    assert (maps["n"] > 0).any()


def test_the_overall_record_says_it_was_scored_over_an_axis(stub):
    record = _scored().metrics()
    assert record["over"] == "time"
    assert record["min_pairs"] == 5
    # reduced over space *and* time: far more pairs than a single map could have
    assert record["n"] > 30


def test_the_overall_number_describes_the_cells_the_maps_show(stub):
    """A number beside a figure that covers a different domain is worse than none."""
    from ocean_skill import metrics as _metrics

    comparison = _scored(min_pairs=25)
    enough = comparison.pointwise_metrics("n")["n"] >= 25
    expected = _metrics.compute(comparison.aligned.where(enough), min_samples=0)
    assert comparison.metrics()["bias"] == pytest.approx(expected["bias"])


def test_thin_cells_are_masked_in_every_map_but_counted_in_n(monkeypatch, lanes):
    """Cloud gaps: bias and corr have to cover the same cells to be readable."""
    import ocean_skill.comparison as comparison

    gapped = lanes["sat"].copy()
    gapped[3:, 0, :] = np.nan  # a southern strip observed on only three days
    monkeypatch.setattr(
        comparison,
        "prepare_source",
        lambda source, *a, **k: (
            {"model": lanes["model"], "sat": gapped}[source],
            None,
        ),
    )
    monkeypatch.setattr(comparison, "_domain_of", lambda name: None)
    scored = _scored(min_pairs=5)
    with pytest.warns(UserWarning, match="valid pairs along"):
        maps = scored.pointwise_metrics("bias", "corr", "n")
    thin = maps["n"] < 5
    assert bool(thin.any()), "the fixture is meant to leave some cells thin"
    for name in ("bias", "corr"):
        assert bool(np.isnan(maps[name].where(thin, np.nan)).all()), name
    assert bool((maps["n"].where(thin) > 0).any()), "n itself is not masked"


def test_maps_need_an_axis_to_score_along(stub):
    from ocean_skill.comparison import Comparison

    plain = Comparison(
        reference="sat",
        test="model",
        variable=NITRATE,
        aggregate={"time": "mean"},
        cache=False,
    )
    with pytest.raises(ValueError, match="over="):
        plain.pointwise_metrics()


# --- what a plain comparison still does ----------------------------------------------


def test_a_plain_comparison_of_the_same_pair_is_still_refused(stub):
    """The gate is intact, and now names over= as a third way forward."""
    from ocean_skill.comparison import Comparison

    with pytest.raises(ValueError) as excinfo:
        Comparison(reference="sat", test="model", variable=NITRATE, cache=False).align()
    message = str(excinfo.value)
    assert "aggregate=" in message
    assert 'over="time"' in message, "the third answer has to be offered"
