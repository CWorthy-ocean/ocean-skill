"""Behavior of ``subtract_mean``: what actually gets removed, recorded, and pooled.

The aligned pair is built by hand rather than through a stubbed regrid (contrast
``tests/test_skill_maps.py``): demeaning is a step *after* alignment, orthogonal to
how the pair got there, and a synthetic gridded pair lets every assertion below
compare against an independently-computed :func:`ocean_skill.metrics.evaluate`
number rather than a hand-derived one -- the same style ``test_skill_maps.py``
itself uses for its own scalar-vs-map cross-checks.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from ocean_skill import metrics as _metrics
from ocean_skill.comparison import Comparison, ComparisonSet

LAT = np.linspace(10, 20, 5)
LON = np.linspace(-90, -80, 6)


def _grid(offset: float, seed: int) -> xr.Dataset:
    """Build a synthetic gridded pair: reference plus a noisy, offset test lane."""
    rng = np.random.default_rng(seed)
    reference = xr.DataArray(
        20 + rng.normal(0, 1, (5, 6)),
        dims=("lat", "lon"),
        coords={"lat": LAT, "lon": LON},
        attrs={"units": "degC"},
    )
    test = reference + offset + rng.normal(0, 0.1, (5, 6))
    test.attrs["units"] = "degC"
    coverage = xr.DataArray(
        np.ones((5, 6)), dims=("lat", "lon"), coords={"lat": LAT, "lon": LON}
    )
    return xr.Dataset(
        {
            "test": test,
            "reference": reference,
            "difference": test - reference,
            "coverage": coverage,
        }
    )


def _comparison(subtract_mean=False, *, aligned=None) -> Comparison:
    """Build a real :class:`Comparison`, its aligned pair set directly, not computed.

    Mirrors the ``comp._aligned = xr.Dataset(...)`` pattern in
    ``tests/test_antimeridian.py``: no catalog entry needs to exist for either name
    since :meth:`Comparison.align` is never called.
    """
    with pytest.warns(UserWarning, match="resolved to standard_name"):
        c = Comparison(
            reference="glodap", test="some_model", variable="temperature",
            cache=False, subtract_mean=subtract_mean,
        )
    c._aligned = (aligned if aligned is not None else _grid(2.0, 0)).copy(deep=True)
    return c


# -- a no-op by default ----------------------------------------------------------------


def test_subtract_mean_off_leaves_the_aligned_pair_untouched():
    c = _comparison(False)
    before = c._aligned.copy(deep=True)
    c._subtract_scalar_means()
    xr.testing.assert_identical(c._aligned, before)
    assert "subtracted_mean_test" not in c._aligned.attrs
    assert "subtracted_mean_reference" not in c._aligned.attrs


# -- both lanes demeaned: bias -> 0, centred moments untouched -------------------------


def test_both_demeaned_zeroes_bias_and_leaves_centred_moments_alone():
    raw = _comparison(False)
    demeaned = _comparison(True, aligned=raw._aligned)
    demeaned._subtract_scalar_means()

    raw_m, demeaned_m = raw.metrics(), demeaned.metrics()
    assert demeaned_m["bias"] == pytest.approx(0, abs=1e-9)
    assert demeaned_m["mean_test"] == pytest.approx(0, abs=1e-9)
    assert demeaned_m["mean_reference"] == pytest.approx(0, abs=1e-9)
    assert demeaned_m["rmse"] == pytest.approx(demeaned_m["crmsd"])
    for key in ("std_test", "std_reference", "corr", "crmsd"):
        assert demeaned_m[key] == pytest.approx(raw_m[key]), key


def test_the_removed_means_match_what_metrics_would_report():
    raw = _comparison(False)
    expected = _metrics.evaluate(
        raw._aligned, ("mean_test", "mean_reference"), dim=None, weighted=True
    )
    demeaned = _comparison(True, aligned=raw._aligned)
    demeaned._subtract_scalar_means()
    attrs = demeaned._aligned.attrs
    assert attrs["subtracted_mean_test"] == pytest.approx(float(expected["mean_test"]))
    assert attrs["subtracted_mean_reference"] == pytest.approx(
        float(expected["mean_reference"])
    )


# -- one lane only: bias shifts by exactly the removed amount, other lane untouched ----


def test_one_sided_demean_shifts_bias_by_exactly_the_removed_mean():
    raw = _comparison(False)
    raw_bias = raw.metrics()["bias"]

    test_only = _comparison("test", aligned=raw._aligned)
    test_only._subtract_scalar_means()
    attrs = test_only._aligned.attrs
    assert "subtracted_mean_test" in attrs
    assert "subtracted_mean_reference" not in attrs
    m = test_only.metrics()
    assert raw_bias - m["bias"] == pytest.approx(attrs["subtracted_mean_test"])
    # untouched by a request that only demeans the other side
    assert m["std_test"] == pytest.approx(raw.metrics()["std_test"])
    assert m["corr"] == pytest.approx(raw.metrics()["corr"])


def test_difference_is_recomputed_but_coverage_is_not():
    raw = _comparison(False)
    demeaned = _comparison({"test": True}, aligned=raw._aligned)
    demeaned._subtract_scalar_means()
    aligned = demeaned._aligned
    expected_difference = aligned["test"] - aligned["reference"]
    xr.testing.assert_allclose(aligned["difference"], expected_difference)
    xr.testing.assert_identical(aligned["coverage"], raw._aligned["coverage"])


# -- metrics columns: always a "demeaned" label, the raw number only when demeaned -----


@pytest.mark.parametrize(
    "subtract_mean, expected_label",
    [
        (False, "raw"),
        (True, "demeaned"),
        ("test", "test demeaned"),
        ("reference", "reference demeaned"),
    ],
)
def test_the_demeaned_column_names_what_was_removed(subtract_mean, expected_label):
    c = _comparison(subtract_mean)
    c._subtract_scalar_means()
    assert c.metrics()["demeaned"] == expected_label


def test_subtracted_mean_columns_are_absent_when_nothing_was_demeaned():
    m = _comparison(False).metrics()
    assert "subtracted_mean_test" not in m
    assert "subtracted_mean_reference" not in m


# -- an all-NaN lane: warn, subtract nothing, record nothing ---------------------------


def test_an_all_nan_lane_warns_and_subtracts_nothing():
    grid = _grid(2.0, 0)
    grid["reference"][:] = np.nan
    grid["difference"] = grid["test"] - grid["reference"]
    c = _comparison({"reference": True}, aligned=grid)
    with pytest.warns(UserWarning, match="no finite mean"):
        c._subtract_scalar_means()
    assert "subtracted_mean_reference" not in c._aligned.attrs


# -- cache sharing: a demeaned run reuses the raw run's cached pair, no recompute ------


def test_a_demeaned_run_reuses_the_raw_runs_cached_pair_without_recomputing():
    """The whole point of caching the raw pair: no second regrid for the demeaned run.

    A raw run's aligned pair is put in the cache under its key. A demeaning run of the
    *same* comparison shares that key (``subtract_mean`` is not part of it), so
    :meth:`Comparison.align` takes the cache-hit branch -- loading the raw pair and
    only subtracting the scalar. The fake source names prove no regrid or source read
    happened: reading either would raise, since neither name is a real source.
    """
    from ocean_skill import cache as _cache

    grid = _grid(2.0, 0)
    with pytest.warns(UserWarning, match="resolved to standard_name"):
        raw = Comparison(
            reference="glodap", test="some_model", variable="temperature", cache=True
        )
    _cache.save(raw._cache_key, grid)

    with pytest.warns(UserWarning, match="resolved to standard_name"):
        demeaned = Comparison(
            reference="glodap", test="some_model", variable="temperature",
            cache=True, subtract_mean=True,
        )
    assert demeaned._cache_key == raw._cache_key  # the shared entry

    aligned = demeaned.align()  # loads from cache; never touches the fake sources
    assert "subtracted_mean_test" in aligned.attrs
    assert demeaned.metrics()["bias"] == pytest.approx(0, abs=1e-9)


# -- pooling raw + demeaned onto one diagram: the workflow this option is for ----------


def test_a_raw_comparison_and_its_demeaned_twin_pool_as_two_distinct_points():
    raw = _comparison(False)
    demeaned = _comparison(True, aligned=raw._aligned)
    demeaned._subtract_scalar_means()

    pooled = ComparisonSet(raw) + ComparisonSet(demeaned)
    assert len(pooled) == 2  # not deduped -- see test_identity_distinguishes_*
    labels = [pooled._label_for(i) for i in range(len(pooled))]
    assert labels == ["raw", "demeaned"]


def test_pooled_taylor_stats_agree_but_target_stats_differ():
    """Taylor is bias-blind by construction; target's whole axis is bias."""
    raw = _comparison(False)
    demeaned = _comparison(True, aligned=raw._aligned)
    demeaned._subtract_scalar_means()

    df = (ComparisonSet(raw) + ComparisonSet(demeaned)).metrics()
    for key in ("std_test", "std_reference", "corr", "crmsd"):
        assert df.iloc[0][key] == pytest.approx(df.iloc[1][key]), key
    assert df.iloc[0]["bias"] != pytest.approx(df.iloc[1]["bias"])
    assert df.iloc[1]["bias"] == pytest.approx(0, abs=1e-9)


def test_pooled_metrics_table_carries_the_subtracted_mean_only_on_the_demeaned_row():
    raw = _comparison(False)
    demeaned = _comparison(True, aligned=raw._aligned)
    demeaned._subtract_scalar_means()

    pooled = ComparisonSet(raw) + ComparisonSet(demeaned)
    df = pooled.metrics()
    assert df["demeaned"].tolist() == ["raw", "demeaned"]
    assert np.isnan(df.iloc[0]["subtracted_mean_test"])
    assert df.iloc[1]["subtracted_mean_test"] == pytest.approx(
        demeaned._aligned.attrs["subtracted_mean_test"]
    )
