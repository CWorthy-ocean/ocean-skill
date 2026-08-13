"""The metric registry: one definition per metric, evaluated over any set of dims.

``compute`` (every dim, one number each) and ``evaluate(dim="time")`` (one map each) run
the *same* registry entries, so the thing worth testing is that they agree: a map's
value at one cell has to equal ``compute`` on that cell's own series. If a metric ever
grows a second spelling for the pointwise case, that assertion is what fails.

The scalar half also has to be unchanged by the refactor, which is asserted against
direct xskillscore calls rather than a golden file — a golden file would happily
record a regression as the new truth.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from ocean_skill import metrics as m


def _pair(nt: int = 8, seed: int = 0) -> xr.Dataset:
    """Build a synthetic (time, lat, lon) test/reference pair with units."""
    rng = np.random.default_rng(seed)
    coords = {
        "time": np.arange("2012-01-01", nt, dtype="datetime64[D]").astype(
            "datetime64[ns]"
        ),
        "lat": np.linspace(18, 26, 6),
        "lon": np.linspace(-98, -90, 5),
    }
    shape = (nt, 6, 5)
    test = xr.DataArray(
        rng.normal(5.0, 1.0, shape),
        dims=("time", "lat", "lon"),
        coords=coords,
        attrs={"units": "mmol m-3"},
    )
    reference = test + rng.normal(0.25, 0.5, shape)
    reference.attrs = {"units": "mmol m-3"}
    return xr.Dataset({"test": test, "reference": reference})


def test_the_scalar_record_keeps_its_columns_and_their_order():
    """The CSV's column order is read by people; it is part of the output."""
    rec = m.compute(_pair())
    assert list(rec) == [
        "bias",
        "rmse",
        "mae",
        "corr",
        "sigma_ratio",
        "std_test",
        "std_reference",
        "crmsd",
        "mean_test",
        "mean_reference",
        "n",
        "weighted",
    ]
    assert list(m.METRICS) == list(rec)[:-1]


@pytest.mark.parametrize(
    ("metric", "call"),
    [("bias", "me"), ("rmse", "rmse"), ("mae", "mae"), ("corr", "pearson_r")],
)
def test_each_primitive_is_the_xskillscore_call_it_claims_to_be(metric, call):
    import xskillscore as xs

    aligned = _pair()
    t, r = aligned["test"], aligned["reference"]
    weights = m.area_weights(r).where(np.isfinite(t) & np.isfinite(r), 0.0)
    expected = float(
        getattr(xs, call)(t, r, dim=list(r.dims), skipna=True, weights=weights)
    )
    assert m.compute(aligned)[metric] == pytest.approx(expected)


def test_the_derived_metrics_stay_algebraically_consistent():
    """``crmsd`` is defined by ``rmse² = bias² + crmsd²``, not computed apart."""
    rec = m.compute(_pair())
    assert rec["crmsd"] ** 2 + rec["bias"] ** 2 == pytest.approx(rec["rmse"] ** 2)
    assert rec["sigma_ratio"] == pytest.approx(rec["std_test"] / rec["std_reference"])


def test_a_map_at_one_cell_is_the_scalar_of_that_cells_own_series():
    """The invariant the registry exists for — and what a wrong ``dim=`` breaks."""
    aligned = _pair()
    maps = m.evaluate(aligned, dim="time", weighted=False)
    cell = aligned.isel(lat=2, lon=3)
    # min_samples=0: eight steps is a fine series and a poor "number of cells"
    scalar = m.compute(cell, weighted=False, min_samples=0)
    for name in m.METRICS:
        assert float(maps[name].isel(lat=2, lon=3)) == pytest.approx(scalar[name]), name


def test_reducing_a_subset_of_dims_leaves_the_others_standing():
    maps = m.evaluate(_pair(), ("bias", "corr"), dim="time", weighted=False)
    for da in maps.values():
        assert da.dims == ("lat", "lon")
        assert np.isfinite(da).all()


def test_asking_for_a_derived_metric_alone_still_computes_what_it_needs():
    """``crmsd`` needs ``rmse`` and ``bias``; a caller should not have to know."""
    out = m.evaluate(_pair(), ("crmsd",), dim="time", weighted=False)
    assert list(out) == ["crmsd"]
    assert np.isfinite(out["crmsd"]).all()


def test_n_counts_pairs_over_the_reduced_axis_not_cells():
    aligned = _pair(nt=8)
    assert m.compute(aligned)["n"] == 8 * 6 * 5
    per_cell = m.evaluate(aligned, ("n",), dim="time", weighted=False)["n"]
    assert per_cell.dims == ("lat", "lon")
    assert (per_cell == 8).all()


def test_a_gap_in_one_member_removes_the_pair_from_every_metric():
    """Cloud gaps are the reason: bias and corr must describe the same cells."""
    aligned = _pair(nt=8)
    aligned["reference"][2:5, 1, 1] = np.nan
    maps = m.evaluate(aligned, ("n", "bias", "corr"), dim="time", weighted=False)
    assert int(maps["n"].isel(lat=1, lon=1)) == 5
    assert np.isfinite(maps["bias"].isel(lat=1, lon=1))


def test_sigma_ratio_is_nan_not_inf_where_the_reference_does_not_vary():
    aligned = _pair(nt=6)
    aligned["reference"][:, 0, 0] = 3.0  # flat in time at one cell
    ratio = m.evaluate(aligned, ("sigma_ratio",), dim="time", weighted=False)[
        "sigma_ratio"
    ]
    assert np.isnan(float(ratio.isel(lat=0, lon=0)))
    assert np.isfinite(ratio.isel(lat=1, lon=1))


def test_every_map_arrives_labelled():
    """The plotting layer labels colorbars from these and cannot derive them."""
    maps = m.evaluate(_pair(), ("bias", "corr", "n"), dim="time", weighted=False)
    assert maps["bias"].attrs["units"] == "mmol m-3"
    assert maps["corr"].attrs["units"] == ""  # dimensionless, not the variable's units
    assert maps["n"].attrs["units"] == "count"
    assert maps["bias"].name == "bias"
    assert "mean error" in maps["bias"].attrs["long_name"]


def test_weighted_is_a_flag_not_a_metric():
    with pytest.raises(KeyError, match="not a metric"):
        m.evaluate(_pair(), ("weighted",), dim="time")


def test_an_unknown_metric_says_what_is_registered():
    with pytest.raises(KeyError, match="willmott"):
        m.evaluate(_pair(), ("willmott",), dim="time")


def test_the_default_map_set_is_the_taylor_and_target_quantities():
    """Not a taste in metrics: these four are what those two diagrams plot."""
    assert set(m.DEFAULT_MAP_METRICS) == {"bias", "crmsd", "corr", "sigma_ratio"}
    assert all(m.REGISTRY[name].kind == "skill" for name in m.DEFAULT_MAP_METRICS)


def test_a_metric_must_be_either_primitive_or_derived():
    with pytest.raises(ValueError, match="exactly one"):
        m.Metric("bogus", "both at once", kind="skill", units="1")
