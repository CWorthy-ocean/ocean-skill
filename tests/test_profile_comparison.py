"""``Comparison`` classification and metrics for a vertical-profile pair.

Builds the aligned pair by hand and injects it as ``Comparison._aligned`` --
mirroring how ``tests/test_compare_times.py`` isolates fan-shape logic from real
data -- so these test ``is_profile``/``family``/``metrics``/``as_item`` without a
catalog, a cache, or ``align()`` itself (already covered end to end by
``tests/test_vertical_match.py``).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import xarray as xr

from ocean_skill.comparison import Comparison

TEMPERATURE = "sea_water_temperature"


def _profile_pair(n: int = 5, *, offset: float = 0.5):
    depth = np.linspace(5.0, 100.0, n)
    reference = xr.DataArray(
        20.0 - 0.1 * depth, dims=("DEPTH",), coords={"DEPTH": depth}
    ).assign_coords(lon=-158.0, lat=22.75)
    reference.attrs["units"] = "degC"
    test = reference + offset
    return xr.Dataset(
        {
            "test": test.rename("test"),
            "reference": reference.rename("reference"),
            "difference": (test - reference).rename("difference"),
        },
        attrs={
            "scored_over": "DEPTH",
            "match_method": "interp",
            "station_lon": -158.0,
            "station_lat": 22.75,
        },
    )


def _profile_comparison(**kwargs) -> Comparison:
    c = Comparison(
        reference="whots_station",
        test="run_new",
        variable=TEMPERATURE,
        select={"depth": [5.0, 25.0, 50.0, 75.0, 100.0]},
        over="Z",
        **kwargs,
    )
    c._aligned = _profile_pair()
    return c


# -- classification ----------------------------------------------------------------


def test_is_profile_true_over_Z_at_a_point():
    c = _profile_comparison()
    assert c.is_profile
    assert not c.is_series
    assert c.family == "profile"


def test_family_reason_names_the_depth_axis():
    c = _profile_comparison()
    assert "depth axis" in c.family_reason


def test_lowercase_z_over_is_not_a_profile():
    """over="z"/"depth" are the pre-existing generic-axis spellings, left alone."""
    c = Comparison(
        reference="a", test="b", variable=TEMPERATURE, over="depth"
    )
    c._aligned = _profile_pair()
    c._aligned.attrs["scored_over"] = "DEPTH"
    assert not c.is_profile
    # not is_series either -- point_of would still say yes, but that's the
    # generic-axis path's own business, not asserted here.


def test_explicit_over_vertical_is_also_recognized():
    c = Comparison(reference="a", test="b", variable=TEMPERATURE, over="vertical")
    c._aligned = _profile_pair()
    assert c.is_profile


# -- metrics --------------------------------------------------------------------------


def test_metrics_reports_depth_levels_not_time_steps():
    """sample_noun is a compute()-only argument (it phrases the thin-sample warning,
    not a field of the record) -- proven here by triggering that warning."""
    c = _profile_comparison()
    with pytest.warns(UserWarning, match="depth levels"):
        record = c.metrics()
    assert record["weighted"] is False
    assert record["station_lon"] == pytest.approx(-158.0)
    assert record["station_lat"] == pytest.approx(22.75)


def test_metrics_does_not_try_to_mask_by_pointwise_n():
    """The is_series/is_profile exclusion in metrics() -- would otherwise call
    pointwise_metrics(), which is refused for a profile."""
    c = _profile_comparison()
    record = c.metrics()  # must not raise
    assert np.isfinite(record["bias"])
    assert record["bias"] == pytest.approx(0.5, abs=1e-6)


def test_pointwise_metrics_is_refused_for_a_profile():
    c = _profile_comparison()
    with pytest.raises(ValueError, match="single water column"):
        c.pointwise_metrics()


# -- as_item ----------------------------------------------------------------------------


def test_as_item_carries_the_aligned_trio_not_metric_maps():
    c = _profile_comparison()
    item = c.as_item()
    assert "aligned" in item
    assert "skill" not in item
    assert item["labels"] == ("run_new", "whots_station")


def test_as_item_metrics_are_the_profile_record():
    c = _profile_comparison()
    item = c.as_item()
    assert item["metrics"]["station_lon"] == pytest.approx(-158.0)


# -- plot() domain guard -----------------------------------------------------------------


def test_plot_does_not_set_a_domain_for_a_profile():
    c = _profile_comparison()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig = c.plot()
    # a profile figure drew (no domain= TypeError from the profile() signature)
    assert fig is not None


def test_plot_sets_labels_for_a_profile():
    import unittest.mock as mock

    import ocean_skill.plot.registry as registry

    captured = {}
    real_render = registry.render

    def spy(spec, **kwargs):
        captured["labels"] = spec.options.get("labels")
        return real_render(spec, **kwargs)

    c = _profile_comparison()
    with mock.patch.object(registry, "render", spy):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            c.plot()
    assert captured["labels"] == ("run_new", "whots_station")
