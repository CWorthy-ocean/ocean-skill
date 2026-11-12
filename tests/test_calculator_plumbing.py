"""Calculator plumbing: what a registered calculator sees, and how often it runs.

A calculator such as a tidal harmonic analysis is expensive and consumes the time
axis itself, so three things matter beyond the formula: the caller's time window
must reach it, its source must be known (to cache by identity), and a Field must
not prepare it twice. Plus the list-valued spec fan (``"constituent": [...]``).
"""

from __future__ import annotations

import warnings
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ocean_skill import cache, operators
from ocean_skill.comparison import prepare_source
from ocean_skill.operators import (
    CALCULATOR_FANS,
    CALCULATORS,
    expand_calculator_fans,
    register_calculator,
)

SEEN: list[dict] = []


@pytest.fixture
def recorder():
    """Register a throwaway calculator recording what it was handed."""
    SEEN.clear()

    @register_calculator("plumbing_probe", fans=("which",))
    def probe(ds, *, which="a"):
        SEEN.append(
            {"n_time": ds.sizes["time"], "source": operators.calculator_source(ds)}
        )
        out = ds["temp"].mean("time")
        out.attrs["units"] = "degC"
        return out.rename(f"probe_{which}")

    yield
    CALCULATORS.pop("plumbing_probe", None)
    CALCULATOR_FANS.pop("plumbing_probe", None)


@pytest.fixture
def source(monkeypatch):
    times = pd.date_range("2000-01-01", periods=48, freq="h")
    ds = xr.Dataset(
        {"temp": (("time", "lat", "lon"), np.ones((48, 2, 3)))},
        coords={"time": times, "lat": [0.0, 1.0], "lon": [0.0, 1.0, 2.0]},
    )
    meta = {"featureType": "grid", "vertical_levels": 1}
    monkeypatch.setattr(
        "ocean_skill.catalog.resolve", lambda name: mock.Mock(metadata=meta)
    )
    monkeypatch.setattr("ocean_skill.read", lambda name, **kw: ds)
    return ds


def test_time_window_reaches_the_calculator(recorder, source):
    window = {"time": slice("2000-01-01T00", "2000-01-01T11")}
    da, _ = prepare_source(
        "probe_src", {"calculate": "plumbing_probe"}, window, None, use_cache=False
    )
    assert SEEN == [{"n_time": 12, "source": "probe_src"}]
    assert da.dims == ("lat", "lon")


def test_bare_dataset_has_no_source():
    assert operators.calculator_source(xr.Dataset()) is None


def test_field_with_calculated_variable_prepares_once(recorder, source):
    from ocean_skill.field import Field

    f = Field("probe_src", {"calculate": "plumbing_probe"})
    assert not f._bare_vertical()
    f.plot()
    assert len(SEEN) == 1


def test_fan_expands_declared_keys_only(recorder):
    spec = {"calculate": "plumbing_probe", "which": ["a", "b"], "other": [1, 2]}
    assert expand_calculator_fans(spec) == [
        {"calculate": "plumbing_probe", "which": "a", "other": [1, 2]},
        {"calculate": "plumbing_probe", "which": "b", "other": [1, 2]},
    ]
    assert expand_calculator_fans("temperature") == ["temperature"]
    assert expand_calculator_fans({"calculate": "plumbing_probe"}) == [
        {"calculate": "plumbing_probe"}
    ]


def test_field_fans_a_list_valued_calculator_kwarg(recorder, source):
    from ocean_skill.field import FieldSet, field

    fs = field("probe_src", {"calculate": "plumbing_probe", "which": ["a", "b"]})
    assert isinstance(fs, FieldSet)
    assert [f.variable["which"] for f in fs.fields] == ["a", "b"]


def test_calculated_cache_kind_round_trips():
    key = cache.key_for_calculated(source="s", name="n", params={"k": 1})
    assert key != cache.key_for_calculated(source="s", name="n", params={"k": 2})
    ds = xr.Dataset({"amplitude": ("x", [1.0, 2.0])})
    cache.save(key, ds, kind="calculated")
    assert cache.load(key, kind="calculated")["amplitude"].values.tolist() == [1.0, 2.0]
    assert cache.clear("calculated") == 1
    assert cache.load(key, kind="calculated") is None


def test_fanned_value_is_in_the_label(recorder):
    """K1 and M2 rows must not draw with identical labels."""
    from ocean_skill.comparison import _short_variable_label, _variable_label

    spec = {"calculate": "plumbing_probe", "which": "a"}
    assert _variable_label(spec) == "plumbing_probe (a)"
    assert "(a)" in _short_variable_label(spec)


# -- layer calculators ----------------------------------------------------------------

LAYER = {"calculate": "twice", "variable": "oxygen"}


@pytest.fixture
def twice():
    """Register a toy layer calculator (doubling), at a version the test can set."""

    def install(version=1):
        @register_calculator("twice", layer=True, version=version)
        def fn(da, *, variable=None):
            return (2 * da).rename("twice_" + str(variable))

    yield install
    for registry in (CALCULATORS, operators.CALCULATOR_VERSIONS):
        registry.pop("twice", None)
    operators.CALCULATOR_LAYER.discard("twice")


def test_layer_spec_is_not_calculated_but_whole_column_calculators_still_are(twice):
    from ocean_skill.comparison import _is_calculated

    twice()
    assert not _is_calculated(LAYER)
    assert not _is_calculated({"test": LAYER, "reference": "oxygen"})
    assert _is_calculated({"calculate": "depth_of", "variable": "oxygen"})
    assert _is_calculated({"test": {"calculate": "mld"}, "reference": "oxygen"})


def test_layer_calculator_splits_the_spec(twice):
    twice()
    spec = {**LAYER, "standard_name": "x", "scale": 3}
    assert operators.layer_calculator(spec) == ("twice", "oxygen", {"scale": 3})
    assert operators.layer_calculator("oxygen") is None
    assert operators.layer_calculator({"calculate": "depth_of"}) is None
    operators.register_derived("twice_o2", LAYER)
    try:
        assert operators.layer_calculator("twice_o2")[0] == "twice"
    finally:
        operators.DERIVED.pop("twice_o2")


def test_calculator_versions_reports_only_versions_above_one(twice):
    twice(version=1)
    assert operators.calculator_versions(LAYER) == {}
    twice(version=2)
    pair = {"test": LAYER, "reference": "oxygen"}
    nested = {"calculate": "other", "variable": LAYER}
    assert operators.calculator_versions(LAYER) == {"twice": 2}
    assert operators.calculator_versions(pair) == {"twice": 2}
    assert operators.calculator_versions({"sum": [LAYER, "oxygen"]}) == {"twice": 2}
    assert operators.calculator_versions(nested) == {"twice": 2}
    assert operators.calculator_versions("oxygen") == {}


def test_a_versioned_calculator_rekeys_only_its_own_specs(twice, monkeypatch):
    kw = {"source": "s", "select": {"depth": 300}}
    plain = cache.key_for_prepared(variable="oxygen", **kw)
    base = cache.key_for_prepared(variable=LAYER, **kw)
    twice(version=2)
    assert cache.key_for_prepared(variable="oxygen", **kw) == plain
    assert cache.key_for_prepared(variable=LAYER, **kw) != base
    both = {"test": "s", "reference": "r", "select": {}, "method": "linear"}
    full = cache.key_for(variable=LAYER, **both)
    twice(version=1)
    assert cache.key_for(variable=LAYER, **both) != full
    # No versioned calculator: the payload never mentions calculators at all, so the
    # key is the one a pre-version cache built.
    monkeypatch.setattr(operators, "calculator_versions", lambda v: {})
    import hashlib
    import json

    payload = json.dumps(
        {"v": cache._FORMAT_VERSION, "source": "s", "variable": "oxygen",
         "select": {"depth": 300}, "definition": ""},
        sort_keys=True,
        default=str,
    )
    assert plain == hashlib.sha256(payload.encode()).hexdigest()[:16]


def _levels_source(monkeypatch):
    from unittest import mock

    depth = np.array([0.0, 100.0, 300.0, 500.0])
    vals = np.broadcast_to(depth[:, None, None] + 1.0, (4, 2, 3)).copy()
    ds = xr.Dataset(
        {"oxygen": (("depth", "lat", "lon"), vals, {"units": "mmol m-3"})},
        coords={"depth": depth, "lat": [0.0, 1.0], "lon": [0.0, 1.0, 2.0]},
    )
    meta = {"featureType": "grid"}
    monkeypatch.setattr(
        "ocean_skill.catalog.resolve", lambda name: mock.Mock(metadata=meta)
    )
    monkeypatch.setattr("ocean_skill.read", lambda name, **kw: ds)


def test_prepare_source_runs_a_layer_calculator_after_the_depth_select(
    twice, monkeypatch
):
    twice()
    _levels_source(monkeypatch)
    plain, _ = prepare_source("src", "oxygen", {"depth": 300}, None, use_cache=False)
    da, _ = prepare_source("src", LAYER, {"depth": 300}, None, use_cache=False)
    assert da.name == "twice_oxygen"
    np.testing.assert_allclose(da.values, 2 * plain.values)
    assert float(da.mean()) == pytest.approx(2 * 301.0)
    # A missing inner variable is "not in this source", not a KeyError.
    missing = {**LAYER, "variable": "no_such_thing"}
    assert prepare_source("src", missing, {"depth": 300}, None, use_cache=False) == (
        None,
        None,
    )


def test_resolve_variable_applies_a_layer_calculator_to_the_given_layer(twice):
    twice()
    layer = xr.Dataset({"oxygen": ("x", [1.0, 2.0])})
    out = operators.resolve_variable(layer, {**LAYER, "standard_name": "doubled"})
    assert out.values.tolist() == [2.0, 4.0]
    assert out.name == "doubled"
    assert operators.resolve_variable(xr.Dataset(), LAYER) is None


def test_whole_column_calculator_still_refuses_a_depth_select():
    from ocean_skill.comparison import _prepare
    from tests.test_mld import DEPTH, TEMP, _water_column

    spec = {"calculate": "mld", "method": "temperature_threshold"}
    with pytest.raises(ValueError, match="already reduces the vertical axis"):
        _prepare(_water_column(DEPTH, TEMP), {"model": "roms"}, spec, {"depth": 100})


def test_roms_layer_calculator_needs_a_fixed_depth(twice):
    from ocean_skill.comparison import _prepare
    from tests.test_compare_horizontal_aggregate_crop import _roms_run

    twice()
    obj, meta = _roms_run(0, 60, nt=1)
    spec = {"calculate": "twice", "variable": "temp"}
    for select in ({}, {"depth": "column"}):
        with pytest.raises(ValueError, match="fixed-depth surface"):
            _prepare(obj, meta, spec, dict(select), depth_method="interp")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        da, _ = _prepare(obj, meta, spec, {"depth": 300}, depth_method="interp")
    assert "s_rho" not in da.dims
    with pytest.warns(UserWarning, match="depth_method='interp'"):
        _prepare(obj, meta, spec, {"depth": 300}, depth_method="nearest")


def test_suite_field_page_allows_a_depth_beside_a_layer_spec_only(twice):
    from ocean_skill.config import SuiteConfig
    from ocean_skill.suites import pages as P

    twice()

    def expand(variable):
        P.expand(
            SuiteConfig.model_validate(
                {
                    "name": "t",
                    "defaults": {"test": "stub"},
                    "pages": [
                        {
                            "title": "x",
                            "field": {
                                "variables": [variable],
                                "select": {"depth": 300},
                            },
                        }
                    ],
                }
            )
        )

    expand(LAYER)  # does not raise
    with pytest.raises(ValueError, match="vertical axis"):
        expand({"calculate": "depth_of", "variable": "oxygen", "extremum": "min"})


def test_probes_look_at_the_inner_variable_of_a_layer_spec(twice, monkeypatch):
    from ocean_skill import comparison

    twice()
    _levels_source(monkeypatch)
    comparison._AVAILABILITY_MEMO.clear()
    assert comparison._variable_available("src", LAYER)
    assert not comparison._variable_available("src", {**LAYER, "variable": "nothing"})
    # The 4-D answer would be the doubled array; the probe must see oxygen's own axes.
    assert comparison._bare_time_is_multistep("src", LAYER, None) is False
