"""Calculator plumbing: what a registered calculator sees, and how often it runs.

A calculator such as a tidal harmonic analysis is expensive and consumes the time
axis itself, so three things matter beyond the formula: the caller's time window
must reach it, its source must be known (to cache by identity), and a Field must
not prepare it twice. Plus the list-valued spec fan (``"constituent": [...]``).
"""

from __future__ import annotations

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
