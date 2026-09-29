"""``select={"time": "latest"}`` as a shorthand in the Python API, not just in suites.

``"latest"`` is resolved to the ISO string of the source's newest native step inside
``Field.__init__`` and ``Comparison.__init__`` -- before any cache key or label is
built from ``select`` -- exactly as a suite page already resolves it in
``workflows.pages.expand``. ``extrema._native_time_index`` is monkeypatched the same
way ``tests/test_extrema.py`` and ``tests/test_workflows_run.py`` do, so none of this
needs a catalog.
"""

from __future__ import annotations

from unittest import mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ocean_skill import catalog as _catalog
from ocean_skill import comparison as _comparison
from ocean_skill.comparison import Comparison
from ocean_skill.field import Field
from ocean_skill.field import field as make_field

INDEX = pd.date_range("2010-01-05", periods=6, freq="7D")
NEWEST = INDEX[-1].isoformat()


@pytest.fixture
def index_calls(monkeypatch):
    """Stub the native time index with ``INDEX``, recording each source asked about."""
    calls: list[str] = []

    def fake(source):
        calls.append(source)
        return INDEX

    monkeypatch.setattr("ocean_skill.extrema._native_time_index", fake)
    return calls


# -- Field / field() -------------------------------------------------------------------


def test_field_resolves_latest_to_the_newest_step_as_an_iso_string(index_calls):
    f = Field("stub", "temperature", select={"depth": "surface", "time": "latest"})
    assert f.select == {"depth": "surface", "time": NEWEST}


def test_field_resolves_the_alternate_time_spellings(index_calls):
    assert Field("stub", "temperature", select={"T": "latest"}).select == {"T": NEWEST}


def test_osk_field_resolves_latest(index_calls):
    f = make_field("stub", "temperature", select={"time": "latest"})
    assert f.select == {"time": NEWEST}


def test_no_latest_means_no_source_read_and_an_untouched_select(index_calls):
    f = Field("stub", "temperature", select={"time": "2010-01", "depth": 50})
    assert f.select == {"time": "2010-01", "depth": 50}
    assert index_calls == []


def test_each_source_of_a_fan_out_resolves_against_its_own_axis(monkeypatch):
    short = pd.date_range("2010-01-01", periods=3, freq="D")
    monkeypatch.setattr(
        "ocean_skill.extrema._native_time_index",
        lambda source: {"a": INDEX, "b": short}[source],
    )
    fs = make_field(["a", "b"], "temperature", select={"time": "latest"})
    assert [f.select["time"] for f in fs] == [
        INDEX[-1].isoformat(),
        short[-1].isoformat(),
    ]


def test_latest_needs_a_decoded_calendar_axis(monkeypatch):
    monkeypatch.setattr(
        "ocean_skill.extrema._native_time_index", lambda source: pd.Index([0.0, 15.0])
    )
    with pytest.raises(ValueError, match="not a decoded calendar axis"):
        Field("clim", "temperature", select={"time": "latest"})


def test_extremum_series_accepts_latest_as_its_time(index_calls, monkeypatch):
    da = xr.DataArray(
        np.array([[1.0, 2.0], [3.0, 50.0]]),
        dims=("lat", "lon"),
        coords={"lat": [10.0, 20.0], "lon": [-100.0, -90.0]},
        name="temperature",
        attrs={"units": "degC"},
    )
    monkeypatch.setattr(_comparison, "prepare_source", lambda *a, **k: (da, None))
    fs = make_field("stub", "temperature").extremum("max").series(time="latest")
    assert fs[0].select["time"] == NEWEST


# -- Comparison / compare() ------------------------------------------------------------


def _cmp(select):
    return Comparison(reference="r", test="t", variable="temperature", select=select)


def test_flat_select_resolves_against_the_test_source(index_calls):
    assert _cmp({"time": "latest"}).select == {"time": NEWEST}
    assert index_calls == ["t"]  # never the reference


def test_pair_spec_resolves_its_test_side_and_leaves_the_reference_alone(index_calls):
    c = _cmp({"test": {"time": "latest"}, "reference": {"time": "2010-01"}})
    assert c.select == {"test": {"time": NEWEST}, "reference": {"time": "2010-01"}}
    assert index_calls == ["t"]


def test_pair_spec_without_latest_reads_nothing(index_calls):
    _cmp({"test": {"time": "2010-01"}, "reference": {}})
    assert index_calls == []


def test_a_reference_side_latest_is_refused_with_a_pointer(index_calls):
    with pytest.raises(ValueError, match=r"select\['reference'\] cannot use time"):
        _cmp({"test": {}, "reference": {"time": "latest"}})


def test_labels_report_the_resolved_date_not_the_word(index_calls):
    assert _comparison._display_time(_cmp({"time": "latest"}).select) == NEWEST


@pytest.fixture
def stubbed_compare():
    """Mock the catalog and record each fanned comparison's resolved select."""
    formed = []
    declared = {"model": {"variables": []}, "obs": {"variables": []}}
    with (
        mock.patch(
            "ocean_skill.catalog.resolve", lambda n: mock.Mock(metadata=declared[n])
        ),
        mock.patch.object(
            Comparison,
            "align",
            lambda self, refresh=False: formed.append(self.select),
        ),
    ):
        yield formed


def test_compare_resolves_a_flat_latest_for_both_lanes(index_calls, stubbed_compare):
    _comparison.compare(
        reference=["obs"],
        test=["model"],
        variables=["temperature"],
        select={"time": "latest"},
    )
    assert stubbed_compare == [{"depth": "surface", "time": NEWEST}]
    assert index_calls == ["model"]


def test_compare_resolves_a_pair_spec_test_side(index_calls, stubbed_compare):
    _comparison.compare(
        reference=["obs"],
        test=["model"],
        variables=["temperature"],
        select={"test": {"time": "latest"}, "reference": {"time": "2010-01"}},
    )
    assert [c["test"]["time"] for c in stubbed_compare] == [NEWEST]
    assert [c["reference"]["time"] for c in stubbed_compare] == ["2010-01"]


# -- the cache: an unmoved run hits, a moved run is a new key --------------------------


def test_comparison_cache_key_follows_the_run(monkeypatch):
    monkeypatch.setattr("ocean_skill.extrema._native_time_index", lambda s: INDEX)
    before = _cmp({"time": "latest"})._cache_key
    same = _cmp({"time": "latest"})._cache_key
    grown = INDEX.append(pd.DatetimeIndex([INDEX[-1] + pd.Timedelta(days=7)]))
    monkeypatch.setattr("ocean_skill.extrema._native_time_index", lambda s: grown)
    after = _cmp({"time": "latest"})._cache_key
    assert before == same
    assert after != before


@pytest.fixture
def counted_prepare(monkeypatch):
    """Count runs of the read/reduce step inside ``prepare_source``.

    Patching one level below it leaves the real on-disk cache-key/hit/miss logic
    running (``isolated_cache`` is autouse).
    """
    calls = {"n": 0}
    da = xr.DataArray(
        np.random.default_rng(0).normal(5.0, 1.0, (8, 10)),
        dims=("lat", "lon"),
        coords={"lat": np.linspace(20, 30, 8), "lon": np.linspace(-100, -90, 10)},
        name="temperature",
        attrs={"units": "degC"},
    )

    def fake_prepare(obj, meta, variable, select, aggregate=None, **kwargs):
        calls["n"] += 1
        return da, None

    monkeypatch.setattr(_comparison, "_prepare", fake_prepare)
    monkeypatch.setattr(_catalog, "resolve", lambda name: mock.Mock(metadata={}))
    monkeypatch.setattr("ocean_skill.read", lambda n: None)
    return calls


def test_a_latest_field_hits_the_cache_while_the_run_stands_still(
    monkeypatch, counted_prepare
):
    monkeypatch.setattr("ocean_skill.extrema._native_time_index", lambda s: INDEX)
    Field("stub", "temperature", select={"time": "latest"}, cache=True).prepare()
    Field("stub", "temperature", select={"time": "latest"}, cache=True).prepare()
    assert counted_prepare["n"] == 1


def test_a_latest_field_recomputes_once_the_run_has_moved(monkeypatch, counted_prepare):
    monkeypatch.setattr("ocean_skill.extrema._native_time_index", lambda s: INDEX)
    Field("stub", "temperature", select={"time": "latest"}, cache=True).prepare()
    assert counted_prepare["n"] == 1

    grown = INDEX.append(pd.DatetimeIndex([INDEX[-1] + pd.Timedelta(days=7)]))
    monkeypatch.setattr("ocean_skill.extrema._native_time_index", lambda s: grown)
    Field("stub", "temperature", select={"time": "latest"}, cache=True).prepare()
    assert counted_prepare["n"] == 2, "a new newest step is a new key, so this misses"
