"""``read()`` under ``truncate_time`` cuts each source once, not once per call.

Cutting a large dask-backed model (``isel`` with an integer list, plus reading the
anchor source) costs seconds, and a page reads the same source dozens of times, so the
cut result is memoized beside the read memo (``sources._TRUNCATED_CACHE``).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import ocean_skill as osk
from ocean_skill import sources as _sources


def _model(n=10):
    return xr.Dataset(
        {"temp": (("time", "y"), np.arange(n * 2.0).reshape(n, 2))},
        coords={"time": pd.date_range("2020-01-01", periods=n, freq="D")},
    )


@pytest.fixture
def memo_sources(monkeypatch, tmp_path):
    """Stub sources ``m`` (10 daily steps) and ``anchor`` with a real catalog file.

    The read memo only stores a source whose catalog file can be stat'd, so the ref
    points at a real path; ``_read_uncached`` counts how often it is reopened.
    """
    path = tmp_path / "c.yaml"
    path.write_text("x")
    monkeypatch.setattr(
        _sources,
        "resolve",
        lambda name: SimpleNamespace(name=name, metadata={}, path=path),
    )
    opened = []

    def uncached(ref, meta, qc, kwargs):
        opened.append(ref.name)
        return _model()

    monkeypatch.setattr(_sources, "_read_uncached", uncached)
    osk.read.cache_clear()
    yield opened
    osk.read.cache_clear()


def _days(obj):
    return list(obj.time.dt.day.values)


def test_a_second_read_does_not_cut_again(memo_sources, monkeypatch):
    cuts = []
    real = _sources._truncated
    monkeypatch.setattr(
        _sources, "_truncated", lambda ref, obj: cuts.append(ref.name) or real(ref, obj)
    )
    with _sources.truncate_time(["m"], 2):
        first = osk.read("m")
        second = osk.read("m")
        third = osk.read("m")
    assert cuts == ["m"]
    assert memo_sources == ["m"]
    assert _days(first) == _days(second) == _days(third) == [1, 2, 9, 10]


def test_the_anchor_is_consulted_once(memo_sources, monkeypatch):
    ends = []
    real = _sources._anchor_end
    monkeypatch.setattr(
        _sources, "_anchor_end", lambda anchor: ends.append(anchor) or real(anchor)
    )
    with _sources.truncate_time(["anchor", "m"], 2, anchor="anchor"):
        for _ in range(3):
            assert _days(osk.read("m")) == [1, 2, 9, 10]
    assert ends == ["anchor"]


def test_a_different_truncation_gets_its_own_cut(memo_sources):
    with _sources.truncate_time(["m"], 1):
        assert _days(osk.read("m")) == [1, 10]
        with _sources.truncate_time(["m"], 3):
            assert _days(osk.read("m")) == [1, 2, 3, 8, 9, 10]
        assert _days(osk.read("m")) == [1, 10]


def test_leaving_truncate_time_empties_the_cut_memo(memo_sources):
    with _sources.truncate_time(["m"], 2):
        osk.read("m")
        assert _sources._TRUNCATED_CACHE
    assert not _sources._TRUNCATED_CACHE
    assert osk.read("m").sizes["time"] == 10


def test_cache_clear_empties_the_cut_memo(memo_sources):
    with _sources.truncate_time(["m"], 2):
        osk.read("m")
        assert _sources._TRUNCATED_CACHE
        osk.read.cache_clear()
        assert not _sources._TRUNCATED_CACHE
        assert _days(osk.read("m")) == [1, 2, 9, 10]
        assert memo_sources == ["m", "m"]  # reopened after the clear


def test_cut_results_are_independent_shallow_copies(memo_sources):
    with _sources.truncate_time(["m"], 2):
        a = osk.read("m")
        a.attrs["mine"] = 1
        a.coords["extra"] = ("time", np.arange(a.sizes["time"]))
        b = osk.read("m")
    assert "mine" not in b.attrs
    assert "extra" not in b.coords
    assert a is not b


def test_a_disabled_memo_still_truncates_without_storing(memo_sources):
    from ocean_skill import cache

    cache.disable()
    try:
        with _sources.truncate_time(["m"], 2):
            assert _days(osk.read("m")) == [1, 2, 9, 10]
            assert not _sources._TRUNCATED_CACHE
    finally:
        cache.enable()


def test_an_unnamed_source_is_not_memoized_as_cut(memo_sources):
    with _sources.truncate_time(["m"], 2):
        assert osk.read("other").sizes["time"] == 10
        assert not _sources._TRUNCATED_CACHE


@pytest.mark.parametrize(
    ("positions", "expected"),
    [
        ([3, 4, 5], slice(3, 6)),
        ([0], slice(0, 1)),
        ([0, 1, 8, 9], [0, 1, 8, 9]),
        ([], []),
    ],
)
def test_contiguous_positions_become_a_slice(positions, expected):
    assert _sources._as_slices(positions) == expected


def test_a_contiguous_cut_is_selected_with_a_slice(memo_sources, monkeypatch):
    seen = []
    real = xr.Dataset.isel

    def spy(self, indexers=None, **kw):
        seen.append(indexers)
        return real(self, indexers, **kw)

    monkeypatch.setattr(xr.Dataset, "isel", spy)
    with _sources.truncate_time(["m"], 5):  # first and last 5 of 10: all of it
        assert _days(osk.read("m")) == list(range(1, 11))
    assert seen == [{"time": slice(0, 10)}]
