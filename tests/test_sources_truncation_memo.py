"""``read()`` under ``truncate_time`` cuts each source once, not once per call.

The cut result is memoized beside the read memo (``sources._TRUNCATED_CACHE``) because a
page reads the same source dozens of times and a cut of a large model costs seconds.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import ocean_skill as osk
from ocean_skill import cache
from ocean_skill import sources as _sources


def _model(n=10):
    return xr.Dataset(
        {"temp": (("time", "y"), np.arange(n * 2.0).reshape(n, 2))},
        coords={"time": pd.date_range("2020-01-01", periods=n, freq="D")},
    )


@pytest.fixture
def memo_sources(monkeypatch, tmp_path):
    """Stub sources with a real catalog file (the read memo stats it); count reopens."""
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
        reads = [osk.read("m") for _ in range(3)]
    assert cuts == ["m"] and memo_sources == ["m"]
    assert all(_days(r) == [1, 2, 9, 10] for r in reads)


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


def test_nested_truncations_get_their_own_cut_and_the_memo_clears_on_exit(
    memo_sources,
):
    with _sources.truncate_time(["m"], 1):
        assert _days(osk.read("m")) == [1, 10]
        with _sources.truncate_time(["m"], 3):
            assert _days(osk.read("m")) == [1, 2, 3, 8, 9, 10]
        assert _days(osk.read("m")) == [1, 10]
        assert _sources._TRUNCATED_CACHE
    assert not _sources._TRUNCATED_CACHE
    assert osk.read("m").sizes["time"] == 10


def test_cache_clear_empties_the_cut_memo(memo_sources):
    with _sources.truncate_time(["m"], 2):
        osk.read("m")
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
    assert "mine" not in b.attrs and "extra" not in b.coords and a is not b


def test_a_disabled_memo_and_an_unnamed_source_are_not_stored(memo_sources):
    with _sources.truncate_time(["m"], 2):
        assert osk.read("other").sizes["time"] == 10  # not named: uncut
        assert not _sources._TRUNCATED_CACHE
        cache.disable()
        try:
            assert _days(osk.read("m")) == [1, 2, 9, 10]  # still cut
            assert not _sources._TRUNCATED_CACHE
        finally:
            cache.enable()


@pytest.mark.parametrize(
    ("positions", "expected"),
    [
        ([3, 4, 5], [slice(3, 6)]),
        ([0, 1, 8, 9], [slice(0, 2), slice(8, 10)]),
        ([2, 5, 6, 9], [slice(2, 3), slice(5, 7), slice(9, 10)]),
        ([], []),
    ],
)
def test_contiguous_runs_become_slices(positions, expected):
    assert _sources._as_slices(positions) == expected


@pytest.mark.parametrize("keep", [5, 2])  # first/last N of 10: all, or two runs
def test_the_cut_is_selected_with_slices_never_an_integer_list(
    memo_sources, monkeypatch, keep
):
    seen = []
    real = xr.Dataset.isel

    def spy(self, indexers=None, **kw):
        seen.append(indexers)
        return real(self, indexers, **kw)

    monkeypatch.setattr(xr.Dataset, "isel", spy)
    with _sources.truncate_time(["m"], keep):
        assert len(_days(osk.read("m"))) == min(2 * keep, 10)
    assert seen and all(isinstance(v, slice) for ix in seen if ix for v in ix.values())
    if keep == 5:
        assert seen == [{"time": slice(0, 10)}]
