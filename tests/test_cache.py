"""Tests for the on-disk cache of aligned comparison results.

Two things have to hold for a cache to be worth having. It must actually skip the
work — asserted here by counting calls to the expensive step, not by timing, which
would be flaky. And a cached result must be *indistinguishable* from a freshly
computed one, since anything else turns "did this come from cache?" into a variable
every downstream bug report has to account for.

The third theme is that a cache must never be able to break a pipeline that would
otherwise have worked: a corrupt entry or an unwritable directory degrades to a
recompute with a warning, never an exception.
"""

from __future__ import annotations

import os
import pathlib
import warnings
from unittest import mock

import numpy as np
import pytest
import xarray as xr

from ocean_skill import align as _align
from ocean_skill import cache
from ocean_skill.comparison import Comparison

NITRATE = "mole_concentration_of_nitrate_in_sea_water"


@pytest.fixture
def aligned():
    """Build an aligned pair: curvilinear test regridded onto a regular obs grid."""
    ny, nx = 12, 14
    lon = np.linspace(260, 270, nx)[None, :] * np.ones((ny, 1))
    lat = np.linspace(18, 28, ny)[:, None] * np.ones((1, nx))
    test = xr.DataArray(
        20 + np.random.default_rng(0).standard_normal((ny, nx)),
        dims=("eta", "xi"),
        coords={"lon": (("eta", "xi"), lon), "lat": (("eta", "xi"), lat)},
        attrs={"units": "mmol/m^3"},
    )
    reference = xr.DataArray(
        19 + np.zeros((8, 9)),
        dims=("lat", "lon"),
        coords={"lat": np.linspace(19, 27, 8), "lon": np.linspace(261, 269, 9)},
        attrs={"units": "mmol/m^3"},
    )
    out = _align.align(test, reference, test_name="test", reference_name="reference")
    out.attrs["actual_depth"] = 100.0
    return out


# -- keys ---------------------------------------------------------------------


def _key(**over):
    args = {
        "test": "model",
        "reference": "woa",
        "variable": NITRATE,
        "select": {"depth": 100},
        "method": "conservative_normed",
    }
    return cache.key_for(**{**args, **over})


def test_same_inputs_give_the_same_key():
    assert _key() == _key()


@pytest.mark.parametrize(
    "different",
    [
        {"test": "other_model"},
        {"reference": "glodap"},
        {"variable": "mole_concentration_of_phosphate_in_sea_water"},
        {"select": {"depth": 50}},
        {"method": "bilinear"},
    ],
)
def test_every_input_participates_in_the_key(different):
    """Anything that changes the result must change the key, or it serves stale data."""
    assert _key(**different) != _key()


def test_key_ignores_dict_ordering():
    """Two identical selections written in a different order are the same request."""
    a = _key(select={"depth": 100, "time": "2012-01"})
    b = _key(select={"time": "2012-01", "depth": 100})
    assert a == b


def test_key_survives_values_plain_json_cannot_encode():
    """A numpy depth or a slice in a selection must hash, not raise."""
    assert _key(select={"depth": np.float64(100.0)})
    assert _key(select={"time": slice("2012-01", "2012-02")})


# -- round-trip fidelity ------------------------------------------------------


def test_cached_result_is_indistinguishable_from_a_fresh_one(aligned):
    cache.save("k", aligned)
    back = cache.load("k")

    assert list(back.data_vars) == list(aligned.data_vars), "variable order changed"
    assert back.attrs == aligned.attrs
    for name, var in aligned.data_vars.items():
        assert np.allclose(back[name], var, equal_nan=True)
        assert back[name].attrs == var.attrs
    for name, coord in aligned.coords.items():
        assert np.allclose(back[name], coord)


def test_saving_does_not_mutate_the_callers_dataset(aligned):
    """save() records bookkeeping of its own; the live object must not sprout it."""
    before = dict(aligned.attrs)
    cache.save("k", aligned)
    assert aligned.attrs == before


def test_miss_returns_none():
    assert cache.load("no_such_key") is None


# -- robustness ---------------------------------------------------------------


def test_corrupt_entry_warns_and_is_treated_as_a_miss(aligned):
    """A half-written store must degrade to a recompute, never an exception."""
    cache.save("k", aligned)
    store = cache.path() / "k.zarr"
    for zarray in list(store.rglob("*.json")) + list(store.rglob("zarr.json")):
        zarray.write_text("{ not json")

    with pytest.warns(UserWarning, match="unreadable cache entry"):
        assert cache.load("k") is None
    assert not store.exists(), "a corrupt entry should be removed, not left to re-warn"


def test_unwritable_cache_warns_but_does_not_raise(aligned, monkeypatch):
    """Being unable to cache a result is not a reason to fail work already done."""
    monkeypatch.setattr(
        cache, "path", lambda: (_ for _ in ()).throw(OSError("read-only fs"))
    )
    with pytest.warns(UserWarning, match="could not cache"):
        cache.save("k", aligned)


def test_disabled_cache_neither_reads_nor_writes(aligned):
    cache.disable()
    try:
        cache.save("k", aligned)
        assert not cache.entries()
        assert cache.load("k") is None
    finally:
        cache.enable()


# -- housekeeping -------------------------------------------------------------


def test_clear_removes_entries_and_reports_the_count(aligned):
    cache.save("a", aligned)
    cache.save("b", aligned)
    assert len(cache.entries()) == 2
    assert cache.clear() == 2
    assert cache.entries() == []


def test_info_reports_state_and_location(aligned):
    assert "empty" in cache.info()
    cache.save("k", aligned)
    text = cache.info()
    assert "1 aligned" in text and str(cache.base_dir() / "cache") in text
    cache.disable()
    try:
        assert "off" in cache.info()
    finally:
        cache.enable()


# -- integration with Comparison ----------------------------------------------


@pytest.fixture
def counted_pipeline(aligned):
    """Patch out the expensive read/reduce step, counting how often it runs."""
    calls = {"n": 0}
    test, reference = aligned["test"], aligned["reference"]

    def fake_prepare(obj, meta, variable, select, aggregate=None, *, source=None):
        calls["n"] += 1
        return (test, None) if meta.get("model") == "roms" else (reference, 100.0)

    def fake_resolve(name):
        return mock.Mock(metadata={"model": "roms"} if name == "model" else {})

    with (
        mock.patch("ocean_skill.comparison._prepare", fake_prepare),
        mock.patch("ocean_skill.catalog.resolve", fake_resolve),
        mock.patch("ocean_skill.read", lambda n: None),
    ):
        yield calls


def _comparison(**kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the "resolved to ..." notice
        return Comparison(
            reference="woa",
            test="model",
            variable="nitrate",
            select={"depth": 100},
            **kw,
        )


def test_second_run_reuses_the_cache_instead_of_recomputing(counted_pipeline):
    """The whole point: a fresh Comparison must not redo the work."""
    first = _comparison().align()
    assert counted_pipeline["n"] == 2  # reference + test

    second = _comparison().align()
    assert counted_pipeline["n"] == 2, "cache hit should have skipped _prepare entirely"
    assert np.allclose(first["difference"], second["difference"], equal_nan=True)
    assert list(first.data_vars) == list(second.data_vars)


def test_cached_run_restores_actual_depth(counted_pipeline):
    """State that lives outside the Dataset must survive the round trip too.

    ``_actual_depth`` feeds the metrics record; a cached comparison that lost it
    would write a subtly different CSV than the run that populated the cache.
    """
    cold = _comparison()
    cold.align()
    warm = _comparison()
    warm.align()
    assert warm._actual_depth == cold._actual_depth == 100.0


def test_cached_and_fresh_metrics_agree(counted_pipeline):
    cold = _comparison()
    cold.align()
    warm = _comparison()
    warm.align()
    assert warm.metrics()["bias"] == cold.metrics()["bias"]


def test_cache_false_bypasses_the_cache(counted_pipeline):
    _comparison().align()
    before = counted_pipeline["n"]
    _comparison(cache=False).align()
    assert counted_pipeline["n"] > before


def test_refresh_recomputes_and_overwrites(counted_pipeline):
    _comparison().align()
    before = counted_pipeline["n"]
    _comparison().align(refresh=True)
    assert counted_pipeline["n"] > before


def test_a_different_selection_is_a_different_entry(counted_pipeline):
    _comparison().align()
    before = counted_pipeline["n"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Comparison(
            reference="woa", test="model", variable="nitrate", select={"depth": 50}
        ).align()
    assert counted_pipeline["n"] > before
    assert len(cache.entries("aligned")) == 2


def test_two_fanned_months_are_two_cache_entries(counted_pipeline):
    """Exactly the shape a `times=` fan produces: same pair, one select per month."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Comparison(
            reference="woa",
            test="model",
            variable="nitrate",
            select={"depth": 100, "time": "2010-01"},
        ).align()
    before = counted_pipeline["n"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Comparison(
            reference="woa",
            test="model",
            variable="nitrate",
            select={"depth": 100, "time": "2010-02"},
        ).align()
    assert counted_pipeline["n"] > before
    assert len(cache.entries("aligned")) == 2

    # and re-running the first month again is a pure cache hit, not a third build
    after_second_month = counted_pipeline["n"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Comparison(
            reference="woa",
            test="model",
            variable="nitrate",
            select={"depth": 100, "time": "2010-01"},
        ).align()
    assert counted_pipeline["n"] == after_second_month
    assert len(cache.entries("aligned")) == 2


def test_one_test_against_several_references_prepares_its_lane_once(counted_pipeline):
    """The lane layer's whole purpose: a model's own work must not repeat per pair.

    Its read + time mean + vertical transform depend only on (source, variable,
    select) — not on which reference it is about to be compared against.
    """
    for reference in ("woa", "glodap"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Comparison(
                reference=reference,
                test="model",
                variable="nitrate",
                select={"depth": 100},
            ).align()

    # 2 references prepared once each + the model prepared once in total = 3.
    # Without the lane cache the model would be prepared twice, giving 4.
    assert counted_pipeline["n"] == 3
    assert len(cache.entries("prepared")) == 3  # woa, glodap, model
    assert len(cache.entries("aligned")) == 2  # one per pair


def test_lane_field_round_trips_with_its_name_and_depth(aligned):
    """A prepared lane carries state outside the array that must survive too."""
    field = aligned["reference"].rename("moles_of_nitrate_per_unit_mass_in_sea_water")
    cache.save_field("k", field, 100.0)
    back, depth = cache.load_field("k")

    assert back.name == field.name
    assert depth == 100.0
    assert np.allclose(back, field, equal_nan=True)
    assert back.attrs == field.attrs, "bookkeeping attrs must not leak onto the array"


def test_clear_can_target_one_kind(aligned):
    cache.save("a", aligned)
    cache.save_field("b", aligned["reference"], None)
    assert cache.clear("prepared") == 1
    assert len(cache.entries("aligned")) == 1
    assert cache.clear() == 1


class TestDownloadLocationFollowsTheCache:
    """Relocating the cache must move *downloads* too, not just the result layers.

    The bug this pins down: ``enable(dir)`` moved ``prepared/`` and ``aligned/`` and
    ``info()`` duly reported the new directory, while every downloaded source file
    kept landing under the old one — because fsspec's location was set once at import
    and never revisited. A half-move that reports itself as complete is worse than no
    move at all, so each half is asserted separately.
    """

    def test_enable_relocates_the_fsspec_download_cache(self, tmp_path):
        import fsspec.config

        cache.enable(tmp_path / "moved")

        assert cache.obs_dir() == tmp_path / "moved" / "cache" / "obs"
        for protocol in cache._FSSPEC_CACHES:
            assert fsspec.config.conf[protocol]["cache_storage"] == str(cache.obs_dir())

    def test_a_location_the_user_set_is_never_overwritten(self, tmp_path):
        """Someone who pointed fsspec at HPC scratch must keep it — and be told so."""
        import fsspec.config

        fsspec.config.conf["simplecache"] = {"cache_storage": "/scratch/mine"}

        cache.enable(tmp_path / "moved")

        assert fsspec.config.conf["simplecache"]["cache_storage"] == "/scratch/mine"
        assert cache.obs_dir() == pathlib.Path("/scratch/mine")
        assert "/scratch/mine" in str(cache.info()), "info must not claim the wrong dir"

    def test_info_reports_where_downloads_go(self, tmp_path):
        cache.enable(tmp_path / "moved")
        assert str(cache.obs_dir()) in str(cache.info())

    def test_pooch_reader_defaults_to_the_same_directory(self, tmp_path):
        """The catalog used to hardcode one machine's home dir; the default resolves."""
        from ocean_skill.readers import PoochTarNetCDF

        cache.enable(tmp_path / "moved")
        with mock.patch("pooch.retrieve", return_value=[]) as retrieve:
            with pytest.raises(FileNotFoundError):  # no members: we only want the call
                PoochTarNetCDF()._read(url="https://example.org/x.tar.gz")

        assert retrieve.call_args.kwargs["path"] == str(cache.obs_dir())

    def test_pooch_reader_still_honours_an_explicit_cache_dir(self, tmp_path):
        from ocean_skill.readers import PoochTarNetCDF

        with mock.patch("pooch.retrieve", return_value=[]) as retrieve:
            with pytest.raises(FileNotFoundError):
                PoochTarNetCDF()._read(
                    url="https://example.org/x.tar.gz", cache_dir="~/somewhere"
                )

        assert retrieve.call_args.kwargs["path"] == os.path.expanduser("~/somewhere")
