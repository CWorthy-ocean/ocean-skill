"""Shared pytest fixtures/config for ocean-skill."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path):
    """Point the aligned-result cache at a temp dir for *every* test.

    Autouse and unconditional: a test that reached the real user cache could both
    pollute it and — worse — pass by reading an entry a previous run left behind.
    Restores the module's own state afterwards rather than leaving an override set —
    including fsspec's, since ``enable`` now relocates the download cache too and that
    lives in fsspec's process-global config, outside this package.
    """
    import fsspec.config

    from ocean_skill import cache

    saved = (cache._enabled, cache._override_dir, cache._announced)
    saved_fsspec = {
        p: dict(fsspec.config.conf.get(p, {})) for p in cache._FSSPEC_CACHES
    }
    saved_applied = dict(cache._fsspec_applied)
    cache.enable(tmp_path / "osk")
    cache._announced = True  # keep the one-time banner out of test output
    yield cache
    cache._enabled, cache._override_dir, cache._announced = saved
    cache._fsspec_applied.clear()
    cache._fsspec_applied.update(saved_applied)
    for protocol, conf in saved_fsspec.items():
        fsspec.config.conf[protocol] = conf


@pytest.fixture
def isolated_catalogs(tmp_path, monkeypatch):
    """Point catalog discovery at an isolated temp dir holding one intake v2 catalog.

    Builds the catalog programmatically (the only correct way to make v2), sets
    ``$OCEAN_SKILL_CATALOGS`` to the temp dir, and chdirs away from the repo's own
    ``./catalogs``. The entry has metadata but is never read. Returns the temp dir.
    """
    import intake
    from intake.readers import datatypes, readers

    cats = tmp_path / "cats"
    cats.mkdir()

    data = datatypes.HDF5(url=str(tmp_path / "foo.nc"))  # never actually read
    reader = readers.XArrayDatasetReader(data)
    reader.metadata.update(
        {"featureType": "grid", "variables": ["sea_water_temperature"]}
    )
    cat = intake.entry.Catalog(metadata={"title": "example catalog"})
    cat["foo"] = reader
    cat.aliases["foo"] = "foo"
    cat.to_yaml_file(str(cats / "example.catalog.yaml"))

    monkeypatch.setenv("OCEAN_SKILL_CATALOGS", str(cats))
    monkeypatch.chdir(tmp_path)
    return cats
