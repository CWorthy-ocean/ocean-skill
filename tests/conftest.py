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


@pytest.fixture(autouse=True)
def fresh_regridder_memo():
    """Give every test its own empty regridder memo.

    :func:`ocean_skill.align._regridder_for` keys on grid *content*, so two tests
    that happen to build numerically identical grids would otherwise see each
    other's cached weights — a correct but confusing cross-test coupling, and one
    that would make a test asserting "the regridder was built" fail depending on
    what ran before it. Unlike ``isolated_cache`` above, this module-level dict
    is never reset on its own.
    """
    from ocean_skill import align

    align.clear_regridder_memo()
    yield
    align.clear_regridder_memo()


@pytest.fixture(autouse=True)
def fast_probe_retries(monkeypatch):
    """Take the sleep out of probe retries so failure-path tests stay instant.

    The build probes with :data:`ocean_skill.build.PROBE_RETRIES` re-attempts and real
    backoff; a test whose read never recovers would otherwise wait out that backoff.
    Zeroing only the wait keeps the retry *count* — and every assertion about it —
    exactly as in production.
    """
    from ocean_skill import build

    monkeypatch.setattr(build, "PROBE_RETRY_BACKOFF", 0.0)


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
