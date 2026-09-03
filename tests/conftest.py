"""Shared pytest fixtures/config for ocean-skill."""

from __future__ import annotations

import matplotlib
import pytest

# Set once, at conftest import time -- before pytest collects (imports) a single test
# module, on every worker. That beats the race individual test files used to guard
# against by calling ``matplotlib.use("Agg")`` at their own module top: whichever file
# pytest happened to collect first won the race and fixed the backend for the rest of
# the process. Doing it here removes the race instead of winning it, so every one of
# those per-file calls (module-level and in-body alike) is now redundant and dropped.
matplotlib.use("Agg")


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
def fresh_availability_memo():
    """Give every test its own empty reference-availability memo.

    :func:`ocean_skill.comparison._variable_available` keys on generic source and
    variable names ("obs", "model", "temperature", ...) that recur across many
    test modules, so a positive result left standing by one test would silently
    short-circuit the read another test means to exercise. Mirrors
    ``fresh_regridder_memo`` above.
    """
    from ocean_skill import comparison

    comparison.clear_availability_memo()
    yield
    comparison.clear_availability_memo()


@pytest.fixture(autouse=True)
def _close_figures():
    """Close every figure after each test.

    The backend is already Agg (set at module import time, above); this just keeps
    one test's figures from accumulating into the next -- more likely to matter once
    tests run concurrently under xdist.
    """
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


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

    Discovery's search path also always includes the packaged reference catalogs
    (``ocean_skill/catalogs/``, unconditionally, so they resolve regardless of
    cwd) -- not something env/cwd can steer away from, unlike the other three
    tiers. A test asserting exact discovery contents or exact parse-call counts
    would otherwise also see the real shipped catalogs, so ``search_paths`` is
    wrapped to drop just that one packaged entry, leaving env/cwd behavior (and
    the real user-config-dir tier, deliberately not isolated here either) as
    ``search_paths`` would actually compute them.
    """
    import intake
    from intake.readers import datatypes, readers

    from ocean_skill import catalog

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

    packaged = catalog.Path(catalog.__file__).parent / "catalogs"
    real_search_paths = catalog.search_paths
    monkeypatch.setattr(
        catalog,
        "search_paths",
        lambda: [p for p in real_search_paths() if p != packaged],
    )
    return cats
