"""A catalog entry redefined under the same name misses the comparison's caches.

``catalog.fingerprint`` / ``cache.key_for*`` (see ``test_cache_definition_key.py``)
make a source's *definition* part of its cache key. This pins the plumbing in
``ocean_skill.comparison``: ``prepare_source`` files a lane under its source's
fingerprint, and ``Comparison._cache_key`` files the aligned pair under both sides' --
so a script that rewrites ``pier`` to point at a different CSV, then compares again with
the cache on, gets the new data back and not the old definition's result.

Offline: a tiny gridded NetCDF stands in for the model and CSVs written to ``tmp_path``
for the observation, both catalogued with :mod:`ocean_skill.build` into the
``isolated_catalogs`` directory (``tests/conftest.py``).
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from intake.readers import datatypes, readers

import ocean_skill as osk
from ocean_skill import build, cache, catalog
from ocean_skill.comparison import Comparison

LON, LAT = -150.0, 61.0
TEMPERATURE = "sea_water_temperature"
N = 6  # hourly steps, enough pairs to score
MODEL = 10.0 + np.arange(N)  # the model's field, the same at every cell

_mtime_bumps = iter(range(1, 10_000))


def _model_file(tmp_path):
    """Write a 3 x 3 grid around the station, ``temp`` rising an hour at a time."""
    path = tmp_path / "model.nc"
    ds = xr.Dataset(
        {
            "temp": (
                ("time", "lat", "lon"),
                np.broadcast_to(MODEL[:, None, None], (N, 3, 3)).astype("float64"),
                {"standard_name": TEMPERATURE, "units": "degC"},
            )
        },
        coords={
            "time": pd.date_range("2024-07-01 12:00", periods=N, freq="h"),
            "lat": ("lat", [LAT - 0.1, LAT, LAT + 0.1], {"units": "degrees_north"}),
            "lon": ("lon", [LON - 0.1, LON, LON + 0.1], {"units": "degrees_east"}),
        },
    )
    ds.to_netcdf(path)
    return path


def _csv(tmp_path, name, values):
    """Write a station table of ``values`` (one per hour) and return its path."""
    path = tmp_path / name
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2024-07-01 12:00", periods=len(values), freq="h"),
            "lon": LON,
            "lat": LAT,
            "temp (degC)": values,
        }
    )
    frame.to_csv(path, index=False)
    return path


def _write(cats, model, obs):
    """Write ``cats/mine.yaml``: the gridded ``model`` and the CSV-backed ``pier``.

    The file's mtime is pushed a whole second past the last write each time -- discovery
    and the fingerprint memo notice a rewrite by ``(mtime_ns, size)``, and two quick
    writes of equal size can share a timestamp on a coarse-clock filesystem.
    """
    cat = build.new_catalog(title="mine")
    build.add_source(cat, "model", model, name_map=None, featureType="grid")
    build.add_source(
        cat,
        "pier",
        reader=readers.PandasCSV(datatypes.CSV(url=str(obs))),
        name_map=None,
        featureType="timeSeries",
        standard_names={"temp (degC)": TEMPERATURE},
        nominal_depth_m=0.0,
        # a mooring is compared at the surface only if its entry says it is there
        geospatial_vertical_min=0.0,
        geospatial_vertical_max=0.0,
    )
    path = build.save(cat, cats / "mine.yaml")
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + next(_mtime_bumps) * 10**9))
    return path


@pytest.fixture
def sources(isolated_catalogs, tmp_path):
    """Return ``(rewrite, model)``: ``rewrite(values)`` points ``pier`` at a new CSV."""
    model = _model_file(tmp_path)
    count = iter(range(1, 100))

    def rewrite(values):
        return _write(
            isolated_catalogs, model, _csv(tmp_path, f"pier{next(count)}.csv", values)
        )

    return rewrite, model


def _compare(**kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return osk.compare(
            reference="pier",
            test="model",
            variables=[TEMPERATURE],
            cache=True,
            **kw,
        )


def _reference_values(out):
    assert len(out) == 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(out[0].aligned["reference"]).reshape(-1)


def test_the_same_name_pointed_at_new_data_is_not_served_the_old_result(sources):
    rewrite, _ = sources
    rewrite([8.0, 9.0, 10.0, 11.0, 12.0, 13.0])
    first = _reference_values(_compare())
    np.testing.assert_allclose(first, [8.0, 9.0, 10.0, 11.0, 12.0, 13.0])

    rewrite([5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    second = _reference_values(_compare())
    np.testing.assert_allclose(second, [5.0, 6.0, 7.0, 8.0, 9.0, 10.0])


def test_an_unchanged_definition_is_still_a_cache_hit(sources, monkeypatch):
    rewrite, _ = sources
    rewrite([8.0, 9.0, 10.0, 11.0, 12.0, 13.0])
    _compare()

    seen = []
    real = Comparison.align

    def spy(self, **kw):
        seen.append(self._cache_key)
        return real(self, **kw)

    monkeypatch.setattr(Comparison, "align", spy)
    # nothing is recomputed: the lanes are read back from the cache, never prepared
    monkeypatch.setattr(
        "ocean_skill.align.align",
        lambda *a, **k: pytest.fail("the aligned pair should have come from the cache"),
    )
    out = _compare()
    assert len(out) == 1
    assert len(seen) == 1


def test_without_a_definition_the_old_result_would_be_served(sources, monkeypatch):
    """The control: with name-only keys (no fingerprints) the second run is stale."""
    monkeypatch.setattr(catalog, "fingerprint", lambda name: "")
    rewrite, _ = sources
    rewrite([8.0, 9.0, 10.0, 11.0, 12.0, 13.0])
    _compare()
    rewrite([5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    stale = _reference_values(_compare())
    np.testing.assert_allclose(stale, [8.0, 9.0, 10.0, 11.0, 12.0, 13.0])


# -- the keys themselves ---------------------------------------------------------


@pytest.fixture
def lane_definitions(monkeypatch):
    """Record the ``(source, definition)`` of every lane key built."""
    seen = []
    real = cache.key_for_prepared

    def spy(*, source, variable, select, definition=""):
        seen.append((source, definition))
        return real(
            source=source, variable=variable, select=select, definition=definition
        )

    monkeypatch.setattr(cache, "key_for_prepared", spy)
    return seen


def test_each_lane_is_filed_under_its_own_sources_fingerprint(
    sources, lane_definitions
):
    rewrite, _ = sources
    rewrite([8.0, 9.0, 10.0, 11.0, 12.0, 13.0])
    _compare()
    definitions = dict(lane_definitions)
    assert definitions["model"] == catalog.fingerprint("model") != ""
    assert definitions["pier"] == catalog.fingerprint("pier") != ""
    assert definitions["model"] != definitions["pier"]


def test_only_the_redefined_lane_changes_its_key(sources, lane_definitions):
    rewrite, _ = sources
    rewrite([8.0, 9.0, 10.0, 11.0, 12.0, 13.0])
    _compare()
    before = dict(lane_definitions)
    lane_definitions.clear()

    rewrite([5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    _compare()
    after = dict(lane_definitions)
    assert after["pier"] != before["pier"]
    assert after["model"] == before["model"]  # the model's lane stays a hit


def _aligned_key(**kw):
    return Comparison(
        reference="pier",
        test="model",
        variable=TEMPERATURE,
        cache=True,
        **kw,
    )._cache_key


def test_the_aligned_key_moves_when_only_a_definition_changes(sources):
    rewrite, _ = sources
    rewrite([8.0, 9.0, 10.0, 11.0, 12.0, 13.0])
    original = _aligned_key()
    assert _aligned_key() == original  # nothing changed, nothing moved

    rewrite([5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    assert _aligned_key() != original


def test_each_sides_definition_is_its_own_field_of_the_key(sources, monkeypatch):
    rewrite, _ = sources
    rewrite([8.0, 9.0, 10.0, 11.0, 12.0, 13.0])
    seen = []
    real = cache.key_for

    def spy(**kw):
        seen.append(kw)
        return real(**kw)

    monkeypatch.setattr(cache, "key_for", spy)
    _aligned_key()
    (kw,) = seen
    assert kw["test_definition"] == catalog.fingerprint("model") != ""
    assert kw["reference_definition"] == catalog.fingerprint("pier") != ""
    assert kw["test_definition"] != kw["reference_definition"]


def test_a_source_with_no_definition_keys_on_its_name_alone(monkeypatch):
    seen = []
    real = cache.key_for

    def spy(**kw):
        seen.append(kw)
        return real(**kw)

    monkeypatch.setattr(cache, "key_for", spy)
    monkeypatch.setattr(catalog, "fingerprint", lambda name: "")
    Comparison(
        reference="nope_ref", test="nope_test", variable=TEMPERATURE, cache=True
    )._cache_key
    assert seen[0]["test_definition"] == ""
    assert seen[0]["reference_definition"] == ""


def test_a_sections_reference_definition_is_its_casts_in_order(
    isolated_catalogs, tmp_path, monkeypatch
):
    """The display name ("cast_1+cast_2") resolves to nothing; the casts count."""
    model = _model_file(tmp_path)
    cat = build.new_catalog(title="casts")
    for i, values in enumerate(([1.0, 2.0], [3.0, 4.0]), start=1):
        build.add_source(
            cat,
            f"cast_{i}",
            reader=readers.PandasCSV(
                datatypes.CSV(url=str(_csv(tmp_path, f"cast_{i}.csv", values)))
            ),
            name_map=None,
            featureType="timeSeriesProfile",
            standard_names={"temp (degC)": TEMPERATURE},
        )
    build.save(cat, isolated_catalogs / "casts.yaml")
    seen = []
    real = cache.key_for

    def spy(**kw):
        seen.append(kw)
        return real(**kw)

    monkeypatch.setattr(cache, "key_for", spy)
    kwargs = dict(
        test="nope_test",
        variable=TEMPERATURE,
        select={"transect": {"from": "reference"}, "depth": [50.0, 200.0]},
        cache=True,
    )
    forward = Comparison(
        reference="cast_1+cast_2", section_casts=["cast_1", "cast_2"], **kwargs
    )
    backward = Comparison(
        reference="cast_2+cast_1", section_casts=["cast_2", "cast_1"], **kwargs
    )
    first, second = catalog.fingerprint("cast_1"), catalog.fingerprint("cast_2")
    assert first and second and first != second
    forward_key, backward_key = forward._cache_key, backward._cache_key
    assert [kw["reference_definition"] for kw in seen] == [
        f"{first}+{second}",
        f"{second}+{first}",
    ]
    assert forward_key != backward_key
    assert model.exists()
