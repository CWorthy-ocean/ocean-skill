"""A catalog entry redefined under the same name must miss the cache, not hit it.

The cache used to identify a source by its *name* alone, so a script that rewrote a
catalog -- ``cast0000`` now pointing at a different CSV, ``model_win`` at a different
time window through a reader chain -- silently got the previous definition's cached
result back. The key now also carries a **fingerprint of the entry's definition**
(:func:`ocean_skill.catalog.fingerprint`): the reader as written in the catalog file --
class, arguments, the URLs/paths they carry, any chained transforms, the ``data``
entries it references -- plus the metadata that changes what a read returns.

Three layers are pinned here: the fingerprint itself (what moves it, what must not, that
it is stable across processes), the three key functions that take it, and one real
cached calculator (:mod:`ocean_skill.tides`) wired through to the whole.

Everything is offline: CSVs written to ``tmp_path`` stand in for sources, and catalogs
are written with :mod:`ocean_skill.build` into the ``isolated_catalogs`` directory
(``tests/conftest.py``), so discovery sees exactly what each test writes.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from intake.readers import datatypes, readers

from ocean_skill import build, cache, catalog

ROOT = Path(__file__).resolve().parents[1]
TEMPERATURE = "sea_water_potential_temperature"

_mtime_bumps = iter(range(1, 10_000))


def _csv(tmp_path, name="cast0000.csv", extra=""):
    """Write a small station table and return its path (``extra`` varies the bytes)."""
    path = tmp_path / name
    path.write_text(
        "time,lon,lat,depth (m),temp (degC)\n"
        "2024-07-01 12:00,-150.0,61.0,5.0,8.0\n"
        "2024-07-01 13:00,-150.0,61.0,5.0,9.0\n"
        "2024-07-01 14:00,-150.0,61.0,5.0,10.0\n" + extra
    )
    return path


def _reader(csv, **kwargs):
    """Return a pandas CSV reader over ``csv`` (chain ``.tail(n)`` for a pipeline)."""
    return readers.PandasCSV(datatypes.CSV(url=str(csv)), **kwargs)


def _write(cats, stem, entries):
    """Write ``entries`` (``{name: (reader, metadata)}``) as ``cats/<stem>.yaml``.

    The file's mtime is pushed a whole second past the last write each time: discovery
    and the fingerprint memo both notice a rewritten catalog by ``(mtime_ns, size)``,
    and two quick writes of equal size can share a timestamp on a coarse-clock
    filesystem.
    """
    cat = build.new_catalog(title=stem)
    for name, (reader, metadata) in entries.items():
        build.add_source(
            cat,
            name,
            reader=reader,
            name_map=None,
            featureType="timeSeries",
            **metadata,
        )
    path = build.save(cat, cats / f"{stem}.yaml")
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + next(_mtime_bumps) * 10**9))
    return path


@pytest.fixture
def cats(isolated_catalogs):
    """Return the catalog directory discovery is pointed at (writes there are seen)."""
    return isolated_catalogs


# -- the fingerprint ----------------------------------------------------------------


def test_fingerprint_is_a_short_stable_hex_digest(cats, tmp_path):
    _write(cats, "mine", {"cast0000": (_reader(_csv(tmp_path)), {})})
    first = catalog.fingerprint("cast0000")
    assert re.fullmatch(r"[0-9a-f]{16}", first)
    assert catalog.fingerprint("cast0000") == first
    # the qualified spelling names the same entry, so it has the same definition
    assert catalog.fingerprint("mine:cast0000") == first


def test_a_different_data_path_is_a_different_definition(cats, tmp_path):
    """The motivating case: ``cast0000`` rewritten to point at a different CSV."""
    a = _csv(tmp_path, "a.csv")
    b = _csv(tmp_path, "b.csv")  # byte-identical contents; only the path differs
    _write(cats, "mine", {"cast0000": (_reader(a), {})})
    before = catalog.fingerprint("cast0000")

    _write(cats, "mine", {"cast0000": (_reader(b), {})})
    after = catalog.fingerprint("cast0000")
    assert after != before, "a redefined entry must not keep the old fingerprint"

    # and writing the old definition back restores the old fingerprint -- nothing in it
    # is random or tied to when the file was written
    _write(cats, "mine", {"cast0000": (_reader(a), {})})
    assert catalog.fingerprint("cast0000") == before


def test_reader_arguments_and_chained_steps_each_move_it(cats, tmp_path):
    """A window written as a reader chain (``.tail(n)``) is part of the definition."""
    csv = _csv(tmp_path)
    variants = {
        "plain": _reader(csv),
        "nrows=2": _reader(csv, nrows=2),
        "nrows=3": _reader(csv, nrows=3),
        "tail(2)": _reader(csv).tail(2),
        "tail(3)": _reader(csv).tail(3),
        "head(2)": _reader(csv).head(2),
    }
    prints = {}
    for label, reader in variants.items():
        _write(cats, "mine", {"model_win": (reader, {})})
        prints[label] = catalog.fingerprint("model_win")
    assert all(prints.values())
    assert len(set(prints.values())) == len(variants), prints


@pytest.mark.parametrize(
    "changed",
    [
        pytest.param({"time_zone": "America/Anchorage"}, id="time_zone"),
        pytest.param({"utc_offset_h": -9}, id="utc_offset_h"),
        pytest.param(
            {"standard_names": {"temp (degC)": TEMPERATURE}}, id="standard_names"
        ),
        pytest.param(
            {"depth_convention": {"origin": "surface", "positive": "up"}},
            id="depth_convention",
        ),
        pytest.param({"axes": {"Z": "temp (degC)"}}, id="axes"),
    ],
)
def test_metadata_that_changes_what_a_read_returns_moves_it(cats, tmp_path, changed):
    csv = _csv(tmp_path)
    _write(cats, "mine", {"cast0000": (_reader(csv), {})})
    plain = catalog.fingerprint("cast0000")
    _write(cats, "mine", {"cast0000": (_reader(csv), changed)})
    assert catalog.fingerprint("cast0000") != plain


@pytest.mark.parametrize(
    ("key", "old", "new"),
    [
        ("description", "the old words", "different words"),
        ("title", "Cast zero", "Cast zero (revised)"),
        ("tags", ["ctd"], ["ctd", "alaska"]),
        ("geospatial_lat_min", 60.0, 59.0),
        ("geospatial_lat_max", 62.0, 63.0),
        ("geospatial_lon_min", -151.0, -152.0),
        ("geospatial_lon_max", -149.0, -148.0),
        ("geospatial_vertical_min", 0.0, 1.0),
        ("time_coverage_start", "2024-07-01", "2024-06-01"),
        ("time_coverage_end", "2024-07-02", "2024-08-02"),
        ("minTime", "2024-07-01T12:00:00Z", "2024-06-01T00:00:00Z"),
        ("maxTime", "2024-07-01T14:00:00Z", "2024-08-01T00:00:00Z"),
        ("minLatitude", 60.0, 59.0),
        ("maxLatitude", 62.0, 63.0),
        ("minLongitude", -151.0, -152.0),
        ("maxLongitude", -149.0, -148.0),
        (
            "domain_outline",
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
            [[0, 0], [2, 0], [2, 2]],
        ),
        ("variables", ["temp"], ["temp", "salt"]),
    ],
)
def test_descriptive_and_derived_metadata_do_not_move_it(cats, tmp_path, key, old, new):
    """Text for people and what the build probe derived *from* the data say nothing new.

    Re-describing an entry, or re-probing it so its extents shift, must not throw away
    every cached result built from it -- the data the entry points at is what decides
    those, and a changed path or reader already moves the fingerprint on its own.
    """
    csv = _csv(tmp_path)
    _write(cats, "mine", {"cast0000": (_reader(csv), {key: old})})
    first = catalog.fingerprint("cast0000")
    assert (
        catalog.resolve("cast0000").metadata[key] == old
    )  # the key really was written

    _write(cats, "mine", {"cast0000": (_reader(csv), {key: new})})
    assert catalog.resolve("cast0000").metadata[key] == new
    assert catalog.fingerprint("cast0000") == first


def test_an_unknown_name_has_no_fingerprint(cats):
    assert catalog.fingerprint("no_such_source") == ""
    assert catalog.fingerprint("mine:no_such_source") == ""


@pytest.mark.parametrize(
    "stub",
    [
        pytest.param(SimpleNamespace(metadata={"featureType": "grid"}), id="namespace"),
        pytest.param(mock.Mock(metadata={}), id="Mock"),
        pytest.param(mock.MagicMock(metadata={}), id="MagicMock"),
        pytest.param(
            SimpleNamespace(metadata={}, name="x", path="/no/such/dir/x.yaml"),
            id="path-missing",
        ),
    ],
)
def test_a_resolved_object_with_no_entry_definition_has_none(monkeypatch, stub):
    """Tests stub ``catalog.resolve`` with bare metadata; this must not crash."""
    monkeypatch.setattr(catalog, "resolve", lambda name: stub)
    assert catalog.fingerprint("anything") == ""


def test_the_catalog_file_is_parsed_once_until_it_changes(cats, tmp_path, monkeypatch):
    csv = _csv(tmp_path)
    _write(cats, "mine", {"a": (_reader(csv), {}), "b": (_reader(csv).tail(2), {})})
    loads = []
    real = catalog._load_catalog_yaml
    monkeypatch.setattr(
        catalog, "_load_catalog_yaml", lambda path: loads.append(path) or real(path)
    )
    first_a, first_b = catalog.fingerprint("a"), catalog.fingerprint("b")
    catalog.fingerprint("a"), catalog.fingerprint("b")
    assert first_a != first_b
    assert len(loads) == 1

    _write(cats, "mine", {"a": (_reader(csv).tail(1), {}), "b": (_reader(csv), {})})
    assert catalog.fingerprint("a") != first_a
    assert len(loads) == 2


def test_the_fingerprint_is_the_same_in_another_process(cats, tmp_path):
    """Plain sha256 over sorted JSON: no hash randomization, no process-local state."""
    metadata = {
        "time_zone": "America/Anchorage",
        "standard_names": {"temp (degC)": "t"},
    }
    _write(cats, "mine", {"cast0000": (_reader(_csv(tmp_path)).tail(2), metadata)})
    here = catalog.fingerprint("cast0000")

    code = "import ocean_skill.catalog as c; print(c.fingerprint('cast0000'))"
    procs = []
    for seed in ("1", "2"):
        env = {
            **os.environ,
            "PYTHONPATH": str(ROOT),
            "PYTHONHASHSEED": seed,  # two different string-hash seeds
            "OCEAN_SKILL_DIR": str(tmp_path / f"osk{seed}"),
            "OCEAN_SKILL_CATALOGS": str(cats),
            "HOME": str(tmp_path / "home"),  # keep the real user catalog dirs out of it
        }
        procs.append(
            subprocess.Popen(
                [sys.executable, "-c", code],
                cwd=tmp_path,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        )
    for proc in procs:
        out, err = proc.communicate(timeout=300)
        assert proc.returncode == 0, err
        assert out.strip().splitlines()[-1] == here


# -- the fingerprint, on parsed catalog documents ------------------------------------
#
# What a catalog file round trip does not reach directly: how far references are
# followed, what a malformed reference does, and which parts of the document count.


def _doc():
    """Return a parsed catalog document: one reader entry over one data description."""
    return {
        "version": 2,
        "aliases": {"a": "a"},
        "entries": {
            "a": {
                "reader": "intake.readers.readers:PandasCSV",
                "kwargs": {"args": ["{data(d1)}"], "nrows": 3},
                "metadata": {"featureType": "timeSeries"},
                "output_instance": "pandas:DataFrame",
                "user_parameters": {},
            }
        },
        "data": {
            "d1": {
                "datatype": "intake.readers.datatypes:CSV",
                "kwargs": {"url": "/data/a.csv", "storage_options": None},
                "metadata": {},
                "user_parameters": {},
            }
        },
        "metadata": {},
        "user_parameters": {},
    }


def test_a_referenced_data_description_is_part_of_the_definition():
    """The entry itself is untouched here: only the data it points at moves."""
    base = catalog._entry_fingerprint(_doc(), "a")

    moved = _doc()
    moved["data"]["d1"]["kwargs"]["url"] = "/data/b.csv"
    assert catalog._entry_fingerprint(moved, "a") != base

    meaningful = _doc()
    meaningful["data"]["d1"]["metadata"] = {"anon": True}
    assert catalog._entry_fingerprint(meaningful, "a") != base

    inert = _doc()  # the same inert-metadata rule applies to what is referenced
    inert["data"]["d1"]["metadata"] = {"geospatial_lat_min": 3.0, "description": "x"}
    assert catalog._entry_fingerprint(inert, "a") == base


def test_a_chain_of_readers_is_followed_to_the_end():
    """A pipeline step names a reader entry, which names a data description."""
    doc = _doc()
    doc["entries"]["r1"] = dict(
        doc["entries"]["a"]
    )  # the bare reader, a chain's step 0
    doc["entries"]["chained"] = {
        "reader": "intake.readers.convert:Pipeline",
        "kwargs": {
            "steps": [
                ["{data(r1)}", [], {}],
                [
                    "{func(intake.readers.transform:Method)}",
                    [],
                    {"method_name": "tail"},
                ],
            ]
        },
        "metadata": {},
        "user_parameters": {},
    }
    base = catalog._entry_fingerprint(doc, "chained")

    doc["data"]["d1"]["kwargs"]["url"] = "/data/elsewhere.csv"  # two hops down
    assert catalog._entry_fingerprint(doc, "chained") != base


def test_dangling_and_cyclic_references_terminate():
    doc = _doc()
    doc["entries"]["a"]["kwargs"]["args"] = ["{data(missing)}"]
    dangling = catalog._entry_fingerprint(doc, "a")
    assert re.fullmatch(r"[0-9a-f]{16}", dangling)
    # a reference to nothing is a different definition from one that points somewhere
    doc["data"]["missing"] = {"datatype": "intake.readers.datatypes:CSV", "kwargs": {}}
    assert catalog._entry_fingerprint(doc, "a") != dangling

    loop = _doc()
    loop["entries"]["a"]["kwargs"]["args"] = ["{data(b)}"]
    loop["entries"]["b"] = {"reader": "r", "kwargs": {"args": ["{data(a)}"]}}
    assert re.fullmatch(r"[0-9a-f]{16}", catalog._entry_fingerprint(loop, "a"))


def test_a_reference_is_resolved_through_an_alias_as_intake_does():
    doc = _doc()
    doc["aliases"]["nick"] = "d1"
    doc["entries"]["a"]["kwargs"]["args"] = ["{data(nick)}"]
    base = catalog._entry_fingerprint(doc, "a")
    doc["data"]["d1"]["kwargs"]["url"] = "/data/b.csv"
    assert catalog._entry_fingerprint(doc, "a") != base


def test_mapping_keys_of_mixed_type_still_hash():
    """YAML allows ``1:`` beside ``x:``; plain ``sort_keys`` cannot order those."""
    doc = _doc()
    doc["entries"]["a"]["kwargs"]["mapping"] = {1: "one", "x": 2}
    first = catalog._entry_fingerprint(doc, "a")
    assert re.fullmatch(r"[0-9a-f]{16}", first)
    doc["entries"]["a"]["kwargs"]["mapping"] = {1: "uno", "x": 2}
    assert catalog._entry_fingerprint(doc, "a") != first


def test_catalog_wide_parameters_count_but_intakes_injected_ones_do_not():
    base = catalog._entry_fingerprint(_doc(), "a")

    param = _doc()
    param["user_parameters"] = {"year": {"default": "2020", "dtype": "str"}}
    with_param = catalog._entry_fingerprint(param, "a")
    assert with_param != base
    param["user_parameters"]["year"]["default"] = "2021"
    assert catalog._entry_fingerprint(param, "a") != with_param

    # where a file was last loaded from leaks into a re-saved catalog; not a definition
    injected = _doc()
    injected["user_parameters"] = {
        "CATALOG_PATH": "/old/place/x.yaml",
        "CATALOG_DIR": "/old/place",
        "STORAGE_OPTIONS": {},
    }
    assert catalog._entry_fingerprint(injected, "a") == base


def test_every_name_in_a_file_gets_a_fingerprint_aliases_included(tmp_path):
    import yaml

    doc = _doc()
    doc["aliases"] = {"nick": "a"}
    path = tmp_path / "hand.yaml"
    path.write_text(yaml.safe_dump(doc))
    prints = catalog._definition_fingerprints(path)
    assert set(prints) == {"a", "nick"}
    assert prints["nick"] == prints["a"] == catalog._entry_fingerprint(doc, "a")


def test_a_yaml_file_that_is_not_a_catalog_has_no_fingerprints(tmp_path):
    for text in ("just: a mapping\n", "- a\n- list\n", "entries: not-a-mapping\n", ""):
        path = tmp_path / "not_a_catalog.yaml"
        path.write_text(text)
        assert catalog._definition_fingerprints(path) == {}


# -- the key functions --------------------------------------------------------------

_KEY = {
    "test": "model",
    "reference": "cast0000",
    "variable": TEMPERATURE,
    "select": {"depth": 100},
    "method": "bilinear",
}


def test_the_cache_format_version_moved_when_the_definition_joined_the_key():
    """Version-8 entries were filed without it; the bump orphans, not reuses, them."""
    assert cache._FORMAT_VERSION >= 9


def test_aligned_key_notices_a_redefined_test_or_reference():
    plain = cache.key_for(**_KEY)
    assert plain == cache.key_for(**_KEY, test_definition="", reference_definition="")
    keys = {
        plain,
        cache.key_for(**_KEY, test_definition="aaaa"),
        cache.key_for(**_KEY, test_definition="bbbb"),
        cache.key_for(**_KEY, reference_definition="aaaa"),
        cache.key_for(**_KEY, reference_definition="bbbb"),
    }
    assert len(keys) == 5
    # the two sides are not interchangeable: the same two definitions swapped differ
    assert cache.key_for(
        **_KEY, test_definition="aaaa", reference_definition="bbbb"
    ) != cache.key_for(**_KEY, test_definition="bbbb", reference_definition="aaaa")


def test_prepared_key_notices_a_redefined_source():
    args = {"source": "model", "variable": TEMPERATURE, "select": {"depth": 100}}
    plain = cache.key_for_prepared(**args)
    assert plain == cache.key_for_prepared(**args, definition="")
    assert (
        len({plain, *(cache.key_for_prepared(**args, definition=d) for d in "ab")}) == 3
    )


def test_calculated_key_notices_a_redefined_source():
    args = {"source": "model", "name": "harmonic_constants", "params": {"k": 1}}
    plain = cache.key_for_calculated(**args)
    assert plain == cache.key_for_calculated(**args, definition="")
    assert (
        len({plain, *(cache.key_for_calculated(**args, definition=d) for d in "ab")})
        == 3
    )


# -- a real cached calculator -------------------------------------------------------


def test_a_redefined_source_misses_the_tidal_calculated_cache(
    cats, tmp_path, monkeypatch
):
    """``harmonic_constants`` caches by source identity; redefining it re-analyses."""
    pytest.importorskip("pyfes")
    from ocean_skill import tides
    from tests.test_tides import WAVES, _ssh

    ds, _ = _ssh({n: 0.1 for n in WAVES}, days=120, source="tide_model")
    calls = []
    real = tides._fit_cell
    monkeypatch.setattr(
        tides, "_fit_cell", lambda *a, **k: calls.append(1) or real(*a, **k)
    )

    a, b = _csv(tmp_path, "a.csv"), _csv(tmp_path, "b.csv")
    _write(cats, "mine", {"tide_model": (_reader(a), {})})

    tides.harmonic_constants(ds, constituents=WAVES)
    cold = len(calls)
    assert cold > 0
    tides.harmonic_constants(ds, constituents=WAVES)
    assert len(calls) == cold, "same definition: the second call is a cache hit"

    _write(cats, "mine", {"tide_model": (_reader(b), {})})  # redefined, same name
    tides.harmonic_constants(ds, constituents=WAVES)
    assert len(calls) > cold, "a redefined source must not be served the old analysis"
