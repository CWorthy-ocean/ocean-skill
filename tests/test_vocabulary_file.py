"""Tests for the shared vocabulary file, ``ocean_skill/vocab/vocabulary.yaml``.

The file is the vocabulary's one source of truth and is written in cf-xarray's own
``custom_criteria`` shape so other packages can load it with no ocean-skill code
(see ``ocean_skill/vocab/README.md``). These pin that promise from both ends: the
file obeys its own rules and round-trips through ocean-skill's parser without loss,
and plain cf-xarray / cf-pandas -- with ocean-skill not even imported -- resolve
names through it. Name resolution itself is ``test_vocabulary.py``'s business.
"""

from __future__ import annotations

import copy
import re
import subprocess
import sys
import textwrap
from importlib import resources
from pathlib import Path

import cf_pandas
import cf_xarray
import numpy as np
import pytest
import xarray as xr
import yaml

from ocean_skill import vocabulary
from tests.test_vocabulary import pristine_vocabulary  # noqa: F401  (fixture)

VOCAB_FILE = Path(str(resources.files("ocean_skill") / "vocab" / "vocabulary.yaml"))
CRITERIA = yaml.safe_load(VOCAB_FILE.read_text(encoding="utf-8"))

MLD_SIGMA_THETA = "ocean_mixed_layer_thickness_defined_by_sigma_theta"


def _wrapped(part: str) -> bool:
    return part.startswith("(?:") and part.endswith(")")


def _options() -> list:
    return list(cf_xarray.options.OPTIONS["custom_criteria"])


@pytest.fixture
def restore_cf_xarray_options(monkeypatch):
    """Put cf-xarray's global criteria (and our record of ours) back after a test."""
    saved = copy.deepcopy(_options())
    monkeypatch.setattr(vocabulary, "_REGISTERED", vocabulary._REGISTERED)
    yield
    cf_xarray.set_options(custom_criteria=saved)


# -- the file obeys its own rules ---------------------------------------------


def test_the_file_is_lossless_through_the_parser():
    """Parsing the file and writing the result back gives exactly the file.

    The invariant that keeps the dict and the file from drifting apart: anything the
    parser cannot represent, or the writer cannot reproduce, fails here -- including
    key order, which the YAML is ordered by.
    """
    parsed = vocabulary._load_vocabulary()
    written = vocabulary._to_criteria(parsed)
    assert written == CRITERIA
    assert list(written) == list(CRITERIA)
    assert list(parsed) == list(CRITERIA)


@pytest.mark.parametrize("key", list(CRITERIA))
def test_every_entry_obeys_the_file_rules(key):
    entry = CRITERIA[key]
    assert set(entry) == {"name", "standard_name"}
    for regex in entry.values():
        assert regex.startswith("^(?i:") and regex.endswith(")$")
        re.compile(regex)

    # `standard_name`: escaped literals only (one, or a broad entry's own + narrower).
    for part in vocabulary._alternatives_of(entry["standard_name"]):
        assert re.escape(vocabulary._unescape(part)) == part
    # `name`: each part is a wrapped pattern or an escaped literal -- never a bare
    # regex, which would be indistinguishable from an alias.
    for part in vocabulary._alternatives_of(entry["name"]):
        if _wrapped(part):
            re.compile(part[3:-1])
        else:
            assert re.escape(vocabulary._unescape(part)) == part, part


def test_every_regex_is_anchored_the_way_roms_tools_requires():
    """ROMS-Tools only accepts patterns that start with ``^`` and end with ``$``."""
    for key, entry in CRITERIA.items():
        for attribute, regex in entry.items():
            assert regex.startswith("^") and regex.endswith("$"), (key, attribute)


def test_a_broad_entry_matches_every_narrower_one():
    """Plain ``ds.cf["mld"]`` must find any definition, by name or by attribute."""
    parsed = vocabulary._load_vocabulary()
    broad = {e["broader"] for e in parsed.values() if "broader" in e}
    assert broad == {"mld"}  # the shipped case; the loop below is the general rule
    for key, entry in parsed.items():
        if "broader" not in entry:
            continue
        parent = CRITERIA[entry["broader"]]
        assert re.match(parent["name"], key)
        assert re.match(parent["name"], entry["standard_name"])
        assert re.match(parent["standard_name"], entry["standard_name"])
        # ... and not the other way round: a specific entry never matches the generic
        # name or a sibling.
        assert not re.match(CRITERIA[key]["name"], entry["broader"])
        for sibling, other in parsed.items():
            if sibling != key and other.get("broader") == entry["broader"]:
                assert not re.match(CRITERIA[key]["name"], sibling)
                assert not re.match(CRITERIA[key]["name"], other["standard_name"])


def test_a_pattern_with_no_regex_metacharacter_stays_a_pattern():
    """``doxy`` looks like an escaped alias; the ``(?:...)`` wrapper says it is not."""
    parsed = vocabulary._load_vocabulary()
    assert re.escape("doxy") == "doxy"
    assert "doxy" in parsed["oxygen"]["patterns"]
    assert "doxy" not in parsed["oxygen"].get("aliases", [])


# -- the parser and writer on awkward input -----------------------------------

#: Every awkward thing the two halves must agree on: regex metacharacters in an alias
#: and a standard_name, a pattern with a top-level ``|``, one with ``|``/``)`` inside
#: a character class, an escaped ``\|``, a nested group, a pattern that looks literal,
#: and a broad entry with two narrower ones (one given as a key).
_TRICKY = {
    "temp": {
        "standard_name": "sea.water_temp",
        "aliases": ["T(C)", "t|c", "TEMP-1", "a b"],
        "patterns": ["doxy", "a|b", r"q[|)]z", r"x\|y", "(?:z)", "[]|]+"],
    },
    "gen": {"standard_name": "gen_std", "aliases": ["generic"]},
    "gen_a": {"standard_name": "gen_std_a", "aliases": ["A-1"], "broader": "gen"},
    "gen_b": {"standard_name": "gen_std_b", "patterns": ["bb"], "broader": "gen"},
}


def test_the_parser_and_writer_agree_on_awkward_input():
    criteria = vocabulary._to_criteria(_TRICKY)
    for regex in (r for entry in criteria.values() for r in entry.values()):
        re.compile(regex)
    parsed = vocabulary._from_criteria(criteria)
    assert parsed == _TRICKY
    assert list(parsed) == list(_TRICKY)
    assert {k: list(v) for k, v in parsed.items()} == {
        k: list(v) for k, v in _TRICKY.items()
    }
    # The regexes do what they say, too: literals match exactly, patterns by family.
    name = criteria["temp"]["name"]
    for text in ["T(C)", "t|c", "doxy", "DOXY", "b", "q|z", "x|y", "z", "]"]:
        assert re.match(name, text), text
    assert not re.match(name, "doxy_qc")
    assert re.match(criteria["gen"]["name"], "bb")  # broad finds a narrower pattern
    assert not re.match(criteria["gen_a"]["name"], "bb")  # siblings stay apart


def test_the_writer_resolves_broader_given_as_any_spelling():
    """``register(..., broader="generic")`` names the broad entry by an alias."""
    vocab = copy.deepcopy(_TRICKY)
    vocab["gen_a"]["broader"] = "GENERIC"
    assert vocabulary._from_criteria(vocabulary._to_criteria(vocab)) == _TRICKY


def test_the_writer_leaves_out_a_broader_link_the_resolver_would_refuse():
    """The links ``_build_narrower`` warns about are silently not written."""
    vocab = copy.deepcopy(_TRICKY)
    vocab["gen_a"]["broader"] = "nothing_like_it"  # names nothing known
    parsed = vocabulary._from_criteria(vocabulary._to_criteria(vocab))
    assert "broader" not in parsed["gen_a"]

    vocab = copy.deepcopy(_TRICKY)
    vocab["gen_b"]["broader"] = "gen_a"  # a chain: gen_a is itself a kind of gen
    parsed = vocabulary._from_criteria(vocabulary._to_criteria(vocab))
    assert parsed["gen_a"]["broader"] == "gen"
    assert "broader" not in parsed["gen_b"]


def test_a_bare_unescaped_alias_in_a_file_is_refused():
    bad = {"a": {"name": "^(?i:a|a_std|x.y)$", "standard_name": "^(?i:a_std)$"}}
    with pytest.raises(ValueError, match="escape"):
        vocabulary._from_criteria(bad)


def test_a_regex_outside_the_file_shape_is_refused():
    bad = {"a": {"name": "(?i)a$", "standard_name": "^(?i:a)$"}}
    with pytest.raises(ValueError, match=r"\^\(\?i:a\|b"):
        vocabulary._from_criteria(bad)


# -- the file is plain cf-xarray / cf-pandas data -----------------------------

_PLAIN_CF_XARRAY = textwrap.dedent(
    """
    import sys

    import cf_xarray
    import numpy as np
    import xarray as xr
    import yaml

    criteria = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))

    def var(**attrs):
        return ("x", np.ones(3), attrs)

    ds = xr.Dataset(
        {
            "NO3": var(),
            "temp_ctd": var(),
            "ocean_mixed_layer_thickness_defined_by_sigma_theta": var(),
        }
    )
    with cf_xarray.set_options(custom_criteria=criteria):
        assert ds.cf["nitrate"].name == "NO3"
        assert ds.cf["temperature"].name == "temp_ctd"
        assert ds.cf["mld"].name == "ocean_mixed_layer_thickness_defined_by_sigma_theta"

        # A raw product name that states its definition only as an attribute.
        sigma_t = "ocean_mixed_layer_thickness_defined_by_sigma_t"
        raw = xr.Dataset({"mld_dt_mean": var(standard_name=sigma_t)})
        assert raw.cf["mld"].name == "mld_dt_mean"

        def finds(dataset, key):
            try:
                dataset.cf[key]
            except KeyError:
                return False
            return True

        # ... which a request for a different definition does not find,
        assert not finds(raw, "mld_by_sigma_theta")
        # and a QC companion is not the data variable.
        assert not finds(xr.Dataset({"NO3_qc_agg": var()}), "nitrate")

    assert not [m for m in sys.modules if m.split(".")[0] == "ocean_skill"]
    """
)


def test_plain_cf_xarray_resolves_names_through_the_file(tmp_path):
    """No ocean-skill import at all: the file alone is enough."""
    result = subprocess.run(
        [sys.executable, "-c", _PLAIN_CF_XARRAY, str(VOCAB_FILE)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_cf_pandas_resolves_names_through_the_file():
    assert cf_pandas.match_criteria_key(["NO3", "x"], "nitrate", CRITERIA) == ["NO3"]
    vocab = cf_pandas.Vocab()
    vocab.vocab.update(CRITERIA)
    columns = ["Temperature_CTD", "temp_ctd_qc", "air_temperature", "x"]
    assert cf_pandas.match_criteria_key(columns, "temperature", vocab.vocab) == [
        "Temperature_CTD"
    ]


# -- registering with cf-xarray merges, never replaces ------------------------

_THIRD_PARTY = {"third_thing": {"name": "^(?i:zzz|third_std)$"}}


def test_a_third_party_criteria_dict_survives_a_refresh(restore_cf_xarray_options):
    cf_xarray.set_options(custom_criteria=[*_options(), _THIRD_PARTY])
    vocabulary._refresh()
    options = _options()
    assert _THIRD_PARTY in options
    assert options.count(vocabulary._REGISTERED) == 1
    assert options[0] == vocabulary._REGISTERED  # ours first: its keys win a clash

    ds = xr.Dataset({"zzz": ("x", np.ones(2)), "NO3": ("x", np.ones(2))})
    assert ds.cf["third_thing"].name == "zzz"  # the other package still resolves
    assert ds.cf["nitrate"].name == "NO3"  # and so do we


def test_refreshing_again_does_not_duplicate_ours(restore_cf_xarray_options):
    cf_xarray.set_options(custom_criteria=[*_options(), _THIRD_PARTY])
    vocabulary._refresh()
    before = _options()
    vocabulary._refresh()
    vocabulary._refresh()
    assert _options() == before
    assert len(_options()) == len(before)


def test_registering_onto_nothing_but_a_third_party_dict(restore_cf_xarray_options):
    """``set_options(custom_criteria=<one dict>)`` leaves a one-item list to extend."""
    cf_xarray.set_options(custom_criteria=_THIRD_PARTY)
    vocabulary._refresh()
    assert _options() == [vocabulary._REGISTERED, _THIRD_PARTY]


def test_a_live_register_keeps_the_third_party_dict(
    restore_cf_xarray_options,
    pristine_vocabulary,  # noqa: F811  (fixture)
):
    cf_xarray.set_options(custom_criteria=[*_options(), _THIRD_PARTY])
    vocabulary.register("my_conc", "standard_x", aliases=["MC"])
    options = _options()
    assert _THIRD_PARTY in options
    assert options.count(vocabulary._REGISTERED) == 1
    assert "my_conc" in vocabulary._REGISTERED
    assert "MC" in vocabulary._REGISTERED
    assert all(options.count(c) == 1 for c in options)  # nothing registered twice
