"""Tests for :mod:`ocean_skill.depth_convention` -- how a source says where its zero is.

A source's vertical values become ``d``: metres, positive down, below an origin that is
either the moving free surface or a fixed position. These tests pin the vocabulary
(:func:`canonicalize`, :func:`merge_probed`), the field-by-field precedence that turns
declared / inferred / data-derived / default into one convention (:func:`resolve`), the
clues a coordinate gives about itself (:func:`infer_from_coordinate`), and the sign and
unit conversion that replaces every ``abs()`` on a vertical coordinate
(:func:`positive_down_values`). The module is pure metadata: nothing here needs a
dataset, and a guard test keeps it importing nothing from the rest of the package.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pydantic
import pytest

from ocean_skill import depth_convention as dc

# -- helpers ------------------------------------------------------------------------


def _error(call, *args, **kwargs) -> ValueError:
    """Return the ``ValueError`` ``call`` raises, which must be the module's own."""
    with pytest.raises(ValueError) as info:
        call(*args, **kwargs)
    assert type(info.value) is ValueError, "a raw pydantic error leaked out"
    assert not isinstance(info.value, pydantic.ValidationError)
    assert str(info.value).startswith("depth_convention: ")
    return info.value


# -- the vocabulary -----------------------------------------------------------------


def test_the_vocabulary_constants():
    assert dc.ORIGINS == ("surface", "fixed")
    assert isinstance(
        dc.ORIGINS, tuple
    )  # so a "seafloor" origin is a one-line addition
    assert dc.POSITIVES == ("up", "down")
    assert dc.UNITS == ("m", "dbar")
    assert dc.SUPPORTS == ("point", "surface", "bottom")
    assert dc.FIELDS == ("origin", "positive", "units", "datum_z_m", "support")
    assert dc.M_PER_DBAR == 1.0
    assert dc.NORMALIZED_ATTR == "depth_normalized"
    assert dc.SURFACE_DEFAULT_FEATURE_TYPES == frozenset(
        {"profile", "trajectoryProfile"}
    )


@pytest.mark.parametrize(
    "field, values",
    [
        ("origin", dc.ORIGINS),
        ("positive", dc.POSITIVES),
        ("units", dc.UNITS),
        ("support", dc.SUPPORTS),
    ],
)
def test_every_canonical_value_is_accepted_as_itself(field, values):
    # Guards the spellings table against a value being added to a tuple without one.
    for value in values:
        assert dc.canonicalize({field: value}) == {field: value}


def test_the_module_imports_nothing_from_the_rest_of_the_package():
    """It must move into a catalog package unchanged: stdlib, numpy, pydantic only."""
    tree = ast.parse(Path(dc.__file__).read_text())
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "relative import"
            roots.add(node.module.split(".")[0])
    assert {r for r in roots if r not in sys.stdlib_module_names} == {
        "numpy",
        "pydantic",
    }


# -- canonicalize -------------------------------------------------------------------


@pytest.mark.parametrize(
    "shorthand, origin",
    [
        ("surface", "surface"),
        ("fixed", "fixed"),
        ("Surface", "surface"),
        ("FIXED", "fixed"),
        ("free_surface", "surface"),
        ("Free-Surface", "surface"),
        ("sea surface", "surface"),
        ("fixed_in_space", "fixed"),
        ("  fixed  ", "fixed"),
    ],
)
def test_a_bare_string_is_shorthand_for_the_origin(shorthand, origin):
    assert dc.canonicalize(shorthand) == {"origin": origin}


@pytest.mark.parametrize(
    "field, raw, canonical",
    [
        ("origin", "Sea_Surface", "surface"),
        ("origin", "Fixed_In_Space", "fixed"),
        ("positive", "UP", "up"),
        ("positive", "Upward", "up"),
        ("positive", "downward", "down"),
        ("positive", "Down", "down"),
        ("units", "M", "m"),
        ("units", "meter", "m"),
        ("units", "Meters", "m"),
        ("units", "metre", "m"),
        ("units", "METRES", "m"),
        ("units", "DBAR", "dbar"),
        ("units", "decibar", "dbar"),
        ("units", "Decibars", "dbar"),
        ("units", "db", "dbar"),
        ("support", "Point", "point"),
        ("support", "BOTTOM", "bottom"),
        ("support", "surface", "surface"),
    ],
)
def test_values_are_normalised_case_insensitively_through_their_aliases(
    field, raw, canonical
):
    assert dc.canonicalize({field: raw}) == {field: canonical}


def test_a_full_declaration_comes_back_in_fields_order_as_plain_json():
    out = dc.canonicalize(
        {
            "support": "bottom",
            "datum_z_m": 2,
            "units": "Metres",
            "positive": "Upward",
            "origin": "FIXED",
        }
    )
    assert out == {
        "origin": "fixed",
        "positive": "up",
        "units": "m",
        "datum_z_m": 2.0,
        "support": "bottom",
    }
    assert list(out) == list(dc.FIELDS)
    assert isinstance(out["datum_z_m"], float)
    assert json.loads(json.dumps(out)) == out


@pytest.mark.parametrize("empty", [None, {}, "", "   ", {"variables": {}}])
def test_nothing_declared_is_none(empty):
    assert dc.canonicalize(empty) is None


def test_unset_values_are_dropped_not_kept_as_none():
    assert dc.canonicalize({"origin": "fixed", "positive": None, "support": None}) == {
        "origin": "fixed"
    }


def test_an_unknown_key_is_rejected_with_the_allowed_keys_and_a_near_miss():
    err = _error(dc.canonicalize, {"orgin": "surface"})
    message = str(err)
    assert "'orgin'" in message
    assert "did you mean 'origin'" in message
    assert "allowed keys" in message
    for key in (*dc.FIELDS, "variables", "inferred"):
        assert key in message


def test_an_unknown_key_deep_in_a_variable_entry_says_where():
    err = _error(dc.canonicalize, {"variables": {"temp": {"posiitve": "up"}}})
    assert "variables['temp']" in str(err)
    assert "'posiitve'" in str(err)


def test_reason_is_only_allowed_in_the_inferred_block():
    assert dc.canonicalize(None, inferred={"origin": "surface", "reason": "why"})
    err = _error(dc.canonicalize, {"reason": "why"})
    assert "'reason'" in str(err)


@pytest.mark.parametrize(
    "field, bad",
    [
        ("origin", "seafloor"),
        ("positive", "sideways"),
        ("units", "feet"),
        ("support", "middle"),
        ("origin", 3),
        ("units", ["m"]),
    ],
)
def test_a_bad_enum_value_names_the_key_the_value_and_what_is_allowed(field, bad):
    err = _error(dc.canonicalize, {field: bad})
    message = str(err)
    assert field in message
    assert repr(bad) in message
    allowed = {
        "origin": dc.ORIGINS,
        "positive": dc.POSITIVES,
        "units": dc.UNITS,
        "support": dc.SUPPORTS,
    }[field]
    for value in allowed:
        assert repr(value) in message


def test_a_near_miss_value_gets_a_suggestion():
    assert "did you mean 'surface'" in str(
        _error(dc.canonicalize, {"origin": "surfce"})
    )
    assert "did you mean 'down'" in str(_error(dc.canonicalize, {"positive": "dwon"}))


def test_a_bad_shorthand_is_an_error_not_silently_ignored():
    _error(dc.canonicalize, "seafloor")


@pytest.mark.parametrize("bad", [5, ["surface"], 1.5])
def test_a_declaration_that_is_neither_mapping_nor_string_is_an_error(bad):
    _error(dc.canonicalize, bad)


@pytest.mark.parametrize(
    "bad", [float("nan"), float("inf"), -float("inf"), "abc", True, [1.0], {"a": 1}]
)
def test_datum_z_m_must_be_a_finite_number(bad):
    err = _error(dc.canonicalize, {"origin": "fixed", "datum_z_m": bad})
    assert "datum_z_m" in str(err)
    assert "finite number" in str(err)


@pytest.mark.parametrize("good, expected", [(2, 2.0), (-0.5, -0.5), ("1.25", 1.25)])
def test_datum_z_m_accepts_numbers_and_numeric_strings(good, expected):
    out = dc.canonicalize({"origin": "fixed", "datum_z_m": good})
    assert out == {"origin": "fixed", "datum_z_m": expected}
    assert isinstance(out["datum_z_m"], float)


def test_numpy_numbers_are_numbers():
    out = dc.canonicalize({"origin": "fixed", "datum_z_m": np.float32(0.5)})
    assert out["datum_z_m"] == 0.5
    assert (
        dc.canonicalize({"origin": "fixed", "datum_z_m": np.int64(3)})["datum_z_m"]
        == 3.0
    )


def test_a_datum_on_a_surface_origin_is_an_error():
    err = _error(dc.canonicalize, {"origin": "surface", "datum_z_m": 1.0})
    assert "datum_z_m only applies to origin: fixed" in str(err)
    assert "'surface'" in str(err)


@pytest.mark.parametrize("zero", [0, 0.0, -0.0, "0"])
def test_a_zero_datum_is_harmless_on_any_origin(zero):
    """It says nothing a surface origin does not, so a resolved key round-trips."""
    assert dc.canonicalize({"origin": "surface", "datum_z_m": zero}) == {
        "origin": "surface",
        "datum_z_m": 0.0,
    }
    assert dc.canonicalize({"datum_z_m": zero}) == {"datum_z_m": 0.0}
    assert dc.canonicalize(
        {"origin": "surface", "variables": {"t": {"datum_z_m": zero}}}
    )
    # ...and any non-zero datum on those origins is still an error
    _error(dc.canonicalize, {"origin": "surface", "datum_z_m": 1e-9})


def test_a_resolved_key_can_be_canonicalised_again():
    for meta in ({"depth_convention": "surface"}, {"featureType": "profile"}, {}):
        key = dc.resolve(meta).key()
        assert dc.canonicalize(key)
        assert dc.resolve({"depth_convention": key}).key() == key


def test_a_datum_on_a_fixed_origin_is_fine():
    assert dc.canonicalize({"origin": "fixed", "datum_z_m": 1.5}) == {
        "origin": "fixed",
        "datum_z_m": 1.5,
    }


def test_a_datum_with_no_origin_anywhere_is_an_error_that_says_to_declare_one():
    err = _error(dc.canonicalize, {"datum_z_m": 1.0})
    assert "datum_z_m only applies to origin: fixed" in str(err)
    assert "not set" in str(err)


def test_a_datum_takes_its_origin_from_the_inferred_tier_when_none_is_declared():
    assert dc.canonicalize({"datum_z_m": 1.0}, inferred={"origin": "fixed"}) == {
        "datum_z_m": 1.0,
        "inferred": {"origin": "fixed"},
    }
    _error(dc.canonicalize, {"datum_z_m": 1.0}, inferred={"origin": "surface"})


def test_a_declared_origin_outranks_the_inferred_one_for_the_datum_check():
    # declared fixed + inferred surface: the datum is fine, because declared wins.
    assert dc.canonicalize(
        {"origin": "fixed", "datum_z_m": 1.0}, inferred={"origin": "surface"}
    ) == {"origin": "fixed", "datum_z_m": 1.0, "inferred": {"origin": "surface"}}
    # ...and declared surface + inferred fixed is still an error.
    _error(
        dc.canonicalize,
        {"origin": "surface", "datum_z_m": 1.0},
        inferred={"origin": "fixed"},
    )


def test_per_variable_entries_accept_shorthand_and_mappings_and_drop_empty_ones():
    out = dc.canonicalize(
        {
            "origin": "fixed",
            "variables": {
                "temp": "surface",
                "sal": {"units": "DBAR", "positive": "Up"},
                "empty": {},
                "none": None,
                "blank": "",
            },
        }
    )
    assert out == {
        "origin": "fixed",
        "variables": {
            "temp": {"origin": "surface"},
            "sal": {"positive": "up", "units": "dbar"},
        },
    }


def test_variable_names_are_kept_exactly_as_written():
    out = dc.canonicalize({"variables": {"Temp": "fixed", "SAL": "surface"}})
    assert list(out["variables"]) == ["Temp", "SAL"]


def test_a_variable_entry_with_a_bad_value_says_which_variable_and_field():
    err = _error(dc.canonicalize, {"variables": {"temp": {"origin": "nope"}}})
    assert "variables['temp'].origin" in str(err)
    assert "'nope'" in str(err)


def test_variables_must_be_a_mapping():
    err = _error(dc.canonicalize, {"variables": ["temp"]})
    assert "variables" in str(err)


def test_a_variable_datum_is_checked_against_the_origin_it_inherits():
    # inherits fixed from the top level: fine
    assert dc.canonicalize(
        {"origin": "fixed", "variables": {"temp": {"datum_z_m": 2.0}}}
    ) == {"origin": "fixed", "variables": {"temp": {"datum_z_m": 2.0}}}
    # inherits surface from the top level: an error that names the variable
    err = _error(
        dc.canonicalize,
        {"origin": "surface", "variables": {"temp": {"datum_z_m": 2.0}}},
    )
    assert "variables['temp']" in str(err)
    assert "datum_z_m only applies to origin: fixed" in str(err)
    # its own origin overrides what it would inherit, either way
    assert dc.canonicalize(
        {
            "origin": "surface",
            "variables": {"temp": {"origin": "fixed", "datum_z_m": 2.0}},
        }
    )
    _error(
        dc.canonicalize,
        {
            "origin": "fixed",
            "variables": {"temp": {"origin": "surface", "datum_z_m": 2.0}},
        },
    )
    # nothing to inherit from at all: an error
    _error(dc.canonicalize, {"variables": {"temp": {"datum_z_m": 2.0}}})
    # ...unless the probe inferred one
    assert dc.canonicalize(
        {"variables": {"temp": {"datum_z_m": 2.0}}}, inferred={"origin": "fixed"}
    )


def test_the_inferred_block_keeps_its_reason_and_is_listed_last():
    out = dc.canonicalize(
        {"origin": "fixed", "variables": {"t": "surface"}},
        inferred={
            "units": "dbar",
            "origin": "surface",
            "reason": "  Z column 'PRES'  ",
        },
    )
    assert out == {
        "origin": "fixed",
        "variables": {"t": {"origin": "surface"}},
        "inferred": {"origin": "surface", "units": "dbar", "reason": "Z column 'PRES'"},
    }
    assert list(out) == ["origin", "variables", "inferred"]
    assert list(out["inferred"]) == ["origin", "units", "reason"]


def test_an_inferred_block_may_be_just_a_reason_or_just_fields():
    assert dc.canonicalize(None, inferred={"reason": "because"}) == {
        "inferred": {"reason": "because"}
    }
    assert dc.canonicalize(None, inferred={"units": "decibars"}) == {
        "inferred": {"units": "dbar"}
    }
    assert dc.canonicalize(None, inferred={"reason": "   "}) is None
    assert dc.canonicalize(None, inferred={}) is None


def test_inferred_values_are_validated_like_declared_ones():
    err = _error(dc.canonicalize, None, inferred={"origin": "nope"})
    assert "inferred.origin" in str(err)
    _error(dc.canonicalize, None, inferred={"reason": 3})
    _error(dc.canonicalize, None, inferred="surface")  # a mapping is required here
    _error(dc.canonicalize, None, inferred={"origin": "surface", "datum_z_m": 1.0})


def test_the_inferred_argument_wins_over_an_inferred_block_in_declared():
    out = dc.canonicalize(
        {"origin": "fixed", "inferred": {"origin": "fixed", "reason": "old"}},
        inferred={"origin": "surface", "reason": "new"},
    )
    assert out["inferred"] == {"origin": "surface", "reason": "new"}
    # an empty inferred= is still "given", and replaces
    assert dc.canonicalize({"inferred": {"origin": "fixed"}}, inferred={}) is None


@pytest.mark.parametrize(
    "declared",
    [
        "surface",
        {"origin": "fixed", "datum_z_m": 1.0, "support": "bottom"},
        {"positive": "up", "units": "dbar"},
        {
            "origin": "fixed",
            "variables": {"a": "surface", "b": {"units": "dbar", "positive": "up"}},
            "inferred": {
                "origin": "surface",
                "units": "dbar",
                "reason": "Z column 'P'",
            },
        },
        {"inferred": {"units": "m"}},
    ],
)
def test_a_canonical_dict_round_trips_unchanged(declared):
    once = dc.canonicalize(declared)
    assert dc.canonicalize(once) == once
    assert dc.canonicalize(json.loads(json.dumps(once))) == once


def test_canonicalize_does_not_mutate_its_input():
    declared = {
        "origin": "Fixed",
        "variables": {"t": "Surface"},
        "inferred": {"units": "M"},
    }
    snapshot = json.loads(json.dumps(declared))
    dc.canonicalize(declared, inferred={"units": "dbar"})
    assert declared == snapshot


def test_errors_are_the_modules_valueerror_never_a_pydantic_one():
    for bad in (
        {"origin": "x"},
        {"orgin": "fixed"},
        {"datum_z_m": "x", "origin": "fixed"},
        {"variables": 3},
        {"variables": {"a": {"origin": "x"}}},
        {"origin": "surface", "datum_z_m": 1},
        5,
    ):
        err = _error(dc.canonicalize, bad)
        assert err.__cause__ is None


def test_every_problem_in_one_declaration_is_reported():
    err = _error(dc.canonicalize, {"origin": "nope", "units": "feet"})
    assert "origin" in str(err) and "units" in str(err)


# -- merge_probed -------------------------------------------------------------------


def test_the_probe_alone_lands_under_inferred():
    probed = {
        "inferred": {"origin": "surface", "units": "dbar", "reason": "Z column 'PRES'"}
    }
    assert dc.merge_probed(probed, None) == probed


def test_declared_values_stay_at_the_top_level_and_the_probes_are_kept_beside_them():
    probed = {
        "inferred": {"origin": "surface", "units": "dbar", "reason": "Z column 'PRES'"}
    }
    out = dc.merge_probed(probed, {"origin": "fixed", "positive": "up"})
    assert out == {"origin": "fixed", "positive": "up", "inferred": probed["inferred"]}


def test_a_declared_origin_wins_over_the_probes_at_resolve_time():
    probed = {"inferred": {"origin": "surface", "positive": "up"}}
    merged = dc.merge_probed(probed, "fixed")
    conv = dc.resolve({"depth_convention": merged})
    assert (conv.origin, conv.provenance["origin"]) == ("fixed", "declared")
    # a field the author left alone falls to the probe
    assert (conv.positive, conv.provenance["positive"]) == ("up", "inferred")


def test_the_probe_replaces_a_stale_inferred_block_from_an_earlier_build():
    stale = {"origin": "fixed", "inferred": {"origin": "fixed", "reason": "old probe"}}
    probed = {"inferred": {"origin": "surface", "reason": "new probe"}}
    out = dc.merge_probed(probed, stale)
    assert out == {
        "origin": "fixed",
        "inferred": {"origin": "surface", "reason": "new probe"},
    }


def test_without_a_new_probe_the_declared_entrys_own_inferred_block_survives():
    declared = {"origin": "fixed", "inferred": {"units": "dbar", "reason": "kept"}}
    assert dc.merge_probed(None, declared) == declared


def test_a_shorthand_declaration_merges_like_a_mapping():
    probed = {"inferred": {"units": "dbar"}}
    assert dc.merge_probed(probed, "surface") == {
        "origin": "surface",
        "inferred": {"units": "dbar"},
    }


def test_nothing_probed_and_nothing_declared_is_none():
    assert dc.merge_probed(None, None) is None
    assert dc.merge_probed({}, {}) is None
    assert dc.merge_probed({"inferred": {}}, None) is None


def test_the_merge_does_not_mutate_what_it_was_given():
    declared = {"origin": "fixed", "inferred": {"origin": "surface"}}
    probed = {"inferred": {"units": "dbar"}}
    dc.merge_probed(probed, declared)
    assert declared == {"origin": "fixed", "inferred": {"origin": "surface"}}
    assert probed == {"inferred": {"units": "dbar"}}


def test_an_invalid_declaration_or_probe_is_an_error_not_swallowed():
    _error(dc.merge_probed, {"inferred": {"origin": "surface"}}, {"origin": "nope"})
    _error(dc.merge_probed, {"inferred": {"origin": "nope"}}, None)
    # declared datum + probed surface origin: the declared fixed origin is required
    _error(dc.merge_probed, {"inferred": {"origin": "surface"}}, {"datum_z_m": 1.0})


# -- resolve ------------------------------------------------------------------------

#: Per field: distinct values for the variable, declared, inferred and data tiers, then
#: the default.
_TIERS = {
    "origin": ("fixed", "surface", "fixed", "surface", "fixed"),
    "positive": ("up", "down", "up", "down", "down"),
    "units": ("dbar", "m", "dbar", "m", "m"),
    "datum_z_m": (3.0, 2.0, 1.0, 0.5, 0.0),
    "support": ("bottom", "surface", "bottom", "surface", "point"),
}


def _tiered(field):
    """Return ``(meta, hint)`` setting only ``field``, in every tier."""
    variable, declared, inferred, data, _ = _TIERS[field]
    top = {field: declared}
    if field == "datum_z_m":
        top["origin"] = "fixed"  # a datum needs a fixed origin to hang on
    meta = {
        "depth_convention": {
            **top,
            "variables": {"temp": {field: variable}},
            "inferred": {field: inferred},
        }
    }
    return meta, {field: data}


@pytest.mark.parametrize("field", dc.FIELDS)
def test_precedence_is_variable_then_declared_then_inferred_then_data_then_default(
    field,
):
    meta, hint = _tiered(field)
    variable, declared, inferred, data, default = _TIERS[field]
    entry = meta["depth_convention"]

    def got(meta, variable_name, hint):
        conv = dc.resolve(meta, variable_name, hint=hint)
        return getattr(conv, field), conv.provenance[field]

    assert got(meta, "temp", hint) == (variable, "declared")
    # no per-variable step when no variable is named (or it has no entry of its own)
    assert got(meta, None, hint) == (declared, "declared")
    assert got(meta, "other", hint) == (declared, "declared")
    # drop the declared top-level field, then the inferred one, then the data hint
    del entry[field]
    assert got(meta, None, hint) == (inferred, "inferred")
    del entry["inferred"]
    assert got(meta, None, hint) == (data, "data")
    assert got(meta, None, None) == (default, "default")


def test_the_variable_is_matched_exactly_then_case_insensitively():
    meta = {
        "depth_convention": {
            "origin": "fixed",
            "variables": {
                "Temp": "surface",
                "temp": {"origin": "fixed", "units": "dbar"},
            },
        }
    }
    assert dc.resolve(meta, "temp").units == "dbar"  # the exact key wins over "Temp"
    assert dc.resolve(meta, "Temp").origin == "surface"
    only_one = {
        "depth_convention": {"origin": "fixed", "variables": {"Temp": "surface"}}
    }
    conv = dc.resolve(only_one, "TEMP")
    assert (conv.origin, conv.provenance["origin"]) == ("surface", "declared")
    assert dc.resolve(only_one, "salt").origin == "fixed"
    assert dc.resolve(only_one, None).origin == "fixed"


def test_each_field_comes_from_the_most_specific_tier_that_sets_it():
    meta = {
        "depth_convention": {
            "origin": "fixed",
            "positive": "up",
            "variables": {"temp": {"units": "dbar"}},
            "inferred": {"support": "bottom", "positive": "down"},
        }
    }
    conv = dc.resolve(meta, "temp", hint={"units": "m", "support": "surface"})
    assert conv.key() == {
        "origin": "fixed",
        "positive": "up",
        "units": "dbar",
        "datum_z_m": 0.0,
        "support": "bottom",
    }
    assert dict(conv.provenance) == {
        "origin": "declared",
        "positive": "declared",
        "units": "declared",
        "datum_z_m": "default",
        "support": "inferred",
    }


def test_no_meta_gives_all_defaults_with_a_fixed_origin():
    for meta in (None, {}, {"depth_convention": None}, {"depth_convention": {}}):
        conv = dc.resolve(meta)
        assert conv.key() == {
            "origin": "fixed",
            "positive": "down",
            "units": "m",
            "datum_z_m": 0.0,
            "support": "point",
        }
        assert set(conv.provenance.values()) == {"default"}
        assert not conv.declared


@pytest.mark.parametrize(
    "feature_type, origin",
    [
        ("profile", "surface"),
        ("trajectoryProfile", "surface"),
        ("trajectoryprofile", "surface"),
        ("TrajectoryProfile", "surface"),
        ("PROFILE", "surface"),
        ("timeSeries", "fixed"),
        ("timeSeriesProfile", "fixed"),
        ("trajectory", "fixed"),
        ("point", "fixed"),
        (None, "fixed"),
        (7, "fixed"),
    ],
)
def test_the_default_origin_follows_the_feature_type(feature_type, origin):
    conv = dc.resolve({"featureType": feature_type})
    assert conv.origin == origin
    assert conv.provenance["origin"] == "default"


def test_dbar_units_default_the_origin_to_the_surface_whatever_declared_them():
    for meta, hint in (
        ({"depth_convention": {"units": "dbar"}}, None),
        ({"depth_convention": {"inferred": {"units": "dbar"}}}, None),
        ({"featureType": "timeSeries"}, {"units": "dbar"}),
        ({"depth_convention": {"variables": {"p": {"units": "dbar"}}}}, None),
    ):
        conv = dc.resolve(meta, "p", hint=hint)
        assert (conv.origin, conv.provenance["origin"]) == ("surface", "default"), meta


def test_a_declared_origin_beats_the_dbar_and_feature_type_defaults():
    meta = {
        "featureType": "profile",
        "depth_convention": {"origin": "fixed", "units": "dbar"},
    }
    conv = dc.resolve(meta)
    assert (conv.origin, conv.provenance["origin"]) == ("fixed", "declared")
    assert conv.declared


def test_a_surface_origin_forces_the_datum_to_zero():
    # surface by feature type, no datum anywhere
    assert dc.resolve({"featureType": "profile"}).datum_z_m == 0.0
    # a variable that overrides a fixed entry to surface does not inherit its datum
    meta = {
        "depth_convention": {
            "origin": "fixed",
            "datum_z_m": 2.5,
            "variables": {"pressure": {"origin": "surface"}},
        }
    }
    surface = dc.resolve(meta, "pressure")
    assert (surface.origin, surface.datum_z_m) == ("surface", 0.0)
    assert surface.provenance["datum_z_m"] == "default"
    fixed = dc.resolve(meta, "temp")
    assert (fixed.origin, fixed.datum_z_m) == ("fixed", 2.5)
    assert fixed.provenance["datum_z_m"] == "declared"
    # a datum arriving from the data hint is dropped the same way
    from_hint = dc.resolve({"featureType": "profile"}, hint={"datum_z_m": 4.0})
    assert from_hint.datum_z_m == 0.0


def test_a_declared_datum_is_used_on_a_fixed_origin():
    conv = dc.resolve({"depth_convention": {"origin": "fixed", "datum_z_m": -1.5}})
    assert conv.datum_z_m == -1.5
    assert conv.provenance["datum_z_m"] == "declared"


@pytest.mark.parametrize(
    "declared, origin",
    [
        ("surface", "surface"),
        ("Free_Surface", "surface"),
        ("FIXED", "fixed"),
        ({"origin": "Sea_Surface", "units": "Decibars", "positive": "UP"}, "surface"),
        ({"origin": "surface"}, "surface"),  # already canonical
    ],
)
def test_the_declaration_may_be_raw_or_canonical_and_is_always_validated(
    declared, origin
):
    conv = dc.resolve({"depth_convention": declared})
    assert conv.origin == origin
    assert conv.provenance["origin"] == "declared"
    canonical = dc.canonicalize(declared)
    assert dc.resolve({"depth_convention": canonical}) == conv


def test_an_invalid_declaration_raises_at_resolve_time():
    err = _error(dc.resolve, {"depth_convention": {"origin": "seafloor"}})
    assert "'seafloor'" in str(err)
    _error(dc.resolve, {"depth_convention": {"origin": "surface", "datum_z_m": 1.0}})
    _error(dc.resolve, {"depth_convention": 5})


def test_a_hint_is_validated_and_may_carry_a_reason():
    conv = dc.resolve(
        {}, hint={"origin": "Surface", "units": "decibar", "reason": "Z col"}
    )
    assert (conv.origin, conv.units) == ("surface", "dbar")
    assert conv.provenance["origin"] == conv.provenance["units"] == "data"
    _error(dc.resolve, {}, hint={"origin": "nope"})
    _error(dc.resolve, {}, hint={"bogus": 1})
    assert dc.resolve({}, hint={}).provenance["origin"] == "default"
    assert dc.resolve({}, hint=None).provenance["origin"] == "default"


def test_the_hint_never_outranks_a_declaration_or_the_probe():
    meta = {"depth_convention": {"origin": "fixed", "inferred": {"positive": "up"}}}
    conv = dc.resolve(
        meta, hint={"origin": "surface", "positive": "down", "units": "dbar"}
    )
    assert (conv.origin, conv.provenance["origin"]) == ("fixed", "declared")
    assert (conv.positive, conv.provenance["positive"]) == ("up", "inferred")
    assert (conv.units, conv.provenance["units"]) == ("dbar", "data")


def test_resolve_takes_a_pressure_hint_from_infer_from_coordinate():
    conv = dc.resolve({}, hint=dc.infer_from_coordinate("PRES"))
    assert (conv.origin, conv.units) == ("surface", "dbar")
    assert conv.provenance["origin"] == "data"


def test_key_and_frame_are_json_serialisable_with_the_documented_shapes():
    meta = {
        "depth_convention": {
            "origin": "fixed",
            "datum_z_m": 1,
            "positive": "up",
            "support": "bottom",
        }
    }
    conv = dc.resolve(meta)
    assert conv.key() == {
        "origin": "fixed",
        "positive": "up",
        "units": "m",
        "datum_z_m": 1.0,
        "support": "bottom",
    }
    assert conv.frame() == {
        "origin": "fixed",
        "datum_z_m": 1.0,
        "support": "bottom",
        "source": "declared",
    }
    for shape in (conv.key(), conv.frame()):
        assert json.loads(json.dumps(shape)) == shape
    assert isinstance(conv.key()["datum_z_m"], float)


@pytest.mark.parametrize(
    "meta, hint, source, declared",
    [
        ({"depth_convention": "surface"}, None, "declared", True),
        ({"depth_convention": {"variables": {"t": "surface"}}}, None, "declared", True),
        (
            {"depth_convention": {"inferred": {"origin": "surface"}}},
            None,
            "inferred",
            False,
        ),
        ({}, {"origin": "surface"}, "data", False),
        ({"featureType": "profile"}, None, "default", False),
    ],
)
def test_frame_source_and_declared_report_where_the_origin_came_from(
    meta, hint, source, declared
):
    conv = dc.resolve(meta, "t", hint=hint)
    assert conv.frame()["source"] == source
    assert conv.provenance["origin"] == source
    assert conv.declared is declared


def test_a_resolved_convention_is_frozen_hashable_and_comparable():
    conv = dc.resolve({"depth_convention": "surface"})
    with pytest.raises(dataclasses.FrozenInstanceError):
        conv.origin = "fixed"
    with pytest.raises(TypeError):
        conv.provenance["origin"] = "default"
    assert dc.resolve({"depth_convention": "surface"}) == conv
    assert hash(dc.resolve({"depth_convention": "surface"})) == hash(conv)
    assert dc.resolve({"depth_convention": "fixed"}) != conv
    assert len({conv, dc.resolve({"depth_convention": "surface"})}) == 1


def test_a_resolved_convention_survives_pickle_and_deepcopy():
    """A comparison carrying one may be pickled by dask or deep-copied by a user."""
    conv = dc.resolve(
        {"depth_convention": {"origin": "fixed", "datum_z_m": 1.5}},
        hint={"units": "dbar"},
    )
    for clone in (
        pickle.loads(pickle.dumps(conv)),
        copy.deepcopy(conv),
        copy.copy(conv),
    ):
        assert clone == conv
        assert hash(clone) == hash(conv)
        assert dict(clone.provenance) == dict(conv.provenance)
        with pytest.raises(TypeError):
            clone.provenance["origin"] = "default"


def test_provenance_reads_like_a_dict_but_cannot_be_edited():
    provenance = dc.resolve({"depth_convention": "surface"}).provenance
    assert provenance["origin"] == "declared"
    assert list(provenance) == list(dc.FIELDS)
    assert len(provenance) == len(dc.FIELDS)
    assert provenance == dict(provenance)
    assert dict(provenance)["support"] == "default"
    with pytest.raises(KeyError):
        provenance["nope"]
    with pytest.raises(TypeError):
        provenance["origin"] = "x"
    with pytest.raises(TypeError):
        del provenance["origin"]
    assert "declared" in repr(provenance)


def test_a_hand_built_convention_gets_a_float_datum_and_complete_provenance():
    conv = dc.ResolvedConvention(
        origin="fixed",
        positive="down",
        units="m",
        datum_z_m=0,
        support="point",
        provenance={"origin": "declared"},
    )
    assert isinstance(conv.datum_z_m, float)
    assert dict(conv.provenance) == {
        "origin": "declared",
        "positive": "default",
        "units": "default",
        "datum_z_m": "default",
        "support": "default",
    }
    assert conv.declared
    assert conv.frame()["source"] == "declared"
    assert json.dumps(conv.key()) == json.dumps({**conv.key(), "datum_z_m": 0.0})


# -- infer_from_coordinate ----------------------------------------------------------


@pytest.mark.parametrize("value", ["up", "UP", "Up", "upward"])
def test_the_positive_attribute_is_read(value):
    assert dc.infer_from_coordinate("depth", {"positive": value}) == {"positive": "up"}


def test_positive_down_attribute():
    assert dc.infer_from_coordinate(None, {"positive": "down"}) == {"positive": "down"}


@pytest.mark.parametrize(
    "units", ["dbar", "DBAR", "decibar", "decibars", "db", " dbar "]
)
def test_decibar_units_mean_a_pressure_below_the_surface(units):
    assert dc.infer_from_coordinate("z", {"units": units}) == {
        "origin": "surface",
        "units": "dbar",
    }


@pytest.mark.parametrize("units", ["m", "M", "meter", "metres", "Meters"])
def test_metre_units_say_nothing_about_the_origin(units):
    assert dc.infer_from_coordinate("z", {"units": units}) == {"units": "m"}


@pytest.mark.parametrize("units", ["cm", "Pa", "feet", "", None, 3])
def test_other_or_missing_units_infer_nothing(units):
    assert dc.infer_from_coordinate("z", {"units": units}) == {}


@pytest.mark.parametrize(
    "standard_name, expected",
    [
        ("sea_water_pressure", {"origin": "surface", "units": "dbar"}),
        ("sea_water_pressure_due_to_sea_water", {"origin": "surface", "units": "dbar"}),
        ("depth", {"positive": "down"}),
        ("height", {"positive": "up"}),
        ("altitude", {"positive": "up"}),
        ("height_above_mean_sea_level", {"origin": "fixed", "positive": "up"}),
        ("height_above_geoid", {"origin": "fixed", "positive": "up"}),
        ("height_above_reference_ellipsoid", {"origin": "fixed", "positive": "up"}),
        ("depth_below_geoid", {"origin": "fixed", "positive": "down"}),
        ("Sea_Water_Pressure", {"origin": "surface", "units": "dbar"}),
        ("sea_water_temperature", {}),
    ],
)
def test_standard_names_that_settle_the_convention(standard_name, expected):
    assert dc.infer_from_coordinate("z", {"standard_name": standard_name}) == expected


@pytest.mark.parametrize(
    "name",
    [
        "PRES",
        "pres",
        "Pressure",
        "pressure",
        "sea_water_pressure",
        "PRDM",
        "prdm",
        "prs",
    ],
)
def test_pressure_like_names_are_a_pressure_below_the_surface(name):
    assert dc.infer_from_coordinate(name) == {"origin": "surface", "units": "dbar"}


@pytest.mark.parametrize(
    "name",
    ["pres (dbar)", "PRES [dbar]", "Pressure(db)", "pres (decibars)", "  Pres  "],
)
def test_a_units_suffix_is_stripped_before_matching_the_name(name):
    assert dc.infer_from_coordinate(name) == {"origin": "surface", "units": "dbar"}


@pytest.mark.parametrize(
    "name", ["depth", "z", "lev", "pressure_flag", "press", "bottom_pressure"]
)
def test_names_that_are_not_pressure_infer_nothing(name):
    assert dc.infer_from_coordinate(name) == {}


def test_units_in_the_name_suffix_are_read_when_there_is_no_units_attribute():
    assert dc.infer_from_coordinate("depth (m)") == {"units": "m"}
    assert dc.infer_from_coordinate("Depth[m]") == {"units": "m"}
    assert dc.infer_from_coordinate("z (dbar)") == {
        "origin": "surface",
        "units": "dbar",
    }
    # ...and the attribute wins over the suffix
    assert dc.infer_from_coordinate("depth (m)", {"units": "dbar"}) == {
        "origin": "surface",
        "units": "dbar",
    }


def test_a_pressure_name_in_metres_is_below_the_surface_but_not_converted():
    assert dc.infer_from_coordinate("PRES", {"units": "m"}) == {
        "origin": "surface",
        "units": "m",
    }
    assert dc.infer_from_coordinate("pres (m)") == {"origin": "surface", "units": "m"}


def test_explicit_attributes_beat_name_heuristics_and_standard_names():
    # an explicit positive beats the standard name's
    assert dc.infer_from_coordinate(
        "z", {"positive": "down", "standard_name": "height"}
    ) == {"positive": "down"}
    # explicit metres beat the name's and the standard name's dbar, but not their origin
    assert dc.infer_from_coordinate(
        "PRES", {"units": "m", "standard_name": "sea_water_pressure"}
    ) == {"origin": "surface", "units": "m"}
    # explicit units that imply an origin are not overridden by a later clue
    assert dc.infer_from_coordinate(
        "z", {"units": "dbar", "standard_name": "height_above_mean_sea_level"}
    ) == {"origin": "surface", "positive": "up", "units": "dbar"}


def test_every_clue_together():
    attrs = {"positive": "up", "units": "dbar", "standard_name": "sea_water_pressure"}
    assert dc.infer_from_coordinate("PRES", attrs) == {
        "origin": "surface",
        "positive": "up",
        "units": "dbar",
    }


def test_inference_returns_fields_in_order_and_nothing_when_there_is_nothing():
    out = dc.infer_from_coordinate(
        "z", {"standard_name": "height_above_geoid", "units": "m", "positive": "up"}
    )
    assert list(out) == ["origin", "positive", "units"]
    assert dc.infer_from_coordinate() == {}
    assert dc.infer_from_coordinate("depth") == {}
    assert dc.infer_from_coordinate("z", {}) == {}
    assert dc.infer_from_coordinate(None, None) == {}
    assert dc.infer_from_coordinate("z", {"positive": "sideways"}) == {}


def test_an_inferred_result_is_always_accepted_back_by_canonicalize_and_resolve():
    for name, attrs in (
        ("PRES", None),
        ("z", {"standard_name": "height_above_mean_sea_level", "units": "m"}),
        ("depth (m)", {"positive": "down"}),
    ):
        found = dc.infer_from_coordinate(name, attrs)
        assert dc.canonicalize(None, inferred={**found, "reason": "test"})
        assert dc.resolve({}, hint=found)


# -- to_positive_down ---------------------------------------------------------------


def test_down_metres_are_unchanged_and_the_result_is_a_new_float_array():
    values = np.array([1, 2, 3])
    out = dc.to_positive_down(values)
    np.testing.assert_array_equal(out, [1.0, 2.0, 3.0])
    assert out.dtype == float
    out[0] = 99.0
    assert values[0] == 1  # never aliases the input
    floats = np.array([1.0, 2.0])
    assert not np.shares_memory(dc.to_positive_down(floats), floats)


def test_up_values_are_negated_and_nan_is_preserved():
    out = dc.to_positive_down([-1.0, -2.5, np.nan, 0.0], positive="up")
    np.testing.assert_array_equal(out, [1.0, 2.5, np.nan, 0.0])
    assert not np.signbit(out[3])  # a depth of zero stays +0.0, not -0.0


def test_dbar_is_converted_at_m_per_dbar(monkeypatch):
    monkeypatch.setattr(dc, "M_PER_DBAR", 1.02)
    np.testing.assert_allclose(
        dc.to_positive_down([10.0, 20.0], units="dbar"), [10.2, 20.4]
    )
    np.testing.assert_allclose(
        dc.to_positive_down([-10.0], positive="up", units="dbar"), [10.2]
    )
    np.testing.assert_array_equal(dc.to_positive_down([10.0], units="m"), [10.0])


def test_dbar_equals_metres_at_the_documented_approximation():
    values = np.array([1.0, 5.0, 100.0])
    np.testing.assert_array_equal(
        dc.to_positive_down(values, units="dbar"), values * dc.M_PER_DBAR
    )


def test_scalars_and_lists_are_accepted_and_spellings_normalised():
    assert dc.to_positive_down(5.0, positive="UP").item() == -5.0
    np.testing.assert_array_equal(
        dc.to_positive_down([1, 2], positive="Upward", units="Metres"), [-1.0, -2.0]
    )


@pytest.mark.parametrize(
    "kwargs", [{"positive": "sideways"}, {"units": "feet"}, {"positive": None}]
)
def test_unknown_positive_or_units_is_an_error(kwargs):
    _error(dc.to_positive_down, [1.0], **kwargs)


# -- coordinate_positive ------------------------------------------------------------


@pytest.mark.parametrize(
    "attr, expected", [("up", "up"), ("DOWN", "down"), ("Up", "up")]
)
def test_the_cf_positive_attribute_wins_over_the_values(attr, expected):
    assert dc.coordinate_positive({"positive": attr}, [5.0, 10.0]) == expected
    assert dc.coordinate_positive({"positive": attr}, [-5.0, -10.0]) == expected
    assert dc.coordinate_positive({"positive": attr}) == expected


@pytest.mark.parametrize(
    "values, expected",
    [
        ([-1.0, -2.0, -30.0], "up"),
        ([0.0, -1.0, -2.0], "up"),
        ([-1.0, np.nan, -2.0], "up"),
        ([1.0, 2.0, 30.0], "down"),
        ([-1.0, 2.0], "down"),
        ([-1.0, 0.0, 1.0], "down"),
        ([0.0, 0.0], "down"),
        ([5.0], "down"),
        ([np.nan, 5.0], "down"),
    ],
)
def test_without_the_attribute_the_sign_of_the_values_decides(values, expected):
    assert dc.coordinate_positive(None, values) == expected
    assert dc.coordinate_positive({}, np.array(values)) == expected


@pytest.mark.parametrize("values", [None, [], [np.nan, np.nan]])
def test_no_attribute_and_no_finite_values_is_none(values):
    assert dc.coordinate_positive(None, values) is None
    assert dc.coordinate_positive({"units": "m"}, values) is None


def test_an_unrecognised_positive_attribute_falls_back_to_the_values():
    assert dc.coordinate_positive({"positive": "sideways"}, [-1.0, -2.0]) == "up"
    assert dc.coordinate_positive({"positive": None}, [1.0]) == "down"


# -- positive_down_values -----------------------------------------------------------


def _conv(declared=None, **tiers):
    """Resolve ``declared`` (the entry's metadata), passing any hint through."""
    meta = {"depth_convention": declared} if declared is not None else {}
    return dc.resolve(meta, **tiers)


def test_normalized_values_are_returned_untouched_whatever_the_convention_says():
    values = np.array([1.0, 2.0, 3.0])
    attrs = {dc.NORMALIZED_ATTR: 1, "positive": "up", "units": "dbar"}
    conv = _conv({"positive": "up", "units": "dbar"})
    out = dc.positive_down_values(values, attrs, convention=conv)
    np.testing.assert_array_equal(out, values)
    assert out.dtype == float
    assert not np.shares_memory(out, values)
    # the attribute has to be truthy
    flipped = dc.positive_down_values(
        values, {dc.NORMALIZED_ATTR: 0, "positive": "up"}, convention=None
    )
    np.testing.assert_array_equal(flipped, -values)


def test_normalized_integer_values_come_back_as_floats():
    out = dc.positive_down_values([1, 2], {dc.NORMALIZED_ATTR: True})
    assert out.dtype == float
    np.testing.assert_array_equal(out, [1.0, 2.0])


def test_a_declared_positive_up_negates():
    conv = _conv({"positive": "up"})
    np.testing.assert_array_equal(
        dc.positive_down_values([-1.0, -2.0, np.nan], None, convention=conv),
        [1.0, 2.0, np.nan],
    )
    # even when the values would have read as down
    np.testing.assert_array_equal(
        dc.positive_down_values([1.0, 2.0], None, convention=conv), [-1.0, -2.0]
    )


def test_a_declared_positive_down_is_trusted_over_the_values_and_attrs():
    conv = _conv({"positive": "down"})
    np.testing.assert_array_equal(
        dc.positive_down_values([-1.0, -2.0], {"positive": "up"}, convention=conv),
        [-1.0, -2.0],
    )


@pytest.mark.parametrize("tier", ["declared", "inferred", "data"])
def test_positive_and_units_are_trusted_from_every_tier_except_the_default(
    tier, monkeypatch
):
    monkeypatch.setattr(dc, "M_PER_DBAR", 2.0)
    fields = {"positive": "up", "units": "dbar"}
    if tier == "declared":
        conv = _conv(fields)
    elif tier == "inferred":
        conv = _conv({"inferred": fields})
    else:
        conv = _conv(None, hint=fields)
    out = dc.positive_down_values([-1.0, -2.0], {}, convention=conv)
    np.testing.assert_array_equal(out, [2.0, 4.0])


def test_a_defaulted_positive_falls_back_to_the_attrs_then_the_values_then_down():
    conv = _conv({"origin": "fixed"})  # positive/units both defaulted
    assert conv.provenance["positive"] == "default"
    # the CF attribute
    np.testing.assert_array_equal(
        dc.positive_down_values([1.0, 2.0], {"positive": "up"}, convention=conv),
        [-1.0, -2.0],
    )
    np.testing.assert_array_equal(
        dc.positive_down_values([-1.0, -2.0], {"positive": "down"}, convention=conv),
        [-1.0, -2.0],
    )
    # the sign heuristic: all <= 0 with some < 0 reads as up...
    np.testing.assert_array_equal(
        dc.positive_down_values([0.0, -1.0, -5.0, np.nan], {}, convention=conv),
        [0.0, 1.0, 5.0, np.nan],
    )
    # ...mixed signs read as down, and are left alone
    np.testing.assert_array_equal(
        dc.positive_down_values([-1.0, 5.0], {}, convention=conv), [-1.0, 5.0]
    )
    # ...and with nothing to go on, down
    np.testing.assert_array_equal(
        dc.positive_down_values([np.nan], {}, convention=conv), [np.nan]
    )
    np.testing.assert_array_equal(
        dc.positive_down_values([3.0], None, convention=conv), [3.0]
    )


def test_no_convention_at_all_uses_the_attrs_and_values():
    np.testing.assert_array_equal(dc.positive_down_values([-3.0, -1.0]), [3.0, 1.0])
    np.testing.assert_array_equal(dc.positive_down_values([3.0, 1.0]), [3.0, 1.0])
    np.testing.assert_array_equal(
        dc.positive_down_values([3.0, 1.0], {"positive": "up"}), [-3.0, -1.0]
    )


def test_units_fall_back_to_the_attrs_then_metres(monkeypatch):
    monkeypatch.setattr(dc, "M_PER_DBAR", 2.0)
    conv = _conv({"origin": "fixed"})
    np.testing.assert_array_equal(
        dc.positive_down_values([1.0, 2.0], {"units": "dbar"}, convention=conv),
        [2.0, 4.0],
    )
    np.testing.assert_array_equal(
        dc.positive_down_values([1.0, 2.0], {"units": "decibars"}), [2.0, 4.0]
    )
    np.testing.assert_array_equal(
        dc.positive_down_values([1.0, 2.0], {"units": "m"}, convention=conv), [1.0, 2.0]
    )
    np.testing.assert_array_equal(
        dc.positive_down_values([1.0, 2.0], {}, convention=conv), [1.0, 2.0]
    )
    # declared units beat the attrs
    declared_m = _conv({"units": "m"})
    np.testing.assert_array_equal(
        dc.positive_down_values([1.0], {"units": "dbar"}, convention=declared_m), [1.0]
    )


def test_a_plain_mapping_convention_is_trusted_for_the_keys_it_sets():
    np.testing.assert_array_equal(
        dc.positive_down_values([-1.0, -2.0], {}, convention={"positive": "down"}),
        [-1.0, -2.0],
    )
    np.testing.assert_array_equal(
        dc.positive_down_values([1.0, 2.0], {}, convention={"positive": "UP"}),
        [-1.0, -2.0],
    )
    # a key it does not set falls through to the attrs / values
    np.testing.assert_array_equal(
        dc.positive_down_values([-1.0, -2.0], {}, convention={"units": "m"}), [1.0, 2.0]
    )
    np.testing.assert_array_equal(
        dc.positive_down_values(
            [1.0], {"positive": "up"}, convention={"origin": "fixed"}
        ),
        [-1.0],
    )


def test_the_frame_and_key_dicts_work_as_mapping_conventions():
    conv = _conv({"positive": "up", "origin": "fixed"})
    np.testing.assert_array_equal(
        dc.positive_down_values([-4.0], {}, convention=conv.key()), [4.0]
    )
    # the frame says nothing about sign or units, so the values decide
    np.testing.assert_array_equal(
        dc.positive_down_values([-4.0], {}, convention=conv.frame()), [4.0]
    )
    np.testing.assert_array_equal(
        dc.positive_down_values([4.0], {}, convention=conv.frame()), [4.0]
    )


def test_a_bad_convention_is_an_error():
    _error(dc.positive_down_values, [1.0], {}, convention={"positive": "sideways"})
    _error(dc.positive_down_values, [1.0], {}, convention="surface")


def test_the_input_is_never_mutated():
    values = np.array([-1.0, -2.0])
    dc.positive_down_values(values, {"positive": "up"})
    np.testing.assert_array_equal(values, [-1.0, -2.0])


def test_it_replaces_abs_where_abs_gets_a_coordinate_that_crosses_zero_wrong():
    """A height of +2 m is 2 m *above* the origin: depth -2, not 2 as ``abs`` says."""
    heights = np.array([2.0, -1.0, -5.0])
    out = dc.positive_down_values(heights, {"positive": "up"})
    np.testing.assert_array_equal(out, [-2.0, 1.0, 5.0])
    assert np.abs(heights).tolist() == [2.0, 1.0, 5.0]
