"""Tests for ``find_variable``'s generic-to-specific matching (mixed layer depth).

Mixed layer depth has one generic CF name (``ocean_mixed_layer_thickness``) and
several definition-specific ones (by sigma_theta, sigma_t, temperature, a model's
mixing scheme). The vocabulary records that each specific name is one *kind of* the
generic one (its ``broader``) and turns that into data: the generic entry's cf-xarray
criteria also match every specific definition. So through plain cf-xarray lookup:

- a request for the generic name finds the one specific variable a dataset carries
  when it has no generic one, and warns which definition it picked;
- a variable literally carrying the generic name wins over a specific one alongside
  it (the literal-name check runs before cf-xarray);
- two different definitions are never chosen between -- they are different
  quantities, so either would be a coin flip: ``None``, with a warning naming them;
- a request for one definition is never answered by the generic name or by a sibling.

Datasets here are tiny synthetic ones. What matters is which *variable* comes back,
so each carries its own constant value and the tests check name and value.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pytest
import xarray as xr

from ocean_skill.units import find_variable

MLD = "ocean_mixed_layer_thickness"
BY_SIGMA_THETA = "ocean_mixed_layer_thickness_defined_by_sigma_theta"
BY_SIGMA_T = "ocean_mixed_layer_thickness_defined_by_sigma_t"
BY_TEMPERATURE = "ocean_mixed_layer_thickness_defined_by_temperature"
BY_MIXING_SCHEME = "ocean_mixed_layer_thickness_defined_by_mixing_scheme"
#: In the sorted order the fallback looks them up (and lists candidates) in.
MLD_DEFINITIONS = (BY_MIXING_SCHEME, BY_SIGMA_T, BY_SIGMA_THETA, BY_TEMPERATURE)
#: The short key a caller types for each definition -- and the "ask for ..." in the
#: ambiguity error.
KEYS = {
    BY_SIGMA_THETA: "mld_by_sigma_theta",
    BY_SIGMA_T: "mld_by_sigma_t",
    BY_TEMPERATURE: "mld_by_temperature",
    BY_MIXING_SCHEME: "mld_by_mixing_scheme",
}
#: Every name a source might carry that is *not* a given specific definition.
GENERIC_NAMES = (MLD, "mld", "mixed_layer_depth", "mixed_layer_thickness")


def _dataset(variables: dict[str, dict]) -> xr.Dataset:
    """Build a 2x2 dataset with one variable per entry (name -> its attrs).

    Each variable is a distinct constant (1.0, 2.0, ... in insertion order), so a
    test can tell *which* variable came back by value as well as by name.
    """
    return xr.Dataset(
        {
            name: (("lat", "lon"), np.full((2, 2), float(i)), attrs)
            for i, (name, attrs) in enumerate(variables.items(), start=1)
        },
        coords={"lat": [10.0, 11.0], "lon": [200.0, 201.0]},
    )


def _find(ds, name):
    """``find_variable`` with its "resolved to ..." notice silenced."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return find_variable(ds, name)


def _find_recording(ds, name):
    """``find_variable``, also returning every UserWarning message it raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        da = find_variable(ds, name)
    return da, [str(w.message) for w in caught if issubclass(w.category, UserWarning)]


def _value(da) -> float:
    return float(da.isel(lat=0, lon=0))


# -- a generic request finds the one definition a dataset carries -----------------

#: The ways a dataset can carry a definition: the variable named by the standard_name
#: itself (a catalog renamed it), named by the short key, or under a raw product
#: name with the standard_name as an attribute alone.
CARRIERS = {
    "standard_name": lambda definition: {definition: {}},
    "short_key": lambda definition: {KEYS[definition]: {}},
    "attribute_only": lambda definition: {"mld_dt_mean": {"standard_name": definition}},
}


@pytest.mark.parametrize("definition", MLD_DEFINITIONS)
@pytest.mark.parametrize("carrier", sorted(CARRIERS))
def test_generic_request_finds_the_one_definition_a_dataset_carries(
    carrier, definition
):
    variables = CARRIERS[carrier](definition)
    (expected,) = variables

    da = _find(_dataset(variables), "mld")

    assert da is not None
    assert da.name == expected
    assert _value(da) == 1.0


@pytest.mark.parametrize(
    "spelling", ["mld", "MLD", "mixed_layer_depth", "Mixed_Layer_Thickness", MLD]
)
def test_every_spelling_of_the_generic_name_takes_the_fallback(spelling):
    da = _find(_dataset({BY_SIGMA_THETA: {}}), spelling)
    assert da is not None and da.name == BY_SIGMA_THETA


def test_the_fallback_keeps_the_datasets_coordinates():
    """As on the ordinary path, a plain ``ds[name]`` -- not cf-xarray's own result."""
    ds = _dataset({BY_SIGMA_THETA: {}}).assign_coords(
        mask=(("lat", "lon"), np.ones((2, 2)))
    )
    assert "mask" in _find(ds, "mld").coords


def test_generic_request_warns_naming_the_definition_it_picked():
    da, messages = _find_recording(_dataset({BY_SIGMA_THETA: {}}), "mld")

    assert da.name == BY_SIGMA_THETA
    assert messages == [
        f"'mld' resolved to '{BY_SIGMA_THETA}', one specific definition of '{MLD}'"
    ]
    # the variable is literally the definition, so the name is not said twice
    assert messages[0].count(BY_SIGMA_THETA) == 1


def test_the_warning_names_the_variable_when_it_differs_from_the_definition():
    ds = _dataset({"mld_dt_mean": {"standard_name": BY_SIGMA_THETA}})

    _, messages = _find_recording(ds, "mld")

    assert messages == [
        (
            f"'mld' resolved to '{BY_SIGMA_THETA}' (variable 'mld_dt_mean'), "
            f"one specific definition of '{MLD}'"
        )
    ]


def test_the_warning_does_not_repeat_the_generic_name_when_asked_for_in_full():
    _, messages = _find_recording(_dataset({BY_SIGMA_T: {}}), MLD)

    assert messages == [
        f"'{MLD}' resolved to '{BY_SIGMA_T}', one specific definition of it"
    ]


def test_the_fallback_warning_blames_the_callers_own_code():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        find_variable(_dataset({BY_SIGMA_THETA: {}}), "mld")
    ours = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(ours) == 1
    assert ours[0].filename == __file__


# -- a variable carrying the generic name always wins -----------------------------


@pytest.mark.parametrize(
    "variables",
    [
        # the literal generic name, with a definition listed after and before it
        {MLD: {}, BY_SIGMA_THETA: {}},
        {BY_SIGMA_THETA: {}, MLD: {}},
        # ... even with two different definitions alongside
        {MLD: {}, BY_SIGMA_THETA: {}, BY_TEMPERATURE: {}},
    ],
)
def test_a_variable_literally_named_the_generic_name_wins(variables):
    """The literal-name check runs before cf-xarray, so the generic name wins outright.

    That is the case real data arrives in: a catalog's rename (ROMS' ``hbls`` on an
    older catalog, say) gives the generic variable exactly this name.
    """
    da, messages = _find_recording(_dataset(variables), "mld")

    assert da.name == MLD
    assert not any("specific definition" in m for m in messages)


@pytest.mark.parametrize(
    "variables",
    [
        {"mixed_layer_depth": {}, BY_TEMPERATURE: {}},  # an alias of the generic name
        {"h": {"standard_name": MLD}, BY_SIGMA_T: {}},  # generic as an attribute only
        {"mld": {}, BY_SIGMA_THETA: {}},  # the short key as a variable name
    ],
)
def test_a_generic_variable_under_another_spelling_ties_with_a_definition(variables):
    """Spelled any other way, the generic variable is one more cf-xarray match.

    Deliberately not a preference coded here: the broad-to-specific matching is the
    registered criteria alone (so another tool gets the same answer from cf-xarray),
    and cf-xarray has no notion of one match outranking another. Two matches, no
    guess -- ``None`` and a warning naming both, as for any ambiguity.
    """
    da, messages = _find_recording(_dataset(variables), "mld")

    assert da is None
    assert any("matches more than one variable" in m for m in messages)


def test_asking_for_the_generic_name_in_full_is_silent_when_a_variable_has_it():
    da, messages = _find_recording(_dataset({MLD: {}, BY_SIGMA_THETA: {}}), MLD)
    assert da.name == MLD
    assert messages == []


# -- two definitions are refused, not chosen between ------------------------------


@pytest.mark.parametrize("pair", list(itertools.combinations(MLD_DEFINITIONS, 2)))
def test_two_definitions_present_are_not_chosen_between(pair):
    """Neither is returned: a coin flip between two quantities is not an answer.

    It is ``None`` -- "not available", which is what lets a comparison skip the pair
    and a combination fall back to its own standard_name -- plus one warning naming
    both candidates, since the variable is plainly there and "not found" alone would
    mislead.
    """
    a, b = pair

    da, messages = _find_recording(_dataset({a: {}, b: {}}), "mld")

    assert da is None
    (message,) = messages
    assert "matches more than one variable" in message
    assert repr(a) in message and repr(b) in message


def test_all_four_definitions_present_lists_all_four_candidates():
    ds = _dataset({definition: {} for definition in reversed(MLD_DEFINITIONS)})

    da, (message,) = _find_recording(ds, "mld")

    assert da is None
    # listed sorted, not in the dataset's own order
    positions = [message.index(repr(d)) for d in MLD_DEFINITIONS]
    assert positions == sorted(positions)


def test_the_ambiguity_warning_names_both_variables_and_both_keys():
    ds = _dataset(
        {
            "a": {"standard_name": BY_SIGMA_THETA},
            "b": {"standard_name": BY_TEMPERATURE},
        }
    )

    da, (message,) = _find_recording(ds, "mld")

    assert da is None
    assert message == (
        "'mld' matches more than one variable in this dataset "
        "('a' (ask for 'mld_by_sigma_theta'), 'b' (ask for 'mld_by_temperature')), "
        "so it is not being resolved to either; ask for the one you mean by name."
    )


def test_definitions_carried_in_different_ways_are_still_ambiguous():
    ds = _dataset({BY_SIGMA_T: {}, "mld_dt_mean": {"standard_name": BY_SIGMA_THETA}})

    da, (message,) = _find_recording(ds, "mld")

    assert da is None
    assert repr(BY_SIGMA_T) in message and "'mld_dt_mean'" in message


def test_the_ambiguous_case_is_not_an_error():
    """``None``, not an exception: callers treat it exactly like a missing variable."""
    ds = _dataset({BY_SIGMA_THETA: {}, BY_TEMPERATURE: {}})
    assert _find(ds, "mld") is None


def test_the_broad_to_specific_match_is_plain_cf_xarray():
    """cf-xarray's own lookup, with the registered criteria, finds the definition.

    No ocean-skill code decides it -- which is what lets another tool that registers
    the same criteria (ROMS-Tools, xroms) behave the same way.
    """
    import cf_xarray  # noqa: F401  (registers the .cf accessor)

    import ocean_skill.vocabulary  # noqa: F401  (registers the criteria)

    ds = _dataset({BY_SIGMA_THETA: {}, "temp": {}})
    assert ds.cf["ocean_mixed_layer_thickness"].name == BY_SIGMA_THETA
    assert ds.cf["mld"].name == BY_SIGMA_THETA
    with pytest.raises(KeyError, match="multiple variables"):
        _dataset({BY_SIGMA_THETA: {}, BY_TEMPERATURE: {}}).cf["mld"]


def test_naming_one_definition_settles_the_ambiguity():
    ds = _dataset({BY_SIGMA_THETA: {}, BY_TEMPERATURE: {}})

    assert _find(ds, "mld_by_sigma_theta").name == BY_SIGMA_THETA
    assert _find(ds, BY_TEMPERATURE).name == BY_TEMPERATURE


def test_one_variable_reachable_under_two_definitions_is_not_ambiguous():
    """Named for one definition but tagged with another: a single hit, not two.

    Contradictory metadata, but the fallback groups hits by the *variable* found (it
    is reachable through the sigma_theta name and, by attribute, the sigma_t one), so
    it stays one candidate rather than tripping a spurious ambiguity.
    """
    ds = _dataset({BY_SIGMA_THETA: {"standard_name": BY_SIGMA_T}})

    da = _find(ds, "mld")

    assert da is not None and da.name == BY_SIGMA_THETA


def test_a_definition_that_matches_several_variables_is_absent_not_raised():
    """cf-xarray's own several-variables error is read as "no match", as it always was.

    So the generic request agrees with asking for that definition directly: neither
    raises, neither guesses.
    """
    ds = _dataset(
        {
            "a": {"standard_name": BY_SIGMA_THETA},
            "b": {"standard_name": BY_SIGMA_THETA},
        }
    )

    assert _find(ds, BY_SIGMA_THETA) is None
    assert _find(ds, "mld") is None


# -- a specific request never falls back ------------------------------------------

#: Every (requested definition, the only name present) pair where the name present is
#: the generic name (in any spelling) or a sibling definition -- never the request.
NOT_THE_REQUESTED = [
    (requested, present)
    for requested in MLD_DEFINITIONS
    for present in (*GENERIC_NAMES, *MLD_DEFINITIONS)
    if present != requested
]


@pytest.mark.parametrize("requested,present", NOT_THE_REQUESTED)
@pytest.mark.parametrize("carried_by", ["name", "attribute"])
def test_a_specific_request_is_never_answered_by_the_generic_or_a_sibling(
    requested, present, carried_by
):
    variables = (
        {present: {}} if carried_by == "name" else {"raw": {"standard_name": present}}
    )
    ds = _dataset(variables)

    assert _find(ds, requested) is None
    assert _find(ds, KEYS[requested]) is None


def test_sigma_t_and_sigma_theta_are_never_confused_for_one_another():
    """One name is a prefix of the other, and cf-xarray matches with ``re.match``."""
    theta_only = _dataset({BY_SIGMA_THETA: {}})
    t_only = _dataset({BY_SIGMA_T: {}})

    assert _find(theta_only, BY_SIGMA_T) is None
    assert _find(theta_only, "mld_by_sigma_t") is None
    assert _find(t_only, BY_SIGMA_THETA) is None
    assert _find(t_only, "mld_by_sigma_theta") is None
    # while the generic request still gets whichever one is actually there
    assert _find(theta_only, "mld").name == BY_SIGMA_THETA
    assert _find(t_only, "mld").name == BY_SIGMA_T


def test_a_specific_request_finds_exactly_its_own_definition_among_the_others():
    ds = _dataset({MLD: {}, **{definition: {} for definition in MLD_DEFINITIONS}})

    for definition in MLD_DEFINITIONS:
        assert _find(ds, definition).name == definition
        assert _find(ds, KEYS[definition]).name == definition
    assert _find(ds, "mld").name == MLD


# -- QC companions stay invisible -------------------------------------------------


def test_a_qc_companion_of_a_definition_is_still_ignored():
    assert _find(_dataset({f"{BY_SIGMA_THETA}_qc": {}}), "mld") is None


def test_a_qc_companion_claiming_a_definitions_standard_name_is_ignored():
    """cf-xarray would match it by attribute; the QC layer hides it from the search."""
    ds = _dataset({"mld_dt_mean_qc": {"standard_name": BY_SIGMA_THETA}})
    assert _find(ds, "mld") is None


def test_the_real_definition_wins_over_its_qc_companion():
    ds = _dataset({BY_SIGMA_THETA: {}, f"{BY_SIGMA_THETA}_qc": {}})

    da = _find(ds, "mld")

    assert da.name == BY_SIGMA_THETA
    assert _value(da) == 1.0


def test_a_qc_companion_of_a_second_definition_is_not_a_second_candidate():
    ds = _dataset(
        {
            BY_SIGMA_THETA: {},
            f"{BY_TEMPERATURE}_qc_agg": {},
            "raw_qc": {"standard_name": BY_SIGMA_T},
        }
    )
    assert _find(ds, "mld").name == BY_SIGMA_THETA  # one candidate, not two


def test_a_request_that_names_a_flag_still_gets_the_flag():
    flag = f"{BY_SIGMA_THETA}_qc"
    ds = _dataset({BY_SIGMA_THETA: {}, flag: {}})
    assert _find(ds, flag).name == flag


def test_a_flag_that_is_the_only_match_for_the_generic_name_is_still_reported():
    """The "closest match is a QC flag" notice survives the end-of-lookup move."""
    ds = _dataset({"mld_qc": {"standard_name": MLD}})

    with pytest.warns(UserWarning, match="QC flag"):
        assert find_variable(ds, "mld") is None


def test_no_flag_notice_when_a_definition_was_found_after_all():
    ds = _dataset({"mld_qc": {"standard_name": MLD}, BY_SIGMA_THETA: {}})

    da, messages = _find_recording(ds, "mld")

    assert da.name == BY_SIGMA_THETA
    assert len(messages) == 1 and "specific definition" in messages[0]


# -- nothing to find --------------------------------------------------------------


def test_an_unknown_name_still_returns_none_without_error_or_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        assert find_variable(_dataset({BY_SIGMA_THETA: {}}), "not_a_variable") is None


def test_the_generic_name_on_a_dataset_with_nothing_mld_like_is_none():
    ds = _dataset({"temp": {}, "salt": {}, "sea_floor_depth_below_geoid": {}})

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        assert find_variable(ds, "mld") is None


def test_two_variables_differing_only_by_case_are_never_coin_flipped():
    """By name, ``_match_name`` refuses; asked for generically, it's ambiguous.

    Either way neither variable is returned: the specific request raises
    ``_match_name``'s case error, the generic one sees two cf-xarray matches.
    """
    ds = _dataset({BY_SIGMA_THETA.upper(): {}, BY_SIGMA_THETA.title(): {}})

    with pytest.raises(ValueError, match="only by case"):
        find_variable(ds, BY_SIGMA_THETA)
    da, messages = _find_recording(ds, "mld")
    assert da is None
    assert any("matches more than one variable" in m for m in messages)
