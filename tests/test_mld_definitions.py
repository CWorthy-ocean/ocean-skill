"""Tests for definition-aware mixed layer depth in the compare layer.

Mixed layer depth has one *broad* name -- ``"mld"``, the generic
``ocean_mixed_layer_thickness`` -- and four *specific* definitions filed under it
(by sigma_theta, by sigma_t, by temperature, by a model's own mixing scheme; each
names the generic entry as its ``broader`` in :mod:`ocean_skill.vocabulary`). The
relation is one-directional (:func:`ocean_skill.vocabulary.covers`): a broad request
is satisfied by a source carrying any one definition, a specific request only by a
source carrying that definition -- never by the generic name, which has not said which
one it is, and never by a sibling.

This file covers what that does to the comparison layer:

- which sources :func:`~ocean_skill.comparison.compare` pairs for a broad request and
  for a specific one, and a dataset carrying two definitions reading as "not
  available" (with a warning) rather than aborting a batch;
- the warning when a plain broad request lands on two *different* definitions -- ROMS'
  KPP ``hbls`` against a Holte & Talley sigma_theta climatology being the case that
  motivated it -- on a freshly computed pair and on one served from the cache
  (:meth:`Comparison._warn_on_definition_mismatch`);
- ``.sel(variable=...)`` and :attr:`Comparison.standard_name`, which speak the same
  rule.

The pair-spec side of the same change (the mismatch warning comparing quantities rather
than spellings, and the folded short label) lives in ``test_pair_spec.py``, next to the
tests it extends.
"""

from __future__ import annotations

import re
import warnings
from unittest import mock

import numpy as np
import pytest
import xarray as xr

from ocean_skill import comparison
from ocean_skill.comparison import (
    Comparison,
    ComparisonSet,
    _aligned_standard_name,
    _lane_standard_name,
    _variable_matches,
)

MLD = "ocean_mixed_layer_thickness"
BY_SIGMA_THETA = "ocean_mixed_layer_thickness_defined_by_sigma_theta"
BY_SIGMA_T = "ocean_mixed_layer_thickness_defined_by_sigma_t"
BY_TEMPERATURE = "ocean_mixed_layer_thickness_defined_by_temperature"
BY_MIXING_SCHEME = "ocean_mixed_layer_thickness_defined_by_mixing_scheme"

#: The sentence every definition-mismatch warning opens its diagnosis with -- what the
#: tests below key on to tell it apart from any other warning a comparison can raise.
DEFINITION_WARNING = "without saying which definition"

#: A pair-spec with a calculated test side and no explicit standard_name: the one
#: shape whose CF name is unknown until it has run.
PAIR = {
    "test": {"calculate": "mld", "method": "density_threshold"},
    "reference": "mld_dt_mean",
}

#: What a hand-built catalog index declares, per source. Every one is a shape a real
#: catalog has: an older ROMS catalog (built before the definition-specific names
#: existed, so KPP's ``hbls`` is still under the generic one), a rebuilt one, a
#: Copernicus-like sigma_theta product, a CMIP-like sigma_t product, and a source that
#: has no mixed layer depth at all.
CATALOG = {
    "gom_old": {"variables": [MLD]},
    "gom_rebuilt": {"variables": [BY_MIXING_SCHEME]},
    "copernicus": {"variables": [BY_SIGMA_THETA]},
    "cmip": {"variables": [BY_SIGMA_T]},
    "temperature_only": {"variables": ["sea_water_potential_temperature"]},
}


# -- helpers ---------------------------------------------------------------------


def _paired(declared, *, reference, test, variables, **kwargs):
    """Run compare() against a hand-built catalog; return the pairs it formed.

    ``align`` is replaced by a recorder, so this exercises exactly the part under
    test -- which sources each side of the request is offered by -- with nothing
    read. Pairs come back as sorted ``(test, reference)`` tuples.
    """
    formed = []
    with (
        mock.patch(
            "ocean_skill.catalog.resolve", lambda n: mock.Mock(metadata=declared[n])
        ),
        mock.patch.object(
            comparison.Comparison,
            "align",
            lambda self, refresh=False: formed.append(
                (self.test_name, self.reference_name)
            ),
        ),
        warnings.catch_warnings(),
    ):
        # "mld" resolves to its standard_name with a notice; not what these are about.
        warnings.simplefilter("ignore", UserWarning)
        comparison.compare(
            reference=reference, test=test, variables=variables, **kwargs
        )
    return sorted(formed)


def _comparison(variable=MLD):
    """Build a Comparison of ``variable`` between two sources that need no catalog."""
    return Comparison(reference="r", test="t", variable=variable)


def _definition_warnings(record):
    """Pick the definition-mismatch messages out of a warnings record."""
    return [str(w.message) for w in record if DEFINITION_WARNING in str(w.message)]


# -- compare(): which sources a broad and a specific request pair ---------------


def test_a_broad_request_pairs_the_generic_name_with_a_specific_definition():
    """``variables=["mld"]`` pairs an older ROMS catalog with a sigma_theta product.

    The test side declares only the generic name and the reference only the
    sigma_theta one; the broad request is satisfied by either, on either side.
    """
    assert _paired(
        CATALOG, reference=["copernicus"], test=["gom_old"], variables=["mld"]
    ) == [("gom_old", "copernicus")]


def test_a_broad_request_pairs_every_definition_and_nothing_else():
    pairs = _paired(
        CATALOG,
        reference=["copernicus", "cmip", "temperature_only"],
        test=["gom_old", "gom_rebuilt", "temperature_only"],
        variables=["mld"],
    )
    assert pairs == sorted(
        (t, r) for t in ("gom_old", "gom_rebuilt") for r in ("copernicus", "cmip")
    )


@pytest.mark.parametrize("request_name", ["mld_by_sigma_theta", BY_SIGMA_THETA])
def test_a_specific_request_does_not_pair_a_reference_declaring_only_the_generic_name(
    request_name,
):
    """The generic name has not said which definition it carries.

    Both lanes are held to the *specific* request, so of the three references only
    the one declaring sigma_theta pairs -- not the generic one, and not the sigma_t
    one whose name merely shares a prefix.
    """
    declared = {
        **CATALOG,
        "obs_generic": {"variables": [MLD]},
        "model_sigma_theta": {"variables": [BY_SIGMA_THETA]},
    }
    assert _paired(
        declared,
        reference=["obs_generic", "copernicus", "cmip"],
        test=["model_sigma_theta"],
        variables=[request_name],
    ) == [("model_sigma_theta", "copernicus")]


def test_a_specific_request_does_not_pair_a_test_declaring_only_the_generic_name(
    capsys,
):
    """An older ROMS catalog is not a sigma_theta source, and is skipped as absent."""
    pairs = _paired(
        CATALOG,
        reference=["copernicus"],
        test=["gom_old"],
        variables=["mld_by_sigma_theta"],
    )
    assert pairs == []
    assert "no test offers" in capsys.readouterr().out


def test_a_pair_spec_holds_each_side_to_its_own_request():
    """A broad test side and a specific reference side are filtered independently.

    ``{"test": "mld", "reference": "mld_by_sigma_theta"}`` accepts any mixed layer depth
    on the test side but only a sigma_theta one on the reference side.
    """
    pairs = _paired(
        {**CATALOG, "obs_generic": {"variables": [MLD]}},
        reference=["obs_generic", "copernicus", "cmip"],
        test=["gom_old", "gom_rebuilt", "temperature_only"],
        variables=[{"test": "mld", "reference": "mld_by_sigma_theta"}],
    )
    assert pairs == [("gom_old", "copernicus"), ("gom_rebuilt", "copernicus")]


# -- an ambiguous dataset reads as "not available", not fatal -------------------


def _two_definitions() -> xr.Dataset:
    """Build a dataset carrying two definitions of mixed layer depth, no generic one."""
    return xr.Dataset(
        {
            BY_SIGMA_THETA: (("lat", "lon"), np.full((2, 2), 1.0), {"units": "m"}),
            BY_TEMPERATURE: (("lat", "lon"), np.full((2, 2), 2.0), {"units": "m"}),
        },
        coords={"lat": [1.0, 2.0], "lon": [1.0, 2.0]},
    )


def test_prepare_treats_an_ambiguous_dataset_as_not_carrying_the_variable():
    """A broad request of two definitions: ``(None, None)``, the "not available" answer.

    The same answer as a variable the dataset lacks, so compare() skips the pair (and
    a combination falls back to its own standard_name) with no special case -- and the
    lookup's warning names both candidates, so the skip is not a mystery.
    """
    from ocean_skill.comparison import _prepare

    with pytest.warns(UserWarning, match="matches more than one variable"):
        da, depth = _prepare(_two_definitions(), {}, MLD, {})
    assert da is None and depth is None


def test_an_ambiguous_dataset_still_answers_a_request_for_one_definition():
    from ocean_skill.comparison import _prepare

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        da, _depth = _prepare(_two_definitions(), {}, "mld_by_sigma_theta", {})
    assert da is not None and da.name == BY_SIGMA_THETA


# -- the definition-mismatch warning: unit level --------------------------------


@pytest.mark.parametrize(
    ("test_name", "reference_name"),
    [
        (MLD, BY_SIGMA_THETA),  # an older ROMS catalog's hbls vs Holte & Talley
        (BY_MIXING_SCHEME, BY_SIGMA_THETA),  # a rebuilt one vs Holte & Talley
        (BY_SIGMA_THETA, BY_MIXING_SCHEME),  # ... and with the sides the other way
        (BY_SIGMA_T, BY_TEMPERATURE),  # two specific definitions, neither generic
        (BY_SIGMA_THETA, MLD),  # the reference is the vague side
    ],
)
def test_a_broad_request_warns_when_the_two_sides_carry_different_definitions(
    test_name, reference_name
):
    c = _comparison()
    with pytest.warns(UserWarning) as record:
        c._warn_on_definition_mismatch(test_name, reference_name)
    assert len(_definition_warnings(record)) == 1


def test_the_warning_names_both_definitions_and_says_how_to_settle_it():
    """Both sides by label and standard_name, why it matters, and both ways out."""
    c = _comparison()
    with pytest.warns(UserWarning) as record:
        c._warn_on_definition_mismatch(BY_MIXING_SCHEME, BY_SIGMA_THETA)
    (message,) = _definition_warnings(record)
    # what was asked for, and which definition each side actually carries
    assert "asked for 'mld'" in message
    assert "'MLD (mixing scheme)' (" + BY_MIXING_SCHEME + ")" in message
    assert "'MLD (σθ)' (" + BY_SIGMA_THETA + ")" in message
    assert re.search(
        r"test side is .*mixing_scheme.*reference side is .*sigma_theta", message
    )
    # why it matters: different quantities the metrics will not tell apart
    assert "different quantities" in message
    assert "metrics will not distinguish" in message
    # both ways out: ask for one definition, or say it is intentional with a pair-spec
    assert "variables=['mld_by_mixing_scheme']" in message
    assert "variables=['mld_by_sigma_theta']" in message
    assert "'standard_name'" in message
    assert "pair-spec" in message


def test_the_warning_says_so_when_a_side_states_no_definition():
    """The generic name is called out as the side that stated none.

    It is the unremarkable-looking side of the pair (it is the very name that was
    asked for), and also the vaguer one; and only the *specific* side is worth
    suggesting as a request, since asking for the generic name again fixes nothing.
    """
    c = _comparison()
    with pytest.warns(UserWarning) as record:
        c._warn_on_definition_mismatch(MLD, BY_SIGMA_THETA)
    (message,) = _definition_warnings(record)
    assert f"the generic {MLD!r} (no definition stated)" in message
    assert "variables=['mld_by_sigma_theta']" in message
    assert "variables=['mld']" not in message


@pytest.mark.parametrize(
    ("test_name", "reference_name"),
    [
        (BY_SIGMA_THETA, BY_SIGMA_THETA),  # the same definition on both sides
        (MLD, MLD),  # two generic lanes: no definition on either to differ
        (MLD, "mixed_layer_depth"),  # two spellings of the one generic name
        (BY_SIGMA_THETA, BY_SIGMA_THETA.upper()),  # ... or of one definition
        ("mld", "mixed_layer_thickness"),  # short key and alias, resolved first
    ],
)
def test_a_broad_request_is_silent_when_the_sides_carry_the_same_definition(
    test_name, reference_name
):
    c = _comparison()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        c._warn_on_definition_mismatch(test_name, reference_name)


@pytest.mark.parametrize(
    ("test_name", "reference_name"),
    [
        (None, BY_SIGMA_THETA),  # a lane that named nothing
        (BY_SIGMA_THETA, None),
        ("", BY_MIXING_SCHEME),  # an empty name is no name
        (None, None),
        ("mld_dt_mean", BY_SIGMA_THETA),  # a raw name the vocabulary cannot judge
        (BY_SIGMA_THETA, "some_other_variable"),
        ("sea_water_temperature", "sea_water_practical_salinity"),  # outside the family
    ],
)
def test_a_missing_or_foreign_name_leaves_nothing_to_check(test_name, reference_name):
    """Only two names *inside* the request's family can disagree about a definition."""
    c = _comparison()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        c._warn_on_definition_mismatch(test_name, reference_name)


def test_a_plain_variable_with_no_definitions_never_warns():
    """``narrower_names`` of a non-family name is empty: nothing to disagree about.

    Even handed two names that *would* mismatch under a broad mixed layer depth
    request, a comparison of temperature has no definitions to tell apart.
    """
    c = _comparison("sea_water_potential_temperature")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        c._warn_on_definition_mismatch(BY_MIXING_SCHEME, BY_SIGMA_THETA)
        c._warn_on_definition_mismatch(
            "sea_water_potential_temperature", "sea_water_temperature"
        )


def test_a_request_that_already_names_one_definition_never_warns():
    """A specific request is answered by that definition or not paired at all.

    ``narrower_names("mld_by_sigma_theta")`` is empty, so this is not the check's
    case even when the lanes somehow disagree -- compare() has already held both
    sources to the one definition.
    """
    c = _comparison(BY_SIGMA_THETA)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        c._warn_on_definition_mismatch(BY_SIGMA_THETA, BY_TEMPERATURE)


def test_a_pair_spec_is_left_to_the_pair_spec_check():
    """A pair-spec wrote its two recipes down itself; this check is for plain names."""
    c = _comparison({"test": BY_MIXING_SCHEME, "reference": BY_SIGMA_THETA})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        c._warn_on_definition_mismatch(BY_MIXING_SCHEME, BY_SIGMA_THETA)


def test_a_combination_is_left_alone():
    """A combination has no single name to have definitions of."""
    c = _comparison({"sum": ["a", "b"], "standard_name": MLD})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        c._warn_on_definition_mismatch(BY_MIXING_SCHEME, BY_SIGMA_THETA)


# -- the resolved lane names the check reads -----------------------------------


def test_a_lane_is_named_by_its_attribute_then_by_its_own_name():
    """``attrs["standard_name"] or .name``, canonicalized -- what ROMS lanes need.

    A ROMS lane carries no ``standard_name`` attribute at all (``roms.standardize``
    renames from the catalog's map and stamps nothing), so its variable *name* is the
    only record of which definition it is.
    """
    stamped = xr.DataArray(1.0, name="mlotst", attrs={"standard_name": BY_SIGMA_THETA})
    keyed = xr.DataArray(1.0, name=BY_MIXING_SCHEME)
    aliased = xr.DataArray(1.0, name="mixed_layer_depth")
    assert _lane_standard_name(stamped) == BY_SIGMA_THETA
    assert _lane_standard_name(keyed) == BY_MIXING_SCHEME
    assert _lane_standard_name(aliased) == MLD  # canonicalized, not passed through


def test_a_lane_that_names_nothing_yields_none():
    assert _lane_standard_name(xr.DataArray(1.0)) is None
    assert _lane_standard_name(xr.DataArray(1.0, name="")) is None
    assert _lane_standard_name(xr.DataArray(1.0, name=7)) is None  # not a string


def test_an_aligned_lane_is_named_by_the_recorded_attr_then_the_arrays_attr():
    """Never by the array's ``.name``, which is literally "test"/"reference"."""
    recorded = xr.Dataset(
        {"test": xr.DataArray(1.0), "reference": xr.DataArray(1.0)},
        attrs={"test_standard_name": BY_MIXING_SCHEME},
    )
    assert _aligned_standard_name(recorded, "test") == BY_MIXING_SCHEME
    assert _aligned_standard_name(recorded, "reference") is None  # .name doesn't count

    older = xr.Dataset(
        {
            "test": xr.DataArray(1.0, attrs={"standard_name": BY_SIGMA_T}),
            "reference": xr.DataArray(1.0),
        }
    )
    assert _aligned_standard_name(older, "test") == BY_SIGMA_T
    assert _aligned_standard_name(older, "reference") is None

    # the recorded attr wins over an array attribute that disagrees with it
    both = xr.Dataset(
        {"test": xr.DataArray(1.0, attrs={"standard_name": BY_SIGMA_T})},
        attrs={"test_standard_name": BY_SIGMA_THETA},
    )
    assert _aligned_standard_name(both, "test") == BY_SIGMA_THETA


# -- the warning on a cached result ---------------------------------------------


def _cached_pair(*, test_attrs=None, reference_attrs=None, **pair_attrs):
    """Build the aligned Dataset a cache hit hands back, with the given attrs.

    ``pair_attrs`` land on the Dataset itself (where ``align()`` records the resolved
    ``test_standard_name``/``reference_standard_name``); the two arrays are literally
    named ``"test"``/``"reference"``, as ``align.align`` names them.
    """
    return xr.Dataset(
        {
            "test": xr.DataArray(1.0, attrs=test_attrs or {}),
            "reference": xr.DataArray(1.0, attrs=reference_attrs or {}),
            "difference": xr.DataArray(0.0),
        },
        attrs=pair_attrs,
    )


def _align_from_cache(c, hit):
    """Align ``c`` with the cache answering ``hit``; return every warning raised."""
    with (
        mock.patch("ocean_skill.cache.load", return_value=hit),
        warnings.catch_warnings(record=True) as record,
    ):
        warnings.simplefilter("always")
        c.align()
    return record


def test_the_warning_fires_on_a_cache_hit_from_the_recorded_names():
    """A hit has only what ``align()`` stored; its arrays are just "test"/"reference".

    Without the recorded names a ROMS lane, which carries no ``standard_name``
    attribute, would leave a cache hit nothing to check -- so every process after the
    one that first computed the pair would silently miss the warning.
    """
    hit = _cached_pair(
        test_standard_name=BY_MIXING_SCHEME, reference_standard_name=BY_SIGMA_THETA
    )
    record = _align_from_cache(_comparison(), hit)
    assert len(_definition_warnings(record)) == 1


def test_a_cache_hit_from_before_the_names_were_recorded_falls_back_to_the_arrays():
    """An older entry still checks, when its arrays carried a ``standard_name``."""
    hit = _cached_pair(
        test_attrs={"standard_name": BY_SIGMA_T},
        reference_attrs={"standard_name": BY_SIGMA_THETA},
    )
    record = _align_from_cache(_comparison(), hit)
    assert len(_definition_warnings(record)) == 1


def test_a_cache_hit_with_nothing_recorded_stays_silent():
    """No recorded names and no array attrs: nothing real to compare, so no warning.

    The arrays are literally named "test" and "reference" -- always unequal -- and
    treating those as names would warn on every hit rather than only a real mismatch
    (the trap ``trust_name_fallback=False`` exists for on the pair-spec check).
    """
    assert list(_align_from_cache(_comparison(), _cached_pair())) == []


def test_a_cache_hit_of_one_definition_on_both_sides_stays_silent():
    hit = _cached_pair(
        test_standard_name=BY_SIGMA_THETA, reference_standard_name=BY_SIGMA_THETA
    )
    assert list(_align_from_cache(_comparison(), hit)) == []


def test_a_cache_hit_on_a_pair_spec_is_left_to_the_pair_spec_check():
    """The recorded names never make a pair-spec warn *here*.

    A pair-spec without an explicit standard_name has its own check, which reads only
    the arrays' attributes on a hit (``trust_name_fallback=False``); this one is for
    plain names, so recorded names differing on a pair-spec must not trip it.
    """
    c = Comparison(
        reference="r",
        test="t",
        variable={"test": BY_MIXING_SCHEME, "reference": BY_SIGMA_THETA},
    )
    hit = _cached_pair(
        test_standard_name=BY_MIXING_SCHEME, reference_standard_name=BY_SIGMA_THETA
    )
    assert list(_align_from_cache(c, hit)) == []


# -- the warning on a freshly computed result ------------------------------------
#
# These run the real align() -- real regridding, real disk cache -- over tiny
# in-memory lanes, replacing only prepare_source (which would otherwise read a
# catalog) so the lane *names and attributes* are exactly what each test says.


def _lane(name, value, *, standard_name=None):
    """Build a small gridded mixed layer depth lane, named ``name`` (maybe ``None``)."""
    attrs = {"units": "m"}
    if standard_name is not None:
        attrs["standard_name"] = standard_name
    return xr.DataArray(
        value + np.zeros((4, 5)),
        dims=("lat", "lon"),
        coords={
            "lat": np.linspace(58.5, 60.5, 4),
            "lon": np.linspace(-153.5, -150.5, 5),
        },
        name=name,
        attrs=attrs,
    )


@pytest.fixture
def served(monkeypatch):
    """Serve lanes from a dict, keyed by source name, in place of ``prepare_source``.

    Returns the dict, so a test fills it before aligning. ``_domain_of`` and
    ``_feature_type`` are stubbed the way the other end-to-end comparison tests stub
    them, so nothing consults a catalog.
    """
    lanes = {}
    monkeypatch.setattr(
        comparison, "prepare_source", lambda source, *a, **k: (lanes[source], None)
    )
    monkeypatch.setattr(comparison, "_domain_of", lambda name: None)
    monkeypatch.setattr(comparison, "_feature_type", lambda source: "grid")
    return lanes


def _align(c):
    """Align ``c``, returning the definition-mismatch messages it warned with."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        c.align()
    return _definition_warnings(record)


def test_a_fresh_pair_warns_once_and_records_each_lanes_resolved_name(served):
    """ROMS' KPP ``hbls`` (a mixing-scheme depth) against Holte & Talley (sigma_theta).

    The test lane carries its definition only as its variable's name -- no
    ``standard_name`` attribute, exactly as a ROMS lane arrives -- while the reference
    is stamped. Both resolved names end up on the aligned pair.
    """
    served["model"] = _lane(BY_MIXING_SCHEME, 20.0)
    served["obs"] = _lane("mld_dt_mean", 22.0, standard_name=BY_SIGMA_THETA)
    c = Comparison(reference="obs", test="model", variable=MLD)
    messages = _align(c)
    assert len(messages) == 1
    assert "'MLD (mixing scheme)'" in messages[0]
    assert "'MLD (σθ)'" in messages[0]
    assert c.aligned.attrs["test_standard_name"] == BY_MIXING_SCHEME
    assert c.aligned.attrs["reference_standard_name"] == BY_SIGMA_THETA


def test_a_cache_hit_of_that_pair_warns_again_from_what_was_stored(served, monkeypatch):
    """The recorded names survive the trip through the disk cache.

    The first ``align()`` writes the pair; ``prepare_source`` is then made to fail, so
    a second comparison of the same request can only be answered by the cache -- and
    the warning still fires, off the two names that were stored.
    """
    served["model"] = _lane(BY_MIXING_SCHEME, 20.0)
    served["obs"] = _lane("mld_dt_mean", 22.0, standard_name=BY_SIGMA_THETA)
    assert len(_align(Comparison(reference="obs", test="model", variable=MLD))) == 1

    def fail(*args, **kwargs):
        raise AssertionError("expected a cache hit, but a lane was prepared")

    monkeypatch.setattr(comparison, "prepare_source", fail)
    again = Comparison(reference="obs", test="model", variable=MLD)
    assert len(_align(again)) == 1
    assert again.aligned.attrs["test_standard_name"] == BY_MIXING_SCHEME
    assert again.aligned.attrs["reference_standard_name"] == BY_SIGMA_THETA


def test_an_older_catalogs_generic_lane_warns_against_a_specific_reference(served):
    """An older catalog's generic name still mismatches against a specific reference."""
    served["model"] = _lane(MLD, 20.0)
    served["obs"] = _lane("mld_dt_mean", 22.0, standard_name=BY_SIGMA_THETA)
    c = Comparison(reference="obs", test="model", variable=MLD)
    (message,) = _align(c)
    assert f"the generic {MLD!r} (no definition stated)" in message


def test_a_fresh_pair_of_one_definition_is_silent_and_still_records_the_names(served):
    served["model"] = _lane(BY_SIGMA_THETA, 20.0, standard_name=BY_SIGMA_THETA)
    served["obs"] = _lane("mld_dt_mean", 22.0, standard_name=BY_SIGMA_THETA)
    c = Comparison(reference="obs", test="model", variable=MLD)
    assert _align(c) == []
    assert c.aligned.attrs["test_standard_name"] == BY_SIGMA_THETA
    assert c.aligned.attrs["reference_standard_name"] == BY_SIGMA_THETA


def test_a_spelling_of_the_name_is_recorded_canonically(served):
    """Aliases are resolved before they are stored, so a later reader compares like."""
    served["model"] = _lane("mixed_layer_depth", 20.0)
    served["obs"] = _lane("mld_dt_mean", 22.0, standard_name="mixed_layer_thickness")
    c = Comparison(reference="obs", test="model", variable=MLD)
    assert _align(c) == []
    assert c.aligned.attrs["test_standard_name"] == MLD
    assert c.aligned.attrs["reference_standard_name"] == MLD


def test_a_lane_that_names_nothing_stores_nothing_and_is_not_warned_about(served):
    """No name is recorded rather than an empty one, and there is nothing to compare."""
    served["model"] = _lane(None, 20.0)
    served["obs"] = _lane("mld_dt_mean", 22.0, standard_name=BY_SIGMA_THETA)
    c = Comparison(reference="obs", test="model", variable=MLD)
    assert _align(c) == []
    assert "test_standard_name" not in c.aligned.attrs
    assert c.aligned.attrs["reference_standard_name"] == BY_SIGMA_THETA


# -- .sel(variable=...): the same one-directional rule ---------------------------


@pytest.mark.parametrize(
    ("wanted", "standard_name", "variable", "expected"),
    [
        # a broad name keeps every definition and the generic name itself
        ("mld", BY_SIGMA_THETA, BY_SIGMA_THETA, True),
        ("mld", BY_TEMPERATURE, BY_TEMPERATURE, True),
        ("mld", BY_MIXING_SCHEME, BY_MIXING_SCHEME, True),
        ("mld", MLD, MLD, True),
        ("mixed_layer_depth", BY_SIGMA_T, BY_SIGMA_T, True),  # an alias, same request
        (MLD, BY_SIGMA_THETA, BY_SIGMA_THETA, True),
        # a specific name keeps only its own definition
        ("mld_by_sigma_theta", BY_SIGMA_THETA, BY_SIGMA_THETA, True),
        (BY_SIGMA_THETA, BY_SIGMA_THETA, BY_SIGMA_THETA, True),
        ("mld_by_sigma_theta", MLD, MLD, False),  # the generic has not said which
        ("mld_by_sigma_theta", BY_TEMPERATURE, BY_TEMPERATURE, False),
        ("mld_by_sigma_theta", BY_SIGMA_T, BY_SIGMA_T, False),  # a prefix, not a match
        ("mld_by_sigma_t", BY_SIGMA_THETA, BY_SIGMA_THETA, False),
        # the plain variable spec answers when no standard_name is known
        ("mld", None, BY_SIGMA_THETA, True),
        ("mld_by_sigma_theta", None, BY_SIGMA_THETA, True),
        ("mld_by_sigma_theta", None, BY_SIGMA_T, False),
        # a calculate-spec has no name to resolve through: not matched by a name
        ("mld", None, {"calculate": "mld", "method": "density_threshold"}, False),
        # everything that was not about definitions is unchanged
        ("temp", "sea_water_potential_temperature", "temperature", True),
        ("temperature", None, "sea_water_potential_temperature", True),
        ("nitrate", "sea_water_potential_temperature", "temperature", False),
    ],
)
def test_variable_matches_is_one_directional_for_definitions(
    wanted, standard_name, variable, expected
):
    assert _variable_matches(standard_name, variable, wanted) is expected


def test_variable_matches_still_compares_a_non_string_request_by_equality():
    spec = {"sum": ["a", "b"], "standard_name": MLD}
    assert _variable_matches(MLD, spec, dict(spec)) is True
    assert _variable_matches(MLD, spec, {"sum": ["a", "c"]}) is False


def _members():
    """Build one comparison per definition, plus the generic name and a bystander."""
    return ComparisonSet(
        [
            Comparison(reference="r", test="t", variable=name)
            for name in (
                BY_SIGMA_THETA,
                BY_SIGMA_T,
                BY_TEMPERATURE,
                MLD,
                "sea_water_potential_temperature",
            )
        ]
    )


def test_comparisonset_sel_by_a_broad_name_keeps_every_mixed_layer_depth():
    kept = _members().sel(variable="mld")
    assert [c.variable for c in kept] == [
        BY_SIGMA_THETA,
        BY_SIGMA_T,
        BY_TEMPERATURE,
        MLD,
    ]


def test_comparisonset_sel_by_a_specific_name_keeps_only_that_definition():
    members = _members()
    assert [c.variable for c in members.sel(variable="mld_by_sigma_theta")] == [
        BY_SIGMA_THETA
    ]
    assert [c.variable for c in members.sel(standard_name=BY_SIGMA_T)] == [BY_SIGMA_T]


def test_comparisonset_sel_by_a_specific_name_refuses_the_generic_and_siblings():
    """Nothing in this set is a sigma_theta depth, so the request matches nothing."""
    only_others = ComparisonSet(
        [
            Comparison(reference="r", test="t", variable=name)
            for name in (MLD, BY_TEMPERATURE, BY_SIGMA_T)
        ]
    )
    with pytest.raises(ValueError, match="no comparisons match"):
        only_others.sel(variable="mld_by_sigma_theta")


def test_comparisonset_sel_accepts_a_list_mixing_broad_and_specific_names():
    kept = _members().sel(variable=["mld_by_temperature", "temperature"])
    assert [c.variable for c in kept] == [
        BY_TEMPERATURE,
        "sea_water_potential_temperature",
    ]


def test_fieldset_sel_speaks_the_same_rule():
    """``FieldSet.sel`` shares :func:`_variable_matches`, so it cannot drift."""
    from ocean_skill.field import Field, FieldSet

    fields = FieldSet(
        [Field("source", name) for name in (BY_SIGMA_THETA, BY_TEMPERATURE, MLD)]
    )
    assert len(fields.sel(variable="mld")) == 3
    assert [f.variable for f in fields.sel(variable="mld_by_sigma_theta")] == [
        BY_SIGMA_THETA
    ]
    only_generic = FieldSet([Field("source", MLD), Field("source", BY_TEMPERATURE)])
    with pytest.raises(ValueError, match="no fields match"):
        only_generic.sel(variable="mld_by_sigma_theta")


# -- Comparison.standard_name falls back to the resolved test lane ---------------


def test_standard_name_is_none_until_aligned_for_a_calculated_pair():
    """Unchanged: an un-aligned, unnamed calculate-spec pair still says nothing."""
    c = Comparison(reference="r", test="t", variable=PAIR)
    assert c.standard_name is None


def test_standard_name_falls_back_to_the_resolved_test_lane_once_aligned():
    """A calculated MLD takes its name, and so its colormap, from the aligned pair.

    The calculator names its output ``..._defined_by_sigma_theta`` for
    ``density_threshold`` -- knowable only after it ran -- and without this fallback
    both renderers drew the pair in the anonymous default colormap.
    """
    from ocean_skill.colormaps import cmaps_for

    c = Comparison(reference="r", test="t", variable=PAIR)
    c._aligned = _cached_pair(test_standard_name=BY_SIGMA_THETA)
    assert c.standard_name == BY_SIGMA_THETA
    assert cmaps_for(c.standard_name)[0].name == "deep"
    c._aligned = None
    assert c.standard_name is None


def test_standard_name_falls_back_to_the_aligned_test_arrays_own_attribute():
    """An aligned pair with no recorded name still answers from its ``test`` array."""
    c = Comparison(reference="r", test="t", variable=PAIR)
    c._aligned = _cached_pair(test_attrs={"standard_name": BY_TEMPERATURE})
    assert c.standard_name == BY_TEMPERATURE


def test_the_recorded_name_beats_the_arrays_attribute():
    c = Comparison(reference="r", test="t", variable=PAIR)
    c._aligned = _cached_pair(
        test_attrs={"standard_name": BY_TEMPERATURE},
        test_standard_name=BY_SIGMA_THETA,
    )
    assert c.standard_name == BY_SIGMA_THETA


def test_standard_name_stays_none_when_the_aligned_pair_names_nothing():
    c = Comparison(reference="r", test="t", variable=PAIR)
    c._aligned = _cached_pair()  # arrays literally named "test"/"reference" don't count
    assert c.standard_name is None


def test_an_explicit_pair_spec_standard_name_still_wins_over_the_aligned_pair():
    c = Comparison(reference="r", test="t", variable={**PAIR, "standard_name": MLD})
    c._aligned = _cached_pair(test_standard_name=BY_SIGMA_THETA)
    assert c.standard_name == MLD


def test_a_name_the_spec_gives_wins_over_the_aligned_pair():
    """Only a spec that says nothing falls back; a plain name is the request itself."""
    c = Comparison(reference="r", test="t", variable=MLD)
    c._aligned = _cached_pair(test_standard_name=BY_SIGMA_THETA)
    assert c.standard_name == MLD


def test_standard_name_never_triggers_an_align_or_a_read():
    """It reads the aligned pair already held, never the ``aligned`` property."""
    c = Comparison(reference="r", test="t", variable=PAIR)

    def boom(self, *args, **kwargs):
        raise AssertionError("standard_name must not align")

    with mock.patch.object(Comparison, "align", boom):
        assert c.standard_name is None
        c._aligned = _cached_pair(test_standard_name=BY_SIGMA_THETA)
        assert c.standard_name == BY_SIGMA_THETA


def test_sel_reaches_a_calculated_member_by_the_name_it_resolved_to():
    """An aligned calculate-spec member matches ``.sel(variable="mld")``."""
    calculated = Comparison(reference="r", test="t", variable=PAIR)
    calculated._aligned = _cached_pair(test_standard_name=BY_SIGMA_THETA)
    plain = Comparison(
        reference="r", test="t", variable="sea_water_potential_temperature"
    )
    both = ComparisonSet([calculated, plain])
    assert list(both.sel(variable="mld")) == [calculated]
    assert list(both.sel(variable="mld_by_sigma_theta")) == [calculated]
    with pytest.raises(ValueError, match="no comparisons match"):
        both.sel(variable="mld_by_temperature")
