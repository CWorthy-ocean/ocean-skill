"""Tests for the variable vocabulary: name resolution and live extension.

The vocabulary is what lets a caller name a variable however they like — a short
key (``"oxygen"``), the canonical CF standard_name, or any spelling a real product
happens to use, in any capitalization — and still reach the same variable. These
cover the resolution rules, the two live-extension entry points, and the spots
where getting it wrong would be silent rather than loud (colliding spellings,
clobbered entries, a QC companion standing in for real data).

The one-level ``broader`` relation (a generic name and its definition-specific kinds,
mixed layer depth being the shipped case) gets its own section near the end:
``narrower_names``/``covers``, ``register(broader=...)``, and the vocabulary's own
integrity.
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pytest
import xarray as xr

from ocean_skill import vocabulary
from ocean_skill.units import find_variable

CHL = "mass_concentration_of_chlorophyll_a_in_sea_water"
OXYGEN = "mole_concentration_of_dissolved_molecular_oxygen_in_sea_water"

# Mixed layer depth: one generic CF name, and the definition-specific names each of
# which is one specific kind of it (the vocabulary's ``broader`` relation).
MLD = "ocean_mixed_layer_thickness"
BY_SIGMA_THETA = "ocean_mixed_layer_thickness_defined_by_sigma_theta"
BY_SIGMA_T = "ocean_mixed_layer_thickness_defined_by_sigma_t"
BY_TEMPERATURE = "ocean_mixed_layer_thickness_defined_by_temperature"
BY_MIXING_SCHEME = "ocean_mixed_layer_thickness_defined_by_mixing_scheme"
#: The four definitions in the sorted order ``narrower_names`` promises (note
#: sigma_t sorts before sigma_theta: one is a prefix of the other).
MLD_DEFINITIONS = (BY_MIXING_SCHEME, BY_SIGMA_T, BY_SIGMA_THETA, BY_TEMPERATURE)
#: Each definition's short key -- the name a caller actually types for it.
MLD_KEYS = {
    BY_SIGMA_THETA: "mld_by_sigma_theta",
    BY_SIGMA_T: "mld_by_sigma_t",
    BY_TEMPERATURE: "mld_by_temperature",
    BY_MIXING_SCHEME: "mld_by_mixing_scheme",
}


@pytest.fixture
def pristine_vocabulary():
    """Restore VOCABULARY after a test mutates it.

    ``register``/``add_alias``/``add_pattern`` mutate module state that would
    otherwise leak into every later test in the session (and into cf-xarray's
    global registration). Each entry's ``aliases``/``patterns`` lists are copied
    too, not just the entry dicts -- ``add_alias``/``add_pattern`` extend an
    *existing* concept's list in place (``setdefault(...).append(...)``), so a
    shallow ``dict(v)`` copy would still share that list object with the "restored"
    snapshot and the mutation would survive teardown.
    """
    saved = {
        k: {
            field: (list(value) if isinstance(value, list) else value)
            for field, value in v.items()
        }
        for k, v in vocabulary.VOCABULARY.items()
    }
    yield vocabulary.VOCABULARY
    vocabulary.VOCABULARY.clear()
    vocabulary.VOCABULARY.update(saved)
    vocabulary._refresh()


def _tiny(varname: str) -> xr.Dataset:
    """Build a 2x2 dataset carrying exactly one variable, named ``varname``."""
    return xr.Dataset(
        {varname: (("lat", "lon"), np.ones((2, 2)))},
        coords={"lat": [10.0, 11.0], "lon": [200.0, 201.0]},
    )


# -- resolution ---------------------------------------------------------------


@pytest.mark.parametrize(
    "spelling",
    [
        "oxygen",  # short key
        OXYGEN,  # canonical standard_name
        "moles_of_oxygen_per_unit_mass_in_sea_water",  # WOA/GLODAP's per-mass name
    ],
)
def test_every_spelling_resolves_to_one_canonical_name(spelling):
    assert vocabulary.resolve_name(spelling) == OXYGEN


@pytest.mark.parametrize(
    "spelling", ["Chlorophyll", "CHLOROPHYLL", "chlorophyll", "ChLoRoPhYlL"]
)
def test_resolution_ignores_case(spelling):
    """Products disagree on capitalization; case never means a different variable."""
    assert vocabulary.resolve_name(spelling) == CHL


def test_unknown_name_keeps_its_own_case():
    """Pass-through must not silently lowercase a name it doesn't recognize."""
    assert vocabulary.resolve_name("Some_Unknown_Var") == "Some_Unknown_Var"


def test_unknown_name_passes_through_unchanged():
    """Most CF names need no vocabulary entry; they must not be mangled."""
    assert vocabulary.resolve_name("sea_floor_depth_below_geoid") == (
        "sea_floor_depth_below_geoid"
    )


def test_resolve_and_report_warns_only_when_the_name_changes():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert vocabulary.resolve_and_report(OXYGEN) == OXYGEN
    assert not caught, "already-canonical name should resolve silently"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert vocabulary.resolve_and_report("oxygen") == OXYGEN
    assert len(caught) == 1
    assert "oxygen" in str(caught[0].message) and OXYGEN in str(caught[0].message)


def test_equivalent_names_spans_the_whole_concept():
    """compare()'s catalog filter relies on this covering every declared spelling."""
    names = vocabulary.equivalent_names("oxygen")
    assert OXYGEN in names
    assert "moles_of_oxygen_per_unit_mass_in_sea_water" in names
    # and an unknown name is its own only equivalent, not an empty set
    assert vocabulary.equivalent_names("not_a_variable") == {"not_a_variable"}


# -- finding the variable in a real dataset -----------------------------------


@pytest.mark.parametrize(
    "stored_as",
    [
        CHL,  # canonical
        "mass_concentration_of_chlorophyll_in_sea_water",  # MODIS catalog's spelling
        "mass_concentration_of_chlorophyll_a_in_sea_water_profiler_depth_enabled",
    ],
)
def test_one_short_key_finds_every_registered_spelling(stored_as):
    """Datasets disagree on chlorophyll's CF name; "chlorophyll" must find them all.

    The MODIS spelling here is a real mismatch found in ``ocean_skill/catalogs/modis_aqua.yaml``
    (it drops the ``_a_``), not a hypothetical — before it was registered,
    ``find_variable(modis_ds, "chlorophyll")`` returned ``None``.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the "resolved to ..." notice; see below
        da = find_variable(_tiny(stored_as), "chlorophyll")
    assert da is not None
    assert da.name == stored_as


def test_find_variable_reports_which_spelling_it_actually_found():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        find_variable(
            _tiny("mass_concentration_of_chlorophyll_in_sea_water"), "chlorophyll"
        )
    assert len(caught) == 1
    msg = str(caught[0].message)
    assert "chlorophyll" in msg
    assert "mass_concentration_of_chlorophyll_in_sea_water" in msg


def test_exact_hit_is_silent():
    """The common case (already CF-renamed) must not warn about anything."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        da = find_variable(_tiny(OXYGEN), OXYGEN)
    assert da is not None and not caught


def test_missing_variable_returns_none():
    assert find_variable(_tiny("something_else"), "oxygen") is None


# -- the short key itself must be dataset-matchable, not just resolver-known --


def _standard_name_and_aliases(key: str) -> set[str]:
    """Every literal spelling the shipped vocabulary itself recognizes for ``key``.

    ``key`` plus everything :func:`~ocean_skill.vocabulary.equivalent_names` reports
    for it -- the same literal set :func:`ocean_skill.comparison.compare`'s catalog
    pre-filter (``_offers``) accepts as "the same variable" a declared column
    resolves to, so this is the set the next test uses to check that anything
    accepted there is also something ``find_variable`` can actually find.
    """
    return {key} | vocabulary.equivalent_names(key)


def test_raw_ctd_column_named_like_the_key_is_found():
    """Regression for the real failure: a raw tabular column named just "Temperature".

    A tabular source's ``standard_name`` attribute is the raw column name itself
    (see :func:`ocean_skill.tabular.to_dataset`), so an attribute-based lookup
    can't rescue this either -- the fix has to be name-based. Before it,
    ``describe()`` reported this column matched (``temperature <- Temperature``)
    while ``find_variable`` returned ``None``.
    """
    ds = _tiny("Temperature")
    ds["Temperature"].attrs.update(standard_name="Temperature", units="degree_C")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the "resolved to ..." notice; see above
        da = find_variable(ds, "temperature")
    assert da is not None
    assert da.name == "Temperature"


@pytest.mark.parametrize("key", sorted(vocabulary.VOCABULARY))
def test_every_short_key_is_findable_as_a_dataset_variable(key):
    """The resolver and cf-xarray must agree on every key, not just temperature's.

    ``_build_index`` (what ``resolve_name``/``describe()`` use) and
    ``_register_custom_criteria`` (what ``find_variable`` uses) used to be built
    from two different lists -- the index included each entry's short key, the
    cf-xarray registration didn't -- so a raw column spelled like the key alone
    (``Oxygen``, ``Pressure``, ...) resolved but could not actually be found.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert find_variable(_tiny(key), key) is not None
        assert find_variable(_tiny(key.capitalize()), key) is not None


@pytest.mark.parametrize("key", sorted(vocabulary.VOCABULARY))
def test_declared_name_acceptance_implies_dataset_side_match(key):
    """Every spelling compare()'s catalog pre-filter accepts must be findable too.

    Closes the whole class, not just the key: a spelling that ``_offers`` would
    call available for a declared column but ``find_variable`` cannot find is
    exactly the "catalog says yes, data says no" bug this fix targets.
    """
    for spelling in _standard_name_and_aliases(key):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert find_variable(_tiny(spelling), key) is not None, spelling


@pytest.mark.parametrize("key", sorted(vocabulary.VOCABULARY))
def test_a_key_named_qc_companion_is_still_ignored(key):
    """Registering the bare key must not loosen the QC-flag exclusion (see above)."""
    assert find_variable(_tiny(f"{key}_qc_agg"), key) is None


def test_register_makes_the_new_key_findable_dataset_side(pristine_vocabulary):
    """A live-registered concept's own key is dataset-matchable immediately too.

    Pins the ``register`` -> ``_refresh`` -> ``_register_custom_criteria`` path,
    the same one the shipped vocabulary now goes through for every key.
    """
    vocabulary.register("my_conc", "standard_x")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert find_variable(_tiny("my_conc"), "my_conc") is not None


def test_near_identical_quantities_resolve_as_plain_aliases():
    """In-situ temperature reaches the "temperature" concept like any other alias.

    These are near-identical rather than identical quantities; they are deliberately
    plain aliases (see the vocabulary module docstring) rather than a separate tier.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        da = find_variable(_tiny("sea_water_temperature"), "temperature")
    assert da is not None and da.name == "sea_water_temperature"
    assert len(caught) == 1  # the usual "resolved to ..." notice, nothing extra


@pytest.mark.parametrize("spelling", ["Fe", "fe", "FE"])
def test_short_symbol_alias_resolves_whole_name_any_case(spelling):
    """ROMS/MARBL's `Fe` reaches iron whatever the case, but only as the whole name."""
    assert vocabulary.resolve_name(spelling) == (
        "mole_concentration_of_dissolved_iron_in_sea_water"
    )


# The ROMS/MARBL tracer short names that must resolve as a typed nickname, not just
# be renamed at build time -- a caller types `NO3`/`O2`/`DIC`/... as readily as the
# long CF name. Kept in sync with build.ROMS_STANDARD_NAMES below; the ones left out
# (zeta/u/v/hbls/FG_CO2) reach the same concept through their friendly keys instead.
_TYPEABLE_TRACERS = [
    "temp", "salt", "w", "NO3", "PO4", "SiO3", "NH4", "Fe", "O2", "DIC", "ALK",
]


@pytest.mark.parametrize("tracer", _TYPEABLE_TRACERS)
def test_model_tracer_name_resolves_as_a_typed_nickname(tracer):
    """Typing the model's own tracer name reaches the same variable a build produces.

    Regression guard for the asymmetry where `NH4`/`Fe` resolved but `NO3`/`O2`/...
    silently passed through: a build-time rename is not enough, resolve_name (used
    everywhere a caller supplies a name) must reach it too.
    """
    from ocean_skill.build import ROMS_STANDARD_NAMES

    assert vocabulary.resolve_name(tracer) == ROMS_STANDARD_NAMES[tracer]


def test_chl_shorthand_resolves_but_does_not_grab_per_pft_tracers():
    """`Chl` is a shorthand for the concept, not a tracer -- spChl/... stay themselves."""
    assert vocabulary.resolve_name("Chl") == (
        "mass_concentration_of_chlorophyll_a_in_sea_water"
    )
    for per_pft in ("spChl", "diatChl", "diazChl"):
        assert vocabulary.resolve_name(per_pft) == per_pft


@pytest.mark.parametrize("not_iron", ["felix", "ferric", "Fe_flux"])
def test_short_symbol_alias_is_not_a_prefix_match(not_iron):
    """A name merely starting with the symbol is a different variable, not iron."""
    assert vocabulary.resolve_name(not_iron) == not_iron
    assert not vocabulary.is_known(not_iron)
    assert find_variable(_tiny(not_iron), "iron") is None


@pytest.mark.parametrize(
    "stored_as",
    [
        "MASS_CONCENTRATION_OF_CHLOROPHYLL_A_IN_SEA_WATER",
        "Mass_Concentration_Of_Chlorophyll_A_In_Sea_Water",
    ],
)
def test_dataset_variable_names_match_regardless_of_case(stored_as):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        da = find_variable(_tiny(stored_as), "chlorophyll")
    assert da is not None and da.name == stored_as


def test_non_vocabulary_name_also_matches_regardless_of_case():
    """Most CF names have no vocabulary entry, so cf-xarray never sees them."""
    da = find_variable(
        _tiny("Sea_Floor_Depth_Below_Geoid"), "sea_floor_depth_below_geoid"
    )
    assert da is not None and da.name == "Sea_Floor_Depth_Below_Geoid"


def test_exact_hit_wins_over_a_case_variant():
    """An exact name is never ambiguous, however the dataset spells its neighbours."""
    ds = _tiny(OXYGEN)
    ds[OXYGEN.upper()] = ds[OXYGEN]
    assert find_variable(ds, OXYGEN).name == OXYGEN


def test_variables_differing_only_by_case_are_rejected_not_guessed():
    """With no exact hit, two case variants are a coin flip — refuse, don't pick."""
    ds = _tiny("Some_Var")
    ds["SOME_VAR"] = ds["Some_Var"]
    with pytest.raises(ValueError, match="only by case"):
        find_variable(ds, "some_var")


def test_qc_companion_is_not_mistaken_for_the_data_variable():
    """cf-xarray matches with re.match, which anchors only the *start* of the name.

    Without an explicit ``$`` the registered pattern also matched anything merely
    prefixed by a real spelling, so an ERDDAP/OOI QC-flag column came back as if it
    were the data — silently, and only when the real variable was absent.
    """
    qc_only = _tiny("mole_concentration_of_nitrate_in_sea_water_qc_agg")
    assert find_variable(qc_only, "nitrate") is None


def test_real_variable_wins_over_its_qc_companion():
    both = _tiny("mole_concentration_of_nitrate_in_sea_water")
    both["mole_concentration_of_nitrate_in_sea_water_qc_agg"] = both[
        "mole_concentration_of_nitrate_in_sea_water"
    ]
    nitrate = "mole_concentration_of_nitrate_in_sea_water"
    assert find_variable(both, "nitrate").name == nitrate


def test_warning_blames_the_callers_own_code_not_ocean_skill_internals():
    """A fixed stacklevel named an internal line, useless for locating the call.

    The depth from ``find_variable`` out to the user varies (``compare`` reaches it
    four frames down; a direct call, one), so the frame is counted rather than
    hard-coded — see :mod:`ocean_skill._stacklevel`.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        find_variable(_tiny("moles_of_oxygen_per_unit_mass_in_sea_water"), "oxygen")
    assert caught[0].filename == __file__, (
        f"warning blamed {caught[0].filename}, not the calling test file"
    )


def test_warning_survives_extra_internal_frames():
    """Reaching find_variable through more ocean-skill frames must not shift blame."""
    from ocean_skill.comparison import _prepare

    ds = _tiny("moles_of_oxygen_per_unit_mass_in_sea_water")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _prepare(ds, {}, "oxygen", {})  # non-ROMS branch: one extra frame
    assert caught, "expected a resolution notice"
    assert caught[0].filename == __file__


def test_find_variable_keeps_coordinates():
    """cf-xarray's own accessor drops non-dimension coords; downstream needs them."""
    ds = _tiny(OXYGEN)
    ds = ds.assign_coords(mask=(("lat", "lon"), np.ones((2, 2))))
    da = find_variable(ds, OXYGEN)
    assert "mask" in da.coords


# -- regex pattern recognition --------------------------------------------------


@pytest.mark.parametrize(
    "spelling,expected",
    [
        ("Temperature_CTD", "sea_water_potential_temperature"),
        ("temp_ctd", "sea_water_potential_temperature"),
        ("CTD_Temperature", "sea_water_potential_temperature"),
        ("PSAL", "sea_water_practical_salinity"),
        ("sal_psu", "sea_water_practical_salinity"),
        ("DOXY", OXYGEN),
        ("chl_a", CHL),
        ("CHLA", CHL),
    ],
)
def test_pattern_spelling_resolves_to_the_canonical_name(spelling, expected):
    assert vocabulary.resolve_name(spelling) == expected


@pytest.mark.parametrize(
    "not_a_match",
    ["my_temp_ctd", "temp_ctd_2", "psalm", "salt_flux", "chlamydomonas"],
)
def test_pattern_matching_is_fullmatch_not_substring(not_a_match):
    """A pattern recognizes the whole name, never a name merely containing it."""
    assert vocabulary.resolve_name(not_a_match) == not_a_match
    assert not vocabulary.is_known(not_a_match)


@pytest.mark.parametrize("spelling", ["Temperature_CTD", "PSAL", "DOXY", "chl_a"])
def test_pattern_spellings_count_as_known(spelling):
    """is_known must track resolve_name's two tiers.

    Otherwise compare()'s absent-vs-unknowable check misjudges a pattern-recognized
    name as genuinely absent.
    """
    assert vocabulary.is_known(spelling)


@pytest.mark.parametrize(
    "flagged", ["Temperature_CTD_flag", "sal_psu_qc_agg", "Temperature_CTD_qc_agg"]
)
def test_pattern_never_claims_a_flag_decorated_name(flagged):
    assert vocabulary.resolve_name(flagged) == flagged
    assert not vocabulary.is_known(flagged)


def test_pattern_never_claims_a_flag_decorated_dataset_variable():
    assert find_variable(_tiny("Temperature_CTD_qc_agg"), "temperature") is None


def test_pattern_match_finds_the_dataset_variant_column():
    """Proves the pattern reached cf-xarray's registration, not just resolve_name."""
    ds = _tiny("Temperature_CTD")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        da = find_variable(ds, "temperature")
    assert da is not None and da.name == "Temperature_CTD"


def test_ambiguous_pattern_match_refuses_to_guess_and_warns(pristine_vocabulary):
    """Two entries whose patterns both claim a name is a vocabulary bug, not a guess."""
    vocabulary.register("concept_a", "standard_a", patterns=["shared_[0-9]"])
    vocabulary.register("concept_b", "standard_b", patterns=["shared_[0-9]"])
    with pytest.warns(UserWarning, match="matches vocabulary patterns"):
        resolved = vocabulary.resolve_name("shared_1")
    assert resolved == "shared_1"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert not vocabulary.is_known("shared_1")


# -- Iceland CTD-profiles / discrete-sample spellings --------------------------


@pytest.mark.parametrize(
    "spelling,expected",
    [
        ("CTDPRES", "sea_water_pressure"),
        ("Oxygen_CTD", OXYGEN),
        ("TA", "sea_water_alkalinity_expressed_as_mole_equivalent"),
        ("pH_T_measured", "sea_water_ph_reported_on_total_scale"),
        ("PAR_CTD", "downwelling_photosynthetic_photon_flux_in_sea_water"),
        ("PAR", "downwelling_photosynthetic_photon_flux_in_sea_water"),
        ("Turbidity_CTD", "sea_water_turbidity"),
        ("Fluor_CTD", "sea_water_chlorophyll_fluorescence"),
        (
            "Nitrate_and_Nitrite",
            "mole_concentration_of_nitrate_and_nitrite_in_sea_water",
        ),
        ("phaeo", "mass_concentration_of_phaeopigments_in_sea_water"),
        ("Ciliate", "number_concentration_of_ciliates_in_sea_water"),
        ("Diatom", "number_concentration_of_diatoms_in_sea_water"),
        ("Dinoflagellate", "number_concentration_of_dinoflagellates_in_sea_water"),
    ],
)
def test_iceland_spelling_resolves_to_the_canonical_name(spelling, expected):
    assert vocabulary.resolve_name(spelling) == expected
    assert vocabulary.is_known(spelling)


@pytest.mark.parametrize("unmatched", ["TEMP_PH", "Total"])
def test_iceland_ambiguous_column_is_deliberately_left_unmatched(unmatched):
    """TEMP_PH (a measurement-condition temperature) and Total (too ambiguous to
    map safely) are deliberately not given a vocabulary entry -- they should pass
    through unresolved rather than being silently folded into an unrelated concept.
    """
    assert vocabulary.resolve_name(unmatched) == unmatched
    assert not vocabulary.is_known(unmatched)


def test_diatom_alias_does_not_grab_the_romsmarbl_per_pft_tracer():
    """`Diatom` resolves, but ROMS/MARBL's per-PFT `diatChl` tracer -- a different,
    un-summed quantity (see the "chlorophyll" entry) -- must not.
    """
    assert vocabulary.resolve_name("diatChl") == "diatChl"
    assert not vocabulary.is_known("diatChl")


def test_chlor_a_is_still_the_live_extension_example(pristine_vocabulary):
    """`chlor_a` is deliberately not a shipped pattern.

    It stays the documented example of extending the vocabulary live via
    add_alias (see the module docstring's "patterns" bullet and
    examples/vocabulary_demo.py).
    """
    assert not vocabulary.is_known("chlor_a")
    vocabulary.add_alias("chlorophyll", "chlor_a")
    assert vocabulary.is_known("chlor_a")


# -- nickname() and match_report() ---------------------------------------------


@pytest.mark.parametrize(
    "spelling,expected",
    [
        ("oxygen", "oxygen"),  # the key itself
        (OXYGEN, "oxygen"),  # canonical standard_name
        ("O2", "oxygen"),  # alias
        ("DOXY", "oxygen"),  # pattern spelling
        ("Fe", "iron"),
    ],
)
def test_nickname_reverses_resolve_name_to_the_short_key(spelling, expected):
    assert vocabulary.nickname(spelling) == expected


def test_nickname_is_none_for_an_unknown_name():
    assert vocabulary.nickname("Instrument_Type") is None


def test_nickname_is_none_for_an_ambiguous_pattern_match(pristine_vocabulary):
    vocabulary.register("concept_a", "standard_a", patterns=["shared_[0-9]"])
    vocabulary.register("concept_b", "standard_b", patterns=["shared_[0-9]"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert vocabulary.nickname("shared_1") is None


def test_match_report_groups_declared_names_by_nickname():
    report = vocabulary.match_report(["PSAL", "DOXY", "Instrument_Type"])
    assert report.matched == {"salinity": ["PSAL"], "oxygen": ["DOXY"]}
    assert report.unmatched == ["Instrument_Type"]


def test_match_report_on_no_variables_is_empty():
    report = vocabulary.match_report([])
    assert report.matched == {}
    assert report.unmatched == []


def test_match_report_collisions_flags_a_nickname_claimed_twice():
    report = vocabulary.match_report(["Temperature", "Temperature_CTD", "PSAL"])
    assert report.collisions == {
        "temperature": ["Temperature", "Temperature_CTD"]
    }
    assert "salinity" not in report.collisions


def test_match_report_suppresses_the_ambiguous_pattern_warning(pristine_vocabulary):
    """The report is where an ambiguous match surfaces, not a repeated warning.

    It shows up as unmatched instead of firing a warning every time someone runs
    a report over it.
    """
    vocabulary.register("concept_a", "standard_a", patterns=["shared_[0-9]"])
    vocabulary.register("concept_b", "standard_b", patterns=["shared_[0-9]"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = vocabulary.match_report(["shared_1"])
    assert not caught
    assert report.unmatched == ["shared_1"]


def test_match_report_str_names_the_nicknames_and_unmatched_variables():
    text = str(vocabulary.match_report(["PSAL", "Instrument_Type"]))
    assert "salinity" in text and "PSAL" in text
    assert "Instrument_Type" in text
    assert "unmatched" in text


def test_match_report_repr_html_escapes_and_wraps():
    html = vocabulary.match_report(["PSAL"])._repr_html_()
    assert html.startswith("<pre")
    assert "PSAL" in html


def test_match_report_html_escapes_angle_brackets_in_a_variable_name():
    html = vocabulary.match_report(["<script>"])._repr_html_()
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


# -- live extension -----------------------------------------------------------


def test_add_alias_takes_effect_immediately(pristine_vocabulary):
    """A new spelling must reach cf-xarray's registration, not just resolve_name."""
    ds = _tiny("chlor_a")
    assert find_variable(ds, "chlorophyll") is None

    vocabulary.add_alias("chlorophyll", "chlor_a")

    assert vocabulary.resolve_name("chlor_a") == CHL
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert find_variable(ds, "chlorophyll").name == "chlor_a"


def test_add_alias_rejects_an_unknown_concept(pristine_vocabulary):
    with pytest.raises(KeyError, match="register"):
        vocabulary.add_alias("not_a_concept", "whatever")


def test_add_alias_is_idempotent(pristine_vocabulary):
    vocabulary.add_alias("chlorophyll", "chlor_a")
    vocabulary.add_alias("chlorophyll", "chlor_a")
    assert vocabulary.VOCABULARY["chlorophyll"]["aliases"].count("chlor_a") == 1


def test_register_adds_a_new_concept(pristine_vocabulary):
    vocabulary.register(
        "ph", "sea_water_ph_reported_on_total_scale", aliases=["PH_TOT"]
    )
    assert vocabulary.resolve_name("ph") == "sea_water_ph_reported_on_total_scale"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert find_variable(_tiny("PH_TOT"), "ph").name == "PH_TOT"


def test_register_warns_before_clobbering_an_existing_concept(pristine_vocabulary):
    """Silently replacing an entry (dropping its aliases) is a typo waiting to bite."""
    with pytest.warns(UserWarning, match="replaces an existing entry"):
        vocabulary.register("nitrate", "some_other_standard_name")


def test_colliding_spellings_warn_rather_than_silently_picking_one(pristine_vocabulary):
    """Two concepts claiming one spelling would otherwise resolve by dict order."""
    vocabulary.register("concept_a", "standard_a", aliases=["shared"])
    with pytest.warns(UserWarning, match="vocabulary collision"):
        vocabulary.register("concept_b", "standard_b", aliases=["shared"])


def test_add_pattern_takes_effect_immediately(pristine_vocabulary):
    """A new pattern must reach cf-xarray's registration, not just resolve_name."""
    ds = _tiny("OXY_UMOLKG")
    assert find_variable(ds, "oxygen") is None

    vocabulary.add_pattern("oxygen", "oxy_umolkg")

    assert vocabulary.resolve_name("OXY_UMOLKG") == OXYGEN
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert find_variable(ds, "oxygen").name == "OXY_UMOLKG"


def test_add_pattern_rejects_an_unknown_concept(pristine_vocabulary):
    with pytest.raises(KeyError, match="register"):
        vocabulary.add_pattern("not_a_concept", "whatever")


def test_add_pattern_is_idempotent(pristine_vocabulary):
    vocabulary.add_pattern("oxygen", "oxy_umolkg")
    vocabulary.add_pattern("oxygen", "oxy_umolkg")
    assert vocabulary.VOCABULARY["oxygen"]["patterns"].count("oxy_umolkg") == 1


def test_register_accepts_patterns_for_a_new_concept(pristine_vocabulary):
    vocabulary.register(
        "ph", "sea_water_ph_reported_on_total_scale", patterns=["ph_tot(?:al)?"]
    )
    assert vocabulary.resolve_name("ph_total") == "sea_water_ph_reported_on_total_scale"


def test_an_invalid_pattern_in_register_is_rejected_before_adding_the_concept(
    pristine_vocabulary,
):
    # "ph" is now a real concept (see VOCABULARY), so use a name that genuinely does
    # not exist -- the point is that a rejected register leaves the concept unadded.
    assert "nonexistent_test_concept" not in vocabulary.VOCABULARY
    with pytest.raises(re.error):
        vocabulary.register(
            "nonexistent_test_concept",
            "sea_water_ph_reported_on_total_scale",
            patterns=["("],
        )
    assert "nonexistent_test_concept" not in vocabulary.VOCABULARY


def test_an_invalid_pattern_in_add_pattern_is_rejected_before_mutating_the_entry(
    pristine_vocabulary,
):
    before = list(vocabulary.VOCABULARY["oxygen"].get("patterns", []))
    with pytest.raises(re.error):
        vocabulary.add_pattern("oxygen", "(")
    assert vocabulary.VOCABULARY["oxygen"].get("patterns", []) == before
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        vocabulary._refresh()  # still clean -- nothing was left half-added


def test_shipped_vocabulary_has_no_collisions():
    """The vocabulary as shipped must be unambiguous — this is the regression guard."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        vocabulary._refresh()


def test_every_total_alkalinity_spelling_is_one_variable():
    """OceanSODA, GLODAP and ROMS/MARBL all carry *total* alkalinity.

    CF defines the canonical name as "the total alkalinity equivalent concentration",
    and the per-mass form as the same quantity per unit mass, so these differ only in
    basis — which units.py converts. Before this, find(variable="alkalinity") returned
    GLODAP and ROMS but silently dropped OceanSODA's `talk`.
    """
    canonical = "sea_water_alkalinity_expressed_as_mole_equivalent"
    for spelling in (
        "total_alkalinity_in_sea_water",  # OceanSODA-ETHZ; not a CF name
        # CF's own per-mass form
        "sea_water_alkalinity_per_unit_mass_expressed_as_mole_equivalent",
        "seawater_alkalinity_per_unit_mass_expressed_as_mole_equivalent",
        "TOTAL_ALKALINITY_IN_SEA_WATER",  # matching ignores case
        "TAlk",  # GLODAP's own raw variable name
        "talk",  # ... any case
    ):
        assert vocabulary.is_known(spelling), spelling
        assert vocabulary.resolve_name(spelling) == canonical, spelling


def test_every_dissolved_inorganic_carbon_spelling_is_one_variable():
    """GLODAP's `TCO2` and ROMS/MARBL's `DIC` are the same quantity, by CF name.

    Before this, build-time probing needed a per-dataset name_map to recognize
    GLODAP's `TCO2` at all -- the vocabulary alone now covers it, the way it already
    covers ROMS/MARBL's `DIC`.
    """
    canonical = "mole_concentration_of_dissolved_inorganic_carbon_in_sea_water"
    for spelling in (
        "DIC",  # ROMS/MARBL tracer name
        "TCO2",  # GLODAP's own raw variable name
        "tco2",  # ... any case
        "moles_of_dissolved_inorganic_carbon_per_unit_mass_in_sea_water",
    ):
        assert vocabulary.is_known(spelling), spelling
        assert vocabulary.resolve_name(spelling) == canonical, spelling


def test_alkalinity_variants_that_are_different_quantities_stay_separate():
    """Preformed and natural-analogue alkalinity are their own CF names, not aliases."""
    for other in (
        "sea_water_preformed_alkalinity_expressed_as_mole_equivalent",
        "sea_water_alkalinity_natural_analogue_expressed_as_mole_equivalent",
    ):
        assert not vocabulary.is_known(other), f"{other} is a distinct quantity"


def test_east_and_x_velocity_are_typeable_nicknames():
    """The short keys a caller types resolve to their own, now-separate standard_names.

    ``east_velocity``/``north_velocity`` (true geographic velocity) and
    ``x_velocity``/``y_velocity`` (ROMS' own grid-relative components) used to share
    one standard_name apiece -- see the next test.
    """
    assert vocabulary.resolve_name("east_velocity") == "eastward_sea_water_velocity"
    assert vocabulary.resolve_name("north_velocity") == "northward_sea_water_velocity"
    assert vocabulary.resolve_name("x_velocity") == "sea_water_x_velocity"
    assert vocabulary.resolve_name("y_velocity") == "sea_water_y_velocity"
    # the pre-split keys still resolve, for a caller who already typed them
    assert vocabulary.resolve_name("eastward_velocity") == "eastward_sea_water_velocity"
    assert (
        vocabulary.resolve_name("northward_velocity") == "northward_sea_water_velocity"
    )


def test_geographic_and_grid_relative_velocity_are_no_longer_the_same_quantity():
    """The intended semantic change: ROMS' `u` is not geographic east on a rotated grid.

    Before the "east_velocity"/"x_velocity" split, both resolved to
    ``sea_water_x_velocity`` and this was True -- which let compare() silently treat
    ROMS' grid-relative x-velocity as if it were an ADCP's own eastward reading.
    """
    assert not vocabulary.same_quantity(
        "sea_water_x_velocity", "eastward_sea_water_velocity"
    )
    assert not vocabulary.same_quantity(
        "sea_water_y_velocity", "northward_sea_water_velocity"
    )


def test_geostrophic_velocity_aliases_true_geographic_velocity_not_grid_relative():
    """DUACS/MULTIOBS geostrophic current is real east/north, not a model grid's x/y."""
    for spelling in (
        "surface_geostrophic_eastward_sea_water_velocity",
        "surface_geostrophic_eastward_sea_water_velocity_assuming_sea_level_for_geoid",
    ):
        assert vocabulary.resolve_name(spelling) == "eastward_sea_water_velocity"
    for spelling in (
        "surface_geostrophic_northward_sea_water_velocity",
        "surface_geostrophic_northward_sea_water_velocity_assuming_sea_level_for_geoid",
    ):
        assert vocabulary.resolve_name(spelling) == "northward_sea_water_velocity"


def test_total_current_wins_over_geostrophic_when_a_dataset_carries_both():
    """Copernicus/DUACS's own shape: total (`uo`) and geostrophic (`ugos`) current.

    Both are renamed at build time (catalogs/copernicus.yaml's ``standard_names`` map)
    to their own literal CF names -- ``eastward_sea_water_velocity`` and
    ``surface_geostrophic_eastward_sea_water_velocity`` respectively. Both are now
    aliases of the same "east_velocity" concept (they were before this split too), but
    ``find_variable``'s literal-name-first check (see its own docstring) matches the
    exact standard_name outright before cf-xarray's alias search ever runs, so asking
    for east_velocity on a dataset carrying both correctly returns the *total*
    current, not an ambiguity error -- unaffected by this fix.
    """
    ds = xr.Dataset(
        {
            "eastward_sea_water_velocity": (("y", "x"), np.zeros((2, 2))),
            "surface_geostrophic_eastward_sea_water_velocity": (
                ("y", "x"),
                np.ones((2, 2)),
            ),
        }
    )
    da = find_variable(ds, "east_velocity")
    assert da.name == "eastward_sea_water_velocity"
    assert bool((da == 0).all())


# -- broader: one-level broad -> specific definitions ---------------------------
#
# A request for "the MLD" should find a source carrying any one definition of it,
# while a request for one definition must never be answered by the generic name or
# by a sibling definition. These pin the relation itself (narrower_names/covers), its
# live extension (register(broader=)), and the vocabulary as shipped;
# ``find_variable``'s use of it is covered in tests/test_find_variable_mld.py.


def test_narrower_names_of_the_generic_mld_are_its_four_definitions_sorted():
    assert vocabulary.narrower_names("mld") == MLD_DEFINITIONS
    assert list(MLD_DEFINITIONS) == sorted(MLD_DEFINITIONS)


@pytest.mark.parametrize(
    "spelling",
    [
        "mld",
        "MLD",
        "Mld",
        "mixed_layer_depth",
        "Mixed_Layer_Thickness",
        MLD,
        MLD.upper(),
    ],
)
def test_narrower_names_takes_the_broad_name_however_it_is_spelled(spelling):
    assert vocabulary.narrower_names(spelling) == MLD_DEFINITIONS


@pytest.mark.parametrize(
    "name",
    [
        "mld_by_sigma_theta",  # a specific definition's key ...
        BY_SIGMA_THETA,  # ... and its standard_name: nothing is narrower than one
        BY_MIXING_SCHEME,
        "temperature",  # a concept nothing is a specific kind of
        "sea_water_potential_temperature",
        "not_a_variable",  # a name the vocabulary has never heard of
    ],
)
def test_narrower_names_is_empty_for_a_specific_childless_or_unknown_name(name):
    assert vocabulary.narrower_names(name) == ()


def test_the_generic_name_still_resolves_to_the_generic_standard_name():
    """``broader`` is a relation *between* entries; it never changes what resolves."""
    for spelling in ("mld", "mixed_layer_depth", "MLD", MLD):
        assert vocabulary.resolve_name(spelling) == MLD


@pytest.mark.parametrize("definition,key", sorted(MLD_KEYS.items()))
def test_each_definition_has_its_own_typeable_key(definition, key):
    assert vocabulary.is_known(key)
    assert vocabulary.resolve_name(key) == definition
    assert vocabulary.nickname(definition) == key


@pytest.mark.parametrize("raw", ["mld_dt_mean", "mlotst", "hbls"])
def test_raw_mld_product_names_are_deliberately_not_global_aliases(raw):
    """``mlotst`` is sigma_theta in Copernicus but sigma_t in CMIP's own naming.

    A raw product name is renamed to its definition-specific standard_name in its own
    catalog, where the definition is known; a global alias would decide it for every
    catalog at once.
    """
    assert not vocabulary.is_known(raw)
    assert vocabulary.resolve_name(raw) == raw


@pytest.mark.parametrize("definition", MLD_DEFINITIONS)
def test_covers_the_generic_request_is_satisfied_by_every_definition(definition):
    assert vocabulary.covers("mld", definition)
    assert vocabulary.covers(MLD, definition)


@pytest.mark.parametrize(
    "requested,declared",
    [
        ("MLD", BY_SIGMA_THETA.upper()),  # capitalization, on both sides
        ("mixed_layer_depth", BY_SIGMA_THETA),  # an alias of the broad name
        ("mixed_layer_thickness", "mld_by_temperature"),  # a definition's own key
        ("Mld", "MLD_BY_MIXING_SCHEME"),
        (MLD, "Mld_By_Sigma_T"),
    ],
)
def test_covers_reaches_definitions_through_aliases_keys_and_case(requested, declared):
    assert vocabulary.covers(requested, declared)


@pytest.mark.parametrize("definition", MLD_DEFINITIONS)
def test_covers_a_specific_request_is_not_met_by_the_generic_name(definition):
    """One direction only: asking for a definition never returns the generic name."""
    for generic in ("mld", "mixed_layer_depth", MLD):
        assert not vocabulary.covers(definition, generic)


@pytest.mark.parametrize("a", MLD_DEFINITIONS)
@pytest.mark.parametrize("b", MLD_DEFINITIONS)
def test_covers_never_bridges_two_sibling_definitions(a, b):
    assert vocabulary.covers(a, b) is (a == b)
    assert vocabulary.covers(MLD_KEYS[a], MLD_KEYS[b]) is (a == b)


def test_sigma_t_and_sigma_theta_do_not_cover_each_other_by_prefix():
    """One name is a prefix of the other; names are compared whole, never by prefix."""
    assert BY_SIGMA_THETA.startswith(BY_SIGMA_T)
    assert not vocabulary.covers(BY_SIGMA_T, BY_SIGMA_THETA)
    assert not vocabulary.covers(BY_SIGMA_THETA, BY_SIGMA_T)


@pytest.mark.parametrize(
    "requested,declared",
    [
        ("mld", "temperature"),
        ("temperature", "mld"),
        ("mld", "not_a_variable"),
        ("not_a_variable", "mld"),
        ("temperature", BY_SIGMA_THETA),
        (BY_SIGMA_THETA, "temperature"),
        ("nitrate", "oxygen"),
    ],
)
def test_covers_is_false_for_unrelated_names(requested, declared):
    assert not vocabulary.covers(requested, declared)


@pytest.mark.parametrize(
    "name",
    ["mld", MLD, BY_SIGMA_THETA, "temperature", "not_a_variable", "Temperature_CTD"],
)
def test_covers_is_true_for_a_name_against_itself(name):
    assert vocabulary.covers(name, name)


def test_covers_still_means_the_same_quantity_through_patterns_and_case():
    """Everything ``same_quantity`` already accepts, ``covers`` accepts too."""
    assert vocabulary.covers("temperature", "Temperature_CTD")
    assert vocabulary.covers("SEA_WATER_POTENTIAL_TEMPERATURE", "temp")
    assert vocabulary.covers("Not_A_Variable", "not_a_variable")
    assert not vocabulary.covers("Not_A_Variable", "another_unknown")


def test_covers_differs_from_same_quantity_only_within_mixed_layer_depth():
    """Outside the MLD family there is no hidden broadening: covers IS same_quantity.

    A pair is compared over every spelling the shipped vocabulary recognizes, so a
    ``broader`` added to some *other* entry later changes this test on purpose --
    it is the one place that would say a new relation went beyond what was intended.
    """
    family = {MLD, *MLD_DEFINITIONS}
    spellings = sorted(
        {
            spelling
            for key in vocabulary.VOCABULARY
            for spelling in {key, *vocabulary.equivalent_names(key)}
        }
    )
    outside = [s for s in spellings if vocabulary.resolve_name(s) not in family]
    assert len(outside) > 50, "the sweep should cover the whole shipped vocabulary"
    for a in outside:
        for b in outside:
            assert vocabulary.covers(a, b) == vocabulary.same_quantity(a, b), (a, b)
    for a in outside:  # and nothing outside is covered by, or covers, an MLD name
        for name in family:
            assert not vocabulary.covers(a, name), (a, name)
            assert not vocabulary.covers(name, a), (name, a)


def test_equivalent_names_and_same_quantity_leave_definitions_out():
    """``broader`` is one-directional; the symmetric "same variable" tests stay put."""
    names = vocabulary.equivalent_names("mld")
    assert names == {MLD, "mixed_layer_depth", "mixed_layer_thickness"}
    for definition in MLD_DEFINITIONS:
        assert not vocabulary.same_quantity("mld", definition)
        assert not vocabulary.same_quantity(definition, "mld")
        assert vocabulary.equivalent_names(definition).isdisjoint(names)
        assert definition not in names


def test_shipped_broader_relation_is_one_level_and_points_at_real_entries():
    """The one-level rule, checked on the vocabulary exactly as shipped.

    Every ``broader`` must be a name the vocabulary knows, must not be the entry
    itself, and must name an entry that has no ``broader`` of its own: no chain, so
    ``narrower_names``/``covers`` never need to follow one.
    """
    by_standard_name = {e["standard_name"]: e for e in vocabulary.VOCABULARY.values()}
    linked = 0
    for key, entry in vocabulary.VOCABULARY.items():
        broader = entry.get("broader")
        if broader is None:
            continue
        linked += 1
        assert vocabulary.is_known(broader), (key, broader)
        parent = vocabulary.resolve_name(broader)
        assert parent != entry["standard_name"], f"{key} is its own broader"
        assert "broader" not in by_standard_name[parent], f"{key} starts a chain"
        assert entry["standard_name"] in vocabulary.narrower_names(broader), key
    assert linked == len(MLD_DEFINITIONS)  # today's only ``broader`` entries


def test_every_shipped_entry_has_its_own_standard_name():
    """``_build_by_standard_name`` keeps the LAST entry per standard_name.

    Two keys sharing one standard_name would silently hide the first from
    ``equivalent_names`` -- exactly the hazard of adding a second entry for
    ``ocean_mixed_layer_thickness`` beside ``mld``.
    """
    standard_names = [e["standard_name"] for e in vocabulary.VOCABULARY.values()]
    duplicated = {n for n in standard_names if standard_names.count(n) > 1}
    assert not duplicated, duplicated


def test_register_with_broader_takes_effect_immediately(pristine_vocabulary):
    vocabulary.register("mld_by_shear", "mld_by_shear_standard", broader="mld")

    expected = tuple(sorted((*MLD_DEFINITIONS, "mld_by_shear_standard")))
    assert vocabulary.narrower_names("mld") == expected
    assert vocabulary.covers("mld", "mld_by_shear_standard")
    assert vocabulary.covers("mld", "mld_by_shear")  # ... or by its own key
    assert not vocabulary.covers("mld_by_shear", "mld")  # one direction only
    assert not vocabulary.covers(BY_SIGMA_THETA, "mld_by_shear_standard")  # a sibling
    assert vocabulary.covers("mld", BY_SIGMA_THETA)  # the shipped ones are unaffected


@pytest.mark.parametrize("broad", ["mld", "MLD", "mixed_layer_depth", MLD])
def test_register_broader_may_be_any_spelling_of_the_broad_entry(
    pristine_vocabulary, broad
):
    vocabulary.register("mld_by_shear", "mld_by_shear_standard", broader=broad)
    assert "mld_by_shear_standard" in vocabulary.narrower_names("mld")


def test_a_live_registered_definition_is_found_dataset_side_from_the_generic_name(
    pristine_vocabulary,
):
    vocabulary.register("mld_by_shear", "mld_by_shear_standard", broader="mld")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        da = find_variable(_tiny("mld_by_shear_standard"), "mld")
    assert da is not None and da.name == "mld_by_shear_standard"


def test_a_new_broad_and_specific_pair_can_be_registered_from_scratch(
    pristine_vocabulary,
):
    vocabulary.register("foo", "standard_foo")
    vocabulary.register("foo_kind", "standard_foo_kind", broader="foo")
    assert vocabulary.narrower_names("foo") == ("standard_foo_kind",)
    assert vocabulary.covers("foo", "standard_foo_kind")
    assert not vocabulary.covers("standard_foo_kind", "foo")
    assert vocabulary.narrower_names("mld") == MLD_DEFINITIONS  # nothing bled over


def _assert_register_rejected(match: str, *args, **kwargs) -> None:
    """``register`` raises ``ValueError`` matching ``match`` and stores nothing.

    Validation runs before ``VOCABULARY`` is touched (as a pattern's does), so a
    rejected call leaves both the entries and what ``narrower_names`` reports exactly
    as they were, and a following ``_refresh`` has nothing to warn about.
    """
    before = {k: dict(v) for k, v in vocabulary.VOCABULARY.items()}
    with pytest.raises(ValueError, match=match):
        vocabulary.register(*args, **kwargs)
    assert {k: dict(v) for k, v in vocabulary.VOCABULARY.items()} == before
    assert vocabulary.narrower_names("mld") == MLD_DEFINITIONS
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        vocabulary._refresh()


@pytest.mark.parametrize(
    "own_spelling", ["mld_x", "mld_x_standard", "mld_x_alias", "MLD_X_ALIAS"]
)
def test_register_rejects_an_entry_that_is_its_own_broader(
    pristine_vocabulary, own_spelling
):
    _assert_register_rejected(
        "names this entry itself",
        "mld_x",
        "mld_x_standard",
        aliases=["mld_x_alias"],
        broader=own_spelling,
    )


def test_register_rejects_a_broader_that_resolves_to_its_own_standard_name(
    pristine_vocabulary,
):
    """A second key for an existing standard_name, naming its twin as its broader."""
    _assert_register_rejected("names this entry itself", "mld_twin", MLD, broader="mld")


def test_register_rejects_several_broaders(pristine_vocabulary):
    """An entry is one specific kind of *one* broader entry, never a list of them."""
    several: object = ["mld", "temperature"]
    with pytest.raises(TypeError, match="one name"):
        vocabulary.register("mld_x", "mld_x_standard", broader=several)
    assert "mld_x" not in vocabulary.VOCABULARY
    assert vocabulary.narrower_names("mld") == MLD_DEFINITIONS


def test_register_rejects_an_unknown_broader(pristine_vocabulary):
    _assert_register_rejected(
        "not a known vocabulary name",
        "mld_x",
        "mld_x_standard",
        broader="not_a_concept",
    )


def test_register_rejects_a_broader_that_is_itself_a_specific_kind(
    pristine_vocabulary,
):
    """One level only: naming a definition as the broader would build a chain."""
    _assert_register_rejected(
        "name 'mld' instead",
        "mld_deeper",
        "mld_deeper_standard",
        broader="mld_by_sigma_theta",
    )


@pytest.mark.parametrize("standard_name", [MLD, "renamed_mld"])
def test_register_rejects_a_broader_on_an_entry_that_already_has_definitions(
    pristine_vocabulary, standard_name
):
    """Making the generic entry itself a kind of something would build a chain.

    The four definitions would become its far end -- and they follow the *key*
    ``mld``, so renaming its standard_name in the same call does not escape it.
    """
    _assert_register_rejected(
        "already has specific entries under it",
        "mld",
        standard_name,
        broader="temperature",
    )


def test_add_alias_keeps_the_broader_link(pristine_vocabulary):
    """A raw product name can join a definition live and stay one kind of ``mld``."""
    vocabulary.add_alias("mld_by_sigma_theta", "mld_dt_mean")

    assert vocabulary.VOCABULARY["mld_by_sigma_theta"]["broader"] == "mld"
    assert vocabulary.narrower_names("mld") == MLD_DEFINITIONS
    assert vocabulary.resolve_name("mld_dt_mean") == BY_SIGMA_THETA
    assert vocabulary.covers("mld", "mld_dt_mean")
    assert not vocabulary.covers("mld_by_temperature", "mld_dt_mean")


def test_add_pattern_keeps_the_broader_link(pristine_vocabulary):
    vocabulary.add_pattern("mld_by_sigma_theta", r"mld_dt_(?:mean|min|max)")

    assert vocabulary.VOCABULARY["mld_by_sigma_theta"]["broader"] == "mld"
    assert vocabulary.narrower_names("mld") == MLD_DEFINITIONS
    assert vocabulary.covers("mld", "MLD_DT_MEAN")
    assert not vocabulary.covers(BY_SIGMA_T, "mld_dt_mean")


def test_reregistering_the_broad_entry_keeps_its_definitions(pristine_vocabulary):
    """The reason ``broader`` lives on the specific entries, not as a list on ``mld``.

    ``register`` replaces the whole dict, dropping the aliases; a list of children kept
    there would go with them, but each definition names its own broader.
    """
    vocabulary.register("mld", MLD)

    assert vocabulary.VOCABULARY["mld"] == {"standard_name": MLD}
    assert vocabulary.narrower_names("mld") == MLD_DEFINITIONS
    assert vocabulary.covers("mld", BY_SIGMA_THETA)
    # ... while the alias the replacement dropped no longer reaches the concept
    assert vocabulary.narrower_names("mixed_layer_depth") == ()


def test_definitions_follow_the_broad_key_when_it_gets_a_new_standard_name(
    pristine_vocabulary,
):
    with pytest.warns(UserWarning, match="replaces an existing entry"):
        vocabulary.register("mld", "renamed_mld")

    assert vocabulary.narrower_names("mld") == MLD_DEFINITIONS
    assert vocabulary.narrower_names("renamed_mld") == MLD_DEFINITIONS
    assert vocabulary.narrower_names(MLD) == ()  # the old name is just unknown now


def test_register_replace_warning_names_broader_among_what_is_discarded(
    pristine_vocabulary,
):
    with pytest.warns(UserWarning, match="aliases, patterns and broader"):
        vocabulary.register("mld_by_temperature", "some_other_standard_name")
    # ... and it really was discarded: it is no longer a kind of mld
    assert "some_other_standard_name" not in vocabulary.narrower_names("mld")
    assert BY_TEMPERATURE not in vocabulary.narrower_names("mld")


@pytest.mark.parametrize(
    "bad,problem",
    [
        ("no_such_name", "is not a known name"),
        ("mld_by_temperature", "is the entry itself"),
        ("mld_by_sigma_t", "itself one specific kind of another"),
    ],
)
def test_a_hand_edited_broader_that_breaks_the_rule_is_warned_about_and_ignored(
    pristine_vocabulary, bad, problem
):
    """``register`` refuses these; a hand edit of ``VOCABULARY`` gets a warning instead.

    Dangling, self-referential and chained links are all dropped from the relation
    (so ``covers`` cannot be fooled by one) rather than half-honoured, and the
    dangling target does not acquire a child under its unknown name either.
    """
    vocabulary.VOCABULARY["mld_by_temperature"]["broader"] = bad
    with pytest.warns(UserWarning, match=f"vocabulary broader.*{problem}"):
        vocabulary._refresh()

    assert BY_TEMPERATURE not in vocabulary.narrower_names("mld")
    assert not vocabulary.covers("mld", BY_TEMPERATURE)
    assert vocabulary.narrower_names(bad) == ()
    assert vocabulary.narrower_names("mld") == (
        BY_MIXING_SCHEME,
        BY_SIGMA_T,
        BY_SIGMA_THETA,
    )


# -- coordinate vocabulary ------------------------------------------------------


def test_matches_axis_recognizes_plain_spellings():
    assert vocabulary.matches_axis("Depth", "Z")
    assert vocabulary.matches_axis("Longitude", "X")
    assert vocabulary.matches_axis("Latitude", "Y")
    assert vocabulary.matches_axis("time", "T")


@pytest.mark.parametrize("name", ["Depth_bottom", "bottom_depth", "BOTTOM_Z"])
def test_matches_axis_refuses_a_bottom_depth_name(name):
    """The motivating fix: "bottom" disqualifies an otherwise depth-shaped name."""
    assert not vocabulary.matches_axis(name, "Z")
    assert vocabulary.excluded_from_axis(name, "Z")


def test_excluded_from_axis_is_false_for_axes_with_no_exclude_list():
    """Only Z has an ``exclude`` entry; T/X/Y never refuse a name this way."""
    assert not vocabulary.excluded_from_axis("bottom", "T")
    assert not vocabulary.excluded_from_axis("bottom", "X")
    assert not vocabulary.excluded_from_axis("bottom", "Y")


def test_matches_axis_direct_only_excludes_pressure_spellings():
    assert vocabulary.matches_axis("pressure", "Z")
    assert not vocabulary.matches_axis("pressure", "Z", direct_only=True)
    assert vocabulary.matches_axis("depth", "Z", direct_only=True)


def test_coord_vocabulary_fallbacks_never_collide_with_their_own_exclude_list():
    """Regression guard backing the claim in :func:`ocean_skill.cf.find_coord`.

    If a fallback name were ever added that an exclude token would also refuse,
    the name-fallback path in ``find_coord`` would need its own exclusion check --
    right now it doesn't, because this can never happen.
    """
    for entry in vocabulary.COORD_VOCABULARY.values():
        for fallback in entry["fallbacks"]:
            for exclude_word in entry.get("exclude", ()):
                assert exclude_word not in fallback.lower()


def test_coord_report_groups_declared_columns_by_axis():
    report = vocabulary.coord_report(
        ["Latitude", "Longitude", "Depth", "Depth_bottom", "Temperature_CTD"]
    )
    assert report.matched == {"X": ["Longitude"], "Y": ["Latitude"], "Z": ["Depth"]}
    assert report.missing == ["T"]


def test_coord_report_on_no_columns_is_all_missing():
    report = vocabulary.coord_report([])
    assert report.matched == {}
    assert report.missing == ["T", "X", "Y", "Z"]


def test_coord_report_collisions_flags_an_axis_claimed_twice():
    report = vocabulary.coord_report(["Depth", "Pressure"])
    assert "Z" in report.collisions
    assert set(report.collisions["Z"]) == {"Depth", "Pressure"}


def test_coord_report_str_names_the_axis_and_its_kind():
    text = str(vocabulary.coord_report(["Depth"]))
    assert "Z (vertical)" in text and "Depth" in text
    assert "missing" in text


def test_coord_report_repr_html_escapes_and_wraps():
    html = vocabulary.coord_report(["Depth"])._repr_html_()
    assert html.startswith("<pre")
    assert "Depth" in html
