"""Tests for the variable vocabulary: name resolution and live extension.

The vocabulary is what lets a caller name a variable however they like — a short
key (``"oxygen"``), the canonical CF standard_name, or any spelling a real product
happens to use, in any capitalization — and still reach the same variable. These
cover the resolution rules, the two live-extension entry points, and the spots
where getting it wrong would be silent rather than loud (colliding spellings,
clobbered entries, a QC companion standing in for real data).
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
