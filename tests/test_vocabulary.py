"""Tests for the variable vocabulary: name resolution and live extension.

The vocabulary is what lets a caller name a variable however they like — a short
key (``"oxygen"``), the canonical CF standard_name, or any spelling a real product
happens to use, in any capitalization — and still reach the same variable. These
cover the resolution rules, the two live-extension entry points, and the spots
where getting it wrong would be silent rather than loud (colliding spellings,
clobbered entries, a QC companion standing in for real data).
"""

from __future__ import annotations

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

    ``register``/``add_alias`` mutate module state that would otherwise leak into
    every later test in the session (and into cf-xarray's global registration).
    """
    saved = {k: dict(v) for k, v in vocabulary.VOCABULARY.items()}
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

    The MODIS spelling here is a real mismatch found in ``catalogs/modis_aqua.yaml``
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
