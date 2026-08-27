"""Tests for variable pair-specs: ``{"test": <spec>, "reference": <spec>}``.

The motivating case is mixed layer depth: a model computes it
(``{"calculate": "mld", "method": "density_threshold"}``) while an observational
climatology already ships it as a plain field (``"mld_dt_mean"``). Comparison/
compare() apply one ``variable=`` to both lanes, so this needs a way to say two
different things and still call them the same comparison.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import xarray as xr

from ocean_skill import cache as _cache
from ocean_skill import comparison
from ocean_skill.comparison import (
    NO_VERTICAL_AXIS,
    Comparison,
    _canonical,
    _display_depth,
    _identity,
    _is_calculated,
    _prepare,
    _short_variable_label,
    _variable_label,
    is_pair_spec,
    variable_for,
)
from ocean_skill.operators import register_calculator, register_derived

CHL = "mass_concentration_of_chlorophyll_a_in_sea_water"
MLD = "ocean_mixed_layer_thickness"

PAIR = {
    "test": {"calculate": "mld", "method": "density_threshold"},
    "reference": "mld_dt_mean",
}


# -- the marker and the accessor -----------------------------------------------


def test_is_pair_spec_requires_both_keys():
    assert is_pair_spec(PAIR)
    assert not is_pair_spec({"test": "x"})  # one-sided: a likely typo, not a pair
    assert not is_pair_spec({"reference": "x"})
    assert not is_pair_spec({"sum": ["a", "b"]})  # an ordinary combination
    assert not is_pair_spec("temperature")


def test_variable_for_picks_the_named_side():
    assert variable_for(PAIR, "test") == PAIR["test"]
    assert variable_for(PAIR, "reference") == "mld_dt_mean"


def test_variable_for_passes_through_a_non_pair_unchanged():
    assert variable_for("temperature", "test") == "temperature"
    assert variable_for({"sum": ["a", "b"]}, "reference") == {"sum": ["a", "b"]}


def test_a_one_sided_pair_names_the_missing_key():
    with pytest.raises(ValueError, match="reference"):
        Comparison(reference="r", test="t", variable={"test": "temperature"})
    with pytest.raises(ValueError, match="test"):
        Comparison(reference="r", test="t", variable={"reference": "temperature"})


# -- resolution: the mechanism the feature exists for --------------------------


def test_prepare_resolves_each_side_independently():
    """Two different sources, two different specs, one physical quantity."""
    test_ds = xr.Dataset(
        {"a": (("lat", "lon"), np.full((2, 2), 1.0), {"units": "m"})},
        coords={"lat": [1.0, 2.0], "lon": [1.0, 2.0]},
    )
    reference_ds = xr.Dataset(
        {"b": (("lat", "lon"), np.full((2, 2), 2.0), {"units": "m"})},
        coords={"lat": [1.0, 2.0], "lon": [1.0, 2.0]},
    )
    pair = {"test": "a", "reference": "b"}
    test_da, _ = _prepare(test_ds, {}, variable_for(pair, "test"), {})
    reference_da, _ = _prepare(
        reference_ds, {}, variable_for(pair, "reference"), {}
    )
    assert float(test_da.mean()) == 1.0
    assert float(reference_da.mean()) == 2.0


def test_comparison_resolves_a_string_on_each_side():
    """A plain-name side gets the same vocabulary treatment a lone variable= does."""
    c = Comparison(
        reference="r", test="t", variable={"test": "temperature", "reference": "temperature"}
    )
    assert c.variable["test"] == "sea_water_potential_temperature"
    assert c.variable["reference"] == "sea_water_potential_temperature"


# -- standard_name: explicit wins, else the test side --------------------------


def test_standard_name_explicit_wins():
    c = Comparison(reference="r", test="t", variable={**PAIR, "standard_name": MLD})
    assert c.standard_name == MLD


def test_standard_name_falls_back_to_the_test_sides_own_standard_name():
    combo_pair = {
        "test": {"sum": ["spChl", "diatChl"], "standard_name": CHL},
        "reference": "modis_chl",
    }
    c = Comparison(reference="r", test="t", variable=combo_pair)
    assert c.standard_name == CHL


def test_standard_name_is_none_when_neither_side_names_one():
    """A calculate-spec carries no standard_name of its own -- same as a bare one."""
    c = Comparison(reference="r", test="t", variable=PAIR)
    assert c.standard_name is None


# -- the mismatch warning --------------------------------------------------------


def test_mismatch_warning_fires_when_sides_resolve_differently():
    c = Comparison(reference="r", test="t", variable=PAIR)
    test_da = xr.DataArray(1.0, attrs={"standard_name": MLD})
    reference_da = xr.DataArray(1.0, attrs={"standard_name": CHL})
    with pytest.warns(UserWarning, match="test side resolves"):
        c._warn_on_pair_spec_mismatch(test_da, reference_da)


def test_mismatch_warning_falls_back_to_the_arrays_own_name():
    """attrs['standard_name'] or .name -- the same fallback find_variable itself uses."""
    c = Comparison(reference="r", test="t", variable=PAIR)
    test_da = xr.DataArray(1.0, attrs={"standard_name": MLD})
    reference_da = xr.DataArray(1.0, name=CHL)
    with pytest.warns(UserWarning, match="test side resolves"):
        c._warn_on_pair_spec_mismatch(test_da, reference_da)


def test_mismatch_warning_is_silent_with_an_explicit_standard_name():
    """The caller has already said the two recipes are the same quantity."""
    c = Comparison(reference="r", test="t", variable={**PAIR, "standard_name": MLD})
    test_da = xr.DataArray(1.0, attrs={"standard_name": MLD})
    reference_da = xr.DataArray(1.0, attrs={"standard_name": CHL})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        c._warn_on_pair_spec_mismatch(test_da, reference_da)  # must not raise


def test_mismatch_warning_is_silent_when_sides_agree():
    c = Comparison(reference="r", test="t", variable=PAIR)
    test_da = xr.DataArray(1.0, attrs={"standard_name": MLD})
    reference_da = xr.DataArray(1.0, name=MLD)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        c._warn_on_pair_spec_mismatch(test_da, reference_da)


def test_a_non_pair_spec_never_warns():
    c = Comparison(reference="r", test="t", variable="temperature")
    test_da = xr.DataArray(1.0, attrs={"standard_name": MLD})
    reference_da = xr.DataArray(1.0, attrs={"standard_name": CHL})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        c._warn_on_pair_spec_mismatch(test_da, reference_da)


# -- dedup and cache keys --------------------------------------------------------


def test_identity_distinguishes_pairs_by_either_side():
    base = Comparison(reference="r", test="t", variable={"test": "a", "reference": "b1"})
    other_ref = Comparison(
        reference="r", test="t", variable={"test": "a", "reference": "b2"}
    )
    other_test = Comparison(
        reference="r", test="t", variable={"test": "a2", "reference": "b1"}
    )
    assert _identity(base) != _identity(other_ref)
    assert _identity(base) != _identity(other_test)


def test_identity_treats_equal_pairs_as_the_same():
    c1 = Comparison(reference="r", test="t", variable={"test": "a", "reference": "b"})
    c2 = Comparison(reference="r", test="t", variable={"test": "a", "reference": "b"})
    assert _identity(c1) == _identity(c2)


def test_cache_key_is_distinct_for_plain_vs_pair_and_across_reference_sides():
    common = dict(test="t", reference="r", select={}, method="conservative_normed")
    plain_key = _cache.key_for(variable="a", **common)
    pair_key_1 = _cache.key_for(variable={"test": "a", "reference": "b1"}, **common)
    pair_key_2 = _cache.key_for(variable={"test": "a", "reference": "b2"}, **common)
    assert len({plain_key, pair_key_1, pair_key_2}) == 3


# -- labels ----------------------------------------------------------------------


def test_variable_label_uses_the_calculator_name_not_its_letters():
    """Regression: {"calculate": "mld"} used to be joined like a components list,
    reading as "m+l+d" -- caught while wiring pair-specs through this function.
    """
    assert (
        _variable_label({"calculate": "mld", "method": "density_threshold"})
        == "mld (density_threshold)"
    )


def test_variable_label_on_a_pair_prefers_the_explicit_standard_name():
    assert (
        _variable_label({**PAIR, "standard_name": MLD})
        == f"{MLD} (density_threshold)"
    )


def test_variable_label_on_a_pair_falls_back_to_the_test_side():
    assert _variable_label(PAIR) == "mld (density_threshold)"


def test_short_variable_label_on_a_named_pair_uses_the_vocabulary():
    assert (
        _short_variable_label({**PAIR, "standard_name": MLD})
        == "ocean mixed layer thickness (density_threshold)"
    )


def test_short_variable_label_recurses_into_a_plain_test_side():
    pair = {"test": "temperature", "reference": "mld_dt_mean"}
    assert _short_variable_label(pair) == "temperature"


def test_two_methods_sharing_a_standard_name_get_different_labels():
    """Regression: two pair-specs computing MLD by different methods, both carrying
    the same explicit standard_name (as the README recommends), used to draw as two
    identically-labelled rows in the same figure -- unreadable, not just cosmetic.
    """
    density = {**PAIR, "standard_name": MLD}
    temperature = {
        "test": {"calculate": "mld", "method": "temperature_threshold"},
        "reference": "mld_tt_mean",
        "standard_name": MLD,
    }
    assert _short_variable_label(density) != _short_variable_label(temperature)
    assert _variable_label(density) != _variable_label(temperature)


def test_a_calculate_spec_with_no_method_gets_no_suffix():
    """The suffix is additive, not assumed -- a calculator that takes no method
    (or wasn't given one) must not grow a stray "(None)".
    """
    assert _variable_label({"calculate": "eke"}) == "eke"
    assert _short_variable_label({"calculate": "eke"}) == "eke"


# -- Field refuses a pair-spec: there is no second lane to give one to ----------


def test_field_refuses_a_pair_spec():
    from ocean_skill.field import Field

    with pytest.raises(TypeError, match="pair-spec"):
        Field("some_source", PAIR)


# -- compare() fan-out: each side checked against its own catalog metadata -----


def test_compare_fan_out_filters_each_side_of_a_pair_spec_independently():
    """The point of the feature: a source is a candidate test lane only if it can
    plausibly supply the *test* spec's inputs, and a candidate reference lane only if
    it declares the *reference* spec's own name -- checked separately, not as a pair.

    The reference side here is the recognized standard_name (what a real catalog
    entry's ``standard_names`` rename map would produce from ``mld_dt_mean``) rather
    than the raw field name itself: a raw name unknown to the vocabulary falls open
    under the documented "absent vs unknowable" rule (see
    ``test_a_combination_over_raw_names_is_never_filtered_out`` in
    test_operators.py), which would make this test pass for the wrong reason.
    """
    from unittest import mock

    temp = "sea_water_potential_temperature"
    salt = "sea_water_practical_salinity"
    nitrate = "mole_concentration_of_nitrate_in_sea_water"
    pair = {
        "test": {"calculate": "mld", "method": "density_threshold"},
        "reference": MLD,
    }
    declared = {
        "GOM_bgc": {"variables": [temp, salt]},  # has what the calculate-spec needs
        "GOM_nutrients_only": {"variables": [nitrate]},  # does not
        "holte_talley": {"variables": [MLD]},  # has the reference field, by CF name
        "other_obs": {"variables": ["some_other_variable"]},  # does not
    }
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
    ):
        comparison.compare(
            reference=["holte_talley", "other_obs"],
            test=["GOM_bgc", "GOM_nutrients_only"],
            variables=[pair],
        )
    assert formed == [("GOM_bgc", "holte_talley")]


def test_offers_reaches_a_source_advertising_a_pattern_spelling():
    """A source declaring only a vocabulary-pattern spelling must still be found.

    same_quantity(), not just equivalent_names()'s literal set, decides whether the
    source offers the requested variable.
    """
    from unittest import mock

    declared = {
        "seanoe_ctd_mooring": {"variables": ["Temperature_CTD"]},
        "other_obs": {"variables": ["some_other_variable"]},
    }
    formed = []
    with (
        mock.patch(
            "ocean_skill.catalog.resolve", lambda n: mock.Mock(metadata=declared[n])
        ),
        mock.patch.object(
            comparison.Comparison,
            "align",
            lambda self, refresh=False: formed.append(self.reference_name),
        ),
    ):
        comparison.compare(
            reference=["seanoe_ctd_mooring", "other_obs"],
            test=["seanoe_ctd_mooring"],
            variables=["temperature"],
        )
    assert formed == ["seanoe_ctd_mooring"]


def test_offers_still_errs_open_for_a_raw_model_component():
    """A raw per-PFT model tracer (spChl) is unknowable from metadata alone.

    True with pattern-aware is_known included -- it must still fall open rather
    than be excluded as "absent".
    """
    from unittest import mock

    declared = {"model": {"variables": [CHL]}, "obs": {"variables": [CHL]}}
    formed = []
    with (
        mock.patch(
            "ocean_skill.catalog.resolve", lambda n: mock.Mock(metadata=declared[n])
        ),
        mock.patch.object(
            comparison.Comparison,
            "align",
            lambda self, refresh=False: formed.append(self.test_name),
        ),
    ):
        comparison.compare(
            reference=["obs"],
            test=["model"],
            variables=[{"sum": ["spChl", "diatChl"], "standard_name": CHL}],
        )
    assert formed == ["model"]


def test_compare_never_injects_a_depth_key_for_a_calculated_variable():
    """compare()'s depth fan-out defaults every variable to depths=("surface",),
    which used to reach _prepare's calculate guard as select={"depth": "surface"}
    and raise -- discovered wiring the pair-spec through compare() end-to-end, and a
    general compare()+calculate interaction, not specific to pairs (a bare
    {"calculate": ...} variable hits the same fan-out).
    """
    from unittest import mock

    declared = {"model": {"variables": []}, "holte_talley": {"variables": [MLD]}}
    formed = []
    with (
        mock.patch(
            "ocean_skill.catalog.resolve", lambda n: mock.Mock(metadata=declared[n])
        ),
        mock.patch.object(
            comparison.Comparison,
            "align",
            lambda self, refresh=False: formed.append(self.select),
        ),
    ):
        comparison.compare(
            reference=["holte_talley"],
            test=["model"],
            variables=[{**PAIR, "standard_name": MLD}],
        )
    assert formed == [{}], f"expected no depth key at all, got {formed}"


def test_compare_resolves_string_sides_of_a_pair_before_filtering():
    """A plain-name side goes through the same vocabulary resolution a lone
    variable= gets, so _offers compares against the canonical standard_name.
    """
    from unittest import mock

    declared = {
        "model": {"variables": ["sea_water_potential_temperature"]},
        "obs": {"variables": ["sea_water_potential_temperature"]},
    }
    formed = []
    with (
        mock.patch(
            "ocean_skill.catalog.resolve", lambda n: mock.Mock(metadata=declared[n])
        ),
        mock.patch.object(
            comparison.Comparison,
            "align",
            lambda self, refresh=False: formed.append(self.variable),
        ),
    ):
        comparison.compare(
            reference=["obs"],
            test=["model"],
            variables=[{"test": "temperature", "reference": "temperature"}],
        )
    assert formed == [
        {
            "test": "sea_water_potential_temperature",
            "reference": "sea_water_potential_temperature",
        }
    ]


# -- fixes from the fable review (2026-08-19) ------------------------------------
#
# Each of these reproduces a bug exactly as the reviewer found it, verified to fail
# without the corresponding fix before it landed.


def test_offers_does_not_fall_closed_for_a_calculator_with_no_registered_inputs():
    """spec_names() returns [] for a calculator that registered no `inputs=` --
    the exact shape of the README's/register_calculator's own `eke` example. The old
    `not all(is_known(n) for opt in options for n in opt)` is vacuously True over an
    empty `options`, so `_offers` returned False and compare() silently produced an
    empty ComparisonSet for the documented no-`inputs=` workflow.
    """
    from unittest import mock

    @register_calculator("eke_test_offers")
    def _eke(ds, **kwargs):
        return ds

    declared = {
        "model": {"variables": ["something"]},
        "obs": {"variables": ["something_else"]},
    }
    formed = []
    with (
        mock.patch(
            "ocean_skill.catalog.resolve", lambda n: mock.Mock(metadata=declared[n])
        ),
        mock.patch.object(
            comparison.Comparison, "align", lambda self, refresh=False: formed.append(1)
        ),
    ):
        comparison.compare(
            reference=["obs"], test=["model"], variables=[{"calculate": "eke_test_offers"}]
        )
    assert formed == [1]


def test_compare_warns_rather_than_silently_dropping_an_explicit_depth():
    """Comparison(..., select={"depth": 50}) raises loudly for a calculated
    variable; compare(..., select={"depth": 50}) used to accept the identical
    request and silently discard it with no message at all.
    """
    from unittest import mock

    declared = {"model": {"variables": []}, "holte_talley": {"variables": [MLD]}}
    pair = {**PAIR, "standard_name": MLD}
    seen = []
    with (
        mock.patch(
            "ocean_skill.catalog.resolve", lambda n: mock.Mock(metadata=declared[n])
        ),
        mock.patch.object(
            comparison.Comparison,
            "align",
            lambda self, refresh=False: seen.append(self.select),
        ),pytest.warns(UserWarning, match="calculated diagnostic")
    ):
        comparison.compare(
            reference=["holte_talley"],
            test=["model"],
            variables=[pair],
            depths=(0, 50, 100),
        )
    assert seen == [{}], "the depth key must still be dropped, just not silently"


def test_compare_stays_silent_for_the_bare_surface_default():
    """Only a genuine, explicit request is worth a warning -- depths=("surface",)
    is indistinguishable from never having asked, and warning on it would make the
    warning fire on the common case rather than the contradiction it exists for.
    """
    from unittest import mock

    declared = {"model": {"variables": []}, "holte_talley": {"variables": [MLD]}}
    pair = {**PAIR, "standard_name": MLD}
    with (
        mock.patch(
            "ocean_skill.catalog.resolve", lambda n: mock.Mock(metadata=declared[n])
        ),
        mock.patch.object(comparison.Comparison, "align", lambda self, refresh=False: None),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("error")
        comparison.compare(reference=["holte_talley"], test=["model"], variables=[pair])
        comparison.compare(
            reference=["holte_talley"],
            test=["model"],
            variables=[pair],
            depths=("surface",),
        )


def test_mismatch_warning_fires_on_a_cache_hit():
    """The mismatch check used to sit only on the freshly-computed path -- every
    later process serving the same pair from disk cache silently skipped it.
    """
    from unittest import mock

    c = Comparison(reference="r", test="t", variable=PAIR)
    hit = xr.Dataset(
        {
            "test": xr.DataArray(1.0, attrs={"standard_name": MLD}),
            "reference": xr.DataArray(1.0, attrs={"standard_name": "some_other_thing"}),
            "difference": xr.DataArray(0.0),
        }
    )
    with mock.patch("ocean_skill.cache.load", return_value=hit):
        with pytest.warns(UserWarning, match="test side resolves"):
            c.align()


def test_mismatch_warning_does_not_trust_the_aligned_datasets_own_names():
    """align.align() literally names its output variables "test"/"reference" (its
    own test_name=/reference_name= parameters) -- .name there is always "test" and
    always "reference", always unequal, regardless of what the fields actually are.
    Falling back to it on a cache hit would warn on every hit lacking a
    standard_name attr rather than only a real mismatch.
    """
    from unittest import mock

    c = Comparison(reference="r", test="t", variable=PAIR)
    hit = xr.Dataset(
        {
            "test": xr.DataArray(1.0),  # no standard_name attr on either side
            "reference": xr.DataArray(1.0),
            "difference": xr.DataArray(0.0),
        }
    )
    with mock.patch("ocean_skill.cache.load", return_value=hit):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            c.align()  # must not warn: "test" != "reference" is not a real mismatch


def test_a_one_sided_pair_in_a_variables_list_is_caught_before_any_fan_out_runs():
    """The one-sided check used to live only in Comparison.__init__, reachable only
    mid-fan-out -- after any earlier, valid variables in the same compare() call had
    already aligned. A bad spec anywhere in the list must be caught up front.
    """
    from unittest import mock

    ran = []
    with mock.patch.object(
        comparison.Comparison, "align", lambda self, refresh=False: ran.append(1)
    ), pytest.raises(ValueError, match="reference"):
        comparison.compare(
            reference=["r"],
            test=["t"],
            variables=[{"test": "temperature"}, "salinity"],
        )
    assert ran == [], "no comparison should have been aligned before the raise"


def test_field_names_a_one_sided_pair_spec_the_same_way_comparison_does():
    from ocean_skill.field import Field

    with pytest.raises(ValueError, match="reference"):
        Field("some_source", {"test": "temperature"})


def test_a_derived_name_wrapping_a_calculate_spec_is_still_calculated():
    """register_derived("x", {"calculate": ...}) makes "x" a calculate-spec in every
    way that matters, even spelled as a plain string.
    """
    register_derived("mld_via_derived", {"calculate": "mld", "method": "temperature_threshold"})
    assert _is_calculated("mld_via_derived")
    assert _is_calculated({"test": "mld_via_derived", "reference": "mld_dt_mean"})


def test_a_derived_calculate_spec_still_refuses_a_depth_selection():
    """Before the fix, _prepare's `calculated` flag only recognized a literal
    {"calculate": ...} dict, so a DERIVED name pointing at one skipped the
    vertical-axis guard entirely and fell through into ROMS's depth machinery.
    """
    register_derived(
        "mld_via_derived_depth", {"calculate": "mld", "method": "temperature_threshold"}
    )
    depths = np.array([2.0, 6.0, 10.0, 20.0, 40.0, 80.0])
    temps = np.array([21.0, 20.0, 20.0, 19.9, 19.0, 15.0])
    shape = (depths.size, 1, 1)
    ds = xr.Dataset(
        {
            "sea_water_potential_temperature": (
                ("s_rho", "eta_rho", "xi_rho"),
                temps[::-1].reshape(shape),
            ),
            "sea_water_practical_salinity": (
                ("s_rho", "eta_rho", "xi_rho"),
                np.full(shape, 35.0),
            ),
        },
        coords={
            "z_rho": (("s_rho", "eta_rho", "xi_rho"), (-depths[::-1]).reshape(shape)),
            "lon": (("eta_rho", "xi_rho"), np.full((1, 1), -90.0)),
            "lat": (("eta_rho", "xi_rho"), np.full((1, 1), 25.0)),
        },
    )
    with pytest.raises(ValueError, match="already reduces the vertical axis"):
        _prepare(ds, {"model": "roms"}, "mld_via_derived_depth", {"depth": 50})


def test_canonical_is_order_insensitive():
    assert _canonical({"test": "a", "reference": "b"}) == _canonical(
        {"reference": "b", "test": "a"}
    )
    assert _canonical({"test": "a", "reference": "b1"}) != _canonical(
        {"test": "a", "reference": "b2"}
    )


def test_identity_agrees_regardless_of_pair_spec_key_order():
    """Two logically-identical pair-specs differing only in dict key order used to
    escape dedup in _flatten (drawn twice) while the disk cache (which already
    sorts keys) treated them as one entry -- the two notions of "same comparison"
    could disagree.
    """
    c1 = Comparison(reference="r", test="t", variable={"test": "a", "reference": "b"})
    c2 = Comparison(reference="r", test="t", variable={"reference": "b", "test": "a"})
    assert _identity(c1) == _identity(c2)


def test_display_depth_is_honest_about_having_no_vertical_axis():
    """A calculated diagnostic used to report "surface" in metrics()/repr/pooled
    labels -- a specific, wrong claim about where in the column the number came
    from, not just an unhelpful default.
    """
    assert _display_depth(PAIR, {}) == NO_VERTICAL_AXIS
    assert _display_depth("temperature", {}) != NO_VERTICAL_AXIS


def test_comparison_metrics_report_no_vertical_axis_for_a_calculated_variable():
    from unittest import mock

    c = Comparison(
        reference="r", test="t", variable={"calculate": "mld", "method": "density_threshold"}
    )
    aligned = xr.Dataset(
        {
            "test": (("lat", "lon"), np.ones((2, 2)), {"units": "m"}),
            "reference": (("lat", "lon"), np.ones((2, 2)), {"units": "m"}),
            "difference": (("lat", "lon"), np.zeros((2, 2))),
        },
        coords={"lat": [1.0, 2.0], "lon": [1.0, 2.0]},
    )
    with mock.patch.object(Comparison, "aligned", aligned):
        record = c.metrics()
    assert record["depth"] == NO_VERTICAL_AXIS
