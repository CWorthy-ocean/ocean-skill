"""Tests for variable combination and dimension reduction.

The design claim these defend is that adding an operator costs no code: reductions
dispatch to xarray methods, combination to the stdlib ``operator`` module. So the
tests exercise a *spread* of reductions that were never individually implemented —
if the dispatch is right they all work, and if it isn't, several break at once.

The other theme is refusing to produce a plausible wrong number: summing fields with
mismatched units, or a partial sum when a component is missing.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import xarray as xr

from ocean_skill.operators import (
    DERIVED,
    aggregate,
    combine,
    register_reducer,
    resolve_variable,
    select,
)

CHL = "mass_concentration_of_chlorophyll_a_in_sea_water"


@pytest.fixture
def marbl():
    """MARBL-shaped chlorophyll: three phytoplankton components, monthly."""
    time = xr.date_range("2012-01-01", periods=12, freq="MS")

    def field(value):
        return (
            ("time", "lat", "lon"),
            np.full((12, 2, 3), value),
            {"units": "mg/m^3"},
        )

    return xr.Dataset(
        {"spChl": field(1.0), "diatChl": field(2.0), "diazChl": field(0.5)},
        coords={"time": time, "lat": [1.0, 2.0], "lon": [1.0, 2.0, 3.0]},
    )


# -- combining variables ------------------------------------------------------


def test_sums_a_list_of_variables(marbl):
    total = resolve_variable(marbl, {"sum": ["spChl", "diatChl", "diazChl"]})
    assert float(total.isel(time=0, lat=0, lon=0)) == pytest.approx(3.5)


def test_combination_keeps_units(marbl):
    """Arithmetic drops attrs in xarray; the units are the whole point of checking."""
    total = resolve_variable(marbl, {"sum": ["spChl", "diatChl"]})
    assert total.attrs["units"] == "mg/m^3"


def test_combination_can_name_its_result(marbl):
    total = resolve_variable(
        marbl, {"sum": ["spChl", "diatChl", "diazChl"], "standard_name": CHL}
    )
    assert total.name == CHL and total.attrs["standard_name"] == CHL


def test_a_named_spec_resolves_by_name(marbl):
    """DERIVED entries mean a recurring combination is written once, then reused."""
    assert DERIVED["total_chlorophyll"]["sum"] == ["spChl", "diatChl", "diazChl"]
    assert float(
        resolve_variable(marbl, "total_chlorophyll").isel(time=0, lat=0, lon=0)
    ) == pytest.approx(3.5)


@pytest.mark.parametrize(
    ("how", "expected"),
    [("sum", 3.0), ("difference", -1.0), ("product", 2.0), ("ratio", 0.5)],
)
def test_every_combiner_works(marbl, how, expected):
    got = resolve_variable(marbl, {how: ["spChl", "diatChl"]})
    assert float(got.isel(time=0, lat=0, lon=0)) == pytest.approx(expected)


def test_mismatched_units_refuse_to_combine(marbl):
    """Xarray would happily add mg/m3 to mmol/m3 and label the result neither."""
    marbl["diatChl"].attrs["units"] = "mmol/m^3"
    with pytest.raises(ValueError, match="different units"):
        resolve_variable(marbl, {"sum": ["spChl", "diatChl"]})


def test_a_missing_component_yields_nothing_not_a_partial_sum(marbl):
    """Three-quarters of a total is not a smaller total, it is a wrong one."""
    assert resolve_variable(marbl, {"sum": ["spChl", "not_present"]}) is None


def test_unknown_combiner_is_rejected(marbl):
    with pytest.raises(KeyError, match="unknown combiner"):
        combine([marbl["spChl"]], "convolve")


# -- reductions ---------------------------------------------------------------


@pytest.mark.parametrize(
    "spec",
    [
        "mean",
        "max",
        "min",
        "sum",
        "std",
        "median",
        "var",
        {"reduce": "mean"},
        {"reduce": "quantile", "q": 0.9},
    ],
)
def test_any_xarray_reduction_works_without_being_registered(marbl, spec):
    """None of these were implemented individually — the dispatch is the feature."""
    out = aggregate(marbl["spChl"], {"time": spec})
    assert "time" not in out.dims
    assert out.attrs["units"] == "mg/m^3", "reductions drop attrs; units must survive"


@pytest.mark.parametrize(("group", "size"), [("month", 12), ("season", 4)])
def test_groupby_gives_a_climatology(marbl, group, size):
    out = aggregate(marbl["spChl"], {"time": {"groupby": group, "reduce": "mean"}})
    assert out.sizes[group] == size
    assert "time" not in out.dims


def test_quantile_is_not_broken_by_positional_dispatch(marbl):
    """``quantile``'s first positional argument is ``q``, not ``dim``.

    Passing the dimension positionally silently lands it in ``q``; this is why the
    dispatch reads the signature instead.
    """
    out = aggregate(marbl["spChl"], {"time": {"reduce": "quantile", "q": 0.5}})
    assert float(out.isel(lat=0, lon=0)) == pytest.approx(1.0)


def test_integrate_uses_its_own_coord_keyword(marbl):
    """``integrate`` names the dimension ``coord``, unlike every other reduction."""
    da = xr.DataArray(
        np.ones((4, 2)),
        dims=("depth", "lat"),
        coords={"depth": [0.0, 10.0, 20.0, 30.0], "lat": [1.0, 2.0]},
    )
    out = aggregate(da, {"depth": {"reduce": "integrate"}})
    assert "depth" not in out.dims
    assert float(out.isel(lat=0)) == pytest.approx(30.0)


def test_reducing_several_dimensions_at_once(marbl):
    out = aggregate(marbl["spChl"], {"time": "mean", "lon": "mean"})
    assert set(out.dims) == {"lat"}


def test_an_absent_dimension_is_skipped_not_an_error(marbl):
    """One spec is shared across variables; not all of them have a depth axis."""
    out = aggregate(marbl["spChl"], {"depth": "mean"})
    assert out.sizes["time"] == 12


def test_empty_spec_is_a_no_op(marbl):
    assert aggregate(marbl["spChl"], None).equals(marbl["spChl"])


def test_unknown_reduction_is_rejected(marbl):
    with pytest.raises(KeyError, match="unknown reduction"):
        aggregate(marbl["spChl"], {"time": "definitely_not_a_method"})


def test_a_spec_without_a_reduction_is_rejected(marbl):
    with pytest.raises(ValueError, match="needs a 'reduce'"):
        aggregate(marbl["spChl"], {"time": {"groupby": "month"}})


def test_a_custom_reducer_can_be_registered(marbl):
    """The escape hatch for the rare reduction xarray has no method for."""

    @register_reducer("range")
    def _range(da, dim, **kw):
        return da.max(dim) - da.min(dim)

    try:
        out = aggregate(marbl["spChl"], {"time": "range"})
        assert float(out.isel(lat=0, lon=0)) == pytest.approx(0.0)
    finally:
        from ocean_skill.operators import REDUCERS

        REDUCERS.pop("range", None)


# -- end to end through the compare layer -------------------------------------


def test_prepare_combines_then_aggregates(marbl):
    """The case that motivated this: sum components, then reduce in time."""
    from ocean_skill.comparison import _prepare

    da, _ = _prepare(
        marbl,
        {},  # non-model source
        {"sum": ["spChl", "diatChl", "diazChl"], "standard_name": CHL},
        {},
        {"time": "mean"},
    )
    assert "time" not in da.dims
    assert float(da.isel(lat=0, lon=0)) == pytest.approx(3.5)


def test_prepare_honours_a_climatology_spec(marbl):
    from ocean_skill.comparison import _prepare

    da, _ = _prepare(
        marbl, {}, "spChl", {}, {"time": {"groupby": "month", "reduce": "mean"}}
    )
    assert da.sizes["month"] == 12


# -- one spec, two storage conventions ----------------------------------------


@pytest.fixture
def modis_style():
    """Build a source carrying the *total*, spelled as MODIS's catalog does."""
    return xr.Dataset(
        {
            "mass_concentration_of_chlorophyll_in_sea_water": (
                ("lat", "lon"),
                np.full((2, 3), 3.0),
                {"units": "mg/m^3"},
            )
        },
        coords={"lat": [1.0, 2.0], "lon": [1.0, 2.0, 3.0]},
    )


SPEC = {"sum": ["spChl", "diatChl", "diazChl"], "standard_name": CHL}


def test_a_combination_falls_back_to_its_standard_name(modis_style):
    """One spec must serve both lanes: MARBL splits chlorophyll, MODIS ships a total.

    Regression: this returned ``None`` for the observational side, so a
    model-vs-MODIS chlorophyll comparison could not be expressed at all.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = resolve_variable(modis_style, SPEC)
    assert got is not None
    assert float(got.mean()) == pytest.approx(3.0)


def test_the_same_spec_still_sums_where_components_exist(marbl):
    assert float(
        resolve_variable(marbl, SPEC).isel(time=0, lat=0, lon=0)
    ) == pytest.approx(3.5)


def test_partially_present_components_warn_loudly(marbl):
    """Some-but-not-all is a typo or truncated output, not a storage convention."""
    marbl["mass_concentration_of_chlorophyll_in_sea_water"] = marbl["spChl"]
    with pytest.warns(UserWarning, match="missing but"):
        resolve_variable(marbl, {"sum": ["spChl", "diatChlTYPO"], "standard_name": CHL})


def test_spec_names_lists_both_ways_to_satisfy_a_combination():
    from ocean_skill.operators import spec_names

    options = spec_names(SPEC)
    assert ["spChl", "diatChl", "diazChl"] in options
    assert [CHL] in options


def test_spec_names_handles_a_plain_name_and_a_derived_key():
    from ocean_skill.operators import spec_names

    assert spec_names("nitrate") == [["nitrate"]]
    assert ["spChl", "diatChl", "diazChl"] in spec_names("total_chlorophyll")


def test_compare_can_filter_sources_by_a_combination_spec():
    """Regression: `_offers` called `.lower()` on the spec and raised AttributeError."""
    from unittest import mock

    from ocean_skill import comparison

    entry = mock.Mock(metadata={"variables": [CHL]})
    with mock.patch("ocean_skill.catalog.resolve", lambda name: entry):
        result = comparison.compare(
            reference=["modis"], test="model", variables=[SPEC], skip_missing=True
        )
    # The reference was *considered* (no AttributeError); alignment is mocked away.
    assert isinstance(result, comparison.ComparisonSet)


# -- selection (distinct from aggregation) ------------------------------------


@pytest.fixture
def two_months():
    """Sixty daily values, so a month-selection is visibly different from the whole."""
    time = xr.date_range("2012-01-01", periods=60, freq="D")
    return xr.DataArray(
        np.arange(60.0)[:, None, None] * np.ones((1, 2, 2)),
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": [1.0, 2.0], "lon": [1.0, 2.0]},
        attrs={"units": "m"},
    )


def test_select_narrows_an_axis_without_collapsing_it(two_months):
    """Selection is not aggregation: the dimension survives, just shorter."""
    got = select(two_months, {"time": "2012-01"})
    assert got.sizes["time"] == 31
    assert "time" in got.dims


def test_select_accepts_a_partial_date_string(two_months):
    assert select(two_months, {"time": "2012-01"}).sizes["time"] == 31


def test_select_accepts_a_slice(two_months):
    got = select(two_months, {"time": slice("2012-01-01", "2012-01-10")})
    assert got.sizes["time"] == 10


def test_select_accepts_a_yaml_friendly_min_max_dict(two_months):
    """YAML has no slice literal, so a suite has to spell ranges as a mapping."""
    got = select(two_months, {"lat": {"min": 1.0, "max": 1.0}})
    assert got.sizes["lat"] == 1


def test_select_falls_back_to_nearest_for_an_inexact_scalar(two_months):
    """A float coordinate almost never matches exactly; failing on that is unhelpful."""
    assert float(select(two_months, {"lat": 1.4}).lat) == 1.0


def test_select_skips_dimensions_the_field_does_not_have(two_months):
    assert select(two_months, {"depth": 100}).sizes == two_months.sizes


def test_select_then_aggregate_is_the_abigale_ordering(two_months):
    """January's mean is select-then-reduce; reversing it averages everything."""
    whole = aggregate(two_months, {"time": "mean"})
    january = aggregate(select(two_months, {"time": "2012-01"}), {"time": "mean"})
    assert float(whole.mean()) == pytest.approx(29.5)
    assert float(january.mean()) == pytest.approx(15.0)


def test_prepare_applies_selection_before_aggregation(two_months):
    """Regression: `select` was accepted but only `depth` was ever read from it."""
    from ocean_skill.comparison import _prepare

    ds = two_months.to_dataset(name="x")
    everything, _ = _prepare(ds, {}, "x", {}, {"time": "mean"})
    january, _ = _prepare(ds, {}, "x", {"time": "2012-01"}, {"time": "mean"})
    assert float(everything.mean()) == pytest.approx(29.5)
    assert float(january.mean()) == pytest.approx(15.0)


def test_a_leftover_dimension_explains_itself(two_months):
    """A groupby against a single-field reference should say so, not fail in xesmf."""
    from ocean_skill import align as _align

    monthly = aggregate(two_months, {"time": {"groupby": "month", "reduce": "mean"}})
    with pytest.raises(ValueError, match="not a single map"):
        _align.align(monthly, monthly, method="bilinear")


def test_a_derived_key_reports_its_cf_standard_name():
    """Otherwise colormaps, plot labels and the metrics table see the key itself."""
    from ocean_skill.comparison import Comparison

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        comparison = Comparison(reference="r", test="t", variable="total_chlorophyll")
    assert comparison.standard_name == CHL


# -- how much needs registering -----------------------------------------------


@pytest.fixture
def abc():
    coords = {"lat": [1.0, 2.0], "lon": [1.0, 2.0]}
    field = lambda v: (  # noqa: E731
        ("lat", "lon"),
        np.full((2, 2), v),
        {"units": "mg/m^3"},
    )
    return xr.Dataset(
        {"a": field(1.0), "b": field(2.0), "c": field(4.0)}, coords=coords
    )


def test_an_inline_combination_needs_no_registration(abc):
    """Registration is for naming and reuse, never a precondition."""
    assert float(resolve_variable(abc, {"sum": ["a", "b"]}).mean()) == 3.0


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ({"ratio": [{"sum": ["a", "b"]}, "c"]}, 0.75),  # (a+b)/c
        ({"difference": [{"sum": ["a", "b"]}, "c"]}, -1.0),  # (a+b)-c
        ({"sum": ["a", {"product": ["b", "c"]}]}, 9.0),  # a+(b*c)
    ],
)
def test_combinations_nest(abc, spec, expected):
    """Compound arithmetic should not require dropping to a registered function."""
    assert float(resolve_variable(abc, spec).mean()) == pytest.approx(expected)


def test_a_missing_leaf_in_a_nested_spec_is_caught(abc):
    with pytest.warns(UserWarning, match="missing"):
        assert resolve_variable(abc, {"ratio": [{"sum": ["a", "nope"]}, "c"]}) is None


def test_spec_names_flattens_a_nested_spec_to_its_leaves():
    """A catalog can only be checked against real variable names."""
    from ocean_skill.operators import spec_names

    assert spec_names({"ratio": [{"sum": ["a", "b"]}, "c"]}) == [["a", "b", "c"]]


def test_a_genuinely_custom_formula_still_needs_registering(abc):
    """Arithmetic is data; a real formula (MLD, EKE) is code and says so."""
    with pytest.raises(NotImplementedError, match="register a calculator"):
        resolve_variable(abc, {"calculate": "mld"})


def test_compare_pairs_each_variable_with_the_stream_that_has_it():
    """A ROMS run writes physics and BGC separately, on different time axes.

    They cannot be one source (GOM_his is 15-minute, GOM_bgc daily), so both are
    passed as `test=` and each variable must find the stream carrying it. Tests used
    to be exempt from this filtering while references were not, so half the
    cross-product was attempted and failed at read time.
    """
    from unittest import mock

    from ocean_skill import comparison

    temp = "sea_water_potential_temperature"
    nitrate = "mole_concentration_of_nitrate_in_sea_water"
    declared = {
        "GOM_his": {"variables": [temp]},
        "GOM_bgc": {"variables": [nitrate]},
        "woa_t": {"variables": [temp]},
        "woa_n": {"variables": [nitrate]},
    }
    formed = []

    def record(self, refresh=False):
        formed.append((self.test_name, self.reference_name, self.variable))

    with (
        mock.patch(
            "ocean_skill.catalog.resolve",
            lambda n: mock.Mock(metadata=declared[n]),
        ),
        mock.patch.object(comparison.Comparison, "align", record),
    ):
        comparison.compare(
            reference=["woa_t", "woa_n"],
            test=["GOM_his", "GOM_bgc"],
            variables=[temp, nitrate],
        )

    assert ("GOM_his", "woa_t", temp) in formed
    assert ("GOM_bgc", "woa_n", nitrate) in formed
    assert len(formed) == 2, f"expected no mismatched pairs, got {formed}"


def test_a_combination_over_raw_names_is_never_filtered_out():
    """Catalogs index CF standard_names, so they cannot speak to `spChl` at all.

    Regression: filtering tests by declared variables silently dropped every
    chlorophyll comparison, because ROMS' component chlorophylls are deliberately
    unmapped and so never appear in a catalog's `variables` list. "Absent" and
    "unknowable" have to be different answers.
    """
    from unittest import mock

    from ocean_skill import comparison

    nitrate = "mole_concentration_of_nitrate_in_sea_water"
    declared = {
        "GOM_bgc": {"variables": [nitrate]},  # no chlorophyll listed at all
        "modis": {"variables": ["mass_concentration_of_chlorophyll_in_sea_water"]},
    }
    formed = []
    with (
        mock.patch(
            "ocean_skill.catalog.resolve",
            lambda n: mock.Mock(metadata=declared[n]),
        ),
        mock.patch.object(
            comparison.Comparison,
            "align",
            lambda self, refresh=False: formed.append(self.test_name),
        ),
    ):
        comparison.compare(
            reference=["modis"],
            test=["GOM_bgc"],
            variables=[{"sum": ["spChl", "diatChl"], "standard_name": CHL}],
        )
    assert formed == ["GOM_bgc"], "a raw-name combination must reach the read"


def test_a_plain_cf_name_the_source_lacks_is_still_filtered():
    """The conservative rule must not disable filtering for names catalogs do index."""
    from unittest import mock

    from ocean_skill import comparison

    nitrate = "mole_concentration_of_nitrate_in_sea_water"
    declared = {
        "modis": {"variables": ["mass_concentration_of_chlorophyll_a_in_sea_water"]}
    }
    formed = []
    with (
        mock.patch(
            "ocean_skill.catalog.resolve",
            lambda n: mock.Mock(metadata=declared[n]),
        ),
        mock.patch.object(
            comparison.Comparison,
            "align",
            lambda self, refresh=False: formed.append(self.test_name),
        ),
    ):
        comparison.compare(reference=["modis"], test=["modis"], variables=[nitrate])
    assert formed == []
