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
import pandas as pd
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


def test_select_snaps_a_full_timestamp_to_the_nearest_step(two_months):
    """An instant behaves like a scalar: output is stamped at offsets the caller
    cannot be expected to know, so 02:00 on midnight-stamped data means the
    nearest step, not a KeyError."""
    got = select(two_months, {"time": "2012-01-30T02:00:00"})
    assert pd.Timestamp(got.time.values) == pd.Timestamp("2012-01-30")


def test_select_snaps_a_missing_day_on_daily_data(two_months):
    """A date at the axis's own resolution is one step's worth of time — an
    instant — even without the clock part; five-daily output between steps."""
    gappy = two_months.isel(time=slice(None, None, 5))  # Jan 1, 6, 11, ...
    got = select(gappy, {"time": "2012-01-04"})
    assert pd.Timestamp(got.time.values) == pd.Timestamp("2012-01-06")


def test_select_still_refuses_an_empty_period(two_months):
    """'2013-01' is a period, not an instant: its nearest neighbour would be
    data from a month the caller did not name."""
    with pytest.raises(KeyError, match="no data within '2013-01'"):
        select(two_months, {"time": "2013-01"})


def test_a_day_of_hourly_data_is_a_period_not_an_instant():
    """On hourly data a bare date names twenty-four steps; when the record skips
    the day entirely, snapping to a neighbouring day would misrepresent."""
    time = xr.date_range("2012-01-01", periods=48, freq="h")
    hourly = xr.DataArray(np.arange(48.0), dims="time", coords={"time": time})
    with pytest.raises(KeyError, match="no data within '2012-01-05'"):
        select(hourly, {"time": "2012-01-05"})


def test_a_method_key_warns_instead_of_vanishing(two_months):
    """xarray's own KeyError advises method='nearest'; a caller who follows that
    advice into the spec should hear why it is not a key here — and still get
    the selection they meant, since nearest is automatic."""
    with pytest.warns(UserWarning, match="Nearest matching is automatic"):
        got = select(
            two_months, {"time": "2012-01-30T02:00:00", "method": "nearest"}
        )
    assert pd.Timestamp(got.time.values) == pd.Timestamp("2012-01-30")


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


def test_an_unregistered_formula_still_needs_registering(abc):
    """Arithmetic is data; a real formula (MLD, EKE) is code and says so."""
    with pytest.raises(NotImplementedError, match="register a calculator"):
        resolve_variable(abc, {"calculate": "eke"})


def test_mld_is_registered_and_dispatches_by_method():
    """The formula this whole mechanism was designed for is no longer a stub.

    Exercised through resolve_variable rather than against `abc` (2-D, no vertical
    axis): a real calculator needs the water column ocean_skill.mld.mld.py's own
    tests build, so this only checks that {"calculate": "mld"} reaches
    ocean_skill.mld.calculate_mld rather than raising -- see tests/test_mld.py for
    the formula itself.
    """
    import ocean_skill.mld  # noqa: F401  (registers CALCULATORS["mld"])
    from ocean_skill.operators import CALCULATORS

    assert "mld" in CALCULATORS
    # calculate_mld raises before ever touching `ds` when method is missing, so
    # None stands in for "a dataset" here -- the water-column fixture this formula
    # actually needs lives in tests/test_mld.py.
    with pytest.raises(KeyError, match="mld needs method"):
        resolve_variable(None, {"calculate": "mld"})


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


# --------------------------------------------------------- ranges on a descending axis
#
# ``.sel`` with a slice follows the coordinate's *stored* order, so a range written
# low-to-high against a north-to-south axis selects nothing — silently. Satellite L3
# products are all stored that way (MODIS latitude runs 89.979 to -89.979), so this was
# not an edge case: every `select={"lat": ...}` against one came back empty.


def _gridded(lat, lon=(-98.0, -96.0, -94.0, -92.0)):
    """Build a field on the given latitude order, letting a test pick the direction."""
    lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    return xr.DataArray(
        np.arange(lat.size * lon.size, dtype=float).reshape(lat.size, lon.size),
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
    )


NORTH_TO_SOUTH = np.arange(40.0, 9.0, -5.0)  # 40, 35, 30, 25, 20, 15, 10
SOUTH_TO_NORTH = NORTH_TO_SOUTH[::-1]


@pytest.mark.parametrize(
    "spec",
    [{"min": 20, "max": 30}, slice(20, 30)],
    ids=["dict", "slice"],
)
@pytest.mark.parametrize(
    ("order", "name"),
    [(NORTH_TO_SOUTH, "descending"), (SOUTH_TO_NORTH, "ascending")],
    ids=["descending", "ascending"],
)
def test_a_range_selects_the_same_band_whichever_way_the_axis_runs(spec, order, name):
    picked = select(_gridded(order), {"lat": spec})
    assert sorted(picked["lat"].values) == [20.0, 25.0, 30.0], name


def test_bounds_written_backwards_keep_working_on_a_descending_axis():
    """The workaround people already had for this must not become the new bug.

    Anyone who discovered the empty result and reversed their bounds to fix it had
    working code; normalizing the range rather than blindly flipping it keeps both
    spellings meaning the same thing.
    """
    picked = select(_gridded(NORTH_TO_SOUTH), {"lat": slice(30, 20)})
    assert sorted(picked["lat"].values) == [20.0, 25.0, 30.0]


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ({"min": 30}, [30.0, 35.0, 40.0]),
        ({"max": 20}, [10.0, 15.0, 20.0]),
    ],
    ids=["min-only", "max-only"],
)
def test_a_one_sided_bound_flips_too(spec, expected):
    picked = select(_gridded(NORTH_TO_SOUTH), {"lat": spec})
    assert sorted(picked["lat"].values) == expected


def test_a_descending_time_axis_is_handled_the_same_way():
    """Not only latitude: some products are written newest-first.

    Note the bounds here are timestamps rather than the partial-date strings
    ``select`` accepts elsewhere. Partial-string slicing goes through pandas, which
    refuses it outright on a decreasing DatetimeIndex — a limitation upstream of this
    orientation fix, not something it can reach.
    """
    times = xr.date_range("2012-01-01", periods=10, freq="D")[::-1]
    field = xr.DataArray(
        np.arange(10, dtype=float), dims=("time",), coords={"time": times}
    )
    picked = select(field, {"time": {"min": times[6], "max": times[4]}})
    assert picked.sizes["time"] == 3


def test_a_range_against_a_dimension_with_no_coordinate_is_refused():
    """``.sel`` would fall back to positional indexing and answer a different question.

    ROMS is what makes this reachable: it ships no coordinate variables, so ``s_rho``
    and an undecoded ``time`` are bare dimensions. Asking for 0-50 m and getting cells
    0 and 1 is a wrong answer that looks like a right one.
    """
    field = xr.DataArray(np.zeros((3, 40)), dims=("time", "s_rho"))
    with pytest.raises(ValueError) as excinfo:
        select(field, {"s_rho": {"min": 0, "max": 50}})
    message = str(excinfo.value)
    assert "no coordinate values" in message
    assert "positional" in message, "say what would have happened instead"
    assert ".isel()" in message, "and how to ask for positions on purpose"


def test_a_scalar_against_a_bare_dimension_is_still_allowed():
    """Only ranges are refused: "the first level" is how xarray reads a scalar."""
    field = xr.DataArray(np.arange(12.0).reshape(3, 4), dims=("time", "s_rho"))
    assert select(field, {"s_rho": 0}).sizes == {"time": 3}


def test_a_single_valued_axis_has_no_direction_to_get_wrong():
    field = _gridded([25.0])
    assert select(field, {"lat": {"min": 20, "max": 30}}).sizes["lat"] == 1


# -- bbox longitude conventions -----------------------------------------------


def _lonlat_grid(lons, lats):
    import numpy as np
    import xarray as xr

    return xr.Dataset(
        {"sst": (("latitude", "longitude"), np.zeros((len(lats), len(lons))))},
        coords={"latitude": lats, "longitude": lons},
    )


def test_a_0_360_bbox_crops_a_180_reference():
    """A global reference must not read as "no overlap" against a 0-360 test.

    MUR is on +/-180 and a North Pacific model on 0-360; cropping the first to the
    second's box asked for longitudes 190-250 on an axis stopping at 180, and the
    empty slice was reported as the two sources not overlapping.
    """
    import numpy as np

    from ocean_skill.align import subset_to_bbox

    ref = _lonlat_grid(np.arange(-180, 181, 1.0), np.arange(-90, 91, 1.0))
    out = subset_to_bbox(ref, (190.268, 13.2256, 250.771, 65.8473))
    assert out.sizes["longitude"] > 0
    assert out.longitude.min() < -100 and out.longitude.max() < 0


def test_a_180_bbox_crops_a_0_360_reference():
    import numpy as np

    from ocean_skill.align import subset_to_bbox

    ref = _lonlat_grid(np.arange(0, 360, 1.0), np.arange(-90, 91, 1.0))
    out = subset_to_bbox(ref, (-169.7, 13.2, -109.2, 65.8))
    assert out.longitude.min() >= 180


def test_a_bbox_already_in_the_same_convention_is_untouched():
    import numpy as np

    from ocean_skill.align import subset_to_bbox

    ref = _lonlat_grid(np.arange(-180, 181, 1.0), np.arange(-90, 91, 1.0))
    out = subset_to_bbox(ref, (10.0, -5.0, 50.0, 20.0))  # valid in both conventions
    assert float(out.longitude.min()) == 9.0
    assert float(out.longitude.max()) == 51.0


def test_a_bbox_straddling_the_seam_says_so_rather_than_dropping_half():
    """Contiguous in the box's convention, split in the reference's.

    Slicing either half would quietly compare against a fragment of the region.
    """
    import numpy as np
    import pytest

    from ocean_skill.align import subset_to_bbox

    ref = _lonlat_grid(np.arange(-180, 181, 1.0), np.arange(-90, 91, 1.0))
    with pytest.raises(ValueError, match="cannot be cropped to one slice"):
        subset_to_bbox(ref, (170.0, -5.0, 190.0, 5.0))


# -- cropping the reference to the test's time window -------------------------


def _daily_maps(start, periods, freq="D"):
    import numpy as np
    import pandas as pd
    import xarray as xr

    return xr.DataArray(
        np.zeros((periods, 4, 5)),
        dims=("time", "latitude", "longitude"),
        coords={
            "time": pd.date_range(start, periods=periods, freq=freq),
            "latitude": np.arange(4.0),
            "longitude": np.arange(5.0),
        },
    )


def test_the_reference_is_cropped_to_the_test_window():
    """Cropping the region but not the window still reads the whole record.

    MUR over a regional model's footprint is a workable map per step, and 2.2 TB
    across its 8838 daily ones.
    """
    from ocean_skill.align import subset_to_time, time_span_of

    ref = _daily_maps("2012-01-01", 1096)
    test = _daily_maps("2013-06-15", 3, freq="MS")
    out = subset_to_time(ref, time_span_of(test))
    assert out.sizes["time"] < 200  # a season, not three years
    assert out.time.values[0] < test.time.values[0]  # padded on both sides
    assert out.time.values[-1] > test.time.values[-1]


def test_a_single_time_test_gets_no_pad():
    """There is no step to measure, so the span is the instant itself."""
    from ocean_skill.align import time_span_of

    lo, hi = time_span_of(_daily_maps("2013-06-15", 3, freq="MS").isel(time=[0]))
    assert lo == hi


def test_a_source_with_no_time_axis_is_left_alone():
    import numpy as np
    import xarray as xr

    from ocean_skill.align import subset_to_time, time_span_of

    static = xr.DataArray(
        np.zeros((4, 5)),
        dims=("latitude", "longitude"),
        coords={"latitude": np.arange(4.0), "longitude": np.arange(5.0)},
    )
    assert time_span_of(static) is None
    window = time_span_of(_daily_maps("2013-06-15", 3, freq="MS"))
    assert subset_to_time(static, window).shape == static.shape


def test_a_reference_sharing_no_span_is_kept_not_emptied():
    """A climatology legitimately shares no calendar span with the test.

    Unlike the bbox crop this is not an error: the comparison's own time handling
    reports it far more precisely than a crop can.
    """
    from ocean_skill.align import subset_to_time, time_span_of

    clim = _daily_maps("1990-01-01", 10)
    window = time_span_of(_daily_maps("2013-06-15", 3, freq="MS"))
    assert subset_to_time(clim, window).sizes["time"] == 10
