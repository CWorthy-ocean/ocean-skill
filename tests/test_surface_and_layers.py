"""``"surface"`` only where there is a surface; layers of one-instrument moorings.

``"surface"`` is a keyword for the model, for gridded datasets, and for point datasets
that are themselves *at* the surface (a surface buoy, a drifter). A cast, a repeat-visit
station and a mooring whose instrument hangs at depth have no surface measurement, so
asking them for one is an error that names what they do have -- or, in a ``compare()``
fan with other references, a warning and a skipped pair.

A depth *band* (a layer, 0-50 m say) of a one-instrument mooring is its one level if the
instrument lies inside it, and nothing if it does not -- so a layer pooled over several
moorings (``ComparisonSet.average(by="variable")``) can be built from each mooring's
saved basic comparison, without reading the model again.

These run the real ``Comparison.align`` pipeline against a small synthetic ROMS grid
(``tests/_tidal_roms.py``), with only ``osk.read``/``catalog.resolve`` stubbed (see
``tests/test_compare_from_pairs.py``, whose fixtures this reuses).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ocean_skill import cache, comparison
from ocean_skill.align import TIME_DEPTH_OVER
from ocean_skill.comparison import (
    SURFACE_DEPTH_TOLERANCE_M,
    Comparison,
    _instrument_depth,
    _reference_at_surface,
    compare,
)
from tests._tidal_roms import tidal_roms
from tests.test_compare_from_pairs import (
    NT,
    POINT,
    START,
    TEMPERATURE,
    World,
    _ref_meta,
    _values,
)


def _deep_model() -> tuple[xr.Dataset, dict]:
    """Build a 120 m deep ROMS model (2 m layers) whose temperature falls with depth."""
    ds, meta = tidal_roms(zeta=2.0 * np.sin(np.arange(NT) / 6.0), h=120.0, n=60)
    ds = ds.assign_coords(time=ds["time"].values.astype("datetime64[ns]"))
    hours = xr.DataArray(np.arange(NT), dims="time", coords={"time": ds["time"]})
    ds[TEMPERATURE] = (15.0 + 0.1 * ds["height"] + 0.01 * hours).assign_attrs(
        units="degC", standard_name=TEMPERATURE
    )
    return ds.drop_vars(["level", "height"]), meta


def _series(depth: float, phase: float = 0.0) -> pd.DataFrame:
    """Return an hourly mooring record at one instrument depth."""
    times = pd.date_range(START, periods=NT, freq="h")
    wave = 13.0 - 0.05 * depth + np.sin(np.arange(NT) / 5.0 + phase)
    return pd.DataFrame(
        {
            "time": times,
            "lon": POINT[0],
            "lat": POINT[1],
            "depth (m)": depth,
            "temperature (degC)": wave,
        }
    )


def _station_meta(depth: float | None, **extra) -> dict:
    if depth is None:
        return _ref_meta("timeSeries", **extra)
    return _ref_meta(
        "timeSeries",
        geospatial_vertical_min=depth,
        geospatial_vertical_max=depth,
        **extra,
    )


class SurfaceWorld(World):
    """The sibling tests' stubbed catalog on a deeper model, plus more stations."""

    def __init__(self, monkeypatch):
        super().__init__(monkeypatch)
        self.sources["his"] = _deep_model()
        for name, depth, phase in [
            ("buoy", 0.5, 0.0),  # a surface buoy: declared extent within 1 m of the top
            ("m10", 10.0, 0.3),
            ("m30", 30.0, 0.6),
            ("m80", 80.0, 0.9),
        ]:
            self.sources[name] = (_series(depth, phase), _station_meta(depth))
        # metadata-only variants (never read): what the at-surface helper looks at
        self.sources["edge"] = (None, _station_meta(1.0))
        self.sources["just_below"] = (None, _station_meta(1.01))
        self.sources["flagged"] = (
            None,
            _station_meta(10.0, depth_convention={"support": "surface"}),
        )
        self.sources["undeclared"] = (None, _station_meta(None))
        self.sources["wide"] = (
            None,
            _ref_meta(
                "timeSeries",
                geospatial_vertical_min=0.0,
                geospatial_vertical_max=200.0,
            ),
        )
        self.sources["pressure_buoy"] = (
            None,
            _station_meta(-0.5, depth_convention={"positive": "up"}),
        )


@pytest.fixture
def world(monkeypatch):
    saved = cache._verbose
    cache.verbose(False)
    yield SurfaceWorld(monkeypatch)
    cache._verbose = saved


def _align(reference: str, **kw) -> xr.Dataset:
    """Align one comparison of ``reference`` against the model and return its pair."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = Comparison(reference=reference, test="his", variable=TEMPERATURE, **kw)
        return c.align()


def _compare(reference, **kw):
    kw.setdefault("variables", [TEMPERATURE])
    return compare(reference=reference, test="his", **kw)


# -- which references are "at the surface" -------------------------------------------


def test_the_surface_tolerance_is_one_metre():
    assert SURFACE_DEPTH_TOLERANCE_M == 1.0


@pytest.mark.parametrize(
    "name, at_surface",
    [
        ("buoy", True),  # declared at 0.5 m
        ("edge", True),  # exactly 1.0 m is within the tolerance
        ("just_below", False),
        ("m10", False),
        ("flagged", True),  # depth_convention.support == "surface", whatever the extent
        ("undeclared", False),  # unknown is not "at the surface"...
        ("cast", False),  # ...and a cast never is
        ("visits", False),
        ("his", False),  # a gridded dataset is not asked
    ],
)
def test_which_references_are_at_the_surface(world, name, at_surface):
    assert _reference_at_surface(name) is at_surface


def test_a_height_declared_positive_up_is_read_through_its_convention(world):
    # -0.5 with positive: up is a depth of 0.5 m
    assert _reference_at_surface("pressure_buoy")
    assert _instrument_depth("pressure_buoy") == 0.0


def test_the_instrument_depth_is_what_the_basic_comparison_is_made_at(world):
    assert _instrument_depth("buoy") == 0.0  # at the surface counts as depth 0
    assert _instrument_depth("m10") == 10.0
    assert _instrument_depth("moor") == 5.0
    assert _instrument_depth("undeclared") is None
    assert _instrument_depth("wide") is None  # really a profiler: no single level
    assert _instrument_depth("cast") is None  # not a fixed station


# -- "surface" of what has none -------------------------------------------------------


def test_a_cast_has_no_surface_and_says_what_it_has(world):
    with pytest.raises(ValueError) as err:
        _align("cast", select={"depth": "surface"})
    assert str(err.value) == (
        "'cast' has no surface measurement -- its shallowest level is 2 m. "
        "Pass depths=[2] or a range like {'min': 0, 'max': 5}."
    )
    assert world.model_reads == 0  # refused before the model is read


def test_surface_inside_a_depth_list_is_refused_too(world):
    with pytest.raises(ValueError, match="no surface measurement"):
        _align("cast", select={"depth": ["surface", 5.0]})
    with pytest.raises(ValueError, match="no surface measurement"):
        _align("visits", select={"depth": ["surface", 6.0]})


def test_a_repeat_visit_station_has_no_surface_either(world):
    with pytest.raises(ValueError, match="shallowest level is 1 m"):
        _align("visits", select={"depth": "surface"})


def test_a_mooring_at_depth_has_no_surface(world):
    with pytest.raises(ValueError) as err:
        _align("m30", select={"depth": "surface"})
    assert str(err.value) == (
        "'m30' has no surface measurement -- its instrument is at 30 m. "
        "Pass depths=[30] or a range like {'min': 0, 'max': 33}."
    )


def test_the_surface_default_is_not_used_where_there_is_no_surface(world):
    # nothing asked at all of a cast: compare() fills in the cast's own levels, and so
    # does a Comparison built directly
    direct = _align("cast")
    assert direct["reference"].sizes[next(iter(direct["reference"].dims))] == 4
    assert direct.attrs["derived_from"] == "lanes"
    np.testing.assert_allclose(_values(direct, "reference"), [10.0, 9.0, 8.0, 7.0])


def test_a_mooring_defaults_to_its_own_depth_not_the_surface(world):
    with pytest.warns(UserWarning, match="sits at ~30 m"):
        c = Comparison(reference="m30", test="his", variable=TEMPERATURE)
        c.align()
    assert c.select == {"depth": 30.0}
    assert c.aligned.attrs["derived_from"] == "lanes"  # it is the basic comparison


def test_a_direct_default_is_the_comparison_compare_makes(world):
    # the same entry: the direct comparison's cache hit is the compare() one
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        (c,) = list(_compare("m30"))
        c.align()
    reads = world.model_reads
    assert reads >= 1
    world.block_model()
    direct = _align("m30")
    assert world.model_reads == reads
    xr.testing.assert_equal(direct["test"], c.aligned["test"])


def test_a_direct_repeat_visit_default_keeps_both_axes(world):
    c = Comparison(reference="visits", test="his", variable=TEMPERATURE)
    assert c.over == TIME_DEPTH_OVER  # decided as if the default were already there
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        aligned = c.align()
    assert c.over == TIME_DEPTH_OVER  # ...and it does not move when it is filled in
    assert c.select["depth"] == [1.0, 3.0, 6.0, 8.0, 12.0]
    assert aligned["reference"].sizes["time"] == 3


def test_an_explicit_over_on_a_station_with_no_depth_is_refused(world):
    # the caller named the axis, so there is no own-levels default to fall back on
    with pytest.raises(ValueError, match="no surface measurement"):
        _align("visits", over="time")


# -- in a compare() fan ---------------------------------------------------------------


def test_a_cast_is_left_out_of_a_fan_not_fatal(world):
    with pytest.warns(UserWarning, match="left out 'cast'") as record:
        out = _compare(["cast", "buoy"], depths=["surface"])
    assert [c.reference_name for c in out] == ["buoy"]
    text = " ".join(str(w.message) for w in record)
    assert "no surface measurement" in text and "shallowest level is 2 m" in text


def test_a_deep_mooring_is_left_out_of_a_surface_fan(world):
    with pytest.warns(UserWarning, match="left out 'm30'"):
        out = _compare(["m30", "buoy"], depths=["surface"])
    assert [c.reference_name for c in out] == ["buoy"]


def test_the_default_depth_of_a_fan_leaves_out_nothing_that_has_one(world):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = _compare(["cast", "m10", "buoy"])
    assert sorted(c.reference_name for c in out) == ["buoy", "cast", "m10"]
    buoy = next(c for c in out if c.reference_name == "buoy")
    assert buoy.select == {"depth": "surface"}  # not the 0.5 m midpoint


def test_depths_with_a_surface_among_levels_leaves_out_only_the_surface(world):
    with pytest.warns(UserWarning, match="left out 'm10'"):
        out = _compare(["m10", "buoy"], depths=["surface", 10])
    got = sorted((c.reference_name, str(c.select["depth"])) for c in out)
    assert got == [("buoy", "10"), ("buoy", "surface"), ("m10", "10")]


def test_nothing_left_after_the_fan_is_an_error_not_an_empty_set(world):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="no surface measurement"):
            _compare("cast", depths=["surface"])
        with pytest.raises(ValueError, match="no surface measurement"):
            _compare(["cast", "visits"], depths=["surface"])


# -- a station at the surface -------------------------------------------------------


def test_a_surface_buoy_compares_the_models_top_cell(world):
    ds = _align("buoy", select={"depth": "surface"})
    assert ds.attrs["derived_from"] == "lanes"  # it *is* the basic comparison
    assert ds.sizes["time"] == NT
    # no depth named at all is the same comparison, the same entry
    again = _align("buoy")
    xr.testing.assert_equal(again["test"], ds["test"])


@pytest.mark.parametrize(
    "variant",
    [
        {
            "select": {
                "depth": "surface",
                "time": {"min": "2024-07-03", "max": "2024-07-05"},
            }
        },
        {
            "select": {"depth": "surface"},
            "aggregate": {"time": {"resample": "1D", "reduce": "mean"}},
        },
        {"select": {"depth": "surface"}, "detide": True},
        {"select": {"depth": "surface", "time": "2024-07-04"}},
    ],
    ids=["time-slice", "daily-mean", "detide", "one-day"],
)
def test_a_surface_buoys_time_variants_never_read_the_model(world, variant):
    base = _align("buoy", select={"depth": "surface"})
    reads = world.model_reads
    assert reads >= 1

    world.block_model()
    ds = _align("buoy", **variant)

    assert ds.attrs["derived_from"] == "pairs"
    assert world.model_reads == reads
    # the derived series is the basic pairs, sliced or averaged
    if "time" in variant["select"] and isinstance(variant["select"]["time"], dict):
        window = base.sel(time=slice("2024-07-03", "2024-07-05T23"))
        np.testing.assert_allclose(_values(ds), _values(window))


def test_a_surface_buoys_derived_variant_says_so_in_the_cache_lines(world, capsys):
    cache.verbose(True)
    window = {"depth": "surface", "time": {"min": "2024-07-03", "max": "2024-07-05"}}
    _align("buoy", select=window)
    assert cache.SAVED_PAIRS in capsys.readouterr().out
    _align(
        "buoy", select={**window, "time": {"min": "2024-07-04", "max": "2024-07-06"}}
    )
    assert cache.USED_PAIRS in capsys.readouterr().out


def test_a_buoy_flagged_as_surface_by_its_convention_is_not_refused(world, monkeypatch):
    # an explicit declaration beats the extent: 10 m of cable, the sensor is the skin
    world.sources["flagged"] = (_series(10.0), world.sources["flagged"][1])
    ds = _align("flagged", select={"depth": "surface"})
    assert ds.sizes["time"] == NT


# -- a layer of single-instrument moorings -------------------------------------------

BAND = {"min": 0, "max": 50}


def test_a_band_holding_the_instrument_is_the_basic_pairs(world):
    basic = _align("m10", select={"depth": 10.0})
    reads = world.model_reads
    world.block_model()

    layer = _align("m10", select={"depth": BAND}, aggregate={"Z": "mean"})

    assert world.model_reads == reads
    assert layer.attrs["derived_from"] == "pairs"
    assert layer.attrs["actual_depth"] == 10.0
    assert layer.attrs["depth_band"] == [0, 50]
    for name in ("test", "reference", "difference"):
        np.testing.assert_allclose(_values(layer, name), _values(basic, name))


def test_a_band_without_a_vertical_mean_is_the_same_layer(world):
    layer = _align("m10", select={"depth": BAND})
    assert layer.attrs["derived_from"] == "pairs"
    assert layer.attrs["depth_band"] == [0, 50]


def test_a_band_with_a_time_variant_derives_both(world):
    _align("m10", select={"depth": 10.0})
    world.block_model()
    window = {"min": "2024-07-03", "max": "2024-07-05"}
    layer = _align(
        "m10",
        select={"depth": BAND, "time": window},
        aggregate={"Z": "mean"},
    )
    assert layer.attrs["derived_from"] == "pairs"
    assert layer.attrs["depth_band"] == [0, 50]
    assert layer.sizes["time"] < NT


def test_a_band_the_instrument_is_outside_has_no_data_and_raises(world):
    with pytest.raises(ValueError) as err:
        _align("m80", select={"depth": BAND}, aggregate={"Z": "mean"})
    assert "'m80'" in str(err.value)
    assert "its instrument at 80 m is outside it" in str(err.value)
    assert "0–50 m layer" in str(err.value)
    assert world.model_reads == 0


def test_the_band_edges_are_inclusive(world):
    ds = _align("m10", select={"depth": {"min": 10, "max": 20}})
    assert ds.attrs["depth_band"] == [10, 20]
    with pytest.raises(ValueError, match="outside"):
        _align("m10", select={"depth": {"min": 10.5, "max": 20}})


def test_a_surface_buoy_is_inside_a_layer_from_the_top(world):
    layer = _align("buoy", select={"depth": BAND}, aggregate={"Z": "mean"})
    assert layer.attrs["derived_from"] == "pairs"
    assert layer.attrs["actual_depth"] == 0.0
    surface = _align("buoy", select={"depth": "surface"})
    np.testing.assert_allclose(_values(layer), _values(surface))
    # ...but not inside a layer that starts below it
    with pytest.raises(ValueError, match="outside"):
        _align("buoy", select={"depth": {"min": 5, "max": 50}})


def test_a_layer_of_an_undeclared_mooring_is_not_derived_from_its_one_level(world):
    # nothing says where the instrument is, so the layer cannot be answered from it
    # (and the comparison is refused before it gets that far: see the undeclared tests)
    world.sources["undeclared"] = (_series(10.0), world.sources["undeclared"][1])
    c = Comparison(
        reference="undeclared",
        test="his",
        variable=TEMPERATURE,
        select={"depth": BAND},
        aggregate={"Z": "mean"},
    )
    assert c._derivation_plan().basic is None
    assert "not declared" in c._derivation_plan().reason


def test_a_mooring_outside_the_layer_is_left_out_of_a_fan(world):
    with pytest.warns(UserWarning, match="left out 'm80'") as record:
        out = _compare(["m10", "m30", "m80"], depths=[BAND], aggregate={"Z": "mean"})
    assert sorted(c.reference_name for c in out) == ["m10", "m30"]
    assert any("its instrument at 80 m is outside" in str(w.message) for w in record)


def test_a_layer_pooled_over_several_moorings_comes_from_the_cache(world):
    request = dict(
        reference=["m10", "m30", "m80"],
        variables=[TEMPERATURE],
        depths=[BAND],
        aggregate={"Z": "mean"},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first = compare(test="his", **request)
    assert sorted(c.reference_name for c in first) == ["m10", "m30"]
    for c in first:
        c.align()
        assert c.aligned.attrs["derived_from"] == "pairs"
        assert c.aligned.attrs["depth_band"] == [0, 50]
    reads = world.model_reads
    assert reads >= 2

    # the second call, with the model unreadable: every mooring answers from its saved
    # basic comparison
    world.block_model()
    with pytest.warns(UserWarning, match="left out 'm80'"):
        second = compare(test="his", **request)
    assert sorted(c.reference_name for c in second) == ["m10", "m30"]
    for c in second:
        c.align()
    assert world.model_reads == reads

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pooled = second.average(by="variable")
    assert len(pooled) == 1
    members = [c for c in second]
    expected = np.mean([_values(c.aligned) for c in members], axis=0)
    got = _values(next(iter(pooled)).aligned)
    assert np.isfinite(got).all()
    np.testing.assert_allclose(got, expected, rtol=1e-6)


def test_every_mooring_outside_the_layer_is_an_error(world):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="outside"):
            _compare(["m80"], depths=[BAND], aggregate={"Z": "mean"})


def test_a_layer_between_a_casts_levels_is_empty_not_the_nearest_level(world):
    # a cast has nothing in a layer none of its levels (2, 5, 9, 14 m) falls in: the
    # nearest level would silently pool an out-of-range depth into the layer's mean
    with pytest.raises(ValueError) as err:
        _align(
            "cast",
            select={"depth": [{"min": 20.0, "max": 30.0}]},
            aggregate={"Z": "mean"},
        )
    assert str(err.value) == (
        "'cast' has no data in the 20–30 m layer -- its nearest levels are "
        "5, 9, 14 m. Pass a layer that holds one of them."
    )
    assert world.model_reads == 0  # refused before the model is read
    assert comparison._single_band({"min": 3, "max": 4}) == (3.0, 4.0)


def test_comparison_helpers_read_nothing(world):
    # the at-surface test and the layer test are catalog-metadata only
    world.block_model()
    assert _reference_at_surface("buoy")
    assert comparison._depth_request_problem("m80", BAND) is not None
    assert comparison._depth_request_problem("m10", BAND) is None
    assert comparison._depth_request_problem("m10", "surface") is not None
    assert comparison._depth_request_problem("buoy", "surface") is None
    assert comparison._depth_request_problem("his", "surface") is None


# -- a mooring that declares no depth is refused "surface" -----------------------------

DECLARE = (
    "Declare its instrument depth in the catalog entry (nominal_depth_m, "
    "geospatial_vertical_min/max, or depth_convention: {support: surface} for a "
    "surface instrument), or pass an explicit depth such as depths=[10]."
)


@pytest.fixture
def undeclared_world(world):
    """Put real data behind the mooring whose catalog declares no depth."""
    world.sources["undeclared"] = (_series(10.0, 0.2), world.sources["undeclared"][1])
    return world


@pytest.mark.parametrize(
    "request_",
    [
        {"select": {"depth": "surface"}},
        {"select": {"depth": ["surface"]}},
        {},  # the unset default is "surface" too
    ],
    ids=["scalar", "in-a-list", "unset"],
)
def test_surface_of_an_undeclared_mooring_is_refused_until_a_depth_is_declared(
    undeclared_world, request_
):
    with pytest.raises(ValueError) as err:
        _align("undeclared", **request_)
    assert str(err.value) == (
        "'undeclared' declares no instrument depth, so it is not known to measure "
        f"the surface. {DECLARE}"
    )
    assert undeclared_world.model_reads == 0


def test_an_undeclared_mooring_is_left_out_of_a_fan_not_fatal(undeclared_world):
    with pytest.warns(UserWarning, match="left out 'undeclared'") as record:
        out = _compare(["undeclared", "buoy"], depths=["surface"])
    assert [c.reference_name for c in out] == ["buoy"]
    text = " ".join(str(w.message) for w in record)
    assert "declares no instrument depth" in text and "geospatial_vertical" in text
    # the unset default of a fan is the same refusal -- no silent surface comparison
    with pytest.warns(UserWarning, match="left out 'undeclared'"):
        out = _compare(["undeclared", "buoy"])
    assert [c.reference_name for c in out] == ["buoy"]


def test_an_undeclared_mooring_alone_in_a_fan_is_an_error(undeclared_world):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="declares no instrument depth"):
            _compare("undeclared")


def test_a_layer_of_an_undeclared_mooring_cannot_be_placed_and_is_refused(
    undeclared_world,
):
    with pytest.raises(ValueError) as err:
        _align("undeclared", select={"depth": BAND}, aggregate={"Z": "mean"})
    assert str(err.value) == (
        "'undeclared' declares no instrument depth, so it cannot be placed in or out "
        f"of the 0–50 m layer. {DECLARE}"
    )
    with pytest.warns(UserWarning, match="left out 'undeclared'"):
        out = _compare(["undeclared", "m10"], depths=[BAND], aggregate={"Z": "mean"})
    assert [c.reference_name for c in out] == ["m10"]


def test_an_explicit_depth_of_an_undeclared_mooring_still_works(undeclared_world):
    ds = _align("undeclared", select={"depth": 10.0})
    assert ds.sizes["time"] == NT
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = _compare(["undeclared", "buoy"], depths=[10])
    assert sorted(c.reference_name for c in out) == ["buoy", "undeclared"]


def test_declaring_the_depth_makes_the_same_mooring_comparable(undeclared_world):
    undeclared_world.sources["undeclared"] = (
        undeclared_world.sources["undeclared"][0],
        _station_meta(10.0),
    )
    with pytest.warns(UserWarning, match="sits at ~10 m"):
        c = Comparison(reference="undeclared", test="his", variable=TEMPERATURE)
        c.align()
    assert c.select == {"depth": 10.0}


@pytest.mark.parametrize(
    "meta, depth, at_surface",
    [
        ({"nominal_depth_m": 10.0}, 10.0, False),
        ({"nominal_depth_m": 0.5}, 0.5, True),
        ({"depth": 25.0}, 25.0, False),
        ({"nominal_depth_m": -10.0}, 10.0, False),  # sign settled as the reader does
        # named outright wins over the extent, in the reader's own order
        (
            {
                "nominal_depth_m": 12.0,
                "geospatial_vertical_min": 30.0,
                "geospatial_vertical_max": 30.0,
            },
            12.0,
            False,
        ),
    ],
    ids=["nominal", "nominal-at-surface", "depth", "nominal-negative", "nominal-wins"],
)
def test_a_depth_named_outright_counts_as_declared(world, meta, depth, at_surface):
    world.sources["named"] = (world.sources["m10"][0], _ref_meta("timeSeries", **meta))
    assert not comparison._undeclared_mooring("named")
    assert comparison._station_depth_from_metadata("named") == depth
    assert _reference_at_surface("named") is at_surface
    problem = comparison._depth_request_problem("named", "surface")
    assert (problem is None) is at_surface
    if problem is not None:
        assert "declares no instrument depth" not in " ".join(map(str, problem))


def test_a_wide_declared_extent_is_declared_not_undeclared(world):
    # a profiler-like extent is not "no declared depth": the new refusal leaves it be
    assert comparison._depth_request_problem("wide", "surface") is None


# -- a layer with no observations in it is empty ---------------------------------------

EMPTY = {"min": 20.0, "max": 30.0}


def test_a_cast_with_no_level_in_the_layer_is_left_out_of_a_fan(world):
    with pytest.warns(UserWarning, match="left out 'cast'") as record:
        out = _compare(["cast", "m30"], depths=[EMPTY], aggregate={"Z": "mean"})
    assert [c.reference_name for c in out] == ["m30"]  # 30 m is inside 20-30
    assert any(
        "has no data in the 20–30 m layer -- its nearest levels are 5, 9, 14 m"
        in str(w.message)
        for w in record
    )


def test_a_cast_layer_without_a_vertical_mean_is_refused_too(world):
    with pytest.raises(ValueError, match="no data in the 20–30 m layer"):
        _align("cast", select={"depth": EMPTY})


def test_a_layer_of_a_cast_that_holds_a_level_still_works(world):
    ds = _align(
        "cast", select={"depth": {"min": 4.0, "max": 10.0}}, aggregate={"Z": "mean"}
    )
    assert ds.attrs["actual_depth"] == pytest.approx(np.mean([5.0, 9.0]))


def test_two_layers_of_a_cast_one_empty_is_nan_there_and_never_reads_the_model(world):
    # the cast's basic comparison (its own levels) is the one model read
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _basic = next(iter(_compare("cast")))
        _basic.align()
    reads = world.model_reads
    assert reads >= 1
    world.block_model()

    bands = [{"min": 0, "max": 6}, EMPTY]
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        (c,) = list(_compare("cast", depths=bands, aggregate={"Z": "mean"}))
        ds = c.align()
    assert world.model_reads == reads
    assert c.aligned.attrs["derived_from"] == "pairs"
    empty_warnings = [w for w in record if "20–30 m layer" in str(w.message)]
    assert len(empty_warnings) == 1
    assert "'cast' has no data in the 20–30 m layer" in str(empty_warnings[0].message)
    for name in ("test", "reference", "difference"):
        values = np.asarray(ds[name])
        assert values.shape == (2,)
        assert np.isnan(values[1])  # the empty layer, both lanes
    # the populated 0-6 m layer is the mean of the cast's 2 and 5 m levels
    np.testing.assert_allclose(np.asarray(ds["reference"])[0], np.mean([10.0, 9.0]))
    zdim = next(iter(ds["reference"].dims))
    assert list(ds[zdim].values) == [3.0, 25.0]


def test_two_layers_both_empty_is_the_one_layer_case(world):
    # nothing to keep together: each layer is asked (and refused) on its own
    both = [EMPTY, {"min": 40, "max": 50}]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="no data in the 20–30 m layer"):
            _compare("cast", depths=both, aggregate={"Z": "mean"})
    with pytest.warns(UserWarning, match="left out 'cast'") as record:
        out = _compare(["cast", "m30"], depths=both, aggregate={"Z": "mean"})
    assert [c.reference_name for c in out] == ["m30"]
    assert sum("left out 'cast'" in str(w.message) for w in record) == 2


def test_a_repeat_visit_station_with_no_level_in_the_layer_is_refused(world):
    with pytest.raises(ValueError) as err:
        _align("visits", select={"depth": EMPTY}, aggregate={"Z": "mean"})
    assert str(err.value) == (
        "'visits' has no data in the 20–30 m layer -- its nearest levels are "
        "6, 8, 12 m. Pass a layer that holds one of them."
    )
    with pytest.warns(UserWarning, match="left out 'visits'"):
        out = _compare(["visits", "m30"], depths=[EMPTY], aggregate={"Z": "mean"})
    assert [c.reference_name for c in out] == ["m30"]


def test_a_layer_holding_a_level_of_only_some_visits_is_not_empty(world):
    # 3 m is sampled on the second visit only: "inside" is across the station's levels
    assert comparison._depth_request_problem("visits", {"min": 2, "max": 4}) is None
    assert comparison._depth_request_problem("visits", EMPTY) is not None


# -- over does not move when the default depth is filled in ----------------------------


@pytest.mark.parametrize(
    "reference, over",
    [("visits", TIME_DEPTH_OVER), ("cast", "Z"), ("m30", "time"), ("buoy", "time")],
)
def test_a_direct_comparisons_axis_is_the_same_before_and_after_align(
    world, monkeypatch, reference, over
):
    # nothing is read to decide it: reading would raise
    def _no_reads(name, **kw):
        raise AssertionError(f"{name!r} was read while constructing a Comparison")

    import ocean_skill as osk

    monkeypatch.setattr(osk, "read", _no_reads)
    monkeypatch.setattr("ocean_skill.sources.read", _no_reads)
    c = Comparison(reference=reference, test="his", variable=TEMPERATURE)
    before = (c.over, c.over_reason)
    monkeypatch.setattr(osk, "read", world._read)
    monkeypatch.setattr("ocean_skill.sources.read", world._read)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c.align()
        family = c.family
        (built,) = list(_compare(reference))
        built.align()

    assert before[0] == over
    assert (c.over, c.over_reason) == before  # unchanged by the align
    assert (built.over, built.over_reason) == before  # what compare() decides
    assert built.family == family
    assert (c.is_series, c.is_profile, c.is_time_depth) == (
        built.is_series,
        built.is_profile,
        built.is_time_depth,
    )
