"""A cast or mooring comparison's variants come from one saved set of model-data pairs.

Against a point-like reference (a mooring, a CTD cast, a repeat-visit station) the
expensive part is pairing the model with the observations at the observations' own times
and levels. Everything a caller then varies -- a time slice, depth levels or bands, a
monthly mean, detiding -- is arithmetic on those pairs, so the *basic* comparison is
built once (:meth:`Comparison._pairs_base`: its own aligned pair) and every variant
derived from it (:mod:`ocean_skill.pairs`) without reading the model again. ("Surface"
is not one of the variants: a cast or a deep mooring has no surface measurement, see
``tests/test_surface_and_layers.py``.)

These run the real ``Comparison.align`` pipeline against a small synthetic ROMS grid
(``tests/_tidal_roms.py``) with only ``osk.read``/``catalog.resolve`` stubbed, and prove
"the model is not read" by making the stubbed read raise for the model once the base
exists.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import ocean_skill as osk
from ocean_skill import cache, catalog, comparison
from ocean_skill.comparison import Comparison, _DerivationPlan, compare
from tests._tidal_roms import tidal_roms

POINT = (200.01, 50.01)  # the middle cell of the fixture's 3 x 3 grid
TEMPERATURE = "sea_water_temperature"
NT = 24 * 14  # two weeks, hourly
START = pd.Timestamp("2024-07-01")
CAST_TIME = pd.Timestamp("2024-07-05 10:00")
CAST_DEPTHS = [2.0, 5.0, 9.0, 14.0]
BANDS = [{"min": 0, "max": 6}, {"min": 7, "max": 16}]


def _model() -> tuple[xr.Dataset, dict]:
    """Build a ROMS model whose temperature falls with depth and rises in time."""
    ds, meta = tidal_roms(zeta=2.0 * np.sin(np.arange(NT) / 6.0))
    ds = ds.assign_coords(time=ds["time"].values.astype("datetime64[ns]"))
    hours = xr.DataArray(np.arange(NT), dims="time", coords={"time": ds["time"]})
    ds[TEMPERATURE] = (15.0 + 0.3 * ds["height"] + 0.01 * hours).assign_attrs(
        units="degC", standard_name=TEMPERATURE
    )
    return ds.drop_vars(["level", "height"]), meta


def _mooring() -> pd.DataFrame:
    times = pd.date_range(START, periods=NT, freq="h")
    wave = (
        12.0
        + np.sin(np.arange(NT) / 5.0)
        + 0.5 * np.sin(2 * np.pi * np.arange(NT) / 12.4)
    )
    return pd.DataFrame(
        {
            "time": times,
            "lon": POINT[0],
            "lat": POINT[1],
            "depth (m)": 5.0,
            "temperature (degC)": wave,
        }
    )


def _cast() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": CAST_TIME,
            "lon": POINT[0],
            "lat": POINT[1],
            "depth (m)": CAST_DEPTHS,
            "temperature (degC)": [10.0, 9.0, 8.0, 7.0],
        }
    )


def _visits() -> pd.DataFrame:
    """Return a repeat-visit station: three casts, ragged depths."""
    rows = []
    for k, stamp in enumerate(
        ["2024-07-03 06:00", "2024-07-05 06:00", "2024-07-07 06:00"]
    ):
        for depth in [1.0, 6.0, 12.0] if k != 1 else [3.0, 8.0]:
            rows.append((stamp, depth, POINT[0], POINT[1], 10.0 - 0.3 * depth + k))
    frame = pd.DataFrame(
        rows, columns=["time", "depth (m)", "lon", "lat", "temperature (degC)"]
    )
    frame["time"] = pd.to_datetime(frame["time"])
    return frame


def _ref_meta(feature: str, **extra) -> dict:
    return {
        "featureType": feature,
        "standard_names": {"temperature (degC)": TEMPERATURE},
        **extra,
    }


class World:
    """The stubbed catalog: sources by name, with the model's reads counted/blocked."""

    def __init__(self, monkeypatch):
        self.monkeypatch = monkeypatch
        self.model, self.model_meta = _model()
        self.sources: dict[str, tuple] = {
            "his": (self.model, self.model_meta),
            "moor": (
                _mooring(),
                _ref_meta(
                    "timeSeries",
                    geospatial_vertical_min=5.0,
                    geospatial_vertical_max=5.0,
                ),
            ),
            "cast": (_cast(), _ref_meta("profile")),
            "visits": (_visits(), _ref_meta("timeSeriesProfile")),
        }
        self.model_reads = 0
        self.model_blocked = False
        monkeypatch.setattr(osk, "read", self._read)
        monkeypatch.setattr("ocean_skill.sources.read", self._read)
        monkeypatch.setattr(
            catalog,
            "resolve",
            lambda name: SimpleNamespace(metadata=self.sources[name][1]),
        )
        monkeypatch.setattr(comparison, "_domain_of", lambda name: None)
        monkeypatch.setattr(
            comparison, "_outline_of", lambda name, convention=None: None
        )

    def _read(self, name, **kw):
        if name == "his":
            if self.model_blocked:
                raise AssertionError("the model was read, but it should not have been")
            self.model_reads += 1
        return self.sources[name][0]

    def block_model(self):
        """From here on, reading the model fails the test."""
        self.model_blocked = True


@pytest.fixture
def world(monkeypatch):
    saved = cache._verbose
    cache.verbose(False)  # most tests are not about the printed lines
    yield World(monkeypatch)
    cache._verbose = saved


def _align(reference: str, **kw) -> xr.Dataset:
    """Align one comparison of ``reference`` against the model and return its pair."""
    kw.setdefault("select", {"depth": 5.0} if reference == "moor" else {})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = Comparison(reference=reference, test="his", variable=TEMPERATURE, **kw)
        return c.align()


def _basic_cast() -> Comparison:
    """Return the cast's basic comparison, as :func:`compare` builds it."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = next(iter(compare(reference="cast", test="his", variables=[TEMPERATURE])))
        c.align()
    return c


def _values(ds: xr.Dataset, name: str = "test") -> np.ndarray:
    return np.asarray(ds[name]).reshape(-1)


# -- a mooring -----------------------------------------------------------------------


def test_the_basic_mooring_request_runs_the_ordinary_pipeline(world):
    ds = _align("moor")
    assert ds.attrs["derived_from"] == "lanes"
    assert "basic comparison" in ds.attrs["derived_reason"]
    assert ds.sizes["time"] == NT


@pytest.mark.parametrize(
    "variant",
    [
        {"select": {"depth": 5.0, "time": {"min": "2024-07-03", "max": "2024-07-05"}}},
        {
            "select": {"depth": 5.0},
            "aggregate": {"time": {"resample": "1D", "reduce": "mean"}},
        },
        {"select": {"depth": 5.0}, "detide": True},
    ],
    ids=["time-slice", "daily-mean", "detide"],
)
def test_a_mooring_variant_never_reads_the_model_once_the_base_exists(world, variant):
    _align(
        "moor",
        select={"depth": 5.0, "time": {"min": "2024-07-02", "max": "2024-07-04"}},
    )
    reads_to_build = world.model_reads
    assert reads_to_build >= 1

    world.block_model()
    ds = _align("moor", **variant)

    assert ds.attrs["derived_from"] == "pairs"
    assert world.model_reads == reads_to_build


def test_a_reference_qc_change_reuses_the_saved_model_lane(world):
    # qc names which obs survive, so it is a different basic comparison -- but the model
    # lane (keyed on the model alone) is saved, so the model is still not read again
    _align("moor")
    reads = world.model_reads
    world.block_model()
    ds = _align("moor", qc={"test": None, "reference": "off"})
    assert ds.attrs["derived_from"] == "lanes"
    assert world.model_reads == reads


def test_demeaning_the_basic_request_reuses_its_saved_pair(world):
    _align("moor")
    world.block_model()
    ds = _align("moor", subtract_mean=True)
    assert "subtracted_mean_test" in ds.attrs
    assert abs(float(ds["test"].mean())) < 1e-9


def test_demeaning_a_derived_variant_works_off_the_saved_pairs(world):
    window = {"depth": 5.0, "time": {"min": "2024-07-03", "max": "2024-07-05"}}
    _align("moor", select=window)
    world.block_model()
    ds = _align("moor", select=window, subtract_mean=True)
    assert ds.attrs["derived_from"] == "pairs"
    assert abs(float(ds["test"].mean())) < 1e-9


def test_a_mooring_time_slice_matches_the_lane_pipeline(world, monkeypatch):
    window = {"depth": 5.0, "time": {"min": "2024-07-03", "max": "2024-07-05"}}
    derived = _align("moor", select=window, cache=False)
    assert derived.attrs["derived_from"] == "pairs"

    # the same request with derivation switched off: the lanes' own reduction
    monkeypatch.setattr(
        Comparison,
        "_derivation_plan",
        lambda self: _DerivationPlan(None, "switched off for the comparison", {}, []),
    )
    lanes = _align("moor", select=window, cache=False)
    assert lanes.attrs["derived_from"] == "lanes"

    xr.testing.assert_equal(derived["time"], lanes["time"])
    np.testing.assert_allclose(_values(derived), _values(lanes))
    np.testing.assert_allclose(
        _values(derived, "reference"), _values(lanes, "reference")
    )


def _lanes_only(monkeypatch):
    monkeypatch.setattr(
        Comparison,
        "_derivation_plan",
        lambda self: _DerivationPlan(None, "switched off for the comparison", {}, []),
    )


def _season_fan(**kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cs = compare(
            test="his",
            variables=[TEMPERATURE],
            times={"groupby": "season", "seasons": ["MJJA"], "reduce": "mean"},
            **kw,
        )
        return [c.align() for c in cs]


def test_a_mooring_season_fan_derives_from_the_saved_pairs(world):
    # the lanes cannot score a seasonal mean against a timeSeries (its model lane loses
    # the time axis the pairs are matched on), so the pairs are the check: the mean of
    # the basic pairs over the season
    basic = _align("moor")
    reads = world.model_reads
    world.block_model()

    (derived,) = _season_fan(reference="moor", select={"depth": 5.0})
    assert derived.attrs["derived_from"] == "pairs"
    assert world.model_reads == reads
    assert derived["test"].ndim == 0
    assert float(derived["test"]) == pytest.approx(float(basic["test"].mean()))
    assert float(derived["reference"]) == pytest.approx(
        float(basic["reference"].mean())
    )


def test_a_visit_station_season_fan_matches_the_lane_pipeline(world, monkeypatch):
    select = {"depth": {"min": 0, "max": 8}}
    kw = {"reference": "visits", "select": select, "aggregate": {"Z": "mean"}}
    _align("visits")  # the basic comparison, so the base exists
    series = _align("visits", select=select, aggregate={"Z": "mean"})
    reads = world.model_reads
    world.block_model()
    (derived,) = _season_fan(**kw)
    assert derived.attrs["derived_from"] == "pairs"
    assert world.model_reads == reads

    world.model_blocked = False
    _lanes_only(monkeypatch)
    (lanes,) = _season_fan(**kw)
    assert lanes.attrs["derived_from"] == "lanes"
    for name in ("test", "reference", "difference"):
        assert derived[name].dims == lanes[name].dims
        assert derived[name].shape == lanes[name].shape
    # a band is the plain mean of the pairs inside it, not the model's
    # thickness-weighted band average (see pairs.py), so against the lanes the values
    # are close, not equal;
    # the season is exact against the band's own series, averaged over the visits
    for name in ("test", "reference"):
        np.testing.assert_allclose(
            _values(derived, name), _values(lanes, name), atol=0.2
        )
    assert float(derived["test"]) == pytest.approx(float(series["test"].mean()))
    assert float(derived["reference"]) == pytest.approx(
        float(series["reference"].mean())
    )


def test_the_basic_result_is_what_the_lane_pipeline_always_gave(world, monkeypatch):
    first = _align("moor", cache=False)
    monkeypatch.setattr(
        Comparison,
        "_derivation_plan",
        lambda self: _DerivationPlan(None, "switched off for the comparison", {}, []),
    )
    second = _align("moor", cache=False)
    for name in ("test", "reference", "difference"):
        xr.testing.assert_equal(first[name], second[name])


def test_a_mooring_at_depth_has_no_surface_to_compare(world):
    # the mooring's one instrument is at 5 m: "surface" used to be silently answered
    # with the model's top cell against that 5 m series
    with pytest.raises(ValueError, match="no surface measurement") as err:
        _align("moor", select={"depth": "surface"})
    assert "its instrument is at 5 m" in str(err.value)
    assert "depths=[5]" in str(err.value)


# -- a cast -------------------------------------------------------------------------


def test_the_basic_cast_is_the_lane_pipelines_own_result(world):
    c = _basic_cast()
    assert c.aligned.attrs["derived_from"] == "lanes"
    assert "test_surface" not in c.aligned
    np.testing.assert_allclose(_values(c.aligned, "reference"), [10.0, 9.0, 8.0, 7.0])


def test_a_cast_building_its_base_reads_the_model_once(world, monkeypatch):
    monkeypatch.setattr(comparison, "_domain_of", lambda name: (*POINT, *POINT))
    ds = _align("cast", select={"depth": [5.0]})
    assert ds.attrs["derived_from"] == "pairs"
    assert world.model_reads == 1


@pytest.mark.parametrize(
    "variant",
    [
        {"select": {"depth": [5.0, 14.0]}},
        {"select": {"depth": 9.0}},
        {"select": {"depth": BANDS[:1]}, "aggregate": {"Z": "mean"}},
        {"select": {"depth": BANDS}, "aggregate": {"Z": "mean"}},
        {"select": {"depth": CAST_DEPTHS}, "qc": {"test": None, "reference": "off"}},
    ],
    ids=["levels", "one-level", "one-band", "two-bands", "qc"],
)
def test_a_cast_variant_never_reads_the_model_once_the_base_exists(world, variant):
    _align("cast", select={"depth": [5.0]})  # builds the base
    reads_to_build = world.model_reads

    world.block_model()
    ds = _align("cast", **variant)

    assert ds.attrs["derived_from"] == "pairs"
    assert world.model_reads == reads_to_build


def test_the_surface_of_a_cast_is_refused_not_mapped_to_its_shallowest_level(world):
    with pytest.raises(ValueError, match="no surface measurement") as err:
        _align("cast", select={"depth": "surface"})
    assert "its shallowest level is 2 m" in str(err.value)
    assert "depths=[2]" in str(err.value)
    assert world.model_reads == 0  # refused before anything is read


def test_cast_levels_are_the_pairs_at_those_levels(world):
    full = _align("cast", select={"depth": CAST_DEPTHS}, cache=False)
    some = _align("cast", select={"depth": [5.0, 14.0]})
    np.testing.assert_allclose(_values(some), _values(full)[[1, 3]])
    np.testing.assert_allclose(_values(some, "reference"), [9.0, 7.0])


def test_a_band_is_the_mean_of_the_pairs_inside_it(world):
    full = _align("cast", select={"depth": CAST_DEPTHS}, cache=False)
    band = _align(
        "cast", select={"depth": [{"min": 4, "max": 10}]}, aggregate={"Z": "mean"}
    )
    np.testing.assert_allclose(float(band["test"]), _values(full)[[1, 2]].mean())
    np.testing.assert_allclose(float(band["reference"]), 8.5)


def test_a_literal_depth_the_cast_never_sampled_falls_back_to_the_lanes(world):
    ds = _align("cast", select={"depth": [3.0, 7.0]})
    assert ds.attrs["derived_from"] == "lanes"


def test_a_pair_spec_variable_falls_back_to_the_lanes(world):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = Comparison(
            reference="cast",
            test="his",
            variable={"test": TEMPERATURE, "reference": TEMPERATURE},
            select={"depth": [5.0]},
        )
        ds = c.align()
    assert ds.attrs["derived_from"] == "lanes"


def test_a_time_select_on_a_single_cast_falls_back_to_the_lanes(world):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = Comparison(
            reference="cast",
            test="his",
            variable=TEMPERATURE,
            select={"depth": [5.0], "time": "2024-07-05"},
        )
    assert c._derivation_plan().basic is None


# -- several layer means of one cast ------------------------------------------------


def test_two_bands_of_one_cast_are_one_comparison_with_two_layers(world):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = compare(
            reference="cast",
            test="his",
            variables=[TEMPERATURE],
            depths=BANDS,
            aggregate={"Z": "mean"},
        )
    assert len(out) == 1
    c = next(iter(out))
    assert c.over == "Z"
    ds = c.aligned
    assert ds.attrs["derived_from"] == "pairs"
    depth = next(d for d in ds["test"].dims)
    assert ds.sizes[depth] == 2
    np.testing.assert_allclose(_values(ds, "reference"), [9.5, 7.5])
    # a two-point profile scores without raising: some metrics may be NaN
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c.metrics()


def test_bands_the_pairs_cannot_answer_keep_the_per_band_fan(world):
    # without a vertical mean a band asks for the model's own band average, which the
    # pairs at the obs levels cannot give -- so the fan stays one comparison per band
    plan = comparison._profile_depth_plan(
        "cast",
        {},
        None,
        False,
        "depth",
        tuple(BANDS),
        BANDS,
        {},
        aggregate={},
    )
    assert plan[0] == tuple(BANDS)


# -- a repeat-visit station ---------------------------------------------------------


def test_a_visit_station_derives_its_variants_too(world):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        basic = next(
            iter(compare(reference="visits", test="his", variables=[TEMPERATURE]))
        )
        basic.align()
    assert basic.aligned.attrs["derived_from"] == "lanes"
    _align("visits", select={"depth": [1.0, 6.0]})  # builds the base

    world.block_model()
    one = _align("visits", select={"depth": 6.0})
    assert one.attrs["derived_from"] == "pairs"
    assert one["test"].sizes["time"] == 3  # all three visits stay, as time points
    assert one.attrs["actual_depth"] == 6.0
    # the visits' own reference values at 6 m (10 - 0.3*6 + visit index), a hole where
    # visit 2 sampled other depths
    got = _values(one, "reference")
    np.testing.assert_allclose(np.sort(got[np.isfinite(got)]), [10 - 1.8, 10 - 1.8 + 2])


# -- what is said ---------------------------------------------------------------------


def test_each_comparison_says_which_saved_layer_answered_it(world, capsys):
    cache.verbose(True)
    window = {"depth": 5.0, "time": {"min": "2024-07-03", "max": "2024-07-05"}}
    other = {"depth": 5.0, "time": {"min": "2024-07-04", "max": "2024-07-06"}}

    _align("moor", select=window)
    assert cache.SAVED_PAIRS in capsys.readouterr().out

    _align("moor", select=other)
    assert cache.USED_PAIRS in capsys.readouterr().out

    _align("moor", select=other)
    assert cache.USED_ALIGNED in capsys.readouterr().out


def test_a_computed_from_scratch_comparison_says_nothing(world, capsys):
    cache.verbose(True)
    _align("moor")
    assert "cache:" not in capsys.readouterr().out


def test_a_lane_hit_is_reported_when_nothing_better_applies(world, capsys):
    cache.verbose(True)
    # same model lane, a different reference qc: the pairs differ but the model lane
    # (keyed on the model alone) is reused
    _align("moor")
    capsys.readouterr()
    world.block_model()
    _align("moor", qc={"test": None, "reference": "off"})
    assert cache.USED_LANE in capsys.readouterr().out


def test_verbose_off_silences_every_line(world, capsys):
    cache.verbose(False)
    window = {"depth": 5.0, "time": {"min": "2024-07-03", "max": "2024-07-05"}}
    _align("moor", select=window)
    _align("moor", select=window)
    assert "cache:" not in capsys.readouterr().out
