"""A pair-spec ``select`` that names the same transect and depths on both lanes.

``select={"test": {"transect": T, "depth": D, "time": ...}, "reference":
{"transect": T, "depth": D}}`` is still one path: the test lane is cut along ``T``
and the reference is sampled at the test lane's own snapped points, each lane's
other keys (a test-only time window, because the reference is a climatology)
applying to that lane alone. A transect on one side only, two different paths, or
two different depth lists are two independently-sampled paths and stay refused.

The model is the small ROMS run of ``test_section_comparison`` given four daily
steps (``temperature = base + 10 * step``), so a window picks out known numbers;
the climatology is its WOA-style gridded reference given one undecoded numeric
``time`` ("months since 1955"), the axis a calendar time window cannot select on.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import ocean_skill as osk
from ocean_skill import catalog
from ocean_skill.align import ALONG_DIM
from ocean_skill.comparison import (
    Comparison,
    ComparisonSet,
    compare,
    transect_pair_problem,
)
from ocean_skill.transect import as_transect
from tests.test_section_comparison import HC, VAR, _climatology, _roms_run

TRANSECT = {"xi_rho": 1}
DEPTHS = [50.0, 200.0]
STEPS = pd.date_range("2011-03-30", periods=4, freq="D")  # Mar 30, 31, Apr 1, 2
WINDOW = {"min": "2011-04-01", "max": "2011-04-30"}  # steps 2 and 3
STEP_OFFSET = 10.0


def _timed_model() -> xr.Dataset:
    """Return the ROMS run with four steps, step ``i`` ``10 * i`` warmer."""
    ds = _roms_run()
    base = ds[VAR]
    timed = xr.concat([base + STEP_OFFSET * i for i in range(len(STEPS))], dim="time")
    timed = timed.assign_coords(time=STEPS).transpose("time", ...)
    timed.attrs = base.attrs
    return ds.drop_vars(VAR).assign({VAR: timed})


def _numeric_climatology() -> xr.Dataset:
    """Return the climatology with WOA's undecoded one-step ``time``."""
    ds = _climatology()
    da = ds[VAR].expand_dims(time=[672.0])
    da["time"].attrs = {"units": "months since 1955-01-01 00:00:00"}
    return da.to_dataset()


@pytest.fixture
def sources(monkeypatch):
    table = {
        "roms_test": (
            _timed_model(),
            {"model": "roms", "vertical": {"s_dim": "s_rho", "hc": HC}},
        ),
        "woa_ref": (_numeric_climatology(), {"featureType": "grid"}),
    }
    monkeypatch.setattr(osk, "read", lambda name, **kw: table[name][0])
    monkeypatch.setattr(
        catalog, "resolve", lambda name: SimpleNamespace(metadata=table[name][1])
    )
    return table


def _pair(test_extra=None, reference_extra=None, *, transect=TRANSECT, depth=DEPTHS):
    return {
        "test": {"transect": transect, "depth": depth, **(test_extra or {})},
        "reference": {"transect": transect, "depth": depth, **(reference_extra or {})},
    }


def _build(select, **kw) -> Comparison:
    return Comparison(
        test="roms_test",
        reference="woa_ref",
        variable=VAR,
        select=select,
        aggregate={"time": "mean"},
        cache=False,
        **kw,
    )


# -- the rule ----------------------------------------------------------------------


def test_the_same_transect_and_depths_with_a_test_only_window_draws_a_section(sources):
    c = _build(_pair({"time": WINDOW}))
    aligned = c.align()
    assert c.is_section and c.family == "section_row"
    assert c._transect_route() is not None
    assert set(aligned["test"].dims) == {"z", ALONG_DIM}
    assert list(aligned["test"][ALONG_DIM].values) == list(
        aligned["reference"][ALONG_DIM].values
    )
    assert np.isfinite(aligned["test"].values).all()
    assert np.isfinite(aligned["reference"].values).all()


def test_the_test_lane_is_the_mean_over_its_own_window(sources):
    """Steps 2 and 3 average to the base run plus 25: not the whole record's 15."""
    windowed = _build(_pair({"time": WINDOW})).align()
    whole = _build(_pair()).align()
    base = _build(_pair({"time": {"min": "2011-03-30", "max": "2011-03-30"}})).align()
    np.testing.assert_allclose(
        windowed["test"].values, base["test"].values + 2.5 * STEP_OFFSET
    )
    np.testing.assert_allclose(
        whole["test"].values, base["test"].values + 1.5 * STEP_OFFSET
    )
    # the climatology carries no window, so the two lanes' difference is the shift
    np.testing.assert_allclose(windowed["reference"].values, base["reference"].values)


def test_it_matches_the_flat_select_on_a_test_source_cut_to_that_window(
    sources, monkeypatch
):
    pair = _build(_pair({"time": WINDOW})).align()
    cut = sources["roms_test"][0].sel(time=slice("2011-04-01", "2011-04-30"))
    monkeypatch.setitem(sources, "roms_test", (cut, sources["roms_test"][1]))
    flat = _build({"transect": TRANSECT, "depth": DEPTHS}).align()
    for lane in ("test", "reference"):
        np.testing.assert_allclose(pair[lane].values, flat[lane].values)


def test_a_numeric_reference_time_works_with_the_pair_spec_and_not_the_flat_window(
    sources,
):
    # the pair-spec form: the window never reaches the climatology's numeric time
    _build(_pair({"time": WINDOW})).align()
    # the flat form hands the same window to both lanes
    flat = _build({"transect": TRANSECT, "depth": DEPTHS, "time": WINDOW})
    with pytest.warns(UserWarning, match="valid data"):
        aligned = flat.align()
    assert not np.isfinite(aligned["reference"].values).any()


def test_each_lane_keeps_its_own_other_keys(sources):
    c = _build(_pair({"time": WINDOW}))
    assert c.select["test"]["time"] == WINDOW
    assert "time" not in c.select["reference"]
    assert c._transect_route() == as_transect(TRANSECT)  # one normalized spec


def test_the_cache_key_tells_a_windowed_pair_spec_from_the_flat_select(sources):
    flat = _build({"transect": TRANSECT, "depth": DEPTHS})
    pair = _build(_pair())
    windowed = _build(_pair({"time": WINDOW}))
    other = _build(_pair({"time": {"min": "2011-03-30", "max": "2011-03-31"}}))
    keys = {flat._cache_key, pair._cache_key, windowed._cache_key, other._cache_key}
    assert len(keys) == 4


def test_the_normalized_spellings_of_one_path_count_as_the_same_transect():
    same = {
        "test": {
            "transect": {"lat": 25, "lon": {"min": -95, "max": -93}},
            "depth": DEPTHS,
        },
        "reference": {
            "transect": {"lon": {"min": -95, "max": -93}, "lat": 25},
            "depth": DEPTHS,
        },
    }
    assert transect_pair_problem(same) is None


# -- what stays refused ------------------------------------------------------------

SAME = "must name the same transect and the same depths"


@pytest.mark.parametrize(
    "select",
    [
        {
            "test": {"transect": TRANSECT, "depth": DEPTHS},
            "reference": {"depth": DEPTHS},
        },
        {
            "test": {"depth": DEPTHS},
            "reference": {"transect": TRANSECT, "depth": DEPTHS},
        },
        _pair(transect={"xi_rho": 1})
        | {"reference": {"transect": {"xi_rho": 2}, "depth": DEPTHS}},
        {
            "test": {"transect": TRANSECT, "depth": DEPTHS},
            "reference": {"transect": TRANSECT, "depth": [50.0, 100.0]},
        },
        {
            "test": {"transect": TRANSECT, "depth": DEPTHS},
            "reference": {"transect": TRANSECT},
        },
    ],
    ids=[
        "test-only",
        "reference-only",
        "different-paths",
        "different-depths",
        "one-depth",
    ],
)
def test_anything_but_one_shared_path_and_depth_list_is_refused(select):
    with pytest.raises(ValueError, match=SAME):
        Comparison(
            test="nope_test",
            reference="nope_ref",
            variable=VAR,
            select=select,
            cache=False,
        )
    with pytest.raises(ValueError, match=SAME):
        compare(test="nope_test", reference="nope_ref", variables=[VAR], select=select)


def test_a_shared_pair_spec_still_needs_a_depth_list():
    with pytest.raises(ValueError, match="explicit list of at least 2"):
        Comparison(
            test="nope_test",
            reference="nope_ref",
            variable=VAR,
            select=_pair(depth=50.0),
            cache=False,
        )


def test_from_reference_inside_a_pair_spec_is_refused():
    with pytest.raises(ValueError, match="not built"):
        compare(
            test="nope_test",
            reference=["a", "b"],
            variables=[VAR],
            select=_pair(transect={"from": "reference"}),
        )


# -- selects= over a pair-spec select carrying a test-only window ------------------


def test_labelled_transects_merge_into_a_pair_spec_base_and_both_draw(sources):
    base = {"test": {"depth": DEPTHS, "time": WINDOW}, "reference": {"depth": DEPTHS}}
    out = compare(
        reference="woa_ref",
        test="roms_test",
        variables=[VAR],
        select=base,
        selects={
            "A": {"transect": {"xi_rho": 0}},
            "B": {"transect": {"lat": 25, "lon": {"min": -95, "max": -93}}},
        },
        aggregate={"time": "mean"},
    )
    assert isinstance(out, ComparisonSet)
    assert [c.label for c in out] == ["A", "B"]
    for c in out:
        assert c.is_section
        assert c.select["test"]["transect"] == c.select["reference"]["transect"]
        assert c.select["test"]["time"] == WINDOW
        assert "time" not in c.select["reference"]
        aligned = c.align()
        assert np.isfinite(aligned["test"].values).any()
        assert np.isfinite(aligned["reference"].values).any()
