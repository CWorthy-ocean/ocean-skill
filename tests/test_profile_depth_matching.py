"""A plain ``profile`` cast with NaN-holed levels: the model must not be shown --
or interpolated -- at a depth the observed variable never actually measured.

Mirrors ``tests/test_tsp_end_to_end.py``'s pinned-cast case (a ragged
``timeSeriesProfile`` station's own union of levels, pruned to what one visit
actually sampled) for the plain ``profile`` featureType, which shares the same
``_prepare`` guard (see ``ocean_skill/comparison.py``'s
``PROFILE_FEATURE_TYPES``-gated ``dropna`` block). Before that guard covered
``profile`` too, a cast's own declared depth axis could carry a level where the
variable itself was never measured (NaN), and the model would still be
interpolated onto it by ``align._match_vertical`` -- exactly the "model and data
allowed to have different depths" symptom this guard now closes for both feature
types alike.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

TEMPERATURE = "sea_water_temperature"

# A cast logged at five nominal levels, but the instrument only actually reported
# a reading at four of them -- 60 m is a hole (e.g. a dropped scan), same shape as
# a ragged timeSeriesProfile visit's own missing levels.
_CAST_DEPTHS = np.array([5.0, 25.0, 60.0, 100.0, 150.0])
_CAST_VALUES = np.array([18.0, 14.0, np.nan, 8.0, 5.0])


def _profile_dataset() -> xr.Dataset:
    return xr.Dataset(
        {"TEMP": (("DEPTH",), _CAST_VALUES, {"units": "degC"})},
        coords={"DEPTH": _CAST_DEPTHS},
    ).assign_coords(lon=-158.0, lat=22.75)


def _profile_meta() -> dict:
    return {
        "axes": {"Z": "DEPTH"},
        "featureType": "profile",
        "standard_names": {"TEMP": "sea_water_temperature"},
    }


def _gridded_test_lane() -> xr.Dataset:
    """A gridded product with a real, fixed depth axis spanning the cast's full
    range, including 60 m -- so if the guard failed to prune it, the model would
    have something to interpolate there and the symptom would not reproduce.
    """
    depth = np.array([0.0, 10.0, 30.0, 50.0, 70.0, 100.0, 150.0, 200.0])
    lon = np.array([-158.5, -157.5])
    lat = np.array([22.25, 23.25])
    base = 19.0 - 0.08 * depth
    values = base[:, None, None] + np.zeros((1, 2, 2))
    da = xr.DataArray(
        values,
        dims=("depth", "lat", "lon"),
        coords={"depth": depth, "lat": lat, "lon": lon},
        name="sea_water_temperature",
        attrs={"units": "degC"},
    )
    return da.to_dataset()


@pytest.fixture
def profile_and_model(monkeypatch):
    import ocean_skill as osk
    from ocean_skill import catalog, comparison, sources

    lanes = {"ctd_cast": _profile_dataset(), "run_new": _gridded_test_lane()}
    metas = {
        "ctd_cast": _profile_meta(),
        "run_new": {"standard_names": {"sea_water_temperature": "sea_water_temperature"}},
    }

    # _profile_reference_depths (comparison.py) reads via ocean_skill.sources.read
    # directly, not the osk.read re-export -- both are patched so either call path
    # sees the same stub lanes.
    monkeypatch.setattr(osk, "read", lambda name, **kw: lanes[name])
    monkeypatch.setattr(sources, "read", lambda name, **kw: lanes[name])
    monkeypatch.setattr(
        catalog, "resolve", lambda name: SimpleNamespace(metadata=metas[name])
    )
    monkeypatch.setattr(comparison, "_domain_of", lambda name: None)
    monkeypatch.setattr(comparison, "_outline_of", lambda name, convention=None: None)
    return lanes


def test_prepare_drops_a_profile_level_the_variable_never_measured():
    """Direct _prepare check: the NaN-holed 60 m level is gone, the other four
    survive -- the same dropna the timeSeriesProfile branch already got.
    """
    from ocean_skill.comparison import _prepare

    obj = _profile_dataset()
    meta = _profile_meta()
    da, _ = _prepare(obj, meta, TEMPERATURE, {"depth": list(_CAST_DEPTHS)})
    assert sorted(float(v) for v in da["DEPTH"].values) == [5.0, 25.0, 100.0, 150.0]


def test_a_profile_compares_on_exactly_the_depths_it_has_data(profile_and_model):
    """End to end: the reference's own declared levels (auto-filled, no explicit
    depths=) drive the compare, but the 60 m hole must not survive into the
    aligned pair -- the model must not be interpolated onto a depth the cast
    never actually reported a value at.
    """
    from ocean_skill.comparison import compare

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = compare(
            reference="ctd_cast",
            test="run_new",
            variables=[TEMPERATURE],
        )
    comparisons = list(result)
    assert len(comparisons) == 1
    c = comparisons[0]
    assert c.over == "Z"
    assert c.is_profile

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        aligned = c.align()

    depths = sorted(float(v) for v in aligned["DEPTH"].values)
    assert depths == [5.0, 25.0, 100.0, 150.0]
    assert 60.0 not in depths
    assert set(aligned.data_vars) >= {"test", "reference", "difference"}
    assert np.isfinite(aligned["reference"].values).all()
    assert np.isfinite(aligned["difference"].values).any()
