"""A WHOTS-shaped ``timeSeriesProfile`` source, read and compared end to end.

Mocks only ``osk.read``/``catalog.resolve`` (mirroring ``tests/test_comparison.py``'s
own pattern), so the real ``_prepare``/``align``/``match_axis`` pipeline runs against
a source shaped exactly like the real catalog entry: ``axes: {Z: DEPTH, T: TIME}``,
``featureType: timeSeriesProfile``, the axis left spelled uppercase since nothing in
the catalog renames it (see ``ocean_skill/catalogs/whots.yaml``).
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

TEMPERATURE = "sea_water_temperature"


def _whots_dataset(n_time: int = 6, n_depth: int = 4) -> xr.Dataset:
    time = pd.date_range("2015-01-01", periods=n_time, freq="MS")
    depth = np.array([2.79, 25.0, 60.0, 118.79])  # real WHOTS-style levels
    base = 24.0 - 0.05 * depth
    values = base[None, :] + 0.1 * np.arange(n_time)[:, None]
    return xr.Dataset(
        {"TEMP": (("TIME", "DEPTH"), values, {"units": "degC"})},
        coords={"TIME": time, "DEPTH": depth},
    ).assign_coords(lon=-158.0, lat=22.75)


def _whots_meta() -> dict:
    return {
        "axes": {"T": "TIME", "X": "LONGITUDE", "Y": "LATITUDE", "Z": "DEPTH"},
        "featureType": "timeSeriesProfile",
        "standard_names": {"TEMP": "sea_water_temperature"},
    }


def _gridded_test_lane(n_time: int = 6) -> xr.Dataset:
    """A gridded product with a real, fixed depth axis -- the non-ROMS shape of the
    "test" side; exercises the same fixed-level select the model side would (see
    tests/test_vertical_match.py for the ROMS-shaped ('z', negative-down) case,
    already covered directly against roms.to_depth's own machinery).
    """
    depth = np.array([2.79, 25.0, 60.0, 118.79])  # positive-down, like WHOTS itself
    lon = np.array([-158.5, -157.5])
    lat = np.array([22.25, 23.25])
    time = pd.date_range("2015-01-01", periods=n_time, freq="MS")
    base = 24.5 - 0.05 * depth
    values = (
        base[None, :, None, None]
        + 0.1 * np.arange(n_time)[:, None, None, None]
        + np.zeros((1, 1, 2, 2))
    )
    da = xr.DataArray(
        values,
        dims=("time", "depth", "lat", "lon"),
        coords={"time": time, "depth": depth, "lat": lat, "lon": lon},
        name="sea_water_temperature",
        attrs={"units": "degC"},
    )
    return da.to_dataset()


@pytest.fixture
def whots_and_model(monkeypatch):
    import ocean_skill as osk
    from ocean_skill import catalog, comparison

    lanes = {"whots_station": _whots_dataset(), "run_new": _gridded_test_lane()}
    metas = {
        "whots_station": _whots_meta(),
        "run_new": {"standard_names": {"sea_water_temperature": "sea_water_temperature"}},
    }

    monkeypatch.setattr(osk, "read", lambda name, **kw: lanes[name])
    monkeypatch.setattr(
        catalog, "resolve", lambda name: SimpleNamespace(metadata=metas[name])
    )
    monkeypatch.setattr(comparison, "_domain_of", lambda name: None)
    monkeypatch.setattr(comparison, "_outline_of", lambda name, convention=None: None)
    return lanes


def test_whots_depth_axis_resolves_despite_being_uppercase(whots_and_model):
    """The core Stage-1/Stage-3 claim: DEPTH (uppercase, no CF attrs) is found."""
    from ocean_skill.comparison import _prepare
    from ocean_skill.tabular import is_frame

    obj = whots_and_model["whots_station"]
    meta = _whots_meta()
    assert not is_frame(obj)  # already an xarray Dataset, not a DataFrame
    da, actual = _prepare(obj, meta, TEMPERATURE, {"depth": [2.79, 25.0]})
    assert "DEPTH" in da.dims
    assert da.sizes["DEPTH"] == 2


def test_a_bare_read_of_whots_implies_a_profile(whots_and_model, monkeypatch):
    """No explicit over=: the featureType alone (both axes present, both unset --
    depth defaults to SURFACE, a scalar) implies a mooring-at-the-surface series,
    not a profile -- the pre-existing reading, unchanged.
    """
    from ocean_skill.comparison import Comparison

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = Comparison(
            reference="whots_station", test="run_new", variable=TEMPERATURE, cache=False
        )
    assert c.over == "time"


def test_a_depth_list_with_time_pinned_reads_as_a_profile_end_to_end(
    whots_and_model,
):
    """select={"depth": [...], "time": <instant>}: time collapses, depth survives
    -- over="Z" is implied, and the comparison draws as a profile.
    """
    from ocean_skill.comparison import Comparison

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = Comparison(
            reference="whots_station",
            test="run_new",
            variable=TEMPERATURE,
            select={"depth": [2.79, 25.0, 60.0, 118.79], "time": "2015-01-01"},
            cache=False,
        )
        assert c.over == "Z"
        aligned = c.align()

    assert c.is_profile
    assert c.family == "profile"
    assert set(aligned.data_vars) >= {"test", "reference", "difference"}
    assert aligned.attrs["match_method"] == "interp"
    assert np.isfinite(aligned["difference"].values).any()

    item = c.as_item()
    assert "aligned" in item and "skill" not in item

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        fig = c.plot()
    ax = fig.axes[0]
    bottom, top = ax.get_ylim()
    assert bottom > top  # surface at the top
