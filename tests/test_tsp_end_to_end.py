"""A Hvalfjörður-shaped ``timeSeriesProfile`` source, read and compared end to end.

Mocks only ``osk.read``/``catalog.resolve`` (mirroring
``tests/test_whots_profile_end_to_end.py``'s own pattern), so the real
``_prepare``/``align``/``match_axis`` pipeline runs against a source shaped exactly
like a real discrete-bottle-sample catalog entry: a **DataFrame** (not already an
xarray Dataset -- ``tabular.to_dataset`` has to build the rectangle for real), ragged
depths across visits, and a station position that wobbles a little from cast to cast.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

TEMPERATURE = "sea_water_temperature"


def _hvalfjordur_frame() -> pd.DataFrame:
    """4 visits, ragged depths, a single-depth cast, a little position wobble."""
    rows = [
        ("2024-01-01", 1, -21.987, 64.2638, 8.1),
        ("2024-01-01", 10, -21.988, 64.2638, 7.9),
        ("2024-01-01", 30, -21.986, 64.2638, 7.5),
        ("2024-02-01", 5, -21.9895, 64.2638, 8.3),  # single-depth cast
        ("2024-03-01", 1, -21.9877, 64.2638, 6.0),
        ("2024-03-01", 15, -21.987, 64.2638, 5.9),
        ("2024-03-01", 31, -21.986, 64.2638, 5.8),
        ("2024-04-01", 2, -21.9865, 64.2638, 7.2),
        ("2024-04-01", 10, -21.9865, 64.2638, 7.0),
    ]
    return pd.DataFrame(rows, columns=["time", "depth (m)", "lon", "lat", "Temperature (degC)"])


def _hvalfjordur_meta() -> dict:
    return {
        "featureType": "timeSeriesProfile",
        "featureType_source": "declared",
        "datasetID": "hvalfjordur_hv1",
        "standard_names": {"Temperature (degC)": TEMPERATURE},
    }


def _gridded_test_lane(n_time: int = 5) -> xr.Dataset:
    """A gridded model spanning the visit dates.

    Depth levels deliberately match the reference's own full union of sampled
    depths exactly: neither this fixture nor the reference declares
    ``model: "roms"``, so both take the same nearest-native-level path (see
    ``_prepare``'s "observational depth axes vary" branch) rather than a real
    vertical interpolation (``roms.to_depth``) -- a coarser, offset grid on this
    side would make two *different* targets both round to the same native level
    (a duplicate-index reindex error) or the same target round to two *different*
    values on the two lanes (an unmergeable scalar-depth conflict on align()'s
    final ``xr.Dataset({test, reference, difference})``). Real ROMS output
    sidesteps both by interpolating to the exact requested depth instead of
    picking a nearby native level; this fixture is deliberately not that, so the
    depths are chosen to make it behave the same way for what these tests check.
    """
    depth = np.array([1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 31.0])
    lon = np.array([-22.0, -21.9])
    lat = np.array([64.2, 64.3])
    time = pd.date_range("2024-01-01", periods=n_time, freq="MS")
    base = 9.0 - 0.1 * depth
    values = (
        base[None, :, None, None]
        + 0.05 * np.arange(n_time)[:, None, None, None]
        + np.zeros((1, 1, 2, 2))
    )
    da = xr.DataArray(
        values,
        dims=("time", "depth", "lat", "lon"),
        coords={"time": time, "depth": depth, "lat": lat, "lon": lon},
        name=TEMPERATURE,
        attrs={"units": "degC"},
    )
    return da.to_dataset()


@pytest.fixture
def hvalfjordur_and_model(monkeypatch):
    import ocean_skill as osk
    from ocean_skill import catalog, comparison

    lanes = {"hvalfjordur_hv1": _hvalfjordur_frame(), "run_new": _gridded_test_lane()}
    metas = {
        "hvalfjordur_hv1": _hvalfjordur_meta(),
        "run_new": {"standard_names": {TEMPERATURE: TEMPERATURE}},
    }

    monkeypatch.setattr(osk, "read", lambda name, **kw: lanes[name])
    # _profile_reference_depths (the per-visit union-of-levels auto-fill) imports
    # sources.read directly rather than going through top-level osk.read -- both
    # need patching, exactly as tests/test_profile_compare_depths.py's own fixture
    # does for the plain "profile" featureType.
    monkeypatch.setattr("ocean_skill.sources.read", lambda name, **kw: lanes[name])
    monkeypatch.setattr(
        catalog, "resolve", lambda name: SimpleNamespace(metadata=metas[name])
    )
    monkeypatch.setattr(comparison, "_domain_of", lambda name: None)
    monkeypatch.setattr(comparison, "_outline_of", lambda name, convention=None: None)
    return lanes


def test_a_bare_compare_keeps_the_time_axis(hvalfjordur_and_model):
    """No select at all: the pre-existing timeSeriesProfile reading, unchanged --
    depth defaults to the surface, time is what survives.
    """
    from ocean_skill.comparison import Comparison

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = Comparison(
            reference="hvalfjordur_hv1",
            test="run_new",
            variable=TEMPERATURE,
            cache=False,
        )
    assert c.over == "time"


def test_a_pinned_visit_reads_as_one_profile_on_its_own_depths(hvalfjordur_and_model):
    """select={"time": <one visit's day>}, through compare()'s fan (the auto-fill
    lives there, in _profile_depth_plan -- a bare Comparison(select={"time": ...})
    with no depth key is still the pre-existing "ambiguous, pass over=" case):
    the comparison is scored on exactly *that cast's* depths, not the station's
    full ragged union (visit 3's own {1, 15, 31} m, not also 10/30/2/5 m from
    other visits) -- the all-NaN levels the rectangular pivot leaves for a level
    this cast never sampled are trimmed once time has narrowed to one instant.
    """
    from ocean_skill.comparison import compare

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = compare(
            reference="hvalfjordur_hv1",
            test="run_new",
            variables=[TEMPERATURE],
            select={"time": "2024-03-01"},
        )
    comparisons = list(result)
    assert len(comparisons) == 1
    c = comparisons[0]
    assert c.over == "Z"
    assert c.is_profile
    assert c.family == "profile"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        aligned = c.align()
    assert sorted(float(v) for v in aligned["depth"].values) == [1.0, 15.0, 31.0]
    assert set(aligned.data_vars) >= {"test", "reference", "difference"}
    # the model was sampled at its nearest snapshot to the cast, not left as a
    # leftover time axis -- the whole point of the singleton-time squeeze
    assert "time" not in aligned.dims
    assert np.isfinite(aligned["difference"].values).any()


def test_times_fan_gives_one_profile_per_visit(hvalfjordur_and_model):
    """times=[...] against a timeSeriesProfile reference: one profile comparison
    per named visit, each on that visit's own depths.
    """
    from ocean_skill.comparison import compare

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = compare(
            reference="hvalfjordur_hv1",
            test="run_new",
            variables=[TEMPERATURE],
            times=["2024-01-01", "2024-02-01"],
        )
    comparisons = list(result)
    assert len(comparisons) == 2
    for c in comparisons:
        assert c.over == "Z"
        assert c.family == "profile"
    # visit 2 (2024-02-01) is the single-depth cast -- one level, not the union
    depths_by_visit = {
        c.label: sorted(float(v) for v in c.aligned["depth"].values) for c in comparisons
    }
    assert depths_by_visit["2024-02-01"] == [5.0]
    assert depths_by_visit["2024-01-01"] == [1.0, 10.0, 30.0]


def test_a_fixed_level_across_visits_warns_when_sparse(hvalfjordur_and_model):
    """The mooring-style recipe (over="time", one fixed depth): a ragged station
    samples different depths on different visits, so most single levels are
    sparse across the record -- and the sparsity is what's worth flagging, not
    silently treated as if this were a mooring's own steady instrument depth.
    """
    from ocean_skill.comparison import Comparison

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = Comparison(
            reference="hvalfjordur_hv1",
            test="run_new",
            variable=TEMPERATURE,
            # 31 m is sampled on only visit 3 of 4 -- well under the 50% floor,
            # unlike 1 m (visits 1 and 3, exactly half, deliberately not sparse
            # enough to warn).
            select={"depth": 31},
            cache=False,
        )
        assert c.over == "time"
        with pytest.warns(UserWarning, match="ragged timeSeriesProfile station"):
            c.align()
