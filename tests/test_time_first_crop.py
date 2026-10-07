"""A ROMS lane is cut along time before space, and read for what it requests only.

On a long lazy model (a kerchunk reference over ~225,000 one-step chunks) every
``isel``/``sel`` on a variable builds a dask graph over *all* of its chunks, so each
comparison spent minutes building graphs that were then thrown away: ``prepare_source``
cropped space first (``subset_to_bbox``) on a dataset still holding every variable, and
only then cut time down to the few steps wanted; and the shared multi-point slab
(:func:`ocean_skill.comparison._build_shared_slabs`) cropped space, then time, and
``.load()``-ed every variable. Now the lane is first narrowed to what the request reads
(its own variable, every variable with no time axis -- grid fields, s-coordinate
parameters -- and the free surface), then cut along time, then space. A request that
reads more than one raw variable (a derived geographic velocity, a calculator, a
combination, an isopycnal slice) keeps them all: this is an optimisation and never
changes a value.

The shared slab is also cut to the *union* of the references' own time windows rather
than their hull -- and not at all when any reference declares none, since a slab
truncated to the others' windows silently starves that reference of data.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

import ocean_skill as osk
from ocean_skill import catalog
from ocean_skill.comparison import (
    _SHARED_SLABS,
    _build_shared_slabs,
    _shared_slab,
    prepare_source,
)

NT, N, NY, NX = 240, 4, 10, 12  # ten days of hourly steps
T0 = np.datetime64("2024-01-01T00:00")
TEST_META = {"model": "roms", "vertical": {"hc": 20.0, "s_dim": "s_rho"}}
STATIC = {"h", "sigma_r", "Cs_r"}
POINT_A = (-89.6, 20.4)
POINT_B = (-88.6, 21.6)


def _model(nt=NT) -> xr.Dataset:
    """Return a lazy, curvilinear ROMS-like dataset with 3-D fields and a surface."""
    rng = np.random.default_rng(7)
    lon = np.tile(np.linspace(-90.0, -88.0, NX), (NY, 1))
    lat = np.tile(np.linspace(20.0, 22.0, NY)[:, None], (1, NX))
    sigma = (np.arange(N) - N + 0.5) / N
    rho = ("time", "s_rho", "eta_rho", "xi_rho")
    ds = xr.Dataset(
        {
            "temp": (rho, rng.normal(15.0, 1.0, (nt, N, NY, NX))),
            "salt": (rho, rng.normal(34.0, 0.2, (nt, N, NY, NX))),
            "oxygen": (rho, rng.normal(200.0, 5.0, (nt, N, NY, NX))),
            "zeta": (("time", "eta_rho", "xi_rho"), rng.normal(0.0, 0.5, (nt, NY, NX))),
            "h": (("eta_rho", "xi_rho"), np.full((NY, NX), 100.0)),
            "sigma_r": ("s_rho", sigma),
            "Cs_r": ("s_rho", sigma),
        },
        coords={
            "time": T0 + np.arange(nt) * np.timedelta64(1, "h"),
            "lon": (("eta_rho", "xi_rho"), lon),
            "lat": (("eta_rho", "xi_rho"), lat),
            "mask_rho": (("eta_rho", "xi_rho"), np.ones((NY, NX))),
        },
    )
    return ds.chunk({"time": 1})


def _stub(monkeypatch, ds, coverage=None):
    """Route ``osk.read``/``catalog.resolve`` for ``"his"`` and fixed-point refs.

    ``coverage`` maps a reference name to its declared ``(start, stop)`` (or ``None``).
    Returns the dict of positions the references sit at.
    """
    coverage = coverage or {}
    positions = {"a": POINT_A, "b": POINT_B}
    reads = {"his": 0}

    def fake_read(name, **kw):
        assert name == "his", name
        reads["his"] += 1
        return ds

    def fake_resolve(name):
        if name == "his":
            return SimpleNamespace(metadata=TEST_META)
        lon, lat = positions[name]
        meta = {
            "featureType": "timeSeries",
            "geospatial_lon_min": lon,
            "geospatial_lon_max": lon,
            "geospatial_lat_min": lat,
            "geospatial_lat_max": lat,
        }
        window = coverage.get(name)
        if window is not None:
            meta["time_coverage_start"], meta["time_coverage_end"] = window
        return SimpleNamespace(metadata=meta)

    monkeypatch.setattr(osk, "read", fake_read)
    monkeypatch.setattr(catalog, "resolve", fake_resolve)
    return positions


@pytest.fixture(autouse=True)
def _clear_slabs():
    _SHARED_SLABS.clear()
    yield
    _SHARED_SLABS.clear()


# -- prepare_source: the order of the crops, and what is left to crop -------------


def _spy_crops(monkeypatch):
    """Record each align crop as ``(name, time-varying variables standing)``."""
    import ocean_skill.align as align

    seen = []

    def spy(name):
        real = getattr(align, name)

        def wrapper(obj, *args, **kwargs):
            seen.append(
                (name, sorted(v for v in obj.data_vars if "time" in obj[v].dims))
            )
            return real(obj, *args, **kwargs)

        monkeypatch.setattr(align, name, wrapper)

    for name in ("subset_to_bbox", "subset_to_time", "subset_to_time_targets"):
        spy(name)
    return seen


def _prepare(variable="temp", select=None, **kw):
    lon, lat = POINT_A
    kw.setdefault(
        "time_window", (T0 + np.timedelta64(24, "h"), T0 + np.timedelta64(72, "h"))
    )
    kw.setdefault("time_targets", np.array([T0 + np.timedelta64(30, "h")]))
    return prepare_source(
        "his",
        variable,
        {"depth": "surface", **(select or {})},
        None,
        use_cache=False,
        bbox=(lon, lat, lon, lat),
        **kw,
    )


def test_prepare_source_cuts_time_before_space(monkeypatch):
    _stub(monkeypatch, _model())
    seen = _spy_crops(monkeypatch)
    da, _ = _prepare()
    assert da is not None
    assert [name for name, _ in seen] == [
        "subset_to_time",
        "subset_to_time_targets",
        "subset_to_bbox",
    ]


def test_prepare_source_narrows_before_any_crop(monkeypatch):
    _stub(monkeypatch, _model())
    seen = _spy_crops(monkeypatch)
    _prepare()
    # the request's own variable and the free surface; salinity, oxygen never read
    assert {tuple(standing) for _, standing in seen} == {("temp", "zeta")}


@pytest.mark.parametrize(
    ("variable", "select", "standing"),
    [
        # a plain name reads that one variable (and the free surface)
        ("salt", None, ["salt", "zeta"]),
        # an isopycnal needs the density, so temperature and salinity both -- and all
        # of them stay, rather than guess which the request reads
        ("oxygen", {"sigma0": 26.5}, ["oxygen", "salt", "temp", "zeta"]),
        # a combination reads each of its components
        ({"sum": ["temp", "salt"]}, None, ["oxygen", "salt", "temp", "zeta"]),
    ],
)
def test_the_standing_variables_follow_what_the_request_reads(
    monkeypatch, variable, select, standing
):
    _stub(monkeypatch, _model())
    seen = _spy_crops(monkeypatch)
    try:
        _prepare(variable, select)
    except Exception:
        pass  # the request itself may not run on this toy grid; only the crops matter
    assert seen, "no crop ran"
    assert seen[0][1] == standing


def test_narrowing_changes_no_value(monkeypatch):
    ds = _model()
    _stub(monkeypatch, ds)
    narrowed, _ = _prepare()
    monkeypatch.setattr(
        "ocean_skill.comparison._narrow_to_variables", lambda obj, *a, **k: obj
    )
    full, _ = _prepare()
    xr.testing.assert_allclose(narrowed, full, atol=0, rtol=0)


# -- the shared slab ---------------------------------------------------------------


def _slab(monkeypatch, variables=None, coverage=None, ds=None):
    positions = _stub(monkeypatch, ds if ds is not None else _model(), coverage)
    _build_shared_slabs(list(positions), ["his"], None, variables)
    return _shared_slab("his", None)


def test_a_slab_built_for_one_variable_holds_that_variable_and_the_statics(monkeypatch):
    slab = _slab(monkeypatch, variables=["temp"])
    assert set(slab.data_vars) == {"temp", "zeta"} | STATIC


def test_a_slab_for_no_named_variables_keeps_them_all(monkeypatch):
    slab = _slab(monkeypatch, variables=None)
    assert set(slab.data_vars) == {"temp", "salt", "oxygen", "zeta"} | STATIC


def test_a_slab_for_several_variables_holds_their_union(monkeypatch):
    slab = _slab(monkeypatch, variables=["temp", {"test": "salt", "reference": "x"}])
    assert set(slab.data_vars) == {"temp", "salt", "zeta"} | STATIC


def test_a_slab_keeps_everything_when_any_variable_cannot_be_narrowed(monkeypatch):
    slab = _slab(monkeypatch, variables=["temp", {"calculate": "mld", "method": "x"}])
    assert set(slab.data_vars) == {"temp", "salt", "oxygen", "zeta"} | STATIC


def test_a_slab_cuts_time_before_it_cuts_space(monkeypatch):
    order = []
    real = xr.Dataset.isel

    def spy(self, indexers=None, *args, **kwargs):
        keys = tuple(sorted(indexers or kwargs))
        if "time" in keys or "eta_rho" in keys:
            order.append(keys)
        return real(self, indexers, *args, **kwargs)

    positions = _stub(
        monkeypatch,
        _model(),
        {"a": ("2024-01-02", "2024-01-02"), "b": ("2024-01-02", "2024-01-02")},
    )
    monkeypatch.setattr(xr.Dataset, "isel", spy)
    _build_shared_slabs(list(positions), ["his"], None, ["temp"])
    first_time = next(i for i, keys in enumerate(order) if "time" in keys)
    first_space = next(i for i, keys in enumerate(order) if "eta_rho" in keys)
    assert first_time < first_space


def test_a_slab_is_cut_to_the_union_of_the_windows_not_their_hull(monkeypatch):
    # two sparse windows, a day each side of Jan 2 and of Jan 9
    slab = _slab(
        monkeypatch,
        ["temp"],
        {"a": ("2024-01-02", "2024-01-02"), "b": ("2024-01-09", "2024-01-09")},
    )
    stamps = slab["time"].values
    # Jan 1 00:00 .. Jan 3 00:00 (49 steps) plus the step after; Jan 8 00:00 ..
    # Jan 10 00:00 (49 steps) plus one either side -- and nothing in between
    assert stamps.size == 50 + 51
    assert not (
        (stamps > np.datetime64("2024-01-03T01:00"))
        & (stamps < np.datetime64("2024-01-07T23:00"))
    ).any()


def test_a_slab_holds_a_bracketing_step_past_each_window_edge(monkeypatch):
    """A model coarser than the window padding still has steps to bracket with."""
    ds = _model().isel(time=slice(0, NT, 48))  # one step every two days
    slab = _slab(monkeypatch, ["temp"], {"a": ("2024-01-04", "2024-01-04")}, ds=ds)
    # the window (Jan 3 .. Jan 5) holds the Jan 3 step; Jan 1 and Jan 5 bracket it
    assert slab is not None
    stamps = slab["time"].values
    assert stamps.min() <= np.datetime64("2024-01-03")
    assert stamps.max() >= np.datetime64("2024-01-05")


def test_a_reference_with_no_declared_window_stops_the_time_crop(monkeypatch):
    """Before: one declared window cropped the slab and starved the other reference."""
    slab = _slab(monkeypatch, ["temp"], {"a": ("2024-01-03", "2024-01-04")})
    assert slab["time"].size == NT


def test_every_reference_declaring_a_window_still_crops(monkeypatch):
    slab = _slab(
        monkeypatch,
        ["temp"],
        {"a": ("2024-01-03", "2024-01-04"), "b": ("2024-01-03", "2024-01-04")},
    )
    assert slab["time"].size < NT


@pytest.mark.parametrize("name", ["a", "b"])
def test_values_through_a_narrowed_cropped_slab_match_a_fresh_read(monkeypatch, name):
    ds = _model()
    coverage = {"a": ("2024-01-02", "2024-01-03"), "b": ("2024-01-08", "2024-01-09")}
    positions = _stub(monkeypatch, ds, coverage)
    lon, lat = positions[name]
    window = (
        np.datetime64(coverage[name][0]) - np.timedelta64(1, "D"),
        np.datetime64(coverage[name][1]) + np.timedelta64(1, "D"),
    )

    def prepared():
        return prepare_source(
            "his",
            "temp",
            {"depth": "surface"},
            None,
            use_cache=False,
            bbox=(lon, lat, lon, lat),
            time_window=window,
        )[0]

    _build_shared_slabs(list(positions), ["his"], None, ["temp"])
    assert _shared_slab("his", None) is not None
    through_slab = prepared()
    _SHARED_SLABS.clear()
    fresh = prepared()
    xr.testing.assert_allclose(through_slab, fresh, atol=0, rtol=0)
    assert through_slab["time"].size == fresh["time"].size > 1


@pytest.mark.parametrize("method", ["nearest", "interp"])
def test_time_targets_before_the_spatial_crop_give_the_same_values(monkeypatch, method):
    """The order is only a cost: the value at the station is what cutting last gives."""
    ds = _model()
    _stub(monkeypatch, ds)
    target = T0 + np.timedelta64(90, "m")  # half way between two hourly steps
    da, _ = _prepare(
        time_window=None, time_targets=np.array([target]), time_targets_method=method
    )

    full = ds["temp"].isel(s_rho=-1)
    if method == "interp":
        # one resolution on both sides, or xarray reads the targets as out of range
        full = full.assign_coords(time=full["time"].values.astype("datetime64[ns]"))
        expected_field = full.interp(time=target.astype("datetime64[ns]"))
    else:
        expected_field = full.sel(time=target, method="nearest")

    def at_point(field):
        d2 = (field["lon"] - POINT_A[0]) ** 2 + (field["lat"] - POINT_A[1]) ** 2
        iy, ix = np.unravel_index(int(np.argmin(d2.values)), d2.shape)
        return float(field.isel(eta_rho=iy, xi_rho=ix).squeeze())

    assert at_point(da) == pytest.approx(at_point(expected_field), abs=1e-12)
