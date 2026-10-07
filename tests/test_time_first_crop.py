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

import warnings
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

    for name in (
        "subset_to_bbox",
        "subset_to_time",
        "preselect_time_targets",
        "subset_to_time_targets",
    ):
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
    # time first (the window, then a pure-indexing pick of the steps the targets
    # use), then space, and only then the targets' own work -- interpolation -- on the
    # few cells that are left
    assert [name for name, _ in seen] == [
        "subset_to_time",
        "preselect_time_targets",
        "subset_to_bbox",
        "subset_to_time_targets",
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


# -- the targets: indexing before the spatial crop, interpolation after it ------


def _no_preselection(monkeypatch):
    """Make the pre-selection a no-op: the bbox-then-targets order this replaces."""
    monkeypatch.setattr(
        "ocean_skill.align.preselect_time_targets",
        lambda obj, targets, method="nearest": (obj, targets),
    )


def test_interpolation_runs_on_the_bracket_steps_and_the_windows_cells(monkeypatch):
    """What reaches the targets' own work: a few steps of a few cells."""
    ds = _model()
    _stub(monkeypatch, ds)
    handed = []
    import ocean_skill.align as align

    real = align.subset_to_time_targets

    def spy(obj, targets, method="nearest"):
        handed.append((dict(obj.sizes), obj["temp"].chunks is not None))
        return real(obj, targets, method=method)

    monkeypatch.setattr(align, "subset_to_time_targets", spy)
    # three half-hour instants: brackets are steps 1-2, 5-6 and 100-101
    targets = np.array([T0 + np.timedelta64(m, "m") for m in (90, 330, 6030)])
    _prepare(
        time_window=None,
        time_targets=targets,
        time_targets_method="interp",
        point_window_cells=1,
    )
    ((sizes, lazy),) = handed
    assert lazy
    assert sizes["time"] == 6
    assert (sizes["eta_rho"], sizes["xi_rho"]) == (
        3,
        3,
    )  # the point window, not 10 x 12


def test_contiguous_bracket_steps_are_taken_as_one_slice(monkeypatch):
    ds = _model()
    _stub(monkeypatch, ds)
    isel_keys = []
    real = xr.Dataset.isel

    def spy(self, indexers=None, *args, **kwargs):
        isel_keys.append(indexers if indexers is not None else kwargs)
        return real(self, indexers, *args, **kwargs)

    monkeypatch.setattr(xr.Dataset, "isel", spy)
    targets = np.array([T0 + np.timedelta64(m, "m") for m in (90, 150, 210)])
    _prepare(time_window=None, time_targets=targets, time_targets_method="interp")
    on_time = [k["time"] for k in isel_keys if "time" in k]
    # steps 1..4 are consecutive: one slice, no index array
    assert isinstance(on_time[0], slice)
    assert on_time[0] == slice(1, 5)


@pytest.mark.parametrize("method", ["nearest", "interp"])
def test_the_preselection_changes_no_value(monkeypatch, method):
    ds = _model()
    _stub(monkeypatch, ds)
    # between steps, on a step, close to the first and to the last, and (for interp)
    # one past the end of the record, which has nothing to interpolate between
    targets = np.array(
        [T0 + np.timedelta64(m, "m") for m in (5, 90, 600, 14395, 14500)]
    )
    kw = dict(time_window=None, time_targets=targets, time_targets_method=method)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with_it, _ = _prepare(**kw)
        _no_preselection(monkeypatch)
        without_it, _ = _prepare(**kw)
    assert with_it["time"].size == without_it["time"].size > 1
    xr.testing.assert_allclose(with_it, without_it, atol=0, rtol=0)


def test_a_target_past_the_record_is_reported_once_with_the_records_own_span(
    monkeypatch,
):
    _stub(monkeypatch, _model())
    targets = np.array([T0 + np.timedelta64(m, "m") for m in (90, 14500)])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _prepare(time_window=None, time_targets=targets, time_targets_method="interp")
    outside = [str(w.message) for w in caught if "fall outside" in str(w.message)]
    assert len(outside) == 1
    # the whole record's span (not the two bracket steps that happen to be left)
    assert "2024-01-01T00:00" in outside[0]
    assert "2024-01-10T23:00" in outside[0]


def _messy(n=12):
    """Return a dataset whose time axis is out of order and repeats a stamp."""
    stamps = T0 + np.array([3, 1, 2, 2, 0, 5, 4, 7, 6, 9, 8, 11]) * np.timedelta64(
        1, "h"
    )
    return xr.Dataset(
        {"v": (("time", "x"), np.arange(n * 2.0).reshape(n, 2))},
        coords={"time": stamps, "x": [0, 1]},
    )


@pytest.mark.parametrize("method", ["nearest", "interp"])
@pytest.mark.parametrize("messy", [False, True])
def test_the_preselection_leaves_the_targets_what_they_would_have_got(method, messy):
    """The two share one decision about which steps a target uses."""
    from ocean_skill.align import preselect_time_targets, subset_to_time_targets

    if messy:
        ds = _messy()
    else:
        ds = _model().isel(time=slice(0, 12))[["temp"]]
    targets = np.array([T0 + np.timedelta64(m, "m") for m in (20, 135, 400, 5000)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        direct = subset_to_time_targets(ds, targets, method=method)
        narrowed, left = preselect_time_targets(ds, targets, method=method)
        via = subset_to_time_targets(narrowed, left, method=method)
    name = next(v for v in ("temp", "v") if v in ds.data_vars)
    xr.testing.assert_allclose(via[name], direct[name], atol=0, rtol=0)
    assert narrowed.sizes["time"] <= ds.sizes["time"]


def test_the_preselection_changes_nothing_when_there_is_nothing_to_cut():
    from ocean_skill.align import preselect_time_targets

    ds = _model().isel(time=slice(0, 3))
    every = np.array([T0 + np.timedelta64(m, "m") for m in (0, 60, 120)])
    same, left = preselect_time_targets(ds, every, method="nearest")
    assert same is ds
    assert left is every
    for empty in (None, np.array([], dtype="datetime64[ns]")):
        same, _ = preselect_time_targets(ds, empty, method="interp")
        assert same is ds
    single = ds.isel(time=slice(0, 1))
    same, _ = preselect_time_targets(single, every, method="interp")
    assert same is single


# -- the same on real comparisons: a mooring and a cast on the tidal fixture ----


def _comparison_values(monkeypatch, make, noop):
    if noop:
        _no_preselection(monkeypatch)
    c = make()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(c.aligned["test"]).reshape(-1)


def _tidal_install(monkeypatch, name, data, meta):
    from tests._tidal_roms import tidal_roms

    ds, roms_meta = tidal_roms()
    ds = ds.assign_coords(time=ds["time"].values.astype("datetime64[ns]"))
    sources = {"his": (ds, roms_meta), name: (data, meta)}
    monkeypatch.setattr(osk, "read", lambda n, **kw: sources[n][0])
    monkeypatch.setattr("ocean_skill.sources.read", lambda n, **kw: sources[n][0])
    monkeypatch.setattr(
        catalog, "resolve", lambda n: SimpleNamespace(metadata=sources[n][1])
    )
    point = (200.01, 50.01, 200.01, 50.01)  # the middle cell of the fixture's grid
    monkeypatch.setattr(
        "ocean_skill.comparison._domain_of", lambda n: None if n == "his" else point
    )
    monkeypatch.setattr(
        "ocean_skill.comparison._outline_of", lambda n, convention=None: None
    )


SPEC = {"test": "height", "reference": "obs", "standard_name": "sea_water_temperature"}


def _mooring_comparison(monkeypatch, time_method):
    import pandas as pd

    from ocean_skill.comparison import Comparison

    frame = pd.DataFrame(
        {
            "time": pd.date_range("2024-07-01 00:30", periods=3, freq="h"),
            "lon": 200.01,
            "lat": 50.01,
            "obs (m)": 0.0,
        }
    )
    meta = {
        "featureType": "timeSeries",
        "nominal_depth_m": 3.0,
        "depth_convention": {"origin": "surface"},
    }
    _tidal_install(monkeypatch, "pier", frame, meta)
    return lambda: Comparison(
        reference="pier",
        test="his",
        variable=SPEC,
        select={"depth": 3.0},
        over="time",
        depth_method="interp",
        time_method=time_method,
        cache=False,
    )


def _cast_comparison(monkeypatch, time_method):
    import pandas as pd

    from ocean_skill.comparison import Comparison

    cast = pd.DataFrame(
        {
            "time": pd.Timestamp("2024-07-01 00:30"),
            "lon": 200.01,
            "lat": 50.01,
            "depth (m)": [3.0, 5.0],
            "obs (m)": [0.0, 0.0],
        }
    )
    _tidal_install(monkeypatch, "cast", cast, {"featureType": "profile"})
    return lambda: Comparison(
        reference="cast",
        test="his",
        variable=SPEC,
        select={"depth": [3.0, 5.0]},
        over="Z",
        depth_method="interp",
        time_method=time_method,
        cache=False,
    )


@pytest.mark.parametrize("time_method", ["auto", "interp"])
@pytest.mark.parametrize("build", [_mooring_comparison, _cast_comparison])
def test_a_comparison_gets_the_same_values_either_way(monkeypatch, build, time_method):
    import ocean_skill.align as align

    make = build(monkeypatch, time_method)
    picked = []
    real = align.preselect_time_targets

    def counting(obj, targets, method="nearest"):
        picked.append(method)
        return real(obj, targets, method=method)

    monkeypatch.setattr(align, "preselect_time_targets", counting)
    with_it = _comparison_values(monkeypatch, make, noop=False)
    if time_method == "interp" or build is _mooring_comparison:
        assert picked, "the pre-selection never ran: the comparison proves nothing"
    without_it = _comparison_values(monkeypatch, make, noop=True)
    np.testing.assert_array_equal(with_it, without_it)
    if time_method == "interp":
        # interpolated onto the observation's own instants: zeta is 1.5, -1.5, -1.5
        expected = {
            _mooring_comparison: [-1.5, -4.5, -4.5],
            _cast_comparison: [-1.5, -3.5],
        }[build]
        np.testing.assert_allclose(with_it, expected)
