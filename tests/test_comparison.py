"""Tests for the compare layer: variable aliasing and the surface/depth=0 distinction.

Both regressions here were found via a real ``osk.compare(..., variables=[OXYGEN])``
call against the GOM MARBL output: the aliased-variable crash
(``ValueError: could not convert string to float: b'T'``, from a bare ``.mean("time")``
falling through to the *whole* dataset including non-numeric fields like ``spherical``)
and the silent conflation of "surface" with "depth=0". Both are reproduced here on a
small synthetic ROMS-shaped dataset — self-contained, no external run directory or
kerchunk reference needed.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import xarray as xr

from ocean_skill import roms
from ocean_skill.comparison import (
    SURFACE,
    _depth_label,
    _prepare,
    _require_reduced,
    is_surface_request,
)

# WOA/GLODAP spell dissolved oxygen per-mass; ROMS/MARBL writes it per-volume (see
# ocean_skill.vocabulary.VOCABULARY) — comparing across that alias is what broke.
OXYGEN_PER_MASS = "moles_of_oxygen_per_unit_mass_in_sea_water"
OXYGEN_PER_VOLUME = "mole_concentration_of_dissolved_molecular_oxygen_in_sea_water"


@pytest.fixture(scope="module")
def gom_bgc():
    """Build a minimal self-contained ROMS/MARBL-shaped dataset.

    Carries a byte-string field (``spherical``) standing in for the real non-numeric
    global ROMS carries — the exact shape that broke a naive ``.mean("time")`` over
    the whole dataset.
    """
    rng = np.random.default_rng(0)
    nt, ns, ny, nx = 3, 4, 5, 6
    lon = np.linspace(260.0, 262.0, nx)[None, :] * np.ones((ny, 1))
    lat = np.linspace(20.0, 22.0, ny)[:, None] * np.ones((1, nx))
    h = np.full((ny, nx), 50.0)  # shallow: puts the top cell centre below z=0
    mask = np.ones((ny, nx))
    sigma_r = np.linspace(-1 + 1 / (2 * ns), -1 / (2 * ns), ns)  # cell centres
    cs_r = sigma_r.copy()  # Vtransform 2, theta_s=theta_b=0 => Cs_r == sigma_r
    o2 = 200 + 5 * rng.standard_normal((nt, ns, ny, nx))

    ds = xr.Dataset(
        {
            "O2": (("time", "s_rho", "eta_rho", "xi_rho"), o2),
            "zeta": (("time", "eta_rho", "xi_rho"), np.zeros((nt, ny, nx))),
            "h": (("eta_rho", "xi_rho"), h),
            "mask_rho": (("eta_rho", "xi_rho"), mask),
            "lon_rho": (("eta_rho", "xi_rho"), lon),
            "lat_rho": (("eta_rho", "xi_rho"), lat),
            "Cs_r": (("s_rho",), cs_r),
            "sigma_r": (("s_rho",), sigma_r),
            "ocean_time": (("time",), np.arange(nt) * 86400.0),
            "spherical": np.bytes_(b"T"),  # the field that broke a whole-dataset mean
        }
    )
    meta = {
        "model": "roms",
        "self_contained_grid": True,
        "standard_names": {"O2": OXYGEN_PER_VOLUME},
        "vertical": {"s_dim": "s_rho", "hc": 300.0, "Vtransform": 2},
        "reference_date": "2000-01-01",
    }
    return roms.standardize(ds, meta), meta


def test_prepare_resolves_aliased_variable(gom_bgc):
    """A per-mass request must resolve to ROMS' per-volume variable.

    Otherwise it falls through to the whole dataset, where a bare ``.mean("time")``
    chokes on non-numeric fields like ``spherical``.
    """
    ds, meta = gom_bgc
    # the reduction is passed explicitly now that there is no default one: the crash
    # this guards needs a mean to actually happen
    da, _ = _prepare(ds, meta, OXYGEN_PER_MASS, {}, {"time": "mean"})
    assert da is not None
    assert np.isfinite(da.values).any()
    assert "time" not in da.dims


def test_prepare_fails_closed_on_missing_variable(gom_bgc):
    """A genuinely absent variable must return (None, None).

    Not fall through to reducing the whole dataset — the same crash risk the alias
    bug hit, for a different reason: nothing found rather than the wrong name found.
    """
    ds, meta = gom_bgc
    da, depth = _prepare(ds, meta, "not_a_real_standard_name", {})
    assert da is None
    assert depth is None


def test_surface_and_depth_zero_are_distinct(gom_bgc):
    """Unset/``"surface"`` uses the model's own top level, with no warning.

    An explicit ``depth=0`` is a real interpolation request and may legitimately warn
    all-NaN.
    """
    ds, meta = gom_bgc
    assert is_surface_request(None)
    assert is_surface_request(SURFACE)
    assert not is_surface_request(0)
    assert not is_surface_request(0.0)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        da_surface, _ = _prepare(ds, meta, OXYGEN_PER_MASS, {})
    assert not any("entirely NaN" in str(w.message) for w in caught)
    assert np.isfinite(da_surface.values).all()

    # The synthetic grid is 50 m deep everywhere with a top cell centre well below
    # 0 m, so an explicit request for the literal surface interpolates to nothing.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _prepare(ds, meta, OXYGEN_PER_MASS, {"depth": 0})
    assert any("entirely NaN" in str(w.message) for w in caught)


def test_nothing_is_reduced_unless_a_reduction_is_named(gom_bgc):
    """No default aggregation: the axes you did not name survive.

    The reverse of what this module used to assume. ``{"time": "mean"}`` was applied
    whenever a caller named nothing, so ``select={"time": "2010-01"}`` returned
    January's mean and the caller had no way to notice but the shape. The choice is now
    the caller's, and a comparison that needs one is told (see
    :func:`test_a_comparison_refuses_an_unreduced_lane`).
    """
    ds, meta = gom_bgc
    kept, _ = _prepare(ds, meta, OXYGEN_PER_MASS, {})
    assert kept.sizes["time"] == ds.sizes["time"], "something reduced time unasked"
    collapsed, _ = _prepare(ds, meta, OXYGEN_PER_MASS, {}, {"time": "mean"})
    assert "time" not in collapsed.dims


def test_a_comparison_refuses_an_unreduced_lane(gom_bgc):
    """And says which lane, which axis, and what to do — before computing anything."""
    ds, meta = gom_bgc
    da, _ = _prepare(ds, meta, OXYGEN_PER_MASS, {})
    with pytest.raises(ValueError) as excinfo:
        _require_reduced(da, "test", "GOM_bgc")
    message = str(excinfo.value)
    assert "GOM_bgc" in message, "the failing lane has to be named"
    assert f"time={ds.sizes['time']}" in message, "so does the axis and its size"
    assert 'aggregate={"time": "mean"}' in message
    assert "osk.field()" in message, "the way to keep the axis instead"


def test_a_reduced_lane_passes_the_check(gom_bgc):
    ds, meta = gom_bgc
    da, _ = _prepare(ds, meta, OXYGEN_PER_MASS, {}, {"time": "mean"})
    out = _require_reduced(da, "test", "GOM_bgc")  # must not raise
    xr.testing.assert_identical(out, da)  # an already-reduced lane comes back unchanged


def test_a_singleton_axis_is_squeezed_not_refused(gom_bgc):
    """A WOA climatology's ``time=1`` needs no ``aggregate`` boilerplate to pass.

    Squeezing a size-1 axis changes no number — the one value already is the mean —
    so :func:`_require_reduced` collapses it itself rather than asking the caller to
    say how, the same way it would refuse a genuinely ambiguous axis.
    """
    ds, meta = gom_bgc
    da, _ = _prepare(ds, meta, OXYGEN_PER_MASS, {})
    one = da.isel(time=slice(0, 1))
    out = _require_reduced(one, "test", "GOM_bgc")  # must not raise
    assert "time" not in out.dims
    np.testing.assert_array_equal(out.values, one.isel(time=0).values)


def test_a_singleton_does_not_excuse_a_real_axis(gom_bgc):
    """A size-1 axis is squeezed away quietly; a genuinely ambiguous one still isn't."""
    ds, meta = gom_bgc
    da, _ = _prepare(ds, meta, OXYGEN_PER_MASS, {})
    one = da.isel(time=slice(0, 1)).expand_dims(depth=[0.0, 10.0])
    with pytest.raises(ValueError) as excinfo:
        _require_reduced(one, "test", "GOM_bgc")
    message = str(excinfo.value)
    assert "depth=2" in message, "the real axis has to be named"
    assert "time=" not in message, "the singleton it squeezed away should not be"


def test_a_cached_lane_is_still_checked_for_reduction(monkeypatch, gom_bgc):
    """The lane cache is shared between callers that disagree about a surviving axis.

    ``key_for_prepared`` deliberately excludes ``require_reduced`` — the same field is
    the same field however it is about to be used — so a lane entry written by a caller
    that keeps the time axis (a ``Field``, or a comparison scoring ``over="time"``) must
    not let a plain comparison past its own gate on the way back out.
    """
    from types import SimpleNamespace

    import ocean_skill as osk
    from ocean_skill import catalog
    from ocean_skill.comparison import prepare_source

    ds, meta = gom_bgc
    monkeypatch.setattr(osk, "read", lambda name: ds)
    monkeypatch.setattr(catalog, "resolve", lambda name: SimpleNamespace(metadata=meta))

    kept, _ = prepare_source("GOM_bgc", OXYGEN_PER_MASS, None, None)
    assert "time" in kept.dims, "the tolerant caller keeps the axis and fills the cache"

    with pytest.raises(ValueError, match="GOM_bgc") as excinfo:
        prepare_source("GOM_bgc", OXYGEN_PER_MASS, None, None, require_reduced="test")
    assert "time=" in str(excinfo.value)


def test_a_cached_singleton_lane_is_squeezed_on_the_way_out(monkeypatch, gom_bgc):
    """The squeeze applies to what a caller receives, never to what the cache holds.

    A field cached with its singleton ``time`` standing (written by a tolerant caller)
    must come back squeezed to a stricter caller on a cache *hit* — the same rule
    :func:`_require_reduced` applies on a cache miss — while the entry itself, read
    again by a tolerant caller, still shows the un-squeezed axis it was written with.
    """
    from types import SimpleNamespace

    import ocean_skill as osk
    from ocean_skill import catalog
    from ocean_skill.comparison import prepare_source

    ds, meta = gom_bgc
    one_step = ds.isel(time=slice(0, 1))
    monkeypatch.setattr(osk, "read", lambda name: one_step)
    monkeypatch.setattr(catalog, "resolve", lambda name: SimpleNamespace(metadata=meta))

    kept, _ = prepare_source("GOM_bgc_1", OXYGEN_PER_MASS, None, None)
    assert kept.sizes["time"] == 1, "the tolerant caller fills the cache unsqueezed"

    hit, _ = prepare_source(
        "GOM_bgc_1", OXYGEN_PER_MASS, None, None, require_reduced="test"
    )
    assert "time" not in hit.dims, "a stricter caller gets the cache hit squeezed"

    again, _ = prepare_source("GOM_bgc_1", OXYGEN_PER_MASS, None, None)
    assert again.sizes["time"] == 1, "the stored entry itself was never squeezed"


def test_depth_label():
    assert _depth_label(None) == "surface"
    assert _depth_label(SURFACE) == "surface"
    assert _depth_label(0) == "0 m"
    assert _depth_label(100.0) == "100 m"


# -- narrowing an ERDDAP lane before it is downloaded -------------------------


@pytest.fixture
def station():
    """Build a year of daily station data, standing in for a moored ERDDAP table."""
    import pandas as pd

    time = pd.date_range("2015-01-01", periods=365, freq="D")
    return xr.Dataset(
        {"temp": ("time", np.arange(365, dtype=float))}, coords={"time": time}
    )


@pytest.fixture
def lane(monkeypatch, station):
    """Run ``prepare_source`` against ``station``, capturing what reached ``read``."""
    from types import SimpleNamespace

    import ocean_skill as osk
    from ocean_skill import catalog
    from ocean_skill.comparison import prepare_source

    seen: list[dict] = []

    def run(meta, select=None, aggregate=None, **kwargs):
        monkeypatch.setattr(
            osk, "read", lambda name, **kw: (seen.append(kw), station)[1]
        )
        monkeypatch.setattr(
            catalog, "resolve", lambda name: SimpleNamespace(metadata=meta)
        )
        da, _ = prepare_source(
            "mooring", "temp", select, aggregate, use_cache=False, **kwargs
        )
        return da, seen[-1]

    return run


#: An ERDDAP tabledap entry's metadata, as `intake_erddap` writes it into a catalog.
TABLEDAP = {"tabledap": "https://example.org/erddap/tabledap/mooring"}


def test_a_time_select_reaches_an_erddap_read_as_a_constraint(lane):
    """The whole point: ERDDAP fetches the table whole, so `select` must travel with it.

    Before this the request went out unconstrained and the twelve-year record came down
    to be trimmed in memory -- the download the caller believed they had narrowed.
    """
    da, kwargs = lane(TABLEDAP, select={"time": slice("2015-02-01", "2015-02-28")})
    assert kwargs["constraints"] == {
        "time>=": "2015-02-01T00:00:00Z",
        "time<=": "2015-02-28T23:59:59Z",
    }
    # ...and the in-memory select still runs, so the result does not depend on whether
    # the server honoured the constraint. The fixture ignores it and returns the year.
    assert da.sizes["time"] == 28


def test_a_derived_window_reaches_an_erddap_read_as_a_constraint(lane):
    """`over=` works the test lane's span out itself, then cropped *after* the read."""
    window = (np.datetime64("2015-03-01"), np.datetime64("2015-03-31"))
    da, kwargs = lane(TABLEDAP, select=None, aggregate=None, time_window=window)
    assert kwargs["constraints"] == {
        "time>=": "2015-03-01T00:00:00Z",
        "time<=": "2015-03-31T00:00:00Z",
    }
    assert da.sizes["time"] == 31


def test_a_non_erddap_lane_is_read_exactly_as_before(lane):
    """The gate: a lazily-opened gridded source neither needs this nor sees it."""
    da, kwargs = lane({}, select={"time": slice("2015-02-01", "2015-02-28")})
    assert kwargs == {}
    assert da.sizes["time"] == 28


def test_a_derived_reference_window_reaches_a_tabledap_test_as_a_constraint(lane):
    """A ``pd.Timestamp`` window -- what ``_time_coverage_of`` produces when deriving
    a crop from a point-like reference's catalog metadata -- survives the same
    stamping path as the ``np.datetime64`` window a skill map derives from its test
    lane's own data (see the test above this one). The two producers disagree on
    dtype; ``erddap_constraints``'s ``_stamp`` must not."""
    import pandas as pd

    window = (pd.Timestamp("2015-03-01"), pd.Timestamp("2015-03-31"))
    da, kwargs = lane(TABLEDAP, select=None, aggregate=None, time_window=window)
    assert kwargs["constraints"] == {
        "time>=": "2015-03-01T00:00:00Z",
        "time<=": "2015-03-31T00:00:00Z",
    }
    assert da.sizes["time"] == 31


def test_a_derived_bbox_and_window_apply_together(monkeypatch):
    """``bbox=`` and ``time_window=`` crop independently and their effects compose --
    the shape a metadata-derived narrowing hands the test lane (spatial *and*
    temporal at once), on a non-ERDDAP (in-memory-cropped) source."""
    from types import SimpleNamespace

    import pandas as pd

    import ocean_skill as osk
    from ocean_skill import catalog
    from ocean_skill.comparison import prepare_source

    time = pd.date_range("2015-01-01", periods=10, freq="D")
    lat = np.array([10.0, 20.0, 30.0])
    lon = np.array([100.0, 110.0, 120.0])
    values = np.arange(10 * 3 * 3, dtype=float).reshape(10, 3, 3)
    ds = xr.Dataset(
        {"temp": (("time", "lat", "lon"), values)},
        coords={"time": time, "lat": lat, "lon": lon},
    )
    monkeypatch.setattr(osk, "read", lambda name, **kw: ds)
    monkeypatch.setattr(catalog, "resolve", lambda name: SimpleNamespace(metadata={}))

    da, _ = prepare_source(
        "grid",
        "temp",
        None,
        None,
        use_cache=False,
        bbox=(105.0, 15.0, 115.0, 25.0),
        time_window=(np.datetime64("2015-01-03"), np.datetime64("2015-01-05")),
    )
    assert list(da["lon"].values) == [110.0]
    assert list(da["lat"].values) == [20.0]
    assert da.sizes["time"] == 3


def test_two_windows_over_one_source_do_not_share_a_cache_entry(monkeypatch, station):
    """A cropped lane is not the uncropped one, and `_bbox` alone did not say so."""
    from ocean_skill.cache import key_for_prepared

    keys = {
        key_for_prepared(
            source="mooring",
            variable="temp",
            select={"_aggregate": None, "_time_window": [str(w) for w in window]},
        )
        for window in (
            (np.datetime64("2015-03-01"), np.datetime64("2015-03-31")),
            (np.datetime64("2016-03-01"), np.datetime64("2016-03-31")),
        )
    }
    assert len(keys) == 2


def test_a_point_bbox_folds_a_marker_into_the_lane_key(monkeypatch):
    """A degenerate (point) bbox is cropped by cells, not by the degree pad a
    region bbox still gets -- a lane cached under the old, wider policy must not
    silently keep being served just because its rounded `_bbox` still matches.
    """
    from types import SimpleNamespace

    import pandas as pd

    import ocean_skill as osk
    from ocean_skill import cache, catalog
    from ocean_skill.comparison import prepare_source

    time = pd.date_range("2015-01-01", periods=3, freq="D")
    ds = xr.Dataset(
        {"temp": (("time", "lat", "lon"), np.ones((3, 2, 2)))},
        coords={"time": time, "lat": [10.0, 20.0], "lon": [100.0, 110.0]},
    )
    monkeypatch.setattr(osk, "read", lambda name, **kw: ds)
    monkeypatch.setattr(catalog, "resolve", lambda name: SimpleNamespace(metadata={}))

    saved_keys = []
    real_save_field = cache.save_field

    def recording_save_field(key, *a, **kw):
        saved_keys.append(key)
        return real_save_field(key, *a, **kw)

    monkeypatch.setattr(cache, "save_field", recording_save_field)

    prepare_source(
        "grid", "temp", None, None, use_cache=True, bbox=(102.0, 12.0, 102.0, 12.0)
    )
    prepare_source(
        "grid", "temp", None, None, use_cache=True, bbox=(100.0, 10.0, 110.0, 20.0)
    )
    assert len(saved_keys) == 2
    assert len(set(saved_keys)) == 2  # a point crop and a region crop never collide

    from ocean_skill.align import POINT_WINDOW_CELLS
    from ocean_skill.cache import key_for_prepared

    point_key = key_for_prepared(
        source="grid",
        variable="temp",
        select={
            "_aggregate": None,
            "_bbox": [102.0, 12.0, 102.0, 12.0],
            "_point_window": POINT_WINDOW_CELLS,
        },
    )
    assert point_key in saved_keys
    pre_existing_key = key_for_prepared(
        source="grid",
        variable="temp",
        select={"_aggregate": None, "_bbox": [102.0, 12.0, 102.0, 12.0]},
    )
    assert pre_existing_key not in saved_keys
