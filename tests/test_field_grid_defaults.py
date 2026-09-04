"""Tests for the two grid defaults a bare :class:`~ocean_skill.field.Field` gets:
a bare vertical select on a catalogued ``featureType: grid`` source draws the
surface, and a bare, genuinely multi-step time axis has no single default instant
and says so rather than guessing one (see ``Field._grid_metadata_if_eligible``,
``Field._surfaced``, ``Field._facet_field_and_depth``).

Mirrors ``tests/test_facet.py``'s ``catalog.resolve`` mocking pattern
(``test_field_plot_draws_no_domain_box_even_with_one_declared``) for the
read-free half of each default, and ``tests/test_field_series.py``'s
``prepare_source`` stub for the post-load fallback half -- ``"stub"``/other
made-up source names are never catalogued, so those tests exercise the
fallback exclusively (``catalog.resolve`` raises ``KeyError`` for them).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

NITRATE = "mole_concentration_of_nitrate_in_sea_water"


def _grid_field(nt: int = 3, with_depth: bool = False):
    time = pd.date_range("2024-01-01", periods=nt, freq="MS")
    lat = np.linspace(60.0, 66.0, 5)
    lon = np.linspace(-25.0, -15.0, 6)
    if not with_depth:
        return xr.DataArray(
            np.random.default_rng(0).normal(5.0, 1.0, (nt, 5, 6)),
            dims=("time", "lat", "lon"),
            coords={"time": time, "lat": lat, "lon": lon},
            name="temperature",
            attrs={"units": "degC"},
        )
    depth = np.array([0.0, 10.0, 50.0])
    return xr.DataArray(
        np.random.default_rng(0).normal(5.0, 1.0, (nt, 3, 5, 6)),
        dims=("time", "depth", "lat", "lon"),
        coords={"time": time, "depth": depth, "lat": lat, "lon": lon},
        name="temperature",
        attrs={"units": "degC"},
    )


def _roms_point_facet(nt: int = 1, ns: int = 4, ny: int = 3, nx: int = 3):
    """A bare ROMS facet: native ``s_rho`` with a 2-D ``z_rho`` -- the shape a
    real catalogued model source leaves standing when nothing narrows depth.
    """
    z_rho = -np.linspace(2.0, 80.0, ns)[:, None, None] * np.ones((1, ny, nx))
    return xr.DataArray(
        np.random.default_rng(1).normal(15.0, 1.0, (nt, ns, ny, nx)),
        dims=("time", "s_rho", "eta_rho", "xi_rho"),
        coords={
            "time": pd.date_range("2024-06-01", periods=nt, freq="D"),
            "z_rho": (("s_rho", "eta_rho", "xi_rho"), z_rho),
            "lon": (
                ("eta_rho", "xi_rho"),
                np.linspace(-20, -19, nx)[None, :] * np.ones((ny, 1)),
            ),
            "lat": (
                ("eta_rho", "xi_rho"),
                np.linspace(64, 65, ny)[:, None] * np.ones((1, nx)),
            ),
        },
        name="temperature",
        attrs={"units": "degC"},
    )


def _resolve(monkeypatch, name: str, metadata: dict) -> None:
    """Mock ``catalog.resolve`` so only ``name`` is catalogued, with ``metadata``."""

    def resolve(source):
        if source == name:
            return SimpleNamespace(metadata=metadata)
        raise KeyError(source)

    monkeypatch.setattr("ocean_skill.catalog.resolve", resolve, raising=True)


def _grid_meta(**overrides) -> dict:
    meta = {
        "featureType": "grid",
        "vertical": {"s_dim": "s_rho"},
        "time_coverage_start": "2024-01-01",
        "time_coverage_end": "2024-06-30",
    }
    meta.update(overrides)
    return meta


def _refuses_to_call(monkeypatch):
    """Make ``comparison.prepare_source`` raise if it is ever reached -- the
    read-free checks must settle before any read is attempted.
    """
    from ocean_skill import comparison

    def boom(*a, **k):
        raise AssertionError("prepare_source was called: not read-free")

    monkeypatch.setattr(comparison, "prepare_source", boom)


def _stub_prepare_source(monkeypatch, field_da, *, capture: dict | None = None):
    from ocean_skill import comparison

    def fake(source, variable, select, aggregate, **kwargs):
        if capture is not None:
            capture["select"] = select
            capture["aggregate"] = aggregate
        return field_da, None

    monkeypatch.setattr(comparison, "prepare_source", fake)


def _make(source: str, **kwargs):
    from ocean_skill.field import field as make_field

    return make_field(source, NITRATE, **kwargs)


# -- the bare-multi-step-time refusal, read-free ------------------------------------------


def test_a_bare_multistep_grid_refuses_read_free(monkeypatch):
    _resolve(monkeypatch, "iceland_his", _grid_meta())
    _refuses_to_call(monkeypatch)

    with pytest.raises(ValueError, match="no single default"):
        _make("iceland_his").plot()


def test_a_timeseriesprofile_entry_is_not_pre_checked(monkeypatch):
    """The same shape of metadata on a ``timeSeriesProfile`` entry (HV5's own
    featureType) never reaches the grid read-free path at all -- it goes
    through ``Field.family`` as a point instead (see ``test_field_time_depth.py``),
    which is why this must NOT raise here.
    """
    from ocean_skill import comparison

    meta = _grid_meta(featureType="timeSeriesProfile")
    _resolve(monkeypatch, "ctd_station_HV5", meta)
    time = pd.date_range("2024-04-01", periods=3, freq="2MS")
    depth = np.array([0.0, 10.0])
    da = xr.DataArray(
        np.random.default_rng(0).normal(5.0, 1.0, (3, 2)),
        dims=("time", "depth"),
        coords={"time": time, "depth": depth},
        name="temperature",
        attrs={"units": "degC"},
    ).assign_coords(lon=-21.8, lat=64.3)
    monkeypatch.setattr(comparison, "prepare_source", lambda *a, **k: (da, None))

    fig = _make("ctd_station_HV5").plot()  # a time_depth panel, not a refusal
    assert fig is not None


def test_a_named_time_bypasses_the_refusal(monkeypatch):
    meta = _grid_meta()
    _resolve(monkeypatch, "iceland_his", meta)
    capture: dict = {}
    _stub_prepare_source(monkeypatch, _grid_field(nt=1), capture=capture)

    _make("iceland_his", select={"time": "2024-02-01"}).plot()
    assert capture["select"]["depth"] == "surface"  # the vertical default still ran


def test_a_time_aggregate_bypasses_the_refusal(monkeypatch):
    meta = _grid_meta()
    _resolve(monkeypatch, "iceland_his", meta)
    _stub_prepare_source(monkeypatch, _grid_field(nt=6))

    fig = _make(
        "iceland_his",
        aggregate={"time": {"resample": "1MS", "reduce": "mean"}},
    ).plot()
    assert fig is not None


def test_movie_is_never_refused_for_bare_time(monkeypatch):
    """``.movie()`` gets only the surface half of the grid defaults -- a bare,
    multi-step time axis is exactly what it is for.
    """
    meta = _grid_meta()
    _resolve(monkeypatch, "iceland_his", meta)
    capture: dict = {}
    _stub_prepare_source(monkeypatch, _grid_field(nt=4), capture=capture)

    _make("iceland_his").movie(save=None)
    assert capture["select"]["depth"] == "surface"


def test_a_short_declared_coverage_falls_through_to_the_post_load_check(monkeypatch):
    """Same-day (or near-enough) declared coverage cannot settle it read-free --
    the post-load check on the *actual* surviving time dim still catches a
    genuinely multi-step field.
    """
    meta = _grid_meta(
        time_coverage_start="2024-01-01", time_coverage_end="2024-01-01"
    )
    _resolve(monkeypatch, "iceland_his", meta)
    _stub_prepare_source(monkeypatch, _grid_field(nt=3, with_depth=False))

    with pytest.raises(ValueError, match="no single default"):
        _make("iceland_his").plot()


# -- the surface default, read-free -------------------------------------------------------


def test_a_bare_vertical_select_is_surfaced_before_reading(monkeypatch):
    meta = _grid_meta()
    _resolve(monkeypatch, "iceland_his", meta)
    capture: dict = {}
    _stub_prepare_source(monkeypatch, _grid_field(nt=1), capture=capture)

    _make("iceland_his", select={"time": "2024-02-01"}).plot()
    assert capture["select"] == {"time": "2024-02-01", "depth": "surface"}


def test_an_explicit_vertical_select_is_left_alone(monkeypatch):
    meta = _grid_meta()
    _resolve(monkeypatch, "iceland_his", meta)
    capture: dict = {}
    _stub_prepare_source(
        monkeypatch, _grid_field(nt=1, with_depth=False), capture=capture
    )

    _make(
        "iceland_his", select={"time": "2024-02-01", "depth": 50}
    ).plot()
    assert capture["select"] == {"time": "2024-02-01", "depth": 50}


def test_a_point_select_is_never_surfaced_read_free(monkeypatch):
    """A select naming lon/lat draws as a line/profile/time_depth instead -- the
    catalog's own vertical metadata says nothing about what *that* shape needs,
    so the read-free surface default must not fire for it.
    """
    meta = _grid_meta()
    _resolve(monkeypatch, "iceland_his", meta)
    capture: dict = {}
    time = pd.date_range("2024-01-01", periods=4, freq="MS")
    da = xr.DataArray(
        np.arange(4.0),
        dims="time",
        coords={"time": time},
        name="temperature",
        attrs={"units": "degC"},
    ).assign_coords(lon=-20.0, lat=64.0)
    _stub_prepare_source(monkeypatch, da, capture=capture)

    f = _make("iceland_his", select={"lon": -20.0, "lat": 64.0})
    assert f.family == "series"
    f.plot()
    assert "depth" not in capture["select"]


# -- the surface default's fallback, post-load ---------------------------------------------


def test_a_bare_labelled_depth_axis_reduces_to_the_top_level(monkeypatch):
    """Uncatalogued source -- the fallback, not the read-free path -- picks the
    level nearest 0 m and labels it "surface".
    """
    _stub_prepare_source(monkeypatch, _grid_field(nt=1, with_depth=True))

    fig = _make("stub").plot()
    assert fig is not None  # one map: depth reduced away, only lat/lon left


def test_a_bare_labelled_depth_axis_reports_surface_in_the_title(monkeypatch):

    _stub_prepare_source(monkeypatch, _grid_field(nt=1, with_depth=True))
    fig = _make("stub", label="run A").plot()
    assert "surface" in (fig._suptitle.get_text() if fig._suptitle else "")


def test_a_bare_roms_facet_reduces_its_native_axis_first(monkeypatch):
    """A ROMS-shaped bare facet (time=1, s_rho, eta, xi) with a 2-D z_rho reduces
    to the top level rather than being refused as labelless -- z_rho is exactly
    the coordinate :func:`ocean_skill.field._top_level` reads for this case.
    """
    from ocean_skill.field import field as make_field

    field = _roms_point_facet(nt=1)
    _stub_prepare_source(monkeypatch, field)

    f = make_field("stub", "sea_water_potential_temperature")
    reduced, depth_label = f._facet_field_and_depth()
    assert "s_rho" not in reduced.dims
    assert depth_label == "surface"
    fig = f.plot()
    assert fig is not None


def test_a_bare_roms_facet_movie_is_not_refused(monkeypatch):
    field = _roms_point_facet(nt=3)
    _stub_prepare_source(monkeypatch, field)

    _make("stub").movie(save=None)


def test_a_bare_grid_with_no_vertical_axis_is_untouched(monkeypatch):
    """Nothing to surface -- the ordinary field_facet path runs exactly as
    before this feature existed.
    """
    from ocean_skill.field import field as make_field

    _stub_prepare_source(monkeypatch, _grid_field(nt=1, with_depth=False))

    fig = make_field("stub", NITRATE).plot()
    assert fig is not None
