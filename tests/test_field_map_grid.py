"""Tests for ``field_map_grid``: several *variables* drawn as maps beside each other.

Mirrors ``tests/test_field_time_depth.py``'s grid section (the reference-free,
multi-member, per-panel-scale precedent this family follows) and
``tests/test_renderers.py``'s ``skill_map`` colorbar tests (the per-quantity
colorbar convention this family borrows) -- but for
:class:`~ocean_skill.field.FieldSet`'s own map composition
(:meth:`ocean_skill.field.Field._map_item`), not a comparison.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

NITRATE = "nitrate"
SILICATE = "silicate"


def _map(name: str, value: float, *, units: str = "mmol m-3"):
    """A single-instant map -- already the one panel a map member draws."""
    lat = np.linspace(20.0, 30.0, 8)
    lon = np.linspace(-100.0, -90.0, 10)
    return xr.DataArray(
        np.full((8, 10), value),
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
        name=name,
        attrs={"units": units},
    )


def _gridded_map(nt: int = 3):
    """An ordinary map with a surviving time facet -- not yet a single map."""
    import pandas as pd

    time = pd.date_range("2012-01-01", periods=nt, freq="MS")
    return xr.DataArray(
        np.random.default_rng(0).normal(5.0, 1.0, (nt, 8, 10)),
        dims=("time", "lat", "lon"),
        coords={
            "time": time,
            "lat": np.linspace(20, 30, 8),
            "lon": np.linspace(-100, -90, 10),
        },
        name=NITRATE,
        attrs={"units": "mmol m-3"},
    )


@pytest.fixture
def stub_by_variable(monkeypatch):
    """Return a setter mapping each variable's own field, keyed by a marker
    substring in the resolved standard_name/variable spec -- unlike
    ``test_field_series.py``'s ``stub`` fixture, which hands every member the
    same field, several of these tests need distinct values/units per member
    to tell "each panel has its own colour scale" apart at all.
    """
    from ocean_skill import comparison

    def use(by_marker: dict[str, xr.DataArray]):
        def fake(source, variable, *a, **k):
            for marker, da in by_marker.items():
                if marker in str(variable):
                    return da, None
            raise AssertionError(f"no stub field for variable={variable!r}")

        monkeypatch.setattr(comparison, "prepare_source", fake)

    return use


def _make_set(variables, **kwargs):
    from ocean_skill.field import field as make_field

    return make_field("stub", variables, **kwargs)


# -- composition: several maps become a grid, one panel per variable --------------------


def test_two_map_variables_draw_as_two_panels_each_its_own_colorbar(stub_by_variable):
    stub_by_variable(
        {"nitrate": _map("nitrate", 5.0), "silicate": _map("silicate", 20.0)}
    )
    fig = _make_set([NITRATE, SILICATE]).plot()
    from matplotlib.collections import QuadMesh

    # panel axes are created (and registered in fig.axes) before any colorbar's
    # own axes, the same ordering tests/test_field_time_depth.py relies on --
    # so the first two are the panels themselves.
    panels = fig.axes[:2]
    assert all(
        any(isinstance(c, QuadMesh) for c in ax.collections) for ax in panels
    )
    # two panels, each with its own colorbar axes (four axes total)
    assert len(fig.axes) == 4

    obj = _make_set([NITRATE, SILICATE]).plot(renderer="holoviews")
    import holoviews as hv

    assert len(obj.traverse(lambda x: x, [hv.QuadMesh])) == 2


def test_a_one_element_list_of_maps_draws_a_single_clean_map(stub_by_variable):
    """A one-variable map list routes through ``field_facet`` itself, the same
    one-element shortcut a line ``FieldSet`` takes -- one map, one colorbar.
    """
    from ocean_skill.field import FieldSet

    stub_by_variable({"nitrate": _map("nitrate", 5.0)})
    fs = _make_set([NITRATE])
    assert isinstance(fs, FieldSet)
    assert len(fs) == 1
    fig = fs.plot()
    assert len(fig.axes) == 2  # one map panel, one colorbar

    obj = fs.plot(renderer="holoviews")
    import holoviews as hv

    assert len(obj.traverse(lambda x: x, [hv.QuadMesh])) == 1


def test_no_two_map_panels_share_a_colour_scale(stub_by_variable):
    stub_by_variable(
        {"nitrate": _map("nitrate", 5.0), "silicate": _map("silicate", 20.0)}
    )
    fig = _make_set([NITRATE, SILICATE]).plot()
    from matplotlib.collections import QuadMesh

    norms = []
    for ax in fig.axes[:2]:  # panel axes only -- see the ordering note above
        mesh = next(c for c in ax.collections if isinstance(c, QuadMesh))
        norms.append((mesh.norm.vmin, mesh.norm.vmax))
    assert len(norms) == 2
    assert norms[0] != norms[1]


def test_each_panel_is_titled_by_its_own_variable(stub_by_variable):
    stub_by_variable(
        {"nitrate": _map("nitrate", 5.0), "silicate": _map("silicate", 20.0)}
    )
    fig = _make_set([NITRATE, SILICATE]).plot()
    titles = {ax.get_title() for ax in fig.axes if ax.get_title()}
    assert "nitrate" in titles
    assert "silicate" in titles


def test_titles_overrides_one_panel_and_keeps_the_other_auto(stub_by_variable):
    stub_by_variable(
        {"nitrate": _map("nitrate", 5.0), "silicate": _map("silicate", 20.0)}
    )
    override = [None, "My Silicate Panel"]
    fig = _make_set([NITRATE, SILICATE]).plot(titles=override)
    titles = {ax.get_title() for ax in fig.axes if ax.get_title()}
    assert titles == {"nitrate", "My Silicate Panel"}

    obj = _make_set([NITRATE, SILICATE]).plot(renderer="holoviews", titles=override)
    import holoviews as hv

    hv_titles = {
        el.opts.get("plot").kwargs.get("title")
        for el in obj.traverse(lambda x: x, [hv.QuadMesh])
    }
    assert hv_titles == {"nitrate", "My Silicate Panel"}


def test_titles_wrong_length_lists_the_current_titles_to_copy(stub_by_variable):
    stub_by_variable(
        {"nitrate": _map("nitrate", 5.0), "silicate": _map("silicate", 20.0)}
    )
    with pytest.raises(ValueError, match="needs one entry per panel"):
        _make_set([NITRATE, SILICATE]).plot(titles=["only one"])


def test_suptitle_carries_only_what_the_set_shares_not_the_variable(stub_by_variable):
    """The variable rides on each panel's own title -- the suptitle carries only
    what every member shares (here, the ``depth`` a common ``select`` narrowed
    to), the same split ``grid_suptitle`` already gives ``time_depth_grid``.
    """
    stub_by_variable(
        {"nitrate": _map("nitrate", 5.0), "silicate": _map("silicate", 20.0)}
    )
    fig = _make_set([NITRATE, SILICATE], select={"depth": "surface"}).plot()
    assert fig._suptitle.get_text() == "surface"


def test_ncols_wraps_the_panels(stub_by_variable):
    stub_by_variable(
        {
            "nitrate": _map("nitrate", 5.0),
            "silicate": _map("silicate", 20.0),
            "oxygen": _map("oxygen", 200.0),
        }
    )
    fig = _make_set([NITRATE, SILICATE, "oxygen"]).plot(ncols=1)
    from matplotlib.collections import QuadMesh

    panels = fig.axes[:3]
    assert all(
        any(isinstance(c, QuadMesh) for c in ax.collections) for ax in panels
    )
    assert fig.axes[0].get_gridspec().ncols == 1


# -- a member still faceted over time refuses -------------------------------------------


def test_a_member_still_faceted_over_time_refuses(stub_by_variable):
    stub_by_variable(
        {"nitrate": _gridded_map(nt=3), "silicate": _map("silicate", 20.0)}
    )
    with pytest.raises(ValueError, match="no single default"):
        _make_set([NITRATE, SILICATE]).plot()


def test_a_member_still_faceted_over_depth_refuses(stub_by_variable):
    depth = np.array([0.0, 50.0, 100.0])
    lat = np.linspace(20, 30, 8)
    lon = np.linspace(-100, -90, 10)
    faceted = xr.DataArray(
        np.random.default_rng(0).normal(5.0, 1.0, (3, 8, 10)),
        dims=("depth", "lat", "lon"),
        coords={"depth": depth, "lat": lat, "lon": lon},
        name="nitrate",
        attrs={"units": "mmol m-3"},
    )
    stub_by_variable({"nitrate": faceted, "silicate": _map("silicate", 20.0)})
    with pytest.raises(ValueError, match="'depth'=3 standing"):
        _make_set([NITRATE, SILICATE], select={"depth": [0.0, 50.0, 100.0]}).plot()


def test_a_size_one_facet_axis_is_not_a_refusal(stub_by_variable):
    """A bare, unsqueezed size-one time axis (a WOA climatology's own shape) is
    not a standing choice -- it draws fine, the same as a solo ``Field.plot()``
    already treats it (see ``NO_AGGREGATION``'s own docstring).
    """
    stub_by_variable(
        {"nitrate": _gridded_map(nt=1), "silicate": _gridded_map(nt=1)}
    )
    fig = _make_set([NITRATE, SILICATE]).plot()
    from matplotlib.collections import QuadMesh

    panels = fig.axes[:2]
    assert all(
        any(isinstance(c, QuadMesh) for c in ax.collections) for ax in panels
    )


# -- movie() refuses, map-aware ----------------------------------------------------------


def test_movie_refuses_a_map_set_with_a_map_aware_message(stub_by_variable):
    stub_by_variable(
        {"nitrate": _map("nitrate", 5.0), "silicate": _map("silicate", 20.0)}
    )
    with pytest.raises(ValueError, match="nothing to play"):
        _make_set([NITRATE, SILICATE]).movie()


# -- save() -------------------------------------------------------------------------------


def test_fieldset_save_writes_one_figure_for_a_map_set(tmp_path, stub_by_variable):
    from ocean_skill import outputs

    outputs.set_base(tmp_path)
    try:
        stub_by_variable(
            {"nitrate": _map("nitrate", 5.0), "silicate": _map("silicate", 20.0)}
        )
        paths = _make_set([NITRATE, SILICATE]).save("proj")
        assert paths["figure"].exists()
    finally:
        outputs.set_base(None)


# -- spec registration ---------------------------------------------------------------------


def test_field_map_grid_is_a_registered_family():
    from ocean_skill.plot.spec import FAMILIES

    assert "field_map_grid" in FAMILIES
