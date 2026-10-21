"""``plot(family="time_depth")``: the colour layout asked for outright.

A field at one place with both time and depth draws as a ``time_depth`` panel by
default -- *except* when the depths were requested as an explicit list
(``select={"depth": [5, 10, 25]}``), which keeps one line per level (see
``Field.is_time_depth``). ``family="time_depth"`` lifts that exception for a field
whose shape admits it (a ROMS boundary slice averaged along the boundary at fixed
depths is the motivating case); ``family="series"`` insists on the lines. Mirrors
``tests/test_field_time_depth.py``'s stub pattern (``comparison.prepare_source``
swapped out).
"""

from __future__ import annotations

import matplotlib.collections as mcoll
import numpy as np
import pandas as pd
import pytest
import xarray as xr

NITRATE = "nitrate"
DEPTHS = [5.0, 10.0, 25.0, 50.0, 100.0]
LISTED = {"depth": DEPTHS}


def _point_time_depth(n: int = 6, depths=tuple(DEPTHS), seed: int = 1):
    time = pd.date_range("2015-01-01", periods=n, freq="MS")
    depth = np.array(depths)
    values = 8.0 + np.random.default_rng(seed).normal(0, 1, (n, depth.size))
    da = xr.DataArray(
        values,
        dims=("time", "depth"),
        coords={"time": time, "depth": depth},
        name=NITRATE,
        attrs={"units": "mmol m-3"},
    )
    return da.assign_coords(lon=-144.245, lat=49.978)


def _gridded_map(nt: int = 1):
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
def stub(monkeypatch):
    from ocean_skill import comparison

    def use(field_da):
        monkeypatch.setattr(
            comparison, "prepare_source", lambda *a, **k: (field_da, None)
        )

    return use


def _make(**kwargs):
    from ocean_skill.field import field as make_field

    return make_field("stub", NITRATE, **kwargs)


def _has_colour_panel(ax) -> bool:
    return any(isinstance(c, mcoll.QuadMesh) for c in ax.collections) or bool(
        [a for a in ax.get_children() if type(a).__name__ == "QuadContourSet"]
    )


# -- (a) the default is unchanged ----------------------------------------------------


def test_an_explicit_depth_list_still_draws_one_line_per_level(stub):
    stub(_point_time_depth())
    f = _make(select=LISTED)
    assert f.family == "series"
    ax = f.plot().axes[0]
    assert len(ax.lines) == len(DEPTHS)
    assert not _has_colour_panel(ax)


def test_a_bare_time_depth_point_is_untouched(stub):
    stub(_point_time_depth())
    f = _make()
    assert f.family == "time_depth"
    assert _has_colour_panel(f.plot().axes[0])


# -- (b) the explicit request --------------------------------------------------------


def test_family_time_depth_draws_colour_from_a_depth_list(stub):
    stub(_point_time_depth())
    f = _make(select=LISTED)
    fig = f.plot(family="time_depth")
    ax = fig.axes[0]
    assert _has_colour_panel(ax)
    assert len(ax.lines) == 0
    # the request is a plot argument: the field itself keeps its own shape
    assert f.family == "series"


def test_the_requested_copy_reports_its_family_and_why(stub):
    stub(_point_time_depth())
    f = _make(select=LISTED)._with_family("time_depth")
    assert f.family == "time_depth" and f.is_time_depth
    assert "family='time_depth'" in f.family_reason


def test_family_series_forces_lines_on_a_bare_time_depth_point(stub):
    stub(_point_time_depth())
    ax = _make().plot(family="series").axes[0]
    assert len(ax.lines) == len(DEPTHS)
    assert not _has_colour_panel(ax)


def test_family_time_depth_on_a_fieldset_draws_one_panel_per_member(monkeypatch):
    from ocean_skill import comparison
    from ocean_skill.field import FieldSet
    from ocean_skill.field import field as make_field

    data = {f"src_{i}": _point_time_depth(seed=i) for i in range(3)}
    monkeypatch.setattr(
        comparison, "prepare_source", lambda source, *a, **k: (data[source], None)
    )
    fs = make_field(list(data), NITRATE, select=LISTED)
    assert isinstance(fs, FieldSet)
    assert {f.family for f in fs} == {"series"}  # the default is still lines

    fig = fs.plot(family="time_depth")
    panels = [
        ax
        for ax in fig.axes
        if ax.get_label() != "<colorbar>" and _has_colour_panel(ax)
    ]
    assert len(panels) == 3
    assert all(len(ax.lines) == 0 for ax in panels)
    assert {f.family for f in fs} == {"series"}  # the set itself is unchanged


# -- (c) holoviews -------------------------------------------------------------------


def test_family_time_depth_renders_in_holoviews(stub):
    import holoviews as hv

    stub(_point_time_depth())
    obj = _make(select=LISTED).plot(family="time_depth", renderer="holoviews")
    assert obj.traverse(lambda x: x, [hv.QuadMesh, hv.Image, hv.HeatMap])
    assert not obj.traverse(lambda x: x, [hv.Curve])
    hv.render(obj)


def test_family_time_depth_on_a_fieldset_renders_in_holoviews(monkeypatch):
    import holoviews as hv

    from ocean_skill import comparison
    from ocean_skill.field import field as make_field

    data = {f"src_{i}": _point_time_depth(seed=i) for i in range(2)}
    monkeypatch.setattr(
        comparison, "prepare_source", lambda source, *a, **k: (data[source], None)
    )
    obj = make_field(list(data), NITRATE, select=LISTED).plot(
        family="time_depth", renderer="holoviews"
    )
    assert len(obj.traverse(lambda x: x, [hv.QuadMesh, hv.Image, hv.HeatMap])) == 2
    hv.render(obj)


# -- (d) refusals --------------------------------------------------------------------


def test_family_time_depth_on_a_map_is_refused_naming_the_shape(stub):
    stub(_gridded_map())
    with pytest.raises(
        ValueError, match=r"family='time_depth'.*draws as 'field_facet'"
    ):
        _make().plot(family="time_depth")


def test_family_time_depth_on_a_series_without_depth_is_refused(stub):
    da = _point_time_depth().isel(depth=0, drop=True)
    stub(da)
    with pytest.raises(ValueError, match=r"draws as 'series'"):
        _make().plot(family="time_depth")


def test_a_fieldset_with_one_map_member_is_refused(monkeypatch):
    from ocean_skill import comparison
    from ocean_skill.field import field as make_field

    data = {"a": _point_time_depth(), "b": _gridded_map()}
    monkeypatch.setattr(
        comparison, "prepare_source", lambda source, *a, **k: (data[source], None)
    )
    with pytest.raises(ValueError, match=r"'b' draws as 'field_facet'"):
        make_field(["a", "b"], NITRATE).plot(family="time_depth")


def test_family_series_needs_a_time_axis(stub):
    stub(_gridded_map())
    with pytest.raises(ValueError, match=r"family='series'.*draws as 'field_facet'"):
        _make().plot(family="series")


def test_an_unknown_family_is_refused_listing_the_two_it_takes(stub):
    stub(_point_time_depth())
    with pytest.raises(ValueError, match=r"'time_depth'.*'series'"):
        _make().plot(family="section")


# -- (e) a suite field: page ---------------------------------------------------------


INDEX = pd.date_range("2010-01-05", periods=10, freq="7D")


def _suite(pages, **top):
    from ocean_skill.config import SuiteConfig

    return SuiteConfig.model_validate({"name": "t", "pages": pages, **top})


def _page(**plot):
    return {
        "title": "bry",
        "field": {"variables": ["temperature"], "select": LISTED},
        "plot": plot,
    }


def test_a_suite_field_page_passes_family_through_to_plot(monkeypatch):
    import ocean_skill as osk
    from ocean_skill.suites import pages as P

    monkeypatch.setattr("ocean_skill.extrema._native_time_index", lambda s: INDEX)
    calls = []

    class FakeObj:
        def plot(self, **opts):
            calls.append(opts)
            return "figure"

    monkeypatch.setattr(osk, "field", lambda *a, **k: FakeObj())
    (p,) = P.expand(_suite([_page(family="time_depth")], defaults={"test": "stub"}))
    assert p.plot["family"] == "time_depth"
    assert P.build(p) == [("", "figure")]
    assert [c["family"] for c in calls] == ["time_depth"]


def test_a_suite_field_page_draws_colour_end_to_end(monkeypatch, stub):
    from ocean_skill.suites import pages as P

    monkeypatch.setattr("ocean_skill.extrema._native_time_index", lambda s: INDEX)
    stub(_point_time_depth())
    (p,) = P.expand(_suite([_page(family="time_depth")], defaults={"test": "stub"}))
    ((_, fig),) = P.build(p)
    assert _has_colour_panel(fig.axes[0])
