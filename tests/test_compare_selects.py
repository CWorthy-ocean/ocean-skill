"""``compare(selects={label: select})``: one comparison per labelled select.

The UCLA report stacks the three transects of one variable on one figure (Figs. 8, 18,
19). ``select=`` keeps meaning the base every entry shares; each labelled entry is laid
over it, and each member's label *is* the entry's label, so a section plot's rows
read as the transects. Like ``test_aggregate_fan.py`` these tests stub the catalog and
``Comparison.align`` and read off what each fanned member was built with; one test draws
a real stacked section figure.
"""

from __future__ import annotations

from unittest import mock

import pandas as pd
import pytest

from ocean_skill import comparison
from ocean_skill.comparison import Comparison, ComparisonSet
from ocean_skill.config import SuiteConfig
from ocean_skill.suites import pages as P

DECLARED = {"model": {"variables": []}, "obs": {"variables": []}}

EQUATOR = {"transect": {"lat": 0, "lon": {"min": 140, "max": 280}}}
W180 = {"transect": {"lon": 190, "lat": {"min": -30, "max": 30}}}
W160 = {"transect": {"lon": 220, "lat": {"min": -30, "max": 30}}}
SELECTS = {"Equator": EQUATOR, "180-160W": W180, "160-120W": W160}
MEAN = {"time": "mean"}
LEVELS = {"depth": [0, 50, 100]}


def _compare(**kwargs):
    kwargs = {
        "reference": "obs",
        "test": "model",
        "variables": ["temperature"],
        "aggregate": MEAN,
        "select": LEVELS,  # a transect needs its levels named
        **kwargs,
    }
    with (
        mock.patch(
            "ocean_skill.catalog.resolve", lambda n: mock.Mock(metadata=DECLARED[n])
        ),
        mock.patch.object(Comparison, "align", lambda self, refresh=False: None),
    ):
        return comparison.compare(**kwargs)


# -- the fan -----------------------------------------------------------------------


def test_one_member_per_labelled_select_in_the_order_given():
    out = _compare(selects=SELECTS)
    assert isinstance(out, ComparisonSet)
    assert [c.label for c in out] == ["Equator", "180-160W", "160-120W"]
    assert [c.select["transect"] for c in out] == [
        EQUATOR["transect"],
        W180["transect"],
        W160["transect"],
    ]


def test_select_is_the_shared_base_and_the_labelled_entry_wins():
    out = _compare(
        select={"time": {"min": "2011-01", "max": "2011-04"}, "depth": [0, 50]},
        selects={"A": {"transect": EQUATOR["transect"]}, "B": {"depth": [0, 100]}},
    )
    a, b = out
    assert a.select["time"] == {"min": "2011-01", "max": "2011-04"}
    assert a.select["depth"] == [0, 50] and "transect" in a.select
    assert b.select["depth"] == [0, 100]  # the entry's depth beats the base's
    assert b.select["time"] == a.select["time"] and "transect" not in b.select


def test_a_pair_spec_base_takes_a_plain_entry_on_both_lanes():
    """What a suite page's time window makes of ``select``: a pair-spec."""
    base = {"test": {"time": {"min": "2011-01", "max": "2011-04"}}, "reference": {}}
    box = {"lat": {"min": -5, "max": 5}}
    (c,) = _compare(select=base, selects={"Tropics": box})
    assert c.select["test"].items() >= {"time": base["test"]["time"], **box}.items()
    assert c.select["reference"].items() >= box.items()
    assert "time" not in c.select["reference"]


def test_a_pair_spec_entry_is_merged_lane_by_lane():
    entry = {"test": {"lat": 5}, "reference": {"depth": 10}}
    (c,) = _compare(select={"time": "2011-01"}, selects={"Eq": entry})
    assert c.select["test"].items() >= {"time": "2011-01", "lat": 5}.items()
    assert c.select["reference"].items() >= {"time": "2011-01", "depth": 10}.items()
    assert "lat" not in c.select["reference"]


def test_a_member_is_what_the_same_call_with_that_select_would_build():
    out = _compare(selects=SELECTS)
    alone = _compare(select={**LEVELS, **W180})
    assert out[1]._cache_key == alone[0]._cache_key
    assert len({c._cache_key for c in out}) == 3


def test_a_single_entry_is_still_a_set_with_its_label():
    out = _compare(selects={"Equator": EQUATOR})
    assert isinstance(out, ComparisonSet) and [c.label for c in out] == ["Equator"]


def test_the_fan_crosses_variables_and_labels_follow_the_member():
    out = _compare(variables=["temperature", "salinity"], selects=SELECTS)
    assert len(out) == 6
    assert [c.label for c in out[:2]] == ["temperature Equator", "salinity Equator"]
    assert len({c.label for c in out}) == 6


def test_the_fan_crosses_the_aggregate_fan():
    out = _compare(selects=SELECTS, aggregate=[MEAN, {"time": "max"}])
    assert len(out) == 6
    assert len({c.label for c in out}) == 6
    assert all(" " in c.label for c in out)


def test_selects_none_is_no_fan():
    (c,) = _compare(selects=None)
    assert c.label == "temperature"


@pytest.mark.parametrize(
    "bad, error",
    [
        ({}, ValueError),
        ([("a", EQUATOR)], ValueError),
        ({"": EQUATOR}, ValueError),
        ({"a": "surface"}, TypeError),
    ],
)
def test_bad_selects_are_refused(bad, error):
    with pytest.raises(error):
        _compare(selects=bad)


def test_the_refusal_comes_before_any_member_is_built():
    built = []
    with (
        mock.patch(
            "ocean_skill.catalog.resolve", lambda n: mock.Mock(metadata=DECLARED[n])
        ),
        mock.patch.object(
            Comparison, "align", lambda self, refresh=False: built.append(self)
        ),
        pytest.raises(TypeError),
    ):
        comparison.compare(
            reference="obs",
            test="model",
            variables=["temperature"],
            aggregate=MEAN,
            select=LEVELS,
            selects={"ok": EQUATOR, "bad": "oops"},
        )
    assert built == []


# -- a suite page -------------------------------------------------------------------

INDEX = pd.date_range("2010-01-05", periods=10, freq="7D")


@pytest.fixture
def _stub_time_index(monkeypatch):
    monkeypatch.setattr("ocean_skill.extrema._native_time_index", lambda source: INDEX)


def _suite(**compare_kwargs):
    # A flat select, as every transect page has: with none, the page window is written
    # as a test/reference pair-spec, which compare() refuses beside a transect.
    return SuiteConfig.model_validate(
        {
            "name": "transects",
            "defaults": {"test": "model"},
            "pages": [
                {
                    "title": "Temperature transects",
                    "compare": {
                        "reference": ["obs"],
                        "variables": ["temperature"],
                        "aggregate": MEAN,
                        "select": LEVELS,
                        **compare_kwargs,
                    },
                }
            ],
        }
    )


def test_a_suite_page_carries_selects_through_expand_unchanged(_stub_time_index):
    (page,) = P.expand(_suite(selects=SELECTS))
    assert page.kwargs["selects"] == SELECTS
    assert page.kwargs["select"] == LEVELS  # the shared base, untouched too


def test_a_suite_page_builds_a_stacked_set(_stub_time_index):
    (page,) = P.expand(_suite(selects=SELECTS))
    plotted = []
    with (
        mock.patch(
            "ocean_skill.catalog.resolve", lambda n: mock.Mock(metadata=DECLARED[n])
        ),
        mock.patch.object(Comparison, "align", lambda self, refresh=False: None),
        mock.patch.object(Comparison, "metrics", lambda self: {"label": self.label}),
        mock.patch.object(
            Comparison,
            "family",
            new_callable=mock.PropertyMock,
            return_value="section_row",
        ),
        mock.patch.object(
            ComparisonSet, "plot", lambda self, **kw: plotted.append(self) or "fig"
        ),
    ):
        ((_, fig),) = P.build(page)
    assert fig == "fig"
    (drawn,) = plotted
    assert [c.label for c in drawn] == ["Equator", "180-160W", "160-120W"]
    assert all(c.select["depth"] == [0, 50, 100] for c in drawn)


# -- end to end: two real sections stacked in one figure ----------------------------


@pytest.fixture
def _model_and_climatology(monkeypatch):
    from types import SimpleNamespace

    import ocean_skill as osk
    from ocean_skill import catalog
    from tests.test_section_comparison import HC, _climatology, _roms_run

    sources = {
        "clim": (_climatology(), {"featureType": "grid"}),
        "roms_test": (
            _roms_run(),
            {"model": "roms", "vertical": {"s_dim": "s_rho", "hc": HC}},
        ),
    }
    monkeypatch.setattr(osk, "read", lambda name, **kw: sources[name][0])
    monkeypatch.setattr(
        catalog, "resolve", lambda name: SimpleNamespace(metadata=sources[name][1])
    )


def _two_sections():
    import ocean_skill as osk
    from tests.test_section_comparison import VAR

    return osk.compare(
        reference="clim",
        test="roms_test",
        variables=[VAR],
        select={"depth": [50.0, 200.0]},
        selects={
            "Line A": {"transect": {"lat": 25, "lon": {"min": -95, "max": -93}}},
            "Line B": {"transect": {"lon": -94, "lat": {"min": 24.5, "max": 27.5}}},
        },
    )


@pytest.mark.usefixtures("_model_and_climatology")
def test_two_transects_stack_as_rows_titled_by_their_labels():
    cs = _two_sections()
    assert [c.label for c in cs] == ["Line A", "Line B"]
    assert {c.family for c in cs} == {"section_row"}
    fig = cs.plot()
    # one figure; each row carries its label and its own x axis
    # (longitude along the zonal line, latitude along the meridional one)
    from matplotlib.text import Text

    words = {" ".join(t.get_text().split()) for t in fig.findobj(Text)}  # rows wrap
    assert {"Line A", "Line B"} <= words
    xlabels = {ax.get_xlabel() for ax in fig.axes if ax.get_xlabel()}
    assert {"longitude", "latitude"} <= xlabels


@pytest.mark.usefixtures("_model_and_climatology")
def test_the_stack_renders_interactively():
    import holoviews as hv

    cs = _two_sections()
    hv.render(cs.plot(renderer="holoviews"), backend="bokeh")


# -- per-entry aggregates ------------------------------------------------------------

SLAB = {"lon": {"min": 180, "max": 200}, "lat": {"min": -30, "max": 30}}
SLAB_MEAN = {"time": "mean", "lon": "mean"}


def test_an_entry_can_carry_its_own_aggregate():
    out = _compare(
        aggregate=MEAN,
        selects={
            "Equator": EQUATOR,  # a plain select: the call's aggregate
            "Slab": {"select": SLAB, "aggregate": SLAB_MEAN},
        },
    )
    eq, slab = out
    assert eq.aggregate == MEAN and slab.aggregate == SLAB_MEAN
    assert slab.select["lon"] == SLAB["lon"] and "transect" not in slab.select
    assert [c.label for c in out] == ["Equator", "Slab"]


def test_an_entry_aggregate_replaces_the_calls_whole_it_is_not_merged():
    (c,) = _compare(
        aggregate={"time": "mean", "lat": "mean"},
        selects={"S": {"select": SLAB, "aggregate": {"lon": "mean"}}},
    )
    assert c.aggregate == {"lon": "mean"}


def test_the_aggregate_key_is_optional_and_a_missing_one_is_the_calls():
    (c,) = _compare(aggregate=MEAN, selects={"E": {"select": EQUATOR}})
    assert c.aggregate == MEAN and "transect" in c.select


def test_an_entry_aggregate_may_be_a_pair_spec_or_a_list():
    pair = {"test": SLAB_MEAN, "reference": {"time": "mean", "lon": "mean"}}
    (c,) = _compare(selects={"S": {"select": SLAB, "aggregate": pair}})
    assert c.aggregate == pair
    both = _compare(
        selects={"S": {"select": EQUATOR, "aggregate": [MEAN, {"time": "max"}]}}
    )
    assert [c.aggregate for c in both] == [MEAN, {"time": "max"}]


def test_an_entry_key_beside_select_and_aggregate_is_read_as_a_select():
    """Only exactly ``select``/``aggregate`` keys make the wrapped form."""
    (c,) = _compare(selects={"S": {"select": SLAB, "lat": 5}})  # a select of 2 axes
    assert c.select["lat"] == 5 and c.select["select"] == SLAB


def test_an_entry_member_is_what_the_same_call_would_build():
    out = _compare(
        selects={"Slab": {"select": SLAB, "aggregate": SLAB_MEAN}, "Eq": EQUATOR}
    )
    alone = _compare(select={**LEVELS, **SLAB}, aggregate=SLAB_MEAN)
    assert out[0]._cache_key == alone[0]._cache_key
    assert out[0]._cache_key != out[1]._cache_key


# -- selects= is the outermost fan: aggregate lists fan within each entry ----------

MAX = {"time": "max"}


def test_a_list_aggregate_fans_within_each_plain_entry():
    out = _compare(aggregate=[MEAN, MAX], selects={"Eq": EQUATOR, "W180": W180})
    # entry by entry, then the call's aggregate fan inside each
    assert [c.aggregate for c in out] == [MEAN, MAX, MEAN, MAX]
    assert [c.select["transect"] for c in out] == [EQUATOR["transect"]] * 2 + [
        W180["transect"]
    ] * 2
    assert len({c.label for c in out}) == 4
    assert all(c.label.endswith(("Eq", "W180")) for c in out)
    # a member is what the plain call with that merged select and one aggregate builds
    alone = _compare(select={**LEVELS, **W180}, aggregate=MAX)
    assert out[3]._cache_key == alone[0]._cache_key


def test_a_list_aggregate_does_not_rerun_an_entry_with_its_own_aggregate():
    out = _compare(
        aggregate=[MEAN, MAX],
        selects={"Eq": EQUATOR, "Slab": {"select": SLAB, "aggregate": SLAB_MEAN}},
    )
    # Eq fans over the call's list; Slab runs once, with its own aggregate only
    assert [c.aggregate for c in out] == [MEAN, MAX, SLAB_MEAN]
    assert [c.label for c in out].count("Slab") == 1
    assert len({c._cache_key for c in out}) == 3
    assert len({c.label for c in out}) == 3


def test_an_entry_whose_own_aggregate_is_a_list_fans_over_it():
    out = _compare(
        aggregate=MEAN,
        selects={
            "Eq": EQUATOR,
            "Slab": {"select": SLAB, "aggregate": [SLAB_MEAN, {"lon": "mean"}]},
        },
    )
    assert [c.aggregate for c in out] == [MEAN, SLAB_MEAN, {"lon": "mean"}]
    assert out[0].label == "Eq"  # one comparison: exactly the entry's label
    assert all(c.label.endswith("Slab") and c.label != "Slab" for c in out[1:])
    assert len({c.label for c in out}) == 3


def test_a_bad_aggregate_list_is_refused_before_any_member_is_built():
    built = []
    with (
        mock.patch(
            "ocean_skill.catalog.resolve", lambda n: mock.Mock(metadata=DECLARED[n])
        ),
        mock.patch.object(
            Comparison, "align", lambda self, refresh=False: built.append(self)
        ),
        pytest.raises(ValueError),
    ):
        comparison.compare(
            reference="obs",
            test="model",
            variables=["temperature"],
            aggregate=MEAN,
            select=LEVELS,
            selects={"ok": EQUATOR, "bad": {"select": SLAB, "aggregate": []}},
        )
    assert built == []


# -- the final-report case, end to end ------------------------------------------------

SALINITY = "sea_water_practical_salinity"


@pytest.fixture
def _two_variable_sources(monkeypatch):
    """Patch in a 3-step ROMS-like model and a gridded climatology (T and S)."""
    from types import SimpleNamespace

    import numpy as np
    import xarray as xr

    import ocean_skill as osk
    from ocean_skill import catalog
    from tests.test_section_comparison import (
        HC,
        VAR,
        _climatology,
        _roms_run_with_time,
    )

    model = _roms_run_with_time(3)
    salt = 35.0 + 0.001 * model["z_rho"] + 0.1 * model["time"].dt.dayofyear * 0
    model = model.assign({SALINITY: salt.transpose(*model[VAR].dims)})
    model[SALINITY].attrs["units"] = "1"
    clim = _climatology()
    clim = clim.assign(
        {SALINITY: (clim[VAR] * 0 + 34.5 - 0.0005 * clim["depth"]).astype(float)}
    )
    clim[SALINITY].attrs["units"] = "1"
    assert isinstance(clim, xr.Dataset) and np.isfinite(clim[SALINITY]).all()
    sources = {
        "woa": (clim, {"featureType": "grid"}),
        "roms_test": (
            model,
            {"model": "roms", "vertical": {"s_dim": "s_rho", "hc": HC}},
        ),
    }
    monkeypatch.setattr(osk, "read", lambda name, **kw: sources[name][0])
    monkeypatch.setattr(
        catalog, "resolve", lambda name: SimpleNamespace(metadata=sources[name][1])
    )


FINAL_SELECT = {"depth": [50.0, 200.0]}
FINAL_SELECTS = {
    "Equator": {"transect": {"lat": 26, "lon": {"min": -95, "max": -93}}},
    "West slab": {
        "select": {
            "lon": {"min": -95.5, "max": -94.0},
            "lat": {"min": 23.5, "max": 28.5},
        },
        "aggregate": {"time": "mean", "lon": "mean"},
    },
    "East slab": {
        "select": {
            "lon": {"min": -94.0, "max": -92.5},
            "lat": {"min": 23.5, "max": 28.5},
        },
        "aggregate": {"time": "mean", "lon": "mean"},
    },
}


def _final_report(variables, **extra):
    import ocean_skill as osk

    return osk.compare(
        reference="woa",
        test="roms_test",
        variables=variables,
        select=FINAL_SELECT,
        aggregate={"time": "mean"},
        selects=FINAL_SELECTS,
        cache=False,
        **extra,
    )


@pytest.mark.usefixtures("_two_variable_sources")
def test_two_variables_over_three_transects_stack_as_six_rows(tmp_path):
    import holoviews as hv
    from matplotlib.text import Text

    cs = _final_report(["temperature", "salinity"])
    assert len(cs) == 6 and {c.family for c in cs} == {"section_row"}
    # entry by entry, then by variable
    assert [c.label for c in cs] == [
        f"{v} {e}"
        for e in ("Equator", "West slab", "East slab")
        for v in ("temperature", "salinity")
    ]
    fig = cs.plot()
    fig.savefig(tmp_path / "stack.png", dpi=70)
    words = {" ".join(t.get_text().split()) for t in fig.findobj(Text)}  # rows wrap
    assert {"temperature Equator", "salinity East slab"} <= words
    hv.render(cs.plot(renderer="holoviews"), backend="bokeh")


@pytest.mark.usefixtures("_two_variable_sources")
def test_contours_from_a_second_compare_pair_with_the_fill_by_position(tmp_path):
    """What a page's ``contours:`` override builds: the same call, other variables.

    ``pages._contour_overrides`` shallow-merges the override over the page's own
    kwargs, so ``selects`` rides along unchanged and the overlay set comes out in the
    same entry-by-entry order -- row ``i`` of the lines is row ``i`` of the fill.
    """
    import holoviews as hv

    fill = _final_report(["temperature", "salinity"])
    lines = _final_report(["salinity", "temperature"])
    assert len(lines) == len(fill) == 6
    # the same transect on the same row, whatever the variable
    for f, c in zip(fill, lines, strict=True):
        assert f.label.split(" ", 1)[1] == c.label.split(" ", 1)[1]
        assert f.select == c.select and f.aggregate == c.aggregate
    fig = fill.plot(contours=lines)
    fig.savefig(tmp_path / "stack_contours.png", dpi=70)
    hv.render(fill.plot(contours=lines, renderer="holoviews"), backend="bokeh")


@pytest.mark.usefixtures("_two_variable_sources")
def test_contours_with_a_different_member_count_are_refused_by_their_length():
    fill = _final_report(["temperature", "salinity"])
    lines = _final_report(["salinity"])  # 3 rows of lines for 6 of fill
    with pytest.raises(ValueError, match="same length"):
        fill.plot(contours=lines)


@pytest.mark.usefixtures("_two_variable_sources")
def test_a_roms_slab_narrower_than_the_grid_cuts_its_bathymetry_to_the_box():
    """The bathymetry rejoined to a slab is cut to the field's box (it was not)."""
    import ocean_skill as osk
    from tests.test_section_comparison import VAR

    f = osk.field(
        "roms_test",
        VAR,
        select={
            "lon": {"min": -95.5, "max": -94.0},
            "lat": {"min": 23.5, "max": 28.5},
            "depth": [50.0, 200.0],
        },
        aggregate={"lon": "mean", "time": "mean"},
        cache=False,
    )
    assert f.family == "section" and f.data.sizes["z"] == 2
