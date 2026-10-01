"""Tests for chained aggregate steps as seen by the spec-reading helpers.

One axis of an ``aggregate=`` may be a *chain* -- ``{"time": [{"groupby": "month",
"reduce": "mean"}, "var"]}`` -- whose last step decides whether the axis ends up
collapsed, folded into a climatology, or resampled. These cover the read-free
helpers in :mod:`ocean_skill.comparison` and :mod:`ocean_skill.field` that
type-check an aggregate value (they used to misread a list), and the title text a
chain earns (:func:`ocean_skill.comparison.describe_aggregate`). Mirrors
``test_select_aggregate_pair_spec.py`` / ``test_field_map_grid.py`` in style.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from ocean_skill.comparison import (
    Comparison,
    _collapses_time,
    _collapses_vertical,
    _display_time_title,
    _time_collapsed,
    _time_is_climatology,
    describe_aggregate,
)

NITRATE = "mole_concentration_of_nitrate_in_sea_water"

MONTHLY_MEAN = {"groupby": "month", "reduce": "mean"}
ANNUAL_MEAN = {"resample": "1YS", "reduce": "mean"}
ANNUAL_MAX = {"resample": "1YS", "reduce": "max"}

# -- the flag helpers read the chain's LAST step ---------------------------------

#: (aggregate value, collapses, climatology) for the time axis.
TIME_CASES = [
    ("mean", True, False),
    ({"reduce": "mean"}, True, False),
    ([MONTHLY_MEAN, "var"], True, False),
    ([ANNUAL_MEAN, "std"], True, False),
    ([ANNUAL_MAX, "mean"], True, False),
    ([MONTHLY_MEAN, {"reduce": "mean", "spread": "std"}], True, False),
    ([{"resample": "1D", "reduce": "max"}, MONTHLY_MEAN], False, True),
    ([ANNUAL_MAX, {"resample": "1MS", "reduce": "mean"}], False, True),
    (MONTHLY_MEAN, False, True),
    (ANNUAL_MEAN, False, True),
    ([MONTHLY_MEAN], False, True),
]


@pytest.mark.parametrize(("value", "collapses", "climatology"), TIME_CASES)
def test_time_flags_follow_the_last_step(value, collapses, climatology):
    agg = {"time": value}
    assert _collapses_time(agg) is collapses
    assert _time_is_climatology(agg) is climatology
    # a bare time instant is not in play here, so select alone never decides
    assert _time_collapsed({"depth": 5}, agg) is collapses


def test_a_time_range_select_does_not_override_a_chain():
    window = {"time": {"min": "2012-01-01", "max": "2012-12-31"}}
    assert _time_collapsed(window, {"time": [MONTHLY_MEAN, "var"]}) is True
    assert _time_collapsed(window, {"time": [ANNUAL_MAX, MONTHLY_MEAN]}) is False


@pytest.mark.parametrize(
    ("value", "collapses"),
    [
        ("mean", True),
        ([{"groupby": "month", "reduce": "mean"}, "var"], True),
        ([{"resample": "10m", "reduce": "mean"}, MONTHLY_MEAN], False),
        ({"groupby": "month", "reduce": "mean"}, False),
    ],
)
def test_collapses_vertical_follows_the_last_step(value, collapses):
    select = {"depth": {"min": 0, "max": 100}}
    assert _collapses_vertical(select, {"depth": value}) is collapses


def test_an_empty_chain_is_refused_not_misread():
    with pytest.raises(ValueError, match="at least one step"):
        _collapses_time({"time": []})


# -- _map_time_label no longer crashes on a chain --------------------------------


def _scalarless_map():
    return xr.DataArray(
        np.ones((2, 3)),
        dims=("lat", "lon"),
        coords={"lat": [1.0, 2.0], "lon": [1.0, 2.0, 3.0]},
    )


WINDOW = {"time": {"min": "2012-01-01", "max": "2012-12-31"}}


@pytest.mark.parametrize(
    ("aggregate", "select", "expected"),
    [
        # unchanged wording for the plain mean
        ({"time": "mean"}, WINDOW, "mean over 2012-01-01–2012-12-31"),
        ({"time": "mean"}, {}, "time mean"),
        ({"time": {"reduce": "mean"}}, {}, "time mean"),
        # a non-mean single reduction names its statistic in words
        ({"time": "var"}, {}, "time variance"),
        ({"time": {"reduce": "max"}}, WINDOW, "maximum over 2012-01-01–2012-12-31"),
        # chains: the regression -- these used to raise AttributeError
        (
            {"time": [MONTHLY_MEAN, "var"]},
            WINDOW,
            "variance of monthly means over 2012-01-01–2012-12-31",
        ),
        ({"time": [ANNUAL_MAX, "mean"]}, {}, "mean of annual maxima"),
        # a chain ending in a kept axis leaves the panels to say when
        ({"time": [ANNUAL_MAX, MONTHLY_MEAN]}, WINDOW, None),
        ({"time": MONTHLY_MEAN}, WINDOW, None),
        (None, WINDOW, None),
    ],
)
def test_map_time_label_reads_a_chain(aggregate, select, expected):
    from ocean_skill.field import _map_time_label

    assert _map_time_label(_scalarless_map(), select, aggregate) == expected


# -- describe_aggregate: the read-free title grammar -----------------------------

DESCRIBE_CASES = [
    (None, None),
    # unremarkable single steps keep every existing title as it was
    ("mean", None),
    ({"reduce": "mean"}, None),
    (MONTHLY_MEAN, None),
    (ANNUAL_MEAN, None),
    # a single non-mean reduction names itself
    ("var", "variance"),
    ("std", "std"),
    ("max", "maximum"),
    ({"reduce": "min"}, "minimum"),
    ("median", "median"),
    ("sum", "sum"),
    ({"reduce": "quantile", "q": 0.9}, "0.9 quantile"),
    # chains: <final statistic> of <earlier step, as its bins>
    ([MONTHLY_MEAN, "var"], "variance of monthly means"),
    ([{"groupby": "season", "reduce": "mean"}, "std"], "std of seasonal means"),
    ([ANNUAL_MEAN, "std"], "std of annual means"),
    ([ANNUAL_MAX, "mean"], "mean of annual maxima"),
    ([{"resample": "1D", "reduce": "range"}, "mean"], "mean of daily ranges"),
    ([{"resample": "1MS", "reduce": "min"}, "max"], "maximum of monthly minima"),
    ([MONTHLY_MEAN, {"reduce": "mean", "spread": "std"}], "mean of monthly means"),
    ([{"resample": "AS", "reduce": "mean"}, "std"], "std of annual means"),
    ([{"resample": "10D", "reduce": "mean"}, "var"], "variance of 10D means"),
    ([{"groupby": "year", "reduce": "max"}, "mean"], "mean of yearly maxima"),
    # three steps nest leftward
    (
        [{"resample": "1D", "reduce": "max"}, MONTHLY_MEAN, "var"],
        "variance of monthly means of daily maxima",
    ),
    # a chain ending in a kept axis names its own bins
    (
        [{"resample": "1D", "reduce": "max"}, MONTHLY_MEAN],
        "monthly means of daily maxima",
    ),
]


@pytest.mark.parametrize(("value", "expected"), DESCRIBE_CASES)
def test_describe_aggregate(value, expected):
    assert describe_aggregate(value) == expected


# -- titles: the comparison's time part and the Field's suptitle -----------------


def test_display_time_title_leads_with_the_statistic():
    window = {"time": {"min": "2012-01-01", "max": "2012-12-31"}}
    chain = {"time": [MONTHLY_MEAN, "var"]}
    assert (
        _display_time_title(window, chain)
        == "variance of monthly means over 2012-01-01–2012-12-31"
    )
    assert _display_time_title({}, chain) == "variance of monthly means"
    # plain mean / no aggregate: the window alone, exactly as before
    assert _display_time_title(window, {"time": "mean"}) == "2012-01-01–2012-12-31"
    assert _display_time_title(window, None) == "2012-01-01–2012-12-31"
    assert _display_time_title({}, {"time": "mean"}) is None


def test_display_time_title_reads_the_test_lane_of_a_pair_spec():
    agg = {"test": {"time": [MONTHLY_MEAN, "var"]}, "reference": {"time": "mean"}}
    assert _display_time_title({}, agg) == "variance of monthly means"


def test_comparison_repr_names_a_chain_statistic():
    c = Comparison(
        reference="obs",
        test="model",
        variable=NITRATE,
        select={"depth": "surface"},
        aggregate={"time": [MONTHLY_MEAN, "var"]},
    )
    assert "@ variance of monthly means" in repr(c)
    plain = Comparison(
        reference="obs",
        test="model",
        variable=NITRATE,
        select={"depth": "surface"},
        aggregate={"time": "mean"},
    )
    assert " @ surface>" in repr(plain)


def test_a_field_map_title_names_a_chain_in_both_renderers(monkeypatch):
    from ocean_skill import comparison
    from ocean_skill.field import field as make_field

    da = xr.DataArray(
        np.ones((8, 10)),
        dims=("lat", "lon"),
        coords={"lat": np.linspace(20, 30, 8), "lon": np.linspace(-100, -90, 10)},
        name="nitrate",
        attrs={"units": "mmol m-3"},
    )
    monkeypatch.setattr(comparison, "prepare_source", lambda *a, **k: (da, None))
    make = lambda **kw: make_field(  # noqa: E731
        "stub",
        ["nitrate", "silicate"],
        select={"depth": "surface", "time": "2012"},
        aggregate={"time": [MONTHLY_MEAN, "var"]},
        **kw,
    )
    # kept short on purpose: each suptitle part is elided past 40 characters
    # (matplotlib_renderer._MAX_TITLE_PART_CHARS), which a full date window would hit
    expected = "surface · variance of monthly means over 2012"
    assert make().plot()._suptitle.get_text() == expected
    obj = make().plot(renderer="holoviews")
    assert obj.opts.get("plot").kwargs.get("title") == expected
    # an explicit title= still wins
    assert make().plot(title="mine")._suptitle.get_text() == "mine"


def test_comparison_as_item_carries_the_chain_title(monkeypatch):
    """``as_item["time"]`` is the one string both renderers title from."""
    aligned = xr.Dataset(
        {
            "reference": xr.DataArray(
                np.ones((2, 2)), dims=("lat", "lon"), attrs={"units": "1"}
            )
        }
    )
    monkeypatch.setattr(Comparison, "aligned", property(lambda self: aligned))
    monkeypatch.setattr(Comparison, "metrics", lambda self: {})
    monkeypatch.setattr(Comparison, "standard_name", property(lambda self: None))

    def item(aggregate):
        c = Comparison(
            reference="obs",
            test="model",
            variable=NITRATE,
            select={"depth": "surface", "time": "2012"},
            aggregate=aggregate,
        )
        c.over = None
        return c.as_item()["time"]

    chain = item({"time": [MONTHLY_MEAN, "var"]})
    assert chain == "variance of monthly means over 2012"
    assert item({"time": "mean"}) == "2012"
    assert item(None) == "2012"
