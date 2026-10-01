"""Tests for the property-property plot (:class:`ocean_skill.xy.XY` and :class:`TS`).

Mirrors ``tests/test_field_series.py``'s stub pattern: ``comparison.prepare_source`` is
swapped for a table keyed on ``(source, short variable name)`` that also records every
select/aggregate it is asked for, and the availability probe reads the same table, so
these exercise ``XY``'s own logic (pairing, region rebuilds, the arrays it hands a
renderer) rather than a catalog. The renderer is captured, not run: drawing is
``tests/test_xy_renderers.py``'s.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import ocean_skill as osk
from ocean_skill import align, comparison
from ocean_skill.field import Field, FieldSet
from ocean_skill.plot import registry
from ocean_skill.vars import short_name
from ocean_skill.xy import TS, XY, normalize_regions

BOX = {"lon": {"min": 155.24, "max": 156.33}, "lat": {"min": 20.51, "max": 21.60}}
PAIR = ["temperature", "salinity"]


class Table:
    """A stand-in for the catalog: ``{(source, short name): DataArray | callable}``."""

    def __init__(self, monkeypatch):
        self.data: dict[tuple[str, str], object] = {}
        self.calls: list[tuple[str, str, dict, dict | None]] = []
        self.probes: list[tuple[str, str]] = []
        monkeypatch.setattr(comparison, "prepare_source", self._prepare)
        monkeypatch.setattr(comparison, "_variable_available", self._available)

    def put(self, source, variable, value):
        self.data[(source, variable)] = value

    def _prepare(self, source, variable, select, aggregate, **kwargs):
        name = short_name(variable)
        self.calls.append((source, name, dict(select), aggregate))
        value = self.data.get((source, name))
        if callable(value):
            value = value(select, aggregate)
        return value, None

    def _available(self, source, variable, **kwargs):
        self.probes.append((source, short_name(variable)))
        return (source, short_name(variable)) in self.data

    def selects(self, source, variable):
        return [c[2] for c in self.calls if c[:2] == (source, variable)]


@pytest.fixture
def table(monkeypatch):
    return Table(monkeypatch)


@pytest.fixture
def captured(monkeypatch):
    """Capture the ``PlotSpec`` (and renderer name) ``.plot()`` hands to ``render``."""
    got: dict = {}

    def fake(spec, *, renderer="matplotlib", **kwargs):
        got["spec"], got["renderer"] = spec, renderer
        return "figure"

    monkeypatch.setattr(registry, "render", fake)
    return got


# -- synthetic members ---------------------------------------------------------------

DEPTHS = np.array([90.0, 60.0, 30.0, 5.0])  # s_rho, bottom (index 0) to top


def _roms(base: float, units: str, standard_name: str, *, nan_corner: bool = False):
    """Build a ROMS box (time=3, s_rho=4, eta=2, xi=3) with a height ``z_rho``."""
    shape = (3, 4, 2, 3)
    values = base + np.arange(np.prod(shape), dtype="float64").reshape(shape) / 10
    if nan_corner:
        values[:, :, 0, 0] = np.nan
    time = pd.date_range("2012-01-01", periods=3, freq="MS")
    z_rho = xr.DataArray(
        np.broadcast_to(-DEPTHS[None, :, None, None], shape).copy(),
        dims=("time", "s_rho", "eta_rho", "xi_rho"),
    )
    return xr.DataArray(
        values,
        dims=("time", "s_rho", "eta_rho", "xi_rho"),
        coords={
            "time": time,
            "z_rho": z_rho,
            "lon_rho": (
                ("eta_rho", "xi_rho"),
                155.5 + np.arange(6.0).reshape(2, 3) / 10,
            ),
            "lat_rho": (
                ("eta_rho", "xi_rho"),
                20.5 + np.arange(6.0).reshape(2, 3) / 10,
            ),
        },
        attrs={"units": units, "standard_name": standard_name},
    )


def _woa(base: float, units: str, standard_name: str):
    """Build a nearest-cell WOA profile (time=1, depth=5, 1, 1), depths unsorted."""
    depth = np.array([100.0, 0.0, 50.0, 10.0, 500.0])
    values = (base + depth / 100.0).reshape(1, 5, 1, 1)
    return xr.DataArray(
        values,
        dims=("time", "depth", "lat", "lon"),
        coords={
            "time": pd.to_datetime(["2000-01-01"]),
            "depth": depth,
            "lat": [21.5],
            "lon": [155.5],
        },
        attrs={"units": units, "standard_name": standard_name},
    )


def _box_mean(base: float, units: str, standard_name: str):
    """Build a box-mean profile: one position, depth standing (a GLORYS-style line)."""
    depth = np.array([0.0, 20.0, 100.0, 300.0])
    return xr.DataArray(
        base + depth / 50.0,
        dims="depth",
        coords={"depth": depth, "lon": 155.8, "lat": 21.05},
        attrs={"units": units, "standard_name": standard_name},
    )


TEMP_UNITS = ("degC", "sea_water_potential_temperature")
SALT_UNITS = ("1", "sea_water_practical_salinity")


def _put_roms(table, source="roms", **kw):
    table.put(source, "temperature", _roms(10.0, *TEMP_UNITS, **kw))
    table.put(source, "salinity", _roms(34.0, *SALT_UNITS, **kw))


def _put_woa(table):
    table.put("woa_t", "temperature", _woa(8.0, "degC", "sea_water_temperature"))
    table.put("woa_s", "salinity", _woa(34.0, *SALT_UNITS))


def _members(select=BOX, **extra):
    members = {"ROMS": osk.field("roms", PAIR, select=select)}
    members.update(extra)
    return members


# -- Field._replace ------------------------------------------------------------------


def test_replace_carries_every_attribute_and_changes_only_what_is_named():
    f = Field(
        "roms",
        "temperature",
        select={"time": "2012-01"},
        aggregate={"time": "mean"},
        label="mine",
        cache=False,
        qc="off",
        detide=True,
    )
    g = f._replace(select={"depth": "surface"})
    assert g is not f
    assert g.select == {"depth": "surface"}
    for name in ("source", "variable", "aggregate", "label", "cache", "qc", "detide"):
        assert getattr(g, name) == getattr(f, name)
    assert f.select == {"time": "2012-01"}  # the original is untouched


def test_replace_starts_unprepared_and_refuses_unknown_attributes():
    f = Field("roms", "temperature")
    f._data = object()
    assert f._replace(label="x")._data is None
    with pytest.raises(TypeError, match="no attribute"):
        f._replace(colour="red")


def test_surfaced_uses_replace_and_adds_the_depth_key():
    f = Field("roms", "temperature", select={"time": "2012-01"}, label="L")
    g = f._surfaced()
    assert g.select == {"time": "2012-01", "depth": "surface"}
    assert g.label == "L"


# -- normalize_regions ---------------------------------------------------------------


def test_normalize_regions_none_passes_through():
    assert normalize_regions(None) is None


def test_normalize_regions_records_box_and_centre_in_the_order_given():
    out = normalize_regions({"b": BOX, "a": {"lon": 155.79, "lat": 21.05}})
    assert list(out) == ["b", "a"]
    assert out["b"]["bbox"] == (155.24, 20.51, 156.33, 21.60)
    assert out["b"]["centre"] == pytest.approx((155.785, 21.055))
    assert out["a"]["bbox"] is None
    assert out["a"]["centre"] == (155.79, 21.05)


def test_normalize_regions_keeps_a_0_360_box_as_given():
    box = {"lon": {"min": 184.59, "max": 185.73}, "lat": {"min": 47.76, "max": 48.59}}
    centre = normalize_regions({"SG": box})["SG"]["centre"]
    assert centre == pytest.approx((185.16, 48.175))


def test_normalize_regions_centres_a_box_across_the_antimeridian():
    box = {"lon": {"min": 170, "max": -170}, "lat": {"min": 0, "max": 10}}
    assert normalize_regions({"seam": box})["seam"]["centre"] == pytest.approx((180, 5))


def test_normalize_regions_accepts_other_horizontal_spellings():
    box = {"longitude": slice(10, 20), "latitude": {"min": 1, "max": 3}}
    assert normalize_regions({"r": box})["r"]["centre"] == pytest.approx((15, 2))


@pytest.mark.parametrize(
    ("regions", "error", "match"),
    [
        ([BOX], TypeError, "dict of name"),
        ({}, ValueError, "names no region"),
        ({1: BOX}, TypeError, "region names"),
        ({"r": "surface"}, TypeError, "select must be a dict"),
        ({"r": {**BOX, "time": "2012"}}, ValueError, "non-horizontal"),
        ({"r": {"lon": {"min": 1, "max": 2}}}, ValueError, "neither a box nor"),
        ({"r": {"lon": 1.0, "lat": {"min": 1, "max": 2}}}, ValueError, "neither a box"),
        ({"r": {"lon": 1.0, "longitude": 2.0, "lat": 3.0}}, ValueError, "neither"),
        ({"r": {}}, ValueError, "neither a box nor"),
    ],
)
def test_normalize_regions_refuses_bad_input(regions, error, match):
    with pytest.raises(error, match=match):
        normalize_regions(regions)


# -- construction reads nothing ------------------------------------------------------


def test_constructors_read_nothing(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the constructor read something")

    monkeypatch.setattr(comparison, "prepare_source", boom)
    monkeypatch.setattr(comparison, "_variable_available", boom)
    members = {
        "ROMS": osk.field("roms", PAIR),
        "WOA": osk.field(["woa_t", "woa_s"], PAIR),
    }
    regions = {"NWP": BOX}
    assert isinstance(XY(members, x="salinity", y="temperature"), XY)
    ts = TS(members, regions=regions, at_center=["WOA"])
    assert isinstance(ts, XY)
    assert (ts.x, ts.y) == ("salinity", "temperature")
    assert "WOA" in repr(ts)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"members": []}, TypeError, "dict of label"),
        ({"members": {}}, ValueError, "members is empty"),
        ({"members": {"A": "roms"}}, TypeError, "member 'A'"),
        ({"members": {"A": [Field("r", "temperature"), "x"]}}, TypeError, "member 'A'"),
        ({"members": {"A": []}}, ValueError, "holds no fields"),
        ({"members": {1: Field("r", "temperature")}}, TypeError, "labels"),
        ({"x": "temp", "y": "temperature"}, ValueError, "same variable"),
        ({"x": ["salinity"]}, TypeError, "x= must be"),
        ({"at_center": ["B"], "regions": {"r": BOX}}, ValueError, "not among"),
        ({"at_center": ["A"]}, ValueError, "needs regions"),
        ({"at_center": "A", "regions": {"r": BOX}}, TypeError, "list of member"),
        ({"regions": {"r": {"time": 1}}}, ValueError, "non-horizontal"),
    ],
)
def test_constructor_validation(kwargs, error, match):
    base = {
        "members": {"A": osk.field("src", PAIR)},
        "x": "salinity",
        "y": "temperature",
    }
    with pytest.raises(error, match=match):
        XY(**{**base, **kwargs})


def test_members_may_be_a_field_a_fieldset_or_a_list_of_fields():
    fs = osk.field("src", PAIR)
    assert isinstance(fs, FieldSet)
    xy = XY(
        {"set": fs, "list": list(fs.fields), "one": fs.fields[0]},
        x="salinity",
        y="temperature",
    )
    assert [len(v) for v in xy.members.values()] == [2, 2, 1]


# -- pairing -------------------------------------------------------------------------


def test_pairs_match_x_and_y_through_vocabulary_aliases(table):
    _put_roms(table)
    xy = XY(_members(), x="salt", y="temp")
    xf, yf = xy.pairs()["ROMS"]
    assert (short_name(xf.variable), short_name(yf.variable)) == (
        "salinity",
        "temperature",
    )


def test_pairs_work_for_a_generic_pair(table):
    table.put("roms", "nitrate", _box_mean(5.0, "mmol m-3", "n"))
    table.put("roms", "phosphate", _box_mean(0.5, "mmol m-3", "p"))
    xy = XY(
        {"ROMS": osk.field("roms", ["nitrate", "phosphate"], select=BOX)},
        x="phosphate",
        y="nitrate",
    )
    xf, yf = xy.pairs()["ROMS"]
    assert (short_name(xf.variable), short_name(yf.variable)) == (
        "phosphate",
        "nitrate",
    )
    (item,) = xy._items()
    assert (item["x_name"], item["y_name"]) == ("phosphate", "nitrate")
    assert (item["x_role"], item["y_role"]) == (None, None)


def test_pairs_are_memoized(table):
    _put_roms(table)
    xy = XY(_members(), x="salinity", y="temperature")
    assert xy.pairs() is xy.pairs()
    assert len(table.probes) == 2  # once per original field, however often it is read


def test_a_member_missing_one_variable_is_refused_saying_what_it_has(table):
    table.put("roms", "temperature", _roms(10.0, *TEMP_UNITS))
    xy = XY(
        {"ROMS": osk.field("roms", "temperature", select=BOX)},
        x="salinity",
        y="temperature",
    )
    with pytest.raises(ValueError, match=r"member 'ROMS'.*roms temperature.*0 match x"):
        xy.pairs()


def test_an_ambiguous_member_is_refused(table):
    _put_roms(table, "roms")
    _put_roms(table, "roms2")
    xy = XY(
        {"ROMS": osk.field(["roms", "roms2"], PAIR, select=BOX)},
        x="salinity",
        y="temperature",
    )
    with pytest.raises(ValueError, match=r"2 match x, 2 match y"):
        xy.pairs()


def test_cross_product_members_drop_the_unavailable_fields_with_a_warning(table):
    _put_woa(table)
    woa = osk.field(["woa_t", "woa_s"], PAIR, select={"lon": 155.79, "lat": 21.05})
    assert len(woa) == 4
    xy = XY({"WOA": woa}, x="salinity", y="temperature")
    with pytest.warns(UserWarning, match=r"skipping 2 field\(s\) whose source doesn't"):
        xf, yf = xy.pairs()["WOA"]
    assert (xf.source, yf.source) == ("woa_s", "woa_t")


def test_a_member_with_no_place_is_refused_before_anything_is_loaded(table):
    _put_roms(table)
    xy = XY(_members(select=None), x="salinity", y="temperature")  # built, not read
    with pytest.raises(ValueError, match="whole domain"):
        xy._items()
    assert table.calls == []


def test_a_point_select_counts_as_a_place(table):
    _put_woa(table)
    woa = osk.field(["woa_t", "woa_s"], PAIR, select={"lon": 155.79, "lat": 21.05})
    with pytest.warns(UserWarning, match="skipping"):
        (item,) = XY({"WOA": woa}, x="salinity", y="temperature")._items()
    assert item["label"] == "WOA"


# -- the arrays a member hands the renderer ------------------------------------------


def test_roms_box_draws_as_points_with_time_and_depth(table):
    _put_roms(table, nan_corner=True)
    ts = TS(_members())
    (item,) = ts._items()
    assert item["mark"] == "points"
    n = 3 * 4 * (2 * 3 - 1)  # one masked column of the 2 x 3 box
    for key in ("x", "y", "depth", "time"):
        assert len(item[key]) == n
    assert item["x"].dtype == item["y"].dtype == item["depth"].dtype == np.float64
    assert np.isfinite(item["x"]).all() and np.isfinite(item["y"]).all()
    assert np.issubdtype(item["time"].dtype, np.datetime64)
    assert set(np.unique(item["depth"])) == set(DEPTHS)  # z_rho height -> depth down
    assert item["x"].min() > 33.9 and item["y"].min() >= 10.0  # salinity x, temp y
    assert (item["label"], item["region"], item["source"]) == ("ROMS", None, "roms")


def test_a_pair_of_each_value_stays_together(table):
    _put_roms(table)
    (item,) = TS(_members())._items()
    # same cell, level and step in both arrays: y - 10 == x - 34 exactly
    assert np.allclose(item["y"] - 10.0, item["x"] - 34.0)


def test_woa_profile_draws_as_a_line_sorted_by_depth(table):
    _put_woa(table)
    woa = osk.field(["woa_t", "woa_s"], PAIR, select={"lon": 155.79, "lat": 21.05})
    with pytest.warns(UserWarning, match="skipping"):
        (item,) = TS({"WOA": woa})._items()
    assert item["mark"] == "line"
    assert list(item["depth"]) == [0.0, 10.0, 50.0, 100.0, 500.0]
    # salinity = 34 + depth/100: sorted along with the depths
    assert list(item["x"]) == pytest.approx([34.0, 34.1, 34.5, 35.0, 39.0])
    assert item["source"] == "woa_s + woa_t"
    assert item["time"] is not None and len(item["time"]) == 5


def test_box_mean_profile_draws_as_a_line(table):
    table.put("glo", "temperature", _box_mean(10.0, *TEMP_UNITS))
    table.put("glo", "salinity", _box_mean(34.0, *SALT_UNITS))
    glorys = osk.field(
        "glo", PAIR, select=BOX, aggregate={"lon": "mean", "lat": "mean"}
    )
    (item,) = TS({"GLORYS": glorys})._items()
    assert item["mark"] == "line"
    assert list(item["depth"]) == [0.0, 20.0, 100.0, 300.0]
    assert item["time"] is None


def _dz_only(base: float, standard_name: str, *, with_month: bool):
    """Build a time-averaged ROMS column: no ``z_rho`` left, only the ``dz`` weights."""
    dz = xr.DataArray([10.0, 20.0, 30.0], dims="s_rho")  # bottom to top
    dims = ("month", "s_rho") if with_month else ("s_rho",)
    shape = (2, 3) if with_month else (3,)
    values = base + np.arange(np.prod(shape), dtype="float64").reshape(shape)
    return xr.DataArray(
        values,
        dims=dims,
        coords={"dz": dz, "lon": 155.8, "lat": 21.0},
        attrs={"standard_name": standard_name},
    )


def test_a_groupby_month_member_with_only_dz_gets_depths_from_the_cell_thickness(table):
    table.put("roms", "temperature", _dz_only(10.0, TEMP_UNITS[1], with_month=True))
    table.put("roms", "salinity", _dz_only(34.0, SALT_UNITS[1], with_month=True))
    (item,) = TS(_members())._items()
    assert item["mark"] == "points"  # twelve profiles are not one line
    # depth of a centre = thickness above it + half its own: [55, 40, 15], per month
    assert list(item["depth"]) == [55.0, 40.0, 15.0, 55.0, 40.0, 15.0]
    assert item["time"] is None  # a month number is not a date


def test_a_single_dz_column_is_a_line_ordered_top_down(table):
    table.put("roms", "temperature", _dz_only(10.0, TEMP_UNITS[1], with_month=False))
    table.put("roms", "salinity", _dz_only(34.0, SALT_UNITS[1], with_month=False))
    (item,) = TS(_members())._items()
    assert item["mark"] == "line"
    assert list(item["depth"]) == [15.0, 40.0, 55.0]
    assert list(item["x"]) == [36.0, 35.0, 34.0]  # the top level (index 2) first


def test_depth_is_none_when_nothing_carries_it(table):
    bare = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("eta_rho", "xi_rho"),
        coords={"lon_rho": (("eta_rho", "xi_rho"), np.full((2, 3), 155.8))},
    )
    table.put("roms", "temperature", bare)
    table.put("roms", "salinity", bare + 30)
    (item,) = TS(_members())._items()
    assert item["depth"] is None and item["time"] is None
    assert item["mark"] == "points"


def test_x_and_y_that_do_not_line_up_are_refused(table):
    table.put("roms", "temperature", _roms(10.0, *TEMP_UNITS))
    table.put("roms", "salinity", _roms(34.0, *SALT_UNITS).isel(time=0))
    with pytest.raises(ValueError, match="same select/aggregate for both"):
        TS(_members())._items()
    other = _woa(34.0, *SALT_UNITS).assign_coords(depth=[1.0, 2.0, 3.0, 4.0, 5.0])
    table.put("roms", "salinity", other)
    table.put("roms", "temperature", _woa(8.0, "degC", "sea_water_temperature"))
    with pytest.raises(ValueError, match="same select/aggregate for both"):
        TS(_members())._items()


def test_no_finite_pair_drops_the_member_with_a_warning(table):
    _put_roms(table)
    _put_woa(table)
    table.put("roms", "salinity", _roms(34.0, *SALT_UNITS) * np.nan)
    woa = osk.field(["woa_t", "woa_s"], PAIR, select={"lon": 155.79, "lat": 21.05})
    ts = TS({"ROMS": osk.field("roms", PAIR, select=BOX), "WOA": woa})
    with pytest.warns(UserWarning) as record:
        items = ts._items()
    assert [i["label"] for i in items] == ["WOA"]
    assert any("skipping member 'ROMS'" in str(w.message) for w in record)


# -- metadata: names, units, roles, places -------------------------------------------


def test_item_names_units_standard_names_and_roles(table):
    _put_roms(table)
    _put_woa(table)
    woa = osk.field(["woa_t", "woa_s"], PAIR, select={"lon": 155.79, "lat": 21.05})
    ts = TS({"ROMS": osk.field("roms", PAIR, select=BOX), "WOA": woa})
    with pytest.warns(UserWarning, match="skipping"):
        roms, woa_item = ts._items()
    assert (roms["x_name"], roms["y_name"]) == ("salinity", "temperature")
    assert (roms["x_units"], roms["y_units"]) == ("1", "degC")
    assert (roms["x_role"], roms["y_role"]) == ("salinity", "temperature")
    assert roms["x_standard_name"] == "sea_water_practical_salinity"
    assert roms["y_standard_name"] == "sea_water_potential_temperature"
    # the prepared array's own name is what a definition mismatch is read from
    assert woa_item["y_standard_name"] == "sea_water_temperature"
    assert (woa_item["x_role"], woa_item["y_role"]) == ("salinity", "temperature")


def test_roles_follow_the_axes_not_the_order(table):
    _put_roms(table)
    xy = XY(_members(), x="temperature", y="salinity")
    (item,) = xy._items()
    assert (item["x_role"], item["y_role"]) == ("temperature", "salinity")
    assert item["x_name"] == "temperature"


def test_standard_name_falls_back_to_the_request_when_the_data_has_none(table):
    da = _box_mean(10.0, *TEMP_UNITS)
    da.attrs.pop("standard_name")
    table.put("glo", "temperature", da)
    table.put("glo", "salinity", _box_mean(34.0, *SALT_UNITS))
    (item,) = TS({"G": osk.field("glo", PAIR, select=BOX)})._items()
    assert item["y_standard_name"] == "sea_water_potential_temperature"


def test_without_regions_the_panel_place_comes_from_the_members(table):
    _put_roms(table)
    da = _roms(10.0, *TEMP_UNITS)
    da.attrs["region"] = [155.24, 20.51, 156.33, 21.60]
    table.put("roms", "temperature", da)
    table.put("roms", "salinity", da + 24)
    (item,) = TS(_members())._items()
    assert item["lon"] == pytest.approx(155.785)
    assert item["lat"] == pytest.approx(21.055)
    assert "°N" in item["region_note"] and "°E" in item["region_note"]


def test_no_place_means_none_not_a_guess(table):
    _put_roms(table)
    (item,) = TS(_members())._items()
    assert (item["lon"], item["lat"], item["region_note"]) == (None, None, None)


# -- regions -------------------------------------------------------------------------


SG = {"lon": {"min": 184.59, "max": 185.73}, "lat": {"min": 47.76, "max": 48.59}}


def test_regions_replace_each_members_horizontal_select(table):
    _put_roms(table)
    members = {"ROMS": osk.field("roms", PAIR, select={"time": "2012-01", "depth": 50})}
    ts = TS(members, regions={"NWP": BOX, "SG": SG})
    items = ts._items()
    assert [(i["region"], i["label"]) for i in items] == [
        ("NWP", "ROMS"),
        ("SG", "ROMS"),
    ]
    sels = table.selects("roms", "temperature")
    assert sels == [
        {"time": "2012-01", "depth": 50, **BOX},
        {"time": "2012-01", "depth": 50, **SG},  # the 0-360 box goes through as given
    ]


def test_regions_order_panels_before_members(table):
    _put_roms(table)
    _put_woa(table)
    woa = osk.field(["woa_t", "woa_s"], PAIR)
    ts = TS(
        {"ROMS": osk.field("roms", PAIR), "WOA": woa},
        regions={"A": BOX, "B": SG},
        at_center=["WOA"],
    )
    with pytest.warns(UserWarning, match="skipping"):
        items = ts._items()
    assert [(i["region"], i["label"]) for i in items] == [
        ("A", "ROMS"),
        ("A", "WOA"),
        ("B", "ROMS"),
        ("B", "WOA"),
    ]


def test_region_panel_carries_its_centre_wrapped_and_a_note(table):
    _put_roms(table)
    ts = TS({"ROMS": osk.field("roms", PAIR)}, regions={"SG": SG})
    (item,) = ts._items()
    assert item["lon"] == pytest.approx(185.16 - 360)
    assert item["lat"] == pytest.approx(48.175)
    assert "°N" in item["region_note"] and "°W" in item["region_note"]


def test_a_point_region_is_noted_as_a_place(table):
    _put_woa(table)
    ts = TS(
        {"WOA": osk.field(["woa_t", "woa_s"], PAIR)},
        regions={"P": {"lon": 155.79, "lat": 21.05}},
    )
    with pytest.warns(UserWarning, match="skipping"):
        (item,) = ts._items()
    assert item["region_note"] == "21.1°N 155.8°E"


def test_regions_warn_once_when_a_member_had_its_own_location(table):
    _put_roms(table)
    ts = TS(_members(), regions={"A": BOX, "B": SG})
    with pytest.warns(UserWarning, match=r"replaces member 'ROMS'.*lat, lon") as record:
        ts._items()
    replaced = [w for w in record if "replaces member" in str(w.message)]
    assert len(replaced) == 1


def test_regions_without_a_prior_location_do_not_warn(table):
    _put_roms(table)
    ts = TS({"ROMS": osk.field("roms", PAIR)}, regions={"A": BOX})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ts._items()


def test_at_center_samples_a_point_and_drops_the_horizontal_mean(table):
    _put_woa(table)
    table.put("glo", "temperature", _box_mean(10.0, *TEMP_UNITS))
    table.put("glo", "salinity", _box_mean(34.0, *SALT_UNITS))
    agg = {"time": "mean", "lon": "mean", "lat": "mean"}
    ts = TS(
        {
            "WOA": osk.field(["woa_t", "woa_s"], PAIR, aggregate=agg),
            "GLORYS": osk.field("glo", PAIR, aggregate=agg),
        },
        regions={"NWP": BOX},
        at_center=["WOA"],
    )
    with pytest.warns(UserWarning, match="skipping"):
        ts._items()
    (sel,) = table.selects("woa_s", "salinity")
    assert sel == {"lon": pytest.approx(155.785), "lat": pytest.approx(21.055)}
    woa_agg = [c[3] for c in table.calls if c[0] == "woa_s"]
    assert woa_agg == [{"time": "mean"}]
    # a member not listed keeps its box and its mean
    (sel,) = table.selects("glo", "salinity")
    assert sel == BOX
    assert next(c[3] for c in table.calls if c[0] == "glo") == agg


def test_at_center_keeps_the_0_360_centre_contiguous(table):
    _put_woa(table)
    ts = TS(
        {"WOA": osk.field(["woa_t", "woa_s"], PAIR)},
        regions={"SG": SG},
        at_center=["WOA"],
    )
    with pytest.warns(UserWarning, match="skipping"):
        ts._items()
    (sel,) = table.selects("woa_s", "salinity")
    assert sel["lon"] == pytest.approx(185.16)


def test_an_empty_box_drops_the_member_with_the_at_center_hint(table):
    _put_roms(table)

    def too_coarse(select, aggregate):
        raise align.EmptySelection("select lon/lat box (1..2, 3..4) selects nothing")

    table.put("woa_t", "temperature", too_coarse)
    table.put("woa_s", "salinity", too_coarse)
    ts = TS(
        {
            "ROMS": osk.field("roms", PAIR),
            "WOA": osk.field(["woa_t", "woa_s"], PAIR),
        },
        regions={"NWP": BOX},
    )
    with pytest.warns(UserWarning) as record:
        items = ts._items()
    assert [i["label"] for i in items] == ["ROMS"]
    dropped = [str(w.message) for w in record if "skipping member" in str(w.message)]
    assert len(dropped) == 1
    assert "'WOA' in 'NWP'" in dropped[0]
    assert "grid may be coarser than the box" in dropped[0]
    assert "at_center=['WOA']" in dropped[0]


def test_nothing_left_is_an_error(table):
    def gone(select, aggregate):
        raise align.NoValidData("masked")

    table.put("roms", "temperature", gone)
    table.put("roms", "salinity", gone)
    with pytest.warns(UserWarning, match="skipping member 'ROMS'"):
        with pytest.raises(ValueError, match="nothing to draw"):
            TS(_members())._items()


def test_a_missing_variable_at_read_time_is_dropped_like_any_other(table):
    _put_roms(table)
    table.put("other", "temperature", _roms(10.0, *TEMP_UNITS))
    table.put("other", "salinity", None)  # probes available, reads as missing
    ts = TS(
        {
            "ROMS": osk.field("roms", PAIR, select=BOX),
            "Other": osk.field("other", PAIR, select=BOX),
        }
    )
    with pytest.warns(UserWarning, match="skipping member 'Other'"):
        items = ts._items()
    assert [i["label"] for i in items] == ["ROMS"]


# -- data ----------------------------------------------------------------------------


def test_data_is_keyed_by_region_and_member(table):
    _put_roms(table)
    xy = TS({"ROMS": osk.field("roms", PAIR)}, regions={"A": BOX, "B": SG})
    data = xy.data
    assert list(data) == ["A", "B"]
    assert set(data["A"]["ROMS"]) == {"x", "y"}
    assert data["A"]["ROMS"]["x"].attrs["standard_name"] == SALT_UNITS[1]
    assert isinstance(data["B"]["ROMS"]["y"], xr.DataArray)


def test_data_without_regions_uses_the_none_key(table):
    _put_roms(table)
    assert list(TS(_members()).data) == [None]


def test_each_field_is_read_once_across_data_and_plot(table, captured):
    _put_roms(table)
    ts = TS(_members())
    ts.data
    ts.plot()
    assert len(table.calls) == 2


# -- plot ----------------------------------------------------------------------------


def test_ts_plot_builds_an_xy_spec_with_density_on(table, captured):
    _put_roms(table)
    out = TS(_members()).plot(title="T-S", ncols=3)
    spec = captured["spec"]
    assert out == "figure"
    assert spec.family == "XY"
    assert spec.options == {"density": True, "title": "T-S", "ncols": 3}
    assert len(spec.items) == 1 and spec.items[0]["label"] == "ROMS"
    assert captured["renderer"] == "matplotlib"


def test_ts_density_can_be_turned_off_and_the_renderer_chosen(table, captured):
    _put_roms(table)
    TS(_members()).plot(density=False, renderer="holoviews")
    assert captured["spec"].options == {"density": False}
    assert captured["renderer"] == "holoviews"


def test_xy_plot_passes_options_through_untouched(table, captured):
    _put_roms(table)
    XY(_members(), x="salinity", y="temperature").plot(color_by="depth", alpha=0.2)
    assert captured["spec"].family == "XY"
    assert captured["spec"].options == {"color_by": "depth", "alpha": 0.2}


def test_save_writes_under_the_figures_dir_and_returns_the_path(
    table, captured, monkeypatch, tmp_path
):
    from ocean_skill import outputs

    _put_roms(table)
    seen = {}

    def figures_dir(project):
        seen["project"] = project
        return tmp_path

    monkeypatch.setattr(outputs, "figures_dir", figures_dir)
    out = TS(_members()).save()
    assert seen["project"] == "roms"
    assert out == {"figure": tmp_path / "temperature_vs_salinity.png"}
    assert captured["spec"].options["save"] == out["figure"]
    assert captured["spec"].options["density"] is True


def test_map_locations_maps_every_region_and_member(table, monkeypatch):
    _put_roms(table)
    got = {}

    def fake(self, **kwargs):
        got["fields"] = self.fields
        return "map"

    monkeypatch.setattr(FieldSet, "map_locations", fake)
    ts = TS({"ROMS": osk.field("roms", PAIR)}, regions={"A": BOX, "B": SG})
    assert ts.map_locations() == "map"
    assert len(got["fields"]) == 4
    assert [f.select for f in got["fields"][::2]] == [BOX, SG]


# -- the pieces around it ------------------------------------------------------------


def test_an_empty_box_raises_the_dedicated_exception_still_a_value_error():
    grid = xr.DataArray(
        np.zeros((3, 3)),
        dims=("lat", "lon"),
        coords={"lat": [20.0, 21.0, 22.0], "lon": [155.0, 156.0, 157.0]},
    )
    assert issubclass(align.EmptySelection, ValueError)
    with pytest.raises(align.EmptySelection, match="selects nothing"):
        align.subset_to_box(grid, (155.24, 20.1, 155.5, 20.9))
    with pytest.raises(ValueError, match="selects nothing"):
        align.subset_to_box(grid, (155.24, 20.1, 155.5, 20.9))


def test_the_items_draw_with_the_static_renderer(table):
    import matplotlib.pyplot as plt

    _put_roms(table, nan_corner=True)
    _put_woa(table)
    ts = TS(
        {
            "ROMS": osk.field("roms", PAIR),
            "WOA": osk.field(["woa_t", "woa_s"], PAIR),
        },
        regions={"A": BOX, "B": SG},
        at_center=["WOA"],
    )
    with pytest.warns(UserWarning, match="skipping"):
        fig = ts.plot(density=False, ncols=2)
    try:
        assert len(fig.axes) >= 2
    finally:
        plt.close(fig)
