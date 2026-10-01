"""Tests for the ``XY`` family: the layout composer and the static renderer.

Items are built by hand in the shape ``ocean_skill.xy.XY`` produces (see
``PlotSpec``'s docstring), so composition and drawing are tested without reading any
data. The interactive renderer's twin of these checks lives in
``tests/test_xy_holoviews.py``; both read the same ``compose()`` layout.
"""

from __future__ import annotations

import warnings
import zlib

import numpy as np
import pytest
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D

from ocean_skill.plot import style as _style
from ocean_skill.plot import xy as _xy
from ocean_skill.plot.registry import render
from ocean_skill.plot.spec import PlotSpec

REGIONS = [
    "North West Pacific",
    "Subpolar Gyre",
    "California Current System",
    "South West Pacific",
    "South Pacific Gyre",
    "Peru Current",
]


def _item(
    label: str = "ROMS",
    region: str | None = "North West Pacific",
    mark: str = "points",
    *,
    n: int | None = None,
    s0: float = 34.5,
    t0: float = 10.0,
    depth: bool = True,
    time: bool = True,
    x_role: str | None = "salinity",
    y_role: str | None = "temperature",
    x_name: str = "salinity",
    y_name: str = "temperature",
    x_units: str | None = None,
    y_units: str | None = "degC",
    x_standard_name: str | None = "sea_water_practical_salinity",
    y_standard_name: str | None = "sea_water_potential_temperature",
    lon: float | None = 155.8,
    lat: float | None = 21.0,
    region_note: str | None = None,
) -> dict:
    """One XY item, shaped as ``XY._items()`` builds it (Contract 1)."""
    n = n or (12 if mark == "line" else 200)
    rng = np.random.default_rng(zlib.crc32(f"{label}|{region}|{mark}".encode()))
    d = np.sort(rng.uniform(5, 1000, n)) if mark == "line" else rng.uniform(5, 1000, n)
    return {
        "label": label,
        "region": region,
        "mark": mark,
        "x": s0 + 0.4 * np.cos(d / 300) + rng.normal(0, 0.01, n),
        "y": t0 + 8 * np.exp(-d / 300) + rng.normal(0, 0.05, n),
        "depth": d if depth else None,
        "time": (
            np.datetime64("2020-01-01")
            + rng.integers(0, 365, n).astype("timedelta64[D]")
            if time
            else None
        ),
        "x_name": x_name,
        "x_units": x_units,
        "x_standard_name": x_standard_name,
        "y_name": y_name,
        "y_units": y_units,
        "y_standard_name": y_standard_name,
        "x_role": x_role,
        "y_role": y_role,
        "lon": lon,
        "lat": lat,
        "region_note": region_note,
        "source": "test",
    }


def _ts_items(regions=REGIONS[:1], members=("ROMS", "WOA23", "GLORYS12")) -> list[dict]:
    """Build the Fig. 7 trio per region: a model cloud, then two profiles."""
    out = []
    for region in regions:
        for label in members:
            out.append(_item(label, region, "points" if label == "ROMS" else "line"))
    return out


def _draw(items, **opts):
    return render(PlotSpec("XY", items, opts), renderer="matplotlib")


def _panel_axes(fig):
    """Return the visible drawing axes (no colorbars), in grid order."""
    return [
        ax for ax in fig.axes if ax.get_visible() and ax.get_label() != "<colorbar>"
    ]


def _scatter(ax):
    return [c for c in ax.collections if isinstance(c, PathCollection)]


def _contours(ax):
    from matplotlib.contour import ContourSet

    return [c for c in ax.collections if isinstance(c, ContourSet)]


def _colorbars(fig):
    return [ax for ax in fig.axes if ax.get_label() == "<colorbar>"]


# --- compose: panels, titles, grid ----------------------------------------------------


def test_panels_follow_first_seen_region_order():
    items = _ts_items(REGIONS[:3])
    layout = _xy.compose(items)
    assert [p.key for p in layout.panels] == REGIONS[:3]
    assert [p.title for p in layout.panels] == REGIONS[:3]
    assert [[s.label for s in p.items] for p in layout.panels] == [
        ["ROMS", "WOA23", "GLORYS12"]
    ] * 3


def test_no_region_titles_the_panel_by_its_note():
    layout = _xy.compose([_item(region=None, region_note="155.2E 20.5N")])
    assert layout.panels[0].key is None
    assert layout.panels[0].title == "155.2E 20.5N"
    # nothing to say when there is no note either
    assert _xy.compose([_item(region=None)]).panels[0].title == ""


def test_grid_defaults_to_one_row_then_three_columns():
    assert (
        _xy.compose(_ts_items(REGIONS[:3])).nrows,
        _xy.compose(_ts_items(REGIONS[:3])).ncols,
    ) == (1, 3)
    six = _xy.compose(_ts_items(REGIONS))
    assert (six.nrows, six.ncols) == (2, 3)
    four = _xy.compose(_ts_items(REGIONS[:4]))
    assert (four.nrows, four.ncols) == (2, 3)
    assert (
        _xy.compose(_ts_items(REGIONS), ncols=2).nrows,
        _xy.compose(_ts_items(REGIONS), ncols=2).ncols,
    ) == (3, 2)
    assert _xy.compose(_ts_items(REGIONS), nrows=1).ncols == 6


def test_titles_override_and_wrong_count():
    items = _ts_items(REGIONS[:2])
    layout = _xy.compose(items, titles=["A", None])
    assert [p.title for p in layout.panels] == ["A", REGIONS[1]]
    with pytest.raises(ValueError, match="titles needs one entry per panel"):
        _xy.compose(items, titles=["only one"])


def test_empty_items_are_refused():
    with pytest.raises(ValueError, match="at least one item"):
        _xy.compose([])


def test_mismatched_lengths_and_marks_are_refused():
    bad = _item()
    bad["y"] = bad["y"][:-1]
    with pytest.raises(ValueError, match="x has 200 values but y has 199"):
        _xy.compose([bad])
    with pytest.raises(ValueError, match="expected 'points' or 'line'"):
        _xy.compose([{**_item(), "mark": "blob"}])


# --- compose: colours and markers -----------------------------------------------------


def test_first_points_member_is_black_and_the_rest_take_the_cycle():
    layout = _xy.compose(_ts_items())
    colors = {s.label: s.color for s in layout.panels[0].items}
    assert colors == {
        "ROMS": "black",
        "WOA23": _style.COLOR_CYCLE[0],
        "GLORYS12": _style.COLOR_CYCLE[1],
    }
    # a lone cloud keeps the plain dot; lines have no marker at all
    markers = {s.label: s.marker for s in layout.panels[0].items}
    assert markers == {"ROMS": "o", "WOA23": None, "GLORYS12": None}


def test_black_goes_to_the_first_points_member_not_the_first_member():
    items = [_item("WOA23", mark="line"), _item("ROMS", mark="points")]
    colors = {s.label: s.color for s in _xy.compose(items).panels[0].items}
    assert colors == {"ROMS": "black", "WOA23": _style.COLOR_CYCLE[0]}


def test_markers_vary_only_when_several_members_are_points():
    items = [_item("A"), _item("B"), _item("C", mark="line")]
    markers = {s.label: s.marker for s in _xy.compose(items).panels[0].items}
    assert markers == {"A": _style.MARKERS[0], "B": _style.MARKERS[1], "C": None}


def test_colors_string_list_and_dict():
    items = _ts_items()
    pinned = {
        s.label: s.color for s in _xy.compose(items, colors="tab:red").panels[0].items
    }
    assert set(pinned.values()) == {"tab:red"}
    listed = _xy.compose(items, colors=["r", "g", "b"]).panels[0].items
    assert [s.color for s in listed] == ["r", "g", "b"]
    partial = _xy.compose(items, colors={"WOA23": "tab:red"}).panels[0].items
    # naming one member leaves the others' defaults exactly where they were
    assert {s.label: s.color for s in partial} == {
        "ROMS": "black",
        "WOA23": "tab:red",
        "GLORYS12": _style.COLOR_CYCLE[1],
    }


def test_colors_errors_name_the_members():
    items = _ts_items()
    with pytest.raises(
        ValueError,
        match=r"names 'ROMZ'.*available members: 'ROMS', 'WOA23', 'GLORYS12'",
    ):
        _xy.compose(items, colors={"ROMZ": "k"})
    with pytest.raises(
        ValueError, match="colors has 2 entries but there are 3 members"
    ):
        _xy.compose(items, colors=["k", "r"])


# --- compose: color_by ----------------------------------------------------------------


def test_color_by_depth_shares_one_scale_across_panels():
    a, b = _item(region="A"), _item(region="B")
    a["depth"] = np.linspace(10, 100, a["x"].size)
    b["depth"] = np.linspace(50, 400, b["x"].size)
    layout = _xy.compose([a, b], color_by="depth")
    scale = layout.colorbar
    assert (scale.field, scale.label, scale.vmin, scale.vmax) == (
        "depth",
        "depth [m]",
        10,
        400,
    )
    assert scale.show and scale.inverted and not scale.is_time
    for panel in layout.panels:
        (member,) = panel.items
        assert member.color_values is not None
        assert member.color == "black"  # the legend swatch keeps the member's colour
    assert layout.colorbar.norm.vmin == 10 and layout.colorbar.norm.vmax == 400


def test_color_by_leaves_lines_solid():
    layout = _xy.compose(_ts_items(), color_by="depth")
    by_label = {s.label: s for s in layout.panels[0].items}
    assert by_label["ROMS"].color_values is not None
    assert by_label["WOA23"].color_values is None
    assert by_label["GLORYS12"].color_values is None


def test_color_by_time_is_days_since_the_epoch_and_viridis():
    import matplotlib

    item = _item()
    item["time"] = np.array(
        ["1970-01-02", "1970-01-12"] * (item["x"].size // 2), dtype="datetime64[ns]"
    )
    scale = _xy.compose([item], color_by="time").colorbar
    assert (scale.vmin, scale.vmax, scale.is_time, scale.inverted) == (
        1.0,
        11.0,
        True,
        False,
    )
    assert scale.label == "time"
    assert scale.cmap.name == matplotlib.colormaps["viridis"].name


def test_color_by_cmap_default_and_override():
    import matplotlib

    from ocean_skill.colormaps import cmaps_for

    default = _xy.compose([_item()], color_by="depth").colorbar.cmap
    assert default.name == cmaps_for("sea_floor_depth")[0].name
    named = _xy.compose([_item()], color_by="depth", cmap="magma").colorbar.cmap
    assert named.name == "magma"
    custom = matplotlib.colormaps["plasma"]
    assert _xy.compose([_item()], color_by="depth", cmap=custom).colorbar.cmap is custom


def test_color_by_missing_array_warns_and_stays_solid():
    items = [_item("ROMS"), _item("OTHER", depth=False)]
    with pytest.warns(
        UserWarning, match=r"color_by='depth': 'OTHER' carries no depth values"
    ):
        layout = _xy.compose(items, color_by="depth")
    by_label = {s.label: s for s in layout.panels[0].items}
    assert by_label["OTHER"].color_values is None
    assert by_label["ROMS"].color_values is not None


def test_color_by_with_nothing_to_colour_has_no_scale():
    with pytest.warns(UserWarning, match="carries no depth values"):
        layout = _xy.compose([_item(depth=False)], color_by="depth")
    assert layout.colorbar is None
    with pytest.warns(UserWarning, match="colours points members, and there are none"):
        assert _xy.compose([_item(mark="line")], color_by="depth").colorbar is None


def test_color_by_unknown_field_is_refused():
    with pytest.raises(
        ValueError, match=r"color_by='salinity'.*one of \('depth', 'time'\)"
    ):
        _xy.compose([_item()], color_by="salinity")


def test_colorbar_false_keeps_the_colours():
    scale = _xy.compose([_item()], color_by="depth", colorbar=False).colorbar
    assert scale is not None and scale.show is False


# --- compose: limits and labels -------------------------------------------------------


def test_automatic_limits_pad_two_percent():
    item = _item()
    layout = _xy.compose([item])
    lo, hi = item["x"].min(), item["x"].max()
    pad = 0.02 * (hi - lo)
    assert layout.panels[0].xlim == pytest.approx((lo - pad, hi + pad))
    lo, hi = item["y"].min(), item["y"].max()
    pad = 0.02 * (hi - lo)
    assert layout.panels[0].ylim == pytest.approx((lo - pad, hi + pad))


def test_limits_are_per_panel_unless_shared_or_pinned():
    a, b = _item(region="A", s0=34.0), _item(region="B", s0=36.0)
    separate = _xy.compose([a, b])
    assert separate.panels[0].xlim != separate.panels[1].xlim
    shared = _xy.compose([a, b], sharex=True)
    assert shared.panels[0].xlim == shared.panels[1].xlim
    assert (
        shared.panels[0].xlim[0] < a["x"].min()
        and shared.panels[0].xlim[1] > b["x"].max()
    )
    # sharing x leaves y alone
    assert (
        shared.panels[0].ylim != shared.panels[1].ylim or a["y"].min() == b["y"].min()
    )
    pinned = _xy.compose([a, b], xlim=(30, 40), ylim=(0, 5))
    assert [p.xlim for p in pinned.panels] == [(30.0, 40.0)] * 2
    assert [p.ylim for p in pinned.panels] == [(0.0, 5.0)] * 2


def test_bad_limits_are_refused():
    with pytest.raises(ValueError, match=r"xlim=.*not a"):
        _xy.compose([_item()], xlim=(1, 2, 3))
    with pytest.raises(ValueError, match=r"ylim=.*finite"):
        _xy.compose([_item()], ylim=(0, np.inf))


def test_axis_labels_use_short_names_and_agreed_units():
    layout = _xy.compose(_ts_items())
    assert layout.xlabel == "salinity"  # no units to give
    assert layout.ylabel == "temperature [degC]"
    # a dimensionless "1" is not worth a bracket
    assert _xy.compose([_item(x_units="1")]).xlabel == "salinity"
    assert _xy.compose([_item(x_units="PSU")]).xlabel == "salinity [PSU]"


def test_label_units_disagreement_warns_and_drops_them():
    items = [_item("A", y_units="degC"), _item("B", y_units="K")]
    with pytest.warns(
        UserWarning, match=r"disagree on the units of the y axis \(degC, K\)"
    ):
        layout = _xy.compose(items)
    assert layout.ylabel == "temperature"


# --- compose: annotations -------------------------------------------------------------


def test_annotations_flat_form_goes_on_every_panel():
    layout = _xy.compose(
        _ts_items(REGIONS[:2]), annotations={"STSW": (34.8, 20.0), "PDW": [34.5, 1]}
    )
    expected = (("STSW", 34.8, 20.0), ("PDW", 34.5, 1.0))
    assert [p.annotations for p in layout.panels] == [expected, expected]


def test_annotations_per_region_form_goes_on_its_own_panel():
    layout = _xy.compose(
        _ts_items(REGIONS[:3]),
        annotations={REGIONS[0]: {"A": (1, 2)}, REGIONS[2]: {"B\nC": (3, 4)}},
    )
    assert [p.annotations for p in layout.panels] == [
        (("A", 1.0, 2.0),),
        (),
        (("B\nC", 3.0, 4.0),),
    ]


def test_annotation_errors():
    items = _ts_items(REGIONS[:2])
    with pytest.raises(ValueError, match="mixes the two forms"):
        _xy.compose(items, annotations={"A": (1, 2), REGIONS[0]: {"B": (3, 4)}})
    with pytest.raises(
        ValueError, match="keyed by region, but this plot has no regions"
    ):
        _xy.compose([_item(region=None)], annotations={"R": {"B": (3, 4)}})
    for bad in ((1,), (1, 2, 3), "xy", (1, "a"), (1, np.nan), None):
        with pytest.raises(ValueError, match=r"'A' must map to an \(x, y\) pair"):
            _xy.compose(items, annotations={"A": bad})
    with pytest.raises(
        ValueError, match=r"annotations\['North West Pacific'\]: 'A' must map"
    ):
        _xy.compose(items, annotations={REGIONS[0]: {"A": (1,)}})
    with pytest.raises(ValueError, match="must be a dict"):
        _xy.compose(items, annotations=[("A", (1, 2))])


def test_annotation_unknown_region_warns_listing_the_panels():
    items = _ts_items(REGIONS[:2])
    with pytest.warns(
        UserWarning,
        match=(
            r"names \['Subpolar Gire'\].*"
            r"The panels are: \['North West Pacific', 'Subpolar Gyre'\]"
        ),
    ):
        layout = _xy.compose(
            items,
            annotations={"Subpolar Gire": {"A": (1, 2)}, REGIONS[0]: {"B": (3, 4)}},
        )
    assert [p.annotations for p in layout.panels] == [(("B", 3.0, 4.0),), ()]


def test_annotations_widen_automatic_limits_only():
    item = _item()
    far = (item["x"].max() + 3.0, item["y"].min() - 5.0)
    layout = _xy.compose([item], annotations={"far": far})
    panel = layout.panels[0]
    assert panel.xlim[1] > far[0] and panel.ylim[0] < far[1]
    # a pinned limit is the caller's word
    pinned = _xy.compose([item], annotations={"far": far}, xlim=(34, 35), ylim=(0, 20))
    assert pinned.panels[0].xlim == (34.0, 35.0)


def test_normalize_annotations_covers_every_panel_key():
    out = _xy.normalize_annotations({"A": (1, 2)}, [None])
    assert out == {None: (("A", 1.0, 2.0),)}
    assert _xy.normalize_annotations(None, ["r", "s"]) == {"r": (), "s": ()}
    assert _xy.normalize_annotations({}, ["r"]) == {"r": ()}


# --- compose: density -----------------------------------------------------------------


def test_density_needs_salinity_and_temperature():
    nutrients = _item(x_role=None, y_role=None, x_name="phosphate", y_name="nitrate")
    with pytest.raises(
        ValueError,
        match=(
            r"needs one axis to be salinity and the other temperature.*"
            r"x is 'phosphate' \(role None\)"
        ),
    ):
        _xy.compose([nutrients], density=True)
    # two salinities is no better
    with pytest.raises(ValueError, match="salinity and the other temperature"):
        _xy.compose([_item(y_role="salinity")], density=True)
    # one member without roles spoils it, and the message says so
    with pytest.raises(ValueError, match="for at least one member"):
        _xy.compose([_item("A"), _item("B", x_role=None)], density=True)
    # off is always fine, nutrients or not
    assert _xy.compose([nutrients], density=False).panels[0].density is None


@pytest.mark.parametrize("bad", [0, -2, [], "sigma", ["a"], [np.nan], 1.5])
def test_density_spec_is_validated(bad):
    pytest.importorskip("gsw")
    with pytest.raises(ValueError, match="density="):
        _xy.compose([_item()], density=bad)


def test_density_grid_salinity_on_x():
    gsw = pytest.importorskip("gsw")
    layout = _xy.compose([_item()], density=True)
    grid = layout.panels[0].density
    xlim, ylim = layout.panels[0].xlim, layout.panels[0].ylim
    assert grid.sigma.shape == (120, 120)
    assert (grid.x[0], grid.x[-1]) == pytest.approx(xlim)
    assert (grid.y[0], grid.y[-1]) == pytest.approx(ylim)
    # sigma[j, i] is at (x[i], y[j]): denser to the salty side, lighter to the warm one
    assert np.all(np.diff(grid.sigma, axis=1) > 0)
    assert np.all(np.diff(grid.sigma, axis=0) < 0)
    s, t = grid.x[40], grid.y[70]
    sa = gsw.SA_from_SP(s, 0.0, 155.8, 21.0)
    assert grid.sigma[70, 40] == pytest.approx(
        float(gsw.sigma0(sa, gsw.CT_from_pt(sa, t)))
    )


def test_density_grid_temperature_on_x():
    pytest.importorskip("gsw")
    swapped = _item(
        x_role="temperature", y_role="salinity", x_name="temperature", y_name="salinity"
    )
    swapped["x"], swapped["y"] = _item()["y"], _item()["x"]
    grid = _xy.compose([swapped], density=True).panels[0].density
    assert np.all(np.diff(grid.sigma, axis=1) < 0)  # warmer to the right: lighter
    assert np.all(np.diff(grid.sigma, axis=0) > 0)  # saltier upward: denser
    straight = _xy.density_grid(
        (34.0, 35.0), (5.0, 20.0), lon=155.8, lat=21.0, levels=True, swap=False
    )
    flipped = _xy.density_grid(
        (5.0, 20.0), (34.0, 35.0), lon=155.8, lat=21.0, levels=True, swap=True
    )
    assert np.allclose(straight.sigma, flipped.sigma.T)
    assert straight.levels == flipped.levels


def test_density_auto_levels_use_a_round_step_and_a_sane_count():
    pytest.importorskip("gsw")
    for xlim, ylim in [
        ((33.0, 36.0), (0.0, 30.0)),
        ((34.3, 34.9), (2.0, 10.0)),
        ((34.0, 34.4), (3.0, 4.0)),
    ]:
        grid = _xy.density_grid(xlim, ylim, lon=0.0, lat=0.0, levels=True)
        levels = np.asarray(grid.levels)
        assert 6 <= levels.size <= 14, (xlim, ylim, levels)
        step = np.unique(np.round(np.diff(levels), 6))
        assert step.size == 1 and step[0] in (1.0, 0.5, 0.25, 0.1, 0.05, 2.0), step
        assert levels.min() >= np.nanmin(grid.sigma) and levels.max() <= np.nanmax(
            grid.sigma
        )
        assert np.allclose(levels / step[0], np.round(levels / step[0]))


def test_density_int_and_list_levels():
    pytest.importorskip("gsw")
    wide = ((33.0, 36.0), (0.0, 30.0))
    few = _xy.density_grid(*wide, lon=0.0, lat=0.0, levels=4).levels
    many = _xy.density_grid(*wide, lon=0.0, lat=0.0, levels=12).levels
    assert len(few) < len(many) and abs(len(few) - 4) <= 2
    assert _xy.density_grid(*wide, lon=0.0, lat=0.0, levels=[26.5, 24.0]).levels == (
        24.0,
        26.5,
    )


def test_density_without_a_position_still_draws():
    pytest.importorskip("gsw")
    item = _item(lon=None, lat=None)
    assert _xy.compose([item], density=True).panels[0].density.levels


def test_density_uses_each_panels_own_position_and_limits():
    pytest.importorskip("gsw")
    a, b = (
        _item(region="A", lon=-170.0, lat=-20.0),
        _item(region="B", lon=155.8, lat=21.0, s0=35.5),
    )
    layout = _xy.compose([a, b], density=True)
    ga, gb = (p.density for p in layout.panels)
    assert not np.allclose(ga.x, gb.x)
    assert ga.levels != gb.levels or not np.allclose(ga.sigma, gb.sigma)


# --- compose: legend, warnings --------------------------------------------------------


def test_legend_entries_say_which_are_dots_and_which_lines():
    layout = _xy.compose(_ts_items())
    entries = layout.panels[0].legend
    assert [(e.label, e.mark, e.marker, e.linestyle) for e in entries] == [
        ("ROMS", "points", "o", "-"),
        ("WOA23", "line", None, "-"),
        ("GLORYS12", "line", None, "-"),
    ]
    assert entries == layout.legend_entries
    assert layout.shared_legend and layout.legend_placement == "auto"


def test_legend_is_not_shared_when_panels_differ():
    items = _ts_items(REGIONS[:2])
    items.pop()  # the second panel loses GLORYS12
    assert _xy.compose(items).shared_legend is False
    assert [e.label for e in _xy.compose(items).legend_entries] == [
        "ROMS",
        "WOA23",
        "GLORYS12",
    ]


def test_legend_placement_forms():
    items = _ts_items()
    assert _xy.compose(items, legend=False).legend_placement == "off"
    assert _xy.compose(items, legend="below").legend_placement == "below"
    forced = _xy.compose(items, legend="lower left")
    assert (
        forced.legend_placement == "corner"
        and forced.panels[0].legend_corner == "lower left"
    )
    with pytest.raises(ValueError, match="legend='middle'"):
        _xy.compose(items, legend="middle")


def test_legend_corner_avoids_the_data():
    # a cloud hugging the upper left leaves the lower right empty
    item = _item()
    item["x"] = np.linspace(0, 1, item["x"].size)
    item["y"] = 1 - item["x"] * 0.0 + np.linspace(0, 0.01, item["x"].size)
    corner = _xy.compose([item], xlim=(0, 10), ylim=(0, 1.2)).panels[0].legend_corner
    assert corner in ("lower right", "upper right")


def test_point_cap_warns(monkeypatch):
    monkeypatch.setattr(_xy, "POINT_CAP", 150)
    with pytest.warns(
        UserWarning, match=r"'North West Pacific' holds 200 points \(more than 150\)"
    ):
        _xy.compose([_item()])
    # lines do not count toward it
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _xy.compose([_item(mark="line", n=500)])


def test_point_cap_is_per_panel(monkeypatch):
    monkeypatch.setattr(_xy, "POINT_CAP", 250)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _xy.compose([_item(region="A"), _item(region="B")])


def test_standard_name_disagreement_warns_naming_both():
    items = [
        _item("ROMS", y_standard_name="sea_water_potential_temperature"),
        _item("WOA23", mark="line", y_standard_name="sea_water_temperature"),
    ]
    with pytest.warns(UserWarning) as caught:
        _xy.compose(items)
    (msg,) = [str(w.message) for w in caught if "disagree" in str(w.message)]
    assert "y axis" in msg
    assert "'sea_water_potential_temperature' (ROMS)" in msg
    assert "'sea_water_temperature' (WOA23)" in msg
    assert "no conversion" in msg


def test_standard_names_that_agree_or_are_unknown_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _xy.compose(_ts_items())
        _xy.compose(
            [
                _item("A", y_standard_name=None),
                _item("B", y_standard_name="sea_water_temperature"),
            ]
        )


# --- renderer: artists ----------------------------------------------------------------


def test_one_panel_draws_a_cloud_and_two_profiles():
    fig = _draw(_ts_items())
    (ax,) = _panel_axes(fig)
    (cloud,) = _scatter(ax)
    assert len(cloud.get_offsets()) == 200
    assert len(ax.lines) == 2
    assert [
        tuple(np.round(line.get_xdata()[:1], 6)) for line in ax.lines
    ]  # drawn, in data order
    assert ax.get_title() == "North West Pacific"
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("salinity", "temperature [degC]")
    assert not _contours(ax)  # density is off unless asked for


def test_dots_use_marker_size_alpha_and_are_rasterized():
    (ax,) = _panel_axes(_draw(_ts_items()))
    (cloud,) = _scatter(ax)
    assert cloud.get_sizes()[0] == 2.0
    assert cloud.get_alpha() == 0.5
    assert cloud.get_rasterized()
    assert cloud.get_linewidths()[0] == 0
    (ax,) = _panel_axes(_draw(_ts_items(), marker_size=9, alpha=0.2))
    (cloud,) = _scatter(ax)
    assert (cloud.get_sizes()[0], cloud.get_alpha()) == (9, 0.2)


def test_dots_sit_under_lines():
    (ax,) = _panel_axes(_draw(_ts_items()))
    (cloud,) = _scatter(ax)
    assert all(line.get_zorder() > cloud.get_zorder() for line in ax.lines)


def test_member_colours_reach_the_artists():
    import matplotlib.colors as mcolors

    (ax,) = _panel_axes(
        _draw(
            _ts_items(),
            colors={"ROMS": "black", "WOA23": "tab:red", "GLORYS12": "tab:blue"},
        )
    )
    (cloud,) = _scatter(ax)
    assert mcolors.same_color(cloud.get_facecolor()[0][:3], "black")
    assert [
        mcolors.same_color(line.get_color(), c)
        for line, c in zip(ax.lines, ("tab:red", "tab:blue"), strict=True)
    ] == [True, True]


def test_line_kwargs_style_the_lines():
    (ax,) = _panel_axes(
        _draw(_ts_items(), line_kwargs={"linewidth": 3.0, "linestyle": "--"})
    )
    assert {(line.get_linewidth(), line.get_linestyle()) for line in ax.lines} == {
        (3.0, "--")
    }


def test_six_regions_make_a_two_by_three_grid_with_outer_labels_only():
    fig = _draw(_ts_items(REGIONS))
    axes = _panel_axes(fig)
    assert [ax.get_title() for ax in axes] == REGIONS
    assert axes[0].get_subplotspec().get_gridspec().get_geometry() == (2, 3)
    xlabels = [bool(ax.get_xlabel()) for ax in axes]
    ylabels = [bool(ax.get_ylabel()) for ax in axes]
    assert xlabels == [False] * 3 + [True] * 3
    assert ylabels == [True, False, False, True, False, False]


def test_a_ragged_grid_hides_its_trailing_cells_and_labels_the_panel_above_them():
    fig = _draw(_ts_items(REGIONS[:4]))
    visible = _panel_axes(fig)
    assert len(visible) == 4 and len([a for a in fig.axes if not a.get_visible()]) == 2
    # panels 2 and 3 of row one have an empty cell below them, so they carry the x label
    assert [bool(ax.get_xlabel()) for ax in visible] == [False, True, True, True]


def test_ncols_and_nrows_wrap_the_grid():
    fig = _draw(_ts_items(REGIONS), ncols=2)
    assert _panel_axes(fig)[0].get_subplotspec().get_gridspec().get_geometry() == (3, 2)
    fig = _draw(_ts_items(REGIONS[:3]), nrows=3)
    assert _panel_axes(fig)[0].get_subplotspec().get_gridspec().get_geometry() == (3, 1)


def test_titles_option_overrides_panel_titles():
    fig = _draw(_ts_items(REGIONS[:2]), titles=["NWP", None])
    assert [ax.get_title() for ax in _panel_axes(fig)] == ["NWP", REGIONS[1]]


def test_title_is_the_suptitle():
    fig = _draw(_ts_items(), title="T-S")
    assert fig._suptitle.get_text() == "T-S"


def test_sharex_and_sharey_share_the_axes():
    a, b = _item(region="A", s0=34.0), _item(region="B", s0=36.0)
    fig = _draw([a, b], sharex=True)
    ax_a, ax_b = _panel_axes(fig)
    assert ax_a.get_xlim() == ax_b.get_xlim()
    assert ax_a.get_xlim()[0] < 34.0 and ax_a.get_xlim()[1] > 36.0
    assert ax_a.get_shared_x_axes().joined(ax_a, ax_b)
    assert not ax_a.get_shared_y_axes().joined(ax_a, ax_b)
    fig = _draw([a, b], sharey=True)
    ax_a, ax_b = _panel_axes(fig)
    assert (
        ax_a.get_shared_y_axes().joined(ax_a, ax_b)
        and ax_a.get_ylim() == ax_b.get_ylim()
    )


def test_xlim_and_ylim_pin_every_panel():
    fig = _draw(_ts_items(REGIONS[:2]), xlim=(33, 36), ylim=(0, 30))
    assert {(ax.get_xlim(), ax.get_ylim()) for ax in _panel_axes(fig)} == {
        ((33.0, 36.0), (0.0, 30.0))
    }


def test_figsize_zoom_and_save(tmp_path):
    out = tmp_path / "deep" / "xy.png"
    fig = _draw(_ts_items(REGIONS[:2]), figsize=(7, 4), save=out)
    assert tuple(fig.get_size_inches()) == (7.0, 4.0)
    assert out.exists() and out.stat().st_size > 0
    small = _draw(_ts_items(REGIONS[:2]), size="column")
    assert small.get_size_inches()[0] == pytest.approx(3.5)


def test_font_scale_enlarges_the_type():
    base = _panel_axes(_draw(_ts_items()))[0].title.get_fontsize()
    bigger = _panel_axes(_draw(_ts_items(), font_scale=1.5))[0].title.get_fontsize()
    assert bigger > base


def test_title_tick_and_suptitle_kwargs_reach_their_text():
    fig = _draw(
        _ts_items(),
        title="T-S",
        title_kwargs={"fontsize": 17, "color": "tab:red"},
        suptitle_kwargs={"fontsize": 23},
        tick_label_kwargs={"color": "tab:green"},
    )
    (ax,) = _panel_axes(fig)
    assert ax.title.get_fontsize() == 17 and ax.title.get_color() == "tab:red"
    assert fig._suptitle.get_fontsize() == 23
    assert {t.get_color() for t in ax.get_xticklabels()} == {"tab:green"}


# --- renderer: density ----------------------------------------------------------------


def test_density_draws_labelled_grey_contours_at_the_composed_levels():
    pytest.importorskip("gsw")
    fig = _draw(_ts_items(), density=True)
    (ax,) = _panel_axes(fig)
    (contours,) = _contours(ax)
    layout = _xy.compose(_ts_items(), density=True)
    assert list(contours.levels) == list(layout.panels[0].density.levels)
    assert 6 <= len(contours.levels) <= 14
    assert len(contours.labelTexts) > 0
    # axis limits stay the data's own: contours never stretch the panel
    assert ax.get_xlim() == pytest.approx(layout.panels[0].xlim)
    assert ax.get_ylim() == pytest.approx(layout.panels[0].ylim)
    # beneath everything else
    assert contours.get_zorder() < _scatter(ax)[0].get_zorder()


def test_density_with_temperature_on_x():
    pytest.importorskip("gsw")
    items = []
    for item in _ts_items():
        item = {
            **item,
            "x": item["y"],
            "y": item["x"],
            "x_role": "temperature",
            "y_role": "salinity",
            "x_name": "temperature",
            "y_name": "salinity",
            "x_standard_name": item["y_standard_name"],
            "y_standard_name": item["x_standard_name"],
            "x_units": item["y_units"],
            "y_units": item["x_units"],
        }
        items.append(item)
    fig = _draw(items, density=[25.0, 26.0, 27.0])
    (ax,) = _panel_axes(fig)
    (contours,) = _contours(ax)
    assert list(contours.levels) == [25.0, 26.0, 27.0]
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("temperature [degC]", "salinity")


def test_density_for_other_variables_is_a_clear_error():
    nutrients = [_item(x_role=None, y_role=None, x_name="phosphate", y_name="nitrate")]
    with pytest.raises(ValueError, match="salinity and the other temperature"):
        _draw(nutrients, density=True)
    # but the same plot without density draws
    fig = _draw(nutrients)
    (ax,) = _panel_axes(fig)
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("phosphate", "nitrate [degC]")


# --- renderer: color_by ---------------------------------------------------------------


def test_color_by_depth_colours_the_dots_on_one_shared_scale_with_one_bar():
    fig = _draw(_ts_items(REGIONS[:3]), color_by="depth")
    axes = _panel_axes(fig)
    clouds = [_scatter(ax)[0] for ax in axes]
    norms = {(c.norm.vmin, c.norm.vmax) for c in clouds}
    assert len(norms) == 1
    ((vmin, vmax),) = norms
    assert vmin < vmax and 5 <= vmin and vmax <= 1000
    assert all(c.get_array() is not None and len(c.get_array()) == 200 for c in clouds)
    assert len({c.get_cmap().name for c in clouds}) == 1
    (bar,) = _colorbars(fig)
    assert bar.get_ylabel() == "depth [m]"
    # shallow water at the top of the bar
    assert bar.yaxis_inverted()
    # lines keep their solid colour
    assert all(len(ax.lines) == 2 for ax in axes)


def test_color_by_time_formats_the_bar_as_dates():
    fig = _draw(_ts_items(), color_by="time")
    (bar,) = _colorbars(fig)
    assert bar.get_ylabel() == "time"
    labels = [t.get_text() for t in bar.get_yticklabels()]
    assert any(label for label in labels)
    assert not any(
        label.replace(".", "").isdigit() and len(label) > 4 for label in labels
    )  # not raw day counts


def test_colorbar_false_colours_without_a_bar():
    fig = _draw(_ts_items(), color_by="depth", colorbar=False)
    assert _colorbars(fig) == []
    (ax,) = _panel_axes(fig)
    assert _scatter(ax)[0].get_array() is not None


def test_color_by_cmap_override():
    (ax,) = _panel_axes(_draw(_ts_items(), color_by="depth", cmap="magma"))
    assert _scatter(ax)[0].get_cmap().name == "magma"


def test_colorbar_kwargs_style_the_bar():
    fig = _draw(
        _ts_items(),
        color_by="depth",
        colorbar_kwargs={"label_size": 21, "orientation": "horizontal"},
    )
    (bar,) = _colorbars(fig)
    assert bar.xaxis.label.get_fontsize() == 21
    assert bar.get_xlabel() == "depth [m]"


def test_color_by_without_depth_warns_at_draw_time_too():
    items = [_item("ROMS", depth=False)]
    with pytest.warns(UserWarning, match="carries no depth values"):
        fig = _draw(items, color_by="depth")
    assert _colorbars(fig) == []
    (cloud,) = _scatter(_panel_axes(fig)[0])
    assert cloud.get_array() is None


# --- renderer: annotations ------------------------------------------------------------


def test_annotations_are_centred_text_at_data_positions():
    fig = _draw(
        _ts_items(REGIONS[:2]),
        annotations={
            REGIONS[0]: {"STSW": (34.8, 12.0)},
            REGIONS[1]: {"Northern\nsurface\nwaters": (34.0, 11.0)},
        },
        annot_kwargs={"color": "tab:red", "fontsize": 13},
    )
    first, second = _panel_axes(fig)
    (text,) = [t for t in first.texts if t.get_text() == "STSW"]
    assert text.get_position() == (34.8, 12.0)
    assert (text.get_ha(), text.get_va(), text.get_color(), text.get_fontsize()) == (
        "center",
        "center",
        "tab:red",
        13,
    )
    assert [t.get_text() for t in second.texts] == ["Northern\nsurface\nwaters"]
    assert all(t.get_text() != "STSW" for t in second.texts)


def test_annotation_beyond_the_data_is_inside_the_axes():
    item = _item()
    far = (item["x"].max() + 2.0, item["y"].max() + 4.0)
    (ax,) = _panel_axes(_draw([item], annotations={"far": far}))
    assert ax.get_xlim()[1] > far[0] and ax.get_ylim()[1] > far[1]


def test_annotation_errors_surface_through_the_renderer():
    with pytest.raises(ValueError, match="mixes the two forms"):
        _draw(_ts_items(), annotations={"A": (1, 2), REGIONS[0]: {"B": (1, 2)}})
    with pytest.raises(ValueError, match="no regions"):
        _draw([_item(region=None)], annotations={"R": {"B": (1, 2)}})


# --- renderer: legend -----------------------------------------------------------------


def test_single_panel_legend_uses_proxy_handles():
    (ax,) = _panel_axes(_draw(_ts_items()))
    legend = ax.get_legend()
    assert [t.get_text() for t in legend.get_texts()] == ["ROMS", "WOA23", "GLORYS12"]
    handles = legend.legend_handles
    assert all(isinstance(h, Line2D) for h in handles)
    dot, *lines = handles
    assert dot.get_linestyle() == "None" and dot.get_marker() == "o"
    assert dot.get_markersize() > 3  # a real swatch, not the plotted s=2 speck
    assert [line.get_linestyle() for line in lines] == ["-", "-"]
    assert all(line.get_marker() in ("None", None, "") for line in lines)


def test_legend_swatch_colours_match_the_members():
    (ax,) = _panel_axes(
        _draw(
            _ts_items(),
            colors={"ROMS": "black", "WOA23": "tab:red", "GLORYS12": "tab:blue"},
        )
    )
    import matplotlib.colors as mcolors

    colors = [h.get_color() for h in ax.get_legend().legend_handles]
    assert all(
        mcolors.same_color(c, w)
        for c, w in zip(colors, ("black", "tab:red", "tab:blue"), strict=True)
    )


def test_several_panels_share_one_key_below_the_figure():
    fig = _draw(_ts_items(REGIONS[:3]))
    assert all(ax.get_legend() is None for ax in _panel_axes(fig))
    (legend,) = fig.legends
    assert [t.get_text() for t in legend.get_texts()] == ["ROMS", "WOA23", "GLORYS12"]


def test_legend_off_and_forced_corner():
    (ax,) = _panel_axes(_draw(_ts_items(), legend=False))
    assert ax.get_legend() is None
    fig = _draw(_ts_items(REGIONS[:2]), legend="lower right")
    assert fig.legends == []
    assert all(ax.get_legend() is not None for ax in _panel_axes(fig))
    assert all(ax.get_legend()._loc == 4 for ax in _panel_axes(fig))  # lower right


def test_legend_kwargs_reach_the_key():
    (ax,) = _panel_axes(
        _draw(_ts_items(), legend_kwargs={"frameon": True, "title": "sources"})
    )
    assert ax.get_legend().get_frame_on()
    assert ax.get_legend().get_title().get_text() == "sources"


def test_two_dot_members_get_different_markers_in_the_key():
    items = [_item("A"), _item("B"), _item("C", mark="line")]
    (ax,) = _panel_axes(_draw(items))
    handles = ax.get_legend().legend_handles
    assert [h.get_marker() for h in handles[:2]] == [
        _style.MARKERS[0],
        _style.MARKERS[1],
    ]
    clouds = _scatter(ax)
    assert len(clouds) == 2


# --- renderer: options and warnings ---------------------------------------------------


def test_unknown_keyword_is_a_type_error_naming_xy_plot():
    with pytest.raises(TypeError, match=r"XY\.plot\(\) got 1 unusable option"):
        _draw(_ts_items(), bogus=1)
    with pytest.raises(TypeError, match=r"'bogus' is not an option of XY\.plot\(\)"):
        _draw(_ts_items(), bogus=1)


def test_a_misplaced_nested_keyword_is_redirected():
    with pytest.raises(TypeError, match="'label_size' goes inside colorbar_kwargs"):
        _draw(_ts_items(), label_size=9)


def test_error_lists_the_accepted_options():
    with pytest.raises(TypeError) as err:
        _draw(_ts_items(), bogus=1)
    text = str(err.value)
    for name in (
        "annotations",
        "density",
        "color_by",
        "annot_kwargs",
        "colorbar",
        "marker_size",
    ):
        assert name in text


def test_options_are_registered_as_top_level():
    from ocean_skill.plot.matplotlib_renderer import _nested_owner, _top_level_options

    for name in (
        "annotations",
        "density",
        "color_by",
        "cmap",
        "colorbar",
        "annot_kwargs",
        "marker_size",
        "alpha",
    ):
        assert name in _top_level_options(), name
        assert _nested_owner(name) is None, name


def test_contract_two_options_are_all_accepted():
    import inspect

    from ocean_skill.plot.matplotlib_renderer import xy

    expected = {
        "title", "annotations", "density", "color_by", "cmap", "colorbar", "colors",
        "legend", "titles", "xlim", "ylim", "sharex", "sharey", "marker_size", "alpha",
        "panel_aspect", "ncols", "nrows", "size", "zoom", "font_scale", "figsize",
        "save", "fit_text", "wspace", "hspace", "title_kwargs", "tick_label_kwargs",
        "suptitle_kwargs", "legend_kwargs", "line_kwargs", "annot_kwargs",
        "colorbar_kwargs",
    }  # fmt: skip
    assert set(inspect.signature(xy).parameters) - {"items"} == expected


def test_defaults_match_the_contract():
    import inspect

    from ocean_skill.plot.matplotlib_renderer import xy

    p = inspect.signature(xy).parameters
    assert (p["density"].default, p["color_by"].default, p["legend"].default) == (
        False,
        None,
        True,
    )
    assert (p["sharex"].default, p["sharey"].default) == (False, False)
    assert (p["marker_size"].default, p["alpha"].default) == (2.0, 0.5)


def test_warnings_surface_once_from_the_renderer(monkeypatch):
    monkeypatch.setattr(_xy, "POINT_CAP", 100)
    items = [
        _item("ROMS", y_standard_name="sea_water_potential_temperature"),
        _item("WOA23", mark="line", y_standard_name="sea_water_temperature"),
    ]
    with pytest.warns(UserWarning) as caught:
        _draw(items)
    messages = [str(w.message) for w in caught]
    assert sum("holds 200 points" in m for m in messages) == 1
    assert sum("disagree on what the y axis is" in m for m in messages) == 1


def test_generic_and_practical_salinity_are_not_a_disagreement():
    # GLORYS and WOA label practical salinity with CF's generic name; a T-S diagram
    # mixing them with ROMS's practical salinity must not warn about the x axis.
    items = [
        _item("ROMS", x_standard_name="sea_water_practical_salinity"),
        _item("GLORYS12", mark="line", x_standard_name="sea_water_salinity"),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _xy.compose(items)


def test_temperature_disagreement_names_the_temperature_offset():
    items = [
        _item("ROMS", y_standard_name="sea_water_potential_temperature"),
        _item("WOA23", mark="line", y_standard_name="sea_water_temperature"),
    ]
    with pytest.warns(UserWarning, match="in situ vs potential temperature"):
        _xy.compose(items)


def test_warning_points_at_the_callers_file():
    items = [
        _item("ROMS", y_standard_name="sea_water_potential_temperature"),
        _item("WOA23", mark="line", y_standard_name="sea_water_temperature"),
    ]
    with pytest.warns(UserWarning) as caught:
        _draw(items)
    assert all(w.filename.endswith("test_xy_renderers.py") for w in caught)


def test_spec_accepts_the_xy_family():
    from ocean_skill.plot.spec import FAMILIES

    assert "XY" in FAMILIES
    spec = PlotSpec("XY", _ts_items(), {"density": False})
    assert spec.family == "XY" and len(spec.items) == 3


def test_large_clouds_draw():
    big = _item(n=60_000)
    (ax,) = _panel_axes(_draw([big]))
    assert len(_scatter(ax)[0].get_offsets()) == 60_000


def test_a_ragged_shared_x_grid_keeps_tick_numbers_on_the_panel_above_a_gap():
    fig = _draw(_ts_items(REGIONS[:4]), sharex=True)
    fig.canvas.draw()
    shown = [
        any(t.get_visible() and t.get_text() for t in ax.get_xticklabels())
        for ax in _panel_axes(fig)
    ]
    # the first panel has a panel below it, so sharex hides its numbers; the other
    # three are the bottom of their column and keep them
    assert shown == [False, True, True, True]


def test_an_explicit_title_size_survives_fitting():
    long = "A region name far too long for a panel of this narrow width " * 2
    fit = _draw(_ts_items(REGIONS[:3]), titles=[long, None, None], figsize=(5, 3))
    pinned = _draw(
        _ts_items(REGIONS[:3]),
        titles=[long, None, None],
        figsize=(5, 3),
        title_kwargs={"fontsize": 20},
    )
    assert _panel_axes(fit)[0].title.get_fontsize() < 20  # shrunk to fit
    assert _panel_axes(pinned)[0].title.get_fontsize() == 20  # the caller's word


def test_missing_gsw_is_a_clear_error(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "gsw", None)
    with pytest.raises(ImportError, match=r"density=.*gsw.*density=False"):
        _draw(_ts_items(), density=True)
    # and nothing needs it when density is off
    _draw(_ts_items())
