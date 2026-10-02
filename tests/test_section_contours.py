"""Contour overlay and filled-band decisions for sections (ocean_skill.plot.section).

A section drawn as smooth filled bands with black contour lines of a second variable
on top is decided in one shared module so both renderers draw identical lines and
bands: :func:`prepare_overlay` (one mesh, or a refusal), :func:`contour_levels` (which
lines, once per figure), :func:`contour_paths` (the lines, for bokeh) and
:func:`fill_edges` (where the bands break). Everything here is pure geometry on small
synthetic arrays; ``contour_paths`` is checked against matplotlib's own ``ax.contour``,
the engine the static renderer calls.
"""

from __future__ import annotations

import dataclasses

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from ocean_skill.align import ALONG_DIM
from ocean_skill.plot import section as section_module
from ocean_skill.plot._colorbar import colorbar_ticks
from ocean_skill.plot.section import (
    DEFAULT_FILL_BANDS,
    ContourLevel,
    contour_levels,
    contour_paths,
    fill_edges,
    prepare_overlay,
    prepare_section,
)

# -- fixtures -----------------------------------------------------------------------


def _raw_section(
    *,
    n_z: int = 6,
    n_along: int = 8,
    depth_max: float = 200.0,
    length: float = 150.0,
    slope: float = 0.1,
    dims: tuple[str, str] = ("z", ALONG_DIM),
) -> xr.DataArray:
    """Build a raw fixed-depth section: negative-down ``z``, ``along`` in km.

    ``slope`` only changes the values (a second "variable" on the same mesh).
    """
    z = -np.linspace(0.0, depth_max, n_z)
    along = np.linspace(0.0, length, n_along)
    values = 25.0 + slope * z[:, None] + 0.02 * along[None, :]
    da = xr.DataArray(
        values,
        dims=("z", ALONG_DIM),
        coords={
            "z": z,
            ALONG_DIM: along,
            "lon": (ALONG_DIM, np.linspace(-95.0, -93.0, n_along)),
            "lat": (ALONG_DIM, np.linspace(24.0, 26.0, n_along)),
        },
    )
    da[ALONG_DIM].attrs["units"] = "km"
    return da.transpose(*dims)


def _raw_native(
    *, n_s: int = 6, n_along: int = 5, bathymetry: float = 1.0, nan_column: bool = False
) -> xr.DataArray:
    """Build a raw native-s section: ``s_rho`` with a 2-D ``z_rho`` (negative-down)."""
    sigma = np.linspace(-0.95, -0.05, n_s)
    z_rho = np.outer(sigma, np.linspace(30.0, 500.0, n_along) * bathymetry)
    if nan_column:
        z_rho[:, 0] = np.nan
    da = xr.DataArray(
        20.0 + 0.01 * z_rho,
        dims=("s_rho", ALONG_DIM),
        coords={
            "z_rho": (("s_rho", ALONG_DIM), z_rho),
            ALONG_DIM: np.linspace(0.0, 100.0, n_along),
            "lon": (ALONG_DIM, np.linspace(-95.0, -93.0, n_along)),
            "lat": (ALONG_DIM, np.linspace(24.0, 26.0, n_along)),
        },
    )
    da[ALONG_DIM].attrs["units"] = "km"
    return da


@pytest.fixture
def panel() -> xr.DataArray:
    """Return the fill's prepared values: what ``prepare_overlay`` gets as ``panel``."""
    return prepare_section(_raw_section())[0]


# -- prepare_overlay ----------------------------------------------------------------


def test_prepare_overlay_accepts_the_same_grid_and_hands_back_the_panels_mesh(panel):
    overlay = _raw_section(slope=0.3)
    out = prepare_overlay(overlay, panel)

    assert out.dims == panel.dims
    np.testing.assert_array_equal(out["distance"], panel["distance"])
    np.testing.assert_array_equal(out["depth"], panel["depth"])
    # the values are the overlay's own, untouched by being put on the panel's mesh
    np.testing.assert_array_equal(out.values, prepare_section(overlay)[0].values)
    assert out["depth"].attrs["units"] == "m"
    assert out["distance"].attrs["units"] == "km"
    assert float(out["depth"].max()) == 200.0  # positive-down, as the panel's


def test_prepare_overlay_puts_a_transposed_overlay_in_the_panels_dim_order(panel):
    overlay = _raw_section(slope=0.3, dims=(ALONG_DIM, "z"))
    out = prepare_overlay(overlay, panel)

    assert out.dims == panel.dims == ("z", ALONG_DIM)
    np.testing.assert_array_equal(
        out.values, _raw_section(slope=0.3).transpose("z", ALONG_DIM).values
    )


def test_prepare_overlay_accepts_native_s_levels_and_equal_nan_depths():
    # a NaN depth column on both sides is "the same mesh": NaN compares equal
    panel_native = prepare_section(_raw_native(nan_column=True))[0]
    overlay = _raw_native(nan_column=True)
    overlay.values[:] = 7.0  # a different variable on the same mesh

    out = prepare_overlay(overlay, panel_native)

    assert out.dims == panel_native.dims == ("s_rho", ALONG_DIM)
    assert bool(np.isnan(np.asarray(out["depth"])[:, 0]).all())


def test_prepare_overlay_refuses_a_native_s_overlay_on_other_bathymetry():
    panel_native = prepare_section(_raw_native())[0]
    with pytest.raises(ValueError, match="depth levels differ") as err:
        prepare_overlay(_raw_native(bathymetry=1.2), panel_native)
    # the same number of levels, but over a deeper range: both sides are spelled out
    assert "overlay 6 levels 1.8–570 m, panel 6 levels 1.5–475 m" in str(err.value)


def test_prepare_overlay_ignores_where_the_two_variables_are_missing(panel):
    overlay = _raw_section(slope=0.3)
    overlay[2:4, 1:3] = np.nan  # a variable can be missing where the panel's is not

    out = prepare_overlay(overlay, panel)

    assert int(np.isnan(out.values).sum()) == 4


def test_prepare_overlay_forgives_round_off_but_not_a_shifted_level(panel):
    nudged = _raw_section().assign_coords(z=lambda d: d["z"] * (1 + 1e-9))
    out = prepare_overlay(nudged, panel)
    # the panel's own arrays, not the overlay's near-identical ones
    np.testing.assert_array_equal(out["depth"], panel["depth"])

    shifted = _raw_section().assign_coords(z=lambda d: d["z"] - 0.5)
    with pytest.raises(ValueError, match="depth levels differ"):
        prepare_overlay(shifted, panel)


def test_prepare_overlay_names_a_differing_depth_list(panel):
    with pytest.raises(ValueError) as err:
        prepare_overlay(_raw_section(n_z=9), panel)
    text = str(err.value)
    assert (
        "depth levels differ (overlay 9 levels 0–200 m, panel 6 levels 0–200 m)" in text
    )
    assert "positions" not in text  # only the axis that differs is named
    # and how to fix it, and that nothing was done behind the caller's back
    assert "same select/aggregate" in text
    assert "one section grid" in text
    assert "nothing is regridded" in text


def test_prepare_overlay_names_differing_positions_along_the_section(panel):
    with pytest.raises(ValueError) as err:
        prepare_overlay(_raw_section(n_along=5), panel)
    text = str(err.value)
    assert (
        "positions along the section differ "
        "(overlay 5 positions 0–150 km, panel 8 positions 0–150 km)"
    ) in text
    assert "depth levels" not in text

    with pytest.raises(ValueError) as err:
        prepare_overlay(_raw_section(length=140.0), panel)  # same count, shorter
    assert "overlay 8 positions 0–140 km, panel 8 positions 0–150 km" in str(err.value)


def test_prepare_overlay_names_both_axes_when_both_differ(panel):
    with pytest.raises(ValueError) as err:
        prepare_overlay(_raw_section(n_z=9, n_along=5), panel)
    text = str(err.value)
    assert "depth levels differ (overlay 9 levels" in text
    assert "positions along the section differ (overlay 5 positions" in text


def test_prepare_overlay_says_when_levels_share_a_count_and_range_but_not_depths(panel):
    uneven = _raw_section().assign_coords(z=-np.array([0, 10, 25, 50, 100, 200.0]))
    with pytest.raises(ValueError) as err:
        prepare_overlay(uneven, panel)
    assert (
        "overlay 6 levels 0–200 m, panel 6 levels 0–200 m -- the same count and range, "
        "but not at the same depths"
    ) in str(err.value)


def test_prepare_overlay_tells_a_latitude_axis_from_a_distance_axis(panel):
    # numbers that coincide with the panel's km, but they are degrees of latitude
    overlay = _raw_section()
    overlay = overlay.assign_coords(lat=(ALONG_DIM, overlay[ALONG_DIM].values))
    overlay[ALONG_DIM].attrs["axis_coord"] = "lat"

    with pytest.raises(ValueError, match="positions along the section differ") as err:
        prepare_overlay(overlay, panel)
    assert "°N" in str(err.value)
    assert "150 km" in str(err.value)


def test_prepare_overlay_refuses_two_kinds_of_vertical_axis(panel):
    with pytest.raises(ValueError, match="vertical axes differ") as err:
        prepare_overlay(_raw_native(), panel)
    assert "'s_rho'" in str(err.value)
    assert "'z'" in str(err.value)


def test_prepare_overlay_wants_a_prepared_panel():
    with pytest.raises(ValueError, match="prepare_section's own return"):
        prepare_overlay(_raw_section(), _raw_section())


# -- contour_levels -----------------------------------------------------------------


def test_true_gives_about_six_round_levels_strictly_inside_the_range():
    assert contour_levels(True, [np.linspace(4.3, 29.87, 50)]) == (
        5.0,
        10.0,
        15.0,
        20.0,
        25.0,
    )
    # a level at the data's own minimum or maximum would be a scrap of the edge
    edge_data = [np.linspace(5.0, 30.0, 50)]
    assert contour_levels(True, edge_data) == (10.0, 15.0, 20.0, 25.0)
    levels = contour_levels(True, [np.linspace(33.81, 36.42, 50)])
    assert levels == (34.0, 34.5, 35.0, 35.5, 36.0)


def test_levels_are_free_of_float_noise():
    levels = contour_levels(True, [np.linspace(0.0, 1.0, 11)])
    assert levels == (0.2, 0.4, 0.6, 0.8)  # not 0.6000000000000001


def test_an_int_asks_for_about_that_many_and_never_more():
    data = [np.linspace(0.0, 30.0, 50)]
    assert contour_levels(3, data) == (10.0, 20.0)
    counts = [len(contour_levels(n, data)) for n in (2, 5, 12, 30)]
    assert all(c <= n for c, n in zip(counts, (2, 5, 12, 30), strict=True))
    assert counts == sorted(counts)  # more asked for, more (or as many) drawn
    assert len(contour_levels(30, data)) == 29  # every integer strictly inside


def test_levels_pool_every_array_so_each_panel_gets_the_same_lines():
    low = np.linspace(0.0, 10.0, 30)
    high = np.linspace(20.0, 30.0, 30)
    pooled = contour_levels(True, [low, high])

    assert pooled == (5.0, 10.0, 15.0, 20.0, 25.0)
    assert pooled == contour_levels(True, [np.concatenate([low, high])])
    # deciding per panel instead would draw different lines in each
    assert contour_levels(True, [low]) != pooled
    assert contour_levels(True, [high]) != pooled


def test_levels_take_dataarrays_one_array_or_a_generator_alike():
    data = np.linspace(4.3, 29.87, 50)
    expected = contour_levels(True, [data])
    assert contour_levels(True, data) == expected
    assert contour_levels(True, xr.DataArray(data)) == expected
    assert contour_levels(True, [xr.DataArray(data)]) == expected
    assert contour_levels(True, (d for d in [data])) == expected
    assert contour_levels(True, [data, np.array([np.nan, np.nan])]) == expected


def test_levels_ignore_nan_and_masked_entries():
    data = np.ma.masked_array([1.0, 2.0, 99.0, 4.0], mask=[0, 0, 1, 0])
    assert max(contour_levels(True, [data])) < 4.0  # the masked 99 is not data
    assert contour_levels(True, [np.array([1.0, np.nan, 4.0, np.inf])]) == (
        contour_levels(True, [np.array([1.0, 4.0])])
    )


def test_an_explicit_list_is_sorted_unique_and_used_as_given():
    assert contour_levels([3, 1, 2, 2.0], None) == (1.0, 2.0, 3.0)
    assert contour_levels((10, 5), [np.arange(3.0)]) == (5.0, 10.0)  # outside the data
    assert contour_levels(np.array([2.5, 0.5]), []) == (0.5, 2.5)
    assert all(type(v) is float for v in contour_levels([1, 2], None))
    assert contour_levels([], [np.arange(3.0)]) == ()


@pytest.mark.parametrize("spec", [False, None, np.False_])
def test_false_and_none_draw_no_lines(spec):
    assert contour_levels(spec, [np.linspace(0.0, 30.0, 50)]) == ()


def test_numpy_bools_mean_what_python_bools_do():
    data = [np.linspace(0.0, 30.0, 50)]
    assert contour_levels(np.True_, data) == contour_levels(True, data)


@pytest.mark.parametrize("spec", [True, 6, 3])
def test_no_finite_data_is_no_levels_and_no_error(spec):
    assert contour_levels(spec, [np.full(5, np.nan)]) == ()
    assert contour_levels(spec, []) == ()
    assert contour_levels(spec, None) == ()
    assert contour_levels(spec, [np.full(3, np.nan), np.array([])]) == ()


def test_a_flat_field_has_nothing_to_contour():
    assert contour_levels(True, [np.full(6, 12.0)]) == ()


def test_an_explicit_list_ignores_whether_there_is_data():
    assert contour_levels([1, 2], [np.full(3, np.nan)]) == (1.0, 2.0)


@pytest.mark.parametrize(
    ("spec", "error"),
    [
        ("auto", TypeError),
        ("6", TypeError),
        (2.5, TypeError),
        (6.0, TypeError),
        ({"a": 1}, TypeError),
        (object(), TypeError),
        ([1, "a"], TypeError),
        (["1", "2"], TypeError),
        ([None, 1.0], TypeError),
        ([[1, 2], [3, 4]], TypeError),
        ([[1, 2], [3]], TypeError),
        ([True, False], TypeError),
        (np.array(["a", "b"]), TypeError),
        (0, ValueError),
        (-3, ValueError),
        ([np.nan, 1.0], ValueError),
        ([1.0, np.inf], ValueError),
    ],
)
def test_bad_contour_specs_raise_and_say_what_is_accepted(spec, error):
    with pytest.raises(error, match="Accepted") as err:
        contour_levels(spec, [np.arange(5.0)])
    text = str(err.value)
    assert "True" in text
    assert "int n >= 1" in text
    assert "list/tuple/array" in text
    assert "False/None" in text


def test_a_bad_spec_raises_even_when_there_is_no_data_to_draw():
    with pytest.raises(TypeError, match="Accepted"):
        contour_levels("auto", [np.full(3, np.nan)])


# -- contour_paths: against matplotlib's own ax.contour -----------------------------


def _mpl_lines(x, y, z, levels):
    """Return ``ax.contour``'s segments, one list of ``(N, 2)`` arrays per level."""
    fig, ax = plt.subplots()
    try:
        cs = ax.contour(x, y, z, levels=list(levels))
        return [[np.asarray(seg) for seg in segs] for segs in cs.allsegs]
    finally:
        plt.close(fig)


def _assert_matches_matplotlib(x, y, z, levels):
    """Assert ``contour_paths`` draws ``ax.contour``'s lines: counts and vertices."""
    ours = contour_paths(x, y, z, levels)
    theirs = _mpl_lines(x, y, z, levels)
    assert len(ours) == len(theirs) == len(levels)
    for entry, segs in zip(ours, theirs, strict=True):
        assert len(entry.lines) == len(segs)
        for line, seg in zip(entry.lines, segs, strict=True):
            assert line.shape == seg.shape
            np.testing.assert_allclose(line, seg, rtol=1e-9, atol=1e-9)
    return ours


@pytest.fixture
def mesh():
    """Return a plain rectilinear mesh: 21 columns of distance, 17 rows of depth."""
    return np.meshgrid(np.linspace(0.0, 100.0, 21), np.linspace(0.0, 500.0, 17))


def test_paths_match_matplotlib_on_a_simple_field(mesh):
    x, y = mesh
    z = 20.0 - 0.03 * y + 3.0 * np.sin(x / 15.0)

    ours = _assert_matches_matplotlib(x, y, z, (10.0, 12.5, 15.0))

    assert [len(entry.lines) for entry in ours] == [1, 1, 1]
    assert [entry.level for entry in ours] == [10.0, 12.5, 15.0]
    assert all(line.shape[1] == 2 for entry in ours for line in entry.lines)


def test_paths_match_matplotlib_for_closed_loops(mesh):
    x, y = mesh
    z = np.hypot(x - 50.0, (y - 250.0) / 5.0)  # rings around the middle

    ours = _assert_matches_matplotlib(x, y, z, (10.0, 20.0, 30.0))

    for entry in ours:
        (ring,) = entry.lines
        np.testing.assert_allclose(ring[0], ring[-1])  # closed: first vertex last


def test_paths_match_matplotlib_when_the_field_is_nan_masked(mesh):
    x, y = mesh
    z = 20.0 - 0.03 * y + 3.0 * np.sin(x / 15.0)
    z[10:14, 5:9] = np.nan  # a hole in the middle (a seamount)
    z[:3, :] = np.nan  # and no data at the top three rows

    ours = _assert_matches_matplotlib(x, y, z, (10.0, 12.5, 15.0))

    # the hole breaks lines into pieces; no vertex is ever NaN
    assert any(len(entry.lines) > 1 for entry in ours)
    assert all(np.isfinite(line).all() for entry in ours for line in entry.lines)


def test_paths_match_matplotlib_with_depth_that_varies_along_the_path():
    # native s-levels: y (depth) is honestly 2-D, deeper along the path
    sigma = np.linspace(-1.0, 0.0, 12)[:, None]
    bathymetry = np.linspace(100.0, 500.0, 25)[None, :]
    y = -sigma * bathymetry
    x = np.broadcast_to(np.linspace(0.0, 300.0, 25)[None, :], y.shape).copy()
    z = 20.0 - 0.01 * y + 0.0001 * x

    ours = _assert_matches_matplotlib(x, y, z, (16.0, 17.0, 18.0, 19.0))

    assert all(len(entry.lines) >= 1 for entry in ours)


def test_a_nan_coordinate_masks_its_point_rather_than_putting_nan_on_a_line(mesh):
    x, y = mesh
    z = 20.0 - 0.03 * y  # the 11.0 line runs level across the section
    x_bad = x.copy()
    x_bad[9, 10] = np.nan  # z is finite there; the vertex would be NaN without a mask

    (whole,) = contour_paths(x, y, z, [11.0])[0].lines
    ours = contour_paths(x_bad, y, z, [11.0])

    assert whole.shape == (21, 2)
    assert all(np.isfinite(line).all() for line in ours[0].lines)
    # the same lines matplotlib draws once that point is masked in z: the line breaks
    z_masked = z.copy()
    z_masked[9, 10] = np.nan
    expected = _mpl_lines(x, y, z_masked, [11.0])[0]
    assert len(ours[0].lines) == len(expected) == 2
    for line, seg in zip(ours[0].lines, expected, strict=True):
        np.testing.assert_allclose(line, seg)


def test_paths_take_dataarrays_straight_from_a_prepared_section(panel):
    overlay = prepare_overlay(_raw_section(slope=0.3), panel)
    levels = contour_levels(True, [overlay])

    ours = _assert_matches_matplotlib(
        overlay["distance"], overlay["depth"], overlay, levels
    )

    assert levels and all(len(entry.lines) >= 1 for entry in ours)


def test_the_anchor_is_the_middle_vertex_of_the_line_it_labels(mesh):
    x, y = mesh
    (entry,) = contour_paths(x, y, y, [260.0])  # one level line across the section

    (line,) = entry.lines
    assert line.shape == (21, 2)
    assert entry.anchor == (50.0, 260.0)
    assert any(np.allclose(entry.anchor, vertex) for vertex in line)  # on the line


def test_the_anchor_sits_on_the_longest_of_several_pieces(mesh):
    x, y = mesh
    z = y.copy()
    z[:, 14] = np.nan  # no data in one column: the line breaks into two pieces

    (entry,) = contour_paths(x, y, z, [260.0])

    lengths = sorted(len(line) for line in entry.lines)
    assert len(entry.lines) == 2 and lengths == [6, 14]
    longest = max(entry.lines, key=len)
    assert any(np.allclose(entry.anchor, vertex) for vertex in longest)
    assert entry.anchor[0] < 70.0  # the left piece, not the stub at the right


def test_the_anchor_is_the_middle_of_the_panel_not_of_the_raw_numbers():
    # level across the section, then a steep rise at the far end -- in km against m the
    # rise dominates raw length and would drag the label to it
    x, y = np.meshgrid(np.linspace(0.0, 60.0, 31), np.linspace(0.0, 1000.0, 21))
    z = y - 300.0 * (x / 60.0) ** 8

    (entry,) = contour_paths(x, y, z, [0.5])

    (line,) = entry.lines
    raw = np.concatenate(
        [[0.0], np.cumsum(np.hypot(np.diff(line[:, 0]), np.diff(line[:, 1])))]
    )
    raw_middle = line[np.argmin(np.abs(raw - raw[-1] / 2))]
    assert raw_middle[0] > 45.0  # what measuring in raw units would pick
    assert 20.0 <= entry.anchor[0] <= 40.0  # the middle of the drawn panel


def test_a_level_that_crosses_nothing_still_gets_an_entry_with_no_anchor(mesh):
    x, y = mesh
    z = 20.0 - 0.03 * y  # runs from 5 to 20

    out = contour_paths(x, y, z, [12.5, 99.0, -4.0])

    assert [entry.level for entry in out] == [12.5, 99.0, -4.0]
    assert out[0].anchor is not None and len(out[0].lines) == 1
    for empty in out[1:]:
        assert empty.lines == ()
        assert empty.anchor is None


def test_levels_come_back_in_the_order_asked_as_floats(mesh):
    x, y = mesh
    out = contour_paths(x, y, 20.0 - 0.03 * y, np.array([15, 10, 12]))

    assert [entry.level for entry in out] == [15.0, 10.0, 12.0]
    assert all(type(entry.level) is float for entry in out)
    assert isinstance(out, list)
    assert contour_paths(x, y, y, []) == []  # no levels asked, none returned


def test_a_field_with_nothing_to_contour_has_empty_entries_not_errors(mesh):
    x, y = mesh
    all_nan = np.full(x.shape, np.nan)
    for out in (
        contour_paths(x, y, all_nan, [1.0, 2.0]),
        contour_paths(x[:1], y[:1], y[:1], [1.0, 2.0]),  # one row: nothing to join
        contour_paths(x[:, :1], y[:, :1], y[:, :1], [1.0, 2.0]),  # one column
    ):
        assert [entry.level for entry in out] == [1.0, 2.0]
        assert all(entry.lines == () and entry.anchor is None for entry in out)


def test_paths_refuse_arrays_that_are_not_two_dimensional_and_alike(mesh):
    x, y = mesh
    with pytest.raises(ValueError, match="2-D arrays of one shape"):
        contour_paths(x, y[:, :-1], y, [1.0])
    with pytest.raises(ValueError, match="2-D arrays of one shape"):
        contour_paths(x[0], y[:, 0], y, [1.0])  # 1-D x and y


def test_a_contour_level_is_frozen():
    entry = ContourLevel(1.0, (), None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.level = 2.0


# -- fill_edges ---------------------------------------------------------------------

RANGES = [
    (0.0, 3.0),  # phosphate: the paper's bar
    (33.8, 36.5),  # salinity
    (-1.8, 1.8),  # a difference panel
    (4.0, 30.0),  # temperature
    (0.0, 52.0),
    (2200.0, 2400.0),  # alkalinity
    (0.0, 0.007),
    (0.0, 5000.0),
]


@pytest.mark.parametrize(("lo", "hi"), RANGES)
def test_fill_edges_are_round_and_every_colorbar_tick_sits_on_one(lo, hi):
    edges = fill_edges(lo, hi, log=False)
    ticks = colorbar_ticks(lo, hi, log=False).values

    assert ticks  # the range has ticks to check
    for tick in ticks:
        assert np.isclose(edges, tick, rtol=0.0, atol=1e-9 * (hi - lo)).any()
    assert edges[0] == pytest.approx(lo) and edges[-1] == pytest.approx(hi)
    assert (np.diff(edges) > 0).all()
    assert edges.dtype == np.float64
    np.testing.assert_array_equal(edges, np.round(edges, 8))  # no 0.30000000000000004


@pytest.mark.parametrize(("lo", "hi"), RANGES)
def test_the_band_count_lands_near_the_default_target(lo, hi):
    bands = len(fill_edges(lo, hi, log=False)) - 1
    # the round widths come in steps of at most 2.5x, so the nearest is within ~1.6x
    assert DEFAULT_FILL_BANDS / 1.8 <= bands <= DEFAULT_FILL_BANDS * 1.8


def test_fill_edges_for_the_papers_phosphate_bar():
    edges = fill_edges(0.0, 3.0, log=False)

    assert len(edges) - 1 == 60  # 0.05 wide: a tenth of the 0.5 tick step
    np.testing.assert_allclose(edges, np.linspace(0.0, 3.0, 61), atol=1e-12)


def test_an_int_is_the_target_band_count_and_a_tie_goes_to_the_finer_width():
    # bands available on a 0-3 bar (tick step 0.5): 6, 12, 30, 60, 120
    assert len(fill_edges(0.0, 3.0, log=False, fill_levels=1)) - 1 == 6  # coarsest
    assert len(fill_edges(0.0, 3.0, log=False, fill_levels=11)) - 1 == 12
    assert len(fill_edges(0.0, 3.0, log=False, fill_levels=40)) - 1 == 30
    assert len(fill_edges(0.0, 3.0, log=False, fill_levels=500)) - 1 == 120  # finest
    # 9 is as far from 6 as from 12: the finer one (0.25 wide) wins
    tied = fill_edges(0.0, 3.0, log=False, fill_levels=9)
    assert len(tied) - 1 == 12
    np.testing.assert_allclose(np.diff(tied), 0.25)
    assert len(fill_edges(0.0, 3.0, log=False, fill_levels=np.int64(11))) - 1 == 12


def test_a_pinned_non_round_end_is_an_edge_itself_with_round_ones_between():
    edges = fill_edges(0.37, 2.71, log=False)

    assert edges[0] == 0.37 and edges[-1] == 2.71  # exactly the colour range
    inner = edges[1:-1]  # multiples of one round width, however the ends fall
    width = np.diff(inner)[1:-1].mean()
    np.testing.assert_allclose(np.diff(inner)[1:-1], width)
    np.testing.assert_allclose(inner / width, np.round(inner / width), atol=1e-9)
    for tick in colorbar_ticks(0.37, 2.71, log=False).values:
        assert np.isclose(edges, tick).any()

    one_end = fill_edges(0.0, 2.71, log=False)
    assert one_end[0] == 0.0 and one_end[-1] == 2.71
    assert fill_edges(0.5, 3.0, log=False)[0] == 0.5  # a round one is not doubled
    assert len(np.unique(fill_edges(0.5, 3.0, log=False))) == len(
        fill_edges(0.5, 3.0, log=False)
    )


def test_a_list_of_edges_is_used_as_given_sorted_whatever_the_range():
    edges = fill_edges(0.0, 1.0, log=False, fill_levels=[0.5, 0.1, 0.1, 0.9])
    np.testing.assert_array_equal(edges, [0.1, 0.5, 0.9])
    assert edges.dtype == np.float64
    np.testing.assert_array_equal(
        fill_edges(0.0, 1.0, log=True, fill_levels=(3, 1)), [1.0, 3.0]
    )
    np.testing.assert_array_equal(
        fill_edges(np.nan, np.nan, log=False, fill_levels=np.array([2.0, 0.0, 1.0])),
        [0.0, 1.0, 2.0],
    )


def test_log_bands_are_geometric_as_before():
    edges = fill_edges(0.01, 10.0, log=True)
    np.testing.assert_allclose(edges, np.geomspace(0.01, 10.0, DEFAULT_FILL_BANDS + 1))

    assert len(fill_edges(0.01, 10.0, log=True, fill_levels=12)) == 13
    # a log scale needs a positive lower end: anything else bands linearly
    np.testing.assert_array_equal(
        fill_edges(0.0, 3.0, log=True), fill_edges(0.0, 3.0, log=False)
    )


@pytest.mark.parametrize(
    ("lo", "hi"),
    [
        (np.nan, 1.0),
        (1.0, np.nan),
        (-np.inf, np.inf),
        (None, None),
        (5.0, 5.0),
        (0.0, 0.0),
        (-3.0, -3.0),
        (3.0, 1.0),
    ],
)
@pytest.mark.parametrize("log", [False, True])
def test_a_range_with_nothing_to_band_still_gives_edges_contourf_accepts(lo, hi, log):
    edges = fill_edges(lo, hi, log=log)

    assert edges.ndim == 1 and len(edges) >= 2
    assert np.isfinite(edges).all() and (np.diff(edges) > 0).all()
    x, y = np.meshgrid(np.arange(4.0), np.arange(3.0))
    fig, ax = plt.subplots()
    try:
        for z in (np.full(x.shape, 5.0), np.full(x.shape, np.nan)):
            ax.contourf(x, y, z, levels=edges)  # raises on non-increasing levels
    finally:
        plt.close(fig)


def test_a_flat_range_is_bracketed_so_the_panel_draws_as_one_band():
    low, high = fill_edges(5.0, 5.0, log=False)
    assert low < 5.0 < high
    low, high = fill_edges(0.0, 0.0, log=False)
    assert low < 0.0 < high
    np.testing.assert_array_equal(fill_edges(np.nan, 1.0, log=False), [0.0, 1.0])


@pytest.mark.parametrize(
    ("spec", "error"),
    [
        ("auto", TypeError),
        (2.5, TypeError),
        (False, TypeError),
        ({"a": 1}, TypeError),
        ([1, "a"], TypeError),
        ([[1, 2], [3, 4]], TypeError),
        (0, ValueError),
        (-4, ValueError),
        ([np.nan, 1.0], ValueError),
        ([1.0], ValueError),  # one edge bounds no band
        ([2.0, 2.0], ValueError),  # two equal edges are one
        ([], ValueError),
    ],
)
def test_bad_fill_levels_raise_and_say_what_is_accepted(spec, error):
    with pytest.raises(error, match="Accepted"):
        fill_edges(0.0, 3.0, log=False, fill_levels=spec)


def test_true_means_the_default_band_count():
    np.testing.assert_array_equal(
        fill_edges(0.0, 3.0, log=False, fill_levels=True),
        fill_edges(0.0, 3.0, log=False),
    )


def test_fill_edges_and_overlay_lines_draw_together_through_matplotlib(panel):
    overlay = prepare_overlay(_raw_section(slope=0.3), panel)
    edges = fill_edges(float(panel.min()), float(panel.max()), log=False)
    levels = contour_levels(True, [overlay])

    fig, ax = plt.subplots()
    try:
        x, y = panel["distance"], panel["depth"]
        filled = ax.contourf(x, y, panel, levels=edges)
        lines = ax.contour(x, y, overlay, levels=levels)
        assert len(filled.levels) == len(edges)
        assert list(lines.levels) == list(levels)
    finally:
        plt.close(fig)


def test_the_module_exports_the_new_names():
    for name in (
        "contour_levels",
        "contour_paths",
        "fill_edges",
        "prepare_overlay",
        "ContourLevel",
        "DEFAULT_FILL_BANDS",
    ):
        assert name in section_module.__all__
        assert hasattr(section_module, name)


def test_a_difference_panel_keeps_a_band_count_but_not_absolute_edges():
    from ocean_skill.plot.section import difference_fill_levels

    assert difference_fill_levels(None) is None
    assert difference_fill_levels(30) == 30
    # edges in the variable's own units would leave a difference panel unfilled
    assert difference_fill_levels([10.0, 15.0, 20.0]) is None


def test_an_unknown_section_mark_is_refused_in_one_wording_for_both_renderers():
    from ocean_skill.plot.section import check_section_options

    with pytest.raises(ValueError, match="not a section mark"):
        check_section_options(has_contours=False, mark="line")
    check_section_options(has_contours=False, mark=None)  # the default: cells
