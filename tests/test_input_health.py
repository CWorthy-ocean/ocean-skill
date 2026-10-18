"""Input integrity: per-variable statistics of forcing, boundary and initial files.

Tiny synthetic roms-tools files on one 5 x 6 grid with two land cells, one of them on
the south boundary row, with known faults planted in them: a negative rain cell, a 75
degC air temperature, a NaN over the ocean (and NaNs and junk over land, which must not
count), a field that is zero everywhere, a constant one, a repeated step, a negative
nitrate on the south boundary with a ``DIC_ALT_CO2`` that copies ``DIC``. A restart of
the run-health fixture covers 3-D fields in blocks of levels and the staggered u and v.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import ocean_skill as osk
from ocean_skill import build, cache, catalog, input_integrity, roms, sources
from ocean_skill.config import INPUT_DRAWS
from ocean_skill.input_integrity import InputHealth, bounds, input_health
from tests.test_run_health import SUBCHUNK, write_run

NY, NX, NS = 5, 6, 3
LAND = [(1, 2), (0, 3)]  # (eta, xi); the second is on the south boundary row
EPOCH = "2011-01-01"
NT = 6
JUNK = 1.0e6  # what land cells hold


def _write_grid(path, mask: bool = True):
    lon = -20.0 + 0.5 * np.arange(NX)[None, :] + 0.01 * np.arange(NY)[:, None]
    lat = 60.0 + 0.5 * np.arange(NY)[:, None] + 0.01 * np.arange(NX)[None, :]
    sigma = (np.arange(NS) - NS + 0.5) / NS
    sigma_w = np.linspace(-1.0, 0.0, NS + 1)
    land = np.ones((NY, NX))
    for cell in LAND:
        land[cell] = 0.0
    rho = ("eta_rho", "xi_rho")
    data = {
        "lon_rho": (rho, lon),
        "lat_rho": (rho, lat),
        "h": (rho, np.full((NY, NX), 500.0)),
        "angle": (rho, np.zeros((NY, NX))),
        "sigma_r": ("s_rho", sigma),
        "Cs_r": ("s_rho", sigma**3),
        "sigma_w": ("s_w", sigma_w),
        "Cs_w": ("s_w", sigma_w**3),
    }
    if mask:
        data["mask_rho"] = (rho, land)
    xr.Dataset(data, attrs={"hc": 50.0, "theta_s": 5.0, "theta_b": 2.0}).to_netcdf(path)
    return lon, lat


def _days(n: int, step_hours: float = 1.0, start: float = 0.0):
    return xr.DataArray(
        start + np.arange(n) * step_hours / 24.0,
        dims="time",
        attrs={
            "long_name": f"relative time: days since {EPOCH} 00:00:00",
            "units": "days",
        },
    )


def _surface(path, *, land_junk: bool = True):
    """Surface physics: six hourly steps with the faults planted in it."""
    shape = (NT, NY, NX)
    t = np.arange(NT)[:, None, None]
    cell = np.arange(NY * NX).reshape(NY, NX)[None]
    f = {
        "Tair": 10.0 + 0.1 * t + 0.01 * cell,
        "qair": 0.01 + 0.0001 * t + 0.00001 * cell,
        "rain": 1e-4 + 1e-6 * t + 1e-7 * cell,
        "swrad": 200.0 + 3.0 * t + cell,
        "lwrad": np.zeros(shape),
        "vwnd": np.full(shape, 3.0),
        "uwnd": np.random.default_rng(1).normal(size=shape),
    }
    f["Tair"][2, 3, 4] = 75.0  # hotter than any air
    f["rain"][4, 0, 1] = -1e-3  # negative rain
    f["qair"][1, 2, 2] = np.nan  # a NaN over the ocean
    f["uwnd"][3] = f["uwnd"][2]  # a step written twice
    if land_junk:
        for cell_ in LAND:
            for name in f:
                f[name][(slice(None), *cell_)] = JUNK
            f["swrad"][(slice(None), *cell_)] = np.nan  # NaN over land: not a fault
            f["lwrad"][(slice(None), *cell_)] = 5.0
    dims = ("time", "eta_rho", "xi_rho")
    xr.Dataset(
        {
            name: (dims, v, {"units": unit})
            for (name, v), unit in zip(
                f.items(),
                ("degC", "kg/kg", "cm/day", "W/m2", "W/m2", "m/s", "m/s"),
                strict=True,
            )
        },
        coords={"time": _days(NT)},
    ).to_netcdf(path)


def _boundary(path, *, month: bool = False):
    """South and west sides of temp, salt, NO3, DIC and a DIC_ALT_CO2 copying DIC."""
    nt = 12 if month else 3
    days = np.arange(nt) * 30.4 if month else np.array([0.0, 1.0, 2.0])
    rng = np.random.default_rng(5)
    data = {}
    for side, n, axis in (("south", NX, "xi_rho"), ("west", NY, "eta_rho")):
        base = np.arange(n)[None, None, :] + np.arange(nt)[:, None, None] * 0.1
        level = 0.01 * np.arange(NS)[None, :, None]
        dims = ("bry_time", "s_rho", axis)
        dic = 2000.0 + 5.0 * base + level + rng.normal(size=(nt, NS, n))
        no3 = 5.0 + 0.5 * base + level
        if side == "south":
            no3[1, 1, 4] = -2.0  # time step 1, level 1, xi 4: ocean
            no3[:, :, 3] = -999.0  # the land cell on the south row
        data[f"temp_{side}"] = (dims, 10.0 + 0.1 * base + level, {"units": "degC"})
        data[f"salt_{side}"] = (dims, 34.0 + 0.1 * base + level, {"units": "PSU"})
        data[f"NO3_{side}"] = (dims, no3, {"units": "mmol/m3"})
        data[f"DIC_{side}"] = (dims, dic, {"units": "mmol/m3"})
        data[f"DIC_ALT_CO2_{side}"] = (dims, dic.copy(), {"units": "mmol/m3"})
    coords = {
        "bry_time": (
            "bry_time",
            days,
            {
                "long_name": f"relative time: days since {EPOCH} 00:00:00",
                "units": "days",
            },
        )
    }
    if month:
        coords["month"] = ("bry_time", np.arange(1, nt + 1))
    xr.Dataset(data, coords=coords).to_netcdf(path)


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    """Write the synthetic inputs once and catalog them; return what the tests need."""
    root = tmp_path_factory.mktemp("inputs")
    cats = root / "cats"
    cats.mkdir()
    patch = pytest.MonkeyPatch()
    patch.setenv("OCEAN_SKILL_CATALOGS", str(cats))
    patch.setattr(catalog, "_added_dirs", [])
    patch.setattr(catalog, "_user_dir", lambda: root / "user")
    patch.setattr(catalog, "_legacy_user_dir", lambda: root / "legacy")
    patch.setattr(cache, "_enabled", False)  # nothing here may reach the real cache
    grid, bare = root / "grid.nc", root / "bare_grid.nc"
    lon, lat = _write_grid(grid)
    _write_grid(bare, mask=False)
    cat = build.new_catalog(title="inputs")
    surf, bry, clim = (root / f for f in ("s.nc", "b.nc", "c.nc"))
    _surface(surf)
    _boundary(bry)
    _boundary(clim, month=True)
    build.add_forcing_source(cat, "surf", surf, grid)
    build.add_forcing_source(cat, "surf_bare", surf, bare)
    build.add_boundary_sources(cat, "bry", bry, grid, sides=("south", "west"))
    build.add_boundary_sources(
        cat, "bclim", clim, grid, sides=("south",), climatology=True
    )
    build.save(cat, cats / "inputs.yaml")
    run = write_run(root / "run", cats, subchunk=SUBCHUNK)
    yield {"root": root, "lon": lon, "lat": lat, "run": run}
    patch.undo()
    sources.read.cache_clear()


@pytest.fixture(scope="module")
def surface(inputs):
    return input_health("surf")


def _row(result: InputHealth, variable: str, source: str | None = None):
    table = result.table()
    rows = table[table["variable"] == variable]
    if source is not None:
        rows = rows[rows["source"] == source]
    assert len(rows) == 1, list(table["variable"])
    return rows.iloc[0]


def _problems(result: InputHealth, kind: str) -> list[dict]:
    return [p for p in result.problems() if p["kind"] == kind]


# ------------------------------------------------------------------ the surface file


def test_a_negative_cell_and_a_hot_one_are_found_where_they_are(surface, inputs):
    rain = _row(surface, "rain")
    ocean = NY * NX - len(LAND)
    assert rain["worst_pct_bad"] == pytest.approx(100.0 / ocean)
    assert rain["worst_bad_time"] == "2011-01-01 04:00"
    assert rain["min"] == pytest.approx(-1e-3)
    assert (rain["min_lon"], rain["min_lat"]) == (
        inputs["lon"][0, 1],
        inputs["lat"][0, 1],
    )
    assert rain["bound"] == "≥ 0" and rain["units"] == "cm/day"
    (problem,) = _problems(surface, "negative")
    assert problem["variable"] == "rain" and problem["source"] == "surf"

    tair = _row(surface, "Tair")
    assert tair["max"] == 75.0 and tair["max_time"] == "2011-01-01 02:00"
    assert (tair["max_lon"], tair["max_lat"]) == (
        inputs["lon"][3, 4],
        inputs["lat"][3, 4],
    )
    assert tair["bound"] == "−60 to 60"
    assert tair["worst_pct_bad"] == pytest.approx(100.0 / ocean)
    kinds = {(p["variable"], p["kind"]) for p in surface.problems()}
    assert ("Tair", "out_of_bounds") in kinds


def test_land_holds_junk_and_nan_that_never_counts(surface):
    ocean = NY * NX - len(LAND)
    # the land cells hold 1e6 (and NaN in swrad); the statistics see none of it
    assert _row(surface, "Tair")["max"] == 75.0
    swrad = _row(surface, "swrad")
    assert swrad["n_nonfinite"] == 0 and swrad["max"] < 400
    assert not _problems(surface, "all_nan")
    # the one NaN over the ocean is counted
    qair = _row(surface, "qair")
    assert qair["n_nonfinite"] == 1 and qair["nonfinite_time"] == "2011-01-01 01:00"
    (problem,) = [p for p in _problems(surface, "nonfinite") if p["variable"] == "qair"]
    assert problem["count"] == 1
    steps = surface.steps
    n_ok = steps[steps["variable"] == "qair"]["n_ok"]
    assert set(n_ok) == {ocean}


def test_a_zero_field_a_constant_one_and_a_repeated_step_are_flagged(surface):
    lwrad = _row(surface, "lwrad")  # land holds 5; the ocean is all zero
    assert lwrad["flags"] == "all zero"
    assert [p["variable"] for p in _problems(surface, "all_zero")] == ["lwrad"]
    vwnd = _row(surface, "vwnd")  # 3 at every ocean cell, 1e6 on land
    assert vwnd["flags"] == "constant" and vwnd["min"] == vwnd["max"] == 3.0
    assert vwnd["std"] == pytest.approx(0.0, abs=1e-6)
    uwnd = _row(surface, "uwnd")
    assert uwnd["n_repeats"] == 1 and "same as previous step x1" in uwnd["flags"]
    (problem,) = _problems(surface, "repeat")
    assert problem["variable"] == "uwnd" and problem["steps"] == 1
    # an ordinary field has no flag
    assert _row(surface, "swrad")["flags"] == ""


def test_mean_and_std_are_over_ocean_cells_and_steps(surface):
    steps = surface.steps
    qair = steps[steps["variable"] == "qair"]
    n = qair["n_fin"].sum()
    table = _row(surface, "qair")
    assert table["mean"] == pytest.approx(qair["total"].sum() / n)
    t = np.arange(NT)[:, None, None]
    cell = np.arange(NY * NX).reshape(NY, NX)[None]
    full = 0.01 + 0.0001 * t + 0.00001 * cell
    wet = np.ones((NY, NX), bool)
    for c in LAND:
        wet[c] = False
    values = np.broadcast_to(full, (NT, NY, NX))[:, wet].copy()
    values[1, list(np.flatnonzero(wet.ravel())).index(2 * NX + 2)] = np.nan  # (2, 2)
    assert table["mean"] == pytest.approx(np.nanmean(values))
    assert table["std"] == pytest.approx(np.nanstd(values))
    assert table["n_steps"] == NT
    assert (table["first_time"], table["last_time"]) == (
        "2011-01-01 00:00",
        "2011-01-01 05:00",
    )


def test_a_time_window_and_max_steps_pick_the_steps_scanned(inputs):
    two = input_health(
        "surf", time={"min": "2011-01-01T01:00", "max": "2011-01-01T02:00"}
    )
    assert _row(two, "Tair")["n_steps"] == 2
    assert _row(two, "Tair")["max"] == 75.0
    assert _row(two, "rain")["min"] > 0  # the negative step (4) is outside the window
    latest = input_health("surf", time="latest")
    assert _row(latest, "Tair")["n_steps"] == 1
    assert _row(latest, "Tair")["first_time"] == "2011-01-01 05:00"
    day = input_health("surf", time="2011-01-01")
    assert _row(day, "Tair")["n_steps"] == NT
    thin = input_health("surf", max_steps=3)
    assert _row(thin, "Tair")["n_steps"] == 3
    assert (_row(thin, "Tair")["first_time"], _row(thin, "Tair")["last_time"]) == (
        "2011-01-01 00:00",
        "2011-01-01 05:00",
    )
    assert thin.sources[0]["thinned"] is True
    with pytest.raises(ValueError, match="selects none of the 6 steps of 'surf'"):
        input_health("surf", time="2012-01")
    with pytest.raises(ValueError, match="not a date"):
        input_health("surf", time="soon")


def test_block_planning_does_not_change_the_answer(inputs, monkeypatch, surface):
    # a block smaller than one step splits it into levels; a tiny one a step at a time
    monkeypatch.setattr(input_integrity, "_BLOCK_BYTES", 1)
    again = input_health("surf", cache=False)
    pd.testing.assert_frame_equal(again.table(), surface.table())


def test_vars_restrict_and_a_missing_one_is_an_error(inputs):
    only = input_health({"name": "surf", "vars": ["Tair", "rain"], "label": "air"})
    assert list(only.table()["variable"]) == ["Tair", "rain"]
    assert only.sources[0]["label"] == "air"
    with pytest.raises(ValueError, match=r"no gridded variable\(s\) \['sst'\]"):
        input_health({"name": "surf", "vars": ["sst"]})


# ----------------------------------------------------------------- the boundary file


def test_a_boundary_side_is_masked_by_its_own_edge_and_found_in_depth(inputs):
    south = input_health("bry_south")
    ocean = NX - 1  # the south row has one land cell, xi = 3
    no3 = _row(south, "NO3")
    assert no3["min"] == -2.0 and no3["min_time"] == "2011-01-02"
    # level 1 of the three, xi 4 on the south row
    z = roms._s_to_z(
        (np.arange(NS)[1] - NS + 0.5) / NS,
        ((np.arange(NS)[1] - NS + 0.5) / NS) ** 3,
        500.0,
        0.0,
        50.0,
        2,
    )
    assert no3["min_level"] == 1 and no3["min_depth"] == pytest.approx(-z)
    assert (no3["min_lon"], no3["min_lat"]) == (
        inputs["lon"][0, 4],
        inputs["lat"][0, 4],
    )
    assert no3["worst_pct_bad"] == pytest.approx(100.0 / (NS * ocean))
    assert no3["bound"] == "≥ 0"
    steps = south.steps
    assert set(steps[steps["variable"] == "NO3"]["n_ok"]) == {NS * ocean}
    # the land cell's -999 never reaches the minimum
    assert no3["min"] > -999
    west = input_health("bry_west")  # no land on the west column
    assert _row(west, "NO3")["min"] > 0
    assert set(west.steps[west.steps["variable"] == "NO3"]["n_ok"]) == {NS * NY}


def test_a_copy_of_another_variable_is_named(inputs):
    south = input_health("bry_south")
    assert _row(south, "DIC")["flags"] == ""
    alt = _row(south, "DIC_ALT_CO2")
    assert alt["copy_of"] == "DIC" and alt["flags"] == "copy of DIC"
    (problem,) = _problems(south, "copy")
    assert problem["variable"] == "DIC_ALT_CO2" and problem["copy_of"] == "DIC"
    # a variable that merely has the same range is not a copy
    assert not _row(south, "temp")["copy_of"]


def test_a_copy_that_differs_in_one_cell_is_not_one(inputs, tmp_path, monkeypatch):
    """Equal counts, range and sums are only candidates; the cells are compared."""
    src = input_integrity._open_source({"name": "bry_south"}, None, None)
    a, b = src.variables["DIC"], src.variables["DIC_ALT_CO2"]
    assert input_integrity._same(src, a, b, 0)
    real = input_integrity._read

    def swapped(src_, v, step, levels):
        x = real(src_, v, step, levels)
        if v is b:
            x = x.copy()
            x[..., 0, 0] += 1.0
        return x

    monkeypatch.setattr(input_integrity, "_read", swapped)
    assert not input_integrity._same(src, a, b, 0)


def test_side_suffixed_names_find_their_bound():
    assert bounds("temp_south") == bounds("temp") == (-2.5, 36.0)
    assert bounds("NO3_west") == (0.0, float("inf"))
    assert bounds("DIC_ALT_CO2_south") == (0.0, float("inf"))
    assert bounds("uwnd") == (-50.0, 50.0) and bounds("u_north") == (-3.0, 3.0)
    assert bounds("qair") == (0.0, 0.05) and bounds("xco2") == (150.0, 1000.0)
    assert bounds("pH_south") == bounds("MARBL_PH_3D") == (6.5, 9.0)


def test_a_climatology_is_scanned_by_month(inputs):
    clim = input_health("bclim_south", time="2011-01")  # a window means nothing here
    row = _row(clim, "temp")
    assert row["n_steps"] == 12
    assert (row["first_time"], row["last_time"]) == ("month 1", "month 12")
    assert row["max_time"].startswith("month")
    assert clim.sources[0]["climatology"] is True
    assert _row(clim, "DIC_ALT_CO2")["copy_of"] == "DIC"


def test_copies_that_share_an_all_nan_step_are_still_copies(inputs):
    """NaN is not equal to itself: an all-NaN step has no min or max to match on."""
    root = inputs["root"]
    t = np.arange(NT)[:, None, None]
    base = 1.0 + 0.1 * t + 0.01 * np.arange(NY * NX).reshape(NY, NX)[None]
    base = np.broadcast_to(base, (NT, NY, NX)).copy()
    base[2] = np.nan  # every cell, one step
    dims = ("time", "eta_rho", "xi_rho")
    xr.Dataset(
        {name: (dims, base.copy()) for name in ("dust", "iron")}
        | {"nox": (dims, base + 1.0)},
        coords={"time": _days(NT)},
    ).to_netcdf(root / "nan.nc")
    cat = build.new_catalog(title="nan")
    build.add_forcing_source(cat, "nansurf", root / "nan.nc", root / "grid.nc")
    build.save(cat, root / "cats" / "nan.yaml")
    result = input_health("nansurf")
    assert _row(result, "iron")["copy_of"] == "dust"
    assert _row(result, "dust")["all_nan_steps"] == 1
    assert not _row(result, "nox")["copy_of"]


def test_a_name_given_twice_is_one_source():
    merged = input_integrity.normalize_sources(
        [
            {"name": "a", "label": "first", "vars": ["x"]},
            {"name": "a", "label": "second", "vars": ["y", "x"]},
            {"name": "b", "vars": ["z"]},
            "b",  # unrestricted: every variable
            {"name": "c", "label": "late"},
            {"name": "c"},
        ]
    )
    assert merged == [
        {"name": "a", "label": "first", "vars": ["x", "y"]},
        {"name": "b"},
        {"name": "c", "label": "late"},
    ]


def test_a_source_listed_twice_is_scanned_once_and_does_not_repeat_itself(inputs):
    result = input_health(
        [{"name": "surf", "vars": ["Tair"]}, {"name": "surf", "vars": ["rain"]}]
    )
    assert list(result.table()["variable"]) == ["Tair", "rain"]
    assert set(result.table()["n_steps"]) == {NT}
    assert not _problems(result, "repeat")


# --------------------------------------------------------------------------- no mask


def test_a_source_with_no_mask_counts_every_cell_and_says_so(inputs):
    bare = input_health({"name": "surf_bare", "vars": ["Tair"]})
    steps = bare.steps
    assert set(steps["n_ok"]) == {NY * NX}
    assert _row(bare, "Tair")["max"] == JUNK  # land holds junk and nothing hides it
    assert bare.sources[0]["no_mask"] is True
    assert any("No ocean mask" in note for note in bare._notes())


# ------------------------------------------------------------- the 3-D restart source


def test_a_restart_with_levels_and_staggered_velocities(inputs, monkeypatch):
    result = input_health(inputs["run"])
    table = result.table().set_index("variable")
    assert table.loc["temp", "min"] == -5.0  # one -5 degC cell at the surface level
    assert (table.loc["temp", "min_lon"], table.loc["temp", "min_lat"]) == (205.0, 14.0)
    assert table.loc["temp", "min_time"] == "2011-05-31"
    assert table.loc["temp", "min_level"] == 3 and table.loc["temp", "min_depth"] > 0
    # u and v are 0.1 except the junk on land points (50), which both neighbors rule out
    assert table.loc["u", "max"] == pytest.approx(0.1)
    assert table.loc["v", "max"] == pytest.approx(0.1)
    assert table.loc["NH4", "flags"] == "all zero"
    assert table.loc["DIC_ALT_CO2", "copy_of"] == "DIC"
    assert table.loc["ALK_ALT_CO2", "copy_of"] == ""
    assert table.loc["NO3", "worst_pct_bad"] > 0
    # blocks of one level, one step at a time, give the same numbers
    monkeypatch.setattr(input_integrity, "_BLOCK_BYTES", 1)
    split = input_health(inputs["run"], cache=False).table().set_index("variable")
    pd.testing.assert_frame_equal(split, table)


def test_it_agrees_with_run_health_on_the_same_restart(inputs):
    from ocean_skill import health

    theirs = health(inputs["run"], cache=False).table().set_index("variable")
    mine = input_health(inputs["run"], cache=False).table().set_index("variable")
    for name in ("temp", "salt", "NO3", "O2", "DIC", "u", "v"):
        assert mine.loc[name, "min"] == theirs.loc[name, "min"]
        assert mine.loc[name, "max"] == theirs.loc[name, "max"]
        assert mine.loc[name, "worst_pct_neg"] == pytest.approx(
            theirs.loc[name, "worst_pct_neg"]
        )
        assert mine.loc[name, "n_nonfinite"] == theirs.loc[name, "n_nonfinite"]


# ---------------------------------------------------------------------- many sources


def test_several_sources_and_nested_lists_make_one_table(inputs):
    result = input_health(
        [
            [{"name": "surf", "label": "surface physics", "vars": ["Tair"]}],
            "bry_south",
            ["bry_south", {"name": "bry_west", "vars": ["temp"]}],  # one is a repeat
        ]
    )
    table = result.table()
    assert list(dict.fromkeys(table["source"])) == ["surf", "bry_south", "bry_west"]
    assert list(table[table["source"] == "surf"]["label"]) == ["surface physics"]
    assert len(table[table["source"] == "bry_south"]) == 5
    assert {p["source"] for p in result.problems()} >= {"surf", "bry_south"}
    with pytest.raises(ValueError, match="at least one source"):
        input_health([])
    with pytest.raises(ValueError, match="needs a 'name'"):
        input_health([{"label": "x"}])


def test_problems_are_json_safe_and_most_severe_first(surface):
    import json

    problems = surface.problems()
    json.dumps(problems)
    order = [input_integrity._SEVERITY.index(p["kind"]) for p in problems]
    assert order == sorted(order)
    assert {
        "nonfinite",
        "negative",
        "out_of_bounds",
        "all_zero",
        "constant",
        "repeat",
    } <= {p["kind"] for p in problems}
    for p in problems:
        assert p["message"] and p["source"] == "surf"


# ----------------------------------------------------------------------------- cache


def test_each_variable_is_cached_and_a_rewritten_file_misses(inputs, monkeypatch):
    first = input_health("surf")
    assert len(cache.entries("inputs")) == 7  # one per variable
    called = []
    real = input_integrity._scan_variable
    monkeypatch.setattr(
        input_integrity,
        "_scan_variable",
        lambda *a: called.append(a) or real(*a),
    )
    again = input_health("surf")
    assert not called  # every variable read from the cache
    pd.testing.assert_frame_equal(again.table(), first.table())
    assert len(cache.entries("inputs")) == 7
    # another window is another entry; cache=False reads and stores nothing
    input_health("surf", time="latest")
    assert len(cache.entries("inputs")) == 14 and len(called) == 7
    called.clear()
    input_health("surf", time={"min": "2011-01-01T02:00"}, cache=False)
    assert len(cache.entries("inputs")) == 14 and len(called) == 7
    called.clear()

    # a bound changed in the code changes what the counts mean: another key
    src = input_integrity._open_source({"name": "surf"}, None, None)
    v = src.variables["Tair"]
    before = input_integrity._key(src, v, "")
    with monkeypatch.context() as patch:
        patch.setitem(input_integrity._FORCING_RANGES, "Tair", (-10.0, 10.0))
        assert input_integrity._key(src, v, "") != before
    assert input_integrity._key(src, v, "") == before

    # a file rewritten in place (same name, other values) is another file: its size and
    # modification time are in every variable's key, and the sample sees the new values
    called.clear()
    path = inputs["root"] / "s.nc"
    saved = path.read_bytes()
    try:
        ds = xr.load_dataset(path)
        ds["Tair"] = ds["Tair"] + 1.0
        ds.close()
        path.unlink()
        ds.to_netcdf(path)
        sources.read.cache_clear()
        input_health("surf")
        assert len(called) == 7
    finally:
        path.write_bytes(saved)


def test_a_cache_lookup_reads_only_small_windows_and_threads_are_capped(
    inputs, monkeypatch
):
    src = input_integrity._open_source({"name": "bry_south"}, None, None)
    reads = []
    real = np.ascontiguousarray
    monkeypatch.setattr(
        np, "ascontiguousarray", lambda a, *k: reads.append(np.size(a)) or real(a, *k)
    )
    input_integrity._sample_hash(src, src.variables["NO3"])
    assert reads and max(reads) <= input_integrity._SAMPLE_EDGE**2
    import dask

    with dask.config.set(num_workers=128):
        assert input_integrity._scan_workers() == input_integrity._MAX_WORKERS
    with dask.config.set(num_workers=2):
        assert input_integrity._scan_workers() == 2


def test_a_variable_named_like_a_flag_raises_no_spurious_problem(surface):
    # the structured row fields, not the text of the flags, decide the problems
    row = _row(surface, "swrad")
    assert not row["zero_steps"] and not row["is_constant"]
    fake = pd.DataFrame([row.to_dict() | {"variable": "constant_x", "flags": ""}])
    assert not [
        p
        for r in fake.itertuples(index=False)
        for p in InputHealth._row_problems(r)
        if p["kind"] in ("constant", "all_zero")
    ]


# ----------------------------------------------------------------------------- figures


def test_the_integrity_figure_fits_a_letter_page(surface, tmp_path):
    ((suffix, fig),) = surface.figures("integrity")
    assert suffix == ""
    width, height = fig.get_size_inches()
    assert width == 8.5 and height <= 9.5
    fig.savefig(tmp_path / "integrity.png", dpi=60)
    texts = {t.get_text() for t in fig.texts}
    assert any("Input integrity: surf" in t for t in texts)
    assert any("Tair\ndegC" in t or t.startswith("Tair") for t in texts)
    assert any(t == "all zero" for t in texts)
    with pytest.raises(ValueError, match="unknown input_health draw"):
        surface.figures("nope")
    assert INPUT_DRAWS == ("integrity",)


def test_a_long_table_continues_on_more_figures(inputs, monkeypatch):
    result = input_health(["surf", "bry_south", "bry_west"])
    monkeypatch.setattr(input_integrity, "_FIG_MAX_H", 3.0)
    figures = result.figures("integrity")
    assert len(figures) > 1
    assert [s for s, _ in figures] == [
        "",
        *(f"-{i}" for i in range(2, len(figures) + 1)),
    ]
    assert all(fig.get_size_inches()[1] <= 3.0 + 1e-9 for _, fig in figures)
    assert any("(cont.)" in t.get_text() for t in figures[1][1].texts)
    names = set(result.table()["variable"])
    rows = sum(
        1
        for _, fig in figures
        for t in fig.texts
        if t.get_text().split("\n")[0] in names
    )
    assert rows == len(result.table())  # every variable is on exactly one figure


def test_test_mode_is_said_on_the_figure_and_thins_the_steps(inputs):
    with sources.truncate_time(["surf"], 1):
        result = input_health("surf", cache=False)
    assert result.test_mode is True
    assert _row(result, "Tair")["n_steps"] == 2  # one step kept at each end
    ((_, fig),) = result.figures("integrity")
    assert any("Test mode" in t.get_text() for t in fig.texts)
    assert osk.input_health is input_health
