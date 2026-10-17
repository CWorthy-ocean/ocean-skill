"""Run-health diagnostics: bounds, copies, inventories, steps, figures and the cache.

A tiny synthetic UCLA-style run (8 x 10 cells, 4 levels, monthly restarts) with known
faults planted in it: negative nitrate at one restart, a -5 degC cell at another, a 5%
DIC jump at a third, a variable that is zero everywhere, a CDR tracer that equals its
twin and one that differs from it by a constant. Constant fields make the inventories
checkable by hand: value times the ocean volume.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ocean_skill import build, cache, catalog, run_health, sources
from ocean_skill.run_health import HEALTH_DRAWS, RunHealth, health

NS, NY, NX = 4, 8, 10
HC = 20.0
STAMPS = ("20110131", "20110228", "20110331", "20110430", "20110531", "20110630")
AREA = 5000.0 * 5000.0
LAND = (2, 3)  # (eta, xi) of the one land cell
BOTTOM = 600.0 + 100.0 * np.arange(NX)  # 600 m to 1500 m, so there is deep water
NEG_STEP, JUMP_STEP, COLD_STEP = 2, 3, 4  # where the faults are planted


def _write_grid(path):
    lon, lat = np.meshgrid(200.0 + np.arange(NX), 10.0 + np.arange(NY))
    h = np.broadcast_to(BOTTOM, (NY, NX)).copy()
    mask = np.ones((NY, NX))
    mask[LAND] = 0.0
    rho = ("eta_rho", "xi_rho")
    xr.Dataset(
        {
            "lon_rho": (rho, lon),
            "lat_rho": (rho, lat),
            "h": (rho, h),
            "mask_rho": (rho, mask),
            "angle": (rho, np.zeros_like(h)),
            "pm": (rho, np.full(h.shape, 1.0 / 5000.0)),
            "pn": (rho, np.full(h.shape, 1.0 / 5000.0)),
        },
        attrs={"hc": HC, "Vtransform": 2, "theta_s": 6.0, "theta_b": 3.0},
    ).to_netcdf(path)


def _fields(i: int, no3_shift: float) -> dict[str, np.ndarray]:
    """Return restart ``i``'s tracers, faults and all, as ``(s, eta, xi)`` arrays."""
    full = (NS, NY, NX)
    f = {
        "temp": np.full(full, 10.0),
        "salt": np.full(full, 35.0),
        "NO3": np.full(full, 20.0 + 0.01 * i + no3_shift),
        "O2": np.full(full, 100.0),
        "NH4": np.zeros(full),
        "ALK": np.full(full, 2300.0),
        "DIC": np.full(full, 2000.0 if i < JUMP_STEP else 2100.0),
        "spC": np.full(full, 1.0 + 0.1 * i),
        "spChl": np.full(full, 0.2),
        "spP": np.full(full, 0.06),
        "spFe": np.full(full, 1e-4),
    }
    f["DIC_ALT_CO2"] = f["DIC"].copy()  # the CDR perturbation that never arrived
    f["ALK_ALT_CO2"] = f["ALK"] + 5.0  # one that did: a constant 5 everywhere
    f["O2"][:, 5, 5] = 40.0  # a hypoxic column
    if i == NEG_STEP:
        f["NO3"][:, 1, 1] = -1.0
    if i == COLD_STEP:
        f["temp"][-1, 4, 5] = -5.0  # surface level, one cell
    return f


def _write_restart(path, i: int, no3_shift: float = 0.0):
    stamp = STAMPS[i]
    day = np.datetime64(f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:]}T00:00:00")
    secs = float((day - np.datetime64("2000-01-01T00:00:00")) / np.timedelta64(1, "s"))
    sw = np.linspace(-1.0, 0.0, NS + 1)
    cs_w = (np.exp(2.0 * sw) - np.exp(-2.0)) / (1.0 - np.exp(-2.0)) - 1.0
    rho3 = ("time", "s_rho", "eta_rho", "xi_rho")
    data = {
        n: (rho3, v[None].astype("float32")) for n, v in _fields(i, no3_shift).items()
    }
    u = np.full((1, NS, NY, NX - 1), 0.1, dtype="float32")
    u[:, :, LAND[0], LAND[1] - 1] = 50.0  # land points hold junk; it must not count
    v = np.full((1, NS, NY - 1, NX), 0.1, dtype="float32")
    v[:, :, LAND[0], LAND[1]] = 50.0
    xr.Dataset(
        {
            **data,
            "u": (("time", "s_rho", "eta_rho", "xi_u"), u),
            "v": (("time", "s_rho", "eta_v", "xi_rho"), v),
            "zeta": (("time", "eta_rho", "xi_rho"), np.zeros((1, NY, NX), "float32")),
            "ocean_time": ("time", [secs], {"units": "second"}),
            "sigma_r": ("s_rho", 0.5 * (sw[1:] + sw[:-1])),
            "Cs_r": ("s_rho", 0.5 * (cs_w[1:] + cs_w[:-1])),
            "sigma_w": ("s_w", sw),
            "Cs_w": ("s_w", cs_w),
        }
    ).to_netcdf(path, unlimited_dims=["time"])


def _kerchunk(root, n_steps: int):
    files = sorted(root.glob("output_rst.*.nc"))[:n_steps]
    return build.make_kerchunk(files, out=root / "refs.json", grid=root / "grid.nc")


def write_run(root, catalogs, steps: int = len(STAMPS)) -> str:
    """Write the synthetic run and its catalog entry; return its nickname."""
    root.mkdir(exist_ok=True)
    _write_grid(root / "grid.nc")
    for i in range(steps):
        _write_restart(root / f"output_rst.{STAMPS[i]}.nc", i)
    refs = _kerchunk(root, steps)
    build.build_catalog({"run": str(refs)}, catalogs / "tiny.yaml", title="tiny run")
    sources.read.cache_clear()
    return "tiny:run"


def _volume() -> float:
    """Return the ocean volume (m3): every column is as deep as its ``h``."""
    wet = np.ones((NY, NX), dtype=bool)
    wet[LAND] = False
    return float((np.broadcast_to(BOTTOM, (NY, NX))[wet] * AREA).sum())


@pytest.fixture(scope="module")
def scanned(tmp_path_factory):
    """Return the synthetic run and its scan, made once for the read-only tests."""
    root = tmp_path_factory.mktemp("health")
    cats = root / "cats"
    cats.mkdir()
    patch = pytest.MonkeyPatch()
    patch.setenv("OCEAN_SKILL_CATALOGS", str(cats))
    patch.setattr(catalog, "_added_dirs", [])
    patch.setattr(catalog, "_user_dir", lambda: root / "user")
    patch.setattr(catalog, "_legacy_user_dir", lambda: root / "legacy")
    patch.setattr(cache, "_enabled", False)  # nothing here may reach the real cache
    nickname = write_run(root / "run", cats)
    result = health(nickname, joins=[pd.Timestamp("2011-04-15")], cache=False)
    yield nickname, result
    patch.undo()
    sources.read.cache_clear()


@pytest.fixture
def result(scanned):
    return scanned[1]


def _problem(result: RunHealth, kind: str, variable: str) -> dict:
    found = [
        p for p in result.problems() if p["kind"] == kind and p["variable"] == variable
    ]
    assert found, [p["message"] for p in result.problems()]
    return found[0]


# ------------------------------------------------------------------ what is caught


def test_planted_negatives_and_bounds_are_caught(result):
    table = result.table().set_index("variable")

    no3 = table.loc["NO3"]
    assert no3["worst_pct_neg"] == pytest.approx(100.0 * 4 / (NS * (NY * NX - 1)))
    assert no3["worst_neg_time"] == pd.Timestamp("2011-03-31")
    assert no3["latest_pct_neg"] == 0.0  # fine again by the latest restart
    assert _problem(result, "negative", "NO3")["time"] == "2011-03-31"

    temp = table.loc["temp"]  # one -5 degC cell, at the surface of cell (4, 5)
    assert temp["min"] == -5.0
    assert (temp["min_lon"], temp["min_lat"]) == (205.0, 14.0)
    assert 0.0 < temp["min_depth"] < BOTTOM[5] / 2  # in the top cell
    assert temp["worst_bad_time"] == pd.Timestamp("2011-05-31")
    assert _problem(result, "out_of_bounds", "temp")["worst_pct"] > 0.1

    # land points holding 50 m/s are not ocean: no velocity problem
    assert not [p for p in result.problems() if p["variable"] in ("u", "v")]
    assert table.loc["u", "max"] == pytest.approx(0.1)


def test_zero_fields_copies_and_steps_are_caught(result):
    assert "all zero" in result.table().set_index("variable").loc["NH4", "flags"]
    assert _problem(result, "all_zero", "NH4")["restarts"] == len(STAMPS)

    copy = _problem(result, "copy", "DIC_ALT_CO2")
    assert copy["message"] == "DIC_ALT_CO2 is identical to DIC in every restart."
    assert not [p for p in result.problems() if p["variable"] == "ALK_ALT_CO2"]

    step = _problem(result, "step", "DIC")
    assert step["time"] == "2011-04-30"
    assert step["change_pct"] == pytest.approx(5.0)
    assert step["at_join"] is True  # the 2011-04-15 join lies between the two restarts
    assert "at a segment join" in step["message"]
    assert not [
        p for p in result.problems() if p["kind"] == "step" and p["variable"] == "salt"
    ]
    json.dumps(result.problems())  # JSON-safe


def test_inventories_equal_value_times_volume(result):
    vol = _volume()

    def series(name):
        return result.steps[result.steps["variable"] == name].set_index("time")["value"]

    assert series("salt").to_numpy() == pytest.approx(35.0 * vol, rel=1e-9)
    assert series("ALK_ALT_CO2").to_numpy() == pytest.approx(2305.0 * vol, rel=1e-9)
    # added carbon: identical tracers cancel; the constant 5 of alkalinity is 5 x volume
    assert series("cdr_DIC").abs().max() == pytest.approx(0.0, abs=1e-6 * vol)
    assert series("cdr_ALK").to_numpy() == pytest.approx(-5.0 * vol, rel=1e-9)
    # deep water (below 1000 m): temperature and salinity are the same everywhere
    assert series("deep_temp").to_numpy() == pytest.approx(10.0)
    assert series("deep_salt").to_numpy() == pytest.approx(35.0)
    # the hypoxic column (5, 5) is below 60 mmol/m3 all the way down, nowhere below 20
    assert series("O2_lt60").iloc[0] == pytest.approx(BOTTOM[5] * AREA, rel=1e-9)
    assert series("O2_lt20").iloc[0] == 0.0


def test_kinetic_energy_masks_land_points(result):
    u = np.full((NY, NX), 0.1)  # on rho points after the average; a wall carries 0
    v = u.copy()
    u[LAND[0], LAND[1] - 1] = u[LAND[0], LAND[1] + 1] = 0.05
    v[LAND[0] - 1, LAND[1]] = v[LAND[0] + 1, LAND[1]] = 0.05
    wet = np.ones((NY, NX), dtype=bool)
    wet[LAND] = False
    expected = 0.5 * ((u**2 + v**2) * BOTTOM[None] * AREA)[wet].sum()

    ke = result.steps[result.steps["variable"] == "KE"]["value"]
    assert ke.to_numpy() == pytest.approx(expected, rel=1e-6)  # u, v are float32


def test_drift_reads_the_slope_of_the_inventory(result):
    drift = result.drift().set_index(["series", "variable"])
    assert drift.loc[("deep mean", "deep_salt"), "pct_per_year"] == pytest.approx(0)
    assert drift.loc[("inventory", "NO3"), "pct_per_year"] > 0.0  # +0.01 per restart
    # temperature, salinity and pH have no inventory: a total of them means nothing
    assert not {"temp", "salt"} & set(drift.loc["inventory"].index)


# ------------------------------------------------------------------------ the figures


def test_every_draw_gives_figures(result):
    for draw in HEALTH_DRAWS:
        pages = result.figures(draw)
        assert pages, draw
        suffixes = [suffix for suffix, _ in pages]
        assert suffixes[0] == "" and all(s.startswith("-") for s in suffixes[1:])
        for _, fig in pages:
            fig.canvas.draw()
    with pytest.raises(ValueError, match="unknown health draw"):
        result.figures("nope")


def test_summary_lists_problems_first_and_never_reassures(result):
    ((_, fig), *_) = result.figures("summary")
    text = [t.get_text() for t in fig.texts]
    drift = next(i for i, t in enumerate(text) if t.startswith("Drift"))
    assert text.index("Problems") < drift
    assert not any("no NaN" in t or "no steps" in t for t in text)
    assert any("DIC_ALT_CO2 is identical to DIC" in t for t in text)


def test_a_run_without_plankton_has_no_plankton_page(result):
    plankton = ("sp", "diat", "diaz", "zoo")
    bare = RunHealth(
        "tiny:run", result.steps[~result.steps["variable"].str.startswith(plankton)]
    )
    assert bare.figures("plankton") == []
    assert bare.figures("inventory")  # the rest still draws


def test_a_single_step_still_draws(scanned):
    one = health(scanned[0], time="latest", cache=False)
    assert one.drift().empty
    for draw in HEALTH_DRAWS:
        for _, fig in one.figures(draw):
            fig.canvas.draw()


def test_test_mode_says_so_and_skips_the_truncation_gap(scanned):
    with sources.truncate_time(["run"], 2):
        result = health(scanned[0], cache=False)

    assert result.test_mode is True
    assert len(result._times()) == 4  # the first two and the last two of six
    assert result._adjacent.tolist() == [True, False, True]  # nothing across the gap
    for draw in HEALTH_DRAWS:
        for _, fig in result.figures(draw):
            says = [t for t in fig.texts if run_health.TEST_MODE_BANNER in t.get_text()]
            assert says, draw


# ------------------------------------------------------------------------- the cache


def test_each_step_is_cached_and_a_rewritten_restart_misses(
    tmp_path, isolated_catalogs, monkeypatch
):
    calls = []
    real = run_health._scan_step

    def counting(run, step, stamp, zeta):
        calls.append(stamp)
        return real(run, step, stamp, zeta)

    monkeypatch.setattr(run_health, "_scan_step", counting)
    nickname = write_run(tmp_path / "run", isolated_catalogs, steps=3)

    first = health(nickname)
    assert len(calls) == 3
    assert len(cache.entries("health")) == 3

    again = health(nickname)  # every step served from the cache
    assert len(calls) == 3
    pd.testing.assert_frame_equal(first.steps, again.steps)

    health(nickname, cache=False)  # bypass: recomputed, and nothing stored
    assert len(calls) == 6
    assert len(cache.entries("health")) == 3

    # the third restart is rewritten in place: only that step is read again
    _write_restart(tmp_path / "run" / f"output_rst.{STAMPS[2]}.nc", 2, no3_shift=3.0)
    _kerchunk(tmp_path / "run", 3)
    sources.read.cache_clear()
    changed = health(nickname)
    assert calls[6:] == [pd.Timestamp("2011-03-31")]
    no3 = changed.steps[changed.steps["variable"] == "NO3"].set_index("time")["value"]
    assert no3.iloc[2] > no3.iloc[1] * 1.1

    # a smoke run keeps its steps in its own scope, away from the real ones
    with cache.scoped("smoke"):
        health(nickname)
        assert len(cache.entries("health")) == 3
    assert len(cache.entries("health")) == 4  # the third step, as it was and as it is
