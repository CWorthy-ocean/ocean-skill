"""Run-health diagnostics: bounds, copies, inventories, steps, figures and the cache.

A tiny synthetic UCLA-style run (8 x 10 cells, 4 levels, monthly restarts) with known
faults planted in it: negative nitrate at one restart, a -5 degC cell at another, a 5%
DIC jump at a third, a variable that is zero everywhere, a CDR tracer that equals its
twin and one that differs from it by a constant. Constant fields make the inventories
checkable by hand: value times the ocean volume.
"""

from __future__ import annotations

import json

import dask
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from dask.callbacks import Callback

from ocean_skill import build, cache, catalog, run_health, sources
from ocean_skill.run_health import HEALTH_DRAWS, RunHealth, health

NS, NY, NX = 4, 8, 10
HC = 20.0
STAMPS = ("20110131", "20110228", "20110331", "20110430", "20110531", "20110630")
AREA = 5000.0 * 5000.0
LAND = (2, 3)  # (eta, xi) of the one land cell
BOTTOM = 600.0 + 100.0 * np.arange(NX)  # 600 m to 1500 m, so there is deep water
NEG_STEP, JUMP_STEP, COLD_STEP = 2, 3, 4  # where the faults are planted
_CHECKS = {
    "Check",
    "NaN / Inf",
    "Entirely NaN",
    "All zero",
    "Negative concentrations",
    "Outside physical bounds",
    "Exact copies",
    "CDR tracers",
    "Sudden step",
    "Repeated restart",
    "Largest drift",
}

#: Stored like a large run's reference: a level by half the rows per chunk.
SUBCHUNK = {"s_rho": 1, "s_w": 1, "eta_rho": NY // 2}


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


def _kerchunk(root, n_steps: int, subchunk=None):
    files = sorted(root.glob("output_rst.*.nc"))[:n_steps]
    return build.make_kerchunk(
        files, out=root / "refs.json", grid=root / "grid.nc", subchunk=subchunk
    )


def write_run(root, catalogs, steps: int = len(STAMPS), subchunk=None) -> str:
    """Write the synthetic run and its catalog entry; return its nickname.

    ``subchunk`` stores it in smaller chunks than a whole variable (see
    :func:`ocean_skill.build.make_kerchunk`), as a large run's reference is.
    """
    root.mkdir(exist_ok=True)
    _write_grid(root / "grid.nc")
    for i in range(steps):
        _write_restart(root / f"output_rst.{STAMPS[i]}.nc", i)
    refs = _kerchunk(root, steps, subchunk)
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
    assert "at a job join" in step["message"]
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


# ------------------------------------------------------------------------ the reads


def _assert_same_steps(got: pd.DataFrame, expected: pd.DataFrame):
    """Assert two step tables agree: exactly, but for the last bits of float sums."""
    got, expected = got.reset_index(drop=True), expected.reset_index(drop=True)
    assert list(got.columns) == list(expected.columns)
    for column in got.columns:
        if got[column].dtype.kind == "f":
            np.testing.assert_allclose(
                got[column], expected[column], rtol=1e-12, err_msg=column
            )
        else:
            pd.testing.assert_series_equal(got[column], expected[column])


def test_a_step_is_read_in_a_few_large_blocks(tmp_path, isolated_catalogs, result):
    """A step is a bounded number of dask tasks, not several per stored chunk.

    Stored a level by half the rows per chunk, the step is 8 chunks per 3-D variable,
    122 in all; reduced at those chunks it was ~3,900 tasks (~600,000 on a large
    grid, most of the time spent ordering the graph). Reopened in blocks of levels,
    each block is one read of a few tasks.
    """
    nickname = write_run(tmp_path / "run", isolated_catalogs, 2, subchunk=SUBCHUNK)
    tasks = []

    class Count(Callback):
        def _pretask(self, key, dsk, state):
            tasks.append(key)

    with dask.config.set(num_workers=2), Count():
        one = health(nickname, time="latest", cache=False)
    assert len(tasks) < 300, len(tasks)
    # the same numbers as the whole-variable store's scan
    stamp = one.steps["time"].iloc[0]
    _assert_same_steps(one.steps, result.steps[result.steps["time"] == stamp])


def test_the_blocks_do_not_change_the_numbers(
    tmp_path, isolated_catalogs, result, monkeypatch
):
    """A level per block over three threads gives what a whole variable per block does.

    Sums add up across blocks, and the minimum and maximum keep their place in the
    whole variable (the -5 degC cell is in the top level of four).
    """
    nickname = write_run(tmp_path / "run", isolated_catalogs, subchunk=SUBCHUNK)
    monkeypatch.setattr(run_health, "_BLOCK_BYTES", 1)  # never more than a level
    with dask.config.set(num_workers=3):
        thin = health(nickname, cache=False)
    _assert_same_steps(thin.steps, result.steps)


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


def test_summary_is_one_table_with_a_row_per_kind_and_never_reassures(result):
    pages = result.figures("summary")
    assert len(pages) == 1
    text = [t.get_text() for t in pages[0][1].texts]
    flat = [t.replace("\n", " ") for t in text]
    checks = [t for t in flat if t in _CHECKS]
    assert checks == [
        "Check",
        "All zero",
        "Negative concentrations",
        "Outside physical bounds",
        "Exact copies",
        "CDR tracers",
        "Sudden step",
        "Repeated restart",
        "Largest drift",
    ]
    assert not any("no NaN" in t or "no steps" in t for t in text)
    assert any(t == "DIC_ALT_CO2 = DIC" for t in text)
    assert any("DIC − DIC_ALT_CO2 = 0 in every restart" in t for t in text)
    assert any(t.startswith("2011-03-31 → 04-30\nat a job join") for t in text)


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


# ------------------------------------------ the pages, on a hand-made run of problems

PLANKTON = ("spC", "spChl", "diatC", "diatFe", "diazP", "zooC")


def _row(time, variable, kind="3d", **kw) -> dict:
    """One (time, variable) row of a steps table, as a scan would write it."""
    n_ocean = kw.pop("n_ocean", 1000)
    n_neg = kw.pop("n_neg", 0)
    n_bad = kw.pop("n_bad", n_neg)
    n_nan = kw.pop("n_nan", 0)
    value = kw.pop("value", 1.0)
    row = dict.fromkeys(run_health._COLUMNS, np.nan)
    row.update(
        time=time,
        variable=variable,
        kind=kind,
        n_ocean=n_ocean,
        n_nan=n_nan,
        n_inf=0,
        n_neg=n_neg,
        n_bad=n_bad,
        pct_neg=100.0 * n_neg / n_ocean,
        pct_bad=100.0 * n_bad / n_ocean,
        all_zero=kw.pop("all_zero", False),
        copy_of=kw.pop("copy_of", ""),
        total=value,
        sumsq=value**2,
        value=value,
    )
    if kind != "derived" and n_nan != n_ocean:
        row.update(
            min=kw.pop("lo", -0.5 if n_neg else 1.0),
            min_lon=238.0,
            min_lat=41.5,
            min_depth=16.0,
            max=kw.pop("hi", 2.0),
            max_lon=240.0,
            max_lat=42.5,
            max_depth=5.0,
        )
    return row


def stress_run(*, test_mode=False) -> RunHealth:
    """Return a run with 40+ negative tracers, copies, NaN fields and several steps."""
    times = pd.date_range("2013-01-02", periods=14, freq="14D")
    names = ["DON", "DOC", "DOP", "DIC", "ALK", "NO3", "O2", "PO4", "SiO3", "NH4"]
    names += [f"TRC{i:02d}" for i in range(30)] + list(PLANKTON)
    steps_up = {5: +3.0, 9: -2.0, 11: +4.0}  # % jumps at these restarts
    rows = []
    for k, t in enumerate(times):
        for i, name in enumerate(names):
            level = 1000.0 + 7 * i
            level *= (
                1 + 1e-4 * k
            )  # a slow drift, so a step has a typical change to beat
            if name in PLANKTON:
                level *= 0.85**k if k < 5 else 1.0  # spin-up: a steep fall at the start
            else:
                level *= 1 + 0.01 * sum(v for j, v in steps_up.items() if k >= j) * (
                    i % 3 == 0
                )
            neg = 20 * (1 + i) if (k == 3 and i < 40) else 0
            nan = 50 if (name in ("NH4", "TRC03", "TRC04") and k == 6) else 0
            copy = {"TRC10": "DIC", "TRC11": "ALK", "TRC12": "NO3"}.get(name, "")
            rows.append(
                _row(
                    t,
                    name,
                    n_neg=neg,
                    n_nan=nan,
                    value=level,
                    copy_of=copy,
                    all_zero=name in ("TRC20", "TRC21"),
                )
            )
        for name in ("MARBL_ABIO_PH_SURF", "MARBL_ABIO_ALK_SURF", "TRC29_FLUX"):
            rows.append(_row(t, name, "2d", n_nan=1000, value=np.nan))
        for name in ("hbls", "hbbl"):
            rows.append(_row(t, name, "2d", n_bad=11 if k == 2 else 0, value=np.nan))
        rows.append(_row(t, "cdr_ALK", "derived", value=3.2e9 * k))
        rows.append(_row(t, "cdr_DIC", "derived", value=0.0))
        rows.append(_row(t, "KE", "derived", value=5e12 * (1 + 1e-3 * k)))
    steps = pd.DataFrame(rows, columns=list(run_health._COLUMNS))
    joins = [times[2] + pd.Timedelta(days=3), times[5] - pd.Timedelta(days=3)]
    return RunHealth("stress:run", steps, joins=joins, test_mode=test_mode)


def _inside(fig) -> list:
    """Return the texts of ``fig`` that reach outside it (there should be none)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    renderer = FigureCanvasAgg(fig).get_renderer()
    box = fig.bbox
    out = []
    for t in fig.texts:
        e = t.get_window_extent(renderer)
        if e.x0 < box.x0 - 0.5 or e.x1 > box.x1 + 0.5:
            out.append(t.get_text())
        if e.y0 < box.y0 - 0.5 or e.y1 > box.y1 + 0.5:
            out.append(t.get_text())
    return out


@pytest.mark.parametrize("test_mode", [False, True])
def test_summary_of_a_run_with_many_problems_fits_one_letter_page(test_mode):
    result = stress_run(test_mode=test_mode)
    assert len(result.problems()) > 80  # 40 negatives, copies, NaN, steps ...

    pages = result.figures("summary")

    assert len(pages) == 1
    fig = pages[0][1]
    width, height = fig.get_size_inches()
    assert (width, height) == (8.5, pytest.approx(height)) and height <= 11.0
    assert not _inside(fig)
    assert min(t.get_fontsize() for t in fig.texts) >= 8.0
    text = "\n".join(t.get_text() for t in fig.texts)
    for what in (
        "NaN / Inf",
        "Entirely NaN",
        "All zero",
        "Negative",
        "Exact copies",
        "CDR tracers",
        "Sudden step",
        "Largest drift",
        "MARBL_ABIO_PH_SURF",
        "TRC10 = DIC",
        "40 of 48 tracers",
        "+",  # lists are capped, saying how many more
    ):
        assert what in text, what
    assert (
        "ALK − ALK_ALT_CO2" in text and "DIC − DIC_ALT_CO2 = 0 in every restart" in text
    )
    assert ("Test mode" in text) is test_mode


def test_one_row_per_step_date_and_a_join_is_named():
    result = stress_run()
    rows = result._step_rows(
        result.problems() and [p for p in result.problems() if p["kind"] == "step"],
        5,
        8,
    )
    dates = [r[2].split("\n")[0] for r in rows]
    assert len(dates) == len(set(dates)) == 3  # the three planted steps
    assert rows[0][2].endswith("at a job join")  # the join sits just before the first
    assert all("× the typical change" in r[1] or "otherwise flat" in r[1] for r in rows)


def test_living_plankton_pools_are_not_checked_for_steps():
    result = stress_run()
    flagged = set(result._jumps()["variable"])
    assert not flagged & set(PLANKTON)  # a spin-up fall of 15% a day is not a step
    assert {"DIC", "DON"} <= flagged  # the planted steps are
    assert not [p for p in result.problems() if p["variable"] in PLANKTON]
    # still drawn, without marks
    panel = result._panel("spC", "Small phytoplankton C", "%")
    assert panel["flagged"] == [] and len(panel["series"]) == 14
    for name in (
        "spC",
        "spChl",
        "diatC",
        "diatFe",
        "diazP",
        "zooC",
        "diatSi",
        "spCaCO3",
    ):
        assert run_health._LIVING.fullmatch(name), name
    for name in ("Fe", "DOC", "Lig", "NO3", "O2", "DIC"):
        assert not run_health._LIVING.fullmatch(name), name


def test_integrity_table_merges_bad_columns_and_cuts_nothing():
    result = stress_run()
    pages = result.figures("integrity")
    texts = [t.get_text() for fig in (f for _, f in pages) for t in fig.texts]

    assert "Worst %\nbad (date)" in texts and "Bound" in texts and "Latest\n%" in texts
    assert not any("negative" in t or "out of bounds" in t for t in texts)
    assert not any(" / " in t for t in texts)  # the latest percentage is one number
    assert "MARBL_ABIO_PH_SURF" in texts and "MARBL_ABIO_ALK_SURF" in texts
    assert "all NaN" in texts  # an entirely-NaN field says so
    assert "≥ 0" in texts and "0 to depth" in texts
    for _, fig in pages:
        assert not _inside(fig)
        assert fig.get_size_inches()[1] <= 11.0

    _, widths, _ = run_health._lay_out(
        [("Variable", "Flags"), ("MARBL_ABIO_PH_SURF", "all NaN"), ("x", "y")],
        7.9,
        6.5,
        flexible=1,
    )
    assert widths[0] >= run_health._text_width("MARBL_ABIO_PH_SURF", 6.5)
    assert sum(widths) == pytest.approx(7.9)
    table = result.table().set_index("variable")
    assert table.loc["MARBL_ABIO_PH_SURF", "all_nan_steps"] == 14
    assert table.loc["NH4", "all_nan_steps"] == 0
    assert result.problems()[0]["kind"] == "nonfinite"
    assert "all_nan" in [p["kind"] for p in result.problems()]


def test_every_panel_title_is_one_size_in_readable_words(result):
    for draw in ("inventory", "plankton"):
        for _, fig in result.figures(draw):
            titled = [ax for ax in fig.axes if ax.get_title()]
            assert {ax.title.get_fontsize() for ax in titled} == {
                run_health._PANEL_TITLE_PT
            }
            for ax in titled:
                title = ax.get_title()
                assert "inventory" not in title and "(% change)" not in title
                assert "_ALT_CO2" not in title.replace(" minus ALT_CO2", "")
                assert (
                    "degC" not in ax.get_ylabel() and "mmol/m3" not in ax.get_ylabel()
                )
    titles = {
        ax.get_title() for _, fig in result.figures("inventory") for ax in fig.axes
    }
    assert {
        "Alkalinity",
        "Alkalinity (ALT_CO2)",
        "DIC (ALT_CO2)",
        "Ammonium",
        "O2 volume < 60 mmol/m³",
        "Deep temperature (> 1000 m)",
        "DIC minus ALT_CO2",
    } <= titles


def test_a_zero_cdr_difference_is_a_note_and_segments_are_shaded_unlabeled(result):
    ((_, fig),) = result.figures("inventory")
    by = {ax.get_title(): ax for ax in fig.axes}

    zero = by["DIC minus ALT_CO2"]  # DIC_ALT_CO2 is a copy of DIC: nothing to plot
    assert [t.get_text() for t in zero.texts] == ["identical to DIC in every restart"]
    assert not zero.lines and not zero.get_yticks().size
    moving = by["Alkalinity minus ALT_CO2"]  # a constant offset: drawn
    assert moving.lines and not moving.texts

    # alternating bands, no "segment N" labels anywhere, and the caption says so
    assert all(
        not t.get_text().startswith("segment") for ax in fig.axes for t in ax.texts
    )
    assert any(ax.patches for ax in fig.axes if ax.get_title())
    assert run_health.SEGMENT_NOTE in [t.get_text() for t in fig.texts]
    assert "job" in run_health.SEGMENT_NOTE
