"""River-forcing files: ``build.add_river_source`` and :mod:`ocean_skill.river_inputs`.

The synthetic file follows roms-tools' river layout -- ``river_volume`` on
``(river_time, nriver)``, ``river_tracer`` on ``(river_time, ntracers, nriver)``,
``river_index`` / ``river_fraction`` on the grid, names, an ``abs_time`` in days
since 2012-12-31 -- with 2,000 rivers, 20 times and 6 tracers, and these planted
values (river = file position, time = step):

* NO3 = -5 at river 10, step 4 (negative concentration)
* ALK = 25000 at river 500, step 7 (huge)
* salt = 5 at river 42, step 2 (a salty "river")
* discharge = -3 at river 99, step 1 (negative)

River 7 enters at two cells (75% and 25%), river 3 at two cells either side of the
dateline (the grid is -180..180).
"""

import io

import numpy as np
import pytest
import xarray as xr
from matplotlib.figure import Figure

from ocean_skill import build, river_inputs, sources
from ocean_skill.river_inputs import RIVER_DRAWS, RiverInputs, rivers

NR, NT = 2000, 20
NY = NX = 50
TRACERS = ["temp", "salt", "NO3", "ALK", "DIC", "DOC"]
UNITS = ["degC", "PSU", "mmol/m3", "mmol/m3", "mmol/m3", "mmol/m3"]
NAMES = [f"River {i + 1:04d}" for i in range(NR)]
SPLIT_7 = 2100  # the second cell of river 7 (fraction 0.25)
SPLIT_3 = 2300  # the second cell of river 3, across the dateline


def _grid_arrays():
    lon = (
        172.0 + 0.4 * np.arange(NX)[None, :] + 0 * np.arange(NY)[:, None] + 180.0
    ) % 360.0 - 180.0
    lat = 50.0 + 0.3 * np.arange(NY)[:, None] + 0.001 * np.arange(NX)[None, :]
    lon, lat = lon.copy(), lat.copy()
    lon.ravel()[2], lon.ravel()[SPLIT_3] = 179.5, -179.5
    return lon, lat


@pytest.fixture(scope="module")
def river_files(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("rivers")
    lon, lat = _grid_arrays()
    xr.Dataset(
        {
            "lon_rho": (("eta_rho", "xi_rho"), lon),
            "lat_rho": (("eta_rho", "xi_rho"), lat),
            "h": (("eta_rho", "xi_rho"), np.full((NY, NX), 100.0)),
            "mask_rho": (("eta_rho", "xi_rho"), np.ones((NY, NX))),
        }
    ).to_netcdf(tmp / "grid.nc")

    index = np.zeros(NY * NX)
    frac = np.zeros(NY * NX)
    index[:NR] = np.arange(1, NR + 1)
    frac[:NR] = 1.0
    frac[6], frac[2] = 0.75, 0.5
    index[SPLIT_7], frac[SPLIT_7] = 7, 0.25
    index[SPLIT_3], frac[SPLIT_3] = 3, 0.5

    rng = np.random.default_rng(0)
    base = 10 ** rng.uniform(-1, 4, NR)  # m3/s, 5 decades
    step = np.arange(NT)
    volume = base[None, :] * (1 + 0.1 * np.sin(step)[:, None])
    volume[1, 98] = -3.0
    level = np.array([10.0, 0.05, 20.0, 1500.0, 1500.0, 300.0])
    tracer = (
        level[None, :, None]
        * (1 + 0.1 * np.cos(step))[:, None, None]
        * (1 + 0.2 * rng.random((NT, 6, NR)))
    )
    tracer[4, 2, 9] = -5.0
    tracer[7, 3, 499] = 25000.0
    tracer[2, 1, 41] = 5.0
    days = step.astype("float64")
    ds = xr.Dataset(
        {
            "river_volume": (("river_time", "nriver"), volume, {"units": "m^3/s"}),
            "river_tracer": (("river_time", "ntracers", "nriver"), tracer),
            "river_index": (("eta_rho", "xi_rho"), index.reshape(NY, NX)),
            "river_fraction": (("eta_rho", "xi_rho"), frac.reshape(NY, NX)),
            "river_name": ("nriver", np.array(NAMES)),
            "tracer_name": ("ntracers", np.array(TRACERS)),
            "tracer_unit": ("ntracers", np.array(UNITS)),
        },
        coords={
            "nriver": ("nriver", np.arange(1, NR + 1)),
            "river_time": (
                "river_time",
                4748.0 + days,
                {
                    "long_name": "relative time: days since 2000-01-01 00:00:00",
                    "units": "days",
                },
            ),
            "abs_time": (
                "river_time",
                days,
                {
                    "long_name": "absolute time",
                    "units": "days since 2012-12-31 00:00:00",
                },
            ),
        },
    )
    ds.to_netcdf(tmp / "rivers.nc")
    return {
        "tmp": tmp,
        "grid": tmp / "grid.nc",
        "file": tmp / "rivers.nc",
        "lon": lon,
        "lat": lat,
    }


@pytest.fixture
def river_source(river_files, isolated_catalogs):
    cat = build.new_catalog(title="rivers")
    added = build.add_river_source(
        cat, "rivers_a", river_files["file"], river_files["grid"]
    )
    assert added == ["rivers_a"]
    build.save(cat, isolated_catalogs / "rivers.yaml")
    return "rivers_a"


@pytest.fixture
def inputs(river_source):
    return rivers(river_source)


def test_the_entry_is_an_ordinary_lazy_source_that_names_its_grid(
    river_source, river_files
):
    meta = sources.resolve(river_source).metadata
    assert meta["river_grid"] == str(river_files["grid"].resolve())
    ds = sources.read(river_source)
    assert ds["river_tracer"].dims == ("river_time", "ntracers", "nriver")
    assert ds.sizes["nriver"] == NR and ds.sizes["river_time"] == NT


def test_a_file_without_river_variables_is_refused(river_files, isolated_catalogs):
    with pytest.raises(ValueError, match="river-forcing"):
        build.add_river_source(
            build.new_catalog(), "x", river_files["grid"], river_files["grid"]
        )


def test_a_list_of_files_joins_along_river_time(river_files, isolated_catalogs):
    ds = xr.open_dataset(river_files["file"], decode_times=False)
    a, b = river_files["tmp"] / "part_a.nc", river_files["tmp"] / "part_b.nc"
    ds.isel(river_time=slice(0, 8)).to_netcdf(a)
    ds.isel(river_time=slice(8, None)).to_netcdf(b)
    cat = build.new_catalog()
    build.add_river_source(cat, "joined", [a, b], river_files["grid"])
    build.save(cat, isolated_catalogs / "joined.yaml")
    assert sources.read("joined").sizes["river_time"] == NT
    assert rivers("joined").times is not None


def test_mouths_are_the_cell_positions_weighted_by_fraction(inputs, river_files):
    lon, lat = river_files["lon"].ravel(), river_files["lat"].ravel()
    one = [i for i in range(NR) if i + 1 not in (3, 7)]
    np.testing.assert_allclose(inputs.lon[one], lon[one])
    np.testing.assert_allclose(inputs.lat[one], lat[one])
    # river 7: 0.75 at its first cell, 0.25 at the second (no dateline near it)
    cells = [6, SPLIT_7]
    w = np.array([0.75, 0.25])
    np.testing.assert_allclose(inputs.lon[6], (w * lon[cells]).sum(), atol=1e-4)
    np.testing.assert_allclose(inputs.lat[6], (w * lat[cells]).sum())


def test_a_river_across_the_dateline_sits_on_it(inputs, river_files):
    lat = river_files["lat"].ravel()
    # 179.5 and -179.5 average to the dateline, not to longitude 0
    assert abs(abs(inputs.lon[2]) - 180.0) < 1e-6
    np.testing.assert_allclose(inputs.lat[2], (lat[2] + lat[SPLIT_3]) / 2)


def test_mouths_follow_a_0_360_grids_convention():
    lon, lat = np.array([[179.0, 181.0, 90.0]]), np.array([[10.0, 12.0, 0.0]])
    got_lon, got_lat = river_inputs.river_mouths(
        [[1, 1, 0]], [[1.0, 1.0, 0.0]], lon, lat, [1, 2]
    )
    assert got_lon[0] == pytest.approx(180.0) and got_lat[0] == pytest.approx(11.0)
    assert np.isnan(got_lon[1]) and np.isnan(got_lat[1])  # river 2 enters nowhere


def test_planted_values_are_problems(inputs):
    by = {(p["kind"], p["variable"]): p for p in inputs.problems()}
    assert by["below_bound", "NO3"]["count"] == 1
    assert by["below_bound", "NO3"]["worst"] == pytest.approx(-5.0)
    assert (
        by["above_bound", "ALK"]["count"] == 1
        and by["above_bound", "ALK"]["n_rivers"] == 1
    )
    assert by["above_bound", "ALK"]["worst"] == pytest.approx(25000.0)
    assert by["above_bound", "salt"]["worst"] == pytest.approx(5.0)
    assert by["negative_discharge", "river_volume"]["count"] == 1
    assert by["negative_discharge", "river_volume"]["worst"] == pytest.approx(-3.0)
    # nothing else is wrong with the file
    assert set(by) == {
        ("below_bound", "NO3"),
        ("above_bound", "ALK"),
        ("above_bound", "salt"),
        ("negative_discharge", "river_volume"),
    }
    import json

    json.dumps(inputs.problems())  # JSON-safe


def test_planted_values_head_the_extremes_table(inputs):
    df = inputs.table()
    top = df[(df["kind"] == "highest") & (df["rank"] == 1)].set_index("tracer")
    assert top.loc["ALK", "value"] == pytest.approx(25000.0)
    assert top.loc["ALK", "river"] == NAMES[499]
    assert top.loc["ALK", "date"] == "2013-01-07"  # step 7, days since 2012-12-31
    assert (
        top.loc["salt", "value"] == pytest.approx(5.0)
        and top.loc["salt", "river"] == NAMES[41]
    )
    assert top.loc["ALK", "unit"] == "mmol/m3" and top.loc["temp", "unit"] == "degC"
    low = df[df["kind"] == "below bound"]
    assert list(low["tracer"]) == ["NO3"]
    row = low.iloc[0]
    assert (
        row["value"] == pytest.approx(-5.0)
        and row["river"] == NAMES[9]
        and row["date"] == "2013-01-04"
    )
    assert row["lon"] == pytest.approx(float(inputs.lon[9]))
    # ten rivers per tracer, each once
    ten = df[(df["kind"] == "highest") & (df["tracer"] == "DOC")]
    assert len(ten) == 10 and ten["river_id"].is_unique


def test_the_sample_is_twelve_distinct_rivers_chosen_by_discharge(river_source):
    a, b = rivers(river_source).sample_rivers(), rivers(river_source).sample_rivers()
    assert a == b  # deterministic
    idx = [i for i, _ in a]
    assert len(idx) == 12 == len(set(idx))
    q = rivers(river_source).discharge
    assert [label for _, label in a].count("largest") == 3
    for label in ("small", "medium", "large"):
        assert [lab for _, lab in a].count(label) == 3
    biggest = set(np.argsort(q)[-3:])
    assert {i for i, lab in a if lab == "largest"} == biggest
    ranked = np.argsort(q)
    terciles = {
        "small": ranked[: NR // 3],
        "medium": ranked[NR // 3 : 2 * NR // 3],
        "large": ranked[2 * NR // 3 :],
    }
    for i, lab in a:
        if lab != "largest":
            assert i in terciles[lab]
    # the median-ranked rivers of each tercile
    mid = (NR // 3 + 2 * NR // 3) // 2
    assert {i for i, lab in a if lab == "medium"} == set(ranked[mid - 1 : mid + 2])


@pytest.mark.parametrize("draw", RIVER_DRAWS)
def test_every_draw_returns_renderable_figures(inputs, draw):
    figs = inputs.figures(draw)
    assert figs and all(isinstance(n, str) and isinstance(f, Figure) for n, f in figs)
    assert len({n for n, _ in figs}) == len(figs)
    for _, fig in figs:
        assert fig.get_figwidth() == pytest.approx(8.5)
        fig.savefig(io.BytesIO(), format="png", dpi=30)


def test_histograms_count_values_outside_bounds_at_the_forcing_time(river_source):
    ((_, fig),) = rivers(river_source, time="2013-01-07").figures("histograms")
    assert len(fig.axes) == 1 + len(TRACERS)  # discharge plus a panel per tracer
    outside = {
        ax.get_title().split(" ")[0]: ax.texts[0].get_text().split("\n")[0]
        for ax in fig.axes
    }
    assert outside["ALK"] == "outside bounds: 1"  # the planted 25000, at step 7
    assert outside["NO3"] == outside["salt"] == "outside bounds: 0"  # planted elsewhere


def test_the_forcing_time_is_the_nearest_step(river_source):
    assert rivers(river_source).time_index == 0
    assert rivers(river_source, time="2013-01-05").time_index == 5
    assert rivers(river_source, time="2013-01-05 20:00").time_index == 6
    with pytest.raises(ValueError, match="Unknown river draw"):
        rivers(river_source).figures("nope")


def test_the_tracer_array_is_only_ever_read_in_slabs(river_source, monkeypatch):
    whole = rivers(river_source).table()
    monkeypatch.setattr(river_inputs, "SLAB_ELEMENTS", 3 * len(TRACERS) * NR)  # 3 steps
    seen = []
    real = RiverInputs._tracer_slab

    def spy(self, t0, t1):
        out = real(self, t0, t1)
        seen.append(out.shape)
        return out

    monkeypatch.setattr(RiverInputs, "_tracer_slab", spy)
    small = rivers(river_source)
    pytest.importorskip("pandas").testing.assert_frame_equal(small.table(), whole)
    assert len(seen) == 7 and max(s[0] for s in seen) == 3  # ceil(20 / 3) slabs
    assert all(s[0] < NT for s in seen)
    # the histograms read the one forcing step; problems() reuses the scan
    seen.clear()
    small.figures("histograms")
    small.problems()
    assert [s[0] for s in seen] == [1]
