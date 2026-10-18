"""River-forcing files: ``build.add_river_source`` and :mod:`ocean_skill.river_inputs`.

The synthetic file follows roms-tools' river layout -- ``river_volume`` on
``(river_time, nriver)``, ``river_tracer`` on ``(river_time, ntracers, nriver)``,
``river_index`` / ``river_fraction`` on the grid, names, an ``abs_time`` in days
since 2012-12-31 -- with 2,000 rivers, 20 times and 8 tracers, and these planted
values (river = file position, time = step):

* NO3 = -5 at river 10, step 4 (negative concentration)
* DOC = 25000 at river 500, step 7 (huge)
* salt = 5 at river 42, step 2 (a salty "river")
* discharge = -3 at river 99, step 1 (negative)
* river 1500 flows but has no cell; river 800 has a cell but no flow and ALK, DIC and
  DOC of 1e6; river 1800 has neither; river 600 has no flow at step 3 and DOC 9e5 there
* zooC is zero everywhere, Lig is 1 everywhere, and DIC is ALK

River 7 enters at two cells (75% and 25%), river 3 at two cells either side of the
dateline (the grid is -180..180).
"""

import gc
import io

import numpy as np
import pytest
import xarray as xr
from matplotlib.figure import Figure
from xarray.backends.file_manager import FILE_CACHE

from ocean_skill import build, river_inputs, sources
from ocean_skill.river_inputs import RIVER_DRAWS, RiverInputs, rivers

NR, NT = 2000, 20
NY = NX = 50
TRACERS = ["temp", "salt", "NO3", "ALK", "DIC", "DOC", "zooC", "Lig"]
UNITS = ["degC", "PSU"] + ["mmol/m3"] * 6
NO_MOUTH = 1499  # river 1500: flow, but no cell in river_index
NO_FLOW = 799  # river 800: a mouth, but zero flow and absurd concentrations
NEITHER = 1799  # river 1800: no cell and no flow
GAPPED = 599  # river 600: flows, except zero flow (and garbage) at step 3
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


@pytest.fixture(autouse=True)
def _release_netcdf_handles():
    """Close every netCDF handle a test opened, on this thread, before the next test.

    ``rivers()`` and ``sources.read`` hand back *lazy* datasets whose dask graphs hold
    open netCDF4/HDF5 handles (xarray's ``FILE_CACHE``), and nothing in a test closes
    them. They were then closed whenever the garbage collector finalized them -- from
    any thread, including a dask worker, while another test's ``open_dataset`` was
    inside the (not thread-safe) HDF5 library. That is the intermittent crash of a
    whole xdist worker, or a read of a half-closed file failing
    ``test_mouths_are_the_cell_positions_weighted_by_fraction``. Finalizing the
    handles here, at a point where no read is running, takes the collector out of it;
    the synchronous scheduler keeps dask worker threads out of the library too.
    """
    import dask

    with dask.config.set(scheduler="synchronous"):
        yield
    gc.collect()
    FILE_CACHE.clear()


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
    index[[NO_MOUTH, NEITHER]] = 0
    frac[[NO_MOUTH, NEITHER]] = 0

    rng = np.random.default_rng(0)
    base = 10 ** rng.uniform(-1, 4, NR)  # m3/s, 5 decades
    step = np.arange(NT)
    volume = base[None, :] * (1 + 0.1 * np.sin(step)[:, None])
    volume[1, 98] = -3.0
    volume[:, [NO_FLOW, NEITHER]] = 0.0
    volume[3, GAPPED] = 0.0
    level = np.array([10.0, 0.05, 20.0, 1500.0, 1500.0, 300.0, 0.0, 1.0])
    tracer = (
        level[None, :, None]
        * (1 + 0.1 * np.cos(step))[:, None, None]
        * (1 + 0.2 * rng.random((NT, len(TRACERS), NR)))
    )
    tracer[4, 2, 9] = -5.0
    tracer[7, 5, 499] = 25000.0  # DOC
    tracer[2, 1, 41] = 5.0
    tracer[:, 3:5, NO_FLOW] = 1e6  # zero-flow river: ALK/DIC absurd
    tracer[:, 5, NO_FLOW] = 1e6
    tracer[3, 5, GAPPED] = 9e5  # no water at that step
    tracer[:, 6] = 0.0  # zooC: zero everywhere
    tracer[:, 7] = 1.0  # Lig: constant
    tracer[:, 4] = tracer[:, 3]  # DIC is ALK
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
        "volume": volume,
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
    a, b = river_files["tmp"] / "part_a.nc", river_files["tmp"] / "part_b.nc"
    with xr.open_dataset(river_files["file"], decode_times=False) as ds:
        ds.isel(river_time=slice(0, 8)).to_netcdf(a)
        ds.isel(river_time=slice(8, None)).to_netcdf(b)
    cat = build.new_catalog()
    build.add_river_source(cat, "joined", [a, b], river_files["grid"])
    build.save(cat, isolated_catalogs / "joined.yaml")
    assert sources.read("joined").sizes["river_time"] == NT
    assert rivers("joined").times is not None


def test_mouths_are_the_cell_positions_weighted_by_fraction(inputs, river_files):
    lon, lat = river_files["lon"].ravel(), river_files["lat"].ravel()
    one = [i for i in range(NR) if i + 1 not in (3, 7) and i not in (NO_MOUTH, NEITHER)]
    np.testing.assert_allclose(inputs.lon[one], lon[one])
    np.testing.assert_allclose(inputs.lat[one], lat[one])
    # river 7: 0.75 at its first cell, 0.25 at the second (no dateline near it)
    cells = [6, SPLIT_7]
    w = np.array([0.75, 0.25])
    np.testing.assert_allclose(inputs.lon[6], (w * lon[cells]).sum(), atol=1e-4)
    np.testing.assert_allclose(inputs.lat[6], (w * lat[cells]).sum())
    assert np.isnan(inputs.lon[[NO_MOUTH, NEITHER]]).all()  # no cell, no mouth


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


def test_rivers_are_classified_by_mouth_and_flow(inputs, river_files):
    cls = inputs.classes()
    assert (
        cls["active"],
        cls["mouth_no_flow"],
        cls["flow_no_mouth"],
        cls["neither"],
    ) == (NR - 3, 1, 1, 1)
    assert not inputs.active[[NO_MOUTH, NO_FLOW, NEITHER]].any()
    q = inputs.discharge
    assert cls["lost_m3s"] == pytest.approx(q[NO_MOUTH])
    assert cls["lost_pct"] == pytest.approx(100 * q[NO_MOUTH] / q[q > 0].sum())
    assert [r["river"] for r in cls["largest"]] == [NAMES[NO_MOUTH]]


def test_planted_values_are_problems(inputs):
    by = {(p["kind"], p["variable"]): p for p in inputs.problems()}
    assert by["below_bound", "NO3"]["count"] == 1
    assert by["below_bound", "NO3"]["worst"] == pytest.approx(-5.0)
    doc = by["above_bound", "DOC"]
    assert doc["count"] == 1 and doc["n_rivers"] == 1  # not the zero-flow 1e6 / 9e5
    assert doc["worst"] == pytest.approx(25000.0)
    assert by["above_bound", "salt"]["worst"] == pytest.approx(5.0)
    assert by["negative_discharge", "river_volume"]["count"] == 1
    assert by["negative_discharge", "river_volume"]["worst"] == pytest.approx(-3.0)
    lost = by["flow_without_mouth", "river_index"]
    assert lost["count"] == 1 and lost["lost_m3s"] == pytest.approx(
        inputs.discharge[NO_MOUTH]
    )
    assert lost["largest"][0]["river"] == NAMES[NO_MOUTH]
    assert NAMES[NO_MOUTH] in lost["message"] and "m3/s" in lost["message"]
    same = by["identical_tracers", "ALK"]
    assert same["equals"] == "DIC" and same["message"] == (
        "river ALK equals river DIC in every river"
    )
    # nothing else is wrong with the file: not the constant or all-zero tracers, and
    # not ALK/DIC at the zero-flow river
    assert set(by) == {
        ("below_bound", "NO3"),
        ("above_bound", "DOC"),
        ("above_bound", "salt"),
        ("negative_discharge", "river_volume"),
        ("flow_without_mouth", "river_index"),
        ("identical_tracers", "ALK"),
    }
    import json

    json.dumps(inputs.problems())  # JSON-safe


def test_planted_values_head_the_extremes_table(inputs, river_files):
    df = inputs.table()
    top = df[(df["kind"] == "highest") & (df["rank"] == 1)].set_index("tracer")
    assert top.loc["DOC", "value"] == pytest.approx(25000.0)
    assert top.loc["DOC", "river"] == NAMES[499]
    assert top.loc["DOC", "date"] == "2013-01-07"  # step 7, days since 2012-12-31
    assert top.loc["DOC", "discharge"] == pytest.approx(river_files["volume"][7, 499])
    assert (
        top.loc["salt", "value"] == pytest.approx(5.0)
        and top.loc["salt", "river"] == NAMES[41]
    )
    assert top.loc["DOC", "unit"] == "mmol/m3" and top.loc["temp", "unit"] == "degC"
    low = df[df["kind"] == "below bound"]
    assert list(low["tracer"]) == ["NO3"]
    row = low.iloc[0]
    assert (
        row["value"] == pytest.approx(-5.0)
        and row["river"] == NAMES[9]
        and row["date"] == "2013-01-04"
    )
    assert row["lon"] == pytest.approx(float(inputs.lon[9]))
    # three rivers per tracer, each once
    three = df[(df["kind"] == "highest") & (df["tracer"] == "DOC")]
    assert len(three) == river_inputs.TOP_N == 3 and three["river_id"].is_unique


def test_zero_flow_concentrations_never_reach_the_extremes(inputs):
    df = inputs.table()
    assert df[["lon", "lat", "discharge"]].notna().all().all()
    assert (df["discharge"] > 0).all()
    assert not {NAMES[NO_FLOW], NAMES[NO_MOUTH], NAMES[NEITHER]} & set(df["river"])
    assert (df["value"] < 1e5).all()
    # the gapped river's 9e5 at its no-flow step is no value either
    assert 9e5 not in set(df["value"])


def test_the_sample_is_twelve_distinct_active_rivers_chosen_by_discharge(
    river_source,
):
    a, b = rivers(river_source).sample_rivers(), rivers(river_source).sample_rivers()
    assert a == b  # deterministic
    idx = [i for i, _ in a]
    assert len(idx) == 12 == len(set(idx))
    inputs = rivers(river_source)
    q = inputs.discharge
    assert not {NO_MOUTH, NO_FLOW, NEITHER} & set(idx)  # active rivers only
    assert [label for _, label in a].count("largest") == 3
    for label in ("small", "medium", "large"):
        assert [lab for _, lab in a].count(label) == 3
    act = np.flatnonzero(inputs.active)
    ranked = act[np.argsort(q[act], kind="stable")]
    n = len(ranked)
    assert {i for i, lab in a if lab == "largest"} == set(ranked[-3:])
    terciles = {
        "small": ranked[: n // 3],
        "medium": ranked[n // 3 : 2 * n // 3],
        "large": ranked[2 * n // 3 :],
    }
    for i, lab in a:
        if lab != "largest":
            assert i in terciles[lab]
    # the median-ranked rivers of each tercile
    mid = (n // 3 + 2 * n // 3) // 2
    assert {i for i, lab in a if lab == "medium"} == set(ranked[mid - 1 : mid + 2])


def test_the_sample_legend_sits_right_under_the_last_panel(inputs):
    ((_, fig),) = inputs.figures("sample")
    fig.canvas.draw()
    (legend,) = fig.legends
    axes_bottom = min(ax.get_position().y0 for ax in fig.axes) * fig.get_figheight()
    legend_top = legend.get_window_extent().y1 / fig.dpi
    assert 0.0 < axes_bottom - legend_top < 0.8  # inches: just the tick labels


@pytest.mark.parametrize("draw", RIVER_DRAWS)
def test_every_draw_returns_renderable_figures(inputs, draw):
    figs = inputs.figures(draw)
    assert figs and all(isinstance(n, str) and isinstance(f, Figure) for n, f in figs)
    assert len({n for n, _ in figs}) == len(figs)
    for _, fig in figs:
        assert fig.get_figwidth() == pytest.approx(8.5)
        fig.savefig(io.BytesIO(), format="png", dpi=30)


def test_histograms_use_active_flowing_rivers_and_skip_empty_tracers(river_source):
    ((_, fig),) = rivers(river_source, time="2013-01-07").figures("histograms")
    # discharge plus the six tracers that vary; zooC and Lig get no panel
    assert len(fig.axes) == 1 + len(TRACERS) - 2
    flagged = {
        ax.get_title().split(" ")[0]: [t.get_text() for t in ax.texts]
        for ax in fig.axes
    }
    assert flagged["DOC"] == ["outside bounds: 1"]  # the planted 25000, at step 7
    # the zero-flow river's 1e6 is no concentration; the others are planted elsewhere,
    # and a panel with nothing outside its bounds says nothing about bounds
    assert flagged["ALK"] == flagged["NO3"] == flagged["salt"] == []
    assert not any("n <= 0" in t.get_text() for ax in fig.axes for t in ax.texts)
    assert any(t.get_text() == "all zero: zooC; constant: Lig = 1" for t in fig.texts)


def test_histogram_panels_are_three_wide_titled_with_display_units(river_source):
    ((_, fig),) = rivers(river_source).figures("histograms")
    titles = [ax.get_title() for ax in fig.axes]
    assert titles[0] == "discharge, time mean (m\u00b3/s)"
    assert "NO3 (mmol/m\u00b3)" in titles and "temp (\u00b0C)" in titles
    first_row = [ax for ax in fig.axes if ax.get_subplotspec().rowspan.start == 0]
    assert len(first_row) == 3 == river_inputs._HIST_COLS
    # a range of several decades has decade ticks, written as powers of ten
    discharge = fig.axes[0]
    assert discharge.get_xscale() == "log"
    fig.canvas.draw()
    labels = [t.get_text() for t in discharge.get_xticklabels() if t.get_text()]
    assert labels and all(lab.startswith("$\\mathdefault{10^") for lab in labels)


def test_a_long_note_is_wrapped_to_stay_inside_the_page():
    note = "all zero: " + ", ".join(f"tracer{i}" for i in range(80))
    wrapped = river_inputs._wrap(note)
    lines = wrapped.split("\n")
    assert len(lines) > 3 and all(len(line) <= river_inputs._WRAP for line in lines)
    assert " ".join(lines) == note
    assert river_inputs._wrap("") == ""


def _inside_the_page(fig):
    """Every text and legend of ``fig`` lies within the figure's own bounds."""
    fig.canvas.draw()
    box = fig.bbox
    items = [*fig.texts, *fig.legends]
    if fig._suptitle is not None:
        items.append(fig._suptitle)
    assert items
    for item in items:
        ext = item.get_window_extent()
        assert ext.x0 >= box.x0 - 1 and ext.x1 <= box.x1 + 1, item
        assert ext.y0 >= box.y0 - 1 and ext.y1 <= box.y1 + 1, item


@pytest.mark.parametrize("draw", RIVER_DRAWS)
def test_figures_have_readable_fonts_and_nothing_runs_off_the_page(inputs, draw):
    from matplotlib.text import Text

    for _, fig in inputs.figures(draw):
        _inside_the_page(fig)
        fig.canvas.draw()
        small = [
            (t.get_text(), t.get_fontsize())
            for t in fig.findobj(Text)
            if t.get_visible() and t.get_text().strip() and t.get_fontsize() < 8.0
        ]
        assert not small, small


def test_the_map_title_is_one_short_line_and_the_counts_are_a_note_below(inputs):
    ((_, fig),) = inputs.figures("map")
    title = fig._suptitle.get_text()
    assert "\n" not in title
    assert title == (
        "River mouths: 1,997 of 2,000 rivers (with a grid cell and flow), sized and "
        "colored by time-mean discharge"
    )
    note = next(t.get_text() for t in fig.texts if t.get_text().startswith("Drawn:"))
    note = " ".join(note.split())  # wrapped over lines
    assert "1 with a grid cell but no flow" in note
    assert "1 with flow but no grid cell of their own" in note
    assert "1 with neither" in note
    assert "lost" not in title.lower() and "lost" not in note.lower()
    # the colorbar says m\u00b3/s at a readable size; marker sizes have a legend
    labels = [ax.get_ylabel() for ax in fig.axes]
    assert "time-mean discharge (m\u00b3/s)" in labels
    legend = next(lg for lg in fig.legends if lg.get_title().get_text())
    assert len(legend.get_texts()) == 3
    assert all(t.get_text().endswith("m\u00b3/s") for t in legend.get_texts())


def test_the_sample_leaves_out_constant_and_all_zero_tracers(river_source, monkeypatch):
    monkeypatch.setattr(
        river_inputs, "SAMPLE_TRACERS", ("temp", "salt", "NO3", "zooC", "Lig")
    )
    ((_, fig),) = rivers(river_source).figures("sample")
    labels = [ax.get_ylabel() for ax in fig.axes]
    assert labels == [
        "discharge (m\u00b3/s)",
        "temp (\u00b0C)",
        "salt (PSU)",
        "NO3 (mmol/m\u00b3)",
    ]


def test_the_constant_tracers_are_found_from_the_scan(inputs):
    constant = {inputs.tracers[k]: v for k, v in inputs.constant_tracers().items()}
    assert constant == {"zooC": 0.0, "Lig": 1.0}


def test_a_constant_tracer_has_no_extremes_but_a_bad_one_still_shows(
    river_source, monkeypatch
):
    inputs = rivers(river_source)
    salt, no3 = inputs.tracers.index("salt"), inputs.tracers.index("NO3")
    monkeypatch.setattr(
        RiverInputs, "constant_tracers", lambda self: {salt: 1.0, no3: 20.0}
    )
    df = inputs.table()
    assert "salt" not in set(df["tracer"])
    # NO3 is "constant" too here, so no highest rows, but its negative value is listed
    assert set(df.loc[df["tracer"] == "NO3", "kind"]) == {"below bound"}


def test_the_extremes_table_fits_two_pages_with_one_cell_for_tracer_and_unit(inputs):
    figs = inputs.figures("extremes")
    assert 1 <= len(figs) <= 2
    ((_, fig),) = figs[:1]
    (ax,) = fig.axes
    (table,) = ax.tables
    cells = {k: c.get_text().get_text() for k, c in table.get_celld().items()}
    heads = [cells[0, c] for c in range(9)]
    assert heads == [
        "tracer",
        "#",
        "value",
        "river",
        "tag",
        "lon",
        "lat",
        "date",
        "Q (m\u00b3/s)",
    ]
    body = [[cells[r, c] for c in range(9)] for r in range(1, 1 + len(cells) // 9 - 1)]
    tracers = [row[0] for row in body if row[0]]
    assert "DOC (mmol/m\u00b3)" in tracers and "temp (\u00b0C)" in tracers
    assert not any(t.startswith(("zooC", "Lig")) for t in tracers)  # constant tracers
    doc = next(row for row in body if row[0] == "DOC (mmol/m\u00b3)")
    assert doc[1:3] == ["1", "25000"]
    # positions read as degrees with a hemisphere, never 0-360 numbers
    assert all(row[5].endswith(("\u00b0E", "\u00b0W", "\u00b0")) for row in body)
    assert all(row[6].endswith(("\u00b0N", "\u00b0S", "\u00b0")) for row in body)
    assert any(row[1] == "low 1" for row in body)  # the planted negative NO3


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
    # the histograms read only the one forcing step
    seen.clear()
    small.figures("histograms")
    assert [s[0] for s in seen] == [1]
    # confirming ALK == DIC rereads in slabs too, never the whole array
    seen.clear()
    small.problems()
    assert seen and max(s[0] for s in seen) == 3


def _marker_positions(ax, gid):
    """Return the map's markers of one kind as ``(lon, lat)`` rows, longitude 0-360."""
    (coll,) = [c for c in ax.collections if c.get_gid() == gid]
    xy = np.asarray(coll.get_offsets())
    return np.column_stack([np.mod(xy[:, 0] + 180.0, 360.0), xy[:, 1]])


def _near(found, lon, lat):
    """Return True when some row of ``found`` is the position ``(lon, lat)``."""
    lon = np.mod(np.asarray(lon, dtype=float), 360.0)
    gap = np.abs(found[:, 0][:, None] - lon[None, :]) + np.abs(
        found[:, 1][:, None] - np.asarray(lat)[None, :]
    )
    return bool((gap.min(axis=0) < 1e-6).all())


def test_the_map_marks_every_sample_river_and_every_extreme_river(inputs):
    ((_, fig),) = inputs.figures("map")
    ax = next(a for a in fig.axes if a.collections and a.texts)
    sample = inputs.tagged_sample()
    assert [t for t, _, _ in sample] == [f"S{n}" for n in range(1, 13)]
    idx = [i for _, i, _ in sample]
    assert _near(
        _marker_positions(ax, "sample_rivers"), inputs.lon[idx], inputs.lat[idx]
    )
    assert len(_marker_positions(ax, "sample_rivers")) == 12
    extremes = inputs.extreme_rivers()
    df = inputs.table()
    highest = df[df["kind"] == "highest"]
    # a river holding several tracers' top values is one marker, one tag
    assert len(extremes) == highest["river_id"].nunique() < len(highest)
    assert [d["tag"] for d in extremes] == [
        f"E{n}" for n in range(1, len(extremes) + 1)
    ]
    found = _marker_positions(ax, "extreme_rivers")
    assert len(found) == len(extremes)
    assert _near(found, [d["lon"] for d in extremes], [d["lat"] for d in extremes])
    # the planted DOC outlier is among them
    assert NAMES[499] in {d["river"] for d in extremes}


def test_map_tags_match_the_sample_legend_and_the_extremes_table(inputs):
    ((_, map_fig),) = inputs.figures("map")
    ax = next(a for a in map_fig.axes if a.collections and a.texts)
    on_map = {tag for t in ax.texts for tag in t.get_text().split()}
    ((_, sample_fig),) = inputs.figures("sample")
    (legend,) = sample_fig.legends
    legend_labels = [t.get_text() for t in legend.get_texts()]
    sample = inputs.tagged_sample()
    for (tag, i, label), text in zip(sample, legend_labels, strict=True):
        assert text == f"{tag} {NAMES[i]} ({label})"
        assert tag in on_map
    figs = inputs.figures("extremes")
    rows = []
    for _, fig in figs:
        (table,) = fig.axes[0].tables
        cells = {k: c.get_text().get_text() for k, c in table.get_celld().items()}
        rows += [
            (cells[r, 3], cells[r, 4], cells[r, 1]) for r in range(1, len(cells) // 9)
        ]
    extremes = {d["river"]: d["tag"] for d in inputs.extreme_rivers()}
    tagged = {(river, tag) for river, tag, rank in rows if tag}
    assert tagged == set(extremes.items())
    assert set(extremes.values()) <= on_map
    # the lowest-value row (a below-bound river) carries no tag
    assert [tag for _, tag, rank in rows if rank.startswith("low")] == [""]
    # one legend entry per marker kind explains them
    texts = [t.get_text() for lg in map_fig.legends for t in lg.get_texts()]
    assert any(t.startswith("S1-S12:") for t in texts)
    assert any(t.startswith("E1-E") for t in texts)


def test_a_river_in_both_lists_gets_one_label_with_both_tags(inputs, monkeypatch):
    first = inputs.tagged_sample()[0][1]
    real = inputs.extreme_rivers

    def both():
        out = real()
        out[0] = {
            **out[0],
            "lon": float(inputs.lon[first]),
            "lat": float(inputs.lat[first]),
        }
        return out

    monkeypatch.setattr(inputs, "extreme_rivers", both)
    ((_, fig),) = inputs.figures("map")
    ax = next(a for a in fig.axes if a.collections and a.texts)
    assert "S1 E1" in {t.get_text() for t in ax.texts}


def test_map_labels_stay_inside_the_axes_and_apart(inputs):
    ((_, fig),) = inputs.figures("map")
    fig.canvas.draw()
    ax = next(a for a in fig.axes if a.collections and a.texts)
    box = ax.get_window_extent()
    texts = [t for t in ax.texts if t.get_text()]
    assert len(texts) >= 12
    extents = [t.get_window_extent() for t in texts]
    for t, ext in zip(texts, extents, strict=True):
        assert ext.x0 >= box.x0 - 1 and ext.x1 <= box.x1 + 1, t.get_text()
        assert ext.y0 >= box.y0 - 1 and ext.y1 <= box.y1 + 1, t.get_text()
    clashes = sum(
        a.overlaps(b) for n, a in enumerate(extents) for b in extents[n + 1 :]
    )
    assert clashes <= 2  # the synthetic mouths are a dense grid


def test_labels_are_placed_around_marks_at_the_axes_corners():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 3))
    try:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        # marks in every corner and one pile of five at the same spot
        marks = [
            (f"S{n}", x, y, "k")
            for n, (x, y) in enumerate(
                [(0, 0), (10, 0), (0, 10), (10, 10)] + [(5, 5)] * 5, start=1
            )
        ]
        texts = river_inputs._label_marks(ax, marks)
        fig.canvas.draw()
        box = ax.get_window_extent()
        extents = [t.get_window_extent() for t in texts]
        for ext in extents:
            assert ext.x0 >= box.x0 - 1 and ext.x1 <= box.x1 + 1
            assert ext.y0 >= box.y0 - 1 and ext.y1 <= box.y1 + 1
        for n, a in enumerate(extents):
            assert not any(a.overlaps(b) for b in extents[n + 1 :])
    finally:
        plt.close(fig)
    assert river_inputs._label_marks(ax, []) == []
