"""``ocean-skill-suite new mid-run-check|input-check`` writes a suite that loads.

Fake run folders are built in ``tmp_path`` in the two layouts C-Star writes on the HPC
(PACMED: one task per month; CCS: named tasks plus a gathered ``rst_files/`` folder):
empty files for the restarts, and a tiny real netCDF for the grid, whose ``eta_rho``
length the command reads. The shipped template is the only copy of the suite, so the
generated file must keep its pages and everything after the setup block.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import yaml

from ocean_skill.config import SuiteConfig
from ocean_skill.workflows import new

RST = "output_rst.{}.nc"


def _grid(path: Path, n_eta: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dims, shape = ("eta_rho", "xi_rho"), (n_eta, 3)
    lat, lon = np.meshgrid(
        np.linspace(10, 20, n_eta), np.linspace(120, 125, 3), indexing="ij"
    )
    xr.Dataset(
        {"h": (dims, np.zeros(shape)), "lon_rho": (dims, lon), "lat_rho": (dims, lat)}
        | {"mask_rho": (dims, np.ones(shape))}
    ).to_netcdf(path)


def _task(root: Path, task: str, stamps: list[str], n_eta: int = 962, grid="a_grid.nc"):
    t = root / "tasks" / task
    (t / "work").mkdir(parents=True, exist_ok=True)
    _grid(t / "input" / "input_datasets" / grid, n_eta)
    for s in stamps:
        (t / "output").mkdir(parents=True, exist_ok=True)
        (t / "output" / RST.format(s)).touch()
    return t


def pacmed(tmp_path: Path, workplan: bool = True) -> Path:
    """Two monthly tasks, a workplan beside tasks/ and a resolved blueprint."""
    root = tmp_path / "pacmed_run"
    tasks = [_task(root, "m201101", ["20110131234500"])]
    tasks.append(_task(root, "m201102", ["20110228234500"]))
    if workplan:
        steps = [{"working_dir": str(t)} for t in tasks]
        (root / "PAC_transformed.yaml").write_text(yaml.safe_dump({"steps": steps}))
        (tasks[0] / "work" / "B_run.ovrd.yaml").write_text(
            yaml.safe_dump({"forcing": {"surface": []}})
        )
    return root


def ccs(tmp_path: Path) -> Path:
    """Named tasks (673 rows, prime) with a gathered ``rst_files/``, no workplan."""
    root = tmp_path / "ccs_run"
    _task(root, "jan_1wk", ["20130102000000", "20130103000000"], 673, "ccs_grid.nc")
    _task(root, "feb_ccs", ["20130201000000"], 673, "ccs_grid.nc")
    for s in ("20130102000000", "20130201000000"):
        (root / "tasks" / "rst_files").mkdir(exist_ok=True)
        (root / "tasks" / "rst_files" / RST.format(s)).touch()
    return root


def _new(tmp_path, *runs, extra=(), template="mid-run-check"):
    """Run ``suite new``; return its exit code and the path it wrote."""
    out = tmp_path / "suite.yaml"
    argv = ["new", template, "--name", "my-run", "-o", str(out), *extra]
    argv += ["--output-dir", str(tmp_path / "diag"), "--shared-catalogs", str(tmp_path)]
    for run in runs:
        argv += ["--run", str(run)]
    return new.main(argv), out


def _source(out: Path) -> dict:
    return yaml.safe_load(out.read_text())["refresh"]["sources"][0]


def test_pacmed_layout_is_written_and_loads(tmp_path, capsys):
    root = pacmed(tmp_path)
    rc, out = _new(tmp_path, root)
    assert rc == 0

    suite = SuiteConfig.model_validate(yaml.safe_load(out.read_text()))
    diag = str(tmp_path / "diag")
    assert suite.output_dir == suite.cache_dir == diag
    assert suite.refresh.catalog == f"{diag}/catalogs/my-run.yaml"
    src = suite.refresh.sources[0]
    assert src.name == "my_run" and src.keep == "latest-per-file"
    assert src.files == str(root / "tasks" / "*" / "output" / "output_rst.*.nc")
    assert src.grid == str(
        root / "tasks" / "m201101" / "input/input_datasets/a_grid.nc"
    )
    assert src.ref == f"{diag}/refs/my_run.parquet"
    assert suite.defaults["test"] == "my_run"
    assert suite.defaults["period"] == {"min": "2011-01-31"}
    assert suite.run_info.workplan == str(root / "PAC_transformed.yaml")
    assert suite.run_info.blueprint.endswith("B_run.ovrd.yaml")
    assert "TODO" not in out.read_text() and f"ocean-skill-run {out}" in out.read_text()
    assert "--smoke 2 --self-check --strict" in capsys.readouterr().out


def test_ccs_layout_prefers_the_gathered_folder(tmp_path):
    root = ccs(tmp_path)
    assert new.find_restarts([root]) == [str(root / "tasks/rst_files/output_rst.*.nc")]
    assert new.find_grid([root]).name == "ccs_grid.nc"
    gathered = root / "gathered_output"
    gathered.mkdir()
    (gathered / RST.format("20130102000000")).touch()
    assert new.find_restarts([root]) == [str(gathered / "output_rst.*.nc")]


def test_several_run_segments_become_a_list_of_globs(tmp_path):
    a, b = tmp_path / "seg_a", tmp_path / "seg_b"
    _task(a, "t1", ["20120131234500"])
    _task(b, "t2", ["20120229234500"])
    rc, out = _new(tmp_path, a, b)
    assert rc == 0
    assert _source(out)["files"] == [
        str(r / "tasks" / "*" / "output" / "output_rst.*.nc") for r in (a, b)
    ]
    assert yaml.safe_load(out.read_text())["defaults"]["period"] == {
        "min": "2012-01-31"
    }


def test_files_flag_names_the_restarts(tmp_path):
    root = pacmed(tmp_path)
    mine = str(root / "tasks" / "m201102" / "output" / "output_rst.*.nc")
    rc, out = _new(tmp_path, root, extra=["--files", mine])
    assert rc == 0 and _source(out)["files"] == mine


def test_grid_must_be_one_file(tmp_path):
    root = pacmed(tmp_path)
    _grid(root / "tasks/m201102/input/input_datasets/other_grid.nc", 10)
    with pytest.raises(new.SuiteNewError, match=r"other_grid.nc.*--grid"):
        new.find_grid([root])
    assert _new(tmp_path, root, extra=["--grid", str(tmp_path / "x.nc")])[0] == 2


def test_no_restarts_is_an_error(tmp_path, capsys):
    (tmp_path / "empty" / "tasks").mkdir(parents=True)
    rc, out = _new(tmp_path, tmp_path / "empty")
    assert rc == 2 and not out.exists() and "--files" in capsys.readouterr().err


@pytest.mark.parametrize(
    "rows, expected",
    [(962, 74), (1346, 673), (674, 337), (673, None), (30, None)],
)
def test_subchunk_divides_eta_rho(tmp_path, rows, expected):
    """A divisor in range wins (962 -> 74), else the smallest one of 32 or more."""
    grid = tmp_path / "g.nc"
    _grid(grid, rows)
    sub, n = new.pick_subchunk(grid)
    assert n == rows and sub.get("eta_rho") == expected and sub["s_rho"] == 1


def test_no_divisor_gives_whole_rows_and_a_warning(tmp_path, capsys):
    rc, out = _new(tmp_path, ccs(tmp_path))
    assert rc == 0
    assert _source(out)["subchunk"] == {"s_rho": 1, "s_w": 1}
    assert "rows stay whole" in out.read_text()
    assert "no divisor" in capsys.readouterr().err


def test_missing_workplan_is_a_todo_not_a_guess(tmp_path, capsys):
    rc, out = _new(tmp_path, pacmed(tmp_path, workplan=False))
    assert rc == 0 and "run_info" in yaml.safe_load(out.read_text())
    assert "# TODO check: run_info.workplan" in out.read_text()
    assert "TODO check" in capsys.readouterr().out


def test_an_existing_file_is_not_overwritten_without_force(tmp_path, capsys):
    root = pacmed(tmp_path)
    out = _new(tmp_path, root)[1]
    out.write_text("keep me")
    assert _new(tmp_path, root)[0] == 2 and out.read_text() == "keep me"
    assert "--force" in capsys.readouterr().err
    assert (
        _new(tmp_path, root, extra=["--force"])[0] == 0 and out.read_text() != "keep me"
    )


def test_other_templates_are_not_supported_yet(tmp_path, capsys):
    rc, out = _new(tmp_path, pacmed(tmp_path), template="final-report")
    err = capsys.readouterr().err
    assert rc == 2 and not out.exists()
    assert "not supported yet" in err and "templates/final-report.yaml" in err


def test_a_missing_required_flag_is_named(tmp_path, capsys):
    assert new.main(["new", "mid-run-check", "--name", "x", "--output-dir", "d"]) == 2
    assert "--run" in capsys.readouterr().err


def test_the_generated_suite_has_the_template_pages(tmp_path):
    rc, out = _new(tmp_path, pacmed(tmp_path))
    assert rc == 0
    source = new.TEMPLATE.read_text()
    generated = out.read_text()
    shipped = yaml.safe_load(source)
    mine = yaml.safe_load(generated)
    assert mine["pages"] == shipped["pages"]
    for key in ("depths", "plot", "snapshot_depths"):  # defaults that are not per run
        assert mine["defaults"][key] == shipped["defaults"][key]
    # only the setup block changes, so the output can be regenerated and diffed
    end = "# <<< setup\n"
    assert generated.count("# >>> setup") == generated.count(end) == 1
    assert generated.split(end, 1)[1] == source.split(end, 1)[1]


def test_quoting_keeps_odd_paths_loadable():
    for odd in ("/a b/c", "/a:b/c", "/a #b", "123", "yes", "/ok/path_1.nc", "*x"):
        assert yaml.safe_load(new._q(odd)) == odd


def _nc(path: Path, name: str, dims: tuple[str, ...]) -> None:
    """Write a netCDF with one variable ``name`` of size-2 dimensions ``dims``."""
    data = {name: (dims, np.zeros((2,) * len(dims)))}
    xr.Dataset(data, coords={dims[0]: np.arange(2)}).to_netcdf(path)


def inputs(tmp_path: Path, restart: bool) -> Path:
    """Build an ``input_datasets`` folder, PACMED-like (restart IC) or CCS-like."""
    root = tmp_path / ("pac" if restart else "ccs")
    rows = ["eta_rho", "xi_rho"]
    folder = root / "tasks" / "first" / "input" / "input_datasets"
    _grid(folder / "x_grid.nc", 1346)
    xr.Dataset(
        {
            "temp_south": (("bry_time", "xi_rho"), np.zeros((2, 2))),
            "temp_west": (("bry_time", "eta_rho"), np.zeros((2, 2))),
        }
    ).to_netcdf(folder / "x_boundary-physics_201101.nc")
    (folder / "x_boundary-physics_201102.nc").touch()
    _nc(folder / "x_surface-physics_201101.nc", "Tair", ["time", *rows])
    _nc(folder / "x_surface-bgc-mbl_co2_2011.nc", "pco2_air", ["xco2_time", *rows])
    _nc(folder / "x_surface-bgc-unified_clim.nc", "iron", ["month", *rows])
    for f in ("x_boundary-bgc-esper-alk_201101", "x_boundary-bgc-unified-chl_clim",
              "x_river", "x_nesting"):  # fmt: skip
        (folder / f"{f}.nc").touch()
    ic = "output_rst.20101231234500" if restart else "x_initial_conditions"
    (folder / f"{ic}.nc").touch()
    return root


@pytest.mark.parametrize("restart", [True, False])
def test_input_check_finds_the_inputs(tmp_path, restart):
    root = inputs(tmp_path, restart)
    rc, out = _new(tmp_path, root, template="input-check")
    assert rc == 0
    suite = SuiteConfig.model_validate(yaml.safe_load(out.read_text()))
    assert suite.name == "input-check" and suite.defaults["test"] == "my_run_ic"
    assert "&id" not in out.read_text()  # rows share no dicts, so no YAML anchors
    assert "\\x" not in out.read_text()  # degree signs written as themselves
    kinds = {f.name: f.kind for f in suite.refresh.forcing}
    assert kinds["my_run_bry_phys"] == "boundary" and kinds["my_run_river"] == "river"
    assert ("my_run_ic" in kinds) == (not restart)  # a restart IC is a plain source
    assert [s.name for s in suite.refresh.sources] == (["my_run_ic"] if restart else [])
    boundary = {f.name: f for f in suite.refresh.forcing}["my_run_bry_bgc"]
    assert boundary.files.endswith("x_boundary-bgc-esper-alk_??????.nc")
    assert boundary.sides == ["south", "west"]
    d = suite.defaults
    assert (d["forcing_time"], d["forcing_month"]) == ("2011-01", 1)
    assert [s["name"] for s in d["sides"]] == ["south", "west"]
    assert d["sides"][0]["transect"] == {"eta_rho": 0}
    assert d["sides"][1]["transect"] == {"xi_rho": 0}
    assert d["sides"][0]["label"] == "lon 120°E to 125°E, lat 10°N to 10°N"
    assert [s["vars"] for s in d["surface"]] == [["Tair"], ["pco2_air"]]
    co2 = kinds["my_run_surf_bgc_mbl_co2"]
    assert co2 == "surface" and [s["vars"] for s in d["surface_clim"]] == [["iron"]]
    entries = {f.name: f for f in suite.refresh.forcing}
    assert entries["my_run_surf_bgc_mbl_co2"].time_coord == "xco2_time"
    assert entries["my_run_surf_bgc_unified_clim"].climatology
    if restart:  # 1346 rows = 2 x 673
        assert suite.refresh.sources[0].subchunk == {
            "s_rho": 1,
            "s_w": 1,
            "eta_rho": 673,
        }


def test_input_check_takes_an_inputs_folder_and_keeps_the_pages(tmp_path):
    folder = inputs(tmp_path, True) / "tasks" / "first" / "input" / "input_datasets"
    out = tmp_path / "suite.yaml"
    argv = [
        "new",
        "input-check",
        "--inputs",
        str(folder),
        "--name",
        "r",
        "-o",
        str(out),
    ]
    argv += ["--output-dir", str(tmp_path / "d"), "--shared-catalogs", str(tmp_path)]
    assert new.main(argv) == 0 and "run_info" not in yaml.safe_load(out.read_text())
    shipped = new.TEMPLATE.with_name("input-check.yaml").read_text()
    assert yaml.safe_load(out.read_text())["pages"] == yaml.safe_load(shipped)["pages"]
    assert new.main(["new", "input-check", "--name", "r", "--output-dir", "d"]) == 2


def test_input_check_picks_the_task_whose_restarts_start_first(tmp_path):
    root = tmp_path / "run"
    for task, stamp in (("a_late", "20130201000000"), ("z_early", "20130102000000")):
        _task(root, task, [stamp])
    assert new._first_inputs([root]).parents[1].name == "z_early"


def test_input_check_pages_expand_one_per_open_side(tmp_path):
    from ocean_skill.workflows import pages

    rc, out = _new(tmp_path, inputs(tmp_path, False), template="input-check")
    assert rc == 0
    titles = [
        p.title
        for p in pages.expand(
            SuiteConfig.model_validate(yaml.safe_load(out.read_text()))
        )
    ]
    for side in ("South", "West"):
        assert (
            f"{side} boundary — temperature and salinity vs WOA23 (2011-01)" in titles
        )
    assert not any(t.startswith(("North", "East")) for t in titles)
    assert "Surface forcing — surface physics (2011-01 mean)" in titles


def test_input_check_starts_at_the_initial_conditions_own_time(tmp_path):
    # roms-tools pads boundary files a month before the start: the IC's ini_time wins
    root = inputs(tmp_path, False)
    folder = root / "tasks" / "first" / "input" / "input_datasets"
    (folder / "x_boundary-physics_201012.nc").write_bytes(
        (folder / "x_boundary-physics_201101.nc").read_bytes()
    )
    ic = xr.Dataset({"temp": ("s_rho", np.zeros(2))})
    ic.attrs["ini_time"] = "2011-01-01 00:00:00"
    ic.to_netcdf(folder / "x_initial_conditions.nc")
    rc, out = _new(tmp_path, root, template="input-check")
    assert rc == 0
    defaults = yaml.safe_load(out.read_text())["defaults"]
    assert (defaults["forcing_time"], defaults["ic_month"]) == ("2011-01", 1)


def test_input_check_leaves_out_a_side_that_is_mostly_land(tmp_path):
    root = inputs(tmp_path, False)
    grid = root / "tasks" / "first" / "input" / "input_datasets" / "x_grid.nc"
    g = xr.load_dataset(grid)
    g["mask_rho"][:, 0] = 0.0  # the west edge is all land
    g.to_netcdf(grid)
    rc, out = _new(tmp_path, root, template="input-check")
    assert rc == 0
    defaults = yaml.safe_load(out.read_text())["defaults"]
    assert [side["name"] for side in defaults["sides"]] == ["south"]


# -- the climatological boundary file, salinity restoring, labels ---------------------


def _folder(root: Path) -> Path:
    return root / "tasks" / "first" / "input" / "input_datasets"


def _entries(out: Path) -> dict:
    suite = SuiteConfig.model_validate(yaml.safe_load(out.read_text()))
    return {f.name: f for f in [*suite.refresh.sources, *suite.refresh.forcing]}


def _expanded(out: Path):
    from ocean_skill.workflows import pages

    suite = SuiteConfig.model_validate(yaml.safe_load(out.read_text()))
    return suite, pages.expand(suite)


def test_the_climatological_boundary_file_is_catalogued_with_its_own_entry(tmp_path):
    root = inputs(tmp_path, False)
    rc, out = _new(tmp_path, root, template="input-check")
    assert rc == 0
    entries = _entries(out)
    clim = entries["my_run_bry_bgc_clim"]
    assert clim.kind == "boundary" and clim.climatology
    assert clim.files.endswith("x_boundary-bgc-unified-chl_clim.nc")
    assert clim.sides == ["south", "west"]
    assert clim.subchunk is None  # one file: no monthly chunks to join
    # the dated BGC entry still reads the monthly files, never the climatology
    assert "clim" not in Path(entries["my_run_bry_bgc"].files).name
    d = yaml.safe_load(out.read_text())["defaults"]
    assert d["bry_bgc_clim"] == "my_run_bry_bgc_clim"
    assert [s["name"] for s in d["clim_sides"]] == ["south", "west"]
    assert "&id" not in out.read_text()


def test_the_climatology_pages_are_model_only_sections_one_per_open_side(tmp_path):
    rc, out = _new(tmp_path, inputs(tmp_path, False), template="input-check")
    assert rc == 0
    _, expanded = _expanded(out)
    clim = [p for p in expanded if " boundary — iron" in p.title]
    assert [p.title.split(" boundary")[0] for p in clim] == ["South", "West"]
    page = clim[0]
    assert page.kind == "field"
    assert page.kwargs["source"] == "my_run_bry_bgc_clim_south"
    assert page.kwargs["variable"][:3] == ["iron", "ammonium", "DOC"]
    select = page.kwargs["select"]
    assert select["month"] == 1 and select["transect"] == {"eta_rho": 0}
    assert len(select["depth"]) == 12  # the upper 1000 m
    assert len(page.kwargs["variable"]) <= 4  # rows per figure


def test_no_climatological_file_means_no_climatology_pages(tmp_path):
    root = inputs(tmp_path, False)
    (_folder(root) / "x_boundary-bgc-unified-chl_clim.nc").unlink()
    rc, out = _new(tmp_path, root, template="input-check")
    assert rc == 0
    assert "my_run_bry_bgc_clim" not in _entries(out)
    assert yaml.safe_load(out.read_text())["defaults"]["clim_sides"] == []
    _, expanded = _expanded(out)
    assert not any(" boundary — iron" in p.title for p in expanded)


def test_a_salinity_restoring_file_is_catalogued_and_compared_with_woa23(tmp_path):
    root = inputs(tmp_path, False)
    _nc(
        _folder(root) / "x_surface-restoring_clim.nc",
        "sss",
        ["month", "eta_rho", "xi_rho"],
    )
    rc, out = _new(tmp_path, root, template="input-check")
    assert rc == 0
    entry = _entries(out)["my_run_surf_restoring_clim"]
    assert entry.kind == "surface" and entry.climatology
    assert entry.label == "salinity restoring climatology"
    d = yaml.safe_load(out.read_text())["defaults"]
    assert d["restoring"] == [
        {"name": "my_run_surf_restoring_clim", "label": entry.label, "var": "sss"}
    ]
    # drawn on its own WOA23 page only, not again as a plain surface map
    assert "my_run_surf_restoring_clim" not in [s["name"] for s in d["surface_clim"]]
    _, expanded = _expanded(out)
    (page,) = [p for p in expanded if p.title.startswith("Salinity restoring vs WOA23")]
    assert page.kind == "compare"
    assert page.kwargs["test"] == "my_run_surf_restoring_clim"
    assert page.kwargs["reference"] == "woa23_salinity_month01"
    assert page.kwargs["select"] == {
        "test": {"month": 1},
        "reference": {"depth": "surface"},
    }


def test_no_restoring_file_means_the_restoring_page_is_skipped_cleanly(tmp_path):
    rc, out = _new(tmp_path, inputs(tmp_path, False), template="input-check")
    assert rc == 0
    assert yaml.safe_load(out.read_text())["defaults"]["restoring"] == []
    _, expanded = _expanded(out)
    assert not any("restoring" in p.title.lower() for p in expanded)
    # (the sources are not catalogued here, so the pages that remain only say that)
    assert all("Unknown source" in p.problem for p in expanded if p.problem)


@pytest.mark.parametrize("restart", [True, False])
def test_every_generated_entry_carries_a_label_for_figures(tmp_path, restart):
    root = inputs(tmp_path, restart)
    _nc(
        _folder(root) / "x_surface-restoring_clim.nc",
        "sss",
        ["month", "eta_rho", "xi_rho"],
    )
    rc, out = _new(tmp_path, root, template="input-check")
    assert rc == 0
    labels = {n: e.label for n, e in _entries(out).items()}
    assert all(labels.values()), labels
    assert labels["my_run_ic"] == "initial conditions"
    assert labels["my_run_bry_phys"] == labels["my_run_bry_bgc"] == "boundary file"
    assert labels["my_run_bry_bgc_clim"] == "climatological boundary file"
    assert labels["my_run_river"] == "river forcing"
    assert labels["my_run_surf_physics"] == "surface physics"
    assert labels["my_run_surf_bgc_unified_clim"] == "surface BGC unified climatology"


def test_a_boundary_figure_names_its_side_and_location(tmp_path):
    rc, out = _new(tmp_path, inputs(tmp_path, False), template="input-check")
    assert rc == 0
    _, expanded = _expanded(out)
    # the section figures; the boundary integrity tables are titled by the page alone
    figures = [p for p in expanded if " boundary — " in p.title and p.kind != "inputs"]
    assert figures
    for p in figures:
        side = p.title.split(" boundary")[0]
        title = p.plot["title"]
        assert title.startswith(f"{side} boundary — ")
        assert "(lon " not in title, title  # the location is in the caption
        assert "2011-01" in title or "month of" in title


def test_the_labels_flow_into_the_config_and_loads(tmp_path):
    from ocean_skill.config import RefreshForcingConfig, RefreshSourceConfig

    src = RefreshSourceConfig(
        name="a", files="/x/*.nc", ref="/r.parquet", label="initial conditions"
    )
    assert src.model_dump(exclude_none=True)["label"] == "initial conditions"
    frc = RefreshForcingConfig(
        name="b", kind="river", files="/x.nc", grid="/g.nc", label="river forcing"
    )
    assert frc.label == "river forcing"


# --- the default transect lines, from the grid's own ocean cells -------------------


def _write_grid(path: Path, lon, lat, mask) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    dims = ("eta_rho", "xi_rho")
    xr.Dataset(
        {"lon_rho": (dims, lon), "lat_rho": (dims, lat), "mask_rho": (dims, mask)}
    ).to_netcdf(path)
    return path


def _regular(tmp_path, lons, lats, mask=None) -> Path:
    lon, lat = np.meshgrid(lons, lats)
    return _write_grid(
        tmp_path / "g_grid.nc",
        lon,
        lat,
        np.ones(lon.shape) if mask is None else mask(lon, lat),
    )


def _lines(found: dict) -> list[dict]:
    zonal, merid = (e["transect"] for e in found.values())
    return [zonal, merid]


def test_pacmed_like_grid_gets_the_equator_and_a_central_meridian(tmp_path):
    # 0..360 longitudes (77E-316E), across the dateline, reaching both sides of 0
    grid = _regular(tmp_path, np.arange(77.0, 316.01, 1.0), np.arange(-40, 60.1, 0.5))
    found = new.default_transects(grid)
    zonal, merid = _lines(found)
    assert zonal == {"lat": 0, "lon": {"min": 77.0, "max": 316.0}}
    assert merid["lat"] == {"min": -40.0, "max": 60.0}
    assert merid["lon"] == 197.0  # median of 77..316, rounded
    assert list(found) == ["Equator (77E-44W)", "163W (40S-60N)"]


def test_zonal_line_uses_the_median_latitude_away_from_the_equator(tmp_path):
    grid = _regular(
        tmp_path, np.arange(-130.0, -119.9, 0.5), np.arange(30.0, 40.1, 0.5)
    )
    found = new.default_transects(grid)
    assert next(iter(found)) == "35N (130W-120W)"
    assert _lines(found)[0] == {"lat": 35, "lon": {"min": -130.0, "max": -120.0}}
    # a domain only just touching the equator does not get the equator line
    grid = _regular(tmp_path, np.arange(10.0, 20.1, 1.0), np.arange(-1.0, 20.1, 0.5))
    assert _lines(new.default_transects(grid))[0]["lat"] == 10
    south = _regular(tmp_path, np.arange(10.0, 20.1, 1.0), np.arange(-30.0, -10.1, 0.5))
    assert next(iter(new.default_transects(south))) == "20S (10E-20E)"


def _tilted(tmp_path, degrees: float, name: str = "rot_grid.nc"):
    """Return a 2 km grid, 150 x 200, turned ``degrees`` on the ground; east is land."""
    i, j = np.meshgrid(np.arange(200), np.arange(150))
    s, th = 0.02, np.radians(degrees)  # degrees of latitude per cell
    stretch = 1 / np.cos(np.radians(32))  # a degree of longitude is shorter at 32N
    lon = -125.0 + stretch * s * (i * np.cos(th) - j * np.sin(th))
    lat = 30.0 + s * (i * np.sin(th) + j * np.cos(th))
    mask = (i < 150).astype(float)  # the last 50 columns are land
    return _write_grid(tmp_path / name, lon, lat, mask), lon, lat, mask


def test_rotated_grid_gets_lines_along_its_row_and_column(tmp_path):
    from ocean_skill.transect import as_transect

    grid, lon, lat, mask = _tilted(tmp_path, 30)
    found, rotation = new._default_transects(grid)
    assert 25 < rotation < 35
    (row_label, row), (col_label, col) = found.items()
    assert row_label.startswith("Middle grid row (")
    assert col_label.startswith("Middle grid column (")
    assert all(as_transect(e["transect"])["kind"] == "waypoints" for e in (row, col))

    def cell(pt):  # the grid cell an endpoint (rounded to 0.01) came from
        d = np.hypot(lon - pt[0], lat - pt[1])
        return np.unravel_index(np.argmin(d), d.shape)

    (w, e), (s_, n) = row["transect"]["waypoints"], col["transect"]["waypoints"]
    assert w[0] < e[0] and s_[1] < n[1]  # west to east, south to north
    (j0, i0), (j1, i1) = cell(w), cell(e)
    assert j0 == j1 == 75 and (i0, i1) == (0, 149)  # the middle row, coast at 149
    assert all(mask[j, i] for j, i in (cell(w), cell(e), cell(s_), cell(n)))
    (j2, i2), (j3, i3) = cell(s_), cell(n)
    assert i2 == i3 == 100 and (j2, j3) == (0, 149)  # the middle column
    ends = " to ".join(f"{new._lat_label(y)} {new._lon_label(x)}" for x, y in (w, e))
    assert row_label == f"Middle grid row ({ends})"


def test_rotated_grid_block_says_so(tmp_path):
    grid, *_ = _tilted(tmp_path, 30)
    lines = new._transect_lines(grid, lambda text: text)
    assert any(
        re.search(r"The grid is rotated (29|30|31) degrees, so the lines follow", x)
        for x in lines
    )
    entries = yaml.safe_load("a:\n" + "\n".join(lines))["a"]["transects"]
    assert len(entries) == 2
    assert all("waypoints" in e["transect"] for e in entries.values())


def test_a_slightly_rotated_grid_keeps_lat_lon_lines(tmp_path):
    grid, lon, _, mask = _tilted(tmp_path, 5)
    found, rotation = new._default_transects(grid)
    assert 4 < rotation < 6
    zonal, merid = _lines(found)
    assert "lat" in zonal and "lon" in merid
    assert "waypoints" not in zonal
    assert zonal["lon"]["max"] < lon[mask == 0].max() - 0.3  # still stops at the coast
    lines = new._transect_lines(grid, lambda text: text)
    assert not any("rotated" in x for x in lines)


def test_the_grids_angle_variable_is_used_when_it_has_one(tmp_path):
    grid, lon, _, _ = _tilted(tmp_path, 0)
    with xr.open_dataset(grid) as ds:
        ds = ds.assign(angle=(("eta_rho", "xi_rho"), np.full(lon.shape, np.pi / 4)))
        ds.load().to_netcdf(tmp_path / "angled_grid.nc")
    found, rotation = new._default_transects(tmp_path / "angled_grid.nc")
    assert rotation == pytest.approx(45.0) and "Middle grid row (" in next(iter(found))


def test_antimeridian_grid_in_180_convention_is_shifted_to_0_360(tmp_path):
    lons = np.r_[np.arange(160.0, 180.0, 1.0), np.arange(-180.0, -139.9, 1.0)]
    grid = _regular(tmp_path, lons, np.arange(-20, 20.1, 0.5))
    found = new.default_transects(grid)
    zonal, merid = _lines(found)
    assert zonal["lat"] == 0
    assert zonal["lon"] == {"min": 160.0, "max": 220.0}
    assert merid["lon"] == 190.0 and merid["lat"] == {"min": -20.0, "max": 20.0}
    assert list(found) == ["Equator (160E-140W)", "170W (20S-20N)"]


def test_greenwich_crossing_0_360_grid_is_shifted_to_180_convention(tmp_path):
    lons = np.r_[np.arange(340.0, 360.0, 1.0), np.arange(0.0, 20.1, 1.0)]
    grid = _regular(tmp_path, lons, np.arange(-10, 10.1, 0.5))
    zonal, merid = _lines(new.default_transects(grid))
    assert zonal["lon"] == {"min": -20.0, "max": 20.0} and merid["lon"] == 0.0


def test_a_grid_with_no_ocean_has_no_default_lines(tmp_path):
    grid = _regular(
        tmp_path, np.arange(0.0, 5.0), np.arange(0.0, 5.0), lambda lon, lat: lon * 0
    )
    with pytest.raises(ValueError, match="no ocean"):
        new.default_transects(grid)


def _setup_block(text: str) -> str:
    return text.split("# >>> setup")[1].split("# <<< setup")[0]


@pytest.mark.parametrize("template", ["mid-run-check", "input-check"])
def test_the_setup_block_carries_the_default_transects(tmp_path, template):
    from ocean_skill.transect import as_transect

    root = (
        pacmed(tmp_path)
        if template == "mid-run-check"
        else inputs(tmp_path, restart=True)
    )
    rc, out = _new(tmp_path, root, template=template)
    assert rc == 0
    assert "  transects:\n" in _setup_block(out.read_text())
    transects = yaml.safe_load(out.read_text())["defaults"]["transects"]
    assert len(transects) == 2 and "section lines" not in out.read_text()
    for label, entry in transects.items():
        assert as_transect(entry["transect"])["kind"] in ("lat_line", "lon_line")
        assert label[0].isalnum() and "(" in label
    # the fake grid: lon 120-125, lat 10-20 (eta_rho rows of 1346 or 962)
    zonal, merid = (e["transect"] for e in transects.values())
    assert zonal["lat"] == 15 and zonal["lon"] == {"min": 120.0, "max": 125.0}
    assert merid["lon"] == 123.0  # median 122.5 rounds half up
    assert merid["lat"] == {"min": 10.0, "max": 20.0}


def test_an_unreadable_grid_leaves_the_block_to_fill_in(tmp_path, capsys):
    root = pacmed(tmp_path)
    for grid in root.glob("tasks/*/input/input_datasets/a_grid.nc"):
        xr.Dataset({"h": (("eta_rho", "xi_rho"), np.zeros((962, 3)))}).to_netcdf(grid)
    rc, out = _new(tmp_path, root)
    assert rc == 0
    text = out.read_text()
    assert "transects: {}  # TODO check: no section lines" in text
    assert "    # Equator (143E-93W): {transect:" in text
    assert yaml.safe_load(text)["defaults"]["transects"] == {}
    assert "marked '# TODO check'" in capsys.readouterr().out
