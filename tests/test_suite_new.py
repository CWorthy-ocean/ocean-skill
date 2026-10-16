"""``ocean-skill-suite new mid-run-check`` writes a suite that loads, from a run folder.

Fake run folders are built in ``tmp_path`` in the two layouts C-Star writes on the HPC
(PACMED: one task per month; CCS: named tasks plus a gathered ``rst_files/`` folder):
empty files for the restarts, and a tiny real netCDF for the grid, whose ``eta_rho``
length the command reads. The shipped template is the only copy of the suite, so the
generated file must keep its pages and everything after the setup block.
"""

from __future__ import annotations

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
    xr.Dataset({"h": (("eta_rho", "xi_rho"), np.zeros((n_eta, 3)))}).to_netcdf(path)


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
    """Named tasks (674 rows) with a gathered ``rst_files/`` folder, no workplan."""
    root = tmp_path / "ccs_run"
    _task(root, "jan_1wk", ["20130102000000", "20130103000000"], 674, "ccs_grid.nc")
    _task(root, "feb_ccs", ["20130201000000"], 674, "ccs_grid.nc")
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


@pytest.mark.parametrize("rows, expected", [(962, 74), (674, None)])
def test_subchunk_is_the_largest_divisor_in_range(tmp_path, rows, expected):
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
