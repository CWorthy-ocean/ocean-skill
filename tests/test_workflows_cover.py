"""``ocean_skill.workflows.cover``: the cover page's content and drawing.

A synthetic C-Star run folder modeled on a real one (workplan one level above
``tasks/``, ``tasks/<step>/work/`` holding blueprints and ``directives.yaml``).
"""

from __future__ import annotations

import io
import json
import os
import warnings

import pandas as pd
import pytest
import yaml
from matplotlib.backends.backend_pdf import PdfPages

from ocean_skill.config import SuiteConfig
from ocean_skill.plot.typography import PAGE_H, PAGE_W
from ocean_skill.workflows.cover import _condense, cover_figures, gather

IN = "/data/input"


def _dump(data):
    return yaml.safe_dump(data, sort_keys=False)


def _bp(run_dir, **extra):
    data_ = lambda *names: {"data": [{"location": f"{IN}/{n}"} for n in names]}  # noqa: E731
    return {
        "name": "SIM12km",
        "description": "a blueprint\nwith folded text",
        "working_dir": str(run_dir),
        "application": "roms_marbl",
        "schema_version": "3.0.0",
        "runtime_params": {"start_date": "2010-01-01", "end_date": "2010-07-01"},
        "code": {
            "roms": {"location": "https://x/ucla-roms.git", "commit": "0.8.0"},
            "run_time": {"location": "/builds/run-time", "branch": "na"},
            "marbl": {"location": "https://x/MARBL.git", "branch": "dev"},
        },
        "initial_conditions": data_("ic.nc"),
        "grid": data_("grid.nc"),
        "forcing": {
            "boundary": data_(
                *[f"S_boundary-physics_2009{m:02d}.nc" for m in (10, 11, 12)],
                "S_boundary-BGC-unified_clim.nc",
                "S_boundary-misc.nc",
            ),
            "surface": data_(
                "S_surface-restoring_clim.nc",
                "S_surface-bgc-mbl_co2_2010.nc",
                "S_surface-bgc-mbl_co2_2011.nc",
            ),
            "river": {"data": "bare_string_river.nc"},
        },
        "partitioning": {"n_procs_x": 4, "n_procs_y": 2, "use_pio": True},
        "namelist_overrides": {"time_stepping": {"dt": 900}, "a": {"b": {"c": 1}}},
        **extra,
    }


def _tree(tmp_path, *, second_workplan_matches=False):
    """``cstar/{wp.yaml, other.yaml, tasks/step_a/{output,work}}``; returns run_dir."""
    root = tmp_path / "cstar"
    run_dir = root / "tasks" / "step_a"
    (run_dir / "output").mkdir(parents=True)
    (run_dir / "work").mkdir()
    work = run_dir / "work"
    (work / "directives.yaml").write_text("{}")
    old, new = work / "B.ovrd.yaml", work / "B.ovrd.cfrom.yaml"
    old.write_text(_dump(_bp(run_dir, name="OLD")))
    new.write_text(_dump(_bp(run_dir)))
    os.utime(old, (1_000_000, 1_000_000))
    os.utime(new, (2_000_000, 2_000_000))
    step = {"name": "step_a", "blueprint": "/bp/B.yaml", "working_dir": str(run_dir)}
    wp = {"name": "WP", "description": "plan", "steps": [step]}
    (root / "wp.yaml").write_text(_dump(wp))
    elsewhere = {"working_dir": str(root / "tasks" / "first_wk")}
    other = {"name": "OTHER", "steps": [step if second_workplan_matches else elsewhere]}
    (root / "other.yaml").write_text(_dump(other))
    return run_dir


def _suite(tmp_path, run_dir=None, run_info=None, **extra):
    cfg = {
        "name": "suite",
        "defaults": {"test": "model"},
        "pages": [{"title": "t", "field": {"variable": "temp"}}],
        **extra,
    }
    if run_dir is not None:
        files = str(run_dir / "output" / "output_rst.*.nc")
        src = {"name": "model", "files": files, "ref": "r.json", "grid": "g"}
        cfg["refresh"] = {"catalog": "cat.yaml", "sources": [src]}
    if run_info is not None:
        cfg["run_info"] = run_info
    return SuiteConfig.model_validate(cfg), tmp_path / "suite.yaml"


def _gather(suite, path, index=None):
    return gather(suite, path, test_source="model", index=index)


def _rows(info, heading):
    return dict(dict(info.sections)[heading])


def test_discovery_picks_newest_blueprint_and_matching_workplan(tmp_path):
    run_dir = _tree(tmp_path)
    index = pd.date_range("2010-01-01", "2010-03-01", freq="D")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        info = _gather(*_suite(tmp_path, run_dir), index=index)
    assert info.blueprint.name == "B.ovrd.cfrom.yaml"
    assert info.workplan.name == "wp.yaml"
    assert info.name == "WP · step_a"
    assert _rows(info, "Simulation")["Name"] == "SIM12km"
    assert _rows(info, "Report")["Model span"] == "2010-01-01 - 2010-03-01"
    assert _rows(info, "Workplan")["Step"] == "step_a"
    assert "a blueprint with folded text" in info.subtitle
    assert "Simulated 2010-01-01 to 2010-07-01" in info.subtitle


def test_two_matching_workplans_warn_and_use_neither(tmp_path):
    run_dir = _tree(tmp_path, second_workplan_matches=True)
    with pytest.warns(UserWarning, match="2 workplans"):
        info = _gather(*_suite(tmp_path, run_dir))
    assert info.workplan is None
    assert info.blueprint is not None
    assert "Workplan" not in dict(info.sections)


def test_explicit_paths_override_discovery_relative_to_suite_file(tmp_path):
    run_dir = _tree(tmp_path)
    (tmp_path / "mine.yaml").write_text(_dump(_bp(run_dir, name="MINE")))
    suite, path = _suite(
        tmp_path,
        run_dir,
        {"blueprint": "mine.yaml", "workplan": "cstar/other.yaml", "name": "Custom"},
    )
    info = _gather(suite, path)
    assert info.blueprint == tmp_path / "mine.yaml"
    assert info.workplan == tmp_path / "cstar" / "other.yaml"
    assert info.name == "Custom"
    assert _rows(info, "Simulation")["Name"] == "MINE"


def test_missing_explicit_path_warns_and_is_absent(tmp_path):
    suite, path = _suite(tmp_path, run_info={"blueprint": "nope.yaml"})
    with pytest.warns(UserWarning, match="nope.yaml"):
        info = _gather(suite, path)
    assert info.blueprint is None


def test_no_refresh_falls_back_with_hint(tmp_path):
    suite, path = _suite(tmp_path)
    info = _gather(suite, path)
    assert info.blueprint is None and info.workplan is None
    assert info.name == "model"
    assert "run_info.blueprint" in _rows(info, "Report")["Simulation"]
    assert [h for h, _ in info.sections] == ["Report"]


def test_garbled_yaml_warns_instead_of_raising(tmp_path):
    (tmp_path / "bad.yaml").write_text("a: [unclosed\n  - : :")
    (tmp_path / "list.yaml").write_text("- 1\n- 2\n")
    suite, path = _suite(
        tmp_path, run_info={"blueprint": "bad.yaml", "workplan": "list.yaml"}
    )
    with pytest.warns(UserWarning):
        info = _gather(suite, path)
    assert info.blueprint is None and info.workplan is None
    assert info.name == "model"


def test_garbled_discovered_blueprint_is_skipped(tmp_path):
    run_dir = _tree(tmp_path)
    newest = run_dir / "work" / "B.ovrd.cfrom.yaml"
    newest.write_text("a: [unclosed")
    info = _gather(*_suite(tmp_path, run_dir))
    assert info.blueprint.name == "B.ovrd.yaml"  # next-newest that parses


def test_condense_collapses_one_digit_run_and_keeps_singletons():
    paths = [f"/d/x_boundary_{m}.nc" for m in ("200912", "201001", "201101")]
    paths += ["/d/x_tidal.nc", "/other/y_2009.nc"]
    assert _condense(paths) == [
        "/d/",
        "  x_boundary_{200912..201101}.nc (3 files)",
        "  x_tidal.nc",
        "/other/",
        "  y_2009.nc",
    ]
    assert _condense(["bare_2009.nc", "bare_2010.nc"]) == [
        "bare_{2009..2010}.nc (2 files)"
    ]


def test_condense_leaves_names_varying_in_two_places_alone():
    paths = ["/d/a1_b1.nc", "/d/a2_b2.nc"]
    assert _condense(paths) == ["/d/", "  a1_b1.nc", "  a2_b2.nc"]


def test_forcing_split_and_code_and_parameters(tmp_path):
    run_dir = _tree(tmp_path)
    info = _gather(*_suite(tmp_path, run_dir))
    forcing = _rows(info, "Forcing")
    assert list(forcing) == [
        "Boundary (physics)",
        "Boundary (BGC)",
        "Boundary (other)",
        "Surface (BGC)",
        "Surface (other)",
        "Rivers",
    ]
    assert (
        "boundary-physics_{200910..200912}.nc (3 files)"
        in forcing["Boundary (physics)"]
    )
    assert "unified_clim" in forcing["Boundary (BGC)"]
    assert "mbl_co2_{2010..2011}.nc (2 files)" in forcing["Surface (BGC)"]
    assert "restoring_clim" in forcing["Surface (other)"]
    assert "bare_string_river.nc" in forcing["Rivers"]
    code = _rows(info, "Code")
    assert code["roms"] == "https://x/ucla-roms.git @ 0.8.0"
    assert code["run_time"] == "/builds/run-time"  # branch: na is no information
    assert code["marbl"] == "https://x/MARBL.git @ dev"
    params = _rows(info, "Parameters")
    assert params["Namelist overrides"] == "time_stepping.dt = 900\na.b.c = 1"
    assert params["Partitioning"] == "4 x 2 (use_pio: True)"


@pytest.mark.parametrize(
    ("run_info", "title"),
    [
        ({"name": "Explicit"}, "Explicit"),
        (None, "WP · step_a"),
    ],
)
def test_name_resolution_order(tmp_path, run_info, title):
    run_dir = _tree(tmp_path)
    assert _gather(*_suite(tmp_path, run_dir, run_info)).name == title


def test_name_falls_back_to_blueprint_then_source_then_suite(tmp_path):
    run_dir = _tree(tmp_path)
    (run_dir.parent.parent / "wp.yaml").unlink()
    (run_dir.parent.parent / "other.yaml").unlink()
    assert _gather(*_suite(tmp_path, run_dir)).name == "SIM12km"
    assert _gather(*_suite(tmp_path)).name == "model"
    suite, path = _suite(tmp_path, defaults={})
    assert gather(suite, path, test_source=None, index=None).name == "suite"


def test_as_dict_is_json_ready_and_notes_keep_order(tmp_path):
    run_dir = _tree(tmp_path)
    notes = {"zeta": 1, "alpha": ["a", "b"]}
    info = _gather(*_suite(tmp_path, run_dir, {"notes": notes}))
    d = json.loads(json.dumps(info.as_dict()))
    assert d["blueprint"].endswith("B.ovrd.cfrom.yaml")
    assert d["sections"][-1] == {
        "heading": "Notes",
        "items": [["zeta", "1"], ["alpha", "['a', 'b']"]],
    }


def test_cover_figures_are_letter_size_and_paginate(tmp_path):
    run_dir = _tree(tmp_path)
    one = cover_figures(_gather(*_suite(tmp_path)))
    assert len(one) == 1
    notes = {f"note {i}": f"value {i} " * 5 for i in range(200)}
    many = cover_figures(_gather(*_suite(tmp_path, run_dir, {"notes": notes})))
    assert len(many) > 1
    for fig in [*one, *many]:
        assert tuple(fig.get_size_inches()) == (PAGE_W, PAGE_H)
    assert "(continued)" in many[1].texts[0].get_text()
    with PdfPages(io.BytesIO()) as pdf:
        for fig in many:
            pdf.savefig(fig)


def test_the_cover_spans_the_whole_workplan_not_the_newest_task(tmp_path):
    """A restarted run is one workplan of several tasks: its dates run first to last."""
    run_dir = _tree(tmp_path)
    root = run_dir.parent.parent
    steps = []
    for name, start, end in (
        ("task_1", "2013-01-02", "2013-02-15"),
        ("task_2", "2013-02-15", "2013-03-26"),
        ("task_3", "2013-03-26 00:00:00", "2013-05-01 00:00:00"),
    ):
        overrides = {"runtime_params": {"start_date": start, "end_date": end}}
        directives = {"apply-overrides": {"overrides": overrides}}
        wd = root / "tasks" / name
        steps.append(
            {
                "name": name,
                "working_dir": str(run_dir if name == "task_3" else wd),
                "directives": directives,
            }
        )
    wp = {"name": "WP", "steps": steps}
    (root / "wp.yaml").write_text(_dump(wp))
    info = _gather(*_suite(tmp_path, run_dir))
    assert "Simulated 2013-01-02 to 2013-05-01" in info.subtitle
    assert "00:00:00" not in info.subtitle
    assert _rows(info, "Simulation")["Run dates"] == "2013-01-02 -> 2013-05-01"


def test_a_generated_blueprint_description_is_not_the_subtitle(tmp_path):
    run_dir = _tree(tmp_path)
    work = run_dir / "work"
    text = "Generated blueprint; overridden keys [runtime_params, working_dir]"
    (work / "B.ovrd.cfrom.yaml").write_text(_dump(_bp(run_dir, description=text)))
    info = _gather(*_suite(tmp_path, run_dir))
    assert "Generated blueprint" not in info.subtitle
    assert _rows(info, "Simulation")["Description"].startswith("Generated blueprint")
