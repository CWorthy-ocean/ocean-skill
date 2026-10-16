"""The PACMED suites, expanded and drawn end to end against a synthetic PACMED.

The shipped ``mid-run-check``, ``final-report`` and ``model-comparison`` suites (under
``ocean_skill/workflows/templates/``) and ``suites/ccs-mid-run-check.yaml`` used
to be debugged one failure at a time, hours into an HPC run. ``_suite_fixture`` builds a
small stand-in for every input they read (UCLA restart files, roms-tools forcing, WOA23,
GLODAP, Holte & Talley, Copernicus, Cravatte -- see its docstring for the quirks it
reproduces) and rewrites a suite to point at it, so the real suite file runs here, in
minutes, on a laptop. A failure's message is the to-do list: one line per page that did
not draw, with the reason.

Three layers:

* the fixture itself looks like what it stands for (cheap, always run);
* each suite expands, and every page's sources resolve (``--list``);
* each suite draws -- once plainly, once with ``smoke=2, self_check=True, strict=True``
  (the guardrails of ``docs/suites.md``), which must come back with no problems.

The drawing cases are ``slow`` (``pytest -m slow``): they refresh the kerchunk
references and draw every page of a suite.
"""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from pathlib import Path

import netCDF4
import numpy as np
import pytest
import xarray as xr

from ocean_skill.workflows.run import run_suite
from tests import _suite_fixture as fx

ROOT = Path(__file__).resolve().parents[1]
#: Where each suite under test lives: the three shipped ones are the package's own
#: templates; the CCS run's instance of mid-run-check stays in ``suites/``.
SUITE_FILES = {
    "mid-run-check": ROOT / "ocean_skill/workflows/templates/mid-run-check.yaml",
    "ccs-mid-run-check": ROOT / "suites/ccs-mid-run-check.yaml",
    "final-report": ROOT / "ocean_skill/workflows/templates/final-report.yaml",
    "model-comparison": ROOT / "ocean_skill/workflows/templates/model-comparison.yaml",
}
PACMED_SUITES = tuple(SUITE_FILES)


@dataclass(frozen=True)
class KnownProblem:
    """A genuine package bug the guarded run reports, tracked until it is fixed.

    ``strict`` in spirit like ``xfail(strict=True)``: the problem *must* still occur (a
    fix, or a suite edit that sidesteps it, makes the entry stale and fails the test, so
    the entry gets removed) and no other problem may hide behind it. Matched on the
    suite, the problem's kind, and regular expressions searching its page title (a
    self-check copy's title starts ``[self-check]``) and its reason.
    """

    suite: str
    kind: str
    title: str
    reason: str
    why: str

    def matches(self, suite: str, problem) -> bool:
        return (
            suite == self.suite
            and problem.kind == self.kind
            and re.search(self.title, problem.title) is not None
            and re.search(self.reason, str(problem.reason)) is not None
        )


#: Package bugs the guarded run is known to report. The "maps only a fraction of an inch
#: wide" layout bug (a long row label overhanging its row; see ``_add_row_label``) was
#: tracked here until it was fixed.
KNOWN_PROBLEMS: tuple[KnownProblem, ...] = ()

HAS_GUARDRAILS = "smoke" in inspect.signature(run_suite).parameters


@pytest.fixture(scope="session")
def pacmed(tmp_path_factory) -> fx.SuiteFixture:
    """Build the synthetic PACMED once per process (a few seconds, ~60 MB)."""
    return fx.build_fixture(tmp_path_factory.mktemp("pacmed_fixture"))


@pytest.fixture
def localized(pacmed, isolated_catalogs, tmp_path):
    """Return ``localize(name) -> Path``: a suite rewritten to run on the fixture.

    ``isolated_catalogs`` keeps the packaged and the user's own catalogs (which hold the
    real ``woa23_*`` and Copernicus entries) out of discovery, so the stand-ins win.
    """

    def localize(name: str) -> Path:
        return fx.localize_suite(SUITE_FILES[name], tmp_path, pacmed)

    return localize


def _failures(result) -> list[str]:
    """One line per page that did not draw, for an assertion message."""
    lines = []
    for page in result.pages:
        if page.kind == "section" or page.status == "ok":
            continue
        reason = " ".join(str(page.reason or page.problem or "").split())
        lines.append(f"[{page.status}] {page.title}: {reason[:300]}")
    return lines


def _problem_lines(problems) -> list[str]:
    return [
        f"[{p.kind}] {p.title}: {' '.join(str(p.reason).split())[:300]}"
        for p in problems
    ]


# --------------------------------------------------------------- the fixture itself


def test_restart_files_hold_two_records_and_the_newer_is_last(pacmed):
    """``keep: latest-per-file`` has something to drop, and the kept one is the last."""
    path = sorted(Path(pacmed.sources["pacmed_pyesper"]["files"]).parent.glob("*.nc"))[
        0
    ]
    with xr.open_dataset(path, decode_times=False) as ds:
        assert ds.sizes["time"] == 2
        t = ds["ocean_time"].values
        assert t[1] > t[0]
        assert not np.allclose(ds["temp"].isel(time=0), ds["temp"].isel(time=1))
        # ROMS writes land as 0, not NaN; the grid is not in the file
        assert "lon_rho" not in ds and "mask_rho" not in ds
        assert (
            {"u", "v"} <= set(ds) and "xi_u" in ds["u"].dims and "eta_v" in ds["v"].dims
        )


def test_the_domain_crosses_the_dateline_with_land_and_a_shallow_north(pacmed):
    with xr.open_dataset(pacmed.model_grid) as g:
        lon, h, mask = g["lon_rho"].values, g["h"].values, g["mask_rho"].values
        assert (
            lon.min() < 180 < lon.max() and lon.max() > 270
        )  # 0-360, past the dateline
        assert (mask == 0).any() and (mask == 1).any()
        north = mask[-1] == 1
        assert np.all(h[-1][north] < 100.0), (
            "the north edge must be shelf, so 200 m is all-NaN"
        )
        assert h[0].min() > 1000.0
        assert {"hc", "theta_s", "theta_b", "Vtransform"} <= set(g.attrs)
        assert {"pm", "pn", "angle"} <= set(g)


def test_zeta_swings_above_five_metres_somewhere(pacmed):
    path = sorted(Path(pacmed.sources["pacmed_pyesper"]["files"]).parent.glob("*.nc"))[
        0
    ]
    with xr.open_dataset(path) as ds:
        assert float(ds["zeta"].max()) > 5.0


def test_forcing_months_are_unequal_and_cdf5_where_the_parser_reads_it(pacmed):
    files = pacmed.forcing["pyesper_bry_phys"]["files"]
    lengths = []
    for f in files:
        with netCDF4.Dataset(f) as nc:
            lengths.append(nc.dimensions["bry_time"].size)
            assert nc.data_model == (
                "NETCDF3_64BIT_DATA"
                if fx._cdf5() == "NETCDF3_64BIT_DATA"
                else "NETCDF3_64BIT_OFFSET"
            )
    assert len(set(lengths)) > 1, lengths
    clim = pacmed.forcing["mld_bry_bgc"]["files"][0]
    with netCDF4.Dataset(clim) as nc:
        assert nc.dimensions["bry_time"].size == 12 and "month" in nc.variables


def test_every_nickname_the_suites_read_is_in_the_catalogs(pacmed):
    import intake

    names = set()
    for cat in pacmed.catalog_dir.glob("*.yaml"):
        names |= set(intake.from_yaml_file(str(cat)))
    needed = {
        "glodap",
        "holte_talley_mld_clim",
        "chl_gapfree_my_daily_geo",
        "ssh_duacs_my_daily_geo",
        "cravatte_u_179e_160w",
        "cravatte_u_160w_120w",
        "glorys_my_daily_timeseries",
    }
    for var in (
        "temperature",
        "salinity",
        "nitrate",
        "phosphate",
        "silicate",
        "oxygen",
    ):
        needed |= {f"woa23_{var}_annual", f"woa23_{var}_monthly"}
        needed |= {f"woa23_{var}_month{m:02d}" for m in range(1, 13)}
    assert needed <= names, sorted(needed - names)


@pytest.mark.parametrize("name", PACMED_SUITES)
def test_localizing_a_suite_points_every_path_at_the_fixture(pacmed, tmp_path, name):
    import yaml

    out = yaml.safe_load(
        fx.localize_suite(SUITE_FILES[name], tmp_path, pacmed).read_text()
    )
    assert out["output_dir"].startswith(str(tmp_path))
    assert out.get("cache_dir", str(tmp_path)).startswith(str(tmp_path))
    assert out["catalog_search_paths"] == [str(tmp_path / "catalogs")]
    refresh = out["refresh"]
    assert refresh["catalog"].startswith(str(tmp_path))
    for entry in [*refresh["sources"], *refresh.get("forcing", [])]:
        files = entry["files"] if isinstance(entry["files"], list) else [entry["files"]]
        assert all(str(pacmed.root) in f for f in files), entry["name"]
        assert str(pacmed.root) in entry["grid"]
        if "ref" in entry:  # initial-conditions entries read one file: no reference
            assert entry["ref"].startswith(str(tmp_path / "refs"))
    text = yaml.safe_dump(out)
    assert "/anvil/" not in text, "a path of the real run was left in"


def test_a_refresh_entry_the_fixture_lacks_is_named_in_the_error(pacmed, tmp_path):
    import yaml

    src = tmp_path / "s.yaml"
    src.write_text(
        yaml.safe_dump(
            {
                "name": "s",
                "refresh": {
                    "catalog": "/x/c.yaml",
                    "sources": [
                        {
                            "name": "not_in_fixture",
                            "files": "/x/*.nc",
                            "ref": "/x/r.parquet",
                        }
                    ],
                },
                "pages": [{"title": "t", "section": ""}],
            }
        )
    )
    with pytest.raises(KeyError, match=r"not_in_fixture.*_suite_fixture"):
        fx.localize_suite(src, tmp_path / "out", pacmed)


def test_localizing_fits_dates_regions_and_subchunks_to_the_fixture(pacmed, tmp_path):
    import yaml

    out = yaml.safe_load(
        fx.localize_suite(SUITE_FILES["final-report"], tmp_path, pacmed).read_text()
    )
    assert out["defaults"]["obs_period"]["max"] == fx.RUN_WINDOW[1]
    assert out["defaults"]["restart_days"] == pacmed.restart_days
    box = next(p for p in out["pages"] if "TS" in p)["TS"]["regions"]["Peru Current"]
    assert box["lon"]["max"] - box["lon"]["min"] == pytest.approx(5.0)
    sub = out["refresh"]["sources"][0]["subchunk"]
    assert fx.NY % sub["eta_rho"] == 0 and sub["eta_rho"] < fx.NY and sub["s_rho"] == 1


# ----------------------------------------------- a bug the fixture turned up (fixed)


def test_a_boundary_row_with_land_cells_still_compares(
    isolated_catalogs, tmp_path, monkeypatch
):
    import ocean_skill as osk
    from ocean_skill import build

    monkeypatch.setattr(fx, "BOUNDARY_LAND", True)
    grid = fx.write_forcing_grid(tmp_path / "grid.nc")
    cat = build.new_catalog(title="bry")
    for run, offset in (("a", 0.0), ("b", 1.0)):
        f = fx.write_boundary(
            tmp_path / f"{run}.nc", [0, 1, 2], fx._ESPER, offset=offset
        )
        build.add_boundary_sources(cat, run, [f], grid, sides=["north"])
    build.save(cat, isolated_catalogs / "bry.yaml")
    try:
        cs = osk.compare(
            test="a_north",
            reference=["b_north"],
            variables=["nitrate"],
            select={"transect": {"eta_rho": 0}, "depth": [5, 25]},
            aggregate={"time": "mean"},
            cache=False,
        )
    except ValueError as exc:
        assert "is not finite" not in str(exc), (
            exc
        )  # the bug; any other error is a real one
        raise
    assert len(cs) == 1


# ------------------------------------------------------------------ the suites on it


@pytest.mark.parametrize("name", PACMED_SUITES)
def test_every_page_of_a_suite_expands_on_the_fixture(localized, name):
    result = run_suite(localized(name), list_only=True)
    broken = [f"{p.title}: {p.problem}" for p in result.pages if p.problem]
    assert not broken, "\n".join(["pages that will be skipped:", *broken])


@pytest.mark.slow
@pytest.mark.parametrize("name", PACMED_SUITES)
def test_a_plain_run_draws_every_page(localized, name):
    """No flags: every page is attempted, a page that cannot draw says why."""
    result = run_suite(localized(name))
    failures = _failures(result)
    assert not failures, "\n".join(["pages that did not draw:", *failures])
    assert result.pdf is not None and result.pdf.exists()


@pytest.mark.slow
@pytest.mark.skipif(
    not HAS_GUARDRAILS, reason="run_suite has no smoke/self_check/strict yet"
)
@pytest.mark.parametrize("name", PACMED_SUITES)
def test_the_guarded_run_finds_no_problems(localized, name):
    """``smoke=2, self_check=True, strict=True`` is the run to make before an HPC one.

    Anything in :data:`KNOWN_PROBLEMS` is a package bug being tracked: it must still
    happen (a stale entry fails here) and is not counted against the suite.
    """
    result = run_suite(localized(name), smoke=2, self_check=True, strict=True)
    known = [k for k in KNOWN_PROBLEMS if k.suite == name]
    unknown = [p for p in result.problems if not any(k.matches(name, p) for k in known)]
    stale = [k for k in known if not any(k.matches(name, p) for p in result.problems)]
    lines = [f"{len(unknown)} problem(s):", *_problem_lines(unknown)]
    assert not unknown, "\n".join(lines)
    assert not stale, (
        "KNOWN_PROBLEMS entries that no longer occur (remove them):\n"
        + "\n".join(f"  {k.title!r} [{k.kind}]" for k in stale)
    )
