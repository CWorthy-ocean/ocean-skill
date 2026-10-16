"""``refresh.sources[].files`` as a list of globs: output spread over task folders.

A run restarted in pieces leaves its restart files in several folders
(``tasks/{first_wk,second_wk,first_half,...}/output/output_rst.*.nc``), and a second run
continues from them in more (``tasks/*/output/...``), some repeating dates (a test
segment). A list of globs joins them into one reference in *time* order -- folder names
do not sort chronologically (``first_half`` before ``first_wk``) -- and where two files
give the same time step after ``keep: latest-per-file`` the later one wins: the later
glob, or within one glob the later path in sorted order.

The synthetic files are the suite fixture's two-record UCLA restarts: the newer record
is the one ``keep: latest-per-file`` must keep, and ``shift`` makes each file's values
its own, so which file supplied a time step can be read off the data.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from ocean_skill import build, sources
from ocean_skill.config import RefreshForcingConfig, RefreshSourceConfig, SuiteConfig
from ocean_skill.workflows.run import _refresh_sources
from tests import _suite_fixture as sf


def _restart(folder: Path, stamp: str, shift: float) -> Path:
    out = folder / "output"
    out.mkdir(parents=True, exist_ok=True)
    return sf.write_restart(out / f"output_rst.{stamp}234500.nc", stamp, shift=shift)


def _times(ref: Path) -> list[str]:
    """Return the reference's time axis as ``YYYY-MM-DD`` days."""
    ds = xr.open_dataset(str(ref), engine="kerchunk", chunks={}, decode_times=False)
    secs = ds["ocean_time"].values
    stamps = np.datetime64("2000-01-01T00:00:00") + secs.astype("timedelta64[s]")
    return [str(t)[:10] for t in stamps]


def _temp(ref: Path, i: int) -> np.ndarray:
    ds = xr.open_dataset(str(ref), engine="kerchunk", chunks={}, decode_times=False)
    return ds["temp"].isel(time=i).values


def _entry(tmp_path: Path, files, **kw) -> dict:
    return {
        "name": "run",
        "files": files,
        "grid": str(sf.write_model_grid(tmp_path / "grid.nc")),
        "ref": str(tmp_path / "refs" / "run.parquet"),
        "keep": "latest-per-file",
        **kw,
    }


@pytest.fixture
def tasks(tmp_path):
    """Write a run's task folders, whose names do not sort in time order."""
    first = tmp_path / "pacmed" / "tasks"
    # Jan 2-3 in first_wk, Jan 4-5 in second_wk, Jan 6-7 in first_half: path order
    # (first_half, first_wk, second_wk) is not time order
    for folder, stamps in (
        ("first_wk", ("20100102", "20100103")),
        ("second_wk", ("20100104", "20100105")),
        ("first_half", ("20100106", "20100107")),
    ):
        for stamp in stamps:
            _restart(first / folder, stamp, shift=0.0)
    return first


def test_a_list_of_globs_joins_the_task_folders_in_time_order(tmp_path, tasks):
    entry = _entry(tmp_path, [str(tasks / "*/output/output_rst.*.nc")])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _refresh_sources([entry], tmp_path / "cat.yaml")
    assert not [w for w in caught if "strictly increasing" in str(w.message)]
    assert _times(Path(entry["ref"])) == [
        "2010-01-02",
        "2010-01-03",
        "2010-01-04",
        "2010-01-05",
        "2010-01-06",
        "2010-01-07",
    ]


def test_without_the_list_the_same_glob_is_still_in_path_order(tmp_path, tasks):
    """The one-glob string reads as it always did, folder order and all."""
    entry = _entry(tmp_path, str(tasks / "*/output/output_rst.*.nc"))
    with pytest.warns(UserWarning, match="not strictly increasing"):
        _refresh_sources([entry], tmp_path / "cat.yaml")
    assert _times(Path(entry["ref"]))[:2] == ["2010-01-06", "2010-01-07"]


def test_a_later_glob_continues_the_series_and_wins_a_repeated_time(
    tmp_path, tasks, capsys
):
    # the second run: Jan 7 repeats (a test segment re-done) and the series goes on
    second = tmp_path / "pacmed_2015" / "tasks"
    _restart(second / "y2010_test", "20100107", shift=2.0)
    _restart(second / "y2010_11", "20100108", shift=2.0)
    _restart(second / "y2010_11", "20100109", shift=2.0)
    entry = _entry(
        tmp_path,
        [
            str(tasks / "*/output/output_rst.*.nc"),
            str(second / "*/output/output_rst.*.nc"),
        ],
    )
    _refresh_sources([entry], tmp_path / "cat.yaml")
    ref = Path(entry["ref"])
    days = _times(ref)
    assert days == [f"2010-01-0{d}" for d in range(2, 10)]
    # Jan 7 is the second run's (shift 2), Jan 6 the first's (shift 0)
    older, newer = _temp(ref, days.index("2010-01-06")), _temp(
        ref, days.index("2010-01-07")
    )
    expected = xr.open_dataset(
        second / "y2010_test/output/output_rst.20100107234500.nc"
    )["temp"].isel(time=-1)
    np.testing.assert_allclose(newer, expected.values)
    assert not np.allclose(older[np.isfinite(older)], newer[np.isfinite(older)])
    # the dropped file is named, with the time and both paths
    log = capsys.readouterr().out
    loser = tasks / "first_half/output/output_rst.20100107234500.nc"
    winner = second / "y2010_test/output/output_rst.20100107234500.nc"
    assert f"refresh: run: 2010-01-07T23:45:00 from {winner} replaces {loser}" in log
    assert "(1 replaced by a later segment)" in log
    assert "<- 8 files" in log


def test_within_one_glob_the_later_path_wins(tmp_path, capsys):
    base = tmp_path / "tasks"
    _restart(base / "a_test", "20100102", shift=1.0)
    _restart(base / "b_real", "20100102", shift=3.0)  # sorts later: wins
    _restart(base / "b_real", "20100103", shift=3.0)
    entry = _entry(tmp_path, [str(base / "*/output/output_rst.*.nc")])
    _refresh_sources([entry], tmp_path / "cat.yaml")
    ref = Path(entry["ref"])
    assert _times(ref) == ["2010-01-02", "2010-01-03"]
    kept = xr.open_dataset(base / "b_real/output/output_rst.20100102234500.nc")
    np.testing.assert_allclose(_temp(ref, 0), kept["temp"].isel(time=-1).values)
    winner = base / "b_real/output/output_rst.20100102234500.nc"
    loser = base / "a_test/output/output_rst.20100102234500.nc"
    log = capsys.readouterr().out
    assert f"2010-01-02T23:45:00 from {winner} replaces {loser}" in log


def test_make_kerchunk_is_handed_time_order_ties_in_glob_order_and_not_resorted(
    tmp_path, monkeypatch
):
    """The order passed on is time, then glob order, and the later-ordered file wins.

    ``z_first`` (glob 1) sorts *after* ``a_second`` (glob 2) by path, so a re-sort by
    path in ``make_kerchunk`` would put the loser last and flip the winner.
    """
    base = tmp_path / "tasks"
    z = _restart(base / "z_first", "20100102", shift=1.0)  # glob 1
    c = _restart(base / "z_first", "20100103", shift=1.0)
    a = _restart(base / "a_second", "20100102", shift=3.0)  # glob 2: wins Jan 2
    seen: list[list[Path]] = []
    real = build.make_kerchunk

    def spy(files, *args, **kwargs):
        seen.append([Path(f) for f in files])
        return real(files, *args, **kwargs)

    monkeypatch.setattr(build, "make_kerchunk", spy)
    refs = []
    for run in range(2):  # twice: deterministic
        entry = _entry(
            tmp_path,
            [str(base / "z_first/output/*.nc"), str(base / "a_second/output/*.nc")],
            keep="last",
            ref=str(tmp_path / "refs" / f"run{run}.parquet"),
        )
        _refresh_sources([entry], tmp_path / "cat.yaml")
        refs.append(Path(entry["ref"]))
    # time order; the Jan 2 tie in glob order (z_first, then the later glob's a_second)
    assert seen == [[z, a, c]] * 2
    for ref in refs:
        # two records a file: the repeated Jan 2 pair collapses, Jan 3 keeps both
        assert _times(ref) == ["2010-01-02", "2010-01-02", "2010-01-03", "2010-01-03"]
        winner = xr.open_dataset(a)["temp"].isel(time=-1).values
        np.testing.assert_allclose(_temp(ref, 1), winner)


def test_a_glob_that_matches_nothing_does_not_stop_the_others(tmp_path, tasks):
    entry = _entry(
        tmp_path,
        [str(tmp_path / "nothing/*.nc"), str(tasks / "*/output/output_rst.*.nc")],
    )
    _refresh_sources([entry], tmp_path / "cat.yaml")
    assert len(_times(Path(entry["ref"]))) == 6


def test_a_list_matching_nothing_keeps_the_existing_entry(tmp_path, tasks, capsys):
    entry = _entry(tmp_path, [str(tasks / "*/output/output_rst.*.nc")])
    _refresh_sources([entry], tmp_path / "cat.yaml")
    ref_before = Path(entry["ref"])
    gone = {**entry, "files": [str(tmp_path / "nothing/*.nc")]}
    _refresh_sources([gone], tmp_path / "cat.yaml")  # must not raise
    assert "kept their existing entries" in capsys.readouterr().out
    assert ref_before.exists()


def test_a_path_named_by_two_globs_counts_once(tmp_path, tasks, capsys):
    pattern = str(tasks / "*/output/output_rst.*.nc")
    entry = _entry(tmp_path, [pattern, pattern])
    _refresh_sources([entry], tmp_path / "cat.yaml")
    assert len(_times(Path(entry["ref"]))) == 6
    assert "replaces" not in capsys.readouterr().out


def test_the_catalog_entry_reads_the_joined_series(tmp_path, tasks, isolated_catalogs):
    """End to end: the refreshed catalog entry opens and is one time-ordered series."""
    entry = _entry(tmp_path, [str(tasks / "*/output/output_rst.*.nc")])
    _refresh_sources([entry], isolated_catalogs / "run.yaml")
    ds = sources.read("run")
    t = ds["time"].values
    assert t.size == 6 and (np.diff(t) > np.timedelta64(0, "s")).all()


def test_order_segments_reports_time_winner_and_loser(tmp_path):
    a = _restart(tmp_path / "a", "20100102", shift=0.0)
    b = _restart(tmp_path / "b", "20100102", shift=1.0)
    c = _restart(tmp_path / "c", "20100101", shift=0.0)
    ordered, replaced = build.order_segments([[a], [b, c]], keep="latest-per-file")
    assert ordered == [c, b]  # time order; b (a later glob) beat a
    assert replaced == [("2010-01-02T23:45:00", b, a)]


def test_a_file_with_several_records_is_never_dropped_for_a_repeat(tmp_path):
    """Without latest-per-file only make_kerchunk's own keep rule collapses repeats."""
    a = _restart(tmp_path / "a", "20100102", shift=0.0)
    b = _restart(tmp_path / "b", "20100102", shift=1.0)
    ordered, replaced = build.order_segments([[a], [b]], keep="last")
    assert ordered == [a, b] and replaced == []


# -- the config ---------------------------------------------------------------------


def test_the_config_takes_a_list_of_globs():
    src = RefreshSourceConfig(
        name="run", files=["a/*.nc", "b/*.nc"], ref="r.parquet"
    )
    assert src.files == ["a/*.nc", "b/*.nc"]
    assert RefreshSourceConfig(name="run", files="a/*.nc", ref="r").files == "a/*.nc"
    with pytest.raises(ValueError, match="at least one glob"):
        RefreshSourceConfig(name="run", files=[], ref="r")


def test_the_cover_reads_a_list_of_globs(tmp_path):
    from ocean_skill.workflows import cover

    run = tmp_path / "cstar_run"
    (run / "work").mkdir(parents=True)
    (run / "work" / "directives.yaml").write_text("{}\n")
    (run / "tasks" / "a" / "output").mkdir(parents=True)
    suite = SuiteConfig.model_validate(
        {
            "name": "s",
            "refresh": {
                "catalog": str(tmp_path / "c.yaml"),
                "sources": [
                    {
                        "name": "run",
                        "files": [
                            str(run / "output_rst.*.nc"),
                            str(run / "tasks" / "*" / "output" / "output_rst.*.nc"),
                        ],
                        "ref": str(tmp_path / "r.parquet"),
                    }
                ],
            },
            "defaults": {"test": "run"},
            "pages": [{"title": "t", "field": {"variables": ["temp"]}}],
        }
    )
    assert cover._run_dir(suite) == run
    assert cover._globs(suite.refresh.sources[0].files)[0].endswith("output_rst.*.nc")


def test_forcing_config_still_validates_the_new_kind_keys():
    ok = RefreshForcingConfig(
        name="ic", kind="initial_conditions", files="ic.nc", grid="g.nc", hc=250.0
    )
    assert ok.hc == 250.0
    with pytest.raises(ValueError, match="do not apply to kind: initial_conditions"):
        RefreshForcingConfig(
            name="ic", kind="initial_conditions", files="ic.nc", grid="g.nc", ref="r"
        )
    with pytest.raises(ValueError, match="strip: is for kind: boundary"):
        RefreshForcingConfig(
            name="s", kind="surface", files="s.nc", grid="g.nc", strip=["west"]
        )
    with pytest.raises(ValueError, match="opposite sides"):
        RefreshForcingConfig(
            name="b",
            kind="boundary",
            files="b.nc",
            grid="g.nc",
            strip=["west", "east"],
        )
    with pytest.raises(ValueError, match="not among sides"):
        RefreshForcingConfig(
            name="b",
            kind="boundary",
            files="b.nc",
            grid="g.nc",
            sides=["south"],
            strip=["west", "south"],
        )
