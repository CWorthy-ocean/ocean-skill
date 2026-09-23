"""``ocean_skill.workflows.run._refresh_sources``: rebuilding a suite's kerchunk refs.

No suite ever had to say ``keep:`` for the ordinary case -- these check that leaving
it out really does let :func:`ocean_skill.build.make_kerchunk`'s own default apply,
rather than ``_refresh_sources`` silently pinning every suite to the old ``"all"``.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from ocean_skill.workflows.run import _refresh_sources


def _segment(path, t0, value):
    """One weekly-segment-like file: two half-daily records starting at ``t0``."""
    xr.Dataset(
        {"NO3": (("ocean_time", "eta_rho", "xi_rho"), np.full((2, 4, 5), value))},
        coords={"ocean_time": ("ocean_time", [t0, t0 + 43200.0], {"units": "second"})},
    ).to_netcdf(path)
    return path


def test_no_keep_key_still_collapses_a_restart_boundary(tmp_path):
    """The exact suite shape that hit the reported bug: no ``keep:`` in the YAML."""
    _segment(tmp_path / "out.0.nc", 0.0, 1.0)
    # restarted from 43200.0 -- same value, a genuine repeat
    _segment(tmp_path / "out.1.nc", 43200.0, 1.0)

    _refresh_sources(
        [
            {
                "name": "GOM_bgc",
                "files": str(tmp_path / "out.*.nc"),
                "ref": str(tmp_path / "refs" / "gom_bgc.json"),
            }
        ],
        tmp_path / "catalog.yaml",
    )

    ds = xr.open_dataset(
        str(tmp_path / "refs" / "gom_bgc.json"),
        engine="kerchunk",
        chunks={},
        decode_times=False,
    )
    assert ds.sizes["ocean_time"] == 3, "the shared 43200.0 record must be collapsed"
    assert list(ds.ocean_time.values) == sorted(ds.ocean_time.values)


def test_an_explicit_keep_key_is_still_forwarded(tmp_path):
    """A suite that opts back into 'all' (or any other keep=) still gets it."""
    _segment(tmp_path / "out.0.nc", 0.0, 1.0)
    _segment(tmp_path / "out.1.nc", 43200.0, 1.0)

    _refresh_sources(
        [
            {
                "name": "GOM_bgc",
                "files": str(tmp_path / "out.*.nc"),
                "ref": str(tmp_path / "refs" / "gom_bgc.json"),
                "keep": "all",
            }
        ],
        tmp_path / "catalog.yaml",
    )

    ds = xr.open_dataset(
        str(tmp_path / "refs" / "gom_bgc.json"),
        engine="kerchunk",
        chunks={},
        decode_times=False,
    )
    assert ds.sizes["ocean_time"] == 4, "keep='all' must still keep every record"


def test_no_keep_key_collapses_a_genuine_disagreement_without_raising(tmp_path):
    """A real overlapping rerun that is not bit-reproducible must not block a build.

    A suite that names no ``keep:`` at all gets the ``"last"`` default: the newer
    segment wins and a loud warning names the divergence, but the refresh
    completes -- this is the exact shape that first surfaced as the Anvil/Iceland
    build raising on a legitimate rerun.
    """
    _segment(tmp_path / "cdr.nc", 0.0, 1.0)
    _segment(tmp_path / "rst.nc", 0.0, 999.0)  # same stamps, different data

    with pytest.warns(UserWarning, match="DISAGREE"):
        _refresh_sources(
            [
                {
                    "name": "GOM_bgc",
                    "files": str(tmp_path / "*.nc"),
                    "ref": str(tmp_path / "refs" / "gom_bgc.json"),
                }
            ],
            tmp_path / "catalog.yaml",
        )

    ds = xr.open_dataset(
        str(tmp_path / "refs" / "gom_bgc.json"),
        engine="kerchunk",
        chunks={},
        decode_times=False,
    )
    assert float(ds.NO3.isel(ocean_time=0, eta_rho=0, xi_rho=0)) == 999.0, (
        "the last-globbed (rst) record must win"
    )


def test_a_suite_can_opt_into_the_strict_raise(tmp_path):
    """``keep: unique`` in the suite YAML still gets the mixed-stream tripwire."""
    _segment(tmp_path / "cdr.nc", 0.0, 1.0)
    _segment(tmp_path / "rst.nc", 0.0, 999.0)  # same stamps, different data

    with pytest.raises(ValueError, match="disagree"):
        _refresh_sources(
            [
                {
                    "name": "GOM_bgc",
                    "files": str(tmp_path / "*.nc"),
                    "ref": str(tmp_path / "refs" / "gom_bgc.json"),
                    "keep": "unique",
                }
            ],
            tmp_path / "catalog.yaml",
        )


def test_a_file_still_being_written_is_skipped_and_the_refresh_still_completes(
    tmp_path,
):
    """The exact case this exists for: refreshing against a run still writing output."""
    _segment(tmp_path / "out.0.nc", 0.0, 1.0)
    bad = tmp_path / "out.1.nc"
    _segment(bad, 43200.0, 1.0)
    data = bad.read_bytes()
    bad.write_bytes(data[: len(data) // 2])  # looks like it, mid-write

    _refresh_sources(
        [
            {
                "name": "GOM_bgc",
                "files": str(tmp_path / "out.*.nc"),
                "ref": str(tmp_path / "refs" / "gom_bgc.json"),
            }
        ],
        tmp_path / "catalog.yaml",
    )

    ds = xr.open_dataset(
        str(tmp_path / "refs" / "gom_bgc.json"),
        engine="kerchunk",
        chunks={},
        decode_times=False,
    )
    assert ds.sizes["ocean_time"] == 2  # only out.0.nc's records


def test_a_stream_matching_only_unfinished_files_keeps_its_existing_entry(tmp_path):
    """Distinct from a real failure: a live run between output steps is routine.

    An entry whose only matched file currently looks unfinished must be treated the
    same as one whose glob matched nothing at all -- keep whatever the catalog
    already has for it, don't raise, and don't touch other entries.
    """
    import intake

    _segment(tmp_path / "out.0.nc", 0.0, 1.0)
    catalog_path = tmp_path / "catalog.yaml"
    spec = [
        {
            "name": "GOM_bgc",
            "files": str(tmp_path / "out.*.nc"),
            "ref": str(tmp_path / "refs" / "gom_bgc.json"),
        }
    ]
    _refresh_sources(spec, catalog_path)
    before = intake.from_yaml_file(str(catalog_path))["GOM_bgc"].read()

    # the run has moved on to a new segment that isn't finished yet
    bad = tmp_path / "out.0.nc"
    data = bad.read_bytes()
    bad.write_bytes(data[: len(data) // 2])

    _refresh_sources(spec, catalog_path)  # must not raise
    after = intake.from_yaml_file(str(catalog_path))["GOM_bgc"].read()
    xr.testing.assert_identical(before, after)
