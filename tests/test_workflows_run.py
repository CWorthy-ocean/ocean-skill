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
                "ref": str(tmp_path / "refs" / "gom_bgc.parquet"),
            }
        ],
        tmp_path / "catalog.yaml",
    )

    ds = xr.open_dataset(
        str(tmp_path / "refs" / "gom_bgc.parquet"),
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
                "ref": str(tmp_path / "refs" / "gom_bgc.parquet"),
                "keep": "all",
            }
        ],
        tmp_path / "catalog.yaml",
    )

    ds = xr.open_dataset(
        str(tmp_path / "refs" / "gom_bgc.parquet"),
        engine="kerchunk",
        chunks={},
        decode_times=False,
    )
    assert ds.sizes["ocean_time"] == 4, "keep='all' must still keep every record"


def test_no_keep_key_still_raises_on_a_genuine_conflict(tmp_path):
    """The safety net applies here too.

    A suite that names no ``keep:`` at all still gets the mixed-stream tripwire,
    not a silent merge.
    """
    _segment(tmp_path / "cdr.nc", 0.0, 1.0)
    _segment(tmp_path / "rst.nc", 0.0, 999.0)  # same stamps, different data

    with pytest.raises(ValueError, match="disagree"):
        _refresh_sources(
            [
                {
                    "name": "GOM_bgc",
                    "files": str(tmp_path / "*.nc"),
                    "ref": str(tmp_path / "refs" / "gom_bgc.parquet"),
                }
            ],
            tmp_path / "catalog.yaml",
        )
