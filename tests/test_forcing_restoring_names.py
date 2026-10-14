"""A roms-tools restoring climatology compares by the name its page asks for.

roms-tools writes ``sss`` with ``standard_name: sea_water_salinity``. The PACMED
model-comparison suite asked for ``sss`` and both of its restoring pages were skipped
with "'sss' not available"; asked for as ``salinity`` it compares, run against run.
"""

from __future__ import annotations

import warnings

import numpy as np
import xarray as xr

import ocean_skill as osk
from ocean_skill import build
from tests.test_forcing_catalog import DAYS_2000, NX, NY, _grid


def _restoring_file(path, offset=0.0):
    """Write a restoring climatology laid out like roms-tools 4.1 writes it."""
    days = 30.4 * np.arange(12)
    sss = 34.0 + offset + np.arange(12)[:, None, None] + np.zeros((12, NY, NX))
    xr.Dataset(
        {
            "sss": (
                ("time", "eta_rho", "xi_rho"),
                sss.astype("float32"),
                {
                    "standard_name": "sea_water_salinity",
                    "long_name": "sea surface salinity",
                    "units": "PSU",
                    "coordinates": "abs_time month sss_time",
                },
            ),
            "month": ("time", np.arange(1, 13), {"long_name": "Month index (1-12)"}),
            "abs_time": ("time", days, {"units": "days since 2000-01-16 00:00:00"}),
            "sss_time": ("time", days, {"units": "days", "long_name": DAYS_2000}),
        },
        attrs={"climatology": "True", "model_reference_date": "2000-01-01 00:00:00"},
    ).to_netcdf(path)


def test_a_restoring_climatology_compares_run_against_run(isolated_catalogs, tmp_path):
    grid, _, _ = _grid(tmp_path)
    cat = build.new_catalog(title="restoring")
    for run, offset in (("a", 0.0), ("b", 0.5)):
        f = tmp_path / f"{run}_restoring_clim.nc"
        _restoring_file(f, offset)
        build.add_forcing_source(cat, f"{run}_restoring", f, grid, climatology=True)
    build.save(cat, isolated_catalogs / "restoring.yaml")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cs = osk.compare(
            test="a_restoring",
            reference=["b_restoring"],
            variables=["salinity"],
            aggregate={"month": "mean"},
            cache=False,
        )
    assert len(cs) == 1
    np.testing.assert_allclose(cs[0].metrics()["bias"], -0.5, atol=1e-5)
