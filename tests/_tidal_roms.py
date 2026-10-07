"""A tiny tidal ROMS grid whose depth-matching answers can be written down by hand.

Shared by the tests that need a free surface that actually moves: the model side
(``test_tidal_depth_matching.py``) and the comparison plumbing built on top of it.
Import it as ``from tests._tidal_roms import tidal_roms`` (``tests`` is a package and
the repo root is on ``sys.path``, whichever directory pytest is started from). Every
number below is chosen so the expected result of a vertical match is a one-line sum.

The grid is 3 x 3 columns (nothing region-specific), ``h = 20`` m deep everywhere,
with ``n = 10`` equally thick layers of 2 m. ``hc = 0`` and ``Cs_r = sigma_r``, so the
s-coordinate is uniform and (for Vtransform 2, and for Vtransform 1 as well)::

    z_rho = zeta + (zeta + h) * sigma_r
    sigma_r = (k - n + 0.5) / n,  k = 0 (bottom) .. n - 1 (top)

``zeta`` is uniform over the grid and moves through four hourly steps, by default
``3 * [1, 0, -1, 0]`` m -- a +/-3 m tide, the macrotidal-estuary case this exists for.
Two data variables ride on it:

* ``level`` -- the ``s_rho`` index ``k`` as a float. A nearest-level match returns the
  index of the cell that was picked, so which cell won is read straight off the result.
* ``height`` -- exactly ``z_rho``. A linear match of a field equal to the vertical
  coordinate itself returns the *target's* own height (``-d`` plus ``zeta`` in the
  surface frame), so interpolation is checked against a closed form too.

Worked answers at ``d = 1`` m (nearest ``level``) and ``d = 3`` m (interp ``height``),
per time step, for the default tide ``zeta = [3, 0, -3, 0]``:

* nearest ``level`` at ``d = 1``: a fixed-in-space target ``z = -1`` is
  ``[8, 9, NaN, 9]`` -- the free surface (``zeta = -3``) is *below* the target at step
  2, so it is above the water; a target 1 m below the moving surface is ``[9, 9, 9,
  9]``.
* interp ``height`` at ``d = 3``: below the surface it is ``z = zeta - 3 = [0, -3, -6,
  -3]``; fixed in space it is ``[-3, -3, -3.85, -3]`` -- at step 2 the target
  ``z = -3`` is exactly the free surface, in the top half-cell, so the top cell's own
  value (``-3 - 0.85``) is taken rather than a NaN.

``temp=True`` adds a third, ``temp`` (degC), that tells the step, the column and the
level apart: ``10 + hour + 0.5 * xi_index + 0.1 * z_rho`` (``hour`` = the step index).

``zeta_name`` chooses the free-surface variable's name, ``"zeta"`` (ROMS's own) or
``"sea_surface_height_above_geoid"`` (what the catalog's ``standard_names`` rename it
to). ``land=True`` masks one corner column (``mask_rho = 0``), which
:func:`ocean_skill.roms.standardize` turns into NaN ``zeta`` / ``level`` / ``height``
there -- and so a NaN frame. ``chunks`` re-chunks the standardized dataset
(``{"time": 1}`` is the usual one) so a test can check that a match stays lazy.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from ocean_skill import roms

REFERENCE_DATE = "2024-07-01"
FREE_SURFACE_STANDARD_NAME = "sea_surface_height_above_geoid"


def tidal_roms(
    *,
    zeta=(3.0, 0.0, -3.0, 0.0),
    h=20.0,
    n=10,
    chunks=None,
    zeta_name="zeta",
    land=False,
    temp=False,
) -> tuple[xr.Dataset, dict]:
    """Return ``(ds, meta)``: a standardized ROMS dataset on a moving free surface.

    ``zeta`` is the per-step free-surface height in metres (uniform over the grid; its
    length is the number of hourly steps), ``h`` the depth everywhere, ``n`` the number
    of layers. See the module docstring for the other options and the worked numbers.
    """
    zeta = np.asarray(zeta, dtype=float)
    nt, ny, nx = zeta.size, 3, 3
    sigma_r = (np.arange(n) - n + 0.5) / n  # bottom (-1) -> top (0), cell centres
    sigma_w = np.linspace(-1.0, 0.0, n + 1)  # interfaces

    zeta_field = np.broadcast_to(zeta[:, None, None], (nt, ny, nx)).copy()
    zeta_4d = zeta[:, None, None, None]
    height = zeta_4d + (zeta_4d + h) * sigma_r[None, :, None, None]
    height = np.broadcast_to(height, (nt, n, ny, nx)).copy()
    level = np.broadcast_to(
        np.arange(n, dtype=float)[None, :, None, None], (nt, n, ny, nx)
    ).copy()

    mask = np.ones((ny, nx))
    if land:
        mask[0, 0] = 0.0

    lon = 200.0 + 0.01 * np.arange(nx)[None, :] + np.zeros((ny, 1))
    lat = 50.0 + 0.01 * np.arange(ny)[:, None] + np.zeros((1, nx))
    rho = ("time", "s_rho", "eta_rho", "xi_rho")
    raw = xr.Dataset(
        {
            "zeta": (("time", "eta_rho", "xi_rho"), zeta_field, {"units": "m"}),
            "level": (rho, level, {"units": "1"}),
            "height": (rho, height, {"units": "m"}),
            "h": (("eta_rho", "xi_rho"), np.full((ny, nx), float(h))),
            "mask_rho": (("eta_rho", "xi_rho"), mask),
            "lon_rho": (("eta_rho", "xi_rho"), lon),
            "lat_rho": (("eta_rho", "xi_rho"), lat),
            "Cs_r": ("s_rho", sigma_r),
            "sigma_r": ("s_rho", sigma_r),
            "Cs_w": ("s_w", sigma_w),
            "sigma_w": ("s_w", sigma_w),
            "ocean_time": ("time", np.arange(nt) * 3600.0),
        }
    )
    if temp:
        hour, xi = np.arange(nt)[:, None, None, None], np.arange(nx)
        raw["temp"] = (rho, 10.0 + hour + 0.5 * xi + 0.1 * height, {"units": "degC"})
    meta = {
        "model": "roms",
        "self_contained_grid": True,
        "vertical": {"s_dim": "s_rho", "hc": 0.0, "Vtransform": 2},
        "reference_date": REFERENCE_DATE,
    }
    if zeta_name == FREE_SURFACE_STANDARD_NAME:
        meta["standard_names"] = {"zeta": FREE_SURFACE_STANDARD_NAME}
    elif zeta_name != "zeta":
        raise ValueError(
            f"zeta_name must be 'zeta' or {FREE_SURFACE_STANDARD_NAME!r}; "
            f"got {zeta_name!r}"
        )

    ds = roms.standardize(raw, meta)
    if chunks is not None:
        ds = ds.chunk(chunks)
    return ds, meta
