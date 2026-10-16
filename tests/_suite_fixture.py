"""A small synthetic PACMED, so the real suites can be expanded *and drawn* off-line.

The shipped ``mid-run-check``, ``final-report`` and ``model-comparison`` suites
(``ocean_skill/workflows/templates/``) read a UCLA-ROMS run's monthly restarts,
roms-tools forcing files and a dozen observational products. Their bugs used to be
found one at a time, hours into an HPC run. This module builds a tiny stand-in for
every input -- same layouts, same names, same units, same awkward corners -- plus
:func:`localize_suite`, which points a suite at it. The test
(``test_suite_smoke_fixture.py``) then runs the real suite file end to end.

What the stand-in reproduces (each of these was a bug that cost an HPC round trip):

* **The model** -- UCLA layout, monthly restart files named ``output_rst.<stamp>.nc``
  with *two* records each (``keep: latest-per-file`` keeps the newer), the grid in its
  own file (``pm``/``pn``/``h``/``mask_rho``/``lon_rho``/``lat_rho``/``angle`` and the
  ``hc``/``theta_s``/``theta_b``/``Vtransform`` attributes), staggered ``u``/``v``, land
  written as 0 not NaN. The domain crosses the dateline (140E-290E, longitudes 0-360)
  and has land, an island and a 75 m shelf along its north edge, so a 200 m selection
  there is all-NaN. ``zeta`` swings above +5 m on that shelf (the mixed layer depth once
  broke on a surface above +5 m).
* **Forcing** -- roms-tools boundary physics and BGC files of *unequal* month lengths in
  CDF-5, a 12-month boundary BGC climatology, a surface restoring climatology (``sss``)
  and the initial-conditions restarts.
* **Observations**, registered in one catalog directory under the exact nicknames the
  suites use, each shaped like its real product (dimension names, units, standard names,
  longitude convention, time axis): WOA23 annual / monthly / ``month01..12``, GLODAPv2
  (0-360, ``depth_surface``), Holte & Talley (classic netCDF3, ``month`` promoted by a
  reader chain), the Copernicus chlorophyll / DUACS / GLORYS products, and the Cravatte
  lon-less zonal-velocity sections.

Sizes are deliberately small: the model grid is 42 x 76 cells with 8 levels, four
restart months. The one concession: the T-S ``regions:`` boxes of the real suite are
about one degree wide, which holds no cell of a 2-degree grid, so :func:`localize_suite`
widens every ``regions:`` box to five degrees around its centre.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from ocean_skill import build

# ----------------------------------------------------------------------------- geometry

LON = 140.0 + 2.0 * np.arange(76)  # 140E..290E, 0-360 convention, crosses the dateline
LAT = -30.0 + 2.0 * np.arange(42)  # 30S..52N
NX, NY = LON.size, LAT.size
NS = 8
HC, THETA_S, THETA_B, VTRANSFORM = 100.0, 6.0, 3.0, 2

#: Month-end restarts of the mid-run-check / final-report stand-in run.
RESTART_STAMPS = ("20110131", "20110228", "20110331", "20110430")
FORCING_EPOCH = "2011-01-01"
ROMS_EPOCH = "2000-01-01"
RUN_WINDOW = ("2011-01-01", "2011-04-30")

#: The model-comparison runs keep the real run's dates (2010-01 on), small: daily
#: restarts for the first days, one at the end of January, then quarterly month ends.
#: Both runs hold the same steps; the folders they are split over mirror the real ones.
MC_DAILY_STAMPS = (
    "20100101",
    "20100102",
    "20100103",
    "20100104",
    "20100105",
    "20100131",
)
MC_MONTH_ENDS = (
    *(
        f"{y}{md}"
        for y in range(2010, 2014)
        for md in ("0131", "0430", "0731", "1031")
        if (y, md) != (2010, "0131")  # already a daily restart
    ),
    "20131231",
)
MC_WINDOW = ("2010-01-01", "2013-12-31")

#: Land that touches the open boundaries (the north-west corner). Off: the real PACMED
#: boundaries are open ocean along their whole length. A test turns it on to show that a
#: boundary row with land cells in it still compares (land is just missing data).
BOUNDARY_LAND = False
#: The land/ocean patterns: an island, a coastal bump and an inlet.
_NORTH_SHELF_LAT = 48.0
_SHELF_DEPTH = 75.0


def _lon_lat() -> tuple[np.ndarray, np.ndarray]:
    return np.meshgrid(LON, LAT)


def bathymetry() -> tuple[np.ndarray, np.ndarray]:
    """Return ``(h, mask_rho)``: deep ocean, a 75 m north shelf, some land."""
    lon, lat = _lon_lat()
    h = 4500.0 - 1500.0 * np.clip((lon - 240.0) / 50.0, 0.0, 1.0)
    # north shelf: 75 m from 48N up, ramping down from 44N
    ramp = np.clip((lat - 44.0) / (_NORTH_SHELF_LAT - 44.0), 0.0, 1.0)
    h = h * (1.0 - ramp) + _SHELF_DEPTH * ramp
    mask = np.ones_like(h)
    mask[(lon >= 170) & (lon <= 174) & (lat >= 8) & (lat <= 12)] = 0.0  # an island
    mask[(lon >= 282) & (lon <= 286) & (lat >= 20) & (lat <= 40)] = 0.0  # a coast bump
    mask[(lon >= 150) & (lon <= 156) & (lat >= 40) & (lat <= 46)] = 0.0  # an inlet
    if BOUNDARY_LAND:
        mask[(lon <= 146) & (lat >= 44)] = 0.0  # land on the north and west boundaries
    h[mask == 0] = 5.0
    return h, mask


def _stretching(sigma: np.ndarray) -> np.ndarray:
    """ROMS ``Vstretching = 4`` ``Cs`` for the fixture's ``theta_s``/``theta_b``."""
    cs = (1.0 - np.cosh(THETA_S * sigma)) / (np.cosh(THETA_S) - 1.0)
    return (np.exp(THETA_B * cs) - 1.0) / (1.0 - np.exp(-THETA_B))


def vertical() -> dict[str, np.ndarray]:
    """Return the s-coordinate arrays (``sigma_r``/``Cs_r``; ``_w`` at the faces)."""
    sigma_r = (np.arange(NS) - NS + 0.5) / NS
    sigma_w = np.linspace(-1.0, 0.0, NS + 1)
    return {
        "sigma_r": sigma_r,
        "Cs_r": _stretching(sigma_r),
        "sigma_w": sigma_w,
        "Cs_w": _stretching(sigma_w),
    }


def z_rho(zeta: np.ndarray) -> np.ndarray:
    """Cell-centre depths ``(s_rho, eta, xi)`` (negative down) under ``zeta``."""
    v = vertical()
    h, _ = bathymetry()
    sig = v["sigma_r"][:, None, None]
    cs = v["Cs_r"][:, None, None]
    s = (HC * sig + h[None] * cs) / (HC + h[None])
    return zeta[None] + (zeta[None] + h[None]) * s


# ------------------------------------------------------------------------------- grids


def write_model_grid(path: Path) -> Path:
    """Write the model's own grid file (the vertical arrays ride in the restarts)."""
    lon, lat = _lon_lat()
    h, mask = bathymetry()
    dx = 111e3 * 2.0 * np.cos(np.deg2rad(lat))
    dy = np.full_like(dx, 111e3 * 2.0)
    rho = ("eta_rho", "xi_rho")
    xr.Dataset(
        {
            "lon_rho": (rho, lon, {"units": "degrees_east"}),
            "lat_rho": (rho, lat, {"units": "degrees_north"}),
            "h": (rho, h, {"units": "meter"}),
            "mask_rho": (rho, mask),
            "angle": (rho, np.zeros_like(h), {"units": "radians"}),
            "pm": (rho, 1.0 / dx),
            "pn": (rho, 1.0 / dy),
        },
        attrs={
            "hc": HC,
            "theta_s": THETA_S,
            "theta_b": THETA_B,
            "Vtransform": VTRANSFORM,
        },
    ).to_netcdf(path)
    return path


def write_forcing_grid(path: Path) -> Path:
    """Write the roms-tools grid file the forcing entries merge in."""
    lon, lat = _lon_lat()
    h, mask = bathymetry()
    v = vertical()
    rho = ("eta_rho", "xi_rho")
    xr.Dataset(
        {
            "lon_rho": (rho, lon),
            "lat_rho": (rho, lat),
            "h": (rho, h),
            "mask_rho": (rho, mask),
            "angle": (rho, np.zeros_like(h)),
            "sigma_r": ("s_rho", v["sigma_r"]),
            "Cs_r": ("s_rho", v["Cs_r"]),
            "sigma_w": ("s_w", v["sigma_w"]),
            "Cs_w": ("s_w", v["Cs_w"]),
        },
        attrs={"hc": HC, "theta_s": THETA_S, "theta_b": THETA_B},
    ).to_netcdf(path)
    return path


# ----------------------------------------------------------- model restart files


def _tracers(
    z: np.ndarray, lat: np.ndarray, season: float
) -> dict[str, tuple[np.ndarray, str]]:
    """Smooth, positive fields on depths ``z`` (m, negative down)."""
    lat = lat[None, :, None]
    a = np.abs(lat)
    seas = np.sin(2.0 * np.pi * season)
    temp = 2.0 + (27.0 - 0.35 * a) * np.exp(z / 450.0) + 0.8 * seas * np.exp(z / 100.0)
    salt = 34.2 + 0.9 * np.exp(z / 250.0) * np.cos(np.deg2rad(lat)) + 0.05 * seas
    no3 = 0.2 + 38.0 * (1.0 - np.exp(z / 400.0)) + 0.3 * seas
    return {
        "temp": (temp, "Celsius"),
        "salt": (salt, "PSU"),
        "NO3": (no3, "mmol/m3"),
        "PO4": (no3 / 15.0 + 0.05, "mmol/m3"),
        "SiO3": (2.0 + 2.5 * no3, "mmol/m3"),
        "O2": (
            260.0
            - 215.0 * np.exp(-(((z + 400.0) / 300.0) ** 2))
            + 30.0 * np.exp(z / 100.0),
            "mmol/m3",
        ),
        "Fe": (5e-4 + 1e-3 * (1.0 - np.exp(z / 500.0)), "mmol/m3"),
        "ALK": (
            2250.0 + 100.0 * (1.0 - np.exp(z / 1000.0)) + 0.4 * lat + 0.5 * seas,
            "meq/m3",
        ),
        "DIC": (1950.0 + 250.0 * (1.0 - np.exp(z / 700.0)) + 0.3 * seas, "mmol/m3"),
        "MARBL_PH_3D": (8.1 - 0.4 * (1.0 - np.exp(z / 600.0)), "1"),
        "spChl": (0.2 * np.exp(z / 40.0) + 0.01, "mg/m3"),
        "diatChl": (0.05 * np.exp(z / 40.0) + 0.002, "mg/m3"),
        "diazChl": (0.01 * np.exp(z / 40.0) + 0.001, "mg/m3"),
    }


def _zeta(season: float, shift: float = 0.0) -> np.ndarray:
    """Free surface: tens of centimetres, and +6 m on the north shelf (a tide)."""
    lon, lat = _lon_lat()
    z = 0.3 * np.sin(np.deg2rad(lon) * 2.0 + 2.0 * np.pi * season) + shift
    shelf = np.clip((lat - 44.0) / 4.0, 0.0, 1.0)
    return z + 6.0 * shelf * np.sin(np.pi * (0.55 + 0.1 * season))


def _restart_records(
    n: int, season0: float, shift: float
) -> tuple[dict[str, tuple[tuple, np.ndarray, dict]], np.ndarray]:
    """Return the ``n`` records' data variables (UCLA dims) at fractions of a year."""
    lon, lat = _lon_lat()
    _, mask = bathymetry()
    out: dict[str, list[np.ndarray]] = {}
    units: dict[str, str] = {}
    zetas = []
    for k in range(n):
        season = season0 - 0.01 * (n - 1 - k)
        zeta = _zeta(season, shift)
        zetas.append(zeta)
        z = z_rho(zeta)
        for name, (vals, unit) in _tracers(z, LAT, season).items():
            vals = (
                np.broadcast_to(vals, z.shape)
                + (0.0 if k == n - 1 else 0.25)
                + 0.5 * shift
            )
            out.setdefault(name, []).append(np.where(mask[None] > 0, vals, 0.0))
            units[name] = unit
        # velocities on their staggered points: u on xi_u, v on eta_v (grid-aligned)
        zc = z[:, :, :-1]
        u = (
            0.6
            * np.exp(-((LAT[None, :, None] / 3.0) ** 2))
            * np.exp(-(((zc + 100.0) / 60.0) ** 2))
        )
        u = u - 0.2 * np.exp(-((LAT[None, :, None] / 6.0) ** 2)) * np.exp(zc / 30.0)
        out.setdefault("u", []).append(u)
        zv = z[:, :-1, :]
        out.setdefault("v", []).append(
            0.05 * np.sin(np.deg2rad(lon[None, :-1, :]) * 3.0) * np.exp(zv / 200.0)
        )
        out.setdefault("hbls", []).append(
            np.where(mask > 0, 30.0 + 40.0 * np.abs(np.sin(np.deg2rad(lat))), 0.0)
        )
        out.setdefault("zeta", []).append(np.where(mask > 0, zeta, 0.0))
    units.update(u="m/s", v="m/s", hbls="m", zeta="m")
    dims = {
        "u": ("time", "s_rho", "eta_rho", "xi_u"),
        "v": ("time", "s_rho", "eta_v", "xi_rho"),
        "hbls": ("time", "eta_rho", "xi_rho"),
        "zeta": ("time", "eta_rho", "xi_rho"),
    }
    rho3 = ("time", "s_rho", "eta_rho", "xi_rho")
    data = {
        name: (
            dims.get(name, rho3),
            np.stack(vals).astype("float32"),
            {"units": units[name]},
        )
        for name, vals in out.items()
    }
    return data, np.stack(zetas)


def write_restart(path: Path, stamp: str, *, shift: float = 0.0) -> Path:
    """Write one two-record UCLA restart file; the *newer* record is the last one.

    The earlier record differs in value (``keep: latest-per-file`` must drop it), is
    three hours older, and the file carries ``sigma_r``/``Cs_r`` but not the grid.
    """
    t = np.datetime64(f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}T23:45:00")
    secs = (
        (t - np.datetime64(f"{ROMS_EPOCH}T00:00:00"))
        .astype("timedelta64[s]")
        .astype(float)
    )
    season = (t.astype("datetime64[D]") - t.astype("datetime64[Y]")).astype(
        float
    ) / 365.0
    data, _ = _restart_records(2, float(season), shift)
    v = vertical()
    ds = xr.Dataset(
        {
            **data,
            "ocean_time": (
                "time",
                np.array([secs - 3 * 3600.0, secs]),
                {"long_name": f"time since {ROMS_EPOCH} 00:00:00", "units": "second"},
            ),
            "sigma_r": ("s_rho", v["sigma_r"]),
            "Cs_r": ("s_rho", v["Cs_r"]),
            "sigma_w": ("s_w", v["sigma_w"]),
            "Cs_w": ("s_w", v["Cs_w"]),
        }
    )
    ds.to_netcdf(path, unlimited_dims=["time"])
    return path


def write_initial_conditions(
    path: Path,
    stamp: str = "20110101",
    *,
    hc_attr: bool = False,
    shift: float = 0.0,
    ini_time: str | None = None,
    reference_date: str = ROMS_EPOCH,
) -> Path:
    """Write a roms-tools initial-conditions file (roms-tools 4.0.0's layout).

    One record on ``(ocean_time, s_rho, eta_rho, xi_rho)`` -- ``ocean_time`` is both the
    dimension and the coordinate, seconds since ``model_reference_date``, with the
    roms-tools ``abs_time`` beside it -- plus ``u``/``v`` on their staggered points,
    ``ubar``/``vbar`` and ``zeta``; ``Cs_r``/``Cs_w`` as variables,
    ``theta_s``/``theta_b`` as global attributes, no ``sigma_r`` and no grid. 4.0.0
    writes no ``hc`` attribute (``hc_attr=True`` adds it, as a later version might).
    ``ini_time`` (default: the day's start, as roms-tools states it) is what the file
    claims its time is.
    """
    day = _stamp_day(stamp)
    t = np.datetime64(f"{day}T00:00:00")
    secs = float(
        (t - np.datetime64(f"{reference_date}T00:00:00"))
        .astype("timedelta64[s]")
        .astype(float)
    )
    season = float(
        (t.astype("datetime64[D]") - t.astype("datetime64[Y]")).astype(float) / 365.0
    )
    data, _ = _restart_records(1, season, shift)
    data.pop("hbls")  # a model diagnostic, not an initial condition
    v = vertical()
    ocean = {
        name: (("ocean_time", *dims[1:]), vals, attrs)
        for name, (dims, vals, attrs) in data.items()
    }
    ocean["ubar"] = (
        ("ocean_time", "eta_rho", "xi_u"),
        data["u"][1].mean(axis=1).astype("float32"),
        {"units": "m/s"},
    )
    ocean["vbar"] = (
        ("ocean_time", "eta_v", "xi_rho"),
        data["v"][1].mean(axis=1).astype("float32"),
        {"units": "m/s"},
    )
    ds = xr.Dataset(
        {
            **ocean,
            "Cs_r": ("s_rho", v["Cs_r"]),
            "Cs_w": ("s_w", v["Cs_w"]),
        },
        coords={
            "ocean_time": (
                "ocean_time",
                np.array([secs]),
                {
                    "long_name": (
                        f"relative time: seconds since {reference_date} 00:00:00"
                    ),
                    "units": "seconds",
                },
            ),
            "abs_time": ("ocean_time", np.array([t], dtype="datetime64[ns]")),
        },
        attrs={
            "title": "ROMS initial conditions file created by ROMS-Tools",
            "roms_tools_version": "4.0.0",
            "ini_time": ini_time or f"{day} 00:00:00",
            "model_reference_date": f"{reference_date} 00:00:00",
            "source": "GLORYS",
            "theta_s": THETA_S,
            "theta_b": THETA_B,
            **({"hc": HC} if hc_attr else {}),
        },
    )
    ds.to_netcdf(path)
    return path


# ------------------------------------------------------------------------- the fixture


@dataclass
class SuiteFixture:
    """Everything :func:`localize_suite` points a suite at (all real files)."""

    root: Path
    catalog_dir: Path
    model_grid: Path
    forcing_grid: Path
    #: refresh.sources name -> {"files": glob, "grid": path}
    sources: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: refresh.forcing name -> {"files": [paths], "grid": path}
    forcing: dict[str, dict[str, Any]] = field(default_factory=dict)
    restart_days: list[str] = field(default_factory=list)
    window: tuple[str, str] = RUN_WINDOW
    #: refresh.sources names whose stand-ins carry the real run's dates (the
    #: model-comparison runs, 2010-2013): a suite built on them is not date-remapped.
    dated_sources: frozenset[str] = frozenset()


def _stamp_day(stamp: str) -> str:
    return f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}"


# ---------------------------------------------------------------------------- forcing

_SIDES = {
    # side: (along dim, size, boundary index on the cross dim, staggered dim)
    "south": ("xi_rho", NX, 0, "xi_u"),
    "north": ("xi_rho", NX, -1, "xi_u"),
    "west": ("eta_rho", NY, 0, "eta_v"),
    "east": ("eta_rho", NY, -1, "eta_v"),
}
_PHYSICS = ("temp", "salt")
_ESPER = ("ALK", "DIC", "NO3", "O2", "PO4", "SiO3")
_UNIFIED = (*_ESPER, "Fe", "NH4")


def _cdf5() -> str:
    """Return CDF-5 (what roms-tools writes), or CDF-2 if the parser cannot read it."""
    return (
        "NETCDF3_64BIT_DATA" if build._netcdf3_reads_cdf5() else "NETCDF3_64BIT_OFFSET"
    )


def _along(side: str, field3d: np.ndarray) -> np.ndarray:
    """Slice ``(s, eta, xi)`` down to the boundary ``side``: ``(s, along)``."""
    _, _, idx, _ = _SIDES[side]
    return field3d[:, idx, :] if side in ("south", "north") else field3d[:, :, idx]


def write_boundary(
    path: Path,
    days,
    names: tuple[str, ...],
    *,
    climatology: bool = False,
    offset: float = 0.0,
    velocities: bool = False,
    fmt: str | None = None,
    epoch: str = FORCING_EPOCH,
) -> Path:
    """Write a roms-tools boundary file: ``<name>_<side>`` on ``(bry_time, ...)``.

    No coordinates but ``bry_time`` (days, epoch in ``long_name``) and ``abs_time``; a
    climatology adds ``month``. ``velocities`` adds the staggered ``u``/``v`` and their
    barotropic means (which the entries must leave out), ``zeta`` rides with them.
    """
    days = np.asarray(days, dtype="float64")
    nt = days.size
    z = z_rho(np.zeros((NY, NX)))
    data: dict[str, Any] = {}
    for side, (along, n, _, stag) in _SIDES.items():
        per: dict[str, list[np.ndarray]] = {name: [] for name in names}
        for k in range(nt):
            fields = _tracers(z, LAT, (days[k] % 365.0) / 365.0)
            fields["NH4"] = (0.05 + 0.02 * fields["NO3"][0] / 38.0, "mmol/m3")
            for name in names:
                vals = np.broadcast_to(fields[name][0], z.shape)
                per[name].append(_along(side, vals) + offset)
        for name in names:
            unit = {"temp": "degrees Celsius", "salt": "PSU"}.get(name, "mmol/m3")
            data[f"{name}_{side}"] = (
                ("bry_time", "s_rho", along),
                np.stack(per[name]).astype("float32"),
                {"units": unit},
            )
        if velocities:
            rng = np.random.default_rng(int(days[0]) + 7)
            data[f"zeta_{side}"] = (
                ("bry_time", along),
                rng.normal(0, 0.1, (nt, n)).astype("float32"),
            )
            for var in ("u", "v"):
                data[f"{var}_{side}"] = (
                    ("bry_time", "s_rho", stag),
                    rng.normal(size=(nt, NS, n - 1)).astype("float32"),
                )
                data[f"{var}bar_{side}"] = (
                    ("bry_time", stag),
                    rng.normal(size=(nt, n - 1)).astype("float32"),
                )
    coords: dict[str, Any] = {
        "bry_time": (
            "bry_time",
            days,
            {
                "long_name": f"relative time: days since {epoch} 00:00:00",
                "units": "days",
            },
        ),
        "abs_time": (
            "bry_time",
            np.datetime64(epoch)
            + (days * 86400).astype("timedelta64[s]").astype("timedelta64[ns]"),
        ),
    }
    if climatology:
        coords["month"] = ("bry_time", np.arange(1, nt + 1))
    xr.Dataset(data, coords=coords).to_netcdf(
        path, format=fmt or _cdf5(), engine="netcdf4"
    )
    return path


def write_surface_climatology(path: Path, names: tuple[str, ...] = ("sss",)) -> Path:
    """Write a roms-tools surface climatology: one field per name plus ``<name>_time``.

    ``month`` is a data variable and there is no ``time`` variable. ``sss`` is the
    salinity restoring, with ``standard_name: sea_water_salinity`` as roms-tools has.
    """
    days = 30.4 * np.arange(12)
    _, lat = _lon_lat()
    seasonal = 0.2 * np.sin(2 * np.pi * np.arange(12) / 12.0)[:, None, None]
    data: dict[str, Any] = {}
    for k, name in enumerate(names):
        vals = (34.0 + k + 0.8 * np.cos(np.deg2rad(lat)) + seasonal) + np.zeros(
            (12, NY, NX)
        )
        attrs = {"long_name": name, "coordinates": f"abs_time month {name}_time"}
        if name == "sss":
            attrs |= {
                "standard_name": "sea_water_salinity",
                "long_name": "sea surface salinity",
                "units": "PSU",
            }
        data[name] = (("time", "eta_rho", "xi_rho"), vals.astype("float32"), attrs)
        data[f"{name}_time"] = (
            "time",
            days,
            {
                "units": "days",
                "long_name": "relative time: days since 2000-01-01 00:00:00",
            },
        )
    data["month"] = ("time", np.arange(1, 13), {"long_name": "Month index (1-12)"})
    data["abs_time"] = ("time", days, {"units": "days since 2000-01-16 00:00:00"})
    xr.Dataset(
        data,
        attrs={"climatology": "True", "model_reference_date": "2000-01-01 00:00:00"},
    ).to_netcdf(path, format=_cdf5(), engine="netcdf4")
    return path


def _dated_month_files(
    root: Path, stem: str, names, *, offset: float, velocities: bool
) -> list[Path]:
    """One boundary file per calendar month of 2010-2013 (``<stem>_YYYYMM.nc``).

    Each holds 2-4 steps five days apart (so month files differ in length, as 28-31 days
    do), in days since the ROMS epoch, as the real roms-tools files are.
    """
    import datetime as dt

    epoch = dt.date.fromisoformat(ROMS_EPOCH)
    files = []
    for year in range(2010, 2014):
        for month in range(1, 13):
            start = (dt.date(year, month, 1) - epoch).days
            n = 2 + (year * 12 + month) % 3
            files.append(
                write_boundary(
                    root / f"{stem}_{year}{month:02d}.nc",
                    start + 5 * np.arange(n),
                    names,
                    offset=offset,
                    velocities=velocities,
                    epoch=ROMS_EPOCH,
                )
            )
    return files


# ------------------------------------------------------------------------ observations

SIGMA_THETA_MLD = "ocean_mixed_layer_thickness_defined_by_sigma_theta"
EAST = "eastward_sea_water_velocity"
_OBS_LAT = np.arange(-41.0, 60.0, 2.0)  # odd degrees: never on the model's even ones
_OBS_LON180 = np.arange(-179.0, 180.0, 2.0)
_WOA_LAT = np.arange(-42.0, 60.0, 4.0)  # 4 degrees: WOA is coarser than the model
_WOA_LON = np.arange(-178.0, 180.0, 4.0)
_WOA_DEPTH = np.array([0, 5, 10, 25, 50, 100, 150, 200, 300, 400, 500, 1000], "float32")
_GLODAP_DEPTH = np.array(
    [0, 10, 20, 30, 50, 75, 100, 125, 150, 200, 250, 300, 400, 500, 1000, 1500],
    "float64",
)
_WOA_VARS = {
    # nickname: (file variable, standard_name, units, model field it stands for)
    "temperature": ("t_an", "sea_water_temperature", "degrees_celsius", "temp"),
    "salinity": ("s_an", "sea_water_practical_salinity", "1", "salt"),
    "nitrate": (
        "n_an",
        "moles_of_nitrate_per_unit_mass_in_sea_water",
        "micromoles_per_kilogram",
        "NO3",
    ),
    "phosphate": (
        "p_an",
        "moles_of_phosphate_per_unit_mass_in_sea_water",
        "micromoles_per_kilogram",
        "PO4",
    ),
    "silicate": (
        "i_an",
        "moles_of_silicate_per_unit_mass_in_sea_water",
        "micromoles_per_kilogram",
        "SiO3",
    ),
    "oxygen": (
        "o_an",
        "moles_of_oxygen_per_unit_mass_in_sea_water",
        "micromoles_per_kilogram",
        "O2",
    ),
}


def _obs_mask(lon180: np.ndarray, lat: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Return ``(depth, lat, lon)`` True at ocean: land and the sea floor are NaN."""
    h, mask = bathymetry()
    lon360 = np.mod(lon180, 360.0)
    # nearest model cell where the model has one; open ocean (4500 m) elsewhere
    ix = np.clip(np.round((lon360 - LON[0]) / 2.0).astype(int), 0, NX - 1)
    iy = np.clip(np.round((lat - LAT[0]) / 2.0).astype(int), 0, NY - 1)
    inside = ((lon360 >= LON[0]) & (lon360 <= LON[-1]))[None, :] & (
        (lat >= LAT[0]) & (lat <= LAT[-1])
    )[:, None]
    hh = np.where(inside, h[np.ix_(iy, ix)], 4500.0)
    wet = np.where(inside, mask[np.ix_(iy, ix)] > 0, True)
    return (depth[:, None, None] <= hh[None]) & wet[None]


def _bounds(centers: np.ndarray) -> np.ndarray:
    half = np.diff(centers).mean() / 2.0
    return np.stack([centers - half, centers + half], axis=1).astype("float32")


def write_woa(path: Path, var: str, month: int | None) -> Path:
    """Write one WOA23-shaped file (``month`` None is the annual field)."""
    short, std, unit, model_name = _WOA_VARS[var]
    z = -_WOA_DEPTH.astype(float)[:, None, None] + np.zeros(
        (1, _WOA_LAT.size, _WOA_LON.size)
    )
    season = 0.5 if month is None else (month - 0.5) / 12.0
    field_ = _tracers(z, _WOA_LAT, season)[model_name][0]
    if unit == "micromoles_per_kilogram":
        field_ = field_ / 1.025
    wet = _obs_mask(_WOA_LON, _WOA_LAT, _WOA_DEPTH.astype(float))
    field_ = np.where(wet, field_, np.nan).astype("float32")
    attrs = {"standard_name": std, "units": unit, "grid_mapping": "crs"}
    ds = xr.Dataset(
        {
            "crs": (
                (),
                np.int32(-2147483647),
                {"grid_mapping_name": "latitude_longitude"},
            ),
            "lat_bnds": (("lat", "nbounds"), _bounds(_WOA_LAT.astype("float32"))),
            "lon_bnds": (("lon", "nbounds"), _bounds(_WOA_LON.astype("float32"))),
            "depth_bnds": (("depth", "nbounds"), _bounds(_WOA_DEPTH)),
            "climatology_bounds": (
                ("time", "nbounds"),
                np.array([[0.0, 1.0]], "float32"),
            ),
            short: (("time", "depth", "lat", "lon"), field_[None], attrs),
        },
        coords={
            "lat": (
                "lat",
                _WOA_LAT.astype("float32"),
                {"standard_name": "latitude", "units": "degrees_north", "axis": "Y"},
            ),
            "lon": (
                "lon",
                _WOA_LON.astype("float32"),
                {"standard_name": "longitude", "units": "degrees_east", "axis": "X"},
            ),
            "depth": (
                "depth",
                _WOA_DEPTH,
                {
                    "standard_name": "depth",
                    "positive": "down",
                    "units": "meters",
                    "axis": "Z",
                },
            ),
            "time": (
                "time",
                np.array([396.5 + (0 if month is None else month - 1)], "float32"),
                {
                    "standard_name": "time",
                    "units": "months since 1955-01-01 00:00:00",
                    "axis": "T",
                    "climatology": "climatology_bounds",
                },
            ),
        },
    )
    ds.to_netcdf(path, format="NETCDF4")
    return path


def write_glodap(path: Path) -> Path:
    """Write a GLODAPv2-shaped climatology: 0-360, ``depth_surface``, ``Depth``."""
    lon = np.arange(21.0, 380.0, 2.0)  # 20.5..379.5 on the real 1 degree grid
    lat = _OBS_LAT
    z = -_GLODAP_DEPTH[:, None, None] + np.zeros((1, lat.size, lon.size))
    tr = _tracers(z, lat, 0.5)
    wet = _obs_mask(np.where(lon > 180, lon - 360.0, lon), lat, _GLODAP_DEPTH)
    attrs = {"units": "micro-mol kg-1"}
    xr.Dataset(
        {
            "TAlk": (
                ("depth_surface", "lat", "lon"),
                np.where(wet, tr["ALK"][0] / 1.025, np.nan),
                {**attrs, "long_name": "moles of alkalinity per unit mass in seawater"},
            ),
            "TCO2": (
                ("depth_surface", "lat", "lon"),
                np.where(wet, tr["DIC"][0] / 1.025, np.nan),
                {
                    **attrs,
                    "long_name": "moles of dissolved inorganic carbon per mass",
                },
            ),
            "Depth": (
                ("depth_surface",),
                _GLODAP_DEPTH,
                {"units": "m", "long_name": "Depth of the standard levels"},
            ),
        },
        coords={
            "lat": ("lat", lat, {"units": "degrees_north"}),
            "lon": ("lon", lon, {"units": "degrees_east"}),
        },
    ).to_netcdf(path)
    return path


def write_holte_talley(path: Path) -> Path:
    """Write a classic-netCDF3 H&T MLD climatology (bare ``iMONTH``/``iLAT``...)."""
    lon = np.arange(21.0, 380.0, 2.0)
    lat = _OBS_LAT
    month = np.arange(1, 13)
    base = 35.0 + 60.0 * np.clip(np.abs(lat) / 50.0, 0, 1)[None, :, None] * (
        1.0
        + 0.4
        * np.cos(
            2 * np.pi * (month[:, None, None] - 1) / 12.0
            + np.deg2rad(lat)[None, :, None]
        )
    )
    shape = (12, lat.size, lon.size)
    dt = np.broadcast_to(base, shape).astype("float64")
    wet = _obs_mask(np.where(lon > 180, lon - 360.0, lon), lat, np.array([0.0]))[0]
    dt = np.where(wet[None], dt, np.nan)
    grid = ("iMONTH", "iLAT", "iLON")
    xr.Dataset(
        {
            "mld_dt_mean": (grid, dt, {"standard_name": "mld_dt_mean", "units": "m"}),
            "mld_da_mean": (
                grid,
                dt + 8.0,
                {"standard_name": "mld_da_mean", "units": "m"},
            ),
            "lat": (
                ("iLAT",),
                lat,
                {"standard_name": "latitude", "units": "degrees_north"},
            ),
            "lon": (
                ("iLON",),
                lon,
                {"standard_name": "longitude", "units": "degrees_east"},
            ),
            "month": (("iMONTH",), month, {"standard_name": "Month"}),
        }
    ).to_netcdf(path, format="NETCDF3_CLASSIC")
    return path


def _stamps_to_days(stamps) -> list[np.datetime64]:
    return [np.datetime64(_stamp_day(s)) for s in stamps]


def write_chlorophyll(path: Path, days: list[np.datetime64]) -> Path:
    """Write the daily gap-free L4 chlorophyll (mg/m3) on ``days``."""
    lon = np.arange(-179.5, 180.0, 1.0)
    lat = np.arange(-40.5, 60.0, 1.0)
    base = (
        0.05 + 0.3 * np.exp(-(((lat[:, None] - 5.0) / 12.0) ** 2)) + 0.0 * lon[None, :]
    )
    wet = _obs_mask(lon, lat, np.array([0.0]))[0]
    vals = np.stack(
        [np.where(wet, base * (1.0 + 0.1 * k), np.nan) for k in range(len(days))]
    )
    xr.Dataset(
        {
            "CHL": (
                ("time", "latitude", "longitude"),
                vals.astype("float32"),
                {
                    "units": "milligram m-3",
                    "long_name": "Chlorophyll-a",
                    "standard_name": "mass_concentration_of_chlorophyll_a_in_sea_water",
                },
            )
        },
        coords={
            "time": ("time", np.array(days, "datetime64[ns]")),
            "latitude": ("latitude", lat, {"units": "degrees_north", "axis": "Y"}),
            "longitude": ("longitude", lon, {"units": "degrees_east", "axis": "X"}),
        },
    ).to_netcdf(path)
    return path


def write_duacs(path: Path, days: np.ndarray) -> Path:
    """Write daily DUACS ``adt`` and ``sla`` on a coarse grid."""
    lon, lat = _OBS_LON180, _OBS_LAT
    t = (days - days[0]).astype(float)
    pattern = (
        0.4 * np.cos(np.deg2rad(lat))[:, None]
        + 0.1 * np.sin(np.deg2rad(lon) * 2.0)[None, :]
    )
    adt = pattern[None] + 0.05 * np.sin(2 * np.pi * t / 120.0)[:, None, None]
    wet = _obs_mask(lon, lat, np.array([0.0]))[0]
    adt = np.where(wet[None], adt, np.nan).astype("float32")
    xr.Dataset(
        {
            "adt": (
                ("time", "latitude", "longitude"),
                adt,
                {"units": "m", "standard_name": "sea_surface_height_above_geoid"},
            ),
            "sla": (
                ("time", "latitude", "longitude"),
                adt - 0.3,
                {"units": "m", "standard_name": "sea_surface_height_above_sea_level"},
            ),
        },
        coords={
            "time": ("time", days.astype("datetime64[ns]")),
            "latitude": ("latitude", lat, {"units": "degrees_north", "axis": "Y"}),
            "longitude": ("longitude", lon, {"units": "degrees_east", "axis": "X"}),
        },
    ).to_netcdf(path)
    return path


_GLORYS_DEPTH = np.array([0.494, 50.0, 200.0, 500.0])


def write_glorys(path: Path, days: np.ndarray) -> Path:
    """Write a GLORYS-shaped temperature/salinity cube (5-day steps keep it small)."""
    lon, lat = _OBS_LON180, _OBS_LAT
    z = -_GLORYS_DEPTH[:, None, None] + np.zeros((1, lat.size, lon.size))
    wet = _obs_mask(lon, lat, _GLORYS_DEPTH)
    out = {}
    for name, key, std, unit in (
        ("thetao", "temp", "sea_water_potential_temperature", "degrees_C"),
        ("so", "salt", "sea_water_practical_salinity", "1e-3"),
    ):
        cube = np.stack(
            [
                np.where(
                    wet,
                    _tracers(z, lat, (d - days[0]).astype(float) / 365.0 + 0.05)[key][
                        0
                    ],
                    np.nan,
                )
                for d in days
            ]
        )
        out[name] = (
            ("time", "depth", "latitude", "longitude"),
            cube.astype("float32"),
            {"standard_name": std, "units": unit},
        )
    xr.Dataset(
        out,
        coords={
            "time": ("time", days.astype("datetime64[ns]")),
            "depth": (
                "depth",
                _GLORYS_DEPTH,
                {"units": "m", "positive": "down", "axis": "Z"},
            ),
            "latitude": ("latitude", lat, {"units": "degrees_north", "axis": "Y"}),
            "longitude": ("longitude", lon, {"units": "degrees_east", "axis": "X"}),
        },
    ).to_netcdf(path)
    return path


_CRAVATTE_DEPTH = np.array(
    [
        30.0,
        40,
        50,
        60,
        70,
        80,
        90,
        100,
        120,
        140,
        160,
        180,
        200,
        225,
        250,
        275,
        300,
        350,
        400,
        450,
        500,
    ]
)


def write_cravatte(path: Path, lat: np.ndarray) -> Path:
    """Write a Cravatte et al. band-mean ADCP section: ``(DEPTH, LATI)``, cm/s."""
    d = _CRAVATTE_DEPTH[:, None]
    la = lat[None, :]
    u = 60.0 * np.exp(-((la / 3.0) ** 2)) * np.exp(
        -(((d - 100.0) / 60.0) ** 2)
    ) - 20.0 * np.exp(-((la / 6.0) ** 2)) * np.exp(-d / 30.0)
    attrs = {"long_name": "Zonal velocity from all SADCP data", "unit": "cm/s"}
    xr.Dataset(
        {
            "U_SADCP": (("DEPTH", "LATI"), u, attrs),
            "U_SADCPD": (("DEPTH", "LATI"), u + 0.5, attrs),
        },
        coords={
            "DEPTH": (
                "DEPTH",
                _CRAVATTE_DEPTH,
                {
                    "units": "meters",
                    "axis": "Z",
                    "positive": "down",
                    "standard_name": "depth",
                },
            ),
            "LATI": (
                "LATI",
                lat,
                {"units": "degrees_north", "axis": "Y", "standard_name": "latitude"},
            ),
        },
    ).to_netcdf(path, format="NETCDF3_CLASSIC")
    return path


# --------------------------------------------------------------------------- catalogs


def _reader_chain_holte_talley(path: str):
    """Return the README's reader chain: promote lat/lon/month, swap the dims."""
    from intake.readers import datatypes, readers

    return (
        readers.XArrayDatasetReader(
            datatypes.HDF5(url=path), engine="scipy", decode_times=False, chunks={}
        )
        .set_coords(["lat", "lon", "month"])
        .swap_dims({"iLAT": "lat", "iLON": "lon", "iMONTH": "month"})
    )


def _reader_chain_cravatte(path: str, lon: float):
    """Return the README's reader chain: give a lon-less band mean its longitude."""
    from intake.readers import datatypes, readers

    return readers.XArrayDatasetReader(
        datatypes.NetCDF3(url=path), engine="scipy", chunks={}
    ).expand_dims(lon=[lon])


def build_observation_catalogs(
    catalog_dir: Path, data_dir: Path, window: tuple[str, str], restart_days: list[str]
) -> None:
    """Write the observation stand-ins and the catalogs naming them."""
    catalog_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # -- WOA23: annual, month01..12 and the stacked monthly entry -------------
    woa = build.new_catalog(title="WOA23 stand-in")
    for var, (short, std, _unit, _m) in _WOA_VARS.items():
        meta = {
            "climatology": True,
            "featureType": "grid",
            "depth_coord": "depth",
            "standard_names": {short: std},
        }
        kw = {"engine": "h5netcdf", "chunks": {}}
        annual = write_woa(data_dir / f"woa23_{var}_annual.nc", var, None)
        build.add_source(
            woa,
            f"woa23_{var}_annual",
            str(annual),
            name_map=None,
            reader_kwargs=kw,
            climatology_period="annual",
            **meta,
        )
        months = [
            write_woa(data_dir / f"woa23_{var}_month{m:02d}.nc", var, m)
            for m in range(1, 13)
        ]
        for m, f in enumerate(months, 1):
            build.add_source(
                woa,
                f"woa23_{var}_month{m:02d}",
                str(f),
                name_map=None,
                reader_kwargs=kw,
                climatology_period=f"month{m:02d}",
                **meta,
            )
        build.add_source(
            woa,
            f"woa23_{var}_monthly",
            [str(f) for f in months],
            name_map=None,
            reader_kwargs={
                **kw,
                "combine": "nested",
                "concat_dim": "time",
                "compat": "override",
                "coords": "minimal",
                "data_vars": "minimal",
            },
            climatology_period="monthly",
            **meta,
        )
    build.save(woa, catalog_dir / "woa.yaml")

    # -- GLODAPv2 --------------------------------------------------------------
    glodap = build.new_catalog(title="GLODAPv2 stand-in")
    build.add_source(
        glodap,
        "glodap",
        str(write_glodap(data_dir / "glodap.nc")),
        name_map=None,
        reader_kwargs={"engine": "h5netcdf", "chunks": {}},
        axes={"X": "lon", "Y": "lat", "Z": "Depth"},
        depth_coord="depth_surface",
        climatology=True,
        featureType="grid",
        standard_names={
            "TAlk": "sea_water_alkalinity_expressed_as_mole_equivalent",
            "TCO2": "mole_concentration_of_dissolved_inorganic_carbon_in_sea_water",
        },
    )
    build.save(glodap, catalog_dir / "glodap.yaml")

    # -- Holte & Talley --------------------------------------------------------
    ht = write_holte_talley(data_dir / "ht.nc")
    build.build_catalog(
        {
            "holte_talley_mld_clim": {
                "reader": _reader_chain_holte_talley(str(ht)),
                "standard_names": {"mld_dt_mean": SIGMA_THETA_MLD},
                "climatology": True,
            }
        },
        catalog_dir / "mld_climatologies.yaml",
        title="MLD climatologies stand-in",
        name_map=None,
    )

    # -- Copernicus Marine: chlorophyll, DUACS, GLORYS -------------------------
    start, end = np.datetime64(window[0]), np.datetime64(window[1])
    daily = np.arange(start, end + np.timedelta64(1, "D"), dtype="datetime64[D]")
    five_day = daily[::5]
    cop = build.new_catalog(title="Copernicus stand-in")
    kw = {"engine": "netcdf4", "chunks": {}, "decode_times": True}
    build.add_source(
        cop,
        "chl_gapfree_my_daily_geo",
        str(
            write_chlorophyll(
                data_dir / "chl.nc", [np.datetime64(d) for d in restart_days]
            )
        ),
        name_map=None,
        reader_kwargs=kw,
        featureType="grid",
        axes={"T": "time", "X": "longitude", "Y": "latitude"},
        standard_names={"CHL": "mass_concentration_of_chlorophyll_a_in_sea_water"},
    )
    build.add_source(
        cop,
        "ssh_duacs_my_daily_geo",
        str(write_duacs(data_dir / "duacs.nc", daily)),
        name_map=None,
        reader_kwargs=kw,
        featureType="grid",
        axes={"T": "time", "X": "longitude", "Y": "latitude"},
        standard_names={
            "adt": "sea_surface_height_above_geoid",
            "sla": "sea_surface_height_above_sea_level",
        },
    )
    build.add_source(
        cop,
        "glorys_my_daily_timeseries",
        str(write_glorys(data_dir / "glorys.nc", five_day)),
        name_map=None,
        reader_kwargs=kw,
        featureType="grid",
        axes={"T": "time", "X": "longitude", "Y": "latitude", "Z": "depth"},
        standard_names={
            "thetao": "sea_water_potential_temperature",
            "so": "sea_water_practical_salinity",
        },
    )
    build.save(cop, catalog_dir / "copernicus.yaml")

    # -- Cravatte et al. (2017) zonal currents: lon-less band means ------------
    crav = build.new_catalog(title="Cravatte stand-in")
    for name, lon, lats in (
        ("cravatte_u_179e_160w", 190.0, np.arange(-20.0, 21.0, 1.0)),
        ("cravatte_u_160w_120w", 220.0, np.arange(-10.0, 19.0, 1.0)),
    ):
        path = write_cravatte(data_dir / f"{name}.cdf", lats)
        build.add_source(
            crav,
            name,
            reader=_reader_chain_cravatte(str(path), lon),
            name_map=None,
            standard_names={"U_SADCPD": EAST},
            units={"U_SADCP": "cm/s", "U_SADCPD": "cm/s"},
            featureType="grid",
            climatology=True,
            doi="10.6096/12",
        )
    build.save(crav, catalog_dir / "cravatte_zonal_currents.yaml")


# ---------------------------------------------------------------------------- assembly


def build_fixture(root: Path) -> SuiteFixture:
    """Write every stand-in under ``root`` and return where each one lives."""
    root.mkdir(parents=True, exist_ok=True)
    grids = root / "grids"
    grids.mkdir(exist_ok=True)
    model_grid = write_model_grid(grids / "PACMED12km_grid.nc")
    forcing_grid = write_forcing_grid(grids / "PACMED12km_forge_grid.nc")
    fx = SuiteFixture(root, root / "catalogs", model_grid, forcing_grid)

    # -- model output: two runs of monthly restarts, and their starting restarts --
    def restarts(folder: str, stamps, **kw) -> Path:
        d = root / "runs" / folder
        d.mkdir(parents=True, exist_ok=True)
        for stamp in stamps:
            write_restart(d / f"output_rst.{stamp}234500.nc", stamp, **kw)
        return d

    test_dir = restarts("pacmed_pyesper", RESTART_STAMPS)
    fx.sources = {
        "pacmed_pyesper": {
            "files": str(test_dir / "output_rst.*.nc"),
            "grid": str(model_grid),
        },
    }
    # the CCS mid-run check reads its own run; here it is the same stand-in
    fx.sources["ccs_4km"] = dict(fx.sources["pacmed_pyesper"])

    # -- the model-comparison pair: both runs from 2010-01, split over task folders ----
    def tasks(folder: str, layout: dict[str, tuple[str, ...]], kw) -> None:
        for task, stamps in layout.items():
            d = root / "runs" / folder / "tasks" / task / "output"
            d.mkdir(parents=True, exist_ok=True)
            for stamp in stamps:
                write_restart(d / f"output_rst.{stamp}234500.nc", stamp, **kw(stamp))

    def split(stamps) -> dict[str, tuple[str, ...]]:
        """Split the stamps over the 2010 task folders, as the real run did."""
        daily = [s for s in stamps if s in MC_DAILY_STAMPS]
        ends = [s for s in stamps if s not in MC_DAILY_STAMPS and s < "2011"]
        return {
            "first_wk": tuple(daily[:3]),
            "second_wk": tuple(daily[3:5]),
            "third_2wks": tuple(daily[5:]),
            "first_half": tuple(e for e in ends if e < "20100601"),
            "all_the_rest": tuple(e for e in ends if e >= "20100601"),
        }

    def drift(stamp: str) -> float:
        """Return the reference run's offset from the test run, growing with time."""
        return 0.2 + 0.15 * (int(stamp[:4]) - 2010) + 0.05 * int(stamp[4:6]) / 12.0

    every = (*MC_DAILY_STAMPS, *MC_MONTH_ENDS)
    later = tuple(s for s in every if s >= "2011")
    # pyesper (test): 2010 in the task folders, the rest in gathered_output
    tasks("cstar_pyesper", split(every), lambda st: {"shift": 0.0})
    gathered = root / "runs" / "cstar_pyesper_2015" / "gathered_output"
    gathered.mkdir(parents=True, exist_ok=True)
    for stamp in later:
        write_restart(gathered / f"output_rst.{stamp}234500.nc", stamp, shift=0.0)
    # MLD-interp (reference): the same, plus a segment's 2011-01-01 test restart that
    # the month-end glob must leave out
    tasks("cstar_mld", split(every), lambda st: {"shift": drift(st)})
    tasks(
        "cstar_mld_2015",
        {
            "m201101": ("20110131", "20110101"),
            "m201104": ("20110430",),
            "rest": tuple(s for s in later if s not in ("20110131", "20110430")),
        },
        lambda st: {"shift": drift(st) if st != "20110101" else 9.0},
    )
    mc_globs = {
        "pyesper": [
            str(root / "runs/cstar_pyesper/tasks/*/output/output_rst.2010*.nc"),
            str(root / "runs/cstar_pyesper_2015/gathered_output/output_rst.*.nc"),
        ],
        "mld_interp": [
            str(root / "runs/cstar_mld/tasks/*/output/output_rst.2010*.nc"),
            str(
                root
                / "runs/cstar_mld_2015/tasks/*/output"
                / "output_rst.??????[23][0-9]234500.nc"
            ),
        ],
    }
    for name, globs in mc_globs.items():
        fx.sources[name] = {"files": globs, "grid": str(model_grid)}
    fx.dated_sources = frozenset(mc_globs)
    fx.restart_days = [_stamp_day(s) for s in RESTART_STAMPS]

    # -- roms-tools forcing ----------------------------------------------------
    f = root / "forcing"
    f.mkdir(exist_ok=True)

    def entry(files) -> dict[str, Any]:
        return {"files": [str(p) for p in files], "grid": str(forcing_grid)}

    clim_days = np.arange(12) * 30.4
    ic = f / "ic"
    ic.mkdir(exist_ok=True)
    fx.forcing = {
        # initial conditions (2010-01-01); pyesper's carries no hc attribute
        "pyesper_ic": entry(
            [write_initial_conditions(ic / "pyesper_ic.nc", "20100101", shift=0.1)]
        ),
        "mld_ic": entry(
            [
                write_initial_conditions(
                    ic / "mld_ic.nc", "20100101", shift=0.3, hc_attr=True
                )
            ]
        ),
        # 2010-2013 monthly boundary files: pyesper's physics and ESPER BGC, the MLD
        # run's physics, and its monthly BGC climatology
        "pyesper_bry_phys": entry(
            _dated_month_files(
                f, "pyesper_bry_phys", _PHYSICS, offset=0.0, velocities=True
            )
        ),
        "pyesper_bry_bgc": entry(
            _dated_month_files(
                f, "pyesper_bry_bgc", _ESPER, offset=0.0, velocities=False
            )
        ),
        "mld_bry_phys": entry(
            _dated_month_files(f, "mld_bry_phys", _PHYSICS, offset=0.2, velocities=True)
        ),
        "mld_bry_bgc": entry(
            [
                write_boundary(
                    f / "mld_bry_bgc_clim.nc",
                    clim_days,
                    _UNIFIED,
                    climatology=True,
                    offset=1.0,
                )
            ]
        ),
        "pyesper_bry_bgcclim": entry(
            [
                write_boundary(
                    f / "pyesper_bry_bgcclim.nc", clim_days, _UNIFIED, climatology=True
                )
            ]
        ),
        "pyesper_surface_restoring_clim": entry(
            [write_surface_climatology(f / "surface-restoring_clim.nc")]
        ),
        "pacmed2015_surface_restoring_clim": entry(
            [write_surface_climatology(f / "surface-restoring_clim_2015.nc")]
        ),
        "pyesper_surface_bgc_clim": entry(
            [
                write_surface_climatology(
                    f / "surface-bgc_clim.nc", ("dust", "iron", "nox", "nhy")
                )
            ]
        ),
        "pacmed2015_surface_bgc_clim": entry(
            [
                write_surface_climatology(
                    f / "surface-bgc_clim_2015.nc", ("dust", "iron", "nox", "nhy")
                )
            ]
        ),
    }

    build_observation_catalogs(fx.catalog_dir, root / "obs", fx.window, fx.restart_days)
    return fx


# ------------------------------------------------------------------- localizing a suite

_ISO_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}")


def _adapt_dates(node: Any, fx: SuiteFixture) -> Any:
    """Fit a suite's dates to the fixture: clip windows, date lists -> restarts.

    A window edge outside the run (a ``min`` after its last restart, a ``max`` before
    its first) falls back to the run's own edge, so "from 2013" still selects the run
    when the stand-in's restarts are all of 2011.
    """
    lo, hi = fx.window
    if isinstance(node, dict):
        out = {k: _adapt_dates(v, fx) for k, v in node.items()}
        for key in ("min", "max"):
            v = node.get(key)
            if isinstance(v, str) and _ISO_DAY.match(v) and set(node) <= {"min", "max"}:
                if key == "min":
                    out[key] = lo if v > hi else max(v, lo)
                else:
                    out[key] = hi if v < lo else min(v, hi)
        return out
    if isinstance(node, list):
        if node and all(isinstance(v, str) and _ISO_DAY.match(v) for v in node):
            return list(fx.restart_days)
        return [_adapt_dates(v, fx) for v in node]
    return node


def _divisor_below(size: int, wanted: int) -> int:
    """Return the largest proper divisor of ``size`` that is at most ``wanted``."""
    cap = min(wanted, max(size // 2, 1))
    return next(d for d in range(cap, 0, -1) if size % d == 0)


def _fit_subchunk(spec: dict[str, int]) -> dict[str, int]:
    """Re-scale a real run's ``subchunk`` (74 rows of 962) to the fixture's grids."""
    sizes = {"eta_rho": NY, "xi_rho": NX, "s_rho": NS, "s_w": NS + 1}
    return {
        dim: (
            n
            if dim not in sizes or (sizes[dim] % n == 0 and n < sizes[dim])
            else _divisor_below(sizes[dim], n)
        )
        for dim, n in spec.items()
    }


def _widen_regions(node: Any, half_width: float = 2.5) -> None:
    """Widen every ``regions:`` box to ``2 * half_width`` degrees around its centre.

    The real T-S boxes are one degree wide; the fixture's grids are two degrees apart,
    so a box that small holds no cell and a page selecting it would draw nothing.
    """
    if isinstance(node, dict):
        regions = node.get("regions")
        if isinstance(regions, dict):
            for box in regions.values():
                for axis in ("lon", "lat"):
                    rng = box.get(axis)
                    if isinstance(rng, dict) and {"min", "max"} <= set(rng):
                        c = 0.5 * (rng["min"] + rng["max"])
                        rng["min"], rng["max"] = c - half_width, c + half_width
        for v in node.values():
            _widen_regions(v, half_width)
    elif isinstance(node, list):
        for v in node:
            _widen_regions(v, half_width)


def localize_suite(src_yaml: str | Path, tmp_path: Path, fx: SuiteFixture) -> Path:
    """Rewrite a suite YAML to run against ``fx`` and write it under ``tmp_path``.

    Generic by walking, not by naming: every ``refresh.sources`` / ``refresh.forcing``
    entry gets its ``files``/``grid`` from the fixture by *name* (a name it does
    not have is a ``KeyError`` saying so), ``ref:`` paths move under ``tmp_path/refs``
    (entries that shared one still share it), ``subchunk:`` is re-scaled to the
    grid, dates fit the fixture's run, T-S ``regions:`` are widened (see
    :func:`_widen_regions`), and the output, cache and catalog locations are
    ``tmp_path``'s.
    The observation catalogs are *copied* into ``tmp_path`` so a run's own
    ``refresh.catalog`` never leaks into another test.
    """
    import yaml

    src = Path(src_yaml)
    raw = yaml.safe_load(src.read_text())
    refresh = raw.pop("refresh", None)
    # a suite on the dated stand-ins (model-comparison) already names real dates
    names = {e["name"] for e in (refresh or {}).get("sources") or []}
    out = raw if names and names <= fx.dated_sources else _adapt_dates(raw, fx)
    _widen_regions(out)

    catalog_dir = tmp_path / "catalogs"
    shutil.copytree(fx.catalog_dir, catalog_dir, dirs_exist_ok=True)
    refs = tmp_path / "refs"
    refs.mkdir(exist_ok=True)

    out["output_dir"] = str(tmp_path / "out")
    if "cache_dir" in out:
        out["cache_dir"] = str(tmp_path / "cache")
    out["catalog_search_paths"] = [str(catalog_dir)]
    if shutil.which("gs") is None:
        out.pop(
            "pdf_images", None
        )  # Ghostscript is optional: not what this test checks
    # run_info paths pointed at C-Star files that are not here: let the cover discover
    # what it can instead (the fixture has no C-Star run folder, so it names nothing)
    info = out.get("run_info")
    if isinstance(info, dict):
        info.pop("workplan", None)
        info.pop("blueprint", None)

    if refresh is not None:
        new: dict[str, Any] = {
            "catalog": str(catalog_dir / Path(refresh["catalog"]).name)
        }
        for kind, table in (("sources", fx.sources), ("forcing", fx.forcing)):
            entries = []
            for entry in refresh.get(kind) or []:
                name = entry["name"]
                if name not in table:
                    raise KeyError(
                        f"refresh.{kind}: {name!r} has no stand-in in "
                        f"tests/_suite_fixture.py (it has {sorted(table)}); "
                        "add one to build_fixture()"
                    )
                e = dict(entry)
                e.update(table[name])
                if "ref" in e:
                    e["ref"] = str(refs / Path(e["ref"]).name)
                if "subchunk" in e:
                    e["subchunk"] = _fit_subchunk(e["subchunk"])
                entries.append(e)
            if entries:
                new[kind] = entries
        out["refresh"] = new

    path = tmp_path / "suites" / src.name
    path.parent.mkdir(exist_ok=True)
    path.write_text(yaml.safe_dump(out, sort_keys=False, width=100))
    return path
