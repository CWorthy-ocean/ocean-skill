"""Run-health diagnostic for ROMS model output: is this run still sane.

:func:`health` makes one streaming pass over a catalog source, a time step (a restart
file, for a run checked mid-way) at a time, over every 2-D and 3-D variable on rho, u or
v points. Per step and variable it records how many ocean cells are NaN/Inf, negative or
outside physical bounds, where the minimum and maximum are, whether the field is zero
everywhere or an exact copy of another variable (``ALK_ALT_CO2`` identical to ``ALK``
means the CDR perturbation never entered), and the volume integral (inventory) of every
3-D rho tracer. A handful of derived series -- total kinetic energy, the volume of
low-oxygen water, the deep (> 1000 m) mean of temperature, salinity, oxygen and DIC, the
added-alkalinity or extra-carbon inventory ``X - X_ALT_CO2`` of a CDR run -- go in the
same table. :class:`RunHealth` then reads drift and sudden steps off those series and
draws four report pages (:data:`HEALTH_DRAWS`).

Everything is masked with ``mask_rho``: UCLA-ROMS writes land as 0, and a statistic over
the whole array would count it as ocean. u and v are ocean where both of their rho
neighbors are; kinetic energy averages them to rho points first. Layer thickness is the
distance between the cell's own two interfaces
(:func:`ocean_skill.vertical._roms_thickness`), so an inventory follows the free
surface.

**One dask compute per variable.** A step's 3-D field is a gigabyte-class array and a
run has ~40 of them, so each variable's reductions are built lazily and evaluated
together, streaming its chunks once; the step's cell volumes and depths are computed
once and shared. (One compute for the whole step ran out of memory on a large grid.)
Only candidate pairs of copies are read a second time.

**Per-step cache.** A restart never changes once written, so each step's table is kept
under :mod:`ocean_skill.cache`'s ``health`` kind and a mid-run re-run reads only the
restarts that are new. The key is :data:`HEALTH_VERSION`, the source's catalog
fingerprint, the step's time stamp and a hash of its free surface and the surface level
of every scanned variable -- a cheap content sample that misses when a restart is
rewritten in place, which an identity key alone would not notice.

**Test mode.** A source read under :func:`ocean_skill.sources.truncate_time`
(``--smoke``) has a strided grid and thinned restarts, so its numbers are not physical.
Every figure says so, and a step is only looked for between restarts that are adjacent
in the full record, never across the gap the truncation leaves.

The module is called ``run_health`` rather than ``health`` so the package can expose
``osk.health`` as the function: a submodule of the same name would be clobbered as a
package attribute (see ``_CallablePlotPackage`` in :mod:`ocean_skill`).
"""

from __future__ import annotations

import hashlib
import json
import textwrap
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ocean_skill.config import HEALTH_DRAWS

__all__ = ["HEALTH_DRAWS", "HEALTH_VERSION", "RunHealth", "health"]

#: Bumped when the statistics a step records, or how they are computed, change, so a
#: cached step from before is not served as one from after. Part of every step's key.
HEALTH_VERSION = 2

#: Said on every figure of a run read under ``truncate_time``.
TEST_MODE_BANNER = (
    "Test mode: values are not physical (grid strided, restarts thinned)."
)

#: Variables never scanned: ROMS's barotropic work arrays, river masks and the clock.
_SKIP = frozenset(
    {
        "DU_avg1",
        "DV_avg1",
        "DU_avg2",
        "DV_avg2",
        "DU_avg_bak",
        "DV_avg_bak",
        "u_slow",
        "v_slow",
        "p_slow",
        "riv_umask",
        "riv_vmask",
        "time_step",
        "ocean_time",
        "time",
    }
)

#: Variables for which a negative value is ordinary; every other one is a concentration.
_NOT_CONCENTRATION = frozenset(
    {"temp", "salt", "u", "v", "zeta", "ubar", "vbar", "hbls", "hbbl"}
)

#: Inclusive ``(low, high)`` bounds outside which a value is out of bounds.
_RANGES = {"temp": (-2.5, 36.0), "salt": (0.0, 42.0)}
#: ``|value|`` above which a velocity (m/s) or the free surface (m) is out of bounds.
_ABS_LIMITS = {"u": 3.0, "v": 3.0, "ubar": 3.0, "vbar": 3.0, "zeta": 10.0}
#: Inclusive bounds for a pH variable (``MARBL_PH_3D``, ``PH_ALT_CO2``, ...).
_PH_RANGE = (6.5, 9.0)
#: The boundary-layer depths (m), bounded below by 0 and above by the local depth ``h``.
_LAYER_DEPTHS = ("hbls", "hbbl")

#: Depth (m) below which a cell counts as deep water for the deep means.
_DEEP_M = 1000.0
#: The variables whose deep (volume-weighted) mean is kept.
_DEEP_MEANS = ("temp", "salt", "O2", "DIC")
#: Oxygen thresholds (mmol m-3) for the volume of low-oxygen water: hypoxic and suboxic.
_HYPOXIC = (60.0, 20.0)

#: A step is a jump when its rate of change is this many times the median rate...
_JUMP_FACTOR = 20.0
#: ...and the change is at least this fraction of the value it started from.
_JUMP_MIN_FRACTION = 1e-3
#: A negative or out-of-bounds fraction of the ocean cells (%) worth reporting.
_PROBLEM_PCT = 0.1
#: Adjacent steps a series needs before a jump can be told from its ordinary rate.
_JUMP_MIN_PAIRS = 3

#: The ``steps`` table's columns, in order: one row per (time, variable).
_COLUMNS = (
    "time",
    "variable",
    "kind",
    "n_ocean",
    "n_nan",
    "n_inf",
    "n_neg",
    "n_bad",
    "pct_neg",
    "pct_bad",
    "min",
    "min_lon",
    "min_lat",
    "min_depth",
    "max",
    "max_lon",
    "max_lat",
    "max_depth",
    "all_zero",
    "copy_of",
    "total",
    "sumsq",
    "value",
)

#: The vertical dimension's name where the catalog entry does not say.
_S_DIM = "s_rho"

#: Plankton groups and the quantities each carries, for the ``plankton`` page.
_GROUPS = (("sp", "small phytoplankton"), ("diat", "diatom"), ("diaz", "diazotroph"))
_PLANKTON_PARTS = ("C", "Chl", "P", "Fe")
_ZOO = "zooC"
#: g C per mmol C, to turn the C:Chl of two inventories (mmol C / mg Chl) into g/g.
_G_C_PER_MMOL = 12.011

#: Readable names for the variables a report shows; anything else is shown as named.
_NAMES = {
    "temp": "Temperature",
    "salt": "Salinity",
    "NO3": "Nitrate",
    "PO4": "Phosphate",
    "SiO3": "Silicate",
    "NH4": "Ammonium",
    "O2": "Oxygen",
    "Fe": "Iron",
    "ALK": "Alkalinity",
    "DIC": "DIC",
}

#: Said of a step whose two restarts a job segment starts between.
JOIN_NOTE = " (at a segment join)"

#: Severity order of :meth:`RunHealth.problems` kinds, most severe first.
_SEVERITY = (
    "nonfinite",
    "negative",
    "out_of_bounds",
    "copy",
    "all_zero",
    "step",
    "repeat",
)

#: Rows and panels a report page holds before it continues on the next.
_TABLE_ROWS = 30
_PANELS_PER_PAGE = 20
_SUMMARY_LINES = 62
_SUMMARY_PROBLEMS = 40


# --------------------------------------------------------------------------------------
# Opening the run


@dataclass
class _Run:
    """What one pass over a source needs, found once: variables, masks and geometry."""

    ds: Any
    meta: dict
    tdim: str
    s_dim: str
    standard: dict  # native name -> the name in the standardized Dataset
    kinds: dict  # native name -> "rho3", "rho2", "u3", "u2", "v3" or "v2"
    mask: np.ndarray  # (eta_rho, xi_rho) bool, True = ocean
    lon: np.ndarray
    lat: np.ndarray
    h: np.ndarray
    area: np.ndarray
    hc: float
    vtransform: int
    sigma_r: np.ndarray
    cs_r: np.ndarray
    zeta_name: str | None

    def dims(self, kind: str) -> tuple[str, ...]:
        """Return the dimensions a ``kind`` of variable is transposed to (no time)."""
        horizontal = {
            "r": ("eta_rho", "xi_rho"),
            "u": ("eta_rho", "xi_u"),
            "v": ("eta_v", "xi_rho"),
        }[kind[0]]
        return (self.s_dim, *horizontal) if kind.endswith("3") else horizontal

    def ocean(self, kind: str) -> np.ndarray:
        """Return the ocean mask on a variable's own points (u/v: both sides wet)."""
        if kind[0] == "u":
            return self.mask[:, :-1] & self.mask[:, 1:]
        if kind[0] == "v":
            return self.mask[:-1] & self.mask[1:]
        return self.mask


def _open_run(ds, meta: dict) -> _Run:
    """Read what a health pass needs off the standardized Dataset ``ds``."""
    from ocean_skill import roms

    if ds.attrs.get("ocean_skill_model") != "roms":
        raise ValueError(
            "health works on ROMS model output; this source is not a ROMS run "
            f"(ocean_skill_model={ds.attrs.get('ocean_skill_model')!r})."
        )
    tdim = meta.get("time_dim", "time")
    s_dim = meta.get("vertical", {}).get("s_dim", _S_DIM)
    if tdim not in ds.dims:
        raise ValueError(f"health needs a time axis ({tdim!r}); this source has none.")
    missing = [
        n
        for n in ("mask_rho", "h", "lon", "lat", "cell_area", "z_rho")
        if n not in ds.coords
    ]
    if missing:
        raise ValueError(
            f"health needs the grid fields {missing}, which the source's grid does not "
            "give (a grid file with pm/pn, h, mask_rho and lon/lat)."
        )
    reverse = {std: nat for nat, std in (meta.get("standard_names") or {}).items()}
    layouts = {
        (s_dim, "eta_rho", "xi_rho"): "rho3",
        ("eta_rho", "xi_rho"): "rho2",
        (s_dim, "eta_rho", "xi_u"): "u3",
        ("eta_rho", "xi_u"): "u2",
        (s_dim, "eta_v", "xi_rho"): "v3",
        ("eta_v", "xi_rho"): "v2",
    }
    standard: dict[str, str] = {}
    kinds: dict[str, str] = {}
    for name, da in ds.data_vars.items():
        native = reverse.get(str(name), str(name))
        if native in _SKIP or tdim not in da.dims:
            continue
        for dims, kind in layouts.items():
            if set(da.dims) == {tdim, *dims}:
                standard[native], kinds[native] = str(name), kind
    if not kinds:
        raise ValueError("health found no 2-D or 3-D variable on rho, u or v points.")
    hc, vtransform = roms._vertical_params(ds, meta)
    zeta_name = next((n for n in roms.FREE_SURFACE_NAMES if n in ds.variables), None)
    return _Run(
        ds=ds,
        meta=meta,
        tdim=tdim,
        s_dim=s_dim,
        standard=dict(sorted(standard.items())),
        kinds=kinds,
        mask=np.asarray(ds["mask_rho"].values == 1),
        lon=np.asarray(ds["lon"].values, dtype=float),
        lat=np.asarray(ds["lat"].values, dtype=float),
        h=np.asarray(ds["h"].values, dtype=float),
        area=np.asarray(ds["cell_area"].values, dtype=float),
        hc=hc,
        vtransform=vtransform,
        sigma_r=np.asarray(ds["sigma_r"].values, dtype=float),
        cs_r=np.asarray(ds["Cs_r"].values, dtype=float),
        zeta_name=zeta_name,
    )


def _positions(times: pd.DatetimeIndex, time) -> range:
    """Return the positions of ``times`` a ``time=`` selection keeps, in order.

    ``None`` keeps every step, ``"latest"`` the last one, a ``{"min": ..., "max": ...}``
    dict the steps from ``min`` to ``max`` inclusive (either key optional; a date-only
    ``max`` includes that whole day). Slices, never an integer list: that is a dask
    ``take`` over the whole record.
    """
    n = len(times)
    if time is None:
        lo, hi = 0, n
    elif isinstance(time, str) and time == "latest":
        lo, hi = n - 1, n
    elif isinstance(time, dict) and set(time) <= {"min", "max"}:
        lo, hi = 0, n
        if time.get("min") is not None:
            lo = int(times.searchsorted(pd.Timestamp(time["min"]), side="left"))
        if time.get("max") is not None:
            top = pd.Timestamp(time["max"])
            if isinstance(time["max"], str) and ":" not in time["max"]:
                top += pd.Timedelta(days=1) - pd.Timedelta(1, "ns")
            hi = int(times.searchsorted(top, side="right"))
    else:
        raise ValueError(
            'health time= is None (every step), "latest", or a dict with "min" and/or '
            f'"max" dates, e.g. {{"min": "2011-01-01"}}; got {time!r}.'
        )
    if hi <= lo:
        raise ValueError(
            f"health time={time!r} selects none of the {n} steps of the run."
        )
    return range(lo, hi)


def _native_adjacent(
    source: str, name: str, tdim: str, times: pd.DatetimeIndex
) -> np.ndarray:
    """Return, per pair of neighboring ``times``, whether they are adjacent restarts.

    Under ``truncate_time`` the record is thinned, so two neighbors of the cut one may
    be months apart in the run: a step between them is the truncation's, not the
    model's. Adjacency is read off the source's full record, read with its truncation
    lifted for the moment.
    """
    from ocean_skill import sources

    saved = sources._TRUNCATE.pop(name)
    try:
        full = sources.read(source)
    finally:
        sources._TRUNCATE[name] = saved
    where = np.asarray(full.indexes[tdim].get_indexer(times))
    return (where[1:] - where[:-1] == 1) & (where[:-1] >= 0)


# --------------------------------------------------------------------------------------
# One step


def _geometry(run: _Run, step):
    """Return the step's lazy ``(volume, depth)``: cell volumes (m3), depths (m, down).

    Volumes come from the layer thickness between each cell's own interfaces
    (:func:`ocean_skill.vertical._roms_thickness`) times the cell area, zero where the
    cell is land; depth is ``-z_rho``. Both are ``(s_rho, eta_rho, xi_rho)`` dask
    arrays.
    """
    import dask.array as dsa

    from ocean_skill import roms, vertical

    depth = (-step["z_rho"]).transpose(run.s_dim, "eta_rho", "xi_rho")
    geom = step
    if all(n in step.variables for n in ("sigma_w", "Cs_w")):
        geom = roms.add_interface_coord(step, run.meta)
    dz = vertical._roms_thickness(geom, depth, run.s_dim).transpose(
        run.s_dim, "eta_rho", "xi_rho"
    )
    vol = dz.chunk().data * run.area
    volume = dsa.where(run.mask & dsa.isfinite(vol), vol, 0.0)
    return volume, depth.chunk().data


def _surface(da):
    """Return the top level of a 3-D variable (``s_rho`` is bottom-up), or a 2-D one."""
    return da.isel({da.dims[0]: -1}) if da.ndim == 3 else da


def _load_sample(run: _Run, step) -> tuple[np.ndarray, dict]:
    """Read the step's free surface and every scanned variable's surface level.

    The cheap content sample the cache key hashes. Returns ``(zeta, surfaces)``;
    ``zeta`` is zeros when the run has no free surface.
    """
    import dask

    zeta = (
        step[run.zeta_name].data
        if run.zeta_name
        else np.zeros(run.mask.shape, dtype="float32")
    )
    top = {
        name: _surface(step[std].transpose(*run.dims(run.kinds[name]))).data
        for name, std in run.standard.items()
    }
    zeta, top = dask.compute(zeta, top)
    return np.asarray(zeta), {k: np.asarray(v) for k, v in top.items()}


def _step_key(source: str, fingerprint: str, stamp: pd.Timestamp, sample) -> str:
    """Return one step's cache key: version, definition, time and content sample."""
    zeta, top = sample
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(zeta).tobytes())
    for name in sorted(top):
        h.update(name.encode())
        h.update(np.ascontiguousarray(top[name]).tobytes())
    payload = json.dumps(
        {
            "v": HEALTH_VERSION,
            "source": source,
            "definition": fingerprint,
            "time": stamp.isoformat(),
            "sample": h.hexdigest(),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _is_ph(name: str) -> bool:
    """Return whether ``name`` is a pH variable (``pH``, ``MARBL_PH_3D``, ...)."""
    return "_PH_" in name or name.upper().split("_")[0] == "PH"


def _out_of_bounds(name: str, x, h):
    """Return the (lazy) boolean of ``x``'s values outside ``name``'s bounds.

    ``h`` is the local depth, broadcastable against ``x``. NaN is neither in nor out
    here: the caller counts it apart.
    """
    if name in _RANGES:
        low, high = _RANGES[name]
        return (x < low) | (x > high)
    if name in _ABS_LIMITS:
        return abs(x) > _ABS_LIMITS[name]
    if _is_ph(name):
        return (x < _PH_RANGE[0]) | (x > _PH_RANGE[1])
    if name in _LAYER_DEPTHS:
        return (x < 0) | (x > h)
    return x < 0  # any concentration


def _mid_lon(a: float, b: float) -> float:
    """Return the longitude halfway between ``a`` and ``b``, whichever way round."""
    return a + 0.5 * ((b - a + 180.0) % 360.0 - 180.0)


def _locate(run: _Run, zeta: np.ndarray, kind: str, index: tuple) -> tuple:
    """Return ``(lon, lat, depth)`` of the cell at ``index`` of a ``kind`` variable.

    A u or v point sits between two rho points: its longitude, latitude and depth are
    theirs, averaged. Depth is metres below the surface, positive down, 0 for a 2-D one.
    """
    from ocean_skill import roms

    e, x = index[-2:]
    pairs = {"r": [(e, x)], "u": [(e, x), (e, x + 1)], "v": [(e, x), (e + 1, x)]}[
        kind[0]
    ]
    lon = _mid_lon(run.lon[pairs[0]], run.lon[pairs[-1]])
    lat = 0.5 * (run.lat[pairs[0]] + run.lat[pairs[-1]])
    depth = 0.0
    if kind.endswith("3"):
        k = index[0]
        z = [
            float(
                roms._s_to_z(
                    run.sigma_r[k],
                    run.cs_r[k],
                    run.h[p],
                    np.nan_to_num(zeta[p]),
                    run.hc,
                    run.vtransform,
                )
            )
            for p in pairs
        ]
        depth = -float(np.mean(z))
    return float(lon), float(lat), depth


def _scan_step(run: _Run, step, stamp: pd.Timestamp, zeta: np.ndarray) -> pd.DataFrame:
    """Return one step's statistics: a row per scanned variable, then derived rows."""
    import dask
    import dask.array as dsa

    from ocean_skill import roms

    # The step's geometry is computed once, in memory (a few full 3-D float arrays);
    # every variable's reductions below share it.
    volume, depth = dask.compute(*_geometry(run, step))
    deep = depth > _DEEP_M
    nz = run.ds.sizes[run.s_dim]
    arrays: dict[str, Any] = {}
    parts: dict[str, dict[str, Any]] = {}
    for name, std in run.standard.items():
        kind = run.kinds[name]
        x = step[std].transpose(*run.dims(kind)).chunk().data
        ok = run.ocean(kind)
        fin = dsa.isfinite(x) & ok
        xs = dsa.where(fin, x, 0.0).astype("float64")
        low = dsa.where(fin, x, np.inf)
        high = dsa.where(fin, x, -np.inf)
        arrays[name] = (x, fin)
        p = {
            "n_fin": fin.sum(),
            "n_nan": (dsa.isnan(x) & ok).sum(),
            "n_inf": (dsa.isinf(x) & ok).sum(),
            "n_neg": (fin & (x < 0)).sum(),
            "n_bad": (fin & _out_of_bounds(name, x, run.h)).sum(),
            "n_nonzero": (fin & (x != 0)).sum(),
            "total": xs.sum(),
            "sumsq": (xs * xs).sum(),
            "min": low.min(),
            "max": high.max(),
            "imin": low.argmin(),
            "imax": high.argmax(),
        }
        if kind == "rho3":
            p["value"] = (xs * volume).sum()
            if name in _DEEP_MEANS:
                p["deep"] = (xs * volume * deep).sum()
            if name == "O2":
                for limit in _HYPOXIC:
                    p[f"o2_{limit:g}"] = (volume * (fin & (x < limit))).sum()
        parts[name] = p
    extras: dict[str, Any] = {"deep_volume": (volume * deep).sum()}
    if run.kinds.get("u") == "u3" and run.kinds.get("v") == "v3":
        u, v = (
            roms._average_to_rho(_wet(run, step, name, kind), stagger, rho).data
            for name, kind, stagger, rho in (
                ("u", "u3", "xi_u", "xi_rho"),
                ("v", "v3", "eta_v", "eta_rho"),
            )
        )
        speed2 = u * u + v * v
        extras["KE"] = (
            0.5
            * (dsa.where(run.mask & dsa.isfinite(speed2), speed2, 0.0) * volume).sum()
        )
    for name in run.kinds:
        alt = f"{name}_ALT_CO2"
        # a CDR pair is a tracer's added amount; pH has an _ALT_CO2 twin too, but a
        # difference of pH is not an amount, so it is left out
        if (
            run.kinds[name] == "rho3"
            and run.kinds.get(alt) == "rho3"
            and not _is_ph(name)
        ):
            diff = arrays[name][0] - arrays[alt][0]
            fin = arrays[name][1] & arrays[alt][1]
            extras[f"cdr_{name}"] = (
                dsa.where(fin, diff, 0.0).astype("float64") * volume
            ).sum()
    # One compute per variable, not one for the whole step: a step of a large run is
    # tens of GB over ~45 variables, and a single graph of all of them let dask hold
    # much of it at once (out of memory on a 1.8 M-cell, 100-level grid). Each
    # variable is still read once.
    got = {name: dask.compute(p)[0] for name, p in parts.items()}
    more = {name: dask.compute(x)[0] for name, x in extras.items()}

    copies = _find_copies(run, arrays, got)
    rows = []
    for name, r in got.items():
        kind = run.kinds[name]
        n_ocean = int(run.ocean(kind).sum()) * (nz if kind.endswith("3") else 1)
        n_fin = int(r["n_fin"])
        row = dict.fromkeys(_COLUMNS, np.nan)
        row.update(
            time=stamp,
            variable=name,
            kind="3d" if kind.endswith("3") else "2d",
            n_ocean=n_ocean,
            n_nan=int(r["n_nan"]),
            n_inf=int(r["n_inf"]),
            n_neg=int(r["n_neg"]),
            n_bad=int(r["n_bad"]),
            pct_neg=100.0 * int(r["n_neg"]) / max(n_ocean, 1),
            pct_bad=100.0 * int(r["n_bad"]) / max(n_ocean, 1),
            all_zero=bool(n_fin and not int(r["n_nonzero"])),
            copy_of=copies.get(name, ""),
            total=float(r["total"]),
            sumsq=float(r["sumsq"]),
        )
        if n_fin:
            shape = arrays[name][0].shape
            for which, flat in (("min", r["imin"]), ("max", r["imax"])):
                index = np.unravel_index(int(flat), shape)
                lon, lat, z = _locate(run, zeta, kind, index)
                row.update(
                    {
                        which: float(r[which]),
                        f"{which}_lon": lon,
                        f"{which}_lat": lat,
                        f"{which}_depth": z,
                    }
                )
        if "value" in r:
            row["value"] = float(r["value"])
        rows.append(row)
    rows += _derived_rows(stamp, got, more)
    return pd.DataFrame(rows, columns=list(_COLUMNS))


def _wet(run: _Run, step, name: str, kind: str):
    """Return the staggered velocity ``name`` with its land points set to zero.

    A wall carries no flow; zeroing it before the average to rho keeps whatever a land
    point holds from reaching the ocean cell beside it.
    """
    import dask.array as dsa

    da = step[run.standard[name]].transpose(*run.dims(kind)).chunk()
    wet = run.ocean(kind) & dsa.isfinite(da.data)
    return da.copy(data=dsa.where(wet, da.data, 0.0))


def _find_copies(run: _Run, arrays: dict, got: dict) -> dict[str, str]:
    """Return ``{variable: the earlier-named variable it equals at every ocean cell}``.

    Cheap first: only variables of one layout whose finite count, minimum, maximum and
    sum all agree are candidates, and only candidates are read again to compare cell by
    cell. A field that is zero everywhere is not a copy of another zero field.
    """
    import dask

    groups: dict[tuple, list[str]] = {}
    for name, r in got.items():
        if not int(r["n_nonzero"]):
            continue
        key = (
            run.kinds[name],
            int(r["n_fin"]),
            float(r["min"]),
            float(r["max"]),
            float(r["total"]),
        )
        groups.setdefault(key, []).append(name)
    pairs = [
        (a, b)
        for names in groups.values()
        for i, a in enumerate(names)
        for b in names[i + 1 :]
    ]
    if not pairs:
        return {}
    same = {}
    for a, b in pairs:
        (xa, fa), (xb, fb) = arrays[a], arrays[b]
        same[(a, b)] = ((xa == xb) | ~(fa | fb)).all()
    equal = dask.compute(same)[0]
    copies: dict[str, str] = {}
    for (a, b), is_same in equal.items():
        if is_same and b not in copies:
            copies[b] = a
    return copies


def _derived_rows(stamp: pd.Timestamp, got: dict, more: dict) -> list[dict]:
    """Return a step's derived series (``kind="derived"``, the number in ``value``)."""
    values: dict[str, float] = {}
    if "KE" in more:
        values["KE"] = float(more["KE"])
    if "O2" in got and "o2_60" in got["O2"]:
        for limit in _HYPOXIC:
            values[f"O2_lt{limit:g}"] = float(got["O2"][f"o2_{limit:g}"])
    deep_volume = float(more["deep_volume"])
    for name in _DEEP_MEANS:
        if name in got and "deep" in got[name] and deep_volume > 0:
            values[f"deep_{name}"] = float(got[name]["deep"]) / deep_volume
    values.update({k: float(v) for k, v in more.items() if k.startswith("cdr_")})
    rows = []
    for name, value in values.items():
        row = dict.fromkeys(_COLUMNS, np.nan)
        row.update(
            time=stamp,
            variable=name,
            kind="derived",
            all_zero=False,
            copy_of="",
            value=value,
        )
        rows.append(row)
    return rows


def _to_dataset(frame: pd.DataFrame):
    """Return a step table as the Dataset :mod:`ocean_skill.cache` stores."""
    return frame.reset_index(drop=True).to_xarray()


def _from_dataset(ds) -> pd.DataFrame:
    """Return the step table :func:`_to_dataset` made, read back."""
    frame = ds.to_dataframe().reset_index(drop=True)
    frame["copy_of"] = frame["copy_of"].fillna("").astype(str)
    frame["variable"] = frame["variable"].astype(str)
    frame["kind"] = frame["kind"].astype(str)
    return frame[list(_COLUMNS)]


# --------------------------------------------------------------------------------------
# The public entry point


def health(
    source: str,
    *,
    time=None,
    joins=None,
    cache: bool = True,
) -> RunHealth:
    """Check a ROMS run's health: bounds, copies, inventories, drift and sudden steps.

    One streaming pass over the catalog entry ``source`` (read with
    :func:`ocean_skill.read`, so a smoke run's truncation applies), one time step at a
    time; see the module docstring for what is recorded and how. Returns a
    :class:`RunHealth` to read as tables, problems or report figures.

    Parameters
    ----------
    source
        Catalog nickname of a ROMS run, typically its restart files.
    time
        Which steps to scan: ``None`` (all), ``"latest"``, or a ``{"min": "2011-01-01",
        "max": "2011-06-30"}`` dict, either key optional.
    joins
        Start dates of the run's job segments, if known. Only used to shade the segments
        on the inventory figures and to say a step falls "at a segment join".
    cache
        Keep each step's table in the ``health`` cache and reuse it (default). ``False``
        recomputes every step and stores nothing; so does a disabled cache.
    """
    from ocean_skill import cache as _cache
    from ocean_skill import catalog, sources

    ref = sources.resolve(source)
    ds = sources.read(source)
    run = _open_run(ds, ref.metadata)
    times = run.ds.indexes[run.tdim]
    chosen = _positions(times, time)
    use_cache = bool(cache) and _cache.enabled()
    fingerprint = catalog.fingerprint(source) if use_cache else ""
    frames = []
    for i in chosen:
        step = run.ds.isel({run.tdim: i})
        stamp = pd.Timestamp(times[i])
        sample = key = None
        if use_cache:
            sample = _load_sample(run, step)
            key = _step_key(source, fingerprint, stamp, sample)
            hit = _cache.load(key, kind="health")
            if hit is not None:
                frames.append(_from_dataset(hit))
                continue
        zeta = sample[0] if sample else _load_sample(run, step)[0]
        frame = _scan_step(run, step, stamp, zeta)
        if key is not None:
            _cache.save(key, _to_dataset(frame), kind="health")
        frames.append(frame)
    steps = pd.concat(frames, ignore_index=True)
    steps["time"] = pd.to_datetime(steps["time"]).astype("datetime64[ns]")
    test_mode = ref.name in sources._TRUNCATE
    adjacent = None
    if test_mode:
        stamps = pd.DatetimeIndex(sorted(steps["time"].unique()))
        adjacent = _native_adjacent(source, ref.name, run.tdim, stamps)
    return RunHealth(
        source,
        steps,
        joins=joins,
        test_mode=test_mode,
        adjacent=adjacent,
    )


# --------------------------------------------------------------------------------------
# The result


def _day(t) -> str:
    """Return a time as ``YYYY-MM-DD``."""
    return pd.Timestamp(t).strftime("%Y-%m-%d")


def _lonlat(lon: float, lat: float) -> str:
    """``147.0W 12.5N``: longitude wrapped to -180..180, hemispheres for signs."""
    lon = (lon + 180.0) % 360.0 - 180.0  # 290E reads as 70W
    east_west, north_south = "E" if lon >= 0 else "W", "N" if lat >= 0 else "S"
    return f"{abs(lon):.1f}{east_west} {abs(lat):.1f}{north_south}"


def _pct(x: float) -> str:
    """Return a percentage to the digits that matter: ``0.31``, ``12``, ``0.0042``."""
    return f"{x:.2g}" if abs(x) < 10 else f"{x:.0f}"


def _name(variable: str) -> str:
    """Return the readable name of a scanned or derived variable."""
    if variable in _NAMES:
        return _NAMES[variable]
    return variable


class RunHealth:
    """The result of :func:`health`: per-step statistics, and what they say.

    Attributes
    ----------
    source
        The catalog nickname that was scanned.
    steps
        :class:`pandas.DataFrame`, one row per (time, variable): the counts of NaN/Inf,
        negative and out-of-bounds ocean cells and their percentages, the minimum and
        maximum with their longitude, latitude and depth, the all-zero flag, the
        variable it is an exact copy of (``copy_of``), the sum and sum of squares (to
        see a repeat of the previous restart), and ``value``: the volume integral of a
        3-D rho variable, or the number of a derived row (``kind == "derived"``: ``KE``,
        ``O2_lt60``, ``O2_lt20``, ``deep_<variable>``, ``cdr_<variable>``).
    joins
        The job-segment start dates given to :func:`health`, as timestamps.
    test_mode
        Whether the source was read under ``truncate_time``; every figure then says so.
    """

    def __init__(
        self,
        source: str,
        steps: pd.DataFrame,
        *,
        joins=None,
        test_mode: bool = False,
        adjacent=None,
    ) -> None:
        self.source = source
        self.steps = steps
        self.joins = sorted(pd.Timestamp(j) for j in (joins or ()))
        self.test_mode = test_mode
        self._adjacent = adjacent  # per neighboring pair of steps; None = all adjacent
        self._problems: list[dict] | None = None

    def __repr__(self) -> str:
        t = self._times()
        span = f"{_day(t[0])} to {_day(t[-1])}" if len(t) else "no steps"
        return f"RunHealth({self.source!r}, {len(t)} steps, {span})"

    # ---- series ------------------------------------------------------------------

    def _times(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(sorted(self.steps["time"].unique()))

    def _series(self, variable: str) -> pd.Series:
        """Return the ``value`` of ``variable`` through time (NaN where it has none)."""
        rows = self.steps[self.steps["variable"] == variable]
        return rows.set_index("time")["value"].sort_index()

    def _inventories(self) -> list[str]:
        """Return the 3-D variables that have an inventory, in scan order."""
        rows = self.steps[(self.steps["kind"] == "3d") & self.steps["value"].notna()]
        # A total of temperature, salinity or pH has no meaning as an amount (it
        # depends on where zero is); their deep means carry the drift instead.
        return [
            v
            for v in dict.fromkeys(rows["variable"])
            if v not in ("temp", "salt") and not _is_ph(v)
        ]

    def _derived(self) -> list[str]:
        rows = self.steps[self.steps["kind"] == "derived"]
        return list(dict.fromkeys(rows["variable"]))

    def _pair_adjacent(self, times: pd.DatetimeIndex) -> np.ndarray:
        """Per neighboring pair of ``times``: are they adjacent restarts of the run."""
        if self._adjacent is None or len(self._adjacent) != len(times) - 1:
            return np.ones(max(len(times) - 1, 0), dtype=bool)
        return np.asarray(self._adjacent, dtype=bool)

    # ---- tables ------------------------------------------------------------------

    def table(self) -> pd.DataFrame:
        """Return the integrity table: a row per variable, its worst case over steps.

        Columns: ``variable``, the worst percentage of ocean cells negative
        (``worst_pct_neg``, ``worst_neg_time``) and out of bounds (``worst_pct_bad``,
        ``worst_bad_time``), the same at the latest step, the run-wide ``min`` and
        ``max`` with ``_lon``, ``_lat``, ``_depth`` and ``_time``, ``n_nonfinite`` (NaN
        plus Inf ocean cells at the worst step) and ``flags``.
        """
        rows = []
        latest = self.steps["time"].max()
        for variable, g in self.steps[self.steps["kind"] != "derived"].groupby(
            "variable", sort=False
        ):
            g = g.sort_values("time")
            last = g[g["time"] == latest]
            last = last.iloc[0] if len(last) else g.iloc[-1]
            neg, bad = g.loc[g["pct_neg"].idxmax()], g.loc[g["pct_bad"].idxmax()]
            row: dict[str, Any] = {
                "variable": variable,
                "worst_pct_neg": float(neg["pct_neg"]),
                "worst_neg_time": neg["time"],
                "worst_pct_bad": float(bad["pct_bad"]),
                "worst_bad_time": bad["time"],
                "latest_pct_neg": float(last["pct_neg"]),
                "latest_pct_bad": float(last["pct_bad"]),
            }
            for which in ("min", "max"):
                has = g[which].notna()
                if has.any():
                    pick = g.loc[
                        g[which][has].idxmin()
                        if which == "min"
                        else g[which][has].idxmax()
                    ]
                    row.update(
                        {
                            which: float(pick[which]),
                            f"{which}_lon": float(pick[f"{which}_lon"]),
                            f"{which}_lat": float(pick[f"{which}_lat"]),
                            f"{which}_depth": float(pick[f"{which}_depth"]),
                            f"{which}_time": pick["time"],
                        }
                    )
                else:
                    row.update(
                        {
                            f"{which}{k}": np.nan
                            for k in ("", "_lon", "_lat", "_depth", "_time")
                        }
                    )
            row["n_nonfinite"] = int((g["n_nan"] + g["n_inf"]).max())
            row["flags"] = self._flags(variable, g)
            rows.append(row)
        return pd.DataFrame(rows)

    def _repeats(self, variable: str) -> pd.Series:
        """Per step of ``variable`` (indexed by time): identical to the previous step.

        True where the count, minimum, maximum, sum and sum of squares of a non-zero
        field all equal the previous step's exactly -- the same restart written twice, a
        field the model no longer updates.
        """
        g = self.steps[self.steps["variable"] == variable].sort_values("time")
        g = g.set_index("time")
        keys = g[["min", "max", "total", "sumsq", "n_ocean"]]
        same = (keys == keys.shift()).all(axis=1) & ~g["all_zero"] & g["min"].notna()
        same.iloc[0:1] = False
        return same

    def _flags(self, variable: str, g: pd.DataFrame) -> str:
        n = len(g)
        flags = []
        zero = int(g["all_zero"].sum())
        if zero:
            flags.append("all zero" if zero == n else f"all zero in {zero}/{n}")
        copy = g[g["copy_of"] != ""]
        if len(copy):
            of = copy["copy_of"].iloc[0]
            flags.append(
                f"copy of {of}"
                if len(copy) == n
                else f"copy of {of} in {len(copy)}/{n}"
            )
        repeat = int(self._repeats(variable).sum())
        if repeat:
            flags.append(f"same as previous restart x{repeat}")
        return "; ".join(flags)

    def drift(self) -> pd.DataFrame:
        """Percent change per year of each inventory and deep mean, by a linear fit.

        Columns ``series`` (``"inventory"`` or ``"deep mean"``), ``variable``, ``label``
        and ``pct_per_year``: the slope of the fit through the steps over the absolute
        value of its mean, times 100. Empty when there are fewer than three steps or
        they span under a day.
        """
        rows = []
        names = [("inventory", v, v) for v in self._inventories()]
        names += [
            ("deep mean", f"deep_{v}", v)
            for v in _DEEP_MEANS
            if f"deep_{v}" in self._derived()
        ]
        for kind, variable, short in names:
            s = self._series(variable).dropna()
            if len(s) < 3:
                continue
            years = (s.index - s.index[0]) / pd.Timedelta(days=365.25)
            mean = float(np.mean(np.abs(s.values)))
            if years[-1] * 365.25 < 1 or mean == 0:
                continue
            slope = float(np.polyfit(np.asarray(years, dtype=float), s.values, 1)[0])
            label = f"{_name(short)} {kind}"
            rows.append(
                {
                    "series": kind,
                    "variable": variable,
                    "label": label,
                    "pct_per_year": 100.0 * slope / mean,
                }
            )
        out = pd.DataFrame(
            rows, columns=["series", "variable", "label", "pct_per_year"]
        )
        return out.reindex(out["pct_per_year"].abs().sort_values(ascending=False).index)

    def _jumps(self) -> pd.DataFrame:
        """Sudden steps of the inventory series, between adjacent steps.

        A step is flagged where the rate of change between two neighboring restarts
        exceeds :data:`_JUMP_FACTOR` times the series' median rate and the change is
        more than :data:`_JUMP_MIN_FRACTION` of its value. Columns ``time`` (the later
        step), ``previous``, ``variable``, ``change_pct``, ``ratio`` (inf if the series
        is otherwise flat) and ``at_join``.
        """
        times = self._times()
        adjacent = self._pair_adjacent(times)
        days = np.asarray((times[1:] - times[:-1]) / pd.Timedelta(days=1), dtype=float)
        usable = adjacent & (days > 0)
        rows = []
        if usable.sum() < _JUMP_MIN_PAIRS:
            return pd.DataFrame(
                columns=[
                    "time",
                    "previous",
                    "variable",
                    "change_pct",
                    "ratio",
                    "at_join",
                ]
            )
        names = [
            *self._inventories(),
            *(v for v in self._derived() if v in ("KE", "O2_lt60", "O2_lt20")),
        ]
        for variable in names:
            v = self._series(variable).reindex(times).to_numpy(dtype=float)
            delta = v[1:] - v[:-1]
            with np.errstate(divide="ignore", invalid="ignore"):
                rate = np.abs(delta) / days
            ok = usable & np.isfinite(rate)
            if ok.sum() < _JUMP_MIN_PAIRS:
                continue
            median = float(np.median(rate[ok]))
            for j in np.flatnonzero(ok):
                big = abs(delta[j]) > _JUMP_MIN_FRACTION * abs(v[j])
                if big and rate[j] > _JUMP_FACTOR * median:
                    rows.append(
                        {
                            "time": times[j + 1],
                            "previous": times[j],
                            "variable": variable,
                            "change_pct": 100.0 * delta[j] / abs(v[j])
                            if v[j]
                            else np.inf,
                            "ratio": rate[j] / median if median > 0 else np.inf,
                            "at_join": any(
                                times[j] < t <= times[j + 1] for t in self.joins
                            ),
                        }
                    )
        return pd.DataFrame(
            rows,
            columns=["time", "previous", "variable", "change_pct", "ratio", "at_join"],
        )

    # ---- problems ----------------------------------------------------------------

    def problems(self) -> list[dict]:
        """Return what is wrong, most severe first, as JSON-safe dicts.

        Each has ``kind`` (``nonfinite``, ``negative``, ``out_of_bounds``, ``copy``,
        ``all_zero``, ``step`` or ``repeat``), ``variable``, a plain-words ``message``
        and the numbers behind it. Variables negative or out of bounds in under 0.1% of
        their ocean cells at every step are in :meth:`table` only.
        """
        if self._problems is None:
            found = [
                *self._bound_problems(),
                *self._copy_problems(),
                *self._step_problems(),
                *self._repeat_problems(),
            ]
            found.sort(
                key=lambda p: (_SEVERITY.index(p["kind"]), -p.get("magnitude", 0.0))
            )
            self._problems = found
        return self._problems

    def _bound_problems(self) -> list[dict]:
        out = []
        for row in self.table().itertuples(index=False):
            variable = row.variable
            if row.n_nonfinite:
                out.append(
                    {
                        "kind": "nonfinite",
                        "variable": variable,
                        "message": (
                            f"{variable} has {row.n_nonfinite} NaN or Inf ocean "
                            "cells at its worst restart."
                        ),
                        "count": int(row.n_nonfinite),
                        "magnitude": float(row.n_nonfinite),
                    }
                )
            if variable not in _NOT_CONCENTRATION and row.worst_pct_neg > _PROBLEM_PCT:
                kind, what = "negative", "negative"
                worst, when, now = (
                    row.worst_pct_neg,
                    row.worst_neg_time,
                    row.latest_pct_neg,
                )
                where = (
                    f" (lowest {row.min:.4g} at {_lonlat(row.min_lon, row.min_lat)}, "
                    f"{_meters(row.min_depth)}, {_day(row.min_time)})"
                )
            elif row.worst_pct_bad > _PROBLEM_PCT:
                kind, what = "out_of_bounds", "outside its physical bounds"
                worst, when, now = (
                    row.worst_pct_bad,
                    row.worst_bad_time,
                    row.latest_pct_bad,
                )
                where = f" (range {row.min:.4g} to {row.max:.4g})"
            else:
                continue
            out.append(
                {
                    "kind": kind,
                    "variable": variable,
                    "message": (
                        f"{variable} is {what} in {_pct(worst)}% of ocean cells at "
                        f"worst ({_day(when)}) and {_pct(now)}% at the latest restart"
                        f"{where}."
                    ),
                    "worst_pct": float(worst),
                    "latest_pct": float(now),
                    "time": _day(when),
                    # oxygen below zero is the one that matters most; it sorts first
                    "magnitude": float(worst) + (1e6 if variable == "O2" else 0.0),
                }
            )
        return out

    def _copy_problems(self) -> list[dict]:
        out = []
        steps = self.steps[self.steps["kind"] != "derived"]
        n_steps = steps["time"].nunique()
        for variable, g in steps.groupby("variable", sort=False):
            copy = g[g["copy_of"] != ""]
            zero = g[g["all_zero"]]
            if len(copy):
                of = copy["copy_of"].iloc[0]
                span = (
                    "in every restart"
                    if len(copy) == n_steps
                    else f"in {len(copy)} of {n_steps} restarts"
                )
                out.append(
                    {
                        "kind": "copy",
                        "variable": variable,
                        "message": f"{variable} is identical to {of} {span}.",
                        "copy_of": of,
                        "restarts": len(copy),
                        "magnitude": float(len(copy)),
                    }
                )
            if len(zero):
                span = (
                    "in every restart"
                    if len(zero) == n_steps
                    else f"in {len(zero)} of {n_steps} restarts"
                )
                out.append(
                    {
                        "kind": "all_zero",
                        "variable": variable,
                        "message": f"{variable} is zero everywhere {span}.",
                        "restarts": len(zero),
                        "magnitude": float(len(zero)),
                    }
                )
        return out

    def _step_problems(self) -> list[dict]:
        out = []
        jumps = self._jumps()
        for when, g in jumps.groupby("time", sort=True):
            g = g.sort_values("ratio", ascending=False)
            at_join = bool(g["at_join"].iloc[0])
            for row in g.head(3).itertuples(index=False):
                size = (
                    "while it is otherwise flat"
                    if np.isinf(row.ratio)
                    else f"{row.ratio:.0f} times the typical step"
                )
                out.append(
                    {
                        "kind": "step",
                        "variable": row.variable,
                        "message": (
                            f"{row.variable} changed {row.change_pct:+.2g}% between "
                            f"{_day(row.previous)} and {_day(row.time)}, {size}"
                            f"{JOIN_NOTE if row.at_join else ''}."
                        ),
                        "time": _day(row.time),
                        "change_pct": float(row.change_pct),
                        "ratio": None if np.isinf(row.ratio) else float(row.ratio),
                        "at_join": bool(row.at_join),
                        "magnitude": 1e3 if np.isinf(row.ratio) else float(row.ratio),
                    }
                )
            rest = g.iloc[3:]
            if len(rest):
                out.append(
                    {
                        "kind": "step",
                        "variable": "(several)",
                        "message": (
                            f"{len(rest)} more series changed suddenly at {_day(when)}"
                            f"{JOIN_NOTE if at_join else ''}: "
                            f"{', '.join(rest['variable'].head(6))}"
                            f"{', ...' if len(rest) > 6 else ''}."
                        ),
                        "time": _day(when),
                        "at_join": at_join,
                        "magnitude": 0.0,
                    }
                )
        return out

    def _repeat_problems(self) -> list[dict]:
        out = []
        names = [
            v for v in self.steps[self.steps["kind"] != "derived"]["variable"].unique()
        ]
        stuck = {v: self._repeats(v) for v in names}
        frame = pd.DataFrame(stuck)
        if frame.empty:
            return out
        share = frame.mean(axis=1)  # the fraction of variables repeating, per step
        whole = share[share >= 0.5]
        for when in whole.index:
            out.append(
                {
                    "kind": "repeat",
                    "variable": "(all)",
                    "message": (
                        f"The restart at {_day(when)} is identical to the previous "
                        f"one in {_pct(100 * share[when])}% of variables."
                    ),
                    "time": _day(when),
                    "magnitude": float(share[when]) + 1.0,
                }
            )
        for variable, flags in stuck.items():
            part = flags[~flags.index.isin(whole.index)]
            if part.any():
                first = part[part].index[0]
                out.append(
                    {
                        "kind": "repeat",
                        "variable": variable,
                        "message": (
                            f"{variable} is identical to the previous restart in "
                            f"{part.sum()} restarts (first at {_day(first)})."
                        ),
                        "time": _day(first),
                        "restarts": int(part.sum()),
                        "magnitude": float(part.sum()),
                    }
                )
        return out

    # ---- figures -----------------------------------------------------------------

    def figures(self, draw: str) -> list[tuple[str, Any]]:
        """Draw one report page: ``(suffix, Figure)`` pairs, ``""`` then ``"-2"``, ...

        ``draw`` is one of :data:`HEALTH_DRAWS`: ``"summary"`` (problems and drift as
        text), ``"integrity"`` (a table per variable), ``"inventory"`` (series of every
        tracer inventory, kinetic energy, low-oxygen volumes, deep means and CDR
        differences) or ``"plankton"`` (each plankton group's carbon, chlorophyll,
        phosphorus and iron, and their ratios; empty for a run without them).
        """
        if draw not in HEALTH_DRAWS:
            raise ValueError(
                f"unknown health draw {draw!r}; choose from {list(HEALTH_DRAWS)}."
            )
        pages = getattr(self, f"_draw_{draw}")()
        return [("" if i == 0 else f"-{i + 1}", fig) for i, fig in enumerate(pages)]

    def _title(self, what: str) -> str:
        t = self._times()
        span = f"{len(t)} restart{'s' if len(t) != 1 else ''}"
        if len(t):
            span += (
                f", {_day(t[0])} to {_day(t[-1])}" if len(t) > 1 else f", {_day(t[0])}"
            )
        return f"Run health: {what}\n{self.source} ({span})"

    def _draw_summary(self) -> list:
        from matplotlib.figure import Figure

        from ocean_skill.plot.typography import PAGE_W

        lines: list[tuple[str, str]] = []  # (style, text)
        problems = self.problems()
        if problems:
            lines.append(("h", "Problems"))
            for i, p in enumerate(problems[:_SUMMARY_PROBLEMS], 1):
                for j, text in enumerate(textwrap.wrap(p["message"], 112)):
                    lines.append(("p", (f"{i}. " if j == 0 else "    ") + text))
            if len(problems) > _SUMMARY_PROBLEMS:
                more = len(problems) - _SUMMARY_PROBLEMS
                lines.append(("p", f"... and {more} more (see the integrity table)."))
        drift = self.drift()
        if len(drift):
            lines.append(("h", "Drift, % per year (linear fit over the record)"))
            for row in drift.head(12).itertuples(index=False):
                lines.append(("kv", f"{row.label}|{row.pct_per_year:+.3g} %/yr"))
        cdr = [v for v in self._derived() if v.startswith("cdr_")]
        if cdr:
            lines.append(
                ("h", "CDR tracers: inventory of X minus X_ALT_CO2 (first, latest)")
            )
            for variable in cdr:
                s = self._series(variable).dropna()
                if len(s):
                    lines.append(
                        (
                            "kv",
                            f"{variable[4:]} added|{s.iloc[0]:.4g}, {s.iloc[-1]:.4g}",
                        )
                    )
        pages = []
        for start in range(0, max(len(lines), 1), _SUMMARY_LINES):
            chunk = lines[start : start + _SUMMARY_LINES]
            height = 1.3 + sum(0.3 if s == "h" else 0.17 for s, _ in chunk)
            fig = Figure(figsize=(PAGE_W, height))
            title = self._title("summary") + (" (cont.)" if start else "")
            y = _draw_title(fig, title, self.test_mode)
            for style, text in chunk:
                y -= (0.3 if style == "h" else 0.17) / height
                if style == "h":
                    fig.text(0.06, y, text, fontsize=10, fontweight="bold", va="top")
                elif style == "kv":
                    left, right = text.split("|")
                    fig.text(0.08, y, left, fontsize=8.5, va="top")
                    fig.text(0.52, y, right, fontsize=8.5, va="top", ha="right")
                else:
                    fig.text(0.06, y, text, fontsize=8.5, va="top")
            pages.append(fig)
        return pages

    def _draw_integrity(self) -> list:
        from matplotlib.figure import Figure
        from matplotlib.table import Table

        from ocean_skill.plot.typography import PAGE_W

        table = self.table()
        head = [
            "Variable",
            "Worst %\nnegative",
            "Worst %\nout of bounds",
            "Latest %\nneg / bad",
            "Minimum",
            "Maximum",
            "NaN/Inf",
            "Flags",
        ]
        widths = [0.9, 0.7, 0.8, 0.65, 1.55, 1.55, 0.45, 1.3]
        rows = []
        for r in table.itertuples(index=False):
            rows.append(
                [
                    r.variable,
                    "–"
                    if r.variable in _NOT_CONCENTRATION
                    else f"{_pct(r.worst_pct_neg)}\n{_day(r.worst_neg_time)}"
                    if r.worst_pct_neg
                    else "0",
                    f"{_pct(r.worst_pct_bad)}\n{_day(r.worst_bad_time)}"
                    if r.worst_pct_bad
                    else "0",
                    "– / " + _pct(r.latest_pct_bad)
                    if r.variable in _NOT_CONCENTRATION
                    else f"{_pct(r.latest_pct_neg)} / {_pct(r.latest_pct_bad)}",
                    _extreme(r, "min"),
                    _extreme(r, "max"),
                    str(r.n_nonfinite) if r.n_nonfinite else "0",
                    "\n".join(textwrap.wrap(r.flags, 22)) if r.flags else "",
                ]
            )
        pages = []
        scale = (PAGE_W - 0.6) / sum(widths)
        widths = [w * scale for w in widths]
        size, line_in = 6.5, 6.5 * 1.35 / 72
        for part, start in enumerate(range(0, max(len(rows), 1), _TABLE_ROWS)):
            chunk = rows[start : start + _TABLE_ROWS]
            heights = [2 * line_in + 0.05] + [
                max(c.count("\n") + 1 for c in row) * line_in + 0.05 for row in chunk
            ]
            fig_h = 1.2 + sum(heights)
            fig = Figure(figsize=(PAGE_W, fig_h))
            title = self._title("integrity by variable") + (" (cont.)" if part else "")
            _draw_title(fig, title, self.test_mode)
            ax = fig.add_axes(
                (0.3 / PAGE_W, 0.1 / fig_h, sum(widths) / PAGE_W, sum(heights) / fig_h)
            )
            ax.axis("off")
            tab = Table(ax, loc="upper left")
            tab.auto_set_font_size(False)
            for i, cells in enumerate([head, *chunk]):
                for j, text in enumerate(cells):
                    tab.add_cell(
                        i,
                        j,
                        widths[j] / sum(widths),
                        heights[i] / sum(heights),
                        text=text,
                        loc="center" if i == 0 or j in (1, 2, 3, 6) else "left",
                        facecolor="0.9" if i == 0 else ("0.96" if i % 2 == 0 else "w"),
                        edgecolor="0.75",
                    )
                    cell = tab[i, j]
                    cell.set_linewidth(0.4)
                    cell.get_text().set_fontsize(size)
                    if i == 0:
                        cell.get_text().set_fontweight("bold")
            ax.add_table(tab)
            pages.append(fig)
        return pages

    def _panel(self, variable: str, title: str, units: str) -> dict | None:
        """One series panel of ``variable``, or ``None`` if the run has no such series.

        ``units="%"`` draws the series as percent change from its first step (so many
        inventories share a scale); any other ``units`` draws it as it is, in those
        units. A series that starts at zero has nothing to scale by and is drawn as it
        is.
        """
        s = (
            self._series(variable).dropna()
            if variable in set(self.steps["variable"])
            else None
        )
        if s is None or s.empty:
            return None
        flagged = self._flagged(variable)
        first = float(s.iloc[0])
        if units == "%" and first != 0:
            s = 100.0 * (s / first - 1.0)
            return {
                "title": f"{title} (% change)",
                "series": s,
                "unit": "% change",
                "flagged": flagged,
            }
        return {
            "title": title,
            "series": s,
            "unit": "" if units == "%" else units,
            "flagged": flagged,
        }

    def _flagged(self, variable: str) -> list:
        """Return the steps at which ``variable`` jumps."""
        jumps = self._jumps()
        return list(jumps.loc[jumps["variable"] == variable, "time"])

    def _series_figures(self, panels: list[dict], what: str) -> list:
        """Lay ``panels`` out as grids of small series panels (the series family).

        Each panel is one line (:class:`~ocean_skill.plot.spec.PlotSpec` ``series``
        family, so the static and interactive renderers draw it alike), with the job
        segments shaded (``highlight=``) and each flagged step ringed in red.
        """
        import xarray as xr

        from ocean_skill.plot.registry import render
        from ocean_skill.plot.spec import PlotSpec
        from ocean_skill.plot.typography import PAGE_W

        pages = []
        for part, start in enumerate(range(0, len(panels), _PANELS_PER_PAGE)):
            chunk = panels[start : start + _PANELS_PER_PAGE]
            items = []
            for p in chunk:
                da = xr.DataArray(
                    p["series"].to_numpy(dtype=float),
                    dims="time",
                    coords={"time": p["series"].index.to_numpy()},
                    name="value",
                )
                items.append(
                    {
                        "aligned": xr.Dataset({"value": da}),
                        "metrics": None,
                        "units": None,
                        "label": p["title"],
                        "standard_name": p["title"],
                        "labels": ("run",),
                    }
                )
            ncols = min(3, len(chunk))
            nrows = -(-len(chunk) // ncols)
            options: dict[str, Any] = {
                "titles": [p["title"] for p in chunk],
                "ncols": ncols,
                "legend": False,
                "encode": {"color": None, "marker": None, "linestyle": None},
                "figsize": (PAGE_W, 0.9 + 2.0 * nrows),
                "title": self._title(what) + (" (cont.)" if part else ""),
            }
            if self._segments():
                options["highlight"] = self._segments()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # a segment with no step in a panel
                spec = PlotSpec(family="series", items=items, options=options)
                fig = render(spec, renderer="matplotlib")
            by_title = {ax.get_title(): ax for ax in fig.axes}
            for p in chunk:
                ax = by_title.get(p["title"])
                if ax is None:
                    continue
                ax.set_ylabel(p["unit"], fontsize=7)
                # plain tick values: an offset label ("1e-5+2.1975") collides with
                # the panel title above it
                ax.ticklabel_format(axis="y", useOffset=False, style="plain")
                marks = [t for t in p["flagged"] if t in p["series"].index]
                if marks:
                    ax.plot(
                        marks,
                        p["series"].loc[marks].to_numpy(),
                        "o",
                        mfc="none",
                        mec="tab:red",
                        ms=7,
                        zorder=5,
                    )
            if self.test_mode:
                fig.text(
                    0.5,
                    0.005,
                    TEST_MODE_BANNER,
                    ha="center",
                    va="bottom",
                    color="firebrick",
                    fontsize=8,
                )
            pages.append(fig)
        return pages

    def _segments(self) -> dict:
        """Return the job segments to shade (every other one), as ``highlight=``."""
        times = self._times()
        joins = [j for j in self.joins if times[0] < j <= times[-1]]
        edges = [times[0], *joins, times[-1]]
        return {
            f"segment {i + 1}": {"min": edges[i], "max": edges[i + 1]}
            for i in range(1, len(edges) - 1, 2)
        }

    def _draw_inventory(self) -> list:
        plankton = {f"{g}{p}" for g, _ in _GROUPS for p in _PLANKTON_PARTS} | {_ZOO}
        tracers = sorted(
            (v for v in self._inventories() if v not in plankton),
            key=lambda v: ({"temp": 0, "salt": 1}.get(v, 2), v),
        )
        wanted = [(v, f"{_name(v)} inventory", "%") for v in tracers[:2]]
        wanted += [
            ("KE", "Kinetic energy", "%"),
            ("O2_lt60", "Volume with oxygen below 60", "%"),
            ("O2_lt20", "Volume with oxygen below 20", "%"),
            ("deep_temp", "Deep temperature (below 1000 m)", "degC"),
            ("deep_salt", "Deep salinity (below 1000 m)", "psu"),
            ("deep_O2", "Deep oxygen (below 1000 m)", "mmol/m3"),
            ("deep_DIC", "Deep DIC (below 1000 m)", "mmol/m3"),
        ]
        wanted += [(v, f"{_name(v)} inventory", "%") for v in tracers[2:]]
        wanted += [
            (v, f"{v[4:]} minus {v[4:]}_ALT_CO2 inventory", "")
            for v in self._derived()
            if v.startswith("cdr_")
        ]
        panels = [self._panel(*w) for w in wanted]
        return self._series_figures([p for p in panels if p], "inventories")

    def _draw_plankton(self) -> list:
        have = set(self.steps["variable"])
        panels = []
        for group, label in _GROUPS:
            for part in _PLANKTON_PARTS:
                panels.append(
                    self._panel(
                        f"{group}{part}", f"{label.capitalize()} {part} inventory", "%"
                    )
                )
        panels.append(self._panel(_ZOO, "Zooplankton C inventory", "%"))
        panels = [p for p in panels if p]
        for group, label in _GROUPS:
            for den, scale, name, unit in (
                ("P", 1.0, "C:P", "mol/mol"),
                ("Chl", _G_C_PER_MMOL, "C:Chl", "g C / g Chl"),
            ):
                if f"{group}C" in have and f"{group}{den}" in have:
                    ratio = (
                        scale
                        * self._series(f"{group}C")
                        / self._series(f"{group}{den}")
                    )
                    ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
                    if len(ratio):
                        title = f"{label.capitalize()} {name}"
                        panels.append(
                            {
                                "title": title,
                                "series": ratio,
                                "unit": unit,
                                "flagged": [],
                            }
                        )
        return self._series_figures(panels, "plankton") if panels else []


def _extreme(r, which: str) -> str:
    """Return the table cell of a run-wide minimum or maximum: value, date, place."""
    value = getattr(r, which)
    if np.isnan(value):
        return ""
    lon, lat, depth = (getattr(r, f"{which}_{k}") for k in ("lon", "lat", "depth"))
    when = _day(getattr(r, f"{which}_time"))
    return f"{value:.4g}  {when}\n{_lonlat(lon, lat)}, {_meters(depth)}"


def _meters(depth: float) -> str:
    """``"130 m"``; a cell just above z = 0 reads ``"0 m"``, not ``"-0 m"``."""
    return f"{max(float(depth), 0.0):.0f} m"


def _draw_title(fig, title: str, test_mode: bool) -> float:
    """Hang the page title (and the test-mode banner) from the top of ``fig``.

    Returns the figure fraction the next line starts below.
    """
    h = fig.get_size_inches()[1]
    fig.text(
        0.5,
        1 - 0.15 / h,
        title,
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
        linespacing=1.3,
    )
    y = 1 - 0.7 / h
    if test_mode:
        fig.text(
            0.5,
            y,
            TEST_MODE_BANNER,
            ha="center",
            va="top",
            fontsize=8,
            color="firebrick",
        )
        y -= 0.3 / h
    return y - 0.1 / h
