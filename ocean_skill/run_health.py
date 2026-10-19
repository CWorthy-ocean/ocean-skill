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

**Blocks of levels, reduced in numpy.** A step's 3-D field is a gigabyte-class array
and a run has ~45 of them. Read at its storage chunks (a subchunked reference: a level
by an eighth of the rows), a step was ~600,000 dask tasks and as many tiny reads, and
ordering and syncing them, not moving the data, took the time; a large run's step took
minutes and tens of GB. The scan instead reopens the source a block of levels per dask
chunk (:func:`_scan_dataset`, about :data:`_BLOCK_BYTES` each), reads each block with
one small compute and reduces it in numpy, a pool of threads working through the
variables in order (:func:`_scan_step`): memory is the step's cell volumes plus a block
per thread, and every value is read once. Only candidate pairs of copies are read a
second time.

**Per-step cache.** A restart never changes once written, so each step's table is kept
under :mod:`ocean_skill.cache`'s ``health`` kind and a mid-run re-run reads only the
restarts that are new. The key is :data:`HEALTH_VERSION`, the source's catalog
fingerprint, the step's time stamp and a hash of its free surface and the surface level
of every scanned variable -- a cheap content sample that misses when a restart is
rewritten in place, which an identity key alone would not notice.

**Sudden steps.** A step is looked for in the inventories, kinetic energy and the
low-oxygen volumes, between adjacent restarts: a change at least 20 times the series'
median rate and over 0.1% of its value. The *living pools* -- the carbon, chlorophyll,
phosphorus, iron, silicon and calcium carbonate of every phytoplankton group (``sp*``,
``diat*``, ``diaz*``) and ``zooC`` -- are left out: a cold start's first days are
spin-up biology (a small phytoplankton pool losing 15% a day), not a restart or
forcing discontinuity. Their series are still drawn, just without step marks.
Nutrients, carbon, oxygen, dissolved organic matter, iron, ligand, physics and kinetic
energy keep the rule.

**Test mode.** A source read under :func:`ocean_skill.sources.truncate_time`
(``--smoke``) has a strided grid and thinned restarts, so its numbers are not physical.
Every figure says so, and a step is only looked for between restarts that are adjacent
in the full record, never across the gap the truncation leaves.

The module is called ``run_health`` rather than ``health`` so the package can expose
``osk.health`` as the function: a submodule of the same name would be clobbered as a
package attribute (see ``_CallablePlotPackage`` in :mod:`ocean_skill`).
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import os
import re
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ocean_skill.config import HEALTH_DRAWS
from ocean_skill.plot._numbers import format_value
from ocean_skill.units import display

__all__ = ["HEALTH_DRAWS", "HEALTH_VERSION", "RunHealth", "health"]

#: Bumped when the statistics a step records, or how they are computed, change, so a
#: cached step from before is not served as one from after. Part of every step's key.
#: 3: sums taken a block of levels at a time, in numpy (the last bits of a total move).
HEALTH_VERSION = 3

#: About how many bytes of one variable a scan block holds: the unit a health step
#: reads and reduces at a time, one per thread (:func:`_workers`).
_BLOCK_BYTES = 64 * 2**20

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
    "Lig": "Ligand",
    "ALK": "Alkalinity",
    "DIC": "DIC",
    "DOCr": "Refractory DOC",
    "DONr": "Refractory DON",
    "DOPr": "Refractory DOP",
    "zooC": "Zooplankton C",
}
#: How a series that is not a scanned variable reads in a list of steps.
_DERIVED_NAMES = {"KE": "KE", "O2_lt60": "O2 < 60 volume", "O2_lt20": "O2 < 20 volume"}
#: The suffix MARBL puts on the twin tracers of a CDR run.
_ALT = "_ALT_CO2"
#: The living plankton pools: the quantities of a group (``spC``, ``diatChl``,
#: ``diazFe``, ...) and the zooplankton carbon. Not looked at for sudden steps.
_LIVING = re.compile(r"(sp|diat|diaz)(C|Chl|P|Fe|Si|CaCO3)|zooC")

#: Under a series figure whose job segments are shaded.
SEGMENT_NOTE = "Shaded bands alternate at each job of the run."

#: Said of a step whose two restarts a job starts between.
JOIN_NOTE = " (at a job join)"

#: Severity order of :meth:`RunHealth.problems` kinds, most severe first.
_SEVERITY = (
    "nonfinite",
    "all_nan",
    "negative",
    "out_of_bounds",
    "copy",
    "all_zero",
    "step",
    "repeat",
)

#: Panels a series figure holds before it continues on the next (five rows of three),
#: and the height of one row (inches): together they keep the figure within
#: :data:`_FIG_MAX_H`.
_PANELS_PER_PAGE = 15
_SERIES_ROW_IN = 1.6
#: One font size for every panel title of the series figures (pt).
_PANEL_TITLE_PT = 8.0
#: A Letter page, in inches; the summary has to fit one.
_LETTER = (8.5, 11.0)
#: The tallest figure a report page holds at its natural size (inches): Letter less
#: the report's top and bottom margins and a two-line caption. Taller figures make
#: the report grow the page past Letter.
_FIG_MAX_H = 9.5
#: Font sizes (pt) of the summary table (never below 8) and the integrity table.
_SUMMARY_PT = 8.0
_INTEGRITY_PT = 6.5
#: A CDR difference this small next to its tracer's own inventory is zero.
_CDR_ZERO = 1e-9


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
    sigma_w: np.ndarray | None  # the interfaces, when the source has them
    cs_w: np.ndarray | None
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
    interfaces = all(n in ds.variables for n in ("sigma_w", "Cs_w"))
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
        sigma_w=np.asarray(ds["sigma_w"].values, dtype=float) if interfaces else None,
        cs_w=np.asarray(ds["Cs_w"].values, dtype=float) if interfaces else None,
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


def _block_bytes(run: _Run) -> int:
    """Return the bytes one scan block holds at most.

    :data:`_BLOCK_BYTES`, or less where a block per thread (:func:`_workers`) would
    add up to more than the largest 3-D variable of a step: what the threads hold at
    once is then about one variable, whatever the number of threads.
    """
    largest = max(
        (
            run.ds[std].nbytes // run.ds.sizes[run.tdim]
            for name, std in run.standard.items()
            if run.kinds[name].endswith("3")
        ),
        default=_BLOCK_BYTES,
    )
    return max(1, min(_BLOCK_BYTES, largest // _workers()))


def _block_levels(run: _Run) -> int | None:
    """Return the levels of one scan block, or ``None`` for a source not read by dask.

    As many levels as fit in :func:`_block_bytes` of the largest 3-D variable, rounded
    down to whole storage chunks along the vertical (a compressed chunk split in two is
    read twice) and never fewer than one of them.
    """
    levels, stored = [], []
    for name, std in run.standard.items():
        if not run.kinds[name].endswith("3"):
            continue
        da = run.ds[std]
        if da.chunks is None:
            return None
        nz = da.sizes[run.s_dim]
        levels.append(da.dtype.itemsize * da.size // (da.sizes[run.tdim] * nz))
        stored.append(max(da.chunks[da.dims.index(run.s_dim)]))
    if not levels:
        return None
    base = math.lcm(*stored)
    return min(max(base, _block_bytes(run) // max(levels) // base * base), nz)


def _scan_dataset(source: str, run: _Run):
    """Return ``source`` reopened for the scan: a block of levels per dask chunk.

    The storage chunks a subchunked reference is opened at (a level by an eighth of
    the rows, say) make one task, and one small read, per sliver of every variable:
    ~600,000 tasks a step, most of the time spent ordering the graph and syncing the
    reads. A chunk of :func:`_block_levels` levels and every row and column is one
    read of all the storage chunks under it. The time axis keeps its stored chunks.
    ``run.ds`` itself is returned for a source without dask chunks, one already
    chunked like that, or one whose reader takes no ``chunks`` (with a warning).
    """
    from ocean_skill import sources

    k = _block_levels(run)
    if k is None:
        return run.ds
    chunks = {d: -1 for d in ("eta_rho", "xi_rho", "eta_v", "xi_u") if d in run.ds.dims}
    chunks[run.s_dim] = k
    blocked = all(
        len(c) == 1 or (dim == run.s_dim and max(c) == k)
        for std in run.standard.values()
        if run.ds[std].chunks is not None
        for dim, c in zip(run.ds[std].dims, run.ds[std].chunks, strict=True)
        if dim != run.tdim
    )
    if blocked:
        return run.ds
    try:
        scan = sources.read(source, chunks=chunks)
    except (TypeError, ValueError) as err:
        warnings.warn(
            f"health could not reopen {source!r} in blocks of {k} levels ({err}); "
            "it is read at its stored chunks instead, which is much slower.",
            stacklevel=3,
        )
        return run.ds
    if scan.sizes.get(run.tdim) != run.ds.sizes[run.tdim]:
        raise RuntimeError(  # the same source, opened twice: cannot happen
            f"health: {source!r} reopened with {scan.sizes.get(run.tdim)} steps, not "
            f"{run.ds.sizes[run.tdim]}."
        )
    return scan


def _workers() -> int:
    """Return how many threads scan a step's blocks: dask's ``num_workers``.

    One under a synchronous dask scheduler, else ``num_workers`` from dask's config,
    else every core. Each holds a block (:func:`_block_bytes`) and its temporaries.
    """
    import dask

    if dask.config.get("scheduler", None) in ("sync", "synchronous", "single-threaded"):
        return 1
    return int(dask.config.get("num_workers", None) or os.cpu_count() or 1)


def _geometry(run: _Run, zeta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the step's cell volumes (m3) and the same, zero above :data:`_DEEP_M`.

    Volumes come from the layer thickness between each cell's own interfaces (the
    transform of :func:`ocean_skill.roms._s_to_z` on ``sigma_w``/``Cs_w``, with the
    step's free surface) times the cell area, zero where the cell is land; a source
    without interfaces estimates the thickness from the level centres
    (:func:`ocean_skill.vertical._roms_thickness`, which warns). Depth is that of the
    cell centre, below the moving surface. Both are ``(s_rho, eta_rho, xi_rho)``
    numpy arrays.
    """
    import xarray as xr

    from ocean_skill import roms, vertical

    def z(sigma, cs):
        return roms._s_to_z(sigma, cs, run.h, zeta, run.hc, run.vtransform)

    nz = len(run.sigma_r)
    if run.sigma_w is None or run.cs_w is None:
        depth = -z(run.sigma_r[:, None, None], run.cs_r[:, None, None])
        frame = xr.DataArray(depth, dims=run.dims("rho3"))
        volume = np.array(vertical._roms_thickness(xr.Dataset(), frame, run.s_dim))
    else:
        # a level at a time: a large grid's full 3-D temporary is over a gigabyte
        volume = np.empty((nz, *run.mask.shape))
        below = z(run.sigma_w[0], run.cs_w[0])
        for k in range(nz):
            above = z(run.sigma_w[k + 1], run.cs_w[k + 1])
            np.subtract(above, below, out=volume[k])
            below = above
    deep = np.empty_like(volume)
    for k in range(nz):
        level = volume[k]
        level *= run.area
        level[~(run.mask & np.isfinite(level))] = 0.0
        is_deep = -z(run.sigma_r[k], run.cs_r[k]) > _DEEP_M
        np.copyto(deep[k], np.where(is_deep, level, 0.0))
    return volume, deep


def _surface(da):
    """Return the top level of a 3-D variable (``s_rho`` is bottom-up), or a 2-D one."""
    return da.isel({da.dims[0]: -1}) if da.ndim == 3 else da


def _zeta(run: _Run, step) -> np.ndarray:
    """Return the step's free surface, zeros when the run has none."""
    if not run.zeta_name:
        return np.zeros(run.mask.shape, dtype="float32")
    return np.asarray(step[run.zeta_name].values)


def _load_sample(run: _Run, step) -> tuple[np.ndarray, dict]:
    """Read the step's free surface and every scanned variable's surface level.

    The cheap content sample the cache key hashes, read from the source at its stored
    chunks (a block of the scan would read many levels for the one). Returns
    ``(zeta, surfaces)``; ``zeta`` is zeros when the run has no free surface.
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


def _units(run: _Run) -> list[tuple[str, ...]]:
    """Return the scanned variables in groups whose blocks are read together.

    ``u`` with ``v`` (kinetic energy needs both), a tracer with its ``_ALT_CO2`` twin
    (the CDR difference; pH has a twin too, but a difference of pH is not an amount),
    every other variable alone.
    """
    units: list[tuple[str, ...]] = []
    taken: set[str] = set()
    ke = run.kinds.get("u") == "u3" and run.kinds.get("v") == "v3"
    for name in run.standard:
        if name in taken:
            continue
        alt = f"{name}_ALT_CO2"
        if ke and name in ("u", "v"):
            unit: tuple[str, ...] = ("u", "v")
        elif (
            run.kinds[name] == "rho3"
            and run.kinds.get(alt) == "rho3"
            and alt not in taken
            and not _is_ph(name)
        ):
            unit = (name, alt)
        else:
            unit = (name,)
        units.append(unit)
        taken.update(unit)
    return units


def _level_blocks(run: _Run, step, name: str) -> list[slice | None]:
    """Return the blocks of levels ``name`` is read in; ``[None]`` for a 2-D variable.

    The dask chunks along the vertical (a block of :func:`_scan_dataset`), merged up to
    :func:`_block_bytes` where they are smaller (a source read at its stored chunks).
    """
    if not run.kinds[name].endswith("3"):
        return [None]
    da = step[run.standard[name]]
    nz = da.sizes[run.s_dim]
    level = da.dtype.itemsize * da.size // nz
    most = max(1, _block_bytes(run) // max(level, 1))
    chunks = da.chunks[da.dims.index(run.s_dim)] if da.chunks else (1,) * nz
    blocks, start, end = [], 0, 0
    for c in chunks:
        if end > start and end + c - start > most:
            blocks.append(slice(start, end))
            start = end
        end += c
    blocks.append(slice(start, end))
    return blocks


def _read(run: _Run, step, name: str, levels: slice | None) -> np.ndarray:
    """Read ``levels`` of the variable ``name`` at one step, as a numpy array.

    One small compute on the calling thread: a block of the scan is one dask chunk.
    """
    da = step[run.standard[name]].transpose(*run.dims(run.kinds[name]))
    if levels is not None:
        da = da.isel({run.s_dim: levels})
    data = da.data
    if hasattr(data, "compute"):
        data = data.compute(scheduler="sync")
    return np.asarray(data)


def _block_stats(
    name: str,
    x: np.ndarray,
    ok: np.ndarray,
    h: np.ndarray,
    offset: int,
    geometry: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[dict[str, Any], np.ndarray]:
    """Return one block's statistics of variable ``name``, and its finite-ocean mask.

    ``x`` is the block (levels ``offset`` on, for a 3-D one), ``ok`` the ocean mask of
    its points, ``h`` the local depth; ``geometry`` is the block's ``(volume, deep
    volume)`` for a 3-D rho variable, whose volume integrals are then kept too.
    Minimum and maximum carry their index in the whole variable.
    """
    fin = np.isfinite(x)
    fin &= ok
    n_ok = int(ok.sum()) * (x.size // ok.size)
    n_fin = int(np.count_nonzero(fin))
    n_nan = int(np.count_nonzero(np.isnan(x) & ok)) if n_fin < n_ok else 0
    xs = np.where(fin, x, 0.0).astype(np.float64, copy=False).reshape(-1)
    p: dict[str, Any] = {
        "n_fin": n_fin,
        "n_nan": n_nan,
        "n_inf": n_ok - n_fin - n_nan,
        "n_neg": int(np.count_nonzero(xs < 0)),
        "n_bad": int(np.count_nonzero(fin & _out_of_bounds(name, x, h))),
        "n_nonzero": int(np.count_nonzero(xs)),
        "total": float(xs.sum()),
        "sumsq": float(xs @ xs),
    }
    for which, fill, pick in (("min", np.inf, np.argmin), ("max", -np.inf, np.argmax)):
        masked = np.where(fin, x, fill)
        flat = int(pick(masked))
        index = np.unravel_index(flat, x.shape)
        p[which] = float(masked.flat[flat])
        p[f"i{which}"] = (int(index[0]) + offset, *map(int, index[1:]))
        del masked
    if geometry is not None:
        volume, deep = (g.reshape(-1) for g in geometry)
        p["value"] = float(xs @ volume)
        if name in _DEEP_MEANS:
            p["deep"] = float(xs @ deep)
        if name == "O2":
            for limit in _HYPOXIC:
                low = (fin & (x < limit)).reshape(-1)
                p[f"o2_{limit:g}"] = float(volume[low].sum())
    return p, fin


def _to_rho(x: np.ndarray, axis: int) -> np.ndarray:
    """Average a staggered velocity onto rho points along ``axis``.

    The numpy twin of :func:`ocean_skill.roms._average_to_rho`: an interior point is
    the mean of the two values either side, an edge point takes the nearest one.
    """
    x = np.moveaxis(x, axis, -1)
    out = np.empty((*x.shape[:-1], x.shape[-1] + 1), dtype=x.dtype)
    out[..., 1:-1] = 0.5 * (x[..., :-1] + x[..., 1:])
    out[..., 0], out[..., -1] = x[..., 0], x[..., -1]
    return np.moveaxis(out, -1, axis)


def _scan_block(
    run: _Run, step, unit: tuple[str, ...], levels: slice | None, geometry: tuple
) -> tuple[dict[str, dict], dict[str, float]]:
    """Read and reduce one block of levels of the variables of ``unit``.

    Returns ``(stats, extras)``: :func:`_block_stats` per variable, and the block's
    share of the kinetic energy (``u``, ``v``) or of the CDR difference (a tracer and
    its ``_ALT_CO2`` twin).
    """
    offset = levels.start if levels is not None else 0
    volume, deep = (g[levels] for g in geometry) if levels is not None else (None,) * 2
    stats, kept = {}, {}
    for name in unit:
        kind = run.kinds[name]
        x = _read(run, step, name, levels)
        geom = (volume, deep) if kind == "rho3" else None
        stats[name], fin = _block_stats(name, x, run.ocean(kind), run.h, offset, geom)
        if len(unit) > 1:
            kept[name] = (x, fin)
        del x, fin
    extras: dict[str, float] = {}
    if unit == ("u", "v"):
        # a wall carries no flow: land (and non-finite) points are zeroed before the
        # average to rho, so whatever a land point holds never reaches the cell beside
        u, v = (
            _to_rho(np.where(fin, x, 0.0), axis)
            for (x, fin), axis in ((kept["u"], -1), (kept["v"], -2))
        )
        speed2 = u * u + v * v
        del u, v
        wet = np.where(run.mask & np.isfinite(speed2), speed2, 0.0)
        extras["KE"] = 0.5 * float(wet.reshape(-1) @ volume.reshape(-1))
    elif len(unit) == 2:
        (x, fin), (x_alt, fin_alt) = kept[unit[0]], kept[unit[1]]
        both = fin & fin_alt
        diff = np.where(both, x - x_alt, 0.0).astype(np.float64, copy=False)
        extras[f"cdr_{unit[0]}"] = float(diff.reshape(-1) @ volume.reshape(-1))
    return stats, extras


def _combine(parts: list[dict]) -> dict:
    """Return one variable's statistics from its blocks', given in block order.

    Counts and sums add up; the minimum and maximum keep the first block's on a tie,
    so the location is the first in the whole variable, as one ``argmin`` would give.
    """
    out = dict(parts[0])
    for p in parts[1:]:
        for k, v in p.items():
            if k not in ("min", "max", "imin", "imax"):
                out[k] += v
        if p["min"] < out["min"]:
            out["min"], out["imin"] = p["min"], p["imin"]
        if p["max"] > out["max"]:
            out["max"], out["imax"] = p["max"], p["imax"]
    return out


def _scan_step(run: _Run, step, stamp: pd.Timestamp, zeta: np.ndarray) -> pd.DataFrame:
    """Return one step's statistics: a row per scanned variable, then derived rows.

    ``step`` is the step of :func:`_scan_dataset`'s source; each block of levels of
    each variable is read once and reduced in numpy, by :func:`_workers` threads in
    variable order, so what is in memory at a time is the geometry and about one
    variable. Only candidate pairs of copies are read a second time.
    """
    geometry = _geometry(run, zeta)
    nz = run.ds.sizes[run.s_dim]
    jobs = [
        (unit, levels)
        for unit in _units(run)
        for levels in _level_blocks(run, step, unit[0])
    ]

    def scan(job):
        return _scan_block(run, step, *job, geometry)

    # a store chunked a whole variable at a time cannot be read in smaller blocks:
    # fewer threads then, so that they still hold about one variable between them
    largest = max(
        step[run.standard[unit[0]]].isel({run.s_dim: levels}).nbytes
        if levels is not None
        else step[run.standard[unit[0]]].nbytes
        for unit, levels in jobs
    )
    room = max(_BLOCK_BYTES, _block_bytes(run) * _workers())
    workers = min(_workers(), len(jobs), max(1, room // max(largest, 1)))
    if workers > 1:
        with ThreadPoolExecutor(workers, thread_name_prefix="health") as pool:
            results = list(pool.map(scan, jobs))
    else:
        results = [scan(job) for job in jobs]
    parts: dict[str, list[dict]] = {}
    sums: dict[str, float] = {}
    for stats, extras in results:
        for name, p in stats.items():
            parts.setdefault(name, []).append(p)
        for k, v in extras.items():
            sums[k] = sums.get(k, 0.0) + v
    got = {name: _combine(parts[name]) for name in run.standard}
    more: dict[str, float] = {"deep_volume": float(geometry[1].sum())}
    if "KE" in sums:
        more["KE"] = sums["KE"]
    more.update((f"cdr_{n}", sums[f"cdr_{n}"]) for n in run.kinds if f"cdr_{n}" in sums)
    geometry = None  # the step's volumes are not needed past here

    copies = _find_copies(run, step, got)
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
            for which in ("min", "max"):
                lon, lat, z = _locate(run, zeta, kind, r[f"i{which}"])
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


def _find_copies(run: _Run, step, got: dict) -> dict[str, str]:
    """Return ``{variable: the earlier-named variable it equals at every ocean cell}``.

    Cheap first: only variables of one layout whose finite count, minimum, maximum and
    sum all agree are candidates, and only candidates are read again, a block at a
    time, to compare cell by cell. A field that is zero everywhere is not a copy of
    another zero field.
    """
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
    copies: dict[str, str] = {}
    for names in groups.values():
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                if b not in copies and _same(run, step, a, b):
                    copies[b] = a
    return copies


def _same(run: _Run, step, a: str, b: str) -> bool:
    """Return whether ``a`` and ``b`` agree wherever either is a finite ocean value."""
    ok = run.ocean(run.kinds[a])
    for levels in _level_blocks(run, step, a):
        xa, xb = _read(run, step, a, levels), _read(run, step, b, levels)
        either = (np.isfinite(xa) | np.isfinite(xb)) & ok
        if not np.all((xa == xb) | ~either):
            return False
    return True


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
    scan = None  # reopened in blocks only once a step is not in the cache
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
        zeta = sample[0] if sample else _zeta(run, step)
        if scan is None:
            scan = _scan_dataset(source, run)
        frame = _scan_step(run, scan.isel({run.tdim: i}), stamp, zeta)
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
    return format_value(x, 2)


def _signed(x: float) -> str:
    """Return a signed number: ``+5``, ``−1.1`` (a true minus sign), ``+inf``."""
    text = format_value(abs(x), 2)
    return f"{'−' if x < 0 else '+'}{text}"


def _name(variable: str) -> str:
    """Return the readable name of a scanned or derived variable."""
    if variable in _NAMES:
        return _NAMES[variable]
    if variable.endswith(_ALT) and variable != _ALT:
        return f"{_name(variable[: -len(_ALT)])} (ALT_CO2)"
    return variable


def _all_nan_steps(g: pd.DataFrame) -> int:
    """Return in how many of a variable's steps no ocean cell holds a valid value."""
    return int(((g["n_ocean"] > 0) & (g["n_nan"] >= g["n_ocean"])).sum())


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
        self._jump_table: pd.DataFrame | None = None

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
        plus Inf ocean cells at the worst step, at ``nonfinite_time``),
        ``all_nan_steps`` (steps in which not one ocean cell is valid, of ``n_steps``)
        and ``flags``.
        For a concentration the out-of-bounds cells *are* the negative ones, so
        ``worst_pct_bad`` is the one percentage to read.
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
            nonfinite = g["n_nan"] + g["n_inf"]
            row["n_nonfinite"] = int(nonfinite.max())
            row["nonfinite_time"] = g.loc[nonfinite.idxmax(), "time"]
            row["all_nan_steps"] = _all_nan_steps(g)
            row["n_steps"] = len(g)
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
        nan = _all_nan_steps(g)
        if nan:
            flags.append("all NaN" if nan == n else f"all NaN in {nan}/{n}")
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

        The living plankton pools are not looked at (see the module docstring). A step
        is flagged where the rate of change between two neighboring restarts
        exceeds :data:`_JUMP_FACTOR` times the series' median rate and the change is
        more than :data:`_JUMP_MIN_FRACTION` of its value. Columns ``time`` (the later
        step), ``previous``, ``variable``, ``change_pct``, ``ratio`` (inf if the series
        is otherwise flat) and ``at_join``.
        """
        if self._jump_table is None:
            self._jump_table = self._find_jumps()
        return self._jump_table

    def _find_jumps(self) -> pd.DataFrame:
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
            *(v for v in self._inventories() if not _LIVING.fullmatch(v)),
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

        Each has ``kind`` (``nonfinite``, ``all_nan``, ``negative``, ``out_of_bounds``,
        ``copy``, ``all_zero``, ``step`` or ``repeat``), ``variable``, a plain-words
        ``message`` and the numbers behind it (a ``step`` has one item per series, with
        the ``previous`` and ``time`` of its two restarts). Variables negative or out of
        bounds in under 0.1% of their ocean cells at every step are in :meth:`table`
        only. The summary page groups these by kind.
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
            if row.all_nan_steps == row.n_steps:
                out.append(
                    {
                        "kind": "all_nan",
                        "variable": variable,
                        "message": f"{variable} is NaN in every ocean cell, always.",
                        "count": int(row.n_nonfinite),
                        "restarts": int(row.n_steps),
                        "n_restarts": int(row.n_steps),
                        "magnitude": float(row.n_nonfinite),
                    }
                )
                continue
            if row.n_nonfinite:
                out.append(
                    {
                        "kind": "nonfinite",
                        "variable": variable,
                        "message": (
                            f"{variable} has {row.n_nonfinite} NaN or Inf ocean "
                            f"cells at its worst restart ({_day(row.nonfinite_time)})."
                        ),
                        "count": int(row.n_nonfinite),
                        "time": _day(row.nonfinite_time),
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
                    f" (lowest {format_value(row.min)} at "
                    f"{_lonlat(row.min_lon, row.min_lat)}, "
                    f"{_meters(row.min_depth)}, {_day(row.min_time)})"
                )
            elif row.worst_pct_bad > _PROBLEM_PCT:
                kind, what = "out_of_bounds", "outside its physical bounds"
                worst, when, now = (
                    row.worst_pct_bad,
                    row.worst_bad_time,
                    row.latest_pct_bad,
                )
                where = f" (range {format_value(row.min)} to {format_value(row.max)})"
            else:
                continue
            out.append(
                {
                    "kind": kind,
                    "variable": variable,
                    "message": (
                        f"{variable} is {what} in {format_value(worst)}% of ocean "
                        f"cells at worst ({_day(when)}) and {format_value(now)}% at "
                        f"the latest restart{where}."
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
                        "n_restarts": int(n_steps),
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
                        "n_restarts": int(n_steps),
                        "magnitude": float(len(zero)),
                    }
                )
        return out

    def _step_problems(self) -> list[dict]:
        out = []
        for row in self._jumps().itertuples(index=False):
            size = (
                "while it is otherwise flat"
                if np.isinf(row.ratio)
                else f"{format_value(row.ratio)} times the typical step"
            )
            out.append(
                {
                    "kind": "step",
                    "variable": row.variable,
                    "message": (
                        f"{row.variable} changed {_signed(row.change_pct)}% between "
                        f"{_day(row.previous)} and {_day(row.time)}, {size}"
                        f"{JOIN_NOTE if row.at_join else ''}."
                    ),
                    "previous": _day(row.previous),
                    "time": _day(row.time),
                    "change_pct": float(row.change_pct),
                    "ratio": None if np.isinf(row.ratio) else float(row.ratio),
                    "at_join": bool(row.at_join),
                    "magnitude": 1e3 if np.isinf(row.ratio) else float(row.ratio),
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
                    "share_pct": float(100 * share[when]),
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

    # ---- the summary page

    def _cdr_is_zero(self, variable: str) -> bool:
        """Return whether ``variable`` (``cdr_X``) is zero in every restart.

        Zero means within :data:`_CDR_ZERO` of the inventory of ``X`` itself: the
        rounding of subtracting two equal numbers, nothing a perturbation could be.
        """
        s = self._series(variable).dropna()
        if s.empty:
            return False
        base = self._series(variable[4:]).dropna()
        reference = float(base.abs().mean()) if len(base) else 0.0
        return float(s.abs().max()) <= _CDR_ZERO * reference

    def _summary_rows(self, cap: int, step_rows: int) -> list[tuple[str, str, str]]:
        """Return the summary table: ``(check, what was found, where/when)`` per kind.

        One row per kind of problem, left out when there is none. A list in a row shows
        at most ``cap`` items and says how many more the integrity table has; at most
        ``step_rows`` sudden-step dates get a row of their own.
        """
        problems = self.problems()
        rows: list[tuple[str, str, str]] = []

        def of(kind: str) -> list[dict]:
            return [p for p in problems if p["kind"] == kind]

        def listing(items: list[str], sep: str = ", ", more: str = "") -> str:
            text = sep.join(items[:cap])
            rest = len(items) - cap
            if rest > 0:
                text += f"{sep.rstrip()} +{rest} more {more or '(see integrity table)'}"
            return text

        def sticky(text: str) -> str:
            return text.replace(" ", "\u00a0")  # a name and its number wrap together

        def span(items: list[dict]) -> str:
            full = all(p["restarts"] == p["n_restarts"] for p in items)
            return "every restart" if full else "some restarts"

        def partial(p: dict) -> str:
            full = p["restarts"] == p["n_restarts"]
            return "" if full else f" ({p['restarts']}/{p['n_restarts']})"

        bad = sorted(of("nonfinite"), key=lambda p: -p["count"])
        if bad:
            top = bad[0]
            items = [sticky(f"{p['variable']} {format_value(p['count'])}") for p in bad]
            rows.append(
                (
                    "NaN / Inf",
                    (
                        f"{len(bad)} variable{'s' if len(bad) != 1 else ''}, worst "
                        f"cell count: {listing(items)}"
                    ),
                    f"worst {top['variable']}, {top['time']}",
                )
            )
        nan = of("all_nan")
        if nan:
            names = [sticky(p["variable"]) for p in nan]
            rows.append(
                (
                    "Entirely NaN",
                    f"no valid ocean cell at all: {listing(names)}",
                    span(nan),
                )
            )
        zero = of("all_zero")
        if zero:
            names = [sticky(p["variable"] + partial(p)) for p in zero]
            rows.append(("All zero", listing(names), span(zero)))

        neg = sorted(of("negative"), key=lambda p: -p["worst_pct"])
        if neg:
            table = self.table()
            n_tracers = int(
                sum(
                    1
                    for v in table["variable"]
                    if v not in _NOT_CONCENTRATION and not _is_ph(v)
                )
            )
            items = [sticky(f"{p['variable']} {_pct(p['worst_pct'])}%") for p in neg]
            still = sum(1 for p in neg if p["latest_pct"] > 0)
            rows.append(
                (
                    "Negative concentrations",
                    f"{len(neg)} of {n_tracers} tracers; worst {listing(items)}",
                    (
                        f"worst {neg[0]['variable']}, {neg[0]['time']}; "
                        + (
                            f"{still} still negative at the latest restart"
                            if still
                            else "none negative at the latest restart"
                        )
                    ),
                )
            )
        oob = sorted(of("out_of_bounds"), key=lambda p: -p["worst_pct"])
        if oob:
            items = [sticky(f"{p['variable']} {_pct(p['worst_pct'])}%") for p in oob]
            rows.append(
                (
                    "Outside physical bounds",
                    f"share of ocean cells at worst: {listing(items)}",
                    f"worst {oob[0]['variable']}, {oob[0]['time']}",
                )
            )
        copies = of("copy")
        if copies:
            items = [
                sticky(f"{p['variable']} = {p['copy_of']}{partial(p)}") for p in copies
            ]
            rows.append(("Exact copies", listing(items, "; "), span(copies)))
        rows += self._cdr_rows()
        rows += self._step_rows(of("step"), cap, step_rows)
        repeats = of("repeat")
        if repeats:
            items = [
                sticky(
                    f"{p['time']}: {_pct(p['share_pct'])}% of variables"
                    if p["variable"] == "(all)"
                    else f"{p['variable']} x{p['restarts']}"
                )
                for p in repeats
            ]
            rows.append(
                (
                    "Repeated restart",
                    f"identical to the previous restart: {listing(items)}",
                    "",
                )
            )
        drift = self.drift()
        # a living pool's drift is mostly spin-up biology, as for the steps
        drift = drift[~drift["variable"].map(lambda v: bool(_LIVING.fullmatch(v)))]
        if len(drift):
            items = []
            for r in drift.head(min(cap, 6)).itertuples(index=False):
                name = _name(r.variable[5:] if r.series == "deep mean" else r.variable)
                name = f"deep {name.lower()}" if r.series == "deep mean" else name
                items.append(sticky(f"{name} {_signed(r.pct_per_year)}"))
            rows.append(
                (
                    "Largest drift",
                    f"{', '.join(items)} %/yr",
                    "linear fit over the record",
                )
            )
        if not rows:
            rows.append(("Result", "nothing flagged by any check", ""))
        return rows

    def _cdr_rows(self) -> list[tuple[str, str, str]]:
        """Return the CDR row: is any added alkalinity or carbon in the tracers."""
        cdr = [v for v in self._derived() if v.startswith("cdr_")]
        if not cdr:
            return []
        zero = [v[4:] for v in cdr if self._cdr_is_zero(v)]
        moving = [v for v in cdr if v[4:] not in zero]
        parts = []
        if zero:
            parts.append(
                " and ".join(f"{x} − {x}{_ALT}" for x in zero) + " = 0 in every restart"
            )
        unit = display("mmol")
        for v in moving:
            s = self._series(v).dropna()
            if len(s):
                parts.append(
                    f"{v[4:]} − {v[4:]}{_ALT} {format_value(float(s.iloc[0]))} → "
                    f"{format_value(float(s.iloc[-1]))} {unit} (first → latest)"
                )
        if not moving:
            parts[-1] += ": no CDR signal in the tracers"
        return [("CDR tracers", "; ".join(parts), "all restarts" if not moving else "")]

    def _step_rows(
        self, steps: list[dict], cap: int, step_rows: int
    ) -> list[tuple[str, str, str]]:
        """One row per pair of restarts a sudden step falls between."""
        pairs: dict[tuple[str, str], list[dict]] = {}
        for p in steps:
            pairs.setdefault((p["previous"], p["time"]), []).append(p)

        def size(g: list[dict]) -> float:
            return max(p["magnitude"] for p in g)

        chosen = sorted(pairs, key=lambda k: -size(pairs[k]))[:step_rows]
        rows = []
        for key in sorted(chosen):
            g = sorted(pairs[key], key=lambda p: -p["magnitude"])
            items = [
                f"{_DERIVED_NAMES.get(p['variable'], p['variable'])} "
                f"{_signed(p['change_pct'])}%".replace(" ", "\u00a0")
                for p in g
            ]
            text = ", ".join(items[:cap])
            if len(items) > cap:
                text += f", +{len(items) - cap} more (see inventory figures)"
            if any(p["ratio"] is None for p in g):
                text += " (otherwise flat)"
            else:
                biggest = format_value(max(p["ratio"] for p in g))
                text += f" (up to {biggest}× the typical change)"
            first, last = key
            last = last[5:] if last[:4] == first[:4] else last
            when = f"{first} → {last}" + ("\nat a job join" if g[0]["at_join"] else "")
            rows.append(("Sudden step", text, when))
        rest = len(pairs) - len(chosen)
        if rest > 0:
            rows.append(
                ("Sudden step", f"+{rest} more step dates", "see inventory figures")
            )
        return rows

    def _draw_summary(self) -> list:
        from matplotlib.figure import Figure

        from ocean_skill.plot.typography import PAGE_W

        # shorten the lists until the table fits the page (one item is the floor)
        for cap in (8, 6, 5, 4, 3, 2, 1):
            rows = self._summary_rows(cap, cap)
            cells, widths, heights = _lay_out(
                [("Check", "What was found", "Where / when"), *rows],
                PAGE_W - 0.6,
                _SUMMARY_PT,
                fixed={0: 1.1, 2: 1.8},
                flexible=1,
            )
            top = _TITLE_IN + (0.3 if self.test_mode else 0.0)
            height = top + sum(heights) + 0.25
            if height <= _LETTER[1]:
                break
        fig = Figure(figsize=(PAGE_W, height))
        _draw_title(fig, self._title("summary"), self.test_mode)
        _draw_grid(fig, 0.3, height - top, widths, heights, cells, _SUMMARY_PT)
        return [fig]

    def _draw_integrity(self) -> list:
        from matplotlib.figure import Figure

        from ocean_skill.plot.typography import PAGE_W

        head = (
            "Variable",
            "Bound",
            "Worst %\nbad (date)",
            "Latest\n%",
            "Minimum",
            "Maximum",
            "NaN/Inf",
            "Flags",
        )
        rows = []
        for r in self.table().itertuples(index=False):
            rows.append(
                (
                    r.variable,
                    _bound_text(r.variable),
                    f"{format_value(r.worst_pct_bad)}\n{_day(r.worst_bad_time)}"
                    if r.worst_pct_bad
                    else "0",
                    format_value(r.latest_pct_bad),
                    _extreme(r, "min"),
                    _extreme(r, "max"),
                    format_value(int(r.n_nonfinite)),
                    r.flags,
                )
            )
        size = _INTEGRITY_PT
        while True:  # the font shrinks only if the columns cannot fit at this one
            cells, widths, heights = _lay_out(
                [head, *rows], PAGE_W - 0.6, size, flexible=7, min_flexible=0.9
            )
            if sum(widths) <= PAGE_W - 0.6 + 1e-9 or size <= 5.0:
                break
            size -= 0.5
        top = _TITLE_IN + (0.3 if self.test_mode else 0.0)
        budget = _FIG_MAX_H - top - 0.25
        pages, chunk, used = [], [], 0.0
        for row, h in zip(cells[1:], heights[1:], strict=True):
            if chunk and used + h > budget - heights[0]:
                pages.append(chunk)
                chunk, used = [], 0.0
            chunk.append((row, h))
            used += h
        pages.append(chunk)
        figures = []
        for part, chunk in enumerate(pages):
            hs = [heights[0], *(h for _, h in chunk)]
            height = top + sum(hs) + 0.25
            fig = Figure(figsize=(PAGE_W, height))
            title = self._title("integrity by variable") + (" (cont.)" if part else "")
            _draw_title(fig, title, self.test_mode)
            _draw_grid(
                fig,
                0.3,
                height - top,
                widths,
                hs,
                [cells[0], *(row for row, _ in chunk)],
                size,
                centered=(1, 2, 3, 6),
            )
            figures.append(fig)
        return figures

    # ---- the series pages

    def _panel(
        self, variable: str, title: str, units: str, note: str | None = None
    ) -> dict | None:
        """One series panel of ``variable``, or ``None`` if the run has no such series.

        ``units="%"`` draws the series as percent change from its first step (so many
        inventories share a scale); any other ``units`` draws it as it is, in those
        units (spelled by :func:`ocean_skill.units.display`). A series that starts at
        zero has nothing to scale by and is drawn as it is. ``note`` replaces the line
        with a centered sentence (a series with nothing to show).
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
            unit = "% change"
        else:
            unit = "" if units == "%" else display(units)
        return {
            "title": title,
            "series": s,
            "unit": unit,
            "flagged": flagged,
            "note": note,
        }

    def _flagged(self, variable: str) -> list:
        """Return the steps at which ``variable`` jumps."""
        jumps = self._jumps()
        return list(jumps.loc[jumps["variable"] == variable, "time"])

    def _series_figures(self, panels: list[dict], what: str) -> list:
        """Lay ``panels`` out as grids of small series panels (the series family).

        Each panel is one line (:class:`~ocean_skill.plot.spec.PlotSpec` ``series``
        family, so the static and interactive renderers draw it alike), every other job
        segment shaded and each flagged step ringed in red. Every title is the same
        size; a panel with a ``note`` shows that sentence instead of its line.
        """
        import xarray as xr
        from matplotlib.ticker import FuncFormatter

        from ocean_skill.plot.registry import render
        from ocean_skill.plot.series import HIGHLIGHT_ALPHA, HIGHLIGHT_COLORS
        from ocean_skill.plot.spec import PlotSpec
        from ocean_skill.plot.typography import PAGE_W

        bands = self._bands()
        footer = [TEST_MODE_BANNER] if self.test_mode else []
        if bands:
            footer.append(SEGMENT_NOTE)
        foot_in = 0.14 * len(footer) + (0.1 if footer else 0.0)
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
            height = 0.9 + _SERIES_ROW_IN * nrows + foot_in
            options: dict[str, Any] = {
                "titles": [p["title"] for p in chunk],
                "ncols": ncols,
                "legend": False,
                "encode": {"color": None, "marker": None, "linestyle": None},
                "figsize": (PAGE_W, height),
                "title": self._title(what) + (" (cont.)" if part else ""),
            }
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                spec = PlotSpec(family="series", items=items, options=options)
                fig = render(spec, renderer="matplotlib")
            if footer:
                fig.get_layout_engine().set(
                    rect=(0, foot_in / height, 1, 1 - foot_in / height)
                )
            by_title = {ax.get_title(): ax for ax in fig.axes}
            for p in chunk:
                ax = by_title.get(p["title"])
                if ax is None:
                    continue
                ax.title.set_fontsize(_PANEL_TITLE_PT)
                ax.set_ylabel(p["unit"], fontsize=7)
                # tick values the way every number here is printed, and no offset label
                # ("1e-5+2.1975"), which collides with the panel title above it
                ax.yaxis.set_major_formatter(FuncFormatter(_tick_label(ax)))
                xlim = ax.get_xlim()
                for low, high in bands:
                    ax.axvspan(
                        low,
                        high,
                        color=HIGHLIGHT_COLORS[0],
                        alpha=HIGHLIGHT_ALPHA,
                        linewidth=0,
                        zorder=0,
                    )
                ax.set_xlim(xlim)
                _date_ticks(ax)
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
                if p.get("note"):
                    for line in list(ax.lines):
                        line.remove()
                    ax.set_yticks([])
                    ax.set_ylabel("")
                    ax.text(
                        0.5,
                        0.5,
                        p["note"],
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="0.35",
                    )
            for k, line in enumerate(footer):
                fig.text(
                    0.5,
                    (0.05 + 0.14 * (len(footer) - 1 - k)) / height,
                    line,
                    ha="center",
                    va="bottom",
                    color="firebrick" if line == TEST_MODE_BANNER else "0.35",
                    fontsize=8 if line == TEST_MODE_BANNER else 7,
                )
            pages.append(fig)
        return pages

    def _bands(self) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """Return the job segments to shade (every other one) as ``(start, end)``."""
        times = self._times()
        if not len(times):
            return []
        joins = [j for j in self.joins if times[0] < j <= times[-1]]
        edges = [times[0], *joins, times[-1]]
        return [(edges[i], edges[i + 1]) for i in range(1, len(edges) - 1, 2)]

    def _draw_inventory(self) -> list:
        plankton = {f"{g}{p}" for g, _ in _GROUPS for p in _PLANKTON_PARTS} | {_ZOO}
        tracers = sorted(
            (v for v in self._inventories() if v not in plankton),
            key=lambda v: ({"temp": 0, "salt": 1}.get(v, 2), v),
        )
        deep = f"(> {format_value(_DEEP_M)} m)"
        conc = display("mmol/m3")
        wanted = [(v, _name(v), "%") for v in tracers[:2]]
        wanted += [
            ("KE", "Kinetic energy", "%"),
            ("O2_lt60", f"O2 volume < 60 {conc}", "%"),
            ("O2_lt20", f"O2 volume < 20 {conc}", "%"),
            ("deep_temp", f"Deep temperature {deep}", "degC"),
            ("deep_salt", f"Deep salinity {deep}", "psu"),
            ("deep_O2", f"Deep oxygen {deep}", "mmol/m3"),
            ("deep_DIC", f"Deep DIC {deep}", "mmol/m3"),
        ]
        wanted += [(v, _name(v), "%") for v in tracers[2:]]
        panels = [self._panel(*w) for w in wanted]
        for v in self._derived():
            if not v.startswith("cdr_"):
                continue
            x = v[4:]
            same = self._cdr_is_zero(v)
            panel = self._panel(
                v,
                f"{_name(x)} minus ALT_CO2",
                "mmol",
                note=f"identical to {x} in every restart" if same else None,
            )
            panels.append(panel)
        return self._series_figures([p for p in panels if p], "inventories")

    def _draw_plankton(self) -> list:
        have = set(self.steps["variable"])
        panels = []
        for group, label in _GROUPS:
            for part in _PLANKTON_PARTS:
                panels.append(
                    self._panel(f"{group}{part}", f"{label.capitalize()} {part}", "%")
                )
        panels.append(self._panel(_ZOO, _name(_ZOO), "%"))
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
                        panels.append(
                            {
                                "title": f"{label.capitalize()} {name}",
                                "series": ratio,
                                "unit": display(unit),
                                "flagged": [],
                                "note": None,
                            }
                        )
        return self._series_figures(panels, "plankton") if panels else []


# --------------------------------------------------------------------------------------
# Tables and text


def _bound_text(variable: str) -> str:
    """Return the bound a variable's "bad" count is measured against, short."""
    if variable in _RANGES:
        low, high = _RANGES[variable]
        return f"{_minus(low)} to {_minus(high)}"
    if variable in _ABS_LIMITS:
        unit = display("m/s" if variable != "zeta" else "m")
        return f"≤ {format_value(_ABS_LIMITS[variable])} {unit}"
    if _is_ph(variable):
        return f"{format_value(_PH_RANGE[0])} to {format_value(_PH_RANGE[1])}"
    if variable in _LAYER_DEPTHS:
        return "0 to depth"
    return "≥ 0"


def _tick_label(ax):
    """Return a tick formatter for ``ax``'s value axis that spells ticks as numbers are.

    Zero is ``0`` however near it the tick landed, and a tick gets as many digits as it
    takes to tell it from its neighbor (a total of 1.0399e13 beside 1.0401e13).
    """

    def label(value: float, _position) -> str:
        ticks = np.asarray(ax.get_yticks(), dtype=float)
        gaps = np.abs(np.diff(ticks))
        step = float(gaps[gaps > 0].min()) if (gaps > 0).any() else 0.0
        if step == 0.0 or abs(value) < 1e-6 * step:
            return "0" if step else format_value(value)
        digits = int(np.ceil(np.log10(abs(value) / step))) + 1
        return format_value(value, max(3, min(digits, 9))).replace("-", "−")

    return label


def _minus(x: float) -> str:
    """Return a number with a true minus sign."""
    return format_value(x).replace("-", "−")


def _extreme(r, which: str) -> str:
    """Return the table cell of a run-wide minimum or maximum: value, date, place."""
    value = getattr(r, which)
    if np.isnan(value):
        return "–"
    lon, lat, depth = (getattr(r, f"{which}_{k}") for k in ("lon", "lat", "depth"))
    when = _day(getattr(r, f"{which}_time"))
    return f"{format_value(value)}  {when}\n{_lonlat(lon, lat)}, {_meters(depth)}"


def _meters(depth: float) -> str:
    """``"130 m"``; a cell just above z = 0 reads ``"0 m"``, not ``"-0 m"``."""
    return f"{format_value(round(max(float(depth), 0.0)))} m"


@functools.lru_cache(maxsize=1)
def _measurer():
    """Return a renderer that measures text in points (72 dpi), without a figure."""
    from matplotlib.backends.backend_agg import RendererAgg

    return RendererAgg(100, 100, 72)


def _text_width(text: str, size: float, bold: bool = False) -> float:
    """Return the width in inches of one line of ``text`` at ``size`` pt."""
    from matplotlib.font_manager import FontProperties

    prop = FontProperties(size=size, weight="bold" if bold else "normal")
    width, _, _ = _measurer().get_text_width_height_descent(text, prop, ismath=False)
    return width / 72.0


def _wrap(text: str, width: float, size: float, bold: bool = False) -> list[str]:
    """Wrap ``text`` to lines no wider than ``width`` inches; break a too-long word."""
    lines: list[str] = []
    for paragraph in text.split("\n"):
        current = ""
        for word in paragraph.split(" "):
            trial = f"{current} {word}" if current else word
            if not current or _text_width(trial, size, bold) <= width:
                current = trial
            else:
                lines.append(current)
                current = word
        lines.append(current)
    out = []
    for line in lines:
        while len(line) > 1 and _text_width(line, size, bold) > width:
            k = max(
                (
                    i
                    for i in range(1, len(line))
                    if _text_width(line[:i], size, bold) <= width
                ),
                default=1,
            )
            out.append(line[:k])
            line = line[k:]
        out.append(line)
    return [line.replace("\u00a0", " ") for line in out]


#: Space (in) around the text of a table cell, and between its lines.
_PAD_X, _PAD_Y = 0.05, 0.035
#: Height (in) the page title and its gap take at the top of a table page.
_TITLE_IN = 0.75


def _lay_out(
    rows: list,
    avail: float,
    size: float,
    *,
    fixed: dict[int, float] | None = None,
    flexible: int,
    min_flexible: float = 0.0,
) -> tuple[list[list[str]], list[float], list[float]]:
    r"""Size a text table from its content: return ``(cells, widths, heights)``.

    Every column is as wide as its widest line (``fixed`` ones have the width given), so
    no cell is ever cut off; the ``flexible`` column takes what is left of ``avail``
    inches and wraps. A cell's lines are its text with the wrapped breaks joined by
    ``\n``; a row is as tall as its tallest cell.
    """
    fixed = fixed or {}
    ncol = len(rows[0])
    widths = [0.0] * ncol
    for j in range(ncol):
        if j == flexible:
            continue
        if j in fixed:
            widths[j] = fixed[j]
            continue
        longest = max(
            _text_width(line, size, bold=i == 0)
            for i, row in enumerate(rows)
            for line in str(row[j]).replace("\u00a0", " ").split("\n")
        )
        widths[j] = longest + 2 * _PAD_X + 0.02
    widths[flexible] = max(avail - sum(widths), min_flexible)
    cells: list[list[str]] = []
    heights: list[float] = []
    line_in = size * 1.3 / 72.0
    for i, row in enumerate(rows):
        wrapped = [
            "\n".join(_wrap(str(text), widths[j] - 2 * _PAD_X, size, bold=i == 0))
            if j in fixed or j == flexible
            else str(text)
            for j, text in enumerate(row)
        ]
        cells.append(wrapped)
        heights.append(max(c.count("\n") + 1 for c in wrapped) * line_in + 2 * _PAD_Y)
    return cells, widths, heights


def _draw_grid(
    fig,
    left: float,
    top: float,
    widths: list[float],
    heights: list[float],
    cells: list[list[str]],
    size: float,
    *,
    centered: tuple[int, ...] = (),
) -> None:
    """Draw ``cells`` as a grid, the top-left corner at ``(left, top)`` inches.

    The first row is the header (gray, bold); the others alternate white and light
    gray. ``centered`` columns center their text, the others align left. ``fig`` is
    labelled a text table (:data:`~ocean_skill.plot.typography.TABLE_FIGURE`).
    """
    from matplotlib.patches import Rectangle

    from ocean_skill.plot.typography import TABLE_FIGURE

    fig.set_label(TABLE_FIGURE)
    w, h = fig.get_size_inches()
    y = top
    for i, (row, height) in enumerate(zip(cells, heights, strict=True)):
        x = left
        face = "0.9" if i == 0 else ("0.96" if i % 2 == 0 else "w")
        for j, (text, width) in enumerate(zip(row, widths, strict=True)):
            fig.add_artist(
                Rectangle(
                    (x / w, (y - height) / h),
                    width / w,
                    height / h,
                    transform=fig.transFigure,
                    facecolor=face,
                    edgecolor="0.75",
                    linewidth=0.4,
                )
            )
            middle = i == 0 or j in centered
            fig.text(
                (x + width / 2 if middle else x + _PAD_X) / w,
                (y - height / 2) / h,
                text,
                ha="center" if middle else "left",
                va="center",
                fontsize=size,
                fontweight="bold" if i == 0 else "normal",
                linespacing=1.15,
            )
            x += width
        y -= height


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


def _date_ticks(ax) -> None:
    """Give a time axis few enough ticks that its labels never run together.

    Three small panels share a page width, so a multi-year record gets one tick a year
    (``2011``, ``2012``), and a shorter one a tick every few months (``Jan``, ``Apr``),
    with the year on its first tick.
    """
    import matplotlib.dates as mdates

    low, high = ax.get_xlim()
    days = high - low
    if days > 540:
        step = max(1, round(days / 365.25 / 4))
        ax.xaxis.set_major_locator(mdates.YearLocator(step))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    else:
        months = 1 if days < 120 else 3
        locator = mdates.MonthLocator(interval=months)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
