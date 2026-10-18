"""Input-file integrity: per-variable statistics of the files a ROMS run is given.

:func:`input_health` makes one streaming pass over catalog sources that hold a run's
*inputs* -- the initial conditions, the open-boundary files (one entry per side), the
surface-forcing, restoring and CO2 files -- and answers, for every data variable of
every source: what range does it span, is any of it NaN, negative or outside what is
physically plausible, is it all zero or the same everywhere, is it a copy of another
variable of the file. It is the input-side twin of :func:`ocean_skill.health` (which
does the same for a run's output): no observations are involved, only the file against
itself and against a plausible range.

Each variable is read a time step (or a few) at a time, and for a 3-D field a block of
levels at a time, and reduced in numpy -- never held whole: a month of hourly 2-D
forcing on a million-cell grid is tens of GB. A block is about :data:`_BLOCK_BYTES`;
a thread also holds the temporaries of reducing it (a float64 copy, masks), so its
peak is about five blocks, and at most :data:`_MAX_WORKERS` threads read at once.
``time`` picks the window to
scan (a month, say) and ``max_steps`` thins it to evenly spread steps; the table says
how many steps, and which span, every row covers.

Everything is masked with ``mask_rho``, which roms-tools and UCLA-ROMS files do not
apply to their fields (land holds values, often junk): a statistic over the whole array
would count it as ocean. A boundary entry carries the grid's own edge row or column, so
its mask is that edge. u and v points are ocean where both rho neighbors are. A source
with no mask is scanned over every cell, and the figure says so.

**Per-variable cache.** A forcing file does not change once written, so each variable's
per-step table is kept under :mod:`ocean_skill.cache`'s ``inputs`` kind and a re-run of
the suite reads none of the data again. The key is :data:`INPUT_VERSION`, the source's
catalog fingerprint, the scanned steps, the variable's bound and units (they are
baked into the counts), the source file's size and modification time where it is known,
and a hash of a content sample: a 64 x 64 window at each end of the field, at the first,
middle and last scanned step and the first and last level. A cache hit reads only that
sample, so a file rewritten in place misses.

The module is called ``input_integrity`` rather than ``input_health`` so the package can
expose ``osk.input_health`` as the function: a submodule of the same name would be
clobbered as a package attribute, as it would for ``osk.health`` (see
:mod:`ocean_skill.run_health`).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ocean_skill.config import INPUT_DRAWS
from ocean_skill.plot._numbers import format_value
from ocean_skill.run_health import (
    _ABS_LIMITS,
    _FIG_MAX_H,
    _INTEGRITY_PT,
    _LAYER_DEPTHS,
    _PH_RANGE,
    _PROBLEM_PCT,
    _RANGES,
    _SKIP,
    _TITLE_IN,
    _combine,
    _day,
    _draw_grid,
    _draw_title,
    _is_ph,
    _lay_out,
    _lonlat,
    _meters,
    _mid_lon,
    _minus,
    _workers,
)
from ocean_skill.units import display

__all__ = ["INPUT_DRAWS", "INPUT_VERSION", "InputHealth", "bounds", "input_health"]

#: The bytes of one variable a scan block holds, and the most threads reading at once.
#: A thread's peak is a block plus the float64 copy and masks of reducing it (about five
#: blocks), and dozens of concurrent netCDF/HDF5 readers on a big node do not go faster.
_BLOCK_BYTES = 16 * 2**20
_MAX_WORKERS = 6

#: The edge (cells) of the window a cache key samples at each end of a field.
_SAMPLE_EDGE = 64

#: Bumped when the statistics a scan records, or how they are computed, change, so a
#: cached variable from before is not served as one from after. Part of every key.
INPUT_VERSION = 2

#: Inclusive ``(low, high)`` plausible range of the variables only forcing files carry
#: (the tracers and physics ROMS output also has are bounded by
#: :mod:`ocean_skill.run_health`'s tables). ``inf`` is no bound on that side. A
#: value outside is suspicious, not proven wrong: a unit slip or a corrupt cell.
#:
#: * ``Tair`` -60..60 degC, ``qair`` 0..0.05 kg/kg, ``swrad`` 0..1500 and ``lwrad``
#:   0..800 W/m2, ``uwnd``/``vwnd`` within 50 m/s: the surface-physics file.
#: * ``rain``, ``dust``, ``iron``, ``nox`` and ``nhy``/``nhx`` are fluxes, never
#:   negative.
#: * ``xco2`` and ``pco2_air`` 150..1000 (umol/mol, uatm): atmospheric CO2.
#: * ``sss`` 0..42 PSU: the salinity restoring field.
_FORCING_RANGES: dict[str, tuple[float, float]] = {
    "Tair": (-60.0, 60.0),
    "qair": (0.0, 0.05),
    "rain": (0.0, math.inf),
    "swrad": (0.0, 1500.0),
    "lwrad": (0.0, 800.0),
    "uwnd": (-50.0, 50.0),
    "vwnd": (-50.0, 50.0),
    "dust": (0.0, math.inf),
    "iron": (0.0, math.inf),
    "nox": (0.0, math.inf),
    "nhy": (0.0, math.inf),
    "nhx": (0.0, math.inf),
    "xco2": (150.0, 1000.0),
    "pco2_air": (150.0, 1000.0),
    "pco2_air_alt": (150.0, 1000.0),
    "sss": (0.0, 42.0),
}

#: The side a boundary file's variable names end with (``temp_south``).
_SIDE_SUFFIX = re.compile(r"_(south|north|east|west)$")

#: Severity order of :meth:`InputHealth.problems` kinds, most severe first.
_SEVERITY = (
    "nonfinite",
    "all_nan",
    "negative",
    "out_of_bounds",
    "copy",
    "all_zero",
    "constant",
    "repeat",
)

#: The horizontal layouts a variable may have: ``(eta dim, xi dim, kind)`` for a rho,
#: u or v point. Anything else (psi points, river or scalar series) is not scanned.
_LAYOUTS = (
    ("eta_rho", "xi_rho", "r"),
    ("eta_rho", "xi_u", "u"),
    ("eta_v", "xi_rho", "v"),
)

#: The columns of a variable's per-step table, in order.
_COLUMNS = (
    "step",
    "time",
    "month",
    "variable",
    "n_ok",
    "n_fin",
    "n_nan",
    "n_inf",
    "n_neg",
    "n_bad",
    "n_nonzero",
    "pct_neg",
    "pct_bad",
    "total",
    "sumsq",
    "min",
    "min_lon",
    "min_lat",
    "min_depth",
    "min_level",
    "max",
    "max_lon",
    "max_lat",
    "max_depth",
    "max_level",
)

#: Sample size cap of :meth:`InputHealth.figures`' notes: steps verified for a copy.
_COPY_CHECKS = 3


# --------------------------------------------------------------------------------------
# Bounds


def bounds(variable: str) -> tuple[float, float]:
    """Return the inclusive ``(low, high)`` plausible range of ``variable``.

    A boundary file's side suffix (``temp_south``) is ignored. Looked up in the
    forcing-only table, then in the bounds :func:`ocean_skill.health` applies to output
    (temperature, salinity, velocities, free surface, pH); any other variable is a
    concentration, bounded below by zero. ``inf`` is no bound on that side.
    """
    base = _SIDE_SUFFIX.sub("", variable)
    if base in _FORCING_RANGES:
        return _FORCING_RANGES[base]
    if base in _RANGES:
        return _RANGES[base]
    if base in _ABS_LIMITS:
        return (-_ABS_LIMITS[base], _ABS_LIMITS[base])
    if _is_ph(base):
        return _PH_RANGE
    if base in _LAYER_DEPTHS:
        return (0.0, math.inf)
    return (0.0, math.inf)


def _bound_text(variable: str) -> str:
    """Return the bound a variable's "bad" percentage is measured against, short."""
    low, high = bounds(variable)
    if math.isinf(high):
        return "≥ 0" if low == 0 else f"≥ {_minus(low)}"
    return f"{_minus(low)} to {_minus(high)}"


# --------------------------------------------------------------------------------------
# Opening a source


@dataclass
class _Var:
    """One scanned variable: what it is called, where its points are, its layers."""

    native: str  # the name the file gives it
    std: str  # the name in the standardized Dataset
    kind: str  # "r", "u" or "v": which points it sits on
    ydim: str
    xdim: str
    lead: tuple[str, ...]  # dimensions between time and the horizontal (levels)
    units: str


@dataclass
class _Source:
    """What one pass over a source needs, found once: variables, masks and geometry."""

    name: str
    label: str
    ds: Any
    tdim: str | None  # "time", "month" or None (no time axis)
    times: pd.DatetimeIndex | None
    months: np.ndarray | None
    positions: list[int]  # the steps of the time axis that are scanned
    variables: dict[str, _Var]
    mask: np.ndarray | None  # (eta_rho, xi_rho) bool, True = ocean; None = no mask
    lon: np.ndarray | None
    lat: np.ndarray | None
    depth: tuple | None  # (h, sigma_r, Cs_r, hc, Vtransform, s_dim), or None
    thinned: bool = False

    def ocean(self, kind: str, shape: tuple[int, int]) -> np.ndarray:
        """Return the ocean mask on a variable's own points (u/v: both sides wet)."""
        if self.mask is None:
            return np.ones(shape, dtype=bool)
        m = self.mask
        if kind == "u":
            return m[:, :-1] & m[:, 1:]
        if kind == "v":
            return m[:-1] & m[1:]
        return m


def _positions(times: pd.DatetimeIndex, time, name: str) -> list[int]:
    """Return the positions of ``times`` a ``time=`` selection keeps, in order.

    ``None`` keeps every step, ``"latest"`` the last, a date string (``"2011-01"``,
    ``"2011-01-15"``, ``"2011"``) the steps inside that month, day or year, a
    ``{"min": ..., "max": ...}`` dict the steps from ``min`` to ``max`` inclusive
    (either key optional; a date-only ``max`` includes that whole day).
    """
    n = len(times)
    lo, hi = 0, n
    if time is None:
        pass
    elif isinstance(time, str) and time == "latest":
        lo = n - 1
    elif isinstance(time, str):
        try:
            period = pd.Period(time)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"input_health time={time!r} is not a date ('2011-01', '2011-01-15')."
            ) from exc
        lo = int(times.searchsorted(period.start_time, side="left"))
        hi = int(times.searchsorted(period.end_time, side="right"))
    elif isinstance(time, Mapping) and set(time) <= {"min", "max"}:
        if time.get("min") is not None:
            lo = int(times.searchsorted(pd.Timestamp(time["min"]), side="left"))
        if time.get("max") is not None:
            top = pd.Timestamp(time["max"])
            if isinstance(time["max"], str) and ":" not in time["max"]:
                top += pd.Timedelta(days=1) - pd.Timedelta(1, "ns")
            hi = int(times.searchsorted(top, side="right"))
    else:
        raise ValueError(
            'input_health time= is None (every step), "latest", a date string '
            f'("2011-01") or a dict with "min" and/or "max" dates; got {time!r}.'
        )
    if hi <= lo:
        span = f"{_day(times[0])} to {_day(times[-1])}" if n else "no steps"
        raise ValueError(
            f"input_health time={time!r} selects none of the {n} steps of {name!r} "
            f"({span})."
        )
    return list(range(lo, hi))


def _thin(positions: list[int], limit: int | None) -> list[int]:
    """Return at most ``limit`` of ``positions``, evenly spread, first and last kept."""
    if limit is None or len(positions) <= limit:
        return positions
    picks = np.unique(np.linspace(0, len(positions) - 1, max(limit, 2)).round())
    return [positions[int(i)] for i in picks]


def _open_source(entry: dict, time, max_steps: int | None) -> _Source:
    """Open one catalog source and find its variables, window, mask and geometry."""
    from ocean_skill import roms, sources

    name = entry["name"]
    ref = sources.resolve(name)
    ds = sources.read(name)
    meta = ref.metadata
    tdim = meta.get("time_dim", "time")
    times = months = None
    if tdim in ds.dims and tdim in ds.indexes:
        times = pd.DatetimeIndex(ds.indexes[tdim]).as_unit("ns")  # a window needs ns
        positions = _positions(times, time, name)
    elif "month" in ds.dims:
        tdim = "month"
        months = np.asarray(ds["month"].values)
        positions = list(range(len(months)))  # a climatology has no window
    else:
        tdim, positions = None, [0]
    thinned = False
    if len(positions) > (max_steps or len(positions)):
        positions, thinned = _thin(positions, max_steps), True

    reverse = {std: nat for nat, std in (meta.get("standard_names") or {}).items()}
    wanted = entry.get("vars")
    found: dict[str, _Var] = {}
    for std, da in ds.data_vars.items():
        native = reverse.get(str(std), str(std))
        if native in _SKIP or (tdim is not None and tdim not in da.dims):
            continue
        for ydim, xdim, kind in _LAYOUTS:
            if ydim in da.dims and xdim in da.dims:
                lead = tuple(d for d in da.dims if d not in (tdim, ydim, xdim))
                units = str(da.attrs.get("units", "") or "")
                found[native] = _Var(native, str(std), kind, ydim, xdim, lead, units)
                break
    if wanted:
        missing = [v for v in wanted if v not in found]
        if missing:
            raise ValueError(
                f"input_health: {name!r} has no gridded variable(s) {missing} "
                f"(it has {sorted(found)})."
            )
        found = {v: found[v] for v in wanted}
    if not found:
        raise ValueError(
            f"input_health: {name!r} has no variable on the ROMS rho, u or v points "
            "(eta_rho/xi_rho, xi_u or eta_v); it is not a gridded roms-tools input."
        )

    def grid(*names: str) -> np.ndarray | None:
        for n in names:
            if n in ds.variables:
                da = ds[n]
                if {"eta_rho", "xi_rho"} <= set(da.dims):
                    return np.asarray(da.transpose("eta_rho", "xi_rho").values)
        return None

    mask = grid("mask_rho")
    lon, lat, h = grid("lon", "lon_rho"), grid("lat", "lat_rho"), grid("h")
    depth = None
    try:
        hc, vtransform = roms._vertical_params(ds, meta)
        s_dim = meta.get("vertical", {}).get("s_dim", "s_rho")
        if h is not None and all(n in ds.variables for n in ("sigma_r", "Cs_r")):
            depth = (
                h,
                np.asarray(ds["sigma_r"].values, dtype=float),
                np.asarray(ds["Cs_r"].values, dtype=float),
                hc,
                vtransform,
                s_dim,
            )
    except ValueError:
        pass  # no hc: levels are reported as levels, not as depths
    return _Source(
        name=name,
        label=entry.get("label") or name,
        ds=ds,
        tdim=tdim,
        times=times,
        months=months,
        positions=positions,
        variables=found,
        mask=None if mask is None else mask == 1,
        lon=lon,
        lat=lat,
        depth=depth,
        thinned=thinned,
    )


def _place(src: _Source, v: _Var, index: tuple) -> tuple[float, float, float, int]:
    """Return ``(lon, lat, depth, level)`` of the cell at ``index`` of variable ``v``.

    A u or v point sits between two rho points: its longitude, latitude and depth are
    theirs, averaged. ``depth`` is metres below the surface (``nan`` for a 2-D
    variable or one with no vertical grid), ``level`` the index along its first
    non-horizontal dimension (``-1`` for none).
    """
    from ocean_skill import roms

    e, x = index[-2:]
    pairs = {"r": [(e, x)], "u": [(e, x), (e, x + 1)], "v": [(e, x), (e + 1, x)]}[
        v.kind
    ]
    lon = lat = np.nan
    if src.lon is not None and src.lat is not None:
        lon = _mid_lon(src.lon[pairs[0]], src.lon[pairs[-1]])
        lat = 0.5 * (src.lat[pairs[0]] + src.lat[pairs[-1]])
    level = int(index[0]) if len(index) > 2 else -1
    depth = np.nan
    if level >= 0 and src.depth is not None and v.lead[0] == src.depth[5]:
        h, sigma, cs, hc, vtransform, _ = src.depth
        z = [
            float(roms._s_to_z(sigma[level], cs[level], h[p], 0.0, hc, vtransform))
            for p in pairs
        ]
        depth = -float(np.mean(z))
    return float(lon), float(lat), depth, level


# --------------------------------------------------------------------------------------
# Reading and reducing blocks


def _plan(src: _Source, v: _Var, positions: list[int]) -> list[tuple]:
    """Return the blocks ``v`` is read in at ``positions``: ``[(time slice, levels)]``.

    A block is a few time steps of the whole variable when a step is smaller than
    :data:`~ocean_skill.run_health._BLOCK_BYTES`; otherwise one step, in blocks of
    levels (whole storage chunks along the vertical) so that a 3-D field is never held
    whole. ``levels`` is a list of slices, ``[None]`` for no split; the time slice is
    ``None`` for a source with no time axis. Contiguous steps share a block, so a
    thinned window is read a step at a time (a list of indices would be a dask ``take``
    over the whole record).
    """
    da = src.ds[v.std]
    sizes = [da.sizes[d] for d in v.lead]
    item = da.dtype.itemsize * da.sizes[v.ydim] * da.sizes[v.xdim]
    step_bytes = item * math.prod(sizes)
    levels: list[slice | None] = [None]
    per_block = max(1, _BLOCK_BYTES // max(step_bytes, 1))
    if step_bytes > _BLOCK_BYTES and v.lead:
        k = max(1, _BLOCK_BYTES // max(item * math.prod(sizes[1:]), 1))
        if da.chunks is not None:
            c = max(da.chunks[da.dims.index(v.lead[0])])
            k = max(c, k // c * c)
        levels = [slice(i, min(i + k, sizes[0])) for i in range(0, sizes[0], k)]
        per_block = 1
    if src.tdim is None:
        return [(None, levels)]
    blocks, run = [], [positions[0]]
    for p in [*positions[1:], None]:
        if p is not None and p == run[-1] + 1:
            run.append(p)
            continue
        blocks.extend(
            (slice(a, min(a + per_block, run[-1] + 1)), levels)
            for a in range(run[0], run[-1] + 1, per_block)
        )
        run = [p]
    return blocks


def _read(src: _Source, v: _Var, step: slice | None, levels: slice | None):
    """Read a block of ``v`` as numpy, ``(time, [levels], eta, xi)``.

    One small compute on the calling thread.
    """
    da = src.ds[v.std]
    order = [d for d in (src.tdim, *v.lead, v.ydim, v.xdim) if d is not None]
    da = da.transpose(*order)
    select = {}
    if step is not None:
        select[src.tdim] = step
    if levels is not None:
        select[v.lead[0]] = levels
    if select:
        da = da.isel(select)
    data = da.data
    if hasattr(data, "compute"):
        data = data.compute(scheduler="sync")
    x = np.asarray(data)
    return x if src.tdim is not None else x[None]


def _stats(
    x: np.ndarray, ok: np.ndarray, bound: tuple[float, float], offset: int
) -> dict[str, Any]:
    """Return the statistics of one step's block ``x`` over its ocean points ``ok``.

    ``x`` is ``([levels], eta, xi)``; ``ok`` its ocean mask, ``(eta, xi)``. Minimum and
    maximum carry their index in the whole variable (``offset`` levels added).
    """
    low, high = bound
    fin = np.isfinite(x)
    fin &= ok
    n_ok = int(ok.sum()) * (x.size // ok.size)
    n_fin = int(np.count_nonzero(fin))
    n_nan = int(np.count_nonzero(np.isnan(x) & ok)) if n_fin < n_ok else 0
    xs = np.where(fin, x, 0.0).astype(np.float64, copy=False).reshape(-1)
    p: dict[str, Any] = {
        "n_ok": n_ok,
        "n_fin": n_fin,
        "n_nan": n_nan,
        "n_inf": n_ok - n_fin - n_nan,
        "n_neg": int(np.count_nonzero(xs < 0)),
        "n_bad": int(np.count_nonzero(fin & ((x < low) | (x > high)))),
        "n_nonzero": int(np.count_nonzero(xs)),
        "total": float(xs.sum()),
        "sumsq": float(xs @ xs),
    }
    for which, fill, pick in (("min", np.inf, np.argmin), ("max", -np.inf, np.argmax)):
        masked = np.where(fin, x, fill)
        flat = int(pick(masked))
        index = np.unravel_index(flat, x.shape)
        p[which] = float(masked.flat[flat])
        shift = offset if x.ndim > 2 else 0
        p[f"i{which}"] = (int(index[0]) + shift, *map(int, index[1:]))
        del masked
    return p


def _scan_block(src: _Source, v: _Var, job: tuple) -> list[dict]:
    """Read one block of ``v``; return one :func:`_stats` dict per time step in it."""
    step, levels = job
    x = _read(src, v, step, levels)
    ok = src.ocean(v.kind, x.shape[-2:])
    offset = levels.start if levels is not None else 0
    bound = bounds(v.native)
    return [_stats(x[t], ok, bound, offset) for t in range(x.shape[0])]


def _stamp(src: _Source, position: int) -> tuple[Any, int]:
    """Return ``(time, month)`` of the step at ``position``: the other is empty."""
    if src.times is not None:
        return pd.Timestamp(src.times[position]), 0
    if src.months is not None:
        return pd.NaT, int(src.months[position])
    return pd.NaT, 0


def _scan_workers() -> int:
    """Return how many threads scan blocks: dask's worker count, at most the cap.

    ``num_workers`` of dask's config (``DASK_NUM_WORKERS``), else every core, as for
    :func:`ocean_skill.health`, but never more than :data:`_MAX_WORKERS`.
    """
    return max(1, min(_workers(), _MAX_WORKERS))


def _scan_variable(src: _Source, v: _Var) -> pd.DataFrame:
    """Return the per-step statistics of ``v`` over the scanned steps, a row each."""
    plan = _plan(src, v, src.positions)
    jobs = [(step, lev) for step, levels in plan for lev in levels]
    # a store chunked a whole variable at a time (a netCDF-3 file) holds that chunk
    # while a block is read: fewer threads then, so that they still hold about
    # _BLOCK_BYTES each between them
    da = src.ds[v.std]
    chunk = da.dtype.itemsize * math.prod(max(c) for c in da.chunks) if da.chunks else 0
    cap = _scan_workers()
    room = _BLOCK_BYTES * cap
    workers = min(cap, len(jobs), max(1, room // max(chunk, _BLOCK_BYTES)))
    if workers > 1:
        with ThreadPoolExecutor(workers, thread_name_prefix="inputs") as pool:
            results = list(pool.map(lambda job: _scan_block(src, v, job), jobs))
    else:
        results = [_scan_block(src, v, job) for job in jobs]
    rows, at, k = [], iter(src.positions), 0
    for _step, levels in plan:
        blocks = results[k : k + len(levels)]
        k += len(levels)
        for t in range(len(blocks[0])):
            p = _combine([b[t] for b in blocks])
            position = next(at)
            when, month = _stamp(src, position)
            row = dict.fromkeys(_COLUMNS, np.nan)
            n_ok = max(int(p["n_ok"]), 1)
            row.update(
                step=position,
                time=when,
                month=month,
                variable=v.native,
                pct_neg=100.0 * int(p["n_neg"]) / n_ok,
                pct_bad=100.0 * int(p["n_bad"]) / n_ok,
                **{
                    key: int(p[key])
                    for key in (
                        "n_ok",
                        "n_fin",
                        "n_nan",
                        "n_inf",
                        "n_neg",
                        "n_bad",
                        "n_nonzero",
                    )
                },
                total=float(p["total"]),
                sumsq=float(p["sumsq"]),
            )
            if p["n_fin"]:
                for which in ("min", "max"):
                    lon, lat, depth, level = _place(src, v, p[f"i{which}"])
                    row.update(
                        {
                            which: float(p[which]),
                            f"{which}_lon": lon,
                            f"{which}_lat": lat,
                            f"{which}_depth": depth,
                            f"{which}_level": level,
                        }
                    )
            rows.append(row)
    return pd.DataFrame(rows, columns=list(_COLUMNS))


# --------------------------------------------------------------------------------------
# Copies


def _same(src: _Source, a: _Var, b: _Var, position: int) -> bool:
    """Return whether ``a`` and ``b`` agree wherever either is a finite ocean value."""
    step = None if src.tdim is None else slice(position, position + 1)
    for _, levels in _plan(src, a, [position]):
        for lev in levels:
            xa, xb = _read(src, a, step, lev), _read(src, b, step, lev)
            ok = src.ocean(a.kind, xa.shape[-2:])
            either = (np.isfinite(xa) | np.isfinite(xb)) & ok
            if not np.all((xa == xb) | ~either):
                return False
    return True


def _find_copies(src: _Source, frames: dict[str, pd.DataFrame]) -> dict[str, str]:
    """Return ``{variable: the earlier-named variable it equals at every ocean cell}``.

    Cheap first: only variables of one layout whose finite count, minimum, maximum and
    sums agree at *every* scanned step are candidates, and only candidates are read
    again, :data:`_COPY_CHECKS` steps of them, to compare cell by cell. A field that is
    zero everywhere is not a copy of another zero field.
    """
    groups: dict[tuple, list[str]] = {}
    for name, g in frames.items():
        if not int(g["n_nonzero"].sum()) or not int(g["n_fin"].sum()):
            continue
        v = src.variables[name]
        key = (
            v.kind,
            tuple((d, src.ds.sizes[d]) for d in v.lead),
            # NaN (an all-NaN step has no min or max) is not equal to itself: None is
            tuple(
                tuple(None if x != x else x for x in row)
                for row in g[["n_fin", "min", "max", "total", "sumsq"]]
                .to_numpy()
                .tolist()
            ),
        )
        groups.setdefault(key, []).append(name)
    positions = src.positions
    checks = sorted({positions[0], positions[len(positions) // 2], positions[-1]})
    copies: dict[str, str] = {}
    for names in groups.values():
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                if b not in copies and all(
                    _same(src, src.variables[a], src.variables[b], p)
                    for p in checks[:_COPY_CHECKS]
                ):
                    copies[b] = a
    return copies


# --------------------------------------------------------------------------------------
# The cache


def _sample_hash(src: _Source, v: _Var) -> str:
    """Return a hash of a small content sample of ``v``: a rewritten file misses.

    A :data:`_SAMPLE_EDGE`-cell window at each end of the field (the low and the high
    corner) at the first, middle and last scanned step and the first and last level of a
    3-D field. The windows are computed together, so a store chunked a whole variable at
    a time is read once for all of them, and one chunked by step or level only reads the
    chunks the windows touch.
    """
    import dask

    da = src.ds[v.std]
    pos = src.positions
    edge = slice(0, _SAMPLE_EDGE), slice(-_SAMPLE_EDGE, None)
    pieces = []
    for p in sorted({pos[0], pos[len(pos) // 2], pos[-1]}):
        for lev in sorted({0, -1}) if v.lead else (None,):
            for window in edge:
                select: dict[str, Any] = {v.ydim: window, v.xdim: window}
                if src.tdim is not None:
                    select[src.tdim] = p
                if v.lead:
                    select[v.lead[0]] = lev
                    select.update(dict.fromkeys(v.lead[1:], 0))
                pieces.append(da.isel(select).data)
    h = hashlib.sha256()
    for piece in dask.compute(*pieces, scheduler="sync"):
        h.update(np.ascontiguousarray(piece))
    return h.hexdigest()


def _file_stamp(src: _Source) -> list:
    """Return the size and modification time of the file the source was opened from.

    Empty where the Dataset does not say (a reference, a list of files): the content
    sample then stands alone.
    """
    import os

    path = src.ds.encoding.get("source")
    try:
        stat = os.stat(path)
    except (TypeError, OSError):
        return []
    return [stat.st_size, stat.st_mtime_ns]


def _key(src: _Source, v: _Var, fingerprint: str) -> str:
    """Return one variable's cache key: version, definition, steps, bound and sample."""
    steps = hashlib.sha256(json.dumps(src.positions).encode()).hexdigest()
    payload = json.dumps(
        {
            "v": INPUT_VERSION,
            "source": src.name,
            "definition": fingerprint,
            "variable": v.native,
            "units": v.units,
            "bound": bounds(v.native),  # the counts of out-of-bounds cells depend on it
            "steps": steps,
            "mask": None if src.mask is None else int(src.mask.sum()),
            "file": _file_stamp(src),
            "sample": _sample_hash(src, v),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _to_dataset(frame: pd.DataFrame):
    return frame.reset_index(drop=True).to_xarray()


def _from_dataset(ds) -> pd.DataFrame:
    frame = ds.to_dataframe().reset_index(drop=True)
    frame["variable"] = frame["variable"].astype(str)
    frame["time"] = pd.to_datetime(frame["time"]).astype("datetime64[ns]")
    return frame[list(_COLUMNS)]


# --------------------------------------------------------------------------------------
# The public entry point


def normalize_sources(sources: Any) -> list[dict]:
    """Return ``sources`` as a flat list of ``{"name", "label"?, "vars"?}`` dicts.

    Each element is a catalog name or a mapping with ``name`` and optionally ``label``
    and ``vars`` (``var`` for one); nested lists (what a ``"{surface}"`` placeholder
    expands to inside a list) are flattened. A name given twice is one source: the
    first label given wins, and the variables are the union of the lists given, or
    every variable if any of the entries names none.
    """
    out: dict[str, dict] = {}
    open_ended: set[str] = set()  # sources some entry did not restrict

    def visit(item: Any) -> None:
        if isinstance(item, str):
            name, label, names = item, None, None
        elif isinstance(item, Mapping):
            if "name" not in item:
                raise ValueError(
                    f"input_health: a source mapping needs a 'name'; got "
                    f"{dict(item)!r}."
                )
            name = str(item["name"])
            label = str(item["label"]) if item.get("label") else None
            names = item.get("vars", item.get("var")) or None
            if isinstance(names, str):
                names = [names]
        elif isinstance(item, Sequence):
            for sub in item:
                visit(sub)
            return
        else:
            raise ValueError(
                f"input_health: a source is a catalog name or a mapping, got {item!r}."
            )
        entry = out.setdefault(name, {"name": name})
        if label and "label" not in entry:
            entry["label"] = label
        if names:
            entry["vars"] = [*entry.get("vars", []), *names]
        else:
            open_ended.add(name)

    visit(sources)
    if not out:
        raise ValueError("input_health needs at least one source.")
    for name, entry in out.items():
        if name in open_ended:
            entry.pop("vars", None)
        elif "vars" in entry:
            entry["vars"] = list(dict.fromkeys(entry["vars"]))
    return list(out.values())


def input_health(
    sources,
    *,
    time=None,
    max_steps: int | None = None,
    cache: bool = True,
) -> InputHealth:
    """Check the input files of a run: ranges, bounds, copies, constants, NaN.

    One streaming pass over every catalog source in ``sources``, one block of time steps
    (levels) at a time; see the module docstring for what is recorded. Returns an
    :class:`InputHealth` to read as a table, problems or report figures.

    Parameters
    ----------
    sources
        A catalog name, or a list of them; an element may be a mapping ``{"name": ...,
        "label": ..., "vars": [...]}`` (``label`` names the source on a figure, ``vars``
        restricts the variables scanned). Nested lists are flattened.
    time
        Which steps to scan: ``None`` (all), ``"latest"``, a date string (``"2011-01"``,
        a month; a day; a year), or a ``{"min": ..., "max": ...}`` dict. Ignored by a
        climatology (``month`` axis), which is always scanned whole.
    max_steps
        Thin each source's window to at most this many evenly spread steps (the first
        and last kept). Under ``ocean-skill-run --smoke`` it is at most twice the
        steps ``--smoke`` keeps, whatever it is set to, and every figure says test mode.
    cache
        Keep each variable's table in the ``inputs`` cache and reuse it (default).
        ``False`` rescans everything and stores nothing; so does a disabled cache.
    """
    from ocean_skill import cache as _cache
    from ocean_skill import catalog
    from ocean_skill import sources as _sources

    entries = normalize_sources(sources)
    test_mode = bool(_sources._TRUNCATE)
    if test_mode:
        smoke = next(iter(_sources._TRUNCATE.values()))[0]
        max_steps = min(max_steps or 2 * smoke, 2 * smoke)
    use_cache = bool(cache) and _cache.enabled()
    frames: list[pd.DataFrame] = []
    infos: list[dict] = []
    for entry in entries:
        src = _open_source(entry, time, max_steps)
        fingerprint = catalog.fingerprint(src.name) if use_cache else ""
        per_var: dict[str, pd.DataFrame] = {}
        for name, v in src.variables.items():
            key = _key(src, v, fingerprint) if use_cache else None
            hit = _cache.load(key, kind="inputs") if key else None
            if hit is not None:
                per_var[name] = _from_dataset(hit)
                continue
            per_var[name] = _scan_variable(src, v)
            if key is not None:
                _cache.save(key, _to_dataset(per_var[name]), kind="inputs")
        copies = _find_copies(src, per_var)
        for name, frame in per_var.items():
            frame.insert(0, "source", src.name)
            frame["copy_of"] = copies.get(name, "")
            frames.append(frame)
        infos.append(
            {
                "name": src.name,
                "label": src.label,
                "units": {n: v.units for n, v in src.variables.items()},
                "no_mask": src.mask is None,
                "thinned": src.thinned,
                "climatology": src.months is not None,
            }
        )
    steps = pd.concat(frames, ignore_index=True)
    return InputHealth(steps, infos, test_mode=test_mode)


# --------------------------------------------------------------------------------------
# The result


def _when(t, intraday: bool) -> str:
    """Return a step's time as a date (and hour); ``""`` for a climatology's month."""
    if pd.isna(t):
        return ""
    return t.strftime("%Y-%m-%d %H:%M") if intraday else t.strftime("%Y-%m-%d")


class InputHealth:
    """The result of :func:`input_health`: per-step statistics, and what they say.

    Attributes
    ----------
    steps
        :class:`pandas.DataFrame`, one row per (source, variable, step): the counts of
        NaN/Inf, negative and out-of-bounds ocean cells and their percentages, the
        minimum and maximum with longitude, latitude, depth and level, the sum and sum
        of squares, and the variable it is an exact copy of (``copy_of``).
    sources
        One dict per source, in the order given: ``name``, ``label``, ``units`` (per
        variable), ``no_mask`` (no ``mask_rho``: every cell was counted), ``thinned``
        and ``climatology``.
    test_mode
        Whether the run is under ``--smoke``; every figure then says so.
    """

    def __init__(
        self, steps: pd.DataFrame, sources: list[dict], *, test_mode: bool = False
    ) -> None:
        self.steps = steps
        self.sources = sources
        self.test_mode = test_mode
        self._table: pd.DataFrame | None = None
        self._problems: list[dict] | None = None

    def __repr__(self) -> str:
        return (
            f"InputHealth({len(self.sources)} source(s), "
            f"{self.steps['variable'].nunique()} variable names, "
            f"{len(self.steps)} scanned steps)"
        )

    # ---- tables ------------------------------------------------------------------

    def table(self) -> pd.DataFrame:
        """Return the integrity table: a row per source and variable, over its steps.

        Columns: ``source``, ``label``, ``variable``, ``units``, ``bound`` (text),
        ``n_steps``, ``first_time`` and ``last_time`` (text; ``"month 1"`` ... for a
        climatology), ``worst_pct_bad`` (percent of ocean cells outside the bound at
        the worst step, at ``worst_bad_time``) and ``worst_pct_neg``/``worst_neg_time``,
        the run-wide ``min`` and ``max`` with ``_lon``, ``_lat``, ``_depth``, ``_level``
        and ``_time``, ``mean`` and ``std`` over every ocean cell and step,
        ``n_nonfinite`` (NaN plus Inf ocean cells at the worst step, at
        ``nonfinite_time``), ``all_nan_steps``, ``zero_steps`` (steps all zero),
        ``is_constant``, ``n_repeats``, ``copy_of`` and ``flags`` (those, as text).
        """
        if self._table is None:
            self._table = self._build_table()
        return self._table

    def _build_table(self) -> pd.DataFrame:
        info = {s["name"]: s for s in self.sources}
        rows = []
        for (source, variable), g in self.steps.groupby(
            ["source", "variable"], sort=False
        ):
            g = g.sort_values("step")
            meta = info[source]
            intraday = bool(
                (g["time"].notna() & (g["time"] != g["time"].dt.normalize())).any()
            )

            def label(r, _intraday=intraday):
                t = _when(r["time"], _intraday)
                return t or (f"month {int(r['month'])}" if r["month"] else "")

            first, last = g.iloc[0], g.iloc[-1]
            neg, bad = g.loc[g["pct_neg"].idxmax()], g.loc[g["pct_bad"].idxmax()]
            n_fin = int(g["n_fin"].sum())
            mean = float(g["total"].sum()) / n_fin if n_fin else np.nan
            spread = float(g["sumsq"].sum()) / n_fin - mean**2 if n_fin else np.nan
            nonfinite = g["n_nan"] + g["n_inf"]
            row: dict[str, Any] = {
                "source": source,
                "label": meta["label"],
                "variable": variable,
                "units": meta["units"].get(variable, ""),
                "bound": _bound_text(variable),
                "n_steps": len(g),
                "first_time": label(first),
                "last_time": label(last),
                "worst_pct_neg": float(neg["pct_neg"]),
                "worst_neg_time": label(neg),
                "worst_pct_bad": float(bad["pct_bad"]),
                "worst_bad_time": label(bad),
                "mean": mean,
                "std": math.sqrt(max(spread, 0.0)) if n_fin else np.nan,
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
                            f"{which}_level": int(pick[f"{which}_level"]),
                            f"{which}_time": label(pick),
                        }
                    )
                else:
                    row.update(
                        {
                            f"{which}{k}": np.nan if k != "_time" else ""
                            for k in ("", "_lon", "_lat", "_depth", "_level", "_time")
                        }
                    )
            row["n_nonfinite"] = int(nonfinite.max())
            row["nonfinite_time"] = label(g.loc[nonfinite.idxmax()])
            row["all_nan_steps"] = int(((g["n_ok"] > 0) & (g["n_fin"] == 0)).sum())
            lo, hi = g["min"].min(), g["max"].max()
            row["zero_steps"] = int(((g["n_fin"] > 0) & (g["n_nonzero"] == 0)).sum())
            row["is_constant"] = bool(lo == hi and lo != 0 and not np.isnan(lo))
            row["n_repeats"] = int(self._repeats(g).sum())
            row["copy_of"] = str(g["copy_of"].iloc[0])
            row["flags"] = self._flags(g, row)
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def _repeats(g: pd.DataFrame) -> pd.Series:
        """Per step of one variable: identical to the previous step.

        True where the count, minimum, maximum, sum and sum of squares of a field that
        is not constant all equal the previous step's exactly: the same step written
        twice, a forcing field that stopped changing.
        """
        keys = g[["n_fin", "min", "max", "total", "sumsq"]]
        same = (keys == keys.shift()).all(axis=1) & (g["min"] != g["max"])
        same.iloc[0:1] = False
        return same

    @staticmethod
    def _flags(g: pd.DataFrame, row: dict) -> str:
        n = len(g)
        flags = []
        nan = row["all_nan_steps"]
        if nan:
            flags.append("all NaN" if nan == n else f"all NaN in {nan}/{n}")
        if not int(g["n_ok"].max()):
            flags.append("no ocean cells")
        zero = row["zero_steps"]
        if zero:
            flags.append("all zero" if zero == n else f"all zero in {zero}/{n}")
        if row["is_constant"]:
            flags.append("constant")
        if row["copy_of"]:
            flags.append(f"copy of {row['copy_of']}")
        if row["n_repeats"]:
            flags.append(f"same as previous step x{row['n_repeats']}")
        return "; ".join(flags)

    # ---- problems ----------------------------------------------------------------

    def problems(self) -> list[dict]:
        """Return what is wrong, most severe first, as JSON-safe dicts.

        Each has ``kind`` (``nonfinite``, ``all_nan``, ``negative``, ``out_of_bounds``,
        ``copy``, ``all_zero``, ``constant`` or ``repeat``), ``source``, ``variable``, a
        plain-words ``message`` and the numbers behind it. A variable negative or out of
        bounds in under 0.1% of its ocean cells at every step is in :meth:`table` only.
        """
        if self._problems is None:
            found = []
            for r in self.table().itertuples(index=False):
                found.extend(self._row_problems(r))
            found.sort(
                key=lambda p: (_SEVERITY.index(p["kind"]), -p.get("magnitude", 0.0))
            )
            self._problems = found
        return self._problems

    @staticmethod
    def _row_problems(r) -> list[dict]:
        out = []
        where = f"{r.variable} of {r.source}"

        def add(kind: str, message: str, **numbers: Any) -> None:
            out.append(
                {
                    "kind": kind,
                    "source": r.source,
                    "variable": r.variable,
                    "message": message,
                    **numbers,
                }
            )

        if r.all_nan_steps == r.n_steps:
            add(
                "all_nan",
                f"{where} is NaN in every ocean cell, always.",
                magnitude=float(r.n_nonfinite),
            )
            return out
        if r.n_nonfinite:
            add(
                "nonfinite",
                f"{where} has {r.n_nonfinite} NaN or Inf ocean cells at its worst step"
                f" ({r.nonfinite_time}).",
                count=int(r.n_nonfinite),
                time=r.nonfinite_time,
                magnitude=float(r.n_nonfinite),
            )
        low = bounds(r.variable)[0]
        if r.worst_pct_bad > _PROBLEM_PCT:
            negative = low == 0 and r.worst_pct_neg >= r.worst_pct_bad
            what = "negative" if negative else "outside its bounds"
            add(
                "negative" if negative else "out_of_bounds",
                f"{where} is {what} ({r.bound}) in {format_value(r.worst_pct_bad)}% of"
                f" ocean cells at worst ({r.worst_bad_time}), range"
                f" {format_value(r.min)} to {format_value(r.max)}.",
                worst_pct=float(r.worst_pct_bad),
                time=r.worst_bad_time,
                magnitude=float(r.worst_pct_bad),
            )
        if r.copy_of:
            add(
                "copy",
                f"{where} is identical to {r.copy_of} at every ocean cell.",
                copy_of=r.copy_of,
                magnitude=1.0,
            )
        if r.zero_steps:
            add(
                "all_zero",
                f"{where} is zero everywhere, "
                + (
                    "always."
                    if r.zero_steps == r.n_steps
                    else f"in {r.zero_steps}/{r.n_steps} steps."
                ),
                steps=int(r.zero_steps),
                magnitude=1.0,
            )
        if r.is_constant:
            add(
                "constant",
                f"{where} has one value, {format_value(r.min)}, at every ocean cell and"
                " step.",
                value=float(r.min),
                magnitude=1.0,
            )
        if r.n_repeats:
            add(
                "repeat",
                f"{where} is identical to the previous step in {r.n_repeats} "
                f"step{'s' if r.n_repeats != 1 else ''}.",
                steps=int(r.n_repeats),
                magnitude=float(r.n_repeats),
            )
        return out

    # ---- figures -----------------------------------------------------------------

    def figures(self, draw: str = "integrity") -> list[tuple[str, Any]]:
        """Draw one report page: ``(suffix, Figure)`` pairs, ``""`` then ``"-2"``, ...

        ``draw`` is one of :data:`INPUT_DRAWS`: ``"integrity"``, the table of every
        variable's range, bound, mean, spread and flags, continued on further figures
        when it is longer than a page.
        """
        if draw not in INPUT_DRAWS:
            raise ValueError(
                f"unknown input_health draw {draw!r}; choose from {list(INPUT_DRAWS)}."
            )
        pages = getattr(self, f"_draw_{draw}")()
        return [("" if i == 0 else f"-{i + 1}", fig) for i, fig in enumerate(pages)]

    def _title(self) -> str:
        labels = [s["label"] for s in self.sources]
        steps = self.table().groupby("source")["n_steps"].max()
        first, last = self.table()["first_time"], self.table()["last_time"]
        if len(labels) == 1:
            n = int(steps.iloc[0])
            span = f"{n} step{'s' if n != 1 else ''}"
            if n > 1 and first.iloc[0] and last.iloc[0]:
                span += f", {first.iloc[0]} to {last.iloc[0]}"
            elif first.iloc[0]:
                span += f", {first.iloc[0]}"
            name = self.sources[0]["name"]
            named = labels[0] if labels[0] == name else f"{labels[0]} ({name})"
            return f"Input integrity: {named}\n{span}"
        return f"Input integrity: {len(labels)} sources\n" + ", ".join(labels)

    def _notes(self) -> list[str]:
        notes = [
            "Statistics are over ocean cells only. Bound: the plausible range; % bad "
            "is the share of ocean cells outside it at the worst step.",
            "Minimum and maximum: value, longitude and latitude, depth (or s-level), "
            "date. NaN/Inf: ocean cells at the worst step.",
        ]
        bare = [s["label"] for s in self.sources if s["no_mask"]]
        if bare:
            notes.append(
                f"No ocean mask for {', '.join(bare)}: every cell was counted."
            )
        thin = [s["label"] for s in self.sources if s["thinned"]]
        if thin:
            notes.append(f"Steps evenly thinned for {', '.join(thin)}.")
        return notes

    def _draw_integrity(self) -> list:
        from matplotlib.figure import Figure

        from ocean_skill.plot.typography import PAGE_W

        table = self.table()
        several = table["source"].nunique() > 1
        head = [
            *(("Source",) if several else ()),
            "Variable",
            "Bound",
            "Worst %\nbad (date)",
            "Minimum",
            "Maximum",
            "Mean",
            "Std",
            "NaN/Inf",
            "Steps",
            "Flags",
        ]
        rows = []
        for r in table.itertuples(index=False):
            unit = display(r.units) if r.units else ""
            rows.append(
                [
                    *((r.label,) if several else ()),
                    f"{r.variable}\n{unit}" if unit else r.variable,
                    r.bound,
                    f"{format_value(r.worst_pct_bad)}\n{r.worst_bad_time}"
                    if r.worst_pct_bad
                    else "0",
                    _extreme(r, "min"),
                    _extreme(r, "max"),
                    _minus(r.mean),
                    format_value(r.std),
                    format_value(int(r.n_nonfinite)),
                    f"{r.n_steps}\n{r.first_time}\n{r.last_time}"
                    if r.n_steps > 1
                    else f"1\n{r.first_time}",
                    r.flags,
                ]
            )
        centered = tuple(
            i + int(several) for i in (1, 2, 3, 4, 5, 6, 7, 8)
        )  # every column but variable, source and flags
        flexible = len(head) - 1
        size = _INTEGRITY_PT
        while True:  # the font shrinks only if the columns cannot fit at this one
            cells, widths, heights = _lay_out(
                [head, *rows], PAGE_W - 0.6, size, flexible=flexible, min_flexible=0.9
            )
            if sum(widths) <= PAGE_W - 0.6 + 1e-9 or size <= 5.0:
                break
            size -= 0.5
        notes = self._notes()
        note_in = len(notes) * size * 1.5 / 72.0 + 0.1
        top = _TITLE_IN + (0.3 if self.test_mode else 0.0)
        budget = _FIG_MAX_H - top - 0.25 - note_in
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
            height = top + sum(hs) + 0.25 + note_in
            fig = Figure(figsize=(PAGE_W, height))
            _draw_title(
                fig, self._title() + (" (cont.)" if part else ""), self.test_mode
            )
            _draw_grid(
                fig,
                0.3,
                height - top,
                widths,
                hs,
                [cells[0], *(row for row, _ in chunk)],
                size,
                centered=centered,
            )
            y = 0.2 + note_in - 0.1
            for line in notes:
                fig.text(
                    0.3 / PAGE_W,
                    y / height,
                    line,
                    ha="left",
                    va="top",
                    fontsize=size,
                    color="0.3",
                )
                y -= size * 1.5 / 72.0
            figures.append(fig)
        return figures


def _extreme(r, which: str) -> str:
    """Return the table cell of a run-wide minimum or maximum: value, place, date."""
    value = getattr(r, which)
    if np.isnan(value):
        return "–"
    lon, lat = getattr(r, f"{which}_lon"), getattr(r, f"{which}_lat")
    depth, level = getattr(r, f"{which}_depth"), getattr(r, f"{which}_level")
    place = _lonlat(lon, lat) if not np.isnan(lon) else ""
    if not np.isnan(depth):
        place += f", {_meters(depth)}"
    elif level >= 0:
        place += f", level {level}"
    when = getattr(r, f"{which}_time")
    return "\n".join(line for line in (_minus(value), place.strip(", "), when) if line)
