"""Check a roms-tools river-forcing file: where the rivers are, and whether it is sane.

A river-forcing file (:func:`ocean_skill.build.add_river_source`) is indexed by river,
not by grid cell: ``river_volume`` is ``(river_time, nriver)`` discharge in m3/s and
``river_tracer`` is ``(river_time, ntracers, nriver)`` -- for a Pacific domain
1829 x 34 x 49,411 values, 12 GB, and the same shape with 100,000 rivers is more again.
:func:`rivers` returns a :class:`RiverInputs` that **never loads ``river_tracer``
whole**: it reads it in slabs of a few time steps (:data:`SLAB_ELEMENTS` values at a
time), keeping only what each figure needs -- per-river maxima and minima with the step
they happened at, counts outside :data:`RIVER_BOUNDS` -- and ``river_volume`` is
streamed the same way for the time-mean discharge that ranks the rivers.

Real files are mostly rivers with no water or no grid cell, and the zero-flow ones carry
absurd concentrations. :meth:`RiverInputs.classes` sorts every river once -- **active**
(a mouth and a positive time-mean discharge), mouth but no flow, flow but no grid cell
(no cell of its own in river_index: reported, to be checked), neither -- and every
concentration view uses active rivers only, at the steps where they actually flow.

Four views (:data:`RIVER_DRAWS`), each a list of ``(name, matplotlib Figure)``:

``map``
    The river mouths over the model domain, sized and colored by time-mean discharge.
``histograms``
    Log-scale histograms of discharge and of every tracer that varies, at the forcing
    time, with the count outside the plausible range printed on a panel that has any;
    all-zero and constant tracers are listed in a note instead of drawn.
``sample``
    Twelve rivers chosen by discharge (three small, three medium, three large, the
    three largest), their discharge and the main concentrations that vary over time.
``extremes``
    A table of each varying main tracer's three highest river concentrations over *all*
    times, and any below its lower bound (:meth:`RiverInputs.table`).

:meth:`RiverInputs.problems` lists what is out of range as JSON-safe records, for a
suite's summary. Everything is matplotlib, built directly (no ``pyplot`` state).

A river's **mouth** is the :data:`river_fraction`-weighted mean position of the grid
cells ``river_index`` gives its ID (a river may enter at several cells), averaged on the
circle so a river at the dateline is not placed at longitude 0.
"""

from __future__ import annotations

import math
import re
import textwrap
from typing import Any

import numpy as np

from ocean_skill.config import RIVER_DRAWS
from ocean_skill.plot._geo_labels import format_lat, format_lon
from ocean_skill.plot._numbers import format_value
from ocean_skill.units import display

__all__ = [
    "MAIN_TRACERS",
    "RIVER_BOUNDS",
    "RIVER_DRAWS",
    "SAMPLE_TRACERS",
    "SLAB_ELEMENTS",
    "RiverInputs",
    "river_mouths",
    "rivers",
]

#: Plausible ``(lower, upper)`` concentration of a river's water, per tracer name, in
#: the units the file states (``tracer_unit``: temperature in degC, everything else in
#: mmol/m3 or the MARBL equivalent). A value outside is *suspicious*, not proven wrong:
#: the upper bounds are generous (several times what a polluted river carries), so
#: crossing one means a unit slip or a corrupt value rather than a dirty river.
#:
#: * ``temp`` -2.5..40 degC: below freezing-point seawater or hotter than a desert
#:   river is wrong.
#: * ``salt`` 0..1: river water is fresh (an estuary mouth is a few PSU, but a river
#:   *input* above 1 is flagged to be looked at).
#: * ``ALK`` / ``DIC`` 0..6000 mmol/m3: hard-water rivers reach 3000-4000; seawater
#:   is ~2300, so 6000 is a unit slip.
#: * ``DOC`` 0..2000, ``DON`` 0..200, ``DOP`` 0..20 mmol/m3: blackwater rivers carry
#:   a few hundred to ~1500 mmol/m3 DOC at the extreme.
#: * ``NO3`` / ``NH4`` 0..500, ``PO4`` 0..50, ``SiO3`` 0..1000, ``Fe`` 0..100, ``Lig``
#:   0..100, ``O2`` 0..600 mmol/m3: heavily polluted rivers carry ~100-300 nitrate;
#:   river water is near saturation (~250-450).
#:
#: Every concentration of any other tracer is bounded below by zero only. The
#: ``(0, inf)`` default is :data:`_DEFAULT_BOUND`.
RIVER_BOUNDS: dict[str, tuple[float, float]] = {
    "temp": (-2.5, 40.0),
    "salt": (0.0, 1.0),
    "NO3": (0.0, 500.0),
    "NH4": (0.0, 500.0),
    "PO4": (0.0, 50.0),
    "SiO3": (0.0, 1000.0),
    "Fe": (0.0, 100.0),
    "Lig": (0.0, 100.0),
    "O2": (0.0, 600.0),
    "ALK": (0.0, 6000.0),
    "ALK_ALT_CO2": (0.0, 6000.0),
    "DIC": (0.0, 6000.0),
    "DIC_ALT_CO2": (0.0, 6000.0),
    "DOC": (0.0, 2000.0),
    "DOCr": (0.0, 2000.0),
    "DON": (0.0, 200.0),
    "DONr": (0.0, 200.0),
    "DOP": (0.0, 20.0),
    "DOPr": (0.0, 20.0),
}
_DEFAULT_BOUND = (0.0, math.inf)

#: The tracers the extremes table lists (those the file has, in this order).
MAIN_TRACERS = (
    "temp",
    "salt",
    "NO3",
    "PO4",
    "SiO3",
    "NH4",
    "Fe",
    "O2",
    "ALK",
    "DIC",
    "DOC",
)

#: The tracers the sample figure draws over time (those the file has).
SAMPLE_TRACERS = ("temp", "salt", "NO3", "ALK", "DIC", "DOC")

#: How many values of ``river_tracer`` (or ``river_volume``) one read may hold: a slab
#: of a few time steps, ~64 MB as float64. The whole PACMED tracer array is 12 GB.
SLAB_ELEMENTS = 8_000_000

#: How many of the highest values per tracer the extremes table lists.
TOP_N = 3

#: Figure fonts (pt) and the letter page's drawable width (in).
_FS = 8.0
_FS_TITLE = 10.0
_WIDTH = 8.5
_TABLE_ROWS_PER_PAGE = 40
_HIST_COLS, _HIST_ROWS = 3, 4
#: Characters per line of a wrapped note at ``_FS`` pt on the letter page's width.
_WRAP = 105

_UNIT_GUESS = {"temp": "degC", "salt": "PSU"}


def rivers(source: str, *, time=None) -> RiverInputs:
    """Open the river-forcing entry ``source`` for inspection.

    Parameters
    ----------
    source
        A catalog nickname made with :func:`ocean_skill.build.add_river_source`.
    time
        The forcing time to show in the histograms: a date string (the nearest time
        step is used) or ``None`` for the first step.
    """
    from ocean_skill import sources

    ref = sources.resolve(source)
    grid = (ref.metadata or {}).get("river_grid")
    if not grid:
        raise ValueError(
            f"{source!r} has no 'river_grid' metadata: catalogue the river file with "
            "ocean_skill.build.add_river_source(cat, name, files, grid)."
        )
    return RiverInputs(source, sources.read(source), str(grid), time=time)


def river_mouths(river_index, river_fraction, lon, lat, ids):
    """Return ``(lon, lat)`` of each river in ``ids``: the mean of its cells.

    ``river_index`` holds each cell's river ID (0 or NaN where no river enters),
    ``river_fraction`` the share of that river's volume entering at the cell, ``lon`` /
    ``lat`` the cell positions (all ``(eta, xi)``), ``ids`` the river IDs in file
    order. Longitudes are averaged on the circle and returned in the grid's own
    convention (0..360 when the grid has values above 180, else -180..180), so a river
    whose cells straddle the dateline sits there. A river with no cell is NaN; one
    whose cells all carry zero or missing fractions is the unweighted mean of them.
    """
    idx = (
        np.nan_to_num(np.asarray(river_index, dtype="float64")).astype(np.int64).ravel()
    )
    frac = np.nan_to_num(np.asarray(river_fraction, dtype="float64")).ravel()
    lon = np.asarray(lon, dtype="float64").ravel()
    lat = np.asarray(lat, dtype="float64").ravel()
    ids = np.asarray(ids, dtype=np.int64)
    n = len(ids)
    order = np.argsort(ids, kind="stable")
    where = np.searchsorted(ids[order], idx)
    where = np.clip(where, 0, max(n - 1, 0))
    valid = (idx > 0) & (n > 0)
    if n:
        valid &= ids[order][where] == idx
    pos = order[where[valid]]
    w = np.maximum(frac[valid], 0.0)
    total = np.bincount(pos, weights=w, minlength=n)
    w = np.where(total[pos] > 0, w, 1.0)
    sw = np.bincount(pos, weights=w, minlength=n)
    rad = np.radians(lon[valid])
    sx = np.bincount(pos, weights=w * np.cos(rad), minlength=n)
    sy = np.bincount(pos, weights=w * np.sin(rad), minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_lat = np.bincount(pos, weights=w * lat[valid], minlength=n) / sw
    mean_lon = np.degrees(np.arctan2(sy, sx))
    wrap360 = np.nanmax(lon) > 180.0 if lon.size else False
    mean_lon = (
        np.mod(mean_lon, 360.0) if wrap360 else (mean_lon + 180.0) % 360.0 - 180.0
    )
    none = sw == 0
    mean_lon[none] = np.nan
    mean_lat[none] = np.nan
    return mean_lon, mean_lat


def _text(values) -> list[str]:
    """Return a string variable's values as plain ``str`` (bytes decoded)."""
    return [v.decode() if isinstance(v, bytes) else str(v) for v in np.asarray(values)]


def _decode_times(ds) -> np.ndarray | None:
    """Return the file's absolute times as ``datetime64[ns]``, or ``None``.

    ``abs_time`` ("days since <date>") is the absolute axis; ``river_time`` is relative
    to another epoch named only in its ``long_name``, so it is the fallback.
    """
    import xarray as xr

    for name in ("abs_time", "river_time"):
        if name not in ds.variables:
            continue
        var = ds[name]
        values = np.asarray(var.values)
        if values.dtype.kind == "M":
            return values.astype("datetime64[ns]")
        units = str(var.attrs.get("units", ""))
        if " since " not in units:
            match = re.search(
                r"(\w+)\s+since\s+([\d-]+(?:[ T][\d:]+)?)",
                str(var.attrs.get("long_name", "")),
            )
            if not match:
                continue
            units = f"{match.group(1)} since {match.group(2)}"
        try:
            return np.asarray(
                xr.coding.times.decode_cf_datetime(values, units),
                dtype="datetime64[ns]",
            )
        except (ValueError, OverflowError):
            continue
    return None


def _wrap(text: str, width: int = _WRAP) -> str:
    """Wrap ``text`` to ``width`` characters a line, so a note stays inside the page."""
    return "\n".join(textwrap.wrap(text, width=width)) if text else ""


def _bound(name: str) -> tuple[float, float]:
    return RIVER_BOUNDS.get(name, _DEFAULT_BOUND)


class RiverInputs:
    """A river-forcing entry, ready to draw and check (see the module docstring).

    Built by :func:`rivers`. Nothing heavy happens until a view is asked for: the
    mouths (grid file and ``river_index``) are found on construction, the discharge
    and the tracer scans on first use, each once.
    """

    def __init__(self, source: str, ds, grid: str, *, time=None):
        import pandas as pd
        import xarray as xr

        self.source = source
        self._ds = ds
        for need in ("river_volume", "river_tracer", "river_index", "river_fraction"):
            if need not in ds.variables:
                raise ValueError(f"{source!r}: the river file has no {need!r}.")
        self.n_rivers = int(ds.sizes["nriver"])
        self.n_times = int(ds.sizes["river_time"])
        self.ids = (
            np.asarray(ds["nriver"].values, dtype=np.int64)
            if "nriver" in ds.variables
            else np.arange(1, self.n_rivers + 1)
        )
        if "river_name" in ds.variables:
            self.names = _text(ds["river_name"].values)
        else:
            self.names = [f"river {i}" for i in self.ids]
        n_tr = int(ds.sizes["ntracers"])
        self.tracers = (
            _text(ds["tracer_name"].values)
            if "tracer_name" in ds.variables
            else [f"tracer{i}" for i in range(n_tr)]
        )
        self.units = (
            _text(ds["tracer_unit"].values)
            if "tracer_unit" in ds.variables
            else [_UNIT_GUESS.get(t, "mmol/m3") for t in self.tracers]
        )
        self.times = _decode_times(ds)
        if time is None:
            self.time_index = 0
        elif self.times is None:
            raise ValueError(
                f"{source!r}: the river file's times cannot be decoded, so "
                f"time={time!r} cannot be matched."
            )
        else:
            want = np.datetime64(pd.Timestamp(time), "ns")
            self.time_index = int(np.argmin(np.abs(self.times - want)))

        with xr.open_dataset(grid, decode_times=False) as g:
            glon = np.asarray(g["lon_rho"].values, dtype="float64")
            glat = np.asarray(g["lat_rho"].values, dtype="float64")
        self._grid_lon, self._grid_lat = glon, glat
        self.lon, self.lat = river_mouths(
            ds["river_index"].values, ds["river_fraction"].values, glon, glat, self.ids
        )
        self._q: dict[str, Any] | None = None
        self._classes: dict[str, Any] | None = None
        self._scan_cache: dict[str, Any] | None = None
        self._identical: list[tuple[int, int]] | None = None

    # -- reading -----------------------------------------------------------------

    def _slab_steps(self, per_step: int) -> int:
        return max(1, SLAB_ELEMENTS // max(per_step, 1))

    def _read(self, name: str, **index):
        """One bounded read of ``river_volume`` / ``river_tracer``, as ``float64``."""
        dims = (
            ("river_time", "ntracers", "nriver")
            if name == "river_tracer"
            else ("river_time", "nriver")
        )
        part = self._ds[name].isel(**index)
        return np.asarray(
            part.transpose(*[d for d in dims if d in part.dims]).values, dtype="float64"
        )

    def _tracer_slab(self, t0: int, t1: int):
        """``river_tracer`` for times ``t0:t1`` as ``(time, ntracers, nriver)``."""
        return self._read("river_tracer", river_time=slice(t0, t1))

    def _volume_slab(self, t0: int, t1: int):
        """``river_volume`` for times ``t0:t1`` as ``(time, nriver)``."""
        return self._read("river_volume", river_time=slice(t0, t1))

    # -- scans ---------------------------------------------------------------------

    def _discharge(self) -> dict[str, Any]:
        """Time-mean, min, max, NaN and negative counts of ``river_volume``, once."""
        if self._q is not None:
            return self._q
        n = self.n_rivers
        total = np.zeros(n)
        count = np.zeros(n)
        lo = np.full(n, np.inf)
        hi = np.full(n, -np.inf)
        n_nan = n_neg = 0
        rivers_neg = np.zeros(n, dtype=bool)
        step = self._slab_steps(n)
        for t0 in range(0, self.n_times, step):
            a = self._volume_slab(t0, t0 + step)
            ok = ~np.isnan(a)
            n_nan += int(a.size - ok.sum())
            neg = ok & (a < 0)
            n_neg += int(neg.sum())
            rivers_neg |= neg.any(axis=0)
            total += np.where(ok, a, 0.0).sum(axis=0)
            count += ok.sum(axis=0)
            lo = np.minimum(lo, np.where(ok, a, np.inf).min(axis=0))
            hi = np.maximum(hi, np.where(ok, a, -np.inf).max(axis=0))
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(count > 0, total / np.maximum(count, 1), np.nan)
        self._q = {
            "mean": mean,
            "min": np.where(np.isfinite(lo), lo, np.nan),
            "max": np.where(np.isfinite(hi), hi, np.nan),
            "n_nan": n_nan,
            "n_negative": n_neg,
            "n_rivers_negative": int(rivers_neg.sum()),
        }
        return self._q

    @property
    def discharge(self) -> np.ndarray:
        """Time-mean discharge of each river, m3/s."""
        return self._discharge()["mean"]

    def classes(self) -> dict[str, Any]:
        """Sort the rivers by whether they have a mouth and a flow, once.

        A river is **active** when it has a mouth (a cell in ``river_index``) and a
        positive time-mean discharge; the rest are ``mouth_no_flow`` (a cell, but no
        water), ``flow_no_mouth`` (water with no grid cell of its own -- roms-tools'
        ``overlap_*`` rivers may enter through another river's cell) or ``neither``.
        Returns the four counts, the masks (``active``,
        ``mouth``, ``flow``) and, for ``flow_no_mouth``, the time-mean discharge lost
        (``lost_m3s``, ``lost_pct`` of all river discharge) and the ``largest`` five
        such rivers.
        """
        if self._classes is not None:
            return self._classes
        q = self.discharge
        mouth = np.isfinite(self.lon) & np.isfinite(self.lat)
        flow = np.isfinite(q) & (q > 0)
        lost = flow & ~mouth
        total = float(q[flow].sum())
        lost_q = float(q[lost].sum())
        biggest = np.flatnonzero(lost)
        biggest = biggest[np.argsort(-q[biggest], kind="stable")][:5]
        self._classes = {
            "active": int((mouth & flow).sum()),
            "mouth_no_flow": int((mouth & ~flow).sum()),
            "flow_no_mouth": int(lost.sum()),
            "neither": int((~mouth & ~flow).sum()),
            "mask_active": mouth & flow,
            "mask_mouth": mouth,
            "mask_flow": flow,
            "lost_m3s": lost_q,
            "lost_pct": 100.0 * lost_q / total if total > 0 else 0.0,
            "largest": [
                {
                    "river": self.names[i],
                    "river_id": int(self.ids[i]),
                    "discharge_m3s": float(q[i]),
                }
                for i in biggest
            ],
        }
        return self._classes

    @property
    def active(self) -> np.ndarray:
        """Boolean mask of the active rivers (a mouth and a positive mean discharge)."""
        return self.classes()["mask_active"]

    def _scan(self) -> dict[str, Any]:
        """One pass over ``river_tracer`` in slabs: per-river extremes and counts.

        Only **active** rivers count, and each only at the steps its discharge is
        positive: a concentration carried by no water is meaningless (and in real
        files absurd). Per tracer and river: the highest and lowest value, the step and
        the discharge there, and the sum; per tracer: counts of NaN and out-of-bounds.
        """
        if self._scan_cache is not None:
            return self._scan_cache
        n_tr, n = len(self.tracers), self.n_rivers
        active = self.active
        hi = np.full((n_tr, n), -np.inf)
        lo = np.full((n_tr, n), np.inf)
        hi_t = np.zeros((n_tr, n), dtype=np.int64)
        lo_t = np.zeros((n_tr, n), dtype=np.int64)
        hi_q = np.zeros((n_tr, n))
        lo_q = np.zeros((n_tr, n))
        total = np.zeros((n_tr, n))
        n_nan = np.zeros(n_tr, dtype=np.int64)
        n_above = np.zeros(n_tr, dtype=np.int64)
        n_below = np.zeros(n_tr, dtype=np.int64)
        lower = np.array([_bound(t)[0] for t in self.tracers])[None, :, None]
        upper = np.array([_bound(t)[1] for t in self.tracers])[None, :, None]
        step = self._slab_steps(n_tr * n)
        cols = np.arange(n)[None, :]
        self.n_slabs = 0
        for t0 in range(0, self.n_times, step):
            a = self._tracer_slab(t0, t0 + step)
            vol = self._volume_slab(t0, t0 + step)
            self.n_slabs += 1
            flowing = ((vol > 0) & active[None, :])[:, None, :]  # (time, 1, river)
            nan = np.isnan(a)
            use = flowing & ~nan
            n_nan += (flowing & nan).sum(axis=(0, 2))
            n_above += ((a > upper) & use).sum(axis=(0, 2))
            n_below += ((a < lower) & use).sum(axis=(0, 2))
            total += np.where(use, a, 0.0).sum(axis=0)
            top = np.where(use, a, -np.inf)
            arg = top.argmax(axis=0)
            val = np.take_along_axis(top, arg[None], axis=0)[0]
            better = val > hi
            hi = np.where(better, val, hi)
            hi_t = np.where(better, t0 + arg, hi_t)
            hi_q = np.where(better, vol[arg, cols], hi_q)
            bot = np.where(use, a, np.inf)
            arg = bot.argmin(axis=0)
            val = np.take_along_axis(bot, arg[None], axis=0)[0]
            better = val < lo
            lo = np.where(better, val, lo)
            lo_t = np.where(better, t0 + arg, lo_t)
            lo_q = np.where(better, vol[arg, cols], lo_q)
        self._scan_cache = {
            "max": np.where(np.isfinite(hi), hi, np.nan),
            "max_t": hi_t,
            "max_q": hi_q,
            "min": np.where(np.isfinite(lo), lo, np.nan),
            "min_t": lo_t,
            "min_q": lo_q,
            "sum": total,
            "n_nan": n_nan,
            "n_above": n_above,
            "n_below": n_below,
        }
        return self._scan_cache

    def identical_tracers(self) -> list[tuple[int, int]]:
        """Return ``(first, other)`` tracer indices equal everywhere that has flow.

        Cheap candidates first -- the scan's per-river maximum, minimum and sum match
        -- then confirmed by comparing the values slab by slab (read only when there
        is a candidate). Constant tracers (all zero, say) are left out: they are
        reported on their own and would all "equal" each other. Each tracer is paired
        with the first earlier one it equals.
        """
        if self._identical is not None:
            return self._identical
        scan = self._scan()
        varies = [
            k
            for k in range(len(self.tracers))
            if np.nanmax(scan["max"][k], initial=-np.inf)
            > np.nanmin(scan["min"][k], initial=np.inf)
        ]
        seen: dict[int, int] = {}
        for j in varies:
            for i in varies:
                if i >= j or j in seen:
                    continue
                if all(
                    np.allclose(scan[key][i], scan[key][j], rtol=1e-12, equal_nan=True)
                    for key in ("max", "min", "sum")
                ):
                    seen[j] = i
        confirmed = dict.fromkeys(seen, True)
        if seen:
            active = self.active
            step = self._slab_steps(len(self.tracers) * self.n_rivers)
            for t0 in range(0, self.n_times, step):
                a = self._tracer_slab(t0, t0 + step)
                flowing = (self._volume_slab(t0, t0 + step) > 0) & active[None, :]
                for j, i in seen.items():
                    if confirmed[j]:
                        x, y = a[:, i], a[:, j]
                        same = (x == y) | (np.isnan(x) & np.isnan(y))
                        confirmed[j] = bool(same[flowing].all())
        self._identical = [(i, j) for j, i in seen.items() if confirmed[j]]
        return self._identical

    def constant_tracers(self) -> dict[int, float]:
        """Return ``{tracer index: value}`` of the tracers that never vary.

        A tracer is constant when every active river carries one value at every step
        it flows (all zero, or ``salt`` = 1 everywhere), judged from the scan's
        per-river extremes. A tracer with no finite value at all is reported as 0.
        """
        scan = self._scan()
        out: dict[int, float] = {}
        for k in range(len(self.tracers)):
            hi = scan["max"][k][np.isfinite(scan["max"][k])]
            lo = scan["min"][k][np.isfinite(scan["min"][k])]
            if not hi.size:
                out[k] = 0.0
            elif hi.max() == lo.min():
                out[k] = float(hi.max())
        return out

    # -- helpers ---------------------------------------------------------------------

    def _unit(self, k: int) -> str:
        """Tracer ``k``'s unit as one spelling (``mmol/m³``, ``°C``)."""
        return display(self.units[k])

    def _date(self, t: int) -> str:
        if self.times is None:
            return f"step {int(t)}"
        return str(np.datetime_as_string(self.times[int(t)], unit="D"))

    def _label(self, i: int) -> str:
        return self.names[i]

    def _tracer_index(self, name: str) -> int | None:
        return self.tracers.index(name) if name in self.tracers else None

    def _tracers_present(self, wanted) -> list[str]:
        return [t for t in wanted if t in self.tracers]

    def _figure(self, height: float):
        from matplotlib.figure import Figure

        return Figure(figsize=(_WIDTH, height))

    # -- the table and the problems ------------------------------------------

    def table(self):
        """Each varying main tracer's highest river concentrations, over all times.

        One row per ``(tracer, river)`` -- a river's highest value over every time, the
        date it happened -- for the :data:`TOP_N` highest rivers of each tracer in
        :data:`MAIN_TRACERS` that is not constant (see :meth:`constant_tracers`: an
        all-zero tracer, or ``salt`` = 1 everywhere, has no extremes), plus
        (``kind == "below bound"``) the lowest values under the tracer's lower bound in
        :data:`RIVER_BOUNDS`, listed even for a constant tracer so a negative
        concentration always shows. Only active rivers (see :meth:`classes`) at steps
        where their discharge is positive are considered, so every row has a position.
        Columns:
        ``tracer``, ``kind``, ``rank``, ``value``, ``unit``, ``river``, ``river_id``,
        ``lon``, ``lat``, ``date``, ``discharge`` (m3/s at that date).
        """
        import pandas as pd

        scan = self._scan()
        constant = self.constant_tracers()
        rows: list[dict[str, Any]] = []

        def add(tracer, kind, rank, value, t, q, unit, i):
            rows.append(
                {
                    "tracer": tracer,
                    "kind": kind,
                    "rank": rank,
                    "value": float(value),
                    "unit": unit,
                    "river": self.names[i],
                    "river_id": int(self.ids[i]),
                    "lon": float(self.lon[i]),
                    "lat": float(self.lat[i]),
                    "date": self._date(t),
                    "discharge": float(q),
                }
            )

        for name in self._tracers_present(MAIN_TRACERS):
            k = self.tracers.index(name)
            unit = self.units[k]
            hi = np.where(np.isnan(scan["max"][k]), -np.inf, scan["max"][k])
            ranked = [] if k in constant else np.argsort(-hi, kind="stable")[:TOP_N]
            for rank, i in enumerate(ranked, start=1):
                if np.isfinite(hi[i]):
                    add(
                        name,
                        "highest",
                        rank,
                        hi[i],
                        scan["max_t"][k, i],
                        scan["max_q"][k, i],
                        unit,
                        int(i),
                    )
            lower = _bound(name)[0]
            lo = np.where(np.isnan(scan["min"][k]), np.inf, scan["min"][k])
            under = np.flatnonzero(lo < lower)
            for rank, i in enumerate(
                under[np.argsort(lo[under], kind="stable")][:TOP_N], start=1
            ):
                add(
                    name,
                    "below bound",
                    rank,
                    lo[i],
                    scan["min_t"][k, i],
                    scan["min_q"][k, i],
                    unit,
                    int(i),
                )
        return pd.DataFrame(
            rows,
            columns=[
                "tracer",
                "kind",
                "rank",
                "value",
                "unit",
                "river",
                "river_id",
                "lon",
                "lat",
                "date",
                "discharge",
            ],
        )

    def problems(self) -> list[dict]:
        """Return what is out of range, as JSON-safe records.

        Each record has ``kind`` (``"negative_discharge"``, ``"nan_discharge"``,
        ``"flow_without_mouth"``, ``"below_bound"``, ``"above_bound"``, ``"nan"``,
        ``"identical_tracers"``), ``variable``, ``message``, ``count`` (river-time
        values) and ``n_rivers``. A bound check also has ``bound`` and ``worst`` (the
        most extreme value); ``flow_without_mouth`` has ``lost_m3s``, ``lost_pct`` and
        the five ``largest`` rivers; ``identical_tracers`` names the other tracer in
        ``equals``. Concentrations are checked for active rivers at steps where they
        have flow only. An empty list is a clean file.
        """
        out: list[dict[str, Any]] = []
        q = self._discharge()
        if q["n_negative"]:
            worst = float(np.nanmin(q["min"]))
            out.append(
                {
                    "kind": "negative_discharge",
                    "variable": "river_volume",
                    "message": (
                        f"{q['n_negative']} negative discharge values in "
                        f"{q['n_rivers_negative']} rivers (lowest {worst:.4g} m3/s)"
                    ),
                    "count": int(q["n_negative"]),
                    "n_rivers": int(q["n_rivers_negative"]),
                    "worst": worst,
                }
            )
        if q["n_nan"]:
            out.append(
                {
                    "kind": "nan_discharge",
                    "variable": "river_volume",
                    "message": f"{q['n_nan']} missing (NaN) discharge values",
                    "count": int(q["n_nan"]),
                    "n_rivers": int(np.isnan(q["mean"]).sum()),
                }
            )
        cls = self.classes()
        if cls["flow_no_mouth"]:
            names = ", ".join(
                f"{r['river']} ({r['discharge_m3s']:.4g} m3/s)" for r in cls["largest"]
            )
            out.append(
                {
                    "kind": "flow_without_mouth",
                    "variable": "river_index",
                    "message": (
                        f"{cls['flow_no_mouth']} rivers have discharge but no grid "
                        f"cell of their own in river_index: {cls['lost_m3s']:.4g} "
                        f"m3/s ({cls['lost_pct']:.3g}% of all river discharge, time "
                        "mean). Check how the model places them -- roms-tools names "
                        "rivers that share a cell overlap_*, and those may enter "
                        f"through another river's cell. Largest: {names}"
                    ),
                    "count": int(cls["flow_no_mouth"]),
                    "n_rivers": int(cls["flow_no_mouth"]),
                    "lost_m3s": cls["lost_m3s"],
                    "lost_pct": cls["lost_pct"],
                    "largest": cls["largest"],
                }
            )
        scan = self._scan()
        for k, name in enumerate(self.tracers):
            lower, upper = _bound(name)
            unit = self.units[k]
            if scan["n_nan"][k]:
                out.append(
                    {
                        "kind": "nan",
                        "variable": name,
                        "message": (
                            f"{name}: {int(scan['n_nan'][k])} missing (NaN) values"
                        ),
                        "count": int(scan["n_nan"][k]),
                        "n_rivers": 0,
                    }
                )
            if scan["n_below"][k]:
                worst = float(np.nanmin(scan["min"][k]))
                n_r = int((scan["min"][k] < lower).sum())
                out.append(
                    {
                        "kind": "below_bound",
                        "variable": name,
                        "message": (
                            f"{name}: {int(scan['n_below'][k])} values below "
                            f"{lower:g} {unit} in {n_r} rivers (lowest {worst:.4g})"
                        ),
                        "count": int(scan["n_below"][k]),
                        "n_rivers": n_r,
                        "bound": float(lower),
                        "worst": worst,
                    }
                )
            if scan["n_above"][k]:
                worst = float(np.nanmax(scan["max"][k]))
                n_r = int((scan["max"][k] > upper).sum())
                out.append(
                    {
                        "kind": "above_bound",
                        "variable": name,
                        "message": (
                            f"{name}: {int(scan['n_above'][k])} values above "
                            f"{upper:g} {unit} in {n_r} rivers (highest {worst:.4g})"
                        ),
                        "count": int(scan["n_above"][k]),
                        "n_rivers": n_r,
                        "bound": float(upper),
                        "worst": worst,
                    }
                )
        for i, j in self.identical_tracers():
            a, b = self.tracers[i], self.tracers[j]
            out.append(
                {
                    "kind": "identical_tracers",
                    "variable": a,
                    "equals": b,
                    "message": f"river {a} equals river {b} in every river",
                    "count": int(self.active.sum()),
                    "n_rivers": int(self.active.sum()),
                }
            )
        return out

    # -- figures ---------------------------------------------------------------

    def figures(self, draw: str) -> list:
        """Draw one view: ``draw`` is one of :data:`RIVER_DRAWS`.

        Returns ``[(name, matplotlib.figure.Figure), ...]``: the names are short and
        unique per view (``"river_map"``, ``"river_histograms_1"``, ...), and every
        figure is letter-page wide.
        """
        if draw not in RIVER_DRAWS:
            raise ValueError(
                f"Unknown river draw {draw!r}; expected one of {list(RIVER_DRAWS)}."
            )
        return getattr(self, f"_draw_{draw}")()

    def _draw_map(self) -> list:
        import cartopy.crs as ccrs
        from matplotlib.colors import LogNorm
        from matplotlib.lines import Line2D

        from ocean_skill.plot.matplotlib_renderer import (
            DEFAULT_GRIDLINE_KWARGS,
            DEFAULT_TICK_LABEL_KWARGS,
            _basemap,
        )

        q = self.discharge
        lon360 = np.mod(self._grid_lon, 360.0)
        lon180 = (self._grid_lon + 180.0) % 360.0 - 180.0
        # the representation with the smaller span is the one that does not split the
        # domain at its seam: a dateline-straddling grid is centred on 180
        central = 180.0 if np.ptp(lon360) < np.ptp(lon180) else 0.0
        proj = ccrs.PlateCarree(central_longitude=central)
        data_lon = lon360 if central else lon180
        x0, x1 = float(np.nanmin(data_lon)), float(np.nanmax(data_lon))
        y0, y1 = float(np.nanmin(self._grid_lat)), float(np.nanmax(self._grid_lat))
        pad_x, pad_y = 0.03 * (x1 - x0 or 1.0), 0.03 * (y1 - y0 or 1.0)
        cls = self.classes()
        notes = _wrap(self._class_note(cls))
        n_note = notes.count("\n") + 1
        # inches, top to bottom: title, map, tick labels, size legend, class note
        top, below, legend_h = 0.5, 0.4, 0.45
        note_h = 0.17 * n_note + 0.15
        map_h = float(
            np.clip(
                (_WIDTH - 1.4) * (y1 - y0 + 2 * pad_y) / (x1 - x0 + 2 * pad_x), 3.5, 8.0
            )
        )
        height = top + map_h + below + legend_h + note_h
        fig = self._figure(height)
        bottom = below + legend_h + note_h
        ax = fig.add_axes(
            (0.08, bottom / height, 0.72, map_h / height), projection=proj
        )
        shift = central
        ax.set_extent(
            (x0 - pad_x - shift, x1 + pad_x - shift, y0 - pad_y, y1 + pad_y), crs=proj
        )
        _basemap(
            ax,
            gridline_kwargs=dict(DEFAULT_GRIDLINE_KWARGS),
            tick_label_kwargs={**DEFAULT_TICK_LABEL_KWARGS, "size": _FS},
        )
        edge = [
            (self._grid_lon[0, :], self._grid_lat[0, :]),
            (self._grid_lon[:, -1], self._grid_lat[:, -1]),
            (self._grid_lon[-1, ::-1], self._grid_lat[-1, ::-1]),
            (self._grid_lon[::-1, 0], self._grid_lat[::-1, 0]),
        ]
        ring_lon = np.concatenate([e[0] for e in edge])
        ring_lat = np.concatenate([e[1] for e in edge])
        ring_lon = (
            np.mod(ring_lon, 360.0) if central else (ring_lon + 180.0) % 360.0 - 180.0
        )
        ax.plot(
            ring_lon - shift,
            ring_lat,
            color="0.35",
            lw=0.6,
            ls="--",
            transform=proj,
            zorder=4,
        )
        mouth_lon = np.mod(self.lon, 360.0) if central else self.lon
        good = cls["mask_active"]
        order = np.flatnonzero(good)[np.argsort(q[good], kind="stable")]  # large on top
        if order.size:
            qs = q[order]
            lo, hi = float(qs.min()), float(qs.max())
            span = max(np.log10(hi) - np.log10(lo), 1e-9)
            crowd = min(
                1.0, math.sqrt(1500.0 / qs.size)
            )  # smaller dots for 100,000 rivers

            def marker_area(value):
                return crowd * (4.0 + 40.0 * (np.log10(value) - np.log10(lo)) / span)

            norm = LogNorm(vmin=lo, vmax=hi if hi > lo else lo * 10)
            sc = ax.scatter(
                mouth_lon[order] - shift,
                self.lat[order],
                c=qs,
                s=marker_area(qs),
                cmap="viridis",
                norm=norm,
                transform=proj,
                zorder=5,
                linewidths=0.2,
                edgecolors="0.2",
                rasterized=True,
            )
            cax = fig.add_axes((0.84, bottom / height, 0.025, map_h / height))
            cb = fig.colorbar(sc, cax=cax)
            cb.set_label(f"time-mean discharge ({display('m3/s')})", fontsize=_FS)
            cb.ax.tick_params(labelsize=_FS, which="both")
            sizes = [lo, math.sqrt(lo * hi), hi] if hi > lo else [lo]
            handles = [
                Line2D(
                    [],
                    [],
                    ls="",
                    marker="o",
                    ms=math.sqrt(float(marker_area(v))),
                    markerfacecolor="0.75",
                    markeredgecolor="0.2",
                    markeredgewidth=0.2,
                )
                for v in sizes
            ]
            fig.legend(
                handles,
                [f"{format_value(v)} {display('m3/s')}" for v in sizes],
                title="marker size: time-mean discharge",
                loc="upper center",
                bbox_to_anchor=(0.44, (below + legend_h + note_h - 0.3) / height),
                ncol=len(sizes),
                fontsize=_FS,
                title_fontsize=_FS,
                frameon=False,
                labelspacing=1.0,
                columnspacing=2.5,
            )
        fig.text(
            0.08,
            (note_h - 0.08) / height,
            notes,
            ha="left",
            va="top",
            fontsize=_FS,
            color="0.2",
        )
        fig.suptitle(
            f"River mouths: {cls['active']:,} of {self.n_rivers:,} rivers (with a grid "
            "cell and flow), sized and colored by time-mean discharge",
            fontsize=_FS_TITLE - 1,
            y=1 - 0.12 / height,
        )
        return [("river_map", fig)]

    @staticmethod
    def _class_note(counts: dict[str, Any]) -> str:
        """Return the note under the map: what was drawn, and why not the rest.

        Rivers with flow but no grid cell of their own are *not* called lost: roms-tools
        names rivers that share a cell ``overlap_*``, and those may enter through
        another river's cell.
        """
        left = []
        if counts["mouth_no_flow"]:
            left.append(f"{counts['mouth_no_flow']:,} with a grid cell but no flow")
        if counts["flow_no_mouth"]:
            left.append(
                f"{counts['flow_no_mouth']:,} with flow but no grid cell of their own "
                f"({format_value(counts['lost_m3s'], 4)} {display('m3/s')}, "
                f"{format_value(counts['lost_pct'], 2)}% of discharge; they may enter "
                "through another river's cell)"
            )
        if counts["neither"]:
            left.append(f"{counts['neither']:,} with neither")
        drawn = f"Drawn: {counts['active']:,} rivers with a grid cell and flow."
        return f"{drawn} Not drawn: {'; '.join(left)}." if left else drawn

    def _draw_histograms(self) -> list:
        from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter

        q = self.discharge
        active = self.active
        ti = self.time_index
        a = self._tracer_slab(ti, ti + 1)[0]  # (ntracers, nriver)
        flowing = active & (self._volume_slab(ti, ti + 1)[0] > 0)
        panels: list[tuple[str, np.ndarray, str, tuple[float, float]]] = [
            (
                "discharge, time mean",
                q[active],
                display("m3/s"),
                (-math.inf, math.inf),
            )
        ]
        zero, constant = [], []
        for k, name in enumerate(self.tracers):
            v = a[k][flowing]
            v = v[np.isfinite(v)]
            if v.size and np.ptp(v) == 0:
                (zero if v[0] == 0 else constant).append(
                    name if v[0] == 0 else f"{name} = {format_value(v[0])}"
                )
                continue
            panels.append((name, v, self._unit(k), _bound(name)))
        notes = _wrap(
            "; ".join(
                f"{label}: {', '.join(items)}"
                for label, items in (("all zero", zero), ("constant", constant))
                if items
            )
        )
        n_note = notes.count("\n") + 1 if notes else 0
        per = _HIST_COLS * _HIST_ROWS
        n_pages = math.ceil(len(panels) / per)
        when = self._date(ti)
        n_flow = int(flowing.sum())
        out = []
        for page in range(n_pages):
            shown = panels[page * per : (page + 1) * per]
            rows = math.ceil(len(shown) / _HIST_COLS)
            top = 0.55 + (0.17 * n_note + 0.3 if notes else 0.0)
            height = top + 0.45 + 2.25 * rows
            fig = self._figure(height)
            fig.subplots_adjust(
                left=0.09,
                right=0.97,
                top=1 - top / height,
                bottom=0.45 / height,
                hspace=0.6,
                wspace=0.4,
            )
            if notes:
                fig.text(
                    0.09,
                    1 - 0.5 / height,
                    notes,
                    ha="left",
                    va="top",
                    fontsize=_FS,
                    color="0.2",
                )
            for slot, (name, vals, unit, (lower, upper)) in enumerate(shown):
                ax = fig.add_subplot(rows, _HIST_COLS, slot + 1)
                v = vals[np.isfinite(vals)]
                pos = v[v > 0]
                # temperature and salinity sit on a narrow linear range and can be 0:
                # a log axis would hide them; everything else spans decades
                linear = name in ("temp", "salt") and v.size
                if linear:
                    ax.hist(v, bins=25, color="#3b6ea5", linewidth=0)
                    pos = v  # every value is drawn
                elif pos.size:
                    lo, hi = pos.min(), pos.max()
                    edges = (
                        np.logspace(np.log10(lo), np.log10(hi), 25)
                        if hi > lo
                        else np.array([lo * 0.9, lo * 1.1])
                    )
                    ax.hist(pos, bins=edges, color="#3b6ea5", linewidth=0)
                    ax.set_xscale("log")
                    if math.floor(math.log10(hi)) - math.ceil(math.log10(lo)) >= 1:
                        # decade ticks: 10^0, 10^1, ...
                        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=6))
                        ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
                    else:
                        # under a decade a power of ten would label nothing, or read
                        # 10^1.04: four plain values across the range instead
                        ticks = np.geomspace(lo, hi, 4)
                        ax.set_xticks(ticks, [format_value(t) for t in ticks])
                    ax.xaxis.set_minor_formatter(NullFormatter())
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "no positive values",
                        transform=ax.transAxes,
                        ha="center",
                        fontsize=_FS,
                        color="0.4",
                    )
                title = f"{name} ({unit})" if unit else name
                ax.set_title(title, fontsize=_FS + 1, pad=4)
                if slot % _HIST_COLS == 0:
                    ax.set_ylabel("rivers", fontsize=_FS)
                outside = int(((v < lower) | (v > upper)).sum())
                nonpos = int((v <= 0).sum())
                flags = []
                if outside:
                    flags.append(f"outside bounds: {outside:,}")
                if nonpos and pos.size and not linear:  # a log axis cannot draw them
                    word = "value" if nonpos == 1 else "values"
                    flags.append(f"{nonpos:,} {word} <= 0 not shown")
                if flags:
                    ax.text(
                        0.97,
                        0.95,
                        "\n".join(flags),
                        transform=ax.transAxes,
                        ha="right",
                        va="top",
                        fontsize=_FS,
                        color="#a02020" if outside else "0.3",
                    )
                ax.tick_params(labelsize=_FS, which="both")
            suffix = f" ({page + 1} of {n_pages})" if n_pages > 1 else ""
            fig.suptitle(
                f"River inputs: distributions across {n_flow:,} active rivers with "
                f"flow at {when}{suffix}",
                fontsize=_FS_TITLE - 1,
                y=1 - 0.15 / height,
            )
            out.append((f"river_histograms_{page + 1}", fig))
        return out

    def sample_rivers(self) -> list[tuple[int, str]]:
        """Twelve rivers ``(index, class)`` by time-mean discharge, deterministically.

        Among the active rivers (see :meth:`classes`): the three largest, then --
        terciles of the rivers ranked from smallest -- the three rivers at the middle
        of each tercile
        (``small``, ``medium``, ``large``), skipping any already taken so the twelve
        are distinct (fewer when the file has fewer rivers).
        """
        q = self.discharge
        ranked = np.flatnonzero(self.active)
        ranked = ranked[np.argsort(q[ranked], kind="stable")]  # smallest first
        taken: list[tuple[int, str]] = []
        used: set[int] = set()

        def take(positions: list[int], label: str, want: int = 3):
            for p in positions:
                i = int(ranked[p])
                if i not in used and want > 0:
                    used.add(i)
                    taken.append((i, label))
                    want -= 1

        n = len(ranked)
        take(list(range(n - 1, -1, -1)), "largest")
        for label, lo, hi in (
            ("large", 2 * n // 3, n),
            ("medium", n // 3, 2 * n // 3),
            ("small", 0, n // 3),
        ):
            mid = (lo + hi) // 2
            # outward from the tercile's median rank: mid, mid-1, mid+1, ...
            outward = [
                mid + s for d in range(hi - lo) for s in ((d,) if d == 0 else (-d, d))
            ]
            take([p for p in outward if lo <= p < hi], label)
        order = {"small": 0, "medium": 1, "large": 2, "largest": 3}
        return sorted(taken, key=lambda t: (order[t[1]], float(q[t[0]])))

    def _draw_sample(self) -> list:
        import matplotlib.dates as mdates
        from matplotlib import colormaps

        chosen = self.sample_rivers()
        idx = [i for i, _ in chosen]
        sel = np.asarray(idx)
        order = np.argsort(sel)
        vol = np.asarray(
            self._ds["river_volume"]
            .isel(nriver=sel[order])
            .transpose("river_time", "nriver")
            .values,
            dtype="float64",
        )
        vol = vol[:, np.argsort(order)]
        names = self._tracers_present(SAMPLE_TRACERS)
        # one read of every sampled tracer for the sampled rivers, not one per tracer
        ks = [self.tracers.index(name) for name in names]
        arr = np.asarray(
            self._ds["river_tracer"]
            .isel(ntracers=ks, nriver=sel[order])
            .transpose("ntracers", "river_time", "nriver")
            .values,
            dtype="float64",
        )
        series = {name: arr[j][:, np.argsort(order)] for j, name in enumerate(names)}
        for data in series.values():  # a concentration carried by no water is no value
            data[~(vol > 0)] = np.nan
        # a tracer that never varies (all zero, salt = 1) draws as a flat line: leave
        # it out, the histogram page lists it
        names = [
            n
            for n in names
            if np.isfinite(series[n]).any()
            and np.ptp(series[n][np.isfinite(series[n])])
        ]
        x = self.times if self.times is not None else np.arange(self.n_times)
        x = (
            mdates.date2num(x.astype("datetime64[ms]").astype(object))
            if self.times is not None
            else x
        )
        n_pan = 1 + len(names)
        legend_h = 0.2 + 0.18 * math.ceil(len(chosen) / 3)  # inches, below the axes
        height = 1.3 * n_pan + 1.0 + 0.45 + legend_h
        fig = self._figure(height)
        axes = fig.subplots(n_pan, 1, sharex=True)
        axes = np.atleast_1d(axes)
        fig.subplots_adjust(
            left=0.1,
            right=0.97,
            top=1 - 0.5 / height,
            bottom=(0.45 + legend_h) / height,
            hspace=0.25,
        )
        colors = colormaps["tab20"].colors
        styles = {"small": ":", "medium": "--", "large": "-.", "largest": "-"}
        panels = [("discharge", display("m3/s"), vol, True)] + [
            (n, self._unit(self.tracers.index(n)), series[n], False) for n in names
        ]
        for ax, (title, unit, data, logy) in zip(axes, panels, strict=True):
            for j, (i, label) in enumerate(chosen):
                ax.plot(
                    x,
                    data[:, j],
                    color=colors[j % len(colors)],
                    ls=styles[label],
                    lw=0.9,
                    label=f"{self.names[i]} ({label})",
                )
            if logy and np.any(data > 0):
                ax.set_yscale("log")
            ax.set_ylabel(f"{title} ({unit})" if unit else title, fontsize=_FS)
            ax.tick_params(labelsize=_FS)
        if self.times is not None:
            axes[-1].xaxis_date()
            locator = mdates.AutoDateLocator()
            axes[-1].xaxis.set_major_locator(locator)
            axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        else:
            axes[-1].set_xlabel("time step", fontsize=_FS)
        handles, labels = axes[0].get_legend_handles_labels()
        # right under the last panel's tick labels
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, legend_h / height),
            ncol=3,
            fontsize=_FS,
            frameon=False,
        )
        fig.suptitle(
            f"Sample rivers: {len(chosen)} of {int(self.active.sum()):,} active rivers "
            "by time-mean discharge",
            fontsize=_FS_TITLE,
        )
        return [("river_sample", fig)]

    def _draw_extremes(self) -> list:
        df = self.table()
        if df.empty:
            fig = self._figure(2.0)
            fig.text(
                0.5,
                0.5,
                "No river concentrations to list.",
                ha="center",
                fontsize=_FS_TITLE,
            )
            return [("river_extremes_1", fig)]
        heads = [
            "tracer",
            "#",
            "value",
            "river",
            "lon",
            "lat",
            "date",
            f"Q ({display('m3/s')})",
        ]
        records = []  # (tracer, label, cells after the label)
        for r in df.itertuples():
            unit = display(r.unit)
            records.append(
                (
                    (r.tracer, r.kind),
                    f"{r.tracer} ({unit})" if unit else str(r.tracer),
                    [
                        str(r.rank) if r.kind == "highest" else f"low {r.rank}",
                        format_value(r.value),
                        str(r.river)[:28],
                        format_lon(r.lon),
                        format_lat(r.lat),
                        str(r.date),
                        format_value(r.discharge),
                    ],
                )
            )
        has_low = bool((df["kind"] == "below bound").any())
        size = _TABLE_ROWS_PER_PAGE
        pages = [records[i : i + size] for i in range(0, len(records), size)]
        out = []
        for n, page in enumerate(pages, start=1):
            # the tracer is named on its first row of the page, blank below it
            rows, groups, seen = [], [], None
            for key, label, cells in page:
                rows.append([label if key != seen else "", *cells])
                groups.append(key[0])
                seen = key
            note = (
                "low N: the lowest values, below the tracer's lower bound "
                if has_low and n == len(pages)
                else ""
            )
            height = 1.0 + 0.22 * (len(rows) + 1) + (0.25 if note else 0.0)
            fig = self._figure(height)
            ax = fig.add_axes(
                (
                    0.03,
                    (0.3 + (0.25 if note else 0.0)) / height,
                    0.94,
                    1 - (0.65 + 0.3 + (0.25 if note else 0.0)) / height,
                )
            )
            ax.axis("off")
            tab = ax.table(
                cellText=rows,
                colLabels=heads,
                colWidths=[0.19, 0.06, 0.09, 0.24, 0.1, 0.09, 0.11, 0.1],
                loc="upper center",
                cellLoc="left",
            )
            tab.auto_set_font_size(False)
            tab.set_fontsize(_FS)
            tab.scale(1.0, 1.25)
            shade = {}
            for g in groups:
                shade.setdefault(g, len(shade) % 2)
            for (r, _c), cell in tab.get_celld().items():
                cell.set_linewidth(0.3)
                cell.set_edgecolor("0.8")
                if r == 0:
                    cell.set_facecolor("0.85")
                    cell.get_text().set_fontweight("bold")
                elif shade[groups[r - 1]]:
                    cell.set_facecolor("0.95")
            if note:
                fig.text(0.03, 0.15 / height, note, fontsize=_FS, color="0.3")
            suffix = f" ({n} of {len(pages)})" if len(pages) > 1 else ""
            fig.suptitle(
                f"Highest river concentrations over all times, top {TOP_N} per "
                f"tracer, active rivers with flow{suffix}",
                fontsize=_FS_TITLE - 1,
                y=1 - 0.15 / height,
            )
            out.append((f"river_extremes_{n}", fig))
        return out
