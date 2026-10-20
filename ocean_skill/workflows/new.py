"""``ocean-skill-suite new mid-run-check|input-check``: fill a template's setup block.

Starting a check by hand means copying ``templates/<name>.yaml`` and editing a few
paths. This finds them in a C-Star run folder (``<run>/tasks/<task>/
{input/input_datasets,output,work}``; ``input-check`` reads the first task's inputs, or
the folder named by ``--inputs``) and rewrites the block between the template's
``# >>> setup`` and ``# <<< setup`` markers. The rest of the file (pages, comments) is
kept byte for byte: the block is spliced in as text, since loading and dumping the YAML
would drop the comments. ``final-report`` and ``model-comparison`` are copied and
edited by hand; guesses are written as ``# TODO check`` comments.

Besides file names and the grid's ``eta_rho`` length, the grid's ``lon_rho``,
``lat_rho`` and ``mask_rho`` are read once, to place the two default section lines
(``defaults.transects``) over the ocean; nothing else is read from the data.
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Any

import yaml

TEMPLATES = ("mid-run-check", "input-check")
TEMPLATE = Path(__file__).parent / "templates" / "mid-run-check.yaml"

#: The shared observation catalogs on Anvil; ``--shared-catalogs`` points elsewhere.
SHARED_CATALOGS = "/anvil/projects/x-ees250129/cstar-forge-data/source-data/catalogs"

#: Restart references are split into one-level chunks of this many rows (the largest
#: divisor of ``eta_rho`` in range, else the smallest one above it) so a page matched to
#: a depth reads whole columns.
SUBCHUNK_RANGE = (32, 128)

_STAMP = re.compile(r"output_rst\.(\d{4})(\d{2})(\d{2})")


class SuiteNewError(Exception):
    """Something the user has to fix before a suite can be written."""


def _abs(path: str) -> Path:
    return Path(os.path.abspath(os.path.expanduser(path)))


def _q(text: object) -> str:
    """``text`` as a YAML scalar: plain when that reads back the same, else quoted."""
    s = str(text)
    plain = (
        re.fullmatch(r"[A-Za-z0-9_./~][A-Za-z0-9_./~+=*-]*", s)
        and yaml.safe_load(s) == s
    )
    return s if plain else json.dumps(s)


def find_restarts(roots: list[Path]) -> list[str]:
    """Return one restart glob per run folder.

    A folder of symlinks to every restart (``gathered_output/``, ``tasks/rst_files/``)
    is preferred where there is one: it de-duplicates restarts two tasks both wrote.
    """
    globs = []
    for root in roots:
        gathered = (root / "gathered_output", root / "tasks" / "rst_files")
        folders = [f for f in gathered if any(f.glob("output_rst.*.nc"))]
        globs.append(
            str(folders[0] / "output_rst.*.nc")
            if folders
            else str(root / "tasks" / "*" / "output" / "output_rst.*.nc")
        )
    if not any(glob.glob(g) for g in globs):
        raise SuiteNewError(
            f"no output_rst.*.nc under {', '.join(map(str, roots))}; "
            "pass --files GLOB to name them"
        )
    return globs


def find_grid(roots: list[Path]) -> Path:
    """Return the one ``*_grid.nc`` in the tasks' inputs, or raise naming them."""
    pattern = "tasks/*/input/input_datasets/*_grid.nc"
    grids = [Path(g) for r in roots for g in sorted(glob.glob(str(r / pattern)))]
    names = sorted({g.name for g in grids})
    if len(names) != 1:
        found = (
            f"{len(names)} different grids ({', '.join(names)})" if names else "no grid"
        )
        raise SuiteNewError(f"{found} in {pattern}; pass --grid FILE")
    return grids[0]


def pick_subchunk(grid: Path) -> tuple[dict[str, int], int]:
    """Return the restart ``subchunk:`` for ``grid`` and its ``eta_rho`` length.

    One level per chunk, and rows so the chunks tile the grid: the largest divisor of
    ``eta_rho`` in :data:`SUBCHUNK_RANGE`, else (1346 = 2 x 673) the smallest divisor
    of at least ``SUBCHUNK_RANGE[0]`` below ``eta_rho``. With none (a prime) the rows
    stay whole.
    """
    import xarray as xr

    with xr.open_dataset(grid) as ds:
        n = int(ds.sizes["eta_rho"])
    divisors = [d for d in range(SUBCHUNK_RANGE[0], n) if n % d == 0]
    in_range = [d for d in divisors if d <= SUBCHUNK_RANGE[1]]
    rows = max(in_range) if in_range else min(divisors, default=0)
    return {"s_rho": 1, "s_w": 1, **({"eta_rho": rows} if rows else {})}, n


#: Boundary side -> the grid dimension and index its edge lies on.
#: A boundary side gets pages only if at least this share of its edge cells is water.
MIN_WET = 0.1
SIDES = {"south": ("eta_rho", 0), "north": ("eta_rho", -1)}
SIDES |= {"west": ("xi_rho", 0), "east": ("xi_rho", -1)}


def _first_inputs(roots: list[Path]) -> Path:
    """Return the ``input_datasets`` folder of the task whose restarts start first."""

    def start(folder: Path) -> tuple[str, str]:
        task = folder.parents[1]
        rst = sorted(p.name for p in (task / "output").glob("output_rst.*.nc"))
        return (rst[0] if rst else "~", task.name)

    dirs = [p for r in roots for p in (r / "tasks").glob("*/input/input_datasets")]
    if not dirs:
        raise SuiteNewError("no tasks/*/input/input_datasets in --run; pass --inputs")
    return min(dirs, key=start)


def _by_glob(folder: Path, pattern: str) -> dict[str, list[Path]]:
    """Group files matching ``pattern`` by name with their date digits as ``?``."""
    groups: dict[str, list[Path]] = {}
    for f in sorted(folder.glob(pattern)):
        key = re.sub(r"\d{4,8}(?=\.nc$)", lambda m: "?" * len(m[0]), str(f))
        groups.setdefault(key, []).append(f)
    return groups


def _is_climatology(path: str) -> bool:
    """Whether a roms-tools file (``..._clim.nc``) is a twelve-month climatology."""
    return Path(path).stem.endswith(("_clim", "-clim"))


def _surface_label(part: str) -> str:
    """Return how a figure names a surface forcing file, from its name's middle.

    ``physics`` reads ``surface physics``, ``bgc-unified_clim`` reads ``surface BGC
    unified climatology`` and a salinity-restoring file ``salinity restoring``.
    """
    if part.startswith("restoring"):
        return "salinity restoring" + (" climatology" if "clim" in part else "")
    words = [w for w in re.split(r"[-_]+", part) if w]
    words = ["climatology" if w == "clim" else w for w in words]
    words = ["BGC" if w == "bgc" else w for w in words]
    return " ".join(["surface", *words])


def _whole(x: float) -> int:
    """Round half up to a whole number."""
    import math

    return math.floor(x + 0.5)


def _outward(lo: float, hi: float) -> tuple[float, float]:
    """Widen ``lo``..``hi`` outward to 0.1 degree (rounded first: float noise)."""
    import math

    return (
        math.floor(round(lo * 10, 6)) / 10,
        math.ceil(round(hi * 10, 6)) / 10,
    )


def _lon_label(lon: float) -> str:
    """Spell a longitude in whole degrees, east or west: ``143E``, ``93W``, ``0E``."""
    deg = _whole((lon + 180) % 360 - 180)
    deg = 180 if deg == -180 else deg
    return f"{abs(deg)}{'W' if deg < 0 else 'E'}"


def _lat_label(lat: float) -> str:
    """Spell a latitude in whole degrees: ``34N``, ``12S``, ``0N``."""
    deg = _whole(lat)
    return f"{abs(deg)}{'S' if deg < 0 else 'N'}"


#: A stretch of land this many degrees wide or more along a default section line ends
#: it: the line keeps the longest unbroken run of ocean (the Pacific equator from
#: Halmahera to Ecuador, not through Sumatra and Borneo), while a smaller island
#: (the Galapagos) is crossed.
LAND_GAP = 2.0


def _longest_run(along: Any, gap: float) -> Any:
    """Return the indices into ``along`` of its longest run without a ``gap`` jump.

    ``along`` is each ocean cell's position along the line (degrees, or cell index
    scaled to degrees); sorted, it is cut wherever the land between two neighbors
    (their distance less the typical step between ocean cells) is ``gap`` or more and
    the distance is over twice that step, so a grid coarser than ``gap`` is not cut at
    every cell. The run spanning the most is kept.
    """
    import numpy as np

    order = np.argsort(along)
    steps = np.diff(along[order])
    distinct = np.diff(np.unique(along))  # several cells of a band share a position
    step = float(np.median(distinct)) if distinct.size else 0.0
    cuts = np.flatnonzero((steps - step >= gap) & (steps > 2.0 * step)) + 1
    runs = np.split(order, cuts)
    return max(runs, key=lambda r: along[r].max() - along[r].min())


def default_transects(grid: Path) -> dict[str, dict]:
    """Return the two default section lines of ``grid``, ``{label: {"transect": ...}}``.

    Reads ``lon_rho``, ``lat_rho`` and ``mask_rho`` only. A zonal line (latitude 0 when
    the ocean reaches at least 2 degrees to each side of it, else the median ocean
    latitude to the whole degree) and a meridional one (the median ocean longitude to
    the whole degree), each running over the ocean cells that lie on it, widened to
    0.1 degree. "On the line" means the cell center is within the half-width the cell
    has across the line: half the sum of the cell's latitude (longitude) change along
    both grid axes. That is half the spacing on a regular grid and wider on a tilted
    one (the California Current domain is rotated ~30 degrees, where a latitude line
    crosses cells row after row), so the band keeps no gaps.

    Longitudes keep the grid's convention unless the grid wraps (a jump of more than
    180 degrees between neighbors: a -180..180 grid crossing the dateline, or a
    0..360 one crossing Greenwich), where they switch to the other so that the line's
    ``min`` is below its ``max``. Raises ``ValueError`` (or ``KeyError``/``OSError``
    for an unreadable grid) when there is nothing to draw.

    A grid rotated more than :data:`MAX_ROTATION` degrees (the median over ocean cells
    of the xi direction's angle from east: the grid's ``angle`` if it has one, else
    from ``lon_rho``/``lat_rho``) gets lines along its own axes instead, since a
    zonal line would cross its coast obliquely: two ``waypoints`` transects, along
    the middle row and the middle column from their first to their last ocean cell.
    """
    return _default_transects(grid)[0]


#: A grid whose rows are rotated more than this many degrees from east gets lines that
#: follow its rows and columns (see :func:`default_transects`).
MAX_ROTATION = 10.0


def _default_transects(grid: Path) -> tuple[dict[str, dict], float]:
    """Return :func:`default_transects` and the grid's rotation in degrees."""
    import numpy as np
    import xarray as xr

    with xr.open_dataset(grid) as ds:
        lon, lat, mask = (
            np.asarray(ds[v].values, dtype=float)
            for v in ("lon_rho", "lat_rho", "mask_rho")
        )
        angle = np.asarray(ds["angle"].values, float) if "angle" in ds else None
    if lon.ndim != 2 or min(lon.shape) < 2:
        raise ValueError(f"lon_rho of {lon.shape} is not a 2-D grid")
    jumps = [np.nanmax(np.abs(np.diff(lon, axis=a))) for a in (0, 1)]
    if max(jumps) > 180:  # the seam is inside the domain: pick the other convention
        lon = lon % 360 if np.nanmin(lon) < 0 else (lon + 180) % 360 - 180
    ocean = (mask > 0) & np.isfinite(lon) & np.isfinite(lat)
    if not ocean.any():
        raise ValueError("the grid has no ocean cells (mask_rho is all 0)")

    # the xi direction's angle from east, in degrees (ROMS stores ``angle`` in radians)
    if angle is not None and angle.shape == lon.shape:
        theta = np.degrees(angle)
    else:
        dlon, dlat = np.gradient(lon, axis=1), np.gradient(lat, axis=1)
        theta = np.degrees(np.arctan2(dlat, dlon * np.cos(np.radians(lat))))
    theta = (theta + 180) % 360 - 180
    rotation = float(np.nanmedian(theta[ocean]))
    if abs(rotation) > MAX_ROTATION:
        return _grid_lines(lon, lat, ocean), rotation

    def half_width(field: np.ndarray) -> np.ndarray:
        return 0.5 * sum(np.abs(np.gradient(field, axis=a)) for a in (0, 1))

    wet_lat = lat[ocean]
    if wet_lat.min() <= -2 and wet_lat.max() >= 2:
        lat0 = 0
    else:
        lat0 = _whole(float(np.median(wet_lat)))
    lon0 = _whole(float(np.median(lon[ocean])))

    on_lat = ocean & (np.abs(lat - lat0) <= half_width(lat))
    on_lon = ocean & (np.abs(lon - lon0) <= half_width(lon))
    # the median line can fall between ocean patches: then span all the ocean
    lons = lon[on_lat] if on_lat.any() else lon[ocean]
    lats = lat[on_lon] if on_lon.any() else lat[ocean]
    lons, lats = (v[_longest_run(v, LAND_GAP)] for v in (lons, lats))
    (x0, x1), (y0, y1) = (
        _outward(lons.min(), lons.max()),
        _outward(lats.min(), lats.max()),
    )
    zonal = "Equator" if lat0 == 0 else _lat_label(lat0)
    return {
        f"{zonal} ({_lon_label(x0)}-{_lon_label(x1)})": {
            "transect": {"lat": lat0, "lon": {"min": x0, "max": x1}}
        },
        f"{_lon_label(lon0)} ({_lat_label(y0)}-{_lat_label(y1)})": {
            "transect": {"lon": float(lon0), "lat": {"min": y0, "max": y1}}
        },
    }, rotation


def _grid_lines(lon, lat, ocean) -> dict[str, dict]:
    """Return waypoint lines along the middle row and column of a rotated grid.

    Each runs over the longest unbroken stretch of ocean (no :data:`LAND_GAP` of land)
    of the row (``eta_rho``) or column (``xi_rho``) nearest the middle that has any,
    its endpoints the cells' lon/lat to 0.01 degree, ordered west to east for the row
    and south to north for the column.
    """
    import numpy as np

    def middle(wet: np.ndarray) -> int:
        have = np.flatnonzero(wet.any(axis=1))
        return int(have[np.argmin(np.abs(have - len(wet) // 2))])

    def ends(cells: np.ndarray, x: np.ndarray, y: np.ndarray, by) -> list[list[float]]:
        # cell index scaled to degrees by the line's median cell width
        step = np.hypot(np.diff(x) * np.cos(np.radians(y[:-1])), np.diff(y))
        wet = np.flatnonzero(cells)
        idx = wet[np.sort(_longest_run(wet * float(np.median(step)), LAND_GAP))]
        pts = [
            [round(float(x[i]), 2), round(float(y[i]), 2)] for i in (idx[0], idx[-1])
        ]
        return sorted(pts, key=by)

    j = middle(ocean)
    i = middle(ocean.T)
    row = ends(ocean[j], lon[j], lat[j], lambda p: p[0])
    column = ends(ocean[:, i], lon[:, i], lat[:, i], lambda p: p[1])
    out = {}
    for what, pts in (("row", row), ("column", column)):
        a, b = (f"{_lat_label(y)} {_lon_label(x)}" for x, y in pts)
        out[f"Middle grid {what} ({a} to {b})"] = {"transect": {"waypoints": pts}}
    return out


def _transect_lines(grid: Path, check) -> list[str]:
    """Return the ``transects:`` lines (a block under ``defaults:``) for ``grid``.

    An unreadable grid or one with no ocean is not an error: the block is left empty
    with a ``# TODO check`` and a commented example to fill in.
    """
    lines = [
        "  # The two sections of the transect pages, also drawn on the bathymetry map:",
        "  # each a fixed lat between two lons or a fixed lon between two lats (the",
        "  # grid's own longitude convention). Edit, add or remove lines freely.",
    ]

    def flow(entry: dict) -> str:
        def num(v: float) -> str:
            return "0" if v == 0 else f"{float(v):.1f}"

        def part(key: str, v) -> str:
            if key == "waypoints":
                pairs = ", ".join(f"[{x:.2f}, {y:.2f}]" for x, y in v)
                return f"{key}: [{pairs}]"
            if isinstance(v, dict):
                return f"{key}: {{min: {num(v['min'])}, max: {num(v['max'])}}}"
            return f"{key}: {num(v)}"

        spec = ", ".join(part(k, v) for k, v in entry["transect"].items())
        return f"{{transect: {{{spec}}}}}"

    try:
        found, rotation = _default_transects(grid)
    except (OSError, KeyError, ValueError) as exc:
        example = {"transect": {"lat": 0, "lon": {"min": 143.0, "max": 267.0}}}
        return [
            *lines,
            f"  transects: {{}}  {check(f'no section lines from {grid.name}: {exc}')}",
            f"    # Equator (143E-93W): {flow(example)}",
        ]
    if abs(rotation) > MAX_ROTATION:
        lines.append(
            f"  # The grid is rotated {abs(_whole(rotation))} degrees, so the lines "
            "follow its rows and columns."
        )
    lines.append("  transects:")
    for label, entry in found.items():
        key = label if yaml.safe_load(label) == label else json.dumps(label)
        lines.append(f"    {key}: {flow(entry)}")
    return lines


def _input_check(args, roots, name, output_dir, catalog, check, warn) -> list[str]:
    """Return the ``refresh:`` and ``defaults:`` lines of an input-check."""
    import textwrap
    from datetime import datetime, timedelta

    import xarray as xr

    from ocean_skill.plot._geo_labels import format_lat, format_lon

    folder = _abs(args.inputs) if args.inputs else _first_inputs(roots)
    grids = [_abs(args.grid)] if args.grid else sorted(folder.glob("*_grid.nc"))
    if len(grids) != 1 or not grids[0].is_file():
        raise SuiteNewError(f"expected one *_grid.nc in {folder}; pass --grid FILE")
    grid = grids[0]
    ics = sorted(folder.glob("*initial_conditions*.nc"))
    ics = ics or sorted(folder.glob("output_rst.*.nc"))
    if len(ics) != 1:
        raise SuiteNewError(f"expected one initial-conditions file in {folder}")
    ic, sources, forcing = ics[0], [], []

    def add(suffix: str, kind: str, files: str | None, *, label: str, **kw) -> str:
        """Add a forcing entry ``<name>_<suffix>`` (a warning if there are no files).

        ``label`` is how figures name the entry (its catalog metadata ``label``).
        """
        entry, refs = f"{name}_{suffix}", f"{output_dir}/refs/{name}_{suffix}.parquet"
        if files is None:
            warn.append(f"no {suffix} files in {folder}; its pages will fail")
        else:
            row = {"name": entry, "kind": kind, "files": files, "grid": str(grid)}
            forcing.append(
                row
                | ({} if kind == "initial_conditions" else {"ref": refs})
                | {"label": label}
                | kw
            )
        return entry

    if ic.name.startswith("output_rst"):  # a restart start: an ordinary model source
        sub, n_eta = pick_subchunk(grid)
        if "eta_rho" not in sub:
            warn.append(f"eta_rho={n_eta} has no divisor of {SUBCHUNK_RANGE[0]}+")
        sources.append(
            {"name": f"{name}_ic", "files": str(ic), "grid": str(grid)}
            | {"ref": f"{output_dir}/refs/{name}_ic.parquet"}
            | {"keep": "latest-per-file", "subchunk": sub}
            | {"label": "initial conditions"}
        )
    else:
        add("ic", "initial_conditions", str(ic), label="initial conditions")

    # open sides: the ones the boundary-physics file has temperature for
    phys = _by_glob(folder, "*boundary-physics_*.nc")
    first = next(iter(phys.values()), [None])[0]
    if first is None:
        raise SuiteNewError(f"no *boundary-physics_*.nc in {folder}")
    with xr.open_dataset(first, decode_times=False) as ds:
        forced = [s for s in SIDES if f"temp_{s}" in ds]
    # a side that is almost all land (a closed edge roms-tools still writes) gets no
    # page: under MIN_WET of its edge cells are water
    with xr.open_dataset(grid) as g:
        mask = g["mask_rho"]
        wet = {s: float(mask.isel({SIDES[s][0]: SIDES[s][1]}).mean()) for s in forced}
    open_sides = [s for s in forced if wet[s] >= MIN_WET]
    if dropped := [f"{s} ({wet[s]:.0%} water)" for s in forced if s not in open_sides]:
        warn.append(f"boundary sides left out as mostly land: {', '.join(dropped)}")
    # the BGC boundary files: monthly ones (dated), and the climatological one (twelve
    # months in one file, whose own ``month`` is picked by the pages)
    bgc_all = _by_glob(folder, "*boundary-bgc-*.nc")
    bgc = [k for k in bgc_all if not _is_climatology(k)]
    bgc_clim = [k for k in bgc_all if _is_climatology(k)]
    if len(bgc_clim) > 1:
        warn.append(f"{len(bgc_clim)} climatological BGC boundary files; using one")

    def bry() -> dict:  # a fresh dict per row: shared ones dump as YAML anchors
        return {"subchunk": {"bry_time": 1}, "sides": list(open_sides)}

    bry_phys = add(
        "bry_phys", "boundary", next(iter(phys)), label="boundary file", **bry()
    )
    bry_bgc = add(
        "bry_bgc", "boundary", next(iter(bgc), None), label="boundary file", **bry()
    )
    # one file, so there are no monthly chunks to join (no subchunk); with none, no
    # warning either: the climatology pages are simply absent
    bry_clim = f"{name}_bry_bgc_clim"
    if bgc_clim:
        add(
            "bry_bgc_clim",
            "boundary",
            bgc_clim[0],
            label="climatological boundary file",
            climatology=True,
            sides=list(open_sides),
        )

    with xr.open_dataset(grid) as g:
        lon, lat = g["lon_rho"].values, g["lat_rho"].values
    sides = []
    for side in open_sides:
        dim, i = SIDES[side]
        cut = (i, slice(None)) if dim == "eta_rho" else (slice(None), i)
        (x0, x1), (y0, y1) = lon[cut][[0, -1]], lat[cut][[0, -1]]
        label = (
            f"lon {format_lon(x0)} to {format_lon(x1)}, "
            f"lat {format_lat(y0)} to {format_lat(y1)}"
        )
        sides.append(
            {"name": side, "title": side.title(), "label": label}
            | {"transect": {dim: i}}
        )

    # surface files, dated or climatology: their 2-D variables, and a time variable
    # that is not called "time" (a CO2 file's own)
    surface: dict[bool, list[dict]] = {False: [], True: []}
    restoring: list[dict] = []  # salinity restoring files, which also get a WOA23 page
    for pattern in (
        "*surface-physics_*.nc",
        "*surface-bgc-*.nc",
        "*surface-restoring*.nc",
    ):
        for key, files in _by_glob(folder, pattern).items():
            filename = Path(key).name
            part = re.sub(
                r"([-_][?\d]+)?\.nc$", "", re.sub(r"^.*surface-", "", filename)
            )
            clim = part.endswith(("_clim", "-clim"))
            with xr.open_dataset(files[0], decode_times=False) as ds:
                names = [
                    str(v)
                    for v, a in ds.data_vars.items()
                    if {"eta_rho", "xi_rho"} <= set(a.dims)
                ]
                times = [
                    str(v) for v in ds.variables if "time" in str(v) and ds[v].ndim == 1
                ]
            kw = {"climatology": True} if clim else {}
            if times and "time" not in times and not clim:
                kw["time_coord"] = times[0]
            if part == "physics" and len(files) > 1:
                kw["subchunk"] = {"time": 1}  # monthly files of 28-31 steps must join
            label = _surface_label(part)
            entry = add(
                "surf_" + re.sub(r"\W", "_", part), "surface", key, label=label, **kw
            )
            if part.startswith("restoring") and "sss" in names:
                # drawn on its own page against WOA23, not again as a plain map
                restoring.append({"name": entry, "label": label, "var": "sss"})
            else:
                surface[clim].append({"name": entry, "label": label, "vars": names})

    rivers = sorted(folder.glob("*river*.nc"))
    add("river", "river", str(rivers[0]) if rivers else None, label="river forcing")

    # the first date: the restart's day (to the nearest), else a roms-tools initial
    # conditions file's own ini_time, else the first boundary month -- which can be a
    # month early, since boundary files are padded before the start
    stamp = re.search(r"output_rst\.(\d{14})", ic.name)
    month = re.search(r"_(\d{4})(\d{2})\.nc$", first.name)
    ini_time = None
    if not stamp:
        try:
            with xr.open_dataset(ic, decode_times=False) as ds:
                ini_time = ds.attrs.get("ini_time")
        except (OSError, ValueError):  # unreadable: fall back to the boundary month
            warn.append(f"could not read ini_time from {ic.name}; check forcing_time")
    if stamp:
        when = datetime.strptime(stamp[1], "%Y%m%d%H%M%S") + timedelta(hours=12)
    elif ini_time:
        when = datetime.fromisoformat(str(ini_time).strip())
    elif month:
        when = datetime(int(month[1]), int(month[2]), 1)
    else:
        raise SuiteNewError(f"no start date in {ic.name} or {first.name}")
    defaults = {"test": f"{name}_ic", "forcing_time": when.strftime("%Y-%m")}
    defaults |= {"forcing_month": when.month, "ic_month": when.month}
    defaults |= {"bry_phys": bry_phys, "bry_bgc": bry_bgc, "bry_bgc_clim": bry_clim}
    defaults |= {"river": f"{name}_river"}
    defaults |= {
        "sides": sides,
        "clim_sides": copy.deepcopy(sides) if bgc_clim else [],  # no YAML anchors
        "restoring": restoring,
        "surface": surface[False],
        "surface_clim": surface[True],
    }

    def dump(data: dict) -> str:
        text = yaml.safe_dump(
            data,
            sort_keys=False,
            default_flow_style=None,
            width=10**6,
            allow_unicode=True,  # "132°W", not "132\xB0W"
        )
        return textwrap.indent(text.rstrip(), "  ")

    refresh = {"catalog": catalog, "sources": sources, "forcing": forcing}
    return [
        *["", "refresh:", dump(refresh), "", "defaults:", dump(defaults)],
        *_transect_lines(grid, check),
    ]


def _mid_run(args, roots, name, output_dir, catalog, check, warn) -> list[str]:
    """Return the ``refresh:`` and ``defaults:`` lines of a mid-run-check."""
    globs = [str(_abs(g)) for g in args.files] if args.files else find_restarts(roots)
    grid = _abs(args.grid) if args.grid else find_grid(roots)
    if not grid.is_file():
        raise SuiteNewError(f"grid {grid} not found")
    sub, n_eta = pick_subchunk(grid)
    stamps = sorted(
        m.groups() for g in globs for p in glob.glob(g) if (m := _STAMP.search(p))
    )
    lines = [
        "",
        "refresh:",
        f"  catalog: {_q(catalog)}",
        "  sources:",
        f"    - name: {_q(name)}",
        f"      files: {json.dumps(globs) if len(globs) > 1 else _q(globs[0])}",
        f"      grid: {_q(grid)}",
        f"      ref: {_q(f'{output_dir}/refs/{name}.parquet')}",
        "      keep: latest-per-file  # restart files: keep each file's main time step",
    ]
    if "eta_rho" in sub:
        lines.append(
            f"      # {sub['eta_rho']}-row, 1-level chunks: depth pages read less."
        )
    else:
        why = f"eta_rho={n_eta} has no divisor of 32 or more: rows stay whole"
        lines.append(f"      # {why}: depth pages over many restarts need more memory.")
        warn.append(f"{why}; depth pages over many restarts use more memory")
    first = "-".join(stamps[0]) if stamps else None
    period = f'  period: {{min: "{first or "2000-01-01"}"}}'
    return [
        *lines,
        f"      subchunk: {json.dumps(sub)}",
        "",
        "defaults:",
        f"  test: {_q(name)}",
        period
        if first
        else f"{period}  {check('restart dates not read from file names')}",
        *_transect_lines(grid, check),
    ]


def render(
    args: argparse.Namespace, roots: list[Path], out: Path
) -> tuple[str, list[str], list[str]]:
    """Return the suite text, its ``# TODO check`` items and its warnings."""
    from ocean_skill.workflows import cover

    tasks = sorted(t for r in roots for t in (r / "tasks").glob("*"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # several workplans: the user names one
        found = {
            "workplan": next(filter(None, map(cover.find_workplan, tasks)), None),
            "blueprint": next(filter(None, map(cover.newest_blueprint, tasks)), None),
        }

    todo, warn = [], []

    def check(text: str) -> str:
        todo.append(text)
        return f"# TODO check: {text}"

    name = re.sub(r"[^A-Za-z0-9_]+", "_", args.name).strip("_")
    output_dir = str(_abs(args.output_dir))
    catalogs = (
        str(_abs(args.catalog_dir)) if args.catalog_dir else f"{output_dir}/catalogs"
    )
    catalog = f"{catalogs}/{name.replace('_', '-')}.yaml"
    lines = [
        f"name: {args.template}",
        f"output_dir: {_q(output_dir)}",
        "pdf_images: jpeg",
        f"catalog_search_paths: [{_q(args.shared_catalogs)}, {_q(catalogs)}]",
        f"cache_dir: {_q(output_dir)}",
    ]
    if roots:
        lines += ["", "run_info:"]
    for key, path in found.items() if roots else ():
        lines.append(
            f"  {key}: {_q(path)}"
            if path
            else f"  {check(f'run_info.{key} not found under {roots[0]}')}"
        )
    build = _input_check if args.template == "input-check" else _mid_run
    lines += build(args, roots, name, output_dir, catalog, check, warn)

    template = TEMPLATE.with_name(f"{args.template}.yaml")
    text = template.read_text()
    opened = re.search(r"^# >>> setup.*\n", text, re.MULTILINE)
    closed = re.search(r"^# <<< setup.*\n", text, re.MULTILINE)
    if opened is None or closed is None:
        raise SuiteNewError(f"{template} has no setup markers")
    head = text[: opened.start()].replace(
        f"ocean-skill-run {args.template}.yaml", f"ocean-skill-run {out}"
    )
    note = (
        f"# Generated by `ocean-skill-suite new {args.template}`; the description "
        "below is the template's.\n"
    )
    block = opened.group(0) + "\n".join(lines) + "\n" + closed.group(0)
    return note + head + block + text[closed.end() :], todo, warn


def validate(path: Path) -> list[str]:
    """Return the problems with the written suite: it must load, its paths exist."""
    from ocean_skill.config import SuiteConfig

    try:
        suite = SuiteConfig.model_validate(yaml.safe_load(path.read_text()))
    except Exception as exc:
        return [f"the suite does not load: {exc}"]
    problems = []
    for src in [*suite.refresh.sources, *suite.refresh.forcing]:
        files = [src.files] if isinstance(src.files, str) else src.files
        problems += [f"no files match {g}" for g in files if not glob.glob(g)]
        if src.grid and not Path(src.grid).is_file():
            problems.append(f"grid {src.grid} not found")
    for key in ("workplan", "blueprint"):
        value = getattr(suite.run_info, key, None)
        if value and not Path(value).is_file():
            problems.append(f"run_info.{key} {value} not found")
    return problems


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ocean-skill-suite")
    new = parser.add_subparsers(dest="command", required=True).add_parser(
        "new",
        help="write a mid-run-check or input-check suite from a C-Star run folder",
    )
    add = new.add_argument
    add("template", help="the suite to start from: mid-run-check or input-check")
    add(
        "--run",
        action="append",
        default=[],
        metavar="DIR",
        help="C-Star run folder (holds tasks/); repeat for segments",
    )
    add("--inputs", metavar="DIR", help="input-check: the input_datasets folder itself")
    add("--name", required=True, help="the model source's name")
    add("--output-dir", required=True, help="reports and cache go here")
    add("--files", nargs="+", metavar="GLOB", help="the restarts, if not found")
    add("--grid", metavar="FILE", help="the grid, if not found")
    add("--catalog-dir", help="this suite's catalog (default <output-dir>/catalogs)")
    add("--shared-catalogs", default=SHARED_CATALOGS)
    add("-o", "--out", help="file to write (default <template>.yaml)")
    add("--force", action="store_true", help="overwrite an existing file")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run ``ocean-skill-suite new TEMPLATE --run DIR --name N --output-dir DIR``.

    Returns 0 when the suite was written and checks out, 1 when it was written but the
    check found problems, 2 when nothing could be written.
    """
    try:
        args = _build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 2
    try:
        if args.template not in TEMPLATES:
            raise SuiteNewError(
                f"template {args.template!r} is not supported yet; copy "
                f"ocean_skill/workflows/templates/{args.template}.yaml and edit its "
                "setup block (between '# >>> setup' and '# <<< setup') by hand"
            )
        out, roots = (
            _abs(args.out or f"{args.template}.yaml"),
            [_abs(r) for r in args.run],
        )
        if not roots and not (args.inputs and args.template == "input-check"):
            raise SuiteNewError(
                "--run DIR is required (input-check also takes --inputs)"
            )
        if out.exists() and not args.force:
            raise SuiteNewError(f"{out} exists; pass --force to overwrite it")
        if missing := [r for r in roots if not r.is_dir()]:
            raise SuiteNewError(f"{missing[0]} is not a directory")
        text, todo, warn = render(args, roots, out)
    except SuiteNewError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)

    problems = validate(out)
    print(f"Wrote {out}" + (f" ({len(todo)} marked '# TODO check')." if todo else "."))
    for line in warn:
        print(f"warning: {line}", file=sys.stderr)
    for line in problems:
        print(f"problem: {line}", file=sys.stderr)
    print(f"Next:\n  ocean-skill-run {out} --list")
    print(f"  ocean-skill-run {out} --smoke 2 --self-check --strict")
    print(f"  ocean-skill-run {out}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
