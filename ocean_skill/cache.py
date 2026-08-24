"""On-disk cache for aligned comparison results, so repeats are reads not recomputes.

:meth:`ocean_skill.comparison.Comparison.align` is by far the expensive step in the
pipeline: it opens both sources (often remote OPeNDAP), reduces each to one 2-D field
(time mean, and for ROMS an xgcm s-coord → z transform), then regrids test onto
reference with xesmf. The result is one small :class:`xarray.Dataset` — test,
reference, difference, coverage. Recomputing that to redraw the same figure with a
bigger font, or because a notebook kernel restarted, costs minutes and buys nothing.

Enabled by default; :func:`disable` turns it off globally, ``cache=False`` per call.
The first time it is used in a process it prints where it lives and how to turn it
off, so it is never silently doing work behind your back.

**The key is identity, not content.** It is a hash of the two source names, the
variable, the selection, and the regrid method (the plan's
``f(sources, variable, select, align-mode)``) — deliberately *not* of the underlying
data, which would mean reading the very files the cache exists to avoid. So if a
model run is rewritten in place at the same catalog path, the cache will happily
serve the old result. Call :func:`clear` after rerunning a model, or pass
``cache=False``. Anything under :func:`path` is reproducible and safe to delete at
any time.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import warnings
from pathlib import Path
from typing import Any

from ocean_skill._display import Text

__all__ = [
    "KINDS",
    "clear",
    "disable",
    "enable",
    "enabled",
    "entries",
    "info",
    "obs_dir",
    "path",
]

#: Bumped when the stored layout changes in a way that makes old entries wrong to
#: reuse. Part of every key, so a bump orphans stale entries instead of loading
#: them into a pipeline that now expects something different.
#:
#: **2** — removing the default aggregation (see
#: :data:`ocean_skill.comparison.NO_AGGREGATION`) changed what an existing key *means*
#: without changing the key: a lane keyed on ``_aggregate: None`` held a time *mean*
#: before and holds every step after, so every pre-change entry now answers a different
#: question than the one it was filed under. That is the case this counter exists for,
#: and the one it is easiest to miss — the stored layout is untouched and nothing fails
#: loudly; a stale hit simply returns a single map where the caller now expects an axis,
#: and the error surfaces somewhere else entirely (a movie with nothing to animate).
#:
#: **3** — spatial alignment now regrids onto the *coarser* of the two grids instead
#: of the reference's unconditionally (see :func:`ocean_skill.align._regrid_target`).
#: A pre-change entry for a coarse-test/fine-reference pair sits on the reference's
#: fine grid; the same key today would resolve onto the test's coarse one instead, so
#: reusing it silently hands a caller the wrong grid rather than recomputing.
_FORMAT_VERSION = 3

#: Zarr stores variables in its own (alphabetical) order, so a round trip would
#: otherwise hand back ``coverage, difference, reference, test`` where the pipeline
#: built ``test, reference, difference, coverage``. Nothing downstream indexes by
#: position today — but "a cached result behaves exactly like a fresh one" is the
#: invariant worth keeping, rather than one every future caller has to know about.
_ORDER_ATTR = "_osk_var_order"

_enabled = True
_override_dir: Path | None = None
_announced = False


def base_dir() -> Path:
    """Return ocean-skill's base directory: override -> ``$OCEAN_SKILL_DIR`` -> default.

    The default is ``platformdirs``' user cache dir, which is the conventional home
    for regenerable data on each platform and is what gets cleaned up by OS tooling.
    """
    if _override_dir is not None:
        return _override_dir
    env = os.environ.get("OCEAN_SKILL_DIR")
    if env:
        return Path(env).expanduser()
    import platformdirs

    return Path(platformdirs.user_cache_dir("ocean-skill"))


#: fsspec caches that a catalog URL can invoke with a ``<protocol>::`` prefix.
_FSSPEC_CACHES = ("simplecache", "blockcache", "filecache")


#: What :func:`configure_fsspec_cache` last wrote, per protocol. A value fsspec is
#: still carrying from us is ours to move; anything else came from the user and is
#: left alone. See the ``relocate`` note below.
_fsspec_applied: dict[str, str] = {}


def configure_fsspec_cache(*, relocate: bool = False) -> None:
    """Point fsspec's file caches at ocean-skill's cache directory.

    A catalog says *what* to read; where a downloaded copy lands is a property of the
    machine, not of the dataset. Left unset, fsspec caches to a temp dir that is wiped
    between sessions, so the location has to come from somewhere — and baking it into
    the catalog put one developer's absolute home path into all 78 WOA entries.

    Never a plain assignment: fsspec applies ``~/.config/fsspec/*.json`` and
    ``FSSPEC_*`` environment variables at *its* import, which happens before this runs.
    Overwriting would silently undo someone who pointed the cache at HPC scratch rather
    than a home quota.

    ``relocate`` is for :func:`enable` moving the base directory mid-process. Setting
    the location once at import was not enough: the caller saw :func:`info` report the
    new directory while downloads kept landing in the old one, because only the two
    result caches had actually moved. So a value we set earlier gets rewritten to the
    new base — but a value the *user* set still wins, which is why this tracks what it
    wrote rather than overwriting whatever it finds.
    """
    import fsspec.config

    target = str(base_dir() / "cache" / "obs")
    for protocol in _FSSPEC_CACHES:
        conf = fsspec.config.conf.setdefault(protocol, {})
        current = conf.get("cache_storage")
        ours = current is None or (
            relocate and current == _fsspec_applied.get(protocol)
        )
        if not ours:
            continue
        conf["cache_storage"] = target
        _fsspec_applied[protocol] = target


def obs_dir() -> Path:
    """Return the directory downloaded source files land in.

    Both download paths resolve here — fsspec's ``simplecache::`` URLs via
    :func:`configure_fsspec_cache`, and :class:`ocean_skill.readers.PoochTarNetCDF`
    via its ``cache_dir`` default — so there is one answer to "where did that file
    go?" and it moves with the base directory.

    If the user pointed fsspec somewhere themselves, that is the honest answer and it
    is what gets reported and what pooch follows: this must not claim a location that
    downloads are not actually using, which is the bug it exists to close.
    """
    import fsspec.config

    conf = fsspec.config.conf.get("simplecache") or {}
    current = conf.get("cache_storage")
    if current and current != _fsspec_applied.get("simplecache"):
        return Path(current).expanduser()
    return base_dir() / "cache" / "obs"


configure_fsspec_cache()


#: The two things worth caching, each its own directory under ``<base>/cache``.
#:
#: ``prepared`` is one file per *lane*: one source reduced to a single comparable 2-D
#: field (variable resolved, time-averaged, vertically interpolated, units converted)
#: — keyed on ``(source, variable, select)`` alone, with no reference and no regrid
#: method in it. ``aligned`` is one file per *pair*, the regridded test+reference+
#: difference.
#:
#: Both earn their place. The aligned entry is the fast path: one read serves a whole
#: repeat plot with no regridding. The prepared entries make a *miss* cheap — comparing
#: one model against several references, or at several regrid methods, otherwise
#: re-reads and re-transforms that model's lane once per pair, and the vertical
#: transform is the most expensive step in the pipeline.
KINDS = ("prepared", "aligned")


def path(kind: str = "aligned") -> Path:
    """Return the directory holding cached results of one :data:`KINDS` kind."""
    return base_dir() / "cache" / kind


def enable(directory: str | Path | None = None) -> None:
    """Turn caching on (the default), optionally relocating it to ``directory``."""
    global _enabled, _override_dir
    _enabled = True
    if directory is not None:
        _override_dir = Path(directory).expanduser()
        # Downloads have to follow, or the move is only half done — see the
        # ``relocate`` note in configure_fsspec_cache.
        configure_fsspec_cache(relocate=True)


def disable() -> None:
    """Turn caching off for this process; nothing is read from or written to disk."""
    global _enabled
    _enabled = False


def enabled() -> bool:
    """Report whether caching is currently on."""
    return _enabled


def key_for(
    *,
    test: str,
    reference: str,
    variable: Any,
    select: dict[str, Any],
    method: str,
) -> str:
    """Return the cache key for one aligned comparison.

    Stable across processes and across dict ordering (``sort_keys``), and
    ``default=str`` keeps values a plain ``json.dumps`` would choke on — a numpy
    float depth, a ``slice`` in a selection — from raising instead of hashing.
    """
    payload = json.dumps(
        {
            "v": _FORMAT_VERSION,
            "test": test,
            "reference": reference,
            "variable": variable,
            "select": select,
            "method": method,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def key_for_prepared(*, source: str, variable: Any, select: dict[str, Any]) -> str:
    """Return the cache key for one *lane*: a single source reduced to a 2-D field.

    Deliberately excludes the other source and the regrid method — that is the whole
    point of the lane layer. The same model, variable and depth reduce to the same
    field whether it is about to be compared against WOA, against GLODAP, or with a
    different regridder, so all of those should hit one entry.
    """
    payload = json.dumps(
        {
            "v": _FORMAT_VERSION,
            "source": source,
            "variable": variable,
            "select": select,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _announce() -> None:
    """Print where the cache lives and how to turn it off — once per process."""
    global _announced
    if _announced:
        return
    _announced = True
    print(
        f"ocean-skill: caching aligned results in {path()}\n"
        "  (reused automatically on repeat; osk.cache.disable() to turn off, "
        "osk.cache.clear() to empty.\n"
        "   Keyed on source/variable/selection, NOT file contents — clear it after "
        "rerunning a model.)"
    )


def load(key: str, kind: str = "aligned"):
    """Return the cached Dataset for ``key``, or ``None`` on a miss.

    A corrupt or unreadable entry is a miss, not an error: it warns and is deleted so
    the caller recomputes and rewrites it. A cache should never be able to break a
    pipeline that would otherwise have worked.
    """
    if not _enabled:
        return None
    import xarray as xr

    store = None
    try:
        store = path(kind) / f"{key}.zarr"
        if not store.exists():
            return None
        # consolidated=False to match save(); left at its default, xarray tries the
        # consolidated metadata first and warns loudly when it falls back.
        ds = xr.open_zarr(store, consolidated=False).load()
    except Exception as exc:  # unreadable/corrupt/half-written entry, or no cache dir
        warnings.warn(
            f"ignoring unreadable cache entry {store or key} ({exc}); recomputing.",
            stacklevel=2,
        )
        if store is not None:
            shutil.rmtree(store, ignore_errors=True)
        return None
    order = [v for v in ds.attrs.pop(_ORDER_ATTR, []) if v in ds.data_vars]
    if order:
        ds = ds[[*order, *(v for v in ds.data_vars if v not in order)]]
        ds.attrs.pop(_ORDER_ATTR, None)  # indexing re-attaches the parent's attrs
    _announce()
    return ds


#: Encoding keys that carry a codec object. Datasets read from a source zarr store
#: inherit these (e.g. a ``numcodecs.blosc.Blosc`` compressor), and zarr v3's codec
#: pipeline rejects the v2-style objects on write ("Expected a BytesBytesCodec"). We
#: never want the source's codecs on our small cache stores anyway, so drop them and
#: let zarr choose its own v3 defaults. dtype/chunks/fill_value are left intact.
_CODEC_ENCODING_KEYS = ("compressor", "compressors", "filters", "serializer", "codecs")


def _strip_codec_encoding(ds):
    """Return ``ds`` with inherited codec encoding removed from every variable.

    Works on a shallow copy so the caller's live dataset keeps its encoding: xarray's
    shallow copy gives each variable an independent ``encoding`` dict over shared data.
    """
    out = ds.copy(deep=False)
    for var in (*out.variables.values(),):
        for k in _CODEC_ENCODING_KEYS:
            var.encoding.pop(k, None)
    return out


def save(key: str, ds, kind: str = "aligned") -> None:
    """Write ``ds`` to the cache under ``key``, replacing any existing entry.

    Failures warn rather than raise, for the same reason as :func:`load`: the result
    is already computed and correct, and being unable to *cache* it is not a reason
    to fail the caller's work. Writes to a temporary path and moves it into place, so
    an interrupted write can't leave a half-written entry to be read back later.
    """
    if not _enabled:
        return
    tmp = None
    try:
        # Inside the try: resolving the directory can itself fail (an unset HOME, an
        # $OCEAN_SKILL_DIR that cannot be expanded), and that must warn like any
        # other cache failure rather than propagate into the caller's pipeline.
        store = path(kind) / f"{key}.zarr"
        tmp = path(kind) / f".{key}.tmp.zarr"
        store.parent.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(tmp, ignore_errors=True)
        # assign_attrs rather than mutating: `ds` is the live object the caller is
        # still holding, and it should not sprout a private bookkeeping attr.
        # consolidated=False because zarr v3 warns that consolidated metadata is
        # outside its spec; these stores are small, so the read cost is noise.
        out = _strip_codec_encoding(ds.assign_attrs({_ORDER_ATTR: list(ds.data_vars)}))
        out.to_zarr(tmp, mode="w", consolidated=False)
        shutil.rmtree(store, ignore_errors=True)
        os.replace(tmp, store)
    except Exception as exc:
        warnings.warn(f"could not cache aligned result ({exc}).", stacklevel=2)
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
        return
    _announce()


#: Name the lane field is stored under. Its real name (whatever spelling the source
#: used) and its actual_depth ride in the attrs, so the DataArray reconstructs exactly.
_FIELD_VAR = "field"
_NAME_ATTR = "_osk_field_name"
_DEPTH_ATTR = "actual_depth"


def save_field(key: str, da, actual_depth: float | None) -> None:
    """Cache one lane's prepared 2-D field (a DataArray, not a Dataset).

    ``actual_depth`` is the observational level actually selected, which
    :class:`~ocean_skill.comparison.Comparison` carries separately from the array and
    reports in its metrics — so it has to survive the round trip too, or a cached run
    would write a subtly different metrics row than the run that filled the cache.
    """
    ds = da.to_dataset(name=_FIELD_VAR)
    ds.attrs[_NAME_ATTR] = da.name or _FIELD_VAR
    if actual_depth is not None:
        ds.attrs[_DEPTH_ATTR] = actual_depth
    save(key, ds, kind="prepared")


def load_field(key: str):
    """Return ``(DataArray, actual_depth)`` for a cached lane, or ``None`` on a miss."""
    ds = load(key, kind="prepared")
    if ds is None:
        return None
    da = ds[_FIELD_VAR].rename(ds.attrs.get(_NAME_ATTR, _FIELD_VAR))
    # Dataset-level bookkeeping must not leak onto the array the pipeline sees.
    da.attrs.pop(_NAME_ATTR, None)
    return da, ds.attrs.get(_DEPTH_ATTR)


def entries(kind: str | None = None) -> list[Path]:
    """Return cached entries on disk, for one :data:`KINDS` kind or all of them."""
    found: list[Path] = []
    for k in [kind] if kind else KINDS:
        root = path(k)
        if root.exists():
            found.extend(sorted(root.glob("*.zarr")))
    return found


def _size_bytes() -> int:
    return sum(
        f.stat().st_size
        for k in KINDS
        if path(k).exists()
        for f in path(k).rglob("*")
        if f.is_file()
    )


def info() -> Text:
    """Return a human-readable summary: state, location, entry counts, size on disk.

    Downloads are reported alongside the results, and separately: they are the bigger
    directory, and reporting only the results location once let a relocation look
    complete while downloads carried on landing somewhere else.
    """
    state = "on" if _enabled else "off"
    root = base_dir() / "cache"
    counts = {k: len(entries(k)) for k in KINDS}
    if not sum(counts.values()):
        head = f"ocean-skill cache: {state}, empty ({root})"
    else:
        breakdown = ", ".join(f"{n} {k}" for k, n in counts.items() if n)
        size = _size_bytes() / 1e6
        head = f"ocean-skill cache: {state}, {breakdown}, {size:.1f} MB ({root})"
    return Text(f"{head}\n  downloaded sources: {obs_dir()}")


def clear(kind: str | None = None) -> int:
    """Delete cached entries and return how many were removed.

    The thing to run after rerunning a model in place — see the module docstring on
    why identity-keyed entries cannot notice that themselves. Clears every kind
    unless one is named.
    """
    found = entries(kind)
    for store in found:
        shutil.rmtree(store, ignore_errors=True)
    return len(found)
