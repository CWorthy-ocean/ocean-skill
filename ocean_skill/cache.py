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

**The key is identity, not content.** It is a hash of the two source names, a
fingerprint of each one's catalog *definition* (:func:`ocean_skill.catalog.fingerprint`:
the reader as written, its paths/URLs and transforms, and the metadata that changes what
a read returns), the variable, the selection, and the regrid method (the plan's
``f(sources, variable, select, align-mode)``) — deliberately *not* of the underlying
data, which would mean reading the very files the cache exists to avoid. So an entry
redefined under the same name misses cleanly, but if a model run is rewritten in place
at the same catalog path, the cache will happily serve the old result. Call
:func:`clear` after rerunning a model, or pass ``cache=False``. Anything under
:func:`path` is reproducible and safe to delete at any time.
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
#:
#: **4** — two more keys now answer a different question under the same spelling.
#: A curvilinear (ROMS) lane's ``select`` naming a lon/lat *box* used to fall
#: through untouched (see :func:`ocean_skill.operators.box_in_spec`); it now
#: really crops. And an ``_aggregate`` reducing both horizontal axes by a plain
#: ``"mean"`` used to be two sequential unweighted means (a silent no-op on a
#: curvilinear grid); it is now one area-weighted joint reduction (see
#: :func:`ocean_skill.operators.spatial_mean_in_spec`). Either key means the
#: same thing it always did on a rectilinear select with no box and no spatial
#: mean, but a stale entry filed under either new shape held the old answer.
#:
#: **5** — a ``select`` naming no vertical key at all used to mean "surface" for
#: every caller of :func:`ocean_skill.comparison.prepare_source`; now it means
#: "leave the vertical axis alone" (see
#: :func:`ocean_skill.comparison._prepare`'s ``surface`` flag), and it is the
#: *compare* lane (:meth:`ocean_skill.comparison.Comparison._prepare_lane`) that
#: writes an explicit ``"surface"`` into its own select before ever reaching
#: here. A pre-change entry filed under a depth-less key held surface-reduced
#: data; a :class:`~ocean_skill.field.Field`'s bare call now asks that same key
#: for the whole column instead, and must not be handed the old answer.
#:
#: **6** — a time ``groupby`` result's new dimension now carries a deliberate
#: marker (:data:`ocean_skill.operators.TIME_GROUPBY_ATTR`) so
#: :meth:`ocean_skill.field.Field.plot` can still find "time" once
#: ``aggregate={"time": {"groupby": "month", ...}}`` has renamed it (see
#: :func:`ocean_skill.operators.time_axis_dim`); a season groupby's coordinate
#: attrs are now stripped for the same reason (:mod:`ocean_skill.operators`'s
#: ``_reduce_dim``). Neither changes the *value* a lane holds, only the attrs
#: on its groupby coordinate -- but a pre-change entry lacks the marker
#: entirely, so a stale hit would silently keep the old refuses-to-plot
#: behaviour rather than the new one.
#:
#: **7** — every aggregate step now stamps a CF ``cell_methods`` entry and an
#: internal ``statistic`` attr on its result, and a spread reduction (``var``,
#: ``std``, ``range``) rewrites ``units`` through
#: :func:`ocean_skill.units.for_statistic` (``degC`` -> ``delta_degC^2`` for a
#: variance) instead of copying the field's own. A pre-change entry holds the same
#: values under the wrong units, and the unit conversion that reads them would
#: shift a temperature variance by 273.15.
#:
#: **8** — a lane's *meaning* changed under an otherwise identical key, in two
#: ways. Depth matching now follows a declared/inferred/default **depth convention**
#: (:mod:`ocean_skill.depth_convention`): a surface-referenced observation (a CTD
#: cast, a pressure record -- the default for a profile) is matched *below the moving
#: free surface* rather than at a fixed height, a nearest-level pick is made *per
#: time step* instead of once at a reference time, a target in the top or bottom
#: half-cell is edge-filled with that cell's value instead of NaN, and the free
#: surface is kept through time aggregates, detiding and transects so the frame the
#: depth is read in is the one the field was reduced under. And a source's naive
#: timestamps can now be declared local (:mod:`ocean_skill.time_zone`) rather than
#: always read as UTC. Wherever the key does not itself move (a ``select`` naming no
#: ``depth_origin``, a source declaring nothing new) a pre-change entry still holds
#: the old answer under it, and a stale hit would silently keep it.
#:
#: **9** — a source's *name* stopped being the whole of its identity in a key. Every key
#: now also carries a fingerprint of the catalog entry's **definition**
#: (:func:`ocean_skill.catalog.fingerprint`: the reader as written -- class, arguments,
#: URLs and paths, chained transforms, the data it references -- plus the metadata that
#: changes what a read returns), because an entry redefined under the same name
#: (``cast0000`` rewritten to point at a different CSV, ``model_win`` at a different
#: window through a reader chain) hashed to the very same key and was silently handed
#: the previous definition's result. The stored layout is untouched and nothing fails
#: loudly, which is the case this counter exists for; every version-8 entry was filed
#: without a definition, can no longer be asked for, and is orphaned. A key built with
#: no definition (``""``, a source that is not a catalog entry) is still valid, but it
#: never collides with a fingerprinted one.
#:
#: A related fix that did *not* bump this: :func:`ocean_skill.sources.read`'s
#: singleton-horizontal squeeze (giving an ADCP-shaped station a recoverable
#: scalar lon/lat) changed what a *fresh* read produces without changing what
#: an already-cached lane holds. Bumping the version here would have orphaned
#: every prepared/aligned entry in the cache, not just the (rare) positionless
#: ones the fix was for -- an expensive, indiscriminate way to fix a handful
#: of entries. Instead, :func:`ocean_skill.comparison.prepare_source` checks a
#: cache hit for exactly this shape on the way out (see
#: ``_is_stale_positionless_station`` there) and discards only an entry that
#: actually lacks a position, recomputing and overwriting just that one.
_FORMAT_VERSION = 9

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


#: The three things worth caching, each its own directory under ``<base>/cache``.
#:
#: ``prepared`` is one file per *lane*: one source reduced to a single comparable 2-D
#: field (variable resolved, time-averaged, vertically interpolated, units converted)
#: — keyed on ``(source, variable, select)`` alone, with no reference and no regrid
#: method in it. ``aligned`` is one file per *pair*, the regridded test+reference+
#: difference. ``weights`` is one file per xesmf regridder — the ESMF weight matrix
#: itself (see :func:`ocean_skill.align._regridder_for`), the single most expensive
#: step in the whole pipeline to recompute.
#:
#: All three earn their place. The aligned entry is the fast path: one read serves a
#: whole repeat plot with no regridding. The prepared entries make a *miss* cheap —
#: comparing one model against several references, or at several regrid methods,
#: otherwise re-reads and re-transforms that model's lane once per pair, and the
#: vertical transform is expensive too. The weights entry makes even a prepared/aligned
#: *miss* cheap: rebuilding a Dataset from a fresh model run still regrids onto the
#: same GLODAP or WOA grid it always has, and ESMF weight generation, not the transform,
#: is what actually takes minutes on a fine ROMS grid.
#:
#: ``weights`` is keyed differently from the other two -- see
#: :func:`ocean_skill.align._regridder_for`: its key is a content hash of the two
#: grids' own coordinates plus the method, not an "identity" like a source name, so the
#: "not of the underlying data" caveat above does not apply to it -- two numerically
#: identical grids always hit, from any source name, and a grid that changed shape or
#: values always misses.
#:
#: Downloaded source files are *not* a fourth kind, even though by default they sit
#: right beside these three, in ``<base>/cache/obs`` (see :func:`obs_dir`). They are
#: kept out of this tuple on purpose; :func:`clear` explains why.
#:
#: ``calculated`` holds a calculator's own expensive intermediate -- every constituent
#: of a tidal harmonic analysis, say -- so several fields drawn from one result (K1,
#: then M2) pay for it once. Keyed by identity, like ``prepared``
#: (:func:`key_for_calculated`).
KINDS = ("prepared", "aligned", "weights", "calculated")

#: File extension each :data:`KINDS` entry is stored under -- the two Dataset kinds as
#: zarr stores (a directory), regridder weights as the plain netCDF file
#: ``regridder.to_netcdf`` writes. :func:`entries` and :func:`clear` need this to find
#: and remove the right thing; a weights entry is not a Dataset and never goes through
#: :func:`load`/:func:`save`.
_EXTENSIONS = {
    "prepared": "zarr",
    "aligned": "zarr",
    "weights": "nc",
    "calculated": "zarr",
}


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
    test_definition: str = "",
    reference_definition: str = "",
) -> str:
    """Return the cache key for one aligned comparison.

    Stable across processes and across dict ordering (``sort_keys``), and
    ``default=str`` keeps values a plain ``json.dumps`` would choke on — a numpy
    float depth, a ``slice`` in a selection — from raising instead of hashing.

    ``test_definition`` / ``reference_definition`` are each side's
    :func:`ocean_skill.catalog.fingerprint` (``""`` for a source with none): what lets
    an entry redefined under the same name miss instead of returning the old
    definition's result. They are separate fields, so swapping the two sides'
    definitions is a different key, as swapping the two names is.
    """
    payload = json.dumps(
        {
            "v": _FORMAT_VERSION,
            "test": test,
            "reference": reference,
            "variable": variable,
            "select": select,
            "method": method,
            "test_definition": test_definition,
            "reference_definition": reference_definition,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def key_for_prepared(
    *, source: str, variable: Any, select: dict[str, Any], definition: str = ""
) -> str:
    """Return the cache key for one *lane*: a single source reduced to a 2-D field.

    Deliberately excludes the other source and the regrid method — that is the whole
    point of the lane layer. The same model, variable and depth reduce to the same
    field whether it is about to be compared against WOA, against GLODAP, or with a
    different regridder, so all of those should hit one entry.

    ``definition`` is the source's own :func:`ocean_skill.catalog.fingerprint` (``""``
    for a source with none) -- the source's, and only the source's: the other side's
    definition has no more business in a lane's key than its name does.
    """
    payload = json.dumps(
        {
            "v": _FORMAT_VERSION,
            "source": source,
            "variable": variable,
            "select": select,
            "definition": definition,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def key_for_calculated(
    *, source: str, name: str, params: dict[str, Any], definition: str = ""
) -> str:
    """Return the cache key for one calculator intermediate (see :data:`KINDS`).

    ``params`` is whatever decides the result besides the source: the calculator's
    own options and the time coverage it actually saw, so a narrower window misses.
    ``definition`` is the source's :func:`ocean_skill.catalog.fingerprint` (``""`` for
    none): a redefined entry is a different source, whatever it is still called.
    """
    payload = json.dumps(
        {
            "v": _FORMAT_VERSION,
            "source": source,
            "name": name,
            "params": params,
            "definition": definition,
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
        "   Keyed on source definition/variable/selection, NOT file contents — clear "
        "it after rerunning a model in place.)"
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


def _check_kind(kind: str | None) -> None:
    """Raise :class:`ValueError` unless ``kind`` is ``None`` or one of :data:`KINDS`.

    The check goes by *name*, never by what happens to be on disk. Before it,
    :func:`entries` only found out by indexing ``_EXTENSIONS``, which it reached only
    for a directory that existed. So a bad name gave a bare ``KeyError`` when its
    directory existed, and a silent empty result when it did not: ``clear("obs")`` on
    a fresh machine, or a typo like ``"align"`` anywhere, "succeeded" at removing
    nothing.
    """
    if kind is None or kind in KINDS:
        return
    raise ValueError(
        f"unknown cache kind {kind!r}; try one of {list(KINDS)}, or None for all of "
        "them. Downloaded source files are not a kind, and clear() never touches "
        f"them: they are in osk.cache.obs_dir() ({obs_dir()}), to delete by hand if "
        "you want the space back."
    )


def entries(kind: str | None = None) -> list[Path]:
    """Return cached entries on disk, for one :data:`KINDS` kind or all of them.

    Only ``None`` means all of them. Any name outside :data:`KINDS` raises
    :class:`ValueError`, ``"obs"`` included; :func:`clear` explains why downloaded
    source files are not a kind.
    """
    _check_kind(kind)
    found: list[Path] = []
    for k in KINDS if kind is None else [kind]:
        root = path(k)
        if root.exists():
            found.extend(sorted(root.glob(f"*.{_EXTENSIONS[k]}")))
    return found


def _remove_entry(entry: Path) -> None:
    """Delete one cache entry, a zarr store (directory) or a weights file, either way.

    ``shutil.rmtree`` on a plain file silently does nothing even with
    ``ignore_errors=True`` -- exactly the failure mode a weights entry would hit if
    :func:`clear` kept using it unconditionally, the way it could when every entry was
    a zarr store.
    """
    if entry.is_dir():
        shutil.rmtree(entry, ignore_errors=True)
    else:
        entry.unlink(missing_ok=True)


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

    Downloaded source files are not one of :data:`KINDS`, so even a plain ``clear()``
    never touches them, and ``clear("obs")``, like any name outside :data:`KINDS`,
    raises :class:`ValueError` rather than guessing. A download is a copy of a remote
    file keyed on its URL (in practice an observational reference), so the model rerun
    this call exists for does not make it stale. Downloads are also the bigger
    directory and slow to refill, and :func:`obs_dir` can be a directory the user
    pointed fsspec at themselves (a ``cache_storage`` setting), holding files that are
    not ocean-skill's to delete. To reclaim that space, delete from :func:`obs_dir` by
    hand.

    Also empties :func:`ocean_skill.sources.read`'s own in-process open memo
    (whichever kind is named, if any) — it holds an already-opened,
    already-standardized source, so a rerun-in-place this call exists for needs
    that memo gone too, or a since-edited source would keep being served from
    before the rerun. A refused ``kind`` raises first, so it flushes nothing
    either. Imported locally: :mod:`ocean_skill.sources` imports from
    this module's sibling :mod:`ocean_skill.catalog`, not from here, but importing
    it at this module's top would still invite a cycle as the package grows.
    """
    from ocean_skill import sources as _sources

    # List first: a refused kind raises here, before the memo below is flushed.
    found = entries(kind)
    _sources.read.cache_clear()
    for entry in found:
        _remove_entry(entry)
    return len(found)
