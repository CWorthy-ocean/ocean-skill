"""Catalog auto-discovery and source-name resolution.

Catalogs are native **intake v2** YAML files that ocean-skill discovers on a search path
and merges into a single namespace. Discovery loads each catalog with
``intake.from_yaml_file`` and indexes its entries by name (with their per-entry
``metadata`` — our contract: featureType, standard_names, extents). Opening an entry
(``cat[name].read()``) plus any model-specific standardization happens in
``ocean_skill.sources.read``. Files that don't parse as v2 are skipped with a warning.

Any ``*.yaml`` in a catalog *directory* is picked up. Search precedence (later shadows
earlier, with a collision warning):
    1. packaged reference catalogs    (``ocean_skill/catalogs/``)
    2. shared/team                    (``$OCEAN_SKILL_CATALOGS``, os.pathsep-separated;
                                        or :func:`add_search_path` in code)
    3. legacy user dir                (``platformdirs`` config dir -- back-compat)
    4. user dir                       (``~/.ocean-skill/catalogs``)
    5. project-local                  (``./catalogs``)
"""

from __future__ import annotations

import difflib
import fnmatch
import hashlib
import json
import os
import re
import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ocean_skill import _stacklevel, casts
from ocean_skill._display import Description, Text

__all__ = [
    "Overlap",
    "SourceNames",
    "SourceRef",
    "add_search_path",
    "catalog_metadata",
    "catalog_names",
    "catalogs",
    "coord_report",
    "describe",
    "discover",
    "find",
    "find_catalogs",
    "fingerprint",
    "match_report",
    "overlap",
    "resolve",
    "search_paths",
]

_CATALOG_GLOB = "*.yaml"


@dataclass(frozen=True)
class SourceRef:
    """A resolved reference to one entry within a discovered intake v2 catalog.

    ``path``/``name`` locate the entry; :func:`ocean_skill.sources.read` reopens the
    catalog with intake and calls ``cat[name].read()``.
    """

    name: str
    #: The catalog's **name**: its file stem (e.g. ``"ooi_papa"``), the unique,
    #: space-free identifier used by :func:`find`'s ``catalog=``, :func:`resolve`'s
    #: qualified ``"catalog:source"`` form, and every "Known catalogs:" message. Not
    #: the freeform ``title``/``description`` below, which are for reading and
    #: prose search, not identity.
    catalog: str
    path: Path
    metadata: dict[str, Any] = field(default_factory=dict)
    #: Freeform catalog-level ``metadata["title"]``, if the catalog set one — may
    #: contain spaces. Searchable individually via :func:`find`'s ``title=``.
    catalog_title: str = ""
    #: Freeform catalog-level ``metadata["description"]``, if the catalog set one.
    #: Searchable individually via :func:`find`'s ``description=``.
    catalog_description: str = ""
    #: Set by :func:`discover` when another, lower-precedence catalog declares the
    #: same bare name — carried on the ref rather than warned about immediately, so
    #: only an actual *bare* lookup of this specific name ever surfaces it (see
    #: :func:`resolve`). Merging the index touches every entry; almost none of
    #: them are what the caller is actually asking about.
    shadowed_path: Path | None = None
    #: The id of the cast, when this ref is one cast of a ``trajectoryProfile`` entry
    #: (:mod:`ocean_skill.casts`): ``name``/``path`` stay the entry's, ``metadata`` is
    #: the cast's own.
    cast: str | None = None

    @property
    def qualified(self) -> str:
        """Fully-qualified ``catalog:source`` name (plus ``[cast]`` for a cast)."""
        suffix = f"[{self.cast}]" if self.cast is not None else ""
        return f"{self.catalog}:{self.name}{suffix}"


#: Shared catalog directories registered at runtime via :func:`add_search_path`.
#: Same tier as ``$OCEAN_SKILL_CATALOGS``, appended after it.
_added_dirs: list[Path] = []


def add_search_path(directory: str | Path) -> None:
    """Register a shared catalog directory for this process.

    Joins the same tier as ``$OCEAN_SKILL_CATALOGS`` -- above the packaged reference
    catalogs, below ``~/.ocean-skill/catalogs`` and any project-local ``./catalogs`` --
    so a site or team catalog registered in code can't silently outrank a user's own
    overrides. Repeated calls append; within the tier, later calls shadow earlier ones.

    Takes effect on the very next :func:`discover` call: the index is cached against
    the files the search path currently yields, not against the path list itself.
    """
    _added_dirs.append(Path(directory).expanduser())


def _user_dir() -> Path:
    """Return the per-user catalog directory: ``~/.ocean-skill/catalogs``."""
    return Path.home() / ".ocean-skill" / "catalogs"


def _legacy_user_dir() -> Path | None:
    """Return the pre-dotdir user location, kept as a fallback for existing setups.

    ``platformdirs`` puts this somewhere OS-conventional but easy to lose track of
    (``~/Library/Application Support/ocean-skill/catalogs`` on macOS), which is why
    the user tier moved to a fixed, visible ``~/.ocean-skill/catalogs``. Anyone who
    already has catalogs at the old location keeps being found here, just at lower
    precedence than the new one.
    """
    try:
        import platformdirs
    except Exception:  # platformdirs optional at import time
        return None
    return Path(platformdirs.user_config_dir("ocean-skill")) / "catalogs"


def search_paths() -> list[Path]:
    """Return the ordered catalog search path (lowest to highest precedence)."""
    paths: list[Path] = [Path(__file__).parent / "catalogs"]  # packaged reference catalogs

    # Shared/team tier: $OCEAN_SKILL_CATALOGS, then any add_search_path() registrations.
    env = os.environ.get("OCEAN_SKILL_CATALOGS")
    if env:
        paths.extend(Path(p).expanduser() for p in env.split(os.pathsep) if p)
    paths.extend(_added_dirs)

    legacy = _legacy_user_dir()  # back-compat, ranked just below the dotdir
    if legacy is not None:
        paths.append(legacy)
    paths.append(_user_dir())

    # Project-local: the cwd, then walk up looking for a catalogs/ directory (as git
    # finds .git). Lets a notebook in docs/ or a script in examples/ see the project's
    # catalogs without caring where it was launched from. Nearest match wins.
    here = Path.cwd()
    found: list[Path] = []
    for d in (here, *here.parents):
        candidate = d / "catalogs"
        if candidate.is_dir():
            found.append(candidate)
        if (d / "pyproject.toml").exists() or (d / ".git").exists():
            break  # don't wander above the project root
    paths.extend(reversed(found))  # nearest last => highest precedence

    # Dedup, keeping each directory's highest-precedence occurrence (the legacy dir
    # aliasing the dotdir when platformdirs and $HOME/.ocean-skill coincide, an env
    # entry repeating another tier's directory, etc).
    seen: set[Path] = set()
    deduped: list[Path] = []
    for p in reversed(paths):
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    return list(reversed(deduped))


def _iter_catalog_files() -> list[Path]:
    """Yield catalog YAMLs in precedence order (any ``*.yaml`` in a catalog dir)."""
    files: list[Path] = []
    for d in search_paths():
        if d.is_dir():
            files.extend(sorted(d.glob(_CATALOG_GLOB)))
    return files


def _precedence_rank(paths: set[Path] | list[Path]) -> list[Path]:
    """Sort catalog file paths low precedence to high, per :func:`_iter_catalog_files`.

    A path that isn't among the currently discovered files (e.g. one supplied by a
    test's fake index) sorts last rather than raising, since it can't be placed
    relative to real files anyway.
    """
    order = _iter_catalog_files()
    rank = {p: i for i, p in enumerate(order)}
    return sorted(paths, key=lambda p: rank.get(p, len(order)))


def _entry_metadata(cat, name: str) -> dict[str, Any]:
    """Per-entry metadata straight off the catalog description.

    Read from ``cat.entries`` rather than ``cat[name]``: indexing a v2 catalog
    imports and instantiates the entry's reader class (network-capable for
    ERDDAP entries), and the description already carries the full metadata
    contract our builders write.
    """
    try:
        key = cat.aliases.get(name, name)
        return dict(cat.entries[key].metadata or {})
    except Exception:  # pragma: no cover - defensive
        return {}


def _catalog_fingerprint() -> tuple[tuple[str, int, int], ...]:
    """Cache key for :func:`discover`: each catalog file with (mtime_ns, size).

    Rebuilding a catalog, adding or removing one, or changing the search path
    (cwd, ``$OCEAN_SKILL_CATALOGS``, :func:`add_search_path`, user dir) all change
    this, so the cache can never serve a stale index.
    """
    out: list[tuple[str, int, int]] = []
    for path in _iter_catalog_files():
        try:
            st = path.stat()
            out.append((str(path), st.st_mtime_ns, st.st_size))
        except OSError:  # deleted between glob and stat
            out.append((str(path), -1, -1))
    return tuple(out)


#: (fingerprint, index) of the last discovery; None until the first call.
_discover_cache: (
    tuple[tuple[tuple[str, int, int], ...], dict[str, SourceRef]] | None
) = None


def discover() -> dict[str, SourceRef]:
    """Discover all entries across intake v2 catalogs, merged by entry name.

    Returns a mapping ``entry_name -> SourceRef``. Catalogs that don't load as intake
    v2 are skipped with a warning. On a name collision the higher-precedence (later)
    catalog wins, recorded on the ref's ``shadowed_path`` rather than warned about
    here, since almost every caller (``find``, ``catalogs.names()``, every
    ``resolve``, ...) has nothing to do with whichever two entries happen to
    collide. :func:`resolve` decides whether a collision is actually relevant to
    what was asked for.

    The parsed index is cached against the catalog files' ``(path, mtime_ns, size)``,
    so repeated calls cost a stat of each file rather than a re-parse; any change to
    a file or to the search path invalidates it automatically.
    """
    global _discover_cache
    fingerprint = _catalog_fingerprint()
    if _discover_cache is not None and _discover_cache[0] == fingerprint:
        return dict(_discover_cache[1])

    import intake

    index: dict[str, SourceRef] = {}
    for path in _iter_catalog_files():
        try:
            cat = intake.from_yaml_file(str(path))
        except Exception as exc:  # not a v2 catalog / unreadable
            warnings.warn(
                f"Skipping catalog {path} (not loadable as intake v2): {exc}",
                stacklevel=2,
            )
            continue
        # The catalog's identity is always its file stem -- unique and space-free by
        # construction (ocean_skill.build validates this on save). title/description
        # are separate, freeform metadata carried alongside for reading and prose
        # search, never for identity.
        cat_meta = getattr(cat, "metadata", {}) or {}
        catalog_name = path.stem
        catalog_title = cat_meta.get("title") or ""
        catalog_description = cat_meta.get("description") or ""
        # list(cat) reads `aliases`, which is empty unless the builder set it; fall back
        # to the entry keys so catalogs built without aliases are still discoverable.
        names = list(cat) or list(getattr(cat, "entries", {}))
        for name in names:
            prior = index.get(name)
            shadowed = prior.path if prior is not None and prior.path != path else None
            index[name] = SourceRef(
                name=name,
                catalog=catalog_name,
                path=path,
                metadata=_entry_metadata(cat, name),
                catalog_title=catalog_title,
                catalog_description=catalog_description,
                shadowed_path=shadowed,
            )
    _discover_cache = (fingerprint, index)
    return dict(index)


def _resolve_in(index: dict[str, SourceRef], name: str) -> SourceRef:
    """Resolve one reference string against an already-built index.

    Same semantics as :func:`resolve`, which is just this over a fresh
    :func:`discover`; callers resolving many names build the index once and
    call this directly instead of re-discovering per name.
    """
    if ":" in name:
        cat, _, src = name.partition(":")
        for ref in index.values():
            if ref.name == src and ref.catalog == cat:
                return ref
        in_cat = [ref.name for ref in index.values() if ref.catalog == cat]
        raise KeyError(
            f"No source {src!r} in catalog {cat!r}.{_did_you_mean(src, in_cat)} "
            f"{_find_hint(len(index))}"
        )
    if name in index:
        ref = index[name]
        if ref.shadowed_path is not None:
            warnings.warn(
                f"Entry name {name!r} in {ref.path} shadows {ref.shadowed_path}; "
                f"use {ref.qualified!r} to disambiguate.",
                stacklevel=_stacklevel.find(),
            )
        return ref
    raise KeyError(
        f"Unknown source {name!r}.{_did_you_mean(name, index)} {_find_hint(len(index))}"
    )


def resolve(name: str) -> SourceRef:
    """Resolve a source reference string to a :class:`SourceRef`.

    ``"catalog:source"`` selects explicitly — never a collision warning, since
    naming the catalog *is* resolving the ambiguity. A bare name resolves against
    the merged index (the higher-precedence entry wins) and warns only if *this*
    name is the one that collided — an unrelated bare lookup elsewhere in the same
    call never triggers it. Raises :class:`KeyError` if unknown / ambiguous.

    ``"<entry>[<cast id>]"`` resolves one cast of a ``trajectoryProfile`` entry, as a
    ``profile`` ref at its own time and position (:mod:`ocean_skill.casts`).
    """
    try:
        return _resolve_in(discover(), name)
    except KeyError:
        base, cast = casts.split_name(name)
        if cast is None:
            raise
    return casts.resolve_cast(resolve(base), base, cast)


#: Entry-metadata keys :func:`fingerprint` leaves out: the ones that do not change
#: what a read of the entry returns. Prose for people (``description``, ``title``,
#: ``tags``) says nothing about the data. The rest is *derived from* the data by the
#: build probe -- the extents (``geospatial_*``, and ERDDAP's own spelling of the same
#: bounding box, ``minLatitude`` ... ``maxLongitude``), the time coverage
#: (``time_coverage_*``, ``minTime`` / ``maxTime``), the ``domain_outline`` ring, the
#: ``variables`` list -- so it follows from the reader definition that is hashed
#: anyway, and re-describing an entry, or re-probing it so an extent shifts, must not
#: throw away every cached result built from it. Everything else in the metadata
#: (``featureType``, ``axes``, ``standard_names``, ``units``, ``time_zone``,
#: ``depth_convention``, ``qc``, ...) says how the read is *interpreted*, so it stays
#: in.
_FINGERPRINT_SKIP_KEYS = frozenset(
    {
        "description",
        "title",
        "tags",
        "minTime",
        "maxTime",
        "minLatitude",
        "maxLatitude",
        "minLongitude",
        "maxLongitude",
        "domain_outline",
        "variables",
    }
)
#: The families of derived keys :func:`fingerprint` also leaves out; see above.
_FINGERPRINT_SKIP_PREFIXES = ("geospatial_", "time_coverage_")

#: Globals intake injects into a catalog's ``user_parameters`` when it loads one, and
#: which leak into the file when a loaded catalog is saved again. They record where the
#: file *was* last loaded from, not what it defines, so :func:`fingerprint` ignores
#: them.
_INTAKE_INJECTED_PARAMETERS = frozenset(
    {"CATALOG_PATH", "CATALOG_DIR", "STORAGE_OPTIONS"}
)

#: intake's own syntax for one catalog entity pointing at another -- ``{data(<token>)}``
#: -- which a reader uses for its data description and a chained step uses for the
#: reader before it. The token may carry a ``,<n>`` partial-pipeline suffix, which is
#: dropped.
_ENTITY_REF = re.compile(r"\{?data\(([^),]+)")

#: ``{catalog file: (mtime_ns, size, {entry name: fingerprint})}`` -- the memo behind
#: :func:`fingerprint`. One slot per file, so a rewritten catalog replaces its own
#: stale slot instead of piling up beside it; ``(mtime_ns, size)`` is the freshness
#: signal :func:`discover` and :func:`ocean_skill.sources.read` already use.
_fingerprints: dict[str, tuple[int, int, dict[str, str]]] = {}


def _load_catalog_yaml(path: str | os.PathLike) -> Any:
    """Parse a catalog file as plain YAML -- the document intake itself loads."""
    import yaml

    with open(path, "rb") as stream:
        return yaml.load(stream, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))


def _entity_refs(entity: Any) -> set[str]:
    """Tokens of the entities ``entity`` points at, by ``{data(<token>)}`` reference.

    Its own ``metadata`` is not searched: that is description, not definition.
    """
    found: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, str):
            match = _ENTITY_REF.match(value)
            if match:
                found.add(match.group(1).strip())
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    if isinstance(entity, dict):
        walk({k: v for k, v in entity.items() if k != "metadata"})
    else:
        walk(entity)
    return found


def _hashed_form(entity: Any) -> Any:
    """``entity`` as :func:`fingerprint` hashes it: as written, less inert metadata."""
    if not isinstance(entity, dict):
        return entity
    out = dict(entity)
    metadata = out.get("metadata")
    if isinstance(metadata, dict):
        out["metadata"] = {
            k: v
            for k, v in metadata.items()
            if k not in _FINGERPRINT_SKIP_KEYS
            and not str(k).startswith(_FINGERPRINT_SKIP_PREFIXES)
        }
    return out


def _stringify_keys(value: Any) -> Any:
    """Return ``value`` with every mapping key made a string, all the way down."""
    if isinstance(value, dict):
        return {str(k): _stringify_keys(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_stringify_keys(v) for v in value]
    return value


def _entry_fingerprint(doc: dict[str, Any], key: str) -> str:
    """Return the :func:`fingerprint` of the entry under ``key`` in a parsed catalog."""
    entries, data = doc.get("entries") or {}, doc.get("data") or {}
    aliases = doc.get("aliases") or {}
    # Follow every reference to the end, resolving each as intake does: an alias first,
    # then a reader entry, then a data description. A reference to nothing is recorded
    # as such rather than skipped -- a catalog that dangles is a different definition.
    refs: dict[str, Any] = {}
    pending = _entity_refs(entries[key])
    while pending:
        token = pending.pop()
        if token in refs:
            continue
        target = aliases.get(token, token)
        entity = entries[target] if target in entries else data.get(target)
        refs[token] = _hashed_form(entity)
        pending |= _entity_refs(entity) - refs.keys()
    payload: dict[str, Any] = {"entry": _hashed_form(entries[key]), "refs": refs}
    # Catalog-wide parameters a ``{name}`` template in the entry may read their value
    # from; empty in every catalog this package builds.
    parameters = {
        k: v
        for k, v in (doc.get("user_parameters") or {}).items()
        if k not in _INTAKE_INJECTED_PARAMETERS
    }
    if parameters:
        payload["user_parameters"] = parameters
    try:
        text = json.dumps(payload, sort_keys=True, default=str)
    except TypeError:  # mixed-type keys (1 and "1") cannot be sorted
        text = json.dumps(_stringify_keys(payload), sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _definition_fingerprints(path: str | os.PathLike) -> dict[str, str]:
    """Return ``{name: fingerprint}`` for every entry of a catalog file, aliases too."""
    doc = _load_catalog_yaml(path)
    entries = doc.get("entries") if isinstance(doc, dict) else None
    if not isinstance(entries, dict):
        return {}  # not an intake v2 catalog document
    prints = {key: _entry_fingerprint(doc, key) for key in entries}
    # A name is an alias when the file has one, else the entry key itself; an alias
    # outranks a key of the same spelling, as it does in intake's own lookup.
    for alias, token in (doc.get("aliases") or {}).items():
        if token in prints:
            prints[alias] = prints[token]
    return prints


def fingerprint(name: str) -> str:
    """Return a short fingerprint of ``name``'s catalog *definition*, or ``""`` if none.

    What :mod:`ocean_skill.cache` files a source under besides its name, so an entry
    redefined under the same name -- a script that rewrites a catalog so ``cast0000``
    now points at a different CSV, or ``model_win`` at a different time window through a
    reader chain -- is not handed the result cached for the old one. The first 16 hex
    digits of a sha256 over the sorted JSON of two things, both read from the catalog
    *file* (no data is opened):

    * the **reader definition** as written: the reader class, its arguments (URLs and
      paths included), any chained transforms -- an intake reader chained with
      ``.tail()``/``.head()``/``.drop_vars()`` is serialized as a pipeline of steps --
      and every data description or reader entry it references, followed to the end;
    * the entry's **metadata**, less what does not change a read (prose, and the
      extents/coverage/variable list the build probe derived; see
      :data:`_FINGERPRINT_SKIP_KEYS`). ``time_zone``, ``standard_names``, ``axes``,
      ``depth_convention`` and the rest, which decide how a read is interpreted, are in.

    Paths are part of the definition on purpose: a cache copied to a machine whose
    catalog points at different paths misses and recomputes, and one copied between
    identical catalogs hits. What this cannot see is the data behind an *unchanged*
    definition -- rewriting the same file in place -- which still needs
    :func:`ocean_skill.cache.clear` or ``refresh=True``.

    ``""`` when there is no definition to speak of -- ``name`` does not resolve, or what
    it resolves to is not an entry in a catalog file (tests stub :func:`resolve` with
    bare metadata) -- and never an error: the cache then keys on the name alone, as it
    always did.

    Stable across processes (plain sha256 over sorted JSON; no hash randomization), and
    memoized per catalog file on ``(path, mtime_ns, size)``: one parse per version of a
    file, however many of its entries are asked about.
    """
    try:
        ref = resolve(name)
    except (KeyError, TypeError):
        return ""
    path, entry = getattr(ref, "path", None), getattr(ref, "name", None)
    if not isinstance(path, (str, os.PathLike)) or not isinstance(entry, str):
        return ""
    try:
        st = os.stat(path)
    except (OSError, TypeError, ValueError):
        return ""
    memo_key = os.path.abspath(path)  # a relative search path means the cwd's file
    slot = _fingerprints.get(memo_key)
    if slot is None or slot[:2] != (st.st_mtime_ns, st.st_size):
        try:
            prints = _definition_fingerprints(path)
        except Exception:  # unreadable, not YAML, or not shaped like a v2 catalog
            prints = {}
        slot = (st.st_mtime_ns, st.st_size, prints)
        _fingerprints[memo_key] = slot
    return slot[2].get(entry, "")


def _did_you_mean(name: str, options: Iterable[str], n: int = 5) -> str:
    """A short " Did you mean: ...?" clause, or "" if nothing looks close.

    Used instead of dumping every known name on a lookup miss (there can be
    hundreds). Tries edit-distance matches first (typos), then falls back to
    case-insensitive substring hits (a partial name like ``"nitrate"``), so a
    guess is offered whenever one is plausible without ever listing everything.
    """
    options = list(options)
    lower_to_original = {opt.lower(): opt for opt in options}
    close = difflib.get_close_matches(name.lower(), lower_to_original, n=n)
    matches = [lower_to_original[c] for c in close]
    if not matches:
        needle = name.lower()
        matches = sorted(opt for opt in options if needle in opt.lower())[:n]
    if not matches:
        return ""
    return f" Did you mean: {', '.join(matches)}?"


def _find_hint(n_sources: int) -> str:
    """A brief pointer to searching for a source yourself, for lookup-miss errors."""
    return (
        "Search with osk.find(name=...) (substring/glob) or osk.find(text=...); "
        f"bare osk.find() lists all {n_sources} sources. If a catalog you expected "
        "isn't among them, check osk.catalog.search_paths()."
    )


def _matches_name(source: str, pattern: str) -> bool:
    """Case-insensitively match a source name by glob if wildcarded, else substring.

    Two habits, one argument: ``"papa"`` finds anything containing it, while
    ``"woa23_nitrate_month*"`` means what a shell would mean. Requiring a full glob
    for the common case would be tedious; treating a wildcard as a literal would be
    surprising.
    """
    source, pattern = source.lower(), pattern.lower()
    if any(ch in pattern for ch in "*?["):
        return fnmatch.fnmatch(source, pattern)
    return pattern in source


#: Every spelling of each month, so a search term can be translated into whichever
#: one a given catalog happens to use. Distinctive forms only: the full name, the
#: three-letter abbreviation, and ``monthNN``. Bare numerals are deliberately absent
#: from the *expansion* -- "1" as a substring matches "woa23" and half the index --
#: though a caller who types "01" still gets a literal match on it.
_MONTH_NAMES: tuple[tuple[str, ...], ...] = (
    ("january", "jan"),
    ("february", "feb"),
    ("march", "mar"),
    ("april", "apr"),
    ("may",),
    ("june", "jun"),
    ("july", "jul"),
    ("august", "aug"),
    ("september", "sep", "sept"),
    ("october", "oct"),
    ("november", "nov"),
    ("december", "dec"),
)

#: any spelling -> the canonical ``monthNN`` period code (used by ``climatology=``)
_MONTH_ALIASES: dict[str, str] = {
    alias: f"month{number:02d}"
    for number, names in enumerate(_MONTH_NAMES, start=1)
    for alias in (*names, f"{number:02d}", str(number), f"month{number:02d}")
}

#: any spelling -> every distinctive spelling of that month (used by ``text=``)
_MONTH_FORMS: dict[str, tuple[str, ...]] = {
    alias: (*names, f"month{number:02d}")
    for number, names in enumerate(_MONTH_NAMES, start=1)
    for alias in (*names, f"{number:02d}", str(number), f"month{number:02d}")
}


def _matches_period(meta: dict[str, Any], wanted: str) -> bool:
    """Whether a source's climatology period matches ``wanted``, however it is spelled.

    Catalogs record ``month01``; people type "January". Both, plus "jan", "1" and
    "01", resolve to the same code, and anything else falls back to a substring test
    so ``"annual"`` and future period names work without an entry here.
    """
    period = str(meta.get("climatology_period") or "").lower()
    if not period:
        return False
    wanted = wanted.strip().lower()
    return period == _MONTH_ALIASES.get(wanted, wanted) or wanted in period


#: spoken cadences -> seconds, so ``cadence="daily"`` need not be written "P1D".
_CADENCE_ALIASES: dict[str, float] = {
    "hourly": 3600.0,
    "6-hourly": 21600.0,
    "6hourly": 21600.0,
    "daily": 86400.0,
    "weekly": 604800.0,
    "8-day": 8 * 86400.0,
    "8day": 8 * 86400.0,
    "monthly": 30.4375 * 86400.0,
    "annual": 365.25 * 86400.0,
    "yearly": 365.25 * 86400.0,
}


def _as_range(spec, aliases: dict[str, float] | None = None):
    """Normalize a filter to ``(low, high)``.

    A bare number is an upper bound, because that is what people mean: asking for
    ``resolution=5`` means "5 km or finer", never "exactly 5". A 2-tuple is a
    closed range, and a word is looked up in ``aliases`` and matched within a
    factor to absorb the difference between a nominal month and 30.44 days.
    """
    if isinstance(spec, str):
        if aliases is None:
            raise TypeError(f"expected a number or (low, high), got {spec!r}")
        key = spec.strip().lower()
        if key not in aliases:
            raise ValueError(
                f"unknown cadence {spec!r}; try one of {sorted(aliases)} "
                "or a number of seconds"
            )
        target = aliases[key]
        return target * 0.75, target * 1.25
    if isinstance(spec, tuple | list):
        low, high = spec
        return (
            float(low) if low is not None else None,
            float(high) if high is not None else None,
        )
    return None, float(spec)


def _matches_range(value, spec, aliases: dict[str, float] | None = None) -> bool:
    """Whether ``value`` falls in ``spec``; an unknown value is **kept**.

    Same rule the extents follow: a source that never declared the quantity has
    not said it is outside the range, and dropping un-probed entries would hide
    exactly the sources a search exists to surface.
    """
    if value is None:
        return True
    low, high = _as_range(spec, aliases)
    value = float(value)
    return not (
        (low is not None and value < low) or (high is not None and value > high)
    )


def _haystack(source: str, ref) -> str:
    """Everything about a source that is worth matching free text against.

    Its name, its catalog's name/title/description, and every string in its
    metadata — summary, institution, period, declared variables. Catalogs describe
    themselves unevenly: one records ``climatology: True``, another writes ``period:
    monthly_climatology``, a third says it only in the source name. Free text spans
    all of it so a search does not depend on knowing which convention a given
    catalog happened to use.
    """
    parts = [source, ref.catalog, ref.catalog_title, ref.catalog_description]
    stack = list((ref.metadata or {}).items())
    while stack:
        key, value = stack.pop()
        parts.append(str(key))
        if key == "qc":
            # Machinery, not description: the resolved qc contract's own
            # flag_definitions/scheme text (e.g. "questionable", "missing") would
            # otherwise match nearly every free-text search over a QC'd catalog.
            # The literal word "qc" (the key itself, appended above) still matches.
            continue
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, dict):
            stack.extend(value.items())
        elif isinstance(value, list | tuple):
            parts.extend(str(v) for v in value)
        elif isinstance(value, bool | int | float):
            parts.append(str(value))
    return " ".join(parts).lower()


def _matches_text(source: str, ref, terms) -> bool:
    """Whether every term appears somewhere in the source's searchable text.

    Terms are ANDed, because narrowing is what a second word is for: "modis chl jan"
    should mean all three, not any of them.
    """
    hay = _haystack(source, ref)
    # A month name should find a month climatology however the catalog spelled the
    # period: WOA records `month01`, MODIS writes `jan` into the source name. Without
    # this, free text and the structured `climatology=` filter would disagree about
    # what "january" means, which is exactly the confusion free text exists to avoid.
    return all(
        term in hay or any(form in hay for form in _MONTH_FORMS.get(term, ()))
        for term in terms
    )


def _as_terms(text) -> list[str]:
    """Split free text into lowercase terms; a sequence is taken as given."""
    if isinstance(text, str):
        return [t for t in text.lower().split() if t]
    return [str(t).lower() for t in text if str(t).strip()]


def _declares_variable(meta: dict[str, Any], variable: str) -> bool:
    """Whether a catalog entry's declared variables satisfy a request for ``variable``.

    The one question :func:`find` (``variable=``) and :func:`ocean_skill.comparison
    .compare`'s source pre-filter both put to an entry's metadata, asked in one place so
    a source a search returns is exactly a source a comparison pairs. Read-free: it
    looks only at the entry's stored ``variables`` list. What a caller does when that
    list is empty differs -- a search excludes the entry, a comparison lets the read
    decide -- so that stays with the caller; this answers only for what is declared.

    Two things widen the stored list or the match beyond literal membership:

    * A ROMS entry also offers the true eastward/northward velocity
      :func:`ocean_skill.roms.standardize` derives at read time from grid-relative
      u/v (:func:`ocean_skill.roms.derived_geographic_velocities`) -- a catalog built
      before that derivation existed does not list it, but the source plainly ends up
      offering it.
    * The vocabulary decides what counts as a match. The literal intersection with
      :func:`~ocean_skill.vocabulary.equivalent_names` is the regex-free fast path;
      :func:`~ocean_skill.vocabulary.covers` is the general test -- it reaches a
      declared spelling only a vocabulary pattern recognizes (``"Temperature_CTD"``)
      or differing only by case, and it is where a broad request meets a specific
      declaration: ``"mld"`` is satisfied by an entry declaring any one definition of
      mixed layer depth, ``"mld_by_sigma_theta"`` only by one declaring that
      definition, never by the generic name (which has not said which one it
      carries) or a sibling.
    """
    from ocean_skill.vocabulary import covers, equivalent_names

    declared = set(meta.get("variables") or [])
    if meta.get("model") == "roms":
        from ocean_skill.roms import derived_geographic_velocities

        declared |= set(derived_geographic_velocities(declared))
    return bool(equivalent_names(variable) & declared) or any(
        covers(variable, d) for d in declared
    )


def _overlaps(low, high, other_low, other_high) -> bool:
    """Whether two closed intervals intersect, given all four bounds are known."""
    return not (high < other_low or low > other_high)


def _bbox_overlaps(meta: dict[str, Any], bbox) -> bool | None:
    """Whether a source's declared extent meets ``bbox``; ``None`` if undeclared."""
    keys = (
        "geospatial_lon_min",
        "geospatial_lat_min",
        "geospatial_lon_max",
        "geospatial_lat_max",
    )
    lo, la, hi, ha = (meta.get(k) for k in keys)
    if None in (lo, la, hi, ha):
        return None
    lo, la, hi, ha = float(lo), float(la), float(hi), float(ha)
    want_lo, want_la, want_hi, want_ha = bbox

    if not _overlaps(la, ha, want_la, want_ha):
        return False

    # A globe-spanning source overlaps every box, and must be short-circuited before
    # any normalisation: wrapping -180..180 into +/-180 sends *both* bounds to -180,
    # collapsing the whole world to a point that overlaps nothing.
    if hi - lo >= 359.0:
        return True
    if hi > 180.0:  # declared 0-360 (ROMS' native convention); bring into +/-180
        lo, hi = ((lo + 180) % 360) - 180, ((hi + 180) % 360) - 180
    if lo > hi:  # straddles the anti-meridian; do not guess
        return None
    return _overlaps(lo, hi, want_lo, want_hi)


def _time_overlaps(meta: dict[str, Any], window) -> bool | None:
    """Whether a source's declared time coverage meets ``window``; ``None`` if not."""
    import pandas as pd

    start, end = meta.get("time_coverage_start"), meta.get("time_coverage_end")
    if not start or not end:
        return None
    try:
        have = (pd.Timestamp(start), pd.Timestamp(end))
        want = (pd.Timestamp(window[0]), pd.Timestamp(window[1]))
    except Exception:
        return None
    return _overlaps(have[0], have[1], want[0], want[1])


def _circular_overlap(lon_min_a, lon_max_a, lon_min_b, lon_max_b) -> bool:
    """Whether two longitude intervals intersect on the circle (mod 360).

    :func:`ocean_skill.comparison._domain_of` deliberately returns a
    dateline-straddling domain's bounds in whichever of 0-360/+/-180 keeps them
    contiguous (its own docstring explains why) -- so the two intervals handed
    here are not guaranteed to share a convention, and comparing them as plain
    (non-wrapping) intervals would misjudge exactly the antimeridian domains
    this exists to get right. Sliding one interval by every whole turn that
    could bring it adjacent to the other sidesteps the question of which
    convention either one started in.
    """
    lo_a, hi_a = lon_min_a % 360.0, lon_min_a % 360.0 + (lon_max_a - lon_min_a)
    lo_b, hi_b = lon_min_b % 360.0, lon_min_b % 360.0 + (lon_max_b - lon_min_b)
    return any(
        _overlaps(lo_a, hi_a, lo_b + shift, hi_b + shift) for shift in (-360.0, 0.0, 360.0)
    )


def _boxes_overlap(box_a, box_b) -> bool:
    """Whether two ``(lon_min, lat_min, lon_max, lat_max)`` boxes intersect."""
    lon_min_a, lat_min_a, lon_max_a, lat_max_a = box_a
    lon_min_b, lat_min_b, lon_max_b, lat_max_b = box_b
    if not _overlaps(lat_min_a, lat_max_a, lat_min_b, lat_max_b):
        return False
    return _circular_overlap(lon_min_a, lon_max_a, lon_min_b, lon_max_b)


#: Allowance, in degrees, added to each side of a *gridded* source's declared extent on
#: an axis where that extent has zero width. The extent is the span of the build probe's
#: cell-*centre* coordinates, so a model file holding one cell (or one row/column of
#: them, for the other axis) declares a point -- which says nothing about the size of
#: the cell around it, and a station well inside that cell, a few metres off its
#: centre, would be "provably disjoint" from it and never compared. The pre-compare
#: skip this feeds (:func:`ocean_skill.comparison._provably_disjoint_axes`) is only a
#: read-free optimisation, so it errs generous: letting a pair through that turns out
#: not to meet costs one read, skipping one that did meet costs the comparison. About 5
#: km is deliberately modest -- not a model of the cell (unknowable without a read),
#: only enough that an off-centre station inside a regional-model cell is not refused.
#: Never applied to observation points (stations, casts, moorings) or to a gridded
#: source's real, non-zero-width extent: for those the declared box is exact, and a
#: station just outside a regional model must still be skipped.
POINT_EXTENT_TOLERANCE_DEG = 0.05


def _is_gridded(meta: dict[str, Any]) -> bool:
    """Whether an entry describes a model/regular grid rather than an observation.

    ``featureType: grid`` (:func:`ocean_skill.build.guess_feature_type`'s answer for
    anything with two horizontal dimensions, and what a gridded climatology declares),
    or a ``model`` key -- a ROMS entry carries ``model: roms``, and a hand-written
    catalog need not also set ``featureType`` on it.
    """
    if str(meta.get("featureType") or "").lower() == "grid":
        return True
    return bool(meta.get("model"))


def _declared_box(name: str) -> tuple[float, float, float, float] | None:
    """Return ``name``'s declared lon/lat box as :func:`overlap` should see it.

    A ``(lon_min, lat_min, lon_max, lat_max)`` tuple, or ``None`` when nothing is
    declared: :func:`ocean_skill.comparison._domain_of` (longitude convention
    normalised) plus one allowance -- a *gridded* source's zero-width axis is widened
    by :data:`POINT_EXTENT_TOLERANCE_DEG` on each side (see that constant for why).
    Axis by axis, so one row of cells (a real longitude extent, a single latitude)
    keeps its exact longitude range. Returns a new tuple; the entry's own metadata
    is never touched.
    """
    from ocean_skill.comparison import _domain_of

    box = _domain_of(name)
    if box is None:
        return None
    lon_min, lat_min, lon_max, lat_max = box
    if lon_min != lon_max and lat_min != lat_max:
        return box  # a real extent on both axes: nothing to widen, nothing to look up
    try:
        meta = resolve(name).metadata
    except KeyError:
        return box
    if not _is_gridded(meta):
        return box
    pad = POINT_EXTENT_TOLERANCE_DEG
    if lon_min == lon_max:
        lon_min, lon_max = lon_min - pad, lon_max + pad
    if lat_min == lat_max:
        lat_min, lat_max = lat_min - pad, lat_max + pad
    return lon_min, lat_min, lon_max, lat_max


@dataclass(frozen=True)
class Overlap:
    """Whether two sources' catalog-declared extents overlap in space and time.

    Each axis is ``True`` (the two sources' declared extents overlap),
    ``False`` (they provably do not), or ``None`` (one side has no declared
    extent on that axis -- an ungridded/unprobed entry, or one whose
    coordinate columns this package's build step didn't recognize -- so there
    is nothing to check, not a "no"). ``bool(this)`` reads as "no *known*
    reason to refuse this pair": ``True`` unless an axis is provably disjoint,
    so a caller doesn't block on an axis it simply can't check yet.
    """

    space: bool | None
    time: bool | None

    def __bool__(self) -> bool:
        return self.space is not False and self.time is not False

    def __repr__(self) -> str:
        def fmt(value: bool | None) -> str:
            return "unknown" if value is None else ("yes" if value else "no")

        return f"Overlap(space={fmt(self.space)}, time={fmt(self.time)})"


def overlap(a: str, b: str) -> Overlap:
    """Whether two catalog sources' declared extents overlap in space and time.

    Read-free -- built entirely from each source's catalog metadata (the same
    ``geospatial_*``/``time_coverage_*`` contract :func:`describe` and
    :func:`find` already rely on), so it costs nothing to check *before*
    comparing two sources: two whose catalog-declared time coverage never
    meets will never produce a matched pair, and a multi-GB read is a bad way
    to learn that. :meth:`ocean_skill.comparison.Comparison.align` calls this
    itself and warns when it comes back false, for the same reason.

    A day of padding is folded into the time check (the same pad
    :func:`ocean_skill.comparison._time_coverage_of` applies before narrowing
    a test lane to a reference's record), so this agrees with what narrowing
    would actually find rather than being a stricter, unrelated question.
    Longitude is compared on the circle, convention-agnostic (0-360 vs
    +/-180, and a domain straddling the antimeridian) -- see
    :func:`_circular_overlap`.

    A gridded source (``featureType: grid``, or a ``model`` entry such as ROMS)
    whose declared extent has zero width on an axis -- a model file holding one
    cell -- is widened by :data:`POINT_EXTENT_TOLERANCE_DEG` on that axis first: the
    extent is built from cell *centres*, so it says nothing about the cell's size and
    a station inside the cell would otherwise read as disjoint. Observation points
    are never widened (see :func:`_declared_box`).
    """
    from ocean_skill.comparison import _time_coverage_of

    box_a, box_b = _declared_box(a), _declared_box(b)
    space = None if box_a is None or box_b is None else _boxes_overlap(box_a, box_b)

    window_a, window_b = _time_coverage_of(a), _time_coverage_of(b)
    time = (
        None
        if window_a is None or window_b is None
        else _overlaps(*window_a, *window_b)
    )
    return Overlap(space=space, time=time)


class SourceNames(list):
    """What :func:`find` returns: a plain list of source names, plus ``.map()``.

    Still a ``list`` in every way that matters — indexing, iteration, ``in``,
    equality with a bare list — so nothing downstream changes. The one addition
    makes the common flow read as one line::

        osk.find(variable="nitrate").map()   # where are they?

    For the cases with no query to hang a method on ("map everything", "map this
    catalog"), :func:`ocean_skill.plot.map_locations.map_locations` is the same
    map as a standalone call.

    ``catalog=``/``name=`` match by substring or glob, so more than one catalog
    can satisfy a single query (``catalog="ooi"`` matches both ``ooi_papa`` and
    ``ooi_endurance``) and their sources are pooled into this one flat list --
    nothing about a bare name says which catalog it came from. ``.catalogs`` and
    ``.by_catalog()`` answer that without a second lookup::

        osk.find(catalog="ooi").catalogs        # ['ooi_endurance', 'ooi_papa']
        osk.find(name="papa").by_catalog()      # {'ooi_papa': [...]}
    """

    def __init__(self, names=(), catalog_of: dict[str, str] | None = None):
        super().__init__(names)
        #: Source name -> catalog name, for every source :func:`find` placed
        #: here. Not necessarily complete: a name added by other means (e.g.
        #: ``SourceNames(["foo"])`` or a plain ``.append``) has no entry, and
        #: ``.catalogs``/``.by_catalog()`` simply skip it.
        self._catalog_of: dict[str, str] = dict(catalog_of) if catalog_of else {}

    def _add(self, source: str, catalog: str) -> None:
        """Append ``source`` and record which catalog it came from."""
        self._catalog_of[source] = catalog
        self.append(source)

    @property
    def catalogs(self) -> list[str]:
        """Sorted, deduplicated names of the catalogs these sources came from."""
        return sorted({self._catalog_of[s] for s in self if s in self._catalog_of})

    def by_catalog(self) -> dict[str, list[str]]:
        """These source names grouped by catalog, in catalog-name order.

        A source with no recorded catalog (added some way other than
        :func:`find`) is omitted -- there is nothing to group it under.
        """
        out: dict[str, list[str]] = {}
        for source in self:
            cat = self._catalog_of.get(source)
            if cat is not None:
                out.setdefault(cat, []).append(source)
        return {cat: out[cat] for cat in sorted(out)}

    def map(self, **kwargs):
        """Map where these sources are.

        Parameters
        ----------
        **kwargs
            Forwarded to :func:`ocean_skill.plot.map_locations.map_locations` --
            ``catalog``, ``renderer`` (one of ``"matplotlib"``, ``"holoviews"``),
            ``domain``, plus any other plot option (``extent``, ``title``,
            ``save``, ``tiles``, ``legend``, ...). See
            ``docs/plot_styling_reference.md`` for the full list.

        See :func:`ocean_skill.plot.map_locations.map_locations`.
        """
        from ocean_skill.plot.map_locations import map_locations

        return map_locations(self, **kwargs)


def find(
    *,
    text: str | list[str] | None = None,
    name: str | None = None,
    catalog: str | None = None,
    title: str | None = None,
    description: str | None = None,
    climatology: bool | str | None = None,
    variable: str | None = None,
    featureType: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    time: tuple[str, str] | None = None,
    resolution: float | tuple[float | None, float | None] | None = None,
    cadence: str | float | tuple[float | None, float | None] | None = None,
    vertical: bool | None = None,
) -> SourceNames:
    """Search discovered sources by name and metadata; return matching source names.

    Parameters
    ----------
    text
        Free-text query: ``str`` or list of terms, all ANDed. Matches anywhere in
        a source's name, its catalog's name/title/description, or its metadata
        (summary, institution, period, declared variables). ``None`` (default)
        skips this filter.
    name
        Substring or glob (case-insensitive), matched against a source's own name
        **or its catalog's name**. ``None`` (default) skips this filter.
    catalog
        Substring or glob (case-insensitive), matched only against the catalog's
        **name** — its saved file stem (e.g. ``"ooi_papa"``), never its title or
        description. ``None`` (default) skips this filter.
    title
        Substring or glob (case-insensitive), matched against the catalog's
        freeform ``title`` metadata, if it has one. ``None`` (default) skips this
        filter.
    description
        Substring or glob (case-insensitive), matched against the catalog's
        freeform ``description`` metadata, if it has one. ``None`` (default) skips
        this filter.
    climatology
        ``True``/``False`` to include or exclude climatologies, or a period string
        (``"January"``, ``"jan"``, ``"01"``, ``"month01"``) to match a specific
        one. ``None`` (default) skips this filter.
    variable
        A short vocabulary key, CF ``standard_name``, or alias (any case) — see
        :mod:`ocean_skill.vocabulary`. A broad name such as ``"mld"`` also finds
        sources declaring any one of its specific definitions
        (``"mld_by_sigma_theta"``, ...); a specific name finds only sources declaring
        that definition. ``None`` (default) skips this filter.
    featureType
        Exact match against the entry's declared ``featureType`` (e.g.
        ``"timeSeries"``). ``None`` (default) skips this filter.
    bbox
        ``(lon_min, lat_min, lon_max, lat_max)``; kept on any *overlap* with a
        source's extent, not containment. ``None`` (default) skips this filter.
    time
        ``(start, end)`` date strings; kept on overlap, and climatologies are
        excluded outright since they carry no date range. ``None`` (default) skips
        this filter.
    resolution
        Grid spacing in km: a bare number is an upper bound (``5`` means "5 km or
        finer"), or a ``(low, high)`` tuple for a closed range. ``None`` (default)
        skips this filter.
    cadence
        Time-step upper bound/range in seconds (like ``resolution``), or one of
        the words in ``_CADENCE_ALIASES`` — ``"hourly"``, ``"6-hourly"``,
        ``"6hourly"``, ``"daily"``, ``"weekly"``, ``"8-day"``, ``"8day"``,
        ``"monthly"``, ``"annual"``, ``"yearly"`` — matched within 25%. ``None``
        (default) skips this filter.
    vertical
        ``True``/``False`` to require or exclude a depth axis. ``None`` (default)
        skips this filter.

    Only the filters given are applied::

        osk.find(text="modis chl jan")               # free text, all terms must match
        osk.find(variable="nitrate")                 # any spelling of the variable
        osk.find(variable="mld")                     # any mixed layer depth definition
        osk.find(variable="mld_by_sigma_theta")      # only that one definition
        osk.find(name="papa")                        # substring, case-insensitive
        osk.find(name="woa23_nitrate_month*")        # glob
        osk.find(name="papa", featureType="timeSeries")
        osk.find(catalog="ooi_papa")                 # by catalog name (file stem)
        osk.find(catalog="ooi*")                     # by catalog rather than source
        osk.find(title="Station Papa")               # by the catalog's freeform title
        osk.find(description="climatology")          # by the catalog's description
        osk.find(climatology=True)                   # any climatology
        osk.find(climatology="January")              # climatologies of Januaries
        osk.find(climatology=False)                  # exclude climatologies
        osk.find(bbox=(-98, 18, -80, 31))            # Gulf of Mexico
        osk.find(time=("2012-01-01", "2012-02-01"))
        osk.find(resolution=5)                       # grid spacing 5 km or finer
        osk.find(cadence="daily")                    # or "hourly", "monthly", P1D
        osk.find(vertical=True)                      # has a depth axis

    ``text`` is the catch-all: each whitespace-separated term must appear somewhere
    in the source's name, its catalog's name/title/description, or any of its
    metadata — summary, institution, period, declared variables. Terms are ANDed,
    so ``text="modis chl jan"`` narrows rather than widens. Reach for it when
    catalogs describe the same idea differently (one records ``climatology:
    True``, another ``period: monthly_climatology``) and a structured filter would
    miss half of them; reach for the structured filters when you want a guarantee.

    ``variable`` accepts anything :mod:`ocean_skill.vocabulary` knows — a short key
    (``"nitrate"``), a canonical CF standard_name, or any alias, in any case — and
    matches a source declaring *any* equivalent spelling. That matters because
    products disagree: WOA declares nitrate per unit *mass* while ROMS/MARBL and
    GLODAP declare it per unit *volume*, so searching one exact standard_name finds
    two sources and silently misses the thirteen you would actually compare against.

    A *broad* name also finds every source declaring one of its *specific*
    definitions, in that direction only. Mixed layer depth is the case that exists:
    ROMS' KPP ``hbls`` (the mixing scheme's own boundary layer), Copernicus'
    ``mlotst`` and Holte & Talley's Argo climatology (a potential-density threshold)
    and a CMIP-style ``mlotst`` (a threshold on sigma_t) are all one *kind* of thing,
    and each declares which definition it is. ``variable="mld"`` finds all of them,
    plus any older catalog still declaring only the generic name; but
    ``variable="mld_by_sigma_theta"`` finds only the sources declaring that one --
    neither the generic name, which has not said which definition it carries, nor
    the sigma_t or mixing-scheme products, whose base of the layer is a different
    quantity that merely shares a family. Asking for the broad name is how you
    survey every mixed layer depth on offer; asking for a specific one is how you
    guarantee the definition, which is the difference that decides whether a
    comparison is like-for-like (see :func:`ocean_skill.vocabulary.covers`).

    ``name`` matches a source name **or its catalog's name**, because the useful
    handle is often on the catalog: OOI's sources are opaque dataset ids
    (``ooi-gp02hypm-rim01-02-ctdmog039``) sitting in a catalog named ``ooi_papa``,
    so a ``name="papa"`` that searched only source names would find nothing for the
    one word a person actually knows. Use ``catalog=`` to match only the catalog
    name, or ``title=``/``description=`` to search the catalog's freeform prose
    (its title and description are independent of its name and may contain spaces
    the name can't).

    Asking "does a catalog matching X exist at all" is a different question from
    "which sources match" -- :func:`find_catalogs` answers it directly, returning
    catalog names rather than a pool of sources from however many catalogs matched.

    ``bbox`` is ``(lon_min, lat_min, lon_max, lat_max)`` and both it and ``time``
    test for *overlap*, not containment — a global climatology matches a regional
    box.

    ``resolution`` is **grid spacing** in km, derived from the axis rather than from
    what the product declares — CoastWatch's Metop-C ASCAT dataset advertises 0.25
    degrees in its title while its latitude axis steps 0.3333. Grid spacing is not
    the scale of the smallest feature the data resolves: an interpolated L4 analysis
    resolves rather less than its grid implies (MUR SST is gridded at 0.01 degrees
    but resolves features of roughly 10 km), so a ``resolution=2`` hit describes how
    the data is stored, not what you can see in it.

    A bare number is an upper bound — ``resolution=5`` means "5 km or finer" — and a
    2-tuple is a closed range. ``cadence`` additionally accepts a word (``"daily"``,
    ``"hourly"``, ``"monthly"``) matched within 25%, because a monthly product steps
    28-31 days and no exact number of seconds would match it.

    ``climatology`` takes ``True``/``False`` to include or exclude climatologies, or
    a period to name one: ``"January"``, ``"jan"``, ``"01"`` and ``"month01"`` are
    all the same request. A climatology deliberately carries no
    ``time_coverage_*`` — it represents a calendar slot rather than a date range, so
    a ``time=`` filter would be answering a different question.

    A source that declares no extent is **kept**, not dropped: "unknown" is not
    "outside", and excluding un-probed entries would quietly hide the very sources
    a search is meant to surface. Probe a catalog if you want its entries filtered
    on geography or time.

    The result is a :class:`SourceNames` — a plain list of names that additionally
    offers ``.map()``, drawing where the matches are on a map from their catalog
    metadata alone, and ``.catalogs``/``.by_catalog()`` to see which catalog(s)
    a ``catalog=``/``name=`` substring or glob actually matched.
    """
    terms = _as_terms(text) if text is not None else None

    out = SourceNames()
    for source, ref in discover().items():
        meta = ref.metadata
        if terms and not _matches_text(source, ref, terms):
            continue
        if name and not (
            _matches_name(source, name) or _matches_name(ref.catalog, name)
        ):
            continue
        if catalog and not _matches_name(ref.catalog, catalog):
            continue
        if title and not _matches_name(ref.catalog_title, title):
            continue
        if description and not _matches_name(ref.catalog_description, description):
            continue
        if climatology is not None:
            is_clim = bool(meta.get("climatology"))
            if isinstance(climatology, str):
                if not (is_clim and _matches_period(meta, climatology)):
                    continue
            elif is_clim is not climatology:
                continue
        # An entry declaring nothing cannot be found by variable -- see
        # _declares_variable, the same test compare() pairs sources by.
        if variable and not _declares_variable(meta, variable):
            continue
        if featureType and meta.get("featureType") != featureType:
            continue
        if bbox is not None and _bbox_overlaps(meta, bbox) is False:
            continue
        if resolution is not None and not _matches_range(
            meta.get("grid_resolution_km"), resolution
        ):
            continue
        if cadence is not None and not _matches_range(
            meta.get("time_resolution_s"), cadence, _CADENCE_ALIASES
        ):
            continue
        if vertical is not None and bool(meta.get("vertical_levels")) is not vertical:
            continue
        if time is not None:
            # A climatology is excluded from a time search, not treated as unknown.
            # Missing coverage usually means "not probed, cannot say" -- but here we
            # positively know there is no date range: a January climatology is a
            # calendar slot, and returning it for a July 2012 query would be wrong.
            # Ask for it with climatology= instead.
            if meta.get("climatology"):
                continue
            if _time_overlaps(meta, time) is False:
                continue
        out._add(source, ref.catalog)
    return out


def find_catalogs(
    *,
    name: str | None = None,
    title: str | None = None,
    description: str | None = None,
    text: str | list[str] | None = None,
    climatology: bool | str | None = None,
    variable: str | None = None,
    featureType: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    time: tuple[str, str] | None = None,
    resolution: float | tuple[float | None, float | None] | None = None,
    cadence: str | float | tuple[float | None, float | None] | None = None,
    vertical: bool | None = None,
) -> list[str]:
    """Search discovered *catalogs* (not sources); return matching catalog names.

    The catalog-level counterpart to :func:`find`: where ``find(catalog="ooi")``
    returns every source pooled from however many catalogs matched "ooi",
    ``find_catalogs(name="ooi")`` answers the question directly -- which catalog(s)
    matched -- as a plain list of names. That makes it the natural existence
    check::

        if osk.find_catalogs(name="ooi"):
            ...   # at least one OOI catalog is discoverable

    Parameters
    ----------
    name
        Substring or glob (case-insensitive) against the catalog's **name** --
        its saved file stem, e.g. ``"ooi_papa"`` -- same rule as :func:`find`'s
        ``catalog=``. Unlike :func:`find`'s ``name=``, this does **not** also
        match source names: "find catalogs named X" should not be satisfied by
        a source named X sitting in a differently-named catalog. ``None``
        (default) skips this filter.
    title, description, text
        Same as :func:`find` -- freeform catalog prose, or free text over
        everything.
    climatology, variable, featureType, bbox, time, resolution, cadence, vertical
        Same as :func:`find`, but as an *existence* test on the catalog: a
        catalog is kept when **at least one** of its sources satisfies the
        filter. ``find_catalogs(variable="nitrate")`` means "which catalogs
        have any nitrate", not "every source in this catalog has nitrate".

    Returns a sorted, deduplicated list of catalog names -- built from
    :func:`find`'s own result via its ``.catalogs`` accessor, so the matching
    logic lives in exactly one place. A catalog with zero sources is invisible
    to :func:`discover` and so invisible here too, same as :func:`catalog_names`.

    For an *exact* name rather than a substring/glob, skip searching entirely::

        "ooi_papa" in osk.catalogs.catalog_names()

    ::

        osk.find_catalogs(name="ooi")           # ['ooi_endurance', 'ooi_papa']
        osk.find_catalogs(variable="nitrate")   # catalogs holding any nitrate
        osk.find_catalogs()                     # every discovered catalog
    """
    return find(
        catalog=name,
        title=title,
        description=description,
        text=text,
        climatology=climatology,
        variable=variable,
        featureType=featureType,
        bbox=bbox,
        time=time,
        resolution=resolution,
        cadence=cadence,
        vertical=vertical,
    ).catalogs


def catalog_names() -> list[str]:
    """Sorted names of all discovered catalogs (each ``SourceRef``'s ``catalog``).

    A catalog's name is its saved file stem (e.g. ``"ooi_papa"``) — the unique,
    space-free identifier, as opposed to its optional freeform ``title``/
    ``description`` (see :func:`catalog_metadata`, and :func:`find`'s
    ``title=``/``description=`` filters). For a filtered, substring/glob search
    over catalog names, see :func:`find_catalogs`.
    """
    return sorted({ref.catalog for ref in discover().values()})


def catalog_metadata(catalog: str) -> dict[str, Any]:
    """Return catalog-level metadata for a discovered catalog.

    Title/description/extents — as opposed to :func:`resolve`, which returns one
    *entry's* metadata. When two search-path tiers each hold a file with this
    catalog's name, reads the highest-precedence one -- the file whose entries
    actually win on a name collision (see :func:`discover`).
    """
    import intake

    paths = {ref.path for ref in discover().values() if ref.catalog == catalog}
    if not paths:
        raise KeyError(f"Unknown catalog {catalog!r}. Known: {catalog_names()}")
    winner = _precedence_rank(paths)[-1]
    cat = intake.from_yaml_file(str(winner))
    return dict(getattr(cat, "metadata", {}) or {})


def _declared_variables(name: str, index: dict[str, SourceRef]) -> list[str]:
    """The declared variable names to run a vocabulary match report over.

    A source's own declared ``variables``; a catalog's is the union across every
    source it contains (sorted, deduplicated) -- the same rollup
    :func:`ocean_skill.build.save`'s ``_rollup_metadata`` computes when writing the
    catalog. Raises the same "neither a known source nor catalog" error
    :func:`describe` does, since both dispatch on the same lookup.
    """
    if name in index:
        return list(index[name].metadata.get("variables") or [])
    names = catalog_names()
    if name in names:
        seen: set[str] = set()
        for ref in index.values():
            if ref.catalog == name:
                seen.update(ref.metadata.get("variables") or [])
        return sorted(seen)
    raise KeyError(
        f"{name!r} is neither a known source nor catalog."
        f"{_did_you_mean(name, [*index, *names])} "
        f"Known catalogs: {names}. {_find_hint(len(index))}"
    )


def _declared_columns(name: str, index: dict[str, SourceRef]) -> list[str]:
    """The raw declared column names to run a coordinate report over.

    Unlike :func:`_declared_variables`, this must include coordinate columns --
    the very thing :func:`ocean_skill.build`'s probe excludes from ``variables``
    (see :func:`ocean_skill.tabular.is_coordinate_column`). So it's the union of a
    source's ``standard_names`` keys (every declared column, raw name, coordinates
    included) and its ``axes`` map's values (the column an entry's own build
    already picked out as an axis -- needed for a reader that declared no
    ``standard_names`` map at all, or for a *stale* entry whose ``axes`` names a
    column the vocabulary no longer recognizes there, see
    :func:`_coord_staleness_notes`). A catalog's is the union across every source
    it contains. Raises the same "neither a known source nor catalog" error
    :func:`_declared_variables` does, since both dispatch on the same lookup.
    """

    def _columns(ref: SourceRef) -> set[str]:
        md = ref.metadata
        cols = set((md.get("standard_names") or {}).keys())
        cols.update((md.get("axes") or {}).values())
        return cols

    if name in index:
        return sorted(_columns(index[name]))
    names = catalog_names()
    if name in names:
        seen: set[str] = set()
        for ref in index.values():
            if ref.catalog == name:
                seen.update(_columns(ref))
        return sorted(seen)
    raise KeyError(
        f"{name!r} is neither a known source nor catalog."
        f"{_did_you_mean(name, [*index, *names])} "
        f"Known catalogs: {names}. {_find_hint(len(index))}"
    )


def _coord_staleness_notes(name: str, index: dict[str, SourceRef]) -> list[str]:
    """Notes when a source's stored ``axes`` are now excluded by the coordinate vocabulary.

    A catalog's ``axes`` map is a snapshot written at build time (see
    :mod:`ocean_skill.build`'s probes); the vocabulary is live (see
    :class:`ocean_skill.vocabulary.CoordReport`). If the vocabulary later adds an
    exclusion -- exactly what ``COORD_VOCABULARY``'s ``"bottom"`` token did to any
    catalog already built with ``axes["Z"] = "Depth_bottom"`` -- the stored map goes
    stale. Deliberately checks only :func:`ocean_skill.vocabulary.excluded_from_axis`
    (an *active* refusal), not whether the name would satisfy a full token match:
    plenty of legitimately-stored axis names never token-match at all -- gridded
    sources routinely reach ``axes["Z"]`` via cf-xarray's own attribute-based
    detection (rather than name matching), and ``"altitude"`` in particular is
    excluded from :func:`ocean_skill.tabular.coord_axis_of`'s token match by design
    (no agreed sign convention) without being wrong as a stored Z name. Re-deriving
    "is this still the *best* match" would need rereading the data, which this
    (like :func:`describe`) deliberately does not do.
    """
    from ocean_skill import tabular, vocabulary

    refs = (
        [index[name]]
        if name in index
        else [ref for ref in index.values() if ref.catalog == name]
    )
    multiple = len(refs) > 1
    notes: list[str] = []
    for ref in refs:
        axes = ref.metadata.get("axes") or {}
        for axis, col in axes.items():
            base = tabular.split_units(str(col))[0]
            if vocabulary.excluded_from_axis(base, axis):
                prefix = f"{ref.name}: " if multiple else ""
                notes.append(
                    f"{prefix}stored axes[{axis!r}] = {col!r} is now excluded from "
                    "the coordinate vocabulary -- rebuild this catalog"
                )
    return notes


def describe(name: str) -> Description:
    """Human-readable summary of a source or a catalog — whichever ``name`` is.

    Parameters
    ----------
    name
        A discovered source name or catalog name (``str``); which one determines
        which summary is produced.

    For a source: its catalog path and full entry metadata (featureType,
    standard_names, extents, ...), followed by a live vocabulary match report over
    its declared variables, then a live coordinate report over its declared
    columns (which of T/X/Y/Z are recognized, and as which column). For a catalog:
    its title/description/extents, the catalog file(s) it is read from, the sources
    it contains, and any of its entries that a differently-named catalog file now
    wins (a ``shadowed`` line per winning file), followed by the same two reports
    over the union of every source's columns. Meant for interactive use, e.g.
    ``osk.describe(name)``. See :func:`match_report` and :func:`coord_report` for
    either report alone, and :class:`ocean_skill.vocabulary.MatchReport`/
    :class:`~ocean_skill.vocabulary.CoordReport` for why neither is ever cached or
    stored: both always reflect the vocabulary as it stands right now.

    Returns
    -------
    Description
        A ``str`` (so anything expecting text keeps working) that also carries the
        summary's structured fields as attributes: ``kind`` (``"source"`` or
        ``"catalog"``), ``catalog``, ``catalog_path`` (the file currently winning
        for this name), ``catalog_paths`` (every file contributing to it, low to
        high precedence), ``metadata``, ``sources``, and ``shadowed`` (``{entry
        name: winning file}`` for this catalog's entries that another catalog file
        defines under the same name and wins; empty for a source). e.g.
        ``osk.describe("glodap").catalog_path``.
    """
    from ocean_skill.vocabulary import coord_report as _vocab_coord_report
    from ocean_skill.vocabulary import match_report as _vocab_match_report

    index = discover()
    if name in index:
        ref = index[name]
        lines = [f"source: {ref.qualified}", f"  catalog path: {ref.path}"]
        for k, v in sorted(ref.metadata.items()):
            lines.append(f"  {k}: {v}")
        fields = {
            "kind": "source",
            "name": ref.name,
            "catalog": ref.catalog,
            "catalog_path": ref.path,
            "catalog_paths": (ref.path,),
            "metadata": dict(ref.metadata),
            "sources": (ref.name,),
            "shadowed": {},
        }
    elif name in catalog_names():
        md = catalog_metadata(name)
        srcs = sorted(ref.name for ref in index.values() if ref.catalog == name)
        cat_paths = {ref.path for ref in index.values() if ref.catalog == name}
        paths = tuple(_precedence_rank(cat_paths))
        lines = [f"catalog: {name}"]
        lines.extend(f"  catalog path: {p}" for p in paths)
        for k, v in sorted(md.items()):
            lines.append(f"  {k}: {v}")
        lines.append(f"  sources ({len(srcs)}): {', '.join(srcs)}")
        # A catalog's own entries that another file now wins: the winner's
        # `shadowed_path` is one of this catalog's files. `ref.catalog != name`
        # leaves out entries only a same-named file in a higher tier beat -- those
        # still belong here. (`shadowed_path` is just the one file directly beaten.)
        own_files = {p for p in _iter_catalog_files() if p.stem == name}
        shadowed = {
            ref.name: ref.path
            for ref in sorted(index.values(), key=lambda r: r.name)
            if ref.catalog != name and ref.shadowed_path in own_files
        }
        by_file: dict[Path, list[str]] = {}
        for entry, path in shadowed.items():
            by_file.setdefault(path, []).append(entry)
        for path in _precedence_rank(list(by_file)):
            entries = by_file[path]
            lines.append(f"  shadowed ({len(entries)}) by {path}: {', '.join(entries)}")
        fields = {
            "kind": "catalog",
            "name": name,
            "catalog": name,
            "catalog_path": paths[-1],
            "catalog_paths": paths,
            "metadata": dict(md),
            "sources": tuple(srcs),
            "shadowed": shadowed,
        }
    else:
        raise KeyError(
            f"{name!r} is neither a known source nor catalog."
            f"{_did_you_mean(name, [*index, *catalog_names()])} "
            f"Known catalogs: {catalog_names()}. {_find_hint(len(index))}"
        )
    report = _vocab_match_report(_declared_variables(name, index))
    lines.append("  vocabulary:")
    lines.extend(f"    {line}" for line in str(report).splitlines())
    coords = _vocab_coord_report(_declared_columns(name, index))
    lines.append("  coordinates:")
    lines.extend(f"    {line}" for line in str(coords).splitlines())
    lines.extend(f"    note: {note}" for note in _coord_staleness_notes(name, index))
    return Description("\n".join(lines), **fields)


def match_report(name: str) -> Text:
    """Live vocabulary match report for a source's or catalog's declared variables.

    Parameters
    ----------
    name
        A discovered source name or catalog name (``str``); which one determines
        whether the report covers one source's declared variables or the union
        across a catalog's sources.

    For a source: which of its declared ``variables`` resolve to a vocabulary
    nickname (and to which), and which don't. For a catalog: the same report over
    the union of every source's variables (see :func:`_declared_variables`).
    Always computed against the vocabulary as it stands *right now* -- never
    stored anywhere, so it can't go stale the way a snapshot written into a
    catalog file would the moment a new alias or pattern ships (see
    :class:`ocean_skill.vocabulary.MatchReport`). Meant for interactive use, e.g.
    ``osk.match_report(name)`` -- or as the ``vocabulary:`` section
    :func:`describe` already appends. Also includes the same live coordinate
    report :func:`coord_report` gives (and :func:`describe` appends as its own
    ``coordinates:`` section) -- a coordinate report whenever there's a match
    report, so a mismatch like ``Depth`` losing out to ``Depth_bottom`` is never
    more than one call away from the variable report that prompted the look.
    """
    from ocean_skill.vocabulary import coord_report as _vocab_coord_report
    from ocean_skill.vocabulary import match_report as _vocab_match_report

    index = discover()
    names = _declared_variables(name, index)  # raises if name is unknown
    header = f"source: {name}" if name in index else f"catalog: {name}"
    lines = [header, str(_vocab_match_report(names))]
    lines.append("coordinates:")
    coords = _vocab_coord_report(_declared_columns(name, index))
    lines.extend(f"  {line}" for line in str(coords).splitlines())
    lines.extend(f"  note: {note}" for note in _coord_staleness_notes(name, index))
    return Text("\n".join(lines))


def coord_report(source) -> Text:
    """Live coordinate report: which of T/X/Y/Z is recognized, and as what.

    Parameters
    ----------
    source
        A catalog/source name (``str``), a :class:`pandas.DataFrame`, or an
        :class:`xarray.Dataset`/:class:`~xarray.DataArray`. Which type determines
        whether the report is over stored metadata (name) or a live, per-axis scan
        of the object's actual columns/coordinates (DataFrame/Dataset/DataArray).

    ``source`` may be:

    - a catalog/source **name** (``str``): the same declared-column report
      :func:`describe`/:func:`match_report` already append, plus a note for any
      stored ``axes`` entry the vocabulary no longer agrees with (see
      :func:`_coord_staleness_notes`) -- no data is read.
    - a :class:`pandas.DataFrame`: the live per-axis report
      :func:`ocean_skill.tabular.coord_column` gives right now, over the frame's
      actual columns.
    - an :class:`xarray.Dataset`/:class:`~xarray.DataArray`: the live report
      :func:`ocean_skill.cf.find_coord` gives, cf-xarray attribute matching
      included -- not just name matching.

    Meant for interactive use, e.g. ``osk.coord_report("glodap")`` or
    ``osk.coord_report(df)`` after a manual :func:`ocean_skill.sources.read`. See
    :func:`match_report` for the analogous variable-vocabulary report, and
    :class:`ocean_skill.vocabulary.CoordReport` for why a live report is never
    cached: it always reflects the vocabulary as it stands right now.
    """
    from ocean_skill import cf, tabular
    from ocean_skill.vocabulary import CoordReport
    from ocean_skill.vocabulary import coord_report as _vocab_coord_report

    if isinstance(source, str):
        index = discover()
        names = _declared_columns(source, index)  # raises if name is unknown
        header = f"source: {source}" if source in index else f"catalog: {source}"
        lines = [header, str(_vocab_coord_report(names))]
        lines.extend(_coord_staleness_notes(source, index))
        return Text("\n".join(lines))

    if tabular.is_frame(source):
        matched = {
            axis: [col]
            for axis in ("T", "X", "Y", "Z")
            if (col := tabular.coord_column(source, axis)) is not None
        }
    else:
        from ocean_skill.vocabulary import COORD_VOCABULARY

        matched = {}
        for axis, entry in COORD_VOCABULARY.items():
            found = cf.find_coord(source, entry["kind"])
            name = getattr(found, "name", None)
            if name is not None:
                matched[axis] = [str(name)]
    missing = [axis for axis in ("T", "X", "Y", "Z") if axis not in matched]
    return Text(str(CoordReport(matched=matched, missing=missing)))


class _CatalogRegistry:
    """Lazy, dict-like view over discovered sources (exposed as ``osk.catalogs``)."""

    def _index(self) -> dict[str, SourceRef]:
        return discover()

    def __iter__(self):
        return iter(self._index())

    def __len__(self) -> int:
        return len(self._index())

    def __contains__(self, name: object) -> bool:
        return name in self._index()

    def __getitem__(self, name: str) -> SourceRef:
        return resolve(name)

    def names(self) -> list[str]:
        """Sorted list of all discovered source names."""
        return sorted(self._index())

    def catalog_names(self) -> list[str]:
        """Sorted list of all discovered catalog names (not source names)."""
        return catalog_names()

    def describe(self, name: str) -> Description:
        """Human-readable summary of a source or catalog (whichever ``name`` is)."""
        return describe(name)

    def match_report(self, name: str) -> Text:
        """Live vocabulary match report for a source or catalog (whichever ``name``)."""
        return match_report(name)

    def coord_report(self, name: str) -> Text:
        """Live coordinate report for a source or catalog (whichever ``name``)."""
        return coord_report(name)

    def __repr__(self) -> str:
        idx = self._index()
        by_cat: dict[str, list[str]] = {}
        titles: dict[str, str] = {}
        for ref in idx.values():
            by_cat.setdefault(ref.catalog, []).append(ref.name)
            if ref.catalog_title:
                titles[ref.catalog] = ref.catalog_title
        lines = [f"<ocean_skill.catalogs: {len(idx)} sources>"]
        for cat, srcs in sorted(by_cat.items()):
            label = f"{cat} ({titles[cat]})" if cat in titles else cat
            lines.append(f"  {label}: {', '.join(sorted(srcs))}")
        return "\n".join(lines)


catalogs = _CatalogRegistry()
