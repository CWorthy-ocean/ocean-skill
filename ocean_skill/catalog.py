"""Catalog auto-discovery and source-name resolution.

Catalogs are native **intake v2** YAML files that ocean-skill discovers on a search path
and merges into a single namespace. Discovery loads each catalog with
``intake.from_yaml_file`` and indexes its entries by name (with their per-entry
``metadata`` — our contract: featureType, standard_names, extents). Opening an entry
(``cat[name].read()``) plus any model-specific standardization happens in
``ocean_skill.sources.read``. Files that don't parse as v2 are skipped with a warning.

Any ``*.yaml`` in a catalog *directory* is picked up. Search precedence (later shadows
earlier, with a collision warning):
    1. packaged example catalogs      (``ocean_skill/catalogs/``)
    2. user dir                       (``platformdirs`` / ``~/.ocean-skill/catalogs``)
    3. ``$OCEAN_SKILL_CATALOGS``      (os.pathsep-separated dirs)
    4. project-local                  (``./catalogs``)
"""

from __future__ import annotations

import fnmatch
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ocean_skill._display import Text

__all__ = [
    "SourceRef",
    "catalog_metadata",
    "catalog_names",
    "catalogs",
    "describe",
    "discover",
    "find",
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
    catalog: str
    path: Path
    metadata: dict[str, Any] = field(default_factory=dict)
    #: Set by :func:`discover` when another, lower-precedence catalog declares the
    #: same bare name — carried on the ref rather than warned about immediately, so
    #: only an actual *bare* lookup of this specific name ever surfaces it (see
    #: :func:`resolve`). Building the merged index touches every entry on every
    #: call; almost none of them are what the caller is actually asking about.
    shadowed_path: Path | None = None

    @property
    def qualified(self) -> str:
        """Fully-qualified ``catalog:source`` name."""
        return f"{self.catalog}:{self.name}"


def search_paths() -> list[Path]:
    """Return the ordered catalog search path (lowest to highest precedence)."""
    paths: list[Path] = [Path(__file__).parent / "catalogs"]  # packaged examples

    try:
        import platformdirs

        paths.append(Path(platformdirs.user_config_dir("ocean-skill")) / "catalogs")
    except Exception:  # platformdirs optional at import time
        paths.append(Path.home() / ".ocean-skill" / "catalogs")

    env = os.environ.get("OCEAN_SKILL_CATALOGS")
    if env:
        paths.extend(Path(p) for p in env.split(os.pathsep) if p)

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
    return paths


def _iter_catalog_files() -> list[Path]:
    """Yield catalog YAMLs in precedence order (any ``*.yaml`` in a catalog dir)."""
    files: list[Path] = []
    for d in search_paths():
        if d.is_dir():
            files.extend(sorted(d.glob(_CATALOG_GLOB)))
    return files


def _entry_metadata(cat, name: str) -> dict[str, Any]:
    """Best-effort per-entry metadata (reader instance, else the description)."""
    try:
        md = getattr(cat[name], "metadata", None)
        if md:
            return dict(md)
    except Exception:  # pragma: no cover - defensive
        pass
    try:
        key = cat.aliases.get(name, name)
        return dict(cat.entries[key].metadata or {})
    except Exception:  # pragma: no cover - defensive
        return {}


def discover() -> dict[str, SourceRef]:
    """Discover all entries across intake v2 catalogs, merged by entry name.

    Returns a mapping ``entry_name -> SourceRef``. Catalogs that don't load as intake
    v2 are skipped with a warning. On a name collision the higher-precedence (later)
    catalog wins, recorded on the ref's ``shadowed_path`` rather than warned about
    here: this runs on every catalog-touching call (``find``, ``catalogs.names()``,
    every ``resolve``, ...), almost all of which have nothing to do with whichever
    two entries happen to collide. :func:`resolve` decides whether a collision is
    actually relevant to what was asked for.
    """
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
        catalog_name = (getattr(cat, "metadata", {}) or {}).get("title") or path.stem
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
                shadowed_path=shadowed,
            )
    return index


def resolve(name: str) -> SourceRef:
    """Resolve a source reference string to a :class:`SourceRef`.

    ``"catalog:source"`` selects explicitly — never a collision warning, since
    naming the catalog *is* resolving the ambiguity. A bare name resolves against
    the merged index (the higher-precedence entry wins) and warns only if *this*
    name is the one that collided — an unrelated bare lookup elsewhere in the same
    call never triggers it. Raises :class:`KeyError` if unknown / ambiguous.
    """
    if ":" in name:
        cat, _, src = name.partition(":")
        for ref in discover().values():
            if ref.name == src and ref.catalog == cat:
                return ref
        raise KeyError(f"No source {src!r} in catalog {cat!r}.")
    index = discover()
    if name in index:
        ref = index[name]
        if ref.shadowed_path is not None:
            warnings.warn(
                f"Entry name {name!r} in {ref.path} shadows {ref.shadowed_path}; "
                f"use {ref.qualified!r} to disambiguate.",
                stacklevel=2,
            )
        return ref
    raise KeyError(f"Unknown source {name!r}. Known: {sorted(index)}")


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
                f"unknown cadence {spec!r}; try one of {sorted(aliases)} or a number of seconds"
            )
        target = aliases[key]
        return target * 0.75, target * 1.25
    if isinstance(spec, tuple | list):
        low, high = spec
        return (float(low) if low is not None else None,
                float(high) if high is not None else None)
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
    return not ((low is not None and value < low) or (high is not None and value > high))


def _haystack(source: str, ref) -> str:
    """Everything about a source that is worth matching free text against.

    Its name, its catalog's name, and every string in its metadata — title, summary,
    institution, period, declared variables. Catalogs describe themselves unevenly:
    one records ``climatology: True``, another writes ``period:
    monthly_climatology``, a third says it only in the source name. Free text spans
    all of it so a search does not depend on knowing which convention a given
    catalog happened to use.
    """
    parts = [source, ref.catalog]
    stack = list((ref.metadata or {}).items())
    while stack:
        key, value = stack.pop()
        parts.append(str(key))
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


def find(
    *,
    text: str | list[str] | None = None,
    name: str | None = None,
    catalog: str | None = None,
    climatology: bool | str | None = None,
    variable: str | None = None,
    featureType: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    time: tuple[str, str] | None = None,
    resolution: float | tuple[float | None, float | None] | None = None,
    effective_resolution: float | tuple[float | None, float | None] | None = None,
    cadence: str | float | tuple[float | None, float | None] | None = None,
    vertical: bool | None = None,
) -> list[str]:
    """Search discovered sources by name and metadata; return matching source names.

    Only the filters given are applied::

        osk.find(text="modis chl jan")               # free text, all terms must match
        osk.find(variable="nitrate")                 # any spelling of the variable
        osk.find(name="papa")                        # substring, case-insensitive
        osk.find(name="woa23_nitrate_month*")        # glob
        osk.find(name="papa", featureType="timeSeries")
        osk.find(catalog="OOI*")                     # by catalog rather than source
        osk.find(climatology=True)                   # any climatology
        osk.find(climatology="January")              # climatologies of Januaries
        osk.find(climatology=False)                  # exclude climatologies
        osk.find(bbox=(-98, 18, -80, 31))            # Gulf of Mexico
        osk.find(time=("2012-01-01", "2012-02-01"))
        osk.find(resolution=5)                       # grid spacing 5 km or finer
        osk.find(effective_resolution=25)            # *actually* resolves 25 km
        osk.find(cadence="daily")                    # or "hourly", "monthly", P1D
        osk.find(vertical=True)                      # has a depth axis

    ``text`` is the catch-all: each whitespace-separated term must appear somewhere
    in the source's name, its catalog's name, or any of its metadata — title,
    summary, institution, period, declared variables. Terms are ANDed, so
    ``text="modis chl jan"`` narrows rather than widens. Reach for it when catalogs
    describe the same idea differently (one records ``climatology: True``, another
    ``period: monthly_climatology``) and a structured filter would miss half of
    them; reach for the structured filters when you want a guarantee.

    ``variable`` accepts anything :mod:`ocean_skill.vocabulary` knows — a short key
    (``"nitrate"``), a canonical CF standard_name, or any alias, in any case — and
    matches a source declaring *any* equivalent spelling. That matters because
    products disagree: WOA declares nitrate per unit *mass* while ROMS/MARBL and
    GLODAP declare it per unit *volume*, so searching one exact standard_name finds
    two sources and silently misses the thirteen you would actually compare against.

    ``name`` matches a source name **or its catalog's**, because the useful handle is
    often on the catalog: OOI's sources are opaque dataset ids
    (``ooi-gp02hypm-rim01-02-ctdmog039``) sitting in a catalog called "OOI Station
    Papa", so a ``name="papa"`` that searched only source names would find nothing
    for the one word a person actually knows. Use ``catalog=`` to match only the
    catalog.

    ``bbox`` is ``(lon_min, lat_min, lon_max, lat_max)`` and both it and ``time``
    test for *overlap*, not containment — a global climatology matches a regional
    box.

    ``resolution`` and ``effective_resolution`` are both in km and are **not the
    same question**. ``resolution`` is grid spacing, derived from the axis.
    ``effective_resolution`` is the scale of the smallest feature the data can
    actually resolve, which for any interpolated L4 analysis is coarser — often by
    an order of magnitude. MUR SST is published on a 0.01 degree grid (1.1 km) but
    resolves features of about 10 km; DUACS altimetry is gridded at 0.25 degrees
    yet resolves roughly 200 km wavelengths at midlatitude. Someone filtering
    ``resolution=2`` and concluding they can see 2 km fronts in MUR would be wrong,
    so ask with ``effective_resolution`` whenever the question is about what the
    data can *see* rather than how it is stored. It is a curated value: absent
    unless a product documents one, and absent means unknown, not fine.

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
    """
    from ocean_skill.vocabulary import equivalent_names

    terms = _as_terms(text) if text is not None else None

    out: list[str] = []
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
        if climatology is not None:
            is_clim = bool(meta.get("climatology"))
            if isinstance(climatology, str):
                if not (is_clim and _matches_period(meta, climatology)):
                    continue
            elif is_clim is not climatology:
                continue
        if variable and not (
            equivalent_names(variable) & set(meta.get("variables") or [])
        ):
            continue
        if featureType and meta.get("featureType") != featureType:
            continue
        if bbox is not None and _bbox_overlaps(meta, bbox) is False:
            continue
        if resolution is not None and not _matches_range(
            meta.get("grid_resolution_km"), resolution
        ):
            continue
        if effective_resolution is not None and not _matches_range(
            meta.get("effective_resolution_km"), effective_resolution
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
        out.append(source)
    return out


def catalog_names() -> list[str]:
    """Sorted names of all discovered catalogs (each ``SourceRef``'s ``catalog``)."""
    return sorted({ref.catalog for ref in discover().values()})


def catalog_metadata(catalog: str) -> dict[str, Any]:
    """Return catalog-level metadata for a discovered catalog.

    Title/description/extents — as opposed to :func:`resolve`, which returns one
    *entry's* metadata.
    """
    import intake

    for ref in discover().values():
        if ref.catalog == catalog:
            cat = intake.from_yaml_file(str(ref.path))
            return dict(getattr(cat, "metadata", {}) or {})
    raise KeyError(f"Unknown catalog {catalog!r}. Known: {catalog_names()}")


def describe(name: str) -> Text:
    """Human-readable summary of a source or a catalog — whichever ``name`` is.

    For a source: its catalog, path, and full entry metadata (featureType,
    standard_names, extents, ...). For a catalog: its title/description/extents plus
    the sources it contains. Meant for interactive use, e.g. ``osk.describe(name)``.
    """
    index = discover()
    if name in index:
        ref = index[name]
        lines = [f"source: {ref.qualified}", f"  path: {ref.path}"]
        for k, v in sorted(ref.metadata.items()):
            lines.append(f"  {k}: {v}")
        return Text("\n".join(lines))
    if name in catalog_names():
        md = catalog_metadata(name)
        srcs = sorted(ref.name for ref in index.values() if ref.catalog == name)
        lines = [f"catalog: {name}"]
        for k, v in sorted(md.items()):
            lines.append(f"  {k}: {v}")
        lines.append(f"  sources ({len(srcs)}): {', '.join(srcs)}")
        return Text("\n".join(lines))
    raise KeyError(
        f"{name!r} is neither a known source nor catalog. "
        f"Known sources: {sorted(index)}. Known catalogs: {catalog_names()}"
    )


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

    def describe(self, name: str) -> Text:
        """Human-readable summary of a source or catalog (whichever ``name`` is)."""
        return describe(name)

    def __repr__(self) -> str:
        idx = self._index()
        by_cat: dict[str, list[str]] = {}
        for ref in idx.values():
            by_cat.setdefault(ref.catalog, []).append(ref.name)
        lines = [f"<ocean_skill.catalogs: {len(idx)} sources>"]
        for cat, srcs in sorted(by_cat.items()):
            lines.append(f"  {cat}: {', '.join(sorted(srcs))}")
        return "\n".join(lines)


catalogs = _CatalogRegistry()
