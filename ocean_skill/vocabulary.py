"""The one editable vocabulary mapping a concept to its known real-world spellings.

The data is ``ocean_skill/vocab/vocabulary.yaml``, written in `cf-xarray <https://
cf-xarray.readthedocs.io>`_'s own ``custom_criteria`` shape so any tool can load it
with plain ``cf_xarray.set_options(custom_criteria=yaml.safe_load(...))`` and no
ocean-skill code (``ocean_skill/vocab/README.md`` has the file rules). Edit that file
and nothing else needs to change: it is parsed at import into :data:`VOCABULARY`, and
:func:`resolve_name` (used everywhere a variable name is accepted, from
:func:`ocean_skill.comparison.compare` down to :func:`ocean_skill.units.find_variable`,
:func:`ocean_skill.vars.lookup` and :func:`ocean_skill.colormaps.cmaps_for`) and the
cf-xarray registration below are both derived from that dict automatically.

Each entry of :data:`VOCABULARY` is one concept, keyed by a short mnemonic (the name to
actually type — it need not be a real CF name, it's just what's easy to read and call
by):

- ``standard_name``: the canonical CF standard_name used internally everywhere else
  in ocean-skill (as ``da.attrs["standard_name"]``, dict keys in
  :mod:`ocean_skill.vars`/:mod:`ocean_skill.colormaps`, etc).
- ``aliases`` (optional): other real spellings of the same quantity — e.g.
  WOA/GLODAP's per-mass CF name for a species ROMS/MARBL writes per-volume, or a
  particular instrument's naming convention. Any number, not just one.
- ``patterns`` (optional): regexes recognizing a whole *family* of spellings an
  enumerated alias list would be tedious to spell out (``Temperature_CTD``,
  ``CTD_Temperature``, ``ctd_temp``, ...). Matched **fullmatch, case-insensitive**
  against the whole name — never a substring — so a pattern must be as narrow as an
  alias is exact. Write enumerated decorations (``"temp(?:erature)?_ctd"``), never an
  open-ended ``.*`` tail, and never a ``qc``/``flag``/``qartod`` token (that is the QC
  layer's job, not the vocabulary's). A name matched by more than one entry's patterns
  is ambiguous and is refused (warned, passed through unresolved) rather than guessed.
  This is the same idea as cf-pandas' ``guess_regex``, deliberately narrower: bare
  ``"month"`` matching cf-pandas' time regex is exactly the looseness these patterns
  must not repeat (see :mod:`ocean_skill.tabular`'s coordinate matcher for the same
  rule applied to column axes).
- ``broader`` (optional): the short key, standard_name or any spelling of a *broader*
  entry this one is one specific kind of — ``mld_by_sigma_theta`` names ``"mld"``,
  because a potential-density mixed layer depth is one particular *definition* of
  "mixed layer depth". One level only (no chains, no self-reference), and recorded on
  the *specific* entry rather than as a list on the broad one, so that
  :func:`register` replacing the broad entry — which replaces the whole dict —
  cannot wipe out what the specific entries say about it. It changes nothing about
  what a name *resolves* to, and :func:`equivalent_names` and :func:`same_quantity`
  stay the symmetric "same variable" tests they always were; it adds a separate,
  one-directional relation, read through :func:`narrower_names` and :func:`covers`:
  asking for the broad name is satisfied by any specific definition, while asking
  for a specific one is never satisfied by the broad name or by a sibling definition.

A caller may use the short key, the canonical ``standard_name``, any alias, or
anything one entry's ``patterns`` fullmatch, interchangeably, **in any
capitalization** — :func:`resolve_name` maps all of them to the same canonical
``standard_name``, and :func:`resolve_and_report` additionally warns once, naming
both, so it's never silently unclear which variable was actually used. Matching
against a *dataset's* own variable names ignores case too (see
:func:`ocean_skill.units.find_variable`). A name with no entry here at all passes
through unchanged — most CF standard_names need no vocabulary entry; this exists
only for the ones spelled more than one way in the wild.

A few aliases pair quantities that are *near*-identical rather than identical —
potential vs in-situ temperature, practical vs absolute salinity — which are
routinely compared anyway (they differ by a few tenths of a degree by ~1000 m, and
much less near the surface). They are plain aliases here deliberately: a separate
"approximate" tier existed briefly and earned nothing, because it changed only the
wording of a message and never how a variable was matched. If that distinction ever
matters for real work, this is the point to reintroduce it — with resolution
behaviour attached, not just a label.

``broader`` is that tier reintroduced, for the case that finally made the distinction
matter: mixed layer depth has one generic CF name and several *definition-specific*
ones (by sigma_theta, sigma_t, temperature, or a model's own mixing scheme), and a
request for "the MLD" should find a source carrying any one of them without ever
letting a request for one specific definition be quietly answered by another. That is
a one-level broad-to-specific relation, not near-identity (so not an alias), with
resolution behaviour attached in the two places a variable is actually looked for:
**search**, where :func:`covers` says whether what a source declares satisfies what
was asked for (the right question for a catalog or source filter to put to each
declared variable — the symmetric :func:`same_quantity` can only say "the same
variable"), and **dataset lookup**, where the broad entry's cf-xarray criteria also
match its specific entries' spellings (see :func:`_register_custom_criteria`), so a
generic request finds the one specific definition a dataset carries. The dataset
half is deliberately *data*, not lookup code: cf-xarray acts on the registered
criteria by itself, so another tool registering the same criteria gets the same
behaviour; a dataset carrying several definitions is cf-xarray's own "multiple
variables" case, which :func:`ocean_skill.units.find_variable` reports rather than
resolves. Catalog search has no cf-xarray equivalent — a catalog's declared names
are not a Dataset — so :func:`covers` is the one piece that stays code.

The in-memory shape above is ours, not cf-xarray's: cf-xarray's own
``custom_criteria`` only understands attribute names as keys (``"name"``,
``"standard_name"``, ``"units"``, ..., matched against each variable's own
attributes) — it has no native concept of an "alias". The file is written in
cf-xarray's shape (one ``{"name": regex, "standard_name": regex}`` per key);
:func:`_from_criteria` reads it into the grouping above, :func:`_to_criteria` writes
the grouping back out as the file's shape, and :func:`_register_custom_criteria`
flattens each entry into the one ``{"name": pattern}`` per spelling that cf-xarray
does understand.
"""

from __future__ import annotations

import html
import importlib.resources
import re
import warnings
from collections.abc import Iterable
from dataclasses import dataclass

import yaml

from ocean_skill import _stacklevel

__all__ = [
    "COORD_AXIS_BY_KIND",
    "COORD_FALLBACKS",
    "COORD_VOCABULARY",
    "VOCABULARY",
    "CoordReport",
    "MatchReport",
    "add_alias",
    "add_pattern",
    "coord_report",
    "covers",
    "equivalent_names",
    "excluded_from_axis",
    "is_known",
    "match_report",
    "matches_axis",
    "narrower_names",
    "nickname",
    "register",
    "resolve_and_report",
    "resolve_name",
    "same_quantity",
]


def _split_alternatives(regex: str) -> list[str]:
    r"""Split ``a|b|(?:c|d)`` on its top-level ``|`` only.

    A ``|`` inside a group, inside a character class, or escaped (``\|``) belongs to
    its alternative. Enough regex awareness for the file rules and no more: the
    alternatives of a ``name`` are escaped literals and ``(?:...)``-wrapped patterns.
    """
    parts: list[str] = []
    depth, start, i, in_class = 0, 0, 0, False
    while i < len(regex):
        ch = regex[i]
        if ch == "\\":
            i += 1  # the escaped character is never structure
        elif in_class:
            in_class = ch != "]"
        elif ch == "[":
            in_class = True
            if regex.startswith("^]", i + 1):  # a leading ] is a literal member
                i += 2
            elif regex.startswith("]", i + 1):
                i += 1
        elif ch in "()":
            depth += 1 if ch == "(" else -1
        elif ch == "|" and depth == 0:
            parts.append(regex[start:i])
            start = i + 1
        i += 1
    return [*parts, regex[start:]]


def _alternatives_of(regex: str) -> list[str]:
    """Split a file regex shaped ``^(?i:a|b|...)$`` into its top-level alternatives."""
    if not (regex.startswith("^(?i:") and regex.endswith(")$")):
        raise ValueError(f"vocabulary.yaml: {regex!r} is not shaped '^(?i:a|b|...)$'")
    return _split_alternatives(regex[len("^(?i:") : -len(")$")])


def _is_wrapped(part: str) -> bool:
    """Whether an alternative is a ``(?:...)`` pattern, not an escaped alias."""
    return part.startswith("(?:") and part.endswith(")")


def _unescape(literal: str) -> str:
    """Invert :func:`re.escape` (which only ever prefixes a backslash)."""
    return re.sub(r"\\(.)", r"\1", literal)


def _from_criteria(criteria: dict[str, dict[str, str]]) -> dict[str, dict[str, object]]:
    """Parse the vocabulary file's cf-xarray shape into :data:`VOCABULARY`'s.

    The inverse of :func:`_to_criteria`; the file rules are in
    ``ocean_skill/vocab/README.md``. A ``standard_name`` with one alternative is that
    entry's canonical name; a *broad* entry lists its own plus its narrower entries',
    so its canonical one is the alternative that is not another key's, and each
    narrower entry gets ``broader`` set to it. From an entry's ``name`` alternatives
    drop its key, its canonical name and (broad entries) its narrower entries'
    alternatives; of the rest a ``(?:...)`` one is a pattern and any other an escaped
    alias.
    """
    std = {k: _alternatives_of(c["standard_name"]) for k, c in criteria.items()}
    names = {k: _alternatives_of(c["name"]) for k, c in criteria.items()}
    single = {alts[0]: k for k, alts in std.items() if len(alts) == 1}
    narrower = {
        k: [single[a] for a in alts if a in single] if len(alts) > 1 else []
        for k, alts in std.items()
    }
    broader_of = {n: k for k, ns in narrower.items() for n in ns}
    vocab: dict[str, dict[str, object]] = {}
    for key, alts in std.items():
        if len(alts) == 1:
            canonical = alts[0]
        else:  # a broad entry: its own name, among the narrower entries' (each a sole)
            canonical = next(a for a in alts if a not in single)
        drop = {re.escape(key), canonical}
        for n in narrower[key]:
            drop.update(names[n])
        rest = [p for p in names[key] if p not in drop]
        patterns = [p[3:-1] for p in rest if _is_wrapped(p)]
        literals = [p for p in rest if not _is_wrapped(p)]
        aliases = [_unescape(p) for p in literals]
        if [re.escape(a) for a in aliases] != literals:
            raise ValueError(f"vocabulary.yaml: {key!r}: an alias is not re.escape()d")
        entry: dict[str, object] = {"standard_name": _unescape(canonical)}
        if aliases:
            entry["aliases"] = aliases
        if patterns:
            entry["patterns"] = patterns
        if key in broader_of:
            entry["broader"] = broader_of[key]
        vocab[key] = entry
    return vocab


def _load_vocabulary() -> dict[str, dict[str, object]]:
    """Read the packaged ``vocab/vocabulary.yaml`` into :data:`VOCABULARY`'s shape."""
    source = importlib.resources.files("ocean_skill") / "vocab" / "vocabulary.yaml"
    return _from_criteria(yaml.safe_load(source.read_text(encoding="utf-8")))


#: Mutable on purpose: :func:`register`/:func:`add_alias`/:func:`add_pattern` edit it
#: in memory, never the file it was loaded from.
VOCABULARY: dict[str, dict[str, object]] = _load_vocabulary()


def _all_names(entry: dict[str, object]) -> list[str]:
    """Every spelling one entry recognizes: its standard_name plus its aliases."""
    return [entry["standard_name"], *entry.get("aliases", [])]  # type: ignore[misc]


def _matchable_names(key: str, entry: dict[str, object]) -> list[str]:
    """Every literal spelling one entry recognizes, its short key included.

    The single source both :func:`_build_index` (resolver side) and
    :func:`_register_custom_criteria` (dataset side) derive from, so a name that
    resolves is guaranteed findable in an actual dataset -- the two used to be built
    from different lists (the index included the key, the criteria didn't), which let
    ``resolve_name("temperature")`` and a real variable literally named
    ``"Temperature"`` disagree about whether it existed. Deduped: a few entries'
    keys equal their own standard_name.
    """
    return list(dict.fromkeys([key, *_all_names(entry)]))


def _alternatives(key: str, entry: dict[str, object]) -> list[str]:
    """One entry's own ``name`` alternatives: escaped literals, then wrapped patterns.

    The literals are :func:`_matchable_names` escaped; each pattern is wrapped
    ``(?:...)`` so it can sit in an alternation whatever it contains -- and so a
    pattern that happens to have no regex metacharacter (``doxy``) can still be told
    from an escaped alias when :func:`_from_criteria` reads the file back. The one
    builder behind both the file shape (:func:`_to_criteria`) and cf-xarray's
    registration (:func:`_register_custom_criteria`).
    """
    return [
        *(re.escape(n) for n in _matchable_names(key, entry)),
        *(f"(?:{p})" for p in entry.get("patterns", [])),  # type: ignore[union-attr]
    ]


def _anchor(alternatives: Iterable[str]) -> str:
    """Join alternatives (deduplicated, in order) as ``^(?i:a|b|...)$``."""
    return "^(?i:" + "|".join(dict.fromkeys(alternatives)) + ")$"


def _narrower_keys(vocab: dict[str, dict[str, object]]) -> dict[str, list[str]]:
    """Map each broad entry's key to its narrower entries' keys, in ``vocab`` order.

    :func:`_build_narrower` for a bare dict: each ``broader`` is resolved against
    ``vocab`` itself (any spelling or pattern match of the broad entry) rather than
    through the live index, and a link that function would warn about -- naming
    nothing known, the entry itself, or an entry that is itself one specific kind of
    another -- is left out, silently.
    """

    def target(broader: object) -> str | None:
        wanted = str(broader)
        for k, e in vocab.items():
            if wanted.lower() in {n.lower() for n in _matchable_names(k, e)} or any(
                re.fullmatch(p, wanted, re.IGNORECASE)
                for p in e.get("patterns", [])  # type: ignore[union-attr]
            ):
                return k
        return None

    kids: dict[str, list[str]] = {}
    for key, entry in vocab.items():
        parent = target(entry["broader"]) if "broader" in entry else None
        if (
            parent is not None
            and parent != key
            and vocab[parent].get("broader") is None
            and vocab[parent]["standard_name"] != entry["standard_name"]
        ):
            kids.setdefault(parent, []).append(key)
    return kids


def _to_criteria(vocab: dict[str, dict[str, object]]) -> dict[str, dict[str, str]]:
    """Write ``vocab`` in the vocabulary file's shape (see :func:`_from_criteria`).

    One ``{"name": ..., "standard_name": ...}`` per key, each an anchored
    case-insensitive ``^(?i:...)$``: ``name`` lists the entry's own alternatives
    (:func:`_alternatives`), ``standard_name`` its canonical name. A *broad* entry
    (one that others name as their ``broader``) additionally lists every narrower
    entry's, in both, so a plain cf-xarray ``ds.cf["mld"]`` finds any definition. What
    a ``save()`` would write; the lossless-parse test pins that loading the packaged
    file and writing it back gives the same dict.
    """
    kids = _narrower_keys(vocab)
    criteria: dict[str, dict[str, str]] = {}
    for key, entry in vocab.items():
        names = _alternatives(key, entry)
        canonical = [re.escape(str(entry["standard_name"]))]
        for kid in kids.get(key, []):
            names += _alternatives(kid, vocab[kid])
            canonical.append(re.escape(str(vocab[kid]["standard_name"])))
        criteria[key] = {"name": _anchor(names), "standard_name": _anchor(canonical)}
    return criteria


def _build_index() -> dict[str, str]:
    """Map every short key and spelling, **lowercased**, -> its canonical standard_name.

    Keyed lowercase so matching ignores case throughout (``"Chlorophyll"``,
    ``"CHL"``, ``"chlorophyll"`` are one name): products disagree on
    capitalization as readily as on spelling, and a case difference is never a
    meaningful distinction between two variables.

    Rebuilt by :func:`_refresh` whenever :data:`VOCABULARY` changes. Warns rather
    than silently picking a winner if two entries claim the same spelling — that
    is always a vocabulary bug (one physical quantity, two concepts), and the
    resulting resolution would otherwise depend on nothing more than dict order.
    """
    index: dict[str, str] = {}
    claimed_by: dict[str, str] = {}  # spelling -> the key that claimed it first
    for key, entry in VOCABULARY.items():
        sn = entry["standard_name"]
        for name in _matchable_names(key, entry):
            lowered = name.lower()
            previous = index.get(lowered)
            if previous is not None and previous != sn:
                warnings.warn(
                    f"vocabulary collision: {name!r} is claimed by both "
                    f"{claimed_by[lowered]!r} (-> {previous!r}) and {key!r} "
                    f"(-> {sn!r}); {key!r} wins. Remove it from one of the two "
                    "entries.",
                    stacklevel=3,
                )
            index[lowered] = sn  # type: ignore[assignment]
            claimed_by[lowered] = key
    return index


#: standard_name -> its full vocabulary entry, for equivalent_names below.
def _build_by_standard_name() -> dict[str, dict[str, object]]:
    return {entry["standard_name"]: entry for entry in VOCABULARY.values()}  # type: ignore[misc]


#: standard_name -> the short key a caller actually types, for nickname() below.
#: One entry per standard_name is guaranteed by the same collision check
#: _build_index runs over every entry's spellings (a shipped-vocabulary regression
#: test keeps this true: test_shipped_vocabulary_has_no_collisions).
def _build_key_by_standard_name() -> dict[str, str]:
    return {entry["standard_name"]: key for key, entry in VOCABULARY.items()}  # type: ignore[misc]


def _build_narrower() -> dict[str, tuple[str, ...]]:
    """Map a broad entry's standard_name to its specific entries' standard_names.

    The table :func:`narrower_names` and :func:`covers` read: every entry with a
    ``broader`` is filed, by its own standard_name, under the standard_name that
    ``broader`` resolves to, each tuple sorted so results never depend on dict order.
    ``broader`` may be any spelling of the broad entry (its key, standard_name, an
    alias), so it is resolved the way a caller's name is -- which is why
    :func:`_refresh` builds this only *after* the index and patterns exist.

    :func:`register` refuses a ``broader`` that breaks the one-level rule, but
    :data:`VOCABULARY` can also be edited by hand, so a link that names nothing
    known, names the entry itself, or names an entry that is itself one specific kind
    of something else is warned about and left out here, exactly as
    :func:`_build_index` warns rather than guess on a colliding spelling. Nothing
    below ever follows a chain, so :func:`covers` cannot become transitive by accident.
    """
    children: dict[str, set[str]] = {}
    for key, entry in VOCABULARY.items():
        broader = entry.get("broader")
        if broader is None:
            continue
        sn = str(entry["standard_name"])
        parent = resolve_name(str(broader))
        parent_entry = _BY_STANDARD_NAME.get(parent)
        if parent == sn:
            problem = "is the entry itself"
        elif parent_entry is None:
            problem = "is not a known name"
        elif parent_entry.get("broader") is not None:
            problem = f"resolves to {parent!r}, itself one specific kind of another"
        else:
            children.setdefault(parent, set()).add(sn)
            continue
        warnings.warn(
            f"vocabulary broader: {key!r} (-> {sn!r}) names broader={broader!r}, "
            f"which {problem}; ignoring the link. A broader entry must be a known "
            "concept that is not itself one specific kind of another (one level "
            "only).",
            stacklevel=3,
        )
    return {parent: tuple(sorted(kids)) for parent, kids in children.items()}


def _build_patterns() -> list[tuple[re.Pattern[str], str]]:
    """Compile every entry's ``patterns`` into one ``(regex, standard_name)`` pair.

    One alternation per entry (not per pattern), case-insensitive. Raises
    :class:`re.error` immediately -- at :func:`_refresh` time, before a caller ever
    resolves a name -- if a pattern is not valid regex.
    """
    compiled: list[tuple[re.Pattern[str], str]] = []
    for entry in VOCABULARY.values():
        patterns = entry.get("patterns")
        if not patterns:
            continue
        alt = "|".join(f"(?:{p})" for p in patterns)  # type: ignore[union-attr]
        compiled.append((re.compile(alt, re.IGNORECASE), entry["standard_name"]))  # type: ignore[arg-type]
    return compiled


_INDEX: dict[str, str] = {}
_BY_STANDARD_NAME: dict[str, dict[str, object]] = {}
_KEY_BY_STANDARD_NAME: dict[str, str] = {}
_PATTERNS: list[tuple[re.Pattern[str], str]] = []
_NARROWER: dict[str, tuple[str, ...]] = {}


def _pattern_lookup(name: str) -> str | None:
    """Run the regex tier shared by :func:`resolve_name` and :func:`is_known`.

    Only reached once ``name`` has already missed the exact :data:`_INDEX` lookup.
    Fullmatch (never a substring) against every entry with ``patterns``; a name
    matched by more than one entry's patterns is ambiguous, so — mirroring
    :func:`_build_index`'s refusal to silently pick a winner on a literal
    collision — this warns and returns ``None`` rather than guessing.
    """
    hits = {sn for pattern, sn in _PATTERNS if pattern.fullmatch(name)}
    if len(hits) > 1:
        warnings.warn(
            f"{name!r} matches vocabulary patterns from more than one entry "
            f"({sorted(hits)!r}); refusing to guess. Narrow the patterns so only "
            "one entry claims it.",
            stacklevel=_stacklevel.find(),
        )
        return None
    return next(iter(hits), None)


def resolve_name(name: str) -> str:
    """Return the canonical CF standard_name for any spelling in :data:`VOCABULARY`.

    Accepts a short key (``"oxygen"``), the canonical standard_name itself, any
    alias, or anything one entry's ``patterns`` fullmatch (``"Temperature_CTD"``),
    **in any capitalization** (``"Oxygen"``, ``"OXYGEN"``) — all resolve to the same
    standard_name. A name with no vocabulary entry at all passes through unchanged
    (keeping its original case), so this is safe to call on any variable name,
    curated or not.
    """
    exact = _INDEX.get(name.lower())
    if exact is not None:
        return exact
    return _pattern_lookup(name) or name


def resolve_and_report(name: str, *, context: str = "") -> str:
    """:func:`resolve_name`, warning once (naming both) when the name actually changes.

    Used at the point a caller hands over a variable name they chose themselves
    (e.g. :meth:`ocean_skill.comparison.Comparison.__init__`), so using a short name
    or an alias is never silently unclear about which variable is actually meant.
    """
    canonical = resolve_name(name)
    if canonical != name:
        suffix = f" ({context})" if context else ""
        warnings.warn(
            f"{name!r} resolved to standard_name {canonical!r}{suffix}",
            stacklevel=_stacklevel.find(),
        )
    return canonical


def is_known(name: str) -> bool:
    """Whether the vocabulary recognizes ``name`` at all.

    True for a key, standard_name, alias, or a fullmatch against some entry's
    ``patterns``. Lets a caller tell "this source does not have X" from "I cannot
    tell": catalog metadata indexes CF standard_names, so a name the vocabulary has
    never heard of (a raw model variable like ``spChl``) is unknowable from metadata
    alone and must not be treated as absent. Kept consistent with
    :func:`resolve_name`'s two tiers on purpose -- :func:`ocean_skill.comparison.
    compare`'s absent-vs-unknowable check depends on that consistency.
    """
    return name.lower() in _INDEX or _pattern_lookup(name) is not None


def equivalent_names(name: str) -> set[str]:
    """Return every spelling this vocabulary treats as the same variable as ``name``.

    The canonical standard_name plus all its aliases —
    the set that counts as "the same variable" when matching a requested variable
    against a source's declared ``variables`` (see
    :func:`ocean_skill.comparison.compare`). A name with no vocabulary entry
    returns just itself. This is always a *finite* set: an entry's ``patterns``
    recognize a whole family of spellings with no enumeration, so they never appear
    here -- use :func:`same_quantity` to compare a possibly pattern-matched name
    against a declared one instead of set membership.

    Specific definitions (:func:`narrower_names`) are deliberately not included: a
    potential-density mixed layer depth is one *kind* of mixed layer depth, not the
    same variable as the generic one, and this set stays the symmetric "same
    variable" one. See :func:`covers` for the one-directional question of whether a
    declared name satisfies a request once definitions count.
    """
    entry = _BY_STANDARD_NAME.get(resolve_name(name))
    return {name} if entry is None else set(_all_names(entry))


def same_quantity(a: str, b: str) -> bool:
    """Whether two spellings resolve to the same canonical quantity.

    Unlike :func:`equivalent_names`' set membership, this also reaches spellings
    only a pattern recognizes (``"Temperature_CTD"``) and two declared names
    differing only by case -- case is never a meaningful distinction here (see
    :func:`_build_index`). Two names the vocabulary has never heard of compare
    equal only when they are literally the same spelling (case-insensitively),
    since :func:`resolve_name` passes an unknown name through unchanged.

    Symmetric, and about *identity* only: a generic name and one of its specific
    definitions are not the same quantity here (see :func:`covers` for that).
    """
    return resolve_name(a).lower() == resolve_name(b).lower()


def narrower_names(name: str) -> tuple[str, ...]:
    """Return the standard_names of the specific definitions one level under ``name``.

    ``name`` may be any spelling :func:`resolve_name` accepts for the *broad* entry
    -- ``"mld"``, ``"mixed_layer_depth"``, the full standard_name, in any
    capitalization -- and the result is the sorted standard_names of every entry
    that names it as its ``broader`` (see the module docstring): for the generic
    ``"mld"``, the four definition-specific mixed layer depths. ``()`` for a name
    that is itself a specific definition, one the vocabulary has never heard of, or
    a concept nothing is a kind of (``"temperature"``): the relation is one level
    and one direction, so a specific name has nothing narrower, and this never
    reports a *broader* name.

    >>> from ocean_skill import vocabulary
    >>> vocabulary.narrower_names("mld")[0]
    'ocean_mixed_layer_thickness_defined_by_mixing_scheme'
    >>> vocabulary.narrower_names("mld_by_sigma_theta")
    ()
    """
    return _NARROWER.get(resolve_name(name), ())


def covers(requested: str, declared: str) -> bool:
    """Whether a source declaring ``declared`` satisfies a request for ``requested``.

    True when the two are the same quantity (:func:`same_quantity`: every spelling,
    alias, pattern match and capitalization of one variable), *or* when ``declared``
    is one of the specific definitions under ``requested``
    (:func:`narrower_names`) -- asking for ``"mld"`` is satisfied by a source that
    declares ``ocean_mixed_layer_thickness_defined_by_sigma_theta``.

    One-directional, the way a request is. Asking for a specific definition is never
    satisfied by the generic name (a source declaring plain
    ``ocean_mixed_layer_thickness`` has not said which definition it carries) nor by
    a sibling definition: the ``sigma_theta`` and ``sigma_t`` names do not cover each
    other in either direction, though one is a prefix of the other -- names are
    compared whole, never by prefix.

    >>> from ocean_skill import vocabulary
    >>> vocabulary.covers("mld", "ocean_mixed_layer_thickness_defined_by_sigma_theta")
    True
    >>> vocabulary.covers("mld_by_sigma_theta", "mld")
    False
    """
    wanted = resolve_name(requested)
    got = resolve_name(declared)
    if wanted.lower() == got.lower():  # same_quantity(requested, declared)
        return True
    return got.lower() in (n.lower() for n in _NARROWER.get(wanted, ()))


def nickname(name: str) -> str | None:
    """Return the short vocabulary key ``name`` resolves to, or ``None``.

    The reverse of typing a nickname: given a real-world spelling (an alias, a
    pattern-recognized name like ``"Temperature_CTD"``, or the canonical
    standard_name itself), return the short key a caller would have typed instead
    (``"temperature"``). ``None`` for anything :func:`is_known` also says no to --
    unrecognized or ambiguous (patterns from more than one entry).
    """
    return _KEY_BY_STANDARD_NAME.get(resolve_name(name))


#: Coordinate-recognition vocabulary -- the axis analogue of :data:`VOCABULARY`
#: above. One entry per axis (``T``/``X``/``Y``/``Z``):
#:
#: - ``kind``: the name :func:`ocean_skill.cf.find_coord` takes for this axis
#:   (``"time"``, ``"longitude"``, ``"latitude"``, ``"vertical"``).
#: - ``tokens``: whole-token spellings that name it, matched case-insensitively
#:   against a units-stripped column/variable name (see :func:`matches_axis`).
#: - ``direct`` (``Z`` only): the subset of ``tokens`` naming an actual reading
#:   rather than a pressure conversion -- ranked ahead of the full set so
#:   :func:`ocean_skill.tabular.coord_column` prefers a depth-shaped column over a
#:   pressure-shaped one.
#: - ``exclude``: whole-token spellings that DISQUALIFY an otherwise-matching name.
#:   ``bottom`` is the motivating case: ``Depth_bottom``/``bottom_depth`` states the
#:   seafloor/station depth -- a data column, never the vertical coordinate --
#:   even though "depth" is right there in the name.
#: - ``fallbacks``: exact-name fallbacks :func:`ocean_skill.cf.find_coord` tries
#:   when cf-xarray detects nothing at all (ROMS writes ``units="degrees East"``
#:   and ``ocean_time`` with ``units="second"``, so cf-xarray finds neither).
#:   Order matters -- rho-points before the staggered/coarse variants. ``sigma0``
#:   is the isopycnal axis :func:`ocean_skill.roms.to_sigma0` produces -- a
#:   vertical axis in every way that matters here, just not a depth.
#:
#: One definition, consumed by :mod:`ocean_skill.tabular` (column-name matching)
#: and :mod:`ocean_skill.cf` (cf-xarray + gridded name-fallback matching), so the
#: two matchers -- and any future one -- cannot drift apart the way
#: :data:`VOCABULARY` keeps variable matching from drifting between callers.
COORD_VOCABULARY: dict[str, dict[str, object]] = {
    "T": {
        "kind": "time",
        "tokens": ("time", "date", "datetime", "timestamp"),
        "fallbacks": ("time", "ocean_time", "t", "T"),
    },
    "X": {
        "kind": "longitude",
        "tokens": ("longitude", "lon", "long"),
        "fallbacks": (
            "lon_rho",
            "lon",
            "longitude",
            "nav_lon",
            "x_rho",
            "lon_u",
            "lon_v",
        ),
    },
    "Y": {
        "kind": "latitude",
        "tokens": ("latitude", "lat"),
        "fallbacks": (
            "lat_rho",
            "lat",
            "latitude",
            "nav_lat",
            "y_rho",
            "lat_u",
            "lat_v",
        ),
    },
    "Z": {
        "kind": "vertical",
        "tokens": ("depth", "z", "pressure", "pres"),
        "direct": ("depth", "z"),
        "exclude": ("bottom",),
        "fallbacks": (
            "depth",
            "z",
            "lev",
            "s_rho",
            "z_rho",
            "s_w",
            "z_w",
            "depth_surface",
            "sigma0",
        ),
    },
}


def _token_pattern(*words: str) -> re.Pattern:
    """Compile a whole-token, case-insensitive alternation over ``words``.

    Split on any run of non-alphanumeric characters, underscore included (the same
    idiom :data:`ocean_skill.tabular._QC_NAME` uses), so a word buried inside a
    longer name never counts: "month" is not "time", "latency" is not "lat", "zone"
    is not "z", "along" is not "lon". This is the spirit of cf-pandas'/cf-xarray's
    own coordinate-criteria regexes (``cf_pandas.criteria.guess_regex``,
    ``cf_xarray.criteria``), deliberately narrower: cf-pandas' own time regex
    (``(?=.*time|min|hour|day|week|month|year)[0-9]*``) matches bare "month" for
    exactly this reason; this refuses that on purpose rather than inheriting it.
    """
    alt = "|".join(re.escape(w) for w in words)
    return re.compile(rf"(?:^|[^0-9A-Za-z])(?:{alt})(?:[^0-9A-Za-z]|$)", re.IGNORECASE)


#: kind -> axis (``"vertical"`` -> ``"Z"``), for callers like
#: :func:`ocean_skill.cf.find_coord` that take a kind rather than an axis letter.
COORD_AXIS_BY_KIND: dict[str, str] = {
    entry["kind"]: axis for axis, entry in COORD_VOCABULARY.items()  # type: ignore[misc]
}

#: kind -> exact-name fallbacks, derived from :data:`COORD_VOCABULARY` so the two
#: cannot drift apart.
COORD_FALLBACKS: dict[str, tuple[str, ...]] = {
    entry["kind"]: entry["fallbacks"] for entry in COORD_VOCABULARY.values()  # type: ignore[misc]
}

_COORD_PATTERNS: dict[str, re.Pattern] = {
    axis: _token_pattern(*entry["tokens"]) for axis, entry in COORD_VOCABULARY.items()  # type: ignore[misc]
}
#: Z's direct-reading subset (see ``COORD_VOCABULARY["Z"]["direct"]``) -- an axis
#: with no ``direct`` entry falls back to its full token set, so this dict always
#: has one pattern per axis.
_COORD_DIRECT_PATTERNS: dict[str, re.Pattern] = {
    axis: _token_pattern(*entry.get("direct", entry["tokens"]))  # type: ignore[arg-type]
    for axis, entry in COORD_VOCABULARY.items()
}
_COORD_EXCLUDE_PATTERNS: dict[str, re.Pattern | None] = {
    axis: (_token_pattern(*entry["exclude"]) if entry.get("exclude") else None)  # type: ignore[arg-type]
    for axis, entry in COORD_VOCABULARY.items()
}


def excluded_from_axis(base: str, axis: str) -> bool:
    """Whether ``base`` (a units-stripped name) is disqualified from ``axis``.

    True only for :data:`COORD_VOCABULARY`'s ``exclude`` tokens (currently just
    ``Z``'s ``"bottom"``) -- checked as a *separate* pass rather than folded into
    the matching pattern as a lookahead, because :meth:`re.Pattern.search`'s
    lookahead only sees text *after* the attempted match position: a
    ``(?!.*bottom)`` prefix on the ``Z`` pattern would still match the ``depth`` in
    ``bottom_depth`` (there, the exclude token comes *before* the match), so it
    must be tried independently over the whole name instead.
    """
    pattern = _COORD_EXCLUDE_PATTERNS.get(axis)
    return pattern is not None and bool(pattern.search(base))


def matches_axis(base: str, axis: str, *, direct_only: bool = False) -> bool:
    """Whether ``base`` (a units-stripped name) is a plausible name for ``axis``.

    True when ``axis``'s token pattern matches and ``base`` isn't
    :func:`excluded_from_axis` for it -- callers never need to check exclusion
    themselves. ``direct_only`` (``Z`` only) restricts to the ``direct`` reading
    subset (depth/z), skipping the pressure-conversion spellings -- see
    :func:`ocean_skill.tabular.coord_column`, which tries it both ways.
    """
    patterns = _COORD_DIRECT_PATTERNS if direct_only else _COORD_PATTERNS
    pattern = patterns.get(axis)
    return (
        pattern is not None
        and bool(pattern.search(base))
        and not excluded_from_axis(base, axis)
    )


@dataclass(frozen=True)
class MatchReport:
    """Which declared variable names the vocabulary recognizes, and as what.

    Built by :func:`match_report`. Always computed live against the *current*
    vocabulary -- never persisted anywhere (a catalog's declared variables don't
    change, but which of them the vocabulary recognizes can, the moment a new
    alias or pattern ships) -- so a report is only ever as current as the moment
    it was printed, and asking again after a vocabulary change gets a fresh answer.
    """

    #: nickname -> the declared names (sorted) that resolve to it.
    matched: dict[str, list[str]]
    #: Declared names the vocabulary does not recognize at all, sorted.
    unmatched: list[str]

    @property
    def collisions(self) -> dict[str, list[str]]:
        """Nicknames claimed by more than one declared name.

        Not necessarily a mistake -- a raw/flag/qc triplet or a genuine duplicate
        column both land here -- but always worth a curator's second look.
        """
        return {k: v for k, v in self.matched.items() if len(v) > 1}

    def __str__(self) -> str:
        lines: list[str] = []
        if self.matched:
            lines.append(f"matched ({len(self.matched)}):")
            width = max(len(k) for k in self.matched)
            for key in sorted(self.matched):
                names = self.matched[key]
                suffix = f"   ({len(names)} variables)" if len(names) > 1 else ""
                lines.append(f"  {key:<{width}}  <- {', '.join(names)}{suffix}")
        else:
            lines.append("matched (0)")
        if self.unmatched:
            lines.append(f"unmatched ({len(self.unmatched)}):")
            lines.append(f"  {', '.join(self.unmatched)}")
        else:
            lines.append("unmatched (0)")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return str(self)

    def _repr_html_(self) -> str:
        """Notebook rendering: monospace, wrapped, and escaped (see :mod:`_display`)."""
        return (
            "<pre style='white-space:pre-wrap; margin:0'>"
            f"{html.escape(str(self))}</pre>"
        )


def match_report(names: Iterable[str]) -> MatchReport:
    """Group ``names`` by the vocabulary nickname each resolves to.

    For every name: :func:`nickname` decides which bucket it lands in, exactly
    the same resolution :func:`resolve_name`/:func:`is_known` use everywhere
    else -- this reports what the vocabulary *already* does, it doesn't add a
    second notion of matching. An ambiguous pattern match (claimed by more than
    one entry) warns from :func:`_pattern_lookup`; that warning is suppressed
    here and the name simply lands in ``unmatched`` -- the report is the place
    that surfaces it, not a warning fired once per :func:`osk.describe` call.
    """
    matched: dict[str, list[str]] = {}
    unmatched: list[str] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name in names:
            key = nickname(name)
            if key is None:
                unmatched.append(name)
            else:
                matched.setdefault(key, []).append(name)
    for names_list in matched.values():
        names_list.sort()
    return MatchReport(matched=matched, unmatched=sorted(unmatched))


@dataclass(frozen=True)
class CoordReport:
    """Which declared column names the coordinate vocabulary recognizes as which axis.

    Built by :func:`coord_report`. The coordinate analogue of :class:`MatchReport`:
    live, never persisted -- which axis a name resolves to can change the moment
    :data:`COORD_VOCABULARY` changes (the ``bottom`` exclusion this class exists to
    make visible is exactly that kind of change), so a report is only ever as
    current as the moment it was printed.
    """

    #: axis (``"T"``/``"X"``/``"Y"``/``"Z"``) -> the declared names (sorted) that
    #: resolve to it.
    matched: dict[str, list[str]]
    #: Axes of ``T``/``X``/``Y``/``Z`` no declared name resolves to, in that order.
    missing: list[str]

    @property
    def collisions(self) -> dict[str, list[str]]:
        """Axes claimed by more than one declared name.

        Not necessarily a mistake -- a reading/pressure pair both named Z-shaped is
        the common case -- but always worth a curator's second look, the same as
        :attr:`MatchReport.collisions`.
        """
        return {k: v for k, v in self.matched.items() if len(v) > 1}

    def __str__(self) -> str:
        lines: list[str] = []
        if self.matched:
            lines.append(f"matched ({len(self.matched)}):")
            labels = {
                axis: f"{axis} ({COORD_VOCABULARY[axis]['kind']})"
                for axis in self.matched
            }
            width = max(len(label) for label in labels.values())
            for axis in ("T", "X", "Y", "Z"):
                if axis not in self.matched:
                    continue
                names = self.matched[axis]
                suffix = f"   ({len(names)} columns)" if len(names) > 1 else ""
                lines.append(f"  {labels[axis]:<{width}}  <- {', '.join(names)}{suffix}")
        else:
            lines.append("matched (0)")
        if self.missing:
            lines.append(f"missing ({len(self.missing)}): {', '.join(self.missing)}")
        else:
            lines.append("missing (0)")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return str(self)

    def _repr_html_(self) -> str:
        """Notebook rendering: monospace, wrapped, and escaped (see :mod:`_display`)."""
        return (
            "<pre style='white-space:pre-wrap; margin:0'>"
            f"{html.escape(str(self))}</pre>"
        )


def coord_report(names: Iterable[str]) -> CoordReport:
    """Group ``names`` by the coordinate axis (``T``/``X``/``Y``/``Z``) each resolves to.

    For every name: :func:`ocean_skill.tabular.coord_axis_of` decides which axis
    (if any) it lands under -- the same resolution every other coordinate-column
    caller uses, units-stripping and the ``bottom`` exclusion included -- so this
    reports what matching *already* does, exactly as :func:`match_report` does for
    variables. A name matching no axis (or excluded from the only axis it would
    otherwise match, like ``Depth_bottom``) simply doesn't appear in ``matched`` --
    unlike :func:`match_report`, there's no separate "unmatched names" list, since
    most declared names are legitimately not coordinates at all. What matters here
    is which of the four axes got a match and which did not, hence ``missing``.
    """
    from ocean_skill import tabular

    matched: dict[str, list[str]] = {}
    for name in names:
        axis = tabular.coord_axis_of(name)
        if axis is not None:
            matched.setdefault(axis, []).append(name)
    for names_list in matched.values():
        names_list.sort()
    missing = [axis for axis in ("T", "X", "Y", "Z") if axis not in matched]
    return CoordReport(matched=matched, missing=missing)


#: The dict :func:`_register_custom_criteria` last handed to cf-xarray, so the next
#: registration can find it again among whatever else is registered (see there).
_REGISTERED: dict[str, dict[str, str]] = {}


def _register_custom_criteria() -> None:
    """Register every :data:`VOCABULARY` entry's spellings with cf-xarray.

    One ``{"name": pattern}`` entry per spelling (all of an entry's short key,
    standard_name, and aliases -- see :func:`_matchable_names` -- registered under
    each so asking for any one finds the others), matched against the variable's own
    *name* only, deliberately not its ``standard_name`` attribute. Data is renamed to
    the standard_name directly (e.g.
    by :func:`ocean_skill.roms.standardize`, or the generic rename in
    :func:`ocean_skill.sources.read`) far more often than it carries that as a
    separate attribute, so name-matching already covers the real case — and
    attribute-matching actively misfires on WOA, whose auxiliary companion variables
    (``n_dd``/``n_se``/... — sample size, standard error) all carry the *same*
    ``standard_name`` as the actual data variable, which cf-xarray then (correctly,
    on that input) reports as ambiguous.

    Each pattern is ``^(?i:...)$``, the shape the shared vocabulary file uses:
    case-insensitive (see :func:`_build_index`) and **anchored at both ends**. The
    trailing ``$`` is load-bearing — cf-xarray matches with :func:`re.match`, which
    anchors only the *start*, so an unanchored pattern also matches anything merely
    *prefixed* by a registered spelling. That silently returned an ERDDAP/OOI QC-flag
    companion (``..._in_sea_water_qc_agg``) as if it were the data variable whenever
    the real one was absent — found by testing, not hypothetical.

    An entry's ``patterns`` join the same alternation, each wrapped ``(?:...)`` (see
    :func:`_alternatives`) so they inherit the anchors without escaping them. They are
    never registered as criteria *keys* the way literal spellings are — every
    dataset-side lookup (:func:`ocean_skill.units.find_variable`) arrives here already
    canonicalized by :func:`resolve_name`, and the canonical standard_name is always a
    literal key.

    A *broad* entry — one that others name as their ``broader`` (see
    :func:`narrower_names`) — also matches every spelling and pattern of those
    specific entries, one level down: ``ds.cf["ocean_mixed_layer_thickness"]``
    finds a variable named ``..._defined_by_sigma_theta``, and also one under a raw
    product name whose ``standard_name`` *attribute* is one of those specific names
    (the one attribute criterion registered here — for the specific names only,
    exactly what cf-xarray's built-in matching already does when asked for a specific
    name, so the broad name finds the same variables they do). That is the dataset half
    of the ``broader`` relation, and it lives here, in the criteria, rather than in
    lookup code on purpose: it is plain data that cf-xarray alone acts on, so any
    tool registering these criteria (ROMS-Tools, xroms) gets the same broad-to-specific
    lookup with no ocean-skill code at all. The specific entries keep their own
    criteria, so a request for one definition never matches the generic name or a
    sibling. A dataset carrying two definitions matches the broad key twice, which
    is cf-xarray's own "multiple variables" error — reported, not guessed at (see
    :func:`ocean_skill.units.find_variable`).

    The criteria are *merged* into cf-xarray's global list, not set over it:
    ``cf_xarray.set_options(custom_criteria=...)`` replaces the whole list, which
    would wipe another package's criteria (ROMS-Tools', xroms', the user's own). What
    is registered is ``[ours, *others]``, where ``others`` is the list as it stands
    minus the dict this function registered last -- found by equality with
    :data:`_REGISTERED`, since cf-xarray deep-copies what it is given and identity
    cannot be tracked -- so refreshing never duplicates ours, and leaves the rest as
    it was. Ours comes first because cf-xarray looks a key up through a
    :class:`~collections.ChainMap`, where the first dict defining it wins.
    """
    import cf_xarray

    global _REGISTERED

    # One standard_name per entry is guaranteed (see _build_key_by_standard_name), so
    # the specific entries' own alternations can be looked up by standard_name.
    by_standard_name = {
        str(entry["standard_name"]): _alternatives(key, entry)
        for key, entry in VOCABULARY.items()
    }
    criteria: dict[str, dict[str, str]] = {}
    for key, entry in VOCABULARY.items():
        sn = str(entry["standard_name"])
        narrower = _NARROWER.get(sn, ())
        parts = list(by_standard_name[sn])
        for specific in narrower:
            parts += by_standard_name.get(specific, [])
        entry_criteria = {"name": _anchor(parts)}
        if narrower:
            # A variable under a raw product name that states its specific definition
            # only as an attribute: cf-xarray already finds it by that attribute when
            # asked for the specific name (its built-in standard_name matching), so the
            # broad name finds it the same way. Only the *specific* names are listed --
            # the broad entry's own standard_name attribute is cf-xarray's built-in
            # match already, and widening it is what misfires on WOA-style companions.
            entry_criteria["standard_name"] = _anchor(re.escape(n) for n in narrower)
        for name in _matchable_names(key, entry):
            criteria[name] = entry_criteria
    registered = cf_xarray.options.OPTIONS["custom_criteria"]
    others = [c for c in registered if c != _REGISTERED]
    cf_xarray.set_options(custom_criteria=[criteria, *others])
    _REGISTERED = criteria


def _refresh() -> None:
    """Rebuild the resolver index and cf-xarray registration from :data:`VOCABULARY`.

    Called once at import time, and again by :func:`register`/:func:`add_alias`/
    :func:`add_pattern` after any live edit -- editing ``VOCABULARY`` directly (e.g.
    ``VOCABULARY["chlorophyll"]["aliases"].append(...)``) does *not* alone update
    :func:`resolve_name` or cf-xarray's own registration, since both are cached at
    module load; call this afterwards if you edit the dict by hand instead of
    through those functions.

    Order matters for one table: ``_NARROWER`` (the ``broader`` relation, see
    :func:`_build_narrower`) is built *after* the index and patterns, because it
    resolves each entry's ``broader`` with :func:`resolve_name`.
    """
    global _INDEX, _BY_STANDARD_NAME, _KEY_BY_STANDARD_NAME, _PATTERNS, _NARROWER
    _INDEX = _build_index()
    _BY_STANDARD_NAME = _build_by_standard_name()
    _KEY_BY_STANDARD_NAME = _build_key_by_standard_name()
    _PATTERNS = _build_patterns()
    _NARROWER = _build_narrower()
    _register_custom_criteria()


def _check_broader(
    key: str, standard_name: str, broader: str, aliases: list[str] | None
) -> None:
    """Raise :class:`ValueError` unless ``broader`` keeps the relation one level deep.

    Run by :func:`register` before it touches :data:`VOCABULARY`, and about the entry
    as it *will* be rather than as the vocabulary currently stands: registering over
    an existing key replaces that entry's spellings, so ``broader`` is checked
    against the new entry's own key, standard_name and aliases as well as against
    what is already registered. Four ways to get it wrong, each of which would leave
    :func:`covers` answering something other than "one specific kind of":

    - ``broader`` names the entry itself (by any of its own spellings, or by another
      entry's spelling of the same standard_name);
    - ``broader`` is not a name the vocabulary knows, so there is nothing to be a
      specific kind of;
    - ``broader`` names an entry that is itself one specific kind of another (a
      chain);
    - the entry being registered already has specific entries under it, so giving it
      a ``broader`` too would turn them into the far end of a chain.

    A ``broader`` that is not a single name (a list of several broader entries, say)
    is a :class:`TypeError`: an entry is one specific kind of *one* broader entry.
    """
    if not isinstance(broader, str):
        raise TypeError(
            f"register({key!r}, ...): broader must be one name -- the key, "
            "standard_name or any spelling of a single broader entry -- not "
            f"{broader!r}; an entry is one specific kind of one broader entry."
        )
    own = {n.lower() for n in (key, standard_name, *(aliases or []))}
    if broader.lower() in own or resolve_name(broader).lower() == standard_name.lower():
        raise ValueError(
            f"register({key!r}, ...): broader={broader!r} names this entry itself, "
            "and an entry cannot be one specific kind of itself. broader must be a "
            "different, broader concept."
        )
    if not is_known(broader):
        raise ValueError(
            f"register({key!r}, ...): broader={broader!r} is not a known vocabulary "
            "name, so there is nothing for this entry to be one specific kind of. "
            "Register the broader concept first, or check VOCABULARY for its key."
        )
    parent = resolve_name(broader)
    grandparent = _BY_STANDARD_NAME.get(parent, {}).get("broader")
    if grandparent is not None:
        raise ValueError(
            f"register({key!r}, ...): broader={broader!r} is {parent!r}, which is "
            f"itself one specific kind of {grandparent!r}. The relation is one level "
            f"only (broad -> specific), so name {grandparent!r} instead, or leave "
            "broader out."
        )
    replaced = VOCABULARY.get(key)
    under_it = _NARROWER.get(standard_name) or (
        _NARROWER.get(str(replaced["standard_name"])) if replaced else None
    )
    if under_it:
        raise ValueError(
            f"register({key!r}, ...): this entry already has specific entries under "
            f"it ({list(under_it)!r}), so it cannot itself be one specific kind of "
            f"{broader!r}; the relation is one level only (broad -> specific)."
        )


def register(
    key: str,
    standard_name: str,
    *,
    aliases: list[str] | None = None,
    patterns: list[str] | None = None,
    broader: str | None = None,
) -> None:
    """Add (or replace) one vocabulary entry, live -- no restart needed.

    For a wholly new concept. To add a spelling or pattern to a concept that
    already has an entry, prefer :func:`add_alias`/:func:`add_pattern` instead, so
    you don't have to repeat its existing ``standard_name``/aliases just to add one
    more.

    ``broader`` makes the new entry one *specific kind of* an existing, broader one
    -- any spelling of it, ``"mld"`` for instance (see the module docstring's
    ``broader`` bullet): asking for the broad name then finds this entry too, while
    asking for this one never finds the broad name or a sibling. It is validated,
    like ``patterns``, before anything is stored, and raises :class:`ValueError` if
    it is not a known name, names the entry itself, names an entry that is itself
    one specific kind of another (one level only, no chains), or if the entry being
    registered already has specific entries of its own under it. Like ``aliases``
    and ``patterns`` it belongs to the entry that a re-registration replaces, so
    pass it again when re-registering a specific entry to keep it.

    >>> from ocean_skill import vocabulary
    >>> vocabulary.register(
    ...     "ph", "sea_water_ph_reported_on_total_scale", aliases=["ph_total"]
    ... )
    >>> vocabulary.resolve_name("ph")
    'sea_water_ph_reported_on_total_scale'
    """
    for pattern in patterns or []:
        re.compile(pattern)  # validate before touching VOCABULARY at all
    if broader is not None:
        _check_broader(key, standard_name, broader, aliases)  # same: nothing stored
    existing = VOCABULARY.get(key)
    if existing is not None and existing["standard_name"] != standard_name:
        warnings.warn(
            f"register({key!r}, ...) replaces an existing entry pointing at "
            f"{existing['standard_name']!r} with {standard_name!r}; its current "
            "aliases, patterns and broader are discarded. Use "
            "add_alias()/add_pattern() to extend an entry instead.",
            stacklevel=2,
        )
    entry: dict[str, object] = {"standard_name": standard_name}
    if aliases:
        entry["aliases"] = list(aliases)
    if patterns:
        entry["patterns"] = list(patterns)
    if broader is not None:
        entry["broader"] = broader
    VOCABULARY[key] = entry
    _refresh()


def add_alias(key: str, *names: str) -> None:
    """Add one or more new spellings to an *existing* concept, live, no restart needed.

    ``key`` must already be in :data:`VOCABULARY` (use :func:`register` to add a
    new concept from scratch). Duplicates are ignored. The entry is extended in
    place, so everything else it says -- its patterns, its ``broader`` -- is left
    exactly as it was: aliasing a raw product name onto ``mld_by_sigma_theta`` keeps
    it one specific kind of ``mld``.

    >>> from ocean_skill import vocabulary
    >>> vocabulary.add_alias("chlorophyll", "chlor_a")
    >>> vocabulary.resolve_name("chlor_a")
    'mass_concentration_of_chlorophyll_a_in_sea_water'
    """
    if key not in VOCABULARY:
        raise KeyError(
            f"{key!r} is not a known vocabulary entry -- use register() to add a "
            "new concept, or check VOCABULARY for the existing key to extend.",
        )
    existing = VOCABULARY[key].setdefault("aliases", [])
    for name in names:
        if name not in existing:
            existing.append(name)  # type: ignore[union-attr]
    _refresh()


def add_pattern(key: str, *patterns: str) -> None:
    """Add regexes recognizing a *family* of spellings to an existing concept, live.

    ``key`` must already be in :data:`VOCABULARY` (use :func:`register` to add a
    new concept from scratch). Each pattern is matched fullmatch, case-insensitive
    (see the module docstring's "patterns" bullet for the narrowness this requires
    -- enumerated decorations, never an open-ended ``.*`` tail). Duplicates are
    ignored; an invalid regex raises :class:`re.error` before anything is stored.
    Like :func:`add_alias`, it extends the entry in place, leaving its ``broader``
    (and everything else) untouched.

    >>> from ocean_skill import vocabulary
    >>> vocabulary.add_pattern("oxygen", "oxy(?:_umolkg)?")
    >>> vocabulary.resolve_name("OXY_UMOLKG")
    'mole_concentration_of_dissolved_molecular_oxygen_in_sea_water'
    """
    if key not in VOCABULARY:
        raise KeyError(
            f"{key!r} is not a known vocabulary entry -- use register() to add a "
            "new concept, or check VOCABULARY for the existing key to extend.",
        )
    for pattern in patterns:
        re.compile(pattern)  # validate before touching VOCABULARY at all
    existing = VOCABULARY[key].setdefault("patterns", [])
    for pattern in patterns:
        if pattern not in existing:
            existing.append(pattern)  # type: ignore[union-attr]
    _refresh()


_refresh()
