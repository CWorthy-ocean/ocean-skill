"""The one editable vocabulary mapping a concept to its known real-world spellings.

Edit :data:`VOCABULARY` and nothing else needs to change: :func:`resolve_name` (used
everywhere a variable name is accepted, from :func:`ocean_skill.comparison.compare`
down to :func:`ocean_skill.units.find_variable`, :func:`ocean_skill.vars.lookup` and
:func:`ocean_skill.colormaps.cmaps_for`) and the `cf-xarray <https://cf-xarray.
readthedocs.io>`_ registration below are both derived from it automatically.

Each entry is one concept, keyed by a short mnemonic (the name to actually type — it
need not be a real CF name, it's just what's easy to read and call by):

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

Note this vocabulary's shape is ours, not cf-xarray's: cf-xarray's own
``custom_criteria`` only understands attribute names as keys (``"name"``,
``"standard_name"``, ``"units"``, ..., matched against each variable's own
attributes) — it has no native concept of an "alias". That grouping is this
module's; :func:`_register_custom_criteria` below flattens each entry into the one
``{"name": pattern}`` shape cf-xarray does understand.
"""

from __future__ import annotations

import html
import re
import warnings
from collections.abc import Iterable
from dataclasses import dataclass

from ocean_skill import _stacklevel

__all__ = [
    "VOCABULARY",
    "MatchReport",
    "add_alias",
    "add_pattern",
    "equivalent_names",
    "is_known",
    "match_report",
    "nickname",
    "register",
    "resolve_and_report",
    "resolve_name",
    "same_quantity",
]

VOCABULARY: dict[str, dict[str, object]] = {
    "nitrate": {
        "standard_name": "mole_concentration_of_nitrate_in_sea_water",
        "aliases": [
            "moles_of_nitrate_per_unit_mass_in_sea_water",
            "NO3",  # ROMS/MARBL tracer name; see the `Fe` note below on matching
        ],
    },
    "phosphate": {
        "standard_name": "mole_concentration_of_phosphate_in_sea_water",
        "aliases": [
            "moles_of_phosphate_per_unit_mass_in_sea_water",
            "PO4",  # ROMS/MARBL tracer name; see the `Fe` note below on matching
        ],
    },
    "silicate": {
        "standard_name": "mole_concentration_of_silicate_in_sea_water",
        "aliases": [
            "moles_of_silicate_per_unit_mass_in_sea_water",
            "SiO3",  # ROMS/MARBL tracer name; see the `Fe` note below on matching
        ],
    },
    "ammonium": {
        "standard_name": "mole_concentration_of_ammonium_in_sea_water",
        "aliases": [
            "moles_of_ammonium_per_unit_mass_in_sea_water",  # WOA/GLODAP per-mass form
            "NH4",  # ROMS/MARBL's own tracer name; see the `Fe` note below on matching
        ],
    },
    "iron": {
        "standard_name": "mole_concentration_of_dissolved_iron_in_sea_water",
        "aliases": [
            # WOA/GLODAP's per-mass spelling, the same per-volume/per-mass split the
            # other nutrients above carry.
            "moles_of_dissolved_iron_per_unit_mass_in_sea_water",
            # ROMS/MARBL's own tracer name (build.py's ROMS_STANDARD_NAMES maps `Fe`
            # here). Short, but matched whole -- the index is an exact lookup and the
            # cf-xarray pattern is anchored, so `Fe`/`fe`/`FE` resolve while a name
            # merely starting with them (e.g. `felix`) does not.
            "Fe",
        ],
    },
    "oxygen": {
        "standard_name": (
            "mole_concentration_of_dissolved_molecular_oxygen_in_sea_water"
        ),
        "aliases": [
            "moles_of_oxygen_per_unit_mass_in_sea_water",
            "O2",  # ROMS/MARBL tracer name; see the `Fe` note below on matching
        ],
        "patterns": [
            # Argo/OceanSITES's own code, and the common mooring spelling --
            # neither is a single literal worth enumerating case variants of (see
            # the module docstring's "patterns" bullet). Must not claim
            # "oxygen_saturation" above (a different quantity, not this pattern's
            # alternation) or a `_qc`/`_flag`-decorated companion (no such token
            # appears in it).
            "doxy",
            "dissolved_oxygen",
        ],
    },
    "oxygen_saturation": {
        # A different quantity from "oxygen" above -- percent saturation, not a
        # concentration -- so deliberately its own key/standard_name rather than an
        # alias, the same reason "sea_level_anomaly" stays separate from "ssh".
        "standard_name": "fractional_saturation_of_oxygen_in_sea_water",
    },
    "dissolved_inorganic_carbon": {
        "standard_name": (
            "mole_concentration_of_dissolved_inorganic_carbon_in_sea_water"
        ),
        "aliases": [
            "moles_of_dissolved_inorganic_carbon_per_unit_mass_in_sea_water",
            "DIC",  # ROMS/MARBL tracer name; see the `Fe` note below on matching
        ],
    },
    "alkalinity": {
        # All of these are *total* alkalinity: CF defines the canonical name as "the
        # total alkalinity equivalent concentration (including carbonate, nitrogen,
        # silicate, and borate components)". They differ only in basis (per volume vs
        # per mass), which units.py converts through the seawater-density context --
        # the same split nitrate has between WOA and ROMS/GLODAP.
        "standard_name": "sea_water_alkalinity_expressed_as_mole_equivalent",
        "aliases": [
            # CF's per-mass form (mol kg-1); the canonical name is mol m-3.
            "sea_water_alkalinity_per_unit_mass_expressed_as_mole_equivalent",
            # Kept for the "seawater" (no underscore) spelling seen in the wild --
            # it is NOT a CF name, so it matches only products that write it that way.
            "seawater_alkalinity_per_unit_mass_expressed_as_mole_equivalent",
            # OceanSODA-ETHZ's `talk` (ocean_skill/catalogs/oceansoda.yaml). Also not CF: the
            # table has no total_alkalinity_in_sea_water entry or alias. Without it,
            # find(variable="alkalinity") returned GLODAP and ROMS but silently
            # dropped OceanSODA, and compare() would not pair them.
            "total_alkalinity_in_sea_water",
            "ALK",  # ROMS/MARBL tracer name; see the `Fe` note below on matching
        ],
    },
    "temperature": {
        "standard_name": "sea_water_potential_temperature",
        # in-situ, not strictly the same quantity as potential temperature -- see
        # the module docstring on why that is an alias here and not its own tier
        "aliases": [
            "temp",  # ROMS tracer name; see the `Fe` note below on matching
            "sea_water_temperature",
            # Satellite SST, in the three flavours GHRSST distinguishes. They are
            # *not* identical: skin is the radiometric top micron, subskin the top
            # millimetre, and foundation the temperature free of diurnal warming --
            # skin and foundation can differ by a few tenths of a degree on a calm
            # sunny afternoon. They are aliased here for the same reason
            # sea_water_temperature is: without it every satellite SST product is
            # invisible to find(variable="temperature"), which was the whole point
            # of asking. MUR alone declares sea_surface_foundation_temperature.
            "sea_surface_foundation_temperature",
            # The plain surface spelling, used by every gridded surface product we
            # carry (OceanSODA-ETHZ's `temperature`, ocean_skill/catalogs/oceansoda.yaml, as well
            # as the CoastWatch L3 and Geo-Polar SST fields). The sampling depth is a
            # property of where the measurement was taken, which a comparison reports
            # separately (the metrics row's `obs_depth`, and align_series' depth
            # caveat) rather than by refusing to pair a mooring with a surface field.
            "sea_surface_temperature",
            "sea_surface_subskin_temperature",
            "sea_surface_skin_temperature",
        ],
        "patterns": [
            # SEANOE's CTD-export column style (`Temperature_CTD`, `CTD_Temperature`,
            # `temp_ctd`, any case) -- a family of decorations around one instrument
            # name, not worth enumerating literally. Must not claim
            # `temperature_qc`/`Temperature_flag` (no qc/flag token here), plain
            # `air_temperature` (a different quantity), `atemp`, or bare `ctd`.
            r"(?:sea_water_)?temp(?:erature)?_ctd",
            r"ctd_temp(?:erature)?",
        ],
    },
    "sea_ice": {
        # NSIDC's CDR (cdr_seaice_conc) and the ice field bundled into MUR, OISST
        # and the Geo-Polar blend all declare this one name.
        "standard_name": "sea_ice_area_fraction",
    },
    "sea_level_anomaly": {
        # Deliberately not an alias of "ssh": DUACS ships both, and they are
        # different quantities -- adt is height above the geoid, sla is the
        # departure from a mean surface. Aliasing them would let compare() pair a
        # ~1 m field against a ~0.1 m one and call it agreement.
        "standard_name": "sea_surface_height_above_sea_level",
    },
    "eastward_velocity": {
        "standard_name": "sea_water_x_velocity",
        "aliases": [
            "eastward_sea_water_velocity",
            # DUACS/MULTIOBS ugos and ugosa. The "_assuming_sea_level_for_geoid"
            # form is the one computed from sla rather than adt.
            "surface_geostrophic_eastward_sea_water_velocity",
            "surface_geostrophic_eastward_sea_water_velocity_assuming_sea_level_for_geoid",
        ],
    },
    "northward_velocity": {
        "standard_name": "sea_water_y_velocity",
        "aliases": [
            "northward_sea_water_velocity",
            "surface_geostrophic_northward_sea_water_velocity",
            "surface_geostrophic_northward_sea_water_velocity_assuming_sea_level_for_geoid",
        ],
    },
    "upward_velocity": {
        # ROMS' vertical velocity `w` (build.py's ROMS_STANDARD_NAMES). `w` is a
        # single letter but matched whole (see the `Fe` note), so only a variable
        # named exactly `w`/`W` resolves, not one merely starting with it. The
        # horizontal siblings above deliberately carry no `u`/`v` short name yet; add
        # them alongside this if the model-momentum triple is ever wanted as a set.
        "standard_name": "upward_sea_water_velocity",
        "aliases": ["w"],
    },
    "eastward_wind": {"standard_name": "eastward_wind"},
    "northward_wind": {"standard_name": "northward_wind"},
    "wind_speed": {"standard_name": "wind_speed"},
    "kd490": {
        # Verified from CoastWatch's VIIRS kd_490: "diffuse_", not the "volume_"
        # spelling the CF table also carries.
        "standard_name": (
            "diffuse_attenuation_coefficient_of_downwelling_radiative_flux_in_sea_water"
        ),
    },
    "salinity": {
        "standard_name": "sea_water_practical_salinity",
        "aliases": [
            "salt",  # ROMS tracer name; see the `Fe` note above on matching
            "sea_water_salinity",  # near-identical; see "temperature" above
            "sea_surface_salinity",  # the surface spelling, as for temperature
        ],
        "patterns": [
            # Argo/OceanSITES's `PSAL`, and the plain/CTD-export `sal`/`sal_psu`
            # family. Must not claim `psalm`, `salt_flux`, `sla` (sea_level_anomaly
            # is its own entry), `salinity_flag`, or `basalt` -- fullmatch refuses
            # all of them. Deliberately does not extend to `salt` itself; that
            # stays the exact literal alias above.
            r"p?sal(?:inity)?(?:_(?:psu|ctd))?",
        ],
    },
    "ssh": {
        "standard_name": "sea_surface_height_above_geoid",
    },
    "conductivity": {
        # OOI Papa's own name (ocean_skill/catalogs/ooi_papa.yaml) needs no alias here
        # -- it already spells the CF name in full.
        "standard_name": "sea_water_electrical_conductivity",
    },
    "pressure": {
        # Also ROMS/MARBL's own concept via build.py's ROMS_STANDARD_NAMES (there
        # mapped from `hbls`'s sibling depth-related tracers); tabular.py's
        # depth_of() treats this standard_name as the pressure-to-depth conversion
        # rung, so this key resolving correctly matters beyond just find(variable=).
        "standard_name": "sea_water_pressure",
    },
    "sigma_theta": {
        # ROMS' own diagnostic (ocean_skill.mld computes it the same way offline; see
        # mld.py's sigma0.name assignment) and the standard CTD-derived quantity a
        # mooring reports directly.
        "standard_name": "sea_water_sigma_theta",
    },
    "ph": {
        # SEANOE's SeapHOx mooring members report this after the QC recipe's build-
        # time rename drops the provider's own (wrong -- pH is unitless) units label
        # ``pH_qc[mL/L]`` down to the plain ``pH`` column this key/alias resolves.
        "standard_name": "sea_water_ph_reported_on_total_scale",
        "aliases": ["pH", "ph_total"],
    },
    "co2_flux": {
        "standard_name": "surface_downward_mole_flux_of_carbon_dioxide",
    },
    "chlorophyll": {
        "standard_name": "mass_concentration_of_chlorophyll_a_in_sea_water",
        "aliases": [
            # Common shorthand for the concept. Unlike NO3/O2/... above, `Chl` is NOT
            # a single model tracer -- ROMS/MARBL carry per-PFT spChl/diatChl/diazChl
            # summed via {"sum": [...]} (see ocean_skill.operators). Matched whole (the
            # `Fe` note), so `Chl`/`chl`/`CHL` resolve but spChl/diatChl/diazChl do not.
            "Chl",
            "mass_concentration_of_chlorophyll_in_sea_water",
            # OOI Papa's profiler-mounted fluorometer (ocean_skill/catalogs/ooi_papa.yaml) --
            # same quantity, a different instrument's naming convention.
            "mass_concentration_of_chlorophyll_a_in_sea_water_profiler_depth_enabled",
            # NOAA CoastWatch's ERDDAP griddap MODIS datasets (erdMH1chla1day and
            # relatives) drop the "mass_" prefix. Without this, the one entry that
            # carries the whole daily MODIS record is invisible to find(variable=...).
            "concentration_of_chlorophyll_in_sea_water",
        ],
        "patterns": [
            # The common short spellings. Must not claim the per-PFT ROMS/MARBL
            # tracers spChl/diatChl/diazChl (a different, un-summed quantity --
            # test_chl_shorthand_resolves_but_does_not_grab_per_pft_tracers pins
            # this) or a `_qc`-decorated companion. Deliberately excludes
            # `chlor_a`: that spelling is kept as the documented example of
            # extending the vocabulary live via add_alias (see its docstring and
            # examples/vocabulary_demo.py) rather than shipped recognized already.
            "chla",
            "chl_a",
            "chlorophyll_a",
        ],
    },
    "mld": {
        "standard_name": "ocean_mixed_layer_thickness",
        # Two ways to reach the same standard_name: ROMS' own KPP boundary-layer
        # depth (build.py's ROMS_STANDARD_NAMES maps `hbls` here directly) is the
        # model's own diagnostic, while {"calculate": "mld", "method": ...}
        # (ocean_skill.mld) computes it offline from T/S by a chosen criterion.
        # Both are legitimately "the model's MLD" and share the CF name on purpose
        # -- comparing them against each other is exactly the useful thing to do,
        # not a collision to avoid.
        "aliases": ["mixed_layer_depth", "mixed_layer_thickness"],
    },
}


def _all_names(entry: dict[str, object]) -> list[str]:
    """Every spelling one entry recognizes: its standard_name plus its aliases."""
    return [entry["standard_name"], *entry.get("aliases", [])]  # type: ignore[misc]


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
        for name in [key, *_all_names(entry)]:
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
    """
    return resolve_name(a).lower() == resolve_name(b).lower()


def nickname(name: str) -> str | None:
    """Return the short vocabulary key ``name`` resolves to, or ``None``.

    The reverse of typing a nickname: given a real-world spelling (an alias, a
    pattern-recognized name like ``"Temperature_CTD"``, or the canonical
    standard_name itself), return the short key a caller would have typed instead
    (``"temperature"``). ``None`` for anything :func:`is_known` also says no to --
    unrecognized or ambiguous (patterns from more than one entry).
    """
    return _KEY_BY_STANDARD_NAME.get(resolve_name(name))


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


def _register_custom_criteria() -> None:
    """Register every :data:`VOCABULARY` entry's spellings with cf-xarray.

    One ``{"name": pattern}`` entry per spelling (all of an entry's standard_name +
    aliases, registered under each so asking for any one finds the
    others), matched against the variable's own *name* only, deliberately not its
    ``standard_name`` attribute. Data is renamed to the standard_name directly (e.g.
    by :func:`ocean_skill.roms.standardize`, or the generic rename in
    :func:`ocean_skill.sources.read`) far more often than it carries that as a
    separate attribute, so name-matching already covers the real case — and
    attribute-matching actively misfires on WOA, whose auxiliary companion variables
    (``n_dd``/``n_se``/... — sample size, standard error) all carry the *same*
    ``standard_name`` as the actual data variable, which cf-xarray then (correctly,
    on that input) reports as ambiguous.

    Each pattern is ``(?i)(?:...)$``: case-insensitive (see :func:`_build_index`)
    and **anchored at both ends**. The trailing ``$`` is load-bearing — cf-xarray
    matches with :func:`re.match`, which anchors only the *start*, so an unanchored
    pattern also matches anything merely *prefixed* by a registered spelling. That
    silently returned an ERDDAP/OOI QC-flag companion
    (``..._in_sea_water_qc_agg``) as if it were the data variable whenever the real
    one was absent — found by testing, not hypothetical.

    An entry's ``patterns`` join the same alternation, each wrapped ``(?:...)`` so
    they inherit the anchors without escaping them. They are never registered as
    criteria *keys* the way literal spellings are — every dataset-side lookup
    (:func:`ocean_skill.units.find_variable`) arrives here already canonicalized by
    :func:`resolve_name`, and the canonical standard_name is always a literal key.
    """
    import cf_xarray

    criteria: dict[str, dict[str, str]] = {}
    for entry in VOCABULARY.values():
        names = _all_names(entry)
        parts = [re.escape(n) for n in names]
        parts += [f"(?:{p})" for p in entry.get("patterns", [])]  # type: ignore[union-attr]
        pattern = "(?i)(?:" + "|".join(parts) + ")$"
        for name in names:
            criteria[name] = {"name": pattern}
    cf_xarray.set_options(custom_criteria=criteria)


def _refresh() -> None:
    """Rebuild the resolver index and cf-xarray registration from :data:`VOCABULARY`.

    Called once at import time, and again by :func:`register`/:func:`add_alias`/
    :func:`add_pattern` after any live edit -- editing ``VOCABULARY`` directly (e.g.
    ``VOCABULARY["chlorophyll"]["aliases"].append(...)``) does *not* alone update
    :func:`resolve_name` or cf-xarray's own registration, since both are cached at
    module load; call this afterwards if you edit the dict by hand instead of
    through those functions.
    """
    global _INDEX, _BY_STANDARD_NAME, _KEY_BY_STANDARD_NAME, _PATTERNS
    _INDEX = _build_index()
    _BY_STANDARD_NAME = _build_by_standard_name()
    _KEY_BY_STANDARD_NAME = _build_key_by_standard_name()
    _PATTERNS = _build_patterns()
    _register_custom_criteria()


def register(
    key: str,
    standard_name: str,
    *,
    aliases: list[str] | None = None,
    patterns: list[str] | None = None,
) -> None:
    """Add (or replace) one vocabulary entry, live -- no restart needed.

    For a wholly new concept. To add a spelling or pattern to a concept that
    already has an entry, prefer :func:`add_alias`/:func:`add_pattern` instead, so
    you don't have to repeat its existing ``standard_name``/aliases just to add one
    more.

    >>> from ocean_skill import vocabulary
    >>> vocabulary.register(
    ...     "ph", "sea_water_ph_reported_on_total_scale", aliases=["ph_total"]
    ... )
    >>> vocabulary.resolve_name("ph")
    'sea_water_ph_reported_on_total_scale'
    """
    for pattern in patterns or []:
        re.compile(pattern)  # validate before touching VOCABULARY at all
    existing = VOCABULARY.get(key)
    if existing is not None and existing["standard_name"] != standard_name:
        warnings.warn(
            f"register({key!r}, ...) replaces an existing entry pointing at "
            f"{existing['standard_name']!r} with {standard_name!r}; its current "
            "aliases and patterns are discarded. Use add_alias()/add_pattern() to "
            "extend an entry instead.",
            stacklevel=2,
        )
    entry: dict[str, object] = {"standard_name": standard_name}
    if aliases:
        entry["aliases"] = list(aliases)
    if patterns:
        entry["patterns"] = list(patterns)
    VOCABULARY[key] = entry
    _refresh()


def add_alias(key: str, *names: str) -> None:
    """Add one or more new spellings to an *existing* concept, live, no restart needed.

    ``key`` must already be in :data:`VOCABULARY` (use :func:`register` to add a
    new concept from scratch). Duplicates are ignored.

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
