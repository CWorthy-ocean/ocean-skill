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

A caller may use the short key, the canonical ``standard_name``, or any alias
interchangeably, **in any capitalization** — :func:`resolve_name` maps all of them
to the same canonical ``standard_name``, and :func:`resolve_and_report` additionally
warns once, naming both, so it's never silently unclear which variable was actually
used. Matching against a *dataset's* own variable names ignores case too (see
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

import re
import warnings

from ocean_skill import _stacklevel

__all__ = [
    "VOCABULARY",
    "add_alias",
    "equivalent_names",
    "is_known",
    "register",
    "resolve_and_report",
    "resolve_name",
]

VOCABULARY: dict[str, dict[str, object]] = {
    "nitrate": {
        "standard_name": "mole_concentration_of_nitrate_in_sea_water",
        "aliases": ["moles_of_nitrate_per_unit_mass_in_sea_water"],
    },
    "phosphate": {
        "standard_name": "mole_concentration_of_phosphate_in_sea_water",
        "aliases": ["moles_of_phosphate_per_unit_mass_in_sea_water"],
    },
    "silicate": {
        "standard_name": "mole_concentration_of_silicate_in_sea_water",
        "aliases": ["moles_of_silicate_per_unit_mass_in_sea_water"],
    },
    "oxygen": {
        "standard_name": (
            "mole_concentration_of_dissolved_molecular_oxygen_in_sea_water"
        ),
        "aliases": ["moles_of_oxygen_per_unit_mass_in_sea_water"],
    },
    "dissolved_inorganic_carbon": {
        "standard_name": (
            "mole_concentration_of_dissolved_inorganic_carbon_in_sea_water"
        ),
        "aliases": ["moles_of_dissolved_inorganic_carbon_per_unit_mass_in_sea_water"],
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
            # OceanSODA-ETHZ's `talk` (catalogs/oceansoda.yaml). Also not CF: the
            # table has no total_alkalinity_in_sea_water entry or alias. Without it,
            # find(variable="alkalinity") returned GLODAP and ROMS but silently
            # dropped OceanSODA, and compare() would not pair them.
            "total_alkalinity_in_sea_water",
        ],
    },
    "temperature": {
        "standard_name": "sea_water_potential_temperature",
        # in-situ, not strictly the same quantity as potential temperature -- see
        # the module docstring on why that is an alias here and not its own tier
        "aliases": [
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
            # carry (OceanSODA-ETHZ's `temperature`, catalogs/oceansoda.yaml, as well
            # as the CoastWatch L3 and Geo-Polar SST fields). The sampling depth is a
            # property of where the measurement was taken, which a comparison reports
            # separately (the metrics row's `obs_depth`, and align_series' depth
            # caveat) rather than by refusing to pair a mooring with a surface field.
            "sea_surface_temperature",
            "sea_surface_subskin_temperature",
            "sea_surface_skin_temperature",
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
            "sea_water_salinity",  # near-identical; see "temperature" above
            "sea_surface_salinity",  # the surface spelling, as for temperature
        ],
    },
    "ssh": {
        "standard_name": "sea_surface_height_above_geoid",
    },
    "co2_flux": {
        "standard_name": "surface_downward_mole_flux_of_carbon_dioxide",
    },
    "chlorophyll": {
        "standard_name": "mass_concentration_of_chlorophyll_a_in_sea_water",
        "aliases": [
            "mass_concentration_of_chlorophyll_in_sea_water",
            # OOI Papa's profiler-mounted fluorometer (catalogs/ooi_papa.yaml) --
            # same quantity, a different instrument's naming convention.
            "mass_concentration_of_chlorophyll_a_in_sea_water_profiler_depth_enabled",
            # NOAA CoastWatch's ERDDAP griddap MODIS datasets (erdMH1chla1day and
            # relatives) drop the "mass_" prefix. Without this, the one entry that
            # carries the whole daily MODIS record is invisible to find(variable=...).
            "concentration_of_chlorophyll_in_sea_water",
        ],
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


_INDEX: dict[str, str] = {}
_BY_STANDARD_NAME: dict[str, dict[str, object]] = {}


def resolve_name(name: str) -> str:
    """Return the canonical CF standard_name for any spelling in :data:`VOCABULARY`.

    Accepts a short key (``"oxygen"``), the canonical standard_name itself, or any
    alias, **in any capitalization** (``"Oxygen"``, ``"OXYGEN"``)
    — all resolve to the same standard_name. A name with no vocabulary entry at all
    passes through unchanged (keeping its original case), so this is safe to call on
    any variable name, curated or not.
    """
    return _INDEX.get(name.lower(), name)


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
    """Whether the vocabulary recognizes ``name`` at all (key, standard_name, alias).

    Lets a caller tell "this source does not have X" from "I cannot tell": catalog
    metadata indexes CF standard_names, so a name the vocabulary has never heard of
    (a raw model variable like ``spChl``) is unknowable from metadata alone and must
    not be treated as absent.
    """
    return name.lower() in _INDEX


def equivalent_names(name: str) -> set[str]:
    """Return every spelling this vocabulary treats as the same variable as ``name``.

    The canonical standard_name plus all its aliases —
    the set that counts as "the same variable" when matching a requested variable
    against a source's declared ``variables`` (see
    :func:`ocean_skill.comparison.compare`). A name with no vocabulary entry
    returns just itself.
    """
    entry = _BY_STANDARD_NAME.get(resolve_name(name))
    return {name} if entry is None else set(_all_names(entry))


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
    """
    import cf_xarray

    criteria: dict[str, dict[str, str]] = {}
    for entry in VOCABULARY.values():
        names = _all_names(entry)
        pattern = "(?i)(?:" + "|".join(re.escape(n) for n in names) + ")$"
        for name in names:
            criteria[name] = {"name": pattern}
    cf_xarray.set_options(custom_criteria=criteria)


def _refresh() -> None:
    """Rebuild the resolver index and cf-xarray registration from :data:`VOCABULARY`.

    Called once at import time, and again by :func:`register`/:func:`add_alias`
    after any live edit -- editing ``VOCABULARY`` directly (e.g.
    ``VOCABULARY["chlorophyll"]["aliases"].append(...)``) does *not* alone update
    :func:`resolve_name` or cf-xarray's own registration, since both are cached at
    module load; call this afterwards if you edit the dict by hand instead of
    through those two functions.
    """
    global _INDEX, _BY_STANDARD_NAME
    _INDEX = _build_index()
    _BY_STANDARD_NAME = _build_by_standard_name()
    _register_custom_criteria()


def register(
    key: str,
    standard_name: str,
    *,
    aliases: list[str] | None = None,
) -> None:
    """Add (or replace) one vocabulary entry, live -- no restart needed.

    For a wholly new concept. To add a spelling to a concept that already has an
    entry, prefer :func:`add_alias` instead, so you don't have to repeat its
    existing ``standard_name``/aliases just to add one more.

    >>> from ocean_skill import vocabulary
    >>> vocabulary.register(
    ...     "ph", "sea_water_ph_reported_on_total_scale", aliases=["ph_total"]
    ... )
    >>> vocabulary.resolve_name("ph")
    'sea_water_ph_reported_on_total_scale'
    """
    existing = VOCABULARY.get(key)
    if existing is not None and existing["standard_name"] != standard_name:
        warnings.warn(
            f"register({key!r}, ...) replaces an existing entry pointing at "
            f"{existing['standard_name']!r} with {standard_name!r}; its current "
            "aliases are discarded. Use add_alias() to extend an entry instead.",
            stacklevel=2,
        )
    entry: dict[str, object] = {"standard_name": standard_name}
    if aliases:
        entry["aliases"] = list(aliases)
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
            "new concept, or check VOCABULARY for the existing key to extend."
        )
    existing = VOCABULARY[key].setdefault("aliases", [])
    for name in names:
        if name not in existing:
            existing.append(name)  # type: ignore[union-attr]
    _refresh()


_refresh()
