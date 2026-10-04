"""Vertical conventions for observed depths: where zero is, and which way is down.

A model's vertical coordinate is a *height*: ROMS builds
``z_rho = zeta + (zeta + h) * s``, metres relative to mean sea level (positive up), so
every layer rides up and down with the free surface. An observation's "depth" is not
one thing. A CTD cast, a pressure record or a float's sensor measures **below the
instantaneous free surface** -- a 1 m sample is 1 m under wherever the surface is at
that moment -- while a pier sonde or a bottom-mounted ADCP sits at a position **fixed
in space**; and a source may store the value as a height (positive up) or as pressure
in decibars. Matching all of them to the model with one rule (target ``z = -depth``)
puts a tide-sized error into every surface-referenced sample: in a macrotidal estuary
(tide +/-3 m) a 1 m CTD sample is compared up to 3 m too deep at high water, and falls
above the model's free surface -- NaN -- at low water.

This module is the vocabulary for that distinction, and nothing more. A source's
vertical values ``v`` are turned into **``d``: metres, positive down, below an origin**,
described by five fields:

* ``origin`` -- ``surface`` (``d`` is measured below the moving free surface; the model
  target is ``z = zeta - d``) or ``fixed`` (a position fixed in space; the model target
  is ``z = datum_z_m - d``);
* ``datum_z_m`` -- for ``fixed``, the height of the obs datum in the model's z frame
  (model mean sea level is 0; default 0);
* ``positive`` -- ``up`` or ``down``, the sign of the raw values (``up`` means
  ``d = -v``);
* ``units`` -- ``m`` or ``dbar`` (``d = v * M_PER_DBAR``, the same 1 dbar ~ 1 m
  approximation :mod:`ocean_skill.tabular` already uses -- approximate, and flagged as
  such wherever it is applied);
* ``support`` -- ``point`` (match at the depth), ``surface`` (the model's top cell) or
  ``bottom`` (the model's bottom cell).

Where the five values come from is as important as what they are, because a person's
word must outlive a re-probe. They are kept in separate tiers and merged only at read
time (:func:`resolve`), most specific first: a **per-variable** entry, the entry's own
**declared** fields, what the build-time probe **inferred** (kept apart under an
``inferred`` block so rebuilding a catalog never overwrites something a person wrote),
a **data** hint derived at read time from the vertical coordinate's own name and CF
attributes (:func:`infer_from_coordinate`), and finally the **default**. Every resolved
field carries the tier it came from, so a warning can say whether a convention was
declared or merely assumed.

Pure metadata, deliberately: this imports only the standard library, numpy and pydantic
-- nothing from the rest of the package -- so that it can move into a catalog package
unchanged. The model-side matching that consumes a resolved convention lives with the
ROMS code; this module never touches a dataset.
"""

from __future__ import annotations

import difflib
import math
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any, NoReturn, TypeVar

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

__all__ = [
    "FIELDS",
    "M_PER_DBAR",
    "NORMALIZED_ATTR",
    "ORIGINS",
    "POSITIVES",
    "SUPPORTS",
    "SURFACE_DEFAULT_FEATURE_TYPES",
    "UNITS",
    "ResolvedConvention",
    "canonicalize",
    "coordinate_positive",
    "infer_from_coordinate",
    "merge_probed",
    "positive_down_values",
    "resolve",
    "to_positive_down",
]

#: Where ``d = 0`` is. ``surface``: the moving free surface; ``fixed``: a position fixed
#: in space (see ``datum_z_m``). A tuple rather than a pair of constants so a further
#: origin (the seafloor, a layer interface) is one more entry here plus its spellings in
#: :data:`_ALIASES` -- nothing below assumes there are exactly two, beyond the explicit
#: default/pressure/datum rules named next to :data:`_DEFAULT_ORIGIN`.
ORIGINS = ("surface", "fixed")

#: Sign of the raw values: ``up`` is a height above the origin, ``down`` a depth below.
POSITIVES = ("up", "down")

#: Units of the raw values (``dbar`` is converted at :data:`M_PER_DBAR`).
UNITS = ("m", "dbar")

#: What a requested depth selects on the model side: the cell at that depth (``point``),
#: the top cell (``surface``) or the bottom cell (``bottom``).
SUPPORTS = ("point", "surface", "bottom")

#: The five fields of a convention, in the order a canonical dict lists them.
FIELDS = ("origin", "positive", "units", "datum_z_m", "support")

#: Metres per decibar, near enough for matching: the exact factor varies with latitude
#: and water column (``gsw.z_from_p``), and gsw is not a dependency here. Same
#: approximation as ``tabular._M_PER_DBAR``; anything converted with it is approximate,
#: not measured.
M_PER_DBAR = 1.0

#: Coordinate attribute stamped on a vertical coordinate whose values are *already*
#: positive-down metres (the read path normalises tabular depths once, at the door), so
#: :func:`positive_down_values` knows not to convert them a second time.
NORMALIZED_ATTR = "depth_normalized"

#: ``featureType`` values whose depth is, absent a declaration, measured below the free
#: surface (casts and profilers) rather than fixed in space.
SURFACE_DEFAULT_FEATURE_TYPES = frozenset({"profile", "trajectoryProfile"})

#: Spellings accepted for each field's values, case-insensitively (``-`` and spaces read
#: as ``_``), mapped to the canonical value. Every canonical value in :data:`ORIGINS`,
#: :data:`POSITIVES`, :data:`UNITS` and :data:`SUPPORTS` is its own key.
_ALIASES: dict[str, dict[str, str]] = {
    "origin": {
        "surface": "surface",
        "free_surface": "surface",
        "sea_surface": "surface",
        "fixed": "fixed",
        "fixed_in_space": "fixed",
    },
    "positive": {"up": "up", "upward": "up", "down": "down", "downward": "down"},
    "units": {
        "m": "m",
        "meter": "m",
        "meters": "m",
        "metre": "m",
        "metres": "m",
        "dbar": "dbar",
        "decibar": "dbar",
        "decibars": "dbar",
        "db": "dbar",
    },
    "support": {"point": "point", "surface": "surface", "bottom": "bottom"},
}

#: Each enumerated field's canonical values, in the order error messages list them.
_CANONICAL: dict[str, tuple[str, ...]] = {
    "origin": ORIGINS,
    "positive": POSITIVES,
    "units": UNITS,
    "support": SUPPORTS,
}

#: Origins that take a ``datum_z_m``: those defined by a position in the model's own z
#: frame. A datum on any other origin has nothing to be the height *of*.
_DATUM_ORIGINS = ("fixed",)

#: Origin when nothing says otherwise: today's behaviour, a sensor at a fixed depth.
_DEFAULT_ORIGIN = "fixed"

#: Origin a pressure-derived depth implies: pressure is measured below the free surface.
_PRESSURE_ORIGIN = "surface"

#: Defaults for the fields that have one independent of the others (``origin``'s default
#: depends on ``featureType`` and ``units``, so it is decided in :func:`resolve`).
_DEFAULTS: dict[str, Any] = {
    "positive": "down",
    "units": "m",
    "datum_z_m": 0.0,
    "support": "point",
}

#: CF standard names that settle a vertical coordinate's convention on their own.
#: ``height``/``altitude`` fix only the sign (they say nothing about the reference);
#: the ``*_above_*``/``*_below_geoid`` names state a reference, hence ``fixed``.
_STANDARD_NAMES: dict[str, dict[str, str]] = {
    "sea_water_pressure": {"origin": "surface", "units": "dbar"},
    "sea_water_pressure_due_to_sea_water": {"origin": "surface", "units": "dbar"},
    "depth": {"positive": "down"},
    "height": {"positive": "up"},
    "altitude": {"positive": "up"},
    "height_above_mean_sea_level": {"origin": "fixed", "positive": "up"},
    "height_above_geoid": {"origin": "fixed", "positive": "up"},
    "height_above_reference_ellipsoid": {"origin": "fixed", "positive": "up"},
    "depth_below_geoid": {"origin": "fixed", "positive": "down"},
}

#: Column/variable names (units suffix stripped, case-folded) that mean pressure.
#: ``prdm`` is Sea-Bird's primary pressure, ``prs`` a common short form.
_PRESSURE_NAMES = frozenset({"pres", "pressure", "sea_water_pressure", "prdm", "prs"})

#: ``"<name> (<units>)"`` and ``"<name>[<units>]"`` -- the two spellings tabular sources
#: pack units into a column name with (the same pair ``tabular.split_units`` reads, kept
#: here so this module needs nothing from it).
_NAME_PAREN_UNITS = re.compile(r"^(.+?)\s*\((.+)\)$")
_NAME_BRACKET_UNITS = re.compile(r"^(.+?)\s*\[(.+)\]$")


# -- value spellings -------------------------------------------------------------------


def _spelling(value: Any) -> str:
    """Case-folded spelling with runs of whitespace/hyphens read as one underscore."""
    return re.sub(r"[\s-]+", "_", str(value).strip().casefold())


def _canonical_value(field: str, value: Any) -> str | None:
    """Return the canonical ``field`` value ``value`` spells, or ``None``."""
    if not isinstance(value, str):
        return None
    return _ALIASES[field].get(_spelling(value))


def _bad_value(field: str, value: Any) -> str:
    """Explain why ``value`` is not a ``field`` value: what is allowed, what is near."""
    allowed = []
    for canonical in _CANONICAL[field]:
        others = [
            a for a, c in _ALIASES[field].items() if c == canonical and a != canonical
        ]
        allowed.append(
            repr(canonical) + (f" (or {', '.join(others)})" if others else "")
        )
    guess = ""
    if isinstance(value, str):
        near = difflib.get_close_matches(_spelling(value), list(_ALIASES[field]), n=1)
        if near:
            guess = f"; did you mean {_ALIASES[field][near[0]]!r}?"
    return f"{value!r} is not one of {', '.join(allowed)}{guess}"


def _checked(field: str, value: Any) -> str:
    """Return :func:`_canonical_value`, or raise the module's ``ValueError``."""
    canonical = _canonical_value(field, value)
    if canonical is None:
        raise ValueError(f"depth_convention: {field}: {_bad_value(field, value)}")
    return canonical


def _finite_number(value: Any) -> float:
    """``value`` as a finite float (a number or numeric string; never a bool)."""
    if isinstance(value, bool):
        raise ValueError(f"{value!r} is not a finite number")
    try:
        number = float(value) if isinstance(value, (Real, str)) else math.nan
    except ValueError:
        number = math.nan
    if not math.isfinite(number):
        raise ValueError(
            f"{value!r} is not a finite number (metres, in the model's z frame)"
        )
    return number


# -- the validated shape ---------------------------------------------------------------


class _Entry(BaseModel):
    """One tier's five fields: a top-level block or a per-variable entry."""

    model_config = ConfigDict(extra="forbid")

    origin: str | None = None
    positive: str | None = None
    units: str | None = None
    datum_z_m: float | None = None
    support: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _known_keys(cls, data: Any) -> Any:
        # Checked here, ahead of pydantic's own extra="forbid", so the message can name
        # the keys that *are* allowed and offer the near miss (``orgin`` -> ``origin``).
        allowed = tuple(cls.model_fields)
        if not isinstance(data, Mapping):
            raise ValueError(
                f"expected a mapping with keys from {', '.join(allowed)}, "
                f"got {type(data).__name__}"
            )
        unknown = [k for k in data if k not in allowed]
        if unknown:
            notes = []
            for key in unknown:
                near = difflib.get_close_matches(str(key), allowed, n=1)
                notes.append(
                    repr(key) + (f" (did you mean {near[0]!r}?)" if near else "")
                )
            raise ValueError(
                f"unknown key {', '.join(notes)}; allowed keys: {', '.join(allowed)}"
            )
        return data

    @field_validator("origin", "positive", "units", "support", mode="before")
    @classmethod
    def _spelled(cls, value: Any, info: ValidationInfo) -> Any:
        if value is None:
            return None
        canonical = _canonical_value(info.field_name, value)
        if canonical is None:
            raise ValueError(_bad_value(info.field_name, value))
        return canonical

    @field_validator("datum_z_m", mode="before")
    @classmethod
    def _datum(cls, value: Any) -> Any:
        return None if value is None else _finite_number(value)


class _Inferred(_Entry):
    """The probe's block: the same fields, plus the ``reason`` it concluded them for."""

    reason: str | None = None

    @field_validator("reason", mode="before")
    @classmethod
    def _text(cls, value: Any) -> Any:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(f"{value!r} is not text")
        return value.strip() or None


class _Declared(_Entry):
    """A whole entry: top-level fields, per-variable overrides, ``inferred`` block."""

    variables: dict[str, _Entry] | None = None
    inferred: _Inferred | None = None

    @field_validator("variables", mode="before")
    @classmethod
    def _variable_entries(cls, value: Any) -> Any:
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise ValueError(
                "expected a mapping of variable name -> convention, "
                f"got {type(value).__name__}"
            )
        entries: dict[Any, Any] = {}
        for name, entry in value.items():
            if entry is None or (isinstance(entry, str) and not entry.strip()):
                entry = {}
            elif isinstance(entry, str):
                entry = {"origin": entry}  # the same shorthand the top level takes
            entries[name] = entry
        return entries

    @model_validator(mode="after")
    def _datum_needs_a_fixed_origin(self) -> _Declared:
        # A datum is the height of the obs datum in the model's z frame; on a surface
        # origin there is no such thing (the origin *is* the moving surface). The
        # origin that counts is the entry's own, else whatever it inherits -- a
        # variable entry from the top level, the top level from what the probe inferred.
        # A datum of exactly zero says nothing a surface origin does not already, so it
        # passes anywhere (which lets a resolved ``key()`` be canonicalised again).
        probed = self.inferred.origin if self.inferred is not None else None

        def check(
            where: str, datum: float | None, chain: tuple[str | None, ...]
        ) -> None:
            if datum is None or datum == 0:
                return
            origin = next((o for o in chain if o is not None), None)
            if origin not in _DATUM_ORIGINS:
                shown = "not set" if origin is None else repr(origin)
                raise ValueError(
                    f"{where}datum_z_m only applies to origin: fixed (the origin here "
                    f"is {shown}) -- declare origin: fixed beside it"
                )

        check("", self.datum_z_m, (self.origin, probed))
        for name, entry in (self.variables or {}).items():
            check(
                f"variables[{name!r}]: ",
                entry.datum_z_m,
                (entry.origin, self.origin, probed),
            )
        if self.inferred is not None:
            check(
                "inferred: ",
                self.inferred.datum_z_m,
                (self.inferred.origin, self.origin),
            )
        return self


_Block = TypeVar("_Block", bound=_Entry)


def _as_mapping(declared: Any) -> dict[str, Any]:
    """Return a declaration as a fresh dict (``"surface"`` -> ``{"origin": ...}``)."""
    if declared is None:
        return {}
    if isinstance(declared, str):
        return {"origin": declared} if declared.strip() else {}
    if isinstance(declared, Mapping):
        return dict(declared)
    raise ValueError(
        f"depth_convention: expected a mapping of {', '.join(FIELDS)} (plus variables, "
        f"inferred) or a shorthand origin such as 'surface' or 'fixed', got "
        f"{type(declared).__name__}"
    )


def _location(loc: tuple[Any, ...]) -> str:
    """Format a pydantic error location as a path: ``variables['temp'].origin``."""
    path = ""
    for i, part in enumerate(loc):
        if i == 0:
            path = str(part)
        elif loc[i - 1] == "variables":  # the one dict-valued field: its keys are names
            path += f"[{part!r}]"
        else:
            path += f".{part}"
    return path


def _raise_value_error(exc: ValidationError) -> NoReturn:
    """Re-raise a pydantic ``ValidationError`` as the module's own ``ValueError``.

    Callers hand this module catalog metadata and user ``select=`` entries; neither
    should ever meet a raw pydantic traceback. Each problem keeps where it was (the path
    into the entry) and what is allowed, and the validators' own wording is passed
    through verbatim rather than pydantic's ``Value error, ...`` wrapping of it.
    """
    problems = []
    for err in exc.errors(include_url=False):
        inner = (err.get("ctx") or {}).get("error")
        message = str(inner) if isinstance(inner, Exception) else err["msg"]
        where = _location(err["loc"])
        problems.append(f"{where}: {message}" if where else message)
    raise ValueError("depth_convention: " + "; ".join(problems)) from None


def _entry_fields(entry: _Entry) -> dict[str, Any]:
    """Return an entry's set fields as plain values, in :data:`FIELDS` order."""
    return {f: getattr(entry, f) for f in FIELDS if getattr(entry, f) is not None}


def _validate_block(data: Mapping[str, Any], model: type[_Block]) -> _Block:
    """Validate ``data`` as ``model``; a pydantic failure becomes a ``ValueError``."""
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        _raise_value_error(exc)


# -- canonical form --------------------------------------------------------------------


def canonicalize(
    declared: Any, *, inferred: Mapping[str, Any] | None = None
) -> dict | None:
    """Validate a ``depth_convention`` declaration and return it in canonical form.

    ``declared`` is what a catalog author writes: ``None``; a shorthand origin
    (``"surface"``/``"fixed"`` -> ``{"origin": ...}``); or a mapping of any of
    :data:`FIELDS`, optionally with ``variables`` (``{name: mapping-or-shorthand}``,
    per-variable overrides) and ``inferred`` (the probe's tier, which an already-
    canonical entry carries back in). ``inferred=`` supplies that tier explicitly and
    wins over one inside ``declared``; it takes a subset of :data:`FIELDS` plus an
    optional ``reason`` (why the probe concluded it).

    Values are normalised case-insensitively (``Surface``, ``free_surface`` and
    ``sea_surface`` are ``surface``; ``metres``/``meter`` are ``m``; ``decibar``/``db``
    are ``dbar``; ``upward``/``downward`` are ``up``/``down``). ``datum_z_m`` must be a
    finite number, and a non-zero one only means something on an ``origin: fixed`` --
    the entry's own origin, else the one it inherits (a variable entry from the top
    level, the top level from the ``inferred`` tier). Unknown keys, unknown values and a
    non-zero datum on any other origin are all reported as ``ValueError`` with a
    message starting ``"depth_convention: "`` that names the offending key or value and
    what is allowed -- never a raw pydantic error.

    Returns a plain, JSON-able dict (set fields in :data:`FIELDS` order, then
    ``variables`` if any, then ``inferred`` if any) or ``None`` when nothing is
    declared, so ``canonicalize(canonicalize(x)) == canonicalize(x)``.
    """
    data = _as_mapping(declared)
    if inferred is not None:
        data["inferred"] = inferred
    model = _validate_block(data, _Declared)

    out = _entry_fields(model)
    variables = {
        name: fields
        for name, entry in (model.variables or {}).items()
        if (fields := _entry_fields(entry))
    }
    if variables:
        out["variables"] = variables
    if model.inferred is not None:
        block = _entry_fields(model.inferred)
        if model.inferred.reason:
            block["reason"] = model.inferred.reason
        if block:
            out["inferred"] = block
    return out or None


def merge_probed(probed: Mapping[str, Any] | None, declared: Any) -> dict | None:
    """Combine a probe's finding with what the caller declared, for a catalog entry.

    ``probed`` is the probe's own ``depth_convention`` entry -- ``{"inferred": {...}}``
    -- or ``None`` when it found nothing; ``declared`` is whatever the caller passed
    (a shorthand, a mapping, or an already-canonical entry from an earlier build).
    Declared values stay at the top level, where :func:`resolve` ranks them above
    anything inferred; the probe's finding is stored under ``inferred``, replacing any
    stale one an earlier build left inside ``declared``. Only ``probed["inferred"]`` is
    read from the probe's entry: the probe never speaks for the author.

    Returns the canonical entry (see :func:`canonicalize`), or ``None`` when both are
    empty.
    """
    found = None
    if probed is not None:
        found = canonicalize(probed)
        found = None if found is None else found.get("inferred")
    base = _as_mapping(declared)
    earlier = base.pop("inferred", None)
    return canonicalize(base, inferred=found if found is not None else earlier)


# -- resolution ------------------------------------------------------------------------


class _Provenance(Mapping):
    """An immutable, hashable mapping of field -> tier that pickles and deep-copies.

    A plain dict could be edited behind a frozen dataclass's back, and
    :class:`types.MappingProxyType` can be neither hashed nor pickled -- so a
    :class:`ResolvedConvention` holding one could not be deep-copied along with the
    comparison that carries it.
    """

    __slots__ = ("_pairs",)

    def __init__(self, pairs: Mapping[str, str]) -> None:
        self._pairs = tuple(pairs.items())

    def __getitem__(self, key: str) -> str:
        for name, tier in self._pairs:
            if name == key:
                return tier
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (name for name, _ in self._pairs)

    def __len__(self) -> int:
        return len(self._pairs)

    def __hash__(self) -> int:
        return hash(self._pairs)

    def __repr__(self) -> str:
        return repr(dict(self._pairs))


@dataclass(frozen=True)
class ResolvedConvention:
    """One source's convention, with where each field came from.

    ``provenance`` maps every field to ``"declared"`` (the entry's own metadata, top
    level or a per-variable override), ``"inferred"`` (the build-time probe), ``"data"``
    (read from the data's own coordinate at read time) or ``"default"``. It is stored
    read-only, and the whole object is hashable and picklable.
    """

    origin: str
    positive: str
    units: str
    datum_z_m: float
    support: str
    provenance: Mapping[str, str]

    def __post_init__(self) -> None:
        # Normalised through object.__setattr__ because the class is frozen: a plain
        # float, so ``key()`` serialises the same whether the caller wrote 0 or 0.0, and
        # a read-only copy of the provenance, complete for every field.
        object.__setattr__(self, "datum_z_m", float(self.datum_z_m))
        given = dict(self.provenance)
        object.__setattr__(
            self,
            "provenance",
            _Provenance({f: given.get(f, "default") for f in FIELDS}),
        )

    def __hash__(self) -> int:
        return hash((*(getattr(self, f) for f in FIELDS), self.provenance))

    def key(self) -> dict[str, Any]:
        """Return the five fields as a JSON-able dict (for a cache key or identity)."""
        return {f: getattr(self, f) for f in FIELDS}

    def frame(self) -> dict[str, Any]:
        """Return what the model side needs to build the matching frame.

        ``source`` is where the *origin* came from (``declared``/``inferred``/``data``/
        ``default``): the one field whose guess can silently put an obs on the wrong
        side of the tide, so it decides whether a mismatch is worth a warning.
        """
        return {
            "origin": self.origin,
            "datum_z_m": self.datum_z_m,
            "support": self.support,
            "source": self.provenance["origin"],
        }

    @property
    def declared(self) -> bool:
        """Whether the origin was stated by the metadata (not inferred or assumed)."""
        return self.provenance["origin"] == "declared"


def _variable_entry(
    variables: Mapping[str, Any] | None, name: Any
) -> Mapping[str, Any]:
    """Return the per-variable entry for ``name``: exact key, then case-insensitive."""
    if not variables or name is None:
        return {}
    if name in variables:
        return variables[name]
    folded = str(name).casefold()
    for key, entry in variables.items():
        if key.casefold() == folded:
            return entry
    return {}


def _hint_fields(hint: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate a data-derived hint (:func:`infer_from_coordinate`'s output)."""
    if not hint:
        return {}
    block = _validate_block(_as_mapping(hint), _Inferred)
    return _entry_fields(block)


def resolve(
    meta: Mapping[str, Any] | None,
    variable: str | None = None,
    *,
    hint: Mapping[str, Any] | None = None,
) -> ResolvedConvention:
    """Resolve a source's depth convention, field by field, with provenance.

    Reads ``meta["depth_convention"]`` (canonical or raw -- it is always validated, so
    an invalid declaration raises ``ValueError`` here, at read time, rather than being
    trusted because it came from a catalog). For each field the first tier that sets it
    wins: the entry for ``variable`` (matched exactly, then case-insensitively; ``None``
    skips this tier), the entry's own declared fields, the probe's ``inferred`` block,
    ``hint`` (derived from the data's coordinate at read time; provenance ``"data"``),
    then the default.

    Defaults: ``positive="down"``, ``units="m"``, ``datum_z_m=0.0``,
    ``support="point"``, and an ``origin`` of ``surface`` when ``meta["featureType"]``
    is a profile type (:data:`SURFACE_DEFAULT_FEATURE_TYPES`, case-insensitive) or the
    resolved units are ``dbar`` -- measured from a cast or a pressure sensor, so below
    the free surface -- else ``fixed``, which is what every source got before
    conventions existed. A surface origin has no datum: ``datum_z_m`` is ``0.0``
    whatever other tiers say (a variable that overrides a fixed entry to ``surface``
    simply does not inherit its datum). ``meta=None`` gives all defaults.
    """
    meta = {} if meta is None else meta
    declared = canonicalize(meta.get("depth_convention")) or {}

    tiers: list[tuple[str, Mapping[str, Any]]] = []
    entry = _variable_entry(declared.get("variables"), variable)
    if entry:
        tiers.append(("declared", entry))
    top = {f: declared[f] for f in FIELDS if f in declared}
    if top:
        tiers.append(("declared", top))
    if declared.get("inferred"):
        tiers.append(("inferred", declared["inferred"]))
    data = _hint_fields(hint)
    if data:
        tiers.append(("data", data))

    values: dict[str, Any] = {}
    provenance: dict[str, str] = {}
    for field in FIELDS:
        values[field], provenance[field] = None, "default"
        for source, fields in tiers:
            if fields.get(field) is not None:
                values[field], provenance[field] = fields[field], source
                break

    for field, default in _DEFAULTS.items():
        if values[field] is None:
            values[field] = default
    if values["origin"] is None:
        feature_type = meta.get("featureType")
        profile_like = isinstance(feature_type, str) and feature_type.casefold() in {
            t.casefold() for t in SURFACE_DEFAULT_FEATURE_TYPES
        }
        pressure_like = values["units"] == "dbar"
        values["origin"] = (
            _PRESSURE_ORIGIN if profile_like or pressure_like else _DEFAULT_ORIGIN
        )
    if values["origin"] not in _DATUM_ORIGINS:
        values["datum_z_m"], provenance["datum_z_m"] = 0.0, "default"

    return ResolvedConvention(provenance=provenance, **values)


# -- inferring from a coordinate -------------------------------------------------------


def _split_name_units(name: Any) -> tuple[str, str | None]:
    """``"Pressure (dbar)"`` -> ``("Pressure", "dbar")``; a bare name has no units."""
    text = str(name).strip()
    match = _NAME_PAREN_UNITS.match(text) or _NAME_BRACKET_UNITS.match(text)
    return (match.group(1), match.group(2)) if match else (text, None)


def infer_from_coordinate(
    name: str | None = None, attrs: Mapping[str, Any] | None = None
) -> dict[str, str]:
    """Infer convention fields from a vertical coordinate's own name and CF attributes.

    Returns the subset of :data:`FIELDS` that can be concluded -- ``{}`` when nothing
    can. Explicit attributes outrank name heuristics, and each field takes the first
    clue that offers it, in this order:

    1. ``attrs["positive"]`` -> ``positive``; ``attrs["units"]`` (else a ``(...)`` or
       ``[...]`` suffix on ``name``) -> ``units``, where a decibar spelling also means a
       pressure and so ``origin="surface"``.
    2. ``attrs["standard_name"]``: ``sea_water_pressure`` (and ``..._due_to_sea_water``)
       -> dbar below the surface; ``depth`` -> positive down; ``height``/``altitude`` ->
       positive up; ``height_above_mean_sea_level``/``height_above_geoid``/
       ``height_above_reference_ellipsoid`` -> fixed, positive up; ``depth_below_geoid``
       -> fixed, positive down.
    3. A pressure-like ``name`` (``pres``, ``pressure``, ``sea_water_pressure``,
       ``prdm``, ``prs``, ignoring case and a units suffix) -> ``origin="surface"``, and
       dbar unless the units say metres.

    Used twice: by the build-time probe, whose result lands in the catalog's
    ``inferred`` tier, and at read time as :func:`resolve`'s ``hint``.
    """
    attrs = attrs or {}
    base, suffix_units = _split_name_units(name) if name is not None else ("", None)
    found: dict[str, str] = {}

    def offer(field: str, value: str) -> None:
        found.setdefault(field, value)

    positive = _canonical_value("positive", attrs.get("positive"))
    if positive:
        offer("positive", positive)

    units_text = attrs.get("units")
    if units_text is None or str(units_text).strip() == "":
        units_text = suffix_units
    units = _canonical_value("units", units_text)
    if units == "dbar":
        offer("units", "dbar")
        offer("origin", _PRESSURE_ORIGIN)
    elif units == "m":
        offer("units", "m")

    standard = _spelling(attrs.get("standard_name") or "")
    for field, value in _STANDARD_NAMES.get(standard, {}).items():
        offer(field, value)

    if _spelling(base) in _PRESSURE_NAMES:
        offer("origin", _PRESSURE_ORIGIN)
        if units != "m":
            offer("units", "dbar")

    return {f: found[f] for f in FIELDS if f in found}


# -- converting values -----------------------------------------------------------------


def to_positive_down(
    values: Any, *, positive: str = "down", units: str = "m"
) -> np.ndarray:
    """Raw vertical values as float metres, positive down.

    Negated when ``positive == "up"`` and multiplied by :data:`M_PER_DBAR` when
    ``units == "dbar"`` (approximate -- see there). NaN is preserved, and the result is
    always a new array, so a caller may edit it in place without touching its input.
    """
    sign = _checked("positive", positive)
    unit = _checked("units", units)
    out = np.array(values, dtype=float)
    if sign == "up":
        np.subtract(0.0, out, out=out)  # 0 - x rather than -x: a depth of 0 stays +0.0
    if unit == "dbar":
        out *= M_PER_DBAR
    return out


def coordinate_positive(
    attrs: Mapping[str, Any] | None, values: Any = None
) -> str | None:
    """Which way a vertical coordinate counts: its CF ``positive``, else its values'.

    ``attrs["positive"]`` when it is set to something recognisable. Otherwise, from
    ``values``: all finite values ``<= 0`` with at least one ``< 0`` reads as ``up`` (a
    height, or depths stored negative), any other finite values -- all zeros included --
    as ``down``. ``None`` when neither attrs nor values say anything, and the caller
    decides what to assume.
    """
    declared = _canonical_value("positive", (attrs or {}).get("positive"))
    if declared:
        return declared
    if values is None:
        return None
    finite = np.asarray(values, dtype=float).ravel()
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    if bool(np.all(finite <= 0)) and bool(np.any(finite < 0)):
        return "up"
    return "down"


def _convention_value(convention: Any, field: str) -> str | None:
    """``field`` from ``convention`` *when it was actually established*, else ``None``.

    A :class:`ResolvedConvention` knows how each field was decided: a declared, inferred
    or data-derived value is a statement about this source, but ``default`` is only
    "nobody said" -- it must not override what the coordinate's own attributes or values
    can tell. A plain mapping has no provenance, so a value it sets is taken as stated.
    """
    if convention is None:
        return None
    if isinstance(convention, ResolvedConvention):
        if convention.provenance[field] == "default":
            return None
        return getattr(convention, field)
    if isinstance(convention, Mapping):
        value = convention.get(field)
        return None if value is None else _checked(field, value)
    raise ValueError(
        "depth_convention: convention must be a ResolvedConvention or a mapping, "
        f"got {type(convention).__name__}"
    )


def positive_down_values(
    values: Any, attrs: Mapping[str, Any] | None = None, *, convention: Any = None
) -> np.ndarray:
    """Return a vertical coordinate as float metres, positive down.

    The one place that decides how a coordinate counts. It replaces every ``abs()`` on
    a vertical coordinate, which read a height as a depth and a depth as itself whatever
    the source said. If ``attrs[NORMALIZED_ATTR]`` is truthy the values were already
    normalised on the way in and come back as floats, untouched. Otherwise the sign is
    the convention's ``positive`` when it was declared, inferred or read from the data
    (or when ``convention`` is a plain mapping that sets it), else
    :func:`coordinate_positive` of ``attrs`` and ``values``, else ``down``; the units
    follow the same precedence, then ``attrs["units"]`` (a decibar spelling means
    dbar), then metres. ``convention`` is a :class:`ResolvedConvention` or a mapping
    with ``positive``/``units`` keys.
    """
    attrs = attrs or {}
    if attrs.get(NORMALIZED_ATTR):
        return np.array(values, dtype=float)
    raw = np.asarray(values, dtype=float)
    positive = _convention_value(convention, "positive")
    if positive is None:
        positive = coordinate_positive(attrs, raw) or "down"
    units = _convention_value(convention, "units")
    if units is None:
        units = (
            "dbar" if _canonical_value("units", attrs.get("units")) == "dbar" else "m"
        )
    return to_positive_down(raw, positive=positive, units=units)
