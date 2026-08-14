"""Unit handling, delegated to `pint <https://pint.readthedocs.io>`_.

Models and climatologies express the same quantity differently: ROMS/MARBL carries
nitrate as mmol m-3, while WOA and GLODAP carry umol kg-1. Reconciling that by hand in
every script is the easiest way to publish a wrong number, so it lives here.

This used to be three hand-maintained sets of unit *strings* and one hard-coded
multiply. pint replaces the arithmetic and, more importantly, adds **dimensional
analysis** — the thing that lets :func:`compatible` refuse to subtract a per-mass field
from a per-volume one instead of silently returning a plausible wrong number.

One registry is shared with ``pint-pandas`` and ``pint-xarray`` via
:func:`pint.get_application_registry`, so a quantity in a DataFrame (point/timeseries
observations) and one in a Dataset (gridded) are directly comparable. Mixing registries
raises in pint, so everything must go through :func:`registry`.

**pint alone is not enough for ocean data**, which was measured rather than assumed: of
21 unit spellings taken from the real WOA/GLODAP/MODIS/OOI catalogs, pint's default
registry parsed 7. The rest are CF/UDUNITS conventions it does not implement —
``kg-1`` exponents, ``micromoles_per_kilogram``, ``meq``, ``PSU``. :func:`normalize`
bridges that gap with rewrite *rules* rather than a lookup table, so a spelling nobody
has seen yet still parses if it follows the conventions; only genuine typos
(:data:`_FIXED_SPELLINGS`) need enumerating.
"""

from __future__ import annotations

import re
import warnings

from ocean_skill import _stacklevel
from ocean_skill.vocabulary import resolve_name

__all__ = [
    "RHO_SEAWATER",
    "compatible",
    "convert_units",
    "find_variable",
    "normalize",
    "parse",
    "registry",
    "to_units",
]

#: Nominal seawater density (kg m-3) for per-mass <-> per-volume conversion. A constant
#: is an approximation: true density varies with T/S/P by a few parts per thousand near
#: the surface. Use ``gsw`` with in-situ T/S if that matters for your comparison.
RHO_SEAWATER = 1025.0

#: Spellings no rule can recover, because they are mistakes or free text rather than a
#: convention. Kept deliberately tiny — anything that *follows* a convention belongs in
#: :func:`normalize`'s rules instead, or it becomes the string table this replaced.
_FIXED_SPELLINGS = {
    "degrees celcius": "degC",  # sic: this misspelling is in real WOA-derived files
    "degrees celsius": "degC",
    "degree celsius": "degC",
    "practical salinity units": "PSU",
    "practical_salinity_units": "PSU",
    # Parts-per-thousand spellings for salinity. `ppt` has to be listed because pint
    # reads it as *pico-pint* -- a real unit, dimensionally wrong, and silently so:
    # compatible("1e-3", "ppt") came back False, which made a mooring-versus-product
    # salinity comparison refuse to run at all (OceanSODA-ETHZ writes `ppt`).
    "ppt": "PSU",
    "ppth": "PSU",
    "parts per thousand": "PSU",
    "psu": "PSU",  # pint's own `psu` works; here so a stray case never reaches it
}

#: ``micro-mol`` -> ``micromol``: UDUNITS hyphenates SI prefixes, pint does not.
_PREFIX_HYPHEN = re.compile(
    r"\b(micro|milli|nano|kilo|centi|deci|pico|femto|deca|hecto)-"
)
#: ``kg-1``/``m3`` -> ``kg**-1``/``m**3``: UDUNITS writes exponents adjacent, and pint
#: reads a bare ``-1`` as subtraction (a TypeError, not a wrong answer, thankfully).
_UDUNITS_EXPONENT = re.compile(r"(?<=[a-zA-Z])\s*(-?\d+)(?![0-9])")
#: A bare number is a scale factor, not a unit — WOA writes salinity as ``1e-3``.
_NUMERIC = re.compile(r"^[0-9.]+([eE][-+]?[0-9]+)?$")

_registry = None


def registry():
    """Return the shared pint registry, defining ocean units and contexts once.

    Deliberately the *application* registry rather than a private one: pint refuses
    to combine quantities from different registries, so a private one here would
    make ocean-skill's units unusable alongside a user's own pint code, or alongside
    pint-pandas in the same session.
    """
    global _registry
    if _registry is not None:
        return _registry

    import pint

    ureg = pint.get_application_registry()
    _define(ureg, "equivalent = mole = eq")  # alkalinity: 1 eq = 1 mol of charge
    _define(ureg, "practical_salinity_unit = [] = PSU = psu")
    _define(ureg, "@alias degree_Celsius = Celsius = degrees_celsius = degrees_C")

    # Per-mass <-> per-volume is not a unit conversion: it needs a density, which is
    # physics, not arithmetic. A context is pint's way of saying "this transformation
    # is available when you ask for it" without making it silently automatic.
    ctx = pint.Context("seawater")
    ctx.add_transformation(
        "[substance] / [mass]",
        "[substance] / [length] ** 3",
        lambda ureg_, x: x * (RHO_SEAWATER * ureg_.kg / ureg_.m**3),
    )
    ctx.add_transformation(
        "[substance] / [length] ** 3",
        "[substance] / [mass]",
        lambda ureg_, x: x / (RHO_SEAWATER * ureg_.kg / ureg_.m**3),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # re-adding an identically named context
        ureg.add_context(ctx)
    _registry = ureg
    return ureg


def _define(ureg, definition: str) -> None:
    """Add a definition, tolerating one that a previous import already added."""
    try:
        ureg.define(definition)
    except Exception:  # RedefinitionError, or a newer pint already shipping it
        pass


def normalize(unit_string) -> str:
    """Rewrite a CF/UDUNITS unit string into something pint parses.

    Rules, not a lookup table: ``_per_`` becomes ``/``, ``.`` becomes a space,
    hyphenated SI prefixes are joined, and adjacent exponents (``kg-1``, ``m3``)
    become explicit (``kg**-1``, ``m**3``). Empty or purely numeric strings are
    dimensionless. Only :data:`_FIXED_SPELLINGS` is an enumeration, because a
    misspelling follows no rule.
    """
    text = str(unit_string or "").strip()
    if not text:
        return "dimensionless"
    if text.lower() in _FIXED_SPELLINGS:
        return _FIXED_SPELLINGS[text.lower()]
    if _NUMERIC.match(text):
        return "dimensionless"
    text = text.replace("_per_", "/").replace(".", " ")
    text = _PREFIX_HYPHEN.sub(r"\1", text)
    return _UDUNITS_EXPONENT.sub(r"**\1", text)


def parse(unit_string):
    """Return a pint ``Unit`` for a CF unit string, or ``None`` if unparseable.

    ``None`` rather than raising: an unrecognized unit should degrade to "cannot
    check this" and let the caller decide, not abort a comparison whose numbers may
    be perfectly fine.
    """
    try:
        return registry().Unit(normalize(unit_string))
    except Exception:
        return None


def compatible(a, b) -> bool | None:
    """Report whether two unit strings describe the same physical quantity.

    ``True``/``False`` when both parse, ``None`` when either does not — "unknown" is
    a third answer here, and collapsing it into ``False`` would block comparisons
    over a spelling problem rather than a physics problem.

    Per-mass and per-volume concentrations count as compatible: they are inter-
    convertible through the seawater density context, which is exactly the
    conversion this module exists to do.
    """
    ua, ub = parse(a), parse(b)
    if ua is None or ub is None:
        return None
    if ua.is_compatible_with(ub):
        return True
    return bool(ua.is_compatible_with(ub, "seawater"))


def to_units(da, target, *, rho: float | None = None):
    """Return ``da`` converted to ``target`` units, or unchanged if it cannot be.

    Unchanged (rather than raising) when either side is unparseable or the two are
    dimensionally unrelated — :func:`compatible` is where a caller asks the question
    and decides. Conversion goes through the ``seawater`` context so per-mass and
    per-volume concentrations interconvert.
    """
    source = da.attrs.get("units")
    ua, ub = parse(source), parse(target)
    if ua is None or ub is None or ua == ub:
        return da
    ureg = registry()
    if rho is not None and rho != RHO_SEAWATER:
        raise NotImplementedError(
            "a per-call density is not supported yet; set units.RHO_SEAWATER instead"
        )
    try:
        with ureg.context("seawater"):
            factor = ureg.Quantity(1.0, ua).to(ub).magnitude
    except Exception:
        return da  # dimensionally unrelated; the caller checks compatible()
    out = da * factor
    out.attrs = dict(da.attrs)
    out.attrs["units"] = str(target)
    if factor != 1.0:
        out.attrs["unit_conversion"] = f"{source} -> {target} (x{factor:g})"
    return out


def convert_units(da, target: str = "mmol/m^3", rho: float = RHO_SEAWATER):
    """Convert ``da`` to ``target`` when the two are the same physical quantity.

    Applied per lane so both sides of a comparison land in one convention. A field
    that is a *different* quantity (chlorophyll in mg m-3, temperature in degC)
    passes through untouched and, unlike the previous string-matching version,
    **without a spurious warning** — "not the target quantity" is the normal case for
    every variable that is not a nutrient, not something to report.
    """
    global RHO_SEAWATER
    if rho != RHO_SEAWATER:
        RHO_SEAWATER = rho  # keep the context's density in step with an explicit rho
        globals()["_registry"] = None
    if compatible(da.attrs.get("units"), target):
        return to_units(da, target)
    return da


#: Name components that mark a variable as a *flag about* a measurement rather than the
#: measurement. Matched as whole `_`-delimited tokens, not substrings: "qc" inside a
#: legitimate name (``qcm_index``) is not a flag, and excluding it would be worse than
#: the problem being solved.
_QC_TOKENS = frozenset({"qc", "qartod", "flag", "flags"})

_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")


def is_qc_name(name) -> bool:
    """Whether ``name`` looks like a QC flag rather than a measurement."""
    return bool(_QC_TOKENS & set(_TOKEN_SPLIT.split(str(name).lower())))


def _warn_if_only_a_flag_matched(ds, standard_name: str) -> None:
    """Say a QC flag was the sole match, rather than reporting a bare "not found".

    "This dataset does not have temperature" is confusing when the file plainly contains
    ``sea_water_temperature_qc_agg``, so the reason the match was refused is named. Only
    reached once nothing else matched, so it costs a second lookup on the failure path
    and nothing on the common one.
    """
    try:
        candidate = str(ds.cf[standard_name].name)
    except (KeyError, AttributeError):
        return
    if is_qc_name(candidate):
        warnings.warn(
            f"{standard_name!r} is not in this dataset; the closest match, "
            f"{candidate!r}, is a QC flag rather than the measurement it flags, so it "
            "is being ignored. Ask for it by name if the flags are what you want.",
            stacklevel=_stacklevel.find(),
        )


def _match_name(ds, standard_name: str, *, allow_qc: bool = False) -> str | None:
    """Return ``ds``'s own name for ``standard_name``, ignoring case, else ``None``.

    An exact hit wins outright; only failing that is a case-insensitive sweep run,
    so a dataset that spells the variable exactly right never pays for one. This
    covers names with no :data:`~ocean_skill.vocabulary.VOCABULARY` entry, which
    cf-xarray is therefore never asked about — most CF names need no entry, but
    they should still match regardless of capitalization.

    Raises if two variables differ only by case: which one was meant is genuinely
    unknowable, and picking either would be a coin flip on the returned data.
    """
    if standard_name in ds.variables:
        return standard_name
    lowered = standard_name.lower()
    hits = [str(v) for v in ds.variables if str(v).lower() == lowered]
    if not allow_qc:
        hits = [h for h in hits if not is_qc_name(h)]
    if len(hits) > 1:
        raise ValueError(
            f"{standard_name!r} matches {sorted(hits)} in this dataset, which differ "
            "only by case; rename one so the request is unambiguous."
        )
    return hits[0] if hits else None


def find_variable(ds, name: str):
    """Return the variable in ``ds`` matching ``name`` or a known equivalent.

    ``name`` may be anything :func:`ocean_skill.vocabulary.resolve_name` recognizes —
    a short vocabulary key, the canonical CF standard_name, or any alias, in any
    capitalization — resolved to the canonical standard_name first. Matching against
    the dataset's own variable names ignores case too, both here
    (:func:`_match_name`) and in cf-xarray's registered patterns.

    The canonical name, if it's already a literal variable — the common case, since
    CF-renaming (:func:`ocean_skill.roms.standardize`, the generic rename in
    :func:`ocean_skill.sources.read`) already did this for the primary variable —
    is checked *before* ever touching cf-xarray, deliberately. cf-xarray's
    ``.cf[...]`` independently checks every variable's ``standard_name`` attribute
    for *any* key shaped like one, regardless of what's registered in
    :func:`ocean_skill.vocabulary._register_custom_criteria` — and some real
    products (WOA) give auxiliary companion variables (``n_dd``/``n_se``/... —
    sample size, standard error) the *same* ``standard_name`` as the actual data
    variable, which turns a query that needed no aliasing at all into a spurious
    ambiguity error. cf-xarray is therefore only consulted to resolve the
    equivalent-spelling case, and even then only to resolve *which name* matches —
    the value returned is a plain ``ds[name]`` lookup by that name, not
    ``.cf[...]``'s own result, since its accessor also drops every coordinate that
    doesn't share a dimension with the match (lon/lat, the grid, ``z_rho``, ...),
    which every downstream step here depends on carrying along.

    Warns once, naming both, whenever ``name`` isn't literally what this dataset
    calls the variable — including the exact spelling actually found, which may
    differ per dataset even when every caller asks by the same canonical name.
    Returns ``None`` if nothing matches; raises if a dataset carries *both*
    spellings of the same concept at once (cf-xarray's own ambiguity error) rather
    than silently choosing one.
    """
    standard_name = resolve_name(name)
    # A request that names a flag gets a flag; a request that names a measurement never
    # does. Station tables carry `<var>_qc_agg`/`<var>_qc_tests` companions, and gridded
    # products carry flag variables that sometimes claim the *same* standard_name as the
    # data they flag -- which is a path into cf-xarray below that no anchored pattern
    # closes, since it matches on the attribute rather than the name.
    allow_qc = is_qc_name(name) or is_qc_name(standard_name)
    found_name = _match_name(ds, standard_name, allow_qc=allow_qc)
    if found_name is None:
        # Flags are made *invisible* to the search rather than fatal when one matches:
        # a dataset can carry both a flag claiming the canonical standard_name and the
        # real variable under an alias spelling, and the real one has to still win.
        searchable = (
            ds
            if allow_qc
            else ds.drop_vars(
                [str(v) for v in ds.variables if is_qc_name(v)], errors="ignore"
            )
        )
        try:
            found_name = str(searchable.cf[standard_name].name)
        except KeyError:
            if not allow_qc:
                _warn_if_only_a_flag_matched(ds, standard_name)
            return None
    da = ds[found_name]

    found = da.attrs.get("standard_name") or found_name
    if name != found:
        detail = f"{name!r} resolved to {found!r}"
        if standard_name not in (name, found):
            detail += f" (standard_name {standard_name!r})"
        warnings.warn(detail, stacklevel=_stacklevel.find())
    return da
