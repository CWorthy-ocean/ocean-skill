"""Unit handling, delegated to `pint <https://pint.readthedocs.io>`_.

Models and climatologies express the same quantity differently: ROMS/MARBL carries
nitrate as mmol m-3, while WOA and GLODAP carry umol kg-1 -- and a CTD's Winkler-
titrated dissolved oxygen often ships as a gas volume, mL/L, against a model's molar
mmol/m3. Reconciling that by hand in every script is the easiest way to publish a
wrong number, so it lives here.

This used to be three hand-maintained sets of unit *strings* and one hard-coded
multiply. pint replaces the arithmetic and, more importantly, adds **dimensional
analysis** — the thing that lets :func:`compatible` refuse to subtract a per-mass field
from a per-volume one instead of silently returning a plausible wrong number.

One registry is shared with ``pint-pandas`` and ``pint-xarray``, so a quantity in a
DataFrame (point/timeseries observations) and one in a Dataset (gridded) are directly
comparable. Mixing registries raises in pint, so everything must go through
:func:`registry`. That registry is ``cf_xarray.units.units`` -- the one cf-xarray itself
installs as pint's application registry, which is what those two libraries read -- with
the ocean units (``oxygen_ml_per_l``, equivalents) and the ``seawater`` density context
added once from ``ocean_skill/vocab/units.txt``, a plain pint definitions file other
packages can load into the same registry.

**pint alone is not enough for ocean data**, which was measured rather than assumed: of
21 unit spellings taken from the real WOA/GLODAP/MODIS/OOI catalogs, pint's default
registry parsed 7. The rest are CF/UDUNITS conventions it does not implement —
``kg-1`` exponents, ``micromoles_per_kilogram``, ``meq``, ``PSU``. :func:`normalize`
bridges that gap with rewrite *rules* rather than a lookup table, so a spelling nobody
has seen yet still parses if it follows the conventions; only genuine typos
(:data:`_FIXED_SPELLINGS`) need enumerating.
"""

from __future__ import annotations

import importlib.resources
import re
import warnings

from ocean_skill import _stacklevel
from ocean_skill.vocabulary import narrower_names, nickname, resolve_name

__all__ = [
    "O2_MMOL_PER_ML",
    "RHO_SEAWATER",
    "compatible",
    "convert_units",
    "display",
    "find_variable",
    "for_statistic",
    "is_spread",
    "normalize",
    "parse",
    "registry",
    "to_units",
]

#: Nominal seawater density (kg m-3) for per-mass <-> per-volume conversion. A constant
#: is an approximation: true density varies with T/S/P by a few parts per thousand near
#: the surface. Use ``gsw`` with in-situ T/S if that matters for your comparison.
RHO_SEAWATER = 1025.0

#: mmol of O2 per mL of O2 gas at standard temperature and pressure -- the UNESCO/WOCE
#: molar volume (22.392 L per mole of O2, Garcia & Gordon 1992's own figure) most
#: Winkler-titration and CTD oxygen sensor software already assumes when it reports
#: dissolved oxygen as a gas volume (``mL/L``) rather than a molar concentration. A
#: constant here for the same reason :data:`RHO_SEAWATER` is: real gas behavior shifts
#: it a little with temperature and pressure, and a source whose own processing used a
#: different figure should override this. Unlike :data:`RHO_SEAWATER`, though, the value
#: that conversions use lives in ``vocab/units.txt`` (the ``oxygen_ml_per_l`` unit, a
#: concrete pint definition rather than a context parameter), so changing this name
#: changes nothing -- edit the definition there. A test keeps the two equal.
O2_MMOL_PER_ML = 1000.0 / 22.392  # ~44.661 mmol/m3 per mL/L

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

#: A dissolved-oxygen gas volume, in every spelling this package has seen
#: (``mL/L``, ``ml.l-1``, ``ml*l-1``, ``milliliters per liter``, ...). To pint, ``mL/L``
#: is bare dimensionless -- the same dimensionality :data:`practical_salinity_unit`
#: (``PSU``) already occupies, so a generic "any dimensionless quantity converts to a
#: molar concentration" rule (a context keyed on abstract dimensionality, the way the
#: ``seawater`` per-mass/per-volume one is) would just as happily, and wrongly, treat a
#: salinity field as a convertible oxygen reading — pint contexts dispatch on
#: dimensionality, not on which named unit symbol got you there. Rewriting *this*
#: specific spelling to its own concretely-dimensioned unit
#: (``oxygen_ml_per_l``, :func:`registry`) sidesteps that collision entirely: PSU's own
#: spellings never match this pattern, so nothing about it changes, and no context is
#: needed at all -- the new unit's dimensionality already matches ``mmol/m3`` directly.
_OXYGEN_ML_PER_L = re.compile(
    r"^(?:ml|milli-?liters?)[\s_]*[/.*]?[\s_]*(?:per[\s_]+)?l(?:iters?)?(?:-1)?$",
    re.IGNORECASE,
)

_registry = None


def registry():
    """Return the shared pint registry: cf-xarray's, plus the ocean definitions.

    Deliberately *cf-xarray's* registry (``cf_xarray.units.units``) rather than a
    private one, or pint's bare application registry: pint refuses to combine
    quantities from different registries, so a private one here would make
    ocean-skill's units unusable alongside a user's own pint code, or alongside
    pint-pandas in the same session. Importing ``cf_xarray.units`` also installs its
    registry as the application registry (:func:`pint.set_application_registry`), and
    :func:`pint.get_application_registry` is a wrapper that follows that, so
    pint-pandas and pint-xarray end up on this very registry however the imports are
    ordered. Building a registry of our own on the application registry instead would
    be silently orphaned the moment anything imported ``cf_xarray.units`` afterwards.

    Everything cf-xarray lacks comes from ``ocean_skill/vocab/units.txt``, loaded once
    (:func:`_load_definitions`).
    """
    global _registry
    if _registry is not None:
        return _registry

    import cf_xarray.units

    ureg = cf_xarray.units.units
    _load_definitions(ureg)
    _registry = ureg
    return ureg


def _definitions_loaded(ureg) -> bool:
    """Whether ``units.txt`` is already in ``ureg``: its last unit and its context."""
    if "oxygen_ml_per_l" not in ureg:
        return False
    try:
        with ureg.context("seawater"):
            return True
    except KeyError:  # pint's "no such context"
        return False


def _load_definitions(ureg) -> None:
    """Add ``vocab/units.txt`` to ``ureg`` unless something already has.

    pint answers a second load of the same definitions with "Redefining ..." log
    warnings for every unit and context, and cf-xarray's registry is process-wide, so
    another package (ROMS-Tools loads the same file) or an earlier call can have got
    there first. Skipping is what makes loading it from several places quiet.
    """
    if _definitions_loaded(ureg):
        return
    source = importlib.resources.files("ocean_skill") / "vocab" / "units.txt"
    with importlib.resources.as_file(source) as path:
        ureg.load_definitions(path)


def _seawater(ureg):
    """Return the ``seawater`` density context at the current :data:`RHO_SEAWATER`.

    Density is the context's ``rho`` parameter (kg m-3), passed on every use rather
    than baked in, so :data:`RHO_SEAWATER` is read when a conversion happens and a
    new value needs no registry rebuild -- which matters now that the registry is
    cf-xarray's, shared with everyone else in the process.
    """
    return ureg.context("seawater", rho=RHO_SEAWATER)


def normalize(unit_string) -> str:
    """Rewrite a CF/UDUNITS unit string into something pint parses.

    Rules, not a lookup table: ``_per_`` becomes ``/``, ``.`` becomes a space,
    hyphenated SI prefixes are joined, and adjacent exponents (``kg-1``, ``m3``)
    become explicit (``kg**-1``, ``m**3``). Empty or purely numeric strings are
    dimensionless. Only :data:`_FIXED_SPELLINGS` is an enumeration, because a
    misspelling follows no rule.

    A dissolved-oxygen gas-volume spelling (:data:`_OXYGEN_ML_PER_L`, checked
    first since it would otherwise just read as bare dimensionless) rewrites to
    ``oxygen_ml_per_l`` -- a real unit, not a lookup-table shortcut, since this
    domain never reports anything else in ``mL/L``: see that pattern's own
    comment for why a generic dimensionless-to-molar rule is not safe here.
    """
    text = str(unit_string or "").strip()
    if not text:
        return "dimensionless"
    if _OXYGEN_ML_PER_L.match(text):
        return "oxygen_ml_per_l"
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

    A missing attribute (``None``) is deliberately not the same as an *empty* one
    (``""``, which CF legitimately uses for a dimensionless quantity): normalizing
    ``None`` to ``""`` would parse as "dimensionless" and make :func:`compatible`
    report a real physical mismatch against a variable that simply never recorded
    its units.
    """
    if unit_string is None:
        return None
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
    if ua.is_compatible_with(ub, "seawater", rho=RHO_SEAWATER):
        return True
    # A variance in (mmol/m^3)^2 against one in (umol/kg)^2: the seawater context is
    # keyed on the *un-squared* dimensionality and never fires for these, so ask about
    # the square roots instead (see :func:`_square_roots`).
    roots = _square_roots(ua, ub)
    return bool(roots) and roots[0].is_compatible_with(
        roots[1], "seawater", rho=RHO_SEAWATER
    )


def _square_roots(ua, ub):
    """Return ``(sqrt(ua), sqrt(ub))`` when both are squared quantities, else ``None``.

    "Squared" is read from the *dimensionality*, not the spelling: every dimension's
    exponent even and at least one non-zero, so ``(mmol/m^3)^2``, ``delta_degC^2`` and
    ``mg^2 m-6`` all qualify and ``mg/m^3`` or a bare ``m^3`` do not. That is what a
    variance is (see :func:`for_statistic`), and it is the only case this exists for.

    Needed because pint's ``seawater`` context is keyed on the *un-squared*
    ``[substance]/[mass]`` <-> ``[substance]/[length]**3`` dimensionalities, so a
    variance in ``(mmol/m^3)^2`` against one in ``(umol/kg)^2`` is "dimensionally
    unrelated" to it -- it neither converts nor reports compatible. Converting the
    square roots and squaring the factor is exactly right (``var(a x) = a^2 var(x)``),
    and uses the same density, applied twice.
    """
    for unit in (ua, ub):
        exponents = [e for _, e in unit.dimensionality.items()]
        if not exponents or any(float(e) % 2 for e in exponents):
            return None
    return ua**0.5, ub**0.5


def _is_offset(ureg, unit) -> bool:
    """Whether ``unit`` has a shifted zero (degC, degF) rather than only a scale."""
    try:
        return float(ureg.Quantity(0.0, unit).to_root_units().magnitude) != 0.0
    except (TypeError, ValueError):  # pint's own errors subclass these
        return False


def _delta(ureg, unit):
    """Return the difference-of-temperatures twin of an offset ``unit``.

    Anything without a shifted zero is returned as it came. A *difference* of two
    temperatures -- which is what a standard deviation or a range of temperatures is --
    scales like a temperature but is not shifted by 273.15, and pint models that as a
    separate ``delta_`` unit (degC -> delta_degC).
    """
    if not _is_offset(ureg, unit):
        return unit
    return ureg.Unit(f"delta_{unit}")


def _scale_and_shift(ureg, ua, ub):
    """Return ``(factor, shift)`` taking ``ua`` to ``ub``; raise if they cannot.

    ``factor`` is the image of 1 and ``shift`` that of 0, so a pure scale has
    ``shift == 0`` and ``factor`` its multiplier, and a shifted zero shows as
    ``shift != 0``. A squared pair the context cannot see through
    (:func:`_square_roots`) is converted via its roots: the factor is squared and the
    shift is zero (an offset unit cannot be squared; pint already refuses one).
    """
    try:
        with _seawater(ureg):
            # float(): cf-xarray's registry hands back 0-d arrays (force_ndarray_like)
            return (
                float(ureg.Quantity(1.0, ua).to(ub).magnitude),
                float(ureg.Quantity(0.0, ua).to(ub).magnitude),
            )
    except Exception:
        roots = _square_roots(ua, ub)
        if roots is None:
            raise
    factor, shift = _scale_and_shift(ureg, *roots)
    if shift != 0.0:
        raise ValueError("an offset unit cannot be squared")
    return factor**2, 0.0


def to_units(da, target, *, rho: float | None = None):
    """Return ``da`` converted to ``target`` units, or unchanged if it cannot be.

    Unchanged (rather than raising) when either side is unparseable or the two are
    dimensionally unrelated — :func:`compatible` is where a caller asks the question
    and decides. Conversion goes through the ``seawater`` context so per-mass and
    per-volume concentrations interconvert. Offset units (degC, degF) are converted by
    value, not by a scale factor, so a K field against degC obs lands correctly.

    **A spread is not a value.** A standard deviation, range or variance of a
    temperature must never pick up the 273.15 of a K <-> degC conversion: a std of 0.3 K
    is a std of 0.3 degC, not 273.45. So when the field is a difference -- its units
    already say so (``delta_degC``), its ``statistic`` attribute is a spread statistic
    (:func:`is_spread`), or the *target* is a ``delta_`` unit -- both sides are read as
    differences (:func:`_delta`) and the conversion is a pure scale. Variances arrive
    squared (:func:`for_statistic`) and convert with the squared factor: a variance in
    ``(mmol/m^3)^2`` becomes ``(umol/kg)^2`` through the density *twice*
    (:func:`_square_roots`). ``target`` is returned as spelled, so a std converted "to
    degC" still reads ``degC`` -- the numbers are right and the ``statistic`` attribute
    says what they are.
    """
    source = da.attrs.get("units")
    ua, ub = parse(source), parse(target)
    if ua is None or ub is None:
        return da
    ureg = registry()
    if (
        "delta_" in str(ua)
        or "delta_" in str(ub)
        or is_spread(da.attrs.get("statistic"))
    ):
        ua, ub = _delta(ureg, ua), _delta(ureg, ub)
    if ua == ub:
        return da
    if rho is not None and rho != RHO_SEAWATER:
        raise NotImplementedError(
            "a per-call density is not supported yet; set units.RHO_SEAWATER instead"
        )
    try:
        factor, shift = _scale_and_shift(ureg, ua, ub)
        if shift != 0.0:
            with _seawater(ureg):
                data = ureg.Quantity(da.data, ua).to(ub).magnitude
    except Exception:
        return da  # dimensionally unrelated; the caller checks compatible()
    # A shifted zero (degC/degF against K) is not a scale: 10 degC is 283.15 K, not
    # 10 x 274.15. Pint converts the values themselves; plain arithmetic on ``da.data``
    # keeps a dask-backed array lazy.
    if shift == 0.0:
        out, note = da * factor, f"x{factor:g}"
    else:
        out, note = da.copy(data=data), "offset"
    out.attrs = dict(da.attrs)
    out.attrs["units"] = str(target)
    if factor != 1.0 or shift != 0.0:
        out.attrs["unit_conversion"] = f"{source} -> {target} ({note})"
    return out


#: Statistics whose result is the *square* of the field's units. ``statistic`` is how a
#: reduction names itself in ``attrs["statistic"]`` (see ``operators.aggregate``).
_VARIANCE_STATISTICS = frozenset({"var", "variance"})

#: Statistics whose result is a *difference* of two values of the field: same magnitude
#: units, but zero means "no spread", so an offset unit must not be shifted. ``mad`` and
#: ``iqr`` are here for the same reason as ``std``; a ``quantile`` or ``median`` is a
#: value of the field, so it is not.
_DIFFERENCE_STATISTICS = frozenset(
    {
        "std",
        "stddev",
        "std_dev",
        "standard_deviation",
        "sem",
        "range",
        "ptp",
        "peak_to_peak",
        "mad",
        "median_abs_deviation",
        "iqr",
    }
)


def _statistic_name(statistic) -> str:
    return statistic.strip().lower() if isinstance(statistic, str) else ""


def is_spread(statistic) -> bool:
    """Whether ``statistic`` measures how much a field varies rather than its level.

    True for variance, standard deviation, range and the like (the sets in
    :func:`for_statistic`). A spread has no business on a variable's own pinned display
    range or log scale -- the variance of chlorophyll is nowhere near 0.01-10 mg/m3 --
    and no business being shifted by a K <-> degC offset. ``None``, ``mean``, ``max``,
    ``median``... are all False.
    """
    name = _statistic_name(statistic)
    return name in _VARIANCE_STATISTICS or name in _DIFFERENCE_STATISTICS


#: A unit that is a single bare symbol (``K``, ``degC``, ``delta_degC``, ``PSU``), as
#: opposed to a compound that needs brackets before it is raised to a power: ``mg/m^3``
#: squared is ``(mg/m^3)^2``, never ``mg/m^3^2``.
_BARE_SYMBOL = re.compile(r"^[A-Za-z_°µ][A-Za-z_0-9°µ]*$")


def _as_difference(unit_string: str) -> str:
    """Spell the units of a *difference* of two ``unit_string`` values.

    Unchanged unless the unit has a shifted zero, when it is its ``delta_`` twin
    (``degC`` -> ``delta_degC``). Unparseable units are returned as they came: there is
    nothing to be shifted if nobody can convert them anyway.
    """
    ureg = registry()
    unit = parse(unit_string)
    if unit is None or not _is_offset(ureg, unit):
        return unit_string
    spelled = f"delta_{normalize(unit_string)}"
    return spelled if parse(spelled) is not None else f"delta_{unit}"


def _squared(unit_string: str) -> str:
    """Spell ``unit_string`` squared, in a form :func:`normalize` reads.

    ``K`` -> ``K^2``; ``mg/m^3`` -> ``(mg/m^3)^2``. ``^`` rather than ``**`` because
    that is how this package's own unit strings are written (``mg/m^3``). A bare number
    (WOA's salinity ``1e-3``) is squared arithmetically, because ``normalize`` would
    read the ``e-3`` of ``(1e-3)^2`` as a unit exponent.

    Bare means bare *once normalized*: ``micromoles_per_kilogram`` looks like one
    symbol but :func:`normalize` reads it as ``micromoles/kilogram``, so an unbracketed
    ``micromoles_per_kilogram^2`` would square only the kilogram.
    """
    text = unit_string.strip()
    if _NUMERIC.match(text):
        return f"{float(text) ** 2:g}"
    try:
        bare = _BARE_SYMBOL.match(normalize(text)) is not None
    except Exception:  # noqa: BLE001 -- an unparseable unit still gets bracketed
        bare = False
    return f"{text}^2" if bare and _BARE_SYMBOL.match(text) else f"({text})^2"


def for_statistic(unit_string, statistic: str | None):
    """Units of ``statistic`` taken over a field carrying ``unit_string``.

    A reduction does not always leave a field's units alone: the variance of a
    temperature is in degrees squared, and a standard deviation of one is a
    *difference* of temperatures. Handing the field's own units to either is how a std
    gets "converted to kelvin" by adding 273.15, or a variance gets its density factor
    applied once instead of twice. ``operators.aggregate`` therefore runs the result's
    units through here after every reduction.

    * ``var`` / ``variance``: the units squared -- ``K^2``, ``(mmol/m^3)^2``. Offset
      units (degC, degF) become the square of their delta form, ``delta_degC^2``, so a
      conversion scales and never shifts. ``(<units>)^2`` is also what an *unparseable*
      unit becomes (best effort, no crash; it simply stays unconvertible).
    * ``std``, ``range``/``ptp``, ``mad``, ``iqr``, ``sem`` (:func:`is_spread`): the
      same magnitude, as a difference -- ``delta_degC`` for degC, anything without a
      shifted zero unchanged (``mg/m^3`` stays ``mg/m^3``).
    * anything else -- ``mean``, ``max``, ``min``, ``median``, ``quantile``, ``None`` --
      returns ``unit_string`` untouched, as do an empty or missing ``unit_string``
      (``None`` stays ``None``, ``""`` stays ``""``). ``sum``/``integrate`` change units
      too but are a different bookkeeping and not handled here.

    The strings are all valid input to :func:`normalize`/:func:`parse`/:func:`to_units`
    and survive a round trip. Not idempotent -- squaring a variance's units again gives
    a fourth power -- so call it once per reduction, on the *field's* units. Use
    :func:`display` for a colorbar or axis label.
    """
    name = _statistic_name(statistic)
    if unit_string is None or not str(unit_string).strip():
        return unit_string
    text = str(unit_string).strip()
    if name in _VARIANCE_STATISTICS:
        return _squared(_as_difference(text))
    if name in _DIFFERENCE_STATISTICS:
        return _as_difference(text)
    return unit_string


#: ``^2`` / ``**2`` -> ``²``: the exponent characters a label can use.
_SUPERSCRIPTS = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
_EXPONENT = re.compile(r"(?:\^|\*\*)(-?\d+)")
_DELTA_DEGREE = re.compile(r"\bdelta_(?:degree_)?(?:deg)?(C|Celsius|F|Fahrenheit)\b")
_DEGREE = re.compile(r"\bdeg(?:ree)?s?_?(C|Celsius|F|Fahrenheit)\b")


def display(unit_string) -> str:
    """Return ``unit_string`` spelled for a colorbar or axis label.

    The strings :func:`for_statistic` produces are written for pint, not for people:
    ``delta_degC^2`` reads as ``°C²`` here and ``(mg/m^3)^2`` as
    ``(mg/m³)²``. ``delta_`` is dropped (a spread of temperatures is plainly
    in degrees), ``degC``/``degF`` become ``°C``/``°F`` and ``^n`` / ``**n``
    become superscripts. A string with none of those is returned as it came, so it is
    safe to run over every label; ``None`` is ``""``.
    """
    text = "" if unit_string is None else str(unit_string)
    text = _DELTA_DEGREE.sub(lambda m: "°" + m.group(1)[0], text)
    text = _DEGREE.sub(lambda m: "°" + m.group(1)[0], text)
    text = re.sub(r"\bdelta_", "", text)
    return _EXPONENT.sub(lambda m: m.group(1).translate(_SUPERSCRIPTS), text)


def convert_units(da, target: str = "mmol/m^3", rho: float = RHO_SEAWATER):
    """Convert ``da`` to ``target`` when the two are the same physical quantity.

    Applied per lane so both sides of a comparison land in one convention. A field
    that is a *different* quantity (chlorophyll in mg m-3, temperature in degC)
    passes through untouched and, unlike the previous string-matching version,
    **without a spurious warning** — "not the target quantity" is the normal case for
    every variable that is not a nutrient, not something to report.

    A spread statistic (``attrs["statistic"]``, see :func:`for_statistic`) is moved to
    the *same statistic of* ``target`` -- a nitrate variance in ``(umol/kg)^2`` lands
    in ``(mmol/m^3)^2`` -- so a variance lane sits in the same convention as the mean
    lane beside it rather than staying in whatever its source shipped.
    """
    global RHO_SEAWATER
    if rho != RHO_SEAWATER:
        RHO_SEAWATER = rho  # the context reads it on every use (see _seawater)
    have = da.attrs.get("units")
    if compatible(have, target):  # a target already spelled as the statistic's units
        return to_units(da, target)
    statistic = da.attrs.get("statistic")
    if is_spread(statistic):
        goal = for_statistic(target, statistic)
        if compatible(have, goal):
            return to_units(da, goal)
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


def _cf_name(searchable, name: str, standard_name: str) -> str | None:
    """Return the one variable cf-xarray finds for ``standard_name``, else ``None``.

    ``searchable`` is the dataset to ask (with any QC flags already dropped from
    it). cf-xarray raises a plain ``KeyError`` both for "no such key" and for
    *several* variables matching it, and a name that cannot be pinned to one
    variable is treated as not found rather than guessed at -- but the second case is
    worth saying, since the variable is plainly there: the list form of the same
    lookup (``.cf[[key]]``) succeeds only when there were several, and names them.
    The usual way to land here is a generic mixed layer depth asked of a dataset
    carrying two of its definitions (the generic entry's criteria match every
    specific one -- see :func:`ocean_skill.vocabulary._register_custom_criteria`),
    and the answer is the same as for any other ambiguity: ask for one by name.
    """
    try:
        return str(searchable.cf[standard_name].name)
    except KeyError:
        pass
    try:
        candidates = sorted(str(v) for v in searchable.cf[[standard_name]].data_vars)
    except KeyError:
        return None  # genuinely absent
    if len(candidates) > 1:
        listed = ", ".join(_describe_candidate(searchable, c) for c in candidates)
        warnings.warn(
            f"{name!r} matches more than one variable in this dataset ({listed}), so "
            "it is not being resolved to either; ask for the one you mean by name.",
            stacklevel=_stacklevel.find(),
        )
    return None


def _describe_candidate(ds, variable: str) -> str:
    """``'var'``, plus the vocabulary key to ask for it by, when it has one."""
    known = resolve_name(str(ds[variable].attrs.get("standard_name") or variable))
    key = nickname(known)
    if key and key != variable:
        return f"{variable!r} (ask for {key!r})"
    return repr(variable)


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

    **A generic name can be answered by one specific definition** — with no code
    here for it: the generic mixed layer depth's cf-xarray criteria also match each of
    its definitions' spellings (see
    :func:`ocean_skill.vocabulary._register_custom_criteria`), so ``"mld"`` finds a
    variable named ``..._defined_by_sigma_theta`` through the same ``.cf[...]`` step
    as any alias. A variable literally carrying the generic name still wins, because
    the literal-name check runs first; a request for a specific definition never
    finds the generic name or a sibling, since its own criteria name only itself.

    Warns once, naming both, whenever ``name`` isn't literally what this dataset
    calls the variable — including the exact spelling actually found, which may
    differ per dataset even when every caller asks by the same canonical name.

    Returns ``None`` if nothing matches. That includes a name cf-xarray finds on
    *several* variables at once — two definitions of mixed layer depth, or two alias
    spellings of one concept — which is neither raised nor silently resolved to one
    of them: it warns, naming the candidates, and returns ``None``, so a comparison
    skips the pair as not available and a combination still falls back to its own
    ``standard_name``. Raises :class:`ValueError` if two variables differ only by
    case (:func:`_match_name`).
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
        found_name = _cf_name(searchable, name, standard_name)
    if found_name is None:
        if not allow_qc:
            _warn_if_only_a_flag_matched(ds, standard_name)
        return None
    da = ds[found_name]

    found = da.attrs.get("standard_name") or found_name
    if name != found:
        detail = f"{name!r} resolved to {found!r}"
        if resolve_name(found) in narrower_names(standard_name):
            # The generic name's criteria reached one of its specific definitions.
            if found_name != found:
                detail += f" (variable {found_name!r})"
            of = "it" if name == standard_name else repr(standard_name)
            detail += f", one specific definition of {of}"
        elif standard_name not in (name, found):
            detail += f" (standard_name {standard_name!r})"
        warnings.warn(detail, stacklevel=_stacklevel.find())
    return da
