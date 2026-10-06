"""Colormap resolution: xcmocean's own tables, extended in place for BGC species.

The single place both renderers get their colours from, so there is one policy to
maintain and a plot looks the same whichever backend draws it (see
:mod:`ocean_skill.plot.matplotlib_renderer` and
:mod:`ocean_skill.plot.holoviews_renderer`).

Two kinds of thing get coloured here, and they are kept apart. A **variable** —
nitrate, chlorophyll — resolves through :func:`cmaps_for`/:func:`norm_for` off its CF
standard_name. A **metric** — a map of bias, of correlation — resolves through
:func:`metric_colors` off its name in :data:`ocean_skill.metrics.REGISTRY`. Both are
edit-one-dict tables (:data:`_SEQUENTIAL_CMAPS`/:data:`_RANGES` for variables,
:data:`_METRIC_CMAPS`/:data:`_METRIC_RANGES` for metrics), but the metric tables are
deliberately *not* registered into xcmocean's own lookup the way the variable ones are:
that lookup is keyed by CF name and matches by substring; a metric is not a variable.

There is no separate ocean-skill colormap registry. :data:`_SEQUENTIAL_CMAPS` below —
the BGC species xcmocean has no opinion on, plus SSH, where we deliberately differ
from its default — is registered directly into xcmocean's own ``REGEX``/``SEQ``
tables at import time, via :func:`_register_colormaps`. Editing a variable's color
means editing that one dict; nothing else needs to change, and code that reaches for
xcmocean's own ``da.cmo.seq`` accessor directly sees the same colors this module
resolves.

Registration inserts our entries *ahead of* xcmocean's own table (see
:func:`_register_colormaps`) rather than appending them, because classification is
first-match-wins over the table in iteration order, and xcmocean's built-in ``"dye"``
vartype matches on the substring ``"concentration"`` — which is in nearly every CF
``mole_concentration_of_..._in_sea_water`` name. Appending would leave that pre-
existing, broader pattern matching first regardless of a later, more specific
registration, silently giving nitrate/phosphate/oxygen/DIC/chlorophyll the same
colormap (this was tried and is why it's called out here, not a hypothetical).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any

__all__ = [
    "METRIC_LIMIT_GROUPS",
    "MetricColors",
    "cmaps_for",
    "difference_cmap",
    "center_for",
    "is_log",
    "metric_colors",
    "norm_for",
    "variable_limits",
]

#: BGC species colors — edit here, nothing else needs to change. Values are cmocean
#: colormap names (``cmo.<name>``) or, for a map cmocean does not have, a matplotlib
#: one (``plasma``) -- see :func:`_resolve_cmap`. Reused as both the xcmocean "vartype"
#: key and its own (escaped, exact) match pattern.
#:
#: Policy: the twelve BGC species (nitrate, phosphate, silicate, ammonium, iron,
#: oxygen, DIC, alkalinity, chlorophyll, PAR, turbidity, pH) must each get a
#: *different* sequential map -- a figure that puts two of them side by side must not
#: paint them alike (this table's whole reason to exist: ammonium and iron used to
#: both fall through to xcmocean's "dye" default and collide). Everything else
#: (physics) follows xcmocean's own family conventions -- temperature/thermal,
#: salinity/haline, speed/speed, density/dense, depth/deep, ice/ice -- and *may*
#: share a hue with a BGC species, since the two are a different family and unlikely
#: to sit in the same figure. cmocean has 15 sequential maps and ~40 vocabulary
#: variables, so some cross-family sharing is unavoidable; where it happens it's a
#: deliberate choice, listed here rather than left as an accident to rediscover:
#:   - oxygen cmo.dense_r vs sigma_theta cmo.dense (opposite directions)
#:   - nitrate / mld / pressure / bathymetry: all cmo.deep -- "mld" meaning all five
#:     mixed-layer thickness names, the generic one and the four CF
#:     ``..._defined_by_<criterion>`` ones. That includes the sigma_theta-defined
#:     one: it is a depth in metres, not a density, so it takes the MLD map even though
#:     its name contains "sigma_theta" (see the ordering note at the "mixed_layer"
#:     entry below)
#:   - kd490 / turbidity: both cmo.turbid (same optical-quantity family)
#:   - sea_ice cmo.ice vs DIC cmo.ice_r (opposite directions, rarely adjacent)
#:   - pH cmo.speed_r vs the speed family's cmo.speed
#:   - iron cmo.amp also colours the rmse/mae/crmsd/std metric panels (see
#:     _METRIC_CMAPS below) -- a separate table by design, not a leak
#:   - ammonium cmo.gray_r likewise also colours the "n" (sample count) metric panel
#:   - iron / SSH tidal amplitude: both cmo.amp (different families, rarely adjacent)
#: Left on xcmocean's "dye" fallthrough (cmo.matter) rather than given a dedicated
#: entry, out of scope for this table's BGC-species guarantee: alkalinity (explicit
#: entry below, but still matter), phaeopigment, ciliate, diatom, dinoflagellate --
#: a cell-count/pigment family unlikely to sit beside the twelve species above.
_SEQUENTIAL_CMAPS: dict[str, str] = {
    # Height above the geoid has an arbitrary datum -- like temperature it is an
    # interval quantity, only differences mean anything -- so neither a white-to-colour
    # magnitude map (cmo.amp, xcmocean's own default for this vartype: white reads as
    # "none") nor a diverging one (cmo.balance: its white midpoint would mean
    # something it does not) fits. A plain perceptually-uniform sequential map does.
    # plasma rather than viridis because viridis is already xcmocean's fallback for
    # any variable it cannot match, so it would read as "no opinion" -- plasma is a
    # deliberate choice, and a matplotlib name, which :func:`_resolve_cmap` allows.
    "sea_surface_height_above_geoid": "plasma",
    # full standard_name: xcmocean's own "vel" pattern matches the substring "vel" in
    # "sea_le-vel-", which would otherwise give sea-level anomaly a velocity map.
    # (SLA, unlike ADT above, is a signed anomaly about zero. The rule for every such
    # variable: *if zero matters it is in* :data:`_CENTERED`, and :func:`cmaps_for`
    # then draws it with limits equal about the centre. :data:`_CENTERED` decides
    # *whether* a variable is centred; this table decides *which* diverging map --
    # balance unless the entry names another -- so every centred variable's entry here
    # must be a diverging map.)
    "sea_surface_height_above_sea_level": "cmo.balance",
    # Signed velocity components (the geostrophic ugos/vgos names are vocabulary aliases
    # of the first two): direction is the point, zero is "no flow", so diverging and
    # centred -- cmo.delta, the velocity family's own diverging map, which also keeps
    # a current section from reading like an SLA or CO2-flux one (balance). A velocity
    # *magnitude* is a different quantity, sequential from zero: see sea_water_speed
    # and wind_speed (cmo.speed). The explicit full names also stop xcmocean's own
    # substring "vel" pattern giving the components cmo.speed through ``da.cmo.seq``.
    "eastward_sea_water_velocity": "cmo.delta",
    "northward_sea_water_velocity": "cmo.delta",
    "sea_water_x_velocity": "cmo.delta",
    "sea_water_y_velocity": "cmo.delta",
    "upward_sea_water_velocity": "cmo.delta",
    "sea_water_speed": "cmo.speed",
    # Tidal harmonics of SSH (osk-custom names): amplitude is a magnitude, phase is
    # cyclic. The full names are no substring of the SSH keys above, so neither can
    # claim the other. The harmonic real/imaginary parts are left to the fallthrough.
    "sea_surface_height_tidal_amplitude": "cmo.amp",
    "sea_surface_height_tidal_phase": "cmo.phase",
    # x/y/eastward/northward baroclinic pressure flux: signed (energy flows either
    # way), so centred (:data:`_CENTERED`) and diverging.
    "baroclinic_pressure_flux": "cmo.balance",
    "nitrate": "cmo.deep",
    "phosphate": "cmo.rain",
    "silicate": "cmo.tempo",
    # Light is low, dark is high, like the other nutrient maps -- gray_r, not gray;
    # only oxygen (below) needs its low end dark.
    "ammonium": "cmo.gray_r",
    "iron": "cmo.amp",
    # Dark is low, light is high -- cmo.dense *reversed*. Unreversed, dense (like the
    # old gray_r before it) drew low oxygen as near-white, which on a white page reads
    # as "nothing there" for the one quantity whose low end (hypoxia) is the thing to
    # see. The substring key also covers oxygen_saturation, the other "oxygen"
    # standard_name.
    "oxygen": "cmo.dense_r",
    # "mole_concentration_of_dissolved_molecular_oxygen_in_sea_water": "cmo.oxy",
    "dissolved_inorganic_carbon": "cmo.ice_r",
    "sea_water_alkalinity_expressed_as_mole_equivalent": "cmo.matter",
    "surface_downward_mole_flux_of_carbon_dioxide": "cmo.balance",
    "mass_concentration_of_chlorophyll_a_in_sea_water": "cmo.algae",
    "turbidity": "cmo.turbid",
    "fluorescence": "cmo.algae",
    # keyed by full standard_name: the resolved name has no "par" substring, so a
    # short "par" key would never match re.search (unlike turbidity/fluorescence).
    "downwelling_photosynthetic_photon_flux_in_sea_water": "cmo.solar",
    # full standard_name: a bare "ph" key would also match phosphate, photon flux
    # (PAR) and phaeopigment, all of which contain "ph".
    "sea_water_ph_reported_on_total_scale": "cmo.speed_r",
    # kd490: shares turbidity's map -- both are water-clarity/optical measures.
    "diffuse_attenuation": "cmo.turbid",
    # Signed wind components: velocity components like the ocean ones above, so in
    # :data:`_CENTERED` and cmo.delta here (see the note on SSH/SLA above). Spelled out
    # by full name rather than the old bare "wind" key, which as a substring also
    # matched wind_speed -- a true speed, which keeps its own explicit cmo.speed entry
    # below. No other standard_name contains "wind".
    "eastward_wind": "cmo.delta",
    "northward_wind": "cmo.delta",
    "wind_speed": "cmo.speed",
    "sea_ice": "cmo.ice",
    # Every mixed-layer-thickness name: the generic ``ocean_mixed_layer_thickness``
    # and CF's ``..._defined_by_sigma_theta``/``_sigma_t``/``_temperature``/
    # ``_mixing_scheme``. ORDER MATTERS: ``cmaps_for`` takes the *first* key in this
    # table that ``re.search`` finds in the name, and a definition-specific MLD name
    # embeds the name of its criterion variable. ``..._defined_by_sigma_theta`` has
    # "sigma_theta" in it, so with that entry listed first it gets the density map
    # (cmo.dense) although it is a thickness in metres and belongs with the other
    # MLDs. This entry must therefore stay above every entry keyed by a criterion
    # variable -- today only "sigma_theta", but a "temperature" or "sigma_t" key added
    # later would shadow its MLD name just the same. xcmocean's own patterns need no
    # such care: ours are inserted ahead of them (see _register_colormaps), which is
    # also what stops its "temp" pattern claiming ``..._defined_by_temperature``.
    "mixed_layer": "cmo.deep",
    "sigma_theta": "cmo.dense",
    "conductivity": "cmo.haline",
    # full standard_name: a bare "pressure" key is fine too, but spelled out for
    # symmetry with the other full-standard_name entries above.
    "sea_water_pressure": "cmo.deep",
}

#: Display range/log-scale — concerns xcmocean has no notion of at all, so they stay
#: separate from the colormap table above. ``(vmin, vmax, log)``; any left as
#: ``None`` falls back to percentile-derived limits.
_RANGES: dict[str, tuple[float | None, float | None, bool]] = {
    "mass_concentration_of_chlorophyll_a_in_sea_water": (0.01, 10.0, True),
}

#: Variables recognized by an *exact* spelling rather than a substring. Every key in
#: :data:`_SEQUENTIAL_CMAPS` is escaped and matched with ``re.search``, which is what a
#: full CF standard_name wants and exactly what a one-letter name cannot have: ``"h"``
#: as a substring pattern would colour half the vocabulary. ``vartype -> (regex,
#: colormap)``, the regex fullmatched against the lower-cased name.
#:
#: ``"bathymetry"``: ROMS calls its seafloor depth ``h``, a grid constant that
#: :func:`ocean_skill.roms.standardize` keeps under that name (``to_depth`` reads it),
#: so ``field(src, "h")`` reaches this table with the string ``"h"`` and nothing to
#: resolve it by. Deliberately *not* a vocabulary alias: an alias would make
#: ``find_variable`` look for ``sea_floor_depth_below_geoid`` in a dataset that only
#: has ``h``. xcmocean's own ``depths`` type covers the word "bathymetry" but not the
#: CF ``sea_floor_depth_*`` names, which fall through to the default map.
_ANCHORED_CMAPS: dict[str, tuple[str, str]] = {
    "bathymetry": (r"^(h|bathymetry|sea_floor_depth(_below_\w+)?)$", "cmo.deep"),
}

#: The one source of truth for "zero is meaningful": a variable listed here is a signed
#: quantity (anomaly, flux, velocity component) whose sequential panel uses a
#: diverging map (:func:`cmaps_for`: its :data:`_SEQUENTIAL_CMAPS` entry, else
#: cmo.balance) over limits equal about the centre
#: (:func:`variable_limits`), so white is always "none". Its *spread* (std, variance)
#: is a non-negative magnitude and is not centred. The colour scale is made symmetric
#: about this value, so the diverging map's white sits at "no anomaly" whatever the data's own extremes are (percentile-or-min/max limits
#: of -0.1..0.4 would otherwise put white at +0.15, saying the wrong thing). Keyed by
#: full standard_name, looked up after :func:`ocean_skill.vocabulary.resolve_name`.
_CENTERED: dict[str, float] = {
    "sea_surface_height_above_sea_level": 0.0,
    "surface_downward_mole_flux_of_carbon_dioxide": 0.0,
    # signed velocity components; true speeds (wind_speed, ...) are magnitudes, absent
    "eastward_sea_water_velocity": 0.0,
    "northward_sea_water_velocity": 0.0,
    "sea_water_x_velocity": 0.0,
    "sea_water_y_velocity": 0.0,
    "upward_sea_water_velocity": 0.0,
    "eastward_wind": 0.0,
    "northward_wind": 0.0,
    # signed baroclinic pressure flux (energy flows either way); zero is "no flux"
    "x_baroclinic_pressure_flux": 0.0,
    "y_baroclinic_pressure_flux": 0.0,
    "eastward_baroclinic_pressure_flux": 0.0,
    "northward_baroclinic_pressure_flux": 0.0,
}

_registered = False


def _resolve_cmap(name: str):
    """Resolve a colormap name from a table: ``cmo.<name>`` or a matplotlib name.

    A ``cmo.`` prefix means cmocean (falling back to ``cmo.matter`` for a name it does
    not have, as before); anything else is looked up in matplotlib's registry, so a
    table entry can name a map cmocean lacks (``plasma``) without a second mechanism.
    A bad matplotlib name raises rather than silently becoming matter -- a typo in a
    table should be loud. cmocean maps are matplotlib Colormaps too, so every consumer
    (norm-based matplotlib draw, holoviews' ``cmap(float)``/``get_under`` palette
    sampling) treats the two kinds alike.
    """
    if name.startswith("cmo."):
        return _cmocean(name)
    import matplotlib

    return matplotlib.colormaps[name]


def _register_colormaps() -> None:
    """Insert :data:`_SEQUENTIAL_CMAPS` into xcmocean's own tables.

    Ahead of its built-ins, so a more specific pattern here is never shadowed by a
    broader pre-existing one (see the module docstring). Runs once (idempotent).
    """
    global _registered
    if _registered:
        return
    import xcmocean.options as xopts

    regexin = {name: re.escape(name) for name in _SEQUENTIAL_CMAPS}
    seqin = {name: _resolve_cmap(cmap) for name, cmap in _SEQUENTIAL_CMAPS.items()}
    # anchored spellings first: they fullmatch, so they can shadow nothing else
    anchored = {name: pattern for name, (pattern, _) in _ANCHORED_CMAPS.items()}
    regexin = anchored | regexin
    seqin |= {name: _resolve_cmap(cmap) for name, (_, cmap) in _ANCHORED_CMAPS.items()}
    # dict order is insertion order; rebuilding with ours first, then xcmocean's
    # existing table, makes ours the entries checked first without disturbing
    # anything already registered (including by a user's own xcmocean.set_options
    # call elsewhere) — must snapshot the original contents before clear().
    original = dict(xopts.REGEX)
    xopts.REGEX.clear()
    xopts.REGEX.update({**regexin, **original})
    xopts.SEQ.update(seqin)
    _registered = True


def cmaps_for(standard_name: str | None, statistic: str | None = None):
    """Return ``(sequential, diverging)`` colormaps for a variable's standard_name.

    Accepts anything :func:`ocean_skill.vocabulary.resolve_name` recognizes (a short
    vocabulary key or alias, not just the canonical standard_name) — resolved first so
    a lookup by short key still finds :data:`_SEQUENTIAL_CMAPS` entries keyed by the
    full standard_name (SSH, alkalinity, CO2 flux, chlorophyll above).

    Otherwise entirely xcmocean's own ``REGEX``/``SEQ``/``DIV`` tables — see the
    module docstring for why ocean-skill's BGC entries are inserted into them
    directly rather than kept as a second, separate lookup here. Falls back to
    xcmocean's own default (``viridis``/``balance``) if nothing matches.

    A centred variable (:data:`_CENTERED`: zero is meaningful) gets a diverging map as
    its sequential map: its own :data:`_SEQUENTIAL_CMAPS` entry (``cmo.delta`` for a
    velocity component), else ``cmo.balance``. :data:`_CENTERED` decides *whether*,
    the table only *which* -- no substring match can make a centred variable
    sequential.

    ``statistic`` is the field's ``attrs["statistic"]``. A spread (variance, std,
    range -- :func:`ocean_skill.units.is_spread`) is a non-negative magnitude whatever
    it is a spread *of*, so its sequential map is ``cmo.amp`` (the same map
    ``_METRIC_CMAPS["std_test"]`` gives a std panel) read from zero, rather than the
    variable's own map; the diverging map is unchanged. ``None`` changes nothing.
    """
    _register_colormaps()
    from xcmocean.options import DIV, REGEX, SEQ

    from ocean_skill.units import is_spread
    from ocean_skill.vocabulary import resolve_name

    name = resolve_name(standard_name or "").lower()
    # Decided here, before the regex table, so no substring match (xcmocean's "vel")
    # can override it: a zero-meaningful variable is diverging, a spread is amp.
    if is_spread(statistic):
        override = _cmocean("cmo.amp")
    elif center_for(standard_name) is not None:
        override = _resolve_cmap(_SEQUENTIAL_CMAPS.get(name, "cmo.balance"))
    else:
        override = None
    for vartype, pattern in REGEX.items():
        if re.search(pattern, name):
            return (override if override is not None else SEQ[vartype]), DIV[vartype]
    # No match: xcmocean's own defaultdict fallback (viridis / balance), called
    # directly rather than via SEQ[None]/DIV[None] so a bogus "None" key doesn't get
    # permanently inserted into its shared, module-global tables.
    seq = override if override is not None else SEQ.default_factory()
    return seq, DIV.default_factory()


def difference_cmap():
    """Return the diverging colormap used for the (test − reference) panel."""
    return cmaps_for(None)[1]


def _pinned(standard_name: str | None, statistic: str | None):
    """Return ``(resolved standard_name, its _RANGES entry)``, empty for a spread.

    A variance, standard deviation or range of a variable is not a value of that
    variable: the variance of chlorophyll sits orders of magnitude from the 0.01-10
    mg/m3 its own display range was chosen for, and it is not log-distributed the way a
    concentration is. So a field carrying a spread ``statistic``
    (:func:`ocean_skill.units.is_spread`, i.e. ``attrs["statistic"]``) gets no pinned
    range and no log scale -- limits come from the data, as for any unnamed variable.
    """
    from ocean_skill.units import is_spread
    from ocean_skill.vocabulary import resolve_name

    standard_name = resolve_name(standard_name or "")
    if is_spread(statistic):
        return standard_name, (None, None, False)
    return standard_name, _RANGES.get(standard_name, (None, None, False))


def center_for(standard_name: str | None, statistic: str | None = None) -> float | None:
    """Return the value a variable's sequential scale is centred on, or ``None``.

    Only signed anomalies (:data:`_CENTERED`: sea-level anomaly, air-sea CO2 flux)
    have one. A spread of such a variable (its std, say) is a non-negative magnitude,
    not a signed anomaly, so it has none -- same rule as :func:`_pinned`. Accepts any
    spelling :func:`ocean_skill.vocabulary.resolve_name` recognizes.
    """
    from ocean_skill.units import is_spread
    from ocean_skill.vocabulary import resolve_name

    if is_spread(statistic):
        return None
    return _CENTERED.get(resolve_name(standard_name or ""))


def is_log(standard_name: str | None, statistic: str | None = None) -> bool:
    """Return whether :data:`_RANGES` marks ``standard_name`` log-scale.

    Public so both :func:`norm_for` (matplotlib) and the holoviews renderer's
    ``logz=`` can ask the same question, rather than each reaching into the
    private ``_RANGES`` dict (or, worse, a since-removed ``VarInfo.log`` — this is
    the fix for exactly that regression). Accepts any spelling
    :func:`ocean_skill.vocabulary.resolve_name` recognizes, same as :func:`cmaps_for`.

    ``statistic`` is the field's ``attrs["statistic"]``, when it has one: a spread
    (variance, std, range) is never log-scale whatever it is a spread *of*, see
    :func:`_pinned`. Optional, and ``None`` changes nothing.
    """
    return _pinned(standard_name, statistic)[1][2]


def variable_limits(
    standard_name: str | None,
    vmin: float,
    vmax: float,
    *,
    user_vmin: float | None = None,
    user_vmax: float | None = None,
    statistic: str | None = None,
) -> tuple[float, float]:
    """Return the sequential ``(lo, hi)`` colour limits for a variable.

    The one place the limit policy lives, so :func:`norm_for` (matplotlib) and the
    holoviews renderer's ``clim`` cannot disagree. ``vmin``/``vmax`` are the data-
    derived (snapped) limits, shared across whatever panels share the scale; this
    layers on, in order:

    1. the variable's declared :data:`_RANGES` pin (none for a spread, see
       :func:`_pinned`);
    2. a **spread** statistic (std, variance, range) reads from 0 -- it is a
       non-negative magnitude, and ``cmo.amp``'s white end must mean "no spread";
    3. a **centred** variable (:func:`center_for`) is made symmetric about its centre,
       half-width ``max(|lo-c|, |hi-c|)`` snapped to a round value the way
       :func:`metric_colors` snaps a centred metric's, so the diverging map's white sits
       at "no anomaly". Limits of a centred variable are *always* equal about the
       centre: a single user end is **mirrored** (``user_vmin=-0.2`` gives
       ``(-0.2, 0.2)``); both user ends are honoured exactly as given;
    4. otherwise ``user_vmin``/``user_vmax`` -- a caller's own limits -- always win,
       each moving only its own end.

    A log variable is returned as pinned/derived; flooring it above zero is the
    caller's business (LogNorm rejects ``vmin <= 0``).
    """
    from ocean_skill.plot._colorbar import round_limits
    from ocean_skill.units import is_spread

    _, (r_vmin, r_vmax, _log) = _pinned(standard_name, statistic)
    lo = r_vmin if r_vmin is not None else vmin
    hi = r_vmax if r_vmax is not None else vmax
    if is_spread(statistic):
        lo = 0.0
    else:
        center = center_for(standard_name)
        if center is not None:
            if user_vmin is not None and user_vmax is not None:
                return user_vmin, user_vmax  # both given: exactly as asked
            if user_vmin is not None or user_vmax is not None:
                # one end given: mirror it, the user's number is not rounded
                half = abs((user_vmin if user_vmin is not None else user_vmax) - center)
            else:
                half = max(abs(lo - center), abs(hi - center))
                if half > 0:
                    half = round_limits(-half, half, log=False)[1]
            if half > 0:  # all-at-centre data keeps its own (degenerate) limits
                return center - half, center + half
    return (
        user_vmin if user_vmin is not None else lo,
        user_vmax if user_vmax is not None else hi,
    )


def norm_for(
    standard_name: str | None,
    vmin: float,
    vmax: float,
    *,
    user_vmin: float | None = None,
    user_vmax: float | None = None,
    statistic: str | None = None,
) -> Any:
    """Return a matplotlib ``Normalize`` for a variable's sequential panels.

    Uses :class:`~matplotlib.colors.LogNorm` when :data:`_RANGES` marks
    ``standard_name`` log-scale (chlorophyll — linear color across 0.01-10 mg/m3 hides
    everything but the brightest blooms), and its own ``vmin``/``vmax`` in place of
    the percentile-derived ones when it declares them. Purely a range/scale concern —
    xcmocean has no equivalent, so nothing here duplicates it. Accepts any spelling
    :func:`ocean_skill.vocabulary.resolve_name` recognizes, same as :func:`cmaps_for`.

    ``user_vmin``/``user_vmax`` are a caller's own explicit limits (``Field.plot(vmin=,
    vmax=)``) and outrank everything else, including a variable's declared
    :data:`_RANGES` — a user who names a number gets that number, not the package's
    default display range. The scale itself (log vs linear) is unaffected; only its
    limits move.

    ``statistic`` (the field's ``attrs["statistic"]``) switches :data:`_RANGES` off for
    a spread -- variance, std, range -- so the variance of chlorophyll is drawn on a
    linear scale between its own ``vmin``/``vmax``, not squeezed into 0.01-10 on a log
    one. A caller's ``user_vmin``/``user_vmax`` still win.
    """
    import matplotlib.colors as mcolors

    log = is_log(standard_name, statistic)
    lo, hi = variable_limits(
        standard_name,
        vmin,
        vmax,
        user_vmin=user_vmin,
        user_vmax=user_vmax,
        statistic=statistic,
    )
    if log:
        lo = max(lo, 1e-6)  # LogNorm rejects vmin <= 0
        return mcolors.LogNorm(vmin=lo, vmax=hi)
    return mcolors.Normalize(vmin=lo, vmax=hi)


# --------------------------------------------------------------------------- metrics

#: Metric colors — edit here, nothing else needs to change. The counterpart of
#: :data:`_SEQUENTIAL_CMAPS` for metrics, and read the same way: values are cmocean
#: colormap names (``cmo.<name>``), or matplotlib ones, via :func:`_resolve_cmap`.
#:
#: A metric absent from this table takes its colormap from the **variable** being scored
#: instead: ``bias`` uses the variable's own *diverging* map, so a bias panel and the
#: ``test − reference`` panel of a comparison are the same colours for one quantity, and
#: ``mean_test``/``mean_reference`` use its *sequential* map, being the field itself.
_METRIC_CMAPS: dict[str, str] = {
    # error magnitudes: one sequential map, so a figure's rmse and mae panels read alike
    "rmse": "cmo.amp",
    "mae": "cmo.amp",
    "crmsd": "cmo.amp",
    "std_test": "cmo.amp",
    "std_reference": "cmo.amp",
    "corr": "cmo.balance",
    "sigma_ratio": "cmo.tarn",
    "n": "cmo.gray_r",
}

#: Display range per metric — ``(vmin, vmax, center)``, the counterpart of
#: :data:`_RANGES` for metrics. ``None`` means "derive it from the data". A ``center``
#: makes the scale diverging and *symmetric about that value*, which is the whole point
#: for a signed metric: white must sit at no error (0), or at equal variability (1), or
#: the colours say the wrong thing.
#:
#: Where both ``vmin`` and ``vmax`` are given the limits are **fixed** and the data is
#: ignored. Correlation is the case that matters: it has an absolute scale, and
#: percentile limits would paint 0.9 the same shade as 1.0 in a figure where the
#: difference is the finding.
#:
#: A ``vmin`` beside a ``center`` is a *floor*, not a limit: it bounds how far the
#: symmetric spread may reach so the low end cannot become negative (a variability ratio
#: of −0.4 is not a thing).
_METRIC_RANGES: dict[str, tuple[float | None, float | None, float | None]] = {
    "bias": (None, None, 0.0),
    "corr": (-1.0, 1.0, 0.0),
    "sigma_ratio": (0.0, None, 1.0),
    # magnitudes: zero is pinned so that "no error" is the same colour in every figure
    "rmse": (0.0, None, None),
    "mae": (0.0, None, None),
    "crmsd": (0.0, None, None),
    "std_test": (0.0, None, None),
    "std_reference": (0.0, None, None),
    "n": (0.0, None, None),
}

#: Metrics whose upper limit is the exact maximum rather than a robust percentile. A
#: count has a real maximum worth showing; an error magnitude has a noisy tail that
#: would flatten every other cell.
_METRIC_EXACT_MAX = ("n",)

#: Metrics that must share one colour scale when drawn in the same figure: the two
#: members of each pair are the same physical quantity for the two fields, and
#: per-panel scaling would make the one comparison the panels exist to invite —
#: "is the model more variable than the observations?" — impossible to read.
METRIC_LIMIT_GROUPS: tuple[tuple[str, ...], ...] = (
    ("mean_test", "mean_reference"),
    ("std_test", "std_reference"),
)

#: Metrics coloured as the *variable* rather than as a score, so they alone may consult
#: :func:`norm_for`/:func:`is_log`. Everything else must not: ``_RANGES`` pins
#: chlorophyll to a ``LogNorm`` over 0.01-10 mg/m3, which is the field's display range
#: and is meaningless for a signed bias (negatives) or an rmse (wrong quantity).
_VARIABLE_LIKE_METRICS = ("mean_test", "mean_reference")

#: Half-width used when a diverging metric's data gives nothing to measure (all-NaN, or
#: exactly at the centre everywhere). Arbitrary, but a degenerate zero-width scale draws
#: a uniformly white panel that looks like missing data rather than perfect agreement.
_DEGENERATE_SPREAD = 1.0


@dataclass(frozen=True)
class MetricColors:
    """How to colour one metric panel: a colormap and the limits to stretch it over.

    Returned rather than applied so both renderers can ask the same question and get the
    same answer in their own vocabulary — :meth:`norm` for matplotlib, :meth:`clim` for
    bokeh. A matplotlib-only norm (``TwoSlopeNorm`` for a ratio, say) is deliberately
    never produced: bokeh's ``clim`` is a linear pair with no equivalent, so one would
    make the two renderers draw different pictures from one spec.
    """

    cmap: Any
    vmin: float
    vmax: float
    log: bool = False
    #: The finite extremes of the values the limits came from, if known. The limits are
    #: a percentile, so these are how a renderer tells whether data lies beyond an end
    #: of the bar and should say so.
    data_min: float | None = None
    data_max: float | None = None

    def clim(self) -> tuple[float, float]:
        """``(vmin, vmax)`` for hvplot/bokeh."""
        return (self.vmin, self.vmax)

    def data_range(self) -> tuple[float, float] | None:
        """``(data_min, data_max)``, or ``None`` when the data was not measured."""
        if self.data_min is None or self.data_max is None:
            return None
        return (self.data_min, self.data_max)

    def covering(self, values) -> MetricColors:
        """The same scale, told about more data drawn on it: only the extremes grow.

        The limits stay where they were fitted -- a skill map's station dots are painted
        on the scale its interpolated surface earned, and letting them move it would
        change the surface's colours -- but a dot past an end of the bar is clipped just
        as a cell is, so the bar's arrow has to know about it.
        """
        finite = _finite(values)
        if not finite.size:
            return self
        lo, hi = float(finite.min()), float(finite.max())
        if self.data_min is not None and self.data_max is not None:
            lo, hi = min(lo, self.data_min), max(hi, self.data_max)
        return replace(self, data_min=lo, data_max=hi)

    def norm(self):
        """Return a matplotlib ``Normalize`` (or ``LogNorm``) over the same limits.

        Carries :meth:`data_range` on the norm as ``_osk_data_range``, which is what the
        static renderer's colourbar reads to decide where to draw its extension arrows.
        """
        import matplotlib.colors as mcolors

        if self.log:
            norm = mcolors.LogNorm(vmin=max(self.vmin, 1e-6), vmax=self.vmax)
        else:
            norm = mcolors.Normalize(vmin=self.vmin, vmax=self.vmax)
        norm._osk_data_range = self.data_range()
        return norm


def _cmocean(name: str):
    """Resolve a ``cmo.<name>`` string to the colormap, falling back to cmo.matter."""
    import cmocean

    return getattr(cmocean.cm, name.removeprefix("cmo."), cmocean.cm.matter)


def _finite(values) -> Any:
    """Return the finite values of ``values`` as a flat array (possibly empty)."""
    import numpy as np

    if values is None:
        return np.empty(0)
    arr = np.asarray(values, dtype="float64").ravel()
    return arr[np.isfinite(arr)]


def metric_colors(metric: str, values=None, *, standard_name: str | None = None):
    """Return the :class:`MetricColors` for one metric panel.

    The single place either renderer decides what a metric map looks like, so a skill
    figure drawn statically and the same figure drawn interactively cannot disagree on
    either the colours or the range. ``values`` is the data the limits come from — pass
    the pooled values of a :data:`METRIC_LIMIT_GROUPS` pair to give both members one
    scale. ``standard_name`` is the *compared variable's* name, needed for the metrics
    coloured as the variable rather than as a score.

    An unregistered metric is not an error: its limits follow the sign of its own data,
    diverging about zero if the values straddle it and sequential from zero if they do
    not. So a metric added by :func:`ocean_skill.metrics.register` draws sensibly before
    anyone gets round to giving it a row in :data:`_METRIC_RANGES`.
    """
    import numpy as np

    from ocean_skill.plot._colorbar import round_limits

    seq_cmap, div_cmap = cmaps_for(standard_name)
    finite = _finite(values)
    # Only the variable-like metrics can be log; a log bar cannot show a value <= 0, so
    # such a value is not what an end arrow would point at.
    log = metric in _VARIABLE_LIKE_METRICS and is_log(standard_name)
    seen = finite[finite > 0] if log else finite
    extremes = (
        {"data_min": float(seen.min()), "data_max": float(seen.max())}
        if seen.size
        else {}
    )

    if metric in _VARIABLE_LIKE_METRICS:
        # the field itself: the variable's own colours, range and log-ness
        lo = float(np.percentile(finite, 10)) if finite.size else 0.0
        hi = float(np.percentile(finite, 90)) if finite.size else 1.0
        # snapped outward to round values like every automatic colour range (see
        # ocean_skill.plot._colorbar); a variable's declared range still wins below
        lo, hi = round_limits(lo, hi, log=log)
        norm = norm_for(standard_name, lo, hi)
        return MetricColors(
            cmap=seq_cmap,
            vmin=float(norm.vmin),
            vmax=float(norm.vmax),
            log=is_log(standard_name),
            **extremes,
        )

    vmin, vmax, center = _METRIC_RANGES.get(metric, (None, None, None))
    if metric not in _METRIC_RANGES and finite.size:
        # unregistered: let the data say whether it is signed
        center = 0.0 if (finite.min() < 0 < finite.max()) else None
        vmin = None if center is not None else 0.0
    cmap = (
        _resolve_cmap(_METRIC_CMAPS[metric])
        if metric in _METRIC_CMAPS
        else (div_cmap if center is not None else seq_cmap)
    )

    if center is not None:
        if vmin is not None and vmax is not None:
            return MetricColors(
                cmap=cmap, vmin=float(vmin), vmax=float(vmax), **extremes
            )
        spread = (
            float(np.percentile(np.abs(finite - center), 98)) if finite.size else 0.0
        )
        if not np.isfinite(spread) or spread <= 0:
            spread = _DEGENERATE_SPREAD
        else:  # a round half-range: ±1.734 reads ±1.8
            spread = round_limits(-spread, spread, log=False)[1]
        if vmin is not None:  # a floor: keep the symmetric low end above it
            spread = min(spread, center - float(vmin))
        return MetricColors(
            cmap=cmap, vmin=center - spread, vmax=center + spread, **extremes
        )

    lo = (
        float(vmin)
        if vmin is not None
        else (float(finite.min()) if finite.size else 0.0)
    )
    if vmax is not None:
        hi = float(vmax)
    elif not finite.size:
        hi = lo + 1.0
    elif metric in _METRIC_EXACT_MAX:
        hi = float(finite.max())
    else:
        hi = float(np.percentile(finite, 98))
    if hi <= lo:  # a constant field: give the bar somewhere to go
        hi = lo + 1.0
    # round the data-derived ends outward; a registered end, and a count's true
    # maximum (_METRIC_EXACT_MAX), stay exactly as they are
    lo, hi = round_limits(
        lo,
        hi,
        log=False,
        keep_lo=vmin is not None,
        keep_hi=vmax is not None or metric in _METRIC_EXACT_MAX,
    )
    return MetricColors(cmap=cmap, vmin=lo, vmax=hi, **extremes)
