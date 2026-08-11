"""Colormap resolution: xcmocean's own tables, extended in place for BGC species.

The single place both renderers get their colours from, so there is one policy to
maintain and a plot looks the same whichever backend draws it (see
:mod:`ocean_skill.plot.matplotlib_renderer` and
:mod:`ocean_skill.plot.holoviews_renderer`).

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
from typing import Any

__all__ = ["cmaps_for", "difference_cmap", "is_log", "norm_for"]

#: BGC species colors — edit here, nothing else needs to change. Values are cmocean
#: colormap names (``cmo.<name>``). Reused as both the xcmocean "vartype" key and its
#: own (escaped, exact) match pattern.
_SEQUENTIAL_CMAPS: dict[str, str] = {
    # xcmocean's own default for this vartype ("zeta") is a sequential cmo.amp; SSH is
    # signed, not a magnitude, so we use a diverging-look map for its sequential panel
    # too -- a deliberate override, not an oversight.
    "sea_surface_height_above_geoid": "cmo.balance",
    "nitrate": "cmo.turbid",
    "phosphate": "cmo.speed",
    "silicate": "cmo.tempo",
    "oxygen": "cmo.gray_r",
    # "mole_concentration_of_dissolved_molecular_oxygen_in_sea_water": "cmo.oxy",
    "dissolved_inorganic_carbon": "cmo.ice_r",
    "sea_water_alkalinity_expressed_as_mole_equivalent": "cmo.matter",
    "surface_downward_mole_flux_of_carbon_dioxide": "cmo.balance",
    "mass_concentration_of_chlorophyll_a_in_sea_water": "cmo.algae",
}

#: Display range/log-scale — concerns xcmocean has no notion of at all, so they stay
#: separate from the colormap table above. ``(vmin, vmax, log)``; any left as
#: ``None`` falls back to percentile-derived limits.
_RANGES: dict[str, tuple[float | None, float | None, bool]] = {
    "mass_concentration_of_chlorophyll_a_in_sea_water": (0.01, 10.0, True),
}

_registered = False


def _register_colormaps() -> None:
    """Insert :data:`_SEQUENTIAL_CMAPS` into xcmocean's own tables.

    Ahead of its built-ins, so a more specific pattern here is never shadowed by a
    broader pre-existing one (see the module docstring). Runs once (idempotent).
    """
    global _registered
    if _registered:
        return
    import cmocean
    import xcmocean.options as xopts

    regexin = {name: re.escape(name) for name in _SEQUENTIAL_CMAPS}
    seqin = {
        name: getattr(cmocean.cm, cmap.removeprefix("cmo."), cmocean.cm.matter)
        for name, cmap in _SEQUENTIAL_CMAPS.items()
    }
    # dict order is insertion order; rebuilding with ours first, then xcmocean's
    # existing table, makes ours the entries checked first without disturbing
    # anything already registered (including by a user's own xcmocean.set_options
    # call elsewhere) — must snapshot the original contents before clear().
    original = dict(xopts.REGEX)
    xopts.REGEX.clear()
    xopts.REGEX.update({**regexin, **original})
    xopts.SEQ.update(seqin)
    _registered = True


def cmaps_for(standard_name: str | None):
    """Return ``(sequential, diverging)`` colormaps for a variable's standard_name.

    Accepts anything :func:`ocean_skill.vocabulary.resolve_name` recognizes (a short
    vocabulary key or alias, not just the canonical standard_name) — resolved first so
    a lookup by short key still finds :data:`_SEQUENTIAL_CMAPS` entries keyed by the
    full standard_name (SSH, alkalinity, CO2 flux, chlorophyll above).

    Otherwise entirely xcmocean's own ``REGEX``/``SEQ``/``DIV`` tables — see the
    module docstring for why ocean-skill's BGC entries are inserted into them
    directly rather than kept as a second, separate lookup here. Falls back to
    xcmocean's own default (``viridis``/``balance``) if nothing matches.
    """
    _register_colormaps()
    from xcmocean.options import DIV, REGEX, SEQ

    from ocean_skill.vocabulary import resolve_name

    name = resolve_name(standard_name or "").lower()
    for vartype, pattern in REGEX.items():
        if re.search(pattern, name):
            return SEQ[vartype], DIV[vartype]
    # No match: xcmocean's own defaultdict fallback (viridis / balance), called
    # directly rather than via SEQ[None]/DIV[None] so a bogus "None" key doesn't get
    # permanently inserted into its shared, module-global tables.
    return SEQ.default_factory(), DIV.default_factory()


def difference_cmap():
    """Return the diverging colormap used for the (test − reference) panel."""
    return cmaps_for(None)[1]


def is_log(standard_name: str | None) -> bool:
    """Return whether :data:`_RANGES` marks ``standard_name`` log-scale.

    Public so both :func:`norm_for` (matplotlib) and the holoviews renderer's
    ``logz=`` can ask the same question, rather than each reaching into the
    private ``_RANGES`` dict (or, worse, a since-removed ``VarInfo.log`` — this is
    the fix for exactly that regression). Accepts any spelling
    :func:`ocean_skill.vocabulary.resolve_name` recognizes, same as :func:`cmaps_for`.
    """
    from ocean_skill.vocabulary import resolve_name

    return _RANGES.get(resolve_name(standard_name or ""), (None, None, False))[2]


def norm_for(standard_name: str | None, vmin: float, vmax: float) -> Any:
    """Return a matplotlib ``Normalize`` for a variable's sequential panels.

    Uses :class:`~matplotlib.colors.LogNorm` when :data:`_RANGES` marks
    ``standard_name`` log-scale (chlorophyll — linear color across 0.01-10 mg/m3 hides
    everything but the brightest blooms), and its own ``vmin``/``vmax`` in place of
    the percentile-derived ones when it declares them. Purely a range/scale concern —
    xcmocean has no equivalent, so nothing here duplicates it. Accepts any spelling
    :func:`ocean_skill.vocabulary.resolve_name` recognizes, same as :func:`cmaps_for`.
    """
    import matplotlib.colors as mcolors

    from ocean_skill.vocabulary import resolve_name

    standard_name = resolve_name(standard_name or "")
    r_vmin, r_vmax, _ = _RANGES.get(standard_name, (None, None, False))
    lo = r_vmin if r_vmin is not None else vmin
    hi = r_vmax if r_vmax is not None else vmax
    if is_log(standard_name):
        lo = max(lo, 1e-6)  # LogNorm rejects vmin <= 0
        return mcolors.LogNorm(vmin=lo, vmax=hi)
    return mcolors.Normalize(vmin=lo, vmax=hi)
