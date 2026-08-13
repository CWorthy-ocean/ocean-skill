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
from dataclasses import dataclass
from typing import Any

__all__ = [
    "METRIC_LIMIT_GROUPS",
    "MetricColors",
    "cmaps_for",
    "difference_cmap",
    "is_log",
    "metric_colors",
    "norm_for",
]

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


# --------------------------------------------------------------------------- metrics

#: Metric colors — edit here, nothing else needs to change. The counterpart of
#: :data:`_SEQUENTIAL_CMAPS` for metrics, and read the same way: values are cmocean
#: colormap names (``cmo.<name>``).
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

    def clim(self) -> tuple[float, float]:
        """``(vmin, vmax)`` for hvplot/bokeh."""
        return (self.vmin, self.vmax)

    def norm(self):
        """Return a matplotlib ``Normalize`` (or ``LogNorm``) over the same limits."""
        import matplotlib.colors as mcolors

        if self.log:
            return mcolors.LogNorm(vmin=max(self.vmin, 1e-6), vmax=self.vmax)
        return mcolors.Normalize(vmin=self.vmin, vmax=self.vmax)


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

    seq_cmap, div_cmap = cmaps_for(standard_name)
    finite = _finite(values)

    if metric in _VARIABLE_LIKE_METRICS:
        # the field itself: the variable's own colours, range and log-ness
        lo = float(np.percentile(finite, 10)) if finite.size else 0.0
        hi = float(np.percentile(finite, 90)) if finite.size else 1.0
        norm = norm_for(standard_name, lo, hi)
        return MetricColors(
            cmap=seq_cmap,
            vmin=float(norm.vmin),
            vmax=float(norm.vmax),
            log=is_log(standard_name),
        )

    vmin, vmax, center = _METRIC_RANGES.get(metric, (None, None, None))
    if metric not in _METRIC_RANGES and finite.size:
        # unregistered: let the data say whether it is signed
        center = 0.0 if (finite.min() < 0 < finite.max()) else None
        vmin = None if center is not None else 0.0
    cmap = (
        _cmocean(_METRIC_CMAPS[metric])
        if metric in _METRIC_CMAPS
        else (div_cmap if center is not None else seq_cmap)
    )

    if center is not None:
        if vmin is not None and vmax is not None:
            return MetricColors(cmap=cmap, vmin=float(vmin), vmax=float(vmax))
        spread = (
            float(np.percentile(np.abs(finite - center), 98)) if finite.size else 0.0
        )
        if not np.isfinite(spread) or spread <= 0:
            spread = _DEGENERATE_SPREAD
        if vmin is not None:  # a floor: keep the symmetric low end above it
            spread = min(spread, center - float(vmin))
        return MetricColors(cmap=cmap, vmin=center - spread, vmax=center + spread)

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
    return MetricColors(cmap=cmap, vmin=lo, vmax=hi)
