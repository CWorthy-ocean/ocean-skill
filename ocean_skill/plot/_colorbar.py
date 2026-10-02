"""Round colour-bar ticks and limits, decided once for both renderers.

A colour bar is read by its tick labels, so they should be numbers a reader can take in
at a glance -- ``0.0, 0.5, 1.0`` rather than the ``0.466, 0.911`` a filled-contour bar
gets when its ticks land on band edges, or the ``9.98×10⁻²`` of a log bar. This module
decides three things, and both renderers ask it rather than deciding themselves, so the
static and interactive bars always read the same:

* :func:`colorbar_ticks` -- which values a bar ticks and how each one is spelled;
* :func:`round_limits` -- where an *automatically chosen* colour range ends: snapped
  outward to a round value one step finer than the ticks (2.987 → 3.0), so a bar ends
  on a number too and a clipped end reads ``≥ 3.0`` rather than ``≥ 2.987``;
* :func:`difference_limit` -- a difference panel's symmetric half-range, the 98th
  percentile of ``|difference|`` snapped up the same way.

Limits a caller pinned (``vmin=``/``vmax=``, or a variable's declared display range in
:func:`ocean_skill.colormaps.norm_for`) are never moved: a user who names a number gets
that number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["Ticks", "colorbar_ticks", "difference_limit", "round_limits", "tick_step"]

#: Tick steps a colour bar may use, as multiples of a power of ten. matplotlib's default
#: list also has 2.5, left out here because it puts two decimals on every other label
#: (0.25, 0.75) -- exactly what this module exists to avoid.
_STEPS = (1, 2, 5, 10)

#: The most intervals a bar is split into (one less than its most ticks). Eight keeps a
#: symmetric difference bar such as ±1.8 at seven ticks (−1.5 … 1.5) rather than three.
_NBINS = 8

#: How much finer than the tick step an automatic limit is rounded to, at most: 2.987
#: on a bar ticked every 0.5 becomes 3.0, not 3.5 -- round, without widening the range
#: by more than a fifth of a tick interval. The snap itself is the largest 1, 2 or 5
#: times a power of ten within that fifth (see :func:`_snap_step`), so a bar ticked
#: every 2 snaps to 0.2, never to 0.4 -- 24.87 becomes 25.0, not 25.2.
_SNAP_DIVISOR = 5

#: Beyond this many decimals a label switches to scientific notation: 0.0000005 is not
#: easier to read than 5e−07.
_MAX_DECIMALS = 6

#: A log bar spanning at least this many decades ticks only the decades themselves; a
#: narrower one ticks 1, 2 and 5 times each power of ten (or 1 and 5, if that would be
#: more ticks than a linear bar may carry).
_DECADE_SPAN = 3.0

#: A log bar never carries more decade ticks than this -- a wider span ticks every
#: other decade (or every third, ...) instead.
_MAX_DECADE_TICKS = 7

_MINUS = "\N{MINUS SIGN}"


def _finite_range(vmin, vmax) -> bool:
    return (
        vmin is not None
        and vmax is not None
        and math.isfinite(vmin)
        and math.isfinite(vmax)
        and vmin < vmax
    )


def tick_step(vmin: float, vmax: float) -> float:
    """Return the linear tick spacing a bar from ``vmin`` to ``vmax`` uses.

    A round step -- 1, 2 or 5 times a power of ten -- giving at most
    :data:`_NBINS` intervals across the range, matplotlib's ``MaxNLocator`` rule with
    this module's own step list. Also what :func:`round_limits` snaps to a fifth of,
    and what filled-contour band edges are a fraction of, so ticks always sit on a
    band edge.
    """
    from matplotlib.ticker import MaxNLocator

    values = MaxNLocator(nbins=_NBINS, steps=list(_STEPS)).tick_values(vmin, vmax)
    return float(values[1] - values[0]) if len(values) > 1 else 1.0


def _decimals(step: float) -> int:
    """How many decimals a multiple of ``step`` needs: 0.5 → 1, 0.05 → 2, 5 → 0."""
    return max(0, -math.floor(math.log10(step) + 1e-9))


def _clean(value: float, step: float) -> float:
    """Return ``value`` without the float noise of ``k * step`` (3.0000000000000004)."""
    return float(round(value, _decimals(step) + 2))


def _plain(value: float) -> str:
    """Spell a log tick as a plain decimal -- ``0.01``, ``0.5``, ``10``."""
    if value == 0:
        return "0"
    if 1e-4 <= abs(value) < 1e6:
        return np.format_float_positional(value, precision=6, unique=True, trim="-")
    return f"{value:.2g}"


def _fixed(value: float, decimals: int) -> str:
    """``value`` with exactly ``decimals`` decimals, never ``-0.0``."""
    if decimals > _MAX_DECIMALS:
        return f"{value:.2g}"
    text = f"{value:.{decimals}f}"
    if float(text) == 0:
        text = f"{0:.{decimals}f}"
    return text


@dataclass(frozen=True)
class Ticks:
    """A colour bar's ticks: where they sit and how each one reads.

    ``labels`` already carry a Unicode minus, so a renderer sets them verbatim.
    ``decimals`` is how many a linear bar's labels share (``None`` on a log bar, whose
    labels are plain decimals each spelled for itself). :meth:`text` spells any other
    value on this bar the same way -- a forced end label such as ``≥ 3.0``.
    """

    values: tuple[float, ...]
    labels: tuple[str, ...]
    decimals: int | None = None

    def text(self, value: float) -> str:
        """Spell ``value`` the way this bar's own labels are spelled.

        A value the bar's decimals hold exactly reads like a tick (``3.0`` on a bar
        ticked ``2.0, 2.5``). One that needs more -- an automatic limit is snapped to a
        fifth of the tick step, so 4.2 can end a bar ticked every 1 -- keeps the extra
        digit rather than being rounded into a number the bar does not end at.
        """
        value = float(value)
        if self.decimals is None:
            text = _plain(value)
        else:
            decimals = self.decimals
            tol = 1e-9 * max(1.0, abs(value))
            while decimals < _MAX_DECIMALS and not math.isclose(
                round(value, decimals), value, rel_tol=0.0, abs_tol=tol
            ):
                decimals += 1
            text = _fixed(value, decimals)
        return text.replace("-", _MINUS)


def _linear_ticks(vmin: float, vmax: float) -> Ticks:
    step = tick_step(vmin, vmax)
    tol = step * 1e-9
    first = math.ceil((vmin - tol) / step)
    last = math.floor((vmax + tol) / step)
    values = tuple(_clean(k * step, step) for k in range(first, last + 1))
    decimals = _decimals(step)
    labels = tuple(_fixed(v, decimals).replace("-", _MINUS) for v in values)
    return Ticks(values, labels, decimals)


def _log_ticks(vmin: float, vmax: float) -> Ticks:
    lo_exp, hi_exp = math.log10(vmin), math.log10(vmax)
    tol = 1e-9
    if hi_exp - lo_exp >= _DECADE_SPAN:
        exps = list(range(math.ceil(lo_exp - tol), math.floor(hi_exp + tol) + 1))
        stride = max(1, math.ceil(len(exps) / _MAX_DECADE_TICKS))
        values = [float(f"1e{e}") for e in exps[::stride]]
    else:
        for mantissas in ((1, 2, 5), (1, 5)):
            values = [
                float(f"{m}e{e}")
                for e in range(math.floor(lo_exp) - 1, math.ceil(hi_exp) + 1)
                for m in mantissas
                if vmin * (1 - tol) <= float(f"{m}e{e}") <= vmax * (1 + tol)
            ]
            if len(values) <= _NBINS + 1:
                break
        if len(values) < 2:
            # Narrower than about half a decade: 1-2-5 leaves at most one tick, so
            # tick it as the linear bar it nearly is (still on a log scale).
            linear = _linear_ticks(vmin, vmax)
            values = [v for v in linear.values if v > 0]
    values = sorted(values)
    return Ticks(tuple(values), tuple(_plain(v).replace("-", _MINUS) for v in values))


def colorbar_ticks(vmin: float, vmax: float, *, log: bool) -> Ticks:
    """Return the round ticks a colour bar from ``vmin`` to ``vmax`` carries.

    **Linear:** multiples of :func:`tick_step` inside the range -- 1, 2 or 5 times a
    power of ten, at most nine of them -- all labelled with the decimals that step
    needs (``0.0, 0.5, 1.0``; ``34, 35, 36``), with a Unicode minus and never an
    offset (``+3.4e1``).

    **Log:** the decades inside the range when it spans at least three (``0.01, 0.1,
    1, 10``), otherwise 1, 2 and 5 times each power of ten (``0.05, 0.1, 0.2, 0.5``),
    each spelled as a plain decimal rather than ``10⁻²``.

    Empty for a range with nothing to tick (non-finite, or ``vmin >= vmax``) -- a
    renderer then leaves the bar's own ticks alone.
    """
    if not _finite_range(vmin, vmax):
        return Ticks((), ())
    vmin, vmax = float(vmin), float(vmax)
    if log and vmin > 0:
        return _log_ticks(vmin, vmax)
    return _linear_ticks(vmin, vmax)


def _snap_step(step: float) -> float:
    """Return the round spacing a linear limit snaps to: within a fifth of ``step``.

    The largest 1, 2 or 5 times a power of ten no bigger than ``step /``
    :data:`_SNAP_DIVISOR` -- 0.1 for a tick step of 0.5, 0.2 for 1 *and* for 2, 1 for
    5 -- so a snapped end is itself a round number, whatever the tick step.
    """
    target = step / _SNAP_DIVISOR
    exp = math.floor(math.log10(target) + 1e-9)
    mantissa = target / 10.0**exp
    digit = max(m for m in (1, 2, 5) if m <= mantissa + 1e-9)
    return float(f"{digit}e{exp}")


def _snap_log(value: float, *, up: bool) -> float:
    """``value`` rounded outward to one significant digit: 0.0213 → 0.02, 2.987 → 3."""
    exp = math.floor(math.log10(value))
    mantissa = value / 10.0**exp
    digit = math.ceil(mantissa - 1e-9) if up else math.floor(mantissa + 1e-9)
    return float(f"{max(digit, 1)}e{exp}")


def round_limits(
    lo: float,
    hi: float,
    *,
    log: bool,
    keep_lo: bool = False,
    keep_hi: bool = False,
) -> tuple[float, float]:
    """Return ``(lo, hi)`` snapped outward to round values.

    For an automatically chosen colour range -- the data's own min/max, or the
    percentiles ``robust=`` picks -- so a bar ends on a number (0.0213-2.987 → 0-3)
    and every label on it reads round. Outward only: nothing the range covered is
    clipped by the snap.

    **Linear:** each end moves to the next multiple of a round step within a fifth of
    :func:`tick_step` (2.987 → 3.0 on a bar ticked every 0.5; 33.81 → 33.8; 24.87 →
    25.0 on one ticked every 2). **Log:** each end moves to one significant digit
    (0.0213 → 0.02, 2.987 → 3), never to zero or below.

    ``keep_lo``/``keep_hi`` leave that end exactly as given -- an end a caller pinned
    is theirs, not this function's. A range with nothing to snap (non-finite, or
    ``lo >= hi``) comes back unchanged.
    """
    if not _finite_range(lo, hi):
        return lo, hi
    lo, hi = float(lo), float(hi)
    if log:
        if lo <= 0:
            # no log value to round the bottom to (a renderer floors it itself); the
            # top still reads round
            return lo, hi if keep_hi else _snap_log(hi, up=True)
        return (
            lo if keep_lo else _snap_log(lo, up=False),
            hi if keep_hi else _snap_log(hi, up=True),
        )
    fine = _snap_step(tick_step(lo, hi))
    new_lo = lo if keep_lo else _clean(math.floor(lo / fine + 1e-9) * fine, fine)
    new_hi = hi if keep_hi else _clean(math.ceil(hi / fine - 1e-9) * fine, fine)
    return new_lo, new_hi


def difference_limit(difference) -> float:
    """Return a difference panel's symmetric half-range, snapped up to a round value.

    The 98th percentile of ``|difference|`` -- wide enough for nearly every cell,
    narrow enough that one outlier does not wash the rest of the panel out -- then
    snapped up as :func:`round_limits` snaps the top of ``(-d, d)``, so the bar reads
    ``−1.8 … 1.8`` rather than ``−1.734 … 1.734``. ``1.0`` for a difference with no
    finite values or one that is zero everywhere, so the diverging norm still has a
    range to span.
    """
    values = np.abs(np.asarray(difference, dtype="float64")).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1.0
    dmax = float(np.percentile(values, 98))
    if not dmax > 0:
        return 1.0
    return round_limits(-dmax, dmax, log=False)[1]
