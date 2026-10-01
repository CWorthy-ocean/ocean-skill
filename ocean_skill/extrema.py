"""Locate a field's min/max, then follow it through time.

``Field.extremum()`` answers the question a surface map naturally raises: where
exactly is that hot spot, and how did it get there? The locator (:func:`_locate`)
runs over *every* standing dimension, not just the horizontal ones, so a field
faceted over time or depth reports the facet coordinate the extremum fell on for
free — the same idiom :func:`ocean_skill.align._nearest_indices` uses for a nearest-
cell lookup, generalized from one dimension pair to however many survive.

Scoped to :class:`~ocean_skill.field.Field` for now. The locator itself
(:func:`_locate`) is generic over any eager, unreduced ``xr.DataArray`` and knows
nothing about a ``Field`` or a ``Comparison`` — so a later
``Comparison.extremum(kind, on="difference")`` can run it against
``aligned["difference"]`` without reshaping this module. :attr:`Extremum.grid` exists
for the same reason: today it is always "the source's own grid", but a comparison's
aligned pair lives on the coarser lane's regrid target, and that case will want to
name it rather than let a caller assume the test source's native grid.

The global min/max of a model field is usually a coastline or river-mouth cell, and
the "k-th lowest value" is more of the same, so two options widen the search. ``n=``
returns the ``n`` most extreme *distinct* places as an :class:`Extrema` (hits closer
than ``separation`` cells merge into one). ``local=True`` changes what is ranked: each
wet cell is scored against the median of its wet neighbors, which is what a one-cell
speck -- unremarkable in absolute value, nothing like its surroundings -- stands out
on, while a straight front or a coastal gradient scores near zero. The neighborhood
counts only wet cells, and every hit reports how many there were.

By default that score is ``z``: the departure in units of how much the neighbors vary
among themselves. A river plume is steep a few cells offshore, so its cells depart from
their neighbors by far more than a speck does, but their neighbors differ just as much;
dividing by that spread ranks the speck first. ``score="departure"`` ranks by the
departure alone. ``interior=k`` additionally drops every cell with land or the grid's
edge within ``k`` cells, for either kind of search, and every hit reports its
``land_distance`` so the coast is visible in the table.

The follow-on time series is deliberately *not* a new plot family: writing the
extremum's ``lon``/``lat`` into a fresh ``select`` and re-entering
:func:`ocean_skill.field.field` produces an ordinary ``Field``/``FieldSet`` whose
point select already draws the ``series`` family in both renderers (see
:attr:`ocean_skill.field.Field.family`) — so a new capability costs no renderer code.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any

import numpy as np

from ocean_skill import _stacklevel

__all__ = ["Extrema", "Extremum", "field_extremum"]

#: Native time steps kept on each side of the snapshot by :meth:`Extremum.series`'s
#: default window. Ten steps each way is enough to see an event grow and decay at
#: the record's own cadence without silently reading a large span; ``time=``
#: overrides it with the full :func:`ocean_skill.operators.select` grammar.
DEFAULT_PAD_STEPS = 10

#: Neighborhood side, in cells, for ``local=True`` when ``window=`` is not given.
DEFAULT_WINDOW = 3

#: Minimum grid-index spacing between hits of a global ``n > 1`` search when
#: ``separation=`` is not given -- wide enough that one river plume or coastal
#: pile-up is one place rather than ten adjacent cells. A ``local=True`` search
#: defaults to its own ``window`` instead (one speck, one hit).
DEFAULT_GLOBAL_SEPARATION = 10

#: A cell with fewer wet neighbors than this (a lone wet pixel, a one-cell inlet) has
#: no neighborhood to stand out from, so ``local=True`` does not score it.
MIN_WET_NEIGHBORS = 3

#: Scales a median absolute deviation to a standard deviation for normal data, so a
#: ``spread`` reads on the same footing as a standard error of the neighbors.
_MAD_TO_SIGMA = 1.4826

#: Why a search can come back empty, appended to by whatever else emptied it.
_NO_DATA_HINT = (
    "The selection may fall entirely on land/masked cells, or outside the "
    "source's coverage -- widen select= or check the depth/time asked for."
)

#: The two ways ``local=True`` can rank a cell. ``"z"`` divides the departure from the
#: neighbors' median by how much those neighbors vary among themselves, so a cell in a
#: steep plume or front (large departure, but its neighbors differ just as much) ranks
#: below a one-cell speck in smooth water; ``"departure"`` is the departure alone, in
#: the field's units.
SCORES = ("z", "departure")


def _rank(
    score,
    kind: str,
    *,
    n: int = 1,
    separation: int = 1,
    hdims: tuple[str, ...] = (),
    source: str,
    what: str = "prepared field",
    hint: str = _NO_DATA_HINT,
) -> list[dict[str, int]]:
    """Return the indices of ``score``'s ``n`` most extreme *distinct* places.

    Most extreme first.

    The unravel-index idiom :func:`ocean_skill.align._nearest_indices` uses for one
    dimension pair, generalized to however many dims ``score`` still carries -- which
    is what lets a field faceted over time or depth report the facet coordinate the
    extremum fell on, for free, once the caller reads the point back off the indices.

    After each pick, every cell within ``separation - 1`` grid indices of it on
    ``hdims`` is masked out -- across *every* other dimension, so a feature that
    persists through the record is reported once, at the slice where it is most
    extreme. Fewer than ``n`` picks come back when the field runs out of finite
    cells; none at all raises.
    """
    if kind not in ("max", "min"):
        raise ValueError(f'kind must be "max" or "min", got {kind!r}.')
    finder = np.nanargmax if kind == "max" else np.nanargmin
    values = np.asarray(score.values)
    if n > 1:
        values = np.array(values, dtype=float)  # a writable copy to mask into
    radius = max(int(separation) - 1, 0)
    picks: list[dict[str, int]] = []
    for _ in range(n):
        try:
            flat = int(finder(values))
        except ValueError as err:
            if picks:
                break
            raise ValueError(
                f"cannot locate a {kind}: {source!r}'s {what} is NaN "
                f"everywhere ({dict(score.sizes)}). {hint}"
            ) from err
        where = np.unravel_index(flat, values.shape)
        picks.append({str(d): int(i) for d, i in zip(score.dims, where)})
        if len(picks) == n:
            break
        if hdims:
            box = tuple(
                slice(max(i - radius, 0), i + radius + 1) if d in hdims else slice(None)
                for d, i in zip(score.dims, where)
            )
        else:
            box = tuple(slice(i, i + 1) for i in where)
        values[box] = np.nan
    return picks


def _locate(da, kind: str, *, source: str) -> dict[str, int]:
    """Indices of ``da``'s global nan-``kind`` extremum, over every dim it has."""
    return _rank(da, kind, n=1, source=source)[0]


def _horizontal_dims(da) -> tuple[str, ...]:
    """Return the dims ``da``'s lon/lat coordinates live on, in ``da``'s dim order.

    ``("eta_rho", "xi_rho")`` on a curvilinear grid, ``("lat", "lon")`` on a
    rectilinear one, one dim on a transect's ``along`` axis, none when the field
    carries no lon/lat at all.
    """
    from ocean_skill.align import _lat_name, _lon_name

    found: set[str] = set()
    for name in (_lon_name(da), _lat_name(da)):
        if name is not None and name in da.coords:
            found.update(str(d) for d in da.coords[name].dims)
    return tuple(str(d) for d in da.dims if str(d) in found)


def _median_along_first(stack, wet):
    """Median over the leading axis of ``stack`` of its ``wet`` finite values.

    ``stack`` must already be sorted along axis 0 -- NaNs sort last, so the ``wet``
    finite values lead -- and ``wet`` counts them per cell. Read off the two middle
    ranks rather than through :func:`numpy.nanmedian`, which is slow at this size and
    warns on an all-NaN neighborhood.
    """
    lo = np.clip((wet - 1) // 2, 0, None)[None]
    hi = (wet // 2)[None]
    return 0.5 * (
        np.take_along_axis(stack, lo, axis=0)[0]
        + np.take_along_axis(stack, hi, axis=0)[0]
    )


def _slice_departure(sl, window: int):
    """Score one 2-D slice: departure from the wet-neighbor median, and the spread.

    Returns ``(departure, median, spread, wet)`` arrays shaped like ``sl``. ``wet``
    counts the finite neighbors among the ``window**2 - 1`` around each cell (the cell
    itself excluded); ``median`` is over those alone, so land never drags it, and
    ``spread`` is their median absolute deviation about it, scaled to a standard
    deviation. Cells beyond the grid's edge are not neighbors (no periodic wrap).

    ``departure`` (``sl`` minus ``median``) and ``spread`` are NaN where the cell
    itself is masked or where ``wet`` is under :data:`MIN_WET_NEIGHBORS`.
    """
    r = window // 2
    ny, nx = sl.shape
    padded = np.full((ny + 2 * r, nx + 2 * r), np.nan)
    padded[r : r + ny, r : r + nx] = np.where(np.isfinite(sl), sl, np.nan)
    stack = np.stack(
        [
            padded[dy : dy + ny, dx : dx + nx]
            for dy in range(window)
            for dx in range(window)
            if (dy, dx) != (r, r)
        ]
    )
    wet = np.isfinite(stack).sum(axis=0)
    stack.sort(axis=0)
    median = _median_along_first(stack, wet)
    deviation = np.abs(stack - median[None])  # NaN stays NaN, so it still sorts last
    deviation.sort(axis=0)
    spread = _MAD_TO_SIGMA * _median_along_first(deviation, wet)
    centre = padded[r : r + ny, r : r + nx]
    scored = np.isfinite(centre) & (wet >= MIN_WET_NEIGHBORS)
    return (
        np.where(scored, centre - median, np.nan),
        median,
        np.where(scored, spread, np.nan),
        wet,
    )


def _neighbor_departure(da, hdims: tuple[str, ...], window: int):
    """Return ``(departure, median, spread, wet)`` DataArrays on ``da``'s coordinates.

    :func:`_slice_departure` at every horizontal slice of ``da``.

    Leading (time, depth, ...) dims are looped over one 2-D slice at a time, so the
    working set is ``window**2 - 1`` copies of one slice, not of the whole field. The
    returned arrays are transposed to put the horizontal dims last.
    """
    lead = [d for d in da.dims if d not in hdims]
    da_t = da.transpose(*lead, *hdims)
    values = np.asarray(da_t.values, dtype=float)
    departure = np.full(values.shape, np.nan)
    median = np.full(values.shape, np.nan)
    spread = np.full(values.shape, np.nan)
    wet = np.zeros(values.shape, dtype=np.int16)
    for idx in np.ndindex(*values.shape[:-2]):
        departure[idx], median[idx], spread[idx], wet[idx] = _slice_departure(
            values[idx], window
        )
    return tuple(da_t.copy(data=a) for a in (departure, median, spread, wet))


def _z_floor(spread, magnitude_of) -> float:
    """Return the smallest neighbor spread a z-score divides by.

    The field's own typical local variability -- the median ``spread`` over every
    scored cell -- so a patch that happens to be nearly constant cannot turn a
    departure of a fraction of a unit into an enormous z. A field that is constant
    everywhere has a zero median, so the floor never drops below floating-point
    resolution at the field's own magnitude, which keeps a lone speck's z finite.
    """
    finite = spread[np.isfinite(spread)]
    typical = float(np.median(finite)) if finite.size else 0.0
    has = np.isfinite(magnitude_of).any()
    magnitude = float(np.nanmax(np.abs(magnitude_of))) if has else 1.0
    return max(typical, float(np.finfo(float).eps) * (magnitude or 1.0))


def _near_land(sl, reach: int):
    """Boolean array shaped like ``sl``: a dry cell or the grid's edge within ``reach``.

    "Within" is Chebyshev distance -- the ``(2 * reach + 1)`` square around each cell
    -- and cells past the edge of the grid count as dry, so the mask is true for every
    cell that does not have a full ``reach`` of wet cells all round it. A summed-area
    table gives each square's dry count in constant time, so the cost is independent
    of ``reach``.
    """
    ny, nx = sl.shape
    padded = np.ones((ny + 2 * reach, nx + 2 * reach), dtype=np.int32)
    padded[reach : reach + ny, reach : reach + nx] = ~np.isfinite(sl)
    table = np.zeros((padded.shape[0] + 1, padded.shape[1] + 1), dtype=np.int64)
    table[1:, 1:] = padded.cumsum(axis=0).cumsum(axis=1)
    w = 2 * reach + 1
    dry = (
        table[w : w + ny, w : w + nx]
        - table[0:ny, w : w + nx]
        - table[w : w + ny, 0:nx]
        + table[0:ny, 0:nx]
    )
    return dry > 0


def _near_land_mask(da, hdims: tuple[str, ...], reach: int):
    """:func:`_near_land` at every horizontal slice of ``da``, as a DataArray.

    Transposed to put the horizontal dims last, like :func:`_neighbor_departure`.
    """
    lead = [d for d in da.dims if d not in hdims]
    da_t = da.transpose(*lead, *hdims)
    values = np.asarray(da_t.values, dtype=float)
    near = np.zeros(values.shape, dtype=bool)
    for idx in np.ndindex(*values.shape[:-2]):
        near[idx] = _near_land(values[idx], reach)
    return da_t.copy(data=near)


def _land_distance(da, hdims: tuple[str, ...], indices: dict[str, int]) -> int:
    """Cells from the hit at ``indices`` to the nearest dry cell or the grid's edge.

    Chebyshev distance on the hit's own horizontal slice, so ``1`` means touching
    land (or the edge) and a hit is kept by ``interior=k`` exactly when this exceeds
    ``k``. The distance to the edge bounds the answer, so only the square that far
    around the hit is searched for dry cells.
    """
    lead = {d: i for d, i in indices.items() if d not in hdims}
    sl = np.asarray(da.isel(lead).transpose(*hdims).values, dtype=float)
    i, j = (indices[d] for d in hdims)
    ny, nx = sl.shape
    edge = min(i + 1, j + 1, ny - i, nx - j)
    r = edge - 1
    box = sl[i - r : i + r + 1, j - r : j + r + 1]
    rows, cols = np.nonzero(~np.isfinite(box))
    if rows.size == 0:
        return int(edge)
    return int(np.maximum(np.abs(rows - r), np.abs(cols - r)).min())


def _time_reason(time: Any, select: dict[str, Any]) -> str:
    """Why :attr:`Extremum.time` came out the way it did -- read by the repr and by
    :meth:`Extremum.series`'s default-window fallback.
    """
    if time is not None:
        return "the time coordinate at the extremum"
    from ocean_skill.sources import _TIME_KEYS

    if any(k in select for k in _TIME_KEYS):
        return (
            "no snapshot at the extremum (time was aggregated away); the series "
            "defaults to the recipe's own time selection"
        )
    return (
        "no snapshot at the extremum and no time selection on the recipe; the "
        "series defaults to the full record"
    )


@dataclasses.dataclass(frozen=True)
class Extremum:
    """Where a field's min/max value is: value, position, grid indices, snapshot.

    Parameters
    ----------
    kind
        One of ``"max"``, ``"min"`` -- which extremum was located.
    value
        The extremum's scalar value (``float``).
    units
        The field's units (``str``), or ``None`` if it has none.
    variable
        The field's variable spec, in whatever form the parent recipe used.
    standard_name
        The field's CF ``standard_name`` (``str``), or ``None`` if unresolved.
    source
        The parent field's source name (``str``).
    lon, lat
        The extremum's horizontal position (``float``), or ``None`` if the field
        has no lon/lat coordinate.
    lon_convention
        The longitude convention the field's grid uses (``str``, e.g.
        ``"-180-180"`` or ``"0-360"`` -- see :func:`ocean_skill.align.natural_convention`).
    indices
        ``dict`` of every dim of the prepared field at the extremum, e.g.
        ``{"eta_rho": 112, "xi_rho": 387}`` on a curvilinear grid.
    coords
        ``dict`` of non-horizontal, non-time coordinate values at the extremum
        (depth, sigma0, a climatology's ``month``, ...), keyed by coordinate name.
    time
        The snapshot time at the extremum, or ``None`` if none is available (see
        ``time_reason``).
    time_reason
        ``str`` explaining why ``time`` came out the way it did.
    grid
        ``str`` naming which grid ``indices`` is into -- always ``"the source's
        own grid"`` for a ``Field`` today.
    mode
        ``"global"`` (ranked by value) or ``"local"`` (ranked by how far a cell
        sits from its wet neighbors, ``Field.extremum(..., local=True)``).
    rank
        ``int``, 1 for the most extreme hit, 2 for the next distinct one, ...
    anomaly, neighborhood, spread, z, wet_neighbors, window, score
        Local mode only (``None`` otherwise): ``anomaly`` is ``value -
        neighborhood`` in the field's units, ``neighborhood`` the median of the
        ``wet_neighbors`` finite cells among the ``window**2 - 1`` around the hit,
        ``spread`` those neighbors' own variability (a scaled median absolute
        deviation, in the field's units), ``z`` the ``anomaly`` in units of that
        spread (never divided by less than the field's typical spread), and
        ``score`` which of ``"z"``/``"departure"`` the hits were ranked by.
    land_distance
        ``int``, cells to the nearest land (masked) cell or the edge of the grid,
        by Chebyshev distance -- 1 means touching -- or ``None`` on a field
        without a 2-D horizontal grid.

    Built by :func:`field_extremum` (reached as ``Field.extremum()``), never
    directly. :meth:`series` follows the same location through time.
    """

    kind: str
    value: float
    units: str | None
    variable: Any
    standard_name: str | None
    source: str
    lon: float | None
    lat: float | None
    lon_convention: str
    #: Every dim of the prepared field at the extremum, e.g. ``{"eta_rho": 112,
    #: "xi_rho": 387}`` on a curvilinear (ROMS) grid, ``{"lat": 41, "lon": 220}`` on
    #: a rectilinear one -- named by the field's own dims, whichever those are.
    indices: dict[str, int]
    #: Non-horizontal, non-time coordinate values at the extremum (depth, sigma0,
    #: a climatology's ``month``, ...), keyed by their coordinate name.
    coords: dict[str, Any]
    time: Any | None
    time_reason: str
    #: Which grid :attr:`indices` are into. Always ``"the source's own grid"`` for a
    #: Field; reserved for a future ``Comparison.extremum``, whose aligned pair lives
    #: on the coarser lane's regrid target rather than either source's native grid.
    grid: str
    _parent: Any = dataclasses.field(repr=False, compare=False)
    mode: str = "global"
    rank: int = 1
    anomaly: float | None = None
    neighborhood: float | None = None
    wet_neighbors: int | None = None
    window: int | None = None
    spread: float | None = None
    z: float | None = None
    score: str | None = None
    land_distance: int | None = None

    def __repr__(self) -> str:
        from ocean_skill.comparison import _short_variable_label

        name = _short_variable_label(self.variable)
        units = f" {self.units}" if self.units else ""
        if self.lon is not None and self.lat is not None:
            where = f"lon {self.lon:.4f}, lat {self.lat:.4f} ({self.lon_convention})"
        else:
            where = "no horizontal position (this field has no lon/lat coordinate)"
        when = (
            f"time {self.time}"
            if self.time is not None
            else f"no snapshot time ({self.time_reason})"
        )
        extra = f", {self.coords}" if self.coords else ""
        label = f"local {self.kind}" if self.mode == "local" else self.kind
        contrast = ""
        if self.mode == "local" and self.anomaly is not None:
            side = "below" if self.anomaly < 0 else "above"
            w = self.window or DEFAULT_WINDOW
            z = f", z = {self.z:.3g}" if self.z is not None else ""
            contrast = (
                f"  {abs(self.anomaly):.6g}{units} {side} the median of its {w}x{w} "
                f"wet neighbors ({self.neighborhood:.6g}; "
                f"{self.wet_neighbors}/{w * w - 1} wet){z}\n"
            )
        if self.land_distance is not None:
            cells = "cell" if self.land_distance == 1 else "cells"
            contrast += (
                f"  {self.land_distance} {cells} from the nearest land or grid edge\n"
            )
        return (
            f"{label} {name} = {self.value:.6g}{units} at {where}\n"
            f"{contrast}"
            f"  grid indices {self.indices}, {when}{extra}\n"
            f"  source={self.source!r}, grid={self.grid!r}"
        )

    def series(
        self,
        variables: list[Any] | None = None,
        *,
        time: Any = None,
        pad: int = DEFAULT_PAD_STEPS,
        label: str | None = None,
        cache: bool | None = None,
    ):
        """Follow this extremum through time: a point series at its location.

        Parameters
        ----------
        variables
            Extra variable(s) to add as more lines on the same figure; the
            extremum's own variable is always plotted first. A single variable
            (a string, or a combination/``calculate`` spec ``dict``) is accepted
            the same as a one-element list -- it is never iterated over
            character by character.
        time
            The ordinary :func:`ocean_skill.operators.select` grammar (a slice, a
            partial date, a ``{"min", "max"}`` range) to control the time window
            directly. ``None`` (default) uses the extremum's own snapshot, padded
            by ``pad`` native steps each side.
        pad
            ``int``, native time steps kept on each side of the snapshot when
            ``time`` is not given (default :data:`DEFAULT_PAD_STEPS`).
        label
            ``str`` label for the resulting field/series, or ``None`` (default)
            to reuse the parent field's own label.
        cache
            ``bool | None`` -- ``None`` (default) inherits the parent field's own
            ``cache``; pass ``True``/``False`` to override it for this series
            call alone (a suite page uses this to mark a window that reaches the
            run's latest step as uncached, the same rule an ordinary page's own
            ``time: latest`` already follows).

        Builds an ordinary :func:`ocean_skill.field.field` call from the parent
        field's own recipe -- same source, same vertical selection, same
        aggregate minus its time entry, same ``qc``/``detide`` -- with the
        horizontal axes pinned to this extremum's position and the time axis
        narrowed to a window around its snapshot. ``variables`` adds more lines to
        the same figure (the extremum's own variable is always first, on the
        primary axis); a single-element list still returns a
        :class:`~ocean_skill.field.FieldSet`, the same as
        :func:`ocean_skill.field.field` itself.

        The default window is :data:`DEFAULT_PAD_STEPS` native time steps each
        side of the snapshot, clamped to the record's own ends -- read lazily off
        the source's own time coordinate, so this is the one call in the
        ``extremum()`` -> ``series()`` -> ``plot()`` chain that may reopen the
        catalog entry. Pass ``time=`` (the ordinary :func:`ocean_skill.operators
        .select` grammar -- a slice, a partial date, a ``{"min", "max"}`` range)
        to skip that and control the window directly.

        When the extremum carries no snapshot (:attr:`time_reason` explains why --
        typically the parent's ``aggregate`` collapsed time entirely), the parent's
        own ``select={"time": ...}`` is reused unchanged if it had one; lacking
        that too, the series defaults to the source's full record and warns, since
        that read can be large.

        A native s-level axis with no coordinate of its own (a bare parent select at
        a ROMS point) is pinned by position, ``select={"s_rho": {"index": k}}``,
        rather than by depth: ``z_rho`` is a height that moves with the tide and can
        put the extremum's level above mean sea level, where no depth request can
        name it.

        A variable that does not share this field's vertical axis is unaffected by
        the depth/sigma0/s-level pin: :func:`ocean_skill.operators.select` skips a key
        naming an axis a given variable does not have.

        The returned set carries a default suptitle (its
        :attr:`~ocean_skill.field.FieldSet.title`) saying what the point *is*, since
        each panel's own title only names a place and a period and so reads like a
        station record or a domain-wide trend::

            Time series at the surface alkalinity minimum: 126.1 mmol/m^3 at
            16.1°N 97.6°E on 2010-10-31

        The depth phrase (``surface``, ``100 m``) comes from the parent field's own
        single-level vertical select and is left out for none, a band or a list; a
        ``local=True`` hit reads ``local minimum``, the second of an ``n=`` search
        ``minimum (#2)``; the position and date are left out when unknown.
        ``.plot(title=...)`` replaces it (so does a suite page's ``plot: {title:
        ...}``), and ``title=""`` draws none.

        Refused when the parent's own ``select`` names a ``transect`` -- once cut
        to a transect, lon/lat lie along the cut's own ``along`` axis rather than
        standing as ordinary coordinates, so there is no longer a plain lon/lat
        pair here to pin a point series to.
        """
        from ocean_skill.comparison import _ANY_VERTICAL_KEYS
        from ocean_skill.field import field
        from ocean_skill.operators import (
            _POINT_LAT_KEYS,
            _POINT_LON_KEYS,
            resolve_dim,
        )
        from ocean_skill.sources import _TIME_KEYS

        parent = self._parent
        if self.lon is None or self.lat is None:
            raise ValueError(
                f"{self.source!r} has no lon/lat coordinate to sample a series "
                "at -- extremum() found a value but no position to follow it "
                "from."
            )

        parent_select = dict(parent.select)
        if "transect" in parent_select:
            raise ValueError(
                f"{parent.source!r}'s select includes a transect -- .series() "
                "cannot follow a point through a transect's own along-axis. Call "
                "extremum()/series() on a plain map or point-select field instead."
            )
        parent_time_key = next((k for k in _TIME_KEYS if k in parent_select), None)
        parent_time_value = (
            parent_select.get(parent_time_key) if parent_time_key else None
        )

        sel = {
            k: v
            for k, v in parent_select.items()
            if k not in _POINT_LON_KEYS
            and k not in _POINT_LAT_KEYS
            and k not in _TIME_KEYS
        }
        sel["lon"], sel["lat"] = self.lon, self.lat

        vkey = next((k for k in _ANY_VERTICAL_KEYS if k in sel), None)
        zdim = resolve_dim(parent.data, "Z")
        if vkey is not None:
            if isinstance(sel[vkey], list) and zdim is not None and zdim in self.coords:
                sel[vkey] = self.coords[zdim]
        elif zdim is not None and zdim in self.coords:
            sel[zdim] = self.coords[zdim]
        elif zdim is not None and "z_rho" in self.coords:
            # A native s-level axis (a bare parent select=) carries no coordinate
            # of its own named zdim to read the extremum's own level back off of --
            # the level's height rode on z_rho instead, which the extremum's own
            # isel already reduced to a scalar. Pin the child field to that one
            # level explicitly, or an empty sel here would leave the whole column
            # standing again rather than following the extremum at the one level
            # it was actually found on.
            #
            # Pinned by position, not by depth: z_rho is a height (see
            # plot/profile.py:positive_down) carrying the free surface, so under
            # a raised sea surface the extremum's level can be above mean sea
            # level -- a negative depth, which is no request select={"depth": ...}
            # is written to take -- and even a positive one names a fixed depth
            # the tide moves the level off of, letting the series drift out of
            # the column it was pinned to. The s-level index is the one thing that
            # stays put.
            sel[zdim] = {
                "index": _source_index(
                    sel.get(zdim), self.indices[zdim], source=parent.source
                )
            }
        elif zdim is not None:
            raise ValueError(
                f"{parent.source!r}'s vertical axis ({zdim!r}) carries no "
                "coordinate and no z_rho either, so this extremum's own depth "
                "cannot be read back to pin the follow-up series to it. Narrow "
                "the parent field's own select= to a coordinate-bearing depth "
                "request first."
            )

        if time is not None:
            sel["time"] = time
        elif self.time is not None:
            index = _native_time_index(parent.source)
            sel["time"] = _window_select(index, self.time, pad)
        elif parent_time_value is not None:
            sel["time"] = parent_time_value
        else:
            warnings.warn(
                f"{parent.source!r} has no snapshot time at the extremum and no "
                f"time selection to fall back to ({self.time_reason}) -- the "
                "series defaults to the source's full time record. Pass time= "
                "to narrow it.",
                stacklevel=_stacklevel.find(),
            )

        agg = {
            k: v for k, v in (parent.aggregate or {}).items() if k not in _TIME_KEYS
        }

        if variables is None:
            extra: list[Any] = []
        elif isinstance(variables, (list, tuple)):
            extra = list(variables)
        else:
            # A single variable spec -- a plain name or a combination/``calculate``
            # dict -- wrapped the same as field()'s own variable= would treat it.
            # *variables would otherwise unpack a bare string character by
            # character, or a dict's keys.
            extra = [variables]

        made = field(
            parent.source,
            [parent.variable, *extra],
            select=sel,
            aggregate=agg or None,
            label=label if label is not None else parent.label,
            cache=parent.cache if cache is None else cache,
            qc=parent.qc,
            detide=parent.detide,
        )
        # field() returns a FieldSet here: the variable is always a list, even of
        # one. The default title rides on the set (FieldSet.title) rather than
        # being a plot() argument, so every route to a drawing -- this series'
        # own .plot(), Extremum.plot(), a suite page -- starts from it, and an
        # explicit plot(title=...) still wins (FieldSet.plot only setdefaults).
        made.title = self._series_title()
        return made

    def _series_title(self) -> str:
        """Return the default suptitle of :meth:`series`: what the point *is*.

        ``"Time series at the surface alkalinity minimum: 126.1 mmol/m^3 at 16.1°N
        97.6°E on 2010-10-31"``. Each panel's own title names a place and a period
        (``alkalinity · 16.1°N 97.6°E · 2010-07 to 2010-10``), and read alone that is
        indistinguishable from a station's record -- or worse, from a domain-wide
        trend. This line says the point was *found* as an extremum, which one, and
        how extreme it was, so nobody has to guess.

        Parts, each left out rather than guessed when it is not known:

        * ``the <depth> `` -- from the parent field's own vertical select: the
          literal ``"surface"``, or a single depth spelled as every other label
          here spells it (``"100 m"``; an isopycnal ``sigma0`` request reads
          ``"σ₀ = 26 kg/m³"``). Omitted for no vertical select, or a band, list or
          anything else that does not name one level.
        * the variable, in the vocabulary's short name
          (:func:`ocean_skill.comparison._short_variable_label`).
        * ``minimum``/``maximum``, or ``local minimum``/``local maximum`` for a
          ``local=True`` hit, with `` (#n)`` after it for any but the first of an
          ``n=`` search.
        * the value (four significant figures) and its units.
        * `` at <lon/lat>`` -- spelled as the panel titles spell a station
          (:func:`ocean_skill.plot.series.lonlat_label`).
        * `` on YYYY-MM-DD`` -- the snapshot the extremum was found on; omitted when
          there is none (:attr:`time_reason` says why).

        Set on the :class:`~ocean_skill.field.FieldSet` :meth:`series` returns, as its
        :attr:`~ocean_skill.field.FieldSet.title`, and so applied by ``.plot()`` unless
        a ``title=`` is passed there (a suite page's ``plot: {title: ...}`` does).
        """
        from ocean_skill.comparison import _short_variable_label
        from ocean_skill.plot.series import lonlat_label

        depth = self._parent_depth_phrase()
        kind = {"max": "maximum", "min": "minimum"}.get(self.kind, self.kind)
        if self.mode == "local":
            kind = f"local {kind}"
        if self.rank > 1:
            kind = f"{kind} (#{self.rank})"
        name = _short_variable_label(self.variable)
        subject = f"{depth} {name}" if depth else name
        value = f"{self.value:.4g}" + (f" {self.units}" if self.units else "")
        title = f"Time series at the {subject} {kind}: {value}"
        if self.lon is not None and self.lat is not None:
            title += f" at {lonlat_label(self.lon, self.lat)}"
        date = _date_label(self.time)
        if date is not None:
            title += f" on {date}"
        return title

    def _parent_depth_phrase(self) -> str | None:
        """How the parent field's vertical select names one level, or ``None``.

        ``"surface"`` for the sentinel, a single depth through
        :func:`ocean_skill.comparison._depth_label` (``"100 m"``), a single
        ``sigma0`` through the same function's isopycnal spelling, and ``None`` for
        everything that does not name exactly one level -- no vertical select, a
        band (``{"min", "max"}``), a list, a ``"column"`` request.
        """
        from ocean_skill.comparison import _ANY_VERTICAL_KEYS, _depth_label

        select = self._parent.select or {}
        key = next((k for k in _ANY_VERTICAL_KEYS if k in select), None)
        if key is None:
            return None
        value = select[key]
        if isinstance(value, str):
            return "surface" if value.lower() == "surface" else None
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            return None
        return _depth_label({"sigma0": value} if key == "sigma0" else value)

    def plot(self, *, renderer: str = "matplotlib", **kwargs: Any):
        """Shortcut for ``.series().plot(...)`` -- the default window, drawn now.

        Parameters
        ----------
        renderer
            One of ``"matplotlib"``, ``"holoviews"`` (default ``"matplotlib"``).
        **kwargs
            Plot styling kwargs forwarded to the resulting series' ``.plot()`` --
            see ``docs/plot_styling_reference.md`` for the full list. ``title=``
            replaces the default suptitle (see :meth:`series`).

        Use :meth:`series` directly when ``variables=``/``time=``/``pad=`` need to
        be set; this only forwards ``renderer=`` and plot styling kwargs.
        """
        return self.series().plot(renderer=renderer, **kwargs)


@dataclasses.dataclass(frozen=True)
class Extrema:
    """The ``n`` most extreme distinct places of a field, most extreme first.

    Returned by ``Field.extremum(..., n=...)`` for ``n > 1``. Each item is an
    ordinary :class:`Extremum`, so ``hits[2].series()`` / ``.plot()`` follow that
    place through time exactly as a single extremum's do. ``len()``, iteration and
    indexing work as on a tuple; :meth:`to_dataframe` gives the table to filter and
    sort, and the repr prints it.

    Parameters
    ----------
    kind
        ``"max"`` or ``"min"``.
    mode
        ``"global"`` or ``"local"`` -- see :class:`Extremum`.
    window
        Neighborhood side in cells (local mode), or ``None``.
    separation
        Minimum grid-index spacing between hits on the horizontal dims.
    interior
        ``int``, the ``interior=`` reach: hits are at least ``interior + 1`` cells
        from land or the grid's edge (``0`` when nothing was excluded).
    score
        ``"z"`` or ``"departure"`` (local mode), or ``None``.
    items
        The hits, as a tuple of :class:`Extremum`.
    """

    kind: str
    mode: str
    window: int | None
    separation: int
    interior: int
    items: tuple[Extremum, ...]
    score: str | None = None

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __getitem__(self, i):
        return self.items[i]

    def to_dataframe(self):
        """One row per hit, indexed by ``rank``.

        Columns: ``value``, then (local mode) ``anomaly``, ``neighborhood``,
        ``spread``, ``z`` and ``wet_neighbors``, then ``land_distance``, ``lon``,
        ``lat``, ``time``, one ``i_<dim>`` column of grid indices per dimension, and
        each scalar coordinate (depth, ...).
        """
        import pandas as pd

        rows = []
        for e in self.items:
            row: dict[str, Any] = {"value": e.value}
            if self.mode == "local":
                row.update(
                    anomaly=e.anomaly,
                    neighborhood=e.neighborhood,
                    spread=e.spread,
                    z=e.z,
                    wet_neighbors=e.wet_neighbors,
                )
            if e.land_distance is not None:
                row["land_distance"] = e.land_distance
            row.update(lon=e.lon, lat=e.lat, time=e.time)
            row.update({f"i_{d}": i for d, i in e.indices.items()})
            for name, val in e.coords.items():
                row.setdefault(name, val)
            rows.append(row)
        return pd.DataFrame(rows, index=pd.Index(range(1, len(rows) + 1), name="rank"))

    def __repr__(self) -> str:
        from ocean_skill.comparison import _short_variable_label

        first = self.items[0]
        how = f">= {self.separation} cells apart"
        if self.mode == "local":
            w = self.window or DEFAULT_WINDOW
            how = f"{w}x{w} wet-neighbor median, ranked by {self.score}, {how}"
        if self.interior:
            cells = "cell" if self.interior == 1 else "cells"
            how += f", no land within {self.interior} {cells}"
        label = f"local {self.kind}" if self.mode == "local" else self.kind
        name = _short_variable_label(first.variable)
        units = f" [{first.units}]" if first.units else ""
        return (
            f"{len(self)} {label} {name}{units} on {first.source!r} ({how})\n"
            f"{self.to_dataframe().to_string()}"
        )


def _source_index(narrowed: Any, k: int, *, source: str) -> int:
    """Map position ``k`` in a positionally narrowed axis back to the source's own.

    ``k`` counts along the *parent's prepared* axis, so if the parent's own select
    already cut that axis by position -- ``{"index": {"min": 10, "max": 20}}``, a
    ``{"index": [3, 5, 7]}`` list, or a bare list -- the child field, which
    re-selects from the whole source, needs the position in the source rather than
    in that cut. ``None`` (nothing narrowed the axis) is the identity.

    A slice is only mapped back when its origin is knowable without the source's
    full length: a non-negative ``start`` and a positive ``step``. A negative one
    counts from the far end of an axis whose length is not to hand here, so it is
    refused rather than guessed at, as is any selection that is not positional at
    all (a coordinate-value one cannot address a bare native axis).
    """
    if narrowed is None:
        return int(k)
    from ocean_skill.operators import _index_spec

    picked = _index_spec(narrowed)
    if picked is None:
        picked = narrowed
    if isinstance(picked, list | tuple | np.ndarray):
        return int(picked[k])
    if isinstance(picked, slice):
        start = 0 if picked.start is None else int(picked.start)
        step = 1 if picked.step is None else int(picked.step)
        if start >= 0 and step > 0:
            return start + int(k) * step
    raise ValueError(
        f"{source!r}'s native vertical axis was narrowed with select={narrowed!r}, "
        "which cannot be mapped back to a single s-level to pin the follow-up "
        "series to. Narrow it with an explicit non-negative {'index': ...} range "
        "or list, or leave it whole, before calling extremum()."
    )


def _native_time_index(source: str):
    """The source's full, native time index, read lazily off the catalog entry.

    Coordinate-only -- like :func:`ocean_skill.comparison._time_bins`, which this
    mirrors: a time index is a small in-memory array, so resolving it against a
    lazily-opened, multi-file model run costs nothing like loading the data itself
    would. Kept as its own function so a default-window build never has to reopen
    the catalog more than once, and so tests can monkeypatch it without a catalog.
    """
    from ocean_skill import operators
    from ocean_skill.sources import read

    obj = read(source)
    dim = operators.resolve_dim(obj, "time")
    if dim is None:
        raise ValueError(
            f"{source!r} has no time axis, so a default time window has nothing "
            "to pad around. Pass time= explicitly."
        )
    index = obj.indexes.get(dim)
    if index is None:
        raise ValueError(
            f"{source!r}'s time axis ({dim!r}) carries no coordinate values -- "
            "an undecoded axis has no native steps to pad by. Pass time= "
            "explicitly, selecting the axis's own numeric values."
        )
    return index


def _scalar_time(coord) -> Any:
    """The scalar value of a 0-d time coordinate, as a type that round-trips.

    ``numpy.datetime64`` scalars at nanosecond resolution collapse to a bare int
    under ``.item()`` -- Python's ``datetime`` can't hold nanoseconds, so numpy
    hands back the raw ns-since-epoch integer. That int then can't be nearest-
    matched against the source's native time index in :func:`_window_select`
    whenever the index isn't itself ``datetime64[ns]`` (a ``datetime64[s]`` index
    raises "Cannot compare dtypes datetime64[s] and int64"). Route datetime64
    through :class:`pandas.Timestamp` so the snapshot stays a datetime regardless
    of the coordinate's resolution; leave everything else -- numeric time axes,
    cftime objects -- to ``.item()``, which already yields the right Python type.
    """
    values = coord.values
    if np.issubdtype(values.dtype, np.datetime64):
        import pandas as pd

        return pd.Timestamp(values)
    return values.item()


def _date_label(time: Any) -> str | None:
    """``"2010-10-31"`` for a snapshot time, or ``None`` when there is no date in it.

    Reads the first ten characters of the time's own string, which the three
    shapes :func:`_scalar_time` hands back all spell the same way -- a
    :class:`pandas.Timestamp` (any ``datetime64`` resolution is routed through one),
    a ``cftime`` date on a model run's own calendar, a ``datetime``. A bare number (a
    time axis nobody decoded) has no date to print, so it comes back ``None`` rather
    than as ``"3652.5"`` passed off as one.
    """
    import re

    if time is None:
        return None
    if isinstance(time, np.datetime64):
        import pandas as pd

        time = pd.Timestamp(time)
    match = re.match(r"\d{4}-\d{2}-\d{2}", str(time))
    return match.group(0) if match else None


def _window_select(index, snapshot: Any, pad: int) -> dict[str, str]:
    """Return a ``{"min", "max"}`` range spanning ``pad`` native steps each side of
    ``snapshot``, clamped to ``index``'s own ends.
    """
    try:
        pos = int(index.get_indexer([snapshot], method="nearest")[0])
    except (TypeError, NotImplementedError):
        # A calendar get_indexer doesn't support nearest-matching for (e.g. some
        # CFTimeIndex builds): fall back to a plain sorted-position search.
        values = np.asarray(index.values)
        pos = int(np.searchsorted(values, np.asarray(snapshot)))
        pos = min(max(pos, 0), len(values) - 1)
    lo = max(0, pos - pad)
    hi = min(len(index) - 1, pos + pad)
    return {"min": str(index[lo]), "max": str(index[hi])}


def _build(fld, da, indices: dict[str, int], kind: str, **extra: Any) -> Extremum:
    """One :class:`Extremum` for the cell of ``da`` at ``indices``.

    ``extra`` carries the fields only a ``local=True`` hit has (``mode``,
    ``rank``, ``anomaly``, ``neighborhood``, ``wet_neighbors``, ``window``).
    """
    from ocean_skill.align import (
        _lat_name,
        _lon_name,
        _time_name,
        natural_convention,
    )

    point = da.isel(indices)

    lon_name, lat_name = _lon_name(point), _lat_name(point)
    lon = float(point[lon_name]) if lon_name is not None else None
    lat = float(point[lat_name]) if lat_name is not None else None

    time_name = _time_name(point)
    time = (
        _scalar_time(point[time_name])
        if time_name is not None and time_name in point.coords
        else None
    )

    skip = {n for n in (lon_name, lat_name, time_name) if n is not None}
    coords = {
        str(name): point[name].values.item()
        for name in point.coords
        if name not in skip and point[name].ndim == 0
    }

    return Extremum(
        kind=kind,
        value=float(point),
        units=da.attrs.get("units"),
        variable=fld.variable,
        standard_name=fld.standard_name,
        source=fld.source,
        lon=lon,
        lat=lat,
        lon_convention=natural_convention(da),
        indices={str(d): indices[str(d)] for d in da.dims},
        coords=coords,
        time=time,
        time_reason=_time_reason(time, fld.select),
        grid="the source's own grid",
        _parent=fld,
        **extra,
    )


def _interior_reach(interior: Any, window: int | None) -> int:
    """Cells of clearance ``interior=`` asks for around every scored cell.

    ``False``/``None``/``0`` ask for none; ``True`` for the reach of the local window
    (``window // 2``, one cell for the default 3 x 3); an integer for exactly that many.
    """
    if interior is None or interior is False:
        return 0
    if interior is True or isinstance(interior, np.bool_):
        return (window or DEFAULT_WINDOW) // 2 if interior else 0
    if isinstance(interior, (int, np.integer)) and interior >= 0:
        return int(interior)
    raise ValueError(
        "interior must be True/False or an integer >= 0 (cells of clearance from "
        f"land or the grid's edge), got {interior!r}."
    )


def field_extremum(
    fld,
    kind: str = "max",
    *,
    n: int = 1,
    local: bool = False,
    window: int | None = None,
    interior: bool | int = False,
    separation: int | None = None,
    score: str | None = None,
) -> Extremum | Extrema:
    """Build an :class:`Extremum` (or, for ``n > 1``, an :class:`Extrema`) for
    ``fld`` -- the implementation behind ``Field.extremum()``.

    Parameters
    ----------
    fld
        A :class:`~ocean_skill.field.Field` whose prepared data has not already
        been reduced to a single point.
    kind
        One of ``"max"``, ``"min"`` (default ``"max"``) -- which extremum to
        locate.
    n
        ``int >= 1`` (default 1) -- how many distinct places to return. One gives an
        :class:`Extremum`; more give an :class:`Extrema`, holding fewer if the field
        runs out of scoreable cells.
    local
        ``bool`` (default ``False``). Rank cells by how far they sit from the median
        of their wet neighbors rather than by value, which surfaces features one
        cell wide -- a speck that is unremarkable in absolute value but nothing like
        the cells around it. ``"min"`` is then the most negative. Needs a 2-D
        horizontal grid.
    window
        Local mode only: odd ``int >= 3``, the neighborhood's side in cells
        (default :data:`DEFAULT_WINDOW`).
    interior
        ``False`` (default) considers every wet cell. An ``int`` k keeps only cells
        with no land and not the grid's edge within k cells (Chebyshev distance);
        ``True`` is k = ``window // 2``, i.e. a cell whose whole neighborhood is
        wet. Works for a global search too. Needs a 2-D horizontal grid.
    separation
        ``int >= 1``, minimum spacing in grid cells between reported hits on the
        horizontal dims (default: ``window`` when ``local``, else
        :data:`DEFAULT_GLOBAL_SEPARATION`).
    score
        Local mode only. ``"z"`` (the default there) ranks by the departure divided
        by the neighbors' own spread, so a steep plume or front -- large departure,
        but its neighbors differ just as much -- ranks below a speck in smooth water;
        ``"departure"`` ranks by the departure alone, in the field's units.
    """
    from ocean_skill.align import point_of

    da = fld.data
    if point_of(da) is not None:
        raise ValueError(
            f"{fld.source!r} has already been reduced to one place "
            f"({fld.family_reason}), so there is no spatial extremum to locate "
            "-- an extremum of one cell is the value .plot() already shows. "
            "Widen select= to keep a horizontal extent."
        )
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError(f"n must be an integer >= 1, got {n!r}.")
    if score is not None and score not in SCORES:
        raise ValueError(f'score must be "z" or "departure", got {score!r}.')
    if not local:
        stray = [k for k, v in (("window", window), ("score", score)) if v]
        if stray:
            raise ValueError(
                f"{' and '.join(stray)} only apply to local=True (neighbor-contrast) "
                "extrema; pass local=True or drop them."
            )
    if separation is not None and (
        isinstance(separation, bool)
        or not isinstance(separation, (int, np.integer))
        or separation < 1
    ):
        raise ValueError(f"separation must be an integer >= 1, got {separation!r}.")

    hdims = _horizontal_dims(da)
    w = None
    if local:
        w = DEFAULT_WINDOW if window is None else window
        if isinstance(w, bool) or not isinstance(w, (int, np.integer)):
            raise ValueError(f"window must be an odd integer >= 3, got {window!r}.")
        w = int(w)
        if w < 3 or w % 2 == 0:
            raise ValueError(f"window must be an odd integer >= 3, got {window!r}.")
        if len(hdims) != 2:
            raise ValueError(
                "local=True needs a 2-D horizontal grid to take neighbors on, but "
                f"{fld.source!r}'s prepared field has horizontal dims "
                f"{hdims or 'none'} (a transect or along-axis field has no "
                "neighbors on both sides). Use the default global search for it."
            )
    reach = _interior_reach(interior, w)
    if reach and len(hdims) != 2:
        raise ValueError(
            "interior needs a 2-D horizontal grid to measure the distance to land "
            f"on, but {fld.source!r}'s prepared field has horizontal dims "
            f"{hdims or 'none'}."
        )
    cells = "cell" if reach == 1 else "cells"
    note = (
        f" (interior={interior!r} also drops every cell within {reach} {cells} of "
        "land or the grid's edge)"
        if reach
        else ""
    )

    if local:
        chosen = "z" if score is None else score
        departure, median, spread, wet = _neighbor_departure(da, hdims, w)
        floor = _z_floor(spread.values, median.values)
        z = departure / np.maximum(spread, floor)
        ranking = z if chosen == "z" else departure
        what = "neighbor-departure field"
        hint = (
            f"Every cell is masked or has fewer than {MIN_WET_NEIGHBORS} wet "
            f"neighbors{note} -- widen select= or check the depth/time asked for."
        )
        gap = w if separation is None else int(separation)
    else:
        chosen = None
        ranking = da
        what = "prepared field"
        hint = _NO_DATA_HINT + note
        gap = DEFAULT_GLOBAL_SEPARATION if separation is None else int(separation)
    if reach:
        near = _near_land_mask(da, hdims, reach)
        ranking = ranking.where(~near.transpose(*ranking.dims))

    picks = _rank(
        ranking,
        kind,
        n=int(n),
        separation=gap,
        hdims=hdims,
        source=fld.source,
        what=what,
        hint=hint,
    )
    hits = []
    for i, ix in enumerate(picks, start=1):
        extra: dict[str, Any] = {"rank": i}
        if len(hdims) == 2:
            extra["land_distance"] = _land_distance(da, hdims, ix)
        if local:
            extra.update(
                mode="local",
                anomaly=float(departure.isel(ix)),
                neighborhood=float(median.isel(ix)),
                spread=float(spread.isel(ix)),
                z=float(z.isel(ix)),
                wet_neighbors=int(wet.isel(ix)),
                window=w,
                score=chosen,
            )
        hits.append(_build(fld, da, ix, kind, **extra))

    if n == 1:
        return hits[0]
    return Extrema(
        kind=kind,
        mode="local" if local else "global",
        window=w,
        separation=gap,
        interior=reach,
        items=tuple(hits),
        score=chosen,
    )
