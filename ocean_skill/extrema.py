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

__all__ = ["Extremum", "field_extremum"]

#: Native time steps kept on each side of the snapshot by :meth:`Extremum.series`'s
#: default window. Ten steps each way is enough to see an event grow and decay at
#: the record's own cadence without silently reading a large span; ``time=``
#: overrides it with the full :func:`ocean_skill.operators.select` grammar.
DEFAULT_PAD_STEPS = 10


def _locate(da, kind: str, *, source: str) -> dict[str, int]:
    """Indices of ``da``'s global nan-``kind`` extremum, over every dim it has.

    The unravel-index idiom :func:`ocean_skill.align._nearest_indices` uses for one
    dimension pair, generalized to however many dims ``da`` still carries — which is
    what lets a field faceted over time or depth report the facet coordinate the
    extremum fell on, for free, once the caller reads the point back off the indices.
    """
    if kind not in ("max", "min"):
        raise ValueError(f'kind must be "max" or "min", got {kind!r}.')
    finder = np.nanargmax if kind == "max" else np.nanargmin
    values = np.asarray(da.values)
    try:
        flat = int(finder(values))
    except ValueError as err:
        raise ValueError(
            f"cannot locate a {kind}: {source!r}'s prepared field is NaN "
            f"everywhere ({dict(da.sizes)}). The selection may fall entirely on "
            "land/masked cells, or outside the source's coverage -- widen "
            "select= or check the depth/time asked for."
        ) from err
    return {
        str(d): int(i)
        for d, i in zip(da.dims, np.unravel_index(flat, values.shape))
    }


def _time_reason(time: Any, select: dict[str, Any]) -> str:
    """Why :attr:`Extremum.time` came out the way it did -- read by the repr and by
    :meth:`Extremum.series`'s default-window fallback."""
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
        return (
            f"{self.kind} {name} = {self.value:.6g}{units} at {where}\n"
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
    ):
        """Follow this extremum through time: a point series at its location.

        Builds an ordinary :func:`ocean_skill.field.field` call from the parent
        field's own recipe -- same source, same vertical selection, same
        aggregate minus its time entry -- with the horizontal axes pinned to this
        extremum's position and the time axis narrowed to a window around its
        snapshot. ``variables`` adds more lines to the same figure (the extremum's
        own variable is always first, on the primary axis); a single-element list
        still returns a :class:`~ocean_skill.field.FieldSet`, the same as
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

        A variable that does not share this field's vertical axis is unaffected by
        the depth/sigma0 pin: :func:`ocean_skill.operators.select` skips a key
        naming an axis a given variable does not have.
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

        return field(
            parent.source,
            [parent.variable, *(variables or [])],
            select=sel,
            aggregate=agg or None,
            label=label if label is not None else parent.label,
            cache=parent.cache,
        )

    def plot(self, *, renderer: str = "matplotlib", **kwargs: Any):
        """Shortcut for ``.series().plot(...)`` -- the default window, drawn now.

        Use :meth:`series` directly when ``variables=``/``time=``/``pad=`` need to
        be set; this only forwards ``renderer=`` and plot styling kwargs.
        """
        return self.series().plot(renderer=renderer, **kwargs)


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


def _window_select(index, snapshot: Any, pad: int) -> dict[str, str]:
    """Return a ``{"min", "max"}`` range spanning ``pad`` native steps each side of
    ``snapshot``, clamped to ``index``'s own ends."""
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


def field_extremum(fld, kind: str = "max") -> Extremum:
    """Build an :class:`Extremum` for ``fld`` -- the implementation behind
    ``Field.extremum()``."""
    from ocean_skill.align import (
        _lat_name,
        _lon_name,
        _time_name,
        natural_convention,
        point_of,
    )

    da = fld.data
    if point_of(da) is not None:
        raise ValueError(
            f"{fld.source!r} has already been reduced to one place "
            f"({fld.family_reason}), so there is no spatial extremum to locate "
            "-- an extremum of one cell is the value .plot() already shows. "
            "Widen select= to keep a horizontal extent."
        )

    indices = _locate(da, kind, source=fld.source)
    point = da.isel(indices)

    lon_name, lat_name = _lon_name(point), _lat_name(point)
    lon = float(point[lon_name]) if lon_name is not None else None
    lat = float(point[lat_name]) if lat_name is not None else None

    time_name = _time_name(point)
    time = (
        point[time_name].values.item()
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
        indices=indices,
        coords=coords,
        time=time,
        time_reason=_time_reason(time, fld.select),
        grid="the source's own grid",
        _parent=fld,
    )
