"""Property-property plots: one variable against another, per data source and region.

A T-S diagram is the famous case, but the shape is general -- nitrate against phosphate,
alkalinity against salinity, DIC against alkalinity. :class:`XY` takes *members* (a
label for each data source, holding the ordinary :func:`ocean_skill.field` results that
carry the two variables) and draws ``y`` against ``x`` for each, one panel per region.
:class:`TS` is the preset with salinity on x and temperature on y that also draws
density contours by default.

The marks follow the data rather than an option. A member whose two variables sit at
one position with only a vertical axis left -- a profile from a nearest-cell sample, or
a box mean -- draws as a depth-ordered line. Anything else (every cell, level and
snapshot of a model box) draws as dots.

Everything here goes through :func:`ocean_skill.comparison.prepare_source` like any
other :class:`~ocean_skill.field.Field`, so a member prepared for an :class:`XY` and the
same field prepared on its own share one cache entry. Nothing is read until the figure
is asked for: constructing an :class:`XY` only checks its arguments. It deliberately
does not go through :meth:`Field.plot <ocean_skill.field.Field.plot>` or
:attr:`Field.family <ocean_skill.field.Field.family>`, whose layout rules (a ROMS
``(time, s, eta, xi)`` member has no facet to name; a one-step WOA time reads as a
time-depth panel) answer a different question.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from ocean_skill import _stacklevel, align, comparison, operators
from ocean_skill._docs import graft_from
from ocean_skill.field import Field, FieldSet

__all__ = ["TS", "XY", "normalize_regions"]

#: Every spelling of a horizontal key in a ``select`` -- what ``regions=`` replaces.
_HORIZONTAL_KEYS = operators._POINT_LON_KEYS | operators._POINT_LAT_KEYS

#: The two roles a density contour needs (and the only ones that mean something to it).
_ROLES = ("salinity", "temperature")


def normalize_regions(regions: Mapping[str, Any] | None) -> dict[str, dict] | None:
    """Validate ``regions`` (``{name: select}``) and record each box and its centre.

    Parameters
    ----------
    regions
        ``dict[str, dict] | None`` -- region name -> a horizontal ``select``: a lon/lat
        box (``{"lon": {"min": 155.24, "max": 156.33}, "lat": {"min": 20.51, "max":
        21.60}}``) or a point (``{"lon": 155.79, "lat": 21.05}``). ``None`` passes
        through. Only horizontal keys are allowed -- a region says *where*, and
        ``time``/``depth`` belong to each member's own ``select``.

    Returns
    -------
    ``None``, or ``{name: {"select", "bbox", "centre"}}`` in the order given: ``select``
    is the region's own dict, ``bbox`` is ``(lon_min, lat_min, lon_max, lat_max)`` for a
    box (``None`` for a point), and ``centre`` is ``(lon, lat)`` -- the box midpoint,
    kept contiguous across the antimeridian (``{"min": 170, "max": -170}`` is centred
    on 180, and a 0-360 box stays in 0-360) -- or the point itself.

    Pure: reads no data, so a suite can check it before anything is loaded. Raises
    ``TypeError`` for a malformed container and ``ValueError`` for a region that is
    neither a box nor a point.
    """
    if regions is None:
        return None
    if not isinstance(regions, Mapping):
        raise TypeError(
            f"regions must be a dict of name -> select, got {regions!r}. For example "
            'regions={"North West Pacific": {"lon": {"min": 155.24, "max": 156.33}, '
            '"lat": {"min": 20.51, "max": 21.60}}}.'
        )
    if not regions:
        raise ValueError(
            "regions={} names no region to draw. Pass at least one, or leave "
            "regions=None to use each member's own select."
        )
    out: dict[str, dict] = {}
    for name, spec in regions.items():
        if not isinstance(name, str) or not name:
            raise TypeError(
                f"region names must be non-empty strings (they title the panels), "
                f"got {name!r}."
            )
        select = comparison.as_select(spec)
        extra = sorted(str(k) for k in select if k not in _HORIZONTAL_KEYS)
        if extra:
            raise ValueError(
                f"region {name!r} holds non-horizontal key(s) {extra}: a region only "
                "says where (lon/lat). Put time and depth in each member's own select."
            )
        n_lon = sum(k in operators._POINT_LON_KEYS for k in select)
        n_lat = sum(k in operators._POINT_LAT_KEYS for k in select)
        box = operators.box_in_spec(select) if (n_lon, n_lat) == (1, 1) else None
        point = operators.point_in_spec(select) if (n_lon, n_lat) == (1, 1) else None
        if box is not None:
            _, _, (lon_lo, lon_hi), (lat_lo, lat_hi) = box
            out[name] = {
                "select": select,
                "bbox": (lon_lo, lat_lo, lon_hi, lat_hi),
                "centre": (0.5 * (lon_lo + lon_hi), 0.5 * (lat_lo + lat_hi)),
            }
        elif point is not None:
            _, _, lon, lat = point
            out[name] = {"select": select, "bbox": None, "centre": (lon, lat)}
        else:
            raise ValueError(
                f"region {name!r} is {select!r}, which is neither a box nor a point. "
                'A box gives lon and lat each as a range ({"lon": {"min": 155.24, '
                '"max": 156.33}, "lat": {"min": 20.51, "max": 21.60}}); a point gives '
                'both as numbers ({"lon": 155.79, "lat": 21.05}).'
            )
    return out


def _normalize_members(members: Any) -> dict[str, list[Field]]:
    """``{label: [Field, ...]}`` from the ways a caller may hand members over."""
    if not isinstance(members, Mapping):
        raise TypeError(
            f"members must be a dict of label -> Field (or FieldSet, or list of "
            f"Fields), got {members!r}. For example "
            '{"ROMS": osk.field("run", ["temperature", "salinity"], select=box)}.'
        )
    if not members:
        raise ValueError("members is empty: there is nothing to draw.")
    out: dict[str, list[Field]] = {}
    for label, member in members.items():
        if not isinstance(label, str):
            raise TypeError(
                f"member labels must be strings (they label the legend), got {label!r}."
            )
        if isinstance(member, Field):
            fields = [member]
        elif isinstance(member, FieldSet):
            fields = list(member.fields)
        elif isinstance(member, (list, tuple)) and all(
            isinstance(f, Field) for f in member
        ):
            fields = list(member)
        else:
            raise TypeError(
                f"member {label!r} is {member!r}: expected a Field, a FieldSet, or a "
                "list of Fields -- the result of osk.field()."
            )
        if not fields:
            raise ValueError(f"member {label!r} holds no fields.")
        out[label] = fields
    return out


def _same_variable(x: Any, y: Any) -> bool:
    """Whether two variable specs name one variable (``"temp"``, ``"temperature"``)."""
    if isinstance(x, str) and isinstance(y, str):
        from ocean_skill.vocabulary import resolve_name

        return resolve_name(x) == resolve_name(y)
    return comparison._canonical(x) == comparison._canonical(y)


def _role(field: Field) -> str | None:
    """``"salinity"`` or ``"temperature"`` when ``field`` is one, else ``None``."""
    for role in _ROLES:
        if comparison._variable_matches(field.standard_name, field.variable, role):
            return role
    return None


def _described(fields: Iterable[Field]) -> str:
    """``"woa_t temperature; woa_s salinity"`` -- what a list of fields is."""
    return "; ".join(
        f"{f.source} {comparison._short_variable_label(f.variable)}" for f in fields
    )


def _wrap_lon(lon: float) -> float:
    """``lon`` in the ±180 convention (a contiguous 185.2 reads as -174.8)."""
    return ((float(lon) + 180.0) % 360.0) - 180.0


def _centre_of(da) -> tuple[float, float] | None:
    """Return the ``(lon, lat)`` a prepared array stands at: box midpoint or point."""
    box = da.attrs.get("region")
    if box is not None:
        lon_min, lat_min, lon_max, lat_max = box
        return 0.5 * (float(lon_min) + float(lon_max)), 0.5 * (
            float(lat_min) + float(lat_max)
        )
    return align.point_of(da)


def _note_for(bbox: Any, centre: tuple[float, float] | None) -> str | None:
    """Return a panel's place: a box (``"20.5–21.6°N, 155°E–156°E"``) or a point."""
    if bbox is not None:
        return comparison._region_label(bbox)
    if centre is not None:
        from ocean_skill.plot.series import lonlat_label

        return lonlat_label(_wrap_lon(centre[0]), centre[1])
    return None


def _spread(var: xr.Variable, like) -> np.ndarray | None:
    """``var``'s values broadcast onto ``like``'s dims and raveled, or ``None``.

    ``None`` when ``var`` carries a dimension ``like`` does not -- a coordinate that
    outlived a reduction of the data it rode on says nothing per point.
    """
    if not set(var.dims) <= set(like.dims):
        return None
    sizes = dict(zip(like.dims, like.shape, strict=True))
    return np.asarray(var.set_dims(sizes).values).ravel()


def _depth_variable(da, zdim: str | None) -> xr.Variable | None:
    """Positive-down depth riding on ``da``'s dims, or ``None`` if nothing carries it.

    In order: the vertical coordinate the data's own axis carries
    (:func:`ocean_skill.operators.vertical_coord_on`), ROMS's ``z_rho``, a scalar
    vertical coordinate (a single level), and last the ``dz`` cell-weight coordinate
    (:data:`ocean_skill.roms.WEIGHT_COORD`). The last is the one that survives a time
    average on ROMS, which drops ``z_rho`` with the time axis it rode on and keeps only
    the thickness of each cell: depth is then rebuilt from the cumulative thickness
    above each cell's centre, assuming a flat sea surface (``zeta = 0``) and s-levels
    ordered bottom to top, as ROMS stores them. Approximate by that much, and enough for
    ordering a profile and colouring by depth.
    """
    from ocean_skill.cf import find_coord
    from ocean_skill.plot.profile import positive_down
    from ocean_skill.roms import WEIGHT_COORD

    coord = operators.vertical_coord_on(da, zdim) if zdim else None
    if coord is None:
        coord = da.coords.get("z_rho")
    if coord is None:
        found = find_coord(da, "vertical")
        coord = found if found is not None and found.ndim == 0 else None
    if coord is not None:
        return xr.Variable(coord.dims, positive_down(coord))
    dz = da.coords.get(WEIGHT_COORD)
    if zdim and dz is not None and zdim in dz.dims:
        reverse = {zdim: slice(None, None, -1)}
        from_top = dz.isel(reverse).cumsum(zdim).isel(reverse)
        return (from_top - dz / 2.0).variable
    return None


def _xy_arrays(xda, yda) -> dict[str, Any]:
    """Return the paired, finite, flat arrays one member draws, and its mark.

    ``{"mark", "x", "y", "depth", "time"}``: ``mark`` is ``"line"`` when the two
    variables sit at one position with at most the vertical axis standing (sorted by
    depth, shallow first), else ``"points"``; ``depth`` (positive-down metres) and
    ``time`` are per-point arrays, or ``None`` where the data carries none.

    Size-1 dimensions are squeezed first (WOA's one-step time, a one-cell box), so the
    marks depend on what is left, not on how it was read. Raises ``ValueError`` when
    the two arrays do not share one shape and ``align.NoValidData`` when no point has
    both values.
    """
    from ocean_skill.cf import find_coord

    xda, yda = xda.squeeze(), yda.squeeze()
    mismatch = (
        f"x and y do not line up (x is {dict(xda.sizes)!r}, y is {dict(yda.sizes)!r}): "
        "use the same select/aggregate for both variables, so each point of x has a "
        "y to go with it."
    )
    if set(xda.dims) != set(yda.dims):
        raise ValueError(mismatch)
    try:
        xda, yda = xr.align(xda, yda, join="exact")
    except ValueError as exc:
        raise ValueError(mismatch) from exc
    yda = yda.transpose(*xda.dims)

    zdim = operators.resolve_dim(xda, "Z")
    vertical_only = set(xda.dims) <= ({zdim} if zdim else set())
    mark = "line" if vertical_only and align.point_of(xda) is not None else "points"

    x = np.asarray(xda.values, dtype="float64").ravel()
    y = np.asarray(yda.values, dtype="float64").ravel()
    depth_var = _depth_variable(xda, zdim)
    depth = None if depth_var is None else _spread(depth_var, xda)
    tcoord = find_coord(xda, "time")
    time = None
    if tcoord is not None and np.issubdtype(tcoord.dtype, np.datetime64):
        time = _spread(tcoord.variable, xda)

    keep = np.flatnonzero(np.isfinite(x) & np.isfinite(y))
    if keep.size == 0:
        raise align.NoValidData(
            "no point has both an x and a y value (every one is masked or missing)."
        )
    if mark == "line" and depth is not None:
        keep = keep[np.argsort(depth[keep].astype("float64"), kind="stable")]
    return {
        "mark": mark,
        "x": x[keep],
        "y": y[keep],
        "depth": None if depth is None else depth[keep].astype("float64"),
        "time": None if time is None else time[keep],
    }


def _axis_meta(field: Field, da) -> dict[str, Any]:
    """``name``, ``units`` and ``standard_name`` of one axis, as an item reports them.

    The standard_name is the prepared array's own where it states one (so a source that
    ships in-situ temperature reports that, not the potential temperature the request
    resolved to) and falls back to the field's request.
    """
    units = da.attrs.get("units")
    standard = da.attrs.get("standard_name") or field.standard_name
    return {
        "name": comparison._short_variable_label(field.variable),
        "units": str(units) if units else None,
        "standard_name": str(standard) if standard else None,
    }


class XY:
    """One variable against another, per data source ("member") and per region.

    Parameters
    ----------
    members
        ``dict[str, Field | FieldSet | list[Field]]`` -- label -> what that data source
        holds: the result of :func:`ocean_skill.field` (a :class:`~ocean_skill.field.
        FieldSet` when it fanned over variables and/or sources). The label names the
        member in the legend and in ``colors=``. Each member needs exactly one field
        carrying ``x`` and one carrying ``y``; a field whose source does not carry its
        variable is dropped with a warning, which is how a list of sources crossed
        with a list of variables (``osk.field([temp_src, salt_src], ["temperature",
        "salinity"])``) pairs up.
    x, y
        Variable specs, as in :func:`ocean_skill.field` -- a name or alias
        (``"salinity"``, ``"temp"``, ``"nitrate"``), matched through the vocabulary.
        They must differ.
    regions
        ``dict[str, dict] | None`` -- one panel per entry, each a lon/lat box or point
        (see :func:`normalize_regions`). It *replaces* every member's own horizontal
        select, so one set of members is drawn in every region. ``None`` (default)
        keeps each member's select, and then each must name a box or a point of its
        own: a member with neither would load its whole domain, every level and time
        step.
    at_center
        ``list[str]`` -- labels of members to sample at the nearest cell to each
        region's centre instead of cropping to its box. For a source whose grid is
        coarser than the box (a 1-degree product against a ~100 km box can hold no cell
        centre at all, and cropping then raises). Needs ``regions=``.

    Nothing is read here. :meth:`plot` reads each member and draws it; :attr:`data`
    returns what was read.

    Example
    -------
    ::

        box = {
            "lon": {"min": 155.24, "max": 156.33},
            "lat": {"min": 20.51, "max": 21.60},
        }
        osk.XY(
            {"ROMS": osk.field("all_the_rest", ["nitrate", "phosphate"], select=box)},
            x="phosphate", y="nitrate",
        ).plot(color_by="depth")
    """

    def __init__(
        self,
        members: Mapping[str, Field | FieldSet | list[Field]],
        *,
        x: Any,
        y: Any,
        regions: Mapping[str, dict] | None = None,
        at_center: Iterable[str] = (),
    ):
        self.members = _normalize_members(members)
        for axis, spec in (("x", x), ("y", y)):
            if not isinstance(spec, (str, dict)):
                raise TypeError(
                    f"{axis}= must be a variable name or spec, as in osk.field(), "
                    f"got {spec!r}."
                )
        if _same_variable(x, y):
            raise ValueError(
                f"x={x!r} and y={y!r} are the same variable: a property-property plot "
                "needs two different ones."
            )
        self.x, self.y = x, y
        self.regions = normalize_regions(regions)
        if isinstance(at_center, str) or not isinstance(at_center, Iterable):
            raise TypeError(
                f"at_center must be a list of member labels, got {at_center!r}. "
                f"Known members: {list(self.members)}."
            )
        self.at_center = list(dict.fromkeys(at_center))
        unknown = [m for m in self.at_center if m not in self.members]
        if unknown:
            raise ValueError(
                f"at_center names {unknown}, not among the members "
                f"{list(self.members)}."
            )
        if self.at_center and self.regions is None:
            raise ValueError(
                "at_center samples each region's centre, so it needs regions= -- "
                "without regions, give that member a point in its own select."
            )
        self._pairs: dict[str, tuple[Field, Field]] | None = None
        self._located: dict[tuple[str | None, str], tuple[Field, Field]] = {}

    # -- members -> (x field, y field) ------------------------------------------------

    def pairs(self) -> dict[str, tuple[Field, Field]]:
        """Return ``{label: (x field, y field)}``: each member's two fields.

        Drops fields whose source does not carry the requested variable (a warning per
        member, worded like :meth:`ocean_skill.field.FieldSet.plot`'s), then picks the
        one field matching ``x`` and the one matching ``y`` through the vocabulary.
        Raises ``ValueError`` for a member that has no such pair or an ambiguous one,
        saying what it does hold. Probes each source cheaply (no data is loaded) and
        remembers the answer.
        """
        if self._pairs is None:
            pairs = {
                label: self._pair(label, fields)
                for label, fields in self.members.items()
            }
            self._check_locations(pairs)
            self._pairs = pairs
        return self._pairs

    def _pair(self, label: str, fields: list[Field]) -> tuple[Field, Field]:
        usable = [
            f
            for f in fields
            if comparison._variable_available(
                f.source, f.variable, select=f.select, qc=f.qc
            )
        ]
        missing = [f for f in fields if f not in usable]
        if missing:
            warnings.warn(
                f"skipping {len(missing)} field(s) whose source doesn't carry the "
                f"requested variable: {_described(missing)}",
                stacklevel=_stacklevel.find(),
            )
        xs = [
            f
            for f in usable
            if comparison._variable_matches(f.standard_name, f.variable, self.x)
        ]
        ys = [
            f
            for f in usable
            if comparison._variable_matches(f.standard_name, f.variable, self.y)
        ]
        if len(xs) != 1 or len(ys) != 1 or xs[0] is ys[0]:
            have = _described(usable) or "no usable field"
            raise ValueError(
                f"member {label!r} needs exactly one field for x={self.x!r} and one "
                f"for y={self.y!r}, but it has {have} ({len(xs)} match x, {len(ys)} "
                f"match y). Build it from osk.field(source, [{self.y!r}, {self.x!r}]) "
                "(or a list of the two fields it should use)."
            )
        return xs[0], ys[0]

    def _check_locations(self, pairs: dict[str, tuple[Field, Field]]) -> None:
        """Refuse a member with no place when nothing supplies one; note what is cut."""
        for label, fields in pairs.items():
            if self.regions is None:
                for f in fields:
                    box = operators.box_in_spec(f.select)
                    if box is None and operators.point_in_spec(f.select) is None:
                        raise ValueError(
                            f"member {label!r} ({f.source}) has no lon/lat box or "
                            "point in its select, so drawing it would load the whole "
                            "domain, every level and time step. Give it select="
                            "{'lon': {'min': ..., 'max': ...}, 'lat': {'min': ..., "
                            "'max': ...}} (or a point), or pass regions=."
                        )
                continue
            own = sorted(
                {str(k) for f in fields for k in f.select if k in _HORIZONTAL_KEYS}
            )
            if own:
                warnings.warn(
                    f"regions= replaces member {label!r}'s own horizontal select "
                    f"({', '.join(own)}): every region is drawn from its own box.",
                    stacklevel=_stacklevel.find(),
                )

    def _located_pair(self, region: str | None, label: str) -> tuple[Field, Field]:
        """Return a member's two fields rebuilt for one region (as given, with none)."""
        key = (region, label)
        if key not in self._located:
            pair = self.pairs()[label]
            if region is not None:
                spec = self.regions[region]
                pair = tuple(
                    self._relocate(f, spec, label in self.at_center) for f in pair
                )
            self._located[key] = pair
        return self._located[key]

    @staticmethod
    def _relocate(field: Field, region: dict, at_center: bool) -> Field:
        """``field`` with its horizontal select replaced by ``region``'s."""
        select = {k: v for k, v in field.select.items() if k not in _HORIZONTAL_KEYS}
        aggregate = field.aggregate
        if at_center:
            lon, lat = region["centre"]
            select.update({"lon": lon, "lat": lat})
            aggregate = comparison._without_horizontal_mean(aggregate) or None
        else:
            select.update(region["select"])
        return field._replace(select=select, aggregate=aggregate)

    def _region_names(self) -> list[str | None]:
        return list(self.regions) if self.regions else [None]

    def _member_data(self, region: str | None, label: str):
        """``(x array, y array)`` as prepared for one region and member."""
        xf, yf = self._located_pair(region, label)
        return xf.data, yf.data

    # -- data and items ---------------------------------------------------------------

    @property
    def data(self) -> dict[str | None, dict[str, dict[str, xr.DataArray]]]:
        """``{region: {label: {"x": DataArray, "y": DataArray}}}``, as prepared.

        ``region`` is ``None`` when there are no ``regions=``. The arrays are what each
        field's own :attr:`~ocean_skill.field.Field.data` holds -- before the squeeze,
        pairing and finite-filtering :meth:`plot` applies -- and are read on first
        access. A member that cannot be read raises here; :meth:`plot` drops it with a
        warning instead.
        """
        self.pairs()
        out: dict[str | None, dict[str, dict[str, xr.DataArray]]] = {}
        for region in self._region_names():
            out[region] = {}
            for label in self.members:
                xda, yda = self._member_data(region, label)
                out[region][label] = {"x": xda, "y": yda}
        return out

    def _items(self) -> list[dict[str, Any]]:
        """One item per (region, member), ordered by region then member.

        A pair the data cannot give -- a variable the source lacks, a box with no cell
        in it, a point on a masked cell, no point with both values -- is dropped with a
        warning and the rest are drawn. ``ValueError`` when nothing is left.
        """
        self.pairs()
        items: list[dict[str, Any]] = []
        for region in self._region_names():
            spec = self.regions[region] if region is not None else None
            group: list[dict[str, Any]] = []
            own: list[tuple[tuple[float, float] | None, Any]] = []
            for label in self.members:
                try:
                    xda, yda = self._member_data(region, label)
                    arrays = _xy_arrays(xda, yda)
                except (KeyError, align.NoValidData, align.EmptySelection) as exc:
                    self._warn_dropped(region, label, exc)
                    continue
                xf, yf = self._located_pair(region, label)
                xmeta, ymeta = _axis_meta(xf, xda), _axis_meta(yf, yda)
                sources = dict.fromkeys(
                    str(getattr(f.source, "name", f.source)) for f in (xf, yf)
                )
                group.append(
                    {
                        "label": label,
                        "region": region,
                        "mark": arrays["mark"],
                        "x": arrays["x"],
                        "y": arrays["y"],
                        "depth": arrays["depth"],
                        "time": arrays["time"],
                        "x_name": xmeta["name"],
                        "x_units": xmeta["units"],
                        "x_standard_name": xmeta["standard_name"],
                        "y_name": ymeta["name"],
                        "y_units": ymeta["units"],
                        "y_standard_name": ymeta["standard_name"],
                        "x_role": _role(xf),
                        "y_role": _role(yf),
                        "lon": None,
                        "lat": None,
                        "region_note": None,
                        "source": " + ".join(sources),
                    }
                )
                own.append((_centre_of(xda), xda.attrs.get("region")))
            # The panel's centre and note are shared by every member in it: the region's
            # own where there is one, else the first member that stands at a place.
            centre = spec["centre"] if spec else next((c for c, _ in own if c), None)
            bbox = (
                spec["bbox"]
                if spec
                else next((b for _, b in own if b is not None), None)
            )
            note = _note_for(bbox, centre)
            for item in group:
                if centre is not None:
                    item["lon"], item["lat"] = _wrap_lon(centre[0]), float(centre[1])
                item["region_note"] = note
            items.extend(group)
        if not items:
            raise ValueError(
                "nothing to draw: no member gave a point with both an x and a y value "
                "(see the warnings above for each one)."
            )
        return items

    def _warn_dropped(self, region: str | None, label: str, exc: Exception) -> None:
        """Say why one (region, member) pair is not on the figure."""
        where = f" in {region!r}" if region is not None else ""
        reason = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
        hint = ""
        if isinstance(exc, align.EmptySelection) and label not in self.at_center:
            hint = (
                " The grid may be coarser than the box: pass "
                + (
                    f"at_center=[{label!r}]"
                    if self.regions is not None
                    else f"regions= with at_center=[{label!r}]"
                )
                + " to sample its nearest cell to the centre instead."
            )
        warnings.warn(
            f"skipping member {label!r}{where}: {reason}{hint}",
            stacklevel=_stacklevel.find(),
        )

    # -- output ---------------------------------------------------------------------

    @graft_from(
        "plot/matplotlib_renderer.py",
        "xy",
        label="ocean_skill.plot.matplotlib_renderer.xy",
    )
    def plot(self, *, renderer: str = "matplotlib", **opts: Any):
        """Draw ``y`` against ``x`` for every member, one panel per region.

        Parameters
        ----------
        renderer
            One of ``"matplotlib"`` (default, static) or ``"holoviews"`` (interactive).
        **opts
            Plot options forwarded untouched to the renderer: ``title``,
            ``annotations``, ``density``, ``color_by``, ``colors``, ``ncols``, ``xlim``/
            ``ylim``, ``marker_size``, ``alpha``, ``save`` and the ``*_kwargs`` styling
            dicts -- every one is documented under "Forwarded keyword
            arguments" below, and in ``docs/plot_styling_reference.md``.

        Reads each member the first time it is needed. A member that cannot be read is
        dropped with a warning (see :meth:`_items`); ``ValueError`` if none can.
        """
        from ocean_skill.plot.registry import render
        from ocean_skill.plot.spec import PlotSpec

        return render(PlotSpec("XY", self._items(), opts), renderer=renderer)

    def save(
        self,
        project: str | None = None,
        *,
        stem: str | None = None,
        renderer: str = "matplotlib",
        **plot_kwargs: Any,
    ) -> dict[str, Path]:
        """Write this figure under ``output/<project>/figures/``.

        Parameters
        ----------
        project
            ``str | None`` -- the output project name, used to build
            ``output/<project>/figures/``. ``None`` (default) uses the first member's
            source.
        stem
            ``str | None`` -- the figure's filename stem (before ``.png``). ``None``
            (default) is ``"<y>_vs_<x>"``, truncated to 24 characters.
        renderer
            One of ``"matplotlib"`` (default) or ``"holoviews"``, forwarded to
            :meth:`plot`.
        **plot_kwargs
            Forwarded to :meth:`plot`.

        The same layout :meth:`ocean_skill.field.Field.save` writes to.
        """
        from ocean_skill import outputs

        first = next(iter(self.members.values()))[0]
        label = comparison._short_variable_label
        stem = stem or f"{label(self.y)}_vs_{label(self.x)}"[:24]
        path = outputs.figures_dir(project or str(first.source)) / f"{stem}.png"
        self.plot(renderer=renderer, save=path, **plot_kwargs)
        return {"figure": path}

    def map_locations(self, *, renderer: str = "matplotlib", **kwargs: Any):
        """Map where this plot's members sit: each region's box or point, deduped.

        Parameters
        ----------
        renderer
            One of ``"matplotlib"`` (default) or ``"holoviews"``.
        **kwargs
            Plot options forwarded to
            :func:`ocean_skill.plot.map_locations.map_locations`.

        From each member's request alone -- nothing is read. See
        :meth:`ocean_skill.field.FieldSet.map_locations`.
        """
        fields = [
            f
            for region in self._region_names()
            for label in self.members
            for f in self._located_pair(region, label)
        ]
        return FieldSet(fields).map_locations(renderer=renderer, **kwargs)

    def __repr__(self) -> str:
        regions = list(self.regions) if self.regions else None
        return (
            f"{type(self).__name__}(members={list(self.members)!r}, x={self.x!r}, "
            f"y={self.y!r}, regions={regions!r}, at_center={self.at_center!r})"
        )


class TS(XY):
    """A temperature-salinity diagram: :class:`XY` with salinity on x, temperature on y.

    Parameters
    ----------
    members, regions, at_center
        As for :class:`XY`.

    The one difference in use: :meth:`plot` draws density (sigma-0) contours by default,
    ``density=False`` turns them off. Contours need gsw, which treats every temperature
    as potential; members that differ (WOA's in-situ against a model's potential
    temperature) are flagged with a warning, not converted.
    """

    def __init__(
        self,
        members: Mapping[str, Field | FieldSet | list[Field]],
        *,
        regions: Mapping[str, dict] | None = None,
        at_center: Iterable[str] = (),
    ):
        super().__init__(
            members,
            x="salinity",
            y="temperature",
            regions=regions,
            at_center=at_center,
        )

    @graft_from(
        "plot/matplotlib_renderer.py",
        "xy",
        label="ocean_skill.plot.matplotlib_renderer.xy",
    )
    def plot(self, *, density: Any = True, renderer: str = "matplotlib", **opts: Any):
        """Draw temperature against salinity, with density contours by default.

        Parameters
        ----------
        density
            ``True`` (default) draws sigma-0 contours at automatic levels; an int or a
            list of levels sets them; ``False`` draws none.
        renderer
            One of ``"matplotlib"`` (default, static) or ``"holoviews"`` (interactive).
        **opts
            As for :meth:`XY.plot`.
        """
        return super().plot(renderer=renderer, density=density, **opts)
