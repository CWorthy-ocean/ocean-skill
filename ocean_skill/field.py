"""One model field, reduced but not compared.

Everything else in this package is built around a pair — a test source, a reference
source, and the difference between them. A :class:`Field` is the other half of the
same pipeline: one source, reduced by the same :mod:`~ocean_skill.operators` grammar,
stopping short of the regrid that only exists to bring two grids together.

Its reason to exist is the *faceted* case. ``aggregate={"time": "mean"}`` collapses a
run to one map, which a comparison can use; ``{"time": {"resample": "1MS", "reduce":
"mean"}}`` leaves a month axis standing, which a comparison explicitly cannot (see
:func:`ocean_skill.align._require_2d`) but which is exactly what "show me each month
of this run in order" means. So the axis a comparison treats as an error is the one
this class treats as the payload, and the panels come out of it.

A ``select`` that narrows both horizontal axes to one position is the other shape a
reduction can take: nothing left to lay out as columns, so it draws as a line over
whatever axis survives instead of panels (:attr:`Field.family`) — one source, no
reference, the model-only counterpart of a :class:`~ocean_skill.comparison.Comparison`
whose reference is a station.

The reduction, the caching and the vocabulary handling are all the comparison lane's,
reached through :func:`ocean_skill.comparison.prepare_source` — a model field prepared
for a comparison and the same field prepared on its own are the same field, and they
share one cache entry *whenever the select names the same thing*. Vertically they
often do not: a :class:`~ocean_skill.comparison.Comparison` with no vertical select
still means the surface, its own default; a bare :class:`Field` means "leave the
vertical axis alone" — an unset select is not a shared cache entry between the two,
though an explicit one (``select={"depth": ...}``) always is.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ocean_skill._docs import graft_plot_options

__all__ = ["Cross", "Field", "FieldSet", "field"]


def _facet_dims(da, spatial_dims: set[str]) -> tuple[str | None, str | None]:
    """Return ``(row_dim, col_dim)``: the non-spatial axes that become the panels.

    What is left standing after the reduction *is* what varies across the panels —
    there is no need to be told which axis it is, and asking the caller would let the
    answer disagree with the data. ``(None, None)`` is a single map, one axis is a
    series, two are a grid.

    **Which of two goes down the rows is a convention, not a measurement.** The
    vertical axis does, because depth reads top-to-bottom with the surface at the top,
    and time then reads left-to-right along each row. Deliberately *not* chosen by
    which arrangement fits the page better: with one axis there is no meaning in the
    fold and the aspect ratio should decide it (see
    :func:`~ocean_skill.plot.typography.facet_layout`), but here both directions carry
    meaning and geometry must not overrule it. Pass ``row_dim`` to
    :func:`~ocean_skill.plot.matplotlib_renderer.field_facet` directly to swap them.

    Three or more axes are refused rather than guessed at: a figure is two-dimensional,
    so something would have to be silently averaged over or dropped.
    """
    from ocean_skill.operators import resolve_dim

    extra = [str(d) for d in da.dims if d not in spatial_dims]
    if not extra:
        return None, None
    if len(extra) == 1:
        return None, extra[0]
    if len(extra) > 2:
        raise ValueError(
            f"the field has {extra} beyond its horizontal axes, and a figure has only "
            "rows and columns. Collapse all but two with aggregate= (e.g. "
            '{"Z": "mean"}) or narrow them with select=.'
        )
    vertical = resolve_dim(da, "Z")
    if vertical in extra:
        return vertical, next(d for d in extra if d != vertical)
    return extra[0], extra[1]


def _labelless_vertical(da) -> str | None:
    """Return the vertical dim name if it stands with no coordinate to label it.

    A bare native ``s_rho``/``s_w`` dimension -- ROMS ships no coordinate of its own
    for it -- reaches a figure at all only when nothing was named vertically (see
    :func:`ocean_skill.comparison._prepare`'s ``surface`` flag, the default a bare
    :func:`field` leaves in force): unlabeled index positions in a legend or a facet
    row title would say nothing about depth and actively mislead about what varies
    from one to the next. A 1-D ``z_rho`` coordinate on that same dimension is a real
    label instead — the same fallback :func:`ocean_skill.plot.profile._vertical_coord`
    already uses for a profile down native levels — and is not "labelless" here.
    Anything else (an observational product's own reported ``depth``, say -- even one
    riding under a different name than its dimension, see
    :func:`ocean_skill.operators.vertical_coord_on`) already carries a coordinate and
    never triggers this, whatever facet size it draws as.
    """
    from ocean_skill.operators import resolve_dim, vertical_coord_on

    zdim = resolve_dim(da, "Z")
    if zdim is None or zdim not in da.dims:
        return None
    if vertical_coord_on(da, zdim) is not None:
        return None
    z_rho = da.coords.get("z_rho")
    if z_rho is not None and z_rho.ndim == 1 and zdim in z_rho.dims:
        return None
    return zdim


def _facet_dims_of(da) -> tuple[str | None, str | None]:
    """``(row_dim, col_dim)`` for an arbitrary field.

    :attr:`Field.facet_dims`'s own logic, extracted so it can be read off a field
    other than ``self.data`` (a reduced top-level slice, most often -- see
    :meth:`Field._facet_field_and_depth`/:meth:`Field._facet_item`).
    """
    from ocean_skill.align import _lat_name, _lon_name

    lon, lat = _lon_name(da), _lat_name(da)
    spatial: set[str] = set()
    for name in (lon, lat):
        if name is not None:
            spatial |= {str(d) for d in da[name].dims}
    return _facet_dims(da, spatial)


def _grid_has_vertical_axis(meta: dict[str, Any]) -> bool:
    """Whether a catalog entry's own metadata declares a vertical axis at all.

    Read-free -- three ways :func:`ocean_skill.build`'s probes record one:
    ``vertical`` (a ROMS entry's s-coordinate parameters, written by
    ``_roms_metadata``), ``axes["Z"]`` (any entry's declared vertical column/
    dimension), or ``vertical_levels`` (an observational grid's own level count).
    Used by :meth:`Field._grid_metadata_if_eligible`'s caller to decide whether a
    bare vertical select is worth redirecting to the surface before anything is
    read -- a source with none of these has nothing to default *away from*.
    """
    return bool(
        meta.get("vertical")
        or (meta.get("axes") or {}).get("Z")
        or meta.get("vertical_levels")
    )


def _top_level(da, zdim: str, *, source: str):
    """Reduce ``da``'s vertical axis ``zdim`` to its shallowest level.

    The fallback half of the grid surface default (see
    :meth:`Field._facet_field_and_depth`), reached only when the read-free half
    (:meth:`Field._grid_metadata_if_eligible`) could not settle it first --
    an uncatalogued source, or the hand-built stubs tests construct directly.

    A coordinate-bearing axis (an observational product's own reported levels)
    picks the level nearest 0. A coordinate-less native ``s_rho``/``s_w`` axis
    with a same-dim ``z_rho`` (see :func:`_labelless_vertical`) picks the index
    whose ``z_rho`` reads shallowest -- averaged over any other dimension
    ``z_rho`` might carry (a curvilinear grid's own eta/xi), so one index names
    the top level everywhere a facet panel might be drawn, not just at a single
    column. Anything else -- coordinate-less with no ``z_rho`` either -- has no
    way to tell top from bottom and is refused, the same message
    :meth:`Field._series_items`/:meth:`Field._refuse_labelless_facet` give the
    same shape elsewhere.

    Returns ``(field, "surface")`` -- the depth label a caller attaches to the
    item it builds, matching what an explicit ``select={"depth": "surface"}``
    would report via :func:`ocean_skill.comparison._depth_label`.
    """
    import numpy as np

    from ocean_skill.operators import vertical_coord_on

    coord = vertical_coord_on(da, zdim)
    if coord is not None:
        levels = np.asarray(coord.values, dtype="float64")
        k = int(np.argmin(np.abs(levels)))
        top = da.isel({zdim: k})
        top.attrs["actual_depth"] = float(abs(levels[k]))
        return top, "surface"

    z_rho = da.coords.get("z_rho")
    if z_rho is not None and zdim in z_rho.dims:
        reduce_dims = [d for d in z_rho.dims if d != zdim]
        shallowest = z_rho.mean(dim=reduce_dims) if reduce_dims else z_rho
        k = int(np.asarray(shallowest).argmax())
        top = da.isel({zdim: k})
        top.attrs["actual_depth"] = float(np.abs(np.asarray(top["z_rho"])).mean())
        return top.drop_vars([zdim, "z_rho"], errors="ignore"), "surface"

    raise ValueError(
        f"{source!r}'s native vertical axis ({zdim!r}) carries no coordinate to "
        "pick a surface level from, and no z_rho either. Narrow it with "
        "select= (e.g. {'depth': 'surface'}, a number, a list, or 'column' at "
        'a point) or {\'sigma0\': ...}, or collapse it with aggregate= (e.g. '
        '{"Z": "mean"}).'
    )


class Field:
    """One source reduced to a map, a series of maps over one axis, or a line.

    Parameters
    ----------
    source
        ``str`` -- one catalog entry name (or an already-open
        :class:`~ocean_skill.catalog.SourceRef`). A list belongs to :func:`field`,
        which fans it into a :class:`FieldSet` instead.
    variable
        ``str`` or ``dict`` -- one variable name, or a combination/``calculate``
        spec (see :mod:`ocean_skill.operators`). Not a list (pass one to
        :func:`field` instead) and not a ``{"test", "reference"}`` pair-spec -- a
        ``Field`` has one source and nothing to give the other side to.
    select
        ``dict[str, Any] | None`` -- axis name -> selection, e.g. ``{"depth":
        "surface"}``, ``{"lon": -144.25, "lat": 49.98}`` (see
        :func:`ocean_skill.comparison.as_select`). ``None`` (default) selects
        nothing. Not a ``{"test", "reference"}`` pair-spec.
    aggregate
        ``dict[str, Any] | None`` -- axis name -> reduction, e.g. ``{"time":
        "mean"}``, ``{"time": {"resample": "1MS", "reduce": "mean"}}``, or
        ``{"time": {"groupby": "month"}}``. A spec that fully collapses an axis
        gives one panel; ``groupby``/``resample`` leave the axis standing as the
        facet. ``None`` (default) aggregates nothing. Not a pair-spec.
    label
        ``str | None`` -- legend/title label override. ``None`` (default) uses
        ``source``.
    cache
        ``bool | None`` -- whether to reuse an already-prepared result cached on
        disk. ``None`` (default) follows :func:`ocean_skill.cache.enabled`.
    qc
        Per-call QC override forwarded to :func:`ocean_skill.sources.read` (see
        its own ``qc`` parameter for the full spec): a dict such as ``{"keep":
        ["GOOD", "SUSPECT"]}``, or the string ``"off"``. ``None`` (default) uses
        the entry's own saved QC contract. Not a pair-spec.
    detide
        ``bool | dict | None`` -- ``False``/``None`` (default) leaves this field
        untouched; ``True`` tidal-filters it at PL33's own default cutoff; a
        ``{"T": hours}`` dict sets that cutoff explicitly. See
        :func:`ocean_skill.comparison._normalize_detide_side`.

    Parameters mirror :class:`~ocean_skill.comparison.Comparison` where they mean the
    same thing, so moving between the two is a change of class rather than of grammar.
    ``aggregate`` is the one that matters here: a spec that fully collapses time gives
    a single panel, while ``groupby``/``resample`` leave the axis that becomes the
    panels. A ``select`` that narrows both horizontal axes to one position instead
    draws as a line over whatever axis survives — see :attr:`family`.

    Holds exactly one source and one variable. A list of either belongs to
    :func:`field`, which fans it into a :class:`FieldSet` instead of a single
    ``Field``.
    """

    def __init__(
        self,
        source: str,
        variable: Any,
        *,
        select: dict[str, Any] | None = None,
        aggregate: dict[str, Any] | None = None,
        label: str | None = None,
        cache: bool | None = None,
        qc: Any = None,
        detide: Any = False,
    ):
        from ocean_skill.comparison import (
            _normalize_detide_side,
            _require_pair_spec,
            as_select,
            is_pair_spec,
        )
        from ocean_skill.vocabulary import resolve_and_report

        if isinstance(source, (list, tuple)):
            raise TypeError(
                f"{source!r} is a list of sources, and a Field holds exactly one -- "
                "pass the list to osk.field(), which fans it into a FieldSet (one "
                "Field per source, drawn on one figure), or pass the one source this "
                "Field is for."
            )
        if isinstance(variable, (list, tuple)):
            raise TypeError(
                f"{variable!r} is a list of variable specs, and a Field holds exactly "
                "one -- pass the list to osk.field(), which fans it into a FieldSet "
                "(one Field per variable, drawn on one figure), or pass the one spec "
                "this Field is for."
            )
        if isinstance(variable, dict):
            # A one-sided {"test": ...} names the same mistake a full pair-spec
            # does, and deserves the same clear error rather than surfacing later,
            # confusingly, as "unknown combiner 'test'" out of resolve_variable.
            _require_pair_spec(variable)
        if is_pair_spec(variable):
            raise TypeError(
                f"{variable!r} is a {{'test', 'reference'}} pair-spec, which names two "
                "different recipes for two different lanes -- Field has only one "
                "source and nothing to give the other side to. Pass the one spec this "
                "source actually needs, or use osk.compare() for a pair-spec."
            )
        # select/aggregate/qc carry the same pair-spec spelling in a Comparison, one
        # lane's own select/aggregate/qc -- a Field is one source, so a pair here is
        # the same mistake as a pair-spec variable, and gets the same clear error.
        for name, arg in (("select", select), ("aggregate", aggregate), ("qc", qc)):
            if isinstance(arg, dict):
                _require_pair_spec(arg, kind=name)
            if is_pair_spec(arg):
                raise TypeError(
                    f"{arg!r} is a {{'test', 'reference'}} pair-spec {name}, for "
                    "giving two lanes different selections/aggregations -- Field "
                    f"has only one source and no other lane to give the other side "
                    f"to. Pass the one {name} this source actually needs."
                )
        self.source = source
        self.variable = (
            resolve_and_report(variable, context="Field variable=")
            if isinstance(variable, str)
            else variable
        )
        self.select = as_select(select)
        self.aggregate = aggregate
        self.label = label
        self.cache = cache
        self.qc = qc
        # A single lane, unlike Comparison's per-lane {"test": ..., "reference": ...}
        # -- {"T": hours} to detide, or None -- see _normalize_detide_side and
        # ocean_skill.comparison._prepare's detide= paragraph.
        self.detide = _normalize_detide_side(detide)
        self._data = None
        self._actual_depth = None

    @property
    def standard_name(self) -> str | None:
        """The CF name this field represents, for colormaps and labels."""
        from ocean_skill.operators import DERIVED

        spec = self.variable
        if isinstance(spec, str):
            spec = DERIVED.get(spec, spec)
        return spec if isinstance(spec, str) else spec.get("standard_name")

    def _use_cache(self) -> bool:
        from ocean_skill import cache as _cache

        return _cache.enabled() if self.cache is None else self.cache

    def prepare(self, *, refresh: bool = False):
        """Read the source and reduce it; cached on disk like a comparison lane."""
        from ocean_skill.comparison import prepare_source

        da, depth = prepare_source(
            self.source,
            self.variable,
            self.select,
            self.aggregate,
            use_cache=self._use_cache(),
            refresh=refresh,
            qc=self.qc,
            detide=self.detide,
        )
        if da is None:
            raise KeyError(f"{self.variable!r} not available in {self.source!r}")
        self._data = da
        self._actual_depth = depth
        return da

    @property
    def data(self):
        """The prepared field; computed on first access."""
        if self._data is None:
            self.prepare()
        return self._data

    @property
    def facet_dims(self) -> tuple[str | None, str | None]:
        """``(row_dim, col_dim)`` — the axes whose values become panels.

        ``(None, None)`` is a single map, ``(None, "time")`` a series of them,
        ``("z", "time")`` a grid of levels by periods. See :func:`_facet_dims_of`.
        """
        return _facet_dims_of(self.data)

    @property
    def facet_dim(self) -> str | None:
        """The axis across the columns, or ``None`` for a single map."""
        return self.facet_dims[1]

    def _time_axis_dim(self, da) -> str | None:
        """:func:`ocean_skill.operators.time_axis_dim`, with one field-level exception.

        A time groupby's surviving dimension (``month``, ``year``, ...) plays
        time's role by default -- see :func:`~ocean_skill.operators.time_axis_dim`
        -- *unless* the caller also named an explicit list of months,
        ``select={"month": [1, 4, 7]}``. That mirrors :attr:`is_time_depth`'s own
        depth-list exception: naming discrete values is asking to tell them apart
        as separate lines, not to fold them into one axis, so "month" stops being
        read as time and the shape falls through to the profile family's fan
        (:func:`ocean_skill.plot.profile.fan_season`) instead -- the same idiom a
        surviving season axis already draws through.
        """
        from ocean_skill.operators import time_axis_dim

        dim = time_axis_dim(da)
        if dim == "month" and isinstance(self.select.get("month"), list | tuple):
            return None
        return dim

    @property
    def is_series(self) -> bool:
        """Whether this field's prepared data is a place through time, not a map.

        Mirrors :attr:`ocean_skill.comparison.Comparison.is_series`: a select that
        narrows both horizontal axes to one position
        (:func:`ocean_skill.align.point_of`) leaves nothing for :attr:`facet_dims`
        to lay out as columns, and a surviving time axis is exactly what a line
        needs for its x. Read off the data's own shape, never an argument. "Time"
        includes a time groupby's surviving dimension (see :meth:`_time_axis_dim`).
        """
        from ocean_skill.align import point_of

        da = self.data
        return point_of(da) is not None and self._time_axis_dim(da) in da.dims

    @property
    def is_profile(self) -> bool:
        """Whether this field's prepared data is a place through depth, not a map.

        Mirrors :attr:`is_series` one axis over: a select that narrows both
        horizontal axes to one position leaves a surviving vertical axis standing
        with no time axis to draw a line through instead. The water column a
        station is, at one instant.
        """
        from ocean_skill.align import point_of
        from ocean_skill.operators import resolve_dim

        da = self.data
        if point_of(da) is None:
            return False
        tdim = self._time_axis_dim(da)
        zdim = resolve_dim(da, "Z")
        return (tdim is None or tdim not in da.dims) and (
            zdim is not None and zdim in da.dims
        )

    @property
    def is_time_depth(self) -> bool:
        """Whether this field's prepared data is a place through both time and depth.

        Checked *before* :attr:`is_series` in :attr:`family`: a select that keeps
        both time and depth standing at one place used to draw as a series, one
        line per level -- now it draws as one ``time_depth`` panel (colour =
        value, x = time, y = depth) instead, the default shape for a bare
        ``timeSeriesProfile`` station. An *explicit* list of levels
        (``select={"depth": [0, 50, 100]}``) is excluded and keeps the old
        per-level lines: naming discrete levels is asking to tell them apart, not
        to see the whole record, and a list too long to draw as separate lines
        should be narrowed or aggregated, not silently redirected here. "Time"
        includes a time groupby's surviving dimension, with the matching
        exception for an explicit month list (see :meth:`_time_axis_dim`).
        """
        from ocean_skill.align import point_of
        from ocean_skill.comparison import _selected_depth
        from ocean_skill.operators import resolve_dim

        da = self.data
        if point_of(da) is None:
            return False
        tdim = self._time_axis_dim(da)
        zdim = resolve_dim(da, "Z")
        if tdim is None or tdim not in da.dims or zdim is None or zdim not in da.dims:
            return False
        requested = _selected_depth(self.select, default=None)
        return not isinstance(requested, list | tuple)

    @property
    def is_section(self) -> bool:
        """Whether this field's prepared data is a cut through space: a vertical slice.

        Mirrors :attr:`is_series`, one level down: a select that names
        ``{"transect": ...}`` leaves an along-path axis standing
        (:func:`ocean_skill.align.path_of`) instead of collapsing to one place, so
        the shape that answers "is this a section?" is a path, not a point. Read
        off the data's own shape, never an argument -- see :attr:`family`.

        True regardless of whether a vertical axis actually survives alongside the
        path; :meth:`plot` is where a shape that cannot draw as one panel (no
        vertical axis left, or more than one extra axis) is refused, with the
        specifics of what is wrong. This property only answers "is this the kind
        of field a section recipe applies to at all".
        """
        from ocean_skill.align import path_of

        return path_of(self.data) is not None

    @property
    def family(self) -> str:
        """The plot family this field's own shape admits.

        ``time_depth`` (colour = value against time and depth at one place),
        ``series`` (a line over time at one place), ``profile`` (a line down
        depth at one place, one instant), ``section`` (a cut through depth and
        along-path distance), or ``field_facet`` (map panels) — no argument
        selects it, the same as :attr:`ocean_skill.comparison.Comparison.family`
        — the prepared data's shape decides. :attr:`is_time_depth` is checked
        *before* :attr:`is_series`, since both hold for a point where time and
        depth both survive; only an explicit list of depths falls through to
        ``series`` instead (see :attr:`is_time_depth`).
        """
        if self.is_time_depth:
            return "time_depth"
        if self.is_series:
            return "series"
        if self.is_profile:
            return "profile"
        if self.is_section:
            return "section"
        return "field_facet"

    @property
    def family_reason(self) -> str:
        """Why :attr:`family` came out the way it did, for tracing a surprise."""
        if self.is_time_depth:
            return (
                "drawn as depth against time: the selection leaves one place "
                "with both time and depth standing"
            )
        if self.is_series:
            return "drawn as a line: the selection leaves one place, so the surviving time axis is the x"
        if self.is_profile:
            return "drawn as a profile: the selection leaves one place and one instant, so the surviving depth axis is the y"
        if self.is_section:
            return "drawn as a section: select={'transect': ...} leaves a cut through space, with depth on the other axis"
        return "drawn as map panels: a horizontal extent survives"

    def extremum(self, kind: str = "max") -> Any:
        """Locate this field's min/max: value, lon/lat, grid indices, snapshot.

        Parameters
        ----------
        kind
            One of ``"max"`` or ``"min"`` (default ``"max"``) -- which extremum
            to locate.

        Runs over every dim the prepared field still has, not just the horizontal
        ones -- a field faceted over time or depth reports the facet coordinate
        the extremum fell on as part of the answer, the same way :attr:`family`
        reads its shape off the data rather than an argument. Refused when
        :attr:`is_series` (or this field has otherwise been reduced to one place):
        a point has one value already, and there is no spatial extent left to
        search over.

        The result's ``.series()`` follows the located position through time --
        a point selection at this extremum's lon/lat, over a window that
        defaults to :data:`~ocean_skill.extrema.DEFAULT_PAD_STEPS` native steps
        each side of the snapshot -- and ``.plot()`` draws that immediately::

            run = osk.field("pac_dt_ramp", "temperature",
                             select={"time": "2013-06-15", "depth": "surface"})
            ext = run.extremum("max")
            ext.series(variables=["salinity"]).plot()   # both, same place/window

        A :class:`FieldSet` (several variables) has no ``extremum`` of its own --
        each member is its own field with its own map; call it on one member,
        e.g. ``fields[0].extremum()``.

        See :class:`ocean_skill.extrema.Extremum`.
        """
        from ocean_skill.extrema import field_extremum

        return field_extremum(self, kind)

    def _series_items(self) -> list[dict[str, Any]]:
        """Return this field's data as one or more single-source series items.

        One item, unless a vertical axis also survives the reduction — then one
        item per level, each carrying its own ``actual_depth`` so the depth-as-
        marker channel (:mod:`ocean_skill.plot.style`) and the legend
        (:func:`ocean_skill.plot.style.series_label`) tell the levels apart.
        Anything else left standing beyond time and depth is refused, the same as
        :func:`_facet_dims` refuses a third map axis — a line has only one axis to
        give away.

        A bare native ``s_rho``/``s_w`` axis (a model point with nothing narrowed
        vertically — see :func:`_labelless_vertical`) has no coordinate of its own
        to fan into distinct legend entries; a 1-D ``z_rho`` on the same dimension
        supplies one instead, the same fallback a profile line uses. Neither one
        present is refused outright, the same recipe :meth:`Field.plot` gives a
        label-less map facet -- an unlabeled legend that cannot tell its own
        entries apart is worse than no legend at all.
        """
        import xarray as xr

        from ocean_skill.operators import resolve_dim, vertical_coord_on

        da = self.data
        tdim = self._time_axis_dim(da)
        zdim = resolve_dim(da, "Z")
        extra = [str(d) for d in da.dims if d not in (tdim, zdim)]
        if extra:
            raise ValueError(
                f"this series still has {extra} beyond time, and a line has only "
                f"one axis to give away. Collapse it with aggregate= (e.g. "
                f'{{"{extra[0]}": "mean"}}) or narrow it with select=.'
            )

        base = {
            "units": da.attrs.get("units"),
            "standard_name": self.standard_name,
            "label": self.label or self.source,
            "labels": (self.label or self.source,),
        }
        if zdim is None or zdim not in da.dims:
            return [{"aligned": xr.Dataset({"value": da}), "metrics": None, **base}]

        if _labelless_vertical(da) is not None:
            raise ValueError(
                f"{self.source!r}'s native vertical axis ({zdim!r}) carries no "
                "coordinate to tell its levels apart in a legend, and no z_rho "
                "either. Narrow it with select= (e.g. {'depth': [0, 50, 100]}, "
                "'surface', or {'sigma0': ...}) or collapse it with aggregate= "
                '(e.g. {"Z": "mean"}).'
            )

        # The coordinate that carries each level's real value -- ordinarily zdim
        # itself, but see vertical_coord_on for a coordinate riding under a
        # different name (a catalog recipe's `depth` on a dimension still spelled
        # `DEPTH`, say). Read once, outside the loop: isel below slices whatever
        # coordinate shares zdim along with the data, whatever it is named.
        zcoord = vertical_coord_on(da, zdim)
        zcoord_name = str(zcoord.name) if zcoord is not None else None

        items = []
        for k in range(da.sizes[zdim]):
            level = da.isel({zdim: k})
            item = {"aligned": xr.Dataset({"value": level}), "metrics": None, **base}
            # actual_depth lives on the *item's* Dataset, not the DataArray, since
            # that is what _depth_of (plot/series.py) reads -- the same convention
            # Comparison.align() uses for its own aligned pair. abs() unconditionally
            # (matching plot/profile.vertical_values and
            # plot/time_depth.prepare_time_depth) -- a no-op for an already
            # positive-down observational coordinate, and what turns z_rho's
            # negative-down convention into the positive-down metres every other
            # depth label in this package uses. zcoord may itself resolve to z_rho
            # (see vertical_coord_on): a bare native axis with a 1-D z_rho counts as
            # carrying a real coordinate the same way a differently-named observational
            # one does, so both are read the same way here.
            if zcoord_name is not None and zcoord_name in level.coords:
                item["aligned"].attrs["actual_depth"] = abs(float(level[zcoord_name]))
            elif "z_rho" in level.coords and level["z_rho"].ndim == 0:
                item["aligned"].attrs["actual_depth"] = abs(float(level["z_rho"]))
            items.append(item)
        return items

    def _time_depth_item(self) -> dict[str, Any]:
        """Return this field's data as one ``time_depth`` spec item.

        The single-source counterpart of :meth:`_series_items`/:meth:`_profile_items`
        for the shape :attr:`is_time_depth` recognizes: a point with both time and
        depth standing, drawn as one panel rather than fanned into several lines.
        Anything else left standing beyond time and depth is refused, the same
        wording :meth:`_series_items` uses for a line's own third axis.

        A bare native ``s_rho``/``s_w`` axis with no coordinate of its own (see
        :func:`_labelless_vertical`) is refused here too, for the same reason
        :meth:`_series_items` refuses it for a fanned line: an unlabeled index
        position says nothing about which depth a marker or mesh cell is at.
        """
        from ocean_skill.operators import resolve_dim

        da = self.data
        tdim = self._time_axis_dim(da)
        zdim = resolve_dim(da, "Z")
        extra = [str(d) for d in da.dims if d not in (tdim, zdim)]
        if extra:
            raise ValueError(
                f"this time_depth panel still has {extra} beyond time and depth, "
                "and it has only those two axes to give away. Collapse it with "
                f'aggregate= (e.g. {{"{extra[0]}": "mean"}}) or narrow it with '
                "select=."
            )
        if _labelless_vertical(da) is not None:
            raise ValueError(
                f"{self.source!r}'s native vertical axis ({zdim!r}) carries no "
                "coordinate to say which depth each reading is at, and no z_rho "
                "either. Narrow it with select= (e.g. {'depth': [0, 50, 100]}, "
                "'surface', or {'sigma0': ...}) or collapse it with aggregate= "
                '(e.g. {"Z": "mean"}).'
            )
        return {
            "field": da,
            "units": da.attrs.get("units"),
            "standard_name": self.standard_name,
            "label": self.label or self.source,
        }

    def _profile_items(self) -> list[dict[str, Any]]:
        """Return this field's data as one or more single-source profile items.

        The vertical twin of :meth:`_series_items`, and simpler: a profile's
        depth axis survives *whole*, one line down the water column, rather than
        being fanned into one item per level -- depth is the axis a profile line
        draws against, not a fact several series lines are told apart by (see
        :mod:`ocean_skill.plot.profile`). A surviving season axis is one
        exception, fanned into one item per season
        (:func:`ocean_skill.plot.profile.fan_season`) exactly the way
        :meth:`_series_items` fans depth levels -- several seasons in one panel
        are several lines, which is several items, the same idiom either way. A
        surviving *marked* month axis -- a ``{"groupby": "month"}`` result
        narrowed by an explicit ``select={"month": [...]}`` list, see
        :meth:`_time_axis_dim` -- is the other, fanned the identical way. An
        ordinary ``month`` dim built some other way carries no such mark and is
        refused like any other extra axis, the same as :meth:`_series_items`
        refuses a third axis a line has no room for.
        """
        import xarray as xr

        from ocean_skill.operators import TIME_GROUPBY_ATTR, resolve_dim
        from ocean_skill.plot.profile import MONTH_DIM, SEASON_DIM, fan_season

        da = self.data
        zdim = resolve_dim(da, "Z")
        extra = [str(d) for d in da.dims if d != zdim]
        month_coord = da.coords.get(MONTH_DIM)
        fannable = extra == [SEASON_DIM] or (
            extra == [MONTH_DIM]
            and month_coord is not None
            and TIME_GROUPBY_ATTR in month_coord.attrs
        )
        if extra and not fannable:
            raise ValueError(
                f"this profile still has {extra} beyond depth, and a profile "
                f"line has only one axis to give away. Collapse it with "
                f'aggregate= (e.g. {{"{extra[0]}": "mean"}}) or narrow it with '
                "select=."
            )
        base = {
            "units": da.attrs.get("units"),
            "standard_name": self.standard_name,
            "label": self.label or self.source,
            "labels": (self.label or self.source,),
        }
        item = {"aligned": xr.Dataset({"value": da}), "metrics": None, **base}
        return fan_season([item]) if extra else [item]

    def as_item(self) -> dict[str, Any]:
        """Return this field as a spec item."""
        from ocean_skill.comparison import _depth_label, _selected_depth

        # the vertical selection, spelled for a label ("surface", "50 m", "σ₀ = 26.5
        # kg/m³"). A renderer cannot recover it from the field once the transform has
        # collapsed the axis, and a plot of one level that does not say which level is
        # a plot of nothing in particular -- see the interactive movie's title. Unlike
        # a Comparison (whose own default is the surface), a bare Field does nothing
        # vertically unless asked -- default=None here, so an unset select reports no
        # depth at all rather than claiming a "surface" reduction that never happened.
        requested = _selected_depth(self.select, default=None)
        item = {
            "field": self.data,
            "units": self.data.attrs.get("units"),
            "standard_name": self.standard_name,
            "depth": _depth_label(requested) if requested is not None else None,
            "label": self.label or self.source,
        }
        if self.family == "section":
            # facet_dim/row_dim are field_facet's own vocabulary -- a section has
            # neither rows nor columns to lay out, only the one panel, so
            # self.facet_dims (which would read the surviving vertical axis as a
            # facet column to lay out, the wrong reading for it here) is not
            # consulted at all.
            return item
        row_dim, facet_dim = self.facet_dims
        return {**item, "facet_dim": facet_dim, "row_dim": row_dim}

    def _require_section_shape(self) -> None:
        """Raise unless this field's data is exactly (vertical, along) -- a section.

        Two ways to fail this: nothing but ``along`` survives (the vertical axis
        was collapsed away, e.g. ``select={'depth': 'surface'}`` alongside a
        transect), or a further axis survives too (most often time, left standing
        by an ``aggregate`` that does not fully collapse it). Either way this says
        so rather than handing the map-drawing code a shape it does not
        understand, which is what :attr:`is_section` alone does not check --
        see its docstring.
        """
        from ocean_skill.align import ALONG_DIM

        extra = sorted(str(d) for d in self.data.dims if d != ALONG_DIM)
        if len(extra) == 1:
            return
        if not extra:
            raise ValueError(
                f"{self.source!r} has been reduced to one along-path axis with no "
                "vertical axis surviving, so there is nothing to draw as a "
                "section. Omit select={'depth': 'surface'} (or any reduction "
                "that collapses depth) to keep the vertical axis standing."
            )
        raise ValueError(
            f"{self.source!r} still has {extra} beyond its along-path and "
            "vertical axes, so it is not a single section -- a section figure "
            "has only depth and distance. Collapse the rest with aggregate= "
            f'(e.g. {{"{extra[0]}": "mean"}}) or narrow it with select= (e.g. '
            f'{{"{extra[0]}": "2012-01"}}), or reduce time or fan it -- '
            "time-animated sections are a follow-up."
        )

    def _refuse_labelless_facet(self, field: Any = None) -> None:
        """Raise if a facet axis is a vertical dim with no coordinate to label it.

        ``field`` defaults to ``self.data``, but a caller that already ran
        :meth:`_facet_field_and_depth` passes its *result* instead -- that field
        has already had a bare vertical axis reduced away wherever it could be
        (see :func:`_top_level`), so this must not re-examine the unreduced
        ``self.data`` and raise for an axis that no longer stands in what is
        actually about to be drawn.

        A bare native ``s_rho``/``s_w`` axis -- a model field with nothing narrowed
        vertically (see :func:`_labelless_vertical`, the default a bare :func:`field`
        call now leaves in force) -- would otherwise draw or play as an unlabeled row,
        column, or frame count: N panels or frames saying nothing about which level
        each one is, once a facet dim happens to be it. Refused with the same recipe
        :meth:`_series_items` gives a labelless *point* -- data access (``.data``)
        stays whole either way; only drawing it as panels/frames is refused. A
        labeled obs gridded product's own ``depth`` axis is never caught by this,
        whatever facet size it draws as -- see :func:`_labelless_vertical`.
        """
        field = self.data if field is None else field
        zdim = _labelless_vertical(field)
        if zdim is not None and zdim in _facet_dims_of(field):
            raise ValueError(
                f"{self.source!r}'s native vertical axis ({zdim!r}) carries no "
                "coordinate to label its levels with, so drawing it as panels or "
                "frames would say nothing about which level is which. Narrow it "
                "with select= (e.g. {'depth': 'surface'}, a number, a list, or "
                "'column' at a point) or {'sigma0': ...}, or collapse it with "
                'aggregate= (e.g. {"Z": "mean"}).'
            )

    def _catalog_metadata(self) -> dict[str, Any]:
        """Return this field's catalog entry metadata, or ``{}`` if unresolvable.

        Read-free -- an uncatalogued or hand-built ``source`` (a test stub, most
        often) is not an error here the way it would be at read time; it just
        means none of the catalog-driven defaults below
        (:meth:`_grid_metadata_if_eligible`) apply.
        """
        from ocean_skill.catalog import resolve

        try:
            return resolve(self.source).metadata
        except KeyError:
            return {}

    def _grid_metadata_if_eligible(self) -> dict[str, Any] | None:
        """Return this field's catalog metadata, when the grid defaults apply to it.

        The grid defaults (:meth:`_surfaced`, the bare-multi-step-time refusal in
        :meth:`plot`) are read-free shortcuts for the common case -- a catalogued
        ``featureType: grid`` source, narrowed to neither a horizontal point nor a
        transect (either of those draws as a line/section/profile instead, see
        :attr:`family`, where the catalog's own vertical/time metadata says
        nothing about what the *reduced* shape needs). ``None`` here means one of
        those does not hold, or the source is not catalogued at all -- the
        fallback half of the surface default (:func:`_top_level`, called from
        :meth:`_facet_field_and_depth`) and the ordinary time-based refusal still
        apply once the data is actually loaded.
        """
        from ocean_skill.operators import point_in_spec

        if point_in_spec(self.select) is not None or "transect" in self.select:
            return None
        meta = self._catalog_metadata()
        if meta.get("featureType") != "grid":
            return None
        return meta

    def _bare_vertical(self) -> bool:
        """Whether nothing -- select or aggregate -- named a vertical request."""
        from ocean_skill.comparison import _names_vertical, _vertical_only

        return not _names_vertical(self.select) and not _vertical_only(
            self.aggregate
        )

    def _bare_time(self) -> bool:
        """Whether nothing -- select or aggregate -- narrowed the time axis.

        Checked against every accepted spelling: ``time``/``T``/``t`` (see
        :data:`ocean_skill.sources._TIME_KEYS`) plus this source's own declared
        axis name (``axes["T"]``, e.g. an ERDDAP entry's ``"time (UTC)"``), the
        same union :func:`ocean_skill.sources.erddap_constraints` reads. An
        aggregate naming time (a ``resample``/``groupby``) counts as narrowing it
        just as much as a ``select`` does -- either one says the caller has
        already decided what the panels are.
        """
        from ocean_skill.sources import _TIME_KEYS

        meta = self._catalog_metadata()
        time_keys = set(_TIME_KEYS)
        axis_t = (meta.get("axes") or {}).get("T")
        if axis_t:
            time_keys.add(axis_t)
        agg = self.aggregate or {}
        return not any(k in self.select for k in time_keys) and not any(
            k in agg for k in time_keys
        )

    def _surfaced(self) -> Field:
        """Return a new, otherwise identical :class:`Field` selecting the surface.

        The read-free half of the grid surface default: a catalogued
        ``featureType: grid`` source with nothing named vertically defaults to
        ``select={"depth": "surface"}`` -- the same explicit key
        :meth:`ocean_skill.comparison.Comparison._prepare_lane` injects for its
        own unset default, so this shares that lane's cache entry rather than
        opening a new one, and :meth:`as_item` reports the level from the select
        exactly as an explicit request would.
        """
        return Field(
            self.source,
            self.variable,
            select={**self.select, "depth": "surface"},
            aggregate=self.aggregate,
            label=self.label,
            cache=self.cache,
            qc=self.qc,
            detide=self.detide,
        )

    def _refuse_bare_multistep_time(self) -> None:
        """Raise the "no default instant" message :meth:`plot` gives a bare grid.

        Named once so the read-free pre-check (against catalog-declared time
        coverage) and the definitive post-load check (against the surviving time
        dim's own size) raise identically -- a caller should not be able to tell
        which of the two caught it.
        """
        raise ValueError(
            f"{self.source!r} has a whole time record standing with nothing "
            "narrowing it, and .plot() draws one instant, not a movie -- there "
            "is no single default to pick. Name one with select={'time': "
            "'2024-06-15'} (or a slice), reduce it with aggregate= (e.g. "
            "{'time': {'resample': '1MS', 'reduce': 'mean'}} for one panel per "
            "month), or call .movie() to play every step instead."
        )

    def _refuse_bare_multistep_time_precheck(self) -> None:
        """Read-cheap half of the bare-time refusal, against this variable's own axis.

        A lazy peek (:func:`ocean_skill.comparison._bare_time_is_multistep`) opens
        the source the same way :func:`~ocean_skill.comparison.prepare_source`
        eventually would, but stops at dimension metadata -- no crop, no vertical
        transform, no ``.load()`` -- and reports whether *this* variable's own time
        axis survives with more than one step. That is the same verdict the
        definitive post-load check in :meth:`plot` would reach (nothing has
        narrowed time, so the raw axis and the surviving one are the same size),
        just without paying for a full prepare first -- the point of a bare,
        multi-day *source* record like a two-week run is exactly that reading it
        whole is expensive, and a time-invariant variable (ROMS's ``h``, say) has
        no ambiguity to refuse in the first place.

        ``True`` refuses outright. ``False`` (static, or a single-step record) and
        ``None`` (the peek could not answer cheaply -- an unresolvable variable, a
        calculate-spec, a read that failed) both fall through to the definitive
        check, which always has the real data to settle it.
        """
        from ocean_skill.comparison import _bare_time_is_multistep

        if not self._bare_time():
            return
        if _bare_time_is_multistep(self.source, self.variable, self.select, qc=self.qc):
            self._refuse_bare_multistep_time()

    def _facet_field_and_depth(self) -> tuple[Any, str | None]:
        """``(field, depth_label)`` for a ``field_facet``/``facet_movie`` item.

        ``self.data`` unchanged, or its bare vertical axis reduced to the
        shallowest level first (:func:`_top_level`) -- the fallback half of the
        grid surface default, for whatever :meth:`_grid_metadata_if_eligible`'s
        read-free half could not settle before the data was ever read: an
        uncatalogued source, or a hand-built ``prepare_source`` stub. A field
        the read-free half already redirected (:meth:`_surfaced`) reaches here
        with nothing left to reduce -- its own select already named the
        surface, so :attr:`_bare_vertical` is false and this is a no-op.
        """
        from ocean_skill.operators import resolve_dim

        da = self.data
        zdim = resolve_dim(da, "Z")
        if zdim is None or zdim not in da.dims or not self._bare_vertical():
            return da, None
        return _top_level(da, zdim, source=self.source)

    def _facet_item(self, field: Any, depth_override: str | None) -> dict[str, Any]:
        """Build a ``field_facet``/``facet_movie`` item from ``field``.

        The single-source counterpart of :meth:`as_item` for whichever field
        :meth:`_facet_field_and_depth` decided to draw -- ordinarily
        ``self.data`` itself, or its top-level slice. ``facet_dim``/``row_dim``
        are read off ``field``'s own shape (:func:`_facet_dims_of`), not
        ``self.data``'s, so a level already reduced away here does not also
        count as a facet axis.
        """
        from ocean_skill.comparison import _depth_label, _selected_depth

        if depth_override is not None:
            depth = depth_override
        else:
            requested = _selected_depth(self.select, default=None)
            depth = _depth_label(requested) if requested is not None else None
        item = {
            "field": field,
            "units": field.attrs.get("units"),
            "standard_name": self.standard_name,
            "depth": depth,
            "label": self.label or self.source,
        }
        row_dim, facet_dim = _facet_dims_of(field)
        return {**item, "facet_dim": facet_dim, "row_dim": row_dim}

    def _facet_item_or_refuse(self) -> dict[str, Any]:
        """Build this field's ``field_facet`` item, running every refusal a map needs.

        Shared by :meth:`plot`'s own map branch and :meth:`_map_item` (the
        :class:`FieldSet` counterpart), so the two never drift on what a map is
        allowed to draw: the same point-with-no-extent and fixed-station
        diagnostics, the same bare-multistep-time and labelless-facet refusals
        :meth:`plot` has always raised here. Called only once the read-cheap
        grid preamble (the surface default, the bare-multistep-time precheck)
        has already run or does not apply -- this does not re-run it.
        """
        from ocean_skill.align import point_of
        from ocean_skill.operators import resolve_dim

        if point_of(self.data) is not None:
            # A point with neither a surviving time nor depth axis: no line
            # (nothing to run it along, either way) and no map (no horizontal
            # extent left) can be drawn -- field_facet would otherwise try to
            # lay out panels of a field with no axes at all, which fails
            # confusingly further in.
            raise ValueError(
                f"{self.source!r} has been reduced to one place with no "
                f"surviving time or depth axis ({sorted(self.data.dims)} "
                "standing), so there is no horizontal extent left for map "
                "panels and nothing for a line to run along either. Keep time "
                "standing for a series, or depth for a profile (drop a "
                "select= that pins it to one value, or an aggregate that "
                "collapses it), or widen select= to keep a horizontal extent "
                "for a map."
            )
        from ocean_skill.comparison import POINT_FEATURE_TYPES, PROFILE_FEATURE_TYPES

        feature_type = str(self._catalog_metadata().get("featureType") or "")
        if feature_type in (POINT_FEATURE_TYPES | PROFILE_FEATURE_TYPES):
            # The catalog says this source is one fixed station (a mooring,
            # a repeat-visit profile) -- unlike a trajectory, whose lon/lat
            # legitimately varies, this featureType's data always has one
            # recoverable position -- yet point_of found none at all, not
            # merely "narrowed to a point". prepare_source's own cache-hit
            # check (_is_stale_positionless_station) already discards and
            # recomputes exactly this shape of stale entry, so reaching
            # here with cache still on means that repair already ran and
            # still found nothing; the source's own lon/lat is the more
            # likely culprit now. Named explicitly rather than falling
            # through to the field_facet path below, which would otherwise
            # fail on a field with no horizontal extent to lay out panels
            # of, with no hint of why.
            raise ValueError(
                f"{self.source!r} is catalogued as featureType: "
                f"{feature_type!r} (one fixed position), but its prepared "
                "data has no recoverable lon/lat at all. Try field(..., "
                "cache=False) to rule out a stale cache entry; if the "
                "position is still missing, the source's own lon/lat "
                "metadata is what needs fixing."
            )
        # The fallback half of the surface default -- whatever the read-free
        # check above could not settle (an uncatalogued source, a test stub)
        # -- runs first: it raises its own labelless-axis message when the
        # bare vertical axis it finds cannot be reduced at all, before the
        # bare-time check below gets a chance to raise a less specific one for
        # the same field.
        field, depth_override = self._facet_field_and_depth()
        tdim = resolve_dim(field, "T")
        if (
            tdim is not None
            and tdim in field.dims
            and field.sizes[tdim] > 1
            and self._bare_time()
        ):
            self._refuse_bare_multistep_time()
        self._refuse_labelless_facet(field)
        return self._facet_item(field, depth_override)

    def _map_item(self) -> dict[str, Any]:
        """This field's own single map item, refusing anything that isn't one.

        The self-contained counterpart of :meth:`plot`'s map branch for a
        caller that never goes through :meth:`plot` at all --
        :meth:`~ocean_skill.field.FieldSet._map_items`, building several
        fields' maps to place beside each other. Runs the same read-cheap grid
        preamble :meth:`plot` runs at its own top (the surface default via
        :meth:`_surfaced`, the bare-multistep-time precheck) before deferring
        to :meth:`_facet_item_or_refuse` for the rest -- so a :class:`FieldSet`
        member gets exactly the defaults a bare :meth:`plot` call on that same
        field would have, not a second, looser set of rules.

        Beyond that shared ground, this also refuses a field whose item still
        carries a facet axis of more than one step (several timesteps, several
        depths) -- a set of several variables drawn beside each other has room
        for one map per member, not a grid of grids. A facet/row dim of size
        *one* (a WOA climatology's bare ``time``, unsqueezed but never really
        a choice -- see :data:`~ocean_skill.comparison.NO_AGGREGATION`'s own
        docstring) is not this: :func:`~ocean_skill.plot.matplotlib_renderer
        .field_facet` already draws it as the single panel it is, the same as
        a solo :meth:`plot` call would. Narrow a genuinely standing axis the
        same way that solo call would draw it instead: ``select=`` to one
        instant, or ``aggregate=`` to collapse it.
        """
        grid_meta = self._grid_metadata_if_eligible()
        if grid_meta is not None:
            if self._bare_vertical() and _grid_has_vertical_axis(grid_meta):
                return self._surfaced()._map_item()
            self._refuse_bare_multistep_time_precheck()

        item = self._facet_item_or_refuse()
        field = item["field"]
        standing = next(
            (
                d
                for d in (item["facet_dim"], item["row_dim"])
                if d is not None and field.sizes[d] > 1
            ),
            None,
        )
        if standing is not None:
            raise ValueError(
                f"{self.source!r} still has {standing!r}={field.sizes[standing]} "
                "standing beyond its map, so it is not a single map to place "
                f"beside the others in this set. Narrow it with select= to one "
                f"{standing} (e.g. select={{{standing!r}: ...}}) or collapse it "
                f"with aggregate= (e.g. aggregate={{{standing!r}: 'mean'}}), so "
                "every member is one map."
            )
        axes = [d for d in (item["facet_dim"], item["row_dim"]) if d is not None]
        if axes:
            # Every named axis survives, but only at size one -- field_facet's own
            # single panel already isels exactly this away (see the loop over
            # n == 1 there); field_map_grid's items carry no such isel step of
            # their own, so it has to happen here instead. drop=False keeps it as
            # the scalar coordinate field_suptitle/grid_suptitle already know how
            # to read for context (e.g. a WOA climatology's bare "time").
            field = field.squeeze(axes, drop=False)
            item = {**item, "field": field, "facet_dim": None, "row_dim": None}
        return item

    @graft_plot_options()
    def plot(self, *, renderer: str = "matplotlib", **kwargs: Any):
        """Draw this field: map panels, a section, a profile, a line, or depth vs time.

        Parameters
        ----------
        renderer
            One of ``"matplotlib"`` (default, static) or ``"holoviews"``
            (interactive) -- goes through the same renderer registry either way.
        **kwargs
            Plot options forwarded to the renderer: option families such as
            ``color_by``, ``marker_by``, ``labels``, ``title``, ``domain``,
            ``robust``, ``figsize``, ``save``, and the ``*_kwargs`` styling
            dicts (``title_kwargs``, ``colorbar_kwargs``, ``legend_kwargs``,
            and the rest). See ``docs/plot_styling_reference.md`` for the full
            list.

        Which of the five is decided by the prepared data's own shape, never an
        argument — see :attr:`family`. The one exception is a catalogued
        ``featureType: grid`` source with a bare vertical select, which is
        narrowed to the surface before :attr:`family` is even consulted (see
        :func:`field`'s own docstring) -- a map has no vertical axis to draw at
        all, so this is the drawing-time default that keeps the data itself,
        read through :attr:`data`, whole either way. Goes through the renderer
        registry, so ``renderer="holoviews"`` gives the interactive version of
        the same plot with no other change.
        """
        from ocean_skill.plot.registry import render
        from ocean_skill.plot.spec import PlotSpec

        # The two grid defaults: a bare vertical select on a catalogued grid draws
        # the surface (recurse on the surfaced field, which then runs this same
        # method start to finish -- including the time check right below, now that
        # the vertical question is settled), read-free; a bare, genuinely
        # multi-step time axis has no single default instant, and says so rather
        # than guessing one -- read-cheap (a lazy peek at this variable's own axis,
        # see _refuse_bare_multistep_time_precheck), never a full prepare.
        grid_meta = self._grid_metadata_if_eligible()
        if grid_meta is not None:
            if self._bare_vertical() and _grid_has_vertical_axis(grid_meta):
                return self._surfaced().plot(renderer=renderer, **kwargs)
            self._refuse_bare_multistep_time_precheck()

        if self.family == "time_depth":
            spec = PlotSpec(
                family="time_depth", items=[self._time_depth_item()], options=kwargs
            )
        elif self.family == "series":
            spec = PlotSpec(
                family="series", items=self._series_items(), options=kwargs
            )
        elif self.family == "profile":
            spec = PlotSpec(
                family="profile", items=self._profile_items(), options=kwargs
            )
        elif self.family == "section":
            self._require_section_shape()
            spec = PlotSpec(family="section", items=[self.as_item()], options=kwargs)
        else:
            # Every other refusal a map needs -- the point-with-no-extent and
            # fixed-station diagnostics, the bare-multistep-time and
            # labelless-facet checks -- lives in _facet_item_or_refuse, shared
            # with FieldSet._map_items so the two never draw a map by different
            # rules.
            spec = PlotSpec(
                family="field_facet",
                items=[self._facet_item_or_refuse()],
                options=kwargs,
            )
        return render(spec, renderer=renderer)

    @graft_plot_options()
    def movie(self, *, renderer: str = "matplotlib", **kwargs: Any):
        """Play :attr:`facet_dim` instead of laying it out: this field as a movie.

        Parameters
        ----------
        renderer
            One of ``"matplotlib"`` (default, static ``.mp4``/``.gif``) or
            ``"holoviews"`` (interactive, a slider).
        **kwargs
            Plot options forwarded to the renderer: ``save`` (filename; its
            extension picks the format), ``every`` (keep every Nth step), plus
            the same option families as :meth:`plot` (``title``, ``domain``,
            ``robust``, ``figsize``, the ``*_kwargs`` styling dicts, including
            movie-only ``frame_label_kwargs``). See
            ``docs/plot_styling_reference.md`` for the full list.

        The same axis :meth:`plot` turns into panels becomes the frames here, so the two
        are one field read two ways::

            run = osk.field("GOM_bgc", "salinity",
                            select={"time": "2012-01", "depth": "surface"})
            run.plot()                          # every step as a panel
            run.movie(save="salt.mp4")          # every step as a frame
            run.movie(renderer="holoviews")     # every step on a slider

        Nothing is reduced unless asked (see
        :data:`~ocean_skill.comparison.NO_AGGREGATION`), so that is a frame per step of
        January. Pass an ``aggregate`` to play a coarser cadence — a ``resample`` for
        one frame per day or month, a ``groupby`` for a climatology — or ``every=`` to
        keep every Nth step of the one you have.

        Which is the better reading depends on how many steps there are: a handful of
        monthly means are best seen at once, where a month of daily output is forty
        panels too small to read and forty frames a drag apart.

        ``save`` names the file and its extension picks the format — ``.mp4`` (needs
        ffmpeg) or ``.gif`` (needs nothing extra) statically, ``.html`` interactively.
        See :func:`ocean_skill.plot.matplotlib_renderer.facet_movie` for the rest, and
        ``docs/movies.md`` for the whole picture.

        A field the reduction left as a *single map* has no axis to play, and says so
        rather than writing a one-frame movie. A field reduced to a *point* (see
        :attr:`family`) has no map at all, and says so too — it draws as a line,
        which :meth:`plot` already shows in full; there is nothing left to play.
        """
        from ocean_skill.plot.registry import render
        from ocean_skill.plot.spec import PlotSpec

        # Only the surface half of the grid defaults applies to a movie -- a bare,
        # multi-step time axis is exactly what .movie() is for, so there is no
        # time refusal here the way there is in .plot() (see
        # _grid_metadata_if_eligible).
        grid_meta = self._grid_metadata_if_eligible()
        if (
            grid_meta is not None
            and self._bare_vertical()
            and _grid_has_vertical_axis(grid_meta)
        ):
            return self._surfaced().movie(renderer=renderer, **kwargs)

        if self.is_time_depth:
            raise ValueError(
                f"{self.source!r} draws as depth against time (see .family), not "
                "a map -- the whole record is already on one panel. Use .plot() "
                "instead; it already shows the whole record."
            )
        if self.is_series:
            raise ValueError(
                f"{self.source!r} has been reduced to a point and draws as a line "
                "over time (see .family), not a map -- there is nothing to play as "
                "a movie. Use .plot() instead; it already shows the whole series."
            )
        if self.is_profile:
            raise ValueError(
                f"{self.source!r} has been reduced to a point and draws as a "
                "profile down depth (see .family), not a map -- there is no time "
                "axis here to play as a movie. Use .plot() instead; it already "
                "shows the whole profile."
            )
        if self.is_section:
            raise ValueError(
                f"{self.source!r} draws as a vertical section (see .family), not "
                "map panels -- there is no axis here to play as a movie yet. "
                "Time-animated sections are a follow-up. Use .plot() instead; it "
                "already shows the whole section."
            )
        field, depth_override = self._facet_field_and_depth()
        self._refuse_labelless_facet(field)
        spec = PlotSpec(
            family="facet_movie",
            items=[self._facet_item(field, depth_override)],
            options=kwargs,
        )
        return render(spec, renderer=renderer)

    def map_locations(self, *, renderer: str = "matplotlib", **kwargs: Any):
        """Map where this field's data sits: the selection over the source's domain.

        Parameters
        ----------
        renderer
            One of ``"matplotlib"`` (default) or ``"holoviews"``.
        **kwargs
            Plot options forwarded to
            :func:`ocean_skill.plot.map_locations.map_locations`. See
            ``docs/plot_styling_reference.md`` for the general styling families.

        From the request (``select``) and catalog metadata alone — nothing is
        opened, so this costs the same whether :meth:`plot`/:meth:`movie` have
        already run or not. See
        :func:`ocean_skill.plot.map_locations.map_locations`.
        """
        from ocean_skill.plot.map_locations import map_locations as _map_locations

        return _map_locations(self, renderer=renderer, **kwargs)

    def save(
        self,
        project: str | None = None,
        *,
        stem: str | None = None,
        renderer: str = "matplotlib",
        **plot_kwargs: Any,
    ) -> dict[str, Path]:
        """Write this field's figure under ``output/<project>/figures/``.

        Parameters
        ----------
        project
            ``str | None`` -- the output project name, used to build
            ``output/<project>/figures/``. ``None`` (default) uses ``self.source``.
        stem
            ``str | None`` -- the figure's filename stem (before ``.png``).
            ``None`` (default) uses this field's standard name, truncated to 24
            characters.
        renderer
            One of ``"matplotlib"`` (default) or ``"holoviews"``, forwarded to
            :meth:`plot`.
        **plot_kwargs
            Forwarded to :meth:`plot` -- see its own ``**kwargs`` entry and
            ``docs/plot_styling_reference.md``.

        The same layout :meth:`ocean_skill.comparison.ComparisonSet.save` writes to,
        minus the metrics table — there is no reference here, so there is nothing to
        score against and no row to write.
        """
        from ocean_skill import outputs

        stem = stem or str(self.standard_name or "field")[:24]
        path = outputs.figures_dir(project or self.source) / f"{stem}.png"
        self.plot(renderer=renderer, save=path, **plot_kwargs)
        return {"figure": path}

    def __repr__(self) -> str:
        facet = self.facet_dims if self._data is not None else "?"
        return (
            f"Field({self.source!r}, {self.variable!r}, "
            f"select={self.select!r}, aggregate={self.aggregate!r}, facet={facet!r})"
        )


class FieldSet:
    """Several fields -- variables and/or sources -- drawn together as one figure.

    Parameters
    ----------
    fields
        ``list[Field]`` -- the members to draw together. Built by :func:`field`
        from a list ``source`` and/or ``variable``; constructing a ``FieldSet``
        directly from hand-built :class:`Field` objects works the same way but
        is not the ordinary path.

    ``osk.field()`` builds one :class:`Field` per entry whenever ``source`` and/or
    ``variable`` is a list, sharing the same ``select``/``aggregate``/``label``/
    ``cache`` (there is no per-entry select yet — see :func:`field`), and pools
    them here. Every member has to reduce the same way for that to mean one
    figure -- all a :attr:`Field.family` of ``"series"``, all ``"profile"``, all
    ``"time_depth"``, or all ``"field_facet"`` (a map) -- and :meth:`plot` says
    so rather than guessing which one it should be when they don't.

    **Series or profile** members overlay: the layout is whatever
    :mod:`ocean_skill.plot.series` or :mod:`ocean_skill.plot.profile` already does
    with several lines -- one panel with a twin axis for two variables (a
    right-hand y axis for a series, a top x axis for a profile —
    ``secondary_y``/``secondary_x`` respectively), one row/column per variable for
    three or more, everything sharing a variable overlaid within a panel and told
    apart by source (dashed by default; pass ``encode={"color": "source"}`` to
    colour by source instead).

    **Time_depth** members stack: one panel per member, down the page.

    **Map** (``field_facet``) members stack too, but side by side rather than
    down a page -- one map panel per member, each with its own colour scale and
    colorbar (different variables, different units, different ranges), under one
    suptitle naming whatever the set shares (see :func:`Field._map_item`). Each
    member must already reduce to *one* map -- a member still faceted over time
    or depth (several timesteps, several levels) is refused, the same way a bare
    :meth:`Field.plot` call on it would ask you to narrow it first, since a set
    of several variables has room for one map per member, not a grid of grids.

    There is nothing to configure beyond what :meth:`Field.plot` already exposes,
    because the composition rule *is* the feature.
    """

    def __init__(self, fields: list[Field]):
        for f in fields:
            if not isinstance(f, Field):
                raise TypeError(
                    f"expected Fields, got {f!r}. A FieldSet is built by osk.field() "
                    "from a list of sources and/or variables -- construct it that "
                    "way rather than by hand."
                )
        self.fields = list(fields)

    def __len__(self) -> int:
        return len(self.fields)

    def __iter__(self):
        return iter(self.fields)

    def __getitem__(self, i):
        return self.fields[i]

    def __repr__(self) -> str:
        return f"FieldSet({self.fields!r})"

    def _items(self) -> list[dict[str, Any]]:
        """Every member's items (series or profile), concatenated into one figure."""
        if self.fields and self.fields[0].family == "profile":
            return [item for f in self.fields for item in f._profile_items()]
        return [item for f in self.fields for item in f._series_items()]

    def _time_depth_items(self) -> list[dict[str, Any]]:
        """Every member's own single ``time_depth`` item, one panel each."""
        return [f._time_depth_item() for f in self.fields]

    def _map_items(self) -> list[dict[str, Any]]:
        """Every member's own single map item, one panel each.

        See :meth:`Field._map_item`.
        """
        return [f._map_item() for f in self.fields]

    @graft_plot_options()
    def plot(self, *, renderer: str = "matplotlib", **kwargs: Any):
        """Draw every member on one figure, laid out by :mod:`plot.series`,
        :mod:`plot.profile`, one panel per member for ``time_depth``, or one map
        panel per member for a set of maps.

        Parameters
        ----------
        renderer
            One of ``"matplotlib"`` (default, static) or ``"holoviews"``
            (interactive).
        **kwargs
            Plot options forwarded to the renderer: option families such as
            ``color_by``, ``marker_by``, ``labels``, ``title``, ``domain``,
            ``robust``, ``figsize``, ``save``, ``secondary_y``/``secondary_x``,
            ``encode`` (e.g. ``{"color": "source"}``), and the ``*_kwargs``
            styling dicts. See ``docs/plot_styling_reference.md`` for the full
            list.

        Every member has to draw the same way -- all a :attr:`Field.family` of
        ``"series"`` (a point over time), all ``"profile"`` (a point down depth,
        at one instant), all ``"time_depth"`` (depth against time, at one
        point -- drawn as a stacked column of panels rather than overlaid or
        faceted lines), or all ``"field_facet"`` (a map, one panel per member,
        each with its own colour scale -- see :meth:`_map_items`) -- for that to
        mean anything. A set that mixes any of those has no single figure that
        is all of them, so this refuses rather than picking one arbitrarily.
        """
        from ocean_skill.comparison import _short_variable_label
        from ocean_skill.plot.registry import render
        from ocean_skill.plot.spec import PlotSpec

        time_depth = [f for f in self.fields if f.family == "time_depth"]
        if time_depth and len(time_depth) < len(self.fields):
            multi_source = len({f.source for f in self.fields}) > 1
            detail = "; ".join(
                (
                    f"{f.source} {_short_variable_label(f.variable)}"
                    if multi_source
                    else _short_variable_label(f.variable)
                )
                + f": {f.family_reason}"
                for f in self.fields
            )
            raise ValueError(
                f"some fields draw as depth against time (see .family) and "
                f"others do not -- {detail}. A time_depth panel has no overlay "
                "or facet composition with a series or profile line, so plot "
                "each group separately with osk.field(source, variable)."
            )
        if time_depth:
            spec = PlotSpec(
                family="time_depth", items=self._time_depth_items(), options=kwargs
            )
            return render(spec, renderer=renderer)
        maps = [f for f in self.fields if f.family == "field_facet"]
        if maps and len(maps) == len(self.fields):
            # Every member draws as a map -- one panel each, its own colour
            # scale and colorbar (different variables, different units), unlike
            # the shared-scale rows below. _map_item refuses any member still
            # faceted over time or depth, so every item reaching here is
            # already a single map.
            items = self._map_items()
            family = "field_facet" if len(items) == 1 else "field_map_grid"
            spec = PlotSpec(family=family, items=items, options=kwargs)
            return render(spec, renderer=renderer)
        not_lines = [f for f in self.fields if f.family not in ("series", "profile")]
        mixed = len({f.family for f in self.fields} & {"series", "profile"}) > 1
        if not_lines or mixed:
            multi_source = len({f.source for f in self.fields}) > 1
            detail = "; ".join(
                (
                    f"{f.source} {_short_variable_label(f.variable)}"
                    if multi_source
                    else _short_variable_label(f.variable)
                )
                + f": {f.family_reason}"
                for f in self.fields
            )
            raise ValueError(
                f"several fields on one figure draw as overlaid lines, but not "
                f"every one of them reduced the same way -- {detail}. Narrow "
                "select= to one lon/lat position -- keeping time standing draws "
                "a series, keeping depth standing with no time draws a profile "
                "-- so every member draws the same way, or plot each one's "
                "maps separately with osk.field(source, variable)."
            )
        family = self.fields[0].family
        spec = PlotSpec(family=family, items=self._items(), options=kwargs)
        return render(spec, renderer=renderer)

    def movie(self, *, renderer: str = "matplotlib", **kwargs: Any):
        """Refuse: a set of fields has nothing shared left to play as frames.

        Parameters
        ----------
        renderer
            Accepted for signature parity with :meth:`Field.movie` but never
            consulted -- this always raises before drawing anything.
        **kwargs
            Accepted for the same reason; never consulted.

        Deliberately does not consult :attr:`Field.family` to tailor this to
        which shape the set happens to be -- that would force every member's
        full prepare just to word an error about not proceeding, the opposite
        of every other read-cheap refusal in this module. Lines over time or
        down depth already show the whole record in one :meth:`plot`; a set of
        maps (see :meth:`Field._map_item`) has had every member narrowed to one
        instant to draw beside the others, with no shared time axis left
        standing on any of them either. Either way, there is nothing left to
        animate as a set -- only one field at a time keeps its own facet axis
        to play.
        """
        raise ValueError(
            "a FieldSet has nothing to play as frames -- lines over time or "
            "down depth already show the whole record in .plot(), and a set "
            "of maps has had every member narrowed to one instant to draw "
            "beside the others, with no shared time axis left to animate. "
            "Movie one variable at a time with osk.field(source, variable), "
            "which keeps its own time facet to play."
        )

    def map_locations(self, *, renderer: str = "matplotlib", **kwargs: Any):
        """Map where this set's fields sit: each member's selection, deduped.

        Parameters
        ----------
        renderer
            One of ``"matplotlib"`` (default) or ``"holoviews"``.
        **kwargs
            Plot options forwarded to
            :func:`ocean_skill.plot.map_locations.map_locations`. See
            ``docs/plot_styling_reference.md`` for the general styling families.

        From each member's request and catalog metadata alone — nothing is
        opened. Members sharing one point/region draw once, not once per
        member. See :func:`ocean_skill.plot.map_locations.map_locations`.
        """
        from ocean_skill.plot.map_locations import map_locations as _map_locations

        return _map_locations(self, renderer=renderer, **kwargs)

    def save(
        self,
        project: str | None = None,
        *,
        stem: str | None = None,
        renderer: str = "matplotlib",
        **plot_kwargs: Any,
    ) -> dict[str, Path]:
        """Write this set's figure under ``output/<project>/figures/``.

        Parameters
        ----------
        project
            ``str | None`` -- the output project name, used to build
            ``output/<project>/figures/``. ``None`` (default) uses the first
            member's ``source``.
        stem
            ``str | None`` -- the figure's filename stem (before ``.png``).
            ``None`` (default) joins the members' deduped variable labels (and
            source labels, if more than one source), truncated to 24 characters.
        renderer
            One of ``"matplotlib"`` (default) or ``"holoviews"``, forwarded to
            :meth:`plot`.
        **plot_kwargs
            Forwarded to :meth:`plot` -- see its own ``**kwargs`` entry and
            ``docs/plot_styling_reference.md``.

        The same layout :meth:`Field.save` writes to, minus the metrics table -- there
        is no reference for any member here either.
        """
        from ocean_skill import outputs
        from ocean_skill.comparison import _short_variable_label

        variable_labels = list(
            dict.fromkeys(_short_variable_label(f.variable) for f in self.fields)
        )
        source_labels = list(dict.fromkeys(f.source for f in self.fields))
        # Distinct sources sharing a variable would otherwise collide on one
        # filename -- fold the source in too, deduped, when there is more than one.
        parts = variable_labels if len(source_labels) <= 1 else (
            variable_labels + source_labels
        )
        stem = stem or "_".join(parts)[:24]
        path = outputs.figures_dir(project or self.fields[0].source) / f"{stem}.png"
        self.plot(renderer=renderer, save=path, **plot_kwargs)
        return {"figure": path}


class Cross:
    """Two vertical sections through one point, one along each grid direction.

    Parameters
    ----------
    along
        :class:`Field` -- the section along ``eta_rho`` (``xi_rho`` held fixed).
    across
        :class:`Field` -- the section along ``xi_rho`` (``eta_rho`` held fixed).
    labels
        Keyword-only ``tuple[str, str]`` -- the two panels' labels, one per
        direction, in ``(along, across)`` order.

    Built by :func:`field` from ``select={"transect": {"cross": ...}}`` (see
    :mod:`ocean_skill.transect`) -- the vertical structure either side of one
    lon/lat point or grid-index pair, cut both ways: one section running
    along ``eta_rho`` (``xi_rho`` held fixed), one along ``xi_rho`` (``eta_rho``
    held fixed), each windowed to a half-width of grid cells either side of
    the shared point rather than the whole line. The two directions share no
    axis to align onto, so each is its own independent :class:`Field` --
    its own (variable, aggregation, depth) reduction, prepared and cached
    separately -- and :meth:`plot` is what draws the pair together, through
    the ``cross`` plot family.

    There is no ``.movie()`` here: a time-animated section is a follow-up (see
    :meth:`Field._require_section_shape`), and a cross would be two of those.
    """

    def __init__(self, along: Field, across: Field, *, labels: tuple[str, str]):
        self.along = along
        self.across = across
        self.labels = labels

    def __repr__(self) -> str:
        return f"Cross({self.along!r}, {self.across!r})"

    def plot(
        self,
        *,
        renderer: str = "matplotlib",
        orientation: str = "vertical",
        **kwargs: Any,
    ):
        """Draw both sections on one figure: stacked (default), or side by side.

        Parameters
        ----------
        renderer
            One of ``"matplotlib"`` (default, static) or ``"holoviews"``
            (interactive).
        orientation
            One of ``"vertical"`` (default, stacked column) or ``"horizontal"``
            (side by side).
        **kwargs
            Plot options forwarded to the renderer: the same option families as
            :meth:`Field.plot` (``title``, ``domain``, ``robust``, ``figsize``,
            ``save``, the ``*_kwargs`` styling dicts). See
            ``docs/plot_styling_reference.md`` for the full list.

        Each direction must itself already draw as a section (see
        :attr:`Field.family`) -- a ``select=``/``aggregate=`` that collapses
        the vertical axis, or leaves some further axis standing, on either
        field raises the same way a lone :meth:`Field.plot` call on it would
        (:meth:`Field._require_section_shape`), since a cross panel has the
        same one shape (depth, along-path distance) a lone section does.
        ``orientation="horizontal"`` lays the two panels side by side instead
        of the default stacked column. ``renderer="holoviews"`` gives the
        interactive version, with no other change.
        """
        from ocean_skill.plot.registry import render
        from ocean_skill.plot.spec import PlotSpec

        self.along._require_section_shape()
        self.across._require_section_shape()
        item_along, item_across = self.along.as_item(), self.across.as_item()
        item_along["label"], item_across["label"] = self.labels
        spec = PlotSpec(
            family="cross",
            items=[item_along, item_across],
            options={"orientation": orientation, **kwargs},
        )
        return render(spec, renderer=renderer)

    def save(
        self,
        project: str | None = None,
        *,
        stem: str | None = None,
        renderer: str = "matplotlib",
        **plot_kwargs: Any,
    ) -> dict[str, Path]:
        """Write this cross's figure under ``output/<project>/figures/``.

        Parameters
        ----------
        project
            ``str | None`` -- the output project name, used to build
            ``output/<project>/figures/``. ``None`` (default) uses
            ``self.along.source``.
        stem
            ``str | None`` -- the figure's filename stem (before ``.png``).
            ``None`` (default) uses the ``along`` field's standard name (truncated
            to 18 characters) with ``"_cross"`` appended.
        renderer
            One of ``"matplotlib"`` (default) or ``"holoviews"``, forwarded to
            :meth:`plot`.
        **plot_kwargs
            Forwarded to :meth:`plot` -- see its own ``**kwargs`` entry and
            ``docs/plot_styling_reference.md``.

        The same layout :meth:`Field.save` writes to, minus the metrics table --
        there is no reference for either direction here either.
        """
        from ocean_skill import outputs

        # Truncated *before* appending "_cross", not after -- a standard_name at or
        # past Field.save()'s own 24-char budget (most of them; CF names run long)
        # would otherwise have the suffix sliced away entirely, leaving a cross
        # figure named indistinguishably from a plain Field's.
        base = str(self.along.standard_name or "field")[:18]
        stem = stem or f"{base}_cross"
        path = outputs.figures_dir(project or self.along.source) / f"{stem}.png"
        self.plot(renderer=renderer, save=path, **plot_kwargs)
        return {"figure": path}


def _expand_cross_transect(
    select: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], tuple[str, str]]:
    """Split ``select={"transect": {"cross": ...}}`` into its two directions.

    Returns ``(select_along, select_across, labels)`` -- two ordinary windowed
    grid-transect ``select``s (see :func:`ocean_skill.transect.as_transect`),
    identical to ``select`` except for ``"transect"``, plus a short label per
    direction naming which grid dimension it holds fixed. A ``{"lon": ...,
    "lat": ...}`` point is passed through to each *unresolved* -- resolving it
    (which cell it lands on, for each direction) happens once each is actually
    prepared (:func:`ocean_skill.transect.grid_slice`), against that field's
    own grid, so building the two selects here reads nothing.
    """
    from ocean_skill.transect import as_transect

    parsed = as_transect(select["transect"])
    point, half_width = parsed["point"], parsed["half_width"]
    rest = {k: v for k, v in select.items() if k != "transect"}

    if "lon" in point and "lat" in point:
        anchor = {"lon": point["lon"], "lat": point["lat"]}
        dim_along, dim_across = point.get("dims", ("eta_rho", "xi_rho"))
        transect_along = {dim_across: anchor, "half_width": half_width}
        transect_across = {dim_along: anchor, "half_width": half_width}
    else:
        (dim_along, idx_along), (dim_across, idx_across) = point.items()
        transect_along = {
            dim_across: idx_across,
            "center": idx_along,
            "half_width": half_width,
        }
        transect_across = {
            dim_along: idx_along,
            "center": idx_across,
            "half_width": half_width,
        }

    labels = (f"along {dim_along}", f"along {dim_across}")
    return (
        {**rest, "transect": transect_along},
        {**rest, "transect": transect_across},
        labels,
    )


def _cross_field(
    source: Any,
    variable: Any,
    select: dict[str, Any] | None,
    *,
    aggregate: dict[str, Any] | None,
    label: str | None,
    cache: bool | None,
    qc: Any,
    detide: Any,
) -> Cross:
    """Build a :class:`Cross`: :func:`field`'s own handling of ``cross``.

    Splits ``select`` by :func:`_expand_cross_transect` into the two
    directions' own selects, then builds one :class:`Field` per direction,
    sharing everything else (``variable``/``aggregate``/``label``/``cache``/
    ``qc``/``detide``) -- the same members :func:`field`'s ordinary,
    non-cross path would build, just two of them instead of one.
    """
    select_along, select_across, labels = _expand_cross_transect(select)
    along = Field(
        source,
        variable,
        select=select_along,
        aggregate=aggregate,
        label=label,
        cache=cache,
        qc=qc,
        detide=detide,
    )
    across = Field(
        source,
        variable,
        select=select_across,
        aggregate=aggregate,
        label=label,
        cache=cache,
        qc=qc,
        detide=detide,
    )
    return Cross(along, across, labels=labels)


def field(
    source: Any,
    variable: Any,
    *,
    select: dict[str, Any] | None = None,
    aggregate: dict[str, Any] | None = None,
    label: str | None = None,
    cache: bool | None = None,
    qc: Any = None,
    detide: Any = False,
) -> Field | FieldSet:
    """Build a :class:`Field`: one model source, no reference.

    Parameters
    ----------
    source
        ``str`` or a list of them -- one catalog entry name, or several to fan
        into a :class:`FieldSet` (one :class:`Field` per source, sharing every
        other argument here).
    variable
        ``str``, ``dict``, or a list of them -- one variable name or a
        combination/``calculate`` spec, or several to fan into a
        :class:`FieldSet`. Not a ``{"test", "reference"}`` pair-spec -- a
        ``Field`` has one source and nothing to give the other side to.
    select
        ``dict[str, Any] | None`` -- axis name -> selection, e.g. ``{"depth":
        "surface"}``, ``{"lon": ..., "lat": ...}``, or ``{"sigma0": ...}`` for an
        isopycnal (ROMS sources only). ``select={"transect": {"cross": ...}}``
        builds a :class:`Cross` instead (source and variable must each be a
        single, non-list value for that). ``None`` (default) leaves the
        vertical axis whole, unlike :func:`ocean_skill.comparison.compare`,
        whose own default is ``"surface"``.
    aggregate
        ``dict[str, Any] | None`` -- axis name -> reduction, e.g. ``{"time":
        "mean"}``, ``{"time": {"resample": "1MS", "reduce": "mean"}}``, or
        ``{"time": {"groupby": "month"}}``. ``None`` (default) aggregates
        nothing.
    label
        ``str | None`` -- legend/title label override, shared by every member
        when ``source``/``variable`` fan out. ``None`` (default) uses each
        member's own source name.
    cache
        ``bool | None`` -- whether to reuse an already-prepared result cached on
        disk. ``None`` (default) follows :func:`ocean_skill.cache.enabled`.
    qc
        Per-call QC override forwarded to :func:`ocean_skill.sources.read` (see
        its own ``qc`` parameter for the full spec): a dict such as ``{"keep":
        ["GOOD", "SUSPECT"]}``, or the string ``"off"``. ``None`` (default) uses
        each entry's own saved QC contract.
    detide
        ``bool | dict | None`` -- ``False``/``None`` (default) leaves every
        member untouched; ``True`` tidal-filters all of them at PL33's own
        default cutoff; a ``{"T": hours}`` dict sets that cutoff explicitly. See
        :func:`ocean_skill.comparison._normalize_detide_side`.

    The counterpart of :func:`ocean_skill.comparison.compare` for the case where there
    is nothing to compare against — a run shown on its own, most usefully as a series
    of panels over time::

        osk.field(
            "gom_bgc",
            "chlorophyll",
            select={"time": slice("2012-01", "2012-06"), "depth": "surface"},
            aggregate={"time": {"resample": "1MS", "reduce": "mean"}},
        ).plot()

    Six monthly means, laid out to suit the domain's own shape. Swap ``resample`` for
    ``{"groupby": "month"}`` and the same call gives a twelve-panel climatology
    instead; the panels label themselves differently, so the two are told apart in the
    figure and not just in the code.

    ``select={"depth": ...}`` is a surface of constant depth; ``select={"sigma0":
    ...}`` asks for an isopycnal instead (ROMS sources only) — see
    :func:`ocean_skill.roms.to_sigma0`. Leaving depth out of ``select`` entirely
    keeps the vertical axis standing, whole — every native s-level for a model
    source (with real depths and thickness weights attached, the same shape
    ``select={"depth": "column"}`` gives a comparison), every reported level for
    an observational one. This differs from :func:`ocean_skill.comparison.compare`,
    whose own unset default is ``"surface"`` — pass that explicitly here for the
    same behavior. A full model domain with nothing narrowed can be large; see the
    memory-use warning this prints if it is.

    **The one exception**: :meth:`Field.plot`/:meth:`Field.movie` on a catalogued
    ``featureType: grid`` source narrow a bare vertical axis to the surface
    themselves, right before drawing — a map has no vertical axis to put on the
    page at all, so the whole-column default above would otherwise mean "facet
    over depth too", almost never what a bare call meant. ``.data`` itself is
    never touched by this; it is a drawing-time default only, and an explicit
    ``select={"depth": ...}``/``aggregate={"Z": ...}`` always wins. A bare,
    genuinely multi-step time axis has no equivalent single default instant, and
    ``.plot()`` says so rather than guessing one — narrow it with ``select=
    {"time": ...}``, reduce it with ``aggregate=``, or call ``.movie()`` to play
    every step instead.

    A ``select`` that narrows both horizontal axes to one position draws as a line
    over whatever axis survives instead of map panels — never a separate call, the
    same ``.plot()``::

        osk.field(
            "run_new", "temperature",
            select={"lon": -144.25, "lat": 49.98, "time": slice("2012-01", "2012-12")},
        ).plot()

    One solid line, no reference to compare against. Keeping *both* time and depth
    standing at that same point instead — a bare ``timeSeriesProfile`` station's
    own shape — draws a third way, one ``time_depth`` panel (colour = value, x =
    time, y = depth)::

        osk.field("ctd_station_HV5", "sea_water_temperature").plot()

    See :attr:`Field.family`/``family_reason`` for why a given call drew what it
    drew. ``renderer="holoviews"`` gives the interactive version with no other
    change.

    ``variable`` also accepts a list, fanning like :func:`ocean_skill.comparison
    .compare`'s ``variables=`` — one :class:`Field` per entry, sharing this same
    ``select``/``aggregate``/``label``/``cache`` (there is no per-variable select yet),
    pooled into a :class:`FieldSet`::

        run = osk.field(
            "run_new", ["temperature", "salinity"],
            select={"lon": -144.25, "lat": 49.98},
        )
        run.plot()                       # 2 variables: one panel, salinity on a
                                          # secondary y-axis (docs/plot_styling_reference.md)
        run.plot(secondary_y=False)      # ...or two stacked panels instead
        run.plot(renderer="holoviews")   # same figure, interactive

    The first entry takes the left (primary) axis when there are exactly two. A dict
    entry (a combination or ``calculate`` spec) should carry its own ``standard_name``
    — panel grouping, colour and the legend all key on it. Exact repeats (including
    alias repeats, like ``"temp"`` and ``"temperature"``) are dropped with a note
    rather than drawn twice. A single-element list still returns a ``FieldSet``, for
    the same reason ``compare(variables=[v])`` still returns a set.

    ``variable="all"`` expands to every variable ``source`` declares in its catalog
    metadata (the same list :func:`ocean_skill.describe` prints), then fans out exactly
    like an explicit list above::

        osk.field("ctd_station_HV1", "all").plot()

    Three or more variables lay out as one profile panel each automatically (see
    :func:`ocean_skill.plot.profile.compose`); ``.plot(cols="variable")`` forces the
    same one-panel-per-variable layout for one or two. ``"all"`` only expands a single
    ``source`` — pass one source name (not a list) when using it, since different
    sources in a list can declare different variables.

    A ``select`` that keeps a horizontal extent standing (a map, not a point) composes
    the same way -- several variables become several map panels side by side, each with
    its own colour scale and colorbar, since different variables carry unrelated units
    and ranges::

        osk.field(
            "second_2wks", ["h", "temperature"],
            select={"time": "2024-06-15", "depth": "surface"},
        ).plot()

    One panel for bathymetry, one for temperature, under one suptitle naming whatever
    the set shares (here, ``surface``). Each member still has to reduce to *one* map
    first -- a member left with several timesteps or levels standing is refused the
    same way a bare :meth:`Field.plot` call on it would ask you to narrow it first
    (see :meth:`Field._map_item`); a time-invariant variable like ``h`` needs no
    ``select={"time": ...}`` of its own, since it has no time axis to narrow.

    ``source`` accepts a list the same way -- one :class:`Field` per source, sharing
    this same ``variable``/``select``/``aggregate``/``label``/``cache``, pooled into
    a :class:`FieldSet`. This is what :func:`ocean_skill.catalog.find` chains into
    directly, since it always returns a list of names::

        osk.field(osk.find(catalog="CTD mooring"), "temperature").plot()

    One line per mooring, overlaid on one figure (dashed by source; pass
    ``encode={"color": "source"}`` to colour by source instead of variable). Passing
    a list for *both* ``source`` and ``variable`` fans the full cross product --
    one ``Field`` per ``(source, variable)`` pair, deduplicated the same way.
    """
    source_is_list = isinstance(source, (list, tuple))
    variable_is_list = isinstance(variable, (list, tuple))
    if source_is_list and not source:
        raise ValueError(
            "source=[] names nothing to draw -- often osk.find() matching "
            "nothing; check its filters. Pass one source name, or a list of "
            'them -- osk.field(["src_a", "src_b"], "temperature").'
        )
    if variable_is_list and not variable:
        raise ValueError(
            "variable=[] names nothing to draw. Pass one spec (a name or a "
            'dict), or a list of them -- osk.field(src, ["temperature", '
            '"salinity"]).'
        )
    if isinstance(variable, str) and variable == "all":
        if source_is_list:
            raise ValueError(
                'variable="all" expands one source\'s declared variables; a '
                "list of sources may each declare different ones. Call "
                "osk.field() once per source, or pass the explicit variable "
                "names you want."
            )
        from ocean_skill import catalog

        names = list(catalog.resolve(source).metadata.get("variables") or [])
        if not names:
            raise ValueError(
                f'{source!r} declares no variables to expand for variable='
                f'"all" -- check osk.describe({source!r}), or pass explicit '
                "names."
            )
        variable = names
        variable_is_list = True
    transect = select.get("transect") if isinstance(select, dict) else None
    is_cross = isinstance(transect, dict) and "cross" in transect
    if is_cross and (source_is_list or variable_is_list):
        raise ValueError(
            "select={'transect': {'cross': ...}} draws two sections from one "
            "source and one variable -- it has no per-source/per-variable "
            "fan-out the way an ordinary select= does (see the module "
            "docstring on FieldSet's own shared-select limit). Call "
            "osk.field() once per source/variable instead."
        )
    if not source_is_list and not variable_is_list:
        if is_cross:
            return _cross_field(
                source,
                variable,
                select,
                aggregate=aggregate,
                label=label,
                cache=cache,
                qc=qc,
                detide=detide,
            )
        return Field(
            source,
            variable,
            select=select,
            aggregate=aggregate,
            label=label,
            cache=cache,
            qc=qc,
            detide=detide,
        )
    from ocean_skill.comparison import _canonical
    from ocean_skill.vocabulary import resolve_and_report

    sources = list(source) if source_is_list else [source]
    variables = list(variable) if variable_is_list else [variable]
    # Resolve each requested variable to its canonical standard_name once, up
    # front -- so the one "name resolved to..." warning fires once per variable
    # rather than once per (source, variable) pair a source-list fans out to
    # (mirrors compare()'s identical up-front resolution, see
    # ocean_skill.comparison.compare). Field.__init__ resolves again on the way
    # in, but a name already canonical triggers no second warning.
    variables = [
        resolve_and_report(v, context="Field variable=") if isinstance(v, str) else v
        for v in variables
    ]
    members = [
        Field(
            s,
            v,
            select=select,
            aggregate=aggregate,
            label=label,
            cache=cache,
            qc=qc,
            detide=detide,
        )
        for s in sources
        for v in variables
    ]
    kept: list[Field] = []
    seen: set[tuple[str, str]] = set()
    dropped = 0
    for f in members:
        key = (f.source, _canonical(f.variable))
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        kept.append(f)
    if dropped:
        print(f"  dropped {dropped} duplicate(s)")
    return FieldSet(kept)
