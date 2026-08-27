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
share one cache entry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["Field", "FieldSet", "field"]


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


class Field:
    """One source reduced to a map, a series of maps over one axis, or a line.

    Parameters mirror :class:`~ocean_skill.comparison.Comparison` where they mean the
    same thing, so moving between the two is a change of class rather than of grammar.
    ``aggregate`` is the one that matters here: a spec that fully collapses time gives
    a single panel, while ``groupby``/``resample`` leave the axis that becomes the
    panels. A ``select`` that narrows both horizontal axes to one position instead
    draws as a line over whatever axis survives — see :attr:`family`.

    Holds exactly one variable. A list of them belongs to :func:`field`, which fans it
    into a :class:`FieldSet` instead of a single ``Field``.
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
    ):
        from ocean_skill.comparison import _require_pair_spec, as_select, is_pair_spec
        from ocean_skill.vocabulary import resolve_and_report

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
        # select/aggregate carry the same pair-spec spelling in a Comparison, one
        # lane's own select/aggregate -- a Field is one source, so a pair here is the
        # same mistake as a pair-spec variable, and gets the same clear error.
        for name, arg in (("select", select), ("aggregate", aggregate)):
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
        ``("z", "time")`` a grid of levels by periods. See :func:`_facet_dims`.
        """
        from ocean_skill.align import _lat_name, _lon_name

        da = self.data
        lon, lat = _lon_name(da), _lat_name(da)
        spatial: set[str] = set()
        for name in (lon, lat):
            if name is not None:
                spatial |= {str(d) for d in da[name].dims}
        return _facet_dims(da, spatial)

    @property
    def facet_dim(self) -> str | None:
        """The axis across the columns, or ``None`` for a single map."""
        return self.facet_dims[1]

    @property
    def is_series(self) -> bool:
        """Whether this field's prepared data is a place through time, not a map.

        Mirrors :attr:`ocean_skill.comparison.Comparison.is_series`: a select that
        narrows both horizontal axes to one position
        (:func:`ocean_skill.align.point_of`) leaves nothing for :attr:`facet_dims`
        to lay out as columns, and a surviving time axis is exactly what a line
        needs for its x. Read off the data's own shape, never an argument.
        """
        from ocean_skill.align import point_of
        from ocean_skill.operators import resolve_dim

        da = self.data
        return point_of(da) is not None and resolve_dim(da, "T") in da.dims

    @property
    def is_profile(self) -> bool:
        """Whether this field's prepared data is a place through depth, not a map.

        Mirrors :attr:`is_series` one axis over: a select that narrows both
        horizontal axes to one position leaves a surviving vertical axis standing
        with no time axis to draw a line through instead (see :attr:`is_series`,
        checked first in :attr:`family` -- a select that keeps *both* time and
        depth standing still draws as a series, one line per level, exactly as
        before). The water column a station is, at one instant.
        """
        from ocean_skill.align import point_of
        from ocean_skill.operators import resolve_dim

        da = self.data
        if point_of(da) is None:
            return False
        tdim = resolve_dim(da, "T")
        zdim = resolve_dim(da, "Z")
        return (tdim is None or tdim not in da.dims) and (
            zdim is not None and zdim in da.dims
        )

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

        ``series`` (a line over time at one place), ``profile`` (a line down
        depth at one place, one instant), ``section`` (a cut through depth and
        along-path distance), or ``field_facet`` (map panels) — no argument
        selects it, the same as :attr:`ocean_skill.comparison.Comparison.family`
        — the prepared data's shape decides.
        """
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
        if self.is_series:
            return "drawn as a line: the selection leaves one place, so the surviving time axis is the x"
        if self.is_profile:
            return "drawn as a profile: the selection leaves one place and one instant, so the surviving depth axis is the y"
        if self.is_section:
            return "drawn as a section: select={'transect': ...} leaves a cut through space, with depth on the other axis"
        return "drawn as map panels: a horizontal extent survives"

    def extremum(self, kind: str = "max") -> Any:
        """Locate this field's min/max: value, lon/lat, grid indices, snapshot.

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
        """
        import xarray as xr

        from ocean_skill.operators import resolve_dim

        da = self.data
        tdim = resolve_dim(da, "T")
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

        items = []
        for k in range(da.sizes[zdim]):
            level = da.isel({zdim: k})
            item = {"aligned": xr.Dataset({"value": level}), "metrics": None, **base}
            # actual_depth lives on the *item's* Dataset, not the DataArray, since
            # that is what _depth_of (plot/series.py) reads -- the same convention
            # Comparison.align() uses for its own aligned pair.
            if zdim in level.coords:
                item["aligned"].attrs["actual_depth"] = float(level[zdim])
            items.append(item)
        return items

    def _profile_items(self) -> list[dict[str, Any]]:
        """Return this field's data as one single-source profile item.

        The vertical twin of :meth:`_series_items`, and simpler: a profile's
        depth axis survives *whole*, one line down the water column, rather than
        being fanned into one item per level -- depth is the axis a profile line
        draws against, not a fact several series lines are told apart by (see
        :mod:`ocean_skill.plot.profile`). Anything beyond it left standing is
        refused, the same as :meth:`_series_items` refuses a third axis a line
        has no room for.
        """
        import xarray as xr

        from ocean_skill.operators import resolve_dim

        da = self.data
        zdim = resolve_dim(da, "Z")
        extra = [str(d) for d in da.dims if d != zdim]
        if extra:
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
        return [{"aligned": xr.Dataset({"value": da}), "metrics": None, **base}]

    def as_item(self) -> dict[str, Any]:
        """Return this field as a spec item."""
        from ocean_skill.comparison import _depth_label, _selected_depth

        # the vertical selection, spelled for a label ("surface", "50 m", "σ₀ = 26.5
        # kg/m³"). A renderer cannot recover it from the field once the transform has
        # collapsed the axis, and a plot of one level that does not say which level is
        # a plot of nothing in particular -- see the interactive movie's title.
        requested = _selected_depth(self.select)
        item = {
            "field": self.data,
            "units": self.data.attrs.get("units"),
            "standard_name": self.standard_name,
            "depth": _depth_label(requested),
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

    def plot(self, *, renderer: str = "matplotlib", **kwargs: Any):
        """Draw this field: as map panels, a vertical section, a profile, or a line.

        Which of the four is decided by the prepared data's own shape, never an
        argument — see :attr:`family`. Goes through the renderer registry, so
        ``renderer="holoviews"`` gives the interactive version of the same plot
        with no other change.
        """
        from ocean_skill.align import point_of
        from ocean_skill.plot.registry import render
        from ocean_skill.plot.spec import PlotSpec

        if self.family == "series":
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
            spec = PlotSpec(
                family="field_facet", items=[self.as_item()], options=kwargs
            )
        return render(spec, renderer=renderer)

    def movie(self, *, renderer: str = "matplotlib", **kwargs: Any):
        """Play :attr:`facet_dim` instead of laying it out: this field as a movie.

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
        spec = PlotSpec(family="facet_movie", items=[self.as_item()], options=kwargs)
        return render(spec, renderer=renderer)

    def map_locations(self, *, renderer: str = "matplotlib", **kwargs: Any):
        """Map where this field's data sits: the selection over the source's domain.

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
    """Several variables of one source, drawn together as one ``series`` figure.

    ``osk.field(source, [v1, v2, ...])`` builds one :class:`Field` per variable,
    sharing the same ``select``/``aggregate``/``label``/``cache`` (there is no
    per-variable select yet — see :func:`field`), and pools them here. The layout is
    whatever :mod:`ocean_skill.plot.series` already does with several variables: one
    panel and a secondary y-axis for two, one row per variable for three or more, all
    overlaid within a row by source. There is nothing to configure beyond what
    :meth:`Field.plot` already exposes, because the composition rule *is* the feature.

    Series only — a set with a member that reduces to a map rather than a point has
    nothing in common to draw as one figure, and :meth:`plot` says so rather than
    guessing which one it should be.
    """

    def __init__(self, fields: list[Field]):
        for f in fields:
            if not isinstance(f, Field):
                raise TypeError(
                    f"expected Fields, got {f!r}. A FieldSet is built by osk.field() "
                    "from a list of variables -- construct it that way rather than "
                    "by hand."
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

    def plot(self, *, renderer: str = "matplotlib", **kwargs: Any):
        """Draw every member's variable on one figure, laid out by :mod:`plot.series`
        or :mod:`plot.profile`.

        Every member has to draw the same way -- all a :attr:`Field.family` of
        ``"series"`` (a point over time) or all ``"profile"`` (a point down
        depth, at one instant) -- for that to mean anything. A set that mixes
        either with a map, or a series with a profile, has no single figure
        that is both, so this refuses rather than picking one arbitrarily.
        """
        from ocean_skill.comparison import _short_variable_label
        from ocean_skill.plot.registry import render
        from ocean_skill.plot.spec import PlotSpec

        not_lines = [f for f in self.fields if f.family not in ("series", "profile")]
        mixed = len({f.family for f in self.fields} & {"series", "profile"}) > 1
        if not_lines or mixed:
            detail = "; ".join(
                f"{_short_variable_label(f.variable)}: {f.family_reason}"
                for f in self.fields
            )
            raise ValueError(
                f"several variables on one figure draw as overlaid lines, but not "
                f"every one of them reduced the same way -- {detail}. Narrow "
                "select= to one lon/lat position -- keeping time standing draws "
                "a series, keeping depth standing with no time draws a profile "
                "-- so every member draws the same way, or plot each variable's "
                "maps separately with osk.field(source, variable)."
            )
        family = self.fields[0].family
        spec = PlotSpec(family=family, items=self._items(), options=kwargs)
        return render(spec, renderer=renderer)

    def movie(self, *, renderer: str = "matplotlib", **kwargs: Any):
        """Refuse: a set of variables drawn as lines has nothing to play as frames."""
        raise ValueError(
            "this set holds several variables of one source drawn as lines over "
            "time or down depth -- there is nothing to play as a movie. Use "
            ".plot(); it already shows the whole figure. For a movie of maps, "
            "give osk.field() one variable."
        )

    def map_locations(self, *, renderer: str = "matplotlib", **kwargs: Any):
        """Map where this set's fields sit: each member's selection, deduped.

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

        The same layout :meth:`Field.save` writes to, minus the metrics table -- there
        is no reference for any member here either.
        """
        from ocean_skill import outputs
        from ocean_skill.comparison import _short_variable_label

        stem = stem or "_".join(
            _short_variable_label(f.variable) for f in self.fields
        )[:24]
        path = outputs.figures_dir(project or self.fields[0].source) / f"{stem}.png"
        self.plot(renderer=renderer, save=path, **plot_kwargs)
        return {"figure": path}


def field(
    source: str,
    variable: Any,
    *,
    select: dict[str, Any] | None = None,
    aggregate: dict[str, Any] | None = None,
    label: str | None = None,
    cache: bool | None = None,
) -> Field | FieldSet:
    """Build a :class:`Field`: one model source, no reference.

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
    :func:`ocean_skill.roms.to_sigma0`.

    A ``select`` that narrows both horizontal axes to one position draws as a line
    over whatever axis survives instead of map panels — never a separate call, the
    same ``.plot()``::

        osk.field(
            "run_new", "temperature",
            select={"lon": -144.25, "lat": 49.98, "time": slice("2012-01", "2012-12")},
        ).plot()

    One solid line, no reference to compare against (see
    :attr:`Field.family`/``family_reason`` for why a given call drew what it drew).
    ``renderer="holoviews"`` gives the interactive version with no other change.

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
    """
    if isinstance(variable, (list, tuple)):
        if not variable:
            raise ValueError(
                "variable=[] names nothing to draw. Pass one spec (a name or a "
                'dict), or a list of them -- osk.field(src, ["temperature", '
                '"salinity"]).'
            )
        from ocean_skill.comparison import _canonical

        members = [
            Field(source, v, select=select, aggregate=aggregate, label=label, cache=cache)
            for v in variable
        ]
        kept: list[Field] = []
        seen: set[str] = set()
        dropped = 0
        for f in members:
            key = _canonical(f.variable)
            if key in seen:
                dropped += 1
                continue
            seen.add(key)
            kept.append(f)
        if dropped:
            print(f"  dropped {dropped} duplicate variable(s)")
        return FieldSet(kept)
    return Field(
        source,
        variable,
        select=select,
        aggregate=aggregate,
        label=label,
        cache=cache,
    )
