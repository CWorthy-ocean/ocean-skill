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
    def family(self) -> str:
        """The plot family this field's own shape admits: ``series`` or ``field_facet``.

        No argument selects it, the same as :attr:`ocean_skill.comparison
        .Comparison.family` — the prepared data's shape decides.
        """
        return "series" if self.is_series else "field_facet"

    @property
    def family_reason(self) -> str:
        """Why :attr:`family` came out the way it did, for tracing a surprise."""
        if self.is_series:
            return "drawn as a line: the selection leaves one place, so the surviving time axis is the x"
        return "drawn as map panels: a horizontal extent survives"

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

    def as_item(self) -> dict[str, Any]:
        """Return this field as a spec item."""
        from ocean_skill.comparison import _depth_label, _selected_depth

        row_dim, facet_dim = self.facet_dims
        # the vertical selection, spelled for a label ("surface", "50 m", "σ₀ = 26.5
        # kg/m³"). A renderer cannot recover it from the field once the transform has
        # collapsed the axis, and a plot of one level that does not say which level is
        # a plot of nothing in particular -- see the interactive movie's title.
        requested = _selected_depth(self.select)
        return {
            "field": self.data,
            "facet_dim": facet_dim,
            "row_dim": row_dim,
            "units": self.data.attrs.get("units"),
            "standard_name": self.standard_name,
            "depth": _depth_label(requested),
            "label": self.label or self.source,
        }

    def plot(self, *, renderer: str = "matplotlib", **kwargs: Any):
        """Draw this field: as map panels, or as a line over time at one place.

        Which of the two is decided by the prepared data's own shape, never an
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
        else:
            if point_of(self.data) is not None:
                # A point with no surviving time: neither a map (no horizontal
                # extent left) nor a line (nothing to run it along) can be drawn --
                # field_facet would otherwise try to lay out panels of a field with
                # no axes at all, which fails confusingly further in.
                raise ValueError(
                    f"{self.source!r} has been reduced to one place with no "
                    f"surviving time axis ({sorted(self.data.dims)} standing), so "
                    "there is no horizontal extent left for map panels and no time "
                    "axis for a line either. Keep time standing for a series (drop "
                    "a select={'time': ...} that pins it to an instant, or an "
                    "aggregate that collapses it), or widen select= to keep a "
                    "horizontal extent for a map."
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
        spec = PlotSpec(family="facet_movie", items=[self.as_item()], options=kwargs)
        return render(spec, renderer=renderer)

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
        """Every member's series items, concatenated into one figure's worth."""
        return [item for f in self.fields for item in f._series_items()]

    def plot(self, *, renderer: str = "matplotlib", **kwargs: Any):
        """Draw every member's variable on one figure, laid out by :mod:`plot.series`.

        Every member has to draw as a line (see :attr:`Field.family`) for that to mean
        anything -- a set that mixes a point and a map has no single figure that is
        both, so this refuses rather than picking one arbitrarily.
        """
        from ocean_skill.comparison import _short_variable_label
        from ocean_skill.plot.registry import render
        from ocean_skill.plot.spec import PlotSpec

        not_series = [f for f in self.fields if f.family != "series"]
        if not_series:
            detail = "; ".join(
                f"{_short_variable_label(f.variable)}: {f.family_reason}"
                for f in not_series
            )
            raise ValueError(
                f"several variables on one figure draw as overlaid lines, but not "
                f"every one of them reduced to a point -- {detail}. Narrow select= "
                "to one lon/lat position so each variable is a series, or plot each "
                "variable's maps separately with osk.field(source, variable)."
            )
        spec = PlotSpec(family="series", items=self._items(), options=kwargs)
        return render(spec, renderer=renderer)

    def movie(self, *, renderer: str = "matplotlib", **kwargs: Any):
        """Refuse: a set of variables drawn as lines has nothing to play as frames."""
        raise ValueError(
            "this set holds several variables of one source drawn as lines over "
            "time -- there is nothing to play as a movie. Use .plot(); it already "
            "shows the whole series. For a movie of maps, give osk.field() one "
            "variable."
        )

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
