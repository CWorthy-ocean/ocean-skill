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

The reduction, the caching and the vocabulary handling are all the comparison lane's,
reached through :func:`ocean_skill.comparison.prepare_source` — a model field prepared
for a comparison and the same field prepared on its own are the same field, and they
share one cache entry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["Field", "field"]


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
    """One source reduced to a map, or to a series of maps over one axis.

    Parameters mirror :class:`~ocean_skill.comparison.Comparison` where they mean the
    same thing, so moving between the two is a change of class rather than of grammar.
    ``aggregate`` is the one that matters here: a spec that fully collapses time gives
    a single panel, while ``groupby``/``resample`` leave the axis that becomes the
    panels.
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
        """Draw one panel per value of :attr:`facet_dim`.

        Goes through the renderer registry, so ``renderer="holoviews"`` gives the
        interactive version of the same plot with no other change.
        """
        from ocean_skill.plot.registry import render
        from ocean_skill.plot.spec import PlotSpec

        spec = PlotSpec(family="field_facet", items=[self.as_item()], options=kwargs)
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
        rather than writing a one-frame movie.
        """
        from ocean_skill.plot.registry import render
        from ocean_skill.plot.spec import PlotSpec

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


def field(
    source: str,
    variable: Any,
    *,
    select: dict[str, Any] | None = None,
    aggregate: dict[str, Any] | None = None,
    label: str | None = None,
    cache: bool | None = None,
) -> Field:
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
    """
    return Field(
        source,
        variable,
        select=select,
        aggregate=aggregate,
        label=label,
        cache=cache,
    )
