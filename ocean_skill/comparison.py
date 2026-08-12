"""The compare layer: ``Comparison`` (one pair) and ``compare`` (fan-out).

A :class:`Comparison` holds a reference and a test source for one variable plus a
selection; it reads both, reduces them to a comparable 2-D field, aligns them
(test → reference), and exposes the difference, metrics and a plot. Roles are assigned
here, not in the catalog: ``diff = test − reference``, alignment brings test onto
reference. :func:`compare` fans over the reference × test × variable × depth
cross-product and collects the results into a :class:`ComparisonSet` that can write one
tidy metrics table and one stacked figure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "SURFACE",
    "Comparison",
    "ComparisonSet",
    "compare",
    "is_surface_request",
    "prepare_source",
]


#: Sentinel meaning "the model's own top level" as distinct from an explicit request
#: for the field interpolated to literal 0 m — see :func:`_prepare`.
SURFACE = "surface"


def _domain_of(source: str) -> tuple[float, float, float, float] | None:
    """Return ``(lon_min, lat_min, lon_max, lat_max)`` for a source's catalog extent.

    Used to draw the model-domain outline on a map (as in Abigale Wyatt's
    ``Obs_comparisons.ipynb``) without the caller having to know or repeat the model's
    bounding box. Returns ``None`` if the source is unresolvable or the catalog entry
    doesn't declare a geospatial extent, in which case no box is drawn.
    """
    from ocean_skill.catalog import resolve

    try:
        meta = resolve(source).metadata
    except KeyError:
        return None
    keys = (
        "geospatial_lon_min",
        "geospatial_lat_min",
        "geospatial_lon_max",
        "geospatial_lat_max",
    )
    lon_min, lat_min, lon_max, lat_max = (meta.get(k) for k in keys)
    if None in (lon_min, lat_min, lon_max, lat_max):
        return None
    # Catalogs may declare 0-360 (ROMS' native convention); maps are drawn in ±180, so
    # normalize. This assumes the domain doesn't itself straddle the anti-meridian.
    lon_min, lon_max = (((lo + 180) % 360) - 180 for lo in (lon_min, lon_max))
    return lon_min, lat_min, lon_max, lat_max


def is_depth_band(depth: Any) -> bool:
    """Report whether ``depth`` asks for an average over a band rather than a level.

    ``{"min": 0, "max": 10}`` — the YAML-friendly spelling of a range, since YAML has
    no slice literal. A band is averaged, not interpolated: see
    :func:`ocean_skill.roms.depth_average`.
    """
    return isinstance(depth, dict) and {"min", "max"} <= set(depth)


def is_surface_request(depth: Any) -> bool:
    """Report whether ``depth`` means "surface" rather than a literal depth in metres.

    Unset (``None``) and the ``"surface"`` sentinel both mean "the model's own top
    level" (:func:`ocean_skill.roms.surface`). A literal ``0``/``0.0`` is a real depth
    request instead — interpolated like any other depth via
    :func:`ocean_skill.roms.to_depth`, which may legitimately come back all-NaN (with a
    warning) if the topmost cell centre already sits below 0 m. Conflating the two
    silently hid that distinction; now only the explicit sentinel gets the shortcut.
    """
    return depth is None or (isinstance(depth, str) and depth.lower() == SURFACE)


def _depth_label(depth: Any) -> str:
    """Format a depth for labels/repr: ``"surface"``, ``"0-10 m"`` or ``"<n> m"``."""
    if is_surface_request(depth):
        return SURFACE
    if is_depth_band(depth):
        return f"{float(depth['min']):g}-{float(depth['max']):g} m"
    return f"{float(depth):g} m"


def _variable_label(spec: Any) -> str:
    """Short display name for a variable spec, whether a name or a combination."""
    if isinstance(spec, str):
        return spec.split("_of_")[-1]
    if name := spec.get("standard_name"):
        return str(name).split("_of_")[-1]
    how, names = next(iter((k, v) for k, v in spec.items() if k != "standard_name"))
    return f"{how}({'+'.join(map(str, names))})"


#: Keys naming the vertical axis in a `select`, in any accepted spelling.
_VERTICAL_KEYS = frozenset({"depth", "Z", "vertical", "z"})


def _vertical_only(agg: dict[str, Any] | None) -> dict[str, Any]:
    """Return the part of an aggregation spec addressing the vertical axis."""
    return {k: v for k, v in (agg or {}).items() if k in _VERTICAL_KEYS}


def _without_vertical(agg: dict[str, Any] | None) -> dict[str, Any]:
    """Return the part of an aggregation spec for every axis but the vertical."""
    return {k: v for k, v in (agg or {}).items() if k not in _VERTICAL_KEYS}


#: Time reduction applied when a caller names none — the long-standing behaviour,
#: now expressed in the same grammar as any other aggregation.
DEFAULT_AGGREGATE: dict[str, Any] = {"time": "mean"}


def _prepare(
    obj,
    meta: dict[str, Any],
    variable: Any,
    select: dict[str, Any],
    aggregate: dict[str, Any] | None = None,
):
    """Reduce a source to one comparable 2-D field (variable, aggregation, depth).

    ``variable`` is anything :func:`ocean_skill.operators.resolve_variable` accepts:
    a name, or a combination such as ``{"sum": ["spChl", "diatChl", "diazChl"]}``.
    ``aggregate`` is a :func:`ocean_skill.operators.aggregate` spec, defaulting to
    :data:`DEFAULT_AGGREGATE`.

    Resolving the variable *first*, and bailing out when it is absent, is
    deliberate: falling through to the whole dataset is both wasteful and unsafe —
    a bare ``.mean("time")`` chokes on whatever non-numeric fields ride along (ROMS'
    ``spherical`` flag), so "variable not found" must fail closed.
    """
    from ocean_skill import operators, roms, units

    depth = next((select[k] for k in _VERTICAL_KEYS if k in select), None)
    surface = is_surface_request(depth)
    band = is_depth_band(depth)
    agg = DEFAULT_AGGREGATE if aggregate is None else aggregate

    da = operators.resolve_variable(obj, variable)
    if da is None:
        return None, None

    # Order matters twice over. Selection precedes reduction, or "the mean of
    # January" would average the whole record. And the *non-vertical* reduction runs
    # before the vertical step, so an expensive s-coordinate transform sees one
    # time-mean field rather than every time step -- the vertical part of the
    # aggregation then collapses whatever the vertical selection left standing.
    horizontal = {k: v for k, v in select.items() if k not in _VERTICAL_KEYS}
    da = operators.select(da, horizontal)
    da = operators.aggregate(da, _without_vertical(agg))

    if meta.get("model") == "roms":
        # The vertical transform needs a Dataset carrying the grid; a DataArray
        # brings its coordinates (h, mask, Cs_r, ...) along, so this round trip
        # keeps everything roms.surface/to_depth reads.
        name = da.name or "field"
        sub = da.to_dataset(name=name)
        # A DataArray only carries coordinates sharing its dimensions, so the
        # interface-grid variables (on s_w, which a tracer has no part of) are
        # dropped by to_dataset. They are exactly what depth_average needs.
        #
        # Static grid fields only -- deliberately not `zeta`, which still carries the
        # time dimension this field has already been averaged over; re-attaching it
        # would make z_rho time-varying against a time-less field and break the xgcm
        # transform. Both depth routines fall back to zeta=0, which is the
        # approximation already in force here and is small against metre-scale cells.
        for grid_var in ("sigma_w", "Cs_w", "sigma_r", "Cs_r", "h"):
            if grid_var in obj.variables and grid_var not in sub.variables:
                sub = sub.assign({grid_var: obj[grid_var]})
        if surface:
            sub = roms.surface(sub, meta)
        elif band:
            # A band is averaged over native cells with thickness weights, not
            # interpolated: above the shallowest cell *centre* -- 7 m down in deep
            # water on this grid -- there is nothing to interpolate from, so a
            # target grid over 0-10 m would be mostly NaN offshore.
            # A *selection*: keeps the cells and their thickness weights, so the
            # vertical aggregation below decides how to collapse them.
            sub = roms.depth_band(sub, meta, depth["min"], depth["max"])
        else:
            # A list interpolates to several levels in one field, which the vertical
            # aggregation then collapses; a scalar gives one level and no axis.
            targets = (
                [float(d) for d in depth]
                if isinstance(depth, list | tuple)
                else float(depth)
            )
            sub = roms.to_depth(sub, meta, targets)
        da = sub[name]
        # Squeeze only a single interpolated level: a scalar depth request collapses
        # the axis by itself (as `.sel` does everywhere), while a list or band leaves
        # several levels for the vertical aggregation to reduce. Squeezing
        # unconditionally used to discard every level but the first, silently.
        if "z" in da.dims and da.sizes["z"] == 1:
            da = da.isel(z=0)
    else:
        # observational depth axes vary: real metres, or an index with depths alongside
        zname = next(
            (n for n in ("depth", "depth_surface", "lev") if n in da.dims), None
        )
        if zname is not None:
            levels = (
                np.asarray(obj["Depth"])
                if "Depth" in obj.variables
                else np.asarray(da[zname])
            )
            if band:
                # Observational products report at standard levels, so a band is just
                # the levels inside it -- no cell thicknesses to weight by, and
                # inventing some would imply structure the product does not claim.
                # Left standing for the vertical aggregation, like the model side.
                inside = np.where((levels >= depth["min"]) & (levels <= depth["max"]))[
                    0
                ]
                if inside.size == 0:  # band falls between levels; take the nearest
                    inside = [int(np.abs(levels - depth["min"]).argmin())]
                attrs = dict(da.attrs)
                da = da.isel({zname: list(inside)})
                da.attrs = attrs
                da.attrs["actual_depth"] = float(np.mean(levels[list(inside)]))
            elif isinstance(depth, list | tuple):
                keep = [int(np.abs(levels - float(d)).argmin()) for d in depth]
                attrs = dict(da.attrs)
                da = da.isel({zname: keep})
                da.attrs = attrs
                da.attrs["actual_depth"] = float(np.mean(levels[keep]))
            else:
                target = 0.0 if surface else float(depth)
                k = int(np.abs(levels - target).argmin())
                da = da.isel({zname: k})
                da.attrs["actual_depth"] = float(levels[k])

    if da is None:
        return None, None
    # Now the vertical reduction, on whatever the vertical selection left: one level
    # (already collapsed), several interpolated levels, or a weighted band.
    da = operators.aggregate(da, _vertical_only(agg))
    return units.convert_units(da), da.attrs.get("actual_depth")


def prepare_source(
    source: str,
    variable: Any,
    select: dict[str, Any] | None,
    aggregate: dict[str, Any] | None,
    *,
    use_cache: bool = True,
    refresh: bool = False,
):
    """Reduce one source to its prepared field, via the lane cache.

    Keyed on this source alone, so the same model/variable/depth is prepared once
    however many references it is compared against — the vertical transform is the
    pipeline's most expensive step and does not depend on the other side of the
    comparison at all. That independence is also why this is a module-level function
    rather than a :class:`Comparison` method: a model-only
    :class:`~ocean_skill.field.Field` wants exactly this lane and exactly this cache
    entry, and a second copy of the keying rules would be a second chance to key them
    differently.

    Returns ``(DataArray, actual_depth)``, or ``(None, None)`` if the source does not
    carry the variable.
    """
    import ocean_skill as osk
    from ocean_skill import cache as _cache
    from ocean_skill.catalog import resolve

    key = _cache.key_for_prepared(
        source=source,
        variable=variable,
        select={**(select or {}), "_aggregate": aggregate},
    )
    if use_cache and not refresh:
        hit = _cache.load_field(key)
        if hit is not None:
            return hit

    da, depth = _prepare(
        osk.read(source),
        resolve(source).metadata,
        variable,
        dict(select or {}),
        aggregate,
    )
    # Computed here rather than left lazy: the vertical transform is the most
    # expensive step in the pipeline, and a dask-backed lane re-runs it on every
    # consumer that touches values -- the metrics, then each panel of each plot.
    # to_zarr below computes the same graph anyway and discards the result, so
    # this costs nothing when caching is on and saves a full recompute when a
    # save fails or caching is off. See align() for the same move on the pair.
    if da is not None:
        da = da.load()
    # A miss is cached; "this source doesn't carry this variable" is not, since
    # that is cheap to rediscover and would otherwise persist past a fixed catalog.
    if use_cache and da is not None:
        _cache.save_field(key, da, depth)
    return da, depth


class Comparison:
    """One reference↔test comparison for a single variable at a single depth."""

    def __init__(
        self,
        *,
        reference: str,
        test: str,
        variable: Any,
        select: dict[str, Any] | None = None,
        aggregate: dict[str, Any] | None = None,
        method: str = "conservative_normed",
        label: str | None = None,
        cache: bool | None = None,
    ):
        from ocean_skill.vocabulary import resolve_and_report

        self.reference_name = reference
        self.test_name = test
        # A plain name resolves through the vocabulary (short name, canonical
        # standard_name, or any alias, in any case) so everything downstream sees
        # one consistent name. A dict is a combination spec for
        # ocean_skill.operators and is carried through as given.
        self.variable = (
            resolve_and_report(variable, context="Comparison variable=")
            if isinstance(variable, str)
            else variable
        )
        self.select = dict(select or {})
        self.aggregate = aggregate
        self.method = method
        self.label = label
        # None = follow the global setting (on unless osk.cache.disable()); an
        # explicit True/False overrides it for this comparison only.
        self.cache = cache
        self._aligned = None
        self._metrics = None
        self._actual_depth = None

    @property
    def standard_name(self) -> str | None:
        """The CF name this comparison represents, for colormaps/labels/metrics.

        A combination carries it explicitly (``{"sum": [...], "standard_name": ...}``);
        without one there is no single CF name, and downstream falls back to defaults.
        """
        from ocean_skill.operators import DERIVED

        spec = self.variable
        if isinstance(spec, str):
            # A DERIVED key is a name for a spec, not a CF name -- expand it, or
            # colormaps, plot labels and the metrics table all see the key.
            spec = DERIVED.get(spec, spec)
        return spec if isinstance(spec, str) else spec.get("standard_name")

    @property
    def _cache_key(self) -> str:
        from ocean_skill import cache as _cache

        return _cache.key_for(
            test=self.test_name,
            reference=self.reference_name,
            variable=self.variable,
            select={**self.select, "_aggregate": self.aggregate},
            method=self.method,
        )

    def _use_cache(self) -> bool:
        """Whether this comparison caches: its own setting, else the global one."""
        from ocean_skill import cache as _cache

        return _cache.enabled() if self.cache is None else self.cache

    def _prepare_lane(self, source: str, use_cache: bool, refresh: bool):
        """Reduce one source to its comparable 2-D field, via the lane cache."""
        return prepare_source(
            source,
            self.variable,
            self.select,
            self.aggregate,
            use_cache=use_cache,
            refresh=refresh,
        )

    # -- pipeline ---------------------------------------------------------------
    def align(self, *, refresh: bool = False):
        """Read both sources, reduce them, and regrid test onto reference.

        The result is always computed, never lazy — see the ``.load()`` note below —
        so repeat consumers (metrics, a redrawn figure) read values rather than
        re-running the pipeline that produced them.

        The result is cached to disk (see :mod:`ocean_skill.cache`) and reused on a
        later call with the same sources, variable, selection and method — including
        from a fresh process, which is the case that matters, since an in-process
        repeat is already served by :attr:`aligned`'s own memo. Pass
        ``refresh=True`` to recompute and overwrite a stale entry, or construct the
        comparison with ``cache=False`` to bypass disk entirely.
        """
        from ocean_skill import align as _align
        from ocean_skill import cache as _cache

        use_cache = self._use_cache()
        if use_cache and not refresh:
            hit = _cache.load(self._cache_key)
            if hit is not None:
                self._aligned = hit
                # actual_depth rides along in attrs precisely so a cached result
                # restores the same state a freshly computed one would have.
                self._actual_depth = hit.attrs.get("actual_depth")
                return self._aligned

        r, r_depth = self._prepare_lane(self.reference_name, use_cache, refresh)
        t, _ = self._prepare_lane(self.test_name, use_cache, refresh)
        if r is None or t is None:
            missing = self.reference_name if r is None else self.test_name
            raise KeyError(f"{self.variable!r} not available in {missing!r}")
        self._actual_depth = r_depth
        # .load() so a computed result is a computed result: a cache hit hands back
        # eager arrays (open_zarr(...).load()), and a miss must leave the comparison
        # in the same state or the session that *fills* the cache is the slowest one
        # -- every plot re-reading the sources through a graph that was already
        # evaluated once to write the entry.
        self._aligned = _align.align(
            t, r, method=self.method, test_name="test", reference_name="reference"
        ).load()
        if r_depth is not None:
            self._aligned.attrs["actual_depth"] = r_depth
        if use_cache:
            _cache.save(self._cache_key, self._aligned)
        return self._aligned

    @property
    def aligned(self):
        """The aligned pair (test, reference, difference); computed on first access."""
        if self._aligned is None:
            self.align()
        return self._aligned

    data = aligned  # alias: prepared arrays for bespoke plotting

    def metrics(self, **extra: Any) -> dict[str, Any]:
        """Compute (and cache) the metric record for this comparison."""
        from ocean_skill import metrics as _metrics

        if self._metrics is None:
            self._metrics = _metrics.compute(
                self.aligned,
                test_name="test",
                reference_name="reference",
                variable=self.standard_name or str(self.variable),
                test=self.test_name,
                reference=self.reference_name,
                depth=self.select.get("depth", SURFACE),
                obs_depth=self._actual_depth,
                regrid=self.method,
                **extra,
            )
        return self._metrics

    def difference(self):
        """Return the ``test − reference`` field on the reference grid."""
        return self.aligned["difference"]

    def as_item(self) -> dict[str, Any]:
        """Return this comparison as a spec item (aligned pair plus metadata)."""
        return {
            "aligned": self.aligned,
            "metrics": self.metrics(),
            "units": self.aligned["reference"].attrs.get("units"),
            "standard_name": self.standard_name,
            "label": self.label,
            # this comparison's own source names, for its row's column titles —
            # not necessarily the same pair as other rows in the same set (a
            # compare() fan-out commonly pairs one variable per reference source).
            "labels": (self.test_name, self.reference_name),
        }

    def plot(self, *, renderer: str = "matplotlib", **kwargs: Any):
        """Render as a ``test | reference | difference`` row.

        Goes through the renderer registry, so ``renderer="holoviews"`` gives the
        interactive version of the same plot with no other change.
        """
        from ocean_skill.plot.registry import render
        from ocean_skill.plot.spec import PlotSpec

        kwargs.setdefault("labels", (self.test_name, self.reference_name))
        # Outline the test (model) source's own declared extent, matching Abigale
        # Wyatt's side-by-side plots — pass domain=None to suppress it.
        kwargs.setdefault("domain", _domain_of(self.test_name))
        spec = PlotSpec(family="field_row", items=[self.as_item()], options=kwargs)
        return render(spec, renderer=renderer)

    def save(
        self,
        project: str | None = None,
        *,
        stem: str | None = None,
        figure: bool = True,
        metrics: bool = True,
        renderer: str = "matplotlib",
        **plot_kwargs: Any,
    ) -> dict[str, Path]:
        """Write this comparison's figure and metrics row under ``output/<project>/``.

        The single-row counterpart of :meth:`ComparisonSet.save`; see there and
        :mod:`ocean_skill.outputs` for the layout and why deliverables are kept out
        of the cache.
        """
        stem = stem or _variable_label(self.variable)[:24]
        return ComparisonSet([self]).save(
            project or f"{self.test_name}_vs_{self.reference_name}",
            stem=stem,
            figure=figure,
            metrics=metrics,
            renderer=renderer,
            **plot_kwargs,
        )

    def __repr__(self) -> str:
        return (
            f"<Comparison {_variable_label(self.variable)[:24]} "
            f"{self.test_name} vs {self.reference_name} "
            f"@ {_depth_label(self.select.get('depth', SURFACE))}>"
        )


class ComparisonSet:
    """A set of comparisons: stacked rows in one figure, one tidy metrics table."""

    def __init__(self, comparisons: list[Comparison]):
        self.comparisons = comparisons

    def __len__(self) -> int:
        return len(self.comparisons)

    def __iter__(self):
        return iter(self.comparisons)

    def __getitem__(self, i):
        return self.comparisons[i]

    def metrics(self):
        """Return every comparison's metrics as a tidy DataFrame (one row each)."""
        import pandas as pd

        return pd.DataFrame([c.metrics() for c in self.comparisons])

    def write_metrics(self, out_dir: str | Path, stem: str = "metrics") -> Path:
        """Write the tidy metrics table to ``<out_dir>/metrics/<stem>.csv``."""
        from ocean_skill import metrics as _metrics

        return _metrics.write(
            [c.metrics() for c in self.comparisons], out_dir, stem=stem
        )

    def save(
        self,
        project: str | None = None,
        *,
        stem: str = "comparison",
        figure: bool = True,
        metrics: bool = True,
        renderer: str = "matplotlib",
        **plot_kwargs: Any,
    ) -> dict[str, Path]:
        """Write this set's figure and metrics table under ``output/<project>/``.

        Deliverables, not cache — see :mod:`ocean_skill.outputs` for why they are
        kept apart. Returns the paths written, keyed ``"figure"``/``"metrics"``.

        ``project`` defaults to a slug of the first comparison's ``test vs
        reference``. Extra keyword arguments are forwarded to the renderer, so the
        styling knobs in ``docs/plot_styling_reference.md`` all work here too.
        """
        from ocean_skill import outputs

        if not self.comparisons:
            raise ValueError("nothing to save: this set has no comparisons")
        first = self.comparisons[0]
        project = project or f"{first.test_name}_vs_{first.reference_name}"

        written: dict[str, Path] = {}
        if metrics:
            written["metrics"] = self.write_metrics(
                outputs.project_dir(project), stem=stem
            )
        if figure:
            fig_path = outputs.figures_dir(project) / f"{stem}.png"
            self.plot(renderer=renderer, save=fig_path, **plot_kwargs)
            written["figure"] = fig_path
        return written

    def _items(self) -> list[dict[str, Any]]:
        """Spec items for every comparison in the set."""
        return [{**c.as_item(), "row_label": c.label} for c in self.comparisons]

    def plot(self, *, renderer: str = "matplotlib", **kwargs: Any):
        """Render all comparisons as stacked rows in one figure."""
        from ocean_skill.plot.registry import render
        from ocean_skill.plot.spec import PlotSpec

        items = self._items()
        if not self.comparisons:
            raise ValueError(
                "no comparisons to plot: every pair was skipped. Check that the "
                "reference actually offers the requested variables (its catalog "
                "metadata lists them under 'variables')."
            )
        first = self.comparisons[0]
        kwargs.setdefault("labels", (first.test_name, first.reference_name))
        # Outlines the first row's test (model) extent; rows sharing one test source
        # (the common case) all get the same box. Pass domain=None to suppress it, or
        # your own bbox if rows mix test sources with different domains.
        kwargs.setdefault("domain", _domain_of(first.test_name))
        return render(
            PlotSpec(family="field_grid", items=items, options=kwargs),
            renderer=renderer,
        )

    def taylor(self, *, renderer: str = "matplotlib", **kwargs: Any):
        """Taylor diagram of the set (correlation + variability; blind to bias)."""
        from ocean_skill.plot.registry import render
        from ocean_skill.plot.spec import PlotSpec

        return render(
            PlotSpec(family="taylor", items=self._items(), options=kwargs),
            renderer=renderer,
        )

    def target(self, *, renderer: str = "matplotlib", **kwargs: Any):
        """Target diagram of the set (bias vs signed centred RMSD)."""
        from ocean_skill.plot.registry import render
        from ocean_skill.plot.spec import PlotSpec

        return render(
            PlotSpec(family="target", items=self._items(), options=kwargs),
            renderer=renderer,
        )

    def summary(self, *, renderer: str = "matplotlib", **kwargs: Any):
        """Taylor and Target side by side for the whole set."""
        from ocean_skill.plot.registry import render
        from ocean_skill.plot.spec import PlotSpec

        return render(
            PlotSpec(family="paired", items=self._items(), options=kwargs),
            renderer=renderer,
        )

    def __repr__(self) -> str:
        return f"<ComparisonSet: {len(self)} comparisons>"


def compare(
    *,
    reference,
    test,
    variables: list[Any],
    depths=None,
    select: dict[str, Any] | None = None,
    aggregate: dict[str, Any] | None = None,
    method: str = "conservative_normed",
    skip_missing: bool = True,
    cache: bool | None = None,
    refresh: bool = False,
) -> ComparisonSet:
    """Fan over reference × test × variable × depth and return a :class:`ComparisonSet`.

    ``reference`` and ``test`` each take a source name or a list; ``variables`` is a
    list of anything :mod:`ocean_skill.vocabulary` recognizes — a short vocabulary key
    (``"oxygen"``), a canonical CF standard_name, or any alias (in any case) —
    resolved to its canonical standard_name once here (warning once per variable if
    the name given wasn't already the canonical form, so it's never unclear which
    variable was actually used). Pairs where the variable is absent are skipped with a
    message rather than failing the whole run, unless ``skip_missing=False``.

    A variable may also be a *combination* — ``{"sum": ["spChl", "diatChl",
    "diazChl"], "standard_name": CHL}`` — see :mod:`ocean_skill.operators`.
    ``aggregate`` names how dimensions collapse (default ``{"time": "mean"}``);
    ``{"time": {"groupby": "month", "reduce": "mean"}}`` gives a climatology, and
    ``{"time": {"resample": "1MS", "reduce": "mean"}}`` consecutive months. Both keep
    an axis standing, which a comparison cannot use — see
    :func:`ocean_skill.field.field` for the model-only path that plots those as panels.

    ``variables`` is required, deliberately not inferred from the sources' catalog
    metadata: a reference or test with several declared variables gives no way to
    tell *which* of them the caller actually wants compared, so guessing would either
    have to pick arbitrarily or silently expand to "compare everything in common" —
    neither is a default worth having quietly happen. Naming the variable(s) is the
    one thing this call cannot infer on your behalf.

    ``depths`` defaults to ``("surface",)`` — the model's own top level, via
    :func:`ocean_skill.roms.surface`. A literal ``0`` is a *different*, real request:
    the field interpolated to exactly 0 m, which for a model whose topmost cell centre
    sits a few metres down legitimately comes back all-NaN (with a warning) rather than
    silently reusing the surface field.

    Each pair's aligned result is cached to disk and reused on a later run with the
    same arguments (see :mod:`ocean_skill.cache`, which prints where once per
    process). ``cache=False`` bypasses it; ``refresh=True`` recomputes and overwrites
    — what to use after rerunning a model, since entries are keyed on source and
    selection rather than on file contents.
    """
    from ocean_skill.catalog import resolve
    from ocean_skill.vars import short_name
    from ocean_skill.vocabulary import equivalent_names, resolve_and_report

    # depths defaults to the vertical entry already in `select`, if any, so the two
    # spellings agree instead of one clobbering the other.
    if depths is None:
        vertical = next(
            (select[k] for k in _VERTICAL_KEYS if select and k in select), SURFACE
        )
        depths = (vertical,)

    refs = [reference] if isinstance(reference, str) else list(reference)
    tests = [test] if isinstance(test, str) else list(test)
    # Resolve each requested variable to its canonical standard_name once, up front
    # -- both so _offers() below matches against catalog metadata correctly (which
    # declares the canonical name), and so the one "name resolved to..." warning
    # fires once per variable rather than once per (ref, test, depth) combination
    # this fans out to.
    variables = [
        resolve_and_report(v, context="compare variables=") if isinstance(v, str) else v
        for v in variables
    ]

    def _offers(source: str, variable: Any) -> bool:
        """Report whether the source advertises this variable (or an equivalent).

        A combination has more than one way to be satisfied — every component, or
        the ``standard_name`` it falls back to for sources using the other
        convention (MODIS ships total chlorophyll; MARBL ships the components).
        Any one complete option is enough.
        """
        from ocean_skill.operators import spec_names
        from ocean_skill.vocabulary import is_known

        try:
            declared = resolve(source).metadata.get("variables")
        except KeyError:
            return True
        if not declared:
            return True  # no metadata to filter on; let the read decide
        declared = set(declared)
        options = spec_names(variable)
        if any(all(equivalent_names(n) & declared for n in opt) for opt in options):
            return True  # positively offered

        # Not positively offered -- but "absent" and "unknowable" are different.
        # A catalog's `variables` list holds CF standard_names, so it can never
        # confirm or deny a raw model variable like `spChl`; concluding "absent"
        # from that silently drops a comparison that would have worked. Only
        # exclude when every name involved is one the catalog could have listed.
        return not all(is_known(n) for opt in options for n in opt)

    out: list[Comparison] = []
    for var in variables:
        # Pair each variable with the sources that actually carry it, rather than
        # forming a blind cross-product. Observational catalogs are usually one
        # entry per variable (WOA ships nitrate and phosphate separately), and so
        # is model output: a ROMS run writes physics and BGC to different streams,
        # on different time axes, so they cannot be one source. Filtering both
        # sides lets you pass every stream as `test=` and have each variable find
        # the one that has it, instead of failing half the cross-product.
        matching = [r for r in refs if _offers(r, var)]
        matching_tests = [t for t in tests if _offers(t, var)]
        if not matching:
            print(f"  no reference offers {var!r}; skipped")
            continue
        if not matching_tests:
            print(f"  no test offers {var!r}; skipped")
            continue
        for ref in matching:
            for tst in matching_tests:
                for d in depths:
                    # `depths` is sugar for fanning select's vertical entry over a
                    # list -- one comparison per value. Writing it into the same key
                    # the caller may have used means a select={"Z": ...} is honoured
                    # rather than silently overridden by the default of "surface".
                    sel = {**(select or {})}
                    for key in _VERTICAL_KEYS:
                        sel.pop(key, None)
                    sel["depth"] = d
                    # Label only what varies across the set: repeating the variable
                    # name on every point of a single-variable fan-out just collides.
                    short = (
                        short_name(var)
                        if isinstance(var, str)
                        else (_variable_label(var))
                    )
                    many_vars, many_depths = len(variables) > 1, len(depths) > 1
                    if many_vars and many_depths:
                        label = f"{short} {_depth_label(d)}"
                    elif many_depths:
                        label = _depth_label(d)
                    else:
                        label = short
                    c = Comparison(
                        reference=ref,
                        test=tst,
                        variable=var,
                        select=sel,
                        aggregate=aggregate,
                        method=method,
                        label=label,
                        cache=cache,
                    )
                    try:
                        c.align(refresh=refresh)
                    except KeyError as exc:
                        if not skip_missing:
                            raise
                        print(f"  skipped: {exc}")
                        continue
                    out.append(c)
    return ComparisonSet(out)
