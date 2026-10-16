r"""Pydantic models for a suite YAML: the declarative form of ``osk.field``/``compare``.

A suite is a *list of pages* -- each one a single :func:`ocean_skill.field.field`,
:func:`ocean_skill.comparison.compare`, :func:`ocean_skill.comparison.summary`, or
:class:`ocean_skill.xy.XY`/:class:`~ocean_skill.xy.TS` call, expressed as YAML instead
of Python -- plus shared defaults, output settings, and an
optional live-run refresh step. :mod:`ocean_skill.workflows.pages` expands a
:class:`SuiteConfig` into a flat list of fully-resolved page dicts (``for_each``
fanned out, ``{placeholder}``\ s filled in, ``time: latest``/``month: run`` resolved
to literal values); :mod:`ocean_skill.workflows.run` draws each one and assembles a
report directory. See ``docs/suites.md`` for the full grammar and ``suites/*.yaml``
for worked examples.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "COMPARE_DRAWS",
    "SUMMARY_KINDS",
    "SUMMARY_SCOPES",
    "PageConfig",
    "RefreshConfig",
    "RefreshSourceConfig",
    "RunInfoConfig",
    "SuiteConfig",
]

#: The mutually exclusive page kinds, in the order their error messages list them.
_PAGE_KINDS = ("field", "compare", "summary", "section", "XY", "TS")

#: The ``draw:`` values a ``compare:`` page accepts: which
#: :class:`~ocean_skill.comparison.ComparisonSet` method draws it, with the page's
#: ``plot:`` mapping forwarded as that method's keyword arguments. ``plot`` (the
#: default) is the per-family ``.plot()``; the rest each draw the whole set as one
#: figure.
COMPARE_DRAWS = ("plot", "metrics_series", "target", "taylor")


#: What a ``summary:`` page's ``kind:`` may name: the overviews
#: :func:`ocean_skill.comparison.summary` draws, and ``table`` (the pooled metrics as
#: a table figure). A page may give one, or a list -- one figure each.
SUMMARY_KINDS = ("both", "taylor", "target", "portrait", "table")

#: Which compare pages a ``summary:`` pools: ``all`` of those drawn so far (the
#: default) or only ``section`` -- those since the last ``section:`` divider.
SUMMARY_SCOPES = ("all", "section")

#: The ``kind:`` of a ``summary:`` page that gives none.
DEFAULT_SUMMARY_KINDS = ("both", "table")


def summary_kinds(title: str, summary: dict[str, Any]) -> list[str]:
    """Return a ``summary:`` page's ``kind:`` as a list, raising on anything unknown.

    The one check of a summary page's ``kind:`` and ``scope:``, called at load time
    (before ``{placeholders}`` are filled) and again after they are, by
    :func:`ocean_skill.workflows.pages.expand`. A missing ``kind:`` is
    :data:`DEFAULT_SUMMARY_KINDS`.
    """
    kind = summary.get("kind", list(DEFAULT_SUMMARY_KINDS))
    kinds = [kind] if isinstance(kind, str) else kind
    if not isinstance(kinds, list) or not kinds:
        raise ValueError(
            f"page {title!r}: summary: kind: must be one kind or a non-empty list "
            f"of them, from {list(SUMMARY_KINDS)} -- got {kind!r}"
        )
    for k in kinds:
        if k not in SUMMARY_KINDS:
            raise ValueError(
                f"page {title!r}: summary: kind: {k!r} is not a known kind -- "
                f"choose from {list(SUMMARY_KINDS)}"
            )
    if len(set(kinds)) != len(kinds):
        raise ValueError(
            f"page {title!r}: summary: kind: {kinds} names a kind twice -- each "
            "kind is one figure"
        )
    scope = summary.get("scope", "all")
    if scope not in SUMMARY_SCOPES:
        raise ValueError(
            f"page {title!r}: summary: scope: {scope!r} is not a known scope -- "
            f"choose from {list(SUMMARY_SCOPES)}"
        )
    return kinds


def check_draw(title: str, draw: str, contours: Any) -> None:
    """Raise unless ``draw`` is a known drawing method that takes these ``contours``.

    The one check of a compare page's ``draw:`` value, called at load time (before
    ``{placeholders}`` are filled) and again after they are, by
    :func:`ocean_skill.workflows.pages.expand`.
    """
    if draw not in COMPARE_DRAWS:
        raise ValueError(
            f"page {title!r}: draw: {draw!r} is not a known drawing "
            f"method -- choose one of {list(COMPARE_DRAWS)}"
        )
    if draw != "plot" and contours:
        raise ValueError(
            f"page {title!r}: contours: only applies to draw: plot (this "
            f"page draws with {draw})"
        )


class RefreshSourceConfig(BaseModel):
    """One stream to rebuild before the suite runs -- forwarded to ``make_kerchunk``.

    ``subchunk`` (``{dim: chunk length}``) splits the stored chunks further, e.g.
    ``{s_rho: 1, eta_rho: 74}`` so a page that needs whole water columns (anything
    matched to a depth) reads a band of rows without loading every level of the
    whole domain at once.

    ``files`` is a glob, or a list of globs for output spread over several folders (the
    task folders of a run restarted in pieces): the matched files are joined in time
    order, and where two give the same time step the later one wins -- the later glob of
    the list, within one glob the later path in sorted order
    (:func:`ocean_skill.workflows.run._refresh_sources`).
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    files: str | list[str]
    ref: str
    grid: str | None = None
    keep: str | None = None
    min_age: float | None = None
    subchunk: dict[str, int] | None = None

    @field_validator("files")
    @classmethod
    def _files_not_empty(cls, files: str | list[str]) -> str | list[str]:
        if isinstance(files, list) and not files:
            raise ValueError("refresh.sources files: needs at least one glob")
        return files


class RefreshForcingConfig(BaseModel):
    """One roms-tools forcing file set to catalog before the suite runs.

    ``kind: boundary`` is forwarded to :func:`ocean_skill.build.add_boundary_sources`
    (one entry per side, ``<name>_<side>``), ``kind: surface`` to
    :func:`ocean_skill.build.add_forcing_source` (one entry, ``<name>``) and ``kind:
    initial_conditions`` to :func:`ocean_skill.build.add_initial_conditions` (one entry,
    ``<name>``, one file). ``files`` is a glob or a list of globs, each expanded and
    sorted, joined in the order given.
    ``ref``, when given, is where a kerchunk reference over the files is kept; it is
    rebuilt only when the matched files change, and opens in seconds where the files
    themselves can take minutes.

    ``subchunk`` (``{dim: chunk length}``) splits the files' stored chunks along the
    time dimension, e.g. ``{bry_time: 1}``, so files whose time chunks differ in length
    (monthly files of 28 to 31 steps) can be joined into one reference, which they
    cannot otherwise. It is only valid for uncompressed files (netCDF-3, or netCDF-4
    without compression) and needs ``ref``.

    ``strip`` (boundary only) names sides to also join into one more entry,
    ``<name>_strip``: ``[west, south]`` is the west boundary run north to south then the
    south boundary west to east (:func:`ocean_skill.build.join_boundary_sides`). ``hc``
    is the s-coordinate's critical depth, for a grid file (and, for initial conditions,
    a data file) that states none.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    kind: Literal["boundary", "surface", "initial_conditions"]
    files: str | list[str]
    grid: str
    variables: list[str] | None = None
    climatology: bool = False
    sides: list[str] | None = None
    strip: list[str] | None = None
    time_coord: str | None = None
    hc: float | None = None
    ref: str | None = None
    subchunk: dict[str, int] | None = None

    @model_validator(mode="after")
    def _kind_keys(self) -> RefreshForcingConfig:
        if self.subchunk is not None and self.ref is None:
            raise ValueError(
                f"refresh.forcing {self.name!r}: subchunk: shapes the kerchunk "
                "reference, so it needs ref:"
            )
        if self.kind != "boundary" and self.sides is not None:
            raise ValueError(
                f"refresh.forcing {self.name!r}: sides: is for kind: boundary"
            )
        if self.kind != "boundary" and self.strip is not None:
            raise ValueError(
                f"refresh.forcing {self.name!r}: strip: is for kind: boundary"
            )
        if self.kind != "surface" and self.time_coord is not None:
            raise ValueError(
                f"refresh.forcing {self.name!r}: time_coord: is for kind: surface "
                "(a boundary file's time is always bry_time)"
            )
        if self.kind == "initial_conditions":
            extra = [
                key
                for key in ("ref", "subchunk")
                if getattr(self, key) is not None
            ] + (["climatology"] if self.climatology else [])
            if extra:
                raise ValueError(
                    f"refresh.forcing {self.name!r}: {extra} do not apply to kind: "
                    "initial_conditions (one file, read as it is)"
                )
        if self.strip is not None:
            from ocean_skill.transect import check_boundary_sides

            strip = check_boundary_sides(
                self.strip, f"refresh.forcing {self.name!r}: strip: {self.strip!r}"
            )
            sides = self.sides if self.sides is not None else strip
            outside = [s for s in strip if s not in sides]
            if outside:
                raise ValueError(
                    f"refresh.forcing {self.name!r}: strip: names {outside}, which are "
                    f"not among sides: {self.sides}"
                )
        return self


class RefreshConfig(BaseModel):
    """``refresh:`` block: rebuild a live run's kerchunk references first.

    ``sources`` is forwarded, unchanged in shape, to
    :func:`ocean_skill.workflows.run._refresh_sources`; ``forcing`` to
    :func:`ocean_skill.workflows.run._refresh_forcing`, into the same catalog.
    """

    model_config = ConfigDict(extra="forbid")

    catalog: str
    sources: list[RefreshSourceConfig] = Field(default_factory=list)
    forcing: list[RefreshForcingConfig] = Field(default_factory=list)


class RunInfoConfig(BaseModel):
    """What ``report.pdf``'s cover page says about the simulation (``run_info:``).

    Every key is optional. ``blueprint``/``workplan`` name a C-Star resolved blueprint
    and workplan directly -- absolute, or relative to the suite file -- instead of
    letting :func:`ocean_skill.workflows.cover.gather` find them from ``refresh:``;
    ``name`` overrides the simulation name the cover is titled with; ``notes`` is
    printed on the cover as written, in order.
    """

    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    blueprint: str | None = None
    workplan: str | None = None
    notes: dict[str, Any] = Field(default_factory=dict)


class PageConfig(BaseModel):
    """One page of a report: ``field``, ``compare``, ``summary``, ``section``, XY or TS.

    ``field``/``compare``/``summary`` are the raw keyword dicts a page will pass to
    :func:`ocean_skill.field.field`, :func:`ocean_skill.comparison.compare`, or
    :func:`ocean_skill.comparison.summary` -- left as ``dict[str, Any]`` (not
    individually typed) because those functions already validate their own
    arguments; a page's job is only to say *which one* and *with what*, and to let
    ``for_each``/``{placeholder}`` reach into any of its values.

    ``XY`` and ``TS`` are the property-property plots (:class:`ocean_skill.xy.XY`;
    ``TS`` is the salinity-temperature preset): a ``members:`` mapping of label -> one
    ``osk.field()`` call each, plus ``x:``/``y:`` (``XY`` only), ``regions:`` and
    ``at_center:``. Their shape is checked in :mod:`ocean_skill.workflows.pages`, like
    ``then:``'s. They take ``for_each:`` and ``plot:`` but not ``then:`` (there is no
    single field to chain on).

    ``section`` is the odd one out: not a figure at all but a text-only divider page in
    ``report.pdf`` (a title plus the notes text it holds -- ``""`` for a title-only
    divider; a YAML block scalar, ``>-`` or ``|``, works for longer notes). It has no
    data to fan out, chain, or plot, so ``for_each:``, ``then:`` and a non-empty
    ``plot:`` beside it are schema errors rather than silently ignored keys. Written
    ``section:`` with nothing after it, YAML reads ``None`` -- which is "no kind at
    all" here, not an empty divider; the error says to write ``section: ""``.

    ``for_each`` fans this one page into several -- one page per element of the
    Cartesian product of its lists (see :mod:`ocean_skill.workflows.pages`).

    ``then`` runs a chain of methods on the object ``field:`` builds, before
    ``plot:`` draws it -- the suite-YAML form of a Python chain like
    ``osk.field(...).extremum("min").series(variables=[...])``. Only the step
    *names* and their basic shape (a bare name, or one ``{name: args}`` mapping
    per list entry) are checked here; the fuller check -- argument validation, and
    that consecutive steps' types actually fit together -- happens in
    :func:`ocean_skill.workflows.pages.expand`, which is also where the whole
    chain runs against real step names (see :data:`~ocean_skill.workflows.pages
    .STEP_REGISTRY`) rather than a hardcoded list here, so the two can never drift
    apart. ``field:``-only for now.

    ``caption`` is a line or paragraph of text drawn under the page's figure in
    ``report.pdf`` (and recorded in ``manifest.json``): any page that draws a figure,
    but not a ``section:`` divider, whose own text is its ``section:`` string.

    ``contours`` overlays contour lines on a ``field:``/``compare:`` page's figure --
    ``true`` for the plotted object's own, or a mapping of overrides for a second
    object (see :func:`ocean_skill.workflows.pages.expand`, which checks its value
    once placeholders are filled; only the page kind is checked here).
    """

    model_config = ConfigDict(extra="forbid")

    title: str
    field: dict[str, Any] | None = None
    compare: dict[str, Any] | None = None
    summary: dict[str, Any] | None = None
    section: str | None = None
    XY: dict[str, Any] | None = None
    TS: dict[str, Any] | None = None
    for_each: dict[str, Any] | None = None
    plot: dict[str, Any] = Field(default_factory=dict)
    then: list[str | dict[str, Any]] | None = None
    contours: Any = None
    draw: str | None = None
    caption: str | None = None

    @model_validator(mode="after")
    def _exactly_one_kind(self) -> PageConfig:
        kinds = [k for k in _PAGE_KINDS if getattr(self, k) is not None]
        if len(kinds) != 1:
            hint = ""
            if not kinds and "section" in self.model_fields_set:
                # ``section:`` (or ``section: null``) reads as None, i.e. absent --
                # the title-only divider is spelled with an empty string.
                hint = (
                    ' -- a title-only divider is written section: "", not a bare '
                    "section: (YAML reads that as null)"
                )
            raise ValueError(
                f"page {self.title!r} must have exactly one of field:/compare:/"
                f"summary:/section:/XY:/TS: -- found {kinds or 'none'}{hint}"
            )
        return self

    @property
    def kind(self) -> Literal["field", "compare", "summary", "section", "XY", "TS"]:
        """Which page kind this is (``field``, ``compare``, ... ``XY`` or ``TS``)."""
        for k in _PAGE_KINDS:
            if getattr(self, k) is not None:
                return k  # type: ignore[return-value]
        raise AssertionError("_exactly_one_kind already enforces this")

    @model_validator(mode="after")
    def _section_is_text_only(self) -> PageConfig:
        if self.kind != "section":
            return self
        # Checked before the then: validator below, so a section carrying a chain
        # gets this message (a divider has no data to chain on) rather than the
        # generic "then: is only supported on field: pages".
        for key, present in (
            ("for_each", self.for_each is not None),
            ("then", self.then is not None),
            ("plot", bool(self.plot)),
            ("caption", self.caption is not None),
        ):
            if present:
                raise ValueError(
                    f"page {self.title!r}: a section: page is a text-only divider "
                    f"in report.pdf -- it takes no {key}: (only title: and the "
                    "section: text itself)"
                )
        return self

    @model_validator(mode="after")
    def _then_is_field_only_and_well_shaped(self) -> PageConfig:
        if self.then is None:
            return self
        if self.kind != "field":
            raise ValueError(
                f"page {self.title!r}: then: is only supported on field: pages "
                f"(this page is {self.kind}:)"
            )
        # Lazy import: keeps this module's own import light (see the module
        # docstring) -- ocean_skill.workflows.pages imports only the stdlib and
        # pydantic at module level, so this costs nothing beyond the import
        # itself, paid once per suite load, not once per page.
        from ocean_skill.workflows.pages import STEP_REGISTRY

        for item in self.then:
            if isinstance(item, str):
                name = item
            else:
                if len(item) != 1:
                    raise ValueError(
                        f"page {self.title!r}: then: {item!r} must name exactly "
                        "one step per list entry"
                    )
                name = next(iter(item))
            if name not in STEP_REGISTRY:
                raise ValueError(
                    f"page {self.title!r}: then: {name!r} is not a known step "
                    f"-- choose one of {sorted(STEP_REGISTRY)}"
                )
        return self

    @model_validator(mode="after")
    def _contours_is_field_or_compare_only(self) -> PageConfig:
        if self.contours is None:
            return self
        if self.kind not in ("field", "compare"):
            raise ValueError(
                f"page {self.title!r}: contours: is only supported on field: and "
                f"compare: pages (this page is {self.kind}:)"
            )
        return self

    @model_validator(mode="after")
    def _summary_kind_and_scope_are_known(self) -> PageConfig:
        if self.summary is not None and "{" not in repr(
            (self.summary.get("kind"), self.summary.get("scope"))
        ):
            summary_kinds(self.title, self.summary)
        return self

    @model_validator(mode="after")
    def _draw_is_compare_only_and_known(self) -> PageConfig:
        if self.draw is None:
            return self
        if self.kind != "compare":
            raise ValueError(
                f"page {self.title!r}: draw: is only supported on compare: pages "
                f"(this page is {self.kind}:)"
            )
        if "{" not in self.draw:
            check_draw(self.title, self.draw, self.contours)
        return self


class SuiteConfig(BaseModel):
    """A suite YAML: the whole file. See ``docs/suites.md`` for the grammar.

    ``name`` is a label the user chooses -- it names the report directory and prefixes
    the metrics CSV, not a catalog source name. ``defaults.test`` (when given) must be a
    single source string: ``time: latest``, ``month: run``, and (with
    ``timestamped_dir``) the report directory's time range are all read off *one*
    source's time axis, and several sources have no single "latest" to mean.

    ``timestamped_dir`` picks where a run's report goes: the default ``False`` refreshes
    one stable ``<output_dir>/<name>/`` in place each run (clearing only the previous
    run's own artifacts); ``True`` writes a new
    ``<name>_<test>_<t0>_to_<t1>_<time>_<id>`` directory per run, which is also the safe
    choice if two runs could overlap.

    ``cover`` (default ``True``) puts a cover page describing the simulation first in
    ``report.pdf``; ``run_info`` (:class:`RunInfoConfig`) overrides what it is built
    from. See "Cover page" in ``docs/suites.md``.

    ``catalog_search_paths`` registers extra shared catalog directories -- absolute, or
    relative to this suite file (not the working directory) -- in the same tier as
    ``$OCEAN_SKILL_CATALOGS``, so a suite handed to a collaborator or run from cron (which
    does not source a shell rc) still finds its observational catalogs without relying on
    environment setup. See ``docs/suites.md``.

    ``cache_dir``, resolved the same way (absolute, or relative to this suite file), is
    the same fix applied to the cache: it calls :func:`ocean_skill.cache.enable` for the
    whole process before the suite touches any source, so a suite run from cron -- which
    would otherwise fall back to ``$OCEAN_SKILL_DIR`` or a per-user platformdirs cache --
    keeps reading from and adding to one particular directory across repeated runs. See
    "Caching" in ``docs/suites.md``.

    ``pdf_images`` picks how ``report.pdf`` stores the rasterized map images that are
    ~95% of its bytes: ``"lossless"`` (the default, matplotlib's own output, untouched)
    or ``"jpeg"`` (re-encoded afterwards with Ghostscript -- see
    :func:`ocean_skill.workflows.report.compress_pdf_images`; roughly a third smaller
    with no visible change, and a warning plus the lossless PDF if ``gs`` is not
    available). PNGs are never touched, and it does nothing under ``pdf: false``.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    output_dir: str | None = None
    timestamped_dir: bool = False
    pdf: bool = True
    cover: bool = True
    run_info: RunInfoConfig | None = None
    pdf_images: Literal["lossless", "jpeg"] = "lossless"
    cache: bool = True
    cache_dir: str | None = None
    refresh: RefreshConfig | None = None
    catalog_search_paths: list[str] = Field(default_factory=list)
    defaults: dict[str, Any] = Field(default_factory=dict)
    pages: list[PageConfig]

    @field_validator("pages")
    @classmethod
    def _pages_not_empty(cls, v: list[PageConfig]) -> list[PageConfig]:
        if not v:
            raise ValueError("a suite needs at least one page under pages:")
        return v

    @field_validator("catalog_search_paths")
    @classmethod
    def _catalog_search_paths_not_blank(cls, v: list[str]) -> list[str]:
        for entry in v:
            if not entry or not entry.strip():
                raise ValueError(
                    f"catalog_search_paths: entries must be non-empty paths, "
                    f"got {entry!r}"
                )
        return v

    @field_validator("cache_dir")
    @classmethod
    def _cache_dir_not_blank(cls, v: str | None) -> str | None:
        if v is not None and not v.strip():
            raise ValueError(f"cache_dir: must be a non-empty path, got {v!r}")
        return v

    @model_validator(mode="after")
    def _single_test_source(self) -> SuiteConfig:
        test = self.defaults.get("test")
        if test is not None and not isinstance(test, str):
            raise ValueError(
                f"defaults.test must be a single source name, got {test!r} -- "
                "time: latest, month: run, and the report directory's time range "
                "are all read off one source's own time axis"
            )
        return self
