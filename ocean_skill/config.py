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

__all__ = ["PageConfig", "RefreshConfig", "RefreshSourceConfig", "SuiteConfig"]

#: The mutually exclusive page kinds, in the order their error messages list them.
_PAGE_KINDS = ("field", "compare", "summary", "section", "XY", "TS")


class RefreshSourceConfig(BaseModel):
    """One stream to rebuild before the suite runs -- forwarded to ``make_kerchunk``."""

    model_config = ConfigDict(extra="forbid")

    name: str
    files: str
    ref: str
    grid: str | None = None
    keep: str | None = None
    min_age: float | None = None


class RefreshConfig(BaseModel):
    """``refresh:`` block: rebuild a live run's kerchunk references first.

    Forwarded, unchanged in shape, to
    :func:`ocean_skill.workflows.run._refresh_sources`.
    """

    model_config = ConfigDict(extra="forbid")

    catalog: str
    sources: list[RefreshSourceConfig]


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


class SuiteConfig(BaseModel):
    """A suite YAML: the whole file. See ``docs/suites.md`` for the grammar.

    ``name`` is a label the user chooses -- it prefixes the report directory and the
    metrics CSV, not a catalog source name. ``defaults.test`` (when given) must be a
    single source string: ``time: latest``, ``month: run``, and the report
    directory's time range are all read off *one* source's time axis, and several
    sources have no single "latest" to mean.

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
    pdf: bool = True
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
