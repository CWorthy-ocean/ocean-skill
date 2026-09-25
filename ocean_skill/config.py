r"""Pydantic models for a suite YAML: the declarative form of ``osk.field``/``compare``.

A suite is a *list of pages* -- each one a single :func:`ocean_skill.field.field`,
:func:`ocean_skill.comparison.compare`, or :func:`ocean_skill.comparison.summary` call,
expressed as YAML instead of Python -- plus shared defaults, output settings, and an
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
    """One page of a report: exactly one of ``field``, ``compare``, or ``summary``.

    ``field``/``compare``/``summary`` are the raw keyword dicts a page will pass to
    :func:`ocean_skill.field.field`, :func:`ocean_skill.comparison.compare`, or
    :func:`ocean_skill.comparison.summary` -- left as ``dict[str, Any]`` (not
    individually typed) because those functions already validate their own
    arguments; a page's job is only to say *which one* and *with what*, and to let
    ``for_each``/``{placeholder}`` reach into any of its values.

    ``for_each`` fans this one page into several -- one page per element of the
    Cartesian product of its lists (see :mod:`ocean_skill.workflows.pages`).
    """

    model_config = ConfigDict(extra="forbid")

    title: str
    field: dict[str, Any] | None = None
    compare: dict[str, Any] | None = None
    summary: dict[str, Any] | None = None
    for_each: dict[str, Any] | None = None
    plot: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _exactly_one_kind(self) -> PageConfig:
        kinds = [
            k for k in ("field", "compare", "summary") if getattr(self, k) is not None
        ]
        if len(kinds) != 1:
            raise ValueError(
                f"page {self.title!r} must have exactly one of field:/compare:/"
                f"summary: -- found {kinds or 'none'}"
            )
        return self

    @property
    def kind(self) -> Literal["field", "compare", "summary"]:
        """Which of ``field``/``compare``/``summary`` this page is."""
        for k in ("field", "compare", "summary"):
            if getattr(self, k) is not None:
                return k  # type: ignore[return-value]
        raise AssertionError("_exactly_one_kind already enforces this")


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
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    output_dir: str | None = None
    pdf: bool = True
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
