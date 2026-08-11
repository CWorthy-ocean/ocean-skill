"""Pydantic config models for comparisons and workflow suites.

These are the single spec the Python API builds and the YAML workflow deserializes into.
An entity is a ``(source, variable, selection)`` with a role; a comparison pairs a
reference and a test; a suite is a list of comparisons plus shared defaults, a plot
spec, and an output project. A ``provenance`` block (content hash + tool versions +
catalog entries used) is written alongside outputs.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

__all__ = ["ComparisonConfig", "EntityConfig", "PlotConfig", "SuiteConfig"]

Role = Literal["reference", "test"]


class EntityConfig(BaseModel):
    """One comparand: a source + variable + optional per-entity select/aggregate."""

    source: str
    variable: str | None = None
    role: Role | None = None
    select: dict[str, Any] = Field(default_factory=dict)
    aggregate: dict[str, Any] = Field(default_factory=dict)


class PlotConfig(BaseModel):
    """Plot spec: family (usually inferred), marks, and bounded composition knobs."""

    spec: str | None = None  # e.g. "field_row" | "line" | "map"; inferred if None
    mark: str | None = None
    rows: str | None = None
    cols: str | None = None
    overlay: str | None = None
    secondary_y: str | None = None
    renderer: Literal["matplotlib", "holoviews"] = "matplotlib"
    mode: Literal["single", "animate"] = "single"


class ComparisonConfig(BaseModel):
    """A single reference↔test comparison for one variable."""

    reference: str
    test: str
    variable: str
    select: dict[str, Any] = Field(default_factory=dict)
    aggregate: dict[str, Any] = Field(default_factory=dict)


class SuiteConfig(BaseModel):
    """A declarative comparison suite (the workflow YAML schema)."""

    name: str
    project: str
    defaults: dict[str, Any] = Field(default_factory=dict)
    comparisons: list[dict[str, Any]] = Field(default_factory=list)
    plot: PlotConfig = Field(default_factory=PlotConfig)
    outputs: dict[str, bool] = Field(
        default_factory=lambda: {"figures": True, "metrics": True}
    )
