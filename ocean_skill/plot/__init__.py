"""Plotting: backend-agnostic PlotSpecs + pluggable renderers.

A :mod:`~ocean_skill.plot.spec` describes *what* to draw (family, marks, composition);
a renderer (matplotlib now, holoviews later), looked up in
:mod:`~ocean_skill.plot.registry`, decides *how*. The same spec renders static or
interactive by renderer choice.
"""

from __future__ import annotations

__all__ = ["get_renderer", "render"]

from ocean_skill.plot.registry import get_renderer, render
