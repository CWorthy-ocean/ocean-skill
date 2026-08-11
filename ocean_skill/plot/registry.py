"""Renderer registry: pick a backend by name and hand it a :class:`PlotSpec`.

Renderers register under a name (``"matplotlib"``, ``"holoviews"``) and implement
``render(spec, **kwargs)``. Choosing a renderer is the only change needed to move a plot
between static and interactive output — which is the point of routing every plot through
here rather than calling a backend function directly.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

__all__ = [
    "get_renderer",
    "register_renderer",
    "render",
    "renderers",
    "side_by_side",
]

_RENDERERS: dict[str, Callable] = {}

#: Modules to import on demand when a renderer is first requested by name, so importing
#: ocean_skill does not drag in matplotlib/holoviews until something is actually drawn.
_LAZY = {
    "matplotlib": "ocean_skill.plot.matplotlib_renderer",
    "holoviews": "ocean_skill.plot.holoviews_renderer",
}


def register_renderer(name: str, fn: Callable) -> None:
    """Register a renderer callable under ``name``."""
    _RENDERERS[name] = fn


def renderers() -> list[str]:
    """Names of all renderers that can be loaded."""
    return sorted(set(_RENDERERS) | set(_LAZY))


def get_renderer(name: str) -> Callable:
    """Return the renderer registered under ``name``, importing it if needed."""
    if name not in _RENDERERS and name in _LAZY:
        import importlib

        importlib.import_module(_LAZY[name])
    try:
        return _RENDERERS[name]
    except KeyError:
        raise KeyError(f"No renderer {name!r}; known: {renderers()}") from None


def side_by_side(spec, **kwargs: Any):
    """Return static and interactive renderings of one spec, in one panel Row.

    matplotlib and bokeh cannot share a figure, so panel hosts both: the matplotlib
    figure on the left as an image pane, the live holoviews object on the right. Useful
    for showing that a renderer swap changes nothing but the backend.
    """
    import panel as pn

    pn.extension()
    static = get_renderer("matplotlib")(spec, **kwargs)
    interactive = get_renderer("holoviews")(spec, **kwargs)
    return pn.Row(
        pn.Column(
            "### static (matplotlib)", pn.pane.Matplotlib(static, dpi=110, tight=True)
        ),
        pn.Column("### interactive (holoviews)", interactive),
    )


def render(spec, *, renderer: str = "matplotlib", **kwargs: Any):
    """Draw ``spec`` with the named renderer.

    ``renderer="both"`` returns the static and interactive versions side by side.
    """
    if renderer == "both":
        return side_by_side(spec, **kwargs)
    return get_renderer(renderer)(spec, **kwargs)
