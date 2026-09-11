"""Graft low-level docstrings onto the user-facing wrappers that forward to them.

Several methods on :class:`~ocean_skill.comparison.Comparison`,
:class:`~ocean_skill.comparison.ComparisonSet`, :class:`~ocean_skill.field.Field`, and
:class:`~ocean_skill.field.FieldSet` take a closed set of named parameters plus
``**kwargs``, and forward that ``**kwargs`` untouched to a lower-level plotting
function. Their own docstrings can only name a handful of the forwarded keywords by
hand without drifting out of sync with the function that actually defines them.

This module appends the real keyword documentation onto those wrappers' ``__doc__`` at
class-definition time, so ``help(fn)``/``fn?`` shows the full surface without any of it
being retyped. It does this by parsing the target's source file with :mod:`ast` rather
than importing it: the dynamic plotters (``.plot()``/``.movie()``) resolve their actual
family -- ``field_row``, ``skill_map``, ``series``, ... -- from the data at call time,
and those families live in :mod:`ocean_skill.plot.matplotlib_renderer`, a module heavy
with cartopy/matplotlib imports that ``import ocean_skill`` deliberately does not pull
in. Reading source text imports nothing, so it can't reintroduce that weight or create
an import-order cycle with the modules doing the decorating.

Only ``__doc__`` is touched -- no signature, behavior, or runtime path changes.
"""

from __future__ import annotations

import ast
import functools
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

_PKG = Path(__file__).resolve().parent

_F = TypeVar("_F", bound=Callable)


@functools.cache
def _parsed(rel_path: str) -> ast.Module:
    """Parse ``ocean_skill/<rel_path>`` once, cached by path."""
    path = _PKG / rel_path
    return ast.parse(path.read_text(), filename=str(path))


def _extract_docstring(rel_path: str, func_name: str) -> str:
    """Return ``func_name``'s docstring from ``ocean_skill/<rel_path>``.

    Walks the module's top-level and class bodies for a function or method definition
    named ``func_name`` and returns its docstring. Raises ``LookupError`` if none is
    found, so a future rename of the target breaks loudly at import time rather than
    silently shipping a wrapper with no forwarded documentation.
    """
    tree = _parsed(rel_path)
    func_types = (ast.FunctionDef, ast.AsyncFunctionDef)
    for node in ast.walk(tree):
        if isinstance(node, func_types) and node.name == func_name:
            doc = ast.get_docstring(node, clean=True)
            if doc:
                return doc
            raise LookupError(f"{func_name!r} in {rel_path} has no docstring to graft")
    raise LookupError(f"no function named {func_name!r} found in {rel_path}")


def _append(fn: _F, block: str) -> _F:
    """Append ``block`` to ``fn.__doc__``, after its own docstring."""
    base = (fn.__doc__ or "").rstrip()
    fn.__doc__ = f"{base}\n\n{block}" if base else block
    return fn


def graft_from(rel_path: str, func_name: str, *, label: str) -> Callable[[_F], _F]:
    """Append ``func_name``'s docstring (from ``ocean_skill/<rel_path>``) as a section.

    Parameters
    ----------
    rel_path
        Path to the source file, relative to the ``ocean_skill`` package directory,
        e.g. ``"plot/summary.py"``.
    func_name
        Name of the module-level function or method whose docstring is the single
        source of truth for the forwarded keywords.
    label
        Fully-qualified dotted name to reference in the graft's header, e.g.
        ``"ocean_skill.plot.summary.taylor"``.
    """

    def decorator(fn: _F) -> _F:
        doc = _extract_docstring(rel_path, func_name)
        header = (
            "Forwarded keyword arguments\n"
            "---------------------------\n"
            f"Forwarded to :func:`{label}`, whose docstring is the single source of\n"
            "truth for these:\n"
        )
        return _append(fn, f"{header}\n{doc}")

    return decorator


#: Shared keyword documentation for the *dynamic* plotting entry points --
#: ``Comparison``/``ComparisonSet``/``Field``/``FieldSet``'s ``.plot()``/``.movie()``.
#: These resolve their actual plot family (``field_row``, ``field_grid``, ``series``,
#: ``profile``, ``section_row``, ``time_depth``, ``skill_map``, ...) from the data at
#: call time, so unlike the single-target wrappers above there is no one function whose
#: docstring is authoritative -- this is written once by hand and shared, rather than
#: grafted, and is the one place to update if the shared option families change.
_PLOT_OPTIONS_DOC = """\
Forwarded keyword arguments
---------------------------
Forwarded to whichever renderer family the data selects (``field_row``,
``field_grid``, ``field_facet``, ``series``, ``profile``, ``section_row``,
``time_depth``, or ``skill_map`` -- see :attr:`family`), so the exact set
accepted varies with the data rather than with this method. Option families
shared across most of them: ``color_by``/``marker_by`` (grouping), ``labels``,
``title``, ``domain`` (map extent), ``robust`` (colour-limit clipping),
``figsize``/``size``/``zoom``/``font_scale`` (sizing), ``save``, ``ncols``/
``nrows`` (grid layout), ``shared_limits``/``shared_axes``, and the
``*_kwargs`` styling dicts (``title_kwargs``, ``colorbar_kwargs``,
``gridline_kwargs``, ``legend_kwargs``, ...), each merging onto a built-in
default and unpacked straight into one matplotlib/cartopy call. See
``docs/plot_styling_reference.md`` for the full, per-family list with
defaults and examples.
"""


def graft_plot_options() -> Callable[[_F], _F]:
    """Append the shared :data:`_PLOT_OPTIONS_DOC` block to a dynamic plot method."""

    def decorator(fn: _F) -> _F:
        return _append(fn, _PLOT_OPTIONS_DOC)

    return decorator
