"""Where deliverables go: metrics tables and figures, organized by project.

The counterpart to :mod:`ocean_skill.cache`, and deliberately a *different* place.
Cached alignments are regenerable intermediates, hash-named, and safe for anything —
you, or the operating system reclaiming its cache directory — to delete at any time.
Metrics and figures are the output of the work: you name them, keep them, put them in
a paper. Writing those under a cache directory would mean the OS could quietly reclaim
a figure you meant to keep.

So they live in a visible, project-scoped tree, matching what ``examples/`` already
wrote by hand::

    output/<project>/
        figures/<stem>.png
        metrics/<stem>.csv        (+ .txt, a human-readable rendering)

The base is ``$OCEAN_SKILL_OUTPUT`` if set, otherwise ``./output`` relative to the
working directory — chosen over a platformdirs location because a figure you have to
go hunting for under ``~/Library/Application Support`` may as well not have been
written. :func:`set_base` overrides it in code.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from ocean_skill._display import Text

__all__ = [
    "base_dir",
    "figures_dir",
    "info",
    "metrics_dir",
    "project_dir",
    "set_base",
    "slug",
]

_override_base: Path | None = None
_announced: set[str] = set()


def base_dir() -> Path:
    """Return the output root.

    Resolved as :func:`set_base` -> ``$OCEAN_SKILL_OUTPUT`` -> ``./output``.
    """
    if _override_base is not None:
        return _override_base
    env = os.environ.get("OCEAN_SKILL_OUTPUT")
    return Path(env).expanduser() if env else Path("output")


def set_base(directory: str | Path | None) -> None:
    """Point outputs at ``directory``; ``None`` restores the default resolution."""
    global _override_base
    _override_base = None if directory is None else Path(directory).expanduser()


def slug(text: str) -> str:
    """Return ``text`` reduced to a safe path component.

    Source names carry characters a path should not: a qualified name like
    ``"GOM offline run:GOM_bgc"`` has both a colon and spaces.
    """
    return re.sub(r"[^0-9A-Za-z._-]+", "_", text).strip("_") or "unnamed"


def project_dir(project: str) -> Path:
    """Return ``<base>/<project>``, created if needed."""
    d = base_dir() / slug(project)
    d.mkdir(parents=True, exist_ok=True)
    _announce(d)
    return d


def figures_dir(project: str) -> Path:
    """Return ``<base>/<project>/figures``, created if needed."""
    d = project_dir(project) / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def metrics_dir(project: str) -> Path:
    """Return ``<base>/<project>/metrics``, created if needed."""
    d = project_dir(project) / "metrics"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _announce(directory: Path) -> None:
    """Say where a project's outputs are going — once per project per process."""
    key = str(directory)
    if key in _announced:
        return
    _announced.add(key)
    print(f"ocean-skill: writing outputs to {directory.resolve()}")


def info() -> Text:
    """Return a human-readable summary of the output tree."""
    root = base_dir()
    if not root.exists():
        return Text(f"ocean-skill outputs: nothing written yet ({root.resolve()})")
    projects = sorted(p.name for p in root.iterdir() if p.is_dir())
    if not projects:
        return Text(f"ocean-skill outputs: no projects ({root.resolve()})")
    return Text(
        f"ocean-skill outputs: {len(projects)} project"
        f"{'' if len(projects) == 1 else 's'} ({', '.join(projects)}) "
        f"in {root.resolve()}"
    )
