"""Point warnings at the caller's own code rather than ocean-skill's internals.

``warnings.warn(..., stacklevel=N)`` decides which file:line Python blames for a
warning. A fixed ``stacklevel=2`` blames whoever called the warning's own function
— which for anything raised deep in the pipeline is another ocean-skill module, not
the user. A variable-resolution notice from
:func:`ocean_skill.units.find_variable`, for instance, reached the user as::

    ocean_skill/comparison.py:105: UserWarning: 'nitrate' resolved to ...

naming an internal line that tells them nothing about which of *their* calls
triggered it — and every variable in a ``compare()`` fan-out cites the same one.

The call depth is not fixed either (``compare`` -> ``Comparison.align`` ->
``_prepare`` -> ``find_variable`` is four frames; calling ``find_variable``
directly is one), so no hard-coded number is right for both. :func:`find` counts
the frames instead. Same approach pandas and xarray take, for the same reason.
"""

from __future__ import annotations

import inspect
from pathlib import Path

__all__ = ["find"]

#: The package directory whose frames are "internal" and should be skipped.
_PACKAGE_ROOT = str(Path(__file__).parent)


def find() -> int:
    """Return the ``stacklevel`` naming the first frame outside ocean-skill.

    Falls back to ``2`` (the immediate caller) if every frame is internal — which
    happens when ocean-skill's own tests or scripts are the outermost caller, and
    where blaming the immediate caller is the best available answer anyway.
    """
    frame = inspect.currentframe()
    try:
        # 0 = this function; 1 = the ocean-skill function that wants to warn. Start
        # the count there, so the first *external* frame found is what gets blamed.
        level = 1
        frame = frame.f_back if frame else None
        while frame is not None:
            if not frame.f_code.co_filename.startswith(_PACKAGE_ROOT):
                return level
            frame = frame.f_back
            level += 1
        return 2
    finally:
        del frame  # break the reference cycle CPython warns about for frame objects
