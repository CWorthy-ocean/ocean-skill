"""Quality control for observations (QARTOD via ioos_qc). Phase 3.

Applied as a per-source operation on observational data (never a carried attribute),
driven by per-variable QARTOD config dictionaries referenced from the catalog entry's
``qc_config``.
"""

from __future__ import annotations

from typing import Any

__all__ = ["qc"]


def qc(obj, config: dict[str, Any]):
    """Run QARTOD tests (ioos_qc) on ``obj`` and attach/apply flags."""
    raise NotImplementedError
