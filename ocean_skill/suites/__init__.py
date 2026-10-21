"""A suite (YAML) and its runner.

A suite (``ocean_skill.config.SuiteConfig``) is an ordered list of *pages* -- each
one a single ``osk.field``, ``osk.compare``, ``osk.summary``, ``osk.XY`` or ``osk.TS``
call -- plus shared defaults and output settings. The same suite is invoked manually, on
a schedule, or by a during-run hook. See :mod:`ocean_skill.suites.run` and
``docs/suites.md``.
"""

from __future__ import annotations

__all__ = ["run_suite"]

from ocean_skill.suites.run import run_suite
