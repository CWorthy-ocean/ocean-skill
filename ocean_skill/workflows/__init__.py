"""Declarative comparison suites (YAML) and their runner.

A suite (``config.SuiteConfig``) lists comparisons + shared defaults + a plot spec + an
output project. The same suite is invoked manually, on a schedule, or by a during-run
hook. See :mod:`ocean_skill.workflows.run`.
"""

from __future__ import annotations

__all__ = ["run_suite"]

from ocean_skill.workflows.run import run_suite
