"""A suite's summary page draws every pooled comparison, not just the first.

The summary de-duplicates by each comparison's specification, which a pooled metric row
does not have -- so every row looked identical and all but one were dropped ("pooled:
dropped 42 duplicate comparison(s)"), leaving a Taylor diagram with a single point.
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")

import ocean_skill as osk
from ocean_skill.comparison import _flatten
from ocean_skill.workflows.pages import MetricRecord


def _row(variable, depth, corr, bias):
    return {
        "variable": variable,
        "test": "model",
        "reference": "obs",
        "depth": depth,
        "corr": corr,
        "bias": bias,
        "sigma_ratio": 1.0,
        "std_test": 1.0,
        "std_reference": 1.0,
        "crmsd": 0.2,
        "rmse": 0.3,
        "n": 10,
    }


def _records(rows):
    return [MetricRecord(r, label=r["variable"]) for r in rows]


def test_distinct_rows_all_survive_pooling(capsys):
    rows = [
        _row("sea_water_potential_temperature", "surface", 0.99, -0.2),
        _row("sea_water_potential_temperature", "200", 0.98, 0.5),
        _row("sea_water_practical_salinity", "surface", 0.97, 0.0),
        _row("sea_surface_height_above_geoid", "surface", 0.92, -0.4),
    ]
    assert len(_flatten(_records(rows))) == 4
    assert "duplicate" not in capsys.readouterr().out


def test_two_statistics_of_one_pair_are_two_points():
    """A mean and a seasonal std of the same pair differ only in their metrics."""
    rows = [
        _row("salinity", "surface", 0.98, 0.0),
        _row("salinity", "surface", 0.76, 0.01),
    ]
    assert len(_flatten(_records(rows))) == 2


def test_an_exact_repeat_is_still_dropped():
    row = _row("salinity", "surface", 0.98, 0.0)
    assert len(_flatten(_records([row, dict(row)]))) == 1


def test_the_summary_figure_draws_every_record():
    rows = [
        _row("sea_water_potential_temperature", "surface", 0.99, -0.2),
        _row("sea_water_practical_salinity", "surface", 0.97, 0.0),
        _row("sea_surface_height_above_geoid", "surface", 0.92, -0.4),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig = osk.summary(_records(rows), kind="target", color_by="variable")
    points = sum(len(c.get_offsets()) for a in fig.axes for c in a.collections)
    assert points >= 3
