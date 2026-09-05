"""``detide=`` through the real pipeline: ``compare()`` and ``field()``.

Not just ``ocean_skill.detide.detide`` called directly (see ``tests/test_detide.py``)
or the normalization/cache-key/identity plumbing checked in isolation (see
``tests/test_detide_plumbing.py``). Mocks only ``osk.read``/``catalog.resolve``
(mirroring ``tests/test_tsp_end_to_end.py``'s own pattern), so the real ``_prepare``/
``align`` pipeline runs against two hourly station DataFrames -- a synthetic tide
gauge and a synthetic model run, both carrying the same M2+K1 tidal signal on top of
a slower, shared trend -- and ``detide=True`` has to actually reach through
``Comparison._prepare_lane`` -> ``prepare_source`` -> ``_prepare`` and come out the
other side with the tidal band gone, before ``align`` ever sees the pair.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("oceans", reason="detide needs the 'oceans' package (pl33tn)")

#: The literal variable name tabular.to_dataset derives from the "zeta (m)" column
#: below (unit stripped) -- requested by this same literal spelling, rather than a
#: real CF standard_name, so resolution needs no vocabulary alias/rename step (which
#: osk.read ordinarily provides and this fixture's monkeypatch deliberately bypasses,
#: mirroring tests/test_tsp_end_to_end.py's own "Temperature (degC)"/read-bypass
#: pattern) -- see ocean_skill.units.find_variable's literal-name-first path.
ZETA = "zeta"
N = 800
EDGE = 40


def _signal(offset: float = 0.0, seed: int = 0):
    t_hours = np.arange(0, N, 1.0)
    rng = np.random.default_rng(seed)
    trend = 0.5 * np.sin(2 * np.pi * t_hours / (24 * 10))
    m2 = 1.0 * np.sin(2 * np.pi * t_hours / 12.42)
    k1 = 0.6 * np.sin(2 * np.pi * t_hours / 24.0)
    noise = rng.normal(0, 0.02, N)
    return trend + m2 + k1 + offset + noise


def _frame(offset: float, seed: int) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=N, freq="h")
    return pd.DataFrame(
        {"time": idx, "zeta (m)": _signal(offset, seed), "lon": -122.5, "lat": 45.0}
    )


@pytest.fixture
def tide_gauge_and_model(monkeypatch):
    import ocean_skill as osk
    from ocean_skill import catalog, comparison

    lanes = {"tide_gauge": _frame(0.0, seed=1), "his": _frame(0.3, seed=2)}
    metas = {"tide_gauge": {}, "his": {}}
    monkeypatch.setattr(osk, "read", lambda name, **kw: lanes[name])
    monkeypatch.setattr("ocean_skill.sources.read", lambda name, **kw: lanes[name])
    monkeypatch.setattr(
        catalog, "resolve", lambda name: SimpleNamespace(metadata=metas[name])
    )
    monkeypatch.setattr(comparison, "_domain_of", lambda name: None)
    monkeypatch.setattr(comparison, "_outline_of", lambda name, convention=None: None)
    return lanes


def _compared(**kwargs):
    """Return ``(ComparisonSet, the one Comparison in it, its aligned pair)``."""
    from ocean_skill.comparison import compare

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = compare(
            reference="tide_gauge", test="his", variables=[ZETA], over="time", **kwargs,
        )
    comparisons = list(result)
    assert len(comparisons) == 1
    c = comparisons[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        aligned = c.align()
    return result, c, aligned


def test_detide_true_reaches_the_pipeline_and_removes_the_tidal_band(
    tide_gauge_and_model,
):
    _, _, raw = _compared()
    _, _, detided = _compared(detide=True)

    interior = slice(EDGE, -EDGE)
    raw_std = float(raw["reference"].values[interior].std())
    detided_std = float(detided["reference"].values[interior].std())
    # the tidal band (M2 ~1.0 + K1 ~0.6 amplitude) dominates the raw std; once
    # removed, only the slow trend (~0.35 std) plus a little noise should remain
    assert detided_std < 0.6 * raw_std


def test_detide_metrics_carry_a_detided_column(tide_gauge_and_model):
    _, c_raw, _ = _compared()
    _, c_detided, _ = _compared(detide=True)
    assert c_raw.metrics()["detided"] == "raw"
    assert c_detided.metrics()["detided"] == "detided"


def test_raw_and_detided_pool_as_two_distinct_points(tide_gauge_and_model):
    raw_set, _, _ = _compared()
    detided_set, _, _ = _compared(detide=True)
    pooled = raw_set + detided_set
    assert len(list(pooled)) == 2


def test_field_detide_true_reaches_the_pipeline(tide_gauge_and_model):
    from ocean_skill.field import field

    f_raw = field("tide_gauge", ZETA)
    f_detided = field("tide_gauge", ZETA, detide=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw_data = f_raw.data
        detided_data = f_detided.data
    interior = slice(EDGE, -EDGE)
    assert float(detided_data.values[interior].std()) < 0.6 * float(
        raw_data.values[interior].std()
    )
