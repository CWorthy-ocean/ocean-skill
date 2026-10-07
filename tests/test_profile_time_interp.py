"""``time_method="interp"`` reaches a ``profile`` reference, not only a time series.

A CTD cast is one instant, and the model lane is matched to it by the snapshot nearest
the cast (:func:`ocean_skill.align._sample_test_at_instant`). ``time_method="interp"``
used to be silently ignored there: the discrete cast-time targets that interpolate a
whole lane onto a reference's own instants (:func:`ocean_skill.align.
subset_to_time_targets`, whole dataset, so the free surface and the depth frame move
with it) were only ever offered to a repeat-visit or fixed-position reference, never
to a ``profile``. Now they are, and the default (``"auto"``/``"nearest"``) is untouched.

The fixture (``tests/_tidal_roms.py``) has hourly steps from 2024-07-01 with a free
surface ``zeta = [3, 0, -3, 0]`` m; ``height`` is exactly ``z_rho``, so a linear match
of it returns the *target's* own height -- 3 m below the surface is ``z = zeta - 3``,
5 m is ``zeta - 5`` -- and ``zeta`` itself interpolates linearly in time.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import ocean_skill as osk
from ocean_skill import catalog, comparison
from ocean_skill.align import NoValidData
from ocean_skill.comparison import Comparison, compare
from tests._tidal_roms import tidal_roms

POINT = (200.01, 50.01)  # the middle cell of the fixture's 3 x 3 grid
HEIGHT_SPEC = {
    "test": "height",
    "reference": "obs",
    "standard_name": "sea_water_temperature",
}


def _install(monkeypatch, sources, reads=None):
    """Stub the catalog and the reader for ``{name: (data, metadata)}``.

    ``reads``, when given, collects the name of every source read.
    """

    def read(name, **kw):
        if reads is not None:
            reads.append(name)
        return sources[name][0]

    monkeypatch.setattr(osk, "read", read)
    monkeypatch.setattr("ocean_skill.sources.read", read)
    monkeypatch.setattr(
        catalog, "resolve", lambda name: SimpleNamespace(metadata=sources[name][1])
    )
    monkeypatch.setattr(comparison, "_domain_of", lambda name: None)
    monkeypatch.setattr(comparison, "_outline_of", lambda name, convention=None: None)


def _cast(time, depths=(3.0, 5.0)):
    """Return a CTD cast at the fixture's middle cell, at ``time``."""
    return pd.DataFrame(
        {
            "time": pd.Timestamp(time),
            "lon": POINT[0],
            "lat": POINT[1],
            "depth (m)": list(depths),
            "obs (m)": [0.0] * len(depths),
        }
    )


def _comparison(monkeypatch, time, reads=None, test_meta=None, **kw):
    ds, meta = tidal_roms()
    # nanosecond stamps, like every obs frame (see test_depth_convention_compare)
    ds = ds.assign_coords(time=ds["time"].values.astype("datetime64[ns]"))
    _install(
        monkeypatch,
        {
            "his": (ds, {**meta, **(test_meta or {})}),
            "cast": (_cast(time), {"featureType": "profile"}),
        },
        reads,
    )
    return Comparison(
        reference="cast",
        test="his",
        variable=HEIGHT_SPEC,
        select={"depth": [3.0, 5.0]},
        over="Z",
        depth_method="interp",
        cache=False,
        **kw,
    )


def _test_values(c):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(c.aligned["test"]).reshape(-1)


def test_interp_moves_the_model_onto_the_casts_instant(monkeypatch):
    c = _comparison(monkeypatch, "2024-07-01 00:30", time_method="interp")
    # zeta is 1.5 half way between +3 and 0, so 3 m / 5 m below it: -1.5 and -3.5 --
    # the mean of the 00:00 answers (0, -2) and the 01:00 ones (-3, -5)
    np.testing.assert_allclose(_test_values(c), [-1.5, -3.5])


def test_interp_weights_follow_the_distance_to_each_step(monkeypatch):
    c = _comparison(monkeypatch, "2024-07-01 00:20", time_method="interp")
    # a third of the way from +3 to 0: zeta = 2
    np.testing.assert_allclose(_test_values(c), [-1.0, -3.0])


def test_linear_is_a_spelling_of_interp(monkeypatch):
    c = _comparison(monkeypatch, "2024-07-01 00:30", time_method="linear")
    np.testing.assert_allclose(_test_values(c), [-1.5, -3.5])


@pytest.mark.parametrize("time_method", ["auto", "nearest"])
def test_the_default_still_takes_the_nearest_step(monkeypatch, time_method):
    c = _comparison(monkeypatch, "2024-07-01 00:20", time_method=time_method)
    # 00:00 is the nearest step: zeta = +3, so 3 m and 5 m below it are 0 and -2
    np.testing.assert_allclose(_test_values(c), [0.0, -2.0])
    assert c._reference_time_targets() is None


def test_the_cast_time_is_the_target_only_for_interp(monkeypatch):
    c = _comparison(monkeypatch, "2024-07-01 00:30", time_method="interp")
    targets = c._reference_time_targets()
    assert [str(t)[:16] for t in np.asarray(targets, dtype="datetime64[m]")] == [
        "2024-07-01T00:30"
    ]


def test_interp_and_nearest_do_not_share_a_cache_entry(monkeypatch):
    interp = _comparison(monkeypatch, "2024-07-01 00:30", time_method="interp")
    nearest = _comparison(monkeypatch, "2024-07-01 00:30")
    assert interp._cache_key != nearest._cache_key


def test_a_cast_after_the_models_last_step_is_not_snapped_to_it(monkeypatch):
    c = _comparison(monkeypatch, "2024-07-01 05:00", time_method="interp")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(NoValidData, match="interpolat"):
            c.align()


def test_a_cast_before_the_models_first_step_is_not_snapped_to_it(monkeypatch):
    c = _comparison(monkeypatch, "2024-06-30 20:00", time_method="interp")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(NoValidData, match="interpolat"):
            c.align()


def test_a_cast_outside_the_declared_record_is_refused_before_the_model_is_read(
    monkeypatch,
):
    reads: list[str] = []
    record = {
        "time_coverage_start": "2024-07-01",
        "time_coverage_end": "2024-07-01",
    }
    c = _comparison(
        monkeypatch,
        "2024-07-20 00:30",
        reads=reads,
        test_meta=record,
        time_method="interp",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(NoValidData, match="interpolat"):
            c.align()
    assert "his" not in reads


def test_a_cast_inside_the_declared_record_still_interpolates(monkeypatch):
    record = {
        "time_coverage_start": "2024-07-01",
        "time_coverage_end": "2024-07-01",
    }
    c = _comparison(
        monkeypatch, "2024-07-01 00:30", test_meta=record, time_method="interp"
    )
    np.testing.assert_allclose(_test_values(c), [-1.5, -3.5])


def test_the_default_still_snaps_a_cast_past_the_record_to_the_last_step(monkeypatch):
    """Unchanged behaviour: nearest has no span to refuse a cast on."""
    c = _comparison(monkeypatch, "2024-07-01 05:00")
    # the last step (03:00) has zeta = 0
    np.testing.assert_allclose(_test_values(c), [-3.0, -5.0])


def test_compare_skips_a_cast_outside_the_record_and_keeps_the_others(
    monkeypatch, capsys
):
    ds, meta = tidal_roms()
    ds = ds.assign_coords(time=ds["time"].values.astype("datetime64[ns]"))
    _install(
        monkeypatch,
        {
            "his": (ds, meta),
            "inside": (_cast("2024-07-01 00:30"), {"featureType": "profile"}),
            "outside": (_cast("2024-07-01 05:00"), {"featureType": "profile"}),
        },
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = compare(
            reference=["inside", "outside"],
            test="his",
            variables=[HEIGHT_SPEC],
            depth_method="interp",
            time_method="interp",
            cache=False,
        )
    assert [c.reference_name for c in out] == ["inside"]
    printed = capsys.readouterr().out
    assert "skipped" in printed
    # a cast outside the record is the expected kind of skip: no error summary
    assert not [
        w for w in caught if "failed with an error other than" in str(w.message)
    ]
    np.testing.assert_allclose(
        np.asarray(out[0].aligned["test"]).reshape(-1), [-1.5, -3.5]
    )
