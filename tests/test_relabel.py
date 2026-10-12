"""``ComparisonSet.relabel``: rename the depth/time labels records carry, once.

A pooled skill table reads ``0-5 m`` / ``MAMJJASON``; a figure wants ``Surface`` /
``Mar–Nov``. ``relabel`` renames them on the comparisons themselves (copies, so the
original set is untouched), so the new names are what ``metrics()``, ``sel()`` and the
summary diagrams' items all read. Built like ``test_pool.py``: aligned pairs are
injected as ``Comparison._aligned``, so nothing needs a catalog entry.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ocean_skill.comparison import Comparison, ComparisonSet

TEMPERATURE = "sea_water_potential_temperature"
TIMES = pd.date_range("2024-01-01", periods=6, freq="D")
SURFACE_BAND = {"min": 0, "max": 5}
DEEP_BAND = {"min": 20, "max": 65}


def _aligned(offset: float = 0.0) -> xr.Dataset:
    ref = xr.DataArray(
        np.array([11.0, 12.0, 13.0, 14.0]) + offset,
        dims=("time",),
        coords={"time": TIMES[:4]},
    )
    tst = ref + np.array([0.5, -0.5, 1.0, 0.0])
    ref.attrs["units"] = tst.attrs["units"] = "degC"
    return xr.Dataset(
        {
            "test": tst.rename("test"),
            "reference": ref.rename("reference"),
            "difference": (tst - ref).rename("difference"),
        }
    )


def _comparison(depth, season, *, reference="A", offset=0.0) -> Comparison:
    c = Comparison(
        reference=reference,
        test="his",
        variable=TEMPERATURE,
        select={"depth": depth, "time": {"season": season}},
        cache=False,
    )
    c._aligned = _aligned(offset)
    return c


def _fan() -> ComparisonSet:
    """Two depth bands x two time windows, in one station-less series each."""
    return ComparisonSet(
        [
            _comparison(SURFACE_BAND, "JJA", offset=0.0),
            _comparison(SURFACE_BAND, "MAMJJASON", offset=1.0),
            _comparison(DEEP_BAND, "JJA", offset=2.0),
            _comparison(DEEP_BAND, "MAMJJASON", offset=3.0),
        ]
    )


def _pooled() -> ComparisonSet:
    return _fan().pool(by=["variable", "depth", "time"])


pytestmark = pytest.mark.filterwarnings("ignore:only .* (samples|valid cells)")

RENAMES = {
    "time": {"MAMJJASON": "Mar–Nov"},
    "depth": {"0-5 m": "Surface", "20-65 m": "Deep"},
}


# -- the new names are what records say ------------------------------------------------


def test_metrics_columns_carry_the_new_names():
    table = _pooled().relabel(**RENAMES).metrics()
    assert set(table["depth"]) == {"Surface", "Deep"}
    assert set(table["time"]) == {"JJA", "Mar–Nov"}  # JJA is not renamed


def test_the_pivot_columns_read_the_new_names():
    table = _pooled().relabel(**RENAMES).metrics()
    wide = table.pivot_table(index="depth", columns="time", values="n")
    assert list(wide.columns) == ["JJA", "Mar–Nov"]
    assert set(wide.index) == {"Deep", "Surface"}


def test_metrics_computed_before_the_rename_are_brought_up_to_date():
    pooled = _pooled()
    before = pooled.metrics()  # caches every record
    assert set(before["time"]) == {"JJA", "MAMJJASON"}
    after = pooled.relabel(**RENAMES).metrics()
    assert set(after["time"]) == {"JJA", "Mar–Nov"}
    # the numbers are the cached ones, not recomputed differently
    assert list(after["bias"]) == list(before["bias"])
    assert set(pooled.metrics()["time"]) == {"JJA", "MAMJJASON"}


def test_unpooled_members_are_renamed_too_and_others_keep_their_spelling():
    fan = _fan().relabel(depth={"0-5 m": "Surface"})
    depth = list(fan.metrics()["depth"])
    # the band that was renamed is its new name; the other stays what it was
    assert depth[:2] == ["Surface", "Surface"]
    assert depth[2:] == [DEEP_BAND, DEEP_BAND]


def test_summary_items_carry_the_new_name():
    items = _pooled().relabel(**RENAMES)._metric_items()
    assert {i["metrics"]["time"] for i in items} == {"JJA", "Mar–Nov"}
    assert {i["metrics"]["depth"] for i in items} == {"Surface", "Deep"}


# -- sel ------------------------------------------------------------------------------


def test_sel_matches_the_new_name_and_not_the_old_one():
    relabeled = _pooled().relabel(**RENAMES)
    assert len(relabeled.sel(depth="Surface")) == 2
    assert len(relabeled.sel(time=" mar–nov ", depth="deep")) == 1
    with pytest.raises(ValueError, match="have: Deep, Surface"):
        relabeled.sel(depth="0-5 m")
    with pytest.raises(ValueError, match="have: JJA, Mar–Nov"):
        relabeled.sel(time="MAMJJASON")


def test_relabel_survives_sel_and_add():
    relabeled = _pooled().relabel(**RENAMES)
    surface = relabeled.sel(depth="Surface")
    assert set(surface.metrics()["time"]) == {"JJA", "Mar–Nov"}
    both = surface + relabeled.sel(depth="Deep")
    assert set(both.metrics()["depth"]) == {"Surface", "Deep"}


# -- the original is left alone --------------------------------------------------------


def test_the_original_set_and_its_comparisons_are_untouched():
    pooled = _pooled()
    pooled.metrics()
    relabeled = pooled.relabel(**RENAMES)
    assert set(pooled.metrics()["depth"]) == {"0-5 m", "20-65 m"}
    assert set(pooled.metrics()["time"]) == {"JJA", "MAMJJASON"}
    assert len(pooled.sel(depth="0-5 m")) == 2
    assert all(not c._relabels for c in pooled)
    assert all(a is not b for a, b in zip(pooled, relabeled, strict=True))


def test_the_sets_label_overrides_are_kept():
    base = ComparisonSet(_pooled().comparisons, labels=["a", "b", "c", "d"])
    assert base.relabel(**RENAMES).labels == ["a", "b", "c", "d"]


# -- errors ---------------------------------------------------------------------------


def test_an_unknown_key_names_the_allowed_ones():
    with pytest.raises(ValueError, match=r"does not know 'variable'.*depth=.*time="):
        _pooled().relabel(variable={"temp": "T"})


def test_a_value_must_be_a_dict_of_strings():
    with pytest.raises(ValueError, match="dict of strings"):
        _pooled().relabel(depth="Surface")
    with pytest.raises(ValueError, match="dict of strings"):
        _pooled().relabel(depth={"0-5 m": 3})


def test_nothing_to_rename_is_an_error():
    with pytest.raises(ValueError, match="at least one of"):
        _pooled().relabel()


def test_a_level_the_set_does_not_have_lists_the_ones_it_has():
    with pytest.raises(
        ValueError, match=r"relabel\(depth='0-9 m'\).*have: 0-5 m, 20-65 m"
    ):
        _pooled().relabel(depth={"0-9 m": "Nope"})
    with pytest.raises(
        ValueError, match=r"relabel\(time='DJF'\).*have: JJA, MAMJJASON"
    ):
        _pooled().relabel(time={"DJF": "Winter"})


def test_matching_ignores_case_and_spaces():
    table = _pooled().relabel(depth={"0-5M": "Surface"}).metrics()
    assert "Surface" in set(table["depth"])


# -- composition ----------------------------------------------------------------------


def test_relabeling_twice_composes_later_wins():
    once = _pooled().relabel(**RENAMES)
    twice = once.relabel(depth={"Surface": "Top"}, time={"Mar–Nov": "Growing season"})
    table = twice.metrics()
    assert set(table["depth"]) == {"Top", "Deep"}
    assert set(table["time"]) == {"JJA", "Growing season"}
    # the first rename still stands for what the second left alone
    assert len(twice.sel(depth="Deep")) == 2
    # and the old names are gone, in the first step's set too
    with pytest.raises(ValueError, match="have: Deep, Top"):
        twice.sel(depth="Surface")
    assert set(once.metrics()["depth"]) == {"Surface", "Deep"}


def test_a_renamed_value_is_addressed_by_its_new_name_only():
    once = _pooled().relabel(depth={"0-5 m": "Surface"})
    with pytest.raises(ValueError, match="have: 20-65 m, Surface"):
        once.relabel(depth={"0-5 m": "Top"})


# -- relabel, then pool or average ------------------------------------------------


def test_relabel_then_pool_names_groups_by_the_new_names():
    pooled = _fan().relabel(**RENAMES).pool(by=["variable", "depth", "time"])
    table = pooled.metrics()
    assert set(table["depth"]) == {"Surface", "Deep"}
    assert set(table["time"]) == {"JJA", "Mar–Nov"}
    assert len(pooled.sel(depth="Surface", time="Mar–Nov")) == 1
    assert any("Surface" in c.label and "Mar–Nov" in c.label for c in pooled)


def test_two_labels_renamed_alike_pool_together():
    fan = _fan().relabel(time={"JJA": "Season", "MAMJJASON": "Season"})
    pooled = fan.pool(by=["variable", "depth", "time"])
    assert len(pooled) == 2
    assert set(pooled.metrics()["time"]) == {"Season"}
    # 4 pairs per member, two members per pile
    assert set(pooled.metrics()["n"]) == {8}


def test_relabel_then_average_carries_the_names():
    fan = _fan().sel(time="JJA").relabel(depth={"0-5 m": "Surface"})
    averaged = fan.average(by=["variable", "depth", "time"])
    # the renamed band is its new name; the other keeps the dict it always had
    assert list(averaged.metrics()["depth"]) == ["Surface", DEEP_BAND]
    assert len(averaged.sel(depth="Surface")) == 1
