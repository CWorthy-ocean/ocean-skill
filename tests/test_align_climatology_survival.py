"""``align.align``'s climatology-dim survival gate, for ``over="Z"``.

A groupby (``month``, renamed) or resample (``time``, kept) fold standing on
*both* lanes is the shape a profile comparison scored down depth is meant to
draw one row per bin from -- not the leftover, un-reduced axis
:func:`~ocean_skill.align._require_2d` otherwise refuses. This is the gate at
the top of :func:`~ocean_skill.align.align`, tested directly (no ``Comparison``,
no catalog) via hand-built ``(time-or-fold, depth, lat, lon)`` pairs run through
``operators.aggregate`` -- exactly the shape a real ``timeSeriesProfile``
comparison's two lanes take once ``aggregate={"time": {...}}`` has run on both.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ocean_skill import align as A
from ocean_skill import operators


def _pair(agg):
    """A (time-ish, depth) reference and a (time-ish, depth, lat, lon) test,
    both reduced by ``agg`` -- the shape a real profile comparison's two lanes
    take once a shared ``aggregate={"time": {...}}`` has run on each.
    """
    time = pd.date_range("2024-04-01", periods=13, freq="MS")
    depth = np.array([5.0, 25.0])
    reference = xr.DataArray(
        (24.0 - 0.05 * depth)[None, :] + 0.1 * np.arange(13)[:, None],
        dims=("time", "depth"),
        coords={"time": time, "depth": depth},
        name="TEMP",
        attrs={"units": "degC"},
    ).assign_coords(lon=-21.0, lat=64.0)
    lon, lat = np.array([-21.5, -20.5]), np.array([63.5, 64.5])
    test = xr.DataArray(
        (24.5 - 0.05 * depth)[None, :, None, None]
        + 0.1 * np.arange(13)[:, None, None, None]
        + np.zeros((1, 1, 2, 2)),
        dims=("time", "depth", "lat", "lon"),
        coords={"time": time, "depth": depth, "lat": lat, "lon": lon},
        name="TEMP",
        attrs={"units": "degC"},
    )
    return operators.aggregate(test, agg), operators.aggregate(reference, agg)


GROUPBY = {"time": {"groupby": "month", "reduce": "mean", "spread": "std"}}
RESAMPLE = {"time": {"resample": "1MS", "reduce": "mean", "spread": "std"}}


@pytest.mark.parametrize(
    ("agg", "dim", "size"), [(GROUPBY, "month", 12), (RESAMPLE, "time", 13)]
)
def test_a_matching_fold_on_both_lanes_survives(agg, dim, size):
    """Both groupby (renamed ``month``) and resample (kept ``time``) fold --
    marked identically on both lanes -- survive ``_require_2d`` and land in the
    aligned result with depth, mean, and spread intact.
    """
    test, reference = _pair(agg)
    out = A.align(test, reference, method="conservative_normed", over="Z")
    assert out.sizes[dim] == size
    assert "depth" in out.sizes or "z" in out.sizes
    assert {"test", "reference", "difference", "test_spread", "reference_spread"} <= set(
        out.data_vars
    )


def test_an_unreduced_time_axis_is_still_refused():
    """Neither lane's ``time`` carries a fold marker -- this is the ordinary
    un-reduced axis a caller forgot to collapse, unchanged by either marker.
    """
    test, reference = _pair(None)
    with pytest.raises(ValueError, match="beyond its horizontal axes"):
        A.align(test, reference, method="conservative_normed", over="Z")


@pytest.mark.parametrize("agg", [GROUPBY, RESAMPLE])
def test_only_one_lane_marked_is_still_refused(agg):
    """A fold on only one lane (the other left raw) must not survive -- the
    gate requires the *same* dim, marked, on *both* lanes.
    """
    test, reference = _pair(agg)
    raw_reference = _pair(None)[1]
    # raw_reference still carries the same (time, depth) shape as test's fold
    # partner would, just unreduced and unmarked -- exactly a caller who
    # aggregated one lane and forgot the other.
    if "month" in test.dims:
        # A raw 24-step time axis has no "month" dim to match test's -- the
        # dims themselves already disagree, so _extra_dims never even reaches
        # the marker check. Confirm it still raises, for a different reason
        # baked into the same safety property.
        with pytest.raises(ValueError):
            A.align(test, raw_reference, method="conservative_normed", over="Z")
    else:
        with pytest.raises(ValueError, match="beyond its horizontal axes"):
            A.align(test, raw_reference, method="conservative_normed", over="Z")
