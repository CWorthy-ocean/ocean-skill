"""A profile reference with a single depth still compares (one pair, not an error).

A ``profile`` reference whose cast reports exactly one level -- or whose caller asked
for ``depths=[5]`` -- stands that one-element list as its depth axis. ROMS's vertical
transform returns it as a length-one ``z`` dimension, and ``_prepare`` used to squeeze
*any* length-one ``z`` away, so the test lane lost the very axis ``over="Z"`` scores
along and ``match_axis`` raised "the test lane has no 'Z' axis to score over" -- which
``compare()`` did not catch, aborting the whole call. The axis is now squeezed only
when the request was a scalar depth; a list keeps it, however short.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import ocean_skill as osk
from ocean_skill import catalog, comparison, depth_convention
from ocean_skill.comparison import _prepare, compare
from tests._tidal_roms import tidal_roms

POINT = (200.01, 50.01)  # the middle cell of the fixture's 3 x 3 grid
SURFACE = {
    "origin": "surface",
    "datum_z_m": 0.0,
    "support": "point",
    "source": "declared",
}
HEIGHT_SPEC = {
    "test": "height",
    "reference": "obs",
    "standard_name": "sea_water_temperature",
}


def _install(monkeypatch, sources):
    """Stub the catalog and the reader for ``{name: (data, metadata)}``."""
    monkeypatch.setattr(osk, "read", lambda name, **kw: sources[name][0])
    monkeypatch.setattr("ocean_skill.sources.read", lambda name, **kw: sources[name][0])
    monkeypatch.setattr(
        catalog, "resolve", lambda name: SimpleNamespace(metadata=sources[name][1])
    )
    monkeypatch.setattr(comparison, "_domain_of", lambda name: None)
    monkeypatch.setattr(comparison, "_outline_of", lambda name, convention=None: None)


def _cast(depths, time="2024-07-01"):
    """Return a CTD cast at the fixture's middle cell, one row per depth."""
    return pd.DataFrame(
        {
            "time": pd.Timestamp(time),
            "lon": POINT[0],
            "lat": POINT[1],
            "depth (m)": list(depths),
            "obs (m)": [0.0] * len(depths),
        }
    )


def _model():
    ds, meta = tidal_roms()
    # nanosecond stamps, like every obs frame (see test_depth_convention_compare)
    return ds.assign_coords(time=ds["time"].values.astype("datetime64[ns]")), meta


def _compare(monkeypatch, casts, depth_method="interp", cache=False, **kw):
    """Run ``compare`` over ``{name: cast frame}`` against the tidal fixture."""
    ds, meta = _model()
    sources = {"his": (ds, meta)}
    sources.update({n: (c, {"featureType": "profile"}) for n, c in casts.items()})
    _install(monkeypatch, sources)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return compare(
            reference=list(casts),
            test="his",
            variables=[HEIGHT_SPEC],
            depth_method=depth_method,
            cache=cache,
            **kw,
        )


@pytest.mark.parametrize(
    ("depth_method", "expected"),
    # 3 m below the surface at the first step (zeta = +3) is z = 0: interpolated it is
    # exactly that, read off the native cells (2.3 m thick, centres at 1.85, -0.45)
    # the nearest one wins
    [("interp", 0.0), ("nearest", -0.45)],
)
def test_a_one_depth_profile_forms_a_comparison_with_one_pair(
    monkeypatch, depth_method, expected
):
    out = _compare(monkeypatch, {"cast": _cast([3.0])}, depth_method=depth_method)
    assert len(out) == 1
    aligned = out[0].aligned
    assert aligned["test"].size == 1
    assert aligned["reference"].size == 1
    np.testing.assert_allclose(np.asarray(aligned["test"]).reshape(-1), [expected])
    # scored, not refused: the one pair counts (the metrics warn that it is only one)
    with pytest.warns(UserWarning, match="only 1 depth levels"):
        assert out[0].metrics()["n"] == 1


def test_a_one_depth_profile_survives_the_cache_round_trip(monkeypatch):
    fresh = _compare(monkeypatch, {"cast": _cast([3.0])}, cache=True)
    cached = _compare(monkeypatch, {"cast": _cast([3.0])}, cache=True)
    assert len(fresh) == len(cached) == 1
    np.testing.assert_array_equal(
        np.asarray(fresh[0].aligned["test"]), np.asarray(cached[0].aligned["test"])
    )
    assert cached[0].aligned["test"].size == 1


def test_a_depths_list_of_one_keeps_the_axis_for_a_deeper_cast(monkeypatch):
    out = _compare(monkeypatch, {"cast": _cast([1.0, 3.0, 5.0])}, depths=[5.0])
    assert len(out) == 1
    aligned = out[0].aligned
    assert aligned["test"].size == 1
    np.testing.assert_allclose(np.asarray(aligned["test"]).reshape(-1), [-2.0])


def test_a_two_depth_and_a_one_depth_cast_both_compare(monkeypatch):
    out = _compare(
        monkeypatch,
        {"deep": _cast([3.0, 5.0]), "shallow": _cast([3.0])},
    )
    assert len(out) == 2
    sizes = sorted(int(c.aligned["test"].size) for c in out)
    assert sizes == [1, 2]


def test_a_scalar_depth_is_still_squeezed_but_a_list_of_one_is_not():
    ds, meta = tidal_roms()

    def lane(depth):
        return _prepare(
            ds,
            meta,
            "height",
            {"depth": depth},
            None,
            depth_method="interp",
            depth_convention=SURFACE,
            obs_convention=depth_convention.resolve({}),
        )[0]

    assert "z" not in lane(3.0).dims
    kept = lane([3.0])
    assert kept.sizes["z"] == 1
    np.testing.assert_allclose(
        np.asarray(kept.isel(eta_rho=1, xi_rho=1)).reshape(-1), [0.0, -3.0, -6.0, -3.0]
    )


def test_a_one_level_list_lane_is_keyed_apart_from_a_squeezed_one(monkeypatch):
    """A lane cached before the axis was kept must not be served in its place."""
    from ocean_skill import cache
    from ocean_skill.comparison import prepare_source

    seen = []
    real = cache.key_for_prepared

    def spy(*, source, variable, select, **kw):
        seen.append(select)
        return real(source=source, variable=variable, select=select, **kw)

    monkeypatch.setattr(cache, "key_for_prepared", spy)
    ds, meta = _model()
    _install(monkeypatch, {"his": (ds, meta)})

    def keyed(request):
        seen.clear()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prepare_source(
                "his",
                "height",
                {"depth": request},
                None,
                use_cache=False,
                depth_method="interp",
                depth_convention=SURFACE,
            )
        return "_unit_vertical_axis" in seen[-1]

    assert keyed([3.0]) is True
    assert keyed((3.0,)) is True
    assert keyed(3.0) is False
    assert keyed([3.0, 5.0]) is False
    assert keyed({"min": 0.0, "max": 10.0}) is False
