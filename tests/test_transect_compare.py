"""A CTD transect compared against a model whose tide is known by hand.

``line`` is one CSV transect (``trajectoryProfile``; ``line_noid`` the same rows without
the station column, cut on the time gap; ``line_bare`` the same, declaring nothing) of
three casts, with the surface-referenced depths a CTD has. An entry is compared as one
section, a list of its cast names cast by cast. The model is the tidal ROMS fixture
(``tests/_tidal_roms``): hourly steps from 2024-07-01 00:00, ``zeta = [3, 0, -3,
0]`` m, ``height`` exactly ``z_rho``, and ``temp = 10 + hour + 0.5 * xi + 0.1 *
z_rho`` so a wrong time or a wrong cell changes every number::

    cast  time   lon (a sample every 10 s)           depths   zeta(time)
    S1    00:30  200.00                              1..5 m   +1.5
    S2    01:30  200.01                              3 m      -1.5
    S3    02:20  200.019 .. 200.021 (median 200.02)  1..5 m   -2.0   (-3.0 at 02:00)

Only the model ``his`` is stubbed; the obs are read from real catalogs written into the
isolated catalog directory. ``p1``..``p3`` are the casts as profile entries of their
own, what one registers by hand today.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from intake.readers import datatypes, readers

import ocean_skill as osk
from ocean_skill import build, casts, catalog, comparison, sources
from ocean_skill.align import ALONG_DIM
from ocean_skill.comparison import Comparison
from tests._tidal_roms import tidal_roms

HEIGHT = {
    "test": "height",
    "reference": "obs",
    "standard_name": "sea_water_temperature",
}
TEMP = {**HEIGHT, "test": "temp", "reference": "temp_obs"}  # in degC, as temp is
CASTS = {  # id: (start, a longitude per sample, depths in m)
    "S1": ("2024-07-01 00:30:00", [200.0] * 5, [1, 2, 3, 4, 5]),
    "S2": ("2024-07-01 01:30:00", [200.01], [3]),
    "S3": (
        "2024-07-01 02:20:00",
        [200.019, 200.0195, 200.02, 200.0205, 200.021],
        [1, 2, 3, 4, 5],
    ),
}
ZETA = {"S1": 1.5, "S2": -1.5, "S3": -2.0}  # the free surface at the cast's own time
HOURS = {"S1": 0.5, "S2": 1.5, "S3": 2 + 20 / 60}  # that time, in model steps
XI = {"S1": 0, "S2": 1, "S3": 2}  # the model column the cast's place falls in
LANES = ("test", "reference", "difference")


def _z(zeta, depths):
    """Return the model's ``height`` ``depths`` below a free surface at ``zeta``.

    ``zeta - d``, except that a target in the top half-cell (above the top cell's
    centre, ``0.95 * zeta - 1`` on this grid) takes that cell's own value -- which only
    S1's 1 m sample, at ``zeta = 1.5``, reaches.
    """
    return np.minimum(zeta - np.asarray(depths, float), 0.95 * zeta - 1.0)


def _rows():
    """Return the transect's samples: ``obs`` = 100 * the cast's number + its depth."""
    return pd.DataFrame(
        {
            "station": sid,
            "time": pd.Timestamp(t0) + pd.Timedelta(seconds=10 * i),
            "lon": lon,
            "lat": 50.01,
            "depth (m)": d,
            "obs (m)": 100.0 * k + d,
            "temp_obs (degC)": 15.0,
        }
        for k, (sid, (t0, lons, depths)) in enumerate(CASTS.items(), 1)
        for i, (lon, d) in enumerate(zip(lons, depths, strict=True))
    )


def _raw_rows(obs):
    """Return ``obs`` with every cast at depths of its own (+0.1 m, +0.2 m, +0.3 m)."""
    raw = obs.copy()
    raw["depth (m)"] += 0.1 * raw["station"].map({"S1": 1, "S2": 2, "S3": 3})
    return raw


def _nc(path):
    """Write the transect as a multidimensional CF trajectoryProfile (profile x z)."""
    obs = _rows()
    pad = lambda v: [*v, *[np.nan] * (5 - len(v))]  # noqa: E731
    xr.Dataset(
        {
            "obs": (
                ("profile", "z"),
                [pad(obs.loc[obs["station"] == s, "obs (m)"]) for s in CASTS],
                {"units": "m"},
            ),
            "station": ("profile", list(CASTS), {"cf_role": "profile_id"}),
        },
        {
            "time": ("profile", pd.to_datetime([c[0] for c in CASTS.values()])),
            "lon": ("profile", [float(np.median(c[1])) for c in CASTS.values()]),
            "lat": ("profile", [50.01] * 3),
            "depth": (
                ("profile", "z"),
                [pad(c[2]) for c in CASTS.values()],
                {"units": "m", "positive": "down"},
            ),
        },
        {"featureType": "trajectoryProfile"},
    ).to_netcdf(path)
    return path


def _write(tmp_path, name="c", **entries):
    """Write catalog ``name`` of ``entry=(CSV samples | netCDF path, metadata)``."""
    cat = build.new_catalog(title=name)
    for entry, (source, meta) in entries.items():
        meta = {
            "featureType": "trajectoryProfile",
            "depth_convention": {"origin": "surface"},
            **meta,
        }
        if isinstance(source, Path):
            build.add_source(cat, entry, str(source), name_map=None, **meta)
            continue
        source.to_csv(tmp_path / f"{entry}.csv", index=False)
        csv = readers.PandasCSV(datatypes.CSV(url=str(tmp_path / f"{entry}.csv")))
        build.add_source(cat, entry, None, reader=csv, name_map=None, **meta)
    return build.save(cat, Path(os.environ["OCEAN_SKILL_CATALOGS"]) / f"{name}.yaml")


def _only_his(real, stub):
    """Return ``real``, except that for the model ``his`` it gives ``stub``."""
    return lambda name, *a, **kw: stub if name == "his" else real(name, *a, **kw)


@pytest.fixture
def world(tmp_path, monkeypatch, isolated_catalogs):
    """Stub the model ``his``; write the transect and its by-hand profile twins."""
    ds, meta = tidal_roms(temp=True)
    ds = ds.assign_coords(time=ds["time"].values.astype("datetime64[ns]"))  # as obs
    for owner, attr, stub in [
        (osk, "read", ds),
        (sources, "read", ds),
        (catalog, "resolve", SimpleNamespace(metadata=meta)),
        (comparison, "_domain_of", None),
        (comparison, "_outline_of", None),
    ]:
        monkeypatch.setattr(owner, attr, _only_his(getattr(owner, attr), stub))
    obs = _rows()
    entries = {
        "line": (obs, {"casts": {"id": "station"}}),
        "line_noid": (obs.drop(columns="station"), {"casts": {"gap": "15min"}}),
        "line_bare": (obs.drop(columns="station"), {}),
        "line_one": (obs[obs["station"] == "S1"], {"casts": {"id": "station"}}),
        "line_raw": (_raw_rows(obs), {"casts": {"id": "station"}}),
    }
    for sid in CASTS:  # a profile has one position: the cast's, 200.02 for S3
        rows = obs[obs["station"] == sid].drop(columns="station")
        profile = rows.assign(lon=rows["lon"].median())
        entries[f"p{sid[1]}"] = (profile, {"featureType": "profile"})
    _write(tmp_path, **entries)
    return tmp_path


def _compare(reference, spec=HEIGHT, **kw):
    """Return ``osk.compare`` of the model against ``reference``, interpolating."""
    kw = {"time_method": "interp", "depth_method": "interp", "cache": False, **kw}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return osk.compare(test="his", reference=reference, variables=[spec], **kw)


def _lane(c, lane="test"):
    return np.asarray(c.aligned[lane])


def _assert_same(got, want):
    """Check two comparisons hold the same aligned values and the same metrics."""
    for lane in LANES:
        np.testing.assert_allclose(_lane(got, lane), _lane(want, lane), atol=1e-9)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # "metrics are weakly constrained"
        numbers = [
            {
                k: v
                for k, v in c.metrics().items()
                if isinstance(v, int | float | np.number) and not isinstance(v, bool)
            }
            for c in (got, want)
        ]
    assert numbers[0] == pytest.approx(numbers[1], nan_ok=True)


def test_the_casts_are_listed_in_time_order(world):
    assert osk.cast_names("line") == ["line[S1]", "line[S2]", "line[S3]"]
    assert osk.cast_names("line_noid") == [f"line_noid[{i}]" for i in "123"]


@pytest.mark.parametrize("entry", ["line", "line_noid"])
def test_the_casts_of_a_transect_compare_like_profiles(world, entry):
    want = _compare(["p1", "p2", "p3"])
    by_name = _compare(osk.cast_names(entry))
    by_over = _compare(entry, over="Z")  # a vertical over= is the casts' profiles too
    assert [c.reference_name for c in by_name] == osk.cast_names(entry)
    assert [c.reference_name for c in by_over] == osk.cast_names(entry)
    assert len(by_name) == len(by_over) == len(want) == 3
    for got in (by_name, by_over):
        for a, b in zip(got, want, strict=True):
            _assert_same(a, b)


def test_depths_follow_the_surface_at_each_casts_own_time(world):
    for c, sid in zip(_compare(osk.cast_names("line")), CASTS, strict=True):
        np.testing.assert_allclose(_lane(c), _z(ZETA[sid], CASTS[sid][2]), atol=1e-6)
        np.testing.assert_array_equal(_lane(c, "depth"), CASTS[sid][2])


def test_the_default_time_method_takes_the_nearest_step(world):
    (c,) = _compare("line[S3]", time_method="auto")  # 02:20 is nearest 02:00: -3 m
    np.testing.assert_allclose(_lane(c), _z(-3.0, CASTS["S3"][2]), atol=1e-6)


def test_the_model_is_read_at_the_casts_own_time_and_place(world):
    for c, sid in zip(_compare(osk.cast_names("line"), TEMP), CASTS, strict=True):
        z = _z(ZETA[sid], CASTS[sid][2])
        want = 10 + HOURS[sid] + 0.5 * XI[sid] + 0.1 * z
        np.testing.assert_allclose(_lane(c), want, atol=1e-6)


def test_the_casts_make_one_section_of_their_own_times_and_places(world):
    depths = [1.0, 3.0]
    (section,) = _compare(
        "line", select={"transect": {"from": "reference"}, "depth": depths}
    )
    assert section.is_section
    a = section.aligned.transpose(ALONG_DIM, "z")
    assert a.sizes[ALONG_DIM] == 3 and (np.diff(a[ALONG_DIM]) > 0).all()
    times = np.array([c[0] for c in CASTS.values()], dtype="datetime64[ns]")
    np.testing.assert_array_equal(a["cast_time"], times)
    np.testing.assert_allclose(a["lon"] % 360, [200.0, 200.01, 200.02], atol=1e-9)
    # no averaging across casts: each column is the model at its own cast's time
    want = np.stack([_z(ZETA[sid], depths) for sid in CASTS])
    np.testing.assert_allclose(a["test"], want, atol=1e-6)
    # S2 has no sample at 1 m
    nan = np.nan
    np.testing.assert_array_equal(a["reference"], [[101, 103], [nan, 203], [301, 303]])
    for k, sid in enumerate(CASTS):
        (child,) = _compare(f"line[{sid}]", select={"depth": depths})
        for lane in LANES:
            np.testing.assert_allclose(a[lane].values[k], _lane(child, lane), atol=1e-6)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")  # netCDF4 under numpy 2.5
def test_a_netcdf_transect_compares_like_the_csv_one(world):
    _write(world, "n", line_nc=(_nc(world / "line.nc"), {}))
    assert osk.cast_names("line_nc") == [f"line_nc[{s}]" for s in CASTS]  # profile_id
    names = {e: osk.cast_names(e) for e in ("line_nc", "line")}
    for got, want in zip(*(_compare(n) for n in names.values()), strict=True):
        _assert_same(got, want)  # cast by cast
    (got,), (want,) = _compare("line_nc"), _compare("line")  # and as one section
    assert got.is_section and want.is_section
    _assert_same(got, want)


def test_a_casts_cache_key_is_its_own_and_follows_the_declaration(world):
    keys = [
        Comparison(reference=f"line[{sid}]", test="his", variable=HEIGHT)._cache_key
        for sid in ("S1", "S2", "S1")
    ]
    assert keys[0] != keys[1] and keys[0] == keys[2]
    before = catalog.fingerprint("line_noid[1]")
    gap = {"casts": {"gap": "30min"}}  # still three casts: 50+ minutes apart
    _write(world, line_noid=(_rows().drop(columns="station"), gap))
    assert osk.cast_names("line_noid")[0] == "line_noid[1]"
    assert before and catalog.fingerprint("line_noid[1]") != before


def test_a_transect_entry_is_one_section_on_its_casts_own_levels(world):
    (section,) = _compare("line")
    assert section.is_section and section.reference_name == "line (3 casts)"
    a = section.aligned.transpose(ALONG_DIM, "z")
    assert a.sizes[ALONG_DIM] == 3
    levels = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_allclose(-a["z"], levels)
    nan = np.nan
    np.testing.assert_array_equal(
        a["reference"],
        [
            [101, 102, 103, 104, 105],
            [nan, nan, 203, nan, nan],
            [301, 302, 303, 304, 305],
        ],
    )
    for k, sid in enumerate(CASTS):
        own = np.isin(levels, CASTS[sid][2])  # S2 only at 3 m: not smeared to 2 and 4
        np.testing.assert_allclose(
            a["test"].values[k][own], _z(ZETA[sid], CASTS[sid][2]), atol=1e-6
        )
        assert np.isnan(a["test"].values[k][~own]).all()
        (child,) = _compare(f"line[{sid}]")
        for lane in LANES:
            np.testing.assert_allclose(
                a[lane].values[k][own], _lane(child, lane), atol=1e-6
            )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # "metrics are weakly constrained"
        assert section.metrics()["n"] == 11  # every (cast, level) pair, 5 + 1 + 5


def test_named_levels_are_the_sections_levels(world):
    depths = [1, 3]
    (section,) = _compare("line", depths=depths)
    (explicit,) = _compare("line", select={"depth": depths})
    assert section.is_section and explicit.is_section
    a = section.aligned.transpose(ALONG_DIM, "z")
    np.testing.assert_allclose(-a["z"], depths)
    # named levels: the model at both for every cast, S2's 1 m included (its own
    # reading is NaN there, 2 m from its one sample)
    np.testing.assert_allclose(
        a["test"], np.stack([_z(ZETA[s], depths) for s in CASTS])
    )
    _assert_same(section, explicit)


def test_a_one_cast_transect_is_a_profile_comparison(world):
    (c,) = _compare("line_one")
    assert not c.is_section and c.reference_name == "line_one[S1]"


def test_a_cast_named_beside_its_entry_is_also_compared_alone(world):
    section, alone = _compare(["line", "line[S2]"])
    assert section.is_section and section.reference_name == "line (3 casts)"
    _assert_same(alone, _compare("line[S2]")[0])


def test_levels_that_are_not_a_list_compare_the_casts_one_by_one(world, monkeypatch):
    formed = []
    monkeypatch.setattr(
        Comparison,
        "align",
        lambda self, refresh=False: formed.append(self.reference_name),
    )
    for kw in ({"depths": [3]}, {"select": {"depth": 3}}):
        formed.clear()
        osk.compare(test="his", reference="line", variables=[HEIGHT], **kw)
        assert formed == osk.cast_names("line"), kw


class _Stop(Exception):
    pass


@pytest.mark.parametrize(
    "kw, whole",
    [
        ({}, True),
        ({"depths": [1, 3]}, True),
        ({"depths": [{"min": 0, "max": 4}]}, False),
        ({"times": ["2024-07-01"]}, False),
        ({"select": {"sigma0": 26.5}}, False),
        (
            {"select": {"test": {"depth": [1, 3]}, "reference": {"depth": [1, 3]}}},
            False,
        ),
        ({"select": {"transect": {"from": "reference"}, "depth": [1, 3]}}, False),
        ({"aggregate": {"depth": "mean"}}, False),
    ],
)
def test_what_keeps_a_transect_whole(world, monkeypatch, kw, whole):
    seen = []

    def stop(refs, over, *, one_transect):
        seen.append(one_transect)
        raise _Stop

    monkeypatch.setattr(comparison, "_expand_transects", stop)
    with pytest.raises(_Stop):
        _compare("line", **kw)
    assert seen == [whole]


def test_unidentified_casts_are_a_moving_platform_with_a_warning(world, monkeypatch):
    with pytest.raises(casts.NoCasts):
        osk.cast_names("line_bare")
    formed = []
    monkeypatch.setattr(
        Comparison, "align", lambda self, refresh=False: formed.append(self)
    )
    with pytest.warns(UserWarning, match="not identified"):
        osk.compare(test="his", reference="line_bare", variables=[HEIGHT], depths=[1])
    assert [(c.reference_name, c._section_casts) for c in formed] == [
        ("line_bare", None)
    ]


def test_a_non_vertical_over_expands_nothing_and_reads_nothing(world, monkeypatch):
    monkeypatch.setattr(casts, "names", lambda *_: pytest.fail("read the casts"))
    assert comparison._expand_transects(["line"], "time") == (["line"], {})


def test_unbinned_casts_need_named_levels_or_the_cast_by_cast_route(world):
    with pytest.raises(ValueError, match="not binned onto shared levels"):
        _compare("line_raw")
    (section,) = _compare("line_raw", select={"depth": [1, 3]})
    np.testing.assert_allclose(-section.aligned["z"], [1, 3])
    assert len(_compare(osk.cast_names("line_raw"))) == 3


def test_a_transect_sections_cache_key_follows_how_its_levels_were_made(world):
    levels = [1, 2, 3, 4, 5]
    explicit = {"transect": {"from": "reference"}, "depth": levels}
    (default,), (named,) = _compare("line"), _compare("line", select=explicit)
    assert default._cache_key != named._cache_key
    kw = {"reference": "x", "test": "his", "variable": HEIGHT, "select": explicit}
    kw["section_casts"] = osk.cast_names("line")
    assert (
        Comparison(**kw)._cache_key
        != Comparison(**kw, section_own_levels=True)._cache_key
    )
