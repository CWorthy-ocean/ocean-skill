"""roms-tools forcing files (boundary physics/BGC, climatology, surface) as sources.

``build.add_boundary_sources`` / ``build.add_forcing_source`` catalogue a run's ROMS
forcing files with ordinary intake reader chains (subset, rename, ``expand_dims``, grid
``merge``) -- no custom reader -- so a run's boundary conditions read like ROMS output:
two runs' forcing compare, and a boundary is a section that compares with a climatology.
The README's "ROMS forcing files (roms-tools)" recipe is this build.

The synthetic files follow roms-tools' shapes: the boundary file has no coordinates but
``bry_time`` (days, the epoch in ``long_name``) and ``abs_time``; each variable appears
once per side on ``(bry_time, s_rho, xi_rho|eta_rho)``; ``u``/``v`` sit on staggered
dims; the grid file has ``lon_rho``/``lat_rho``/``h``/``mask_rho``/``sigma_r``/``Cs_r``
and ``hc``/``theta_s``/``theta_b`` as global attributes.
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import ocean_skill as osk
from ocean_skill import build, roms, sources

NY, NX, NS = 5, 6, 3
LON0, LAT0 = -20.0, 60.0
EPOCH = "2011-01-01"
TEMP = "sea_water_potential_temperature"
SALT = "sea_water_practical_salinity"
NITRATE = "mole_concentration_of_nitrate_in_sea_water"
SIDES = ("south", "north", "east", "west")


def _grid(tmp_path):
    lon = LON0 + 0.5 * np.arange(NX)[None, :] + 0.01 * np.arange(NY)[:, None]
    lat = LAT0 + 0.5 * np.arange(NY)[:, None] + 0.01 * np.arange(NX)[None, :]
    sigma = (np.arange(NS) - NS + 0.5) / NS
    sigma_w = np.linspace(-1.0, 0.0, NS + 1)
    cs = sigma**3  # any monotonic stretching will do
    ds = xr.Dataset(
        {
            "lon_rho": (("eta_rho", "xi_rho"), lon),
            "lat_rho": (("eta_rho", "xi_rho"), lat),
            "h": (("eta_rho", "xi_rho"), np.full((NY, NX), 500.0)),
            "mask_rho": (("eta_rho", "xi_rho"), np.ones((NY, NX))),
            "angle": (("eta_rho", "xi_rho"), np.zeros((NY, NX))),
            "sigma_r": ("s_rho", sigma),
            "Cs_r": ("s_rho", cs),
            "sigma_w": ("s_w", sigma_w),
            "Cs_w": ("s_w", sigma_w**3),
        },
        attrs={"hc": 50.0, "theta_s": 5.0, "theta_b": 2.0},
    )
    path = tmp_path / "grid.nc"
    ds.to_netcdf(path)
    return path, lon, lat


def _bry_time(days):
    t = xr.DataArray(
        np.asarray(days, dtype="float64"),
        dims="bry_time",
        attrs={
            "long_name": f"relative time: days since {EPOCH} 00:00:00",
            "units": "days",
        },
    )
    return t


def _boundary_file(path, days, offset=0.0, *, month=False):
    """One boundary file: temp/salt/NO3/zeta per side plus the staggered u/v."""
    days = np.asarray(days)
    nt = len(days)
    rng = np.random.default_rng(int(days[0]) + 1)
    data, along = {}, {"south": NX, "north": NX, "east": NY, "west": NY}
    for side, n in along.items():
        axis = "xi_rho" if side in ("south", "north") else "eta_rho"
        base = np.arange(n)[None, None, :] + np.arange(nt)[:, None, None] * 0.1
        for var, scale, units in (
            ("temp", 1.0, "degrees Celsius"),
            ("salt", 0.1, "PSU"),
            ("NO3", 0.5, "mmol/m3"),
        ):
            vals = scale * base + offset + 0.01 * np.arange(NS)[None, :, None]
            data[f"{var}_{side}"] = (
                ("bry_time", "s_rho", axis),
                vals,
                {"units": units},
            )
        data[f"zeta_{side}"] = (("bry_time", axis), 0.1 * base[:, 0, :])
        stag = {"south": "xi_u", "north": "xi_u", "east": "eta_v", "west": "eta_v"}[
            side
        ]
        m = n - 1
        data[f"u_{side}"] = (("bry_time", "s_rho", stag), rng.normal(size=(nt, NS, m)))
        data[f"ubar_{side}"] = (("bry_time", stag), rng.normal(size=(nt, m)))
    bt = _bry_time(days)
    coords = {
        "bry_time": bt,
        "abs_time": (
            "bry_time",
            np.datetime64(EPOCH)
            + (days * 86400).astype("timedelta64[s]").astype("timedelta64[ns]"),
        ),
    }
    if month:
        coords["month"] = ("bry_time", np.arange(1, nt + 1))
    xr.Dataset(data, coords=coords).to_netcdf(path)


def _surface_file(path, offset=0.0):
    nt = 6
    hours = np.arange(nt) / 24.0
    dims = ("time", "eta_rho", "xi_rho")
    rng = np.random.default_rng(3)
    ds = xr.Dataset(
        {
            v: (dims, offset + rng.normal(size=(nt, NY, NX)))
            for v in ("Tair", "swrad", "uwnd")
        },
        coords={
            "time": (
                "time",
                hours,
                {
                    "long_name": f"relative time: days since {EPOCH} 00:00:00",
                    "units": "days",
                },
            )
        },
    )
    ds.to_netcdf(path)


@pytest.fixture
def forcing(isolated_catalogs, tmp_path):
    """Two runs (B = A + 1) on one grid: two-file boundary, a climatology, surface."""
    grid, lon, lat = _grid(tmp_path)
    cat = build.new_catalog(title="forcing")
    for run, offset in (("a", 0.0), ("b", 1.0)):
        f1, f2 = tmp_path / f"{run}_bry_201101.nc", tmp_path / f"{run}_bry_201102.nc"
        _boundary_file(f1, [0, 1, 2], offset)
        _boundary_file(f2, [31, 32, 33], offset)
        build.add_boundary_sources(cat, f"{run}_bry", [f1, f2], grid)
    clim = tmp_path / "a_clim.nc"
    _boundary_file(clim, np.arange(12) * 30.4, 0.0, month=True)
    build.add_boundary_sources(cat, "a_clim", clim, grid, climatology=True)
    surf = tmp_path / "a_surface.nc"
    _surface_file(surf)
    build.add_forcing_source(cat, "a_surface", surf, grid)
    build.save(cat, isolated_catalogs / "forcing.yaml")
    return {"lon": lon, "lat": lat, "tmp": tmp_path}


DEPTHS = [
    20.0,
    100.0,
    200.0,
]  # z_rho of the synthetic grid runs from about -300 to -10 m


def test_every_side_is_an_entry_that_reads_like_roms_output(forcing):
    ds = sources.read("a_bry_south")
    assert {TEMP, SALT, NITRATE, "sea_surface_height_above_geoid"} <= set(ds.data_vars)
    assert ds[TEMP].dims == ("time", "s_rho", "eta_rho", "xi_rho")
    assert ds.sizes["eta_rho"] == 1 and ds.sizes["xi_rho"] == NX
    assert {"z_rho", "lon", "lat"} <= set(ds.coords) and ds["z_rho"].sizes[
        "s_rho"
    ] == NS
    # two monthly files concatenated, the days decoded from the epoch in the long_name
    assert ds["time"].dtype.kind == "M"
    assert list(ds["time"].values.astype("datetime64[D]")) == list(
        np.datetime64(EPOCH) + np.array([0, 1, 2, 31, 32, 33])
    )
    # each side carries its own grid row/column: south the first eta row, east the last
    # xi column, and so on
    lon, lat = forcing["lon"], forcing["lat"]
    rows = {
        "south": ("eta_rho", lon[:1], lat[:1]),
        "north": ("eta_rho", lon[-1:], lat[-1:]),
        "west": ("xi_rho", lon[:, :1], lat[:, :1]),
        "east": ("xi_rho", lon[:, -1:], lat[:, -1:]),
    }
    for side, (dim, want_lon, want_lat) in rows.items():
        d = sources.read(f"a_bry_{side}")
        assert d.sizes[dim] == 1
        np.testing.assert_allclose(d["lon"].values, want_lon)
        np.testing.assert_allclose(d["lat"].values, want_lat)
        assert sources.resolve(f"a_bry_{side}").metadata["boundary_side"] == side


def test_the_entry_metadata_says_roms_with_vertical_time_and_names(forcing):
    meta = sources.resolve("a_bry_south").metadata
    assert meta["model"] == "roms" and meta["featureType"] == "grid"
    assert meta["vertical"] == {
        "s_dim": "s_rho",
        "hc": 50.0,
        "theta_s": 5.0,
        "theta_b": 2.0,
        "Vtransform": 2,
    }
    assert (meta["time_coord"], meta["time_dim"], meta["time_units"]) == (
        "time",
        "time",
        "days",
    )
    assert meta["reference_date"] == EPOCH
    # the probe recorded the time coverage of both files from the days axis
    assert (meta["time_coverage_start"], meta["time_coverage_end"]) == (
        "2011-01-01",
        "2011-02-03",
    )
    assert set(meta["variables"]) == {
        TEMP,
        SALT,
        NITRATE,
        "sea_surface_height_above_geoid",
    }


def test_velocities_stay_out_of_boundary_entries(forcing, isolated_catalogs):
    ds = sources.read("a_bry_north")
    assert not {"u", "v", "ubar", "vbar"} & set(ds.variables)
    with pytest.raises(ValueError, match="velocities"):
        build.add_boundary_sources(
            build.new_catalog(),
            "x",
            forcing["tmp"] / "a_clim.nc",
            forcing["tmp"] / "grid.nc",
            variables=["temp", "u"],
            climatology=True,
        )


def test_a_south_entry_is_a_section_along_the_boundary(forcing):
    f = osk.field(
        source="a_bry_south", variable=TEMP, select={"transect": {"eta_rho": 0}}
    )
    da = f.data
    assert set(da.dims) == {"time", "s_rho", "along"} and da.sizes["along"] == NX
    np.testing.assert_allclose(da["path_lon"].values, forcing["lon"][0])
    np.testing.assert_allclose(da["path_lat"].values, forcing["lat"][0])
    assert (np.diff(da["along"].values) > 0).all()
    # temp = i along the boundary + 0.1 per step + 0.01 per level, as written
    np.testing.assert_allclose(da.isel(time=1, s_rho=2).values, np.arange(NX) + 0.12)


def test_two_runs_forcing_compares_along_the_boundary(forcing):
    result = osk.compare(
        test="a_bry_south",
        reference="b_bry_south",
        variables=[TEMP],
        select={"transect": {"eta_rho": 0}, "depth": DEPTHS},
        aggregate={"time": "mean"},
    )
    (c,) = result.comparisons
    assert c.is_section and c.family == "section_row"
    aligned = c.aligned
    assert aligned.sizes["along"] == NX and aligned.sizes["z"] == len(DEPTHS)
    # run B is run A plus one degree
    np.testing.assert_allclose(aligned["difference"].values, -1.0)


def test_a_climatology_entry_selects_a_month(forcing):
    ds = sources.read("a_clim_south")
    assert ds[TEMP].dims[0] == "month" and list(ds["month"].values) == list(
        range(1, 13)
    )
    assert "time" not in ds.variables and "abs_time" not in ds.variables
    meta = sources.resolve("a_clim_south").metadata
    assert meta["climatology"] is True and meta["climatology_period"] == "monthly"
    f = osk.field(source="a_clim_south", variable=TEMP, select={"month": 3})
    assert "month" not in f.data.dims
    # month 3 is the third step of the file, 0.2 along the time axis of the temp ramp
    np.testing.assert_allclose(f.data.isel(s_rho=0, eta_rho=0, xi_rho=0).values, 0.2)


def test_surface_forcing_reads_on_the_grid_by_raw_name(forcing):
    ds = sources.read("a_surface")
    assert {"Tair", "swrad", "uwnd"} <= set(ds.data_vars)
    assert ds["Tair"].dims == ("time", "eta_rho", "xi_rho")
    assert ds["time"].dtype.kind == "M"
    assert ds["time"].values[1] - ds["time"].values[0] == np.timedelta64(3600, "s")
    np.testing.assert_allclose(ds["lon"].values, forcing["lon"])
    assert sources.resolve("a_surface").metadata["model"] == "roms"
    f = osk.field(source="a_surface", variable="Tair")
    assert f.data.sizes["time"] == 6


def test_a_boundary_section_compares_with_a_woa_like_climatology(
    forcing, isolated_catalogs
):
    """The model's boundary against a 1-degree observed climatology, along its path."""
    lon, lat = np.arange(-23.0, -14.0), np.arange(58.0, 65.0)
    depth = np.array([0.0, 10.0, 50.0, 100.0, 500.0])
    month = np.arange(1, 13)
    temp = np.broadcast_to(
        5.0 + 0.1 * month[:, None, None, None], (12, depth.size, lat.size, lon.size)
    )
    xr.Dataset(
        {
            "t_an": (
                ("month", "depth", "lat", "lon"),
                temp,
                {"standard_name": TEMP, "units": "degC"},
            )
        },
        coords={
            "month": month,
            "depth": ("depth", depth, {"positive": "down"}),
            "lat": lat,
            "lon": lon,
        },
    ).to_netcdf(forcing["tmp"] / "woa.nc")
    build.build_catalog(
        {
            "woa_like": {
                "url": str(forcing["tmp"] / "woa.nc"),
                "climatology": True,
                "climatology_period": "monthly",
            }
        },
        isolated_catalogs / "woa.yaml",
        title="WOA-like",
        name_map=None,
    )
    result = osk.compare(
        test="a_bry_south",
        reference="woa_like",
        variables=[TEMP],
        select={"transect": {"eta_rho": 0}, "depth": DEPTHS},
        aggregate={"test": {"time": "mean"}, "reference": {"month": "mean"}},
    )
    (c,) = result.comparisons
    assert c.is_section
    aligned = c.aligned
    assert aligned.sizes["z"] == len(DEPTHS) and aligned.sizes["along"] >= 2
    # the reference is the (month-mean) climatology, 5.65 everywhere
    np.testing.assert_allclose(aligned["reference"].values, 5.65)


def test_building_raises_on_a_grid_that_is_not_roms_tools(forcing):
    bad = forcing["tmp"] / "bad_grid.nc"
    xr.Dataset({"h": (("eta_rho", "xi_rho"), np.ones((NY, NX)))}).to_netcdf(bad)
    with pytest.raises(ValueError, match="not a roms-tools grid"):
        build.add_forcing_source(
            build.new_catalog(), "s", forcing["tmp"] / "a_surface.nc", bad
        )


def test_building_a_complete_set_warns_about_nothing(isolated_catalogs, tmp_path):
    grid, _, _ = _grid(tmp_path)
    f = tmp_path / "bry.nc"
    _boundary_file(f, [0, 1, 2])
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        names = build.add_boundary_sources(build.new_catalog(), "r", f, grid)
    assert names == [f"r_{side}" for side in SIDES]


def test_roms_decodes_days_and_hours_from_the_epoch_the_file_states():
    t = xr.DataArray(
        [0.0, 1.5, 31.0],
        dims="time",
        attrs={
            "units": "days",
            "long_name": "relative time: days since 2011-01-01 00:00:00",
        },
    )
    ds = xr.Dataset(coords={"time": t})
    out = roms._decode_time(ds, {"time_coord": "time", "time_units": "days"})
    assert list(out["time"].values) == [
        np.datetime64("2011-01-01T00:00:00"),
        np.datetime64("2011-01-02T12:00:00"),
        np.datetime64("2011-02-01T00:00:00"),
    ]
    # the entry's own reference_date wins, and hours work the same way
    hours = xr.Dataset(
        coords={"time": ("time", np.array([0.0, 36.0]), {"units": "hours"})}
    )
    out = roms._decode_time(
        hours,
        {"time_coord": "time", "time_units": "hours", "reference_date": "2000-03-01"},
    )
    assert list(out["time"].values) == [
        np.datetime64("2000-03-01T00:00:00"),
        np.datetime64("2000-03-02T12:00:00"),
    ]


def test_roms_time_in_days_needs_an_epoch_and_other_units_still_raise():
    ds = xr.Dataset(coords={"time": ("time", np.array([0.0]), {"units": "days"})})
    with pytest.raises(ValueError, match="no 'reference_date'"):
        roms._decode_time(ds, {"time_coord": "time", "time_units": "days"})
    with pytest.raises(ValueError, match="Unsupported time_units"):
        roms._decode_time(ds, {"time_coord": "time", "time_units": "months"})


def _time_file(tmp_path, units, values=(0.0, 90.0)):
    path = tmp_path / "t.nc"
    t = xr.DataArray(
        np.asarray(values),
        dims="time",
        attrs={"units": units, "long_name": f"relative time: since {EPOCH} 00:00:00"},
    )
    xr.Dataset(coords={"time": t}).to_netcdf(path)
    return path


def test_forcing_time_keeps_a_recognized_unit_and_decodes_minutes(tmp_path):
    path = _time_file(tmp_path, "minutes")
    with xr.open_dataset(path, decode_times=False) as ds:
        md = build._forcing_time(ds, "time", False, {}, path)
    assert md["time_units"] == "minutes" and md["reference_date"] == EPOCH
    out = roms._decode_time(ds, {"time_coord": "time", "time_dim": "time", **md})
    assert list(out["time"].values) == [
        np.datetime64("2011-01-01T00:00:00"),
        np.datetime64("2011-01-01T01:30:00"),
    ]


def test_forcing_time_defaults_to_days_only_when_no_unit_is_stated(tmp_path):
    path = _time_file(tmp_path, "days")
    with xr.open_dataset(path, decode_times=False) as ds:
        assert build._forcing_time(ds, "time", False, {}, path)["time_units"] == "days"
        ds["time"].attrs.pop("units")
        assert build._forcing_time(ds, "time", False, {}, path)["time_units"] == "days"


def test_forcing_time_in_an_unknown_unit_raises_naming_file_variable_and_unit(
    tmp_path,
):
    path = _time_file(tmp_path, "months since 2011-01-01")
    with xr.open_dataset(path, decode_times=False) as ds:
        with pytest.raises(ValueError, match=r"t\.nc.*'time'.*'months'"):
            build._forcing_time(ds, "time", False, {}, path)
    # and it fails at catalog-build time, not by quietly reading months as days
    grid, _, _ = _grid(tmp_path)
    surf = tmp_path / "surf.nc"
    _surface_file(surf)
    with xr.open_dataset(surf) as src:
        bad = src.copy(deep=True)
    bad["time"].attrs["units"] = "months"
    bad.to_netcdf(tmp_path / "bad.nc")
    with pytest.raises(ValueError, match=r"bad\.nc.*'months'"):
        build.add_forcing_source(build.new_catalog(), "m", tmp_path / "bad.nc", grid)


def test_roms_seconds_still_decode_as_they_did():
    ds = xr.Dataset(coords={"ocean_time": ("time", np.array([0.0, 86400.0]))})
    out = roms._decode_time(ds, {"reference_date": "2000-01-01"})
    assert list(out["time"].values) == [
        np.datetime64("2000-01-01"),
        np.datetime64("2000-01-02"),
    ]


def test_the_probe_decodes_days_with_the_epoch_in_the_long_name():
    var = xr.DataArray(
        [0.0, 2.0],
        dims="t",
        attrs={
            "units": "days",
            "long_name": "relative time: days since 2011-01-01 00:00:00",
        },
    )
    got = build._decode_times(None, var)
    assert list(got) == [np.datetime64("2011-01-01"), np.datetime64("2011-01-03")]
    # undecodable units are still left alone
    assert build._decode_times(None, var.assign_attrs(units="months")) is None


# ======================================================================================
# A suite's ``refresh.forcing:`` block: the same entries, built into the refresh catalog
# ======================================================================================


def _forcing_files(tmp_path):
    """One run's boundary forcing split over two folders, plus a surface file."""
    grid, _, _ = _grid(tmp_path)
    for year_dir, days in (("y2011", [0, 1, 2]), ("y2012", [31, 32, 33])):
        (tmp_path / year_dir).mkdir()
        _boundary_file(tmp_path / year_dir / f"bry_{year_dir}.nc", days)
    _surface_file(tmp_path / "surface.nc")
    return grid


def test_refresh_forcing_joins_glob_lists_into_the_refresh_catalog(
    isolated_catalogs, tmp_path
):
    import intake

    from ocean_skill.workflows.run import _refresh_forcing

    grid = _forcing_files(tmp_path)
    catalog_path = isolated_catalogs / "run.yaml"
    model = build.new_catalog(title="run")
    _surface_file(tmp_path / "model.nc")
    build.add_forcing_source(model, "model_output", tmp_path / "model.nc", grid)
    build.save(model, catalog_path)

    _refresh_forcing(
        [
            {
                "name": "bry",
                "kind": "boundary",
                # listed latest folder last: the order given is the time order
                "files": [str(tmp_path / "y2011/*.nc"), str(tmp_path / "y2012/*.nc")],
                "grid": str(grid),
                "sides": ["south", "north"],
                "variables": ["temp", "salt"],
            },
            {
                "name": "surf",
                "kind": "surface",
                "files": str(tmp_path / "surface.nc"),
                "grid": str(grid),
                "time_coord": "time",
            },
        ],
        catalog_path,
    )

    names = set(intake.from_yaml_file(str(catalog_path)))
    assert names == {"model_output", "bry_south", "bry_north", "surf"}
    ds = sources.read("bry_south")
    assert {TEMP, SALT} <= set(ds.data_vars) and NITRATE not in ds
    assert list(ds["time"].values.astype("datetime64[D]")) == list(
        np.datetime64(EPOCH) + np.array([0, 1, 2, 31, 32, 33])
    )
    assert "Tair" in sources.read("surf")


def test_refresh_forcing_keeps_entries_whose_files_are_gone(
    isolated_catalogs, tmp_path
):
    import intake

    from ocean_skill.workflows.run import _refresh_forcing

    grid = _forcing_files(tmp_path)
    catalog_path = isolated_catalogs / "run.yaml"
    spec = [
        {
            "name": "surf",
            "kind": "surface",
            "files": str(tmp_path / "surface.nc"),
            "grid": str(grid),
        }
    ]
    _refresh_forcing(spec, catalog_path)
    spec[0]["files"] = str(tmp_path / "nothing_here_*.nc")
    _refresh_forcing(spec, catalog_path)  # must not raise
    assert set(intake.from_yaml_file(str(catalog_path))) == {"surf"}


def test_refresh_forcing_reports_a_failed_entry_and_builds_the_rest(
    isolated_catalogs, tmp_path, capsys
):
    import intake

    from ocean_skill.workflows.run import _refresh_forcing

    grid = _forcing_files(tmp_path)
    catalog_path = isolated_catalogs / "run.yaml"
    surface = {
        "kind": "surface",
        "files": str(tmp_path / "surface.nc"),
        "grid": str(grid),
    }
    _refresh_forcing(
        [
            {"name": "bad", **surface, "variables": ["not_a_variable"]},
            {"name": "good", **surface},
        ],
        catalog_path,
    )
    out = capsys.readouterr().out
    assert "FAILED bad" in out and "surface.nc" in out
    assert set(intake.from_yaml_file(str(catalog_path))) == {"good"}


@pytest.mark.parametrize(
    ("kind", "extra", "message"),
    [
        ("surface", {"sides": ["south"]}, "sides: is for kind: boundary"),
        ("boundary", {"time_coord": "time"}, "time_coord: is for kind: surface"),
        ("lateral", {}, "boundary"),
    ],
)
def test_refresh_forcing_keys_are_checked_against_the_kind(kind, extra, message):
    from pydantic import ValidationError

    from ocean_skill.config import RefreshConfig

    with pytest.raises(ValidationError, match=message):
        RefreshConfig.model_validate(
            {
                "catalog": "c.yaml",
                "forcing": [
                    {"name": "f", "kind": kind, "files": "*.nc", "grid": "g.nc"} | extra
                ],
            }
        )


# ======================================================================================
# The real roms-tools layouts: climatology with a plain ``month`` variable, a time
# variable named unlike its dimension (CO2), and ``ref=`` (a kerchunk reference)
# ======================================================================================

DAYS_2000 = "relative time: days since 2000-01-01 00:00:00"


def _clim_surface_file(path, names):
    """Write a roms-tools surface climatology.

    ``month`` is a data variable and there is no ``time`` variable, only one
    ``<name>_time`` per field and an ``abs_time``.
    """
    days = 30.4 * np.arange(12)
    data: dict = {}
    for k, v in enumerate(names):
        vals = k + np.arange(12)[:, None, None] + np.zeros((12, NY, NX))
        data[v] = (("time", "eta_rho", "xi_rho"), vals)
        data[f"{v}_time"] = (
            "time",
            days,
            {"units": "days", "long_name": DAYS_2000},
        )
    data["month"] = ("time", np.arange(1, 13), {"long_name": "Month index (1-12)"})
    data["abs_time"] = (
        "time",
        days,
        {"units": "days since 2000-01-16 00:00:00"},
    )
    attrs = {"climatology": "True", "model_reference_date": "2000-01-01 00:00:00"}
    xr.Dataset(data, attrs=attrs).to_netcdf(path)


CO2_YEARS = (2011, 2012, 2013)
CO2_STEP, CO2_N = 20.0, 16  # days between records, records per file


def _co2_days(year):
    """Return the days since 2000-01-01 of a year's CO2 records."""
    start = np.datetime64(f"{year}-01-01") - np.datetime64("2000-01-01")
    return start.astype(int) + CO2_STEP * np.arange(CO2_N)


def _co2_file(path, year):
    """Write one year of the CO2 file.

    ``xco2_time`` is the time variable, on dimension ``time``, and there is no variable
    called ``time``.
    """
    days = _co2_days(year)
    vals = (
        (year - 2010)
        + np.arange(CO2_N)[:, None, None] * 0.01
        + np.zeros((CO2_N, NY, NX))
    )
    xr.Dataset(
        {
            "xco2_air": (("time", "eta_rho", "xi_rho"), vals),
            "xco2_air_alt": (("time", "eta_rho", "xi_rho"), vals + 0.5),
            "abs_time": (
                "time",
                days * 1440.0,
                {"units": "minutes since 2011-01-01 00:00:00"},
            ),
            "xco2_time": ("time", days, {"units": "days", "long_name": DAYS_2000}),
        }
    ).to_netcdf(path)


def _physics_file(path, month):
    """Write one month of hourly surface physics.

    ``time`` is both the dimension and the variable, with a second ``rad_time`` and an
    ``abs_time`` beside it.
    """
    nt = 5
    days = 4018 + 31.0 * month + np.arange(nt) / 24.0
    rng = np.random.default_rng(month)
    dims = ("time", "eta_rho", "xi_rho")
    data = {v: (dims, rng.normal(size=(nt, NY, NX))) for v in ("uwnd", "vwnd", "Tair")}
    data["abs_time"] = (
        "time",
        days * 24.0,
        {"units": "hours since 1900-01-01 00:00:00"},
    )
    data["rad_time"] = ("time", days, {"units": "days", "long_name": DAYS_2000})
    xr.Dataset(
        data,
        coords={"time": ("time", days, {"units": "days", "long_name": DAYS_2000})},
    ).to_netcdf(path)


def _case_clim_bgc(tmp_path, grid):
    f = tmp_path / "surface-bgc-unified_clim.nc"
    _clim_surface_file(f, ["dust", "iron", "nox", "nhy"])
    return "surface", [f], {"climatology": True}


def _case_clim_restoring(tmp_path, grid):
    f = tmp_path / "surface-restoring_clim.nc"
    _clim_surface_file(f, ["sss"])
    return "surface", [f], {"climatology": True}


def _case_co2(tmp_path, grid):
    files = []
    for year in CO2_YEARS:
        files.append(tmp_path / f"surface-bgc-mbl_co2_{year}.nc")
        _co2_file(files[-1], year)
    return "surface", files, {"time_coord": "xco2_time"}


def _case_physics(tmp_path, grid):
    files = []
    for month in (0, 1):
        files.append(tmp_path / f"surface-physics_2011{month + 1:02d}.nc")
        _physics_file(files[-1], month)
    return "surface", files, {}


def _case_boundary(tmp_path, grid):
    files = [tmp_path / "bry_201101.nc", tmp_path / "bry_201102.nc"]
    _boundary_file(files[0], [0, 1, 2])
    _boundary_file(files[1], [31, 32, 33])
    return "boundary", files, {"sides": ["south", "west"]}


def _case_boundary_clim(tmp_path, grid):
    f = tmp_path / "bry_clim.nc"
    _boundary_file(f, np.arange(12) * 30.4, month=True)
    return "boundary", [f], {"climatology": True, "sides": ["south"]}


CASES = {
    "clim_bgc": _case_clim_bgc,
    "clim_restoring": _case_clim_restoring,
    "co2": _case_co2,
    "physics": _case_physics,
    "boundary": _case_boundary,
    "boundary_clim": _case_boundary_clim,
}


def _add(cat, name, case, grid, **extra):
    """Add one case's entries to ``cat`` under ``name``; return their names."""
    kind, files, kwargs = case
    if kind == "boundary":
        return build.add_boundary_sources(cat, name, files, grid, **kwargs, **extra)
    return [build.add_forcing_source(cat, name, files, grid, **kwargs, **extra)]


@pytest.fixture
def layouts(isolated_catalogs, tmp_path):
    grid, lon, lat = _grid(tmp_path)
    return {
        "grid": grid,
        "lon": lon,
        "lat": lat,
        "tmp": tmp_path,
        "catalogs": isolated_catalogs,
    }


def _save(layouts, cat, stem="layouts"):
    build.save(cat, layouts["catalogs"] / f"{stem}.yaml")


@pytest.mark.parametrize(("key", "n_vars"), [("clim_bgc", 4), ("clim_restoring", 1)])
def test_a_surface_climatology_with_a_plain_month_variable_builds(layouts, key, n_vars):
    case = CASES[key](layouts["tmp"], layouts["grid"])
    cat = build.new_catalog(title=key)
    (name,) = _add(cat, key, case, layouts["grid"])
    _save(layouts, cat, key)
    ds = sources.read(name)
    fields = {k: v for k, v in ds.data_vars.items() if "month" in v.dims}
    assert len(fields) == n_vars
    assert all(v.dims == ("month", "eta_rho", "xi_rho") for v in fields.values())
    assert list(ds["month"].values) == list(range(1, 13))
    assert "time" not in ds.variables and "abs_time" not in ds.variables
    np.testing.assert_allclose(ds["lon"].values, layouts["lon"])
    np.testing.assert_allclose(ds["lat"].values, layouts["lat"])
    meta = sources.resolve(name).metadata
    assert meta["climatology"] is True and meta["climatology_period"] == "monthly"
    # month 3 is the third step of the file: 2 along the monthly ramp (the first
    # variable in the file has no offset of its own)
    f = osk.field(source=name, variable=next(iter(fields)), select={"month": 3})
    assert "month" not in f.data.dims
    np.testing.assert_allclose(f.data.isel(eta_rho=0, xi_rho=0).values, 2.0)


def test_a_time_variable_named_unlike_its_dimension_joins_the_files(layouts):
    case = _case_co2(layouts["tmp"], layouts["grid"])
    cat = build.new_catalog(title="co2")
    (name,) = _add(cat, "co2", case, layouts["grid"])
    _save(layouts, cat, "co2")
    ds = sources.read(name)
    assert {"xco2_air", "xco2_air_alt"} <= set(ds.data_vars)
    assert ds["xco2_air"].dims == ("time", "eta_rho", "xi_rho")
    assert ds.sizes["time"] == 3 * CO2_N
    assert "xco2_time" not in ds.variables
    want = np.concatenate(
        [
            np.datetime64("2000-01-01") + _co2_days(y).astype("timedelta64[D]")
            for y in CO2_YEARS
        ]
    )
    assert ds["time"].dtype.kind == "M"
    assert list(ds["time"].values.astype("datetime64[D]")) == list(want)
    meta = sources.resolve(name).metadata
    assert (meta["time_coord"], meta["time_dim"], meta["time_units"]) == (
        "time",
        "time",
        "days",
    )
    assert meta["reference_date"] == "2000-01-01"
    # the second year's first record is where its own file put it
    np.testing.assert_allclose(
        ds["xco2_air"].isel(time=CO2_N, eta_rho=0, xi_rho=0), 2.0
    )


def test_a_time_variable_the_file_does_not_have_is_named_in_the_error(layouts):
    f = layouts["tmp"] / "surface-restoring_clim.nc"
    _clim_surface_file(f, ["sss"])
    with pytest.raises(ValueError, match="No time coordinate 'time'"):
        build.add_forcing_source(build.new_catalog(), "s", f, layouts["grid"])
    with pytest.raises(ValueError, match=r"No time coordinate 'time'.*sss_time"):
        build.add_forcing_source(
            build.new_catalog(),
            "s",
            f,
            layouts["grid"],
            ref=layouts["tmp"] / "r.parquet",
        )


@pytest.mark.parametrize(
    ("key", "suffix"),
    [(k, ".parquet") for k in CASES] + [("co2", ".json"), ("boundary", ".json")],
)
def test_ref_reads_the_same_values_as_the_files(layouts, key, suffix):
    case = CASES[key](layouts["tmp"], layouts["grid"])
    cat = build.new_catalog(title=key)
    # the reference first: kerchunking opens each file lock-free, which HDF5 refuses
    # while the same process still holds one open with locking (a plain entry's probe)
    ref = layouts["tmp"] / f"{key}{suffix}"
    viaref = _add(cat, "viaref", case, layouts["grid"], ref=ref)
    plain = _add(cat, "plain", case, layouts["grid"])
    _save(layouts, cat, key)
    assert ref.exists() and Path(str(ref) + ".sources.json").is_file()
    assert [n.replace("viaref", "plain") for n in viaref] == plain
    for a, b in zip(plain, viaref):
        want, got = sources.read(a).load(), sources.read(b).load()
        xr.testing.assert_allclose(got, want)
        assert set(got.coords) == set(want.coords)
        for dim in want.dims:
            assert got.sizes[dim] == want.sizes[dim]
        ma, mb = sources.resolve(a).metadata, sources.resolve(b).metadata
        for k in (
            "time_coord",
            "time_dim",
            "time_units",
            "reference_date",
            "climatology",
            "variables",
            "time_coverage_start",
            "time_coverage_end",
        ):
            assert ma.get(k) == mb.get(k), k


def test_ref_sidecar_records_the_files_and_the_arguments(layouts):
    import json

    cat = build.new_catalog()
    case = _case_co2(layouts["tmp"], layouts["grid"])
    ref = layouts["tmp"] / "co2.parquet"
    _add(cat, "co2", case, layouts["grid"], ref=ref)
    state = json.loads(Path(str(ref) + ".sources.json").read_text())
    files = case[1]
    assert [f["path"] for f in state["files"]] == [str(f.resolve()) for f in files]
    assert state["files"][0]["size"] == files[0].stat().st_size
    assert state["files"][0]["mtime_ns"] == files[0].stat().st_mtime_ns
    assert state["concat_dim"] == "time"
    assert state["loadable_variables"] == ["xco2_time"]
    assert state["keep"] == "last"
    clim = _case_clim_bgc(layouts["tmp"], layouts["grid"])
    _add(cat, "clim", clim, layouts["grid"], ref=layouts["tmp"] / "clim.parquet")
    state = json.loads((layouts["tmp"] / "clim.parquet.sources.json").read_text())
    assert state["concat_dim"] == "time" and state["loadable_variables"] == ["month"]


class _Watch:
    """Count ``make_kerchunk`` calls and record every forcing file xarray opens."""

    def __init__(self, monkeypatch, forcing_files):
        self.builds = 0
        self.opened: list[str] = []
        self.names = {Path(f).name for f in forcing_files}
        real = build.make_kerchunk

        def make(*a, **k):
            self.builds += 1
            return real(*a, **k)

        monkeypatch.setattr(build, "make_kerchunk", make)
        for fn in ("open_dataset", "open_mfdataset"):
            monkeypatch.setattr(xr, fn, self._wrap(getattr(xr, fn)))

    def _wrap(self, real):
        def opener(first, *a, **k):
            paths = first if isinstance(first, (list, tuple)) else [first]
            self.opened += [str(p) for p in paths if Path(str(p)).name in self.names]
            return real(first, *a, **k)

        return opener


@pytest.mark.parametrize("key", ["co2", "clim_bgc", "boundary", "physics"])
def test_a_fresh_ref_is_reused_and_no_forcing_file_is_opened(layouts, monkeypatch, key):
    case = CASES[key](layouts["tmp"], layouts["grid"])
    ref = layouts["tmp"] / "r.parquet"
    watch = _Watch(monkeypatch, case[1])
    cat = build.new_catalog()
    first = _add(cat, "one", case, layouts["grid"], ref=ref)
    assert watch.builds == 1
    watch.opened.clear()
    second = _add(cat, "two", case, layouts["grid"], ref=ref)
    assert watch.builds == 1, "a fresh reference was rebuilt"
    assert watch.opened == []
    _save(layouts, cat, key)
    for name in second:
        sources.read(name).load()
    assert watch.opened == []
    assert [n.replace("one", "two") for n in first] == second


def test_touching_or_adding_a_file_rebuilds_the_ref(layouts, monkeypatch):
    case = _case_co2(layouts["tmp"], layouts["grid"])
    kind, files, kwargs = case
    ref = layouts["tmp"] / "r.parquet"
    watch = _Watch(monkeypatch, files)
    cat = build.new_catalog()
    _add(cat, "a", case, layouts["grid"], ref=ref)
    assert watch.builds == 1
    st = files[1].stat()
    os.utime(files[1], ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    _add(cat, "b", case, layouts["grid"], ref=ref)
    assert watch.builds == 2
    _add(cat, "c", case, layouts["grid"], ref=ref)
    assert watch.builds == 2
    # a fourth file joins the set
    extra = layouts["tmp"] / "surface-bgc-mbl_co2_2014.nc"
    _co2_file(extra, 2014)
    _add(cat, "d", (kind, [*files, extra], kwargs), layouts["grid"], ref=ref)
    assert watch.builds == 3
    _save(layouts, cat, "co2")
    assert sources.read("d").sizes["time"] == 4 * CO2_N


def test_the_order_of_the_files_is_part_of_freshness(layouts, monkeypatch):
    """The files join in the order given, so another order is another reference."""
    _, files, kwargs = _case_co2(layouts["tmp"], layouts["grid"])
    ref = layouts["tmp"] / "r.parquet"
    watch = _Watch(monkeypatch, files)
    for name, order in (("a", files), ("b", files[::-1])):
        build.add_forcing_source(
            build.new_catalog(), name, order, layouts["grid"], ref=ref, **kwargs
        )
    assert watch.builds == 2


def test_two_entries_sharing_a_ref_build_it_once(layouts, monkeypatch):
    _, files, kwargs = _case_co2(layouts["tmp"], layouts["grid"])
    ref = layouts["tmp"] / "shared.parquet"
    watch = _Watch(monkeypatch, files)
    cat = build.new_catalog()
    build.add_forcing_source(cat, "co2_a", files, layouts["grid"], ref=ref, **kwargs)
    build.add_forcing_source(
        cat, "co2_b", files, layouts["grid"], ref=ref, variables=["xco2_air"], **kwargs
    )
    assert watch.builds == 1
    # the four sides of a boundary share one reference too
    _, bfiles, _ = _case_boundary(layouts["tmp"], layouts["grid"])
    watch = _Watch(monkeypatch, bfiles)
    names = build.add_boundary_sources(
        cat, "bry", bfiles, layouts["grid"], ref=layouts["tmp"] / "bry.parquet"
    )
    assert watch.builds == 1 and len(names) == 4


def test_an_unfinished_file_refuses_the_reference_instead_of_leaving_a_gap(
    layouts, monkeypatch
):
    _, files, kwargs = _case_co2(layouts["tmp"], layouts["grid"])
    with open(files[-1], "r+b") as f:  # a truncated last year
        f.truncate(files[-1].stat().st_size // 2)
    ref = layouts["tmp"] / "r.parquet"
    calls = []
    monkeypatch.setattr(build, "make_kerchunk", lambda *a, **k: calls.append(a))
    with pytest.raises(ValueError, match="cannot build the reference"):
        build.add_forcing_source(
            build.new_catalog(), "a", files, layouts["grid"], ref=ref, **kwargs
        )
    assert calls == []
    assert not Path(str(ref) + ".sources.json").exists()


def test_a_failed_build_leaves_no_sidecar(layouts, monkeypatch):
    _, files, kwargs = _case_co2(layouts["tmp"], layouts["grid"])
    ref = layouts["tmp"] / "r.parquet"
    side = Path(str(ref) + ".sources.json")
    real = build.make_kerchunk

    def broken(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(build, "make_kerchunk", broken)
    with pytest.raises(RuntimeError, match="disk full"):
        build.add_forcing_source(
            build.new_catalog(), "a", files, layouts["grid"], ref=ref, **kwargs
        )
    assert not side.exists() and not Path(str(side) + ".tmp").exists()
    # and a good sidecar does not outlive the rebuild that failed after it
    monkeypatch.setattr(build, "make_kerchunk", real)
    build.add_forcing_source(
        build.new_catalog(), "a", files, layouts["grid"], ref=ref, **kwargs
    )
    assert side.is_file()
    st = files[0].stat()
    os.utime(files[0], ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    monkeypatch.setattr(build, "make_kerchunk", broken)
    with pytest.raises(RuntimeError, match="disk full"):
        build.add_forcing_source(
            build.new_catalog(), "a", files, layouts["grid"], ref=ref, **kwargs
        )
    assert not side.exists()


def test_a_ref_over_remote_files_is_refused(layouts):
    with pytest.raises(ValueError, match="local forcing files"):
        build._forcing_reference(
            ["https://example.org/a.nc"],
            layouts["tmp"] / "r.parquet",
            concat_dim="time",
            loadable_variables=["time"],
        )
