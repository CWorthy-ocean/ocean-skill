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


def _boundary_file(path, days, offset=0.0, *, month=False, fmt=None):
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
    xr.Dataset(data, coords=coords).to_netcdf(
        path, format=fmt, **({"engine": "netcdf4"} if fmt else {})
    )


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


# ======================================================================================
# Monthly files of different lengths: subchunk= makes their time chunks equal
# ======================================================================================

#: roms-tools writes uncompressed netCDF-3 (CDF-5); one chunk per variable per file
NC3 = "NETCDF3_64BIT_DATA"
MONTH_DAYS = ([0, 1, 2], [31, 32, 33, 34], [59, 60])


def _uneven_months(tmp_path):
    """Three monthly boundary files with 3, 4 and 2 time steps (uncompressed CDF-5)."""
    files = []
    for i, days in enumerate(MONTH_DAYS):
        files.append(tmp_path / f"bry_2011{i + 1:02d}.nc")
        _boundary_file(files[-1], days, fmt=NC3)
    return files


def test_files_of_different_lengths_need_subchunk_and_the_error_says_so(layouts):
    files = _uneven_months(layouts["tmp"])
    ref = layouts["tmp"] / "bry.parquet"
    with pytest.raises(ValueError, match="inconsistent chunk shapes") as err:
        build.add_boundary_sources(
            build.new_catalog(), "bry", files, layouts["grid"], ref=ref
        )
    assert "subchunk: {bry_time: 1}" in str(err.value)
    assert not Path(str(ref) + ".sources.json").exists()


def test_subchunk_joins_files_of_different_lengths_and_reads_every_step(layouts):
    files = _uneven_months(layouts["tmp"])
    ref = layouts["tmp"] / "bry.parquet"
    cat = build.new_catalog()
    build.add_boundary_sources(
        cat,
        "bry",
        files,
        layouts["grid"],
        sides=["south"],
        ref=ref,
        subchunk={"bry_time": 1},
    )
    _save(layouts, cat)
    ds = sources.read("bry_south").load()
    days = np.concatenate(MONTH_DAYS)
    assert list(ds["time"].values.astype("datetime64[D]")) == list(
        np.datetime64(EPOCH) + days
    )
    expected = np.concatenate([xr.open_dataset(f)["temp_south"].values for f in files])
    got = ds[TEMP].squeeze("eta_rho").transpose("time", "s_rho", "xi_rho").values
    np.testing.assert_allclose(got, expected)


def test_changing_subchunk_rebuilds_the_ref_and_records_it(layouts, monkeypatch):
    import json

    files = _uneven_months(layouts["tmp"])
    ref = layouts["tmp"] / "bry.parquet"
    side = Path(str(ref) + ".sources.json")
    watch = _Watch(monkeypatch, files)

    def build_it(**kw):
        build.add_boundary_sources(
            build.new_catalog(),
            "bry",
            files,
            layouts["grid"],
            sides=["south"],
            ref=ref,
            **kw,
        )

    build_it(subchunk={"bry_time": 1})
    assert json.loads(side.read_text())["subchunk"] == {"bry_time": 1}
    build_it(subchunk={"bry_time": 1})
    assert watch.builds == 1, "the same subchunk must reuse the reference"
    build_it(subchunk={"bry_time": 1, "s_rho": 1})
    assert watch.builds == 2
    assert json.loads(side.read_text())["subchunk"] == {"bry_time": 1, "s_rho": 1}


def test_subchunk_without_a_ref_is_refused(layouts):
    files = _uneven_months(layouts["tmp"])
    with pytest.raises(ValueError, match="needs ref="):
        build.add_boundary_sources(
            build.new_catalog(),
            "bry",
            files,
            layouts["grid"],
            subchunk={"bry_time": 1},
        )
    with pytest.raises(ValueError, match="needs ref="):
        build.add_forcing_source(
            build.new_catalog(),
            "surf",
            files[0],
            layouts["grid"],
            subchunk={"time": 1},
        )


def test_refresh_forcing_subchunk_builds_a_set_that_otherwise_fails(
    isolated_catalogs, tmp_path, capsys
):
    from ocean_skill.workflows.run import _refresh_forcing

    grid, _, _ = _grid(tmp_path)
    _uneven_months(tmp_path)
    entry = {
        "name": "bry",
        "kind": "boundary",
        "files": str(tmp_path / "bry_*.nc"),
        "grid": str(grid),
        "sides": ["south"],
        "ref": str(tmp_path / "bry.parquet"),
    }
    catalog_path = isolated_catalogs / "run.yaml"
    _refresh_forcing([entry], catalog_path)
    out = capsys.readouterr().out
    assert "FAILED bry" in out and "subchunk: {bry_time: 1}" in out
    _refresh_forcing([{**entry, "subchunk": {"bry_time": 1}}], catalog_path)
    assert "reference rebuilt" in capsys.readouterr().out
    assert sources.read("bry_south").sizes["time"] == 9


def test_refresh_forcing_subchunk_needs_a_ref():
    from pydantic import ValidationError

    from ocean_skill.config import RefreshConfig

    with pytest.raises(ValidationError, match="subchunk: shapes the kerchunk"):
        RefreshConfig.model_validate(
            {
                "catalog": "c.yaml",
                "forcing": [
                    {
                        "name": "f",
                        "kind": "boundary",
                        "files": "*.nc",
                        "grid": "g.nc",
                        "subchunk": {"bry_time": 1},
                    }
                ],
            }
        )


# ======================================================================================
# kind: initial_conditions -- one model state that reads like restart output
# ======================================================================================


def _ic_grid(tmp_path, *, hc=True):
    """Write the grid file an entry merges in, with or without an ``hc`` attribute."""
    from tests import _suite_fixture as sf

    path = sf.write_forcing_grid(tmp_path / "ic_grid.nc")
    if not hc:
        ds = xr.open_dataset(path).load()
        ds.close()
        ds.attrs.pop("hc")
        path.unlink()
        ds.to_netcdf(path)
    return path


@pytest.fixture
def ic(isolated_catalogs, tmp_path):
    """Write an IC file as roms-tools 4.0.0 does (no ``hc``), dated 2010-01-01."""
    from tests import _suite_fixture as sf

    grid = _ic_grid(tmp_path)
    path = sf.write_initial_conditions(tmp_path / "ic.nc", "20100101")
    cat = build.new_catalog(title="ic")
    build.add_initial_conditions(cat, "ic_a", path, grid)
    ic_b = sf.write_initial_conditions(tmp_path / "ic_b.nc", "20100101", shift=0.5)
    build.add_initial_conditions(cat, "ic_b", ic_b, grid)
    build.save(cat, isolated_catalogs / "ic.yaml")
    return {"tmp": tmp_path, "grid": grid, "file": path}


def test_an_ic_entry_reads_like_a_restart(ic):
    ds = sources.read("ic_a")
    # model-output layout: one decoded time step, rho fields on (time, s_rho, eta, xi)
    assert ds[TEMP].dims == ("time", "s_rho", "eta_rho", "xi_rho")
    assert ds.sizes["time"] == 1
    assert {"z_rho", "lon", "lat", "mask_rho", "h", "sigma_r", "Cs_r"} <= set(ds.coords)
    assert {TEMP, SALT, NITRATE} <= set(ds.data_vars)
    # the BGC tracers keep their model names where the table maps them
    assert "mole_concentration_of_dissolved_inorganic_carbon_in_sea_water" in ds


def test_the_ic_time_decodes_to_the_files_ini_time(ic):
    ds = sources.read("ic_a")
    assert ds["time"].values.astype("datetime64[s]").tolist() == [
        np.datetime64("2010-01-01T00:00:00").tolist()
    ]
    meta = sources.resolve("ic_a").metadata
    assert (meta["time_coverage_start"], meta["time_coverage_end"]) == (
        "2010-01-01",
        "2010-01-01",
    )
    assert (meta["time_coord"], meta["time_dim"], meta["reference_date"]) == (
        "ocean_time",
        "time",
        "2000-01-01",
    )


def test_an_ic_time_that_disagrees_with_ini_time_is_refused(tmp_path):
    from tests import _suite_fixture as sf

    path = sf.write_initial_conditions(
        tmp_path / "ic.nc", "20100101", ini_time="2011-01-01 00:00:00"
    )
    with pytest.raises(ValueError, match="ini_time"):
        build.add_initial_conditions(
            build.new_catalog(), "ic", path, _ic_grid(tmp_path)
        )


def test_the_ic_vertical_takes_theta_from_the_file_and_hc_from_the_grid(ic):
    """roms-tools 4.0.0 writes theta_s/theta_b but no hc: hc is the grid file's."""
    from tests import _suite_fixture as sf

    meta = sources.resolve("ic_a").metadata
    assert meta["vertical"] == {
        "s_dim": "s_rho",
        "hc": sf.HC,
        "theta_s": sf.THETA_S,
        "theta_b": sf.THETA_B,
        "Vtransform": 2,
    }


def test_an_ic_hc_attribute_beats_the_grids_and_an_explicit_hc_beats_both(tmp_path):
    from tests import _suite_fixture as sf

    grid = _ic_grid(tmp_path)
    ds = xr.open_dataset(grid).load()
    ds.close()
    ds.attrs["hc"] = 77.0
    other = tmp_path / "other_grid.nc"
    ds.to_netcdf(other)
    path = sf.write_initial_conditions(tmp_path / "ic.nc", "20100101", hc_attr=True)
    cat = build.new_catalog()
    build.add_initial_conditions(cat, "from_file", path, other)
    assert cat["from_file"].metadata["vertical"]["hc"] == sf.HC
    build.add_initial_conditions(cat, "explicit", path, other, vertical={"hc": 33.0})
    assert cat["explicit"].metadata["vertical"]["hc"] == 33.0


def test_an_ic_with_no_hc_anywhere_needs_one_and_says_so(tmp_path):
    from tests import _suite_fixture as sf

    grid = _ic_grid(tmp_path, hc=False)
    path = sf.write_initial_conditions(tmp_path / "ic.nc", "20100101")
    with pytest.raises(ValueError, match=r"No 'hc'.*hc: on a refresh\.forcing entry"):
        build.add_initial_conditions(build.new_catalog(), "ic", path, grid)
    cat = build.new_catalog()
    build.add_initial_conditions(cat, "ic", path, grid, vertical={"hc": 100.0})
    assert cat["ic"].metadata["vertical"]["hc"] == 100.0


def test_an_ic_is_one_file_and_names_what_it_does_not_have(tmp_path):
    from tests import _suite_fixture as sf

    grid = _ic_grid(tmp_path)
    a = sf.write_initial_conditions(tmp_path / "a.nc", "20100101")
    b = sf.write_initial_conditions(tmp_path / "b.nc", "20100102")
    with pytest.raises(ValueError, match="one file, got 2"):
        build.add_initial_conditions(build.new_catalog(), "ic", [a, b], grid)
    with pytest.raises(ValueError, match="not variables of"):
        build.add_initial_conditions(
            build.new_catalog(), "ic", a, grid, variables=["temp", "nope"]
        )
    cat = build.new_catalog()
    build.add_initial_conditions(cat, "ic", a, grid, variables=["temp", "salt"])
    assert set(cat["ic"].metadata["variables"]) == {TEMP, SALT}


def test_an_ic_selects_a_depth_as_a_restart_does(ic):
    for depth in ("surface", 200):
        f = osk.field(source="ic_a", variable=TEMP, select={"depth": depth})
        assert {"eta_rho", "xi_rho"} <= set(f.data.dims) and "s_rho" not in f.data.dims
        assert np.isfinite(f.data.values).any()
    surface = osk.field(source="ic_a", variable=TEMP, select={"depth": "surface"})
    deep = osk.field(source="ic_a", variable=TEMP, select={"depth": 200})
    assert float(surface.data.mean()) > float(deep.data.mean())  # warm at the top


def test_two_ics_compare_at_a_depth(ic):
    result = osk.compare(
        test="ic_b",
        reference="ic_a",
        variables=[TEMP],
        select={"depth": 200},
    )
    (c,) = result.comparisons
    diff = c.aligned["difference"]
    assert np.isfinite(diff.values).any() and float(diff.mean()) > 0  # b is the warmer


def test_refresh_forcing_builds_an_ic_entry_and_takes_an_hc(
    isolated_catalogs, tmp_path, capsys
):
    from ocean_skill.workflows.run import _refresh_forcing
    from tests import _suite_fixture as sf

    grid = _ic_grid(tmp_path, hc=False)
    sf.write_initial_conditions(tmp_path / "ic.nc", "20100101")
    entry = {
        "name": "ic_run",
        "kind": "initial_conditions",
        "files": [str(tmp_path / "ic.nc")],
        "grid": str(grid),
        "climatology": False,  # the config's own default must not reach the builder
    }
    catalog_path = isolated_catalogs / "run.yaml"
    _refresh_forcing([entry], catalog_path)
    # no hc anywhere: said, not raised
    assert "FAILED ic_run" in capsys.readouterr().out
    _refresh_forcing([{**entry, "hc": 100.0}], catalog_path)
    assert "-> ic_run" in capsys.readouterr().out
    assert sources.read("ic_run").sizes["time"] == 1
    assert not sources.resolve("ic_run").metadata.get("climatology")


# ======================================================================================
# strip: -- the boundary sides joined into one section
# ======================================================================================


def _strip_catalog(
    isolated_catalogs, tmp_path, strip=("west", "south"), grid_path=None
):
    """Two dated runs (B = A + 1) and a monthly climatology, each with a strip."""
    grid, lon, lat = _grid(tmp_path)
    grid = grid_path or grid
    cat = build.new_catalog(title="strip")
    for run, offset in (("a", 0.0), ("b", 1.0)):
        f = tmp_path / f"{run}_bry.nc"
        _boundary_file(f, [0, 1, 2], offset)
        build.add_boundary_sources(cat, f"{run}_bry", f, grid, strip=list(strip))
    clim = tmp_path / "clim.nc"
    _boundary_file(clim, np.arange(12) * 30.4, 0.0, month=True)
    build.add_boundary_sources(
        cat, "c_bry", clim, grid, climatology=True, strip=list(strip)
    )
    build.save(cat, isolated_catalogs / "strip.yaml")
    return lon, lat


def _model_strip(lon, lat, sides):
    """Return the strip ``transect: {boundary: sides}`` draws from model output."""
    from ocean_skill.transect import boundary_strip

    ds = xr.Dataset(
        {"x": (("eta_rho", "xi_rho"), np.zeros(lon.shape))},
        coords={
            "lon": (("eta_rho", "xi_rho"), lon),
            "lat": (("eta_rho", "xi_rho"), lat),
        },
    )
    return boundary_strip(ds, list(sides))


def _entry_strip(source, select=None, variable=TEMP):
    f = osk.field(
        source=source,
        variable=variable,
        select={"transect": {"eta_rho": 0}, **(select or {})},
    )
    return f.data


@pytest.mark.parametrize(
    "sides",
    [
        ("west", "south"),
        ("south", "east"),
        ("east", "north"),
        ("north", "west", "south"),
        ("west",),
    ],
)
def test_a_strip_walks_the_sides_exactly_as_the_model_output_strip_does(
    isolated_catalogs, tmp_path, sides
):
    lon, lat = _strip_catalog(isolated_catalogs, tmp_path, strip=sides)
    want = _model_strip(lon, lat, sides)
    da = _entry_strip("a_bry_strip")
    # same cells in the same order, the corner(s) once, along in km
    np.testing.assert_allclose(da["path_lon"].values, want["path_lon"].values)
    np.testing.assert_allclose(da["path_lat"].values, want["path_lat"].values)
    np.testing.assert_allclose(da["along"].values, want["along"].values)
    assert da["along"].attrs["boundary_sides"] == ",".join(sides)
    np.testing.assert_allclose(
        da["along"].attrs["boundary_corner_km"],
        want["along"].attrs["boundary_corner_km"],
    )
    assert da.sizes["along"] == sum(
        {"west": NY, "east": NY, "south": NX, "north": NX}[s] for s in sides
    ) - (len(sides) - 1)


def test_the_west_south_strip_values_come_from_each_sides_own_file_data(
    isolated_catalogs, tmp_path
):
    """West run north to south, then south run west to east, the corner once.

    ``_boundary_file`` writes ``temp = index along the side + 0.1 per step + 0.01 per
    level``, so which side and which cell a column came from is readable off its value.
    """
    lon, lat = _strip_catalog(isolated_catalogs, tmp_path)
    da = _entry_strip("a_bry_strip", {"depth": [20.0]}).isel(time=0, z=0)
    # west is the eta axis (0 = south end): north to south counts down NY-1 ... 0, and
    # the south row then counts up 1 ... NX-1 -- the corner (index 0 of both) once
    ramp = np.concatenate([np.arange(NY)[::-1], np.arange(1, NX)])
    np.testing.assert_allclose(da.values, ramp + 0.02, atol=0.011)
    np.testing.assert_allclose(da["path_lon"].values[:NY], lon[::-1, 0])
    np.testing.assert_allclose(da["path_lon"].values[NY - 1 :], lon[0, :])
    np.testing.assert_allclose(da["path_lat"].values[: NY - 1 + 1], lat[::-1, 0])
    km = da["along"].attrs["boundary_corner_km"]
    assert len(km) == 1 and km[0] == pytest.approx(float(da["along"].values[NY - 1]))


def test_a_strip_entry_is_one_row_that_selects_a_section(isolated_catalogs, tmp_path):
    _strip_catalog(isolated_catalogs, tmp_path)
    ds = sources.read("a_bry_strip")
    assert ds[TEMP].dims == ("time", "s_rho", "eta_rho", "xi_rho")
    assert ds.sizes["eta_rho"] == 1 and ds.sizes["xi_rho"] == NY + NX - 1
    meta = sources.resolve("a_bry_strip").metadata
    assert meta["boundary_strip"] == ["west", "south"]
    da = _entry_strip("a_bry_strip", {"depth": [20.0, 100.0]})
    assert set(da.dims) == {"time", "along", "z"}


def test_the_strip_leaves_land_missing_like_the_sides(isolated_catalogs, tmp_path):
    grid, _, _ = _grid(tmp_path)
    ds = xr.open_dataset(grid).load()
    ds.close()
    ds["mask_rho"][2, 0] = 0.0  # a land cell on the west column, third from the south
    land = tmp_path / "land_grid.nc"
    ds.to_netcdf(land)
    _strip_catalog(isolated_catalogs, tmp_path, grid_path=land)
    da = _entry_strip("a_bry_strip", {"depth": [20.0]}).isel(time=0, z=0)
    # west runs north to south: eta index 2 is NY - 1 - 2 = 2 places in
    assert np.isnan(da.values[NY - 1 - 2])
    assert np.isfinite(np.delete(da.values, NY - 1 - 2)).all()


def test_two_runs_strips_compare_and_draw_in_both_renderers(
    isolated_catalogs, tmp_path
):
    import holoviews as hv

    _strip_catalog(isolated_catalogs, tmp_path)
    result = osk.compare(
        test="a_bry_strip",
        reference="b_bry_strip",
        variables=[TEMP],
        select={"transect": {"eta_rho": 0}, "depth": DEPTHS},
        aggregate={"time": "mean"},
    )
    (c,) = result.comparisons
    assert c.is_section and c.family == "section_row"
    assert c.aligned.sizes["along"] == NY + NX - 1
    np.testing.assert_allclose(c.aligned["difference"].values, -1.0)
    attrs = c.aligned["along"].attrs
    assert attrs["boundary_sides"] == "west,south"
    assert len(attrs["boundary_corner_km"]) == 1
    # the static renderer reads the corner off the section it draws ...
    from ocean_skill.plot.matplotlib_renderer import boundary_corners

    assert boundary_corners(c.aligned["test"], "along") == pytest.approx(
        attrs["boundary_corner_km"]
    )
    fig = result.plot()
    assert fig is not None
    # ... and the interactive one renders without error
    hv.render(result.plot(renderer="holoviews"))


def test_a_dated_strip_compares_with_a_climatology_strip(isolated_catalogs, tmp_path):
    _strip_catalog(isolated_catalogs, tmp_path)
    clim = sources.read("c_bry_strip")
    # the strip of a climatology keeps its month axis, as the side entries do
    assert clim[TEMP].dims == ("month", "s_rho", "eta_rho", "xi_rho")
    assert list(clim["month"].values) == list(range(1, 13))
    assert "time" not in clim.variables
    assert sources.resolve("c_bry_strip").metadata["climatology"] is True
    f = osk.field(
        source="c_bry_strip",
        variable=TEMP,
        select={"transect": {"eta_rho": 0}, "month": 3, "depth": DEPTHS},
    )
    assert "month" not in f.data.dims and f.data.sizes["along"] == NY + NX - 1
    result = osk.compare(
        test="a_bry_strip",
        reference="c_bry_strip",
        variables=[TEMP],
        select={"transect": {"eta_rho": 0}, "depth": DEPTHS},
        aggregate={"test": {"time": "mean"}, "reference": {"month": "mean"}},
    )
    (c,) = result.comparisons
    assert c.is_section and c.aligned.sizes["along"] == NY + NX - 1


def test_a_strip_of_sides_that_do_not_meet_is_refused(tmp_path):
    grid, _, _ = _grid(tmp_path)
    f = tmp_path / "bry.nc"
    _boundary_file(f, [0, 1, 2])
    with pytest.raises(ValueError, match="opposite sides"):
        build.add_boundary_sources(
            build.new_catalog(), "x", f, grid, strip=["west", "east"]
        )
    with pytest.raises(ValueError, match="not among"):
        build.add_boundary_sources(
            build.new_catalog(), "x", f, grid, sides=["south"], strip=["west", "south"]
        )


def test_a_strip_keeps_only_the_variables_every_side_has(tmp_path):
    grid, _, _ = _grid(tmp_path)
    f = tmp_path / "bry.nc"
    _boundary_file(f, [0, 1, 2])
    ds = xr.open_dataset(f).load()
    ds.close()
    f.unlink()
    ds.drop_vars("NO3_south").to_netcdf(f)  # the south side has no nitrate
    cat = build.new_catalog()
    with pytest.warns(UserWarning, match="leaves out.*NO3"):
        build.add_boundary_sources(cat, "x", f, grid, strip=["west", "south"])
    assert set(cat["x_strip"].metadata["variables"]) == {
        TEMP,
        SALT,
        "sea_surface_height_above_geoid",
    }
    assert NITRATE in cat["x_west"].metadata["variables"]


def test_refresh_forcing_builds_a_strip_entry(isolated_catalogs, tmp_path, capsys):
    from ocean_skill.workflows.run import _refresh_forcing

    grid, _, _ = _grid(tmp_path)
    _boundary_file(tmp_path / "bry.nc", [0, 1, 2])
    entry = {
        "name": "bry",
        "kind": "boundary",
        "files": [str(tmp_path / "bry.nc")],
        "grid": str(grid),
        "sides": ["west", "south"],
        "strip": ["west", "south"],
    }
    _refresh_forcing([entry], isolated_catalogs / "run.yaml")
    assert "-> bry_west, bry_south, bry_strip" in capsys.readouterr().out
    assert _entry_strip("bry_strip", {"depth": [20.0]}).sizes["along"] == NY + NX - 1


def test_a_label_reaches_the_entry_metadata_and_names_the_source(
    isolated_catalogs, tmp_path
):
    """``label=`` (a suite's ``refresh: ... label:``) is how figures name a source."""
    grid, _, _ = _grid(tmp_path)
    cat = build.new_catalog(title="labelled")
    f = tmp_path / "bry.nc"
    _boundary_file(f, [0, 1, 2])
    build.add_boundary_sources(cat, "run_bry", f, grid, label="boundary file")
    surf = tmp_path / "surf.nc"
    _surface_file(surf)
    build.add_forcing_source(cat, "run_surf", surf, grid, label="surface physics")
    build.save(cat, isolated_catalogs / "labelled.yaml")
    assert sources.resolve("run_bry_south").metadata["label"] == "boundary file"
    assert sources.resolve("run_surf").metadata["label"] == "surface physics"
    from ocean_skill.catalog import source_label

    assert source_label("run_bry_west") == "boundary file"
    assert source_label("run_surf") == "surface physics"


def test_refresh_forcing_label_is_how_figures_name_each_entry(
    isolated_catalogs, tmp_path
):
    from ocean_skill import catalog
    from ocean_skill.workflows.run import _refresh_forcing

    grid = _forcing_files(tmp_path)
    catalog_path = isolated_catalogs / "run.yaml"
    _refresh_forcing(
        [
            {
                "name": "bry",
                "kind": "boundary",
                "files": [str(tmp_path / "y2011/*.nc")],
                "grid": str(grid),
                "sides": ["south"],
                "variables": ["temp", "salt"],
                "label": "Boundary physics",
            },
            {
                "name": "surf",
                "kind": "surface",
                "files": str(tmp_path / "surface.nc"),
                "grid": str(grid),
                "time_coord": "time",
                "label": "Surface physics",
            },
        ],
        catalog_path,
    )
    assert catalog.source_label("bry_south") == "Boundary physics"
    assert catalog.source_label("surf") == "Surface physics"
