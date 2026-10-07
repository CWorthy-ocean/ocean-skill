"""A grid carried inside a ROMS store survives the read.

ROMS output files often carry their own grid (``h``, ``mask_rho``, ``lon_rho``,
``Cs_r``, ``hc``, ...). Stacked into one kerchunk store, virtualizarr's concat defaults
(``data_vars="all"``, ``coords="different"``) gave every one of those *static* variables
the record dimension: ``h`` came out ``(ocean_time, eta_rho, xi_rho)``, ``hc``
``(ocean_time,)``, and a classic file's valued ``s_rho`` went 2-D -- which stopped
:func:`ocean_skill.build._roms_metadata` recognising the store as ROMS at all. With no
``model: roms`` the entry never reached :mod:`ocean_skill.roms` on read, so the grid it
carried was dropped (no ``lon``/``lat``, no land mask, no depth coordinate) and a user
had to hand ``grid=`` a separate file to get one.

Three layers, each held on its own and then together:

* :func:`ocean_skill.build.make_kerchunk` keeps a variable without the concat dimension
  static (the cause);
* ``_roms_metadata`` still recognises a store built the old way (existing stores);
* :func:`ocean_skill.roms.standardize` reads the grid from the first record of any grid
  field that still carries the record dimension (existing stores, read time).

Synthetic throughout. The stores are JSON references (not parquet: the parquet reference
reader has a known fsspec flake).
"""

from __future__ import annotations

import intake
import numpy as np
import pytest
import xarray as xr

import ocean_skill as osk
from ocean_skill import build, roms

NS, NY, NX = 4, 3, 5
HC = 5.0
LAND = (0, 0)  # (eta, xi) cell with mask_rho == 0
# The grid fields a ROMS file carries, by layout (the classic ones plus the metrics).
STATIC = ("h", "mask_rho", "angle", "pm", "pn", "Cs_r", "Cs_w", "hc", "Vtransform")
CLASSIC_STATIC = (*STATIC, "lon_rho", "lat_rho", "s_rho", "s_w")
FREE_SURFACE = "sea_surface_height_above_geoid"


def _grid_arrays() -> dict[str, np.ndarray]:
    """Return the numbers every file's grid carries (the same in all of them)."""
    s_w = np.linspace(-1.0, 0.0, NS + 1)
    lon, lat = np.meshgrid(np.linspace(-150.0, -148.0, NX), np.linspace(58.0, 60.0, NY))
    mask = np.ones((NY, NX))
    mask[LAND] = 0.0
    return {
        "s_w": s_w,
        "s_rho": 0.5 * (s_w[:-1] + s_w[1:]),
        "cs_r": np.linspace(-0.9, -0.1, NS),
        "cs_w": np.linspace(-1.0, 0.0, NS + 1),
        "h": np.linspace(20.0, 200.0, NY * NX).reshape(NY, NX),
        "mask": mask,
        "lon": lon,
        "lat": lat,
    }


def _zeta_of_step(step: int) -> float:
    """Free-surface height of global time step ``step``: distinct at every step."""
    return 0.5 + 0.25 * step


def _temp_of_steps(first_step: int, nt: int) -> np.ndarray:
    """Temperature ``(time, s_rho, eta, xi)``, a function of the global step alone.

    So a record two files both hold (a restart repeating one) is identical in each.
    """
    return np.stack(
        [
            1.0 + np.random.default_rng(first_step + i).random((NS, NY, NX))
            for i in range(nt)
        ]
    )


def _classic_part(first_step: int, nt: int = 2) -> xr.Dataset:
    """One classic-layout ROMS history file that carries its grid.

    ``ocean_time`` is its own dimension, ``hc``/``Vtransform`` (1) are 0-d variables and
    ``s_rho`` is a valued coordinate. ``zeta`` is uniform but different at every time
    step, so a dropped or reordered record shows in the depth coordinate.
    """
    g = _grid_arrays()
    rho2 = ("eta_rho", "xi_rho")
    zeta = np.array([_zeta_of_step(first_step + i) for i in range(nt)])
    return xr.Dataset(
        {
            "Cs_r": ("s_rho", g["cs_r"]),
            "Cs_w": ("s_w", g["cs_w"]),
            "hc": ((), np.float64(HC)),
            "Vtransform": ((), np.int32(1)),
            "h": (rho2, g["h"]),
            "mask_rho": (rho2, g["mask"]),
            "angle": (rho2, np.zeros((NY, NX))),
            "pm": (rho2, np.full((NY, NX), 1e-3)),
            "pn": (rho2, np.full((NY, NX), 1e-3)),
            "zeta": (
                ("ocean_time", *rho2),
                np.broadcast_to(zeta[:, None, None], (nt, NY, NX)).copy(),
            ),
            "temp": (
                ("ocean_time", "s_rho", *rho2),
                _temp_of_steps(first_step, nt),
            ),
        },
        coords={
            "s_rho": ("s_rho", g["s_rho"]),
            "s_w": ("s_w", g["s_w"]),
            "ocean_time": (
                "ocean_time",
                (first_step + np.arange(nt)) * 3600.0,
                {"units": "seconds since 1970-01-01"},
            ),
            "lon_rho": (rho2, g["lon"]),
            "lat_rho": (rho2, g["lat"]),
        },
    )


def _ucla_part(first_step: int, nt: int = 2) -> xr.Dataset:
    """One UCLA-layout ROMS file that carries its grid: ``sigma_r``, bare ``s_rho``.

    The record dimension is called ``time`` (``ocean_time`` is a variable on it) and the
    vertical-grid scalars are global attributes, Vtransform 2.
    """
    g = _grid_arrays()
    rho2 = ("eta_rho", "xi_rho")
    zeta = np.array([_zeta_of_step(first_step + i) for i in range(nt)])
    ds = xr.Dataset(
        {
            "Cs_r": ("s_rho", g["cs_r"]),
            "Cs_w": ("s_w", g["cs_w"]),
            "sigma_r": ("s_rho", g["s_rho"]),
            "sigma_w": ("s_w", g["s_w"]),
            "h": (rho2, g["h"]),
            "mask_rho": (rho2, g["mask"]),
            "angle": (rho2, np.zeros((NY, NX))),
            "pm": (rho2, np.full((NY, NX), 1e-3)),
            "pn": (rho2, np.full((NY, NX), 1e-3)),
            "lon_rho": (rho2, g["lon"]),
            "lat_rho": (rho2, g["lat"]),
            "ocean_time": (
                "time",
                (first_step + np.arange(nt)) * 3600.0,
                {"long_name": "time since 2012-01-01 00:00:00"},
            ),
            "zeta": (
                ("time", *rho2),
                np.broadcast_to(zeta[:, None, None], (nt, NY, NX)).copy(),
            ),
            "temp": (("time", "s_rho", *rho2), _temp_of_steps(first_step, nt)),
        }
    )
    ds.attrs.update(hc=HC, Vtransform=2, theta_s=5.0, theta_b=1.0)
    return ds


def _time_expanded(parts: list[xr.Dataset], dim: str) -> xr.Dataset:
    """Concatenate ``parts`` the way virtualizarr's defaults did: statics stacked too.

    Every variable present in every part gains ``dim`` -- ``h`` becomes ``(dim, eta_rho,
    xi_rho)``, ``hc`` ``(dim,)`` -- and a classic file's ``s_rho``/``s_w`` index
    coordinates turn into 2-D ``(dim, s_rho)`` variables, as in a store opened from such
    a reference (xarray accepts them when they come in as ``Variable`` objects).
    """
    ds = xr.concat(parts, dim=dim, data_vars="all", coords="all")
    n = ds.sizes[dim]
    for label in ("s_rho", "s_w"):
        if label in ds.coords and ds[label].dims == (label,):
            vals = ds[label].values
            ds = ds.drop_vars(label).assign_coords(
                {label: xr.Variable((dim, label), np.tile(vals, (n, 1)))}
            )
    return ds


def _expected_z_rho(h, zeta_by_step, vtransform):
    """Depths ``(time, s_rho, eta, xi)`` from the raw numbers alone (no roms.py)."""
    g = _grid_arrays()
    sigma = g["s_rho"][None, :, None, None]
    cs = g["cs_r"][None, :, None, None]
    zeta = np.asarray(zeta_by_step)[:, None, None, None]
    h = h[None, None]
    if vtransform == 1:
        z0 = HC * (sigma - cs) + cs * h
        return z0 + zeta * (1 + z0 / h)
    s = (HC * sigma + h * cs) / (HC + h)
    return zeta + (zeta + h) * s


def _check_standardized(out: xr.Dataset, n_steps: int, vtransform: int):
    """Assert what a correctly read ROMS store looks like, whatever its layout."""
    g = _grid_arrays()
    ocean = g["mask"] == 1

    # positions are the 2-D grid, not a record-by-grid stack
    assert out["lon"].dims == ("eta_rho", "xi_rho")
    assert out["lat"].dims == ("eta_rho", "xi_rho")
    np.testing.assert_allclose(out["lon"].values, g["lon"])
    # static grid fields are static
    for name in ("h", "mask_rho", "angle", "Cs_r", "sigma_r"):
        assert "time" not in out[name].dims, name
    assert out["h"].dims == ("eta_rho", "xi_rho")
    np.testing.assert_allclose(out["h"].values, g["h"])
    assert out["sigma_r"].dims == ("s_rho",)
    # the time axis is the records, in order
    assert out.sizes["time"] == n_steps

    # the free surface is there, per step, land-masked
    zeta = out[FREE_SURFACE]
    assert zeta.dims == ("time", "eta_rho", "xi_rho")
    np.testing.assert_allclose(
        zeta.values[:, ocean][:, 0], [_zeta_of_step(i) for i in range(n_steps)]
    )
    assert np.isnan(zeta.values[:, ~ocean]).all()

    # land is masked in the data, the ocean is not
    temp = out["sea_water_potential_temperature"]
    assert np.isnan(temp.values[..., ~ocean]).all()
    assert np.isfinite(temp.values[..., ocean]).all()

    # and depth rides on that free surface
    expected = _expected_z_rho(
        g["h"], [_zeta_of_step(i) for i in range(n_steps)], vtransform
    )
    assert out["z_rho"].dims == ("time", "s_rho", "eta_rho", "xi_rho")
    np.testing.assert_allclose(
        out["z_rho"].values[..., ocean], expected[..., ocean], rtol=1e-12
    )


# ----------------------------------------------------------------------- make_kerchunk


def _write(tmp_path, parts):
    """Write each part as ``his_<i>.nc``; return the paths."""
    paths = []
    for i, part in enumerate(parts):
        path = tmp_path / f"his_{i}.nc"
        part.to_netcdf(path)
        paths.append(path)
    return paths


def _open_reference(path) -> xr.Dataset:
    return xr.open_dataset(str(path), engine="kerchunk", chunks={}, decode_times=False)


def test_make_kerchunk_keeps_the_grid_each_file_carries_static(tmp_path):
    """Variables without the record dimension stay without it in the store.

    Two files, each with ``h``/``mask_rho``/``lon_rho``/``Cs_r``/``hc``/... and a free
    surface and temperature over ``ocean_time``: only those two stack.
    """
    paths = _write(tmp_path, [_classic_part(0), _classic_part(2)])

    store = _open_reference(build.make_kerchunk(paths, out=tmp_path / "refs.json"))

    assert store.sizes["ocean_time"] == 4
    for name in CLASSIC_STATIC:
        assert "ocean_time" not in store[name].dims, f"{name} gained the record dim"
    assert store["h"].dims == ("eta_rho", "xi_rho")
    assert store["hc"].dims == ()
    assert float(store["hc"]) == HC  # a scalar reads back as itself, not as its fill
    assert int(store["Vtransform"]) == 1
    assert store["s_rho"].dims == ("s_rho",)
    assert store["zeta"].dims == ("ocean_time", "eta_rho", "xi_rho")
    assert store["temp"].dims == ("ocean_time", "s_rho", "eta_rho", "xi_rho")
    np.testing.assert_allclose(store["h"].values, _grid_arrays()["h"])


def test_dedup_of_a_repeated_timestamp_does_not_stack_the_grid_either(tmp_path):
    """The collapse of an overlapping record rebuilds the store -- statics stay static.

    The second file repeats the first one's last record (a restart), so
    ``keep="last"`` re-concatenates the surviving records.
    """
    first, second = _classic_part(0), _classic_part(1)  # steps 0,1 then 1,2
    paths = _write(tmp_path, [first, second])

    with pytest.warns(UserWarning, match="repeated a timestamp"):
        store = _open_reference(build.make_kerchunk(paths, out=tmp_path / "refs.json"))

    assert store.sizes["ocean_time"] == 3
    for name in CLASSIC_STATIC:
        assert "ocean_time" not in store[name].dims, f"{name} gained the record dim"


def test_any_static_variable_stays_static_not_just_a_roms_grid(tmp_path):
    """The rule is the concat dimension's, not ROMS': ``area`` stays ``(lat, lon)``."""
    paths = []
    for i in range(2):
        path = tmp_path / f"day{i}.nc"
        xr.Dataset(
            {
                "chlor_a": (("time", "lat", "lon"), np.full((1, 3, 4), float(i))),
                "area": (("lat", "lon"), np.full((3, 4), 7.0)),
            },
            coords={
                "time": [np.datetime64("2012-01-01") + np.timedelta64(i, "D")],
                "lat": np.linspace(31, 18, 3),
                "lon": np.linspace(-98, -80, 4),
            },
        ).to_netcdf(path)
        paths.append(path)

    store = _open_reference(build.make_kerchunk(paths, out=tmp_path / "refs.json"))

    assert store["area"].dims == ("lat", "lon")
    assert store["chlor_a"].dims == ("time", "lat", "lon")
    assert store.sizes["time"] == 2
    assert float(store["chlor_a"].isel(time=1, lat=0, lon=0)) == 1.0
    assert float(store["area"].isel(lat=0, lon=0)) == 7.0


@pytest.mark.parametrize("suffix", [".json", ".parquet"])
def test_a_scalar_variable_reads_back_as_itself_from_either_reference_format(
    tmp_path, suffix
):
    """0-d variables (``hc``, ``Vtransform``, ...) are written where a reader looks.

    virtualizarr files a scalar's one chunk under ``hc/`` and zarr reads ``hc/0``: a
    JSON reference gave back the fill value (NaN) and a parquet one could not be opened
    at all. Keeping statics static makes every ROMS scalar 0-d, so this is what a
    classic file's vertical-grid parameters stand on.
    """
    paths = _write(tmp_path, [_classic_part(0), _classic_part(2)])

    store = _open_reference(build.make_kerchunk(paths, out=tmp_path / f"refs{suffix}"))

    assert store["hc"].dims == ()
    assert float(store["hc"]) == HC
    assert int(store["Vtransform"]) == 1
    np.testing.assert_allclose(store["Cs_r"].values, _grid_arrays()["cs_r"])


def test_a_scalar_in_a_merged_grid_file_reads_back_as_itself(tmp_path):
    """The ``grid=`` merge brings its scalars (a grid's ``xl``/``el``) in 0-d too."""
    grid = tmp_path / "grid.nc"
    xr.Dataset(
        {"xl": ((), np.float64(1234.5)), "el": ((), np.float64(678.0))}
    ).to_netcdf(grid)
    paths = _write(tmp_path, [_classic_part(0), _classic_part(2)])

    store = _open_reference(
        build.make_kerchunk(paths, out=tmp_path / "refs.json", grid=grid)
    )

    assert float(store["xl"]) == 1234.5
    assert float(store["el"]) == 678.0


def test_make_kerchunk_rejects_a_reference_format_it_cannot_write(tmp_path):
    """The format check ``to_kerchunk`` made is still made."""
    paths = _write(tmp_path, [_classic_part(0)])

    with pytest.raises(ValueError, match="Unrecognized output format"):
        build.make_kerchunk(paths, out=tmp_path / "refs.json", fmt="xml")


# --------------------------------------------------------------- catalog -> read, whole


def test_a_kerchunked_run_that_carries_its_grid_reads_with_the_grid(
    tmp_path, isolated_catalogs
):
    """``make_kerchunk`` -> catalog -> ``osk.read``, no ``grid=`` anywhere.

    The store is recognised as ROMS with its grid inside (``self_contained_grid``), and
    reading it gives the 2-D lon/lat, the static ``h`` and mask, the free surface, the
    land-masked temperature and a depth coordinate on the right free surface per step.
    """
    paths = _write(tmp_path, [_classic_part(0), _classic_part(2)])
    refs = build.make_kerchunk(paths, out=tmp_path / "refs.json")
    out = build.build_catalog(
        {"run": str(refs)}, isolated_catalogs / "carried.yaml", title="carried grid"
    )

    md = intake.from_yaml_file(str(out))["run"].metadata
    assert md["model"] == "roms"
    assert md["self_contained_grid"] is True
    assert md["vertical"] == {"s_dim": "s_rho", "hc": HC, "Vtransform": 1}
    assert "grid" not in md

    ds = osk.read("carried:run")

    assert ds.attrs["ocean_skill_model"] == "roms"
    _check_standardized(ds, n_steps=4, vtransform=1)


# ------------------------------------------ stores built the old way (time-expanded)


@pytest.mark.parametrize("chunked", [False, True], ids=["numpy", "dask"])
def test_a_time_expanded_store_is_still_recognised_as_roms(chunked):
    """Stacked ``hc``/``Vtransform``/``s_rho`` no longer hide the ROMS tell.

    The detection wants ``Cs_r`` plus the sigma values (a 1-D ``s_rho`` there) and
    reads ``hc``/``Vtransform`` as 0-d; in a store that stacked them they are
    ``(ocean_time,)`` and ``(ocean_time, s_rho)``.
    """
    stacked = _time_expanded([_classic_part(0), _classic_part(2)], "ocean_time")
    assert stacked["hc"].dims == ("ocean_time",)  # the fixture is what it claims
    if chunked:  # as a store opened from a reference is: dask has no ``.item()``
        stacked = stacked.chunk()

    md = build._roms_metadata(stacked)

    assert md["model"] == "roms"
    assert md["loader"] == "ocean_skill.roms"
    assert md["self_contained_grid"] is True
    assert md["vertical"] == {"s_dim": "s_rho", "hc": HC, "Vtransform": 1}
    assert type(md["vertical"]["hc"]) is float
    assert type(md["vertical"]["Vtransform"]) is int
    assert md["time_coord"] == "ocean_time"


def _meta_of(stacked: xr.Dataset, **override) -> dict:
    """Return the catalog entry a probe of ``stacked`` writes (free surface renamed)."""
    return {
        **build._roms_metadata(stacked),
        "standard_names": {
            "zeta": FREE_SURFACE,
            "temp": "sea_water_potential_temperature",
        },
        **override,
    }


@pytest.mark.parametrize("chunked", [False, True], ids=["numpy", "dask"])
def test_standardize_reads_the_grid_from_a_classic_store_that_stacked_it(chunked):
    """Every grid field that carries ``ocean_time`` is read at its first record."""
    stacked = _time_expanded([_classic_part(0), _classic_part(2)], "ocean_time")
    assert stacked["h"].dims == ("ocean_time", "eta_rho", "xi_rho")
    assert stacked["s_rho"].dims == ("ocean_time", "s_rho")
    if chunked:
        stacked = stacked.chunk()

    out = roms.standardize(stacked, _meta_of(stacked))

    _check_standardized(out, n_steps=4, vtransform=1)


def test_standardize_takes_hc_and_vtransform_from_a_stacked_store_too():
    """With no ``vertical`` block, ``hc``/``Vtransform`` come from the data itself.

    The fallback reads the file's own variables, which a stacking store made
    ``(ocean_time,)`` arrays -- ``float()`` of those is an error, not a number.
    """
    stacked = _time_expanded([_classic_part(0), _classic_part(2)], "ocean_time")
    meta = _meta_of(stacked)
    meta.pop("vertical")

    out = roms.standardize(stacked, meta)

    _check_standardized(out, n_steps=4, vtransform=1)


def test_standardize_reads_the_grid_from_a_ucla_store_that_stacked_it():
    """A UCLA file's record dimension is ``time``: ``sigma_r``/``h``/... stack on it."""
    stacked = _time_expanded([_ucla_part(0), _ucla_part(2)], "time")
    assert stacked["sigma_r"].dims == ("time", "s_rho")
    assert stacked["h"].dims == ("time", "eta_rho", "xi_rho")

    meta = _meta_of(stacked)
    assert meta["vertical"]["Vtransform"] == 2

    out = roms.standardize(stacked, meta)

    _check_standardized(out, n_steps=4, vtransform=2)


def _legacy_reference(paths, out):
    """Write the reference ``make_kerchunk`` used to: virtualizarr's concat defaults.

    Every variable present in every file is stacked along the record dimension (the call
    is what ``make_kerchunk`` made before it kept statics static), so this is a store as
    users built them -- and as they exist on disk now.
    """
    from obspec_utils.registry import ObjectStoreRegistry
    from virtualizarr import open_virtual_mfdataset

    urls = dict(build._store_for(p) for p in paths)
    concat_dim, loadable = build.detect_concat(paths[0])
    vds = open_virtual_mfdataset(
        urls=list(urls),
        registry=ObjectStoreRegistry(urls),
        parser=build._parser_for(paths[0]),
        combine="nested",
        concat_dim=concat_dim,
        loadable_variables=list(loadable),
    )
    vds.vz.to_kerchunk(str(out), format="json")
    return out


@pytest.mark.parametrize("layout", ["classic", "ucla"])
def test_a_store_built_by_the_old_make_kerchunk_reads_with_its_grid(
    layout, tmp_path, isolated_catalogs
):
    """The stores already on disk: stacked grid, real dask-backed read, whole pipeline.

    Catalogued and read through the public path, not handed to ``standardize`` by hand:
    recognised as ROMS, grid and all, and read exactly as a store built the new way is.
    """
    part, vtransform = {
        "classic": (_classic_part, 1),
        "ucla": (_ucla_part, 2),
    }[layout]
    paths = _write(tmp_path, [part(0), part(2)])
    refs = _legacy_reference(paths, tmp_path / "legacy.json")
    assert _open_reference(refs)["h"].dims[0] in ("ocean_time", "time")  # stacked
    out = build.build_catalog(
        {"run": str(refs)}, isolated_catalogs / "legacy.yaml", title="legacy"
    )

    md = intake.from_yaml_file(str(out))["run"].metadata
    assert md["model"] == "roms"
    assert md["self_contained_grid"] is True
    assert md["vertical"]["Vtransform"] == vtransform

    _check_standardized(osk.read("legacy:run"), n_steps=4, vtransform=vtransform)


def test_a_clean_store_is_read_as_before():
    """Nothing to reduce: a store whose grid is already static is untouched by it."""
    clean = xr.concat(
        [_classic_part(0), _classic_part(2)],
        dim="ocean_time",
        data_vars="minimal",
        coords="minimal",
        compat="override",
    )
    assert clean["h"].dims == ("eta_rho", "xi_rho")

    out = roms.standardize(clean, _meta_of(clean))

    _check_standardized(out, n_steps=4, vtransform=1)


def test_a_genuinely_time_varying_wetdry_mask_is_not_reduced():
    """``wetdry_mask_rho`` is written along the record dimension by ROMS: it is data.

    Reading the grid at its first record applies to the fields ROMS holds static, not to
    this one -- it keeps every record.
    """
    parts = []
    for first in (0, 2):
        part = _classic_part(first)
        flags = np.zeros((2, NY, NX))
        flags[:, 1, 1] = [first, first + 1]  # differs record to record
        parts.append(
            part.assign(wetdry_mask_rho=(("ocean_time", "eta_rho", "xi_rho"), flags))
        )
    stacked = _time_expanded(parts, "ocean_time")

    out = roms.standardize(stacked, _meta_of(stacked))

    wet = out["wetdry_mask_rho"]
    assert wet.dims == ("time", "eta_rho", "xi_rho")
    np.testing.assert_array_equal(wet.values[:, 1, 1], [0, 1, 2, 3])


def test_a_store_without_any_record_dimension_on_its_grid_is_left_alone():
    """A single file (no record-stacked grid) is the common case and must not change."""
    single = _classic_part(0)

    out = roms.standardize(single, _meta_of(single))

    _check_standardized(out, n_steps=2, vtransform=1)
