"""A monthly climatology read as one twelve-step source, and grouped by month.

WOA ships twelve single-month files, catalogued as twelve ``monthNN`` entries. The
seasonal cycle's own statistics (``aggregate={"time": [{"groupby": "month", "reduce":
"mean"}, "var"]}``) need all twelve in one object, so ``build.add_source`` takes
the twelve files as one list-of-files entry and ``sources.read`` gives its undecoded
time a ``month`` coordinate to group by. Nothing here touches the network: the twelve
"months" are tiny local NetCDF files written in WOA's own shape (a one-step float
``time`` at the middle of the month, ``months since`` units).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import xarray as xr

from ocean_skill import build, operators, sources

#: population variance of 1..12: the seasonal variance of a field equal to its month
VAR_1_12 = 143.0 / 12.0

#: WOA23's decadal files count months from this epoch, a January
UNITS = "months since 1955-01-01 00:00:00"


def _month_file(path, number, *, units=UNITS, epoch_offset=396):
    """Write one WOA-shaped month: one ``time`` step at ``epoch_offset + number - 0.5``.

    The field's value is the month number, everywhere, so the variance across the
    twelve is analytic.
    """
    lat, lon = [10.0, 20.0], [100.0, 110.0, 120.0]
    ds = xr.Dataset(
        {
            "t_an": (
                ("time", "lat", "lon"),
                np.full((1, 2, 3), float(number), dtype="float32"),
                {"standard_name": "sea_water_temperature", "units": "degrees_celsius"},
            ),
        },
        coords={
            "time": (
                "time",
                np.array([epoch_offset + number - 0.5], dtype="float32"),
                {"units": units, "axis": "T", "standard_name": "time"},
            ),
            "lat": ("lat", lat, {"units": "degrees_north", "axis": "Y"}),
            "lon": ("lon", lon, {"units": "degrees_east", "axis": "X"}),
        },
    )
    ds.to_netcdf(path)
    return path


@pytest.fixture
def twelve_months(tmp_path):
    """Build twelve single-month entries, the way the shipped WOA catalog has them."""
    cat = build.new_catalog(title="WOA-like")
    for number in range(1, 13):
        path = _month_file(tmp_path / f"woa_m{number:02d}.nc", number)
        build.add_source(
            cat,
            f"toy_temperature_month{number:02d}",
            path,
            probe=False,
            climatology=True,
            climatology_period=f"month{number:02d}",
            woa_period_code=f"{number:02d}",
            featureType="grid",
            geospatial_lat_min=10.0,
            geospatial_lat_max=20.0,
            standard_names={"t_an": "sea_water_temperature"},
            variables=["sea_water_temperature"],
            institution="toy",
            title=f"toy temperature (month{number:02d})",
        )
    return cat


#: what a twelve-file climatology's reader takes: one step per file, stacked along
#: time, the grid read once from the first file
CONCAT = {
    "combine": "nested",
    "concat_dim": "time",
    "data_vars": "minimal",
    "coords": "minimal",
    "compat": "override",
}


def _add_monthly(cat, tmp_path):
    """Add ``toy_temperature_monthly``: the twelve month files as one entry."""
    build.add_source(
        cat,
        "toy_temperature_monthly",
        [tmp_path / f"woa_m{number:02d}.nc" for number in range(1, 13)],
        probe=False,
        reader_kwargs=CONCAT,
        climatology=True,
        climatology_period="monthly",
        featureType="grid",
        standard_names={"t_an": "sea_water_temperature"},
        variables=["sea_water_temperature"],
        title="toy temperature (monthly)",
    )


# ------------------------------------------------------------------ list of files


def test_add_source_takes_a_list_of_files_as_one_reader_in_order(
    twelve_months, tmp_path
):
    _add_monthly(twelve_months, tmp_path)
    reader = twelve_months["toy_temperature_monthly"]
    urls = reader.kwargs["args"][0].url
    assert isinstance(urls, list) and len(urls) == 12
    assert [u.rsplit("_m", 1)[1] for u in urls] == [f"{m:02d}.nc" for m in range(1, 13)]
    assert all(u.startswith("/") for u in urls), "local paths are stored absolute"
    assert reader.kwargs["combine"] == "nested"
    assert reader.kwargs["concat_dim"] == "time"
    assert reader.kwargs["decode_times"] is False


def test_add_source_refuses_an_empty_list(twelve_months):
    with pytest.raises(ValueError, match="empty list"):
        build.add_source(twelve_months, "x", [], probe=False)


def test_the_shipped_catalog_has_a_monthly_entry_per_variable_and_period():
    """Each ``woa23_<var>_monthly`` stacks that variable's own twelve month files."""
    import re
    from pathlib import Path

    import intake

    import ocean_skill

    path = Path(ocean_skill.__file__).parent / "catalogs" / "woa.yaml"
    cat = intake.from_yaml_file(str(path))
    monthly = [n for n in cat if n.endswith("_monthly")]
    assert len(monthly) == len([n for n in cat if n.endswith("_month01")]) > 0
    for name in monthly:
        variable = name.split("_")[1]
        reader = cat[name]
        urls = reader.kwargs["args"][0].url
        codes = [re.search(r"_(\w)(\d\d)_01\.nc$", u) for u in urls]
        assert len(urls) == 12 and all(codes)
        assert [int(c.group(2)) for c in codes] == list(range(1, 13)), name
        assert all(f"/{variable}/" in u for u in urls), name
        assert reader.metadata["climatology_period"] == "monthly"
        assert reader.metadata["title"] == f"WOA23 {variable} (monthly, 1 deg)"
        # the same urls the single months use
        for number, url in enumerate(urls, start=1):
            single = cat[f"woa23_{variable}_month{number:02d}"]
            assert single.kwargs["args"][0].url == url


# ---------------------------------------------------------------------------- read


@pytest.fixture
def monthly_read(isolated_catalogs, twelve_months, tmp_path):
    """Build ``toy_temperature_monthly``, save it, and read it back."""
    _add_monthly(twelve_months, tmp_path)
    build.save(twelve_months, isolated_catalogs / "toy_woa.yaml")
    return sources.read("toy_temperature_monthly")


def test_a_monthly_read_has_twelve_undecoded_steps_and_a_month_coordinate(
    monthly_read,
):
    ds = monthly_read
    assert ds.sizes["time"] == 12
    assert not np.issubdtype(ds["time"].dtype, np.datetime64), "left undecoded"
    assert list(ds["month"].values) == list(range(1, 13))
    assert ds["month"].dims == ("time",)
    # the grid is one copy, not twelve
    assert ds["sea_water_temperature"].dims == ("time", "lat", "lon")
    assert list(ds["sea_water_temperature"].isel(lat=0, lon=0).values) == list(
        range(1, 13)
    )


def test_the_seasonal_variance_chain_gives_the_analytic_answer(monthly_read):
    da = monthly_read["sea_water_temperature"]
    out = operators.aggregate(
        da, {"time": [{"groupby": "month", "reduce": "mean"}, "var"]}
    )
    assert out.dims == ("lat", "lon")
    np.testing.assert_allclose(out.values, VAR_1_12, rtol=1e-6)


def test_a_single_month_entry_is_left_without_a_month_coordinate(
    isolated_catalogs, twelve_months
):
    build.save(twelve_months, isolated_catalogs / "toy_woa.yaml")
    ds = sources.read("toy_temperature_month03")
    assert ds.sizes["time"] == 1
    assert "month" not in ds.coords


# ------------------------------------------------------------ month from the values


def _undecoded(values, *, units=UNITS, size=None):
    """Return an undecoded climatology with the given ``time`` values."""
    values = np.asarray(values, dtype=float)
    n = values.size if size is None else size
    return xr.Dataset(
        {"v": (("time", "x"), np.arange(n * 2.0).reshape(n, 2))},
        coords={"time": ("time", values, {"units": units, "axis": "T"})},
    )


META = {"climatology": True, "climatology_period": "monthly"}


def test_the_month_is_read_off_the_values_not_the_order_of_the_steps():
    """WOA stamps mid-month: 396.5 is January, 407.5 December, in any order."""
    values = 396 + np.arange(12) + 0.5
    shuffled = np.random.default_rng(0).permutation(values)
    ds = sources._with_month_coordinate(_undecoded(shuffled), META)
    expected = (np.floor(shuffled).astype(int) % 12) + 1
    assert list(ds["month"].values) == list(expected)
    assert sorted(ds["month"].values) == list(range(1, 13))
    # 396.5 -> January, 397.5 -> February, 407.5 -> December
    by_time = dict(zip(shuffled, ds["month"].values))
    assert (by_time[396.5], by_time[397.5], by_time[407.5]) == (1, 2, 12)


def test_the_epochs_own_month_is_honoured_not_assumed_to_be_january():
    """Counting from 1965-07-01, step 0.5 is July, step 6.5 is January."""
    values = np.arange(12) + 0.5
    ds = sources._with_month_coordinate(
        _undecoded(values, units="months since 1965-07-01"), META
    )
    assert list(ds["month"].values) == [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]


def test_an_undeclared_twelve_step_climatology_is_recognised_too():
    ds = sources._with_month_coordinate(
        _undecoded(396.5 + np.arange(12)), {"climatology": True}
    )
    assert list(ds["month"].values) == list(range(1, 13))


@pytest.mark.parametrize(
    "meta,values",
    [
        ({"climatology": True, "climatology_period": "month01"}, [396.5]),
        ({"climatology": True, "climatology_period": "annual"}, [6.5]),
        ({"climatology_period": "monthly"}, list(396.5 + np.arange(12))),
    ],
    ids=["one month", "annual", "not a climatology"],
)
def test_anything_but_a_twelve_step_climatology_is_left_alone(meta, values):
    ds = _undecoded(values)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = sources._with_month_coordinate(ds, meta)
    assert "month" not in out.coords


def test_a_declared_monthly_climatology_that_cannot_be_read_warns():
    twice = np.r_[396.5 + np.arange(11), 396.5]  # January twice: not twelve months
    with pytest.warns(UserWarning, match="twelve\\s+different months"):
        out = sources._with_month_coordinate(_undecoded(twice), META)
    assert "month" not in out.coords

    with pytest.warns(UserWarning, match="no 'month' coordinate"):
        out = sources._with_month_coordinate(
            _undecoded(np.arange(12.0), units="days since 1990-01-01"), META
        )
    assert "month" not in out.coords


def test_an_existing_month_coordinate_is_not_replaced():
    ds = _undecoded(396.5 + np.arange(12)).assign_coords(month=("time", np.arange(12)))
    out = sources._with_month_coordinate(ds, META)
    assert list(out["month"].values) == list(range(12))
