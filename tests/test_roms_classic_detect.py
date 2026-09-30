"""Detection of classic Rutgers ROMS output (:func:`ocean_skill.build._roms_metadata`).

``_roms_metadata`` used to recognize only UCLA-ROMS output (a ``sigma_r`` variable, the
vertical-grid scalars as global attributes). Classic Rutgers ROMS -- xroms' example
file, for one -- lays the same information out differently: the sigma values are the
``s_rho``/``s_w`` *coordinates* themselves, ``hc`` and ``Vtransform`` are 0-d data
variables, and ``ocean_time`` is a dimension coordinate. Left undetected, such a file
would skip :mod:`ocean_skill.roms` entirely -- no land mask, no depth coordinate --
which is a silent correctness bug since ROMS writes 0.0, not NaN, on land.

Synthetic throughout, like ``tests/test_roms_velocity.py``: no example file needed.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from ocean_skill import build

NS, NY, NX, NT = 4, 3, 5, 2


def _classic_roms(**overrides):
    """Build a small Dataset laid out like classic Rutgers ROMS output.

    ``overrides`` replace data variables by name; pass ``None`` to drop one.
    """
    rng = np.random.default_rng(0)
    s_rho = (np.arange(1, NS + 1) - NS - 0.5) / NS  # -0.875 .. -0.125
    s_w = np.linspace(-1.0, 0.0, NS + 1)
    lon1d = np.linspace(-150.0, -148.0, NX)
    lat1d = np.linspace(58.0, 60.0, NY)
    lon, lat = np.meshgrid(lon1d, lat1d)
    data = {
        "Cs_r": ("s_rho", np.linspace(-0.9, -0.1, NS)),
        "Cs_w": ("s_w", np.linspace(-1.0, 0.0, NS + 1)),
        "hc": ((), np.float64(5.0)),
        "Vtransform": ((), np.int32(1)),
        "h": (("eta_rho", "xi_rho"), np.linspace(10.0, 200.0, NY * NX).reshape(NY, NX)),
        "mask_rho": (("eta_rho", "xi_rho"), np.ones((NY, NX))),
        "pm": (("eta_rho", "xi_rho"), np.full((NY, NX), 1e-3)),
        "pn": (("eta_rho", "xi_rho"), np.full((NY, NX), 1e-3)),
        "angle": (("eta_rho", "xi_rho"), np.zeros((NY, NX))),
        "zeta": (("ocean_time", "eta_rho", "xi_rho"), rng.random((NT, NY, NX))),
        "temp": (
            ("ocean_time", "s_rho", "eta_rho", "xi_rho"),
            rng.random((NT, NS, NY, NX)),
        ),
        "salt": (
            ("ocean_time", "s_rho", "eta_rho", "xi_rho"),
            rng.random((NT, NS, NY, NX)),
        ),
    }
    for k, v in overrides.items():
        if v is None:
            data.pop(k, None)
        else:
            data[k] = v
    return xr.Dataset(
        data,
        coords={
            "s_rho": ("s_rho", s_rho),
            "s_w": ("s_w", s_w),
            "ocean_time": (
                "ocean_time",
                np.arange(NT) * 3600.0,
                {
                    "units": "seconds since 1970-01-01",
                    "long_name": "time since initialization",
                },
            ),
            "lon_rho": (("eta_rho", "xi_rho"), lon),
            "lat_rho": (("eta_rho", "xi_rho"), lat),
        },
    )


def _ucla_roms(**attrs):
    """Build a small Dataset laid out like UCLA-ROMS (roms-tools) output.

    ``sigma_r`` is a data variable, ``s_rho`` a bare dimension, and the vertical-grid
    scalars are global attributes.
    """
    rng = np.random.default_rng(0)
    sigma = (np.arange(1, NS + 1) - NS - 0.5) / NS
    ds = xr.Dataset(
        {
            "Cs_r": ("s_rho", np.linspace(-0.9, -0.1, NS)),
            "sigma_r": ("s_rho", sigma),
            "lon_rho": (("eta_rho", "xi_rho"), rng.random((NY, NX))),
            "temp": (
                ("time", "s_rho", "eta_rho", "xi_rho"),
                rng.random((NT, NS, NY, NX)),
            ),
            "ocean_time": (
                "time",
                np.arange(NT) * 3600.0,
                {"long_name": "time since 2012-01-01 00:00:00"},
            ),
        }
    )
    ds.attrs.update(attrs)
    return ds


def test_classic_rutgers_roms_is_detected_with_vertical_scalars_from_data_variables():
    """``hc``/``Vtransform`` are 0-d variables here, not global attrs; read them."""
    md = build._roms_metadata(_classic_roms())
    assert md["model"] == "roms"
    assert md["loader"] == "ocean_skill.roms"
    assert md["self_contained_grid"] is True
    assert md["vertical"] == {"s_dim": "s_rho", "hc": 5.0, "Vtransform": 1}


def test_classic_vertical_scalars_are_plain_python_types_for_yaml():
    """A numpy scalar in the catalog metadata would break ``to_yaml_file``."""
    vertical = build._roms_metadata(_classic_roms())["vertical"]
    assert type(vertical["hc"]) is float
    assert type(vertical["Vtransform"]) is int


def test_classic_time_coordinate_metadata_is_read_from_the_dimension_coordinate():
    """``ocean_time`` being its own dim coord must not change the time contract."""
    md = build._roms_metadata(_classic_roms())
    assert md["time_coord"] == "ocean_time"
    assert md["time_dim"] == "time"
    assert md["time_units"] == "seconds"
    assert md["reference_date"] == "1970-01-01"


def test_ucla_roms_with_sigma_r_and_global_attrs_is_still_detected_unchanged():
    """The pre-existing UCLA layout keeps its metadata, Vtransform 2 included."""
    md = build._roms_metadata(_ucla_roms(theta_s=5.0, theta_b=2.0, hc=300.0))
    assert md["model"] == "roms"
    assert md["vertical"] == {
        "s_dim": "s_rho",
        "theta_s": 5.0,
        "theta_b": 2.0,
        "hc": 300.0,
        "Vtransform": 2,
    }
    assert md["time_units"] == "seconds"
    assert md["reference_date"] == "2012-01-01"


def test_ucla_roms_without_a_since_clause_in_its_time_units_keeps_seconds():
    """The units guard only fires on ``<word> since ...``; older UCLA files lack it."""
    ds = _ucla_roms(theta_s=5.0, theta_b=2.0, hc=300.0)
    ds["ocean_time"].attrs["units"] = "second"
    assert build._roms_metadata(ds)["time_units"] == "seconds"


def test_ucla_roms_without_the_vertical_scalars_omits_the_vertical_block():
    """No ``hc`` anywhere means no ``vertical``; the adapter cannot build depth."""
    md = build._roms_metadata(_ucla_roms())
    assert md["model"] == "roms"
    assert "vertical" not in md


def test_classic_vertical_transform_2_data_variable_is_carried_through():
    """``Vtransform`` is read, not assumed: a Vtransform-2 classic file says 2."""
    ds = _classic_roms(Vtransform=((), np.int32(2)))
    assert build._roms_metadata(ds)["vertical"]["Vtransform"] == 2


def test_classic_vertical_transform_defaults_to_2_when_not_recorded_anywhere():
    """No attr and no variable: fall back to the UCLA-ROMS default of 2."""
    ds = _classic_roms(Vtransform=None)
    assert build._roms_metadata(ds)["vertical"] == {
        "s_dim": "s_rho",
        "hc": 5.0,
        "Vtransform": 2,
    }


def test_classic_theta_s_and_theta_b_are_recorded_only_when_present():
    """Classic files carry no theta_s/theta_b; if a file does, they come through."""
    ds = _classic_roms(theta_s=((), np.float64(7.0)), theta_b=((), np.float64(0.1)))
    vertical = build._roms_metadata(ds)["vertical"]
    assert vertical["theta_s"] == 7.0
    assert vertical["theta_b"] == 0.1
    assert type(vertical["theta_s"]) is float


def test_classic_without_hc_omits_the_vertical_block():
    """Without ``hc`` the depth coordinate cannot be built, so claim nothing."""
    md = build._roms_metadata(_classic_roms(hc=None))
    assert md["model"] == "roms"
    assert "vertical" not in md


def test_global_attribute_wins_over_a_data_variable_of_the_same_name():
    """Attributes are looked up first, so UCLA-style metadata is never shadowed."""
    ds = _classic_roms()
    ds.attrs["hc"] = 300.0
    assert build._roms_metadata(ds)["vertical"]["hc"] == 300.0


def test_time_units_are_taken_from_the_since_clause_so_a_days_file_fails_loudly():
    """``roms._decode_time`` raises on non-seconds units; record the real unit."""
    ds = _classic_roms()
    ds["ocean_time"].attrs["units"] = "days since 1858-11-17"
    md = build._roms_metadata(ds)
    assert md["time_units"] == "days"
    assert md["reference_date"] == "1858-11-17"


def test_time_units_without_a_since_clause_fall_back_to_seconds():
    """``units = "second"`` has no ``since``; today's "seconds" default stands."""
    ds = _classic_roms()
    ds["ocean_time"].attrs["units"] = "second"
    assert build._roms_metadata(ds)["time_units"] == "seconds"


def test_time_units_match_is_case_insensitive_and_lowercased():
    """Some writers capitalize the unit word; the recorded value is normalized."""
    ds = _classic_roms()
    ds["ocean_time"].attrs["units"] = "  Seconds Since 1970-01-01"
    assert build._roms_metadata(ds)["time_units"] == "seconds"


def test_no_cs_r_is_not_roms():
    """``Cs_r`` is required: a valued ``s_rho`` alone does not make a dataset ROMS."""
    assert build._roms_metadata(_classic_roms(Cs_r=None)) == {}


def test_cs_r_without_sigma_r_or_a_valued_s_rho_is_not_roms():
    """With ``Cs_r`` but no sigma values anywhere there is no depth to build."""
    ds = _classic_roms().drop_vars("s_rho")
    assert "s_rho" not in ds.variables
    assert build._roms_metadata(ds) == {}


def test_probe_drops_sigma_values_recorded_as_metres_for_a_classic_roms_source():
    """The generic probe reads a valued ``s_rho`` as "vertical", sigma as metres.

    For ROMS that is wrong (depth comes from z_rho at read time), and UCLA output --
    bare ``s_rho`` dim -- records none, so ROMS sources must look alike regardless of
    layout. ``vertical_levels`` is a true count and stays.
    """
    ds = _classic_roms()

    # Teeth: without the ROMS tell the generic probe does record sigma "metres".
    plain = build._probe(ds.drop_vars("Cs_r"), None)
    assert "geospatial_vertical_min" in plain

    md = build._probe(ds, None)
    assert md["model"] == "roms"
    for key in (
        "geospatial_vertical_min",
        "geospatial_vertical_max",
        "vertical_resolution_min",
        "vertical_resolution_max",
    ):
        assert key not in md
    assert md["vertical_levels"] == NS


def test_probe_of_a_ucla_roms_source_records_no_vertical_extent_either():
    """UCLA and classic ROMS now produce the same vertical-extent keys (none)."""
    md = build._probe(_ucla_roms(theta_s=5.0, theta_b=2.0, hc=300.0), None)
    assert md["model"] == "roms"
    assert "geospatial_vertical_min" not in md
    assert "geospatial_vertical_max" not in md
