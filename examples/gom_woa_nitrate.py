"""ROMS (GOM offline, Jan 2012) vs WOA23 January climatology — surface nitrate.

The first end-to-end comparison: read both through the catalogs, reduce to a comparable
2-D field, align (harmonize longitude -> regrid model onto the WOA grid), compute
area-weighted metrics, and draw the model | obs | difference row.

Run:  python examples/gom_woa_nitrate.py
"""

from pathlib import Path

import numpy as np

import ocean_skill as osk
from ocean_skill import align as osk_align
from ocean_skill import metrics as osk_metrics
from ocean_skill import roms
from ocean_skill.catalog import resolve
from ocean_skill.plot.matplotlib_renderer import field_row

OUT = Path("output/gom_woa")
STANDARD_NAME = "mole_concentration_of_nitrate_in_sea_water"
WOA_NAME = "moles_of_nitrate_per_unit_mass_in_sea_water"
RHO_SEAWATER = 1025.0  # kg m-3, for umol/kg -> mmol/m3

# --- model: surface nitrate, time-mean over January ---------------------------
model = osk.read("GOM_bgc")
meta = resolve("GOM_bgc").metadata
surf = roms.surface(model, meta)
m = surf[STANDARD_NAME].mean("time", keep_attrs=True)
m.attrs.setdefault("units", "mmol/m^3")
print(f"model  : {m.dims} {tuple(m.sizes.values())} units={m.attrs.get('units')}")

# --- observations: WOA January, surface level ---------------------------------
obs_ds = osk.read("woa23_nitrate_month01")
o = obs_ds[WOA_NAME].isel(time=0).isel(depth=0)
# WOA is umol/kg; ROMS is mmol/m3 -> convert obs to the model's units
o = o * (RHO_SEAWATER / 1000.0)
o.attrs["units"] = "mmol/m^3"
print(f"obs    : {o.dims} {tuple(o.sizes.values())} units={o.attrs['units']}")

# --- align: harmonize longitude, subset to the model box, regrid model -> WOA --
aligned = osk_align.align(
    m, o, method="bilinear", test_name="model", reference_name="obs"
)
print(
    f"aligned: {dict(aligned.sizes)}  ({aligned.attrs['regrid_method']}, "
    f"{aligned.attrs['lon_convention']})"
)

# --- metrics ------------------------------------------------------------------
rec = osk_metrics.compute(
    aligned,
    test_name="model",
    reference_name="obs",
    variable=STANDARD_NAME,
    test="GOM_bgc",
    reference="woa23_nitrate_month01",
    depth="surface",
    period="2012-01",
    units="mmol/m^3",
)
csv = osk_metrics.write([rec], OUT)
print(
    "metrics:",
    {
        k: (round(v, 4) if isinstance(v, float) else v)
        for k, v in rec.items()
        if k in osk_metrics.METRICS
    },
)
print("wrote  :", csv)

# --- plot ---------------------------------------------------------------------
fig = field_row(
    aligned,
    test_name="model",
    reference_name="obs",
    labels=("ROMS GOM (Jan 2012 mean)", "WOA23 (Jan climatology)"),
    title="Surface nitrate — model vs WOA23",
    units="mmol m$^{-3}$",
    standard_name=STANDARD_NAME,
    metrics=rec,
    domain=(
        float(np.nanmin(aligned["lon"])),
        float(np.nanmin(aligned["lat"])),
        float(np.nanmax(aligned["lon"])),
        float(np.nanmax(aligned["lat"])),
    ),
    save=OUT / "figures" / "surface_nitrate.png",
)
print("wrote  :", OUT / "figures" / "surface_nitrate.png")
