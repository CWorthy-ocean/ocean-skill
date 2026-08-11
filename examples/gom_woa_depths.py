"""ROMS (GOM offline, Jan 2012) vs WOA23 January climatology — nitrate at depth.

Exercises the vertical path: the model's s-coordinate fields are transformed onto fixed
depths with xgcm (``roms.to_depth``) and compared against the matching WOA depth level,
one ``model | obs | difference`` row per depth.

Run:  python examples/gom_woa_depths.py
"""

from pathlib import Path

import numpy as np

import ocean_skill as osk
from ocean_skill import align as osk_align
from ocean_skill import metrics as osk_metrics
from ocean_skill import roms
from ocean_skill.catalog import resolve
from ocean_skill.plot.matplotlib_renderer import field_grid

OUT = Path("output/gom_woa")
METHOD = "conservative_normed"
RHO_SEAWATER = 1025.0
STD_NAME = "mole_concentration_of_nitrate_in_sea_water"
WOA_ENTRY, WOA_VAR = (
    "woa23_nitrate_month01",
    "moles_of_nitrate_per_unit_mass_in_sea_water",
)
DEPTHS = [0.0, 50.0, 100.0, 300.0]  # metres

# --- model: one variable, time-mean, transformed onto fixed depths -------------
ds = osk.read("GOM_bgc")
meta = resolve("GOM_bgc").metadata
sub = ds[[STD_NAME]].mean("time", keep_attrs=True)
at_depth = roms.to_depth(sub, meta, DEPTHS)
print(f"model on depths: {dict(at_depth.sizes)}")

obs_all = osk.read(WOA_ENTRY)[WOA_VAR].isel(time=0)

comparisons, records = [], []
for i, d in enumerate(DEPTHS):
    # 0 m is the surface field (the top s-level): interpolating to exactly 0 m returns
    # NaN, since the shallowest model cell centre sits a few metres down.
    if d == 0:
        m = roms.surface(sub, meta)[STD_NAME]
    else:
        m = at_depth[STD_NAME].isel(z=i)
    m.attrs.setdefault("units", "mmol/m^3")

    # WOA's own level nearest the requested depth (its grid is 0,5,10,...)
    o = obs_all.sel(depth=d, method="nearest")
    actual = float(o["depth"])
    o = o * (RHO_SEAWATER / 1000.0)  # umol/kg -> mmol/m3
    o.attrs["units"] = "mmol/m^3"

    aligned = osk_align.align(
        m, o, method=METHOD, test_name="model", reference_name="obs"
    )
    rec = osk_metrics.compute(
        aligned,
        test_name="model",
        reference_name="obs",
        variable=STD_NAME,
        test="GOM_bgc",
        reference=WOA_ENTRY,
        depth=d,
        obs_depth=actual,
        period="2012-01",
        units="mmol/m^3",
        regrid=METHOD,
    )
    records.append(rec)
    comparisons.append(
        {
            "aligned": aligned,
            "metrics": rec,
            "units": "mmol m$^{-3}$",
            "standard_name": STD_NAME,
            "row_label": f"{d:g} m",
        }
    )
    finite = int(np.isfinite(aligned["model"]).sum())
    print(
        f"{d:6.0f} m (WOA {actual:5.0f} m)  bias={rec['bias']:+7.3f} "
        f"rmse={rec['rmse']:6.3f} corr={rec['corr']:+.3f} "
        f"sigma_ratio={rec['sigma_ratio']:.3f} n={rec['n']:4d} cells={finite}"
    )

csv = osk_metrics.write(records, OUT, stem="gom_woa_nitrate_depths")
print("wrote:", csv)

fig = field_grid(
    comparisons,
    test_name="model",
    reference_name="obs",
    labels=("ROMS GOM (Jan 2012 mean)", "WOA23 (Jan climatology)"),
    title="Nitrate vs depth — ROMS GOM vs WOA23 (Jan 2012)",
    save=OUT / "figures" / "nitrate_depths.png",
)
print("wrote:", OUT / "figures" / "nitrate_depths.png")
