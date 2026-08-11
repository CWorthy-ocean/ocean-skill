"""ROMS (GOM offline, Jan 2012) vs GLODAPv2.2016b — DIC and alkalinity at depth.

The carbonate counterpart to the WOA nutrient comparison, and the case the example run
actually targets (it is a carbonate/CDR configuration). GLODAP is an *annual* mapped
climatology, so this compares a January model mean against an annual mean — a real
caveat for the surface, less so at depth.

Two GLODAP quirks handled here: ``depth_surface`` is an index (0-32) with the true
depths in the ``Depth`` variable, and longitudes run 20.5 -> 379.5 (past 360), which the
alignment step harmonizes.

Run:  python examples/gom_glodap_carbon.py
"""

from pathlib import Path

import numpy as np

import ocean_skill as osk
from ocean_skill import align as osk_align
from ocean_skill import metrics as osk_metrics
from ocean_skill import roms
from ocean_skill.catalog import resolve
from ocean_skill.plot.matplotlib_renderer import field_grid

OUT = Path("output/gom_glodap")
METHOD = "conservative_normed"
RHO_SEAWATER = 1025.0  # kg m-3, for umol/kg -> mmol/m3

VARIABLES = [
    ("sea_water_alkalinity_expressed_as_mole_equivalent", "ALK"),
    ("mole_concentration_of_dissolved_inorganic_carbon_in_sea_water", "DIC"),
]
DEPTHS = [0.0, 100.0, 500.0]  # metres; GLODAP has levels at exactly these

model = osk.read("GOM_bgc")
meta = resolve("GOM_bgc").metadata
obs_all = osk.read("glodap")
glodap_depths = np.asarray(obs_all["Depth"])

comparisons, records = [], []
for std_name, short in VARIABLES:
    sub = model[[std_name]].mean("time", keep_attrs=True)
    at_depth = roms.to_depth(sub, meta, [d for d in DEPTHS if d > 0])

    for d in DEPTHS:
        # 0 m = the top s-level; interpolating to exactly 0 m yields NaN
        if d == 0:
            m = roms.surface(sub, meta)[std_name]
        else:
            m = at_depth[std_name].sel(z=-d, method="nearest")
        m.attrs.setdefault("units", "mmol/m^3")

        # GLODAP indexes depth by position; find the level nearest the target
        k = int(np.abs(glodap_depths - d).argmin())
        o = obs_all[std_name].isel(depth_surface=k)
        o = o * (RHO_SEAWATER / 1000.0)  # umol/kg -> mmol/m3
        o.attrs["units"] = "mmol/m^3"

        aligned = osk_align.align(
            m, o, method=METHOD, test_name="model", reference_name="obs"
        )
        rec = osk_metrics.compute(
            aligned,
            test_name="model",
            reference_name="obs",
            variable=std_name,
            test="GOM_bgc",
            reference="glodap",
            depth=d,
            obs_depth=float(glodap_depths[k]),
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
                "standard_name": std_name,
                "row_label": f"{short} {d:g} m",
            }
        )
        print(
            f"{short:4s} {d:5.0f} m  bias={rec['bias']:+8.2f} "
            f"rmse={rec['rmse']:7.2f} corr={rec['corr']:+.3f} "
            f"sigma_ratio={rec['sigma_ratio']:.3f} n={rec['n']:4d}"
        )

csv = osk_metrics.write(records, OUT, stem="gom_glodap_carbon")
print("wrote:", csv)

fig = field_grid(
    comparisons,
    test_name="model",
    reference_name="obs",
    labels=("ROMS GOM (Jan 2012 mean)", "GLODAPv2.2016b (annual)"),
    title="Carbonate system — ROMS GOM vs GLODAPv2.2016b",
    save=OUT / "figures" / "carbon_depths.png",
)
print("wrote:", OUT / "figures" / "carbon_depths.png")
