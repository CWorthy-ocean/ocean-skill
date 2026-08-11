"""ROMS (GOM offline, Jan 2012) vs WOA23 January climatology — surface nutrients + O2.

Extends the single-variable example to a multi-row suite: one
``model | obs | difference`` row per variable, conservatively regridded, with
area-weighted metrics written to a tidy CSV.

Conservative regridding (not bilinear) is the right operator here: the ROMS grid is
~5-8 km against WOA's 1 degree, so the model must be *area-averaged* onto the
observation cells rather than sampled.

Run:  python examples/gom_woa_nutrients.py
"""

from pathlib import Path

import ocean_skill as osk
from ocean_skill import align as osk_align
from ocean_skill import metrics as osk_metrics
from ocean_skill import roms
from ocean_skill.catalog import resolve
from ocean_skill.plot.matplotlib_renderer import field_grid

OUT = Path("output/gom_woa")
METHOD = "conservative_normed"
RHO_SEAWATER = 1025.0  # kg m-3, for the umol/kg -> mmol/m3 conversion

# model standard_name, WOA entry, WOA standard_name, short label
VARIABLES = [
    (
        "mole_concentration_of_nitrate_in_sea_water",
        "woa23_nitrate_month01",
        "moles_of_nitrate_per_unit_mass_in_sea_water",
        "NO3",
    ),
    (
        "mole_concentration_of_phosphate_in_sea_water",
        "woa23_phosphate_month01",
        "moles_of_phosphate_per_unit_mass_in_sea_water",
        "PO4",
    ),
    (
        "mole_concentration_of_silicate_in_sea_water",
        "woa23_silicate_month01",
        "moles_of_silicate_per_unit_mass_in_sea_water",
        "SiO3",
    ),
    (
        "mole_concentration_of_dissolved_molecular_oxygen_in_sea_water",
        "woa23_oxygen_month01",
        "moles_of_oxygen_per_unit_mass_in_sea_water",
        "O2",
    ),
]

model = osk.read("GOM_bgc")
meta = resolve("GOM_bgc").metadata
surface = roms.surface(model, meta)

comparisons, records = [], []
for std_name, woa_entry, woa_var, short in VARIABLES:
    m = surface[std_name].mean("time", keep_attrs=True)
    m.attrs.setdefault("units", "mmol/m^3")

    o = osk.read(woa_entry)[woa_var].isel(time=0).isel(depth=0)
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
        reference=woa_entry,
        depth="surface",
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
            "row_label": short,
        }
    )
    print(
        f"{short:5s} bias={rec['bias']:+.3f} rmse={rec['rmse']:.3f} "
        f"corr={rec['corr']:+.3f} sigma_ratio={rec['sigma_ratio']:.3f} n={rec['n']}"
    )

csv = osk_metrics.write(records, OUT, stem="gom_woa_surface_nutrients")
print("wrote:", csv)

fig = field_grid(
    comparisons,
    test_name="model",
    reference_name="obs",
    labels=("ROMS GOM (Jan 2012 mean)", "WOA23 (Jan climatology)"),
    title=f"Surface nutrients & oxygen — ROMS GOM vs WOA23 ({METHOD} regrid, Jan 2012)",
    save=OUT / "figures" / "surface_nutrients.png",
)
print("wrote:", OUT / "figures" / "surface_nutrients.png")
