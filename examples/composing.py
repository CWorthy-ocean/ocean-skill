"""How to compose comparisons — the same figures as the other examples, in a few lines.

Everything below goes through ``osk.compare``, which reads both sources through their
catalogs, reduces them to comparable fields (surface level or depth transform), matches
the variable across CF naming conventions, converts units, regrids test onto reference,
and computes area-weighted metrics. Each block is one figure plus one metrics table.

Run:  python examples/composing.py
"""

from pathlib import Path

import ocean_skill as osk

OUT = Path("output/composed")
NO3 = "mole_concentration_of_nitrate_in_sea_water"
PO4 = "mole_concentration_of_phosphate_in_sea_water"
SIO3 = "mole_concentration_of_silicate_in_sea_water"
O2 = "mole_concentration_of_dissolved_molecular_oxygen_in_sea_water"

# 1. ONE comparison ------------------------------------------------------------
c = osk.Comparison(
    reference="woa23_nitrate_month01",
    test="GOM_bgc",
    variable=NO3,
    select={"depth": 0},
    # required, not decorative: the model lane has a time axis and a comparison is one
    # map, so it has to be told what happens to it. There is no default reduction.
    aggregate={"time": "mean"},
)
print(
    "1. single :",
    c,
    "->",
    {k: round(v, 3) for k, v in c.metrics().items() if k in ("bias", "rmse", "corr")},
)
c.plot(title="Surface nitrate", save=OUT / "1_single.png")

# 2. MANY VARIABLES, one row each ----------------------------------------------
nutrients = osk.compare(
    reference=[
        "woa23_nitrate_month01",
        "woa23_phosphate_month01",
        "woa23_silicate_month01",
        "woa23_oxygen_month01",
    ],
    test="GOM_bgc",
    variables=[NO3, PO4, SIO3, O2],
    depths=(0,),
    aggregate={"time": "mean"},
)
print(f"2. by variable: {nutrients}")
nutrients.plot(title="Surface nutrients — GOM vs WOA23", save=OUT / "2_nutrients.png")
nutrients.write_metrics(OUT, stem="nutrients")

# 3. ONE VARIABLE, many depths -------------------------------------------------
depths = osk.compare(
    reference="woa23_nitrate_month01",
    test="GOM_bgc",
    variables=[NO3],
    depths=(0, 50, 100, 300),
    aggregate={"time": "mean"},
)
print(f"3. by depth   : {depths}")
depths.plot(title="Nitrate vs depth — GOM vs WOA23", save=OUT / "3_depths.png")
depths.write_metrics(OUT, stem="nitrate_depths")

# 4. The metrics table, ready for a scorecard or a report ----------------------
df = depths.metrics()
print("\n4. tidy metrics table:")
print(
    df[["variable", "depth", "bias", "rmse", "corr", "sigma_ratio", "n"]].to_string(
        index=False, float_format=lambda v: f"{v:.3f}"
    )
)

# 5. The escape hatch: prepared arrays for a bespoke figure --------------------
aligned = depths[0].aligned  # test / reference / difference on a common grid
print(f"\n5. escape hatch: {list(aligned.data_vars)} on {dict(aligned.sizes)}")
print(f"   wrote figures + metrics to {OUT}/")
