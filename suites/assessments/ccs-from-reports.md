# CCS 4 km run (ccs_4km_v6, 2013) - assessment from `ccs-mid-run-check.pdf` only

## 1. Summary
The report covers about one model year (2013-01-02 to 2013-12-31) of a planned five-year run (2013-2018), described as "spin up of CCS". The large-scale picture is credible. Domain-mean seasonal cycles of temperature, nitrate, DIC and chlorophyll look sensible. Against WOA23 and GLODAPv2 the nutrients, oxygen and alkalinity have spatial correlations of 0.78-0.95, so the basin-scale structure is right. The weak spots are:
- Surface chlorophyll against the satellite is poor: correlation 0.19 and a variance ratio of 0.41.
- Surface DIC is biased high by 28 mmol/m3 with correlation only 0.51.
- Surface nitrate is too high in the north, and surface oxygen is too low.
- The surface alkalinity/DIC extremes at the river-mouth cells at 49N and 60N are not physical (alkalinity 9104 and 404 mmol/m3) and point to river or coastal forcing problems.
- Small negative nitrate and oxygen values exist.

One year with spin-up transients cannot separate drift from seasonality. Nothing was shown about physics skill (SST, SSH, currents) or the coastal upwelling system the domain is named for.

## 2. Findings
| # | kind | finding | evidence | confidence | severity |
|---|---|---|---|---|---|
| 1 | problem | Surface alkalinity maximum is 9104 mmol/m3 at 49.0N 122.8W (Fraser/Salish Sea area), 17 Dec 2013. The series is spiky, 2500 at start, 3000-4500 mid-year, 7000-9000 in Nov-Dec. DIC tracks alkalinity almost 1:1. Salinity at that cell is only 25.5-27.3. High alkalinity at moderate salinity cannot come from dilution, so river alkalinity/DIC loading is likely wrong or accumulating in an under-flushed cell. | p6; snapshot maps p7 (alk max 8.75e3, DIC max 8.44e3) | high that it is unphysical; medium on cause | high |
| 2 | problem | The same cell has a one-snapshot nitrate spike to about 47 mmol/m3 in about mid-April, then back to about 0 within two snapshots. | p6 bottom panel | medium (monthly or sparser sampling) | medium |
| 3 | notable feature | Alkalinity minimum 404.6 mmol/m3 at 60.1N 141.4W (Copper River/Gulf of Alaska coast), 6 Aug. Alkalinity and DIC fall from about 2100 to about 400 as salinity falls from 31 to about 5.5. Nitrate is about 0 from April to October. Temperature is -2 to -1 C in Jan-Mar (at or below the freezing point, no ice model evident) and 15 C in summer. A freshwater plume with very low river alkalinity is plausible, but 400 is extreme. | p5; snapshot colorbar mins p7 (alk min 1.15e3, DIC min 1.1e3, salinity min 18) | medium | medium |
| 4 | numerical artifact | Negative tracers: surface nitrate min -0.56 mmol/m3 (p7); 200 m oxygen min -0.77 mmol/m3 (p8). Surface phosphate min 3.4e-22 (essentially zero). Extent not given. | p7, p8 colorbar min labels | high on existence, low on extent | low-medium |
| 5 | drift | Surface domain means over 2013 are about closed. Nitrate goes from 4.9 to 5.0 (seasonal range 2.7-6.0). Temperature is 13.4 at start vs 14.5 at end (about +1 C). DIC is 2079 vs 2073. Alkalinity is 2286 vs 2289. Salinity is 33.37 vs 33.42 (+0.05). The Jan 2013 transients (alkalinity drops about 2.5 mmol/m3 in the first weeks, salinity dips to 33.35) look like spin-up adjustment. | p4 | medium (read off small plots) | medium |
| 6 | notable feature | Domain-mean surface chlorophyll peaks at about 0.7 mg/m3 in mid-May (baseline 0.1, 0.07 in Jan). The domain-wide bloom is sharp and short (Mar-Jul). | p4 bottom panel | medium | low-medium |
| 7 | skill vs obs | Chlorophyll vs Copernicus L4 on 31 Dec 2013 only: bias -0.161 mg/m3, RMSE 0.505, corr 0.19, sigma ratio 0.41, n=683,547. Model is far lower than the satellite north of about 40N (differences beyond -0.2, saturated) and slightly higher in the southern and coastal Baja area (red). Model spatial variance is much too low. Single winter day, so high-latitude retrievals are suspect. | p9 (map, stats box), p10 (table) | high on numbers, medium on interpretation | medium |
| 8 | skill vs obs | MLD (0.03 sigma-theta from 10 m, Dec 2013) vs Holte & Talley: bias +0.46 m, RMSE 14.8 m, corr 0.56, sigma ratio 0.88, n=1223. Difference field is a noisy +/-15 m pattern with no clear coherent bias. Blank (NaN) cells in the model interior. | p10 | high | low-medium |
| 9 | skill vs obs | Surface vs WOA23 (Dec): nitrate +1.31 (RMSE 2.66, corr 0.935, sigma ratio 1.29), phosphate +0.060, silicate -0.19, oxygen -5.79 mmol/m3 (RMSE 11.6, corr 0.955). Nitrate and phosphate are too high in the subpolar/Gulf of Alaska region (differences beyond +3 and +0.2, saturated). Oxygen is too low across the north-central domain (differences beyond -15, saturated). The oxygen unit basis of the reference (mmol/m3 vs umol/kg, about 2.5%) is not stated. | p11, p13 | high | medium |
| 10 | skill vs obs | 100 m vs WOA23: nitrate +0.75, phosphate +0.055, silicate -1.15, oxygen +3.82 mmol/m3 (RMSE 23.5, corr 0.78, sigma ratio 1.02). Oxygen differences are large and alternate in sign: dark red (model high) in the north, blue (model low) in the southeast/California Current offshore (beyond -30). Oxygen at 100 m is the weakest nutrient/oxygen field in the metrics. | p12, p13 | high | medium |
| 11 | skill vs obs | Carbon vs GLODAPv2: surface DIC bias +28.2 (RMSE 37.6, corr 0.51), alkalinity +5.28 (RMSE 16.7, corr 0.95); at 100 m DIC +13.5 (RMSE 29, corr 0.85), alkalinity +3.29 (RMSE 13, corr 0.93). DIC excess is in the northern half (beyond +40, saturated) and near zero in the south. DIC high relative to alkalinity would imply elevated surface pCO2 and lower pH (inference, not shown). | p14, p15 | high on numbers; low on the pCO2 inference | medium-high |
| 12 | notable feature | The 200 m maps show a blotchy, eddy-like texture. At 200 m, oxygen ranges -0.77 to 307 mmol/m3 and nitrate 2.2-43.8. Salinity at 200 m has a min of 31.2, which is odd for 200 m (a shallow-water or river-influenced cell). | p8 | low | low |
| 13 | artifact | WOA comparisons (n about 1300) are on the 1-degree grid, so the 4 km model is coarsened to blocks. The chlorophyll panels are blanked north of about 55N where the satellite product has no data, so the Gulf of Alaska river-mouth region is excluded from that comparison. | p9, p11 | high | low |
| 14 | notable feature | Grid depth shows a visible straight discontinuity near 40N across 160W-125W and an Aleutian trench line near 53N. The straight edge is not geographic, so it is likely a bathymetry source or tile seam. | p3 | low-medium (could be a rendering/blend artifact) | low-medium |
| 15 | skill vs obs | Taylor/target: nutrients and oxygen cluster at correlation 0.8-0.95 with sigma ratios 0.8-1.3. Chlorophyll and surface DIC are the clear outliers (correlation <0.55). Surface nitrate sigma ratio is 1.29 (too much spatial variance). | p16 | high | medium |

## 3. What the reports cannot tell me
- Physics skill: no SST, SSH, currents/transport, or mixed-layer seasonality comparison (only one month of MLD).
- Coastal upwelling system: no nearshore structure, upwelling index, or cross-shelf sections. The headline CCS processes are invisible at the sizes shown.
- Boundary behavior: open boundaries on the west and south are not shown. Boundary files are listed but not assessed.
- Vertical structure: no sections, no profiles, no deep or intermediate values. The extremes and snapshots are surface or 200 m only. The domain-trend text promises 100 m and 200 m lines but only surface is plotted.
- Time evolution: no multi-year drift (only one year is shown), no inventory or conservation checks for heat, salt, carbon, nutrients or oxygen. The observations are compared at one December month, and GLODAP/WOA are annual climatologies in places, so seasonal aliasing is not separated.
- The oxygen minimum zone, and whether oxygen goes to zero or negative in the California Undercurrent (only a min of -0.77 at 200 m is quoted).
- Numerical stability: no timestep or CFL diagnostics, no count of negative cells, no check that the cell at the 9104 alkalinity maximum is an isolated point or a region.
- River forcing: no river discharge or concentration plots; which rivers produce the extremes is inferred from coordinates only.
- Carbon-system derived fields (pH, pCO2, air-sea CO2 flux, saturation state) and iron are not shown.
- Chlorophyll skill outside a single day (31 Dec), and any comparison to in-situ data (CalCOFI, moorings).
- Sea ice or freezing-point behavior (temperature reaches -2.3 C).

## 4. Report quality notes
- The "mid-run-check" covers only 2013 of a 5-year run, with no flag saying so. The header's "Simulated ... to 2018" is the plan, not what was run.
- The domain-trends intro (p4) says surface, 100 m and 200 m, but the figure and caption show surface only.
- Many "extreme" and snapshot colorbars are robust (central 80%) while min/max labels are not, so the visible colors hide the 10x outliers. The "[PSU]" and "[mmol/m3]" colorbar labels overlap the colorbars on p7 and p8.
- The chlorophyll difference colorbar (+/-0.2 mg/m3, linear) saturates over most of the north, hiding the size of the bias; map panels are log scale and difference is linear.
- Longitude is given as both degrees East and West on p5/p6 (218.6E and 141.4W for the same point).
- The MLD map is sparsely sampled (n=1223) with white gaps and a mismatched domain outline.
- There is no unit conversion note for WOA oxygen. The wide-open metric table has no n of time samples. The target plot and Taylor plot show variable types without depth labels (surface and 100 m are mixed).
