# Phase 5C: what the reports catch vs what the model output shows

Two blind assessments per run (2026-10-16), by separate agents with the same model:

| | **From reports only** (5A) | **From the output itself** (5B) |
|---|---|---|
| CCS 4 km, 2013 | `ccs-from-reports.md`: `ccs-mid-run-check.pdf` (23 pp) | `ccs-from-output.md`: scan of all 103 restarts, every variable and level |
| PACMED pyesper | `pacmed-from-reports.md`: Phase 1 `mid-run-check.pdf` + `final-report.pdf`, 2011-2013 | `pacmed-from-output.md`: scan of 78 restarts, 2010-2013 |

The output scan (`scan_restarts.py`) is independent of ocean_skill: for each variable and s-level it records stats over ocean cells, extremes with their locations, volume inventories, hypoxic volumes, deep means, and boundary-band means. Its limits:
- it has no observations, so it cannot judge skill;
- its per-level means mix depths, because it works on s-levels;
- it gives one extreme location per variable per file, not maps.

The scan's headline CCS finding was checked against its tables by the planner. The 1% step at the March/April task join is real: the weekly change before and after is 0.001-0.02%.

## Summary

**The reports get the model's skill against observations right, but miss most of what is wrong inside the run.** They show the state at a few depths against climatologies. They are blind to:
- **jumps between job segments,**
- **spin-up and drift of the biogeochemical pools,**
- **how widespread negative and impossible values are,**
- **broken diagnostic fields,**
- **anything below 500 m.**

Every one of these is cheap to detect from the output. In both runs, the report reader's verdict of "credible, with a few local problems" understates the biogeochemistry problems the output shows.

## CCS

| Finding | Reports | Output | Verdict |
|---|---|---|---|
| **~1% drop in NO3/PO4/SiO3/O2/DIC/ALK between 2013-03-29 and 04-02 (march_ccs to april_ccs), bottom-heavy (deep DIC -30 mmol/m3), T/S/Fe unchanged** | not seen; domain means "about closed" | high confidence; ~150x a normal weekly change | **gap, high**. The surface-only domain-mean page (surface change 0.05%) cannot show it. The 100/200 m lines were dropped after running out of memory. |
| Plankton initial state out of balance (spC -87%, zooC -61%; diat C:P 11, diaz C:P 3.9 at start) | not seen (chlorophyll only) | high | **gap, medium** |
| Extent of negative values: diatC < 0 in 36% of cells, diazC 26%, spC 22%, zooC 18%, DOC 32%, NH4 2% by December | "small negatives exist" (NO3 -0.56, O2 -0.77) | high | **gap, medium**: the reports show existence (colorbar minimum), not extent |
| Negative O2 to -11.5 in Southern California Borderland basins (600-900 m), -6.2 offshore by December | only -0.77 at 200 m | high | **gap, medium** (depths not shown) |
| pH broken: ABIO pH = 0 everywhere; pH = 0 in 2,251 columns; ALT_CO2 fields are exact copies | not shown | high | **gap, medium** |
| Fraser/Salish ALK 9,104, DIC 8,776; Copper River ALK ~400 | yes (both, with coordinates) | yes, plus DOC 990, NO3 207, O2 639 at the same cells | **both**; the output adds the full set of variables |
| DOC/DON/DOP inventories -7 to -8% over the year, still falling | not shown | high | gap, low-medium |
| Isolated PWS salinity spike to 36.1; diazP maximum on the north boundary row; currents > 3 m/s in SE Alaska channels | not shown | medium | gap, low |
| Sub-freezing temperature (-2.3 C, Cook Inlet) | noted under "cannot tell" | 65 of 103 files | both (weakly) |
| Surface warming ~+1 C over the year | yes | yes | both |
| Chlorophyll skill (corr 0.19), DIC +28 vs GLODAP, northern nitrate high, O2 bias pattern, MLD | yes | cannot | **reports only** (their strength) |
| Bathymetry seam near 40N | yes | cannot | reports only (worth checking) |

## PACMED

| Finding | Reports | Output | Verdict |
|---|---|---|---|
| **OMZ too weak:** O2 +17 mmol/m3 at 200 m, OMZ 57 m too thin | yes (state vs WOA) | **and getting weaker:** suboxic volume -44% over 4 years, accelerating (-8.5% to -15.6%/yr); O2 rising at 100-540 m | **complementary, high**. Together they say the OMZ is too weak *and still eroding*. The reports cannot show the trend: no O2 inventory or deep series. |
| Plankton/DOM initial state inconsistent (phyto P 10-30x too high vs C); phytoplankton still drifting in 2013 (diat -12 to -17%/yr) | not seen | high | **gap, high** |
| DOC/DON/DOP negative in ~1/3 of cells from day 2, and decaying ~5%/yr (-19% in 4 years) | not seen | high | **gap, medium-high** |
| pH = 0 in 2,596 cells per level, every file; ABIO pH all zero | pH maps look normal (8.05-8.10, OMZ tongue) | high | **gap, and the report is misleading**: robust color limits hide the zeros, and any pH metric would include them |
| Deep (> 1000 m) warming 0.012 C/yr; O2 loss near the seabed ~4 mmol/m3/yr | nothing below 500 m | medium | **gap, medium** |
| Physical spikes: T -5.3 C in the open ocean off Sanriku; zeta ±5 m in the Visayan/Sibuyan seas in 54/78 files; > 3 m/s speeds | T -2.64 noted only | medium-high | **gap, medium**. The zeta spikes may also feed the report's SSH variability ratio of 4.3, which it attributes to tides. |
| Extent of negatives (plankton 55-57% of cells at depth, NO3 1-2%, O2 down to -35, Fe) | "small negatives" | high | gap, medium |
| River-mouth extremes (Chao Phraya ALK 5,541; Martaban 43) | yes | yes (+ DOC 1,074, Fe 72 nM, salt 0.15 at the Meghna) | **both** |
| Surface ALK rising (~+3-4 / +10 mmol/m3) without salinity change | yes (domain means) | yes (interior +10.6) | both |
| No jump at task joins or at the 2010 to 2011 continuation | (cannot tell) | none seen at monthly sampling | output only (reassuring) |
| Skill vs WOA/GLODAP/satellite/MLD/SSH/Cravatte; halved seasonal variability at 200 m; T-S water masses | yes | cannot | **reports only** |

## What would close the gaps (suite changes, ranked)

1. **A "run integrity" table** for mid-run-check (and final-report). For every variable in the restart, at all depths, for the latest snapshot and the worst over the run, list:
   - the % of ocean cells negative or outside physical bounds, NaN/Inf counts;
   - min/max with lon/lat/depth and whether the cell is near a river or boundary;
   - fields that are all zero or exact copies of another field.
   
   Catches: the extent of negatives, negative O2 at depth, pH zeros and ABIO zeros, ALT_CO2 copies, T -5.3 C, zeta ±5 m, PWS salinity, speeds.
   
   Cost: one streaming pass, like the scan (CCS 17 min, PACMED 22 min on 32 cores).
2. **An inventory time-series page.** The volume integral of every tracer (DOM, plankton, Fe included), kinetic energy, the O2 < 60 and < 20 volumes, and deep (> 1000 m) means, with **job-segment boundaries marked and steps flagged** (a change > N x the typical step).
   
   Catches: the CCS step, OMZ erosion, DOM decay, spin-up, deep drift.
   
   Cost: needs no depth matching, so it avoids the memory problem that removed CCS's 100/200 m lines.
3. **A plankton/stoichiometry spin-up page:** phytoplankton and zooplankton C, Chl, P, Fe pools and C:P, C:Chl through the run.
4. **Mask derived fields:** exclude pH = 0 cells from pH maps and metrics, and say how many were masked.
5. **Below 500 m:** one deep page (a 1000-4000 m profile or section of T, O2, DIC drift since the start).
6. **Captions:** when color limits are robust, say what fraction of cells lies outside them and the true min/max, so hidden outliers are visible.

Items 1-2 would have caught the most important problem in each run (the CCS restart step; PACMED's eroding OMZ and pool spin-up). They also cost the least. They could become built-in page kinds (`integrity:`, `inventory:`) that every C-Star suite gets from the `ocean-skill-suite new` templates.

## Caveats
- **5A:** one reader, one model. Another reader might catch more from the same pages, such as the CCS DOM or plankton fields, but those pages don't exist.
- **The scan's period differs from the reports'.** For PACMED it starts in 2010 and the reports in 2011, so the 2010 kinetic-energy spin-up is out of the reports' scope by design.
- **The causes named in the output assessments are inferences:** the April re-initialization, too much P in the initial state, and the river end-members. The runs' authors need to confirm them.
