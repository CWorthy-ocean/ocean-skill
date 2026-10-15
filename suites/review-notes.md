# PACMED suite review — overnight 2026-10-14/15

Every page of each report looked at by Claude after the run. "Model" notes are about the
simulation itself (worth a look by the modelers); "Suite" notes are about the plots or
the ocean-skill code.

## mid-run-check — `diagnostics/mid-run-check/report.pdf`

Run in the Jupyter session, ocean-skill c8cd8fa, 8 dask threads: 11/11 pages drawn,
36 min, peak 30.5 GB (just under the 30 GB-ish limit of a 16-core shared allocation).

| Page | Looks right? | Notes |
|---|---|---|
| Domain-mean time series (surface, 100, 200 m) | yes | Surface SST ~22 C over the whole 54S–66N domain; chlorophyll's spring peak each year. 1417 s: not served from the cache (see "Speed"). |
| Alkalinity minimum — time series | yes, but | **Model:** the minimum is a river mouth (16.1N 97.6E, Irrawaddy/Andaman Sea): salinity 1–17 PSU, Alk/DIC 40–670. **Suite:** extremes will always land on river mouths; an open-ocean mask (or a salinity > 30 condition) would make these pages say more. |
| Alkalinity maximum — time series | yes, but | **Model:** the maximum is also a river mouth (13.4N 100.3E, Chao Phraya, Gulf of Thailand): Alk/DIC 3000–5500 and nitrate up to 107 mmol/m3 at salinity 28–31 — very high; worth checking the river BGC concentrations in the river forcing. |
| Latest snapshot — surface | yes, flags | Patterns look right. **Model:** min temperature −2.64 C (below the seawater freezing point, about −1.9 C), negative nitrate (−1.85) and phosphate (−0.04), Alk/DIC maxima ~4000+ (the river mouths above). |
| Latest snapshot — 200 m | yes, flags | Eastern tropical Pacific OMZ and nutrient maxima where expected. **Model:** O2 min −3.9 and nitrate min −0.62 (negative concentrations in the OMZ). |
| Chlorophyll vs satellite — latest snapshot | yes | (Fixed today: was skipped.) Model chlorophyll too high along the equator (tropical-instability-wave cusps), too low in the north and along the coasts; corr 0.23. Both cut at ~55N (no December satellite data at high latitude). |
| MLD vs Holte & Talley — December 2013 | yes | Model ~40–60 m too deep in the Kuroshio Extension winter mixed layer, too shallow in the subtropical South Pacific; corr 0.78. |
| Nutrients vs WOA23 — Dec 2013, surface / 100 m | yes | **Model:** at 100 m the eastern tropical Pacific OMZ is too oxygenated (up to +100 mmol/m3, bias +17) and the subpolar NW Pacific too low in nitrate/silicate. |
| Alk & DIC vs GLODAPv2 — Dec 2013, surface / 100 m | yes | Alk bias +1.5, DIC −9 at 100 m; corr 0.89–0.94. |

Small plot issues seen on every map page (follow-ups, not blockers):
- The dashed model-domain outline is a straight-edged box, not the curved domain edge.
- On multi-row comparison maps the right-hand "test − reference" colorbar labels of
  neighbouring rows overlap.
- Panel titles on along-boundary and boundary-mean pages say "domain mean".

## model-comparison — `diagnostics/model-comparison/report.pdf` (batch job 21355855, new YAML)

The first run tonight (Jupyter session, the earlier 25-page YAML) was killed out of memory
on "Bias, RMSE and correlation by month (surface)" after the 200 m domain mean took 56 min;
its figures are kept in `logs/model-comparison-run1/` (no PDF). Rerun as a 32-core batch
job with the updated YAML: **all 35 pages drawn**, exit 0, 4 h 01 min, peak 36 GB (more than
the 30 GB of a 16-core allocation, so the 32-core job was needed). Five 200 m pages took
30–48 min each (domain mean, bias by month, overlap mean, each run vs WOA).

| Page | Looks right? | Notes |
|---|---|---|
| Temperature along each boundary, depth vs time | yes | New colour version, one panel per boundary, own colour scale each. South/east/west: seasonal warming in the top ~100 m, steady below. **North** is a shallow shelf: data only in the top ~75 m, the rest of the 0–1000 m axis is grey (a per-panel depth range would use the space better). |
| Nitrate along each boundary, depth vs time | yes | ESPER nitrate: nitracline ~100–200 m on the south/east/west boundaries, seasonal surface drawdown on the east; the north shelf (top ~75 m) varies 2–17 with the seasons. |
| Boundary T and S at 200 m — **north** | **no: empty page** | The north boundary is a ~75 m shelf, so there is nothing at 200 m: an empty plot with a 1970 time axis and `nan` statistics. **Suite:** drop the north page or use a shallower depth there. **Package:** an all-NaN comparison should be skipped with a reason, not drawn empty. |
| Boundary T and S at 200 m, pyesper vs pacmed_2015 (south/east/west) | yes, cosmetic issues | The two runs' boundary physics differ by ~0.002 C and ~0.0003 PSU: effectively identical, as expected. **Suite:** temperature and salinity share one panel on twin y axes, the stats box and legend overlap the lines, and the difference strip shows only temperature. Next run: `plot: {residual: true, rows: variable}` (one row per variable, each with its own difference strip). |

## final-report — `diagnostics/final-report-batch/report.pdf` (batch jobs 21355172, then 21356975)

First batch run: 2 h 11 min, peak 29 GB, 29 of 35 pages drawn; kept in
`logs/final-report-batch-run1/`. The 6 skipped pages were my YAML mistake: merging
nitrate+phosphate and oxygen+silicate onto one section page broke their WOA contour
overlay (one contour set per plotted variable). Split back to one variable per page and
rerun (21356975), with the iron fix below.

| Page | Looks right? | Notes |
|---|---|---|
| Tracers vs WOA23, mean and seasonal std (T, S, NO3, PO4, O2, SiO3) | yes | e.g. temperature: surface bias −0.17 C, 200 m +0.53 C, seasonal std corr 0.98 at the surface. **Suite:** uneven row gaps, overlapping colorbar labels, units written two ways (`degrees_celsius` vs `°C`). |
| MLD vs Holte & Talley, annual mean / seasonal std | yes | (Fixed today: were skipped, tides lifted z above +5 m.) Model ~5 m deeper on average, much deeper in the Kuroshio Extension and Southern Ocean winter; corr 0.81 / 0.84. 25 min per page (MLD over every restart, full columns). |
| T-S diagrams | (as this afternoon) | |
| T and S sections (equator, 180–160W, 160–120W) | yes | Now one page per section; 180–160W: corr 0.99 / 0.97, the model's 20 C isotherm deviates north of the equator (difference up to ±2 C at 100–300 m). |
| SSH vs DUACS | yes, expected caveats | Mean offset −0.40 m (different datums, expected). Model seasonal std too high (bias +0.14 m): the restart zeta is an instantaneous snapshot that includes the tide. |
| Zonal velocity 180–160W / 160–120W | yes | (160–120W fixed today: was skipped.) Equatorial Undercurrent present at ~150 m; model EUC broader and deeper than Cravatte (difference up to +30 cm/s at 200–350 m), bias 3 cm/s, corr 0.83. |
| Iron, surface / 200 m | **fixed in the rerun** (labels still a follow-up) | Rerun with robust limits: surface dissolved iron ~0.05–0.2 nM, low in the subtropical gyres, high in the dust-fed NW Pacific and along the coasts. Before the fix: | Both maps titled just "iron" (mean and std indistinguishable) and the colour scale set by river-mouth maxima, so the open ocean is blank white. Rerun has `robust: true`; telling the two panels apart needs a plot-label fix (follow-up). Units also written two ways (`mmol/m^3`, `mmol/m³`). |
| Chlorophyll vs satellite, mean and seasonal std | **no: layout broken** | Numbers fine (36 restart days; corr 0.54, bias −0.04 mg/m3) but the maps are tiny with large blank gaps and the difference panels collapse ("maps are only 0.58in wide"). A renderer layout bug on this two-row satellite page — needs code work, not a YAML change. The single-row version in mid-run-check draws fine. |
| Minimum oxygen in the water column | **no (second fix running)** | Was showing *surface* oxygen (title "oxygen · surface", values 200–250): a compare page defaults to the surface level, so `Z: min` took the minimum of one level. `select: {depth: column}` was refused on the WOA side (model-only spelling); per-lane vertical selects are refused too (the vertical select must be shared). Now one shared band, `select: {depth: {min: 0, max: 6000}}` (the model keeps every native cell, WOA every level) -- in the local YAML for the next run. |
| Nutrient and O2/Si sections (×12, with WOA isotherms / isohalines) | yes | Back to one variable per page: all drew in the second run. Nitrate along the equator: model nitracline shallower in the west and deeper in the east than WOA; bias 0.57, corr 0.99. |
| Alk & DIC vs GLODAP | yes | Surface Alk bias +6.7, DIC +23; 200 m corr 0.95–0.96. |
| Model pH | yes | Surface ~8.0–8.1, 200 m down to ~7.5 in the eastern tropical and subpolar North Pacific, as expected. |
| Summary (Taylor + target) | **no: package bug, fixed (uncommitted)** | Showed one point: the suite pools metric rows, which have none of the attributes the de-duplication reads, so all 43 looked identical and 42 were dropped as "duplicates". Fixed in `comparison._identity` + `workflows/pages.MetricRecord` with tests; redrawn from tonight's 43 rows it shows all 43. Needs merging before it shows in a report. |

## Speed (for the plan)

- **Refresh invalidates the model cache.** mid-run-check's 25-min domain-mean page was
  recomputed after tonight's refresh even though the restarts had not changed: the lane
  cache key includes the catalog entry's definition fingerprint, which the refresh
  rewrites. Worth keying on the files' sizes/mtimes instead.
- **Section and box-mean pages are never cached.** The 12 nutrient/O2 sections and 3 T/S
  sections took 4–5 min each in *every* final-report run (≈50 of the 59 min of a rerun
  with everything else cached).
- **Depth pages re-read whole columns per page** (the plan's matched-field cache):
  model-comparison's 200 m domain mean took 56 min.
- **By-month pages hold every month's map pair in memory** (36 × 6 variables × 3 maps):
  the surface page needed >30 GB; it only draws metrics.
- MLD pages take ~25 min each (MLD over every restart's full columns).

### model-comparison, remaining pages

| Page | Looks right? | Notes |
|---|---|---|
| Boundary BGC, ESPER vs climatology (×4 sides, mean + seasonal std) | numbers yes, **layout no** | **Model:** ESPER boundary alkalinity and DIC are ~60 mmol/m3 *higher* than the climatology everywhere on the south boundary (corr 0.96–0.98); nitrate +0.7, O2 +6. The climatology has more month-to-month variability than ESPER at depth. **Suite:** 12 rows (6 variables × 2 statistics) on one page: row labels overprint each other and titles collide. Split mean and seasonal std onto separate pages (or the variables into two groups). |
| Salinity restoring vs WOA23 surface salinity | yes | (Fixed today: was skipped.) Restoring field matches WOA annual surface salinity closely (corr 0.998, rmse 0.06). Difference-colorbar tick labels overlap. |
| Initial conditions, pyesper vs pacmed_2015 (surface, 200 m) | yes, scale issue | (Fixed today: was skipped.) Temperature differs by ±6 C (the restarts are Dec 31 vs Oct 31: seasonal). **Model:** alkalinity +58 and DIC +59 already in pyesper's starting state. **Suite:** colour scales set by river-mouth extremes (alkalinity 0–4000), so the alkalinity/DIC/nutrient maps look uniform; add `plot: {robust: true}`. Title shows only the test date. |
| Domain mean, surface / 200 m / SSH (lines + difference strip) | **yes (corrected)** | T, S, O2, chlorophyll: the runs agree to within 0.01 C, 0.001 PSU, 0.3 and 0.02 — this morning's 5 C "difference" was the crop bug. **Model:** pyesper is **+58 mmol/m3 alkalinity** and **+0.8 nitrate** at the surface throughout 2011–2013 — inherited from the initial condition (+58) and fed by the ESPER boundary (+60). |
| Bias, RMSE and correlation by month (surface, 200 m) | yes, cosmetic | Alkalinity (bias 58) shares a panel with nitrate and oxygen (same units), flattening them; one row per variable would read better. Chlorophyll correlation cycles 0.68–0.93 seasonally. |
| Overlap-mean maps (surface, 200 m, SSH) | yes | Good page: the nitrate difference is concentrated in the subpolar North Pacific and along the equator; alkalinity +58 uniformly. |
| December 2013 maps | yes | as the overlap means. |
| Each run vs WOA23, mean and seasonal std (×4) | numbers yes, **layout no** | Same renderer problem as final-report's chlorophyll page: 12 rows (6 variables × 2 statistics) leave the maps 0.35 in wide with huge gaps — unreadable. Split mean/std (or variables) across pages until the renderer is fixed. |
| Skill summary | **no: package bug, fixed (uncommitted)** | One point only (see final-report's summary). |

## Morning summary

**Reports** (all on Anvil under `diagnostics/`):
- `mid-run-check/report.pdf` — complete, 11/11 pages.
- `final-report-batch/report.pdf` — 34/35 pages (minimum oxygen still skipped; fixed in the
  local YAML). Earlier versions in `logs/final-report-batch-run1/`, `-run2/`.
- `model-comparison/report.pdf` — complete, 35/35 pages. The killed first attempt's figures
  are in `logs/model-comparison-run1/`.

**Biggest findings about the runs:** the two runs' physics are essentially identical
(boundary and interior); pyesper carries **+58 mmol/m3 alkalinity and +59 DIC** from its
initial condition and its ESPER boundary (~+60 vs the climatology), plus ~+0.8 nitrate.
Both runs: OMZ too oxygenated at 100 m, small negative tracer values, river-mouth extremes
(Alk/DIC up to 5500 at the Chao Phraya), mixed layer too deep in the Kuroshio Extension.

**Uncommitted in the worktree, ready for you to merge:**
- `ocean_skill/comparison.py` + `ocean_skill/workflows/pages.py` +
  `tests/test_workflows_summary_pool.py`: the summary-page fix (all comparisons drawn).
- `suites/final-report.yaml`: nutrient sections back to one variable per page, iron
  `robust: true`, minimum oxygen over a shared full-depth band.
- `suites/review-notes.md` (this file).

**Suggested next YAML changes** (not made — waiting for what people want to see):
split the mean/std pages that have 12 rows (model-comparison's BGC sections and WOA pages);
`robust: true` on the initial-condition pages; drop or shallow the north boundary's 200 m
page; one row per variable on the boundary line and by-month pages.

**Package follow-ups:** the 12-row layout bug (chlorophyll, WOA pages), "domain mean" in
boundary titles, iron/aggregate panels unlabeled, units spelled two ways, empty all-NaN
pages drawn instead of skipped, and the speed items above.

**Anvil jobs tonight:** final-report 21355172 (16 cores, 2 h 11 m), 21356975 (16, 57 m),
21357771 (16, 59 m); model-comparison 21355855 (32 cores, 4 h 01 m; 21355854 cancelled
before starting); profiling 21355856 (8 cores, 23 s). **≈194 SU** in total (cap 300).
