# PACMED suite comparisons

Every page each suite draws (section dividers left out), from the suite YAMLs in this folder. "Last Anvil run" is the 2026-10-14 run on Anvil: drawn, skipped (reason in run.log), killed (out of memory on that page), or not reached (after the kill). Chlorophyll in final-report now reads the 36 restart days only.

## mid-run-check (19 pages)

| # | Page | Compares | Variables | Where | When | Last Anvil run |
|---|---|---|---|---|---|---|
| 1 | Domain-mean time series | pacmed_pyesper (model only) | temperature; salinity; nitrate; alkalinity; dissolved_inorganic_carbon; chlorophyll (sum of 3) | surface, 100 m, 200 m | whole run → area mean | drawn |
| 2 | Alkalinity minimum — time series | pacmed_pyesper (model only) | alkalinity | surface | whole run | drawn |
| 3 | Alkalinity maximum — time series | pacmed_pyesper (model only) | alkalinity | surface | whole run | drawn |
| 4 | Snapshot — temperature | pacmed_pyesper (model only) | temperature | surface, 100 m, 200 m | snapshots [0, -3, -2, -1] | drawn |
| 5 | Snapshot — salinity | pacmed_pyesper (model only) | salinity | surface, 100 m, 200 m | snapshots [0, -3, -2, -1] | drawn |
| 6 | Snapshot — nitrate | pacmed_pyesper (model only) | nitrate | surface, 100 m, 200 m | snapshots [0, -3, -2, -1] | drawn |
| 7 | Snapshot — phosphate | pacmed_pyesper (model only) | phosphate | surface, 100 m, 200 m | snapshots [0, -3, -2, -1] | drawn |
| 8 | Snapshot — oxygen | pacmed_pyesper (model only) | oxygen | surface, 100 m, 200 m | snapshots [0, -3, -2, -1] | drawn |
| 9 | Snapshot — dissolved_inorganic_carbon | pacmed_pyesper (model only) | dissolved_inorganic_carbon | surface, 100 m, 200 m | snapshots [0, -3, -2, -1] | drawn |
| 10 | Snapshot — alkalinity | pacmed_pyesper (model only) | alkalinity | surface, 100 m, 200 m | snapshots [0, -3, -2, -1] | drawn |
| 11 | Chlorophyll vs satellite — latest snapshot | pacmed_pyesper vs chl_gapfree_my_daily_geo | chlorophyll (sum of 3) vs chlorophyll | surface | 2013-12-31, per 1D | skipped |
| 12 | Mixed layer depth vs Holte & Talley — December 2013 | pacmed_pyesper vs holte_talley_mld_clim | mld (density_threshold) vs mld_by_sigma_theta | — | 2013-12-01 to 2013-12-31 → mean | drawn |
| 13 | Nutrients vs WOA23 — December 2013 (surface) | pacmed_pyesper vs woa23_nitrate_month12, woa23_phosphate_month12, woa23_silicate_month12, woa23_oxygen_month12 | nitrate; phosphate; silicate; oxygen | surface | 2013-12-01 to 2013-12-31 → mean | drawn |
| 14 | Nutrients vs WOA23 — December 2013 (100 m) | pacmed_pyesper vs woa23_nitrate_month12, woa23_phosphate_month12, woa23_silicate_month12, woa23_oxygen_month12 | nitrate; phosphate; silicate; oxygen | 100 m | 2013-12-01 to 2013-12-31 → mean | drawn |
| 15 | Alkalinity & DIC vs GLODAPv2 — December 2013 mean (surface) | pacmed_pyesper vs glodap | alkalinity; dissolved_inorganic_carbon | — | 2013-12-01 to 2013-12-31 → mean | drawn |
| 16 | Alkalinity & DIC vs GLODAPv2 — December 2013 mean (100 m) | pacmed_pyesper vs glodap | alkalinity; dissolved_inorganic_carbon | — | 2013-12-01 to 2013-12-31 → mean | drawn |
| 17 | T-S diagrams — six Pacific regions | T-S: ROMS, WOA23, GLORYS12 | temperature, salinity | — | whole run | drawn |
| 18 | Zonal velocity — 180-160W | pacmed_pyesper vs cravatte_u_179e_160w | east_velocity | 24 levels 0–500 m; lon 180–200; lat -20–20 | whole run → mean | drawn |
| 19 | Zonal velocity — 160-120W | pacmed_pyesper vs cravatte_u_160w_120w | east_velocity | 24 levels 0–500 m; lon 200–240; lat -10–18 | whole run → mean | skipped |

## model-comparison (78 pages)

| # | Page | Compares | Variables | Where | When | Last Anvil run |
|---|---|---|---|---|---|---|
| 1 | Boundary physics, both runs — south boundary | pyesper_bry_phys_south, pacmed2015_bry_phys_south (model only) | temperature; salinity; ssh | — | 2011-01-01 to 2013-12-31 → monthly means | drawn |
| 2 | Boundary physics, both runs — north boundary | pyesper_bry_phys_north, pacmed2015_bry_phys_north (model only) | temperature; salinity; ssh | — | 2011-01-01 to 2013-12-31 → monthly means | drawn |
| 3 | Boundary physics, both runs — east boundary | pyesper_bry_phys_east, pacmed2015_bry_phys_east (model only) | temperature; salinity; ssh | — | 2011-01-01 to 2013-12-31 → monthly means | drawn |
| 4 | Boundary physics, both runs — west boundary | pyesper_bry_phys_west, pacmed2015_bry_phys_west (model only) | temperature; salinity; ssh | — | 2011-01-01 to 2013-12-31 → monthly means | drawn |
| 5 | Boundary BGC (ESPER, pyesper run) — south boundary | pyesper_bry_bgc_south (model only) | alkalinity; dissolved_inorganic_carbon; nitrate; oxygen; phosphate; silicate | — | 2011-01-01 to 2013-12-31 → monthly means | drawn |
| 6 | Boundary BGC (ESPER, pyesper run) — north boundary | pyesper_bry_bgc_north (model only) | alkalinity; dissolved_inorganic_carbon; nitrate; oxygen; phosphate; silicate | — | 2011-01-01 to 2013-12-31 → monthly means | drawn |
| 7 | Boundary BGC (ESPER, pyesper run) — east boundary | pyesper_bry_bgc_east (model only) | alkalinity; dissolved_inorganic_carbon; nitrate; oxygen; phosphate; silicate | — | 2011-01-01 to 2013-12-31 → monthly means | drawn |
| 8 | Boundary BGC (ESPER, pyesper run) — west boundary | pyesper_bry_bgc_west (model only) | alkalinity; dissolved_inorganic_carbon; nitrate; oxygen; phosphate; silicate | — | 2011-01-01 to 2013-12-31 → monthly means | drawn |
| 9 | Boundary physics RMSE and bias by month — south boundary | pyesper_bry_phys_south vs pacmed2015_bry_phys_south | temperature; salinity | 12 levels 5–1000 m; transect eta_rho=0 | whole run, per 1MS | drawn |
| 10 | Boundary physics RMSE and bias by month — north boundary | pyesper_bry_phys_north vs pacmed2015_bry_phys_north | temperature; salinity | 12 levels 5–1000 m; transect eta_rho=0 | whole run, per 1MS | drawn |
| 11 | Boundary physics RMSE and bias by month — east boundary | pyesper_bry_phys_east vs pacmed2015_bry_phys_east | temperature; salinity | 12 levels 5–1000 m; transect xi_rho=0 | whole run, per 1MS | drawn |
| 12 | Boundary physics RMSE and bias by month — west boundary | pyesper_bry_phys_west vs pacmed2015_bry_phys_west | temperature; salinity | 12 levels 5–1000 m; transect xi_rho=0 | whole run, per 1MS | drawn |
| 13 | Boundary physics, pyesper vs pacmed_2015 — south (overlap mean) | pyesper_bry_phys_south vs pacmed2015_bry_phys_south | temperature; salinity | 12 levels 5–1000 m; transect eta_rho=0 | 2011-01-01 to 2013-12-31 → mean | drawn |
| 14 | Boundary physics, pyesper vs pacmed_2015 — north (overlap mean) | pyesper_bry_phys_north vs pacmed2015_bry_phys_north | temperature; salinity | 12 levels 5–1000 m; transect eta_rho=0 | 2011-01-01 to 2013-12-31 → mean | drawn |
| 15 | Boundary physics, pyesper vs pacmed_2015 — east (overlap mean) | pyesper_bry_phys_east vs pacmed2015_bry_phys_east | temperature; salinity | 12 levels 5–1000 m; transect xi_rho=0 | 2011-01-01 to 2013-12-31 → mean | drawn |
| 16 | Boundary physics, pyesper vs pacmed_2015 — west (overlap mean) | pyesper_bry_phys_west vs pacmed2015_bry_phys_west | temperature; salinity | 12 levels 5–1000 m; transect xi_rho=0 | 2011-01-01 to 2013-12-31 → mean | drawn |
| 17 | Boundary BGC, ESPER vs climatology — south (annual mean) | pyesper_bry_bgc_south vs pacmed2015_bry_bgc_south | alkalinity; dissolved_inorganic_carbon; nitrate; oxygen; phosphate; silicate | 12 levels 5–1000 m; transect eta_rho=0 | whole run → mean | drawn |
| 18 | Boundary BGC, ESPER vs climatology — north (annual mean) | pyesper_bry_bgc_north vs pacmed2015_bry_bgc_north | alkalinity; dissolved_inorganic_carbon; nitrate; oxygen; phosphate; silicate | 12 levels 5–1000 m; transect eta_rho=0 | whole run → mean | drawn |
| 19 | Boundary BGC, ESPER vs climatology — east (annual mean) | pyesper_bry_bgc_east vs pacmed2015_bry_bgc_east | alkalinity; dissolved_inorganic_carbon; nitrate; oxygen; phosphate; silicate | 12 levels 5–1000 m; transect xi_rho=0 | whole run → mean | drawn |
| 20 | Boundary BGC, ESPER vs climatology — west (annual mean) | pyesper_bry_bgc_west vs pacmed2015_bry_bgc_west | alkalinity; dissolved_inorganic_carbon; nitrate; oxygen; phosphate; silicate | 12 levels 5–1000 m; transect xi_rho=0 | whole run → mean | drawn |
| 21 | Boundary BGC, ESPER vs climatology — south, month 1 | pyesper_bry_bgc_south vs pacmed2015_bry_bgc_south | alkalinity; dissolved_inorganic_carbon; nitrate; oxygen; phosphate; silicate | 12 levels 5–1000 m; transect eta_rho=0 | {'months': [1]} → mean | drawn |
| 22 | Boundary BGC, ESPER vs climatology — south, month 7 | pyesper_bry_bgc_south vs pacmed2015_bry_bgc_south | alkalinity; dissolved_inorganic_carbon; nitrate; oxygen; phosphate; silicate | 12 levels 5–1000 m; transect eta_rho=0 | {'months': [7]} → mean | drawn |
| 23 | Boundary BGC, ESPER vs climatology — north, month 1 | pyesper_bry_bgc_north vs pacmed2015_bry_bgc_north | alkalinity; dissolved_inorganic_carbon; nitrate; oxygen; phosphate; silicate | 12 levels 5–1000 m; transect eta_rho=0 | {'months': [1]} → mean | drawn |
| 24 | Boundary BGC, ESPER vs climatology — north, month 7 | pyesper_bry_bgc_north vs pacmed2015_bry_bgc_north | alkalinity; dissolved_inorganic_carbon; nitrate; oxygen; phosphate; silicate | 12 levels 5–1000 m; transect eta_rho=0 | {'months': [7]} → mean | drawn |
| 25 | Boundary BGC, ESPER vs climatology — east, month 1 | pyesper_bry_bgc_east vs pacmed2015_bry_bgc_east | alkalinity; dissolved_inorganic_carbon; nitrate; oxygen; phosphate; silicate | 12 levels 5–1000 m; transect xi_rho=0 | {'months': [1]} → mean | drawn |
| 26 | Boundary BGC, ESPER vs climatology — east, month 7 | pyesper_bry_bgc_east vs pacmed2015_bry_bgc_east | alkalinity; dissolved_inorganic_carbon; nitrate; oxygen; phosphate; silicate | 12 levels 5–1000 m; transect xi_rho=0 | {'months': [7]} → mean | drawn |
| 27 | Boundary BGC, ESPER vs climatology — west, month 1 | pyesper_bry_bgc_west vs pacmed2015_bry_bgc_west | alkalinity; dissolved_inorganic_carbon; nitrate; oxygen; phosphate; silicate | 12 levels 5–1000 m; transect xi_rho=0 | {'months': [1]} → mean | drawn |
| 28 | Boundary BGC, ESPER vs climatology — west, month 7 | pyesper_bry_bgc_west vs pacmed2015_bry_bgc_west | alkalinity; dissolved_inorganic_carbon; nitrate; oxygen; phosphate; silicate | 12 levels 5–1000 m; transect xi_rho=0 | {'months': [7]} → mean | drawn |
| 29 | pyesper boundary vs WOA23 / GLODAPv2 — south | pyesper_bry_bgc_south vs woa23_nitrate_annual, woa23_phosphate_annual, woa23_silicate_annual, woa23_oxygen_annual, glodap | alkalinity; dissolved_inorganic_carbon; nitrate; oxygen; phosphate; silicate | 12 levels 5–1000 m; transect eta_rho=0 | whole run → mean | drawn |
| 30 | pyesper boundary vs WOA23 / GLODAPv2 — north | pyesper_bry_bgc_north vs woa23_nitrate_annual, woa23_phosphate_annual, woa23_silicate_annual, woa23_oxygen_annual, glodap | alkalinity; dissolved_inorganic_carbon; nitrate; oxygen; phosphate; silicate | 12 levels 5–1000 m; transect eta_rho=0 | whole run → mean | drawn |
| 31 | pyesper boundary vs WOA23 / GLODAPv2 — east | pyesper_bry_bgc_east vs woa23_nitrate_annual, woa23_phosphate_annual, woa23_silicate_annual, woa23_oxygen_annual, glodap | alkalinity; dissolved_inorganic_carbon; nitrate; oxygen; phosphate; silicate | 12 levels 5–1000 m; transect xi_rho=0 | whole run → mean | drawn |
| 32 | pyesper boundary vs WOA23 / GLODAPv2 — west | pyesper_bry_bgc_west vs woa23_nitrate_annual, woa23_phosphate_annual, woa23_silicate_annual, woa23_oxygen_annual, glodap | alkalinity; dissolved_inorganic_carbon; nitrate; oxygen; phosphate; silicate | 12 levels 5–1000 m; transect xi_rho=0 | whole run → mean | drawn |
| 33 | pacmed_2015 boundary (climatology) vs WOA23 / GLODAPv2 — south | pacmed2015_bry_bgc_south vs woa23_nitrate_annual, woa23_phosphate_annual, woa23_silicate_annual, woa23_oxygen_annual, glodap | alkalinity; dissolved_inorganic_carbon; nitrate; oxygen; phosphate; silicate | 12 levels 5–1000 m; transect eta_rho=0 | whole run | drawn |
| 34 | pacmed_2015 boundary (climatology) vs WOA23 / GLODAPv2 — north | pacmed2015_bry_bgc_north vs woa23_nitrate_annual, woa23_phosphate_annual, woa23_silicate_annual, woa23_oxygen_annual, glodap | alkalinity; dissolved_inorganic_carbon; nitrate; oxygen; phosphate; silicate | 12 levels 5–1000 m; transect eta_rho=0 | whole run | drawn |
| 35 | pacmed_2015 boundary (climatology) vs WOA23 / GLODAPv2 — east | pacmed2015_bry_bgc_east vs woa23_nitrate_annual, woa23_phosphate_annual, woa23_silicate_annual, woa23_oxygen_annual, glodap | alkalinity; dissolved_inorganic_carbon; nitrate; oxygen; phosphate; silicate | 12 levels 5–1000 m; transect xi_rho=0 | whole run | drawn |
| 36 | pacmed_2015 boundary (climatology) vs WOA23 / GLODAPv2 — west | pacmed2015_bry_bgc_west vs woa23_nitrate_annual, woa23_phosphate_annual, woa23_silicate_annual, woa23_oxygen_annual, glodap | alkalinity; dissolved_inorganic_carbon; nitrate; oxygen; phosphate; silicate | 12 levels 5–1000 m; transect xi_rho=0 | whole run | drawn |
| 37 | Boundary physics vs WOA23 — pyesper, south | pyesper_bry_phys_south vs woa23_temperature_annual, woa23_salinity_annual | temperature; salinity | 12 levels 5–1000 m; transect eta_rho=0 | whole run → mean | drawn |
| 38 | Boundary physics vs WOA23 — pyesper, north | pyesper_bry_phys_north vs woa23_temperature_annual, woa23_salinity_annual | temperature; salinity | 12 levels 5–1000 m; transect eta_rho=0 | whole run → mean | drawn |
| 39 | Boundary physics vs WOA23 — pacmed2015, south | pacmed2015_bry_phys_south vs woa23_temperature_annual, woa23_salinity_annual | temperature; salinity | 12 levels 5–1000 m; transect eta_rho=0 | whole run → mean | drawn |
| 40 | Boundary physics vs WOA23 — pacmed2015, north | pacmed2015_bry_phys_north vs woa23_temperature_annual, woa23_salinity_annual | temperature; salinity | 12 levels 5–1000 m; transect eta_rho=0 | whole run → mean | drawn |
| 41 | Boundary physics vs WOA23 — pyesper, east | pyesper_bry_phys_east vs woa23_temperature_annual, woa23_salinity_annual | temperature; salinity | 12 levels 5–1000 m; transect xi_rho=0 | whole run → mean | drawn |
| 42 | Boundary physics vs WOA23 — pyesper, west | pyesper_bry_phys_west vs woa23_temperature_annual, woa23_salinity_annual | temperature; salinity | 12 levels 5–1000 m; transect xi_rho=0 | whole run → mean | drawn |
| 43 | Boundary physics vs WOA23 — pacmed2015, east | pacmed2015_bry_phys_east vs woa23_temperature_annual, woa23_salinity_annual | temperature; salinity | 12 levels 5–1000 m; transect xi_rho=0 | whole run → mean | drawn |
| 44 | Boundary physics vs WOA23 — pacmed2015, west | pacmed2015_bry_phys_west vs woa23_temperature_annual, woa23_salinity_annual | temperature; salinity | 12 levels 5–1000 m; transect xi_rho=0 | whole run → mean | drawn |
| 45 | Surface physics forcing (one month mean, both runs) | pyesper_surface_phys (model only) | Tair; qair; rain; swrad; lwrad; uwnd; vwnd | — | 2011-07-01 to 2011-07-31 → mean | drawn |
| 46 | Surface BGC forcing climatology, run vs run (12-month mean) | pyesper_surface_bgc_clim vs pacmed2015_surface_bgc_clim | dust; iron; nox; nhy | — | 2011-01-31 to 2013-12-31 | drawn |
| 47 | Salinity restoring climatology, run vs run (12-month mean) | pyesper_surface_restoring_clim vs pacmed2015_surface_restoring_clim | sss | — | 2011-01-31 to 2013-12-31 | skipped |
| 48 | Salinity restoring climatology vs WOA23 surface salinity (12-month mean) | pyesper_surface_restoring_clim vs woa23_salinity_annual | sss vs salinity | — | 2011-01-31 to 2013-12-31 | skipped |
| 49 | Atmospheric CO2 forcing, run vs run (overlap mean) | pyesper_surface_co2 vs pacmed2015_surface_co2 | xco2_air | — | 2011-01-01 to 2013-12-31 → mean | drawn |
| 50 | Initial conditions, pyesper vs pacmed_2015 (surface) | ic_pyesper vs ic_pacmed2015 | temperature; salinity; nitrate; phosphate; silicate; oxygen; alkalinity; dissolved_inorganic_carbon | — | 2011-01-31 to 2013-12-31 → mean | skipped |
| 51 | Initial conditions, pyesper vs pacmed_2015 (200 m) | ic_pyesper vs ic_pacmed2015 | temperature; salinity; nitrate; phosphate; silicate; oxygen; alkalinity; dissolved_inorganic_carbon | — | 2011-01-31 to 2013-12-31 → mean | skipped |
| 52 | Initial condition vs WOA23 / GLODAPv2 — ic_pyesper (surface) | ic_pyesper vs woa23_temperature_annual, woa23_salinity_annual, woa23_nitrate_annual, woa23_phosphate_annual, woa23_silicate_annual, woa23_oxygen_annual, glodap | temperature; salinity; nitrate; phosphate; silicate; oxygen; alkalinity; dissolved_inorganic_carbon | — | 2011-01-31 to 2013-12-31 → mean | drawn |
| 53 | Initial condition vs WOA23 / GLODAPv2 — ic_pyesper (200 m) | ic_pyesper vs woa23_temperature_annual, woa23_salinity_annual, woa23_nitrate_annual, woa23_phosphate_annual, woa23_silicate_annual, woa23_oxygen_annual, glodap | temperature; salinity; nitrate; phosphate; silicate; oxygen; alkalinity; dissolved_inorganic_carbon | — | 2011-01-31 to 2013-12-31 → mean | drawn |
| 54 | Initial condition vs WOA23 / GLODAPv2 — ic_pacmed2015 (surface) | ic_pacmed2015 vs woa23_temperature_annual, woa23_salinity_annual, woa23_nitrate_annual, woa23_phosphate_annual, woa23_silicate_annual, woa23_oxygen_annual, glodap | temperature; salinity; nitrate; phosphate; silicate; oxygen; alkalinity; dissolved_inorganic_carbon | — | 2011-01-31 to 2013-12-31 → mean | drawn |
| 55 | Initial condition vs WOA23 / GLODAPv2 — ic_pacmed2015 (200 m) | ic_pacmed2015 vs woa23_temperature_annual, woa23_salinity_annual, woa23_nitrate_annual, woa23_phosphate_annual, woa23_silicate_annual, woa23_oxygen_annual, glodap | temperature; salinity; nitrate; phosphate; silicate; oxygen; alkalinity; dissolved_inorganic_carbon | — | 2011-01-31 to 2013-12-31 → mean | drawn |
| 56 | Domain mean, pyesper vs pacmed_2015 (surface) | pacmed_pyesper vs pacmed_2015 | temperature; salinity; nitrate; alkalinity; oxygen; chlorophyll (sum of 3) | surface | 2011-01-01 to 2013-12-31 → area mean | drawn |
| 57 | Domain mean, pyesper vs pacmed_2015 (200 m) | pacmed_pyesper vs pacmed_2015 | temperature; salinity; nitrate; alkalinity; oxygen; chlorophyll (sum of 3) | 200 m | 2011-01-01 to 2013-12-31 → area mean | killed (out of memory) |
| 58 | Domain mean sea surface height, pyesper vs pacmed_2015 | pacmed_pyesper vs pacmed_2015 | ssh | — | 2011-01-01 to 2013-12-31 → area mean | not reached |
| 59 | Bias, RMSE and correlation by month (surface) | pacmed_pyesper vs pacmed_2015 | temperature; salinity; nitrate; alkalinity; oxygen; chlorophyll (sum of 3) | surface | 2011-01-01 to 2013-12-31, per 1MS | not reached |
| 60 | Bias, RMSE and correlation by month (200 m) | pacmed_pyesper vs pacmed_2015 | temperature; salinity; nitrate; alkalinity; oxygen; chlorophyll (sum of 3) | 200 m | 2011-01-01 to 2013-12-31, per 1MS | not reached |
| 61 | Target diagram through time (surface) | pacmed_pyesper vs pacmed_2015 | temperature; salinity; nitrate; alkalinity; oxygen; chlorophyll (sum of 3) | surface | 2011-01-01 to 2013-12-31, per 1MS | not reached |
| 62 | Target diagram through time (200 m) | pacmed_pyesper vs pacmed_2015 | temperature; salinity; nitrate; alkalinity; oxygen; chlorophyll (sum of 3) | 200 m | 2011-01-01 to 2013-12-31, per 1MS | not reached |
| 63 | Skill maps through time (surface) | pacmed_pyesper vs pacmed_2015 | temperature; salinity; nitrate; alkalinity; oxygen; chlorophyll (sum of 3) | surface | 2011-01-01 to 2013-12-31 | not reached |
| 64 | Skill maps through time (200 m) | pacmed_pyesper vs pacmed_2015 | temperature; salinity; nitrate; alkalinity; oxygen; chlorophyll (sum of 3) | 200 m | 2011-01-01 to 2013-12-31 | not reached |
| 65 | Overlap mean, pyesper vs pacmed_2015 (surface) | pacmed_pyesper vs pacmed_2015 | temperature; salinity; nitrate; alkalinity; oxygen; chlorophyll (sum of 3) | surface | 2011-01-01 to 2013-12-31 → mean | not reached |
| 66 | Overlap mean, pyesper vs pacmed_2015 (200 m) | pacmed_pyesper vs pacmed_2015 | temperature; salinity; nitrate; alkalinity; oxygen; chlorophyll (sum of 3) | 200 m | 2011-01-01 to 2013-12-31 → mean | not reached |
| 67 | Overlap mean sea surface height, pyesper vs pacmed_2015 | pacmed_pyesper vs pacmed_2015 | ssh | — | 2011-01-01 to 2013-12-31 → mean | not reached |
| 68 | December 2013, pyesper vs pacmed_2015 (surface) | pacmed_pyesper vs pacmed_2015 | temperature; salinity; nitrate; alkalinity; oxygen; chlorophyll (sum of 3) | surface | 2013-12-01 to 2013-12-31 → mean | not reached |
| 69 | December 2013, pyesper vs pacmed_2015 (200 m) | pacmed_pyesper vs pacmed_2015 | temperature; salinity; nitrate; alkalinity; oxygen; chlorophyll (sum of 3) | 200 m | 2013-12-01 to 2013-12-31 → mean | not reached |
| 70 | pacmed_pyesper vs WOA23 — mean and seasonal variability (surface) | pacmed_pyesper vs woa23_temperature_monthly, woa23_salinity_monthly, woa23_nitrate_monthly, woa23_phosphate_monthly, woa23_silicate_monthly, woa23_oxygen_monthly | temperature; salinity; nitrate; phosphate; silicate; oxygen | — | 2011-01-01 to 2013-12-31 → mean, seasonal std | not reached |
| 71 | pacmed_pyesper vs WOA23 — mean and seasonal variability (200 m) | pacmed_pyesper vs woa23_temperature_monthly, woa23_salinity_monthly, woa23_nitrate_monthly, woa23_phosphate_monthly, woa23_silicate_monthly, woa23_oxygen_monthly | temperature; salinity; nitrate; phosphate; silicate; oxygen | — | 2011-01-01 to 2013-12-31 → mean, seasonal std | not reached |
| 72 | pacmed_2015 vs WOA23 — mean and seasonal variability (surface) | pacmed_2015 vs woa23_temperature_monthly, woa23_salinity_monthly, woa23_nitrate_monthly, woa23_phosphate_monthly, woa23_silicate_monthly, woa23_oxygen_monthly | temperature; salinity; nitrate; phosphate; silicate; oxygen | — | 2011-01-01 to 2013-12-31 → mean, seasonal std | not reached |
| 73 | pacmed_2015 vs WOA23 — mean and seasonal variability (200 m) | pacmed_2015 vs woa23_temperature_monthly, woa23_salinity_monthly, woa23_nitrate_monthly, woa23_phosphate_monthly, woa23_silicate_monthly, woa23_oxygen_monthly | temperature; salinity; nitrate; phosphate; silicate; oxygen | — | 2011-01-01 to 2013-12-31 → mean, seasonal std | not reached |
| 74 | pacmed_pyesper vs GLODAPv2 — overlap mean (surface) | pacmed_pyesper vs glodap | alkalinity; dissolved_inorganic_carbon | — | 2011-01-01 to 2013-12-31 → mean | not reached |
| 75 | pacmed_pyesper vs GLODAPv2 — overlap mean (200 m) | pacmed_pyesper vs glodap | alkalinity; dissolved_inorganic_carbon | — | 2011-01-01 to 2013-12-31 → mean | not reached |
| 76 | pacmed_2015 vs GLODAPv2 — overlap mean (surface) | pacmed_2015 vs glodap | alkalinity; dissolved_inorganic_carbon | — | 2011-01-01 to 2013-12-31 → mean | not reached |
| 77 | pacmed_2015 vs GLODAPv2 — overlap mean (200 m) | pacmed_2015 vs glodap | alkalinity; dissolved_inorganic_carbon | — | 2011-01-01 to 2013-12-31 → mean | not reached |
| 78 | Skill summary, every comparison above | summary |  | — | whole run | not reached |

## final-report (40 pages)

| # | Page | Compares | Variables | Where | When | Last Anvil run |
|---|---|---|---|---|---|---|
| 1 | Bathymetry | pacmed_pyesper (model only) | h | — | 2011-01-31 to 2013-12-31 | drawn |
| 2 | temperature vs WOA23 — mean and seasonal variability | pacmed_pyesper vs woa23_temperature_monthly | temperature | — | 2011-01-31 to 2013-12-31 → mean, seasonal std | drawn |
| 3 | salinity vs WOA23 — mean and seasonal variability | pacmed_pyesper vs woa23_salinity_monthly | salinity | — | 2011-01-31 to 2013-12-31 → mean, seasonal std | drawn |
| 4 | nitrate vs WOA23 — mean and seasonal variability | pacmed_pyesper vs woa23_nitrate_monthly | nitrate | — | 2011-01-31 to 2013-12-31 → mean, seasonal std | drawn |
| 5 | phosphate vs WOA23 — mean and seasonal variability | pacmed_pyesper vs woa23_phosphate_monthly | phosphate | — | 2011-01-31 to 2013-12-31 → mean, seasonal std | drawn |
| 6 | oxygen vs WOA23 — mean and seasonal variability | pacmed_pyesper vs woa23_oxygen_monthly | oxygen | — | 2011-01-31 to 2013-12-31 → mean, seasonal std | drawn |
| 7 | silicate vs WOA23 — mean and seasonal variability | pacmed_pyesper vs woa23_silicate_monthly | silicate | — | 2011-01-31 to 2013-12-31 → mean, seasonal std | drawn |
| 8 | MLD vs Holte & Talley — annual mean | pacmed_pyesper vs holte_talley_mld_clim | mld (density_threshold) vs mld_by_sigma_theta | — | 2011-01-31 to 2013-12-31 → mean | skipped |
| 9 | MLD vs Holte & Talley — summer (JJA) | pacmed_pyesper vs holte_talley_mld_clim | mld (density_threshold) vs mld_by_sigma_theta | — | JJA → mean | skipped |
| 10 | MLD vs Holte & Talley — winter (DJF) | pacmed_pyesper vs holte_talley_mld_clim | mld (density_threshold) vs mld_by_sigma_theta | — | DJF → mean | drawn |
| 11 | MLD vs Holte & Talley — seasonal variability | pacmed_pyesper vs holte_talley_mld_clim | mld (density_threshold) vs mld_by_sigma_theta | — | 2011-01-31 to 2013-12-31 → seasonal std | skipped |
| 12 | T-S diagrams — six Pacific regions | T-S: ROMS, WOA23, GLORYS12 | temperature, salinity | — | whole run | drawn |
| 13 | temperature along the equator | pacmed_pyesper vs woa23_temperature_annual | temperature | 10 levels 5–500 m; transect lat=0, lon={min: 143, max: 267} | whole run → mean | drawn |
| 14 | salinity along the equator | pacmed_pyesper vs woa23_salinity_annual | salinity | 10 levels 5–500 m; transect lat=0, lon={min: 143, max: 267} | whole run → mean | drawn |
| 15 | temperature — 180-160W mean | pacmed_pyesper vs woa23_temperature_annual | temperature | 10 levels 5–500 m; lon 180–200; lat -30–30 | whole run → mean | drawn |
| 16 | salinity — 180-160W mean | pacmed_pyesper vs woa23_salinity_annual | salinity | 10 levels 5–500 m; lon 180–200; lat -30–30 | whole run → mean | drawn |
| 17 | temperature — 160-120W mean | pacmed_pyesper vs woa23_temperature_annual | temperature | 10 levels 5–500 m; lon 200–240; lat -30–30 | whole run → mean | drawn |
| 18 | salinity — 160-120W mean | pacmed_pyesper vs woa23_salinity_annual | salinity | 10 levels 5–500 m; lon 200–240; lat -30–30 | whole run → mean | drawn |
| 19 | SSH vs DUACS — mean and seasonal variability | pacmed_pyesper vs ssh_duacs_my_daily_geo | ssh | — | 2011-01-31 to 2013-12-31 → mean, seasonal std | drawn |
| 20 | Zonal velocity — 180-160W | pacmed_pyesper vs cravatte_u_179e_160w | east_velocity | 24 levels 0–500 m; lon 180–200; lat -20–20 | whole run → mean | drawn |
| 21 | Zonal velocity — 160-120W | pacmed_pyesper vs cravatte_u_160w_120w | east_velocity | 24 levels 0–500 m; lon 200–240; lat -10–18 | whole run → mean | skipped |
| 22 | Iron at surface — mean and seasonal variability | pacmed_pyesper (model only) | iron | surface | 2011-01-31 to 2013-12-31 → mean, seasonal std | not reached |
| 23 | Iron at 200 m — mean and seasonal variability | pacmed_pyesper (model only) | iron | 200 m | 2011-01-31 to 2013-12-31 → mean, seasonal std | not reached |
| 24 | nitrate along the equator (isotherms) | pacmed_pyesper vs woa23_nitrate_annual | nitrate | 10 levels 5–500 m; transect lat=0, lon={min: 143, max: 267} | whole run → mean | drawn |
| 25 | phosphate along the equator (isotherms) | pacmed_pyesper vs woa23_phosphate_annual | phosphate | 10 levels 5–500 m; transect lat=0, lon={min: 143, max: 267} | whole run → mean | drawn |
| 26 | nitrate — 180-160W mean (isotherms) | pacmed_pyesper vs woa23_nitrate_annual | nitrate | 10 levels 5–500 m; lon 180–200; lat -30–30 | whole run → mean | drawn |
| 27 | phosphate — 180-160W mean (isotherms) | pacmed_pyesper vs woa23_phosphate_annual | phosphate | 10 levels 5–500 m; lon 180–200; lat -30–30 | whole run → mean | drawn |
| 28 | nitrate — 160-120W mean (isotherms) | pacmed_pyesper vs woa23_nitrate_annual | nitrate | 10 levels 5–500 m; lon 200–240; lat -30–30 | whole run → mean | drawn |
| 29 | phosphate — 160-120W mean (isotherms) | pacmed_pyesper vs woa23_phosphate_annual | phosphate | 10 levels 5–500 m; lon 200–240; lat -30–30 | whole run → mean | drawn |
| 30 | oxygen along the equator (isohalines) | pacmed_pyesper vs woa23_oxygen_annual | oxygen | 10 levels 5–500 m; transect lat=0, lon={min: 143, max: 267} | whole run → mean | drawn |
| 31 | silicate along the equator (isohalines) | pacmed_pyesper vs woa23_silicate_annual | silicate | 10 levels 5–500 m; transect lat=0, lon={min: 143, max: 267} | whole run → mean | drawn |
| 32 | oxygen — 180-160W mean (isohalines) | pacmed_pyesper vs woa23_oxygen_annual | oxygen | 10 levels 5–500 m; lon 180–200; lat -30–30 | whole run → mean | drawn |
| 33 | silicate — 180-160W mean (isohalines) | pacmed_pyesper vs woa23_silicate_annual | silicate | 10 levels 5–500 m; lon 180–200; lat -30–30 | whole run → mean | drawn |
| 34 | oxygen — 160-120W mean (isohalines) | pacmed_pyesper vs woa23_oxygen_annual | oxygen | 10 levels 5–500 m; lon 200–240; lat -30–30 | whole run → mean | drawn |
| 35 | silicate — 160-120W mean (isohalines) | pacmed_pyesper vs woa23_silicate_annual | silicate | 10 levels 5–500 m; lon 200–240; lat -30–30 | whole run → mean | drawn |
| 36 | Chlorophyll vs satellite — mean and seasonal variability | pacmed_pyesper vs chl_gapfree_my_daily_geo | chlorophyll (sum of 3) vs chlorophyll | surface | 2011-01-31 to 2013-12-31 → mean, seasonal std | killed (out of memory) |
| 37 | Minimum oxygen in the water column | pacmed_pyesper vs woa23_oxygen_annual | oxygen | — | 2011-01-31 to 2013-12-31 → mean | not reached |
| 38 | Alkalinity & DIC vs GLODAPv2 — run mean | pacmed_pyesper vs glodap | alkalinity; dissolved_inorganic_carbon | — | 2011-01-31 to 2013-12-31 → mean | not reached |
| 39 | Model pH (no observations) | pacmed_pyesper (model only) | MARBL_PH_3D | surface, 200 m | 2011-01-31 to 2013-12-31 → mean | not reached |
| 40 | Fig. 26 — Summary | summary |  | — | whole run | not reached |
