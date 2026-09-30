# Satellite catalogs: NOAA CoastWatch (ERDDAP) and Copernicus Marine (CMEMS)

Server: `https://coastwatch.pfeg.noaa.gov/erddap`
All ranges verified from `allDatasets` + per-dataset `/info/` on **2026-08-12**.

Every dataset below also exists as `<id>_Lon0360` or `<id>_LonPM180` — same data, different
longitude convention. Pick one convention and ignore the twins.

**Frequency column** = composite periods available in that dataset family. The listed ID is the
daily one; swap the period token in the ID to get the others (e.g. `erdMPIC1day_R2022SQ` →
`erdMPIC8day_R2022SQ` → `erdMPICmday_R2022SQ`).

**Grid res vs Effective res** — these are different questions and the gap is often 10×.
Grid res is how the data is stored (derived from the axis, so it disagrees with some titles).
Effective res is the smallest feature the data can actually resolve. Read §8 before using the
grid number to judge what you can see; `jplMURSST41` is genuinely on a 1.1 km grid and
genuinely resolves ~10 km. `≈ grid` marks L3 products, which bin real pixels and so resolve
about what they claim; **unquantified** means the product is an interpolated analysis (so
coarser than its grid) but publishes no figure — unknown, not fine.

---

## 1. SST

| Dataset ID | Variables | Grid res | Effective res | Frequency | Time frame | Type | Notes |
|---|---|---|---|---|---|---|---|
| `jplMURSST41` | `analysed_sst`, `analysis_error`, `mask`, `sea_ice_fraction` | 0.01° (~1 km) | **~10 km** | daily | 2002-06-01 → 2026-08-11 | L4 gap-free | **Default SST.** 1-day lag, no NRT/SQ seam |
| `jplMURSST41mday` | `sst`, `nobs`, `mask` | 0.01° | ~10 km | monthly | 2002-06-16 → 2026-07-16 | L4 gap-free | ⚠️ variable is `sst`, **not** `analysed_sst` |
| `jplMURSST41anom1day` | `sstAnom`, `mask` | 0.01° | ~10 km | daily / monthly | 2002-06-01 → 2026-08-10 | L4 anomaly | Baseline 2003–2014 |
| `jplMURSST41clim` | `mean_sst`, `standard_deviation` | 0.01° | ~10 km | `dayOfYear` 1–365 | climatology | L4 climatology | Base 2003–2014. No `time` axis |
| `jplMURSST42` | `analysed_sst`, `analysis_error`, `mask`, `sea_ice_fraction`, `sst_anomaly` | 0.25° (~28 km) | ~28 km (grid binds) | daily / monthly | 2002-09-01 → 2026-08-11 | L4 gap-free | Coarse MUR; **anomaly bundled in**. ~600× smaller than v4.1 |
| `nesdisGeoPolarSSTN5NRT` | `analysed_sst`, `analysis_error`, `mask`, `sea_ice_fraction` | 0.05° (~5 km) | coarser, **unquantified** | daily | 2002-09-01 → 2026-08-11 | L4 gap-free | GOES-18/19 + Himawari-9 + VIIRS + AVHRR. Better diurnal sampling than MUR |
| `ncdcOisst21Agg` | `sst`, `anom`, `err`, `ice` | 0.25° | ~100 km | daily | 1981-09-01 → 2026-07-28 | L4 gap-free | **Final.** Longest record. ~2 wk lag |
| `ncdcOisst21NrtAgg` | `sst`, `anom`, `err`, `ice` | 0.25° | ~100 km | daily | 2020-05-04 → 2026-08-11 | L4 gap-free | **Preliminary.** NRT twin of above |
| `NOAA_DHW` | `CRW_SST`, `CRW_SSTANOMALY`, `CRW_DHW`, `CRW_HOTSPOT`, `CRW_BAA`, `CRW_BAA_7D_MAX`, `CRW_SEAICE` (+ `_mask` for each) | 0.05° | coarser, **unquantified** | daily | 1985-04-01 → 2026-08-11 | L4 | Coral Reef Watch. Anomaly baseline 1985–2012 MMM |
| `NOAA_DHW_monthly` | `sea_surface_temperature`, `sea_surface_temperature_anomaly`, `mask` | 0.05° | coarser, **unquantified** | monthly | 1985-01-16 → 2026-07-16 | L4 | ⚠️ totally different variable names from the daily |
| `nceiErsstv5` | `sst`, `ssta` | 2.0° (~222 km) | ≫ grid (heavily smoothed) | monthly | 1854-01-01 → 2026-07-15 | in situ only | Not satellite. Climate baseline / long context |
| `erdMH1sstd1day_R2022SQMasked` | `sstMasked` | 0.0417° (4 km) | ≈ grid | daily / 8-day / monthly | 2002-07-04 → 2026-06-30 | L3, gappy | MODIS Aqua **SQ** |
| `erdMH1sstd1day_R2022NRTMasked` | `sstMasked` | 0.0417° | ≈ grid | daily | 2019-11-01 → 2026-08-11 | L3, gappy | MODIS Aqua **NRT** twin |
| `nesdisVHNsstDaily` | `sea_surface_temperature`, `graphics` | 0.0375° (4 km) | ≈ grid | daily | 2020-01-15 → 2026-08-11 | L3, gappy | VIIRS S-NPP. Flagged EXPERIMENTAL |
| `erdMWsstd1day` | `sst` | **0.0125° (~1.4 km)** | ≈ grid | 1/3/8/14-day, monthly | 2002-06-24 → 2026-08-10 | L3, gappy | **West US only.** 3× finer than global |
| `erdMBsstd1day` | `sst` | 0.025° (~2.8 km) | ≈ grid | 1/3/5/8/14-day, monthly | 2006-01-01 → 2026-08-10 | L3, gappy | **Pacific only** (lat −45→65) |

## 2. Ocean color — chlorophyll

| Dataset ID | Variables | Grid res | Effective res | Frequency | Time frame | Type | Notes |
|---|---|---|---|---|---|---|---|
| `nesdisVHNSQchlaDaily` | `chlor_a` | 0.0375° (4 km) | ≈ grid | daily / monthly | 2012-01-02 → 2026-08-02 | L3, gappy | **Best single-sensor chl.** VIIRS S-NPP, **SQ**, only ~10 d lag |
| `nesdisVHNchlaDaily` | `chlor_a` | 0.0375° | ≈ grid | daily | 2025-07-06 → 2026-07-09 | L3, gappy | NRT twin, but **rolling ~1 yr window only** |
| `noaacwNPPN20S3ASCIDINEOF2kmDaily` | `chlor_a` | 0.0208° (~2 km) | coarser, **unquantified** | daily | 2018-01-01 → 2026-08-01 | L4 gap-filled | **Highest-res gap-free chl.** S-NPP + NOAA-20 VIIRS + Sentinel-3A OLCI, **SQ** |
| `nesdisNPPN20S3ASCIDINEOFDaily` | `chlor_a` | 0.0833° (~9 km) | coarser, **unquantified** | daily | 2018-02-09 → 2026-08-01 | L4 gap-filled | Same product at 9 km — **redundant**, skip unless you want small files |
| `nesdisVHNnoaaSNPPnoaa20chlaGapfilledDaily` | `chlor_a` | 0.0833° (~9 km) | coarser, **unquantified** | daily | 2018-05-30 → 2026-07-31 | L4 gap-filled | VIIRS-only gap-filled, **SQ** |
| `nesdisVHNnoaaSNPPnoaa20NRTchlaGapfilledDaily` | `chlor_a` | 0.0833° | coarser, **unquantified** | daily | 2020-05-05 → 2026-08-10 | L4 gap-filled | **NRT** twin of above, 2 d lag |
| `erdMH1chla1day_R2022SQ` | `chlor_a` | 0.0417° (4 km) | ≈ grid | daily / monthly | 2002-07-04 → 2026-05-30 | L3, gappy | MODIS Aqua **SQ**. The only chl before 2012 |
| `erdMH1chla1day_R2022NRT` | **`chlorophyll`** | 0.0417° | ≈ grid | daily / 8-day / monthly | 2022-03-01 → 2026-08-11 | L3, gappy | MODIS Aqua **NRT**. ⚠️ variable renamed vs SQ twin |
| `erdMWchla1day` | `chlorophyll` | **0.0125° (~1.4 km)** | ≈ grid | 1/3/8/14-day, monthly | 2002-07-04 → 2026-08-11 | L3, gappy | **West US only** |
| `erdMBchla1day` | `chlorophyll` | 0.025° (~2.8 km) | ≈ grid | 1/3/5/8/14-day, monthly | 2006-01-01 → 2026-08-11 | L3, gappy | **Pacific only** |
| `erdVHNchla1day` | `chla` | **0.0075° (750 m)** | ≈ grid | 1/3/8-day, monthly | 2015-02-25 → 2026-06-14 | L3, gappy | **North Pacific only.** ⚠️ ~2 mo stale |

## 3. Ocean color — optics

| Dataset ID | Variables | Grid res | Effective res | Frequency | Time frame | Type | Notes |
|---|---|---|---|---|---|---|---|
| `nesdisVHNSQkd490Daily` | `kd_490` | 0.0375° | ≈ grid | daily / monthly | 2012-01-02 → 2026-08-02 | L3, gappy | VIIRS **SQ** |
| `nesdisVHNSQkdparDaily` | `kd_par` | 0.0375° | ≈ grid | daily / monthly | 2012-01-02 → 2026-08-02 | L3, gappy | VIIRS **SQ** |
| `noaacwNPPN20S3AkdSCIDINEOF2kmDaily` | `kd_490` | 0.0208° (~2 km) | coarser, **unquantified** | daily | 2018-01-01 → 2026-08-01 | L4 gap-filled | Gap-free Kd490 |
| `noaacwNPPN20S3AspmSCIDINEOF2kmDaily` | `spm` | 0.0208° (~2 km) | coarser, **unquantified** | daily | 2018-01-02 → 2026-08-01 | L4 gap-filled | Suspended particulate matter |
| `nesdisNPPN20S3AspmSCIDINEOFDaily` | `spm` | 0.0833° (~9 km) | coarser, **unquantified** | daily | 2018-02-09 → 2026-08-01 | L4 gap-filled | 9 km version — redundant |
| `erdMH1kd4901day_R2022SQ` | `Kd_490` | 0.0417° | ≈ grid | daily / 8-day / monthly | 2002-07-04 → **2025-10-31** | L3, gappy | MODIS **SQ**. ⚠️ ~10 mo stale |
| `erdMH1kd4901day_R2022NRT` | `Kd_490` | 0.0417° | ≈ grid | daily / 8-day / monthly | 2022-03-01 → 2026-08-11 | L3, gappy | MODIS **NRT** twin |
| `nesdisVHNSQnLw443Daily` | `nLw_443` | 0.0375° | ≈ grid | daily / monthly | 2012-01-02 → 2026-08-02 | L3, gappy | Also `nLw410`, `nLw486`, `nLw551`, `nLw671` — swap the number in the ID |

## 4. Biogeochemistry

| Dataset ID | Variables | Grid res | Effective res | Frequency | Time frame | Type | Notes |
|---|---|---|---|---|---|---|---|
| `erdMPIC1day_R2022SQ` | `pic` | 0.0417° | ≈ grid | daily / 8-day / monthly | 2002-07-04 → 2025-10-31 | L3, gappy | Particulate inorganic carbon, **SQ**. ⚠️ monthly variant starts **2022-01**, not 2002 |
| `erdMPIC1day_R2022NRT` | `pic` | 0.0417° | ≈ grid | daily / 8-day / monthly | 2022-03-01 → 2026-08-11 | L3, gappy | **NRT** twin |
| `erdMPOC1day_R2022SQ` | `poc` | 0.0417° | ≈ grid | daily / 8-day / monthly | 2002-07-04 → 2026-05-30 | L3, gappy | Particulate organic carbon, **SQ**. Same monthly caveat |
| `erdMPOC1day_R2022NRT` | `poc` | 0.0417° | ≈ grid | daily / 8-day / monthly | 2022-03-01 → 2026-08-11 | L3, gappy | **NRT** twin |
| `erdMH1cflh1day_R2022SQ` | **`nflh`** | 0.0417° | ≈ grid | daily / 8-day / monthly | 2002-07-04 → 2026-05-30 | L3, gappy | Fluorescence line height, **SQ**. ⚠️ ID says `cflh`, variable is `nflh` |
| `erdMH1cflh1day_R2022NRT` | `nflh` | 0.0417° | ≈ grid | daily / 8-day / monthly | 2022-03-01 → 2026-08-11 | L3, gappy | **NRT** twin |
| `erdMH1pp1day` | `productivity` | 0.0417° | ≈ grid (inherits chl) | 1/3/8-day, monthly | 2003-01-01 → 2026-08-07 | L4 derived | MODIS primary productivity. EXPERIMENTAL, no SQ/NRT split |
| `productivity_viirs_snpp_daily` | `productivity`, `chlor_a`, `par`, `sea_surface_temperature` | 0.0417° | ≈ grid (inherits chl) | daily / monthly | 2012-01-19 → 2026-05-31 | L4 derived | VIIRS S-NPP **SQ**. **Bundles 4 vars incl. live PAR** |
| `productivity_viirs_snpp_nrt_daily` | same 4 | 0.0417° | ≈ grid (inherits chl) | daily | 2024-01-01 → 2026-08-10 | L4 derived | **NRT** twin |
| `productivity_viirs_noaa20_daily` | same 4 | 0.0417° | ≈ grid (inherits chl) | daily / monthly | 2018-01-31 → 2025-12-31 | L4 derived | NOAA-20 **SQ** |
| `productivity_viirs_noaa20_daily_nrt` | same 4 | 0.0417° | ≈ grid (inherits chl) | daily | 2025-01-01 → 2026-08-10 | L4 derived | **NRT** twin |

**⚠️ Standalone MODIS PAR is dead.** `erdMH1par01day_R2022NRT` stops **2024-07-14**;
`erdMH1par01day_R2022SQ` stops **2023-09-30**. Get `par` from the VIIRS productivity
datasets above instead — those are current.

## 5. Physical

| Dataset ID | Variables | Grid res | Effective res | Frequency | Time frame | Type | Notes |
|---|---|---|---|---|---|---|---|
| `erdQCwindproducts1day` | `wind_speed`, `wind_direction`, `wind_u`, `wind_v`, `stress`, `stress_u`, `stress_v`, `curl`, `divergence`, `ekman_current`, `ekman_current_u`, `ekman_current_v`, `ekman_upwelling` | **0.333°** | ~37 km (grid binds) | 4-hr, 1/3/7-day, monthly | 2021-09-03 → 2026-08-09 | L4 | Metop-C ASCAT. ⚠️ title claims 0.25° — **grid is actually 0.333°** (verified) |
| `pifscCcmpDailyV21NRT` | `uwnd`, `vwnd`, `nobs` | 0.25° | coarser, **unquantified** | **6-hourly** | 2015-01-16 → 2026-08-11 | L4 gap-free | CCMP. ⚠️ ID says "Daily", data is 6-hourly. Lat ±78° only |
| `coastwatchSMOSv662SSS1day` | `sss`, `sss_dif`, `l2_time`, `l2_lat`, `l2_lon` | 0.25° | coarser (L-band footprint ≫ grid) | 1-day / 3-day | 2010-06-01 → 2026-08-10 | L3 | SMOS salinity. 3-day composite is less noisy |
| `NWW3_Global_Best` | `shgt`, `sper`, `sdir`, `whgt`, `wper`, `wdir`, `Thgt`, `Tper`, `Tdir` | 0.5° | coarser (model) | **hourly** | 2017-01-01 → 2026-08-18 (forecast) | model | WaveWatch III. Model, not satellite. Extends into future |
| `nsidcG02202v6nh1day` | `cdr_seaice_conc`, `cdr_seaice_conc_stdev`, + 3 QA flags | 25 km polar | ≈ grid (PM footprint) | daily / monthly | 1978-10-25 → 2026-08-05 | L3 CDR | Sea ice, N. hemisphere. `...sh1day` for south. Longest record here |
| `nsidcG10016v3nh1day` | same | 25 km polar | ≈ grid (PM footprint) | daily / monthly | 2024-07-01 → 2026-02-22 | L3 NRT | NRT twin. ⚠️ ~6 mo stale |

## 6. Currents / SSH — ⚠️ read the note

| Dataset ID | Variables | Grid res | Effective res | Frequency | Time frame | Type | Notes |
|---|---|---|---|---|---|---|---|
| `erdTAgeo1day` | `u_current`, `v_current` | 0.25° | **~200 km** | "1-day" / monthly | 1992-10-14 → **2012-12-08** | L4 | AVISO. ⚠️ actual spacing ~6 days despite the name. **Dead since 2012** |
| `erdTAssh1day` | `ssh`, `sshd` | 0.25° | ~200 km | "1-day" / monthly | 1992-10-14 → 2012-12-09 | L4 | Same era, same problem |
| `miamicurrents` | `u_current`, `v_current` | 0.2° | ~200 km | daily | 2016-07-01 → 2026-08-11 | L4 | Current, but **lat ±64.7° only**, no SSH, thin metadata |
| `nesdisSSH1day` | `sla`, `ugos`, `vgos` | 0.25° | ~200 km | daily | 2017-02-13 → **2026-03-25** | L4 | EXPERIMENTAL. ⚠️ ~4.5 mo stale |

**There is no continuous long global geostrophic current record on this server.**
Coverage is 1992–2012, then a **gap over 2013–2016**, then two partial products that each
have problems. For a real long record use **Copernicus Marine `SEALEVEL_GLO_PHY_L4`**
(1993 → present, 0.25°, continuous; `sla`, `adt`, `ugos`, `vgos`, `ugosa`, `vgosa`) via the
`copernicusmarine` Python client. Free account required. This is the one variable where I'd
go off this ERDDAP entirely.

---

## 7. Locations (griddap URLs)

`{server}/griddap/{dataset_id}` opens directly with `xr.open_dataset` — **no `intake_erddap`
needed**, which sidesteps the broken search path (see gotcha 6 below).

```python
reader_kwargs={"chunks": "auto"}   # NOT {} -- see below
```

⚠️ **`chunks={}` is unusable here.** It means "use the engine's preferred chunks", and
OPeNDAP reports none (`encoding["chunksizes"]` is `None`), so dask builds a *single* chunk
spanning the whole variable — 45.8 TB for MUR. Every `.sel()` then means "materialise that
chunk, then slice it", and a comparison dies with a `MemoryError` naming the full array
shape even though the selection is small. `"auto"` gives ~130-270 MB chunks and, unlike an
explicit dict, does not depend on the dimension names — the same setting works for MUR
(`time/latitude/longitude`), the DINEOF products (which add `altitude`), the NSIDC polar
grids (`time/y/x`) and the MUR climatology (`dayOfYear/...`).

This is the opposite of the file-based catalogs: netCDF files *do* carry chunk encoding, so
`chunks={}` is right for those, and for the Copernicus Zarr stores in part C.

All 63 URLs below were HTTP-verified on 2026-08-12, and every one opens in xarray **except
where noted** — the two `pae-paha.pacioos.hawaii.edu` entries, which are corrected below.

```python
ERDDAP = "https://coastwatch.pfeg.noaa.gov/erddap"

COASTWATCH = {
  # --- 1 SST ---
  "mur_sst_daily":               f"{ERDDAP}/griddap/jplMURSST41",
  "mur_sst_monthly":             f"{ERDDAP}/griddap/jplMURSST41mday",
  "mur_sst_anom_daily":          f"{ERDDAP}/griddap/jplMURSST41anom1day",
  "mur_sst_anom_monthly":        f"{ERDDAP}/griddap/jplMURSST41anommday",
  "mur_sst_climatology":         f"{ERDDAP}/griddap/jplMURSST41clim",
  "mur_sst_025_daily":           f"{ERDDAP}/griddap/jplMURSST42",
  "mur_sst_025_monthly":         f"{ERDDAP}/griddap/jplMURSST42mday",
  "geopolar_sst_daily":          f"{ERDDAP}/griddap/nesdisGeoPolarSSTN5NRT",
  "oisst_final_daily":           f"{ERDDAP}/griddap/ncdcOisst21Agg",
  "oisst_prelim_daily":          f"{ERDDAP}/griddap/ncdcOisst21NrtAgg",
  "crw_dhw_daily":               "https://pae-paha.pacioos.hawaii.edu/erddap/griddap/dhw_5km",  # NOT pfeg
  "crw_dhw_monthly":             f"{ERDDAP}/griddap/NOAA_DHW_monthly",
  "ersst_monthly":               f"{ERDDAP}/griddap/nceiErsstv5",
  "modis_sst_sq_daily":          f"{ERDDAP}/griddap/erdMH1sstd1day_R2022SQMasked",
  "modis_sst_nrt_daily":         f"{ERDDAP}/griddap/erdMH1sstd1day_R2022NRTMasked",
  "viirs_sst_daily":             f"{ERDDAP}/griddap/nesdisVHNsstDaily",
  "modis_sst_westus_daily":      f"{ERDDAP}/griddap/erdMWsstd1day",
  "modis_sst_pacific_daily":     f"{ERDDAP}/griddap/erdMBsstd1day",
  # --- 2 chlorophyll ---
  "viirs_chl_sq_daily":          f"{ERDDAP}/griddap/nesdisVHNSQchlaDaily",
  "viirs_chl_sq_monthly":        f"{ERDDAP}/griddap/nesdisVHNSQchlaMonthly",
  "viirs_chl_nrt_daily":         f"{ERDDAP}/griddap/nesdisVHNchlaDaily",
  "chl_gapfree_2km_daily":       f"{ERDDAP}/griddap/noaacwNPPN20S3ASCIDINEOF2kmDaily",
  "chl_gapfree_9km_daily":       f"{ERDDAP}/griddap/nesdisNPPN20S3ASCIDINEOFDaily",
  "chl_gapfree_viirs_sq_daily":  f"{ERDDAP}/griddap/nesdisVHNnoaaSNPPnoaa20chlaGapfilledDaily",
  "chl_gapfree_viirs_nrt_daily": f"{ERDDAP}/griddap/nesdisVHNnoaaSNPPnoaa20NRTchlaGapfilledDaily",
  "modis_chl_sq_daily":          f"{ERDDAP}/griddap/erdMH1chla1day_R2022SQ",
  "modis_chl_nrt_daily":         f"{ERDDAP}/griddap/erdMH1chla1day_R2022NRT",
  "modis_chl_westus_daily":      f"{ERDDAP}/griddap/erdMWchla1day",
  "modis_chl_pacific_daily":     f"{ERDDAP}/griddap/erdMBchla1day",
  "viirs_chl_npac_750m_daily":   f"{ERDDAP}/griddap/erdVHNchla1day",
  # --- 3 optics ---
  "viirs_kd490_sq_daily":        f"{ERDDAP}/griddap/nesdisVHNSQkd490Daily",
  "viirs_kdpar_sq_daily":        f"{ERDDAP}/griddap/nesdisVHNSQkdparDaily",
  "kd490_gapfree_2km_daily":     f"{ERDDAP}/griddap/noaacwNPPN20S3AkdSCIDINEOF2kmDaily",
  "spm_gapfree_2km_daily":       f"{ERDDAP}/griddap/noaacwNPPN20S3AspmSCIDINEOF2kmDaily",
  "spm_gapfree_9km_daily":       f"{ERDDAP}/griddap/nesdisNPPN20S3AspmSCIDINEOFDaily",
  "modis_kd490_sq_daily":        f"{ERDDAP}/griddap/erdMH1kd4901day_R2022SQ",
  "modis_kd490_nrt_daily":       f"{ERDDAP}/griddap/erdMH1kd4901day_R2022NRT",
  "viirs_nlw443_sq_daily":       f"{ERDDAP}/griddap/nesdisVHNSQnLw443Daily",
  # --- 4 biogeochemistry ---
  "modis_pic_sq_daily":          f"{ERDDAP}/griddap/erdMPIC1day_R2022SQ",
  "modis_pic_nrt_daily":         f"{ERDDAP}/griddap/erdMPIC1day_R2022NRT",
  "modis_poc_sq_daily":          f"{ERDDAP}/griddap/erdMPOC1day_R2022SQ",
  "modis_poc_nrt_daily":         f"{ERDDAP}/griddap/erdMPOC1day_R2022NRT",
  "modis_nflh_sq_daily":         f"{ERDDAP}/griddap/erdMH1cflh1day_R2022SQ",
  "modis_nflh_nrt_daily":        f"{ERDDAP}/griddap/erdMH1cflh1day_R2022NRT",
  "modis_par_sq_daily":          f"{ERDDAP}/griddap/erdMH1par01day_R2022SQ",   # stalled 2023-09
  "modis_par_nrt_daily":         f"{ERDDAP}/griddap/erdMH1par01day_R2022NRT",  # stalled 2024-07
  "modis_pp_daily":              f"{ERDDAP}/griddap/erdMH1pp1day",
  "viirs_snpp_pp_sq_daily":      f"{ERDDAP}/griddap/productivity_viirs_snpp_daily",
  "viirs_snpp_pp_nrt_daily":     f"{ERDDAP}/griddap/productivity_viirs_snpp_nrt_daily",
  "viirs_n20_pp_sq_daily":       f"{ERDDAP}/griddap/productivity_viirs_noaa20_daily",
  "viirs_n20_pp_nrt_daily":      f"{ERDDAP}/griddap/productivity_viirs_noaa20_daily_nrt",
  # --- 5 physical ---
  "ascat_wind_daily":            f"{ERDDAP}/griddap/erdQCwindproducts1day",
  "ccmp_wind_6hourly":           f"{ERDDAP}/griddap/pifscCcmpDailyV21NRT",
  "smos_sss_daily":              f"{ERDDAP}/griddap/coastwatchSMOSv662SSS1day",
  "smos_sss_3day":               f"{ERDDAP}/griddap/coastwatchSMOSv662SSS3day",
  "ww3_waves_hourly":            "https://pae-paha.pacioos.hawaii.edu/erddap/griddap/ww3_global",  # NOT pfeg
  "seaice_cdr_nh_daily":         f"{ERDDAP}/griddap/nsidcG02202v6nh1day",
  "seaice_cdr_sh_daily":         f"{ERDDAP}/griddap/nsidcG02202v6sh1day",
  "seaice_nrt_nh_daily":         f"{ERDDAP}/griddap/nsidcG10016v3nh1day",
  # --- 6 currents / SSH ---
  "aviso_geo_currents":          f"{ERDDAP}/griddap/erdTAgeo1day",
  "aviso_ssh":                   f"{ERDDAP}/griddap/erdTAssh1day",
  "miami_currents_daily":        f"{ERDDAP}/griddap/miamicurrents",
  "nesdis_ssh_daily":            f"{ERDDAP}/griddap/nesdisSSH1day",
}
```

### ⚠️ Two datasets must NOT use the PFEG URL

`NOAA_DHW` and `NWW3_Global_Best` are `EDDGridFromErddap` **proxies**. PFEG serves their
metadata (`.dds`/`.das` return 200), but **data requests 302-redirect to PacIOOS**:

```
https://coastwatch.pfeg.noaa.gov/erddap/griddap/NOAA_DHW.dods?time
  → 302 → https://pae-paha.pacioos.hawaii.edu/erddap/griddap/dhw_5km.dods?time
```

netCDF-C's DAP2 client does not follow that redirect, so `xr.open_dataset` fails with
`NetCDF: Malformed or inaccessible DAP2 DATADDS or DAP4 DAP response`. It looks like a
corrupt dataset; it is a redirect.

Point at the source instead — both verified working:

| Nickname | Use this URL | Verified |
|---|---|---|
| `crw_dhw_daily` | `https://pae-paha.pacioos.hawaii.edu/erddap/griddap/dhw_5km` | 15101×3600×7200, 12 vars, 1985-04-01→2026-08-11 |
| `ww3_waves_hourly` | `https://pae-paha.pacioos.hawaii.edu/erddap/griddap/ww3_global` | 83644×311×720, 9 vars, 2017-01-01→2026-08-19 |

If you hit this error on any *other* ERDDAP dataset, check for a redirect first:

```bash
curl -sI "<griddap-url>.dods?time" | grep -i location
```

### Building the catalog

```python
build_catalog(COASTWATCH, "catalogs/coastwatch.yaml", title="NOAA CoastWatch",
              reader_kwargs={"chunks": "auto"})
```

---

## 8. Grid resolution vs effective resolution

**These are different numbers and the gap is often an order of magnitude.** Grid spacing is
how the data is stored. Effective resolution is the smallest feature the data can actually
resolve. For any interpolated L4 analysis the second is much coarser than the first.

The clearest case is the one you flagged: **`jplMURSST41` really is on a 0.01° grid (1.1 km)
— that part is right. But it resolves features of about 10 km.** MUR is an optimal-
interpolation analysis; its own paper describes it as improving feature resolution "from
approximately 100 km down to 10 km" over previous SST analyses. So the 1 km grid is real, and
the 1 km *information* is not. Filtering `resolution=2` and concluding you can see 2 km fronts
in MUR would be wrong by a factor of ten.

Same story for altimetry: DUACS maps are gridded at 0.125–0.25° but resolve roughly **200 km
wavelengths at midlatitude** (about 100 km at high latitude, up to 800 km near the equator).
The Copernicus documentation is explicit that "the dynamical content of the 0.25° gridded maps
does not have full 0.25° spatial resolution due to the filtering properties of optimal
interpolation."

| Nickname | Grid | Effective | Basis |
|---|---|---|---|
| `mur_sst_daily` | 0.01° / **1.1 km** | **~10 km** | MUR paper (sourced) |
| `mur_sst_025_daily` | 0.25° / 27.8 km | ~28 km | grid is the binding limit |
| `oisst_final_daily` | 0.25° / 27.8 km | ~100 km | typical of pre-MUR SST analyses |
| `ssh_duacs_my_daily` | 0.125° / 13.9 km | **~200 km** midlat | DUACS QUID (sourced) |
| `ssh_duacs_twosat_daily` | 0.25° / 27.8 km | ~200 km+ | fixed 2-sat sampling |
| `cur_multiobs_my_daily` | 0.25° / 27.8 km | ~200 km | derived from the same altimetry |
| `viirs_chl_sq_daily`, `modis_chl_sq_daily`, all L3 | 4 km | **≈ grid** | L3 bins native L2 pixels — no interpolation |
| `chl_gapfree_2km_daily`, all DINEOF / gap-filled | 2–9 km | coarser, **unquantified** | interpolation fills cloud gaps |
| `geopolar_sst_daily`, `crw_dhw_daily` | 5 km | coarser, **unquantified** | L4 blended analyses |
| `glorys_my_daily` | 1/12° / 9.3 km | coarser, **unquantified** | model effective resolution > grid |

The honest split: **L3 single-sensor products resolve about what their grid says** (they bin
real pixels), while **every L4 analysis resolves less than its grid says**. Where a product
documents a number I've used it; where it doesn't I've left it blank rather than guess, and
blank means unknown — `find()` treats it as "not stated", not "fine".

Note the separate issue for L3: effective resolution ≈ grid, but *sampling* is gappy. You get
honest 4 km where there are no clouds and nothing where there are.

### Feeding the curated values in

`grid_resolution_km` is derived automatically when a source is probed. `effective_resolution_km`
can't be — it's literature, not data — so pass it per source:

```python
EFFECTIVE_KM = {
    "mur_sst_daily": 10.0,
    "mur_sst_025_daily": 28.0,
    "oisst_final_daily": 100.0,
    "ssh_duacs_my_daily": 200.0,
    "ssh_duacs_twosat_daily": 200.0,
    "cur_multiobs_my_daily": 200.0,
}

# A source may be a bare URL, or a dict of per-source keys containing "url".
# Shared options still apply; the per-source keys are added on top.
build_catalog(
    {name: ({"url": url, "effective_resolution_km": EFFECTIVE_KM[name]}
            if name in EFFECTIVE_KM else url)
     for name, url in COASTWATCH.items()},
    "catalogs/coastwatch.yaml", title="NOAA CoastWatch",
    reader_kwargs={"chunks": "auto"},
)
```

Verified end to end — the entry comes back carrying both, the derived one and the curated one:

```
grid_resolution_lat_deg    = 0.01
grid_resolution_km         = 1.112     # derived from the axis
time_resolution            = P1D       # derived from the axis
effective_resolution_km    = 10.0      # curated, passed in
featureType                = grid
```

Then both are queryable:

```python
osk.find(resolution=5)              # grid spacing 5 km or finer
osk.find(effective_resolution=25)   # actually resolves 25 km features
osk.find(cadence="daily")           # or "hourly", "monthly", "8-day"
osk.find(vertical=True)             # has a depth axis
```

Sources: [Chin et al. 2017, *A multi-scale high-resolution analysis of global sea surface
temperature*](https://www.sciencedirect.com/science/article/abs/pii/S0034425717303462) ·
[Ballarotta et al. 2019, *On the resolutions of ocean altimetry
maps*](https://os.copernicus.org/articles/15/1091/2019/) ·
[DUACS L4 product FAQ](https://duacs.cls.fr/faq/what-are-the-product-specification/spatial-and-temporal-resolution-of-the-l4-gridded-products/)

---

## Cross-cutting gotchas

1. **Variable names change between siblings of the same product.** MUR SST is `analysed_sst`
   daily, `sst` monthly, `mean_sst` in the climatology. MODIS chl is `chlor_a` in SQ and
   `chlorophyll` in NRT. CRW is `CRW_SST` daily and `sea_surface_temperature` monthly. Never
   assume the variable name carries across a period or stream change.
2. **NRT records are short.** Most `_R2022NRT` datasets start 2022-03, regardless of what the
   title says. The long record is always in the SQ twin.
3. **SQ lag varies wildly** — 10 days (VIIRS chl) to 10 months (MODIS Kd490) to permanently
   stalled (MODIS PAR). Check `maxTime` before depending on one.
4. **Titles are not reliable** for resolution or frequency. Two confirmed wrong above
   (ASCAT winds 0.333° not 0.25°; CCMP 6-hourly not daily).
5. **`_R2022SQ` monthly variants of PIC/POC start 2022-01**, not 2002, even though the daily
   goes back to 2002. For long monthly PIC/POC you need the legacy `erdMPICmday` (2003–2022)
   stitched to `erdMPICmday_R2022SQ` (2022–2026).

---
---

# Copernicus Marine (CMEMS) — ARCO datasets

Verified from the public STAC catalog (`https://stac.marine.copernicus.eu/metadata/`) on
**2026-08-12**. 307 products total; the global ones relevant here are below.

All of these are **ARCO Zarr** — open them lazily with `copernicusmarine.open_dataset()`
and they behave exactly like the ERDDAP datasets above, no download step.

**Dataset IDs below have their `_YYYYMM` version suffix stripped** (e.g. the catalog lists
`..._P1D_202411`; you pass `..._P1D`). The toolbox resolves to the newest version. Pass
`dataset_version="202411"` explicitly if you need a reproducible pin.

Access needs a free account (`copernicusmarine login`, once).

## C1. Sea level + geostrophic currents ← **the gap-filler**

| Dataset ID | Variables | Grid res | Effective res | Frequency | Time frame | Type | Notes |
|---|---|---|---|---|---|---|---|
| `cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D` | `sla`, `adt`, `ugos`, `vgos`, `ugosa`, `vgosa`, `err_sla`, `err_ugosa`, `err_vgosa`, `flag_ice`, `tpa_correction` | **0.125°** | **~200 km** midlat | daily | **1993-01-01 → 2026-01-16** | L4 gap-free | **The one you want.** DUACS all-sat reprocessed. 33 continuous years |
| `cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.125deg_P1D` | same minus `tpa_correction` | 0.125° | ~200 km midlat | daily | 2024-07-01 → **2026-08-12** | L4 gap-free | **NRT twin.** Overlaps MY by 18 months — stitch inside the overlap |
| `cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1M-m` | `sla` only | 0.125° | ~200 km midlat | monthly | 1993-01-01 → 2025-12-01 | L4 gap-free | ⚠️ monthly drops the currents — only `sla` survives |
| `c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg_P1D` | same as MY daily | 0.25° | ~200 km midlat | daily | 1993-01-01 → 2026-01-16 | L4 gap-free | **Two-satellite** stable version. Use for *trends* — constant sampling avoids jumps as the constellation changed |

**All-sat vs two-sat:** all-sat (0.125°) resolves more eddy detail but its effective resolution
changes as satellites come and go — bad for trend analysis. Two-sat (0.25°) deliberately holds
the constellation fixed. Use all-sat for maps and eddies, two-sat for time series and trends.

## C2. Total surface currents — decomposed

| Dataset ID | Variables | Grid res | Effective res | Frequency | Time frame | Type | Notes |
|---|---|---|---|---|---|---|---|
| `cmems_obs-mob_glo_phy-cur_my_0.25deg_P1D-m` | `uo`/`vo` (total), `ugos`/`vgos` (geostrophic), `ue`/`ve` (Ekman), `utide`/`vtide` (tidal), + `err_*` for each | 0.25° | ~200 km | hourly / daily / monthly | **1993-01-01 → 2026-03-31** | L4 | **Often better than C1 for currents** — gives the components separately. Has a depth axis (surface + 15 m) |
| `cmems_obs-mob_glo_phy-cur_nrt_0.25deg_P1D-m` | same | 0.25° | ~200 km | hourly / daily / monthly | 2022-05-01 → 2026-08-11 | L4 | **NRT twin** |

## C3. Ocean color — longer than anything on CoastWatch

| Dataset ID | Variables | Grid res | Effective res | Frequency | Time frame | Type | Notes |
|---|---|---|---|---|---|---|---|
| `cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D` | `CHL`, `CHL_uncertainty`, `flags` | 4 km | coarser, **unquantified** | daily | **1997-09-04 → 2026-08-04** | L4 gap-free | **29 years of gap-free chl**, incl. the SeaWiFS era. Beats every CoastWatch chl on record length |
| `cmems_obs-oc_glo_bgc-plankton_nrt_l4-gapfree-multi-4km_P1D` | same | 4 km | coarser, **unquantified** | daily | 2023-10-01 → 2026-08-11 | L4 gap-free | **NRT twin**, 1 d lag |
| `cmems_obs-oc_glo_bgc-plankton_my_l4-multi-climatology-4km_P1D` | `CHL_mean`, `CHL_median`, `CHL_std`, `CHL_min`, `CHL_max`, `CHL_count`, `CHL_percentile_3`, `CHL_percentile_97` | 4 km | ~10 km | **day-of-year** | climatology | L4 climatology | **The chl climatology CoastWatch doesn't have.** Time axis is a dummy year (2008) = day-of-year 1–366 |
| `cmems_obs-oc_glo_bgc-transp_my_l4-gapfree-multi-4km_P1D` | `KD490`, `ZSD` (Secchi depth), + uncertainties | 4 km | coarser, **unquantified** | daily | 1997-09-04 → 2026-08-04 | L4 gap-free | Gap-free Kd490 **and Secchi depth** |
| `cmems_obs-oc_glo_bgc-pp_my_l4-multi-4km_P1M` | `PP`, `PP_uncertainty` | 4 km | ≈ grid | monthly | 1997-09-01 → 2026-07-01 | L4 derived | Primary production |
| `cmems_obs-oc_glo_bgc-optics_my_l4-multi-4km_P1M` | `BBP`, `CDM`, + uncertainties | 4 km | ≈ grid | monthly | 1997-09-01 → 2026-07-01 | L4 | Backscatter + colored dissolved/detrital matter |

## C4. SST, wind, salinity, waves

| Dataset ID | Variables | Grid res | Effective res | Frequency | Time frame | Type | Notes |
|---|---|---|---|---|---|---|---|
| `METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2` | `analysed_sst`, `analysis_error`, `mask`, `sea_ice_fraction` | 0.05° | coarser, **unquantified** | daily | 2007-01-01 → 2026-08-11 | L4 gap-free | OSTIA. Same variables as MUR |
| `C3S-GLO-SST-L4-REP-OBS-SST` | `analysed_sst`, `analysed_st`, `analysis_error_sst`, `mask`, `sea_ice_fraction` | 0.05° | coarser, **unquantified** | daily | 1982-01-01 → **2024-12-31** | L4 gap-free | Reprocessed. ⚠️ ends 2024 |
| `ESACCI-GLO-SST-L4-REP-OBS-SST` | `analysed_sst`, `analysed_sst_uncertainty`, `mask`, `sea_ice_fraction` | 0.05° | coarser, **unquantified** | daily | 1980-01-01 → **2021-12-31** | L4 gap-free | ⚠️ ends 2021. Climate record only |
| `cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H` | `eastward_wind`, `northward_wind`, `eastward_stress`, `northward_stress`, `wind_speed`, `air_density`, + bias/sdd for each | 0.125° | coarser, **unquantified** | **hourly** | 2007-01-11 → 2026-04-20 | L4 | Far finer than CoastWatch ASCAT (0.333°, 4-hourly) |
| `cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H` | same | 0.125° | coarser, **unquantified** | hourly | 2020-07-01 → 2026-08-11 | L4 | **NRT twin** |
| `cmems_obs-mob_glo_phy-sss_my_multi-oi_P1W` | `sss`, `sst`, `CT`, `SA`, `rho`, `spiciness0`, `alpha`, `beta`, `ice_mask`, `sss_corr_smos`, `sss_corr_smap`, `sss_isas` | 0.2° | coarser (L-band footprint) | **weekly** | 2010-06-03 → **2025-06-26** | L4 | Multi-sensor SSS + derived TS quantities. ⚠️ ~14 mo stale |
| `cmems_obs-wave_glo_phy-swh_my_multi-l4-0.5deg_P1D-i` | `VAVH_INST`, `VAVH_INST_FLAG` | 0.5° | coarser (altimeter) | daily | 2002-01-15 → **2020-12-31** | L4 | ⚠️ ends 2020. Use WW3 on CoastWatch instead |

## C5. Models (not satellite — 3D with depth)

| Dataset ID | Variables | Grid res | Effective res | Frequency | Time frame | Type | Notes |
|---|---|---|---|---|---|---|---|
| `cmems_mod_glo_phy_my_0.083deg_P1D-m` | `thetao`, `so`, `uo`, `vo`, `zos`, `mlotst`, `bottomT`, `siconc`, `sithick`, `usi`, `vsi` | 1/12° (~8 km) | coarser (model) | daily / monthly | **1993-01-01 → 2026-06-23** | reanalysis | **GLORYS.** Full 3D (depth axis). Everything bundled in one dataset |
| `cmems_mod_glo_phy_anfc_0.083deg_P1D-m` | `zos`, `mlotst`, `siconc`, `sithick`, `sob`, `tob`, `ist`, `pbo`, + sea-ice fields | 1/12° | coarser (model) | daily | 2022-06-01 → 2026-08-21 (forecast) | analysis+forecast | ⚠️ **`thetao`/`so`/`uo`/`vo` are NOT here** — the forecast product splits them into separate datasets (`cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m`, `-so_`, `-cur_`) |
| `cmems_mod_glo_phy_my_0.083deg-climatology_P1M-m` | monthly climatology of the above | 1/12° | coarser (model) | month-of-year | climatology | reanalysis | GLORYS climatology |

## CMEMS vs CoastWatch — where each wins

| Need | Winner | Why |
|---|---|---|
| SSH / geostrophic currents | **CMEMS** | CoastWatch has no continuous record; CMEMS has 1993→present |
| Long gap-free chlorophyll | **CMEMS** | 1997→present vs CoastWatch's 2018→present (DINEOF) |
| Chlorophyll climatology | **CMEMS** | CoastWatch has none |
| **Highest-res** chlorophyll | **CoastWatch** | 2 km DINEOF vs CMEMS 4 km |
| Highest-res SST | **CoastWatch** | MUR 0.01° (~1 km) vs CMEMS 0.05° |
| Longest continuous SST | **CoastWatch** | OISST 1981→present in one dataset; CMEMS reprocessed streams end 2021/2024 |
| PIC / POC / fluorescence | **CoastWatch** | CMEMS doesn't carry them (has `BBP`/`CDM` instead) |
| Wind | **CMEMS** | 0.125° hourly vs 0.333° 4-hourly |
| 3D T/S/currents with depth | **CMEMS** | CoastWatch is surface-only |
| No account needed | **CoastWatch** | CMEMS requires a free login |

## C6. Locations (ARCO Zarr URLs)

**No `copernicusmarine` dependency and no login needed.** The ARCO Zarr stores are readable
anonymously over plain HTTPS. They are **Zarr v2**, and zarr-python 3 probes for v3 first and
treats the resulting 403 as fatal — so you must pass `zarr_format=2`:

```python
reader_kwargs={"zarr_format": 2, "chunks": {}}
```

⚠️ **Do not add `"engine": "zarr"`.** `ocean_skill.build._reader_for` sets the engine from
the `.zarr` suffix, and passing it again raises
`TypeError: got multiple values for keyword argument 'engine'`. `chunks={}` *is* right here,
unlike the ERDDAP entries above: a Zarr store carries its own chunk encoding, so `{}`
resolves to the store's real chunking rather than to one chunk.

All 22 URLs below were opened and verified on 2026-08-12; time ranges in the tables above come
from these stores directly.

**The bucket number varies per product** (`mdl-arco-time-013`, `-025`, `-037`, `-043`, `-045`,
`-062`, …) — there is no constructible pattern. URLs must be read from the STAC catalog. See
the discovery helper at the end.

```python
CMEMS = {
  # --- C1 sea level + geostrophic currents ---
  "ssh_duacs_my_daily":      "https://s3.waw3-1.cloudferro.com/mdl-arco-time-045/arco/SEALEVEL_GLO_PHY_L4_MY_008_047/cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D_202411/timeChunked.zarr",
  "ssh_duacs_my_monthly":    "https://s3.waw3-1.cloudferro.com/mdl-arco-time-045/arco/SEALEVEL_GLO_PHY_L4_MY_008_047/cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1M-m_202411/timeChunked.zarr",
  "ssh_duacs_nrt_daily":     "https://s3.waw3-1.cloudferro.com/mdl-arco-time-045/arco/SEALEVEL_GLO_PHY_L4_NRT_008_046/cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.125deg_P1D_202506/timeChunked.zarr",
  "ssh_duacs_twosat_daily":  "https://s3.waw3-1.cloudferro.com/mdl-arco-time-045/arco/SEALEVEL_GLO_PHY_CLIMATE_L4_MY_008_057/c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg_P1D_202411/timeChunked.zarr",
  # --- C2 total surface currents ---
  "cur_multiobs_my_daily":   "https://s3.waw3-1.cloudferro.com/mdl-arco-time-037/arco/MULTIOBS_GLO_PHY_MYNRT_015_003/cmems_obs-mob_glo_phy-cur_my_0.25deg_P1D-m_202411/timeChunked.zarr",
  "cur_multiobs_nrt_daily":  "https://s3.waw3-1.cloudferro.com/mdl-arco-time-039/arco/MULTIOBS_GLO_PHY_MYNRT_015_003/cmems_obs-mob_glo_phy-cur_nrt_0.25deg_P1D-m_202411/timeChunked.zarr",
  # --- C3 ocean color ---
  "chl_gapfree_my_daily":    "https://s3.waw3-1.cloudferro.com/mdl-arco-time-043/arco/OCEANCOLOUR_GLO_BGC_L4_MY_009_104/cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D_202603/timeChunked.zarr",
  "chl_gapfree_nrt_daily":   "https://s3.waw3-1.cloudferro.com/mdl-arco-time-044/arco/OCEANCOLOUR_GLO_BGC_L4_NRT_009_102/cmems_obs-oc_glo_bgc-plankton_nrt_l4-gapfree-multi-4km_P1D_202311/timeChunked.zarr",
  "chl_climatology_doy":     "https://s3.waw3-1.cloudferro.com/mdl-arco-time-043/arco/OCEANCOLOUR_GLO_BGC_L4_MY_009_104/cmems_obs-oc_glo_bgc-plankton_my_l4-multi-climatology-4km_P1D_202311/timeChunked.zarr",
  "transp_gapfree_my_daily": "https://s3.waw3-1.cloudferro.com/mdl-arco-time-044/arco/OCEANCOLOUR_GLO_BGC_L4_MY_009_104/cmems_obs-oc_glo_bgc-transp_my_l4-gapfree-multi-4km_P1D_202603/timeChunked.zarr",
  "pp_my_monthly":           "https://s3.waw3-1.cloudferro.com/mdl-arco-time-043/arco/OCEANCOLOUR_GLO_BGC_L4_MY_009_104/cmems_obs-oc_glo_bgc-pp_my_l4-multi-4km_P1M_202603/timeChunked.zarr",
  "optics_my_monthly":       "https://s3.waw3-1.cloudferro.com/mdl-arco-time-043/arco/OCEANCOLOUR_GLO_BGC_L4_MY_009_104/cmems_obs-oc_glo_bgc-optics_my_l4-multi-4km_P1M_202603/timeChunked.zarr",
  # --- C4 SST / wind / salinity / waves ---
  "sst_ostia_nrt_daily":     "https://s3.waw3-1.cloudferro.com/mdl-arco-time-045/arco/SST_GLO_SST_L4_NRT_OBSERVATIONS_010_001/METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2/timeChunked.zarr",
  "sst_c3s_rep_daily":       "https://s3.waw3-1.cloudferro.com/mdl-arco-time-046/arco/SST_GLO_SST_L4_REP_OBSERVATIONS_010_024/C3S-GLO-SST-L4-REP-OBS-SST_202506/timeChunked.zarr",
  "sst_esacci_rep_daily":    "https://s3.waw3-1.cloudferro.com/mdl-arco-time-046/arco/SST_GLO_SST_L4_REP_OBSERVATIONS_010_024/ESACCI-GLO-SST-L4-REP-OBS-SST_202603/timeChunked.zarr",
  "wind_my_hourly":          "https://s3.waw3-1.cloudferro.com/mdl-arco-time-062/arco/WIND_GLO_PHY_L4_MY_012_006/cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H_202211/timeChunked.zarr",
  "wind_nrt_hourly":         "https://s3.waw3-1.cloudferro.com/mdl-arco-time-050/arco/WIND_GLO_PHY_L4_NRT_012_004/cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H_202207/timeChunked.zarr",
  "sss_my_weekly":           "https://s3.waw3-1.cloudferro.com/mdl-arco-time-039/arco/MULTIOBS_GLO_PHY_SSS_L4_MY_015_015/cmems_obs-mob_glo_phy-sss_my_multi-oi_P1W_202406/timeChunked.zarr",
  "swh_my_daily":            "https://s3.waw3-1.cloudferro.com/mdl-arco-time-046/arco/WAVE_GLO_PHY_SWH_L4_MY_014_007/cmems_obs-wave_glo_phy-swh_my_multi-l4-0.5deg_P1D-i_202411/timeChunked.zarr",
  # --- C5 models ---
  "glorys_my_daily":         "https://s3.waw3-1.cloudferro.com/mdl-arco-time-025/arco/GLOBAL_MULTIYEAR_PHY_001_030/cmems_mod_glo_phy_my_0.083deg_P1D-m_202311/timeChunked.zarr",
  "glorys_climatology":      "https://s3.waw3-1.cloudferro.com/mdl-arco-time-026/arco/GLOBAL_MULTIYEAR_PHY_001_030/cmems_mod_glo_phy_my_0.083deg-climatology_P1M-m_202311/timeChunked.zarr",
  "glo_anfc_daily":          "https://s3.waw3-1.cloudferro.com/mdl-arco-time-013/arco/GLOBAL_ANALYSISFORECAST_PHY_001_024/cmems_mod_glo_phy_anfc_0.083deg_P1D-m_202406/timeChunked.zarr",
}
```

### Nickname → dataset ID (join key back to tables C1–C5)

| Nickname | Dataset ID | Verified extent |
|---|---|---|
| `ssh_duacs_my_daily` | `cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D` | 1993-01-01→2026-01-16 (12069) |
| `ssh_duacs_my_monthly` | `..._P1M-m` | 1993-01-01→2025-12-01 (396) |
| `ssh_duacs_nrt_daily` | `cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.125deg_P1D` | 2024-07-01→2026-08-12 (773) |
| `ssh_duacs_twosat_daily` | `c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg_P1D` | 1993-01-01→2026-01-16 (12069) |
| `cur_multiobs_my_daily` | `cmems_obs-mob_glo_phy-cur_my_0.25deg_P1D-m` | 1993-01-01→2026-03-31 (12143) |
| `cur_multiobs_nrt_daily` | `cmems_obs-mob_glo_phy-cur_nrt_0.25deg_P1D-m` | 2022-05-01→2026-08-11 (1564) |
| `chl_gapfree_my_daily` | `cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D` | 1997-09-04→2026-08-04 (10562) |
| `chl_gapfree_nrt_daily` | `cmems_obs-oc_glo_bgc-plankton_nrt_l4-gapfree-multi-4km_P1D` | 2023-10-01→2026-08-11 (1046) |
| `chl_climatology_doy` | `cmems_obs-oc_glo_bgc-plankton_my_l4-multi-climatology-4km_P1D` | 366 steps (dummy year 2008) |
| `transp_gapfree_my_daily` | `cmems_obs-oc_glo_bgc-transp_my_l4-gapfree-multi-4km_P1D` | 1997-09-04→2026-08-04 (10562) |
| `pp_my_monthly` | `cmems_obs-oc_glo_bgc-pp_my_l4-multi-4km_P1M` | 1997-09-01→2026-07-01 (347) |
| `optics_my_monthly` | `cmems_obs-oc_glo_bgc-optics_my_l4-multi-4km_P1M` | 1997-09-01→2026-07-01 (347) |
| `sst_ostia_nrt_daily` | `METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2` | 2007-01-01→2026-08-11 (7163) |
| `sst_c3s_rep_daily` | `C3S-GLO-SST-L4-REP-OBS-SST` | 1982-01-01→2024-12-31 (15706) |
| `sst_esacci_rep_daily` | `ESACCI-GLO-SST-L4-REP-OBS-SST` | 1980-01-01→2021-12-31 (15341) |
| `wind_my_hourly` | `cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H` | 2007-01-11→2026-04-20 (168960) |
| `wind_nrt_hourly` | `cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H` | 2020-07-01→2026-08-11 (53592) |
| `sss_my_weekly` | `cmems_obs-mob_glo_phy-sss_my_multi-oi_P1W` | 2010-06-03→2025-06-26 (787) |
| `swh_my_daily` | `cmems_obs-wave_glo_phy-swh_my_multi-l4-0.5deg_P1D-i` | 2002-01-15→2020-12-31 (6926) |
| `glorys_my_daily` | `cmems_mod_glo_phy_my_0.083deg_P1D-m` | 1993-01-01→2026-06-23 (12227) |
| `glorys_climatology` | `cmems_mod_glo_phy_my_0.083deg-climatology_P1M-m` | 12 steps (dummy year 2004) |
| `glo_anfc_daily` | `cmems_mod_glo_phy_anfc_0.083deg_P1D-m` | 2022-06-01→2026-08-21 (1543) |

### Building catalogs from these

```python
# Copernicus — URLs already resolved above
build_catalog(CMEMS, "catalogs/copernicus.yaml", title="Copernicus Marine",
              reader_kwargs={"zarr_format": 2, "chunks": {}})

# CoastWatch ERDDAP — URLs ARE pattern-constructible, unlike CMEMS
ERDDAP = "https://coastwatch.pfeg.noaa.gov/erddap"
IDS = ["jplMURSST41", "jplMURSST41mday", "jplMURSST41clim", "nesdisVHNSQchlaDaily",
       "noaacwNPPN20S3ASCIDINEOF2kmDaily", "ncdcOisst21Agg", "NOAA_DHW"]
build_catalog({i: f"{ERDDAP}/griddap/{i}" for i in IDS},
              "catalogs/coastwatch.yaml", title="NOAA CoastWatch",
              reader_kwargs={"chunks": "auto"})
```

`{server}/griddap/{dataset_id}` opens directly with `xr.open_dataset` — no `intake_erddap`
needed, which sidesteps the broken search path entirely (see notes below).

### Building from a named list instead (shorter)

Once you know which datasets you want, none of the discovery above applies. A CMEMS
entry needs a **pair** — the STAC path is ``/{product_id}/{dataset_id}/``, so a
dataset ID alone cannot be resolved. (CoastWatch needs only the ID, because its URL
is ``{server}/griddap/{id}``.)

All 22 pairs below were verified to resolve on 2026-08-13.

```python
DATASETS = {   # nickname: (product_id, dataset_id)
    "ssh_duacs_my_daily":      ("SEALEVEL_GLO_PHY_L4_MY_008_047",
                               "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D"),
    "ssh_duacs_my_monthly":    ("SEALEVEL_GLO_PHY_L4_MY_008_047",
                               "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1M-m"),
    "ssh_duacs_nrt_daily":     ("SEALEVEL_GLO_PHY_L4_NRT_008_046",
                               "cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.125deg_P1D"),
    "ssh_duacs_twosat_daily":  ("SEALEVEL_GLO_PHY_CLIMATE_L4_MY_008_057",
                               "c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg_P1D"),
    "cur_multiobs_my_daily":   ("MULTIOBS_GLO_PHY_MYNRT_015_003",
                               "cmems_obs-mob_glo_phy-cur_my_0.25deg_P1D-m"),
    "cur_multiobs_nrt_daily":  ("MULTIOBS_GLO_PHY_MYNRT_015_003",
                               "cmems_obs-mob_glo_phy-cur_nrt_0.25deg_P1D-m"),
    "chl_gapfree_my_daily":    ("OCEANCOLOUR_GLO_BGC_L4_MY_009_104",
                               "cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D"),
    "chl_gapfree_nrt_daily":   ("OCEANCOLOUR_GLO_BGC_L4_NRT_009_102",
                               "cmems_obs-oc_glo_bgc-plankton_nrt_l4-gapfree-multi-4km_P1D"),
    "chl_climatology_doy":     ("OCEANCOLOUR_GLO_BGC_L4_MY_009_104",
                               "cmems_obs-oc_glo_bgc-plankton_my_l4-multi-climatology-4km_P1D"),
    "transp_gapfree_my_daily": ("OCEANCOLOUR_GLO_BGC_L4_MY_009_104",
                               "cmems_obs-oc_glo_bgc-transp_my_l4-gapfree-multi-4km_P1D"),
    "pp_my_monthly":           ("OCEANCOLOUR_GLO_BGC_L4_MY_009_104",
                               "cmems_obs-oc_glo_bgc-pp_my_l4-multi-4km_P1M"),
    "optics_my_monthly":       ("OCEANCOLOUR_GLO_BGC_L4_MY_009_104",
                               "cmems_obs-oc_glo_bgc-optics_my_l4-multi-4km_P1M"),
    "sst_ostia_nrt_daily":     ("SST_GLO_SST_L4_NRT_OBSERVATIONS_010_001",
                               "METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2"),
    "sst_c3s_rep_daily":       ("SST_GLO_SST_L4_REP_OBSERVATIONS_010_024",
                               "C3S-GLO-SST-L4-REP-OBS-SST"),
    "sst_esacci_rep_daily":    ("SST_GLO_SST_L4_REP_OBSERVATIONS_010_024",
                               "ESACCI-GLO-SST-L4-REP-OBS-SST"),
    "wind_my_hourly":          ("WIND_GLO_PHY_L4_MY_012_006",
                               "cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H"),
    "wind_nrt_hourly":         ("WIND_GLO_PHY_L4_NRT_012_004",
                               "cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H"),
    "sss_my_weekly":           ("MULTIOBS_GLO_PHY_SSS_L4_MY_015_015",
                               "cmems_obs-mob_glo_phy-sss_my_multi-oi_P1W"),
    "swh_my_daily":            ("WAVE_GLO_PHY_SWH_L4_MY_014_007",
                               "cmems_obs-wave_glo_phy-swh_my_multi-l4-0.5deg_P1D-i"),
    "glorys_my_daily":         ("GLOBAL_MULTIYEAR_PHY_001_030",
                               "cmems_mod_glo_phy_my_0.083deg_P1D-m"),
    "glorys_climatology":      ("GLOBAL_MULTIYEAR_PHY_001_030",
                               "cmems_mod_glo_phy_my_0.083deg-climatology_P1M-m"),
    "glo_anfc_daily":          ("GLOBAL_ANALYSISFORECAST_PHY_001_024",
                               "cmems_mod_glo_phy_anfc_0.083deg_P1D-m"),
}

def arco_url(product_id, dataset_id, chunking="timeChunked"):
    """Current ARCO URL. Resolved rather than hardcoded because CMEMS embeds a
    version tag (..._202411) that changes when a product is republished."""
    col = fetch(f"{STAC}/{product_id}/product.stac.json")
    versioned = next(l["href"].split("/")[0] for l in col["links"]
                     if l.get("rel") == "item"
                     and l["href"].split("/")[0].rsplit("_", 1)[0] == dataset_id)
    item = fetch(f"{STAC}/{product_id}/{versioned}/dataset.stac.json")
    return next(a["href"] for a in item["assets"].values()
                if a.get("type") == "application/vnd+zarr"
                and a["href"].endswith(f"{chunking}.zarr"))

urls = {k: arco_url(*v) for k, v in DATASETS.items()}
build_catalog(urls, "catalogs/copernicus.yaml", title="Copernicus Marine",
              reader_kwargs={"zarr_format": 2, "chunks": {}}, skip_errors=True)
```

The hardcoded URLs above skip even this fetch, but bake in the version tag — treat a
404 from one as "re-run the resolver", not "the dataset is gone".

### Refreshing the URLs

The `_YYYYMM` version tag is baked into each URL, so these pin a specific version — reproducible,
but they go stale when CMEMS republishes. Re-derive with:

```python
import json, urllib.request

def cmems_arco_url(product_id, dataset_id, chunking="timeChunked"):
    """Resolve a CMEMS ARCO Zarr URL from the public STAC catalog. No auth needed."""
    url = f"https://stac.marine.copernicus.eu/metadata/{product_id}/{dataset_id}/dataset.stac.json"
    return json.load(urllib.request.urlopen(url))["assets"][chunking]["href"]

def cmems_datasets(product_id):
    """List (title, dataset_id) for every dataset in a CMEMS product."""
    url = f"https://stac.marine.copernicus.eu/metadata/{product_id}/product.stac.json"
    d = json.load(urllib.request.urlopen(url))
    return [(l.get("title"), l["href"].split("/")[0])
            for l in d["links"] if l.get("rel") == "item"]
```

`geoChunked` is the same data chunked for spatial subsetting — use it for maps/regions,
`timeChunked` for time series at a point.

