# intake v2 ("Take2") reference for ocean-skill

Distilled from the intake 2.0.9 source/docs, `axiom-data-science/cook-inlet-catalogs`,
and community catalogs (Beforerr, OpenADMET, eerie-project/DKRZ). All snippets verified
against an actual intake 2.0.9 install unless noted.

> **Provisional:** we may not stay on intake (evaluating alternatives). If we do, this is
> the canonical pattern. Either way, **catalogs are code-generated, never hand-written.**

## Version timeline
- intake **2.0.0 released 2024-02-02** (alphas from 2023-11). Anything before ~Nov 2023 is v1.
- **v1** = top-level `sources:` + `driver:`/`args:`, Jinja `{{ CATALOG_DIR }}`, `intake.open_catalog`.
- **v2** = `version: 2` + `data:`/`entries:`/`aliases:`, single-brace `{CATALOG_DIR}`,
  `intake.from_yaml_file` / `to_yaml_file`.

## Tell v1 from v2 at a glance
Fastest check: a `reader:` with a `module:Class` colon path + `output_instance:` ⇒ **v2**.
A `sources:` mapping with `driver:`/`args:` ⇒ **v1**.

## Anatomy of a v2 catalog
```yaml
version: 2                # REQUIRED (from_dict rejects != 2)
aliases:                  # friendly-name -> token; what list(cat)/tab-complete use
  woa23_nitrate_annual: woa23_nitrate_annual
data:                     # WHAT the data is (DataDescription): datatype + location
  6cb34b6490cbb4fc:       # key = CONTENT-DERIVED hash (dedups identical sources)
    datatype: intake.readers.datatypes:OpenDAP   # module:Class
    kwargs: {url: https://.../woa23_all_n00_01.nc, options: {}}
    metadata: {}
    user_parameters: {}
entries:                  # HOW to load it (ReaderDescription): reader + output type
  woa23_nitrate_annual:
    reader: intake.readers.readers:XArrayDatasetReader
    output_instance: xarray:Dataset
    kwargs:
      args: ['{data(6cb34b6490cbb4fc)}']   # reference into data: by token
      chunks: {}
      decode_times: false
    metadata: {featureType: grid, standard_names: {n_an: moles_of_nitrate_...}, ...}
    user_parameters: {}
metadata: {...}           # catalog-level (free-form)
user_parameters: {}       # catalog-global templated params
```
- **`data` vs `entries`**: one dataset definition (data), referenced by one-or-many readers
  (entries) via `{data(<token>)}`. Tokens are content hashes (deterministic, dedup) unless
  you name the entry (`cat["x"]=reader` makes the token `x`).
- On load, intake injects globals `CATALOG_PATH`, `CATALOG_DIR`, `STORAGE_OPTIONS`.
- **Quirk:** `cat[name]=reader` leaves `aliases` empty, so `list(cat)` shows nothing.
  Fix: `cat.aliases[name] = name` (then enumeration works).

## Datatypes and readers (import paths)
- Datatypes `intake.readers.datatypes:` — `CSV, Parquet, HDF5, NetCDF3, Zarr, GRIB2,
  OpenDAP, TIFF, GeoJSON, YAMLFile, Feather2, ...`. `FileData(url, storage_options=None, ...)`.
- Readers `intake.readers.readers:` — `PandasCSV`/`PandasParquet` (`pandas:DataFrame`),
  `XArrayDatasetReader`/`XArrayPatternReader` (`xarray:Dataset`), `PolarsFeather`,
  `YAMLCatalogReader` (`intake.readers.entry:Catalog`, for nested catalogs), etc.
- **Pick the datatype for the transport:** local/remote file ⇒ `HDF5`/`NetCDF3` (may need
  `engine="netcdf4"` to avoid the h5py path); **OPeNDAP/dodsC ⇒ `OpenDAP`** (routes to
  xarray's DAP engine — using `HDF5` makes it try fsspec HTTP and 404). `XArrayDatasetReader.implements`
  = {NetCDF3, HDF5, GRIB2, IcechunkRepo, Zarr, OpenDAP, TileDB}.

## Build + read (the only sane way to author)
```python
import intake
from intake.readers import datatypes, readers

data = datatypes.OpenDAP(url="https://.../woa23_all_n00_01.nc")
reader = readers.XArrayDatasetReader(data, decode_times=False, chunks={})
reader.metadata.update({"featureType": "grid", "standard_names": {"n_an": "..."}, ...})

cat = intake.entry.Catalog(metadata={"title": "..."})
cat["woa23_nitrate_annual"] = reader
cat.aliases["woa23_nitrate_annual"] = "woa23_nitrate_annual"   # for list(cat)
cat.to_yaml_file("catalogs/woa.catalog.yaml")

# consumer side
cat = intake.from_yaml_file("catalogs/woa.catalog.yaml")
list(cat)                              # ['woa23_nitrate_annual', ...]
ds = cat["woa23_nitrate_annual"].read()          # -> xr.Dataset
md = cat.entries["woa23_nitrate_annual"].metadata  # reliable metadata access
```
Access metadata via `cat.entries[name].metadata` — `.describe()` can hang in Take2.

## Custom reader (v2 way — subclass BaseReader)
```python
from intake.readers.readers import BaseReader

class PoochTarNetCDF(BaseReader):          # must live in an INSTALLED module (importable by qname)
    output_instance = "xarray:Dataset"     # REQUIRED
    def _read(self, url, known_hash=None, member_glob="*.nc", cache_dir=None, **kw):
        import fnmatch, pooch, xarray as xr
        paths = pooch.retrieve(url, known_hash=known_hash, processor=pooch.Untar(), path=cache_dir)
        paths = [p for p in paths if fnmatch.fnmatch(p, f"*{member_glob}")]
        return xr.open_mfdataset(paths, combine="by_coords", **kw)
```
The catalog references it by import path: `reader: ocean_skill.readers:PoochTarNetCDF`.
Override `_read`; set `output_instance`. (This is how pooch/tar/erddap-style access fits an
intake catalog — pooch lives *inside* the reader, the catalog names the reader.)

## user_parameters (templating) — the rough edge
```python
cat.extract_parameter(item=datatok, name="year", value="2020",
                      cls=SimpleUserParameter, dtype=str)   # url -> ".../data_{year}.csv"
cat.promote_parameter_name("year", level="cat")
cat(year="2021")["mydata"].read()          # override at read time via cat(param=...)
```
Serializes as `user_parameters: {year: {default: '2020', description: '', dtype: str}}`.
Overrides only feed reliably when the UP is at **catalog** scope; leaving it on a data
block leaks the override into the read function. Docs here are marked "to come."

## storage_options / fsspec caching
- `storage_options` goes on the **data** object: `datatypes.CSV(url=..., storage_options={"anon": True})`.
- `PandasCSV`/`Parquet`/`Dask` forward it (class attr `storage_options=True`); `XArray*`
  consume it internally (fsspec.open_files + zarr/kerchunk backend_kwargs).
- Caching = fsspec URL chaining in the url: `simplecache://::https://.../file.nc`, with
  `{"simplecache": {"cache_storage": <dir>, "same_names": True}}`.
- Catalog-file fsspec opts: `from_yaml_file(path, **storage_options)`.

## Known gotchas (from intake issues)
- `{CATALOG_DIR}` can get an fsspec protocol prepended on round-trip, breaking local-path
  readers (#894).
- user_parameters at read time is unintuitive; `cat(param=...)` scope confusion (#856, #868).
- Can't mutate a source in place like v1; use `Catalog.add_entry`/`delete` (#826).
- `list(cat)` empty unless `aliases` set (see quirk above).

## Worked example in this repo
`catalogs/woa.catalog.yaml` (WOA23 nitrate/phosphate/silicate/oxygen, annual 1°, via
OPeNDAP) — generated by the pattern above. `catalogs/... ROMS.yaml` (combined ROMS file,
`XArrayDatasetReader` + `engine: netcdf4`) is another. Both are v2.
