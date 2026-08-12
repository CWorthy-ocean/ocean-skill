# ocean-skill

Modular, composable **model–data validation and analysis** for ocean models.

`ocean-skill` reads any two things through [intake](https://intake.readthedocs.io)
catalogs — labeling which is the `reference` (observations, usually) and which is the
`test` (model output) — then **aligns** them in space and time, computes **skill
metrics**, and **plots** the comparison, with a switch between static and interactive
output. It is model-agnostic; ROMS specifics (kerchunk ingest, s-coordinate depth
transform, curvilinear regridding) live behind a thin adapter.

## Concepts

- **reference / test** — roles assigned *at compare time* (not in the catalog). They set
  the difference direction (`test − reference`), alignment direction (test → reference),
  and styling (reference solid, test dashed).
- **entities** — a thing-being-compared is a `(source, variable, selection)` over time /
  depth / location. Any two entities can be compared; one alone just plots.
- **featureType drives the plot recipe** — `grid`/`timeSeries`/`profile`/`trajectory`/…
  (declared in the catalog) picks the plot family, axes, and marks by default.

## Quick start

```python
import ocean_skill as osk

osk.catalogs                          # every source discovered, across all catalogs
osk.find(variable="nitrate")          # search by variable, bbox, time, name, free text
osk.describe("woa23_nitrate_month01") # full metadata for one source

NITRATE = "mole_concentration_of_nitrate_in_sea_water"

comparison = osk.compare(
    reference="woa23_nitrate_month01",   # observations
    test="GOM_bgc",                      # model output
    variables=[NITRATE],
    depths=["surface"],
)

comparison.plot()                     # test | reference | difference maps
comparison.plot(renderer="holoviews") # the same plot, interactive
comparison.metrics()                  # bias, rmse, corr, sigma_ratio, n
```

A set of comparisons stacks as rows, or plays as frames — the same items either way
(see [docs/movies.md](docs/movies.md)):

```python
depths = osk.compare(reference="woa23_nitrate_month01", test="GOM_bgc",
                     variables=[NITRATE], depths=("surface", 50, 100))

depths.plot()                         # three rows, stacked
depths.movie(save="depths.mp4")       # the same three, played  (.gif also works)
depths.movie(renderer="holoviews")    # the same three, on a slider
```

Roles are set at compare time rather than in the catalog, so comparing two *models*
is the same call with a different `reference`:

```python
runs = osk.compare(reference="run_baseline", test="run_new", variables=[NITRATE])
runs.summary()                        # Taylor + target diagrams side by side
runs.write_metrics("metrics/")        # tidy CSV, one row per comparison
```

## Catalogs

Sources are described by [intake](https://intake.readthedocs.io) v2 catalogs, found
automatically along a search path where **later shadows earlier**:

```
1. ocean_skill/catalogs/     shipped defaults
2. ~/.ocean-skill/catalogs/  your machine
3. $OCEAN_SKILL_CATALOGS     a site or shared-cluster directory
4. ./catalogs/               this project
```

That ordering is what lets the same source name resolve to a colleague's shared
filesystem path on a cluster and to your own download on a laptop, without either of
you editing a tracked file. Remote files are cached under ocean-skill's cache
directory; set `$OCEAN_SKILL_DIR` to move it, or fsspec's own
`FSSPEC_SIMPLECACHE_CACHE_STORAGE` to override it outright.

## Layout

```
ocean_skill/       package (flat layout)
catalogs/          project-local intake catalogs (auto-discovered)
tests/             pytest suite
docs/              MyST / Jupyter Book docs + notebooks
examples/          short runnable scripts
suites/            declarative comparison suites (YAML)
environment.yml    conda scientific stack (source of truth)
```

## Install (development)

```bash
mamba env create -f environment.yml
mamba activate ocean-skill
pytest
```
