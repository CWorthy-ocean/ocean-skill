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

When the reference varies in **time** as well as space — a satellite record rather than a
climatology — averaging it away answers a different question. Name the axis instead and
every metric is computed cell by cell along it, giving a map per metric with its overall
value beside it (see [docs/skill_maps.md](docs/skill_maps.md)):

```python
scored = osk.compare(
    reference="modis_chl_daily", test="GOM_bgc", variables=["chlorophyll"],
    select={"time": slice("2012-01", "2012-06")},
    over="time",                      # keep the axis and score against it
)

scored.plot()                         # bias | crmsd | corr | sigma_ratio, as maps
scored[0].maps("rmse", "n")           # any registered metric, as an xr.Dataset
scored.metrics()                      # the same numbers over space *and* time
```

When the reference is a **place** rather than a field — a mooring, a station — there is no
map to draw: the comparison is that place through time, and it draws as lines. Nothing
extra to ask for, because the catalog already says so (`featureType: timeSeries`), and
each comparison records what decided its figure in `family_reason`:

```python
papa = osk.compare(
    reference="ooi-gp03flma-rim01-02-ctdmog040",   # a Station Papa CTD
    test="oceansoda_ethz",                          # a monthly gridded product
    variables=["temperature"],
    # the mooring samples every 15 minutes and the product is monthly, so say how the
    # mooring is binned — coarsening the *reference* is never done silently
    aggregate={"time": {"resample": "MS", "reduce": "mean"}},
)

papa.plot()                           # reference solid, test dashed, on one time axis
papa.plot(renderer="holoviews")       # the same lines, with hover
papa[0].family_reason                 # why this drew as lines and not as maps
```

The model lane is *sampled at the station* — the nearest cell by default,
`method="bilinear"` to interpolate to the position — and the offset between the station
and the grid is reported in the aligned result's attrs, since at 1° it is ~55 km.

Roles are set at compare time rather than in the catalog, so comparing two *models*
is the same call with a different `reference`:

```python
runs = osk.compare(reference="run_baseline", test="run_new", variables=[NITRATE])
runs.summary()                        # Taylor + target diagrams side by side
runs.write_metrics("metrics/")        # tidy CSV, one row per comparison
```

Comparisons you already have go onto one diagram without being rebuilt — pool any mix
of sets and single comparisons, which is often the only way to get them onto one figure
at all, since a pool may span several references or aggregations:

```python
osk.summary([nutrients, depths, one_off])            # Taylor + target, pooled
osk.summary({"hindcast": nutrients, "forecast": v2}) # or name the groups yourself
osk.summary(nutrients, kind="taylor")                # just the one diagram
pooled = nutrients + depths                          # a real set: .metrics(), .save()
```

Points are named by whatever varies across the pool (variable, depth, model,
reference), so two fan-outs that each called a point `surface` stay distinguishable —
and the comparisons themselves are untouched, keeping the labels their own figures
draw. Unlike `.plot()`, a pool may freely mix a station time series with a gridded
field: a metrics record is a handful of scalars either way, and both diagrams normalize
by the reference's standard deviation, so points are comparable across variables and
units.

One source alone just plots — `osk.field` is the same pipeline without a reference,
most useful when the reduction leaves a time axis standing, which becomes the panels:

```python
run = osk.field(
    "GOM_bgc",
    NITRATE,
    select={"time": slice("2012-01", "2012-06"), "depth": "surface"},
    aggregate={"time": {"resample": "1MS", "reduce": "mean"}},
)
run.plot()                            # six monthly means, one shared colour scale
run.movie(save="nitrate.mp4")         # the same six, played instead of laid out
run.movie(renderer="holoviews")       # the same six, stepped through on a slider
```

**Nothing is reduced unless you ask.** There is no default aggregation anywhere: omit
`aggregate` and every step of the selection survives as its own panel or frame. A
`compare()` needs a single map, so it will tell you to choose rather than average an axis
behind your back:

```python
osk.compare(reference="woa23_nitrate_month01", test="GOM_bgc", variables=[NITRATE])
# ValueError: the test lane ('GOM_bgc') still has time=124 beyond its horizontal axes...
#   aggregate={"time": "mean"}          one map, the mean over time
#   select={"time": <one value>}    or narrow it to a single value instead
```

`resample` gives consecutive periods (`Jan 2012`, `Feb 2012`, …); `{"groupby": "month"}`
gives a climatology (`Jan`, `Feb`, …, every January of the record in one panel). The
panels label themselves differently, so the two can't be confused on the page, and an
axis finer than its label says so — three days of one January are titled `2012-01-16`
rather than `Jan 2012` three times over. The figure's own title names the variable
(`nitrate`), since the panels say when but nothing else says what; `title=""` drops it. A
selection that starts or ends mid-period warns, since those panels average over part of
a month but are labelled like whole ones. The grid's orientation follows the domain's
aspect ratio — a wide box stacks down the page, a tall one spreads across it.

Ask for several depths and the months become the columns, the depths the rows:

```python
osk.field(
    "GOM_bgc",
    NITRATE,
    select={"time": slice("2012-01", "2012-06"), "depth": [0, 50, 100]},
    aggregate={"time": {"resample": "1MS", "reduce": "mean"}},
).plot()                              # 3 depths x 6 months, one colour scale per depth
```

Each depth row keeps its own colour scale, since nitrate at 100 m and at the surface
span unrelated ranges and one scale across both would flatten the shallow rows — pass
`shared_limits=True` if the levels you picked really do share a range.

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
