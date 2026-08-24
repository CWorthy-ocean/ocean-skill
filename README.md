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
osk.find(variable="nitrate").map()    # ...and where the matches are, on one map
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

A `select` that narrows both models to one **lon/lat** works the same way a station
reference does — nothing left to draw a map of, so it draws as lines, `over="time"`
implied the same way a mooring's `featureType` implies it:

```python
point = osk.compare(
    reference="run_baseline", test="run_new", variables=[NITRATE],
    select={"lon": -144.25, "lat": 50.0, "time": slice("2012-01", "2012-12")},
)
point.plot()                          # reference solid, test dashed, at one place
point[0].family_reason                # "the select narrows the reference to one position"
```

The **reference's** grid decides the exact position — its nearest cell to the request,
or its interpolated value with `method="bilinear"` — and the *test* is sampled there
too, so the pair is genuinely co-located rather than each lane picking its own
nearest cell to the raw request (which is what two independent `select={"test": ...,
"reference": ...}` positions would do, and is still available when two real, possibly
different, positions are the point — see `docs/plot_styling_reference.md`).

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

A `select` that narrows both horizontal axes to one **lon/lat** instead draws as a
line over whatever axis survives — never a separate call, still `.plot()`:

```python
osk.field(
    "run_new", NITRATE,
    select={"lon": -144.25, "lat": 50.0, "time": slice("2012-01", "2012-12")},
).plot()                              # one solid line, no reference to compare against
```

Which of the two — map panels or a line — `.plot()` draws is read off the data's own
shape (`Field.family`/`family_reason`), the same rule `compare()` follows between a
score map and a line comparison; there is no argument that picks one over the other.

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

A surface of constant depth is not always the most meaningful slice through a
stratified column — water masses move along surfaces of constant density, not
constant depth. `select={"sigma0": ...}` asks for an isopycnal instead, faceted the
same way a list of depths is:

```python
osk.field(
    "GOM_bgc",
    NITRATE,
    select={"time": slice("2012-01", "2012-06"), "sigma0": [24.5, 25.5, 26.5]},
    aggregate={"time": {"resample": "1MS", "reduce": "mean"}},
).plot()                       # 3 isopycnals x 6 months, rows labelled "σ₀ = 24.5 kg/m³"
```

ROMS sources only (an isopycnal is read off the model's own temperature and
salinity), and not alongside a `"depth"`/`"Z"` select key or `compare()`'s `depths=` —
pick one vertical request. `sigma0` is potential density *anomaly* (density minus
1000 kg/m3, typically 20-28 for seawater); there is no `"rho"`/`"density"` alias, since
ROMS's own `rho` output is in-situ density and would silently name a different surface
at any real depth.

## Variable specs

A plain name is the common case, but `variable=`/`variables=` accepts three other
shapes for the cases a name alone can't cover:

**A combination** sums (or differences, multiplies, divides) several variables into
one field — MARBL splits chlorophyll into three phytoplankton components, MODIS ships
the total under one CF name:

```python
osk.compare(
    reference="modis_chl", test="GOM_bgc",
    variables=[{"sum": ["spChl", "diatChl", "diazChl"],
                "standard_name": "mass_concentration_of_chlorophyll_a_in_sea_water"}],
)
```

**A registered calculator** is for a genuine formula rather than arithmetic — mixed
layer depth, computed from temperature and salinity by a chosen criterion:

```python
osk.field("GOM_bgc", {"calculate": "mld", "method": "density_threshold"})
```

Any function can be plugged in this way, from a notebook, with no codebase change —
`register_calculator` is public API, not an internal detail:

```python
from ocean_skill.operators import register_calculator

@register_calculator("eke")
def eddy_kinetic_energy(ds, **kwargs):
    ...          # read whatever ds carries, return a DataArray
    return da

osk.field("GOM_bgc", {"calculate": "eke"})
```

**A pair-spec** — `{"test": <spec>, "reference": <spec>}` — is for the case a
`Comparison` cannot otherwise express: the two sides need genuinely different recipes
for the same quantity. A model computes mixed layer depth from temperature and
salinity; an observational climatology (e.g. the Holte & Talley Argo product) already
ships it as a plain field:

```python
osk.compare(
    reference="holte_talley_mld_clim", test="GOM_bgc",
    variables=[{
        "test": {"calculate": "mld", "method": "density_threshold"},
        "reference": "mld_dt_mean",
        "standard_name": "ocean_mixed_layer_thickness",
    }],
    aggregate={"time": "mean"},
)
```

`standard_name` is optional but worth setting whenever you know it: it names the
figure precisely, and — since the two sides of a pair-spec can resolve to genuinely
different CF names with nothing else checking that they don't — its absence is also
what turns a mismatch between the two recipes into a warning rather than a number that
looks right and isn't. `Comparison`/`compare()` accept a pair-spec; `Field` does not
(there is no second lane to give the other half of the pair to).

`select` and `aggregate` accept the same `{"test": ..., "reference": ...}` spelling,
for when the two lanes' *axes* don't match, not just their variable recipe — a model
spanning several years against a WOA monthly climatology (its `time` read with
`decode_times=False`, since a climatology has no calendar year to decode):

```python
osk.compare(
    reference="woa23_nitrate_month01", test="GOM_bgc", variables=["nitrate"],
    aggregate={"test": {"time": {"groupby": "month", "reduce": "mean"}},
               "reference": {"time": "mean"}},
    select={"test": {"month": 1}, "reference": {}},
)
```

The model needs a monthly climatology of its own, then one month picked out of it;
the reference just needs its one time step meaned away — a shared `aggregate` would
try to `groupby` the reference's undecoded numeric time (no calendar to group by),
and a shared `select={"time": "2010-01"}` fails the same way trying to match a date
string against it. `select={"month": 1}` is deferred and retried once the aggregate
above has created that axis, since it doesn't exist before then. `depths=`/
`select={"depth": ...}` sugar still applies to both sides of a pair-spec select at
once.

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
