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
  the difference direction (`test − reference`), which lane moves during alignment (the
  finer one, whichever that is), and styling (reference solid, test dashed).
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
comparison.map_locations()            # where the comparison's data sits: the
                                       # selection over the model domain
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

`compare()` fans over **time** too — `times=` gives one comparison per bin instead of
one reduction kept standing, so the set above plots and plays month by month:

```python
months = osk.compare(reference="GOM_bgc_hindcast", test="GOM_bgc_forecast",
                     variables=[NITRATE],
                     times={"resample": "1MS", "reduce": "mean"})

months.plot()                         # one row per month actually present
months.movie(save="months.mp4")       # the same months, played
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
scored[0].pointwise_metrics("rmse", "n")  # any registered metric, as an xr.Dataset
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
    # the mooring samples every 15 minutes and the product is monthly. Left alone,
    # alignment coarsens the mooring into the product's own months itself and warns
    # about it; saying so here puts the choice on the record and runs quietly
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

`subtract_mean` removes a scalar offset (a sea-level datum, a model's drift) from
either lane, or both, before scoring — pool a comparison alongside its demeaned twin
to see the effect directly:

```python
raw = osk.compare(reference="tide_gauge", test="his", variables=["zeta"])
demeaned = osk.compare(reference="tide_gauge", test="his", variables=["zeta"],
                        subtract_mean=True)
(raw + demeaned).target()              # the demeaned point drops toward zero bias
```

The two land on the *same spot* on a Taylor diagram (its statistics are centred
moments, blind to a constant offset by construction) but apart on a target diagram,
whose whole axis is bias. What was removed is never drawn on the figure — it's in
`.metrics()`'s `subtracted_mean_test`/`subtracted_mean_reference` columns instead,
alongside an always-present `demeaned` column (`color_by="demeaned"` splits a mixed
pool the same way any other label dimension does).

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

**More than one variable on the same line plot** — pass a list instead of a name, and
`osk.field` fans it into a `FieldSet` (one `Field` per variable, sharing this same
`select`/`aggregate`), drawn on one figure:

```python
run = osk.field(
    "run_new", ["temperature", "salinity"],
    select={"lon": -144.25, "lat": 50.0, "time": slice("2012-01", "2012-12")},
)
run.plot()                            # 2 variables: one panel, the second on a
                                       # secondary y-axis
run.plot(secondary_y=False)           # ...or two stacked panels instead
run.plot(renderer="holoviews")        # same figure, interactive
```

The layout follows the series composition rule everywhere else in this package: one
variable overlays every source on one panel, two share a panel with the second on a
secondary axis by default, three or more each get their own row (see
`docs/plot_styling_reference.md`). There is no separate multi-variable option to
learn — the same `.plot()` keyword arguments (`rows=`, `cols=`, `secondary_y=`) apply.

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

**A single water column, read down instead of across** — pin `select` to one
`lon`/`lat` and leave depth standing (rather than pinning it to a scalar too, which
draws a `series` over time instead), and a field draws as a `profile`: value on x,
depth on y, inverted so the surface sits at the top and the seafloor at the bottom.

```python
osk.field(
    "run_new", NITRATE,
    select={"lon": -144.25, "lat": 50.0, "time": "2013-06-15",
            "depth": [0, 10, 25, 50, 100]},
).plot()                              # one cast, five interpolated levels
```

`select={"depth": "column"}` reaches every native model level instead of a fixed
list — the model's own resolution, honest about how coarse the water column
actually is (a ROMS grid, unlike an observational product, has no standard levels
of its own to fall back on). Several times at one place fan into one line per
cast, coloured/marked by time rather than by depth (depth is the axis itself
here, not a fact to tell lines apart by):

```python
osk.field(
    "run_new", NITRATE,
    select={"lon": -144.25, "lat": 50.0, "depth": "column",
            "time": ["2013-06-15", "2013-09-15"]},
).plot()                              # two casts overlaid, one legend entry each
```

Two variables at one cast merge onto one panel too, the same way `series` merges
two — just transposed, since a profile's value axis is x rather than y, so its
twin is a top x axis (`secondary_x`) rather than a right-hand y axis:

```python
osk.field(
    "run_new", ["temperature", "salinity"],
    select={"lon": -144.25, "lat": 50.0, "time": "2013-06-15", "depth": "column"},
).plot()                              # 2 variables: one panel, salinity on a top axis
osk.field(..., ...).plot(secondary_x=False)  # ...or two side-by-side columns instead
```

`compare()` draws the same way against a real profile — a WHOTS/Argo-style cast, or
any reference whose catalog `featureType` is `profile`/`timeSeriesProfile` — scored
`over="Z"` instead of `over="time"`:

```python
osk.compare(
    reference="whots_temp", test="run_new", variables=[TEMPERATURE],
    select={"depth": [0, 10, 25, 50, 60, 100], "time": "2013-06-15"},
    over="Z",
).plot()                              # test | reference overlaid, difference in the box
```

The test lane is linearly interpolated onto the reference's own levels — the
vertical counterpart of the coarser-wins rule `over="time"` already follows,
settled once rather than chosen, since a water column has no "composite vs.
instantaneous" question a time axis does. `over=` rarely needs spelling out at
all: a `profile` reference implies it outright (no time axis to draw instead), and
a `timeSeriesProfile` reference — which carries both axes — reads whichever one
your own `select`/`aggregate` narrows to a single value (a `depth=` pinned to one
number keeps the familiar mooring-at-a-depth series; a `time=` pinned to one
instant keeps the cast).

**A vertical slice through the model** — `select={"transect": {"<dim>": <index>}}`
cuts along a named grid dimension instead of narrowing to one place, and draws as
depth (or the model's own s-levels) against along-path distance rather than a map:

```python
osk.field(
    "pac_dt_ramp", NITRATE,
    select={"transect": {"xi_rho": 30}, "time": "2013-06-15"},
).plot()                       # native s-levels, exact -- a free isel, no interpolation
```

Give `depth` a list of fixed levels to interpolate onto instead, the same
`roms.to_depth` a map row uses:

```python
osk.field(
    "pac_dt_ramp", NITRATE,
    select={"transect": {"xi_rho": 30}, "depth": [0, 50, 100, 200]},
    aggregate={"time": "mean"},
).plot()
```

An **arbitrary path** — waypoints, or a fixed longitude/latitude line — samples the
grid instead, by nearest-neighbour (default) or `method="bilinear"`:

```python
osk.field(
    "pac_dt_ramp", NITRATE,
    select={"transect": {"waypoints": [[150.0, 10.0], [175.0, 15.0], [-160.0, 20.0]]},
            "depth": [0, 50, 100, 200]},
    aggregate={"time": "mean"},
).plot()                       # crosses the antimeridian without incident

osk.field("pac_dt_ramp", NITRATE, select={"transect": {"lon": 200.0}}).plot()
```

Waypoints/lines densify to roughly the grid's own resolution before sampling
(`spacing_km` overrides); a point straying off the domain is dropped with a
warning, trimming the section to what the source actually covers. For clicking a
path instead of typing coordinates, in a live notebook:

```python
picker = osk.pick_path("pac_dt_ramp")          # click waypoints on the domain map
osk.field("pac_dt_ramp", NITRATE, select=picker.as_select()).plot()
```

**Matching a section against a dataset** works the same way through `osk.compare()` —
the reference is sampled at exactly the same points the model's own path resolved
to, so comparing the two is pairing columns along the path, not regridding one grid
onto another. The pair lands on whichever lane's along-path spacing is coarser (the
same house rule a map comparison's own regridding follows), and an explicit
`select={"depth": [...]}` list is required — two lanes' native verticals share no
axis to guess a common one from:

```python
osk.compare(
    test="pac_dt_ramp", reference="woa23_nitrate", variables=[NITRATE],
    select={"transect": {"waypoints": [[150.0, 10.0], [175.0, 15.0], [-160.0, 20.0]]},
            "depth": [0, 50, 100, 200, 400, 700, 1000],
            "time": "2013"},
    aggregate={"time": "mean"},
).plot(renderer="both")        # test | reference | difference sections + metrics
```

Every transect form above works here too, and the reference can be another model
run just as well as a climatology.

**Where is the hot spot, and how did it get there?** — a map naturally raises that
question, and `Field.extremum()` answers it: value, position (lon/lat *and* grid
indices), and the snapshot it fell on.

```python
run = osk.field(
    "pac_dt_ramp", NITRATE,
    select={"time": "2013-06-15", "depth": "surface"},
)
ext = run.extremum("max")
ext
# max nitrate = 31.42 mmol m-3 at lon 214.3800, lat 5.1200 (0-360)
#   grid indices {'eta_rho': 112, 'xi_rho': 387}, time 2013-06-15 00:00:00
#   source='pac_dt_ramp', grid="the source's own grid"
```

`lon`/`lat` are reported in whatever convention the grid is actually stored in
(`lon_convention`) — a domain straddling the dateline, like `pac_dt_ramp`, reports past
180° rather than silently wrapping. `indices` is keyed by the field's own dimension
names, so a curvilinear (ROMS) grid shows `eta_rho`/`xi_rho` and a rectilinear one shows
`lat`/`lon` directly. Pass `"min"` for the opposite extreme; running over every
standing dimension means a field faceted over time or depth reports the facet
coordinate the extremum fell on too.

`.series()` follows that position through time — a point selection at the extremum's
lon/lat, defaulting to 10 native time steps either side of the snapshot (clamped to the
record's own ends) — and `.plot()` draws it immediately, the same `series` figure
`osk.field`'s own point selects already draw:

```python
ext.plot()                                        # ±10 steps, one line
ext.series(variables=["salinity"]).plot()          # add a line, same place/window
ext.series(time=slice("2013-05", "2013-07")).plot()  # a wider window instead
```

A field already reduced to one place (see `Field.family`) has nothing left to search
spatially, and `.extremum()` says so rather than returning the one value `.plot()`
already shows.

## Variable specs

A plain name is the common case, but `variable=`/`variables=` accepts three other
shapes for the cases a name alone can't cover — each describing one quantity, however
it has to be built. (A *list* of specs is a different thing: several quantities, one
figure. `compare()`'s `variables=` fans a list into a `ComparisonSet`; `osk.field`'s
`variable=` fans one the same way into a `FieldSet` — see "More than one variable on
the same line plot" above. Any of the shapes below can be one entry in that list.)

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

`times=` is not what this pattern wants — it fans one calendar time axis, read off the
**test** source, so a reference like this one with no decoded calendar (`decode_times=
False`) has nothing for it to fan against. Reach for `times=` when both sides genuinely
share a calendar (two model runs, a model against a satellite record), and the
`aggregate`/`select` pair-spec above when they don't.

`subtract_mean` takes the same `{"test": ..., "reference": ...}` shape, but — unlike
`select`/`aggregate`, where a one-sided dict is refused as a likely typo — naming just
one side (`subtract_mean={"test": True}`) is accepted outright: a bare `True`/`False`
has no partial form a typo could produce, so there's nothing here for that check to
protect against. The side left unnamed defaults to `False`.

## Catalogs

Sources are described by [intake](https://intake.readthedocs.io) v2 catalogs, found
automatically along a search path where **later shadows earlier**:

```
1. ocean_skill/catalogs/     shipped defaults
2. $OCEAN_SKILL_CATALOGS     a team / shared-cluster directory (os.pathsep-separated);
                             or register one in code: osk.catalog.add_search_path(...)
3. ~/.ocean-skill/catalogs/  your machine
4. ./catalogs/               this project
```

Setup is meant to be obvious and take no config file: drop a YAML in
`~/.ocean-skill/catalogs/` for catalogs that are just yours, point
`$OCEAN_SKILL_CATALOGS` at a team directory (a cluster module file or shared
`.bashrc` is the usual place) for catalogs your whole group should see, and build
project catalogs straight into `./catalogs/` — it's auto-discovered and gitignored.
The narrower the scope, the higher the priority, so a catalog you build for one
project always wins there, your own catalogs win over your team's, and the team's
win over the shipped defaults — without anyone editing a tracked file. (A catalog
already at the old `platformdirs` user-config location, e.g.
`~/Library/Application Support/ocean-skill/catalogs` on macOS, still works — it's
scanned just below `~/.ocean-skill/catalogs/`.) Check what's actually on your search
path any time with `osk.catalog.search_paths()`.

On a cluster with data already staged on a shared filesystem, build a catalog that
points straight at it and save it into a `$OCEAN_SKILL_CATALOGS` directory so it
shadows the packaged, internet-backed entry of the same name — see the GLODAP-on-Anvil
recipe in `docs/catalogs.ipynb` (`ocean_skill.readers.PoochTarNetCDF` takes `local_dir=`
as well as `url=`, so the same reader and merge logic runs either way).

Remote files are cached under ocean-skill's cache directory; set `$OCEAN_SKILL_DIR`
to move it, or fsspec's own `FSSPEC_SIMPLECACHE_CACHE_STORAGE` to override it
outright.

## Layout

```
ocean_skill/       package (flat layout); ocean_skill/catalogs/ ships the reference catalogs
catalogs/          project-local catalogs you build (auto-discovered; gitignored)
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
