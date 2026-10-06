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
osk.describe("woa23_nitrate_month01") # full metadata for one source, and the
                                       # catalog file it was found in
                                       # (osk.describe(...).catalog_path)

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

`detide` removes the tide instead of a scalar offset — the PL33 tidal low-pass filter
(`oceans.filters.pl33tn`, a 33-hour half-amplitude FIR filter), run on each lane at its
own native sample rate *before* alignment, so the tidal band is gone before `select`/
`aggregate` ever touch the time axis:

```python
raw = osk.compare(reference="tide_gauge", test="his", variables=["zeta"])
detided = osk.compare(reference="tide_gauge", test="his", variables=["zeta"],
                       detide=True)
(raw + detided).target()               # the detided point moves off the raw one --
                                        # the tide itself was part of what "raw" scored
```

Takes the same shapes `subtract_mean` does (`True`/`"test"`/`"reference"`/a
`{"test": ..., "reference": ...}` pair-spec) plus a cutoff of its own: `detide={"T":
72}` swaps PL33's 33-hour default for a 3-day low-pass, on one or both lanes. Unlike
`subtract_mean` (a post-align scalar shift that shares its raw twin's cache entry),
detiding changes the data itself, so it joins the cache key — a raw run and a detided
run of the same comparison are cached, and pool, as two distinct entries — and lands
in `.metrics()` as an always-present `detided` column, the same four spellings
`demeaned` uses.

The same filter is available standalone, on a time series or `timeSeriesProfile`
(time × depth), whether it arrives as a DataFrame or a Dataset:

```python
subtidal = osk.detide(osk.read("tide_gauge"))          # DataFrame in, DataFrame out
tidal = osk.detide(ds["zeta"], component="tidal")       # the removed tide itself
```

Edges are NaN — PL33's centered window has no full window within about `T` hours of
either end of the record, wider still as `T` grows.

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

A select that leaves **both time and depth** standing at one place — the default shape
of a `timeSeriesProfile` station (`ctd_station_HV5`, say) — draws a third way: one
panel, colour = value, x = time, y = depth (inverted, surface at top), scattered points
for a ragged repeat-visit record or a mesh for a dense one:

```python
osk.field("ctd_station_HV5", "sea_water_temperature").plot()
# one time_depth panel, colour = temperature, no aggregate= or select= needed
```

Naming an explicit list of levels (`select={"depth": [0, 50, 100]}`) still draws the
old way instead — one line per level — since asking for discrete levels is asking to
tell them apart, not to see the whole record. There is no separate call for any of
this: `.plot()` reads it all off `Field.family`.

**A time `groupby` stays plottable, too.** `aggregate={"time": {"groupby": "month",
"reduce": "mean"}}` renames the time axis to `month` (twelve climatological means, one
per calendar month) rather than removing it, and `.plot()` draws that the same way it
would a real time axis — `month` plays time's role, spelled `Jan`..`Dec` on the axis:

```python
osk.field(
    "ctd_station_HV5", "sea_water_temperature",
    aggregate={"time": {"groupby": "month", "reduce": "mean"}},
).plot()
# the same time_depth panel, x = month (Jan..Dec) instead of a real date

osk.field(
    "ctd_station_HV5", "sea_water_temperature",
    select={"Z": {"min": 0, "max": 5}},
    aggregate={"time": {"groupby": "month", "reduce": "mean"}, "Z": "mean"},
).plot()
# depth is gone too, so this draws as a series instead: one line, month on x
```

Every other `.dt`-accessor groupby (`year`, `hour`, `dayofyear`, ...) works the same
way — only `month` gets the calendar spelling; the rest draw their own integer values
on a plain numeric axis. `groupby: "season"` is the one exception: it keeps drawing as
the profile family's one-line-per-season fan it always has, in calendar order. Naming
an explicit list of months (`select={"month": [1, 4, 7]}`) is the `month` climatology's
own version of the depth-list rule above: it turns off "month is time" and fans into
one profile line per named month instead, coloured and legended by month.

**Statistics of a climatology: chain the steps.** An axis's value may be a *list* of
steps, applied in order, each acting on the axis the previous one left — after a
`groupby: "month"` that is the new `month` axis, after a `resample` it is still time. So
the variance of the seasonal cycle is "mean by month, then the variance of those twelve":

```python
osk.field(
    "run_new", "temperature",
    select={"depth": "surface", "time": slice("2010-01", "2012-12")},
    aggregate={"time": [{"groupby": "month", "reduce": "mean"}, "var"]},
).plot()                              # one map, in °C² — not twelve panels
```

The common ones, each reading left to right (`range` is the one reduction added here,
max − min):

| You want | `aggregate={"time": ...}` |
|---|---|
| Seasonal-cycle variance / std | `[{groupby: month, reduce: mean}, var]` (or `std`) |
| Seasonal range | `[{groupby: month, reduce: mean}, range]` |
| Interannual variability | `[{resample: 1YS, reduce: mean}, std]` |
| Variance of monthly means, interannual included | `[{resample: 1MS, reduce: mean}, var]` |
| Mean annual maximum (winter MLD, bloom peak) | `[{resample: 1YS, reduce: max}, mean]` |
| Mean diurnal range | `[{resample: 1D, reduce: range}, mean]` |
| Month-balanced annual mean ± seasonal std | `[{groupby: month, reduce: mean}, {reduce: mean, spread: std}]` |

(Written as Python dicts in your code, `{"groupby": "month", "reduce": "mean"}`; the
table drops the quotes the way a suite YAML does.) The rules are the ones that keep a
chain meaningful: a one-element list is the bare step, an empty list is an error, `spread`
belongs on the last step only, nothing can `groupby`/`resample` after a `groupby` (there
is no time left to bin), and nothing can follow a step that collapsed the axis. The whole
chain is checked before any data is read, and a month `groupby` followed by another step
warns when fewer than twelve months are present, since a "seasonal variance" of three
months is not one.

Units follow the statistic rather than the field: a variance is in squared units
(`degC` becomes `delta_degC^2`, drawn `°C²`; `mg/m^3` becomes `(mg/m^3)^2`), a std or
range in the *difference* of the field's units (no Kelvin offset applied), and a unit
conversion scales a variance by the squared factor. Colour limits that a standard_name
pins for the field itself (chlorophyll's log 0.01–10) are skipped for a spread
statistic, whose values are nowhere near the field's. Titles lead with what was computed
— `variance of monthly means over 2010-01-01–2012-12-31`, `std of annual means`, `mean of
annual maxima` — and a plain mean is titled as it always was; `title=` replaces any of
it. Aggregates run per lane on the native grid *before* regridding, so a variance is
computed on each grid and then regridded, not the other way round. `compare()` takes the
same chains.

**A bare grid defaults to its surface.** `osk.field(source, variable)` with nothing
selected keeps the whole vertical axis standing for a catalogued `featureType: grid`
source (see "Nothing is reduced unless you ask", below) — but its own map panels have
no vertical axis to draw at all, so `.plot()`/`.movie()` narrow to `select={"depth":
"surface"}` first, the same default `compare()` already has. Pass `select={"depth":
...}` (or `{"sigma0": ...}`) explicitly for anything else. A bare, genuinely
multi-step time axis has no equivalent single default instant, and `.plot()` says so
rather than guessing one — narrow it with `select={"time": ...}`, reduce it with
`aggregate=`, or call `.movie()` to play every step instead.

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

**A whole-domain mean through time, at several depths.** `aggregate={"lon": "mean",
"lat": "mean"}` on a model grid is one area-weighted mean of the whole domain (true cell
area where the grid carries it, `cos(latitude)` otherwise), not a station, and a depth *list* keeps one line
per level — `"surface"` is allowed in it, alongside depths in metres:

```python
osk.field(
    "run_new", ["temperature", "salinity", "oxygen"],
    select={"depth": ["surface", 100, 200]},
    aggregate={"lon": "mean", "lat": "mean"},
).plot()
# three panels (one per variable), each with three lines: surface, 100 m, 200 m
# panel titles read e.g. "temperature · domain mean · 2010-07 to 2010-10"
```

The panel title says `domain mean` rather than naming a position (a whole-domain mean has
none); a mean over a box — `select={"lon": {"min": 165, "max": 175}, "lat": {"min": 5,
"max": 15}}` — reads `mean over 5–15°N, 165–175°E` instead. The surface line is
labelled `surface` in the legend, not `0 m`. As for any depth list, colour carries the
variable and a marker tells the levels apart; pass `encode={"color": "depth",
"marker": None}` to colour by level instead, and `rows="variable"` to drop the variable
name from each legend entry (the panel title already says it).

**Nothing is reduced unless you ask** — for an `osk.field()` call itself: an unset
`select`/`aggregate` leaves every native level and every native step standing, unlike
`compare()`'s own unset vertical default (the surface). The one exception is `.plot()`/
`.movie()` on a catalogued grid source, which narrows a bare vertical axis to the
surface right before drawing (see above) since a map has nowhere to put it; `.data`
itself is never touched by that narrowing. There is no default *aggregation* anywhere:
omit `aggregate` and every step of the selection survives as its own panel or frame. A
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
instant, or one entry of `times=[...]`, keeps the cast). Narrow *neither* one and
both axes stay standing instead of picking one to guess at: every `(time, depth)`
pair the station actually sampled is matched against the model and pooled into
one metric —

```python
osk.compare(
    reference="hvalfjordur_hv1", test="run_new", variables=[TEMPERATURE],
).plot()                              # test | reference | difference, time on x, depth on y
```

— `family == "time_depth"`, drawn as a `test | reference | difference` row with
time on x and depth on y (inverted, surface at top), the comparison counterpart
of `osk.field()`'s own `time_depth` panel below. This is the right default for a
ragged discrete-sample station (bottle casts at whatever depths that visit
reached, not a mooring's fixed levels): its near-surface level alone is often
sampled on only a handful of visits, too sparse to score well on its own, where
every visited `(time, depth)` pair together gives a real sample. Narrow *both*
axes (one instant *and* one depth) and there is nothing left to keep — that is
the one shape `over=` still has to name explicitly.

`depths=` needs no spelling out either, for any of these: left unset,
`compare()` reads the reference's own levels straight off the source — for a
`timeSeriesProfile` reference narrowed to one instant (scored over depth alone),
the levels *that visit itself actually sampled*; for a bare call keeping both
axes, every visit's own depths, together, since every sampled pair is what gets
matched and pooled. A discrete-sample station visited repeatedly (bottle casts,
not a continuously logging instrument) reads this way once its table is
catalogued `featureType: timeSeriesProfile`: bare `compare()` pools the whole
ragged record as above, `select={"time": <one visit>}` draws one cast, an
explicit `depths=("surface",)` (or any `select={"depth": ...}`) falls back to
the old mooring-at-a-depth series over whichever visits actually sampled near
that level, and `times=[<visit>, <visit>, ...]` overlays several casts in one
panel, same as a real profile source's own multi-cast overlay above. A bare
`osk.field()` on the same station has no such default and keeps the whole
record instead, which draws as a `time_depth` panel — see above.

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

**A window around a point, in both grid directions** —
`select={"transect": {"cross": ...}}` cuts two windowed grid-aligned transects
through one lon/lat point (or grid-index pair), one along each grid direction,
rather than the whole line:

```python
osk.field(
    "pac_dt_ramp", NITRATE,
    select={"transect": {"cross": {"lon": 200.0, "lat": 15.0}, "half_width": 10}},
).plot()                       # two sections, stacked; orientation="horizontal" for side by side
```

`half_width` (grid cells either side of the point) defaults to 15; a window
reaching past the domain edge is clamped there, with one warning. This returns an
`ocean_skill.field.Cross` (two independent `Field`s, `.along`/`.across`) rather
than a `Field` — see `docs/plot_styling_reference.md` for the full grammar,
including a single windowed direction without the `cross` sugar.

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
run just as well as a climatology. `comparison.map_locations()` draws the
requested waypoint path (or fixed lon/lat line) over the model's domain outline;
a grid-aligned/`cross`/reference-derived transect draws each source's own
footprint instead, since those name no lon/lat without opening a dataset.

**A box averaged along one axis is a section too.** `aggregate={"lon": "mean"}` over a
lon/lat box leaves latitude and depth standing — a meridional slab, drawn against
latitude (°N) instead of along-path distance, titled "mean over 180–200°E". `{"lat":
"mean"}` gives the zonal counterpart, drawn against longitude. On a curvilinear grid
(ROMS) the cells inside the box are binned along the surviving axis at the grid's own
spacing, then averaged. As with a transect, a comparison needs an explicit depth list.

**Several sections, one figure.** Build each one separately, then hand the list to
`osk.plot`. Comparisons stack as `test | reference | difference` rows, and fields
stack one panel per member. A `{label: item}` dict names the rows:

```python
depths = [0, 25, 50, 100, 150, 200, 300, 400]
pair = dict(test="all_the_rest", reference="woa23_temperature_annual",
            variables=["temp"])
box = {"lat": {"min": -30, "max": 30}, "depth": depths}

eq   = osk.compare(**pair, aggregate={"time": "mean"},
                   select={"transect": {"lat": 0, "lon": {"min": 143, "max": 267}},
                           "depth": depths})
b180 = osk.compare(**pair, aggregate={"time": "mean", "lon": "mean"},
                   select={**box, "lon": {"min": 180, "max": 200}})
b160 = osk.compare(**pair, aggregate={"time": "mean", "lon": "mean"},
                   select={**box, "lon": {"min": 200, "max": 240}})

osk.plot({"Eq": eq, "180-160": b180, "160-120": b160}, shared_limits=True)
```

Each row keeps its own x axis (km along the equator for the first row, latitude for
the others), and `shared_limits=True` puts every row on one colour scale.
`osk.plot([field_a, field_b])` does the same for fields. It is the same as
`osk.ComparisonSet([...]).plot()` or `osk.FieldSet([...]).plot()`, without naming the
set. Only plots of one kind combine: sections with sections, maps with maps, lines with
lines. A list mixing a section with a map is refused, because mixed-panel figures
don't exist yet.

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

The figure carries a suptitle saying what it is a series *of* — each panel's own title
names only a place and a period, which on its own reads like a station record or a
domain-wide trend:

```
Time series at the surface alkalinity minimum: 126.1 mmol/m^3 at 16.1°N 97.6°E on 2010-10-31
```

The depth phrase (`surface`, `100 m`) is the parent field's own single-level
`select={"depth": ...}` and is left out for none, a band or a list; a `local=True` hit
reads `local minimum`, and the second of an `n=` search `minimum (#2)`. Pass
`title="..."` to `.plot()` (`ext.plot(title="Where it bottoms out")`, or a suite page's
`plot: {title: ...}`) to replace it, or `title=""` for none.

A field already reduced to one place (see `Field.family`) has nothing left to search
spatially, and `.extremum()` says so rather than returning the one value `.plot()`
already shows.

**Specks, not just the extreme.** On a model field the global min/max is usually a
coastline or river-mouth cell, and the next-lowest values are more of the same. The tiny
minima a `robust=True` colorbar reveals are not extreme in *value* — they are extreme
relative to their *neighbors* — so `local=True` scores every wet cell against the median
of its wet neighbors (a straight front or a smooth gradient scores ~0; only a feature
about one cell wide stands out) and `n=` lists the worst offenders:

```python
hits = run.extremum("min", local=True, n=15)
hits
# 15 local min alkalinity [mmol/m^3] on 'pac_dt_ramp' (3x3 wet-neighbor median, ranked by z, >= 3 cells apart)
#       value  anomaly  neighborhood  spread      z  wet_neighbors  land_distance     lon    lat       time  i_eta_rho  i_xi_rho
# rank
# 1    2288.5    -30.0        2318.5    0.59  -50.6              8              8  166.65  39.58 2010-10-01        795       596
# ...
hits.to_dataframe().query("land_distance > 10")   # well offshore only
hits[0].plot()                                    # each hit follows through time, as above
```

**Ranking.** A river plume defeats a plain departure: a few cells offshore it is steep, so
its cells depart from their neighbors by hundreds of units and crowd out a speck that
departs by tens. By default hits are therefore ranked by `z` — the departure in units of
how much the neighbors vary *among themselves* (`spread`, a scaled median absolute
deviation) — which is large in smooth water and small inside a plume or front. A z is never
divided by less than the field's typical spread, so a flat patch cannot inflate it.
`score="departure"` ranks by the departure in the field's own units instead; every hit
reports both, as `anomaly` and `z`.

**Distance from land.** `land_distance` is the number of cells from a hit to the nearest
land cell or the grid's edge (1 = touching), and `wet_neighbors` counts how many of the
window's other cells were wet, so coast and island hits are easy to tell from open-ocean
ones. `interior=k` drops every cell with land or the edge within `k` cells up front
(`interior=True` is the window's own reach — one cell for the default 3×3) and works for a
plain global `n=` search too, keeping it off the coast:

```python
run.extremum("min", local=True, n=15, interior=10)   # no land within 10 cells
run.extremum("max", n=5, interior=10)                # the highest values away from any coast
```

Each hit is an ordinary `Extremum`, so `.series()` / `.plot()` work on it. `window=5`
widens the neighborhood, and `separation=` sets how many cells apart hits must be
(default: the window; 10 for a plain `n=` search, so one river plume is one place rather
than ten adjacent cells). A field faceted over time is scored one slice at a time, and a
speck that persists is reported once, at the step where it is strongest.

A suite page runs this same chain with no Python at all — see `then:` in
[docs/suites.md](docs/suites.md).

**One property against another — T-S diagrams and beyond.** `osk.XY(members, x=, y=)`
plots one variable against another for every data source you hand it (a *member*, built
from ordinary `osk.field()` results), with one panel per region. `osk.TS` is the preset
with salinity on x and temperature on y, and it draws density (sigma-0) contours by
default. How each member is drawn follows its data: a member whose two variables sit at
one position with only depth left (a nearest-cell profile, a box mean) is a
depth-ordered line, and anything else — every cell, level and snapshot of a model box —
is a cloud of dots.

```python
box = {"lon": {"min": 156.16, "max": 157.23}, "lat": {"min": 19.32, "max": 20.32}}
TS_VARS = ["temperature", "salinity"]

roms = osk.field("all_the_rest", TS_VARS, select=box)                  # dots
woa = osk.field(["woa23_temperature_annual", "woa23_salinity_annual"], TS_VARS,
                select={"lon": 156.70, "lat": 19.82})                   # nearest cell: line
run = {"min": "2010-07-31", "max": "2010-10-31"}                    # the run's own dates
glorys = osk.field("glorys_my_daily_timeseries", TS_VARS, select={**box, "time": run},
                   aggregate={"time": "mean", "lon": "mean", "lat": "mean"})  # line

osk.TS({"ROMS": roms, "WOA23": woa, "GLORYS12": glorys}).plot(
    title="North West Pacific water masses")

osk.XY({"ROMS": osk.field("all_the_rest", ["nitrate", "phosphate"], select=box)},
       x="phosphate", y="nitrate").plot()      # any pair; density contours are off
```

A member needs one field for `x` and one for `y`; a source list crossed with a variable
list (`osk.field([woa_t, woa_s], TS_VARS)`, WOA keeping temperature and salinity in
separate entries) makes the cross product and the combinations a source does not carry
are dropped with a warning. Each member is read only when the figure is drawn, so
building an `XY` costs nothing.

`regions=` gives one panel per region and *replaces* every member's horizontal `select`,
so one set of members is drawn in each. A region is a lon/lat box or point. A ~100 km box
on a 1° grid can sit between cell centres and select nothing; `at_center=` samples the
listed members at the nearest cell to each box centre instead:

```python
PACIFIC = {   # lon (0-360) and lat ranges of Damien et al. Fig. 7's boxes, on their grid
    "North West Pacific":        ((156.16, 157.23), (19.32, 20.32)),
    "Subpolar Gyre":             ((185.86, 186.96), (44.56, 45.34)),
    "California Current System": ((233.87, 235.07), (36.39, 37.35)),
    "South West Pacific":        ((185.18, 185.99), (-15.45, -14.67)),
    "South Pacific Gyre":        ((240.99, 242.00), (-21.26, -20.31)),
    "Peru Current":              ((275.31, 276.41), (-11.48, -10.40)),
}
regions = {name: {"lon": {"min": lon[0], "max": lon[1]},
                  "lat": {"min": lat[0], "max": lat[1]}}
           for name, (lon, lat) in PACIFIC.items()}

ts = osk.TS({"ROMS": osk.field("all_the_rest", TS_VARS),
             "WOA23": osk.field(["woa23_temperature_annual", "woa23_salinity_annual"],
                                TS_VARS),
             "GLORYS12": osk.field("glorys_my_daily_timeseries", TS_VARS,
                                   select={"time": run},
                                   aggregate={"time": "mean", "lon": "mean",
                                              "lat": "mean"})},
            regions=regions, at_center=["WOA23"])
ts.plot(ncols=3, colors={"ROMS": "black", "WOA23": "tab:red", "GLORYS12": "tab:blue"})
ts.plot(color_by="depth", density=False, renderer="holoviews")   # colour the cloud by depth
ts.data                    # {region: {member: {"x": DataArray, "y": DataArray}}}
```

A member that cannot give a panel (a box with no cell in it, a point on land) is dropped
from that panel with a warning that says why. Without `regions=`, each member needs a
lon/lat box or point in its own `select` — otherwise it would load its whole domain, every
level and time step, and `XY` refuses instead. Water-mass names are placed by hand with
`annotations=`; the positions for the six panels above are the `annotations:` block of
the `TS:` page in `suites/pacmed_review.yaml`. The same figure is a suite page — see
`XY:`/`TS:` below.

## Suites: run a whole diagnostic from one YAML

A suite is a YAML file listing **pages** — each one a single `osk.field`, `osk.compare`,
or `osk.summary` call — plus shared defaults (an `XY:`/`TS:` page draws the
property-property plot above from a `members:` dict, with `regions:` and `at_center:`).
One command draws every page and writes PNGs, a PDF, and a metrics CSV, with no Python
required:

```bash
ocean-skill-run suites/roms_marbl_diagnostic.yaml
```

Every run writes its own report directory (nothing is ever overwritten), named after
the model and the run's own time range, plus when the command was run, and (for a
suite with a `refresh:` block) carrying its own copy of the model's kerchunk reference
as it was at that moment. See
[docs/suites.md](docs/suites.md) for the full grammar — `for_each` fan-out,
`{placeholder}` templating, `time: latest`/`month: run`, a `catalog_search_paths:` key
for shared catalog directories, a `cache_dir:` key to pin the suite's cache to one
directory across runs, and the report layout —
and `suites/roms_marbl_diagnostic.yaml`/`suites/roms_marbl_quick.yaml` for a worked
ROMS-MARBL example (latest snapshot, monthly means, WOA23/GLODAPv2/satellite
chlorophyll comparisons).

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
layer depth, computed by a chosen criterion from a ROMS run's temperature (and, for the
density criterion, salinity), so point it at the run's physics output rather than a
biogeochemistry file that carries neither:

```python
osk.field("GOM_his", {"calculate": "mld", "method": "density_threshold"})
```

The result is named by its definition — `ocean_mixed_layer_thickness_defined_by_sigma_theta`
for `density_threshold`, `..._defined_by_temperature` for `temperature_threshold` —
because CF's mixed-layer names encode which variable defines the base of the layer. The
threshold and reference depth (0.03 kg m-3 from 10 m for density, 0.2 °C from 10 m for
temperature; `threshold=`/`ref_depth=` in the spec change them) are not part of the name,
so they ride on the result's `mld_threshold`/`mld_ref_depth` attrs.

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
    reference="holte_talley_mld_clim", test="GOM_his",
    variables=[{
        "test": {"calculate": "mld", "method": "density_threshold"},
        "reference": "mld_by_sigma_theta",   # the name its catalog gives mld_dt_mean
    }],
    select={"test": {"time": "2010-08"}, "reference": {"month": 8}},
    aggregate={"test": {"time": "mean"}, "reference": {}},
)
```

Both sides resolve to `ocean_mixed_layer_thickness_defined_by_sigma_theta` — the same
criterion variable, and at the calculator's defaults the same threshold as `mld_dt_mean`
— so this scores like against like. The two sides are narrowed differently because they
are different kinds of time: the model run is a record (August 2010, averaged over its
steps) and the Argo product is a climatology whose time axis is the calendar `month`,
so the reference picks August with `{"month": 8}` and has nothing left to reduce.
(`holte_talley_mld_clim` is a catalog you build once, with `month` as a real axis; see
"Catalogs" below for the one that gives the file's `mld_dt_mean` that name.)

`standard_name` is optional here — both sides already agree, and the test side names
the figure. Set it to name the figure and its labels yourself
(`"standard_name": "ocean_mixed_layer_thickness_defined_by_sigma_theta"` reads as
"MLD (σθ, density_threshold)" in a pooled figure), or when the two recipes' CF names
legitimately differ: without one a pair-spec is assumed to score a quantity against
itself, so a mismatch between what its two sides resolve to is a warning rather than a
number that looks right and isn't. Swap the test side for
`{"calculate": "mld", "method": "temperature_threshold"}` and it warns — a
temperature-defined mixed layer depth is not the density-defined one the climatology
holds — until you set `standard_name` to say the difference is intended.
`Comparison`/`compare()` accept a pair-spec; `Field` does not (there is no second lane
to give the other half of the pair to).

**Broad and specific names.** CF names a mixed layer depth by the variable that defines
the base of the layer, not by the threshold, so there is a generic `"mld"` and one name
per criterion: `"mld_by_sigma_theta"`, `"mld_by_sigma_t"`, `"mld_by_temperature"`,
`"mld_by_mixing_scheme"`. `"mld"` finds and compares any definition; a specific name
asks for that one, and is never answered by the generic name (which hasn't said which
definition it carries) or by a sibling. The shipped Copernicus `mlotst` products (and
Holte & Talley's climatology, catalogued below) are `"mld_by_sigma_theta"`; ROMS' KPP
`hbls` is `"mld_by_mixing_scheme"`, though an older ROMS catalog keeps the generic name
until it is rebuilt — `"mld"` still finds it, `"mld_by_mixing_scheme"` will not:

```python
osk.find(variable="mld")                  # a mixed layer depth of any definition
osk.find(variable="mld_by_sigma_theta")   # only the potential-density ones
```

So a plain `variables=["mld"]` comparison can pair two different definitions — KPP `hbls`
against a sigma_theta climatology — and warns once when it does (a cached rerun too),
since a bias between them partly measures the definitions; the pair-spec above is the
like-for-like route. Inside one dataset `"mld"` resolves to the single definition it
carries (with a warning naming it); if it carries several, it warns, naming them, and
treats the variable as not available until you ask for one by name. That dataset-side
matching is plain cf-xarray: the generic entry's registered criteria include each
definition's spellings, so `ds.cf["mld"]` behaves the same way outside ocean-skill.
`vocabulary.register(..., broader="mld")` (from `ocean_skill import vocabulary`) adds
another definition under the broad name.

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

**All twelve months of a WOA climatology as one source.** The `woa23_<var>_month01`..
`month12` entries each hold one undecoded step; `woa23_<var>_monthly` (nitrate, oxygen,
phosphate, salinity, silicate, temperature) stacks all twelve into one source, with a
`month` coordinate attached so a month `groupby` works on a time axis that has no
calendar to read one from. That is what a question about the seasonal cycle itself needs
(its variance, its range) rather than one month's map. Build the same thing for another
twelve-file product by giving `build.add_source` the list of files, with
`reader_kwargs={"combine": "nested", "concat_dim": "time"}` and
`climatology=True, climatology_period="monthly"`. WOA's
monthly fields reach 1500 m, so a depth list can go deeper than the annual ones usually
get compared.

**Several statistics in one call.** A *top-level* list for `aggregate=` — a list of whole
specs, not the list of steps under one axis above — fans exactly like `variables=` and
`depths=` do, one member per spec, crossed with the other fans. The mean and the
seasonal-cycle variance of a model against WOA temperature, at the surface and 200 m, is
four members and one `.plot()`:

```python
osk.compare(
    reference="woa23_temperature_monthly", test="all_the_rest",
    variables=["temperature"], depths=["surface", 200],
    aggregate=[{"time": "mean"},
               {"time": [{"groupby": "month", "reduce": "mean"}, "var"]}],
).plot()
```

Each member is labelled by its statistic (`mean`, `variance of monthly means`) so the
rows stay distinguishable; exact repeats are dropped with a note, `[]` is an error, and a
one-element list is that spec (still returning a set, as `variables=[v]` does). Any entry
may itself be a `{"test": ..., "reference": ...}` pair-spec. `osk.field(...,
aggregate=[...])` fans the same way into a `FieldSet` (plain specs only — one source has
no two lanes), and a suite page takes the list under `aggregate:` (docs/suites.md).

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
                             or register one in code: osk.catalog.add_search_path(...);
                             or a suite YAML's own catalog_search_paths: key
                             (docs/suites.md)
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
recipe in `docs/tutorial.ipynb` (the `grid` example under "Data catalogs";
`ocean_skill.readers.PoochTarNetCDF` takes `local_dir=` as well as `url=`, so the same
reader and merge logic runs either way).

Remote files are cached under ocean-skill's cache directory; set `$OCEAN_SKILL_DIR`
to move it, or fsspec's own `FSSPEC_SIMPLECACHE_CACHE_STORAGE` to override it
outright. A remote single-file product takes a `simplecache::` prefix on its URL, which
downloads it once into that cache (the WOA entries do the same).

A product whose own metadata is wrong is corrected in its catalog entry, not in the
file. A per-source `standard_names` map *merges* over what the probe found — your entry
wins for the variables it names and the rest of the probed map stays — and a variable it
renames also has its own `standard_name` attribute set to the catalog's name when the
source is read. The Holte & Talley Argo mixed-layer-depth climatology needs exactly
this: its `standard_name` attributes just repeat each variable's own name, so its
density-threshold field `mld_dt_mean` has to be told what it is. It also stores `lat`,
`lon` and `month` (1 to 12) as plain variables on dimensions with no coordinates
(`iLAT`, `iLON`, `iMONTH`), so a bare URL leaves nothing to select a month, a point or a
box on. Pass a reader *chain* as `"reader"` instead of a `url`: it reads the
file, promotes the three to coordinates and swaps them in for the index dimensions, so
`month` becomes a selectable axis (`select={"month": 8}`):

```python
from intake.readers import datatypes, readers
from ocean_skill import build

URL = "simplecache::https://mixedlayer.ucsd.edu/data/Argo_mixedlayers_monthlyclim_04142022.nc"
ht = (
    readers.XArrayDatasetReader(
        datatypes.HDF5(url=URL),
        engine="scipy",         # a classic netCDF3 file; h5netcdf can't read it
        decode_times=False,     # `month` is a plain 1-12 axis, not a date
        chunks={},
    )
    .set_coords(["lat", "lon", "month"])
    .swap_dims({"iLAT": "lat", "iLON": "lon", "iMONTH": "month"})
)

build.build_catalog(
    {
        "holte_talley_mld_clim": {
            "reader": ht,
            "standard_names": {
                "mld_dt_mean": "ocean_mixed_layer_thickness_defined_by_sigma_theta",
            },
            "climatology": True,
            "doi": "10.1002/2017GL073426",
        },
    },
    "catalogs/mld_climatologies.yaml",
    title="Global mixed layer depth climatologies",
    name_map=None,                  # not ROMS output: skip the ROMS name fallback
)
```

The chain is saved into the catalog entry itself (it is an ordinary intake pipeline),
so nothing about it needs repeating at read time. `HDF5` is only the vehicle for the
URL here — the `engine="scipy"` is what actually reads the classic file.

`osk.find(variable="mld_by_sigma_theta")` now finds it, and `"mld_by_sigma_theta"` is the
name the pair-spec above reads it by.

## Layout

```
ocean_skill/       package (flat layout); ocean_skill/catalogs/ ships the reference catalogs
catalogs/          project-local catalogs you build (auto-discovered; gitignored)
tests/             pytest suite
docs/              MyST / Jupyter Book docs + notebooks
examples/          short runnable scripts
suites/            declarative multi-page suites (YAML) -- run with ocean-skill-run
environment.yml    conda scientific stack (source of truth)
```

## Install (development)

```bash
mamba env create -f environment.yml
mamba activate ocean-skill
pytest
```
