# Skill maps: scoring a model against a reference that varies in time

A `Comparison` normally reduces both sources to a single map and shows you three panels:
the model, the observations, and their difference. That works when the reference *is* a
single map — a WOA climatology, a GLODAP section. It throws away most of what a satellite
record has to offer, because a multi-year L3 product varies in time as well as space, and
averaging it away answers a different question. "Does the model's long-term mean look like
the observed long-term mean?" is not "does the model track the observations?" — a run can
pass the first and fail the second badly.

`over=` keeps the axis and scores against it:

```python
import ocean_skill as osk

osk.compare(
    reference="modis_chl_daily",
    test="gom_bgc",
    variables=["chlorophyll"],
    select={"time": slice("2012-01", "2012-06"), "depth": "surface"},
    over="time",
).plot()
```

Every metric is now computed **cell by cell along time**, so each one is a map: bias says
*where* the model runs high or low, correlation *where* it follows the observed variability
through the record, the variability ratio *where* it is over- or under-dispersed. There is
no test-vs-reference row to draw, so the panels are free to be different metrics of the
same comparison, side by side.

## What you get

Four panels by default — `bias`, `crmsd`, `corr`, `sigma_ratio`. Not a taste in metrics:
these are exactly the quantities the [Taylor and Target diagrams](../ocean_skill/plot/summary.py)
are built from, so the figure is the *spatial decomposition of the single point those two
diagrams plot*. RMSE is not missing, it is implied: `rmse² = bias² + crmsd²`.

Any registered metric works — `metrics=("bias", "rmse", "corr", "n")` at the data layer,
`metric_names=(...)` to pick a subset of what was computed when you draw. `n` is worth
asking for against a cloud-gapped product: it is the number of valid model/observation
pairs at each cell.

Each panel is also annotated with that metric's **overall** value, reduced over space and
time together — the same number that goes in the metrics CSV, and the one a Taylor or
Target diagram would plot. The map and the number are one statistic at two resolutions,
and reading either alone is how a respectable average hides a bad region.

```python
scored = osk.compare(..., over="time")
scored.metrics()                         # the tidy one-row-per-comparison table,
                                          # as always
scored[0].pointwise_metrics()            # the metric maps as an xr.Dataset
scored[0].pointwise_metrics("rmse", "n") # or any other registered metric
scored.taylor()                          # still works: these are the same numbers
scored.save("gom_vs_modis")              # figure + CSV, same layout as any other
                                          # comparison
```

Because they are the same numbers, a scored set pools with an unscored one on a single
diagram — `osk.summary([scored, surface_nutrients])` — even though the two cannot share
a figure of *fields*. Only the metrics record travels, so nothing recomputes the maps.

`renderer="holoviews"` gives the interactive version of the same figure, with the overall
value in each panel's title instead of its corner.

## How the two sources are matched in time

This is the part you cannot infer from the signature, and the part most likely to matter.

**Alignment in time now matches alignment in space exactly.** Both pick direction by
resolution rather than by role — how the finer side moves depends on how much finer it
is, and either lane may be the one that moves. A fine satellite reference scored
against a coarse model lands *on* the model's grid (see
[`align()`](../ocean_skill/align.py)'s `_regrid_target`); a fine mooring reference
scored against a coarse monthly product lands *on* the product's months the same way
(`resolve_match_method`) — with a warning, since coarsening the reference does change
what is being scored against, even though it is not refused.

| Situation | What happens | Spatial analogue |
|---|---|---|
| Test much finer (hourly model, daily product) | test steps are **averaged into** the reference's bins | `conservative_normed` |
| Comparable cadences, offset stamps (a daily mean at 12:00 vs a composite at 00:00) | steps are **paired** by nearest match within half a bin | `bilinear` |
| Reference is instantaneous rather than a composite | test is **sampled** at those instants | `bilinear` |
| Reference finer than the test (a 15-minute mooring against a monthly product) | reference steps are **averaged into** the test's bins, with a warning | the same finer-onto-coarser regrid |

Averaging the finer lane is a default rather than something to request for the same
reason area-averaging is: nobody has to ask for `conservative_normed` either. And the
binning happens *before* the regrid, which is not just tidier — it turns 8760 regridded
fields into 365. Whichever lane is finer, coarsening it is the default; only when the
*reference* is the one being coarsened does it also warn, naming both cadences and two
ways out — an explicit `aggregate={"reference": {"time": {"resample": ..., "reduce":
"mean"}}, "test": {}}` to spell out the same thing deliberately (and silence the
warning), or swapping which lane is the test.

**Bin edges come from the coarser lane's own stamps** — the frame, whichever lane that
turns out to be — as cell edges in longitude and latitude do. Whether a stamp marks the
*start* of its bin or the *middle* is read off the stamps themselves: a product that
labels a bin with its first instant lands on a period boundary (midnight for a daily or
8-day composite, the first of the month for a monthly one), while one that labels the
middle deliberately does not — WOA and OceanSODA stamp the 15th, a ROMS daily average
noon. Both spellings therefore bin their own period correctly. Override with
`bin_anchor="start"`/`"center"` if a product does something else.

**Composite or snapshot** is read from CF `cell_methods` on the variable, then from
`period` in the catalog entry — asked of whichever lane is finer, since that is the one
being averaged or sampled. When neither says, the finer lane is taken to be a composite
and you get a warning naming the assumption, the override (`time_method="nearest"`) and
the permanent fix. That is the same warn-and-proceed `align()` already does for units it
cannot verify.

Everything the matching decided lands in the aligned result's `attrs`, beside
`regrid_method`: `match_method`, `match_target` (which lane's stamps the shared axis
carries — `"reference"` for the historical direction, `"test"` for the mirror),
`match_reason`, `n_matched`, `bin_anchor`, how many bins came out empty or short. Both
are choices a reader of the numbers is entitled to see.

```python
scored[0].aligned.attrs["match_reason"]
# "the test steps every 1 hour and the reference every 1 day, so the test is averaged
#  into its bins"
```

## Thin cells

A satellite pixel behind cloud contributes nothing. Cells with fewer than `min_pairs`
valid pairs (5 by default) are masked in *every* metric map — a correlation from three
clear days is noise, and a figure where `bias` and `corr` cover different cells cannot be
read. How much that removed comes out as a warning, not as text on the figure, and the `n`
map is always available to see the coverage for yourself.

The overall value is computed on the same cells the maps show, so the number beside a
figure describes the figure.

## Depth instead of time

`over=` names an axis, not a concept. `over="Z"` scores down each water column — one
metric per cell computed over the levels — with the same matching rules applied to the
vertical axis. A *second* surviving axis is refused: a pointwise metric reduces over the
axis it was given and draws the rest as a map, and there is nothing it could do with a
third.

## Costs

The aligned pair now holds every matched step rather than one map, which for a long record
is the difference between megabytes and gigabytes. Three things keep that in hand, and one
is up to you:

- the reference lane is cropped to the model's own extent **before** it is read, not after
  (`align()` crops it either way; doing it early is what saves the memory);
- the finer lane is binned down to the coarser one's cadence before anything is
  regridded;
- coverage is computed once rather than per step when the model's valid cells do not move,
  which for a land mask is always;
- and `select={"time": ...}` is still yours to narrow. A lane above ~2 GB says so before
  it is read.

## Interpolated maps for scattered stations

Everything above is one comparison, scored against a reference that itself varies in
space and time — a satellite record, a climatology. A mooring network is a different
shape entirely: dozens of *separate* comparisons, each at one place, each reduced to a
single number per metric (`Comparison.metrics()`, not `.pointwise_metrics()` — a
station has one cell, so there is nothing to score pointwise). Plotting those as
discrete markers on a map answers one metric at a time and reads poorly once two
stations sit close together.

A single CTD cast maps the same way: it is a place through *depth* rather than through
*time*, but it is still one place — `compare()` scores a `profile`-featureType
reference over its own depth axis automatically, so each cast reduces to one full-water-
column number per metric, no `over=` needed. `map_metrics` treats a mooring and a cast
as the same shape (a single-position station); a set can even mix the two.

`osk.map_metrics(mooring_set)` (or `mooring_set.map_metrics()`) instead **fits a smooth
surface through the scattered values** — a cross-validated spline
([verde](https://www.fatiando.org/verde/)) in a local projection centred on the
stations — and draws it with this same `skill_map` family: one panel per metric, the
same `metric_colors` policy (a bias panel symmetric about zero, a correlation panel on
(−1, 1)), the true station values overlaid as dots in that same colour scale:

```python
mooring_set = osk.compare(reference=osk.find(featureType="timeSeries", bbox=...),
                           test="ciofs3", variables=["temperature"])
mooring_set.map_metrics()                              # bias, crmsd, corr, sigma_ratio
mooring_set.map_metrics(metrics=("corr", "n"))
mooring_set.map_metrics(grid="regular")                # skip the model's own grid/mask
osk.map_metrics(a_ciofs_report_metrics_table, test="ciofs3")   # a plain table works too
```

**Every metric here is a full-record statistic** — deliberately *not* split by year.
Splitting first and mapping a median (or a per-year facet) sounds appealing but is the
wrong default: a full-record correlation tests whether the model gets the whole signal
right, inter-annual variability included, which a median-of-per-year-correlations
cannot; and it means every station contributes exactly one clean value instead of a
pile of per-year rows to reconcile. The consequence worth knowing: two stations with
very different record lengths (one moored for two years, one for twenty) are
interpolated as equals — `map_metrics` warns when the spread is wide (see `n`'s own
column), but does not resolve it for you.

**When a time slice matters more than the whole record**, pool it yourself and pass
one entry per slice with `rows=`, which draws as `skill_map`'s ordinary stacked rows —
metrics across, rows down:

```python
mooring_set.map_metrics(rows={"DJF": winter_set, "MAM": spring_set,
                               "JJA": summer_set, "SON": fall_set})
```

**This does not route around land.** The interpolation only knows Euclidean distance
in the map's own plane, so two moorings on opposite shores of a peninsula (Cook Inlet,
most obviously) are blended as if the water between them were open. A model grid's
ocean mask — used automatically when `grid="model"` (the default) can read one — keeps
the surface from being *drawn* on land, but nothing stops a station on the far side of
a barrier from *influencing* it. Read the coastline, not just the colour. Barrier-aware
interpolation ([DIVAnd](https://github.com/gher-uliege/DIVAnd.jl)) is Julia-only and
out of scope here; a distance mask (`maxdist=`) at least keeps the surface from
extrapolating confidently across a large open gap between survey lines.

**Should the surface vary smoothly at all?** The spline above is the default because a
gradual gradient between stations is often plausible, but it can invent one across
water two stations say nothing about — blending a good station into a bad one across a
strait the fit cannot see. `method=` picks a different interpolator when that isn't
the story you want to tell:

```python
mooring_set.map_metrics(method="nearest")       # Voronoi tiles: hard edges, no invented gradients
mooring_set.map_metrics(method="knn", knn_k=8)   # softer than nearest, still local
mooring_set.map_metrics(method="linear")         # faceted; NaN outside the stations' convex hull
```

`"nearest"` also adapts to station density for free — a dense survey cluster gets
small tiles, an isolated mooring a large one — with no smoothing parameter to tune.
`"linear"`/`"cubic"` triangulate instead: faceted rather than smooth, and never
extrapolate past the outermost stations. If a dense cluster of stations threatens to
dominate a sparser region regardless of method (pure vote-counting, not real signal),
pool it first with `block_spacing=` (metres):

```python
mooring_set.map_metrics(method="nearest", block_spacing=15_000)  # pool a 15 km cluster
```

See [`ocean_skill/plot/map_metrics.py`](../ocean_skill/plot/map_metrics.py) for the
full mechanics (duplicate-position pooling, the antimeridian-safe projection, the
model-grid vs. regular-grid fallback), and
[`docs/plot_styling_reference.md`](plot_styling_reference.md#station-dots-skill_map-items-built-by-map_metrics)
for the station-dot overlay and the `groups=` grouping option on `taylor`/`target`.

## See also

- [`docs/plot_styling_reference.md`](plot_styling_reference.md) — the `skill_map` family's
  parameters, its per-panel colorbars, and the metric colour policy.
- [`docs/caching.md`](caching.md) — a scored aligned pair is cached like any other, and is
  larger.
- [`ocean_skill/metrics.py`](../ocean_skill/metrics.py) — the metric registry. Adding a
  skill score (Willmott, Murphy/MSESS) is one `register()` call plus a colour row; which
  definition "skill score" should mean is deliberately still an open question.
