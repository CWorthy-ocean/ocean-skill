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
scored.metrics()            # the tidy one-row-per-comparison table, as always
scored[0].maps()            # the metric maps as an xr.Dataset
scored[0].maps("rmse", "n") # or any other registered metric
scored.taylor()             # still works: these are the same numbers
scored.save("gom_vs_modis") # figure + CSV, same layout as any other comparison
```

Because they are the same numbers, a scored set pools with an unscored one on a single
diagram — `osk.summary([scored, surface_nutrients])` — even though the two cannot share
a figure of *fields*. Only the metrics record travels, so nothing recomputes the maps.

`renderer="holoviews"` gives the interactive version of the same figure, with the overall
value in each panel's title instead of its corner.

## How the two sources are matched in time

This is the part you cannot infer from the signature, and the part most likely to matter.

**Alignment in time mirrors alignment in space.** The test is brought onto the
reference's time axis exactly as it is brought onto the reference's grid: the reference is
the frame, and nothing is done to it. How the test moves depends on which side is coarser.

| Situation | What happens | Spatial analogue |
|---|---|---|
| Test much finer (hourly model, daily product) | test steps are **averaged into** the reference's bins | `conservative_normed` |
| Comparable cadences, offset stamps (a daily mean at 12:00 vs a composite at 00:00) | steps are **paired** by nearest match within half a bin | `bilinear` |
| Reference is instantaneous rather than a composite | test is **sampled** at those instants | `bilinear` |
| Reference finer than the test | **refused**, naming both cadences | — (no counterpart: this package never coarsens the reference on its own) |

Averaging the finer lane is a default rather than something to request for the same
reason area-averaging is: nobody has to ask for `conservative_normed` either. And the
binning happens *before* the regrid, which is not just tidier — it turns 8760 regridded
fields into 365.

**Bin edges come from the reference's own stamps**, as cell edges in longitude and
latitude do. Whether a stamp marks the *start* of its bin or the *middle* is read off the
stamps themselves: a product that labels a bin with its first instant lands on a period
boundary (midnight for a daily or 8-day composite, the first of the month for a monthly
one), while one that labels the middle deliberately does not — WOA and OceanSODA stamp the
15th, a ROMS daily average noon. Both spellings therefore bin their own period correctly.
Override with `bin_anchor="start"`/`"center"` if a product does something else.

**Composite or snapshot** is read from CF `cell_methods` on the variable, then from
`period` in the catalog entry. When neither says, the reference is taken to be a composite
and you get a warning naming the assumption, the override (`time_method="nearest"`) and
the permanent fix. That is the same warn-and-proceed `align()` already does for units it
cannot verify.

Everything the matching decided lands in the aligned result's `attrs`, beside
`regrid_method`: `match_method`, `match_reason`, `n_matched`, `bin_anchor`, how many bins
came out empty or short. Both are choices a reader of the numbers is entitled to see.

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
- the test is binned down to the reference's cadence before anything is regridded;
- coverage is computed once rather than per step when the model's valid cells do not move,
  which for a land mask is always;
- and `select={"time": ...}` is still yours to narrow. A lane above ~2 GB says so before
  it is read.

## See also

- [`docs/plot_styling_reference.md`](plot_styling_reference.md) — the `skill_map` family's
  parameters, its per-panel colorbars, and the metric colour policy.
- [`docs/caching.md`](caching.md) — a scored aligned pair is cached like any other, and is
  larger.
- [`ocean_skill/metrics.py`](../ocean_skill/metrics.py) — the metric registry. Adding a
  skill score (Willmott, Murphy/MSESS) is one `register()` call plus a colour row; which
  definition "skill score" should mean is deliberately still an open question.
