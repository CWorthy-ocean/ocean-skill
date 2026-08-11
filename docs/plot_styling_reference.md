# Plot styling reference

`field_row`/`field_grid` (and `Comparison.plot()`/`ComparisonSet.plot()`, which forward
arbitrary kwargs to them) expose **7 style parameters**, each a dict that merges onto a
built-in default and is unpacked straight into one specific matplotlib/cartopy call —
so any keyword that call accepts works, not just a hand-picked subset. A few more
parameters aren't styling dicts at all (`title`, `metric_keys`, `shared_limits`,
`shared_axis_labels`, `shared_axes`) — see [Other
parameters](#other-parameters-not-styling-dicts) at the end of this doc.

> **The 7 `*_kwargs` dicts are `renderer="matplotlib"` only.** Each maps onto a
> matplotlib or cartopy call, so none of them do anything with `renderer="holoviews"`
> — passing one warns once and is dropped, naming which renderer to use instead.
> **`title` and `metric_keys` are the exception**: both renderers honor them the same
> way — same kwarg, same meaning, no renderer switch required. See their entries under
> [Other parameters](#other-parameters-not-styling-dicts) for exactly what each
> renderer does with them (matplotlib: a `Figure.suptitle`/corner text box; holoviews:
> a `Layout` title/the difference panel's own title — the closest bokeh equivalents).

Every `*_kwargs` parameter merges with its defaults rather than replacing them
wholesale: `title_kwargs={"fontsize": 10}` doesn't require also passing `color`,
`weight`, etc. — those keep their default values.

## Quick reference

| Parameter | Draws | Underlying call |
|---|---|---|
| [`title_kwargs`](#title_kwargs) | each panel's title (test / reference / difference) | [`Axes.set_title`](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_title.html) |
| [`colorbar_kwargs`](#colorbar_kwargs) | both colorbars (shape, label, ticks) | [`Figure.colorbar`](https://matplotlib.org/stable/api/_as_gen/matplotlib.figure.Figure.colorbar.html) + [`Colorbar.set_label`](https://matplotlib.org/stable/api/_as_gen/matplotlib.colorbar.Colorbar.html) + [`Axes.tick_params`](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.tick_params.html) |
| [`gridline_kwargs`](#gridline_kwargs) | the lat/lon grid lines | [`GeoAxes.gridlines`](https://scitools.org.uk/cartopy/docs/latest/reference/generated/cartopy.mpl.geoaxes.GeoAxes.gridlines.html) |
| [`tick_label_kwargs`](#tick_label_kwargs) | the lat/lon tick **labels** | [`Gridliner.xlabel_style`/`ylabel_style`](https://scitools.org.uk/cartopy/docs/latest/reference/generated/cartopy.mpl.gridliner.Gridliner.html) |
| [`row_label_kwargs`](#row_label_kwargs) | the rotated variable name (field_grid only) | [`Axes.text`](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.text.html) |
| [`metrics_kwargs`](#metrics_kwargs) | the bias/rmse/corr corner box | [`Axes.text`](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.text.html) |
| [`suptitle_kwargs`](#suptitle_kwargs) | the overall figure title | [`Figure.suptitle`](https://matplotlib.org/stable/api/_as_gen/matplotlib.figure.Figure.suptitle.html) |

Most of these ultimately configure a matplotlib `Text` object (title, tick label, axes
text, colorbar label all are one) — see [Common Text properties](#common-text-properties)
for the keys you'll reach for most: `fontsize`, `color`, `weight`, `rotation`, `ha`/`va`.

---

## `title_kwargs`

Draws each panel's title via [`Axes.set_title(lab, **title_kwargs)`](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_title.html).
Accepts any `Text` property plus `set_title`'s own `loc`/`pad`/`y`.

**Default:** `{"fontsize": 8}`

```python
c.plot(title_kwargs={"fontsize": 11, "color": "darkred", "weight": "bold"})
```

## `colorbar_kwargs`

One dict, split internally by key prefix into three separate calls — this is the only
parameter that isn't a direct 1:1 passthrough, since a colorbar's shape, its label, and
its tick labels are three different matplotlib methods with non-overlapping keywords:

| Key prefix | Goes to | Doc |
|---|---|---|
| *(none)* | [`Figure.colorbar(im, **kwargs)`](https://matplotlib.org/stable/api/_as_gen/matplotlib.figure.Figure.colorbar.html) | shape/placement: `shrink`, `aspect`, `pad`, `orientation`, `extend`, ... |
| `label_*` | [`Colorbar.set_label(lab, **kwargs)`](https://matplotlib.org/stable/api/_as_gen/matplotlib.colorbar.Colorbar.html) (strip prefix) | the label text — any `Text` property, e.g. `label_size`, `label_color` |
| `tick_*` | [`Axes.tick_params(**kwargs)`](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.tick_params.html) on `cbar.ax` (strip prefix) | tick labels — `tick_labelsize`, `tick_color`, `tick_direction`, ... |

**`Figure.colorbar`'s own keywords worth knowing** (beyond `shrink`/`aspect`/`pad`):
`orientation` (`"vertical"`/`"horizontal"`), `extend` (`"neither"`/`"both"`/`"min"`/`"max"`
— draws a pointed end for out-of-range values), `ticks` (explicit tick locations),
`format` (a format string or `Formatter` for tick labels).

**Defaults** — `field_row` and `field_grid` start from different values (one page-width
horizontal row vs. several stacked vertical ones):

| Key | `field_row` | `field_grid` |
|---|---|---|
| `orientation` | `"horizontal"` | `"vertical"` |
| `pad` | `0.04` | `0.015` |
| `shrink` | `0.85` | `0.8` |
| `aspect` | `30` | `15` |
| `label_size` | `7` | `6` |

```python
c.plot(colorbar_kwargs={
    "shrink": 1.0,        # -> fig.colorbar(..., shrink=1.0)
    "aspect": 10,         # lower = wider
    "label_size": 9,      # -> cbar.set_label(lab, size=9)
    "tick_labelsize": 8,  # -> cbar.ax.tick_params(labelsize=8)  (new: not settable before)
})
```

## `gridline_kwargs`

Draws the lat/lon grid **lines** via
[`GeoAxes.gridlines(draw_labels=True, **gridline_kwargs)`](https://scitools.org.uk/cartopy/docs/latest/reference/generated/cartopy.mpl.geoaxes.GeoAxes.gridlines.html).
Common keys: `linewidth`, `color`, `alpha`, `linestyle`, `xlocs`/`ylocs` (explicit
gridline positions).

**Default:** `{"linewidth": 0.2, "color": "0.6", "alpha": 0.6}`

```python
c.plot(gridline_kwargs={"linewidth": 0.5, "color": "gray", "linestyle": "--"})
```

## `tick_label_kwargs`

Draws the lat/lon tick **labels** (distinct from the grid lines above) via the
[`Gridliner`](https://scitools.org.uk/cartopy/docs/latest/reference/generated/cartopy.mpl.gridliner.Gridliner.html)'s
`xlabel_style`/`ylabel_style` — both set to this same dict, which is any `Text`
property (most commonly `size`/`color`/`rotation`).

**Default:** `{"size": 5}`

```python
c.plot(tick_label_kwargs={"size": 7, "color": "0.3"})
```

## `row_label_kwargs`

`field_grid` only — draws the rotated variable name at the left edge of each row via
[`Axes.text(-0.18, 0.5, row_label, **row_label_kwargs)`](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.text.html).
Any `Text` property.

> The row label *text* appears in both renderers (holoviews prefixes it onto each
> row's first panel title, bokeh having no floating-text equivalent) — it's only
> this **styling dict** that is matplotlib-only.

**Default:** `{"fontsize": 7, "rotation": 90, "va": "center", "ha": "center", "weight": "bold"}`

```python
physics.plot(row_label_kwargs={"fontsize": 9, "color": "navy"})
```

## `metrics_kwargs`

Draws the bias/rmse/corr corner box via
[`Axes.text(0.02, 0.02, txt, **metrics_kwargs)`](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.text.html).
Any `Text` property, including `bbox` (a dict of
[`FancyBboxPatch`](https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.FancyBboxPatch.html)
properties — `facecolor`, `alpha`, `pad`, `edgecolor`, `linewidth`).

**Default:**
```python
{
    "fontsize": 5.5, "va": "bottom", "ha": "left",
    "bbox": {"facecolor": "white", "alpha": 0.75, "pad": 2, "edgecolor": "0.6", "linewidth": 0.4},
}
```

```python
c.plot(metrics_kwargs={"fontsize": 8, "bbox": {"facecolor": "lightyellow", "alpha": 0.9}})
```
Note: passing `bbox` replaces the *whole* bbox dict (not a deep-merge) — repeat any
sub-keys you still want.

## `suptitle_kwargs`

Draws the overall figure title (the `title=` argument's actual text) via
[`Figure.suptitle(title, **suptitle_kwargs)`](https://matplotlib.org/stable/api/_as_gen/matplotlib.figure.Figure.suptitle.html).
Any `Text` property.

**Default:** `{"fontsize": 9}` (`field_row`) / `{"fontsize": 10}` (`field_grid`)

```python
physics.plot(title="ROMS GOM vs. WOA", suptitle_kwargs={"fontsize": 13, "weight": "bold"})
```

---

## Common Text properties

`title_kwargs`, `tick_label_kwargs`, `row_label_kwargs`, `metrics_kwargs`,
`suptitle_kwargs`, and `colorbar_kwargs`'s `label_*` keys all end up as keyword
arguments to a matplotlib [`Text`](https://matplotlib.org/stable/api/_as_gen/matplotlib.text.Text.html)
object. The full property list is at
[matplotlib's Text properties guide](https://matplotlib.org/stable/users/explain/text/text_props.html);
the ones you'll actually reach for:

| Key | Controls | Example values |
|---|---|---|
| `fontsize` / `size` | text size (points) — `set_title`/`text`/`suptitle` use `fontsize`, cartopy's label styles and `Colorbar.set_label` use `size` | `6`, `10`, `"small"`, `"large"` |
| `color` | text color | `"black"`, `"0.3"`, `"#1d4ed8"` |
| `weight` | boldness | `"normal"`, `"bold"` |
| `style` | slant | `"normal"`, `"italic"` |
| `rotation` | degrees counter-clockwise | `0`, `90` |
| `ha` / `va` | horizontal/vertical alignment | `"left"`/`"center"`/`"right"`, `"bottom"`/`"center"`/`"top"` |
| `family` | font family | `"sans-serif"`, `"serif"` |

---

## Other parameters (not styling dicts)

These don't map onto a single matplotlib/cartopy call the way the seven above do —
they control *what data feeds the plot*, *which panels get labelled*, or (for `title`/
`metric_keys`) work identically on **both** renderers.

### `title`

The overall title's *text* — shared by both renderers, unlike everything else on this
page. matplotlib draws it via `Figure.suptitle` (styled by `suptitle_kwargs`, static
only); holoviews sets it as the interactive `Layout`'s own `title` option, which
renders above the whole grid of panels but has no equivalent styling knob (font/size
follow holoviews' own theme, not `suptitle_kwargs`).

```python
physics.plot(title="ROMS GOM vs. WOA (100m, Jan 2001)")                    # matplotlib
physics.plot(title="ROMS GOM vs. WOA (100m, Jan 2001)", renderer="holoviews")  # same text, interactive
```

### `metric_keys`

Picks *which* of `metrics.compute()`'s values are shown, and in what order — a tuple of
any subset of `bias`, `rmse`, `corr`, `sigma_ratio`, or any other key `metrics.compute()`
returns. Also shared by both renderers, just displayed differently: matplotlib draws
them in the corner-box `Axes.text` (styled by `metrics_kwargs`, static only); holoviews
folds the same numbers into the difference panel's own title (`"difference (bias=...,
rmse=..., corr=...)"`) — the closest bokeh equivalent of a free-floating annotation
that survives pan/zoom/resize as cleanly as a title does.

**Default:** `("bias", "rmse", "corr")`

```python
c.plot(metric_keys=("corr", "sigma_ratio"))
```

### `shared_axes` (holoviews only)

The interactive analog of `shared_axis_labels` below, but for *pan/zoom linking*
rather than which panels show coordinate labels: `True` (the default) links every
panel's pan/zoom together — test, reference, and difference within a row via
`field_row`, and every row too via `field_grid`, since they share the same underlying
bokeh `Range` object. Meaningful whenever every panel shares one geographic domain
(the common case). Set `False` if `field_grid`'s rows genuinely cover different
regions and independent zooming makes more sense.

**Default:** `True`

```python
physics.plot(renderer="holoviews", shared_axes=False)   # each row zooms independently
```

### `shared_limits`

`field_grid` only. Makes every row's colour scale — and its difference range —
span *all* rows' data combined, instead of each row scaling to its own. Meaningful
only when every row is the same variable (e.g. one depth or one time per row):
different variables have unrelated ranges and units, so sharing across those makes
the colours meaningless relative to the numbers on the bar. Warns (once) if the
rows' `standard_name`s actually differ.

**Default:** `False` (each row scales independently, as before this option existed)

```python
depths = osk.compare(reference="woa23_nitrate_month01", test="my_run",
                      variables=[NITRATE], depths=("surface", 50, 100))
depths.plot(shared_limits=True)   # same colour scale on every depth row
```

### `shared_axis_labels`

Draws grid lines on every panel either way, but controls whether coordinate
**labels** repeat on every panel or only appear where they're not redundant: the
leftmost column (latitude) and the bottom row (longitude) — the usual convention
for a grid of maps, since three side-by-side copies of the same latitude labels say
nothing three copies didn't already say once. Set `False` to label every panel's
axes independently (`field_row`'s single row is always effectively "bottom", so
this only visibly changes anything for `field_grid`'s left column).

**Default:** `True`

```python
physics.plot(shared_axis_labels=False)   # every panel gets its own lat/lon labels
```

### `labels` (summary diagrams)

`taylor`, `target` and `paired` only. Chooses how each point is identified:

| Value | Effect |
|---|---|
| `"legend"` | a key beneath the axes, one entry per point (or per group, with `color_by`/`marker_by`) |
| `"annotate"` | each label written beside its own marker |
| `None` | neither |

**Defaults:** `"legend"` for `taylor` and `paired`, `"annotate"` for `target` — each
diagram's usual idiom. Both modes work on both diagrams, so the default is only a
starting point.

Pick by how many points there are. Annotation is more direct — no eye-travel between
key and marker — but matplotlib has no label-repel, so labels collide once points
cluster. A legend always stays legible and costs the lookup.

```python
suite.paired(labels="annotate")   # both panels annotated
suite.paired(labels="legend")     # one shared key below both panels
suite.target(labels="legend")     # target keyed like a Taylor
```

`paired` applies **one** choice to both panels, since they show the same points and
labelling them two different ways in one figure reads as two unrelated plots. With
`"legend"` the key is drawn once beneath both panels rather than once per panel.

Honored by **both renderers** for `target`. (`taylor` and `paired` delegate to
matplotlib interactively — bokeh has no floating polar axis — so they label the same
way either way.) Colours are pinned to tab10 by level index in both, so a diagram keeps
its colours when you switch renderer.

### `color_by` / `marker_by`

Two grouping channels, and only two: colour carries one dimension, marker shape the
other. Any field of the metric record works (`variable`, `depth`, `test`,
`reference`, ...):

```python
suite.taylor(color_by="variable", marker_by="test")   # 3 models × 6 variables
```

Naming only `marker_by` colours by the *same* groups, so the legend's swatches match
the points rather than varying with nothing to explain them.

A third channel (size, or filled vs hollow) is possible but deliberately absent: marker
size already reads as magnitude on these diagrams, and three encodings on one point
tend to be slower to decode than two diagrams side by side. If you need a third
dimension, split it across figures.

Interactively, bokeh cannot show two independent legend blocks, so `color_by` +
`marker_by` produces combined entries (`"chl · runA"`) where the static diagram shows
a colour block and a marker block. Same groups, one legend instead of two.
