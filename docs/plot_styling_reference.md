# Plot styling reference

`field_row`/`field_grid`/`field_facet`/`skill_map` (and `Comparison.plot()`/
`ComparisonSet.plot()`/`Field.plot()`, which forward arbitrary kwargs to them) expose
**7 style parameters**,
each a dict that merges onto a
built-in default and is unpacked straight into one specific matplotlib/cartopy call —
so any keyword that call accepts works, not just a hand-picked subset. An eighth,
[`frame_label_kwargs`](#frame_label_kwargs), belongs to `field_movie` alone, there being
no per-frame label on a still. A few more parameters aren't styling dicts at all
(`title`, `metric_keys`, `metric_names`, `shared_limits`, `shared_axis_labels`,
`shared_axes`) — see [Other parameters](#other-parameters-not-styling-dicts) at the end
of this doc.

> **The `*_kwargs` dicts are `renderer="matplotlib"` only.** Each maps onto a
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

**You should rarely need these dicts for font sizes.** Every text size is derived from
the figure's own geometry, so changing `figsize` or the number of rows re-sizes all of
them together — see [Automatic sizing](#automatic-sizing). Reach for a `*_kwargs` dict
when you want a *specific* size, colour, or weight; reach for `font_scale` when you just
want everything bigger or smaller.

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
| [`frame_label_kwargs`](#frame_label_kwargs) | a movie's per-frame timestamp (`field_movie` only) | [`Axes.text`](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.text.html) |
| [`line_kwargs`](#line_kwargs) | every line of a `series` panel (`series` only) | [`Axes.plot`](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html) |
| [`legend_kwargs`](#legend_kwargs) | a `series` panel's key, or the `locations` map's | [`Axes.legend`](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html) |

Most of these ultimately configure a matplotlib `Text` object (title, tick label, axes
text, colorbar label all are one) — see [Common Text properties](#common-text-properties)
for the keys you'll reach for most: `fontsize`, `color`, `weight`, `rotation`, `ha`/`va`.

---

## Automatic sizing

Font sizes and the default figure size are computed together, from the geometry of the
figure being drawn ([`ocean_skill/plot/typography.py`](../ocean_skill/plot/typography.py)).
Nothing needs configuring for this; the section is here so the behaviour isn't surprising.

**Why it isn't a set of constants.** Matplotlib point sizes are absolute, and
`constrained_layout` gives text priority over axes. A title fixed at 8pt is right for a
page-width row and ruinous for a 4-inch one: the three maps get squeezed to about a
third of an inch each to make room for their own labels. Nor can you fix that by
retuning the fonts — the fonts are the cause. Sizes therefore follow the space
available, and the figure's own default height follows the type it will carry.

**How sizes relate.** Every role is one base size times fixed ratios, so they move
together and stay in proportion — there is one number for the overall level
(`BASE_PT_AT_1IN`), and it was set by rendering the same comparison at 8 / 11 / 12 / 14pt
titles and picking, not by derivation. Sizes below are what a default page-width
`field_row` gets:

| Role | Default row | Scales with |
|---|---|---|
| panel title | 11.1pt | the grid **cell** |
| suptitle | 12.9pt | the figure **width** (not the row count) |
| colorbar label, row label, axis label, legend | 9.7pt | the grid cell |
| metrics box, colorbar tick labels | 8.4pt | the grid cell |
| lat/lon tick labels, point annotations | 7.3pt | the grid cell |

Nothing goes below **6pt** (`MIN_PT`), the floor most journal style guides accept at final
printed size.

Panel-level type shrinks when its panel does — an eight-row grid gets smaller titles than
a two-row grid. The suptitle deliberately does not: it labels the whole figure, whose
width the row count doesn't change.

> **These are larger than the sizes shipped before.** The first version of this reproduced
> 8pt titles and 5pt coordinate labels, which were matplotlib-era literals nobody had
> chosen and which turned out to be too small. Figures drawn with the old defaults will
> look different — that is the intended change.

### Scaling a figure up or down

**`size`** names the canvas the figure has to fit; **`zoom`** multiplies it.

```python
physics.plot()                  # "page": 8.5in wide, fits an 11in page
physics.plot(zoom=1.5)          # half again as big, type and panels to match
physics.plot(size="slide")      # 13.33 x 7.5 (16:9)
physics.plot(size="free")       # page width, no height cap — see below
physics.plot(size=6.5)          # 6.5in wide
physics.plot(size=(6.5, None))  # 6.5in wide, uncapped
```

| `size` | Width | Height cap | For |
|---|---|---|---|
| `"page"` *(default)* | 8.5in | 11in | reports, papers |
| `"free"` | 8.5in | none | notebooks — a many-row grid gets longer rather than squeezed |
| `"slide"` | 13.33in | 7.5in | 16:9 presentations |
| `"column"` | 3.5in | none | a single journal column (cramped for three maps — see the warning below) |

The **height cap belongs to the canvas**, which matters for many-row grids: `"page"` keeps
the figure printable by squeezing panels, and `"free"` keeps every panel at full height
and lets the figure grow. Previously the 11-inch cap was unavoidable.

`zoom` grows the type *sub-linearly* — a canvas 1.6× larger gets type 1.32× larger — so a
bigger figure buys detail rather than magnification. Use **`font_scale`** instead when you
want the same figure with more prominent type:

```python
physics.plot(zoom=1.5)         # bigger figure, proportionally more detail
physics.plot(font_scale=1.3)   # same figure, 30% larger type
```

Because row height is computed *from* the type, `font_scale=1.3` makes the figure taller
rather than shrinking the maps. Both work in either renderer — they name sizes, which
bokeh has, unlike the seven `*_kwargs` dicts, which name matplotlib calls.

**Default `figsize`.** A row is as tall as the maps' own aspect ratio
(`lon_span / lat_span`) wants, plus the room its type needs. Aspect ratios beyond 0.3–4.0
are letterboxed instead of obeyed. Interactive frames follow the same aspect ratio, so a
tall narrow domain isn't letterboxed in one renderer and fitted in the other. Passing
`figsize` overrides `size`/`zoom` outright and re-derives the type for what you asked for.

**Too small to work.** Below roughly five inches, three maps plus their titles, colorbars
and coordinate labels do not fit, and no font size fixes it — the type is already at its
6pt floor and the panels take the difference. The figure still draws and **warns**, naming
the parameter to change:

```
UserWarning: the maps are only 0.32in wide on a 3.50in canvas — their own titles,
colorbars and coordinate labels have taken the rest, and the type is already at its
6pt floor. Widen the canvas (size=, zoom=, or figsize=) rather than shrinking the text.
```

**Overlong labels (`fit_text`).** No global size can rescue one 50-character CF standard
name in a 2-inch panel, so after the layout settles each title, axis label and colorbar
label is measured against its own box and shrunk *only if it overflows*, never below 6pt.
Labels that fit keep the size the scale chose. Pass `fit_text=False` to disable it and let
long labels overhang.

**Overriding.** An explicit size always wins outright — automatic sizing is a better
default, not a new constraint:

```python
c.plot(title_kwargs={"fontsize": 14})   # exactly 14pt, whatever the geometry
```

---

## `title_kwargs`

Draws each panel's title via [`Axes.set_title(lab, **title_kwargs)`](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_title.html).
Accepts any `Text` property plus `set_title`'s own `loc`/`pad`/`y`.

**Default:** `{"y": 1.0}` plus a `fontsize` **chosen from the figure's geometry** —
11.1pt at the default page-width row, smaller in a tall grid, larger on a bigger canvas.
See [Automatic sizing](#automatic-sizing) below. `y` is pinned to work around a
matplotlib 3.11 title-placement bug over cartopy axes; leave it alone unless you know
you want it moved.

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
| `shrink` | `1.0` | `1.0` |
| `aspect` | `30` | `15` |
| `label_size` | *automatic* (9.7pt at the default row) | *automatic* |
| `tick_labelsize` | *automatic* (8.4pt at the default row) | *automatic* |

Both text sizes come from the shared type scale — see
[Automatic sizing](#automatic-sizing). `tick_labelsize` used to be unset entirely, so
the bar's numbers fell through to matplotlib's 10pt default while the titles around
them were 8pt and the latitude labels 5pt.

**Placement is refitted after the layout settles.** `fig.colorbar` sizes a bar to the
grid *cell* its panels sit in, which also holds the title above and the longitude
labels below — and a cartopy map shrinks inside its own slot to keep its aspect — so
the bar overshot the map at both ends. `field_row`/`field_grid` therefore re-fit each
bar to the drawn extent of the panels it describes: a vertical bar's top and bottom sit
level with the top and bottom of the axes, a horizontal bar's ends with their left and
right edges. Consequences for the keys above:

- `shrink` is now a fraction of *that* extent (centred), not of the cell — hence the
  `1.0` defaults. `shrink=0.6` gives a bar 60% of the maps' height, still centred on
  them.
- `aspect` still sets length:thickness, but one thickness is used for every bar of the
  same orientation in the figure, taken from the shortest. A field row's shared-scale
  bar spans two panels and its difference bar one; per-bar thickness made the first two
  and a half times fatter than the second.
- `pad` is levelled the same way, and for the same reason: it is a fraction of the
  parent's *own* width (or height), so a grid row's shared-scale bar — padded off a
  two-panel span — sat twice as far from its panels as the difference bar beside it.
  One gap in inches, the widest of the group, is used for every bar of that orientation.
- Pass `align_colorbars=False` (a top-level plot option, not a `colorbar_kwargs` key)
  to hand placement back to `constrained_layout` entirely. The refit also stands the
  layout engine down when it finishes, so the positions survive the redraw that
  `savefig` and Jupyter's inline backend perform; a figure returned with the refit on
  is laid out for good.
- Interactive (`renderer="holoviews"`) needs none of this: bokeh attaches a colorbar to
  the plot frame, so it is already flush with the panel. `align_colorbars` is accepted
  and ignored there rather than warned about.

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

**Default:** a `size` **chosen from the figure's geometry** — 7.3pt at the default
page-width row. See [Automatic sizing](#automatic-sizing).

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

**Default:** `{"rotation": 90, "va": "center", "ha": "center", "weight": "normal"}` plus
an automatic `fontsize` (9.7pt at the default page-width row).

```python
physics.plot(row_label_kwargs={"fontsize": 9, "color": "navy"})
```

## `metrics_kwargs`

Draws the bias/rmse/corr corner box via
[`Axes.text(0.02, 0.02, txt, **metrics_kwargs)`](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.text.html).
Any `Text` property, including `bbox` (a dict of
[`FancyBboxPatch`](https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.FancyBboxPatch.html)
properties — `facecolor`, `alpha`, `pad`, `edgecolor`, `linewidth`).

**Default:** an automatic `fontsize` (8.4pt at the default page-width row) plus
```python
{
    "va": "bottom", "ha": "left",
    "bbox": {"facecolor": "white", "alpha": 0.75, "pad": 2, "edgecolor": "0.6", "linewidth": 0.4},
}
```

```python
c.plot(metrics_kwargs={"fontsize": 8, "bbox": {"facecolor": "lightyellow", "alpha": 0.9}})
```
Note: passing `bbox` replaces the *whole* bbox dict (not a deep-merge) — repeat any
sub-keys you still want.

## `frame_label_kwargs`

`field_movie` only — draws the per-frame label (the timestamp, usually) in the top-left
of the test panel via
[`Axes.text(0.02, 0.98, frame_label, **frame_label_kwargs)`](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.text.html).
Any `Text` property, including `bbox`. Mirrors `metrics_kwargs`' box in the bottom-left
of the difference panel, and shares its box styling so the two read as one figure's
annotations rather than two.

**Default:** an automatic `fontsize` (7pt at the default page-width row — a step above
the metrics box, since the frame label is the one thing in the figure that changes and so
the thing being read) plus
```python
{
    "va": "top", "ha": "left", "family": "monospace",
    "bbox": {"facecolor": "white", "alpha": 0.75, "pad": 2, "edgecolor": "0.6", "linewidth": 0.4},
}
```

`family="monospace"` is deliberate: in a proportional font a counting timestamp jitters
sideways as its digits change width, which is invisible on a still and distracting once
it is animated. Override it if your labels aren't numeric.

```python
runs.movie(save="m.mp4", frame_label_kwargs={"fontsize": 11, "color": "navy"})
runs.movie(save="m.mp4", frame_label=False)      # no label at all
```

Interactively the frame label is the slider's own value (and the test panel's title), so
this dict is matplotlib-only like the rest — see [Movies](movies.md).

## `suptitle_kwargs`

Draws the overall figure title (the `title=` argument's actual text) via
[`Figure.suptitle(title, **suptitle_kwargs)`](https://matplotlib.org/stable/api/_as_gen/matplotlib.figure.Figure.suptitle.html).
Any `Text` property.

**Default:** an automatic `fontsize` — 12.9pt at the default page-width row. Unlike the
per-panel sizes this one does **not** shrink as you add rows: it labels the whole
figure, whose width does not change. See [Automatic sizing](#automatic-sizing).

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

The one-source families default it, the panels having said *when* but nothing having
said *what*. `Field.plot()` composes what a `select=` has taken off the page — the
variable's short name, the depth (unless a row or facet axis already names it), and the
instant (when time has collapsed to a single scalar rather than staying the panels'
own axis) — as `source: variable · depth · time`, each part dropped when it does not
apply (`alkalinity · 50 m · 2013-01-16`, or just `alkalinity` when the panels already
say when and there is no single depth to add). Pass `title=""` to drop it, or any
string to replace it. `Field.movie()` defaults to the plainer `field_title` (variable
only); on an interactive movie the name joins each frame's label (`alkalinity —
2013-01-16`) instead, bokeh's only title there being the panel's own. A single
comparison (`Comparison.plot()`, a `test | reference | difference` row) defaults to the
same `variable · depth · time` through the shared spelling, since a lone row has no
left-edge label to name its variable (that is `field_grid`'s, and only when it stacks
several rows). A `field_grid` of stacked comparisons still draws no per-row title: each
row is named down its left edge, and one overall title sits up top if given.

### `font_scale`

Multiplies **every** text size at once, keeping the proportions between them — the
alternative to editing six `*_kwargs` dicts in step. Shared by both renderers (it names a
size, not a matplotlib call) and honored by the summary diagrams too. Because the default
figure height is computed from the type it has to carry, a larger `font_scale` makes the
figure taller rather than squeezing the maps. See [Automatic sizing](#automatic-sizing).

**Default:** `1.0`

```python
physics.plot(font_scale=1.3)                          # larger type, taller figure
physics.plot(font_scale=1.3, renderer="holoviews")    # same, interactively
comparisons.taylor(font_scale=0.9)
```

### `size` and `zoom`

`size` names the canvas the figure has to fit; `zoom` multiplies it. Height, panel sizes
and every font size are then derived from that, so this replaces working out a `figsize`
tuple by hand. Both are shared by the renderers: interactively they scale the bokeh frame
by the same ratio. See [Scaling a figure up or down](#scaling-a-figure-up-or-down) for the
preset table and how `zoom` differs from `font_scale`.

**Defaults:** `size="page"`, `zoom=1.0`

```python
physics.plot(size="slide")            # 13.33 x 7.5
physics.plot(size="free")             # page width, no height cap
physics.plot(zoom=1.5)                # half again as big
physics.plot(size=(6.5, None))        # 6.5in wide, uncapped
physics.plot(figsize=(7, 2.5))        # overrides both
```

### `fit_text`

After the layout settles, measures each title, axis label and colorbar label against the
box it labels and shrinks *only* the ones that overflow — for one long CF standard name in
a small panel, which no choice of scale can accommodate. Never goes below 6pt, and never
touches a size you set yourself in a `*_kwargs` dict. Set `False` to leave every label at
its nominal size and let long ones overhang. Silently ignored interactively: bokeh lays out
its own text.

**Default:** `True`

```python
physics.plot(fit_text=False)
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

`field_grid` and a two-axis `field_facet`. Makes every row's colour scale — and, in
`field_grid`, its difference range — span *all* rows' data combined, instead of each
row scaling to its own. Meaningful only when every row is the same variable over a
comparable range: different variables have unrelated ranges and units, so sharing
across those makes the colours meaningless relative to the numbers on the bar.
`field_grid` warns (once) if the rows' `standard_name`s actually differ.

The same caution applies to a `field_facet` grid of depths, for a subtler reason: the
rows *are* one variable, but nitrate at 100 m and at the surface still span unrelated
ranges, so one scale across them pushes every surface panel to the bottom of the bar
and hides the monthly change the figure exists to show. Reach for it when the levels
you picked genuinely share a range, not by default.

**Default:** `False` (each row scales independently, as before this option existed)

```python
depths = osk.compare(reference="woa23_nitrate_month01", test="my_run",
                      variables=[NITRATE], depths=("surface", 50, 100))
depths.plot(shared_limits=True)   # same colour scale on every depth row
```

**The movie families default it to `True`** instead, and mean something slightly
different by `False`: a movie's scale is fixed for its whole length either way, and the
option only picks whether it is derived from every frame or from the first. A scale
re-derived per frame would make the ruler move with the field, which is unreadable rather
than merely inconsistent — see
[Movies](movies.md#one-colour-scale-for-the-whole-movie).

### Movie-only parameters

`field_movie` and `facet_movie` add `save`, `fps`, `dpi`, `every`, `frame_label`,
`frame_label_kwargs`, `widget`/`player` (interactive), `progress`, and — interactively
only — `hover`, `rasterize` and `tiles` (on by default; see [Movies](movies.md)). They
are documented together in [Movies](movies.md), since they only mean anything once
there is more than one frame.

### `metric_names` (`skill_map` only)

Which pointwise metric maps become panels, and in what order — `metric_names=("corr",
"bias")` draws those two, in that order. Honored by **both** renderers.

Not to be confused with two neighbours it sits between:

| Parameter | Family | Means |
|---|---|---|
| `metric_names` | `skill_map` | which metrics get a **panel** |
| `metric_keys` | `field_row`/`field_grid` | which scalars appear in the **corner box** |
| `metrics` | (item payload) | the overall scalar **record** itself |

A name the comparison never computed pointwise **raises**, naming what it did compute:
dropping the panel would be invisible, since a figure of three maps looks exactly like a
figure of three maps. Which metrics exist is decided when the maps are prepared —
`compare(..., over="time", metrics=("bias", "rmse", ...))` — because each is a full
reduction over the scored axis and cannot be conjured at draw time. `skill_map` therefore
accepts no `metric_keys`: one metric per panel leaves nothing to select.

### One colorbar per panel (`skill_map` only)

Every other map family gives a colour scale to a *row* (`field_row`, `field_grid`) or to
the *whole grid* (`field_facet`), because in each case the panels sharing it are the same
quantity. `skill_map`'s panels are different quantities — a bias in mmol m-3, a
dimensionless correlation, a variability ratio — so each carries **its own scale and its
own bar**, and there is deliberately no `shared_limits` to ask for one.

The colours come from
[`colormaps.metric_colors`](../ocean_skill/colormaps.py), which both renderers call, so a
bias panel is diverging and symmetric about zero, a correlation panel spans a fixed
(−1, 1), a variability ratio centres on 1, and an error magnitude pins its low end at 0 —
identically in either backend. Edit `_METRIC_CMAPS`/`_METRIC_RANGES` there to change any
of it, the same way `_SEQUENTIAL_CMAPS`/`_RANGES` work for variables.

Because the bars are vertical and per-panel, this family charges `PANEL_W_FRACTION` of
each cell to the map rather than the wider allowance a shared bar leaves. Passing
`colorbar_kwargs={"orientation": "horizontal"}` is honored but warns: the grid's *height*
is not re-charged for a bar under every panel, so pass `figsize=`/`zoom=` with it.

### Station dots (`skill_map` items built by `map_metrics`)

`osk.map_metrics(mooring_set)` interpolates many stations' per-station metrics onto a
smooth surface and draws it with this same `skill_map` family — see
[`docs/skill_maps.md`](skill_maps.md#interpolated-maps-for-scattered-stations) for what
it does and does not do. Its items carry one thing an ordinary scored comparison's
never do: a `stations` entry, drawn as a dot per station in **the same colour scale as
the surface underneath it**, so a reader can tell where the surface has actual support
and where it is only filling a gap between stations. There is no parameter for this —
it rides on the item, not on `skill_map`'s own signature — so it is not something you
pass, only something a `map_metrics` figure always shows.

### `method`, `knn_k`, `block_spacing` (`map_metrics`)

Which interpolator fits the scattered per-station values before `map_metrics` draws
them — see [`docs/skill_maps.md`](skill_maps.md#interpolated-maps-for-scattered-stations)
for the honesty trade-off each one makes. `method="spline"` (the default) is a smooth,
cross-validated fit; `"nearest"` tiles the map by whichever station is closest (hard
edges, no invented gradients, density-adaptive for free); `"knn"` softens that with the
mean of the `knn_k` nearest stations (default 5); `"linear"`/`"cubic"` triangulate
instead (faceted, and only defined inside the stations' convex hull). `block_spacing`
(metres) pools stations within each block to their median before fitting, for any
method — the fix for a dense cluster otherwise outvoting a sparser region.

**Default:** `method="spline"`, `knn_k=5`, `block_spacing=None` (no pre-pooling)

```python
mooring_set.map_metrics(method="nearest")
mooring_set.map_metrics(method="knn", knn_k=8)
mooring_set.map_metrics(method="nearest", block_spacing=15_000)
```

### `shared_limits`, `layout` (`skill_map`)

Two rows of `skill_map` — several comparisons stacked, one row each — default to
independent colour scales per row (see the family's own docstring for why: a
different *metric* genuinely has no shared scale to have). `shared_limits=True` pools
each metric's values over every row before choosing its limits and draws one
colorbar per metric spanning the rows, so the same metric reads on one scale across
comparisons — the CIOFS3-vs-Hindcast case a stacked figure exists for. The static
renderer consolidates the bar itself; the interactive one gives every panel the same
limits but still draws its own bar (bokeh has no cross-panel colorbar).

`layout="columns"` transposes the grid — comparisons across, metrics down — instead
of the default `"rows"` (comparisons down, metrics across). Both options apply only
with more than one item; a single comparison's panels have no rows to share or
transpose.

**Default:** `shared_limits=False`, `layout="rows"`

```python
mooring_set.map_metrics(rows={"CIOFS3": ciofs3_set, "Hindcast": hindcast_set},
                         shared_limits=True, layout="columns")
```

### `ncols` (`field_facet` and `skill_map`)

How many columns the panels are laid out in. By default there is no fixed answer:
[`typography.facet_layout`](../ocean_skill/plot/typography.py) picks the orientation
from the domain's own aspect ratio, because a grid that suits a wide box is wrong for a
tall one and vice versa (`skill_map` uses the same rule for the same reason — four
metrics have no inherent order, so nothing about the fold carries meaning). A Gulf-of-Mexico box stacks down the page one panel per row; a
California-Current box spreads across it. Blank cells are charged for, so an ordered
series doesn't end up scattered through a mostly empty grid just because those cells
happened to be the right shape.

The colorbar follows whatever grid results — horizontal beneath a wide, short grid;
vertical beside a tall one — since a bar on the grid's long edge is the one that stays
the same length as the panels it describes.

Pass an integer to override the layout entirely.

**Default:** `None` (chosen from the domain's aspect ratio)

```python
run.plot()            # orientation follows the domain
run.plot(ncols=3)     # three columns, whatever the domain
```

**With two facet axes there is nothing to choose.** If the reduction leaves both a
time axis and a depth axis standing (`select={"depth": [0, 50, 100]}` alongside a
monthly `aggregate`), the grid is `len(depth)` × `len(time)` — depths down the rows,
months across the columns — and `ncols` is refused rather than ignored, since a count
that disagrees with the data would drop panels instead of re-flowing them. Depth takes
the rows by convention (it reads top-to-bottom, surface first) rather than by whichever
arrangement fits the page; pass `row_dim` to `field_facet` directly to swap them.

Month titles then appear on the top row only, and each row is named down its left edge
(`50 m`) with the same rotated label `field_grid` uses — so `row_label_kwargs` applies
in this case, though `metrics_kwargs` still does not, there being no metrics without a
reference.

> **One facet axis: one shared colour scale and one colorbar**, and that is not
> configurable. The panels are the same quantity at different times, so per-panel
> scaling would draw a doubling between March and April as no change at all.
>
> **Two facet axes: one scale and one colorbar per depth row** — see
> [`shared_limits`](#shared_limits) to collapse them into one.

`ncols` has no counterpart in `facet_movie`: a movie has one panel, and the axis a grid
would arrange is the one it plays instead. A field with *two* facet axes therefore can't
be a movie — one of them would have to become panels, which is what `field_facet` is
for — and is refused rather than quietly animated along one axis.

### `shared_axis_labels`

Draws grid lines on every panel either way, but controls whether coordinate
**labels** repeat on every panel or only appear where they're not redundant: the
leftmost column (latitude) and the bottom row (longitude) — the usual convention
for a grid of maps, since three side-by-side copies of the same latitude labels say
nothing three copies didn't already say once. Set `False` to label every panel's
axes independently (`field_row`'s single row is always effectively "bottom", so
this only visibly changes anything for `field_grid`'s left column). In a `field_facet`
grid the last row can be ragged, so "bottom" means *has no panel below it* rather than
"is in the last row" — the seventh of seven panels and the sixth both get longitude
labels.

**Default:** `True`

```python
physics.plot(shared_axis_labels=False)   # every panel gets its own lat/lon labels
```

`section_row` takes the same parameter for the same reason, on its own single
non-geographic axis: with no lat/lon gridlines to draw, it decides whether the
depth label repeats on all three panels or only the leftmost, since the other
two share the same depth axis.

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

A third *grouping* channel (size, or filled vs hollow) is possible but deliberately
absent: three encodings on one point tend to be slower to decode than two diagrams
side by side. If you need a third dimension, split it across figures.
[`marker_scale`](#colors--alpha--marker_scale-summary-diagrams) below is not this — its
uniform form resizes every point by the same factor, a style knob rather than a way to
tell points apart.

Interactively, bokeh cannot show two independent legend blocks, so `color_by` +
`marker_by` produces combined entries (`"chl · runA"`) where the static diagram shows
a colour block and a marker block. Same groups, one legend instead of two.

### `groups` (summary diagrams)

`color_by`/`marker_by` split on a field the metric record already carries. `groups`
is for splitting on something that isn't a column at all — which region a mooring
sits in, which cruise a cast came from — supplied at plot time as a
`{reference_name: label}` mapping, without injecting it into every comparison's own
metrics first:

```python
regions = {"ciofs3-mooring-kbay01": "Kachemak Bay", "ciofs3-mooring-uci04": "Upper Inlet"}
suite.taylor(groups=regions)                       # colours by group
suite.taylor(groups=regions, marker_by="variable")  # ... shape still free for something else
```

Keyed by each comparison's `reference` (its catalog source name — what every metric
record already carries), falling back to the `label` for a hand-built record with no
`reference` column. When neither `color_by` nor `marker_by` is given, `groups` defaults
`color_by="group"`; an explicit `color_by`/`marker_by` is never overridden, so `groups`
can still supply the *other* channel. Since `groups` becomes `color_by` this way,
[`colors`/`alpha`/`marker_scale`](#colors--alpha--marker_scale-summary-diagrams) below
key on the group labels the same way they'd key on any other `color_by` field.

Honored by **both renderers** for `target` (`taylor`/`paired` delegate to matplotlib
interactively, as `labels` does above).

### `colors` / `alpha` / `marker_scale` (summary diagrams)

These three style what a point looks like — colour, transparency, and size — and all
three key on **the same field colour already groups by**: `color_by` if given, else
`marker_by`, else each comparison's own label. That shared field is what lets them
compose with grouping instead of fighting it, and what lets one call style particular
groups without touching the rest — fade one group, enlarge another, leave everything
else at its default:

```python
suite.target(color_by="variable",
             colors={"sea_water_temperature": "r"},
             alpha={"sea_water_salinity": 0.15},
             marker_scale={"sea_water_temperature": 1.5})
```

Each accepts a few shapes, in order of how much control they give:

| form | effect |
|---|---|
| not given | the usual default (colour cycle, opaque, unscaled) |
| a scalar (`alpha`/`marker_scale`) or single colour (`colors`) | broadcast to every point |
| a list | a palette assigned to the grouping levels, in the order they first appear |
| a `{level: value}` dict | styles only the named levels; every other level keeps its default |

A dict's keys are the field's **raw values** (`"sea_water_temperature"`, not the
legend's short spelling `"temp"`) — pretty names aren't guaranteed unique across
variables, so the raw value is the one key space that always resolves unambiguously. A
key that isn't one of the field's actual levels raises, naming the levels that *are*
available (with their pretty spelling alongside, so the fix is a paste); a level the
dict doesn't mention isn't an error — it just keeps its default, which is what makes
"style only these two groups" possible in the first place.

`colors` used to be silently discarded whenever `color_by` (or `groups`, which sets
`color_by` internally) was also given, and a single colour or a list shorter than the
number of points crashed with an opaque `zip()` error. Both are fixed: `colors` now
always composes with grouping or raises loudly, never dropped without a trace.

`marker_scale`'s reference star and the grey marker-block legend swatch (when colour
and marker are carrying two different fields) belong to no single group, so a dict
`marker_scale` leaves them at their default size rather than guessing which entry
applies — pass a scalar instead if you want the star to scale too. `alpha` never
touches the reference star, point labels, or the static legend's swatches, dict or not
— fading the point everything else is read against, or the text identifying a point,
defeats the point of drawing it. (Interactively the legend glyph is drawn from the
point itself, so it inherits that point's alpha — a small, unavoidable divergence from
the static legend.)

`marker_scale`'s uniform (non-dict) form is a style knob, not a third grouping
channel — see the note at the end of [`color_by` / `marker_by`](#color_by--marker_by)
above; it resizes points by the same factor rather than encoding anything in the size.

Honored by **both renderers** for `target` (`taylor`/`paired` are static-only, as
above).

### `overlay`, `summary_points` (summary diagrams)

A second, emphasized layer drawn on top of the base cloud — for two related reasons.
**Highlighting** a subset: `overlay=` takes the same kind of thing the main argument
does (a list, a `ComparisonSet`, hand-built records) and draws it bigger, opaque, and
black-edged, so it reads as "look here" against the fainter cloud beneath:

```python
suite.taylor(color_by="variable", overlay=[best_station, worst_station])
```

**Summarizing** a group: `summary_points=True` (or `"median"`/`"mean"`) instead builds
one ★-marked centroid per group internally — the reduced position of that group's own
cloud, median by default (robust to the odd outlier a mean isn't):

```python
suite.taylor(color_by="region", summary_points=True)  # one star per region
```

Both compose — an overlay can highlight specific points *and* summarize groups in one
call. Either way, the overlay point is styled to match **the colour its own group
already has in the base cloud** — never independently re-cycled, which matters because
an overlay is usually a small subset (one station, one centroid) whose own encounter
order rarely matches the base cloud's. Neither introduces a new legend entry: the
group already has one from the base cloud.

`overlay_marker_scale`/`overlay_alpha` size and fade the overlay layer specifically
(default `1.8`/`1.0` — bigger and fully opaque, independent of the base layer's own
`marker_scale`/`alpha`), and accept the same `{level: value}` dict form those do. A
centroid's marker is always `"*"` (a bokeh `"star"` interactively); a highlighted
point keeps its group's own marker.

**Default:** `overlay=None`, `overlay_marker_scale=1.8`, `overlay_alpha=1.0`,
`summary_points=False`

Honored by **both renderers** for `target` (`taylor`/`paired` are static-only, as
`marker_scale`/`alpha` are above).

### `circles` (target only)

**Default:** `None` (adaptive — see below)

Radii of the dashed/dotted guide rings, always in the axes' own units. Left at its
default, that is `(0.5, 1.0)` normalized; with `normalize=False` it becomes the same
fractions of the shared reference standard deviation — `(0.5·σ_ref, σ_ref)` — as long
as every comparison shares one. Either way the dashed ring marks where the model's
total RMSD equals that reference standard deviation — inside it, the model
out-performs the observed mean as a predictor. With `normalize=False` and comparisons
that do *not* share a reference standard deviation, there is no single boundary to draw,
so the default becomes no rings at all, with a warning; pass `circles` explicitly (still
in the axes' own units) to draw rings anyway:

```python
suite.target(circles=(0.5, 1.0, 1.5))
```

Honored by **both renderers**.

### `normalize` (Taylor and target)

**Default:** `True`

Divides each standard deviation, centred RMSD, and bias by its own comparison's
reference, so comparisons of different variables or in different units can share one
diagram, with every reference sitting at radius 1 (Taylor) or the origin (target):

```python
suite.taylor(normalize=False)   # native units; only sound with a shared reference
suite.target(normalize=False)   # same, for the target diagram
```

Turn it off only when every comparison shares one reference and the native units
matter more than a common scale. Axis labels then name the shared units when every
comparison agrees on one, and stay unitless otherwise. Two warnings guard against a
diagram that looks fine but isn't: comparisons that span more than one variable (the
axes may then mix units), and — for `taylor` specifically — comparisons whose reference
standard deviations differ (the star, dashed arc, and RMS contours describe only the
first comparison's reference; `target`'s analogous case is covered by `circles` above).
Honored by **both renderers** for `target`; `taylor` (and `paired`, which forwards to
both) delegates to matplotlib interactively, so this applies to both renderers either
way.

---

## The `series` family (time series)

A comparison whose two lanes reduce to one time axis draws as lines rather than maps:
`reference` solid, `test` dashed, on one panel with a statistics box and a key.

```python
osk.compare(reference="ooi-gp03flma-rim01-02-ctdmog040", test="oceansoda_ethz",
            variables=["temperature"]).plot()
```

Two *gridded* runs reach the same family the same way — a `select` that pins both
lon and lat to one position leaves nothing to draw a map of, exactly like a mooring's
`featureType` does, and `over="time"` is implied without being asked:

```python
osk.compare(reference="run_baseline", test="run_new", variables=["temperature"],
            select={"lon": -144.25, "lat": 50.0}).plot()
```

The reference's own grid decides the exact position (nearest cell, or interpolated
with `method="bilinear"`), and the test is sampled *there* — co-located, not each
lane's own nearest cell to the raw request.

### Which channel carries what

Deterministic, so a figure reads the same way every time and adding a source or a
variable does the expected thing:

| Channel | Default field | Notes |
|---|---|---|
| line style | **role** (not a field) | `reference` solid, `test` dashed — always |
| colour | `variable` | one colour per variable, shared by both lanes of a pair |
| dash pattern *within* the test side | `source` | a second model takes `:`, a third `-.` |
| marker | `depth` | drawn only when depth actually varies, on ~20 samples per line |

`role` beats everything: model-versus-model gives the baseline solid and the candidate
dashed, and swapping which is the `reference` swaps the two.

**One source, no comparison.** `osk.field(...).plot()` draws the same family when a
`select` narrows both horizontal axes to one position — one line, role `"value"`
since there is no reference to win against, drawn solid:

```python
osk.field("run_new", "temperature", select={"lon": -144.25, "lat": 50.0}).plot()
```

Colour still comes from `variable`; there is no statistics box (nothing to score
against) and `residual=True` is refused with a clear error rather than silently
drawing nothing, for the same reason.

Pass a list instead of one name and `osk.field` fans it into a `FieldSet` — one
single-source item per variable, pooled into this same figure and following the
Composition table below exactly as a multi-variable comparison set would:

```python
osk.field(
    "run_new", ["temperature", "salinity"],
    select={"lon": -144.25, "lat": 50.0},
).plot()                              # 2 variables: one panel, salinity on the
                                       # secondary axis
```

`encode=` moves a channel onto another field, or switches it off:

```python
.plot(encode={"linestyle": "depth"})   # dash pattern by depth instead of by source
.plot(encode={"marker": None})         # no markers, whatever depth does
```

### Composition

| Distinct variables | Default |
|---|---|
| one | one panel, every source overlaid |
| two | one panel, the second variable on a right-hand y axis |
| three or more | one row per variable |

`secondary_y=False` stacks the two-variable case instead. `rows=` or `cols=` (one, not
both) facet on `variable`, `source`, `reference`, `depth` or `comparison`.

On a two-variable panel, each y-axis label — and its tick numbers — take the colour of
the lines drawn against that axis, so the left and right scales stay identifiable
without reading the units. A single-axis panel keeps the default label colour; so does
any axis whose lines don't share one colour (`encode={"color": "source"}`, for example).

### `series`-only parameters

| Parameter | Default | Effect |
|---|---|---|
| `residual` | `False` | adds a short `test − reference` strip under each panel, sharing its time axis |
| `mark` | `"line"` | `"line+marker"`, `"marker"` or `"step"` |
| `metrics_loc` | `"auto"` | the corner the statistics box takes; `"auto"` picks the emptiest, and the key takes the next emptiest |
| `legend` | `True` | draw the key at all |
| `ylim` | `None` | y limits for every panel |
| `panel_aspect` | `2.6` | width/height of a panel; a line panel has no data aspect to read, unlike a map |

### `line_kwargs`

Anything [`Axes.plot`](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html)
takes, applied to every line — `linewidth`, `alpha`, `zorder`. Colour and line style come
from the policy above and are not overridable here (use `encode=`).

### `legend_kwargs`

Anything [`Axes.legend`](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html)
takes except `loc`, which the layout chooses so the key cannot land on the statistics box.

### Static versus interactive

Both renderers draw the same lines, colours, dash patterns, titles, axis labels, axis
label colours and key *entries*. Two differences, both deliberate:

* **Where the key goes.** When every panel shares one set of entries, the static renderer
  draws a single key below the figure; bokeh has no figure-level legend, so it always
  draws one inside each panel.
* **Where the statistics box goes.** Statically it is pinned to the panel's corner;
  interactively it is an `hv.Text` in data coordinates, so it pans and zooms with the data.

`line_kwargs` and `legend_kwargs` are matplotlib call signatures and only affect the
static renderer, which warns rather than absorbing them silently.

## The `profile` family (vertical profiles)

The vertical twin of `series`: a select that pins `lon`/`lat` to one place and
leaves depth standing, with no time axis surviving alongside it, draws value on x
and depth on y instead of value against time.

```python
osk.compare(reference="whots_temp", test="run_new", variables=["temperature"],
            select={"depth": [0, 10, 25, 50, 100], "time": "2013-06-15"},
            over="Z").plot()
```

**One source, no comparison.** `osk.field(...).plot()` draws the same family for a
single cast — one line, role `"value"`, drawn solid — the same reduction the
`series` family follows at one place through time:

```python
osk.field("run_new", "temperature",
          select={"lon": -144.25, "lat": 50.0, "time": "2013-06-15",
                  "depth": "column"}).plot()
```

### Which channel carries what

The same policy `series` uses, with `time` in place of `depth` as the marker
channel — depth is the axis itself here, not a fact to tell lines apart by, so a
profile's own `LineSpec.depth` is always `None`.

| Channel | Default field | Notes |
|---|---|---|
| line style | **role** (not a field) | `reference` solid, `test` dashed — always |
| colour | `variable` | one colour per variable, shared by both lanes of a pair |
| dash pattern *within* the test side | `source` | a second model takes `:`, a third `-.` |
| marker | `time` | drawn only when a cast's own instant actually varies |

`encode={"marker": "depth"}` (or any channel keyed on `"depth"`, or `rows="depth"`/
`cols="depth"`) is refused with a clear message: depth is the axis every panel
already draws against.

### Composition

| Distinct variables | Default |
|---|---|
| one | one panel, every source/cast overlaid |
| two or more | one column per variable, sharing the depth axis |

No `secondary_y` here — a second value axis at the *top* of a profile panel has no
interactive twin (bokeh has `multi_y`, not `multi_x`), so two or more variables
each take their own column instead. `rows=`/`cols=` (one, not both) facet on
`variable`, `source`, `reference`, `time` or `comparison`.

### Axis conventions

* **y is depth, positive down, inverted** — 0 m draws at the top, the deepest
  sample at the bottom — the same convention `section` uses, and every other
  depth label in this package (`facet_labels`' own `abs()`). A `sigma0` profile
  (an isopycnal comparison) inverts the same way: denser water at the bottom.
* **x is the compared value**, labelled the same way `series`' own y-axis is
  (`"temperature [degC]"`).
* **`ylim=(shallow, deep)`** is always read in positive-down metres — `(0, 200)`,
  not axis order — since the axis is inverted regardless of what is passed.

### `profile`-only parameters

| Parameter | Default | Effect |
|---|---|---|
| `mark` | `"line"` | `"line+marker"` or `"marker"` — no `"step"` (a profile's levels are irregularly spaced, with nothing between them a step-hold represents honestly) |
| `metrics_loc` | `"auto"` | the corner the statistics box takes; `"auto"` picks the emptiest, and the key takes the next emptiest |
| `legend` | `True` | draw the key at all |
| `xlim` | `None` | value-axis limits, every panel |
| `ylim` | `None` | depth limits, `(shallow, deep)` in positive-down metres |
| `panel_aspect` | `0.62` | width/height of a panel — portrait, since a water column reads top-to-bottom |

There is no `residual` option yet (a `test − reference` strip beside each depth
panel is a follow-up) and no `profile_movie` (several casts over time played as
frames, mirroring `facet_movie`) — both are named follow-ups, not gaps in the
family's own logic.

### Static versus interactive

Both renderers draw the same lines, colours, dash patterns, titles, axis labels
and key entries, through the same `ocean_skill.plot.profile.compose` layout —
exactly the guarantee `series` makes. The depth axis reads surface-at-top in both
by construction: statically via `ax.set_ylim(deep, shallow)`, interactively via
`ylim=(deep, shallow)` on the bokeh figure (deliberately *not* also
`invert_yaxis=True`, which would flip an already-descending range back to
ascending). `line_kwargs` and `legend_kwargs` only affect the static renderer,
which warns rather than absorbing them silently.

## The `section` family (vertical slices)

A **grid-aligned** transect — `select={"transect": {"<dim>": <index>}}` — draws as
one panel: depth (or the model's own s-levels) against along-path distance, rather
than a map. It is a plain `isel` along a named grid dimension, so it is exact and
needs no interpolation:

```python
osk.field("run_new", "temperature",
          select={"transect": {"xi_rho": 30}, "time": "2012-06"},
          aggregate={"time": "mean"}).plot()
```

With no `depth` key, the section shows the model's own s-levels — the cheapest and
most exact reading, since nothing is interpolated. Give `depth` a list of fixed
levels to interpolate onto instead (the same `roms.to_depth` a map row uses):

```python
osk.field("run_new", "temperature",
          select={"transect": {"eta_rho": 12}, "depth": [0, 50, 100, 200]}).plot()
```

**Arbitrary paths** — waypoints, or a fixed lon/lat line — sample the grid instead
of indexing it, by nearest-neighbour (default) or bilinear interpolation:

```python
osk.field("run_new", "temperature",
          select={"transect": {"waypoints": [[-96.0, 24.0], [-94.0, 26.0], [-92.0, 28.0]]},
                  "depth": [0, 50, 100, 200]},
          aggregate={"time": "mean"}).plot()

osk.field("run_new", "temperature", select={"transect": {"lon": -94.5}}).plot()
osk.field("run_new", "temperature",
          select={"transect": {"lat": 25.0, "lon": {"min": -96, "max": -93}}}).plot()
```

Waypoints/lines are densified to roughly the model's own grid resolution before
sampling (`spacing_km` overrides); `select={"transect": {"points": [...]}}` samples
exactly the given points instead, with no densification. A path that partly leaves
the domain is trimmed to it, with one warning naming how many points were dropped —
the same trim either sampling method gives, so switching `method="nearest"` to
`"bilinear"` never silently changes how much of the path is covered. Antimeridian
crossings (the `pac_dt_ramp`-style 77°E–316°E domain) are handled the same way
every other longitude-aware function in this package is.

For an interactive alternative to typing waypoints by hand, `osk.pick_path(source)`
opens a click-to-add-waypoints map in a live notebook — see its own docstring.

Matching a section against a gridded dataset (`osk.compare`, not just `osk.field`)
draws through the `section_row` family instead — see below.

### Axis conventions

* **y is depth, positive down, inverted** — 0 m draws at the top, the seafloor at
  the bottom — matching every other depth label in this package (`facet_labels`'
  own `abs()`).
* **x is along-path distance in kilometres** (great-circle, from the sliced grid's
  own lon/lat), labelled `distance along transect (km)`.
* **Below-bathymetry (or off-domain) cells** carry no data and draw as the same
  grey a map's land does — the seafloor's shape is visible without singling those
  cells out.
* **Native s-levels** draw with a true, per-column depth (`z_rho`, curved with the
  bathymetry) rather than a flat sigma-level index; a fixed-depth list draws flat
  rows, one per requested depth.

Both conventions are decided once, in `ocean_skill.plot.section.prepare_section`,
and read by both renderers — a section cannot look different statically than
interactively.

### Static versus interactive

Both draw the same mesh, colour scale and axis conventions. A section has no map to
outline or gridline, so `domain`, `gridline_kwargs` and `tick_label_kwargs` are not
options of `section()` — the static renderer raises naming the reason, the
interactive one warns and drops. `mark="pcolormesh"` (default) or `"contourf"`,
the same two the map families accept; bokeh always draws a mesh regardless (`mark`
has no interactive effect, same as every map family).

## The `section_row` family (a section matched against a dataset)

A comparison whose `select` cuts a transect draws `test | reference | difference`
sections instead of maps — `field_row` with `section`'s depth-by-distance axes
substituted for the map:

```python
osk.compare(
    test="run_new", reference="woa23_nitrate", variables=["nitrate"],
    select={"transect": {"waypoints": [[147.0, 35.0], [175.0, 45.0], [-160.0, 50.0]]},
            "depth": [0, 50, 100, 200, 400, 700, 1000],
            "time": "2013"},
    aggregate={"time": "mean"},
).plot(renderer="both")
```

Every transect form `section` accepts works here too — grid index, waypoints, a
fixed lon/lat line, resolved `points`, or `osk.pick_path`'s output — and the
reference can be another model run just as well as a climatology.

**Both lanes are reduced to columns at the same lon/lat positions along the
path — never a 2-D regrid.** The test lane is prepared first; the reference is
then sampled at exactly those same points, so comparing the two is pairing
columns, not interpolating one grid onto another.

**The pair lands on the coarser lane's along-path resolution** — the same
house rule a map comparison's own regridding follows: whichever lane has fewer
columns along the path keeps its own, and the finer lane's columns are
mean-binned into them (nearest-column grouping, then a per-level mean).
Comparable resolutions pair one-to-one instead, with no averaging — the
identity for two lanes on the same grid. Which lane moved, if either did, is
recorded in the aligned pair's attrs (`section_target`).

A comparison section needs an explicit `select={"depth": [...]}` list — two
lanes' native s-levels or observational levels share no axis to compare on, so
nothing is guessed. A scalar depth, a depth band, a one-element list or a
vertical aggregate are all refused for the same reason, each with a suggestion.
`over=`, `sigma0` and a pair-spec `select` are refused too — each names its own
follow-up.

### Axis conventions and layout

Exactly `section`'s: positive-down depth, y-axis inverted, along-path distance
in kilometres, below-bathymetry/off-domain cells grey. Test and reference share
one colour scale (10th–90th percentile of the pair); the difference panel is
diverging and centred on zero; metrics go in the difference panel's corner box
(statically) or fold into its title (interactively) — precisely as `field_row`
draws a gridded comparison, just against depth and along-path distance rather
than longitude and latitude. The title carries the depth list, time and the
path's own endpoints (`29.0°N, 94.5°W → 27.5°N, 90.0°W`) in place of a region.

There is no `domain`, `region`, `gridline_kwargs`, `tick_label_kwargs` or
`row_label` — a section has no map to outline, and it is always the only (and
so also the bottom) row. `metrics(weighted=False)` — cos-lat area weights mean
nothing for section cells — and `pointwise_metrics()` is refused (there is no
further axis to score over).

A `section_row` is never stacked into a grid: more than one in a
`ComparisonSet.plot()` is refused (`section_grid` is a follow-up), and
`ComparisonSet.movie()` refuses a set containing one (time-animated sections
are a follow-up too).

## The `locations` family (dataset map, and selection map)

`osk.map_locations()` / `osk.find(...).map()` draw where catalog datasets *are*, from
metadata alone — no field, so no colormap, no colorbar and no `mark`. Colour keys the
`featureType` (markers for stations/profiles/tracks, dashed extent boxes for grids),
and the legend is the key to it. The same family also draws where a plotted
*selection* sits — see [Selection maps](#selection-maps) below.

### `locations`-only parameters

| Parameter | Default | Effect |
|---|---|---|
| `extent` | frames every item | `(lon_min, lat_min, lon_max, lat_max)`, the same bbox shape `find(bbox=...)` takes |
| `legend` | `True` | draw the featureType key at all |
| `marker_size` | `80` static / `9` interactive | station marker size (matplotlib points² / bokeh pixels) |
| `tiles` | `"CartoLight"` (interactive only) | any `geoviews.tile_sources` name; `None` for the offline coastline basemap. The static renderer accepts-and-warns so `renderer="both"` can share one set of options |

Of the styling dicts, `title_kwargs`, `gridline_kwargs`, `tick_label_kwargs` and
`legend_kwargs` apply (static renderer only, as ever); the rest describe things this
family does not draw. `size`/`zoom`/`figsize`/`font_scale`/`save` work as everywhere
else.

Interactively, hovering any marker or extent box reads that dataset's record — name,
catalog, featureType, variables, time coverage, cadence, resolution, depth,
institution, title — pre-formatted by `ocean_skill.plot.locations.build_items`, which
is also where entries with no declared geospatial extent are skipped with a warning.

### Selection maps

`comparison.map_locations()`, `comparison_set.map_locations()`, `field.map_locations()`
and `field_set.map_locations()` draw where *that object's own data* sits — the same
`locations` family, fed by `ocean_skill.plot.map_locations` instead of catalog names:

```python
comparisons = osk.compare(reference="papa", test="ciofs3", variables="temperature",
                           select={"lon": -144.3, "lat": 50.0})
comparisons.map_locations()                     # a crimson star at the requested
                                                 # point, over ciofs3's dashed
                                                 # domain outline
comparisons.map_locations(renderer="holoviews") # the same, interactive
```

Every lane draws once: its **requested** selection (a point, a region box, or a
lone-lon/lat slice as a solid line) when `select` pins one, else its own **catalog
footprint** — the same marker/box `osk.map_locations("that_source")` would draw —
so a comparison with no horizontal select still places both the test and the
reference. A `ComparisonSet`/`FieldSet` fan that shares one point or region (ten
variables at one mooring, say) draws it once, not once per member.

This never opens a dataset and never aligns a comparison — it reads only the
*request* (`select`) and catalog metadata, so it costs the same whether `.plot()`
has already run or not. It is therefore also never precise about a snapped-vs-
requested offset (the few kilometres between where you asked and the nearest model
cell): that offset is already a warning where alignment actually happens, and stays
a warning there rather than becoming a second marker or an annotation here.

Selection geometry always draws **crimson** (a colour outside the catalog
featureType palette, so it can never collide with one); the domain outline always
draws **black, dashed** — the same style the `domain=` option draws everywhere
else, so every map in this package agrees on what a model's footprint looks like.
The domain ring is on by default (one per distinct test source); pass `domain=None`
to suppress it, or a `(lon_min, lat_min, lon_max, lat_max)` bbox / `(N, 2)` ring to
override it — the same spelling `.plot()`'s own `domain=` takes. The ring/line paths
carry no hover record in the interactive renderer (there is nothing per-glyph on a
path to report); the selection point/box still hovers with the same fields a
catalog item does, `title` holding a plain-language description ("point at (-144.30,
50.07)", "meridional slice at -150.00°", ...).

All the `locations`-only parameters above apply here too.
