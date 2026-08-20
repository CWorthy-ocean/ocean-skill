# Movies

**A movie is the plot you already have, played rather than laid out.** Both static plot
families that draw a series of maps have an animated twin, and `.movie()` is `.plot()`
with the series moved from the page into time. A frame is drawn by the same code as the
still it would have been, so the two cannot drift apart.

| | laid out | played |
|---|---|---|
| one model field, no reference | `Field.plot()` (`field_facet`) | `Field.movie()` (`facet_movie`) |
| a comparison | `ComparisonSet.plot()` (`field_grid`) | `ComparisonSet.movie()` (`field_movie`) |

## A model field over time

The common case: one run, one variable, no reference. The axis the reduction leaves
standing — which `Field.plot()` turns into panels — becomes the frames.

```python
run = osk.field("GOM_bgc", "salinity",
                select={"time": "2012-01", "depth": "surface"})

run.plot()                        # every step as a panel
run.movie(save="salt.mp4")        # every step as a frame
run.movie(renderer="holoviews")   # every step on a slider
```

Nothing is reduced unless you ask, so a month selected is a month of frames. `aggregate=`
chooses a coarser cadence when the raw one is too fine to watch:

| `aggregate=` | frames |
|---|---|
| omitted (or `{}`) | every step in the selection |
| `{"time": {"resample": "1D", "reduce": "mean"}}` | one per day — thins hourly output |
| `{"time": {"resample": "1MS", "reduce": "mean"}}` | one per month |
| `{"time": {"groupby": "month", "reduce": "mean"}}` | twelve, a climatology |
| `{"time": "mean"}` | **one — a single map, not a movie** |

A reduction and `every=` are not the same thinning: `{"resample": "1D"}` *averages* each
day into a frame, `every=24` keeps every 24th hour and discards the rest. Both give one
frame per day from hourly output; only the first shows you a daily mean.

Which reading is better depends on how many steps there are, and they trade off in
opposite directions: a handful of monthly means are best seen at once, side by side,
where a month of daily output is 31 panels too small to read and 31 frames a drag apart.

Frame labels come from the facet coordinate, spelled as the static panel titles are —
`Jan 2012` for consecutive months, `Jan` for a climatology, `50 m` for a level — including
where that wouldn't tell one frame from another. A movie is as often over the *unreduced*
axis, where every step of January is one month; there the label refines itself to
`2012-01-05`, or to the minute if that is what separates two frames. Panels refine the
same way, for the same reason: a title that fits every panel names none of them. It
matters most on a slider, though, where the labels *are* the frames' keys and duplicates
would collapse frames on top of each other silently.

A movie plays **one** axis. If the reduction leaves two standing (`select={"depth": [0,
50, 100]}` beside a monthly `aggregate`) there is no single sequence to play, and the
movie is refused rather than animated along one axis with the other quietly averaged —
that is what `Field.plot()`'s depth-rows grid is for. A field collapsed all the way to a
single map has nothing to play either, and says so instead of writing a one-frame movie.

## A comparison over frames

`ComparisonSet.movie()` animates the same items `.plot()` stacks down the page — one
`test | reference | difference` row, redrawn per frame, with each comparison's label as
the frame label and its own metrics in the corner box.

```python
runs = osk.compare(reference="run_baseline", test="run_new", variables=[NITRATE],
                   depths=("surface", 50, 100))

runs.plot()                          # three rows, stacked
runs.movie(save="depths.mp4")        # the same three, played
runs.movie(renderer="holoviews")     # the same three, on a slider
```

There is currently no way to fan a comparison out over *time* — `compare()` fans over
variables and depths only — so a comparison movie's frames are whatever varies across the
set you built. A model-vs-data movie through time needs `compare(times=...)`, which is
designed but not built.

A comparison movie keeps the ~260px panels of the row it animates, since it draws three
of them side by side; only the single-field movie takes the whole width.

## Formats

The extension picks the writer. Nothing else selects a format — you have already said
which you want by naming the file.

| Extension | Writer | Needs |
|---|---|---|
| `.mp4`, `.m4v`, `.mov` | matplotlib's `FFMpegWriter` | **ffmpeg** on `PATH` |
| `.gif` | matplotlib's `PillowWriter` | nothing beyond matplotlib |
| `.html` (with `renderer="holoviews"`) | bokeh | nothing beyond holoviews |

ffmpeg is the only external binary in the stack, so it is the only thing here that can
be missing. It ships in `environment.yml`; if it is absent the error says so and names
the two ways out:

```bash
conda install -c conda-forge ffmpeg
```

A `.gif` is the fallback that always works. It is larger than an mp4 for the same
frames and has no seek bar, which is a fair trade for needing nothing installed.

Without `save=`, nothing is written and the animation is only returned — in a notebook,
`HTML(ani.to_jshtml())` plays it inline. Keep a reference to it either way: a
matplotlib animation whose last reference is dropped stops, and says so.

## One colour scale for the whole movie

This is the setting that decides whether an animation is readable, so it is on by
default. `shared_limits=True` takes the colour limits from **every** frame at once;
a value is then the same colour throughout, and what you see change is the field.

Re-deriving the scale per frame — the obvious implementation — makes the ruler move with
the field, and the eye cannot separate the two. Every frame looks correct in isolation,
which is what makes it a bad default rather than an obvious bug.

`shared_limits=False` takes the scale from the first frame instead. Either way the scale
is **fixed for the whole movie**; the option only chooses which frames it is derived
from. Reach for `False` when the first frame is representative and a later outlier would
otherwise flatten everything else.

The interactive renderer fixes the same limits, to the same numbers — dragging the
slider must not move the ruler either. `tests/test_movie.py` asserts the two renderers
agree on them.

## What changes per frame, and what does not

The figure, its layout, its colorbars, its axis labelling and its left margin are built
**once**, from the first frame. Only three things are redrawn:

- the field values, in the one panel or in all three
- the **frame label** — the timestamp, in the panel's top-left corner
- the **metrics box** — that frame's own bias/rmse/corr, in the bottom-left of the
  difference panel (comparison movies only; a model field has nothing to score against)

So nothing shifts, resizes or re-fits as the movie plays. That matters more here than on
a still: the layout machinery (`_align_colorbars`, `_fit_left_margin`, `_fit_text_widths`)
measures the drawn figure and adjusts, and running it per frame would make the colorbars
and margins twitch frame to frame.

The frame label is drawn in **monospace** on purpose. A proportional font makes a
counting timestamp jitter sideways as its digits change width — invisible on a still and
distracting in a movie. Style it with `frame_label_kwargs`, or drop it with
`frame_label=False`; see
[the styling reference](plot_styling_reference.md#frame_label_kwargs).

`mark="pcolormesh"` (the default) is the mark to animate: its values live in an array
that can simply be swapped. `mark="contourf"` works, but a filled contour set is geometry
rather than an image and has to be removed and redrawn every frame.

## Length, and thinning

Every frame is a full cartopy redraw, so frames are the cost. A movie longer than 200
frames says so before spending the time, and suggests a stride:

```python
run.movie(save="year.mp4", every=24)      # hourly output -> daily frames
```

`every=N` keeps every Nth frame. It thins what is *shown*, not what was computed, so it
costs nothing and needs no re-preparation — unlike narrowing `select=` or collapsing
`aggregate=`, which change the data and are the right tools when you want a daily *mean*
rather than every 24th hour.

The warning is not a cap. A year of hourly output really is 8760 frames, and refusing to
draw what was asked for would be worse than taking a while over it.

## Static and interactive

The two renderers differ more in form here than anywhere else, and less in intent.

| | `renderer="matplotlib"` | `renderer="holoviews"` |
|---|---|---|
| Output | mp4 / gif file | slider over the frames |
| `save=` | the video | a standalone `.html` page |
| `fps=` | encoded frame rate | the play speed, with `widget="player"` |
| Frame label | drawn in the panel | the slider's value, and the panel title |
| Metrics | corner box | folded into the difference panel's title |

**`widget=`** picks the control (interactive only):

| | |
|---|---|
| `"slider"` *(default)* | a `DiscreteSlider` — drag or arrow-key through the frames, labelled with the frame's own name |
| `"player"` | play / pause / step, running at `fps` |
| `"dropdown"` | holoviews' own default control, and the bare holoviews object rather than a panel pane |

The default is a slider rather than holoviews' own choice because holoviews picks a
*dropdown* for a string-valued dimension, and a dropdown is the wrong control for an
ordered sequence: the next frame is two clicks and a search, and you cannot drag through
the movie at all.

### Why the interactive movie works anywhere

**Every frame is drawn and embedded up front.** This used to be lazy: a `DynamicMap`
drew only the frame on screen, and asked a live Python kernel to draw the next one the
moment the slider moved. That channel has to be bound perfectly to work at all, and in
practice often isn't — `pn.extension()` running mid-cell, a notebook exported or
reopened with no kernel behind it, an editor with its own comm handling. When the
channel wasn't there, the slider moved and the plot didn't, with no error to say why.
Drawing every frame into a `HoloMap` and baking it into the notebook cell's own output
(the same machinery `save=` has always used) turns stepping through them into plain
client-side JavaScript: no comm, no kernel, nothing to lose. It works in a running
notebook exactly as it does in one you've exported or reopened cold.

**`hover=False` by default.** A hover readout makes bokeh hit-test every quad. That is
worth paying on one map you are reading values off, and pure cost on a movie you are
watching. `hover=True` brings it back.

**`rasterize="auto"`.** Past ~100k cells a frame, datashader renders the mesh to an image
instead of shipping every quad — which is also what makes embedding every frame
affordable rather than a hang. On a 400×550 grid that is **210 MB → 4.4 MB** and
**31s → 1s** to save:

| | raw mesh | rasterized |
|---|---|---|
| payload per frame | 59.7 MB | 1.7 MB |
| 5-frame HTML | 210.4 MB | 4.4 MB |
| time to save | 31.5s | 1.0s |

Below the threshold the raw mesh is kept, because it stays sharp when you zoom in.
Rasterizing is applied eagerly (a `HoloMap`'s frames have to be concrete elements —
nothing downstream can evaluate a lazy aggregation once a frame is embedded), so zooming
magnifies the image rather than re-aggregating — pass `rasterize=False` for a field
small enough to explore that way, or `True` to force it.

What remains is honest: display time and page size both scale with frame count, since
every frame is drawn either way. `every=` thins a long movie, and the mp4 is the artifact
for one long enough that thinning isn't enough.

### Looks

`tiles=True` (the default) puts a basemap under the field — a notebook watching a movie
is already on the web, so there's nothing offline about fetching a few map tiles too.
Pass a source name — `tiles="EsriTerrain"`, `"CartoLight"`, or any [geoviews tile
source](https://geoviews.org/user_guide/Working_with_Bokeh.html) — for a different map,
or `tiles=False` for a notebook that genuinely has to work offline. The view opens framed
on the field's own domain, with the basemap filling in around it and under anywhere the
field is masked.

The panel title says **what** as well as **when** — `GOM_bgc: alkalinity, surface —
2010-01-29`. The variable comes from the CF standard name via `vars.short_name`, the
depth from the `select` that produced the field, the source from its label; any part that
isn't known is left out, and `title=` overrides the lot while keeping the frame stamp.

A single-field movie also gets the **whole page width** (`SOLO_PANEL_WIDTH_PX`, ~680px)
rather than the ~260px a panel gets in a row of three. It is the only panel on the page,
so inheriting the row's width made the one thing on it the smallest thing on it. Its axis
titles are shortened to `longitude`/`latitude` too — a ROMS coordinate's own `long_name`
is "longitude of rho-points (degrees East)", which bokeh truncates anyway. Override
either with `width_px=` / `axis_labels=`.

Stepping is the interactive form of playing: you can hold a frame, step back one, and
hover a cell for its value — none of which an mp4 can do. When it should just run,
`widget="player"` swaps the slider for a play/pause scrubber, driven at the same `fps`.

```python
runs.movie(renderer="holoviews")                          # slider
runs.movie(renderer="holoviews", widget="player", fps=12) # play/pause
runs.movie(renderer="holoviews", save="depths.html")      # a page to send someone
```

**A saved `.html` embeds every frame's data**, so its size grows with the frame count —
four frames of a modest domain is already tens of megabytes, on top of bokeh's own
bundle. That is the price of a page that pans, zooms and hovers with no server behind it.
Displaying the movie in a notebook pays the same cost, for the same reason (see [Why the
interactive movie works anywhere](#why-the-interactive-movie-works-anywhere)); for
anything long the mp4 is the artifact to send instead: same frames, orders of magnitude
smaller, at the cost of interactivity.

The seven matplotlib-only `*_kwargs` dicts — `frame_label_kwargs` among them — mean
nothing interactively and warn once if passed, exactly as they do for any other family.
`title`, `metric_keys`, `font_scale`, `fps`, `every` and `shared_limits` are honored by
both.

## Frame order

Frames play in the set's own order, which for a `compare()` fan-out is the order the fan
produced. That is what you want when **one** thing varies across the set — a depth, a
time, a run — and meaningless when several do: a movie whose frames step through both
variable and depth is a movie of neither.

Interactively this has to be pinned rather than assumed, because a holoviews `HoloMap`
sorts its keys: without it, `2012-01-10` would sort before `2012-01-02`, and a movie over
depths would be reordered alphabetically. The frame dimension carries the author's order
explicitly, and `tests/test_movie.py` guards it.
