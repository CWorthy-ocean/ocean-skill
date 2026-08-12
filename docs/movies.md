# Movies

A movie is the plot you already have, over frames. `ComparisonSet.movie()` animates the
same items `.plot()` stacks down the page — one `test | reference | difference` row,
redrawn per frame — so a frame of the movie *is* the figure `field_row` draws, because
it is drawn by the same code.

```python
runs = osk.compare(reference="run_baseline", test="run_new", variables=[NITRATE],
                   depths=("surface", 50, 100))

runs.plot()                          # three rows, stacked
runs.movie(save="depths.mp4")        # the same three, played
runs.movie(renderer="holoviews")     # the same three, on a slider
```

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

- the field values in each of the three panels
- the **frame label** — the timestamp, in the top-left of the test panel
- the **metrics box** — that frame's own bias/rmse/corr, in the bottom-left of the
  difference panel

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
runs.movie(save="year.mp4", every=24)     # hourly output -> daily frames
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
| `fps=` | encoded frame rate | the play speed, with `player=True` |
| Frame label | drawn in the panel | the slider's value, and the panel title |
| Metrics | corner box | folded into the difference panel's title |

Stepping is the interactive form of playing: you can hold a frame, step back one, and
hover a cell for its value — none of which an mp4 can do. When it should just run,
`player=True` adds a play/pause scrubber, driven at the same `fps`.

```python
runs.movie(renderer="holoviews")                          # slider
runs.movie(renderer="holoviews", player=True, fps=12)     # play/pause
runs.movie(renderer="holoviews", save="depths.html")      # a page to send someone
```

**A saved `.html` embeds every frame's data**, so its size grows with the frame count —
four frames of a modest domain is already tens of megabytes, on top of bokeh's own
bundle. That is the price of a page that pans, zooms and hovers with no server behind it.
Displaying the movie in a notebook doesn't pay it (frames render on demand), and for
anything long the mp4 is the artifact to send: same frames, orders of magnitude smaller,
at the cost of interactivity.

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
