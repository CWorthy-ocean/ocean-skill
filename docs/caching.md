# Caching aligned results

Every comparison's expensive step is `Comparison.align()`: it opens both sources
(often remote OPeNDAP), reduces each to one 2-D field (time mean, and for ROMS an
xgcm s-coord → z transform), then regrids onto the coarser lane's grid with xesmf. The result
is one small Dataset — `test`, `reference`, `difference`, `coverage`.

That result is **cached to disk by default** and reused on a later run with the same
arguments, so redrawing a figure with a bigger font, or picking up after a notebook
kernel restart, is a read rather than a recompute.

```python
import ocean_skill as osk

physics = osk.compare(reference=[...], test="GOM_bgc", variables=[...], depths=["100"])
physics.plot()      # first run: reads, regrids, caches

# ... restart the kernel, change plot styling, come back tomorrow ...

physics = osk.compare(reference=[...], test="GOM_bgc", variables=[...], depths=["100"])
physics.plot()      # same arguments -> served from cache, no reading or regridding
```

The first time the cache is touched in a process it prints where it lives and how to
turn it off, so it is never silently working behind your back:

```
ocean-skill: caching aligned results in /Users/you/Library/Caches/ocean-skill/cache/aligned
  (reused automatically on repeat; osk.cache.disable() to turn off, osk.cache.clear() to empty.
   Keyed on source definition/variable/selection, NOT file contents — clear it after rerunning a model in place.)
```

> **In-session repeats were already free.** `Comparison.aligned` memoizes in memory,
> so calling `.plot()` twice on the same object never recomputed anything. What the
> disk cache adds is reuse *across processes* — a restarted kernel, a rerun script, a
> second notebook.

## Where in the pipeline to cache, and why here

The pipeline runs the reference and test lanes independently, joins them at `align`,
then measures and draws:

```
read → resolve variable → time mean → vertical interp (xgcm) → unit convert   ← per lane
                              ↓ (reference lane)      ↓ (test lane)
             harmonize longitude → subset to overlap → regrid (xesmf) → difference
                                          ↓
                              metrics (xskillscore)   →   render (matplotlib / bokeh)
```

| Stage | Cost | Cached? | Why |
|---|---|---|---|
| Open source | seconds (remote OPeNDAP metadata) | ✔ | lazy; the real cost is what follows |
| Per-lane prepare — time mean, **vertical interpolation**, unit convert | **minutes** | ✔ **`prepared/`**, one file per lane | the xgcm s-coord → z transform is the single most expensive step |
| Align — lon harmonize, subset, **regrid**, difference | **seconds–minutes** | ✔ **`aligned/`**, one file per pair | last point where model and obs are still one reusable object |
| Metrics | milliseconds | ✘ — *written*, not cached | see [outputs](#outputs-are-not-cache) |
| Render | seconds | ✘ — *written*, not cached | see [outputs](#outputs-are-not-cache) |

There are two cache layers, because they answer different questions.

**`aligned/` — one file per pair**, keyed `(test, reference, variable, select,
method)`, each source standing for its name *and* its
[catalog definition](#the-one-thing-to-know-the-key-is-identity-not-content). This is
the fast path, and the right outer boundary because it is exactly
*"everything needed to remake the plot, for both model and data, with no source
access"*: all reading, depth interpolation, unit conversion, regridding and
differencing are behind it, and nothing downstream touches the catalog again. It is
also small — four 2-D fields — so it is instant to reload, unlike the multi-GB
sources it came from.

**`prepared/` — one file per lane**, keyed `(source, variable, select)` (the source
again meaning its name and definition) with *no* reference and *no* regrid method in
it. This one makes a *miss* cheap. A lane's own work depends only on that source, so
comparing one model against several references should prepare it once:

```python
osk.compare(reference=["woa23_nitrate", "glodap"], test="GOM_bgc",
            variables=[NITRATE], depths=["100"])
# GOM_bgc's lane — read + time mean + xgcm depth transform — runs ONCE,
# not once per pair; the two references are prepared once each.
# 3 prepared entries, 2 aligned entries.
```

Without it that model lane ran twice, identically, because each pair hashes to a
different aligned key. It also pays off across runs that change only the regrid
method, or that add a reference to an existing comparison.

### Outputs are not cache

Metrics and figures are **deliverables**, not regenerable intermediates. They go to a
visible, project-scoped tree ([`ocean_skill.outputs`](../ocean_skill/outputs.py)), not
into the cache:

```
~/Library/Caches/ocean-skill/cache/   # regenerable, hash-named, safe to delete
        prepared/<hash>.zarr          #   (and reclaimable by the OS)
        aligned/<hash>.zarr

./output/<project>/                   # deliverables, named by you, meant to be kept
        figures/<stem>.png
        metrics/<stem>.csv  (+ .txt)
```

Keeping them apart is not just tidiness: the cache default is the OS cache directory,
which the operating system may reclaim whenever it likes. A figure you meant to keep
must not live somewhere it can be swept away.

```python
physics = osk.compare(reference=[...], test="GOM_bgc", variables=[...])
physics.save("gom_nutrients")        # -> output/gom_nutrients/{figures,metrics}/
# writes both, prints where, and returns {"figure": Path, "metrics": Path}
```

`save()` forwards any extra keyword arguments to the renderer, so everything in
[plot_styling_reference.md](plot_styling_reference.md) works there too. Base
directory: `osk.outputs.set_base(...)` → `$OCEAN_SKILL_OUTPUT` → `./output`.
`osk.outputs.info()` says what has been written.

Neither is worth *caching*, for its own reason:

- **Metrics** cost milliseconds once the aligned pair exists (xskillscore over four
  small 2-D fields), and are already memoized per `Comparison`. Caching them would
  save nothing measurable while adding a second thing that can go stale.
- **Figures** would have to key on the aligned data *plus* every styling knob
  (the seven `*_kwargs` dicts, `metric_keys`, `shared_limits`, renderer, figsize…).
  That key is both fragile and self-defeating: it changes on every edit, and
  iterating on styling is precisely when you would want the hit. Rendering is
  seconds against a cached alignment, so there is little left to win.

## The one thing to know: the key is identity, not content

An entry is keyed on a hash of **the two source names, a fingerprint of each source's
catalog definition, the variable, the selection, and the regrid method** — deliberately
*not* on the data itself, since hashing the data would mean reading the very files the
cache exists to avoid.

The fingerprint (`osk.catalog.fingerprint(name)`) is read from the catalog *file*, never
from the data: the entry's reader as written — its class and arguments, the URLs and
paths in them, any chained transforms (`.tail()`, `.sel(...)`, …), the data it points
at — plus the metadata that changes what a read returns (`time_zone`, `standard_names`,
`axes`, `depth_convention`, …). Prose (`description`, `title`, `tags`) and what the build
probe derived from the data (extents, time coverage, the variable list) are left out, so
re-describing an entry does not throw its results away.

So **redefining an entry under the same name misses cleanly**: rewrite a catalog so
`cast0000` points at a different CSV, or `model_win` at a different time window, and the
result cached for the old definition is not served. Two things follow:

- **A cache copied to another machine hits only where the catalog entries are
  identical** — paths included. The same catalog over the same paths shares every hit;
  data at different paths misses and is recomputed. Slower, never wrong.
- **Rewriting the same file in place is still invisible.** If you **rerun a model and
  write new output to the same catalog path**, the definition has not changed, so the
  cache cannot tell, and will serve you the old result.

After rerunning a model in place:

```bash
python -c "import ocean_skill as osk; print(osk.cache.clear(), 'entries removed')"
```

or, for a single call, `refresh=True` (recompute and overwrite) or `cache=False`
(bypass disk entirely). Changing the variable, depth, sources, or regrid method — or
redefining a catalog entry — all change the key on their own; those need no special
handling.

### Scored comparisons are bigger entries

A comparison scored over an axis (`compare(..., over="time")` — see
[Skill maps](skill_maps.md)) caches the same way, but its aligned pair holds every matched
step rather than one map, so an entry is as many times larger as there are steps. That is
still the right thing to cache: the expensive part is regridding each of those steps, and
the metric maps derived from them are cheap enough to recompute on the spot. `over`, the
matching method, the tolerance and the bin anchoring all join the key, so a scored pair and
a plain one never collide. `min_pairs` deliberately does not: it masks the maps and leaves
the aligned pair untouched.

The reference *lane* of a scored comparison is also keyed differently, because it is
cropped to the model's own extent before it is read (which is what keeps a global product
from being held whole). One consequence: that lane is not shared with a different model the
way an uncropped one is.

## Casts, moorings and stations: variants come from saved matches

A comparison against a point-like reference — a CTD cast (`profile`), a mooring or station
(`timeSeries`), or repeat casts at one station (`timeSeriesProfile`) — reads the model only
once. The first comparison against that reference saves the **basic** comparison: the model
matched to the observations at their own times and depths, with nothing averaged. Every
later variant is worked out from those saved matches without touching the model:

```python
osk.compare(reference="ctd_0412", test="his", variables=["temp"])
# reads the model at the cast's depths and time; saves the matches

osk.compare(reference="ctd_0412", test="his", variables=["temp"],
            depths=[{"min": 0, "max": 10}, {"min": 50, "max": 100}],
            aggregate={"Z": "mean"})
# 0-10 m and 50-100 m layer means, worked out from the saved matches: no model read
```

| Variant | Worked out from the saved matches as |
|---|---|
| A time window or instants | the matches inside it |
| A list of depths the observations have | the matches at those depths |
| `"surface"` | only for a dataset that is itself at the surface (see below): the model's top cell against it |
| Depth layers + `{"Z": "mean"}` | the plain mean of the matches inside each layer, on both sides alike; against a single cast the layers form a short profile |
| A depth layer against a one-instrument mooring | that mooring's matches when its instrument is inside the layer; left out, with a warning, when it is not |
| Time reductions, `resample`, climatologies | applied to both sides of the matches, then the difference recomputed |
| `detide` | the PL33 filter run on both sides of the matched series (needs roughly hourly, regular sampling; otherwise it warns and leaves the series as is) |
| `subtract_mean`, `qc` | as before; a different `qc` rebuilds the matches, but the model lane underneath is still a cache hit |

**"Surface" is for data that has one.** The model has a top cell and a gridded product has a top
level, but a cast's shallowest reading may be 2 m or 8 m, and a mooring's instrument sits wherever
it was deployed. So against a cast, a repeat-visit station or a mooring at depth, `"surface"` is
an error that names the depths the data does have. Ask for one of those, or a range, instead. A
dataset counts as at the surface when its catalog entry says so (`depth_convention: {support:
surface}`), or when its declared depth is within 1 m of the top: a surface buoy, say. A mooring's
depth is declared by `nominal_depth_m` (or `depth`), else `geospatial_vertical_min/max`. A
mooring that declares none is refused "surface" and depth ranges until it does, because it
can't be placed at the surface or in a layer. In a `compare()` over several references, one
that can't answer is skipped with a warning rather than failing the call.

**A layer with no observations in it is empty.** For casts, stations and moorings, a depth
range containing none of the data's depths is an error naming the depths it does have. In a
`compare()` over several references it is skipped with a warning instead. With several ranges
against one cast, an empty one comes back as NaN with a warning. Gridded products, which report
at standard levels, still take the nearest level for a range falling between two of them.

**Pooling a depth range across moorings.** Ask every mooring for the same layer, then average
them. Each one comes from its own saved matches, and moorings whose instrument is outside the
layer are left out:

```python
layer = osk.compare(reference=["m_10m", "m_30m", "m_80m"], test="his", variables=["temp"],
                    depths=[{"min": 0, "max": 50}], aggregate={"Z": "mean"})
# warns: m_80m left out (its instrument at 80 m is outside 0-50 m)
layer.average(by="variable")   # the 10 m and 30 m moorings pooled into one series
```

Some requests can't be answered from matches, and still go through the model as before:
- a depth the observations didn't sample;
- density and mixed-layer-depth calculators;
- the tidal harmonic calculators;
- gridded references, area means and trajectories.

A different `time_method` or `depth_method` matches the model differently, so it saves a
new set of matches. A result records which way it was made in its attrs: `derived_from` is
`"pairs"` or `"lanes"`, and `derived_reason` says why.

**Two consequences to know:**
- **Averages are taken over the matches.** A layer mean of the model is the mean of the model
  at the observation's depths — the same sampling as the observation side — rather than a
  thickness-weighted mean over the model's own levels. Likewise a monthly mean is a mean of
  matched values.
- **The first comparison against a mooring reads its whole record**, even when it asks for
  one month, so that later windows can be served from it.

### Seeing when the cache was used

Each comparison that used a saved result prints one line saying which:

```
ocean-skill: cache: his vs ctd_0412 (sea_water_temperature) -- reused the saved comparison
ocean-skill: cache: his vs ctd_0412 (sea_water_temperature) -- derived from saved model-data matches (model not read)
ocean-skill: cache: his vs mooring_A (sea_water_temperature) -- saved model-data matches for reuse
ocean-skill: cache: his vs woa23_temp (sea_water_temperature) -- reused the saved model lane
```

A comparison computed entirely fresh prints nothing. `osk.cache.verbose(False)` silences
these lines and the banner.

## Controls

| What | How |
|---|---|
| Where it lives | `osk.cache.path()` (or `path("prepared")`) |
| Where downloaded sources land | `osk.cache.obs_dir()` — never touched by `clear()` (`clear("obs")` raises); delete by hand to reclaim space |
| State, entry counts, size | `osk.cache.info()` |
| Turn off for this session | `osk.cache.disable()` |
| Silence the "cache: … reused …" lines | `osk.cache.verbose(False)` |
| Turn back on | `osk.cache.enable()` |
| Move it elsewhere | `osk.cache.enable("/path/to/dir")`, or set `$OCEAN_SKILL_DIR` |
| Pin a suite's cache | `cache_dir:` in the suite YAML (`docs/suites.md`) |
| Empty it | `osk.cache.clear()` → number removed (`clear("prepared")` for one layer) |
| Skip for one call | `osk.compare(..., cache=False)` / `Comparison(..., cache=False)` |
| Recompute and overwrite | `osk.compare(..., refresh=True)` / `c.align(refresh=True)` |
| Where outputs go | `osk.outputs.info()` / `osk.outputs.set_base(...)` |

Location resolves as `osk.cache.enable(dir)` → `$OCEAN_SKILL_DIR` →
[platformdirs](https://pypi.org/project/platformdirs/) user cache dir (the
conventional home for regenerable data, and what OS cleanup tools know to reclaim).

Moving the base directory moves **downloaded source files** (`cache/obs`) with it, not
just the two result layers — whether the move comes from the environment variable or
from `enable()` mid-session. The one thing that stays put is a location *you* set: an
explicit `cache_storage` in `~/.config/fsspec/*.json` or `$FSSPEC_*`, or a catalog
entry's own `cache_dir`, wins over ocean-skill's default and is never rewritten.

Everything under `osk.cache.path()` is reproducible and **safe to delete at any
time** — by hand, by `clear()`, or by the OS.

## Failure behaviour

A cache must never break a pipeline that would otherwise have worked, so every
failure degrades to "just do the work":

- **Corrupt or half-written entry** → warns, deletes the entry, recomputes.
- **Cache directory unwritable** (read-only filesystem, no `$HOME`) → warns once per
  save, returns the correctly-computed result anyway.
- **Interrupted write** → entries are written to a temporary path and moved into
  place, so a partial store is never left where a later run would read it.

## Format

One [zarr](https://zarr.readthedocs.io/) store per entry, `<key>.zarr`. A cached
result is byte-equivalent to a fresh one — same values, coordinates, variable
attributes, dataset attributes, and even the same variable *order* (zarr stores
alphabetically, so the original order is recorded and restored; nothing indexes
`data_vars` positionally today, but "cached behaves exactly like fresh" is the
invariant worth not having to think about later).

The key embeds a format version, so if what gets stored ever changes shape, old
entries are orphaned rather than loaded into a pipeline expecting something else. (The
definition fingerprint arrived with version 9: entries from before it are never asked for
again, and `osk.cache.clear()` reclaims them.)
