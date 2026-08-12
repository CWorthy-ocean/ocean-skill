# Caching aligned results

Every comparison's expensive step is `Comparison.align()`: it opens both sources
(often remote OPeNDAP), reduces each to one 2-D field (time mean, and for ROMS an
xgcm s-coord → z transform), then regrids test onto reference with xesmf. The result
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
   Keyed on source/variable/selection, NOT file contents — clear it after rerunning a model.)
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
method)`. This is the fast path, and the right outer boundary because it is exactly
*"everything needed to remake the plot, for both model and data, with no source
access"*: all reading, depth interpolation, unit conversion, regridding and
differencing are behind it, and nothing downstream touches the catalog again. It is
also small — four 2-D fields — so it is instant to reload, unlike the multi-GB
sources it came from.

**`prepared/` — one file per lane**, keyed `(source, variable, select)` with *no*
reference and *no* regrid method in it. This one makes a *miss* cheap. A lane's own
work depends only on that source, so comparing one model against several references
should prepare it once:

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

An entry is keyed on a hash of **the two source names, the variable, the selection,
and the regrid method** — deliberately *not* on the data itself, since hashing the
data would mean reading the very files the cache exists to avoid.

So if you **rerun a model and write new output to the same catalog path**, the cache
cannot tell, and will serve you the old result. After rerunning a model:

```bash
python -c "import ocean_skill as osk; print(osk.cache.clear(), 'entries removed')"
```

or, for a single call, `refresh=True` (recompute and overwrite) or `cache=False`
(bypass disk entirely). Changing the variable, depth, sources, or regrid method all
change the key on their own — those need no special handling.

## Controls

| What | How |
|---|---|
| Where it lives | `osk.cache.path()` (or `path("prepared")`) |
| Where downloaded sources land | `osk.cache.obs_dir()` |
| State, entry counts, size | `osk.cache.info()` |
| Turn off for this session | `osk.cache.disable()` |
| Turn back on | `osk.cache.enable()` |
| Move it elsewhere | `osk.cache.enable("/path/to/dir")`, or set `$OCEAN_SKILL_DIR` |
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
entries are orphaned rather than loaded into a pipeline expecting something else.
