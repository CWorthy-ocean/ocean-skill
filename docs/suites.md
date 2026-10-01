# Suites: run a whole diagnostic from one YAML

A suite is a YAML file describing an ordered list of **pages** -- each one a single
`osk.field`, `osk.compare`, or `osk.summary` call (a `field:` page may chain a few more
methods after it -- see `then:`, below), or a text-only `section:` divider that
organizes the PDF -- plus shared defaults and output settings.
Running it draws every page, writes one PNG per figure, collects them into one PDF
(unless `pdf: false`), and writes a metrics CSV and a `manifest.json` recording exactly
what was drawn. The suite YAML is the whole interface: no Python is required to run
one.

```bash
ocean-skill-run suites/roms_marbl_diagnostic.yaml
```

`suites/roms_marbl_diagnostic.yaml` is a worked example for a ROMS-MARBL run: the
latest snapshot and monthly means of the key physics and BGC variables (model only),
plus model-vs-observation maps (WOA23 nutrients, GLODAPv2 alkalinity/DIC, satellite
chlorophyll) and a closing Taylor/target summary. `suites/roms_marbl_quick.yaml` is the
same `defaults:` block with only the snapshot and monthly pages -- the fast, repeated
check with no regridding and no downloads. `suites/pacmed_review.yaml` is a larger worked
example that organizes its PDF with `section:` divider pages and combines domain-mean
time series with observation comparisons at each snapshot. Copy whichever fits, and edit
the lines it calls out (`defaults.test`, `refresh:`, `output_dir:`).

## Before you run one: register the model

A suite reads the model through a catalog entry, the same one `osk.field`/`osk.compare`
would use. Build it once with `build_kerchunk`/`build_catalog` (see `docs/Abigale_plots.ipynb`
cell 4 for the full worked example):

```python
from ocean_skill.build import build_kerchunk, build_catalog

refs = build_kerchunk(
    {"pac_dt_ramp": "tasks/*/joined_output/output_rst.*.nc"},
    root=model_base, grid=grid_loc,
    keep="latest-per-file",  # restart files: keep each file's main time step
    out_dir=".",
)
build_catalog(refs, "catalogs/pac_dt_ramp.yaml", title="pac_dt_ramp")
```

A suite's own `refresh:` block does exactly this rebuild automatically before each run,
so the same suite stays correct against a run that is still writing output -- see
below. Because that shared reference is overwritten by every later run, each report
directory keeps its own copy under `refs/` -- see Output layout.

## Schema

```yaml
name: roms_marbl_diagnostic          # a label you choose: report-dir prefix, metrics CSV stem
output_dir: /path/to/diagnostics      # report dirs go under here; unset -> $OCEAN_SKILL_OUTPUT or ./output
pdf: true                             # report.pdf beside the PNGs; false -> PNGs only
pdf_images: lossless                  # jpeg -> re-encode report.pdf's map rasters with Ghostscript (see Page size)
cache: true                           # false -> never reuse a prepared field from disk (see Caching, below)
catalog_search_paths:                 # optional: shared catalog dirs; relative = to this file
  - /anvil/projects/x-ees250129/catalogs
  - ../shared_catalogs
cache_dir: /anvil/scratch/x-me/osk_cache  # optional: pin this suite's cache; relative = to this file

refresh:                              # optional: rebuild the model's kerchunk reference first
  catalog: catalogs/pac_dt_ramp.yaml
  sources:
    - name: pac_dt_ramp
      files: tasks/*/joined_output/output_rst.*.nc
      grid: tasks/third_1wks/input/input_datasets/grid.nc
      ref: refs/pac_dt_ramp.parquet
      keep: latest-per-file           # forwarded to make_kerchunk; see its own docstring
      min_age: 0.0                    # seconds -- skip a file still being written (build._partition_unfinished)

defaults:                             # merged into every page; a page's own keys win
  test: pac_dt_ramp                   # exactly one source -- required for time: latest/month: run
  depths: [surface, 100]

pages:
  - title: "..."          # required; may contain {placeholder}s (see below)
    field: {...}          # exactly one of field: / compare: / summary: / section:
    section: "..."         # a text-only divider page in report.pdf (see below): no for_each:/then:/plot:
    for_each: {...}        # optional: fan this one page into several (see below)
    then: [...]            # optional, field: pages only: a method chain (see below)
    plot: {...}            # optional: kwargs forwarded to .plot()/.summary(), merged over defaults.plot
                            # (size:/zoom:/figsize: are pinned to the page canvas and warned
                            # about instead while pdf: true -- see Page size, below)
```

`defaults.test` must be a single source name, not a list: `time: latest`, `month: run`,
and the report directory's own time range are all read off *one* source's time axis,
and several sources have no single "latest" between them.

`catalog_search_paths:` registers extra shared catalog directories -- the same tier as
`$OCEAN_SKILL_CATALOGS` (see the Catalogs section of the main README) -- *before* the
suite touches any catalog entry, including under `--list`. This exists because
`$OCEAN_SKILL_CATALOGS` is normally set in a shell rc, which cron does not source, so a
suite that resolves its observational catalogs interactively can otherwise fail
silently (a missing entry is a *skipped page*, not an error) once it's scheduled. It
also means a suite handed to a collaborator is self-contained, matching the rest of this
doc's claim that everything that changes what gets drawn is a YAML key. A relative entry
resolves against **this suite file's own directory**, not the working directory --
deliberately different from `refresh:`/`output_dir` above, which are working-directory
relative -- so the suite means the same thing run from cron, a notebook, or any shell. A
directory that doesn't exist is a usage error (`main` returns `2`), since the suite
names it explicitly. Registering it never outranks a user's own
`~/.ocean-skill/catalogs` or the project's own `./catalogs`.

`cache_dir:` is the same fix applied to the cache rather than the catalogs: it calls
`osk.cache.enable(dir)` for the whole process before the suite touches any source
(resolved the same way as `catalog_search_paths:` -- relative to this suite file, not
the working directory), so a suite scheduled from cron keeps reading from and adding to
one particular directory across repeated runs instead of falling back to
`$OCEAN_SKILL_DIR` or a per-user platformdirs cache. It is a base directory: entries land
under `<cache_dir>/cache/{prepared,aligned,weights,obs}`, same layout as the default.
Several suites can point at the same `cache_dir:` -- cache keys are source-name identity,
not path, so entries from different suites coexist. Unlike `catalog_search_paths:`, a
missing directory is not an error; the cache layer creates what it needs. See "Caching"
below, and `docs/caching.md` for the key format and identity-caveat this inherits.

### `field:` -- model only

Keys map straight onto `osk.field(source, variable, select=, aggregate=, ...)` -- every
keyword `osk.field()` accepts (`qc:`, `detide:`, `label:`, ...) passes straight
through, not only `select:`/`aggregate:`. `cache:` is the one reserved key: it is the
suite's own to set (`cache:`/`cache_dir:` above, and each page's own tracks-or-closed
rule -- see Caching, below), not a per-page kwarg.

```yaml
field:
  variables: [temperature, salinity, ssh, {calculate: mld, method: density_threshold}]
  select: {depth: surface, time: latest}
```

`variables:` is always a list (even for one variable) and becomes `field()`'s
`variable=`; a combination (`{sum: [...]}`) or calculator (`{calculate: ...}`) spec
works the same as it does from Python. A one-variable list draws exactly as
`osk.field(source, variable).plot()` would -- one panel per facet step (a month of a
`resample`, say), rows per depth -- which is why the monthly-means page below fans one
variable per page with `for_each` rather than listing several in one `field:` block.
*Several* variables in one `field:` block draw as one `FieldSet` figure, one map panel
per variable -- fine for a snapshot, but every member must already be a single map:
one still faceted over time or depth (several steps standing) is refused, naming the
offending source and axis.

A depth-selected page cannot include a variable with no vertical axis (`ssh`, `mld`,
`co2_flux`, `PH`, `pCO2`) -- that raises a clear schema error naming the offending
variable.

### `compare:` -- model vs. observations

Keys map onto `osk.compare(test=, reference=, variables=, select=, aggregate=, ...)`
(`test` defaults to `defaults.test`). Sets that end up mixing plot families are drawn
as one figure per family, suffixed in the PNG filename; an empty result (every pair
`skip_missing`-ed) is a skipped page, not a fatal error.

`aggregate:` may be a **list of whole specs**, which fans the page over statistics the
way `variables:` and `depths:` fan it -- one member per spec, one figure, each member
labelled by its statistic. A list *under* an axis is something else: a chain of steps
(`time: [{groupby: month, reduce: mean}, var]`, the variance of the seasonal cycle).
The mean and the seasonal-cycle variance of a model against WOA, at two depths, is one
page (the 12-month `woa23_<var>_monthly` sources exist for exactly this):

```yaml
- title: "Temperature vs WOA23 -- mean and seasonal variance"
  compare:
    reference: [woa23_temperature_monthly]
    variables: [temperature]
    depths: [surface, 200]
    aggregate:
      - {time: mean}
      - {time: [{groupby: month, reduce: mean}, var]}
```

The same list works on a `field:` page (plain specs only). Chains, their rules and
units are in the main README ("Statistics of a climatology").

### `summary:` -- pool the compare pages above

```yaml
summary: {kind: both, color_by: variable}
```

Pools the metric records from every `compare:` page that produced results (Taylor +
target by default; `kind: portrait` for the scorecard heatmap). Skipped when no
compare page succeeded.

### `section:` -- a divider page

```yaml
- title: "Part 2 -- the alkalinity minimum"
  section: >-
    The surface alkalinity minimum on 2010-10-31, then how that point got there.
    Observations are WOA23 and GLODAPv2 where they exist.
- title: "Alkalinity ({test})"
  field: {variables: [alkalinity], select: {depth: surface, time: "2010-10-31"}}
```

A page that is only text: `title:` as a large heading, centred, with the `section:`
string wrapped beneath it, on a US Letter page of its own in `report.pdf`. It lets a long
report have chapters. `section: ""` is a title-only divider; a YAML block scalar (`>-`
folds the lines into one paragraph, `|` keeps your own line breaks, and a long line wraps
either way) is the natural way to write longer notes. `{placeholder}`s in the title and
the text resolve against `defaults:` -- there is no `for_each:` to bind anything else --
and a literal brace needs the usual `{{ }}`.

- **A divider is only drawn when a page after it draws.** It is buffered and written
  into the PDF immediately before the next figure. A later `section:` page replaces one
  still waiting, so a section whose pages all skipped leaves no empty divider behind, and
  one at the very end of the suite is dropped. A run where nothing draws still writes no
  PDF at all.
- **PDF only.** A divider never gets a PNG and does not take a number: the
  `figures/NN_` numbering counts figures alone, so it reads the same with or without
  sections. Under `pdf: false` a section does nothing.
- **Not a figure.** Sections take no part in the exit code or the "N page(s) drawn"
  count (see Exit codes). `manifest.json` gives each one the status `ok` (written) or
  `skipped` with a reason -- `no page after this section drew`, or `pdf: false -- section
  pages appear only in report.pdf`. `run.log` heads it `== section: <title> ==`, and
  `--list` prints `NN. [section] <title>`.
- **`for_each:`, `then:` and a non-empty `plot:` are schema errors** on a section page,
  naming it: there is no data to fan out, chain, or plot. A bare `section:` with nothing
  after it is YAML for null, which is no kind at all and so is refused too -- write
  `section: ""`.

## `for_each`: one page becomes several

```yaml
for_each:
  variable: [temperature, salinity, nitrate]   # a literal list
  depth: "{depths}"                            # or a placeholder resolved against defaults
  month: run                                   # every calendar month the test source's time axis touches
```

Produces the Cartesian product, in YAML order, of every name's own list -- one page per
combination. `month: run` lists `(year, month)` oldest first; `month: {run: {last: N}}`
keeps only the last `N` (use this on a long run to bound the WOA page count). A
`for_each` name may not collide with a `defaults` key.

## `{placeholder}` templating

A page's `title`, and every string inside its `field:`/`compare:`/`summary:`/`plot:`
dict, may contain placeholders resolved against `defaults` plus whatever `for_each`
bound for that combination:

- **Exactly one placeholder and nothing else** (`"{depths}"`, `"{month.window}"`) is
  replaced by the referenced object itself, keeping its type -- a list stays a list, a
  window stays a `{"min", "max"}` dict. This is how `select: {depth: "{depths}"}` gets
  the real list rather than its string form.
- **Anything else** (`"woa23_nitrate_month{month.mm}"`, a title) goes through ordinary
  `str.format`, which already resolves dotted attribute access (`{month.name}`). A
  literal brace needs the usual `{{ }}`.
- `month` exposes `.year`, `.month`, `.mm` (zero-padded), `.name` ("January"), and
  `.window` (`{"min": "...-01T00:00:00", "max": "...-<last day>T23:59:59"}`, inclusive).
- `depth` (bound by a `for_each: {depth: ...}`) formats plainly as its own value
  ("surface", "100") and exposes `.label` ("surface", "100 m").

An unresolvable placeholder is a schema error naming the page and the placeholder text,
not a silent pass-through.

## `time: latest`

The literal word `latest` in a `field:` select, or in a `compare:` `select.test`,
resolves to the newest timestamp on `defaults.test`'s own time axis (one lazy,
coordinate-only read, done once per run regardless of how many pages use it). Never
combine it with a `method:` key -- nearest-matching is automatic and `method` is
otherwise ignored with a warning.

The same shorthand works outside suites: `osk.field(src, select={"time": "latest"})`
and `osk.compare(..., select={"time": "latest"})` (or a pair-spec's `test` side; not its
`reference` side) resolve it to that source's newest step's date as the field or
comparison is built, so the on-disk cache keys on the date, not the word. A run that
has not moved hits the cache; one that has gained a step is a new key. In a long-lived
session the catalog is read once, so call `osk.cache.clear()` to notice steps appended
since.

Every page with no time key at all (a monthly-means page reducing the whole run, a
run-mean comparison like the GLODAP page) gets a literal `{"min", "max"}` window
injected covering the run's full recorded span -- this is what makes the expanded page
list, written into `manifest.json`, reproduce the same figures on replay rather than
silently meaning "whatever the run happens to cover by then."

None of this needs writing into a page's own `title:` to show up on the figure: a
`field:` page whose `select`/`aggregate` collapses time to one answer -- the resolved
`latest` instant, a single facet bin, or a window-reducing mean -- carries it
automatically in the suptitle (`surface · 2012-12-31`, or `surface · mean over
2012-01-01–2012-12-31`), the same way it already carries the depth a `select`
narrowed. A page still faceted over several steps needs no suptitle text either,
since each panel already says its own (`Jan 2012`, `Feb 2012`, ...).

## `then:` -- a method chain after `field:`

A `field:` page draws whatever `osk.field()` returns; `then:` runs a short, fixed
chain of methods on that object first, in order -- the suite-YAML form of a Python
chain like:

```python
run = osk.field("first_half", "alkalinity", select={"depth": "surface", "time": "2010-06-30T2345"})
ext = run.extremum("min")                                   # where is it lowest?
ext.series(variables=["dissolved_inorganic_carbon", "temperature", "salinity",
                       "nitrate", "oxygen"]).plot()          # ... and how did it get there?
```

as a page:

```yaml
- title: "Alkalinity minimum ({test}) -- time series"
  field:
    variables: [alkalinity]
    select: {depth: surface, time: "2010-06-30T2345"}    # or time: latest
  then:
    - extremum: min                              # {kind: min} would do the same
    - series:
        variables: [dissolved_inorganic_carbon, temperature, salinity, nitrate, oxygen]
```

Each step is one of:

- a bare name (`extremum`), run with its defaults;
- `name: <scalar>`, filling in the one argument that reads most naturally as a bare
  value (`extremum: min` -> `kind: min`; `series: temperature` -> `variables:
  [temperature]`);
- `name: {...}`, the step's keyword arguments in full (`series.variables`/`time`/
  `pad`/`label` -- see `Extremum.series()`'s own docstring for what each does).

Only two steps exist today, and each expects what the one before it produced:
`extremum` (`kind: min`/`max`, default `max`) needs a single `Field` and returns an
`Extremum`; `series` needs that `Extremum` and returns the point time series
`FieldSet` that `plot:` then draws. `then:` is refused wherever that shape does not
hold:

- on a `compare:`/`summary:` page (a schema error);
- when `variables:`/`source:` would build more than one `Field` (`extremum` has one
  map to search, not several -- give each variable its own page instead);
- when the steps are out of order, or a step's own arguments don't match what it
  takes (both schema errors, caught by `--list` before any data is read);
- when `extremum` would otherwise search the whole, ever-growing run: the page needs
  an explicit `select.time` (a literal, a range, or `latest`) or an `aggregate.time`
  that collapses it;
- when `extremum` would otherwise search a bare vertical axis `.plot()` would have
  shown as just the surface: the page needs an explicit `select.depth`/`Z`/`z`/
  `vertical`/`sigma0`.

`series`'s own default window -- `DEFAULT_PAD_STEPS` (10) native time steps either
side of wherever `extremum` landed -- pads by the source's *own* cadence, so it means
something different for an hourly stream than for a restart file written every few
weeks; pass `series: {pad: N}` or an explicit `series: {time: ...}` to control it
directly. When the page's own `select.time` is a single fixed instant (including a
resolved `latest`), that window is worked out once, here, and written into
`manifest.json` as a literal `{"min", "max"}` -- so replaying the same manifest later
reproduces the same figure regardless of how much the run has grown by then, the same
guarantee every other page already gets (see `time: latest`, above). It is left for
`Extremum.series()`'s own runtime default only when the page instead searches a
*range* of times, since which instant the minimum/maximum actually falls on is then
only known once the data is read.

The time-series figure's suptitle says what picked its location, by default, in the
shape `Time series at the surface alkalinity minimum: 126.1 mmol/m^3 at 16.1°N 97.6°E on
2010-10-31`: the depth (when the page's `select` leaves a single level), whether it is
the minimum or the maximum, its value with units, where it is, and the snapshot date it
was found on. A page's own `plot: {title: "..."}` replaces it. The same value and
position are also printed (so they land in `run.log`) and recorded per page in
`manifest.json`'s own `results:` list.

## Caching

`osk.field`/`osk.compare` normally reuse an already-prepared field from ocean-skill's
own disk cache (see `docs/caching.md`). A suite page decides whether its own call gets
`cache=True` or `cache=False` from one rule: cache whenever the test lane's time
selection either **tracks** the run's current last step, or is **closed** against it --
anything else keeps `cache=False`, the safe default.

**Tracks**: `time: latest` resolves to the last step's own timestamp, and a page with no
time key at all gets the injected `{"min", "max"}` window described above -- both
already change the moment the run's last step does, whether that is a new step being
appended or the old one being *replaced* (which is what happens to the restart file
still being written, under `keep: latest-per-file`). Both get `cache=True`: rerunning
the suite once the model has stopped hits the cache instead of reprocessing, and
rerunning it while the model is still advancing recomputes -- a new latest step means
a new key -- and refills the cache for next time.

**Closed**: checked against the run's current time axis, not assumed -- a page's
selection picks at least one step, none of them the run's current last step, and it
would pick exactly the same steps with one more step appended or its current last step
taken away. A calendar month still being written, an explicit window or instant
reaching up to the run's end, a flat (non-paired) `compare:` select with no time key at
all (it reads both lanes, so it is never rewritten, only checked), and a `times=` page
(each bin's own value replaces whatever time key the page had, so nothing here can
speak for it) all keep `cache=False`. A `detide:` page's cutoff sits a little earlier
still, since its PL33 filter leaves an edge of the record that keeps changing shape as
the run grows.

Set the suite-level `cache: false` to disable caching for every page regardless of any
of the above -- useful after any change that a select-identity key cannot see on its
own: old output purged, a segment rewritten in place with the same final timestamp, or
a reference that is itself still being appended to.

`--list` prints `latest step of <test>: ...` once, so a comparison between two `--list`
runs shows whether the run has actually moved, and marks each page `(cache)`,
`(no-cache (cache: false))`, or `(no-cache (may change as the run grows))`.

Set `cache_dir:` to pin where those entries accumulate across repeated runs of this
suite (see the schema section above). Entries for a step the run has since moved past
are not pruned automatically -- they simply stop being read, and accumulate until
`osk.cache.clear()`.

## Output layout

Every invocation writes its own report directory -- nothing is ever overwritten:

```
<output_dir>/<name>_<test>_<t0>_to_<t1>_<run-time>/
    report.pdf                (unless pdf: false, or no page drew at all)
    figures/NN_<slug(title)>[_<family>].png      (figures only: section: pages have none)
    metrics/<name>.csv (+ .txt)
    suite.yaml                 -- byte-identical copy of the input
    manifest.json
    run.log                    -- everything printed to the terminal during the run
    refs/<ref name>            -- copy of each refresh: reference, exactly as drawn
<output_dir>/latest.txt        -- path of the newest report dir
```

### Page size

`report.pdf`'s pages are all a fixed US Letter portrait (8.5x11in): each figure's own
ink is placed at the top of the page, horizontally centred, with a small margin above
it -- a wide figure (a time series, three maps in a row) leaves white space below
rather than growing the page, and a tall one (a profile) is unaffected. This is what
makes the report paginate and print like a document regardless of what any one page
draws.

Because of this, `size:`/`zoom:`/`figsize:` in a page's `plot:` (or a `summary:`
page's own kwargs) are ignored while `pdf: true` -- pinned to the `"page"` canvas
instead, with a warning naming the ignored kwarg (once per suite run, not once per
page). Set `pdf: false` to size figures freely; PNGs are still written, as tight crops
around each figure rather than letter pages. A figure whose own ink still doesn't fit
8.5x11 even after pinning (rare -- only an explicit `figsize:` outside this grammar
could do it) grows the page rather than clipping the figure, with its own warning.
A `section:` divider is always an exact letter page.

**Smaller PDFs: `pdf_images: jpeg`.** About 95% of a map report's bytes are its
rasterized map images, which matplotlib's PDF backend can only store losslessly.
`pdf_images: jpeg` (the default, `lossless`, is today's output byte-for-byte) re-encodes
them as JPEG in a post-pass through Ghostscript (`gs`, found on `PATH`), after the PDF is
closed. On the 13-page report it was tuned on, 7.6 MB became 4.8 MB with no visible
change when compared against a lossless render at 300 dpi: chroma is not subsampled, so
coastlines and fronts stay crisp, and the transparency masks that cut land out of the
maps stay lossless. The run prints one line (`report.pdf: 20.1 MB -> 10.4 MB (JPEG via
Ghostscript)`, so it is in `run.log`) and `manifest.json` records `"pdf_images"` (what
the suite asked for) and `"pdf_images_applied"` (whether the pass actually ran). If `gs`
is not installed, fails, takes more than ten minutes, or would not make the file smaller,
the run warns, names the reason, and keeps the lossless PDF. PNGs are never touched, and
the key does nothing under `pdf: false`.

`t0`/`t1` are the test source's own first/last recorded day; `<run-time>` is when the
command was run, down to the second, plus a short random suffix so two runs started in
the same second still land in different directories. `manifest.json` records the fully
expanded page list (every `time: latest`/`month: run`/window already resolved to a
literal value), each page's outcome, the `pdf_images` setting and whether it was applied,
the resolved `catalog_search_paths:` directories,
the effective `cache_dir` (the suite's own if set, otherwise wherever
`osk.cache.base_dir()` already pointed), and (when the suite has a `refresh:` block) a
`"refresh"` key -- enough that a second person with the same suite YAML and the same
catalogs gets the identical report.

`refs/` only appears when the suite has a `refresh:` block. Each source's reference
(`refresh.sources[].ref`, a parquet directory or a `.json` file) is copied there
right after the report directory is created, before any page is drawn -- so it
matches the model state the report was actually drawn from, even though the shared
reference at `ref:` is overwritten by the *next* run's own refresh. `manifest.json`'s
`"refresh"` key records `{"catalog": ..., "sources": [{"name", "ref", "snapshot"}]}`,
where `snapshot` is the copy's path (`null` if no reference existed yet to copy). The
copy pins file paths and byte ranges into the model output, not file contents: a
restart that rewrites an output file in place still changes what the snapshot reads.
Open a snapshot directly with `xr.open_dataset(path, engine="kerchunk", chunks={})`.

`run.log` is a mirror of stdout/stderr for the run: a `== page i/n: <title> ==` header
and a `done in Ns`/`SKIPPED after Ns` line bracket each page, so a warning in between
is attributable to the page that raised it. It also carries the full traceback for
every skipped page and, if the run crashes outright, for the crash itself -- neither of
which the terminal shows. `--list` writes nothing, so no `run.log` is created for it.
A Python-level tee catches everything this package or its warnings print; it does not
catch a C library writing straight to a file descriptor.

## Exit codes

A page is skipped -- never fatal to the rest of the report -- when its variable is
absent, its observational catalog entry isn't on this machine's search path
(`osk.catalog.search_paths()`), or the comparison itself raises; `run.log` has the
traceback. `main`'s exit code: `0` every page drew, `3` the run completed with some
pages skipped, `1` no page drew at all, `2` a schema or usage error (nothing was drawn
or written). `section:` divider pages are not counted either way -- they are neither
"drawn" nor "skipped" as far as these codes and the `N page(s) drawn, M skipped` line
are concerned, so a suite whose only data pages skipped is `1` however many dividers it
has.

## CLI

```
ocean-skill-run SUITE.yaml [--list]
```

`--list` validates the suite, resolves `latest`/`month: run`/`for_each`, and prints the
expanded page list with no drawing and nothing written -- a fast check on an edited
suite, or on how many pages a long run's `month: run` will produce, before committing
to the full run. `--list` still registers `catalog_search_paths:` first, since resolving
`latest`/`month: run` reads a source's time axis. There is no other flag: everything
that changes what gets drawn is a YAML key, so the suite file alone is what reproduces a
report.
