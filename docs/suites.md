# Suites: run a whole diagnostic from one YAML

A suite is a YAML file describing an ordered list of **pages** -- each one a single
`osk.field`, `osk.compare`, or `osk.summary` call -- plus shared defaults and output
settings. Running it draws every page, writes one PNG per figure, collects them into
one PDF (unless `pdf: false`), and writes a metrics CSV and a `manifest.json` recording
exactly what was drawn. The suite YAML is the whole interface: no Python is required to
run one.

```bash
ocean-skill-run suites/roms_marbl_diagnostic.yaml
```

`suites/roms_marbl_diagnostic.yaml` is a worked example for a ROMS-MARBL run: the
latest snapshot and monthly means of the key physics and BGC variables (model only),
plus model-vs-observation maps (WOA23 nutrients, GLODAPv2 alkalinity/DIC, satellite
chlorophyll) and a closing Taylor/target summary. `suites/roms_marbl_quick.yaml` is the
same `defaults:` block with only the snapshot and monthly pages -- the fast, repeated
check with no regridding and no downloads. Copy whichever fits, and edit the lines it
calls out (`defaults.test`, `refresh:`, `output_dir:`).

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
below.

## Schema

```yaml
name: roms_marbl_diagnostic          # a label you choose: report-dir prefix, metrics CSV stem
output_dir: /path/to/diagnostics      # report dirs go under here; unset -> $OCEAN_SKILL_OUTPUT or ./output
pdf: true                             # report.pdf beside the PNGs; false -> PNGs only
cache: true                           # false -> never reuse a prepared field from disk (see Caching, below)
catalog_search_paths:                 # optional: shared catalog dirs; relative = to this file
  - /anvil/projects/x-ees250129/catalogs
  - ../shared_catalogs

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
    field: {...}          # exactly one of field: / compare: / summary:
    for_each: {...}        # optional: fan this one page into several (see below)
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

### `field:` -- model only

Keys map straight onto `osk.field(source, variable, select=, aggregate=)`:

```yaml
field:
  variables: [temperature, salinity, ssh, {calculate: mld, method: density_threshold}]
  select: {depth: surface, time: latest}
```

`variables:` is always a list (even for one variable) and becomes `field()`'s
`variable=`; a combination (`{sum: [...]}`) or calculator (`{calculate: ...}`) spec
works the same as it does from Python. Several variables draw as one `FieldSet` figure
-- fine for a snapshot, but a *faceted* member (several time steps or depths) is
refused by `FieldSet.plot()`, which is why the monthly-means page below fans one
variable per page with `for_each` instead of listing them all in one `field:` block.

A depth-selected page cannot include a variable with no vertical axis (`ssh`, `mld`,
`co2_flux`, `PH`, `pCO2`) -- that raises a clear schema error naming the offending
variable.

### `compare:` -- model vs. observations

Keys map onto `osk.compare(test=, reference=, variables=, select=, aggregate=, ...)`
(`test` defaults to `defaults.test`). Sets that end up mixing plot families are drawn
as one figure per family, suffixed in the PNG filename; an empty result (every pair
`skip_missing`-ed) is a skipped page, not a fatal error.

### `summary:` -- pool the compare pages above

```yaml
summary: {kind: both, color_by: variable}
```

Pools the metric records from every `compare:` page that produced results (Taylor +
target by default; `kind: portrait` for the scorecard heatmap). Skipped when no
compare page succeeded.

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

Every page with no time key at all (a monthly-means page reducing the whole run, a
run-mean comparison like the GLODAP page) gets a literal `{"min", "max"}` window
injected covering the run's full recorded span -- this is what makes the expanded page
list, written into `manifest.json`, reproduce the same figures on replay rather than
silently meaning "whatever the run happens to cover by then."

## Caching

`osk.field`/`osk.compare` normally reuse an already-prepared field from ocean-skill's
own disk cache (see `docs/caching.md`). A page whose window reaches the run's current
latest step -- `time: latest`, the calendar month containing it, or an injected
whole-run window -- is run with caching off, since the run has grown since any earlier
check and a stale hit would silently serve an old mean. A completed month (safely in
the past) keeps caching on. Set the suite-level `cache: false` to disable it for every
page, which matters after a restart rewrites already-seen timestamps (`keep:
latest-per-file`) and an old prepared field could otherwise be served under a select
that now means something different.

## Output layout

Every invocation writes its own report directory -- nothing is ever overwritten:

```
<output_dir>/<name>_<test>_<t0>_to_<t1>_<run-time>/
    report.pdf                (unless pdf: false)
    figures/NN_<slug(title)>[_<family>].png
    metrics/<name>.csv (+ .txt)
    suite.yaml                 -- byte-identical copy of the input
    manifest.json
    run.log                    -- everything printed to the terminal during the run
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

`t0`/`t1` are the test source's own first/last recorded day; `<run-time>` is when the
command was run, down to the second, plus a short random suffix so two runs started in
the same second still land in different directories. `manifest.json` records the fully
expanded page list (every `time: latest`/`month: run`/window already resolved to a
literal value), each page's outcome, and the resolved `catalog_search_paths:`
directories -- enough that a second person with the same suite YAML and the same
catalogs gets the identical report.

`run.log` is a mirror of stdout/stderr for the run: a `== page i/n: <title> ==` header
and a `done in Ns`/`SKIPPED after Ns` line bracket each page, so a warning in between
is attributable to the page that raised it. It also carries the full traceback for
every skipped page and, if the run crashes outright, for the crash itself -- neither of
which the terminal shows. `--list` writes nothing, so no `run.log` is created for it.
A Python-level tee catches everything this package or its warnings print; it does not
catch a C library writing straight to a file descriptor.

## The log page and exit codes

Every PDF/PNG set ends with a plain-text log page: each page's title, whether it drew
or was skipped, why, and how long it took. It is `run.log`'s content rasterized as the
final report page; `run.log` is the greppable version, and the place to look for a
skipped page's traceback. A page is skipped -- never fatal to the rest of the report --
when its variable is absent, its observational catalog entry isn't on this machine's
search path (`osk.catalog.search_paths()`), or the comparison itself raises. `main`'s
exit code: `0` every page drew, `3` the run completed with some pages skipped, `1` no
page drew at all, `2` a schema or usage error (nothing was drawn or written).

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
