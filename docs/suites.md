# Suites: run a whole diagnostic from one YAML

A suite is a YAML file describing an ordered list of **pages** -- each one a single
`osk.field`, `osk.compare`, or `osk.summary` call (a `field:` page may chain a few more
methods after it -- see `then:`, below), a property-property plot (`osk.XY`, or `osk.TS`
for salinity against temperature -- see `XY:` and `TS:`, below), or a text-only
`section:` divider that organizes the PDF -- plus shared defaults and output settings.
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
time series with observation comparisons at each snapshot and a `TS:` page of T-S
diagrams in six Pacific boxes. Copy whichever fits, and edit the lines it calls out
(`defaults.test`, `refresh:`, `output_dir:`). Each run's report lands in one stable
directory, `<output_dir>/<name>/`, with a cover page describing the simulation as page 1
of `report.pdf` -- see Output layout and Cover page.

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

## Starting a suite from a run folder

The three suites that ship with the package -- `mid-run-check`, `final-report`,
`model-comparison` -- live in one place, `ocean_skill/workflows/templates/*.yaml`
(inside the installed package, so `pip install` brings them; find the folder with
`python -c "import ocean_skill.workflows as w, pathlib; print(pathlib.Path(w.__file__).parent / 'templates')"`).
Each file is a complete, runnable suite for the PACMED run it was written for, with its
run-specific block (names, paths, dates) between `# >>> setup` and `# <<< setup` marker
comments. For your own run, copy the file and edit that block by hand (mainly
`refresh:`, `defaults.test`, the dates, `output_dir`, `cache_dir`,
`catalog_search_paths`); that is the only route for `final-report` and
`model-comparison`. For `mid-run-check`, `ocean-skill-suite new` fills the block from a
C-Star run folder and keeps everything else, pages and comments included, as the shipped
file has it:

```bash
ocean-skill-suite new mid-run-check \
    --run /scratch/cstar/my_run --name my_run --output-dir /projects/me/diagnostics \
    -o my-run-check.yaml
ocean-skill-run my-run-check.yaml --list
ocean-skill-run my-run-check.yaml --smoke 2 --self-check --strict
ocean-skill-run my-run-check.yaml
```

`--run` is the folder holding `tasks/`; repeat it for the segments of a run that was
restarted in pieces (they become a list of globs under `files:`). Other options:
`--files GLOB ...` (name the restarts yourself), `--grid FILE`, `--catalog-dir DIR`
(where this suite's own catalog goes; default `<output-dir>/catalogs`),
`--shared-catalogs DIR` (the observation catalogs; default the Anvil directory) and
`--force` (replace an existing `-o` file; default `./mid-run-check.yaml`). The command
reads only file names and the grid's `eta_rho` length and changes nothing but the YAML it
writes. A missing required flag, or another template name, is an error.

**What is discovered**, from the layout C-Star writes (`<run>/tasks/<task>/{input/
input_datasets,output,work}`):

- the restarts, `tasks/*/output/output_rst.*.nc`, or the folder of symlinks that gathers
  them (`gathered_output/`, `tasks/rst_files/`) when there is one, since it de-duplicates
  restarts written by two tasks;
- the grid: the one `*_grid.nc` in the tasks' input folders (an error naming the
  candidates when there are none or several with different names);
- the restart `subchunk:`: one level and the largest divisor of the grid's `eta_rho`
  between 32 and 128 rows (962 rows give 74). A grid with none (674 = 2 x 337) gets one
  level and whole rows, with a comment in the YAML and a warning, because a page matched
  to a depth over many restarts then needs more memory there;
- the workplan and the newest resolved blueprint, for `run_info:`;
- `defaults.period.min`, the first restart's day.

**What to check.** Every place the command was not sure is a `# TODO check` comment in
the YAML (a run folder with no workplan or blueprint, restart dates it could not read).
The pages are the PACMED ones, so their boxes and dates are the first edit for another
domain. The written file is loaded with the suite's own schema and every path under
`refresh:` and `run_info:` must exist; any failure is printed and the exit status is 1.

**Editing a shipped suite.** Edit the file under `ocean_skill/workflows/templates/`;
there is no second copy to keep in step. Keep the `# >>> setup` / `# <<< setup` markers
intact so `ocean-skill-suite new` can still fill the block.

## Schema

```yaml
name: roms_marbl_diagnostic          # a label you choose: the report dir's name, metrics CSV stem
output_dir: /path/to/diagnostics      # report dir goes under here; unset -> $OCEAN_SKILL_OUTPUT or ./output
timestamped_dir: false                # true -> a new <name>_<test>_<t0>_to_<t1>_<time>_<id>/ per run (see Output layout)
pdf: true                             # report.pdf beside the PNGs; false -> PNGs only
cover: true                           # false -> no cover page as page 1 of report.pdf (see Cover page)
run_info:                             # optional: override what the cover page finds on its own
  name: PACMED v2                     # the title; default: found from the C-Star workplan/blueprint
  blueprint: runs/bp_resolved.yaml    # a resolved C-Star blueprint; relative = to this file
  workplan: runs/workplan.yaml        # the C-Star workplan; relative = to this file
  notes: {Purpose: "spin-up check"}   # printed on the cover as written, in order
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
      files: tasks/*/joined_output/output_rst.*.nc   # a glob, or a list of globs (see Refresh: ...)
      grid: tasks/third_1wks/input/input_datasets/grid.nc
      ref: refs/pac_dt_ramp.parquet
      keep: latest-per-file           # forwarded to make_kerchunk; see its own docstring
      min_age: 0.0                    # seconds -- skip a file still being written (build._partition_unfinished)
      subchunk: {s_rho: 1, s_w: 1, eta_rho: 74}  # optional, forwarded to make_kerchunk: smaller stored chunks
                                      # (here 1 level x 74 rows), so depth matching reads bands of whole
                                      # columns, not every level of the domain; the length must divide the dim
  forcing:                            # optional: catalog roms-tools forcing files into the same catalog
    - name: base_bry                  # kind: boundary -> base_bry_south, ..., base_bry_west
      kind: boundary                  # boundary (build.add_boundary_sources) | surface (build.add_forcing_source)
                                      # | initial_conditions (build.add_initial_conditions)
      files: [y2011/*_boundary-physics_*.nc, y2012/*_boundary-physics_*.nc]  # glob or globs, joined in order
      grid: input/grid.nc
      ref: refs/base_bry.parquet      # optional: kerchunk reference, rebuilt only when the files change
      subchunk: {bry_time: 1}         # optional, needs ref:, forwarded to make_kerchunk: split each file's time chunk
                                      # so monthly files of 28-31 steps can be joined (the build otherwise fails with
                                      # "inconsistent chunk shapes"); uncompressed (netCDF-3) files only
      variables: [temp, salt, NO3]    # optional: base names to keep; default all
      climatology: false              # true -> a 12-month _clim file: month replaces time
      sides: [south, north]           # optional, boundary only; default all four
      strip: [west, south]            # optional, boundary only: also write <name>_strip, the sides joined (see below)
      hc: 250.0                       # optional: the s-coordinate's critical depth, for a grid file that states none
    - name: base_co2
      kind: surface
      files: input/*_surface-bgc-mbl_co2_*.nc
      grid: input/grid.nc
      time_coord: xco2_time           # optional, surface only: the time variable (its dimension may be named otherwise); default time
    - name: ic_run                    # kind: initial_conditions -> one entry, ic_run, that reads like a restart
      kind: initial_conditions
      files: input/*_initial_conditions_*.nc   # exactly one file
      grid: input/grid.nc

defaults:                             # merged into every page; a page's own keys win
  test: pac_dt_ramp                   # exactly one source -- required for time: latest/month: run
  depths: [surface, 100]

pages:
  - title: "..."          # required; may contain {placeholder}s (see below)
    field: {...}          # exactly one of field: / compare: / summary: / section: / XY: / TS:
    section: "..."         # a text-only divider page in report.pdf (see below): no for_each:/then:/plot:
    XY: {members: {...}, x: ..., y: ...}   # one variable against another (see below); TS: is XY: with x/y fixed
    for_each: {...}        # optional: fan this one page into several (see below)
    then: [...]            # optional, field: pages only: a method chain (see below)
    contours: true         # optional, field:/compare: pages: contour lines over the fill (see below)
    draw: target           # optional, compare: pages only: plot (default) | metrics_series | target | taylor (see below)
    plot: {...}            # optional: kwargs forwarded to .plot()/.summary(), merged over defaults.plot
                            # (size:/zoom:/figsize: are pinned to the page canvas and warned
                            # about instead while pdf: true -- see Page size, below)
    caption: "..."         # optional, any page but section:: text under the figure in report.pdf (see Captions)
```

### Refresh: several folders, initial conditions, boundary strips

**`refresh.sources[].files` as a list of globs.** A run restarted in pieces leaves its
output in several task folders, and a second run continues in more, some repeating dates
(a test segment):

```yaml
sources:
  - name: pacmed
    files:
      - /anvil/scratch/x-me/cstar/pacmed/tasks/*/output/output_rst.2010*.nc   # first_wk, second_wk, ...
      - /anvil/scratch/x-me/cstar/pacmed_2015/tasks/*/output/output_rst.*.nc
    keep: latest-per-file
    ...
```

(Globs are Python's: `*`, `?` and `[...]`, no `{a,b}`.) The matched
files are joined into one reference in **time order** -- not path order, since folder
names do not sort chronologically (`first_half` comes before `first_wk`). When two files
give the same time step, after `keep: latest-per-file` has cut each to its newest record,
the **later one wins**: a file of a later glob beats one of an earlier glob, and within
one glob the later path in sorted order beats the earlier. Each file dropped that way is
logged:

    refresh: pacmed: 2010-01-07T23:45:00 from <path> replaces <path>

A path two globs both match counts once, at its later place; a glob that matches nothing
does not stop the others. Only a file that gives *one* record is dropped for a repeat (every
file under `latest-per-file`); with another `keep:` the build's own rule decides among
repeated records, the later in the order handed over winning. A single glob written as a
string reads as it always did (every match in path order); write it as a one-item list to
get the time ordering.

**`kind: initial_conditions`.** A roms-tools initial-conditions file becomes one catalog
entry, `<name>`, that reads like model output: `s_rho x eta_rho x xi_rho` fields (`temp`,
`salt`, the BGC tracers, `zeta`, `u`/`v`) at one time step, with `lon`/`lat`, a land mask
and a `z_rho` depth coordinate. So a page can compare it with the run's own first restart
at `depth: surface` or `200` exactly as it compares two restarts. `files` is one file and
`grid:` the run's roms-tools grid file, which supplies the horizontal grid and `sigma_r`.
The file's own `Cs_r`/`Cs_w` and `theta_s`/`theta_b` attributes are used. `hc`
comes from the file's attribute, else the grid file's, else the entry's `hc:` key (roms-tools
4.0.0 writes no `hc` attribute in an initial-conditions file); an explicit `hc:` beats both,
and none of the three is an error that says so. The time is `ocean_time` (seconds since the
file's `model_reference_date`), checked against its `ini_time`: they must agree. `variables:`
limits the fields kept; `ref`, `subchunk`, `climatology` and `sides` do not apply.

**`strip:` on a `kind: boundary` entry.** `strip: [west, south]` also writes an entry
`<name>_strip` beside the per-side ones: the west boundary run north to south, then the south
boundary run west to east, the corner cell once -- the strip `select: {transect: {boundary:
[west, south]}}` draws from model output, as a catalog entry of its own. It is built in the
reader chain (no file is written) and only holds the variables every named side has. Each side
must meet the next at a grid corner (`[west, east]` is an error) and be among `sides:`. Read
it like a south entry whose single row is the whole strip -- `select: {transect: {eta_rho: 0}}`
cuts the section, with `along` in kilometres (the cumulative great-circle distance over the
joined lon/lat) and the corner lines a boundary-following section draws; pages then need an
explicit depth list, as every section comparison does:

```yaml
- title: "Boundary temperature, west + south strip"
  compare:
    test: pyesper_bry_phys_strip
    reference: [pacmed2015_bry_phys_strip]
    variables: [temperature]
    select: {transect: {eta_rho: 0}, depth: [5, 50, 200, 500, 1000], time: "{period}"}
    aggregate: {time: mean}
```

Against a climatology's strip (`climatology: true`, which keeps its `month` axis) give each
side its own aggregate: `aggregate: {test: {time: mean}, reference: {month: mean}}`.

### Suite-wide plot options: `defaults.plot`

`defaults: {plot: {...}}` is merged under every page's own `plot:` (the page's keys win),
so a setting every figure should share is written once:

```yaml
defaults:
  test: pac_dt_ramp
  plot: {robust: true}        # every figure that has a robust color scale gets it
pages:
  - title: "Temperature"
    field: {variables: [temperature]}
  - title: "Salinity, full range"
    field: {variables: [salinity]}
    plot: {robust: false}     # this page's own value wins
```

A default only fills in where the page's draw method takes it. A page whose method does
not (`robust` means nothing to a `draw: metrics_series` page or a T-S diagram) is drawn
without it and a warning names the skipped option; the same option written in that
page's own `plot:` is still an error. `summary:` pages take their keywords from the
`summary:` dict alone, so `defaults.plot` never reaches them, and a `section:` has no plot.
`defaults.plot.size/zoom/figsize` are pinned to the page canvas like a page's own (see
Page size). `max_rows` (see `compare:`) is a plot default that is dropped on the pages it
does not suit without a warning.

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

`source:` may be a **list** (`source: [roms_run_a, roms_run_b]`): `osk.field()` then
builds a `FieldSet` with one member per source, drawn together -- series and profiles
overlay, and `plot: {encode: {color: source}}` colors them by source. Every member
shares the page's one `select:`, so the suite resolves the window from a single source's
time axis: `defaults.test` if it is one of the entries, otherwise **the first entry** (so
put the run whose span the page should follow first). That source's index resolves
`time: latest`, a `month: run` fan, and the whole-run window injected when the page names
no time, and it is also what the cache rule below is checked against; the other members
simply read the same literal window. A list does not combine with `then:` (more than one
member), and an empty list is refused.

A depth-selected page cannot include a variable with no vertical axis (`ssh`, `mld`,
`co2_flux`, `PH`, `pCO2`) -- that raises a clear schema error naming the offending
variable.

### `compare:` -- model vs. observations

Keys map onto `osk.compare(test=, reference=, variables=, select=, aggregate=, ...)`
(`test` defaults to `defaults.test`). Sets that end up mixing plot families are drawn
as one figure per family, suffixed in the PNG filename; an empty result (every pair
`skip_missing`-ed) is a skipped page, not a fatal error.

An optional `draw:` key names which `ComparisonSet` method draws the page, with `plot:`
forwarded as that method's keyword arguments in place of `.plot()`'s:

| `draw:` | draws |
|---|---|
| `plot` (default) | `ComparisonSet.plot()`, one figure per plot family (suffixed in the PNG name) |
| `metrics_series` | `ComparisonSet.metrics_series()`: metrics over the `times=` bins |
| `target` | `ComparisonSet.target()`: bias vs. centered RMSD |
| `taylor` | `ComparisonSet.taylor()`: correlation and variability |

Every `draw:` other than `plot` draws the whole set as **one figure** -- one PNG named
for the page, no per-family split. An unknown value, `draw:` on a non-`compare:` page,
and `contours:` beside any `draw:` but `plot` are all schema errors (caught by `--list`).
`--list` marks a page drawn another way with `+ draw: <method>`, and `manifest.json`
records `draw` in the page's record (`"plot"` for a compare page that leaves it out). The
page-size pinning under `pdf: true` (see Page size) applies the same way.

```yaml
# metrics over monthly bins -- RMSE and bias against the reference, one line per variable
- title: "Monthly RMSE and bias"
  compare:
    reference: [glorys_monthly]
    variables: [temperature, salinity]
    times: {resample: 1MS, reduce: mean}
  draw: metrics_series
  plot: {metrics: [rmse, bias]}

# target diagram with a drift arrow through each run's monthly points
- title: "Monthly target diagram"
  compare:
    reference: [glorys_monthly]
    variables: [temperature, salinity]
    times: {resample: 1MS, reduce: mean}
  draw: target
  plot: {arrows: true, color_by: variable}
```

Two `times:` forms matter for these pages. `times: {resample: 1YS, reduce: pool}` scores
each year from *every* step as samples (the page then draws the year's mean map), where
`reduce: mean` scores the year's mean map; and against a monthly-climatology reference
(WOA23 monthly, a roms-tools boundary climatology, Holte and Talley) every `times:` bin,
a `draw: metrics_series` page and an `over: time` page pair each test step with its own
calendar month automatically -- no `reference: {month: "{month.month}"}` pin needed (an
explicit `month` select or aggregate still means what it says). See `compare()`'s `times=`.

**`max_rows:` -- a long grid becomes several figures.** One row per comparison is
readable up to a few rows; six variables at two statistics is twelve rows, each too small
to read. A compare page that would draw a row grid (maps, sections, `time_depth` stations
or `over:` skill maps: not the profile or series families, which overlay everything on
one panel) with more than `plot: {max_rows: N}` rows (default `4`) is drawn as several
figures of at most `N` rows each, in the set's own order, so a variable's two statistics
stay together when `N` is a multiple of two. The first keeps the page's PNG name; the
next ones get `_part2`, `_part3`, ... (after the family suffix, if there is one) and a
title ending `(cont.)` -- an explicit `plot: {title: ...}` is extended, else the figure's
own suptitle, else the figure is titled just `(cont.)`. `max_rows: null` (or `0`) draws
one figure whatever the number of rows. It applies to `draw: plot` pages only (anywhere
else a page's own `max_rows` is a schema error, and one from `defaults.plot` is ignored),
and is not forwarded to `.plot()`. Everything else sees the whole set once: the metrics
records, `--self-check`, the all-NaN drop and the summary pool are computed before it is
cut, so cutting changes neither the CSV nor the problems list. A `contours:` overlay is
cut with the fill.

```yaml
- title: "Seasonal mean and variance vs WOA23"
  compare: {reference: [woa23_temperature_monthly, ...], variables: [temperature, salinity, nitrate], aggregate: [...]}
  plot: {max_rows: 2}          # one variable (two statistics) per figure
```

**`over:` -- skill maps.** `over: time` (or `Z`) is forwarded to `compare(over=)`: each
metric is computed cell by cell along that axis and drawn as a map instead of a test /
reference / difference row (see [skill_maps.md](skill_maps.md)). A page with it needs no
other key; the figure is one row of metric maps per comparison, so `max_rows` cuts it too.
`--self-check`, the all-NaN skip and the metrics CSV treat it like any compare page (each
record carries `over`).

```yaml
- title: "Chlorophyll skill against satellite"
  compare: {reference: [modis_chl_daily], variables: [chlorophyll], select: {test: {depth: surface}, reference: {}}, over: time}
```

**`selects:`.** `compare(selects={label: select})`: one comparison per labelled select,
each merged over the page's `select:` (stacked transects, say). A page whose `selects:`
name a `transect` needs a flat `select:` (even `select: {depth: [0, 50, 100]}`): without
one the page would get the run window as a `{test, reference}` pair-spec, which `compare()`
refuses beside a transect, so it is an error naming the page. A flat select is used for
both lanes as written and gets no run window. Other `selects:` get the window as usual, and
an entry that names a `time` of its own must be closed against the run to be cached (see
Caching).

An entry may carry its own aggregate as `{select: {...}, aggregate: {...}}` (only those two
keys), which replaces the page's `aggregate:` for that entry as a whole (no per-axis
merge). That lets a transect and lon-mean box slabs stack in one figure:

```yaml
- title: "{variable} sections vs WOA23"
  for_each: {variable: [temperature, salinity]}
  compare:
    reference: ["woa23_{variable}_annual"]
    variables: ["{variable}"]
    select: {depth: [5, 50, 100, 200, 500]}      # the shared base: depths
    selects:
      Equator:
        select: {transect: {lat: 0, lon: {min: 143, max: 267}}}
        aggregate: {time: mean}
      180-160W mean:
        select: {lon: {min: 180, max: 200}, lat: {min: -30, max: 30}}
        aggregate: {time: mean, lon: mean}
  contours: true
```

Rows run entry by entry, then variable by variable, each titled by its entry's label (or
`"<variable> <label>"` when an entry gives several comparisons). A `contours:` overlay on
such a page needs as many members as the fill: the same number of variables.

An optional `notes:` string is display text for a later [`osk.table()`](tables.md): it
goes to `compare(notes=)`, is stored on each comparison and in its metrics record, and
does not enter the cache key. A non-string `notes:` is refused when the suite is expanded.

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

### `XY:` and `TS:` -- one variable against another

```yaml
- title: "T-S diagrams -- six Pacific regions"
  TS:                              # XY: is the same, plus x: and y:
    members:
      ROMS: {}                     # source = defaults.test, variables = [y, x], whole-run window
      WOA23: {source: [woa23_temperature_annual, woa23_salinity_annual]}
      GLORYS12:
        source: glorys_climatology_timeseries
        aggregate: {time: mean, lon: mean, lat: mean}
    regions:                       # one panel per region, in this order
      North West Pacific: {lon: {min: 155.24, max: 156.33}, lat: {min: 20.51, max: 21.60}}
      Subpolar Gyre: {lon: {min: 184.59, max: 185.73}, lat: {min: 47.76, max: 48.59}}
    at_center: [WOA23]
  plot:
    ncols: 3
    colors: {ROMS: black, WOA23: "tab:red", GLORYS12: "tab:blue"}
    annotations: {North West Pacific: {STSW: [34.8, 28.0], NPIW: [34.4, 8.0]}}
```

Plots one variable against another for each data source (a **member**), one panel per
region: `XY:` builds `osk.XY(members, x=, y=, regions=, at_center=)` and `TS:` builds
`osk.TS(members, regions=, at_center=)`, which is `XY` with `x="salinity"`,
`y="temperature"` (and potential-density lines drawn by default). `plot:` is forwarded
to `.plot()`; the options (`ncols`, `colors`, `color_by`, `annotations`, `density`, ...)
are in the XY section of `docs/plot_styling_reference.md`. Unlike `field:` and `compare:`,
the whole page is one figure, one PNG, whatever the number of regions.

- **`members:`** (required, at least one) maps a label to one `osk.field()` call. The
  label is the legend entry and the key for `colors:` and `at_center:`, so `label:` is
  refused in a member, and so is `cache:` (the suite's own to set, as on `field:`). Every
  other key passes to `osk.field()`: `source:`, `variables:`, `select:`, `aggregate:`,
  `qc:`, `detide:`, ... A member with nothing to say is written `{}`.
- **`x:` and `y:`** name the two variables, as `osk.field` does (a nickname, or a
  `{calculate: ...}`/`{sum: [...]}` spec). `XY:` requires both; `TS:` refuses them, since
  it is the shorthand for `x: salinity`, `y: temperature`.
- **Defaults.** A member's `source:` defaults to `defaults.test` (an error if there is
  none), and its `variables:` to `[y, x]` -- `[temperature, salinity]` for `TS:`. A
  `source:` list with a `variables:` list builds every pairing, and the pairs that do not
  exist (WOA's temperature file has no salinity) are dropped with a warning; `XY` then
  matches each member's `x` and `y` fields itself.
- **The test member is windowed like a `field:` page.** The member whose `source:` is
  `defaults.test` gets the same treatment as a `field:` page's `select:` (see
  `time: latest` and Caching): no time key means the run's whole recorded span as a
  literal `{"min", "max"}` window, and `time: latest` becomes the last step. Every other
  member is passed through exactly as written, with no run window, because a climatology
  or a different run has a time axis of its own -- its time is yours to select.
- **`window: run` gives a non-test member the run's window.** A reanalysis compared
  against the run has to cover the run's dates, which change from run to run, so they
  cannot be written into the YAML: `GLORYS12: {source: glorys_my_daily_timeseries,
  window: run}` gets the same literal `{"min", "max"}` `select.time` the test member
  does. It needs `defaults.test`, is refused if the member also names a `time:` in its
  `select:` (give one or the other), and is a no-op on the test member itself. The only
  value is `run`.
- **`regions:`** maps a region name to a horizontal box, `{lon: {min, max}, lat: {min,
  max}}` (or a single point). The name is the panel title. Each region *replaces* its
  members' horizontal `select:`, so write `ROMS: {}`, not the box, and the same members
  serve every region; their `aggregate:` stays, which is how GLORYS12 above is averaged
  over each box. Without `regions:`, each member must already carry its own box or point
  in `select:` -- a member that would load a whole 4-D domain is refused. The boxes are
  checked at `--list` but kept as written in `manifest.json`.
- **`at_center:`** lists members to sample at the nearest grid cell to each region's
  centre instead of selecting the box (and any `lon`/`lat` mean in their `aggregate:` is
  dropped). It exists for coarse sources: a 1-degree grid such as WOA23's can have no
  cell centre inside a ~100 km box, and a box with no cells in it leaves that member
  out of the panel (with a warning).
  Each name must be a member, and it needs `regions:`.
- **`for_each:` and `plot:` work as on any page, `then:` does not** (there is no one
  field to chain on): a schema error. `{month.window}` and friends template into a
  member's `select:` as usual.

Several pages usually want the same regions (or the same `annotations:`). Define the
dict once under `defaults:` and refer to it by an exact placeholder, which keeps the
dict's type (see `{placeholder}` templating) -- each page gets its own copy:

```yaml
defaults:
  test: all_the_rest
  pacific: {North West Pacific: {lon: {min: 155.24, max: 156.33}, lat: {min: 20.51, max: 21.60}}}

pages:
  - title: "T-S"
    TS: {members: {ROMS: {}}, regions: "{pacific}"}
  - title: "Nitrate-phosphate"
    XY: {members: {ROMS: {}}, x: phosphate, y: nitrate, regions: "{pacific}"}
```

`--list` prints these as `[TS     ]`/`[XY     ]`. A page is `(cache)` only if every
member is: see Caching.

### `summary:` -- pool the compare pages above

```yaml
summary: {kind: both, color_by: variable}
```

Pools the metric records of the `compare:` pages that produced results and draws them
as overview figures. Skipped when no compare page succeeded. Besides the keywords
`osk.summary` takes (`color_by`, `marker_by`, `legend_style`, `title`, ...), a summary
page has three of its own:

- **`kind:`** -- one kind or a list, each drawn as its own figure, in the order given:
  `both` (Taylor and target side by side), `taylor`, `target`, `portrait` (the scorecard
  heatmap) or `table` (the pooled metrics as a table). Default: `[both, table]` -- the
  diagram a summary always drew, and the table under it. A page with one kind keeps the
  plain `figures/NN_<title>.png`; with several, each PNG is suffixed with its kind
  (`..._table.png`, `..._target.png`). A duplicate or unknown kind is a schema error.
- **`scope:`** -- `all` (default) pools every compare page drawn so far; `section`
  pools only those since the last `section:` divider (everything so far, if there is
  none). A suite can then close each part with its own summary.
- **`max_rows:`** -- for `table`: comparisons per table figure (default 25, about a
  page); a longer table continues in further figures titled `... (cont.)` (PNGs
  suffixed `_part2`, ...), with the header repeated. A figure also ends early when its
  rows would no longer fit a Letter page -- long labels make tall rows -- so
  `max_rows: 0` means "as many as fit a page". The diagram kinds ignore it, and `table` ignores `color_by` and
  the other diagram options (only `title:`, default the page title, is shared).

```yaml
pages:
  - title: "Part 1 -- temperature and salinity"
    section: "Physics against WOA23."
  - title: "Temperature profiles"
    compare: {reference: [woa23_temperature_annual], variables: [temperature], select: {depth: [0, 50, 200]}}
  - title: "Salinity profiles"
    compare: {reference: [woa23_salinity_annual], variables: [salinity], select: {depth: [0, 50, 200]}}
  - title: "Physics skill"
    summary: {scope: section, kind: [table, target], color_by: variable}
```

The table has one row per pooled comparison (named as its Taylor/target point is, with
depth and time window in a Notes column) and the columns bias, RMSE, CRMSD,
correlation, standard-deviation ratio and n, in the same number formatting as
`osk.table()` (three significant figures for dimensional metrics, two decimals for
dimensionless ones, whole numbers for counts), whose layout it is. Units come from each
record, spelled as the plots do: in the column headers when every row shares them, or
in a `Units` column when they differ. Summary pages are static (matplotlib) like every
suite page, so the table has no interactive form.

Each page that draws a `table` also writes its numbers, unformatted and with a `units`
column, to `metrics/<suite name>_<page title slug>.csv` beside the whole-suite
`metrics/<suite name>.csv` (two summary pages with the same title overwrite each other's
file). Records are pooled once per compare page, however many figures the page was cut
into with `plot: {max_rows}`, and a comparison dropped as all-NaN is not pooled at all
-- so a `--self-check` or partial run's summary counts exactly the comparisons its
compare pages drew.

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
string wrapped beneath it. It heads the PDF page of the figure(s) that follow it when they
fit under it (see Page size) and has a US Letter page of its own, with its text centred,
when the next figure does not. It lets a long
report have chapters. `section: ""` is a title-only divider; a YAML block scalar (`>-`
folds the lines into one paragraph, `|` keeps your own line breaks, and a long line wraps
either way) is the natural way to write longer notes. `{placeholder}`s in the title and
the text resolve against `defaults:` -- there is no `for_each:` to bind anything else --
and a literal brace needs the usual `{{ }}`.

- **A divider is only drawn when a page after it draws.** It is buffered and written
  into the PDF with the next figure: above it on the same page when it fits under it. A
  later `section:` page replaces one still waiting, so a section whose pages all skipped leaves no empty divider behind, and
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
- **`for_each:`, `then:`, `caption:` and a non-empty `plot:` are schema errors** on a
  section page, naming it: there is no data to fan out, chain, plot or caption. A bare `section:` with nothing
  after it is YAML for null, which is no kind at all and so is refused too -- write
  `section: ""`.

### Captions

`caption: "..."` on a `field:`, `compare:`, `summary:`, `XY:` or `TS:` page puts a few
lines of text under its figure in `report.pdf` -- small type, wrapped to the page -- and
records it as `"caption"` in the page's `manifest.json` entry. It is templated like
`title:` (`for_each` placeholders, `defaults`), and must come out as text. The PNG is the
bare figure. A page drawn as several figures (a plot-family split, or `max_rows`) carries
its caption under the first. A `section:` page takes no caption (its text is its
`section:` string).

```yaml
- title: "Nitrate, {depth.label}"
  field: {variables: [nitrate], select: {depth: "{depth}"}}
  for_each: {depth: [surface, 100]}
  caption: "Nitrate at {depth.label}, mean over the run. Gray cells are land."
```

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

A page's `title` and `caption`, and every string inside its `field:`/`compare:`/`summary:`/
`XY:`/`TS:`/`plot:` dict, may contain placeholders resolved against `defaults` plus whatever `for_each`
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

## `contours:` -- contour lines over a `field:`/`compare:` page

`contours:` is the suite-YAML form of `.plot(contours=<object>)` (see "Contour
overlays" in `plot_styling_reference.md`): lines drawn over the filled figure. It sits
beside `plot:`, whose own `contour_levels:`/`contour_kwargs:` set the levels and
styling as usual. Two forms:

```yaml
# true: contour the plotted object itself (zero-lines on a velocity section)
- title: "Zonal velocity, 180-160W"
  compare:
    reference: [cravatte_u_179e_160w]
    variables: [east_velocity]
    select: {lon: {min: 180, max: 200}, lat: {min: -20, max: 20}, depth: [0, 10, 20, 50, 100, 200, 300, 500]}
    aggregate: {test: {time: mean, lon: mean}, reference: {lon: mean}}
  contours: true
  plot: {contour_levels: [0]}

# a mapping: a second object, built from the page's own kwargs with these keys
# replaced (a shallow merge) -- isotherms over phosphate
- title: "Phosphate with isotherms"
  compare: {reference: [woa23_phosphate_annual], variables: [phosphate], select: {...}, aggregate: {...}}
  contours: {reference: [woa23_temperature_annual], variables: [temperature]}
  plot: {contour_levels: 8}
```

A `field:` page takes the same two forms (`contours: {variable: temperature}`;
`variables:` is accepted as the same key). Everything else follows `plot(contours=)`:
over a comparison the test lines go on the test panel and the reference lines on the
reference panel, pairing is by position, and a grid mismatch is refused.

- The mapping's values are templated (`{placeholder}`, `for_each:`) like the page's
  own. The second object gets the same `test`/`source` default, the same run window
  and `time: latest` resolution, and the same `skip_missing`/`cache` settings as the
  page; the page is cached only when both reads are.
- On a `compare:` page each plot-family figure is paired with the contour comparisons
  of the *same family*. A family with no contour counterpart, or a different count,
  is an error naming the page, not a silent mis-pairing.
- `contours:` is refused on `summary:`/`section:`/`XY:`/`TS:` pages and on a `field:`
  page with a `then:` chain (schema/`--list` errors), and when `plot:` also sets
  `contours:`. A value other than `true`/`false` or a non-empty mapping is an error.
- `contours:` is also refused beside a `draw:` other than `plot` (see `compare:`).
- `--list` marks such a page with `+ contours`; `manifest.json` records the resolved
  contour kwargs under the page's own `contours:`.

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

An `XY:`/`TS:` page decides per member, since each member is its own `osk.field()` call.
The member that reads `defaults.test` follows the rule above (an injected whole-run window
or `time: latest` tracks the run; an explicit window must be closed); every other member --
a climatology, say -- has no run window to track and caches whenever the suite does. The
page is marked `(cache)` in `--list` only when all its members are.

A `field:` page whose `source:` is a list is checked against one source's time axis --
`defaults.test` if listed, else the first entry (see `field:` above); the other members
read the same literal window and are not separately vouched for, so a listed run that is
itself still growing inside a closed window is not tracked.

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

By default a suite has one stable report directory, `<output_dir>/<name>/`, refreshed in
place by every run. Just before a run starts writing, it removes the previous run's own
artifacts -- `report.pdf`, `figures/`, `metrics/`, `refs/`, `run_info/`, `manifest.json`,
`run.log`, `suite.yaml` -- and nothing else: any other file or folder you keep in that
directory is left alone (a suite's `cache_dir:` may even be the same directory). Set
`timestamped_dir: true` for the older behavior, a new directory per run that never
overwrites an earlier report; use it if two runs of the same suite might overlap.

```
<output_dir>/<name>/      (timestamped_dir: true -> <name>_<test>_<t0>_to_<t1>_<run-time>_<id>/)
    report.pdf                (unless pdf: false, or no page drew at all)
    figures/NN_<slug(title)>[_<family>][_partN].png      (figures only: section: and the cover have none)
    metrics/<name>.csv (+ .txt)
    suite.yaml                 -- byte-identical copy of the input
    manifest.json
    run.log                    -- everything printed to the terminal during the run
    refs/<ref name>            -- copy of each refresh: reference, exactly as drawn
    run_info/                  -- copies of the C-Star blueprint/workplan the cover page used
<output_dir>/latest.txt        -- path of the newest report dir
<output_dir>/latest            -- symlink to it
```

A run that is not the whole suite as written -- `--smoke`, `--self-check` or `--only` (see
Guardrails under CLI) -- writes to `<name>-smoke/`, `<name>-selfcheck/`, `<name>-only/` (or
their combination, e.g. `<name>-smoke-selfcheck/`) instead, and leaves `latest.txt`/`latest`
alone, so it can never replace the real report.

### Cover page

Page 1 of `report.pdf` says which simulation the report is about. It is titled with the
simulation's name -- `run_info.name`, else the C-Star workplan's name (with the step),
else the blueprint's name, else `defaults.test`, else the suite's `name` -- and lists the
report (suite, generated time, version), then the workplan, the simulation (grid,
period, ...), forcing (boundary and surface split into physics, BGC, and other; rivers;
tides), code (the exact repositories and commits), the parameters, and any `run_info:
notes`. Nothing needs configuring when the suite has a `refresh:` block: the cover finds
the run from that block's output glob -- the run folder is the one holding
`work/directives.yaml`, the blueprint is the newest resolved blueprint in its `work/`,
and the workplan is the one two levels up whose step's `working_dir` matches. Name them
with `run_info.blueprint`/`workplan` instead when that does not apply (relative paths
resolve against the suite file, like `catalog_search_paths:`); `cover: false` turns the
page off. Files that are missing or fail to parse only warn and shorten the page. The
blueprint and workplan used are copied to `run_info/`, and the same content is the
manifest's `"run_info"` key.

The cover is not a figure: it has no PNG and takes no `NN_` number, it is not counted in
the exit code or the "N page(s) drawn" line, and, like a `section:` divider, it is only
written if some page after it drew -- a run where nothing draws still writes no PDF.

### Page size

`report.pdf`'s pages are all a fixed US Letter portrait (8.5x11in), so the report
paginates and prints like a document regardless of what any one page draws.

**Packing.** Consecutive figures share a page when they fit: each is stacked under the one
before it, in the order drawn, until the next would not fit in what is left (greedy; a
figure never moves to an earlier page). A figure wider than the page is scaled down to
its width (keeping its aspect, side margins kept) and then packed like any other. A
`section:` divider is put at the top of the page its first figure lands on; when that
figure does not fit under it, it is scaled down to fit, centred, as long as that keeps
it at 0.7 or more of its size (`MIN_FIT_SCALE`) -- otherwise the divider gets a page of
its own, as before. A figure's caption counts towards the room it takes, so it never
runs off the page or into the next figure. A figure alone on its page (nothing packed
with it, and not scaled) is written exactly as before -- its ink at the top, horizontally
centred, text kept as vector -- unless its page has room to enlarge it (below). A page of
several figures, one whose figure was scaled, or one that is enlarged, is assembled from
the figures' own PNGs (`PNG_DPI` 200), so its text is raster rather
than vector. The PNGs are unaffected, one per figure. Only a figure that is still too
tall for a page after scaling to its width (see below) gets a page of its own.

**Filling.** After a page's figures are chosen (packing does not change for this), they
are all scaled up by one factor so the page's stack -- divider header, figures, captions
and the gaps between them -- fills the usable page height, with no figure wider than the
usable width (8.0in); narrower figures stay centred and proportions are kept. The factor
is capped at 1.6 (`MAX_FILL_SCALE`), so a small figure is not blown up into a poster,
and a page that would gain under 10% (`MIN_FILL_GAIN`) is left alone and stays vector.
Captions and the header keep their type size; only figures grow. An enlarged page is
raster by necessity (matplotlib cannot scale a vector page without reflowing it), so such
a figure is rendered a second time at 200 dpi times its factor (up to 320 dpi), keeping
its text as sharp on paper as its PNG; the cost is that extra render and a larger PDF.

Because pages are laid out this way, `size:`/`zoom:`/`figsize:` in a page's `plot:` (or a `summary:`
page's own kwargs) are ignored while `pdf: true` -- pinned to the `"page"` canvas
instead, with a warning naming the ignored kwarg (once per suite run, not once per
page). Set `pdf: false` to size figures freely; PNGs are still written, as tight crops
around each figure rather than letter pages. A figure whose own ink still doesn't fit
8.5x11 even after pinning (rare -- only an explicit `figsize:` outside this grammar
could do it) grows the page rather than clipping the figure, with its own warning.
A page holding only a `section:` divider is an exact letter page too.

**Smaller PDFs: `pdf_images: jpeg`.** About 95% of a map report's bytes are its
rasterized map images, which matplotlib's PDF backend can only store losslessly (a packed
page's figures are rasterized whole, so the pass recompresses their text too).
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

With `timestamped_dir: true`, `t0`/`t1` are the test source's own first/last recorded
day; `<run-time>` is when the command was run, down to the second, plus a short random
suffix so two runs started in the same second still land in different directories.
`manifest.json` records the fully
expanded page list (every `time: latest`/`month: run`/window already resolved to a
literal value), each page's outcome, the `pdf_images` setting and whether it was applied,
the resolved `catalog_search_paths:` directories,
the effective `cache_dir` (the suite's own if set, otherwise wherever
`osk.cache.base_dir()` already pointed), and (when the suite has a `refresh:` block) a
`"refresh"` key, plus the cover page's content as `"run_info"` -- enough that a second
person with the same suite YAML and the same catalogs gets the identical report.

`refs/` only appears when the suite has a `refresh:` block. Each source's reference
(`refresh.sources[].ref`, a parquet directory or a `.json` file) is copied there
right after the report directory is created, before any page is drawn -- so it
matches the model state the report was actually drawn from, even though the shared
reference at `ref:` is overwritten by the *next* run's own refresh. `manifest.json`'s
`"refresh"` key records `{"catalog": ..., "rebuilt": ..., "sources": [{"name", "ref",
"snapshot"}]}`, where `snapshot` is the copy's path (`null` if no reference existed yet
to copy) and `rebuilt` is `false` for a `--no-refresh` run, which snapshots the
references as it found them. The
copy pins file paths and byte ranges into the model output, not file contents: a
restart that rewrites an output file in place still changes what the snapshot reads.
Open a snapshot directly with `ocean_skill.build.open_reference(path)` (one dask layer, and
no fsspec protocol scan, which can crash on a parquet reference).

`run.log` is a mirror of stdout/stderr for the run: a `== page i/n: <title> ==` header
and a `done in Ns`/`SKIPPED after Ns` line bracket each page, so a warning in between
is attributable to the page that raised it. It also carries the full traceback for
every skipped page and, if the run crashes outright, for the crash itself -- neither of
which the terminal shows. `--list` writes no report, so no `run.log` is created for it
-- but it is not write-free: a `refresh:` block still runs under `--list` (see CLI).
A Python-level tee catches everything this package or its warnings print; it does not
catch a C library writing straight to a file descriptor.

## Exit codes

A page is skipped -- never fatal to the rest of the report -- when its variable is
absent, its observational catalog entry isn't on this machine's search path
(`osk.catalog.search_paths()`), or the comparison itself raises; `run.log` has the
traceback. A page whose own source cannot be read while the suite is expanded (say a
`refresh.forcing` entry that failed to build) is skipped without being tried, and
`--list` already marks it `will be skipped: <why>`. A source with no time axis (a
monthly climatology) is fine: its pages simply get no run window. `main`'s exit code: `0` every page drew and nothing was flagged, `3` the run completed
with some pages skipped or flagged (a source that could not be read, an all-NaN
comparison, a `--strict` layout warning), `4` a page *failed* (its draw raised) or a
`--self-check` found a lane bug -- so a batch script can tell a crash or a real bug from a
benign skip -- `1` no page drew at all, `2` a schema or usage error (nothing was drawn).
See Guardrails under CLI for what is flagged.
`section:` divider pages are not counted either way -- they are neither "drawn" nor
"skipped" as far as these codes and the `N page(s) drawn, M skipped` line are concerned,
so a suite whose only data pages skipped is `1` however many dividers it has.

## CLI

```
ocean-skill-run SUITE.yaml [--list] [--no-refresh] [--smoke [N]] [--smoke-stride K]
                [--self-check] [--only GLOB] [--strict]
```

`--list` validates the suite, resolves `latest`/`month: run`/`for_each`, and prints the
expanded page list without drawing any page or writing a report (no report directory,
`run.log`, PNGs or manifest) -- a fast check on an edited suite, or on how many pages a
long run's `month: run` will produce, before committing to the full run. It is **not**
read-only, though: `--list` still runs the `refresh:` block first, so it rebuilds the
kerchunk reference and the catalog entries that block names (resolving `latest`/`month:
run` needs the refreshed time axis). Add `--no-refresh` for a dry check that writes nothing. `--list` also registers
`catalog_search_paths:` first, since resolving `latest`/`month: run` reads a source's time
axis.

`--no-refresh` skips the `refresh:` block, with or without `--list`, and reads the
catalog and references as they already are (`manifest.json` records `"rebuilt": false`).
It is how several suites that share a `refresh:` target run at once. Their refreshes
would otherwise rewrite the same reference and catalog, at the same time or while another
suite is still reading them. Start one suite normally and wait for its `latest step of
...` line, which prints once its refresh has saved the catalog. Then start the rest with
`--no-refresh`. The one refreshing has to cover every source the others read.

Everything that changes what a page shows is a YAML key, so the suite file alone is what
reproduces a report. The other flags only decide *how much* of it is drawn and how
carefully it is checked: they are the guardrails below, and all of them combine with each
other and with `--list` and `--no-refresh`.

### Guardrails: find the bug in minutes, not after the full run

A full run on an HPC can take hours, and a mistake in the suite or a bug in the code only
shows once a page reaches it. These flags make most of them show early.

**`--smoke [N]`** reads every model source named under `refresh.sources` (by `name`;
with no `refresh:` block, `defaults.test`) with only its **first and last N time steps** (default
2), plus the steps the pages **pin**: the nearest step (within 2 days, else none) to every date a
page names (a date in `select` time, an entry of a `times:` list, including a `{placeholder}` that
expands to one) and the first and last N steps inside every time window under 62 days (a
`{min, max}` window, a period such as `"2010-01"`). A window longer than that, such as the whole-run
`period`, adds nothing, and neither does a `times: {resample: ...}` (it bins whatever is kept), so a
suite whose pages name dates all through a long run still gets them from `--smoke 2`. The pins are
collected from a first, throwaway expansion of the suite made before the cut (so `month: run`
placeholders see the whole run); the run log says how many instants and windows it found and how
many steps of each source are kept. For another source (a reference run), the steps are chosen from
those up to the test source's last time. Observation and reference catalogs and `refresh.forcing`
entries are read whole.
Every `latest`/`month: run` lookup made while the suite is expanded sees the shortened
record too, so the pages expand and draw consistently, just over very little data. Write
the count right after the flag (`--smoke 3` or `--smoke=3`) and put the suite path first.

`--smoke` is also **test mode**: shortening the time axis alone saves little on a big suite (pages pin
most of their dates, every map reads and draws the whole grid, figures render at up to 320 dpi), and a
test run should take minutes, not as long as the real one. So it also:

- keeps only the **first and last entry of a page's `times:` list** (a 16-date seasonal page runs two
  dates) and of a `for_each: {month: ...}` fan; variables, depths and every other fan stay complete, since they
  exercise different code. The pins above are collected after this cut, so fewer steps are kept. The run's page list
  and log mark each such page, e.g. `(test: 2 of 16 dates)`, and the log's `smoke:` line counts them.
  A compare page that sets no `min_pairs` gets `min_pairs: 1`: the few steps kept cannot reach the default of 5,
  which would mask every cell of a per-cell skill map;
- reads every truncated model source on **every `--smoke-stride K`-th horizontal cell** (default 4, about
  1/16 of the cells to read and draw; `1` keeps the whole grid). The rho dims (`xi_rho`, `eta_rho`) take every
  Kth cell, the staggered `xi_u`/`eta_v` are cut so they stay one shorter than their rho dim (the velocity code
  relies on it), and the grid coordinates (lon/lat, `h`, mask, angle, cell area) are strided with the data. A dimension
  the stride would leave under 3 cells is kept whole, with a warning. The values are physically inconsistent
  (u/v no longer sit between the strided cells, cell areas describe the full cells), so this tests the code paths, not the
  numbers. Observations, climatologies, initial conditions and boundary forcing are read whole and regridded to
  the coarser model lane as usual;
- saves figures at **no more than 100 dpi** (PNGs, the PDF's rasterized maps, and the enlarged pages' renders).

Nothing changes without `--smoke`. The stride is part of the smoke-cache hash, so runs with different strides
never share results.

A smoke run never touches the real results:

- its report goes to `<output_dir>/<name>-smoke/` (not `<name>/`), and `latest.txt`/
  `latest` are left pointing at the last full report;
- its cached results (the `prepared`, `aligned` and `calculated` kinds) go to
  `<cache base dir>/smoke-cache/<hash>/` -- `cache_dir:` if the suite sets one, else the
  default -- because the cache key knows nothing about the truncation and a shortened result
  must never be served to a full run. `<hash>` is a short hash of everything the key cannot
  see: each truncated source's name, N, the stride and the time stamps actually kept, pinned ones included (so
  a run that has grown, or pages that now name other dates, keep different steps and start fresh), and the code version
  (the package version, the git `HEAD`, and, if the tree has uncommitted changes, a hash of
  them). A repeat run on the same steps with the same code **reuses the folder** (the run log
  says `smoke cache: <dir> (reused)` or `(new)`, and `manifest.json` `"modes"` records it),
  while any code change, edited file included, starts cold, so a fix is always exercised.
  Without a git checkout to name the code version every run starts cold. Only the newest
  few `<hash>/` folders are kept (`SMOKE_CACHE_KEEP`); older ones, unused for an hour, are
  deleted by the next smoke run. Regrid weights (keyed on the grids' own contents) and
  downloaded observation files stay shared, so a smoke run does not re-download WOA.
  Delete the folder any time.

**`--self-check`** draws every `compare:` page against itself: the reference is replaced
by the test source, and every `{test: ..., reference: ...}` pair-spec (`variables`,
`select`, `aggregate`, `qc`) is collapsed to its test side, so both lanes are the very
same recipe. Identical lanes must show zero bias, zero error and a correlation of 1, so
any comparison with `|bias|` or `rmse` above `1e-6` of the data's scale (the larger of
its mean, spread and 1), or `corr` below `1 - 1e-6`, is reported as a `self-check`
problem with the numbers (a NaN `bias` or `rmse` is a problem too -- identical lanes
always have both; a NaN `corr` alone is not, a constant field has none). It catches lane-asymmetry bugs -- one lane cropped, shifted or
masked differently from the other -- that otherwise look like a plausible model bias.
Pages are titled `[self-check] <title>`, are never cached, and are drawn into
`<output_dir>/<name>-selfcheck/` so the real report is never replaced; `field:`, `TS:`/
`XY:` and `summary:` pages run as written. A compare page with several `test` sources
cannot be paired with itself and is drawn as written (the log says so).

**`--only GLOB`** draws only the pages whose title matches the case-insensitive glob
(`fnmatch`: `*`, `?`, `[...]`), matched against the title as written (not the
`[self-check]` one). A `section:` divider is kept only if a matching page follows it. The
report goes to `<output_dir>/<name>-only/`, so a quick look at one page never replaces
the full report. A `summary:` page needs its compare pages: select them too, or it is
skipped. A glob that matches nothing is an error that lists the page titles.

**`--strict`** turns layout warnings into problems: any warning raised while drawing a
page that says the figure came out cramped -- "the maps are only ...in wide", "panels on
a canvas capped at ...", "the maps may be squeezed", matplotlib's "Tight layout not
applied" -- is reported as a `strict` problem for that page (the first message, and how
many more). Both drawing the figure and saving it are watched, since the layout warnings
matplotlib raises come at save time. The page still draws. Ordinary data warnings (thin samples, partial bins)
are left alone: they are in `run.log`, not failures.

**Problems and the summary.** However the pages are drawn, the end of the run prints (and
appends to `run.log`, and writes to `manifest.json` as `"problems"`) one line per
problem after the usual `N page(s) drawn, M skipped` line:

```
2 problem(s):
  [skipped] Nitrate at 200 m: all-NaN: reference has no data at depth 200 (...) (12.4s, peak 3,102 MB)
  [strict] Salinity grid: the maps are only 1.20in wide: ... (31.0s, peak 4,870 MB)
```

`kind` is `skipped` (not drawn, or part of it left out, for a reason the code knew: a
source that could not be read, or a comparison whose test or reference data is entirely
NaN once aligned -- a 200 m comparison on a 75 m shelf -- which is left out with a
warning rather than drawn as an empty plot, the page skipped only if none remains),
`failed` (the draw raised), `self-check` or `strict`. The time is what drawing the page
took and the peak memory is the process's resident size while it ran, sampled every
0.1 s (so a short spike can be missed; blank without `psutil`, since the fallback would
be a lifetime peak for the whole process). Both are also recorded per page in
`manifest.json` as `"seconds"` and `"peak_mb"`. The exit code is nonzero whenever there
is a problem: `3` for skipped or `strict` problems if anything drew (even when every page
did), `4` for any `failed` or `self-check` problem, `1` if nothing drew. A
smoke/self-check/only/strict run is recorded in `manifest.json` under `"modes"`.

**Recommended workflow.** On the HPC, run the suite quickly first with
`ocean-skill-run SUITE.yaml --smoke --self-check --strict`: a few steps of data, every
compare page checked against itself, cramped layouts caught -- minutes, in
`<name>-smoke-selfcheck/`. Fix what it lists (and use `--only` to redraw just the pages you
changed), then start the full run, which reuses the real cache and refreshes
`<name>/` as usual.
