# Suites: a whole report from one YAML

A suite is a YAML file that lists the pages of a report. Running it draws every page and
writes one PDF, a PNG per figure, a metrics CSV and a `manifest.json` recording exactly
what was drawn. No Python is needed to run one.

## The four reports

The package ships four suites, as templates in `ocean_skill/suites/templates/`:

| Template | What it shows | Set up with |
|---|---|---|
| `input-check` | What the model was given: grid, initial conditions vs WOA23/GLODAP/satellite chlorophyll (maps and sections), each open boundary vs WOA23, surface forcing, salinity restoring, river inputs, and an integrity table of every input file. Reads no model output. | `ocean-skill-suite new` |
| `mid-run-check` | A run still in progress: run health (impossible values, drift, steps, plankton spin-up), domain-mean time series, extreme values, the latest restart, and the latest month vs satellite chlorophyll, Holte & Talley MLD, WOA23 (maps and sections) and GLODAPv2. | `ocean-skill-suite new` |
| `final-report` | A finished run: the figures of the UCLA PACMED validation report (WOA23 tracers, MLD, T-S, sections, SSH, zonal currents, iron, chlorophyll, oxygen minimum, carbon). | by hand |
| `model-comparison` | Two runs against each other: input differences, the start of the run, seasonal differences, per-cell skill maps, skill through time, each run vs WOA23. | by hand |

Each section of a report ends with a metrics table and Taylor and target diagrams. Find
the templates folder with

```bash
python -c "import ocean_skill.suites as w, pathlib; print(pathlib.Path(w.__file__).parent / 'templates')"
```

## Running a report

```bash
ocean-skill-suite new input-check --run /path/to/cstar/run --name my_run \
    --output-dir /path/to/diagnostics -o input-check.yaml
ocean-skill-run input-check.yaml --smoke 2 --self-check --strict
ocean-skill-run input-check.yaml
```

1. **`ocean-skill-suite new`** writes the suite from your C-Star run folder (see below).
2. **The test run** draws every page from a few time steps on a coarsened grid, in
   minutes, and lists anything that fails. Its report goes to `<name>-smoke-selfcheck/`,
   so it never replaces the real one.
3. **The full run** writes `/path/to/diagnostics/input-check/report.pdf`. Rerunning it
   refreshes that folder in place, and reuses the cache for whatever has not changed.

`mid-run-check` works the same way. Run it again as the model writes more output: the
`refresh:` block picks up the new restarts first.

**Redraw one part.** Name a section (or page) title:

```bash
ocean-skill-run input-check.yaml --only "River inputs"
```

The part gets its own folder, here `input-check-only-river-inputs/`, and shares the
cache with the full run. `ocean-skill-run input-check.yaml --list` prints the section
and page titles.

**On a cluster** (Anvil and the like), run on a compute node, not the login node. In an
interactive session:

```bash
salloc -A <allocation> -p shared -c 16 --mem=32G -t 2:00:00
```

or as a batch job:

```bash
#!/bin/bash
#SBATCH -A <allocation>
#SBATCH -p shared
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 2:00:00
ocean-skill-run input-check.yaml
```

**Before the first run:**

- The observation catalogs (WOA23, GLODAPv2, Holte & Talley, ...) must be in a directory
  under `catalog_search_paths:`. `ocean-skill-suite new` points it at the shared Anvil
  directory; use `--shared-catalogs DIR` elsewhere.
- Satellite chlorophyll, DUACS SSH and GLORYS read Copernicus Marine: run
  `copernicusmarine login` once.
- A catalog entry that points at a URL rather than a local file downloads it the first
  time it is read, so that run needs network access.
- `pdf_images: jpeg` needs Ghostscript (`gs`) on `PATH`; without it the PDF stays
  lossless, with a warning.

## Setting up a suite

### `ocean-skill-suite new` (input-check, mid-run-check)

```bash
ocean-skill-suite new mid-run-check --run RUN --name NAME --output-dir DIR [-o FILE]
```

It copies the template and fills in its setup block (between the `# >>> setup` and
`# <<< setup` comments) from the run folder; the pages and comments are kept as the
template has them.

| Option | |
|---|---|
| `--run DIR` | the C-Star run folder, the one holding `tasks/`; repeat it for a run restarted in pieces |
| `--name` | the model's name in the catalog and on the cover page |
| `--output-dir DIR` | where reports, references, catalogs and the cache go |
| `-o FILE` | the suite to write (default `<template>.yaml`); `--force` replaces it |
| `--inputs DIR` | input-check: the `input_datasets` folder, if not under `--run` |
| `--files GLOB ...`, `--grid FILE` | the restarts or the grid, if they are not found |
| `--catalog-dir DIR` | this suite's own catalog (default `<output-dir>/catalogs`) |
| `--shared-catalogs DIR` | the observation catalogs (default: the Anvil directory) |

It finds the grid, the restarts (mid-run-check), the initial conditions, boundary,
surface and river files (input-check, from the first task's `input/input_datasets/`),
the C-Star workplan and blueprint for the cover page, and the run's start date. It also
picks two section lines through the middle of the domain (`defaults.transects`): the
equator (or the middle latitude) and the middle longitude, or, on a rotated grid, the
middle grid row and column. Each line keeps its longest stretch over the ocean.

Anything it was unsure of is marked `# TODO check` in the file. It then loads the suite
and checks every path; problems are printed and the exit status is 1. Read the setup
block before the first run, and change `transects:` if you want other lines.

### By hand (final-report, model-comparison)

Copy the template and edit its setup block: `output_dir`, `cache_dir`,
`catalog_search_paths`, `refresh:` (the model files to catalog, see below) and the
`defaults:` (the test source, the run's dates, the transects). These two templates are
written for the PACMED runs, so their regions and dates are the first things to change
for another domain.

To change a shipped report itself, edit its file in `ocean_skill/suites/templates/`;
there is no second copy. Keep the setup markers so `ocean-skill-suite new` still works.

## Checking a suite

| Flag | |
|---|---|
| `--list` | expand the suite and print its pages without drawing. It still runs `refresh:`; add `--no-refresh` for a check that writes nothing. |
| `--smoke [N]` | test mode: read only the first and last N model time steps (default 2) plus the dates the pages name, every 4th grid cell (`--smoke-stride K`), and draw at low resolution. Report in `<name>-smoke/`, cache in its own folder. |
| `--self-check` | draw every comparison against itself; any nonzero bias or error, or correlation below 1, is a bug in how the two sides were read. Report in `<name>-selfcheck/`. |
| `--strict` | treat cramped-layout warnings ("the maps are only ... wide") as problems. |
| `--only GLOB` | draw only the sections or pages whose title matches (repeatable). Report in `<name>-only-<slug>/`. |
| `--no-refresh` | skip `refresh:`, e.g. for a second suite running beside one that already refreshed the same catalog. |

The run ends with a list of every page that was skipped or failed, with the reason, its
time and peak memory (also in `run.log` and `manifest.json`). Exit codes: `0` every page
drew, `3` some pages skipped or flagged, `4` a page failed or the self-check found a
bug, `1` nothing drew, `2` a schema or usage error.

## Output

```
<output_dir>/<name>/
    report.pdf          cover page, then the pages, packed onto US Letter pages
    figures/NN_<title>.png
    metrics/<name>.csv  one row per comparison (and one CSV per table page)
    manifest.json       the expanded pages, each page's outcome, time and peak memory
    run.log             everything printed, with tracebacks of skipped pages
    suite.yaml          a copy of the suite
    refs/, run_info/    the model reference and C-Star files the report was drawn from
<output_dir>/latest     symlink to the newest full report
```

Each run replaces those files and leaves anything else in the folder alone
(`timestamped_dir: true` makes a new folder per run instead). The cover page names the
simulation and lists its workplan, grid, forcing, code versions and parameters, found
from the C-Star run folder (or `run_info:`).

---

# Writing suites

The rest of this page is the reference for editing a suite or writing a new one.

## The suite file

```yaml
name: mid-run-check              # the report folder's name
output_dir: /path/to/diagnostics
cache_dir: /path/to/diagnostics  # optional: where cached results accumulate
catalog_search_paths: [/anvil/projects/.../catalogs]   # observation catalogs; relative = to this file
timestamped_dir: false           # true: a new report folder per run
pdf: true                        # false: PNGs only
pdf_images: jpeg                 # lossless (default) | jpeg: smaller PDF, via Ghostscript
cover: true
cache: true                      # false: never reuse cached results
run_info:                        # optional: what the cover page shows, if not found
  name: PACMED v2
  workplan: runs/workplan.yaml
  blueprint: runs/bp_resolved.yaml
  notes: {Purpose: "spin-up check"}

refresh: {...}                   # catalog the model files before each run (below)

defaults:                        # merged into every page; a page's own keys win
  test: my_run                   # the model source (one name)
  depths: [surface, 100]         # any value pages refer to as {depths}
  plot: {robust: true}           # merged under every page's plot:

pages:
  - title: "..."
    field: {...}                 # exactly one of the page kinds below
    for_each: {...}              # optional: one page per combination
    plot: {...}                  # optional: keyword arguments to the plot method
    caption: "..."               # optional: text under the figure in the PDF
```

## `refresh:` -- cataloging the model files

```yaml
refresh:
  catalog: catalogs/my_run.yaml
  sources:                                 # model output
    - name: my_run
      files: [/run1/tasks/*/output/output_rst.*.nc, /run2/tasks/*/output/output_rst.*.nc]
      grid: /run1/tasks/first/input/input_datasets/grid.nc
      ref: refs/my_run.parquet
      keep: latest-per-file                # restarts: each file's newest record
      subchunk: {s_rho: 1, s_w: 1, eta_rho: 74}   # smaller stored chunks: depth matching reads less
  forcing:                                 # roms-tools input files
    - {name: bry, kind: boundary, files: input/*_boundary-physics_*.nc, grid: input/grid.nc}
    - {name: sfc, kind: surface, files: input/*_surface-physics_*.nc, grid: input/grid.nc}
    - {name: ic, kind: initial_conditions, files: input/*_initial_conditions_*.nc, grid: input/grid.nc}
    - {name: rivers, kind: river, files: input/*_river*.nc, grid: input/grid.nc}
```

Before each run (and `--list`), the model files are cataloged so the suite sees the
latest output. With a list of globs the files are joined in time order, and where two
give the same time the later glob wins (a rerun segment replaces the original).
`kind: boundary` makes one entry per side (`bry_south`, ...); `sides:` limits them,
`climatology: true` reads a 12-month file, and `strip: [west, south]` adds `bry_strip`,
the sides joined into one line. `kind: initial_conditions` reads like a restart, so it
can be compared at a depth like model output. `variables:` limits what is kept.

## Page kinds

### `field:` -- model only

```yaml
- title: "Surface temperature and salinity, latest"
  field:
    variables: [temperature, salinity]
    select: {depth: surface, time: latest}
```

Keys are `osk.field()`'s (`select`, `aggregate`, `source`, ...); `source` defaults to
`defaults.test`. Several variables draw one map each in one figure. A list of sources
overlays them (series, profiles).

### `compare:` -- model vs. observations or another run

```yaml
- title: "Nitrate vs WOA23 -- {month.name} {month.year}"
  for_each: {month: {run: {last: 1}}}
  compare:
    reference: [woa23_nitrate_monthly]
    variables: [nitrate]
    depths: [surface, 200]
    select: {test: {time: "{month.window}"}, reference: {month: "{month.month}"}}
    aggregate: {test: {time: mean}}
```

Keys are `osk.compare()`'s; `test` defaults to `defaults.test`. A `{test: ..., reference:
...}` pair gives each side its own `select`/`aggregate`/`variables`. The figure is test |
reference | difference, one row per comparison. Other keys:

- **`draw:`** `plot` (default), `metrics_series` (metrics over the `times:` bins),
  `target` or `taylor`.
- **`plot: {max_rows: N}`** (default 4): a taller grid is split into several figures.
- **`plot: {panels: difference}`**: difference maps only, one shared colorbar.
- **`over: time`**: per-cell skill maps (bias, RMSE, correlation, ...) instead of maps
  of the fields; see [skill_maps.md](skill_maps.md).
- **`times:`**: score per time bin, e.g. `{resample: 1YS, reduce: pool}` for one score
  per year from every step. A monthly climatology reference is matched to each step's
  calendar month.
- **`selects: {label: select}`**: one comparison per labelled select, stacked in one
  figure; the templates' section pages use `selects: "{transects}"`. An entry may be
  `{select: ..., aggregate: ...}` to carry its own aggregate.
- **`aggregate:` as a list** of whole specs draws one row per statistic, e.g. the mean
  and the seasonal variance: `[{time: mean}, {time: [{groupby: month, reduce: mean}, var]}]`.

A comparison that is all NaN once aligned (200 m on a shelf) is left out with a warning.

### `summary:` -- metrics of the comparisons above

```yaml
- title: "Nutrients -- metrics"
  summary: {scope: section, kind: [table, both], color_by: variable}
```

`kind:` is one or a list of `table`, `both` (Taylor and target side by side), `taylor`,
`target`, `portrait`. `scope: section` pools the comparisons since the last `section:`
page, `all` (default) every one so far. A table page also writes its numbers to
`metrics/<name>_<title>.csv`.

### `XY:` and `TS:` -- one variable against another

```yaml
- title: "T-S diagrams"
  TS:
    members:
      ROMS: {}                                   # defaults.test, whole run
      WOA23: {source: [woa23_temperature_annual, woa23_salinity_annual]}
    regions: "{ts_boxes}"                        # {name: {lon: {min, max}, lat: {min, max}}}
    at_center: [WOA23]                           # coarse sources: nearest cell to the box center
```

One panel per region, one color per member. `XY:` takes `x:` and `y:` as well; `TS:` is
salinity against temperature with density contours. A member other than the model can get
the run's dates with `window: run`.

### `health:`, `rivers:` and `inputs:` -- diagnostics without observations

```yaml
- title: "Run health -- summary"
  health: {time: "{period}", draw: summary}        # summary | integrity | inventory | plankton
- title: "River inputs -- map"
  rivers: {source: rivers, time: "{forcing_time}", draw: map}   # map | histograms | sample | extremes
- title: "Surface forcing -- integrity"
  inputs: {sources: ["{surface}"], time: "{forcing_time}", max_steps: 200}
```

- `health:` scans the model output (`osk.health`): values out of range, negative
  concentrations, NaNs, all-zero fields, copies, steps between restarts, drift in domain
  totals, plankton pools. Job-segment joins come from the C-Star workplan.
- `rivers:` draws the river forcing (`osk.rivers`): where rivers enter, how big they are,
  a sample of rivers through time, the most extreme concentrations.
- `inputs:` tabulates every variable of the input files (`osk.input_health`): range and
  where it occurs, values out of range, flags such as `all zero` or `copy of X`.

Pages with the same source and time share one scan. The tables are also written as CSVs.

### `section:` -- a divider

```yaml
- title: "Nutrients vs WOA23"
  section: >-
    The latest month against the WOA23 climatology of the same month.
```

A heading and text, at the top of the PDF page of the figures after it. It is dropped when
no page after it draws. `--only` with a section title draws the whole section.

### Captions and figure titles

`caption:` puts text under a figure in the PDF. `plot: {title: "..."}` replaces the
figure's own title (`plot: {title: page}` uses the page title).

## `for_each` and `{placeholders}`

```yaml
for_each:
  variable: [temperature, salinity]   # a list
  depth: "{depths}"                   # a list from defaults:
  month: run                          # every month of the run; {run: {last: N}} for the last N
```

One page per combination. Titles and every string in a page can use `{placeholders}` from
`defaults:` and `for_each`. A string that is exactly one placeholder (`"{depths}"`) keeps
the value's type, so a list stays a list. `month` has `.year`, `.month`, `.mm`, `.name` and
`.window` (the month as a `{min, max}` time range); `depth` has `.label` (`100 m`); a
mapping default reads by attribute (`{period.min}`).

## Time

- **`time: latest`** is the newest step of `defaults.test`.
- **A page with no time** gets the run's whole span written in as a fixed window, so
  `manifest.json` replays the same figures later.
- Figure titles show the time a page reduced to (`surface · 2013-12-31`) without it being
  in the page title.

## `then:` and `contours:`

`then:` runs a short method chain on a `field:` page; the one in use finds an extreme and
draws the time series at its location:

```yaml
- title: "Alkalinity minimum -- time series"
  field: {variables: [alkalinity], select: {depth: surface, time: latest}}
  then:
    - extremum: min
    - series: {variables: [dissolved_inorganic_carbon, temperature, salinity]}
```

`contours: true` draws contour lines of the page's own field over it; a mapping draws
another variable's (`contours: {variables: [temperature]}` over phosphate). `plot:
{contour_levels: ...}` sets the levels.

## Caching

Pages reuse prepared fields from the disk cache (see [caching.md](caching.md)) when their
time window cannot change as the run grows: `time: latest`, the whole run, or a window
that has already closed. A month still being written is recomputed every run. `cache:
false` turns caching off for the whole suite, e.g. after model output was rewritten in
place. `--list` marks each page `(cache)` or `(no-cache ...)`.

## PDF layout

Pages are US Letter. Consecutive figures share a page when they fit, and a page's figures
are enlarged to fill it (up to 1.6x). A section heading goes at the top of the page with
the next figure. `size:`/`figsize:` in `plot:` are ignored while `pdf: true`.
