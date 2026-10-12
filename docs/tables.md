# Metrics tables: `osk.table()`

`osk.table()` / `ComparisonSet.table()` turn comparisons into a publishable table of
*selected* metrics: bias, RMSE, R² and so on, one row per variable, station or depth,
with optional seasonal sub-rows, several values per cell, a Type column and Notes. It is
"portrait without the picture" -- the same metric records, the same pivot and the same
number formatting as [`portrait`](plot_styling_reference.md#the-portrait-family-metrics-scoreboard)
-- and it pools its inputs exactly the way `osk.summary` does: a `Comparison`, any nesting
of lists and tuples of comparisons and `ComparisonSet`s, or a `{name: comparisons}` dict
(the names become a `group` field you can use as `rows=` or `cols=`). That pooling is the
point: one table across all of them, row names matching the Taylor/target point names for
the same pool.

```python
osk.table([ctd, velocity, alk], metrics=("bias", "rmse", "r2"))
cs.table(metrics=("bias", "rmse"), rows="variable", cols="test", save="skill.tex")
```

## Parameters

| Parameter | Default | Effect |
|---|---|---|
| `metrics` | `bias, rmse, crmsd, corr, sigma_ratio, n` | which metrics to show, in column order (`r2` = corr² is available too). A name no record carries raises, naming what they do carry |
| `rows` | `"label"` | a metric-record field (`"variable"`, `"depth"`, `"test"`, ...) or `"label"`, the pooled/dict/`labels=` name |
| `cols` | `None` | `None`: the columns are the metrics. A field name (`"test"`): the portrait grid of every metric side by side, columns `(metric, test)`. Two comparisons landing in one cell raise, with the portrait wording |
| `split` | `None` | `{"bias": "time"}` breaks those metrics out by that field's levels (the seasons of a `times={"seasons": [...]}` comparison, say); the other metrics read the whole-record comparison of the same row. A row with no split-level records gets one line holding its whole-record value. One split field per table |
| `layout` | `"rows"` | how `split=` is drawn: `"rows"` makes sub-rows (the level gets its own column just left of the first split metric, and the unsplit columns are merged across the sub-rows); `"columns"` makes `(metric, level)` column groups plus an "all" column for rows without levels, so nothing needs merging and every format (including CSV) reads the same |
| `join` | `None` | `{"r2": "depth"}` puts the records that differ only in that field into one cell, separated by " / " in level order. A missing level shows "–" in its slot |
| `join_labels` | from `pretty_level` | `{"depth": ["surface", "column"]}` renames the joined levels; the header names them, `R² (surface / column)` |
| `headers` | built in | display names, `{"r2": "R²", "label": "Variable"}` (the defaults are R², RMSE, CRMSD, σ ratio, ...); also renames the Notes and User notes columns |
| `units` | `True` | `bias (°C)` in the header for one variable; a mixed-variable table puts units in a per-row column instead, since they differ by row. Dimensionless metrics stay unitless |
| `round` | `True` | the `portrait` rule per metric: two decimal places for a dimensionless metric, an integer for a count, three significant figures otherwise. `False` writes full precision. `to_dataframe()` is always the unformatted numbers |
| `info` | `None` | `{"SST": {"Type": "CTD"}}`: text columns keyed by row label, supplied by you |
| `notes` | `None` | `{"SST": "ice-free months only"}`: the "User notes" column, keyed by row label. Overrides a note set on the comparison itself (below) |
| `auto_notes` | `True` | the "Notes" column, generated from the record: depth label, time window, "station average" / "sample average" when the row is pooled, demeaned, detided. `False` drops it, a callable `record -> str` replaces the generator, a `{label: text}` dict overrides single rows |
| `save` | `None` | writes the table, format by suffix (below); the table is returned either way |

Index levels are labeled with the same `pretty_level` the summary diagrams use, so a
depth band reads "0-100 m", not a dict. Thin samples warn exactly as `metrics()` does.

### Notes on a comparison

`compare(..., notes="...")` stores display text on the comparison and in its metrics
record, so the note travels with it into any later table. It is not part of the cache
key: changing a note never recomputes anything. In a [suite](suites.md#compare----model-vs-observations)
it is the `notes:` key of a `compare:` page. A `table(notes=...)` entry overrides it.

## Worked example

Variables down the side. Sea surface temperature and salinity get their bias split by
season (MAM, JJA, SON); R² and RMSE are joined as surface / water column in one cell;
velocity and alkalinity have a whole-record row only. The Type column comes from `info=`:

```python
whole = osk.compare(...)                                  # the whole record
seasonal = osk.compare(..., times={"seasons": ["MAM", "JJA", "SON"]})
osk.table(
    [whole, seasonal, velocity, alk],
    metrics=("r2", "rmse", "bias"),
    split={"bias": "time"},                               # bias by season
    join={"r2": "depth", "rmse": "depth"},                # surface / column in one cell
    join_labels={"depth": ["surface", "column"]},
    headers={"label": "Variable", "bias": "Bias"},
    info={"SST": {"Type": "CTD"}, "Salinity": {"Type": "CTD"},
          "Velocity east": {"Type": "ADCP"}, "Alk": {"Type": "Sample"}},
    notes={"SST": "ice-free months only"},
)
```

`to_markdown()` of that table (from the test suite's hand-built comparisons, so the
numbers are made up):

| Variable | R² (surface / column) | RMSE (surface / column) | Season | Bias | Units | Type | User notes |
| :--- | :---: | :---: | :--- | :---: | :--- | :--- | :--- |
| SST | 0.90 / 0.85 | 0.5 / 0.4 | MAM | 0.21 | °C | CTD | ice-free months only |
|  |  |  | JJA | 0.32 |  |  |  |
|  |  |  | SON | -0.13 |  |  |  |
| Salinity | 0.64 / – | 0.11 / – | MAM | 0.03 |  | CTD |  |
|  |  |  | JJA | 0.04 |  |  |  |
| Velocity east | 0.36 / – | 0.07 / – |  | 0.01 | m/s | ADCP |  |
| Alk | 0.49 / – | 9 / – |  | -2 | umol/kg | Sample |  |

A row with no seasonal records (velocity, Alk) gets one line holding its whole-record
value, not empty season sub-rows; a joined level a row lacks shows "–". The Units column
appears because the rows' units differ (with one shared unit it goes in the headers
instead). There is no Notes column here because everything the auto notes would say
(depth, season) is already in the headers. In the notebook and in LaTeX the R², RMSE,
Units, Type and notes cells of SST are one merged cell spanning its three lines; the
Markdown writer leaves the cells below blank instead.

## Output formats

`table()` returns a `MetricsTable`. All its writers read one internal layout (cells,
rowspans, header levels), so the merge logic is written once.

| Use | What you get |
|---|---|
| shown in a notebook | a styled HTML table with real merged (`rowspan`) cells |
| `str(t)` / `print(t)` | aligned plain text for the terminal |
| `t.to_markdown()` | pipe table; a merged cell is written once and the cells below it are left blank |
| `t.to_latex()` | a `booktabs` table with `\multirow` for the merged cells; the preamble needs `\usepackage{booktabs,multirow}` |
| `t.to_html()` | the same HTML as the notebook, as a string |
| `t.to_dataframe()` | the underlying DataFrame (MultiIndex rows) for further pandas work |
| `save="x.tex"` / `.md` / `.html` / `.csv` | one of the above, picked by the file suffix; the CSV is the tidy long form, one line per number |

`layout="columns"` needs no merged cells at all, so it reads the same in every format.
