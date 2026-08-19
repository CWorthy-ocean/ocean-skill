# Searching catalogs

`osk.find()` searches every discovered source across every catalog and returns
matching source names — which are exactly what `osk.read()`, `osk.describe()` and
`compare(reference=…, test=…)` take.

```python
import ocean_skill as osk

osk.find()                                  # every source: 139
osk.find(name="papa")                       # 41
osk.find(climatology="January")             # 6
```

Every filter you pass is ANDed; every one you omit is skipped.

## Filters

| Filter | Matches |
|---|---|
| `text` | free text over everything — all terms must match |
| `name` | source name **or its catalog's** — substring, or glob if wildcarded |
| `catalog` | catalog name only |
| `variable` | a variable, in any spelling the vocabulary knows |
| `featureType` | `grid`, `timeSeries`, `profile`, `trajectory`, … |
| `bbox` | `(lon_min, lat_min, lon_max, lat_max)` — tests **overlap** |
| `time` | `(start, end)` — tests **overlap** |
| `climatology` | `True`/`False`, or a period such as `"January"` |

## Free text

The catch-all, and usually the fastest way to narrow a list you're staring at.
Every whitespace-separated term must appear somewhere in the source's name, its
catalog's name, or any of its metadata:

```python
osk.find(name="modis", variable="chlorophyll")   # 14 — too many
osk.find(name="modis", variable="chlorophyll", text="jan")   # 1
osk.find(text="modis chl jan")                   # 1  ['modis_chl_climatology_jan']
osk.find(text="modis chl daily")                 # 2
osk.find(text=["modis", "jan"])                  # a list works too
```

Terms are **ANDed** — a second word narrows, which is what you want when a
structured search returned too much.

Free text earns its place because catalogs describe the same idea differently. A
monthly climatology is `climatology: True` + `climatology_period: month01` in WOA,
`period: monthly_climatology` in MODIS, and `jan` in the MODIS source name. Free
text spans all three, so you needn't know which convention a catalog used:

```python
osk.find(text="woa nitrate january")   # ['woa23_nitrate_month01']
osk.find(text="modis chl january")     # ['modis_chl_climatology_jan']
```

Month names match whichever spelling the catalog used — `january`, `jan` and
`month01` are interchangeable, in both directions. Bare numerals are matched
literally rather than expanded, since `"1"` as a substring would match most of an
index.

Use free text to explore; use the structured filters below when you want a
guarantee about *what* matched.

## By name

```python
osk.find(name="papa")                       # 41  substring, case-insensitive
osk.find(name="GOM")                        #  3
osk.find(name="woa23_nitrate_month*")       # 12  glob
osk.find(catalog="OOI*")                    # 41  catalog only
```

`name` deliberately matches the **catalog** too. OOI's sources are opaque dataset
ids (`ooi-gp02hypm-rim01-02-ctdmog039`) inside a catalog called *OOI Station Papa*,
so a `name="papa"` that searched only source names would find nothing for the one
word you actually know. Use `catalog=` when you want to match only the catalog.

## By variable and type

```python
osk.find(variable="nitrate")                # 15
osk.find(featureType="timeSeries")          # 41
```

`variable` takes anything the [vocabulary](../ocean_skill/vocabulary.py) knows — a
short key, a canonical CF standard_name, or any alias, in any case. All of these
return the same 15 sources:

```python
osk.find(variable="nitrate")
osk.find(variable="NITRATE")
osk.find(variable="mole_concentration_of_nitrate_in_sea_water")      # ROMS, GLODAP
osk.find(variable="moles_of_nitrate_per_unit_mass_in_sea_water")     # WOA
```

That equivalence is the point rather than a convenience. Products disagree about
which CF name nitrate has: WOA declares it **per unit mass**, while ROMS/MARBL and
GLODAP declare it **per unit volume**. Searching one exact standard_name finds two
sources and silently misses the thirteen you would actually want to compare
against — so the filter matches any equivalent spelling, exactly as `compare()`
does when pairing a variable with the sources that carry it.

Matching is against what a source *declares*, which comes from probing it at
catalog-build time. A catalog built with `probe=False` declares no variables, so it
will not match — see [caching and probing](caching.md) for the trade-off.

## By space and time

```python
GULF = (-98, 18, -80, 31)
osk.find(bbox=GULF)                         # 98
osk.find(time=("2012-01-01", "2012-02-01")) # 98
```

Both test **overlap, not containment**, so a global climatology matches a regional
box — which is usually what you want, but it does mean `bbox` will not thin a list
of global products:

```python
osk.find(name="woa23")                      # 78
osk.find(name="woa23", bbox=GULF)           # 78  — WOA is global, so it overlaps
```

A source that declares **no** extent is **kept**, not dropped. "Unknown" is not
"outside", and excluding un-probed entries would quietly hide exactly the sources a
search is meant to surface. Probe a catalog if you want its entries filtered on
geography or time.

Climatologies are the one exception, and for the opposite reason: their absence of a
date range is *known*, not unknown. A January climatology is a calendar slot, so
returning it for a July 2012 query would be wrong. `time=` skips them; reach for
`climatology=` instead.

## By climatology

A climatology has no meaningful date range — it represents a calendar slot, not a
period — so it carries no `time_coverage_*` and a `time=` filter cannot reach it.
`climatology=` answers the question `time=` cannot:

```python
osk.find(climatology=True)                  # 79  WOA + GLODAP
osk.find(climatology=False)                 # 60  everything with real dates
osk.find(climatology="January")             #  6
osk.find(climatology="annual")              #  6
```

Catalogs record the period as `month01`; you can type whatever you'd say out loud —
all of these are the same request:

```python
osk.find(climatology="January")
osk.find(climatology="january")
osk.find(climatology="jan")
osk.find(climatology="01")
osk.find(climatology="month01")
```

Anything unrecognized falls back to a substring test, so `"annual"` works and future
period names need no new entry.

## Combining filters

```python
osk.find(variable="nitrate", bbox=GULF)                # 15
osk.find(variable="nitrate", climatology=True)         # 14  just the climatologies
osk.find(name="nitrate", bbox=GULF)                    # 13  by *name*, not variable
osk.find(climatology="January", name="nitrate")        #  1  ['woa23_nitrate_month01']
osk.find(name="papa", bbox=(-160, 40, -140, 55))       # 41  North Pacific
osk.find(name="papa", bbox=GULF)                       #  0  wrong ocean
```

That last pair is the clean demonstration: same name filter, two different boxes.

## Seeing where they are

Any result maps in one line — from the catalog metadata alone, so nothing is
opened or read:

```python
osk.find(variable="nitrate").map()          # where every match is, on one map
osk.map_datasets()                          # everything discoverable
osk.map_datasets(catalog="OOI*")            # one catalog
osk.find(name="papa").map(renderer="holoviews")  # interactive
```

Moorings, profiles and tracks draw as markers at their declared position; gridded
datasets draw as dashed extent rectangles — both coloured by `featureType`, with a
legend. The interactive version opens on a web basemap (pass `tiles=None` to work
offline, or any other `geoviews.tile_sources` name) and hovering any marker or
extent box reads that dataset's record: name, catalog, featureType, variables, time
coverage, cadence, resolution, depth, institution, title.

A source that declares no extent cannot be placed, so it is skipped with one
warning naming it — the mapping counterpart of the "unknown is kept" rule above.
Probe the catalog to fill extents in.

## One gotcha: glob vs substring

A wildcard makes `name` a **whole-name glob**, shell-style. Without one it is a
plain substring:

```python
osk.find(name="month01")     #  6  substring — matches anywhere in the name
osk.find(name="month0*")     #  0  glob, anchored — no name *starts* with "month0"
osk.find(name="*month0*")    # 54  what you probably meant
```

This trips people up (it caught out this project's own test suite first time).
The rule: **if you type a `*`, you are writing a glob and it must match the whole
name.** If you just want "contains", leave the wildcards out.

## Related

- `osk.catalogs` — the discovered catalogs
- `osk.describe(name)` — full metadata for one source or catalog
- `osk.read(name)` — open a source
