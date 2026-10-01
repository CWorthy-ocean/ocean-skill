# Shared vocabulary and units

Two data files, each usable by any package with no ocean-skill code:

- `vocabulary.yaml` -- variable spellings, in cf-xarray's `custom_criteria` shape.
- `units.txt` -- ocean units and the seawater-density context, as pint definitions on
  top of cf-xarray's unit registry (see [units.txt](#unitstxt) below).

## vocabulary.yaml

`vocabulary.yaml` maps a short name for a concept (`nitrate`, `temperature`, `mld`) to
the real-world spellings of it -- a variable name, or a `standard_name` attribute.
It is written in [cf-xarray](https://cf-xarray.readthedocs.io)'s own
`custom_criteria` shape, `{key: {attribute: regex}}`, so **any** package can use it
as-is, with no ocean-skill code.

This folder is data only, laid out to move to its own shared home unchanged (so that
ROMS-Tools, ocean-skill, xroms and others can depend on one copy). Product names --
`thetao`, `so`, `t_an`, ... -- are deliberately absent: a product's own variable name
belongs in that product's catalog entry (its `standard_names`), not in a vocabulary
every package shares.

## Using it without ocean-skill

```python
import cf_xarray, yaml

criteria = yaml.safe_load(open("vocabulary.yaml"))

with cf_xarray.set_options(custom_criteria=criteria):
    no3 = ds.cf["nitrate"]          # finds NO3, nitrate, mole_concentration_of_nitrate...
    mld = ds.cf["mld"]              # finds any mixed-layer-depth definition
```

cf-pandas takes the same dict, matching the regexes against column names:

```python
import cf_pandas

vocab = cf_pandas.Vocab()
vocab.vocab.update(criteria)                     # or pass `criteria` directly:
cf_pandas.match_criteria_key(["NO3", "x"], "nitrate", criteria)    # -> ["NO3"]

with cf_pandas.set_options(custom_criteria=criteria):
    df.cf["nitrate"]
```

## File rules

One top-level key per concept. Each value has exactly two entries, both single-quoted
regexes of the form `^(?i:...)$` (anchored at both ends, case-insensitive -- the
anchoring is what keeps `Fe` from matching `felix`, and a `..._qc_agg` flag companion
from standing in for its data variable, since cf-xarray matches with `re.match`):

- `name`: `^(?i:ALT1|ALT2|...)$`, matched against a variable's name. The alternatives,
  in order, are the key, the canonical standard_name, each alias (every one
  `re.escape`d, so it is a literal), then each pattern. **Every pattern is wrapped
  `(?:...)`**, even one with nothing in it a regex treats specially (`(?:doxy)`), so a
  pattern can always be told from an escaped alias; it is otherwise written as given.
  Duplicates are dropped (a few keys equal their own standard_name).
- `standard_name`: `^(?i:CANONICAL)$`, matched against a variable's `standard_name`
  attribute -- one escaped literal.

A **broad** entry -- one that other entries are a specific kind of, today `mld`, with
the definition-specific `mld_by_*` -- additionally lists every narrower entry's
alternatives in `name` and its canonical name in `standard_name`
(`^(?i:OWN|NARROWER1|NARROWER2|...)$`), so plain `ds.cf["mld"]` finds any definition
while `ds.cf["mld_by_sigma_t"]` never finds the generic name or a sibling.

Nothing else is in the file: no `aliases:`/`patterns:` lists, no `broader:` key. Those
are recovered from the two regexes as below. Keep the entry's design notes as YAML
comments above it -- they record why an alias is, or is not, there.

## How ocean-skill reads it

`ocean_skill/vocabulary.py` parses the file at import into the in-memory dict its API
is built on (`{key: {"standard_name", "aliases", "patterns", "broader"}}`; the last
three only when present). Editing the dict at runtime (`register`, `add_alias`,
`add_pattern`) never writes the file.

- The canonical standard_name is the sole alternative of `standard_name`. For a broad
  entry it is the alternative that is not another key's canonical one; the others are
  its narrower entries, each of which gets `broader: <the broad key>`.
- From `name`'s alternatives, drop the key, the canonical name and (broad entries)
  every narrower entry's alternatives. Of what remains, a part that is wrapped
  `(?:...)` is a pattern (unwrapped); any other part is an alias (unescaped), and must
  satisfy `re.escape(unescaped) == part`.
- So a key must not also be listed as its own alias: it is always matched already, and
  the file has nowhere to say it twice.

## Writing patterns

An alias is one exact spelling; a pattern recognizes a whole *family* of spellings that
listing would make tedious (`Temperature_CTD`, `CTD_Temperature`, `ctd_temp`, ...). It
is matched against the **whole** name, never a substring, so a pattern must be as narrow
as an alias is exact:

- enumerate the decorations (`(?:sea_water_)?temp(?:erature)?_ctd`); never an
  open-ended `.*` tail;
- never a `qc`, `flag` or `qartod` token -- a QC companion is the QC layer's business,
  not the vocabulary's;
- a name that two entries' patterns both match is ambiguous, and ocean-skill refuses
  it (warns, passes the name through) rather than guess. Narrow the patterns instead;
- this is the same idea as cf-pandas' `guess_regex`, deliberately narrower: bare
  `month` matching a time regex is exactly the looseness to avoid.

## units.txt

Ocean units on top of cf-xarray's unit registry (`cf_xarray.units.units`), in plain
pint definition syntax. cf-xarray already covers PSU, degC/Celsius, `%` and
UDUNITS-style exponents (`m2 s-2`), so the file adds only what it lacks: `equivalent`/`eq`
(alkalinity), two more Celsius spellings, `oxygen_ml_per_l` (1000/22.392 mmol/m3, the
UNESCO/WOCE molar volume of O2), and the `seawater` context. Load it once into
cf-xarray's registry -- a second load logs pint's "Redefining ..." messages, so check
first (ocean-skill skips it when `oxygen_ml_per_l` and the context are already there):

```python
import cf_xarray.units

ureg = cf_xarray.units.units
if "oxygen_ml_per_l" not in ureg:
    ureg.load_definitions("units.txt")
```

The `seawater` context converts between per-mass (`umol/kg`) and per-volume
(`mmol/m3`) concentrations through a density parameter, `rho` in kg/m3 (default 1025):

```python
with ureg.context("seawater", rho=1030):
    ureg.Quantity(1, "umol/kg").to("mmol/m^3")
```

## Notes

- ocean-skill registers only the `name` regexes with cf-xarray (plus, for a broad entry,
  the narrower `standard_name`s), one key per spelling, and deliberately not each
  entry's own `standard_name`: WOA's companion variables (`n_dd`, `n_se`, ...) carry the
  same `standard_name` as the data variable, so matching on the attribute reports them
  as a second match. Plain cf-xarray loading the file matches on both.
- ocean-skill registers into cf-xarray's global `custom_criteria` by *merging* with
  what is already there, never replacing it.
- `tests/test_vocabulary_file.py` pins these rules: the file parses losslessly, and
  plain cf-xarray and cf-pandas (with no ocean-skill imported) resolve names through it.
