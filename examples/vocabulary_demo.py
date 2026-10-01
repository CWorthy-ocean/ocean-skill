"""How ocean_skill.vocabulary resolves variable names -- and how to extend it.

Self-contained: builds tiny synthetic datasets shaped like real catalog entries
rather than reading anything over the network. Shows:

1. Any spelling registered for a concept resolves to the same variable, even when
   two real catalogs disagree on the exact CF standard_name (a real mismatch this
   demo reproduces, not a hypothetical -- see the ``chlorophyll`` entry below).
2. ``add_alias()`` -- teach an *existing* concept one more real-world spelling,
   live, no restart needed.
3. ``add_pattern()`` -- teach an *existing* concept a whole family of spellings via
   a narrow, anchored regex, when an enumerated alias list would be tedious.
4. ``register()`` -- add a wholly new concept from scratch, live.
5. ``register(..., broader=)`` -- file a new *specific kind of* an existing concept
   under it (mixed layer depth by sigma_theta, by temperature, ...): asking for the
   broad name is satisfied by any of them, asking for a specific one never by another.
6. How to make an addition permanent: edit ``ocean_skill/vocab/vocabulary.yaml``
   (the shared file ``VOCABULARY`` is loaded from) instead of calling these at runtime.

Run:  python examples/vocabulary_demo.py
"""

import warnings

import numpy as np
import xarray as xr

from ocean_skill import vocabulary
from ocean_skill.colormaps import cmaps_for
from ocean_skill.units import find_variable
from ocean_skill.vars import lookup


def _tiny_dataset(varname: str) -> xr.Dataset:
    """Build a minimal 3x3 lon/lat dataset carrying one variable, named ``varname``."""
    return xr.Dataset(
        {varname: (("lat", "lon"), np.linspace(0.1, 0.9, 9).reshape(3, 3))},
        coords={"lat": [10.0, 11.0, 12.0], "lon": [200.0, 201.0, 202.0]},
    )


print("=" * 70)
print("1. One concept, several real-world spellings -- all resolve the same way")
print("=" * 70)

# ocean_skill/catalogs/modis_aqua.yaml standardizes MODIS's chlor_a to this spelling -- missing
# "_a_" -- which is NOT the canonical CF name ocean_skill uses everywhere else
# (mass_concentration_of_chlorophyll_a_in_sea_water, in vars.py/colormaps.py). A real
# find_variable(modis_ds, "chlorophyll") call found nothing until this alias was
# added to VOCABULARY["chlorophyll"] -- this reproduces exactly that.
modis_shaped = _tiny_dataset("mass_concentration_of_chlorophyll_in_sea_water")
# ocean_skill/catalogs/ooi_papa.yaml's profiler-mounted fluorometer -- a different instrument,
# same physical quantity, its own naming convention.
ooi_profiler_shaped = _tiny_dataset(
    "mass_concentration_of_chlorophyll_a_in_sea_water_profiler_depth_enabled"
)

for label, ds in [("MODIS", modis_shaped), ("OOI Papa profiler", ooi_profiler_shaped)]:
    da = find_variable(ds, "chlorophyll")  # same short key both times
    print(f"{label:20s}: 'chlorophyll' -> {da.name if da is not None else None}")

print()
print("Any of these also work in place of the short key -- same result.")
print("Capitalization never matters, on input or in the dataset's own names:")
for spelling in (
    "chlorophyll",
    "Chlorophyll",  # case is ignored
    "CHLOROPHYLL",
    "mass_concentration_of_chlorophyll_a_in_sea_water",  # canonical
    "mass_concentration_of_chlorophyll_in_sea_water",  # MODIS's spelling
):
    print(f"  resolve_name({spelling!r}) -> {vocabulary.resolve_name(spelling)!r}")

# ...and a dataset that shouts its variable names is matched just the same
shouty = _tiny_dataset("MASS_CONCENTRATION_OF_CHLOROPHYLL_A_IN_SEA_WATER")
print(f"  dataset var {'MASS_..._SEA_WATER'!r:22s} <- 'chlorophyll' ->", end=" ")
print(find_variable(shouty, "chlorophyll").name)

print()
print("=" * 70)
print("2. add_alias(): teach an EXISTING concept a new spelling, live")
print("=" * 70)

# Say a new obs product spells chlorophyll "chlor_a" (a common shorthand this
# vocabulary doesn't know about yet).
chlor_a_shaped = _tiny_dataset("chlor_a")
print("before add_alias:", find_variable(chlor_a_shaped, "chlorophyll"))  # None

vocabulary.add_alias("chlorophyll", "chlor_a")

after = find_variable(chlor_a_shaped, "chlorophyll")
print("after  add_alias:", after.name if after is not None else None)
# It's remembered by everything else that resolves variable names too, not just
# find_variable -- same VOCABULARY, same resolve_name() underneath both:
print("vars.lookup('chlor_a').units      ->", lookup("chlor_a").units)
print("colormaps.cmaps_for('chlor_a')    -> (matches canonical chlorophyll's map)")
cmaps_for("chlor_a")  # no error -- resolves before looking up xcmocean's tables

print()
print("=" * 70)
print("3. add_pattern(): teach an EXISTING concept a whole FAMILY of spellings")
print("=" * 70)

# Argo/OceanSITES write dissolved oxygen as OXY_UMOLKG (a per-mass unit baked
# into the name) -- not worth an enumerated alias for every unit variant, so a
# narrow, anchored pattern recognizes the whole family instead. Fullmatch,
# case-insensitive, and refused for anything merely containing it (see the
# vocabulary module docstring's "patterns" bullet).
oxy_shaped = _tiny_dataset("OXY_UMOLKG")
print("before add_pattern:", find_variable(oxy_shaped, "oxygen"))  # None

vocabulary.add_pattern("oxygen", "oxy_umolkg")

after_pattern = find_variable(oxy_shaped, "oxygen")
print("after  add_pattern:", after_pattern.name if after_pattern is not None else None)

print()
print("=" * 70)
print("4. register(): add a wholly NEW concept from scratch, live")
print("=" * 70)

vocabulary.register(
    "ph",
    "sea_water_ph_reported_on_total_scale",
    aliases=["ph_total", "PH_TOT"],
)
ph_shaped = _tiny_dataset("PH_TOT")
found = find_variable(ph_shaped, "ph")
print("find_variable(ph_shaped, 'ph') ->", found.name if found is not None else None)

print()
print("=" * 70)
print("5. broader=: a concept that is one specific KIND of another")
print("=" * 70)

# Mixed layer depth has one generic CF name and four definition-specific ones -- CF
# names only the variable that defines the base of the layer, never the threshold.
# Each specific entry in VOCABULARY says "broader": "mld"; narrower_names() and
# covers() read that. It is a one-way relation, so it is not an alias.
print("narrower_names('mld'):")
for standard_name in vocabulary.narrower_names("mld"):
    print("  ", standard_name)

print()
print("covers(requested, declared) -- does a source declaring `declared` satisfy a")
print("request for `requested`?")
for requested, declared in [
    ("mld", "mld_by_sigma_theta"),  # a broad request, a specific source: yes
    ("mld_by_sigma_theta", "mld"),  # the generic name has not said which: no
    ("mld_by_sigma_theta", "mld_by_sigma_t"),  # siblings never cover each other: no
]:
    answer = vocabulary.covers(requested, declared)
    print(f"  covers({requested!r}, {declared!r}) -> {answer}")

# register(..., broader=) files a NEW definition under the broad name, live. The
# standard_name is made up here -- a stand-in for a definition of your own.
vocabulary.register("mld_by_shear", "my_mld_defined_by_shear", broader="mld")
print()
print("after register('mld_by_shear', ..., broader='mld'):")
print("  covers('mld', 'mld_by_shear') ->", vocabulary.covers("mld", "mld_by_shear"))
print("  covers('mld_by_shear', 'mld') ->", vocabulary.covers("mld_by_shear", "mld"))

# The same rule inside a dataset: one definition answers the broad name (and says
# which it picked); a definition the dataset does not carry is simply not found.
shear_shaped = _tiny_dataset("my_mld_defined_by_shear")
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    da = find_variable(shear_shaped, "mld")
message = caught[0].message if caught else None
print("find_variable(ds, 'mld')                ->", da.name)
print("message shown                           :", message)
not_carried = find_variable(shear_shaped, "mld_by_sigma_theta")
print("find_variable(ds, 'mld_by_sigma_theta') ->", not_carried)

# Two different definitions and no generic variable: choosing either would be a coin
# flip on the numbers, so nothing is returned and the warning names the candidates.
# (This matching is plain cf-xarray -- the generic entry's registered criteria include
# every definition -- so ds.cf["mld"] raises cf-xarray's own "multiple variables".)
two_definitions = _tiny_dataset("my_mld_defined_by_shear")
two_definitions["ocean_mixed_layer_thickness_defined_by_sigma_theta"] = (
    two_definitions["my_mld_defined_by_shear"]
)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    ambiguous = find_variable(two_definitions, "mld")
message = caught[0].message if caught else None
print("two definitions, asked for 'mld'        ->", ambiguous)
print("message shown                           :", message)

print()
print("=" * 70)
print("6. Every resolution says which variable it actually found")
print("=" * 70)

# sea_water_temperature (in-situ) is an alias of sea_water_potential_temperature.
# They are near-identical rather than identical quantities -- a distinction the
# vocabulary deliberately does not model yet (see its module docstring); what it
# does guarantee is that you are always told which spelling was actually used.
temp_shaped = _tiny_dataset("sea_water_temperature")
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    da = find_variable(temp_shaped, "temperature")
print("asked for 'temperature', found:", da.name)
print("message shown :", str(caught[0].message) if caught else None)

print()
print("=" * 70)
print("Making an addition permanent")
print("=" * 70)
print(
    """
add_alias()/add_pattern()/register() only last for the current Python session --
they mutate the in-memory VOCABULARY dict and refresh the resolver + cf-xarray
registration, but a fresh `import ocean_skill` loads the vocabulary file again.

To make a new alias or concept permanent, edit ocean_skill/vocab/vocabulary.yaml
(the file VOCABULARY is parsed from at import; its rules are in the README next to
it) and it's there for every future session -- and for any other package that loads
the same file with plain cf-xarray.
"""
)
