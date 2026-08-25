"""How ocean_skill.vocabulary resolves variable names -- and how to extend it.

Self-contained: builds tiny synthetic datasets shaped like real catalog entries
rather than reading anything over the network. Shows:

1. Any spelling registered for a concept resolves to the same variable, even when
   two real catalogs disagree on the exact CF standard_name (a real mismatch this
   demo reproduces, not a hypothetical -- see the ``chlorophyll`` entry below).
2. ``add_alias()`` -- teach an *existing* concept one more real-world spelling,
   live, no restart needed.
3. ``register()`` -- add a wholly new concept from scratch, live.
4. How to make an addition permanent: edit ``ocean_skill/vocabulary.py``'s
   ``VOCABULARY`` dict directly instead of calling these at runtime.

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
print("3. register(): add a wholly NEW concept from scratch, live")
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
print("4. Every resolution says which variable it actually found")
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
add_alias()/register() only last for the current Python session -- they mutate the
in-memory VOCABULARY dict and refresh the resolver + cf-xarray registration, but a
fresh `import ocean_skill` starts from vocabulary.py's own file again.

To make a new alias or concept permanent, edit VOCABULARY in
ocean_skill/vocabulary.py directly (the same dict these functions mutate at
runtime) and it's there for every future session -- no other file needs to change,
per the module's own docstring.
"""
)
