"""OOI Station Papa mooring vs OceanSODA-ETHZ — temperature as a time series.

The line counterpart of the map examples: a *place* through time rather than a field.
The reference is a moored CTD on Station Papa's flanking mooring A (49.98 N, 144.25 W);
the test is a global monthly gridded product, sampled at the mooring's position.

Four caveats, all of them real and all of them reported by the pipeline rather than
buried here:

* **Cadence.** The mooring samples every 15 minutes and the product is monthly, so the
  mooring is binned to monthly means. That is asked for explicitly — coarsening the
  *reference* changes the thing being scored against, so it is never done silently.
* **Depth.** This entry is titled "CTD (30 meters)" and its ``z`` column is a flat 0.0,
  but its pressure record says otherwise: the instrument sits near 9 m in one deployment
  and near 34 m in another. ``tabular.depth_of`` reads pressure and says so; the product
  is surface-only, so expect a depth-related bias and a warning naming it.
* **Position.** The product's 1-degree cell centre nearest the mooring is ~55 km away.
  That is just what a 1-degree cell is, and it is recorded in the aligned attrs
  (``nearest_distance_km``) rather than swallowed.
* **The download.** ERDDAP has no way to give part of a table unless asked, so the read
  is constrained to the period below. Without ``constraints=`` the whole twelve-year
  record comes down; with it, two years is quick. The lane cache keeps it a one-time
  cost either way.

Swapping in a ROMS run covering Papa is a one-line change: point ``TEST`` at it. The
model lane is sampled at the station whether it is 1-degree and rectilinear or
curvilinear and 4 km.

Run:  python examples/papa_oceansoda_temperature.py
"""

from pathlib import Path

import ocean_skill as osk

OUT = Path("output/papa_oceansoda")

#: The Papa flanking-mooring-A CTD, and the gridded product to score against it.
REFERENCE = "ooi-gp03flma-rim01-02-ctdmog040"
TEST = "oceansoda_ethz"
PERIOD = ("2015-01-01", "2017-01-01")

# Server-side, so the read is two years rather than twelve. osk.read forwards these to
# the ERDDAP reader; a select= would only narrow what has already been downloaded.
constraints = {"time>=": PERIOD[0], "time<=": PERIOD[1]}
station = osk.read(REFERENCE, constraints=constraints)
print(f"{REFERENCE}: {len(station)} rows over {PERIOD[0]} to {PERIOD[1]}")

comparison = osk.compare(
    reference=REFERENCE,
    test=TEST,
    variables=["temperature"],
    select={"time": slice(*PERIOD)},
    # the mooring binned to the product's cadence; see the docstring
    aggregate={"time": {"resample": "MS", "reduce": "mean"}},
)

pair = comparison[0]
print(f"family: {pair.family} — {pair.family_reason}")
print(f"matched: {pair.aligned.attrs.get('match_reason')}")
offset = pair.aligned.attrs.get("nearest_distance_km", float("nan"))
print(f"station offset: {offset:.1f} km")
print(f"metrics: {pair.metrics()}")

comparison.save("papa_oceansoda", stem="temperature")
print(f"wrote {OUT}/figures/temperature.png and {OUT}/metrics/temperature.csv")
