"""Two gridded products at Station Papa's position — a point series, no mooring involved.

The model-vs-model counterpart of ``papa_oceansoda_temperature.py``: neither source is
a station here, so nothing in the catalog says "draw this as lines". A ``select`` that
pins both lon and lat to one position says the same thing a mooring's ``featureType``
does — there is no map left to draw — and ``over="time"`` is implied without being
asked for, the same way.

The two products also disagree about which longitude convention they are stored in
(OISST is 0-360, OceanSODA is -180/180) and run at different native resolutions and
cadences (0.25° daily against 1° monthly) — deliberately, since that is exactly the
case a shared point select has to get right: wrapping the request into each grid's own
convention rather than snapping to the wrong edge column, and letting ``match_axis``
bin the finer record into the coarser one's months rather than needing an ``aggregate``
spelled out by hand.

Run:  python examples/point_series_two_gridded_products.py
"""

from pathlib import Path

import ocean_skill as osk

OUT = Path("output/point_series_two_gridded_products")

#: Station Papa's position (49.98 N, 144.25 W) — reused only as a shared point, not as
#: a mooring; neither source below is one.
LON, LAT = -144.245, 49.978

#: Two independent gridded SST products: 0.25-degree daily (0-360 convention) against
#: 1-degree monthly (-180/180 convention).
TEST = "ncdcOisst21Agg"
REFERENCE = "oceansoda_ethz"
PERIOD = ("2015-01-01", "2016-01-01")

comparison = osk.compare(
    reference=REFERENCE,
    test=TEST,
    variables=["temperature"],
    select={"lon": LON, "lat": LAT, "time": slice(*PERIOD)},
    # no aggregate: match_axis bins the daily test into the monthly reference's own
    # bins by itself, which is part of what this example is demonstrating.
)

pair = comparison[0]
print(f"family: {pair.family} — {pair.family_reason}")
print(f"over: {pair.over} — {pair.over_reason}")
print(f"matched: {pair.aligned.attrs.get('match_reason')}")
offset = pair.aligned.attrs.get("nearest_distance_km", float("nan"))
print(f"test sampled {offset:.1f} km from the reference's own snapped cell")
print(f"metrics: {pair.metrics()}")

comparison.save("point_series_two_gridded_products", stem="temperature")
print(f"wrote {OUT}/figures/temperature.png and {OUT}/metrics/temperature.csv")
