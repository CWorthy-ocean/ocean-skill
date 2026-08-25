"""Read NASA OB.DAAC Level-3 mapped filenames well enough to name a catalog entry.

Their filenames are fully self-describing, which is lucky, because a catalog full of
``AQUA_MODIS.20030101_20220131.L3m.MC.CHL.chlor_a.9km`` is unusable::

    AQUA_MODIS . 20030101_20220131 . L3m .  MC  . CHL.chlor_a . 9km . nc
    ^platform    ^date or range           ^period  ^suite.product ^resolution

:func:`nickname` turns that into "MODIS Aqua chlorophyll a January climatology
2003-2022 9km"; :func:`parse` returns the pieces if you want to build your own.

Only the naming is here. Nothing in this module reads data or touches a catalog, so
it composes with :func:`ocean_skill.build.build_catalog` rather than being wired into
it — no source-specific code in the generic builder.
"""

from __future__ import annotations

import re
from datetime import date

__all__ = [
    "PERIODS",
    "PRODUCTS",
    "catalog_entry",
    "catalog_metadata",
    "nickname",
    "parse",
]

#: OB.DAAC temporal binning codes.
#:
#: Three independent properties, deliberately not one flag:
#:
#: ``window``   the date range is a span, so both endpoints are meaningful and both
#:              get shown. Purely a formatting concern.
#: ``rolling``  the product is a *moving* average, recomputed at a finer step than
#:              its own length. This is a claim about the science, not the layout.
#: ``climatology`` averages the *same* calendar slot across many years — which is why
#:              ``MC`` spanning 2003-2022 means "every January in those years", not
#:              "January 2003 through January 2022".
#:
#: ``8D`` has ``window`` but **not** ``rolling``: NASA's 8-day products are fixed,
#: non-overlapping bins restarting each year, not a moving average. Only the
#: ``R``-prefixed codes roll, and those are matched by rule (:data:`_ROLLING`)
#: rather than listed, so ``R3``/``R8``/``R32`` all work without an entry here.
PERIODS: dict[str, dict[str, object]] = {
    "DAY": {"label": "daily"},
    "8D": {"label": "8-day", "window": True},
    # `redundant` marks a label the date phrase already conveys: naming "January
    # 2003" says monthly on its own, and "monthly January 2003" just reads badly.
    "MO": {"label": "monthly", "redundant": True},
    "YR": {"label": "annual"},
    "CU": {"label": "mission composite"},
    "MC": {"label": "climatology", "climatology": "month"},
    "SCSP": {"label": "spring climatology", "climatology": "season"},
    "SCSU": {"label": "summer climatology", "climatology": "season"},
    "SCAU": {"label": "autumn climatology", "climatology": "season"},
    "SCWI": {"label": "winter climatology", "climatology": "season"},
    "SNSP": {"label": "spring"},
    "SNSU": {"label": "summer"},
    "SNAU": {"label": "autumn"},
    "SNWI": {"label": "winter"},
}

#: ``suite.product`` -> a readable name. Anything absent falls back to the raw
#: product token, so an unlisted product still yields a usable nickname.
PRODUCTS: dict[str, str] = {
    "CHL.chlor_a": "chlorophyll a",
    "CHL.chl_ocx": "chlorophyll (OCx)",
    "SST.sst": "sea surface temperature",
    "NSST.sst": "night sea surface temperature",
    "SST4.sst4": "4um sea surface temperature",
    "POC.poc": "particulate organic carbon",
    "PAR.par": "photosynthetically available radiation",
    "KD.Kd_490": "diffuse attenuation at 490nm",
    "FLH.nflh": "normalized fluorescence line height",
    "PIC.pic": "particulate inorganic carbon",
}

#: Platform tokens as they appear at the front of a filename.
PLATFORMS: dict[str, str] = {
    "AQUA_MODIS": "MODIS Aqua",
    "TERRA_MODIS": "MODIS Terra",
    "SNPP_VIIRS": "VIIRS SNPP",
    "JPSS1_VIIRS": "VIIRS NOAA-20",
    "SEASTAR_SEAWIFS": "SeaWiFS",
}

#: ``R32`` -> a 32-day rolling composite. A rule, so any R-code is understood.
_ROLLING = re.compile(r"^R(\d+)$")

_NAME = re.compile(
    r"^(?P<platform>[A-Z0-9_]+)"
    r"\.(?P<start>\d{8})(?:_(?P<end>\d{8}))?"
    r"\.(?P<level>L3m|L3b)"
    r"\.(?P<period>[A-Z0-9]+)"
    r"\.(?P<product>.+?)"
    r"\.(?P<resolution>\d+km|\d+m)"
    r"(?:\.nc)?$"
)

_MONTHS = (
    "January February March April May June July "
    "August September October November December"
).split()


def _period(code: str) -> dict[str, object]:
    """Describe a temporal code, deriving R-codes rather than enumerating them."""
    if code in PERIODS:
        return PERIODS[code]
    rolling = _ROLLING.match(code)
    if rolling:
        days = int(rolling.group(1))
        return {"label": f"{days}-day rolling", "window": True, "rolling": True}
    return {"label": code.lower()}


def _as_date(token: str) -> date:
    return date(int(token[:4]), int(token[4:6]), int(token[6:8]))


def parse(url_or_name: str) -> dict | None:
    """Return the filename's parts, or ``None`` if it isn't an OB.DAAC L3 name.

    ``None`` rather than raising: this is used while sweeping directory listings
    that also contain palettes, checksums and other files, and a non-match is an
    ordinary outcome there.
    """
    stem = str(url_or_name).rstrip("/").split("/")[-1]
    match = _NAME.match(stem)
    if match is None:
        return None
    parts = match.groupdict()
    period = _period(parts["period"])
    return {
        **parts,
        "platform_label": PLATFORMS.get(parts["platform"], parts["platform"]),
        "product_label": PRODUCTS.get(parts["product"], parts["product"]),
        "period_label": period["label"],
        "start_date": _as_date(parts["start"]),
        "end_date": _as_date(parts["end"]) if parts["end"] else None,
        "climatology": period.get("climatology"),
        "rolling": bool(period.get("rolling")),
        "window": bool(period.get("window")),
        "redundant": bool(period.get("redundant")),
    }


def _when(p: dict) -> str:
    """Phrase the time coverage the way the period code means it."""
    start, end = p["start_date"], p["end_date"]

    # A climatology averages one calendar slot across many years, so the range is
    # years, not a span of dates: MC over 2003-2022 is "every January in 2003-2022".
    if p["climatology"]:
        multiyear = end is not None and end.year != start.year
        span = f"{start.year}-{end.year}" if multiyear else str(start.year)
        if p["climatology"] == "month":
            return f"{_MONTHS[start.month - 1]} climatology {span}"
        return span

    if end is None:
        return start.isoformat()
    if p["window"]:  # a span rather than a calendar unit; both endpoints matter
        return f"{start.isoformat()} to {end.isoformat()}"
    if start.year == end.year:
        if (start.month, end.month) == (1, 12):
            return str(start.year)  # a full calendar year
        if start.month == end.month:
            return f"{_MONTHS[start.month - 1]} {start.year}"
    return f"{start.isoformat()} to {end.isoformat()}"


def catalog_metadata(url_or_name: str) -> dict:
    """Return the catalog metadata an OB.DAAC filename already states.

    The filename is authoritative about things the *file* does not say. A monthly
    climatology's global attributes give only ``time_coverage_start/end`` spanning
    the whole averaging period (2003-2022 for a January climatology), and nothing
    marks it as a climatology at all — so probing alone produces an entry that a
    July 2012 time search would match. The ``MC`` in the name is what settles it.

    Returns ``{}`` for a name this module does not recognize, so it is safe to call
    over a directory listing.
    """
    parsed = parse(url_or_name)
    if parsed is None:
        return {}

    md: dict = {
        "platform": parsed["platform_label"],
        "period": parsed["period_label"],
        "nominal_resolution": parsed["resolution"],
    }
    start, end = parsed["start_date"], parsed["end_date"]

    if parsed["climatology"]:
        md["climatology"] = True
        md["climatology_period"] = (
            f"month{start.month:02d}"
            if parsed["climatology"] == "month"
            else parsed["period_label"].removesuffix(" climatology")
        )
        # The averaging span is real and worth keeping, but it is not time coverage:
        # a January climatology is not "data from 2003 to 2022", it is January. Left
        # in time_coverage_*, it would answer a July query. Recorded under its own
        # key, and the probed values are cleared so `find(time=...)` treats this as
        # what it is -- a calendar slot, reachable via `find(climatology=...)`.
        md["climatology_span_start"] = start.isoformat()
        md["climatology_span_end"] = (end or start).isoformat()
        md["time_coverage_start"] = None
        md["time_coverage_end"] = None
    else:
        md["time_coverage_start"] = start.isoformat()
        md["time_coverage_end"] = (end or start).isoformat()
    return md


def catalog_entry(url: str) -> dict:
    """Return a :func:`ocean_skill.build.build_catalog` spec for one OB.DAAC URL.

    Pairs with :func:`nickname` to build a whole catalog from a directory listing::

        build_catalog(
            {nickname(u): catalog_entry(u) for u in urls if nickname(u)},
            "ocean_skill/catalogs/modis_aqua.yaml", title="MODIS Aqua",
        )
    """
    return {"url": url, **catalog_metadata(url)}


def nickname(url_or_name: str) -> str | None:
    """Return a short human-readable name for an OB.DAAC L3 file, else ``None``.

    Suitable directly as a catalog entry name::

        >>> nickname("AQUA_MODIS.20030101_20220131.L3m.MC.CHL.chlor_a.9km.nc")
        'MODIS Aqua chlorophyll a January climatology 2003-2022 9km'
    """
    p = parse(url_or_name)
    if p is None:
        return None
    # Only a *month* climatology folds its label into the date phrase ("January
    # climatology 2003-2022"); a seasonal one still needs "summer climatology".
    hide_label = p["climatology"] == "month" or p["redundant"]
    period = "" if hide_label else f"{p['period_label']} "
    return (
        (
            f"{p['platform_label']} {p['product_label']} {period}{_when(p)} "
            f"{p['resolution']}"
        )
        .replace("  ", " ")
        .strip()
    )
