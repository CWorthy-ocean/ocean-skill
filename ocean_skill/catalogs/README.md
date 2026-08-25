# Reference catalogs

This directory ships the package's reference-observation catalogs inside the
wheel (lowest search-path precedence) — WOA23, GLODAP, MODIS Aqua, OceanSODA,
OOI Papa, Coastwatch, Copernicus, WHOTS. Every entry points at a remote URL
(HTTPS/ERDDAP/CMEMS); nothing here is machine-specific.

A project-local `./catalogs/*.yaml` (built with `build_catalog`, e.g. for your
own model output) or the user config dir overrides these on a name collision —
see `ocean_skill/catalog.py` for the full discovery search path.
