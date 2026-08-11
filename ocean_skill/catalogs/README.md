# Packaged example catalogs

This directory ships example intake catalogs inside the wheel (lowest search-path
precedence). Project-local `catalogs/*.catalog.yaml` and the user config dir override
them. See `ocean_skill/catalog.py` for the discovery search path.
