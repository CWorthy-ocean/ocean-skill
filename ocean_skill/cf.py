"""CF-standardization helpers (cf-xarray based).

Detect axes (T/Z/Y/X), rename coordinates to a canonical ``time``/``depth``/``lat``/
``lon``, map variables to CF standard_names (from the catalog entry's ``standard_names``
map or dataset attributes), and tag the object with its ``featureType``.
"""

from __future__ import annotations

import contextlib
import warnings

from ocean_skill import vocabulary

__all__ = ["find_coord", "standardize", "tag_feature_type"]


def standardize(
    obj, standard_names: dict[str, str] | None = None, axes: dict | None = None
):
    """Return ``obj`` with canonical axis names and variables keyed by standard_name."""
    # TODO: use cf_xarray for axis/coord detection; apply the standard_names rename map;
    # normalize units where declared.
    raise NotImplementedError


def tag_feature_type(obj, feature_type: str):
    """Attach a ``featureType`` tag to a standardized object."""
    raise NotImplementedError


@contextlib.contextmanager
def quiet_dropped_ancillaries():
    """Silence cf-xarray's "referred to in the CF attributes" warning around a lookup.

    A QC'd variable keeps ``ancillary_variables="TEMP_flag"`` after a subset or a
    flag-dropping step has removed ``TEMP_flag``, and every ``ds.cf[...]`` lookup then
    repeats cf-xarray's "Variables {...} not found in object but are referred to in
    the CF attributes". It is true and no use to anyone: the lookup is for a
    coordinate or a measurement, and the dangling pointer is to a flag nobody asked
    for. Only that one message is filtered, so a warning about anything else raised
    inside the block still reaches the caller.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Variables .* not found in object but are referred to in the CF "
            r"attributes",
            category=UserWarning,
        )
        yield


def find_coord(ds, kind: str):
    """Return the coordinate/variable for ``kind`` (cf-xarray first, then name match).

    ``kind`` is ``longitude``/``latitude``/``time``/``vertical``. Returns ``None`` if
    nothing plausible is present. Name fallbacks and the exclusion below both come
    from :data:`ocean_skill.vocabulary.COORD_VOCABULARY` (ROMS is the motivating
    fallback case: it writes ``units="degrees East"`` and ``ocean_time`` with
    ``units="second"``, so cf-xarray detects nothing by attrs at all).
    """
    import cf_xarray  # noqa: F401  (registers the .cf accessor)

    axis = vocabulary.COORD_AXIS_BY_KIND[kind]
    # A DataArray has no `.variables`; its coords are the equivalent namespace.
    known = getattr(ds, "variables", None)
    if known is None:
        known = ds.coords
    for key in (kind, axis):
        try:
            with quiet_dropped_ancillaries():
                got = ds.cf[key]
        except (KeyError, ValueError):
            continue
        name = getattr(got, "name", None)
        # Ignore bare dimension indices (e.g. a "time" dim with no coordinate variable,
        # which cf-xarray still reports as the T axis) — they carry no real values.
        # Ignore a name COORD_VOCABULARY excludes for this axis (Z's "bottom") --
        # cf-xarray's own stock Z criteria would otherwise happily match a
        # lowercase "depth_bottom" coordinate as the vertical axis.
        if name in known and not vocabulary.excluded_from_axis(str(name), axis):
            return got
    # Case-insensitively: a source's own spelling (WHOTS' `DEPTH`, say) is one a
    # builder chose, not one this list can enumerate every capitalization of. The
    # fallback names themselves can never collide with an exclude token (see the
    # shipped-vocabulary regression test), so no exclusion check is needed here.
    known_by_lower = {str(k).lower(): str(k) for k in known}
    for nm in vocabulary.COORD_FALLBACKS[kind]:
        if nm in known_by_lower:
            return ds[known_by_lower[nm]]
    return None
