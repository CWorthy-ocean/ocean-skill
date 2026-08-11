"""Reading sources: ``osk.read`` opens an intake v2 catalog entry, standardized.

Opens the entry with intake (``cat[name].read()``), then standardizes: ROMS entries
(``metadata.model == "roms"``) route through :mod:`ocean_skill.roms`; other gridded/obs
entries get a light CF rename from the entry's ``standard_names`` map (fuller handling
lives in :mod:`ocean_skill.cf`). Returns a **known type** by featureType: point
featureTypes → :class:`pandas.DataFrame`; gridded/multidim → :class:`xarray.Dataset`.
"""

from __future__ import annotations

from typing import Any

from ocean_skill.catalog import SourceRef, resolve

__all__ = ["read"]


def read(source: str | SourceRef, **kwargs: Any):
    """Open a catalog source and return a CF-standardized Dataset or DataFrame.

    Parameters
    ----------
    source
        An entry name (``"glodap"`` or ``"catalog:name"``) or a :class:`SourceRef`.
    """
    import intake

    ref = source if isinstance(source, SourceRef) else resolve(source)
    meta = ref.metadata

    cat = intake.from_yaml_file(str(ref.path))
    obj = cat[ref.name].read()

    if meta.get("model") == "roms" or meta.get("loader") == "ocean_skill.roms":
        from ocean_skill import roms

        return roms.standardize(obj, meta)

    # Point sources (e.g. ERDDAP tabledap, via add_erddap_source) come back as a
    # DataFrame rather than a Dataset — same metadata contract, different renaming and
    # time-decoding calls, since pandas has no .variables/.assign_coords.
    is_frame = hasattr(obj, "columns")

    # Generic/obs: light CF rename from the entry's standard_names map (cf.standardize
    # will do axis detection + units later). Skip any rename whose target already exists
    # or is claimed twice — the mapping has to stay one-to-one for rename() to work.
    rename: dict[str, str] = {}
    existing = set(obj.columns) if is_frame else set(getattr(obj, "variables", {}))
    for src, dst in (meta.get("standard_names") or {}).items():
        if src not in existing or dst in rename.values() or dst in existing:
            continue
        rename[src] = dst
    if rename:
        obj = obj.rename(columns=rename) if is_frame else obj.rename(rename)

    # Sources are opened with decode_times=False (ocean time units are often non-CF and
    # make xarray refuse the whole file), so decode here instead — otherwise time comes
    # back as raw integers (Dataset) or ISO8601 strings (DataFrame, e.g. ERDDAP's
    # "time (UTC)"). Undecodable units (WOA's "months since ...") return None and are
    # left alone.
    tname = (meta.get("axes") or {}).get("T")
    if is_frame and tname and tname in obj.columns:
        import pandas as pd

        decoded = pd.to_datetime(obj[tname], utc=True, errors="coerce")
        obj = obj.assign(**{tname: decoded})
    elif not is_frame and tname and tname in getattr(obj, "variables", {}):
        from ocean_skill.build import _decode_times

        decoded = _decode_times(obj, obj[tname])
        if decoded is not None:
            obj = obj.assign_coords({tname: decoded})
    return obj
