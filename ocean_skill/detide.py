"""Tidal filtering: remove diurnal/semidiurnal tides with the PL33 low-pass filter.

Wraps :func:`oceans.filters.pl33tn` -- a symmetric 67-weight FIR low-pass filter with a
33-hour half-amplitude period (Rosenfeld 1983, WHOI Technical Report 85-35), resampled
to the series' own sample interval and normalized to sum to 1. It removes diurnal (K1,
~24h) and semidiurnal (M2, ~12.4h) tides, leaving the subtidal (low-frequency)
circulation standing. ``oceans`` is not part of ocean-skill's pip-installable core (see
``environment.yml``); it is imported lazily, here, the same way every other heavy
geoscience dependency in this package is.

Two derived signals, from one filter:

- **subtidal** (the default) -- ``pl33tn(x)``, the low-passed series itself.
- **tidal** -- ``x - pl33tn(x)``, the removed high-frequency residual.

Works on a plain time series (``timeSeries``) and on a repeat-visit station with depth
(``timeSeriesProfile``) alike: :func:`oceans.filters.pl33tn`'s rolling-window
convolution runs along the time axis and broadcasts over any other dimension (depth,
say) for free, so a ``time x depth`` field is filtered independently at every depth
with no special-casing here.

Two version-sensitive details of ``pl33tn`` itself are worked around rather than
inherited:

- Its ``xr.DataArray`` branch finds the sample interval as
  ``(x.cf["T"][1] - x.cf["T"][0]) / np.timedelta64(3_600_000_000_000)`` -- a bare
  integer literal, which numpy/pandas resolve to nanosecond resolution. Since pandas 2.x
  /numpy 2.x default new datetime axes to microsecond (or coarser) resolution, that
  division silently returns a value 1000x too small (an hour miscounted as a
  millisecond) unless the time coordinate happens to already be ``datetime64[ns]``.
  Worked around here by casting a *copy* of the time coordinate to ``datetime64[ns]``
  before the call, then restoring the caller's original coordinate (dtype and attrs
  both) on the result -- callers never see the cast.
- Its ``.cf["T"]`` lookup depends on cf-xarray recognizing the time coordinate, which it
  does only when the coordinate carries CF attributes (``standard_name="time"``, or
  similar) -- a plain ``datetime64`` coordinate named ``"time"`` with no such attrs
  (exactly what :func:`ocean_skill.tabular.to_dataset` produces) is invisible to it. The
  coordinate is located the package's own way first
  (:func:`ocean_skill.cf.find_coord`, cf-xarray then a name-fallback list -- the same
  routine :func:`ocean_skill.operators.resolve_dim` uses for every other axis in this
  pipeline) and then stamped with ``standard_name="time"`` on the working copy, so
  ``pl33tn``'s own internal lookup resolves to the same coordinate regardless of what
  attrs the input actually carried.

Edges are NaN: ``pl33tn``'s centered rolling window has no full window near either end
of the record (about ``T`` hours' worth of samples on each side), and that gap widens
as ``T`` grows -- expected, not a bug, and left unfilled rather than papered over (see
the project's "data caveats warn, don't paper over" convention).

A QC/flag companion variable (:func:`ocean_skill.units.is_qc_name`) or any non-floating
data variable is passed through unfiltered rather than convolved as if it were a
measurement -- filtering a flag code is meaningless.
"""

from __future__ import annotations

from typing import Any

__all__ = ["detide"]

#: PL33's own name for this filter -- recorded on every filtered variable's attrs.
_FILTER_NAME = "PL33"


def _require_oceans():
    """Import :mod:`oceans.filters`, or raise a clear install hint."""
    try:
        import oceans.filters as _filters
    except ImportError as exc:
        raise ImportError(
            "osk.detide needs the 'oceans' package (the PL33 tidal filter, "
            "oceans.filters.pl33tn) -- install it with `pip install oceans` or "
            "`conda install -c conda-forge oceans`."
        ) from exc
    return _filters


def _detide_attrs(original: dict, *, T: float, component: str) -> dict[str, Any]:
    """Attrs for a filtered variable: the original's, plus filter provenance.

    ``units`` is carried over explicitly -- ``pl33tn`` itself drops it (it appends
    ", filtered" to every other attr, on the theory that a filtered field may no
    longer mean what its long_name says, but the physical unit is unchanged by a
    low-pass filter and is worth keeping usable by :mod:`ocean_skill.units`).
    """
    attrs = {k: v for k, v in original.items() if k != "units"}
    if "units" in original:
        attrs["units"] = original["units"]
    attrs["detide_filter"] = _FILTER_NAME
    attrs["detide_period_hours"] = float(T)
    attrs["detide_component"] = component
    return attrs


def _pl33_dataarray(da, *, T: float):
    """Low-pass ``da`` along its time axis with PL33; same shape, edges NaN.

    Finds the time coordinate via :func:`ocean_skill.cf.find_coord` rather than
    trusting ``pl33tn``'s own ``.cf["T"]`` lookup to see it unaided (see the module
    docstring), works around ``pl33tn``'s nanosecond-literal ``dt`` bug on a working
    copy, then restores the caller's original time coordinate verbatim on the result.
    """
    from ocean_skill.cf import find_coord

    filters = _require_oceans()

    time_coord = find_coord(da, "time")
    if time_coord is None or time_coord.name not in da.dims:
        raise ValueError(
            "detide needs a time dimension to filter along, and none was found on "
            f"{da.name!r} (dims: {da.dims!r}). Pass an object with a recognizable "
            "time axis, or (for a station DataFrame) meta= so it can be converted "
            "first."
        )
    time_name = str(time_coord.name)
    original_time = da.coords[time_name]

    working = da.assign_coords(
        {time_name: original_time.astype("datetime64[ns]")}
    )
    working.coords[time_name].attrs = {
        **working.coords[time_name].attrs,
        "standard_name": "time",
    }
    low = filters.pl33tn(working, T=T)
    # pl33tn returns the coordinate it was given (now datetime64[ns], "standard_name"
    # stamped); the caller asked about `da`'s own coordinate, not this working copy's.
    low = low.assign_coords({time_name: original_time})
    low.name = da.name
    low.attrs = _detide_attrs(da.attrs, T=T, component="subtidal")
    return low


def _component_from_dataarray(da, low, *, T: float, component: str):
    """Apply ``component=`` to a filtered DataArray, given its raw field + low-pass."""
    if component == "subtidal":
        return low
    tidal = da - low
    tidal.name = da.name
    tidal.attrs = _detide_attrs(da.attrs, T=T, component="tidal")
    if component == "tidal":
        return tidal
    return low, tidal  # "both"


def _detide_dataarray(da, *, T: float, component: str):
    low = _pl33_dataarray(da, T=T)
    return _component_from_dataarray(da, low, T=T, component=component)


def _detide_dataset(ds, *, T: float, component: str):
    """Filter every floating, non-QC data variable that carries the time dimension.

    A variable without the time axis (a static grid field, a scalar station position)
    rides through unchanged. A QC/flag companion (:func:`ocean_skill.units.is_qc_name`)
    or non-floating variable rides through unchanged too, with a warning naming what
    was skipped and why -- filtering a flag code as if it were a measurement would
    silently produce nonsense.
    """
    import warnings

    import numpy as np

    from ocean_skill import _stacklevel
    from ocean_skill import units as _units
    from ocean_skill.cf import find_coord

    time_coord = find_coord(ds, "time")
    if time_coord is None or time_coord.name not in ds.dims:
        raise ValueError(
            "detide needs a time dimension to filter along, and none was found on "
            f"this dataset (dims: {dict(ds.sizes)!r})."
        )
    time_name = str(time_coord.name)

    ds_out = ds.copy()
    skipped = []
    for name in list(ds.data_vars):
        da = ds[name]
        if time_name not in da.dims:
            continue
        if _units.is_qc_name(name) or not np.issubdtype(da.dtype, np.floating):
            skipped.append(name)
            continue
        low = _pl33_dataarray(da, T=T)
        result = _component_from_dataarray(da, low, T=T, component=component)
        if component == "both":
            subtidal, tidal = result
            del ds_out[name]
            ds_out[f"{name}_subtidal"] = subtidal
            ds_out[f"{name}_tidal"] = tidal
        else:
            ds_out[name] = result
    if skipped:
        warnings.warn(
            f"detide: left {sorted(skipped)} unfiltered (QC/flag companion or "
            "non-floating) -- only floating-point measurements carrying the time "
            "axis are passed through the PL33 filter.",
            stacklevel=_stacklevel.find(),
        )
    return ds_out


def _detide_series(s, *, T: float, component: str):
    """Low-pass one :class:`pandas.Series` (a :class:`~pandas.DatetimeIndex` required).

    ``pl33tn``'s default ``mode="valid"`` truncates the result and drops the original
    index entirely (a `RangeIndex` in its place) -- ``mode="same"`` is passed explicitly
    instead, to keep the same length and the same (datetime) index, with edges NaN.
    ``pl33tn`` also always hands a Series back wrapped in a single-column DataFrame
    (``x.to_frame().apply(...)``); squeezed back to a Series here so the return type
    matches the input's.
    """
    import pandas as pd

    filters = _require_oceans()

    if not isinstance(s.index, pd.DatetimeIndex):
        raise TypeError(
            "detide needs a Series with a DatetimeIndex; got "
            f"{type(s.index).__name__}. Set the time column as the index first, or "
            "pass meta= if this is a station DataFrame column."
        )
    low = filters.pl33tn(s, T=T, mode="same").iloc[:, 0]
    low.name = s.name
    return _component_from_series(s, low, T=T, component=component)


def _component_from_series(s, low, *, T: float, component: str):
    if component == "subtidal":
        return low
    tidal = s - low
    tidal.name = s.name
    if component == "tidal":
        return tidal
    return low, tidal


def _detide_dataframe(df, *, T: float, component: str):
    """Low-pass every numeric, non-flag column of a time-indexed DataFrame.

    Requires a :class:`~pandas.DatetimeIndex` (the same requirement
    :func:`oceans.filters.pl33tn` places on a bare Series) -- a station DataFrame with
    its own time *column* instead should be passed with ``meta=`` so it converts through
    :func:`ocean_skill.tabular.to_dataset` first (see :func:`detide`'s docstring).
    """
    import warnings

    import numpy as np
    import pandas as pd

    from ocean_skill import _stacklevel
    from ocean_skill import units as _units

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            "detide needs a DataFrame with a DatetimeIndex; got "
            f"{type(df.index).__name__}. Set the time column as the index first, or "
            "pass meta= (the catalog entry's metadata) so a station table converts "
            "through ocean_skill.tabular.to_dataset instead."
        )

    subtidal_cols: dict[str, Any] = {}
    tidal_cols: dict[str, Any] = {}
    skipped = []
    for col in df.columns:
        series = df[col]
        if _units.is_qc_name(col) or not np.issubdtype(
            np.asarray(series).dtype, np.floating
        ):
            skipped.append(col)
            if component != "tidal":
                subtidal_cols[col] = series
            continue
        low = _detide_series(series, T=T, component="subtidal")
        subtidal_cols[col] = low
        if component != "subtidal":
            tidal_cols[col] = series - low
    if skipped:
        warnings.warn(
            f"detide: left column(s) {sorted(skipped)} unfiltered (QC/flag companion "
            "or non-numeric) -- only floating-point measurement columns are passed "
            "through the PL33 filter.",
            stacklevel=_stacklevel.find(),
        )
    subtidal = pd.DataFrame(subtidal_cols, index=df.index)
    if component == "subtidal":
        return subtidal
    tidal = pd.DataFrame(tidal_cols, index=df.index)
    if component == "tidal":
        return tidal
    return subtidal, tidal


def detide(
    obj,
    *,
    T: float = 33.0,
    component: str = "subtidal",
    meta: dict[str, Any] | None = None,
):
    """Remove tides from a time series or timeSeriesProfile with the PL33 filter.

    ``obj`` is an :class:`xarray.DataArray`, :class:`xarray.Dataset`,
    :class:`pandas.Series`, or :class:`pandas.DataFrame` -- a plain time series or a
    repeat-visit station with depth (``timeSeriesProfile``) alike; the return type
    matches ``obj``'s (a station DataFrame converted via ``meta=`` is the one
    exception -- see below). ``T`` is the filter's half-amplitude period in hours
    (33, the default, is the plain PL33 filter; ``T=72`` gives a 3-day low-pass that
    also removes longer-period fluctuations near the tidal band).

    ``component`` picks what comes back:

    - ``"subtidal"`` (default) -- the low-passed (detided) signal, ``pl33tn(x)``.
    - ``"tidal"`` -- the removed residual, ``x - pl33tn(x)``.
    - ``"both"`` -- both, as a ``(subtidal, tidal)`` pair for a DataArray/Series/
      DataFrame, or as a single Dataset with ``{name}_subtidal``/``{name}_tidal``
      variables in place of each filtered ``name``.

    ``meta`` is the catalog entry's metadata (as :func:`ocean_skill.read` or
    :func:`ocean_skill.catalog.resolve(source).metadata` returns) -- give it when
    ``obj`` is a station **DataFrame** so it converts through
    :func:`ocean_skill.tabular.to_dataset` first, which understands a
    ``featureType: "timeSeriesProfile"`` entry's time+depth rectangle (a
    DataFrame has no such shape of its own to preserve). That branch returns a
    **Dataset**, not a DataFrame, since a DataFrame cannot represent one. A
    time-indexed DataFrame with no ``meta`` is instead filtered column by column,
    plainly, one time series at a time (see :func:`ocean_skill.tabular.to_dataset`'s
    docstring for the difference).

    Needs the ``oceans`` package (``oceans.filters.pl33tn``) -- not part of
    ocean-skill's pip core, listed in ``environment.yml``; raises with an install hint
    if missing rather than failing on import.

    Edges are NaN: PL33's centered window has no full window within about ``T`` hours
    of either end of the record, and that gap widens as ``T`` grows.

    Examples
    --------
    >>> subtidal = osk.detide(osk.read("tide_gauge"))            # DataFrame in, out
    >>> tidal = osk.detide(ds["zeta"], component="tidal")        # the tide itself
    >>> lp3d = osk.detide(ds, T=72.0)                            # a 3-day low-pass
    """
    if component not in ("subtidal", "tidal", "both"):
        raise ValueError(
            f"component= takes 'subtidal', 'tidal', or 'both'; got {component!r}."
        )

    import pandas as pd
    import xarray as xr

    from ocean_skill import tabular

    if tabular.is_frame(obj) and meta is not None:
        obj = tabular.to_dataset(obj, meta)

    if isinstance(obj, pd.DataFrame):
        return _detide_dataframe(obj, T=T, component=component)
    if isinstance(obj, pd.Series):
        return _detide_series(obj, T=T, component=component)
    if isinstance(obj, xr.Dataset):
        return _detide_dataset(obj, T=T, component=component)
    if isinstance(obj, xr.DataArray):
        return _detide_dataarray(obj, T=T, component=component)
    raise TypeError(
        "osk.detide takes an xarray DataArray/Dataset or a pandas Series/DataFrame; "
        f"got {type(obj)!r}."
    )
