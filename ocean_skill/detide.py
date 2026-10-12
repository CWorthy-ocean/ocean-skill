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

Three version-sensitive details of ``pl33tn`` itself are worked around rather than
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
- Its rolling window and its ``interpolate_na`` both run through
  ``apply_ufunc(dask="parallelized")`` along time, which refuses a time axis split
  across more than one dask chunk -- a lazily-read lane (a ROMS history file, most
  often, chunked one or a few steps at a time) hits this routinely. Worked around by
  rechunking the working copy's time axis alone to one chunk
  (``working.chunk({time_name: -1})``) before the call, when the input is dask-backed
  at all; every other axis stays chunked as given, and a plain numpy-backed array
  (``.chunks is None``) is untouched.

Gaps and the time axis are handled here too, because ``pl33tn`` does neither. It fills
*every* NaN run linearly, however long, so a multi-day outage came back as invented
values and the ~``T`` hours either side of it carried a straight line's tide-free
spectrum into the subtidal series; and it reads the sample interval from the first two
stamps alone, so a record with missing stamps was filtered with the wrong width. Now
(``max_gap=``) a gap up to ``max_gap`` is bridged linearly, a longer one ends a stretch
of record: the output is NaN wherever the input was, and wherever a long gap lies
within the filter's half-window -- exactly what filtering each stretch on its own gives.
Missing stamps are NaN samples on a regular grid (output back on the caller's stamps);
a genuinely irregular axis raises rather than being filtered at the wrong ``dt``.

Edges are NaN: ``pl33tn``'s centered rolling window has no full window near either end
of the record (about ``T`` hours' worth of samples on each side), and that gap widens
as ``T`` grows -- expected, not a bug, and left unfilled rather than papered over (see
the project's "data caveats warn, don't paper over" convention).

A QC/flag companion variable (:func:`ocean_skill.units.is_qc_name`) or any non-floating
data variable is passed through unfiltered rather than convolved as if it were a
measurement -- filtering a flag code is meaningless.
"""

from __future__ import annotations

import datetime
from typing import Any

import numpy as np

__all__ = ["detide"]

#: PL33's own name for this filter -- recorded on every filtered variable's attrs.
_FILTER_NAME = "PL33"

#: ``detide(max_gap=...)``'s default: a longer gap ends a stretch (module docstring).
_DEFAULT_MAX_GAP = "6h"

_HOUR_NS = 3_600_000_000_000


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


def _max_gap_hours(max_gap) -> float | None:
    """Convert ``max_gap=`` to hours (a bare number is hours, else a timedelta)."""
    import numbers

    import pandas as pd

    if max_gap is None:
        return None
    hours = float("nan")
    if isinstance(max_gap, str | np.timedelta64 | datetime.timedelta):
        try:
            hours = pd.Timedelta(max_gap) / pd.Timedelta(hours=1)
        except ValueError:
            pass  # unparseable string: refused below, with the same message
    elif isinstance(max_gap, numbers.Real) and not isinstance(max_gap, bool):
        hours = float(max_gap)
    if not hours >= 0:
        raise ValueError(
            "max_gap= takes a pandas timedelta (a string like '6h', or a Timedelta), "
            f"hours as a number, or None to bridge every gap; got {max_gap!r}."
        )
    return hours


def _max_gap_of(side: dict[str, Any]) -> float | None:
    """Return one normalized ``detide=`` lane spec's ``max_gap`` in hours.

    Falls back to the default when the key is absent (a spec that was not normalized).
    """
    return _max_gap_hours(side.get("max_gap", _DEFAULT_MAX_GAP))


def _regular_steps(stamps):
    """Fit a time axis to a regular grid: ``(positions, step_ns)``.

    ``positions`` is ``None`` when the stamps already are evenly spaced (the common
    case: nothing to rebuild); otherwise each stamp's index on a grid of ``step_ns``,
    so stamps that are merely *missing* become NaN samples. Every step must be a whole
    multiple of the base step (within 1% of it, which tolerates timestamp jitter); the
    median step is tried first, then the smallest. Anything else is irregular and
    raises -- ``pl33tn`` would read ``dt`` off the first two stamps and be wrong.
    """
    ns = np.asarray(stamps).astype("datetime64[ns]").astype("int64")
    if ns.size < 3:
        return None, (int(ns[1] - ns[0]) if ns.size == 2 else None)
    steps = np.diff(ns)
    if steps[0] > 0 and (steps == steps[0]).all():
        return None, int(steps[0])
    for base in (np.median(steps), steps.min()):
        if base <= 0:
            continue
        k = np.rint(steps / base)
        # k.sum() bounds the grid: a stray tiny step must not blow it up
        if (k >= 1).all() and (np.abs(steps - k * base) <= 0.01 * base).all() and (
            k.sum() <= 100 * ns.size + 1000
        ):
            return np.r_[0, np.cumsum(k)].astype("int64"), round(base)
    raise ValueError(
        "detide needs a regularly spaced time axis (stamps may be missing, but every "
        "step must be a whole multiple of one base step); this one is irregular. "
        "Resample it first, e.g. `.resample(time='1h').mean()` on a DataArray or "
        "`.resample('1h').mean()` on a Series."
    )


def _half_window(step_hours: float, T: float) -> int:
    """Return how many samples ``pl33tn`` leaves NaN at each end (``Nt - 1``)."""
    return len(np.arange(0.0, 33.0, float(step_hours) * (33.0 / T))) - 1


def _bad_samples(values, *, max_steps: int, half: int):
    """Mark the samples to blank after filtering, along the last axis of ``values``.

    The input's own NaNs, plus every sample within ``half`` of a *long* gap -- a NaN run
    between two valid samples more than ``max_steps`` steps apart. A leading or trailing
    NaN run is no gap: ``pl33tn`` does not fill it, so it already spreads NaN itself.
    Only columns holding both NaN and valid samples are looked at -- a gap-free column
    needs nothing and an all-NaN one (land) comes out of ``pl33tn`` all-NaN anyway -- so
    a gridded lane that is mostly one or the other costs almost nothing here.
    """
    nan = np.isnan(values)
    n = nan.shape[-1]
    flat = nan.reshape(-1, n)
    bad = np.zeros(flat.shape, dtype=bool)
    mixed = np.flatnonzero(flat.any(axis=-1) & ~flat.all(axis=-1))
    if mixed.size:
        gap = flat[mixed]
        idx = np.arange(n, dtype=np.int32)
        prev = np.maximum.accumulate(np.where(gap, np.int32(-1), idx), axis=-1)
        nxt = np.minimum.accumulate(np.where(gap, np.int32(n), idx)[..., ::-1], axis=-1)
        nxt = nxt[..., ::-1]
        long_gap = gap & (prev >= 0) & (nxt < n) & (nxt - prev > max_steps)
        count = np.zeros((len(mixed), n + 1), dtype=np.int32)
        np.cumsum(long_gap, axis=-1, dtype=np.int32, out=count[:, 1:])
        hi, lo = np.minimum(idx + half + 1, n), np.maximum(idx - half, 0)
        bad[mixed] = gap | (count[:, hi] - count[:, lo] > 0)
    return bad.reshape(nan.shape)


def _pl33_dataarray(da, *, T: float, max_gap=_DEFAULT_MAX_GAP):
    """Low-pass ``da`` along its time axis with PL33; same shape, edges NaN.

    Finds the time coordinate via :func:`ocean_skill.cf.find_coord` rather than
    trusting ``pl33tn``'s own ``.cf["T"]`` lookup to see it unaided (see the module
    docstring), works around ``pl33tn``'s nanosecond-literal ``dt`` bug on a working
    copy, then restores the caller's original time coordinate verbatim on the result.
    The working copy sits on a regular grid (missing stamps become NaN samples), and
    the result is blanked where the input was NaN or a long gap (``max_gap``) is
    within the filter's reach -- see :func:`_bad_samples`.
    """
    import xarray as xr

    from ocean_skill.cf import find_coord

    filters = _require_oceans()
    max_gap_h = _max_gap_hours(max_gap)

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
    stamps = working.coords[time_name].values
    positions, step_ns = _regular_steps(stamps)
    if positions is not None:
        offsets = np.arange(positions[-1] + 1) * step_ns
        working = working.assign_coords(
            {time_name: stamps[0] + (positions * step_ns).astype("timedelta64[ns]")}
        ).reindex({time_name: stamps[0] + offsets.astype("timedelta64[ns]")})
    working.coords[time_name].attrs = {
        **working.coords[time_name].attrs,
        "standard_name": "time",
    }
    if working.chunks is not None:
        # pl33tn's centered rolling and its interpolate_na both go through
        # apply_ufunc(dask="parallelized") along time, which requires the time core
        # dim to be a *single* chunk -- a lazily-read lane (a ROMS history file, an
        # ADCP mooring's model counterpart) is routinely chunked into many steps
        # along time and would otherwise raise "consists of multiple chunks, but is
        # also a core dimension". Rechunk time alone; every other axis (depth,
        # eta/xi) stays chunked, so this only makes the time axis whole -- which a
        # rolling tidal filter needs whole regardless -- never the whole field.
        working = working.chunk({time_name: -1})
    low = filters.pl33tn(working, T=T)
    if max_gap_h is not None and step_ns:
        bad = xr.apply_ufunc(
            _bad_samples,
            working,
            input_core_dims=[[time_name]],
            output_core_dims=[[time_name]],
            kwargs={
                "max_steps": int(np.floor(max_gap_h * _HOUR_NS / step_ns + 1e-9)),
                "half": _half_window(step_ns / _HOUR_NS, T),
            },
            dask="parallelized",
            output_dtypes=[bool],
        )
        if bad.chunks is not None or bool(bad.any()):
            low = low.where(~bad)
    if positions is not None:
        low = low.isel({time_name: positions})
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


def _detide_dataarray(da, *, T: float, component: str, max_gap=_DEFAULT_MAX_GAP):
    low = _pl33_dataarray(da, T=T, max_gap=max_gap)
    return _component_from_dataarray(da, low, T=T, component=component)


def _detide_dataset(ds, *, T: float, component: str, max_gap=_DEFAULT_MAX_GAP):
    """Filter every floating, non-QC data variable that carries the time dimension.

    A variable without the time axis (a static grid field, a scalar station position)
    rides through unchanged. A QC/flag companion (:func:`ocean_skill.units.is_qc_name`)
    or non-floating variable rides through unchanged too, with a warning naming what
    was skipped and why -- filtering a flag code as if it were a measurement would
    silently produce nonsense.
    """
    import warnings

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
        low = _pl33_dataarray(da, T=T, max_gap=max_gap)
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


def _detide_series(s, *, T: float, component: str, max_gap=_DEFAULT_MAX_GAP):
    """Low-pass one :class:`pandas.Series` (a :class:`~pandas.DatetimeIndex` required).

    ``pl33tn``'s default ``mode="valid"`` truncates the result and drops the original
    index entirely (a `RangeIndex` in its place) -- ``mode="same"`` is passed explicitly
    instead, to keep the same length and the same (datetime) index, with edges NaN.
    ``pl33tn`` also always hands a Series back wrapped in a single-column DataFrame
    (``x.to_frame().apply(...)``); squeezed back to a Series here so the return type
    matches the input's. Its ``np.convolve`` spreads any NaN across the whole window
    and never bridges a gap, so interior gaps are filled linearly first and re-blanked
    after, the same rule as the DataArray path (:func:`_bad_samples`).
    """
    import pandas as pd

    filters = _require_oceans()

    if not isinstance(s.index, pd.DatetimeIndex):
        raise TypeError(
            "detide needs a Series with a DatetimeIndex; got "
            f"{type(s.index).__name__}. Set the time column as the index first, or "
            "pass meta= if this is a station DataFrame column."
        )
    max_gap_h = _max_gap_hours(max_gap)
    positions, step_ns = _regular_steps(s.index.to_numpy("datetime64[ns]"))
    full = s
    if positions is not None:
        grid = np.full(positions[-1] + 1, np.nan)
        grid[positions] = s.to_numpy(float)
        stepping = pd.Timedelta(step_ns, "ns")
        full = pd.Series(
            grid, index=pd.date_range(s.index[0], periods=len(grid), freq=stepping)
        )
    filled = full.interpolate(limit_area="inside") if full.isna().any() else full
    low = filters.pl33tn(filled, T=T, mode="same").iloc[:, 0]
    if max_gap_h is not None and step_ns:
        low = low.mask(
            _bad_samples(
                full.to_numpy(float),
                max_steps=int(np.floor(max_gap_h * _HOUR_NS / step_ns + 1e-9)),
                half=_half_window(step_ns / _HOUR_NS, T),
            )
        )
    if positions is not None:
        low = low.iloc[positions]
        low.index = s.index
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


def _detide_dataframe(df, *, T: float, component: str, max_gap=_DEFAULT_MAX_GAP):
    """Low-pass every numeric, non-flag column of a time-indexed DataFrame.

    Requires a :class:`~pandas.DatetimeIndex` (the same requirement
    :func:`oceans.filters.pl33tn` places on a bare Series) -- a station DataFrame with
    its own time *column* instead should be passed with ``meta=`` so it converts through
    :func:`ocean_skill.tabular.to_dataset` first (see :func:`detide`'s docstring).
    """
    import warnings

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
        low = _detide_series(series, T=T, component="subtidal", max_gap=max_gap)
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
    max_gap: Any = _DEFAULT_MAX_GAP,
):
    """Remove tides from a time series or timeSeriesProfile with the PL33 filter.

    Parameters
    ----------
    obj
        :class:`xarray.DataArray`, :class:`xarray.Dataset`, :class:`pandas.Series`,
        or :class:`pandas.DataFrame` -- a plain time series or a repeat-visit
        station with depth (``timeSeriesProfile``); the return type matches
        ``obj``'s (a station DataFrame converted via ``meta=`` is the one
        exception).
    T
        ``float``, the filter's half-amplitude period in hours (default ``33.0``,
        the plain PL33 filter; ``T=72`` gives a 3-day low-pass that also removes
        longer-period fluctuations near the tidal band).
    component
        One of ``"subtidal"``, ``"tidal"``, ``"both"`` (default ``"subtidal"`` --
        see below for what each returns).
    meta
        ``dict`` (the catalog entry's metadata) or ``None`` (default). Give it
        when ``obj`` is a station **DataFrame** so it converts through
        :func:`ocean_skill.tabular.to_dataset` first; see below.
    max_gap
        How long a gap (time between two valid samples) is bridged, as a pandas
        timedelta string (``"6h"``, the default), a :class:`pandas.Timedelta`, or hours
        as a number; ``None`` bridges every gap (see below).

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

    Gaps: a gap up to ``max_gap`` is bridged linearly; a longer one ends a stretch of
    record, so the filter never runs across it -- the output is NaN inside the gap and
    within the filter's half-window (about ``T`` hours) of it, exactly what filtering
    each stretch on its own gives, rather than a straight line's invented values and a
    tide leaking out of it. The output is also NaN wherever the input was (a bridged gap
    is not reported back). ``max_gap=None`` bridges everything, as ``pl33tn`` alone
    would, and returns the invented values. Missing time stamps count as gaps (the
    result comes back on the stamps given); an irregularly spaced axis raises --
    resample it first.

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

    _max_gap_hours(max_gap)  # fail on a bad value up front, whatever obj is

    if tabular.is_frame(obj) and meta is not None:
        obj = tabular.to_dataset(obj, meta)

    kwargs = {"T": T, "component": component, "max_gap": max_gap}
    if isinstance(obj, pd.DataFrame):
        return _detide_dataframe(obj, **kwargs)
    if isinstance(obj, pd.Series):
        return _detide_series(obj, **kwargs)
    if isinstance(obj, xr.Dataset):
        return _detide_dataset(obj, **kwargs)
    if isinstance(obj, xr.DataArray):
        return _detide_dataarray(obj, **kwargs)
    raise TypeError(
        "osk.detide takes an xarray DataArray/Dataset or a pandas Series/DataFrame; "
        f"got {type(obj)!r}."
    )
