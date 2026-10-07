"""Tidal harmonic constants: amplitude and Greenwich phase lag per constituent.

Two calculators, ``tidal_amplitude`` and ``tidal_phase``, used as variable specs::

    {"calculate": "tidal_amplitude", "constituent": "K1"}
    {"calculate": "tidal_phase", "constituent": "K1"}

and each returns a 2-D map (m, or degrees in [0, 360)). They answer the same question
from whichever form the source takes:

- **Re/Im pair** (a roms-tools tidal forcing file's ``ssh_Re``/``ssh_Im`` or a TPXO
  atlas's ``hRe``/``hIm``, named :data:`TIDAL_RE`/:data:`TIDAL_IM` once standardized):
  amplitude is ``hypot(Re, Im)`` in metres and phase ``atan2(-Im, Re)`` in degrees --
  the TPXO/OTIS Greenwich-lag convention, ``h = f A cos(vu - G)``. Exact zeros are the
  files' land fill and come back NaN. Forcing-file amplitudes include the *nodal
  factor of the forcing start date* (``f`` is folded in); that is warned once.
- **SSH time series** (``sea_surface_height_above_geoid``; ROMS ``zeta``): harmonic
  analysis with pyFES (:func:`harmonic_constants`), cell by cell. The fit is a joint
  least-squares over several constituents at once, so K1 is not contaminated by O1 the
  way a spectral-band amplitude is. Nodal modulations and the equilibrium argument
  ``vu`` are applied, so amplitudes are the *mean* constants, comparable to an atlas.

``pyfes`` is optional (see ``environment.yml``) and imported lazily, like ``oceans``
in :mod:`ocean_skill.detide`.
"""

from __future__ import annotations

import warnings

import numpy as np
import xarray as xr

__all__ = ["harmonic_constants", "tidal_amplitude", "tidal_phase"]

TIDAL_AMPLITUDE = "sea_surface_height_tidal_amplitude"  # m
TIDAL_PHASE = "sea_surface_height_tidal_phase"  # degree, Greenwich lag 0-360
TIDAL_RE = "sea_surface_height_tidal_harmonic_real_part"  # m
TIDAL_IM = "sea_surface_height_tidal_harmonic_imaginary_part"  # m
_SSH = "sea_surface_height_above_geoid"

#: Analysed by default: the four semidiurnal and four diurnal principal constituents,
#: cut down to what the record can separate (Rayleigh criterion).
MAJOR8 = ("M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1")
_CONSTITUENT_DIMS = ("ntides", "con", "nc", "constituent")
_CHUNK_BYTES = 100e6


def _require_pyfes():
    """Import :mod:`pyfes`, or raise a clear install hint."""
    try:
        import pyfes
    except ImportError as exc:
        raise ImportError(
            "tidal harmonic analysis needs the 'pyfes' package -- install it with "
            "`conda install -c conda-forge pyfes`."
        ) from exc
    return pyfes


def _pyfes_names(pyfes, names) -> list[str]:
    """Normalize ``names`` to pyFES's capitalization, or raise listing known ones."""
    known = {k.upper(): k for k in pyfes.known_constituents()}
    bad = [n for n in names if str(n).strip().upper() not in known]
    if bad:
        raise ValueError(
            f"unknown tidal constituent(s) {bad}; pyFES knows {sorted(known.values())}."
        )
    return [known[str(n).strip().upper()] for n in names]


def _fit_cell(h, *, wt, f, vu):
    """Fit one cell: ``(complex constants, status)``; 0 land, 1 ok, 2 partial."""
    good = np.isfinite(h)
    if not good.any():
        return np.full(len(wt.constituents), np.nan + 0j), np.int8(0)
    if not good.all():
        return np.full(len(wt.constituents), np.nan + 0j), np.int8(2)
    # harmonic_analysis has no mean term; a nonzero mean would leak into the fit.
    return wt.harmonic_analysis(h - h.mean(), f, vu), np.int8(1)


def _spatial_chunks(sizes: dict[str, int], n_time: int) -> dict[str, int]:
    """Chunk non-time dims so a chunk of float64 series is ~:data:`_CHUNK_BYTES`."""
    budget = max(1, int(_CHUNK_BYTES // (8 * n_time)))
    chunks = {}
    for dim in reversed(list(sizes)):
        chunks[dim] = min(sizes[dim], budget)
        budget = max(1, budget // chunks[dim])
    return chunks


def harmonic_constants(
    ds, *, constituents=None, keep=(), stride: int = 1, time_stride: int = 1
) -> xr.Dataset:
    """Harmonic analysis of the SSH time series in ``ds``, per horizontal cell.

    Returns a Dataset with ``amplitude`` (m) and ``phase`` (degree, Greenwich lag,
    0-360) over a ``constituent`` dimension (pyFES's own order), plus the horizontal
    dims and coordinates. The result is computed (one pass over the record), then
    cached under ``kind="calculated"`` when ``ds`` carries its catalog source, so asking
    for K1 and then M2 analyses once.

    ``constituents`` defaults to :data:`MAJOR8`, cut to those the record length can
    separate by the Rayleigh criterion (the rest are dropped, with a warning -- keeping
    them would destabilize the fit). ``keep`` names constituents analysed regardless,
    warning if the record cannot resolve them. ``stride`` / ``time_stride`` subsample
    the horizontal dims / time. All-NaN (land) cells give NaN; a cell with only some
    NaN also gives NaN, counted in one summary warning.
    """
    from ocean_skill import _stacklevel, cache, catalog, operators
    from ocean_skill.cf import find_coord
    from ocean_skill.units import find_variable

    pyfes = _require_pyfes()
    da = find_variable(ds, _SSH)
    tcoord = None if da is None else find_coord(da, "time")
    if tcoord is None or tcoord.name not in da.dims:
        raise ValueError(
            f"tidal harmonic analysis needs {_SSH!r} with a time dimension; "
            f"dataset has {list(ds.data_vars)}."
        )
    tdim = str(tcoord.name)
    da = da.isel(
        {d: slice(None, None, time_stride if d == tdim else stride) for d in da.dims}
    )
    times = np.asarray(da[tdim].values).astype("datetime64[us]")
    if times.size < 2:
        raise ValueError("tidal harmonic analysis needs at least two time steps.")
    duration = float((times[-1] - times[0]) / np.timedelta64(1, "s"))
    step_h = float(np.median(np.diff(times)) / np.timedelta64(1, "h"))
    if step_h > 4:
        warnings.warn(
            f"tides: sampling interval is {step_h:.1f} h; tidal aliasing is likely.",
            stacklevel=_stacklevel.find(),
        )

    wanted = _pyfes_names(pyfes, MAJOR8 if constituents is None else constituents)
    keep = _pyfes_names(pyfes, keep)
    union = [*wanted, *(k for k in keep if k not in wanted)]
    resolvable = set(
        pyfes.wave_table_factory(pyfes.DARWIN, union).select_waves_for_analysis(
            duration
        )
    )
    dropped = [n for n in wanted if n not in resolvable and n not in keep]
    if dropped:
        warnings.warn(
            f"tides: a {duration / 86400:.0f}-day record cannot separate {dropped} "
            "from their neighbours (Rayleigh criterion); dropped from the analysis.",
            stacklevel=_stacklevel.find(),
        )
    unresolved = [n for n in keep if n not in resolvable]
    if unresolved:
        warnings.warn(
            f"tides: a {duration / 86400:.0f}-day record cannot separate {unresolved} "
            "from neighbouring constituents; its constants are unreliable.",
            stacklevel=_stacklevel.find(),
        )
    names = [n for n in union if n in resolvable or n in keep]
    wt = pyfes.wave_table_factory(pyfes.DARWIN, names)

    source = operators.calculator_source(ds)
    horizontal = {d: da.sizes[d] for d in da.dims if d != tdim}
    key = None
    if source is not None:
        key = cache.key_for_calculated(
            source=source,
            name="harmonic_constants",
            params={
                "time": [str(times[0]), str(times[-1]), int(times.size)],
                "constituents": sorted(wt.constituents),
                "horizontal": horizontal,
                "stride": stride,
                "time_stride": time_stride,
            },
            # A source redefined under the same name is a different record.
            definition=catalog.fingerprint(source),
        )
        hit = cache.load(key, kind="calculated")
        if hit is not None:
            return hit

    # Nodal modulations are latitude-independent: compute once, share across cells.
    f, vu = wt.compute_nodal_modulations(times)
    if da.chunks is not None:
        # The fit needs the whole record of a cell at once, so time must be a single
        # chunk (apply_ufunc's core-dim rule); spatial chunks are sized to ~100 MB.
        da = da.chunk({tdim: -1, **_spatial_chunks(horizontal, times.size)})
    w, status = xr.apply_ufunc(
        _fit_cell,
        da,
        input_core_dims=[[tdim]],
        output_core_dims=[["constituent"], []],
        vectorize=True,
        dask="parallelized",
        kwargs={"wt": wt, "f": f, "vu": vu},
        dask_gufunc_kwargs={"output_sizes": {"constituent": len(wt.constituents)}},
        output_dtypes=[complex, np.int8],
    )
    w, status = w.compute(), status.compute()
    partial, valid = int((status == 2).sum()), int((status > 0).sum())
    if partial:
        warnings.warn(
            f"tides: {partial} of {valid} non-land cells have some NaN in the record; "
            "their constants are NaN.",
            stacklevel=_stacklevel.find(),
        )
    amplitude = np.abs(w)
    # np.angle of NaN+0j is 0; keep the NaN.
    phase = (xr.apply_ufunc(np.angle, w, kwargs={"deg": True}) % 360).where(
        amplitude.notnull()
    )
    out = xr.Dataset(
        {
            "amplitude": amplitude.assign_attrs(units="m", long_name="tidal amplitude"),
            "phase": phase.assign_attrs(
                units="degree", long_name="tidal phase (Greenwich lag)"
            ),
        }
    ).assign_coords(constituent=list(wt.constituents))
    if key is not None:
        cache.save(key, out, kind="calculated")
    return out


def _label(v) -> str:
    return (v.decode() if isinstance(v, bytes) else str(v)).strip().upper()


def _forcing_field(ds, constituent: str, field: str) -> xr.DataArray:
    """Amplitude or phase of ``constituent`` from a Re/Im pair (forcing or atlas)."""
    from ocean_skill import _stacklevel
    from ocean_skill.units import find_variable, to_units

    re, im = find_variable(ds, TIDAL_RE), find_variable(ds, TIDAL_IM)
    # The constituent dim: the one whose labels are strings/bytes (a coordinate, or a
    # TPXO-style `con` label variable), else a conventional name.
    labelled = [
        (d, ds[v].values)
        for v in ds.variables
        for d in ds[v].dims
        if ds[v].dims == (d,) and d in re.dims and ds[v].dtype.kind in "SUO"
    ]
    dim = (
        labelled[0][0]
        if labelled
        else next((d for d in re.dims if d in _CONSTITUENT_DIMS), None)
    )
    if dim is None or not labelled:
        raise ValueError(
            f"cannot find constituent labels on {re.dims}; need a string coordinate."
        )
    labels = [_label(v) for v in labelled[0][1]]
    if constituent.upper() not in labels:
        raise ValueError(f"{constituent!r} not in this source's constituents {labels}.")
    i = labels.index(constituent.upper())
    re, im = (
        re.isel({dim: i}, drop=True).astype(float),
        im.isel({dim: i}, drop=True).astype(float),
    )
    land = (re == 0) & (im == 0)
    if (
        dim == "ntides"
    ):  # roms-tools forcing layout; a raw TPXO atlas has none folded in
        warnings.warn(
            "tides: amplitudes from a tidal forcing file include the nodal factor of "
            "the forcing start date, so they differ slightly from mean harmonic "
            "constants.",
            stacklevel=_stacklevel.find(),
        )
    if field == "amplitude":
        return to_units(
            np.hypot(re, im).assign_attrs(units=re.attrs.get("units", "m")), "m"
        ).where(~land)
    return (np.degrees(np.arctan2(-im, re)) % 360).where(~land)


def _calculate(ds, constituent: str, field: str, *, constituents, stride, time_stride):
    from ocean_skill.units import find_variable

    if (
        find_variable(ds, TIDAL_RE) is not None
        and find_variable(ds, TIDAL_IM) is not None
    ):
        da = _forcing_field(ds, constituent, field)
    else:
        (name,) = _pyfes_names(_require_pyfes(), [constituent])
        res = harmonic_constants(
            ds,
            constituents=constituents,
            keep=[name],
            stride=stride,
            time_stride=time_stride,
        )
        da = res[field].sel(constituent=name, drop=True)
    constituent = constituent.strip().upper()
    std, units, word = (
        (TIDAL_AMPLITUDE, "m", "amplitude")
        if field == "amplitude"
        else (TIDAL_PHASE, "degree", "phase")
    )
    da.name = std
    da.attrs = {
        "standard_name": std,
        "units": units,
        "long_name": f"{constituent} tidal {word}",
        "constituent": constituent,
    }
    return da


def tidal_amplitude(
    ds, *, constituent: str, constituents=None, stride: int = 1, time_stride: int = 1
):
    """2-D map of one constituent's tidal amplitude (m); see the module docstring."""
    return _calculate(
        ds,
        constituent,
        "amplitude",
        constituents=constituents,
        stride=stride,
        time_stride=time_stride,
    )


def tidal_phase(
    ds, *, constituent: str, constituents=None, stride: int = 1, time_stride: int = 1
):
    """2-D map of one constituent's Greenwich phase lag (degrees, 0-360)."""
    return _calculate(
        ds,
        constituent,
        "phase",
        constituents=constituents,
        stride=stride,
        time_stride=time_stride,
    )


def _register() -> None:
    from ocean_skill.operators import register_calculator

    inputs = lambda spec: [[TIDAL_RE, TIDAL_IM], [_SSH]]  # noqa: E731
    for name, fn in (
        ("tidal_amplitude", tidal_amplitude),
        ("tidal_phase", tidal_phase),
    ):
        register_calculator(name, inputs=inputs, fans=("constituent",))(fn)


_register()
