"""Skill metrics — a single engine (xskillscore) over the xarray aligned pair.

Computes bias, RMSE, MAE, correlation and the standard-deviation ratio, area-weighted
by default (cos(latitude) on a regular grid), and writes a tidy one-row-per-comparison
CSV plus a human-readable summary. Because the aligned pair is always xarray (point
DataFrames are converted upstream in :mod:`ocean_skill.align`), one code path serves
both gridded and point comparisons.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

__all__ = ["METRICS", "area_weights", "compute", "write"]

METRICS = (
    "bias",
    "rmse",
    "mae",
    "corr",
    "sigma_ratio",
    "std_test",
    "std_reference",
    "crmsd",
    "mean_test",
    "mean_reference",
    "n",
)


def area_weights(da) -> xr.DataArray | None:
    """cos(latitude) weights for a lat/lon grid, else ``None``.

    A degree of longitude shrinks toward the poles, so unweighted means over-count high
    latitudes. Weights are broadcast against ``da`` and zeroed where ``da`` is missing.
    """
    lat_name = next((n for n in ("lat", "latitude", "lat_rho") if n in da.coords), None)
    if lat_name is None:
        return None
    w = np.cos(np.deg2rad(da[lat_name]))
    return w.where(np.isfinite(da), 0.0).fillna(0.0)


def _single_chunk(obj, dims):
    """Collapse ``dims`` into one dask chunk each; a no-op on eager data.

    Only the dims the object actually has are named, since the weights array need not
    carry every dimension of the field it weights.
    """
    if obj is None or obj.chunks is None:
        return obj
    return obj.chunk({d: -1 for d in dims if d in obj.dims})


def compute(
    aligned: xr.Dataset,
    *,
    test_name: str = "test",
    reference_name: str = "reference",
    weights: xr.DataArray | None = None,
    weighted: bool = True,
    min_samples: int = 30,
    **extra: Any,
) -> dict[str, Any]:
    """Compute the standard metric set for a test/reference pair.

    Only cells finite in *both* members are used, so every metric describes the same
    sample. Returns a flat dict (one comparison = one row) with ``extra`` merged in for
    identifying columns such as variable/depth/period. Warns when fewer than
    ``min_samples`` cells survive, since sparse reference coverage otherwise yields
    confident-looking but meaningless numbers.
    """
    import xskillscore as xs

    t, r = aligned[test_name], aligned[reference_name]
    valid = np.isfinite(t) & np.isfinite(r)
    t, r = t.where(valid), r.where(valid)

    if weights is None and weighted:
        weights = area_weights(r)
    if weights is not None:
        weights = weights.where(valid, 0.0)

    dims = list(r.dims)

    # xskillscore hands `dim` to apply_ufunc as *core* dimensions, and dask's
    # 'parallelized' mode refuses a core dimension that is split across chunks:
    #   "dimension lat ... consists of multiple chunks, but is also a core dimension".
    # Any lazily-read reference hits this — a NetCDF opened with chunks={} inherits the
    # file's internal chunking, which is routinely several chunks in lat. Collapsing
    # them costs nothing here: every metric below reduces over *all* of `dims` to a
    # scalar anyway, and the aligned pair has already been subset to the overlap.
    t, r = _single_chunk(t, dims), _single_chunk(r, dims)
    weights = _single_chunk(weights, dims)

    kw = {"dim": dims, "skipna": True}
    wkw = {**kw, "weights": weights} if weights is not None else kw

    def _f(x) -> float:
        return float(np.asarray(x))

    # xskillscore: me = mean error (bias). sigma_ratio has no single call — compute
    # each member's weighted std and divide.
    std_t = t.weighted(weights).std(dims) if weights is not None else t.std(dims)
    std_r = r.weighted(weights).std(dims) if weights is not None else r.std(dims)

    _bias = _f(xs.me(t, r, **wkw))
    _rmse = _f(xs.rmse(t, r, **wkw))

    rec: dict[str, Any] = {
        "bias": _bias,
        "rmse": _rmse,
        "mae": _f(xs.mae(t, r, **wkw)),
        "corr": _f(xs.pearson_r(t, r, **wkw)),
        "sigma_ratio": _f(std_t) / _f(std_r) if _f(std_r) else float("nan"),
        "std_test": _f(std_t),
        "std_reference": _f(std_r),
        # centred (unbiased) RMSD: RMSD^2 = bias^2 + crmsd^2. Clamped at 0 because
        # floating point can make the difference marginally negative.
        "crmsd": float(np.sqrt(max(_rmse**2 - _bias**2, 0.0))),
        "mean_test": _f(t.weighted(weights).mean(dims))
        if weights is not None
        else _f(t.mean(dims)),
        "mean_reference": _f(r.weighted(weights).mean(dims))
        if weights is not None
        else _f(r.mean(dims)),
        "n": int(valid.sum()),
        "weighted": bool(weights is not None),
    }
    rec.update(extra)
    if rec["n"] < min_samples:
        warnings.warn(
            f"only {rec['n']} valid cells: metrics are weakly constrained. Usually the "
            "reference has sparse coverage over this domain (GLODAP, for instance, is "
            "an open-ocean product and masks most marginal seas).",
            stacklevel=2,
        )
    return rec


def write(records: list[dict], out_dir: str | Path, stem: str = "metrics") -> Path:
    """Write metric records to ``<out_dir>/metrics/<stem>.csv`` (+ a .txt summary)."""
    import pandas as pd

    d = Path(out_dir).expanduser() / "metrics"
    d.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    csv = d / f"{stem}.csv"
    df.to_csv(csv, index=False)
    (d / f"{stem}.txt").write_text(df.to_string(index=False) + "\n")
    return csv
