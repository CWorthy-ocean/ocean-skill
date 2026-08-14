"""Skill metrics — one registry (xskillscore), evaluated over whatever dims are asked.

Computes bias, RMSE, MAE, correlation and the standard-deviation ratio, area-weighted
by default (cos(latitude) on a regular grid), and writes a tidy one-row-per-comparison
CSV plus a human-readable summary. Because the aligned pair is always xarray (point
DataFrames are converted upstream in :mod:`ocean_skill.align`), one code path serves
both gridded and point comparisons.

**Every metric is a registry entry, not a line of :func:`compute`.** The reason is that
the same expressions are wanted over two different reductions. Collapse *every* axis and
each metric is one number describing the whole comparison — the row that goes in the CSV
and the point that goes on a Taylor diagram. Collapse only the axis a
:class:`~ocean_skill.comparison.Comparison` was told to score ``over`` (time, for a
satellite record) and each metric is a *map*: bias at every cell, correlation at every
cell. Those must be the same bias and the same correlation, so there is one definition
of each (:data:`REGISTRY`) evaluated twice (:func:`evaluate`) rather than two spellings
free to drift apart. :func:`compute` is now a thin wrapper that asks for every dim.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

__all__ = [
    "DEFAULT_MAP_METRICS",
    "METRICS",
    "REGISTRY",
    "Metric",
    "area_weights",
    "compute",
    "evaluate",
    "register",
    "write",
]


def _xs():
    """Return the xskillscore module, imported on demand (it is not light)."""
    import xskillscore as xs

    return xs


@dataclass(frozen=True)
class Metric:
    """One metric: how to compute it, what it means, and what it is measured in.

    ``fn`` is a *primitive* — it reads the pair directly — and ``derive`` is the
    alternative for a metric defined in terms of others (``crmsd`` from ``rmse`` and
    ``bias``), which is how those two stay algebraically consistent instead of being
    computed twice from the data. Exactly one of the two is set.

    ``kind`` is what the metric is *for*, and it decides what may be drawn as a default
    panel: ``"skill"`` scores the test against the reference, ``"descriptive"``
    describes one member on its own (a mean, a standard deviation), ``"diagnostic"``
    describes the sample rather than either field (``n``).

    ``units`` says how to label it: ``"same"`` as the compared variable, ``"1"`` for a
    dimensionless ratio or correlation, ``"count"`` for a tally. The plotting layer
    needs this to label a colorbar and cannot get it from the numbers.
    """

    name: str
    long_name: str
    kind: str
    units: str
    fn: Callable[..., xr.DataArray] | None = None
    requires: tuple[str, ...] = ()
    derive: Callable[[dict[str, xr.DataArray]], xr.DataArray] | None = None
    #: How :func:`compute` casts the scalar form. A count is an integer there.
    cast: Callable[[Any], Any] = float

    def __post_init__(self):
        if (self.fn is None) == (self.derive is None):
            raise ValueError(
                f"metric {self.name!r} needs exactly one of fn= (a primitive) or "
                "derive= (defined from other metrics)"
            )


#: Every metric by name, in the order :func:`compute` reports them. Ordered rather than
#: merely collected because that order is the CSV's column order, which people read.
REGISTRY: dict[str, Metric] = {}


def register(metric: Metric) -> None:
    """Add (or replace) a metric in :data:`REGISTRY`.

    The extension point: a skill score — Willmott's index of agreement, Murphy's
    MSESS — is one call here plus a colour policy in
    :mod:`ocean_skill.colormaps`, and both :func:`compute` and the pointwise map path
    pick it up with no further change. Deliberately not populated with one yet: which
    definition "skill score" means is a choice worth making explicitly.
    """
    REGISTRY[metric.name] = metric


def _reduce_kw(dim, weights, skipna) -> dict[str, Any]:
    """Keyword arguments shared by every xskillscore call in the registry."""
    kw: dict[str, Any] = {"dim": dim, "skipna": skipna}
    return {**kw, "weights": weights} if weights is not None else kw


def _pairwise(call: str) -> Callable[..., xr.DataArray]:
    """Build a primitive that hands the pair to xskillscore's ``call``."""

    def fn(t, r, *, dim, weights, skipna, **_):
        return getattr(_xs(), call)(t, r, **_reduce_kw(dim, weights, skipna))

    fn.__name__ = f"_{call}"
    return fn


def _member(which: str, how: str) -> Callable[..., xr.DataArray]:
    """Build a primitive reducing *one* member of the pair (``mean``/``std``)."""

    def fn(t, r, *, dim, weights, **_):
        da = t if which == "test" else r
        if weights is None:
            return getattr(da, how)(dim)
        return getattr(da.weighted(weights), how)(dim)

    fn.__name__ = f"_{how}_{which}"
    return fn


def _crmsd(res: dict[str, xr.DataArray]) -> xr.DataArray:
    """Centred (unbiased) RMSD: ``RMSD^2 = bias^2 + crmsd^2``.

    Clamped at 0 because floating point can make the difference marginally negative.
    """
    return np.sqrt(np.maximum(res["rmse"] ** 2 - res["bias"] ** 2, 0.0))


def _sigma_ratio(res: dict[str, xr.DataArray]) -> xr.DataArray:
    """Ratio of the two members' variability; NaN where the reference is flat.

    ``.where(ref != 0)`` rather than a division that would return ``inf``: a reference
    with no variability at all does not have an over- or under-dispersed counterpart,
    so there is no ratio to report.
    """
    ref = res["std_reference"]
    return res["std_test"] / ref.where(ref != 0)


for _metric in (
    Metric(
        "bias",
        "mean error (test − reference)",
        kind="skill",
        units="same",
        # xskillscore: me = mean error (bias)
        fn=_pairwise("me"),
    ),
    Metric(
        "rmse",
        "root-mean-square error",
        kind="skill",
        units="same",
        fn=_pairwise("rmse"),
    ),
    Metric(
        "mae", "mean absolute error", kind="skill", units="same", fn=_pairwise("mae")
    ),
    Metric(
        "corr",
        "Pearson correlation",
        kind="skill",
        units="1",
        fn=_pairwise("pearson_r"),
    ),
    Metric(
        "sigma_ratio",
        "standard-deviation ratio (test / reference)",
        kind="skill",
        units="1",
        requires=("std_test", "std_reference"),
        derive=_sigma_ratio,
    ),
    Metric(
        "std_test",
        "standard deviation of the test",
        kind="descriptive",
        units="same",
        fn=_member("test", "std"),
    ),
    Metric(
        "std_reference",
        "standard deviation of the reference",
        kind="descriptive",
        units="same",
        fn=_member("reference", "std"),
    ),
    Metric(
        "crmsd",
        "centred root-mean-square difference",
        kind="skill",
        units="same",
        requires=("rmse", "bias"),
        derive=_crmsd,
    ),
    Metric(
        "mean_test",
        "mean of the test",
        kind="descriptive",
        units="same",
        fn=_member("test", "mean"),
    ),
    Metric(
        "mean_reference",
        "mean of the reference",
        kind="descriptive",
        units="same",
        fn=_member("reference", "mean"),
    ),
    Metric(
        "n",
        "valid pairs",
        kind="diagnostic",
        units="count",
        fn=lambda t, r, *, valid, dim, **_: valid.sum(dim),
        cast=int,
    ),
):
    register(_metric)

#: Metric names, in report order. Derived from :data:`REGISTRY` so the two cannot
#: disagree; kept as a name for the tuple that used to be written out by hand.
METRICS = tuple(REGISTRY)

#: The metrics drawn as maps by default — exactly the four quantities the Taylor and
#: Target diagrams are built from (see :mod:`ocean_skill.plot.summary`: Taylor plots
#: ``corr`` against ``std_test``/``std_reference``, Target plots ``bias`` against
#: ``crmsd``, both normalized by ``std_reference``). So a default set of skill maps is
#: the *spatial decomposition of the single point those two diagrams plot*, which is a
#: reason to choose these four rather than a taste in metrics. RMSE is not among them,
#: and is not missing: ``rmse² = bias² + crmsd²``.
DEFAULT_MAP_METRICS = ("bias", "crmsd", "corr", "sigma_ratio")


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


def _resolve_names(names: Iterable[str] | None) -> tuple[str, ...]:
    """Validate a requested metric list, or return every registered name in order."""
    if names is None:
        return METRICS
    names = tuple(names)
    for name in names:
        if name in REGISTRY:
            continue
        if name == "weighted":
            raise KeyError(
                "'weighted' is not a metric: it is a flag compute() adds to say "
                "whether the reduction was area-weighted. Pass weighted= instead."
            )
        raise KeyError(f"unknown metric {name!r}; registered: {sorted(REGISTRY)}")
    return names


def _closure(names: Sequence[str]) -> list[str]:
    """``names`` plus everything they are derived from, dependencies first."""
    order: list[str] = []

    def add(name: str) -> None:
        for dep in REGISTRY[name].requires:
            add(dep)
        if name not in order:
            order.append(name)

    for name in names:
        add(name)
    return order


def _units_for(metric: Metric, reference) -> str:
    """Return the units a metric is measured in, given the compared variable's own."""
    if metric.units == "same":
        return str(reference.attrs.get("units", "") or "")
    return "" if metric.units == "1" else metric.units


def _evaluate(
    aligned: xr.Dataset,
    names: Iterable[str] | None,
    *,
    dim,
    test_name: str,
    reference_name: str,
    weights: xr.DataArray | None,
    weighted: bool,
    skipna: bool,
) -> tuple[dict[str, xr.DataArray], xr.DataArray | None]:
    """Shared body of :func:`evaluate`, also reporting the weights actually used.

    :func:`compute` needs that second value for its ``weighted`` column, which is
    otherwise unknowable from outside: ``weights=None, weighted=True`` resolves to
    cos(latitude) weights on a lat/lon grid and to no weights on anything else.
    """
    requested = _resolve_names(names)
    t, r = aligned[test_name], aligned[reference_name]
    # Only cells finite in *both* members are used, so every metric describes the same
    # sample -- and so that a map of bias and a map of correlation cover the same cells.
    valid = np.isfinite(t) & np.isfinite(r)
    t, r = t.where(valid), r.where(valid)

    if weights is None and weighted:
        weights = area_weights(r)
    if weights is not None:
        weights = weights.where(valid, 0.0)

    if dim is None:
        dim = list(r.dims)
    elif isinstance(dim, str):
        dim = [dim]
    else:
        dim = list(dim)

    # xskillscore hands `dim` to apply_ufunc as *core* dimensions, and dask's
    # 'parallelized' mode refuses a core dimension that is split across chunks:
    #   "dimension lat ... consists of multiple chunks, but is also a core dimension".
    # Any lazily-read reference hits this — a NetCDF opened with chunks={} inherits the
    # file's internal chunking, which is routinely several chunks in lat. Collapsing
    # them costs nothing: the reduction runs over all of `dim` anyway, and the aligned
    # pair has already been subset to the overlap.
    t, r = _single_chunk(t, dim), _single_chunk(r, dim)
    weights = _single_chunk(weights, dim)

    res: dict[str, xr.DataArray] = {}
    for name in _closure(requested):
        metric = REGISTRY[name]
        out = (
            metric.fn(t, r, valid=valid, dim=dim, weights=weights, skipna=skipna)
            if metric.fn is not None
            else metric.derive(res)
        )
        # Named and labelled here so a map arrives at the plotting layer knowing what it
        # is; harmless on the scalar path, which reads the values and drops the rest.
        res[name] = out.rename(name).assign_attrs(
            long_name=metric.long_name, units=_units_for(metric, r)
        )
    return {name: res[name] for name in requested}, weights


def evaluate(
    aligned: xr.Dataset,
    names: Iterable[str] | None = None,
    *,
    dim=None,
    test_name: str = "test",
    reference_name: str = "reference",
    weights: xr.DataArray | None = None,
    weighted: bool = True,
    skipna: bool = True,
) -> dict[str, xr.DataArray]:
    """Evaluate metrics on an aligned pair, reducing over ``dim``.

    ``dim=None`` reduces over every dimension of the reference, which gives one number
    per metric: what :func:`compute` asks for. Naming a *subset* instead leaves the
    others standing, which is how a pointwise skill map is made —
    ``dim="time"`` on a pair that kept its time axis returns one 2-D map per metric,
    each cell scored against its own series. The arrays come back named and carrying
    ``long_name``/``units`` attributes.

    ``weights`` (or ``weighted=True``, which falls back to cos(latitude)) applies to the
    dims being reduced. It is right for a reduction over space and a no-op for one over
    time alone — a single cell's series has a single latitude — so the pointwise path
    passes ``weighted=False`` rather than paying for a constant.
    """
    return _evaluate(
        aligned,
        names,
        dim=dim,
        test_name=test_name,
        reference_name=reference_name,
        weights=weights,
        weighted=weighted,
        skipna=skipna,
    )[0]


def compute(
    aligned: xr.Dataset,
    *,
    test_name: str = "test",
    reference_name: str = "reference",
    weights: xr.DataArray | None = None,
    weighted: bool = True,
    min_samples: int = 30,
    names: Iterable[str] | None = None,
    sample_noun: str = "valid cells",
    **extra: Any,
) -> dict[str, Any]:
    """Compute the standard metric set for a test/reference pair.

    Every dimension is reduced, so each metric is one number and the result is one row.
    Only cells finite in *both* members are used, so every metric describes the same
    sample. Returns a flat dict with ``extra`` merged in for identifying columns such as
    variable/depth/period. Warns when fewer than ``min_samples`` cells survive, since
    sparse reference coverage otherwise yields confident-looking meaningless numbers.

    ``names`` narrows the set to particular metrics; by default every registered one is
    reported, in :data:`REGISTRY` order. For the same metrics as *maps* rather than
    numbers, see :func:`evaluate`.

    ``sample_noun`` is what ``n`` counts, for the thin-sample warning to say. Cells, for
    a pair of maps; ``"time steps"`` for a station's series, where the sparse-coverage
    explanation would be about the wrong thing entirely. The caller names it because
    only the caller knows what it aligned — inferring it from the dimensions here would
    put that knowledge in the wrong module.
    """
    res, used_weights = _evaluate(
        aligned,
        names,
        dim=None,
        test_name=test_name,
        reference_name=reference_name,
        weights=weights,
        weighted=weighted,
        skipna=True,
    )
    rec: dict[str, Any] = {
        name: REGISTRY[name].cast(np.asarray(value)) for name, value in res.items()
    }
    rec["weighted"] = bool(used_weights is not None)
    rec.update(extra)
    if rec.get("n", min_samples) < min_samples:
        why = (
            "Usually the reference has sparse coverage over this domain (GLODAP, for "
            "instance, is an open-ocean product and masks most marginal seas)."
            if sample_noun == "valid cells"
            else "Usually the two records overlap over only a short period, or the "
            "binning left few of them — see the aligned pair's attrs for what matched."
        )
        warnings.warn(
            f"only {rec['n']} {sample_noun}: metrics are weakly constrained. {why}",
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
