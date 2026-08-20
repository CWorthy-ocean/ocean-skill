"""``_warn_if_chunk_is_large`` warns on the store's chunk size, not the reduced result.

:data:`ocean_skill.comparison.LOAD_WARN_BYTES` inspects the field :func:`_prepare` hands
back -- already reduced to whatever ``select``/``aggregate`` left standing -- and so
never fires for the failure this guards: a reduction that runs *before* a selection
narrows it, over a chunk sized by the store rather than by the request.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import xarray as xr

from ocean_skill.comparison import CHUNK_WARN_BYTES, _warn_if_chunk_is_large


def _dask_dataset(chunk_mb: float) -> xr.Dataset:
    """A single-variable Dataset whose one chunk is (about) ``chunk_mb`` megabytes."""
    n = int((chunk_mb * 1024**2 / 8) ** 0.5)  # float64, square 2-D chunk
    return xr.Dataset({"temp": (("y", "x"), np.zeros((n, n)))}).chunk(
        {"y": -1, "x": -1}
    )


def test_a_large_chunk_warns():
    ds = _dask_dataset(chunk_mb=CHUNK_WARN_BYTES / 1024**2 * 4)
    with pytest.warns(UserWarning, match="stored in chunks up to"):
        _warn_if_chunk_is_large(ds, "some_source")


def test_a_small_chunk_does_not_warn():
    ds = _dask_dataset(chunk_mb=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_if_chunk_is_large(ds, "some_source")


def test_an_eager_dataset_does_not_warn():
    """Nothing dask-backed left to hide an oversized chunk behind."""
    ds = xr.Dataset({"temp": (("y", "x"), np.zeros((4, 4)))})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_if_chunk_is_large(ds, "some_source")


def test_none_does_not_raise():
    """``osk.read`` returns ``None`` on a genuine miss (or when mocked out) -- inert."""
    _warn_if_chunk_is_large(None, "some_source")


def test_a_dataframe_does_not_raise():
    """A station is a DataFrame (see ``tabular.is_frame``), with no ``data_vars``."""
    import pandas as pd

    _warn_if_chunk_is_large(pd.DataFrame({"x": [1, 2, 3]}), "some_source")
