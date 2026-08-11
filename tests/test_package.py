"""Smoke tests: the package imports and exposes its public API."""

from __future__ import annotations

import pytest

import ocean_skill as osk


def test_version():
    assert isinstance(osk.__version__, str) and osk.__version__


@pytest.mark.parametrize("name", ["read", "compare", "Comparison", "catalogs", "find"])
def test_public_api_present(name):
    assert hasattr(osk, name)


def test_catalogs_repr_does_not_error():
    # Discovery is lazy; repr should work even with no catalogs on the path.
    assert isinstance(repr(osk.catalogs), str)
