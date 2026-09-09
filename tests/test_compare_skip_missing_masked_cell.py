"""``compare(skip_missing=True)`` skips a station whose position falls in a masked/dry
cell of the test grid -- no valid data to sample there at all.

The motivating failure: a real CTD station near a coastline sat in an entirely masked
ROMS cell, so :func:`ocean_skill.align.sample_at` raised
:class:`~ocean_skill.align.NoValidData` (a ``ValueError`` subclass) from inside
``Comparison.align``. Before this fix, ``compare()``'s align-loop `try`/`except` caught
only ``KeyError`` there, so ``skip_missing=True`` did not save the run -- one uncoverable
station aborted every other station's comparison too, after minutes of real reads.

Mirrors ``tests/test_compare_overlap_skip.py``'s fan-recording idiom: ``catalog.resolve``
is stubbed with real metadata, and ``Comparison.align`` is replaced rather than reading
anything, so this checks only the skip *decision*, not a real masked-cell sample.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest

from ocean_skill import align, comparison

TEMPERATURE = "sea_water_potential_temperature"

HIS = {"variables": [TEMPERATURE]}
DECLARED = {
    "his": HIS,
    "wet_station": {"variables": [TEMPERATURE]},
    "dry_station": {"variables": [TEMPERATURE]},
}


def _align_dry_station_raises(self, refresh=False):
    if self.reference_name == "dry_station":
        raise align.NoValidData(
            "the test lane has no valid data at (-21.884, 64.298) -- the nearest "
            "value is missing everywhere (offset 0.1 km, cell ~0.3 km). The station "
            "may sit in a masked cell; check the source covers it."
        )


def test_a_masked_cell_station_is_skipped_not_fatal(capsys):
    with mock.patch(
        "ocean_skill.catalog.resolve", lambda n: SimpleNamespace(metadata=DECLARED[n])
    ), mock.patch.object(comparison.Comparison, "align", _align_dry_station_raises):
        out = comparison.compare(
            reference=["wet_station", "dry_station"],
            test="his",
            variables=[TEMPERATURE],
        )
    assert [c.reference_name for c in out.comparisons] == ["wet_station"]
    printed = capsys.readouterr().out
    assert "skipped temperature: the test lane has no valid data" in printed
    assert "1 comparison(s) formed; 1 skipped" in printed


def test_skip_missing_false_raises_the_masked_cell_error_instead():
    with (
        mock.patch(
            "ocean_skill.catalog.resolve",
            lambda n: SimpleNamespace(metadata=DECLARED[n]),
        ),
        mock.patch.object(comparison.Comparison, "align", _align_dry_station_raises),
        pytest.raises(align.NoValidData, match="no valid data"),
    ):
        comparison.compare(
            reference=["wet_station", "dry_station"],
            test="his",
            variables=[TEMPERATURE],
            skip_missing=False,
        )


def test_no_valid_data_is_a_value_error_subclass():
    """Backward-compat contract: every existing ``except ValueError`` caller (and any
    ``skip_missing=False`` user checking the type by hand) still sees a ValueError."""
    assert issubclass(align.NoValidData, ValueError)
