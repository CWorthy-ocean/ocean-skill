"""``compare()`` skips pairs whose catalog-declared extents provably never meet.

The motivating hang: a reference list of hundreds of CTD casts against one model
source, where the vast majority of casts fall outside the model's declared time
coverage. Without this skip, each such pair still went through a full read of the
test lane (an empty derived time crop deliberately falls back to uncropped -- see
``ocean_skill.align.subset_to_time``) before the mismatch was ever discovered, once
per stray cast. ``osk.catalog.overlap`` already knows this read-free, from metadata
alone -- this module checks that ``compare()`` now acts on that up front instead of
only warning about it (``Comparison._warn_if_no_overlap``, covered in
``tests/test_series.py``).

Mirrors ``tests/test_profile_compare_depths.py``'s and ``tests/test_pair_spec.py``'s
fan-shape mocking: ``catalog.resolve`` is stubbed with real metadata keys (so the
package's own ``_domain_of``/``_time_coverage_of``/``overlap`` do the actual work) and
``Comparison.align`` records each fanned comparison instead of reading anything.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest import mock

import pytest

from ocean_skill import comparison

TEMPERATURE = "sea_water_potential_temperature"

# `his`'s declared domain/window -- every reference below is placed relative to this.
HIS = {
    "variables": [TEMPERATURE],
    "geospatial_lon_min": -150.0,
    "geospatial_lon_max": -130.0,
    "geospatial_lat_min": 55.0,
    "geospatial_lat_max": 62.0,
    "time_coverage_start": "2024-02-01",
    "time_coverage_end": "2024-11-29",
}

# Well inside `his`'s window -- the ±1-day catalog pad (_TIME_COVERAGE_PAD_DAYS)
# makes weeks of separation the safe margin for a *disjoint* case.
_OVERLAPPING = {
    "variables": [TEMPERATURE],
    "geospatial_lon_min": -145.0,
    "geospatial_lon_max": -140.0,
    "geospatial_lat_min": 57.0,
    "geospatial_lat_max": 58.0,
    "time_coverage_start": "2024-06-01",
    "time_coverage_end": "2024-06-02",
}

# Same place, a year later -- time-disjoint only.
_TIME_DISJOINT = {
    **_OVERLAPPING,
    "time_coverage_start": "2025-06-01",
    "time_coverage_end": "2025-06-02",
}

# Nowhere near `his`'s domain -- space-disjoint (time left overlapping on purpose,
# so a test can tell which axis actually triggered the skip).
_SPACE_DISJOINT = {
    **_OVERLAPPING,
    "geospatial_lon_min": 10.0,
    "geospatial_lon_max": 12.0,
    "geospatial_lat_min": -40.0,
    "geospatial_lat_max": -38.0,
}


@contextmanager
def _fan_recorded(declared: dict[str, dict], *, forbid_reads: bool = False):
    """Stub the catalog and record each fanned pair align() would otherwise read.

    Yields the ``formed`` list -- one ``(test_name, reference_name)`` tuple per
    comparison that actually reached :meth:`~ocean_skill.comparison.Comparison.align`.
    With ``forbid_reads=True``, also fails the test outright if
    :func:`ocean_skill.sources.read` is ever called -- proving a skip happened
    *before* the reference was opened, not merely that no comparison came out.
    """
    formed: list[tuple[str, str]] = []

    def _record(self, refresh=False):
        formed.append((self.test_name, self.reference_name))

    def _forbidden_read(name, **kw):
        pytest.fail(f"read {name!r}: a provably disjoint pair must never be read")

    patches = [
        mock.patch(
            "ocean_skill.catalog.resolve",
            lambda n: SimpleNamespace(metadata=declared[n]),
        ),
        mock.patch.object(comparison.Comparison, "align", _record),
    ]
    if forbid_reads:
        patches.append(mock.patch("ocean_skill.sources.read", _forbidden_read))
    with ExitStack() as stack:
        for patch in patches:
            stack.enter_context(patch)
        yield formed


def test_a_time_disjoint_pair_is_skipped_before_align(capsys):
    declared = {"his": HIS, "cast_2025": _TIME_DISJOINT}
    with _fan_recorded(declared) as formed:
        comparison.compare(reference="cast_2025", test="his", variables=[TEMPERATURE])
    assert formed == []
    out = capsys.readouterr().out
    assert "skipped 'his' vs 'cast_2025': no declared overlap in time" in out


def test_an_overlapping_pair_still_compares(capsys):
    declared = {"his": HIS, "cast_2024": _OVERLAPPING}
    with _fan_recorded(declared) as formed:
        comparison.compare(reference="cast_2024", test="his", variables=[TEMPERATURE])
    assert formed == [("his", "cast_2024")]
    out = capsys.readouterr().out
    assert "no declared overlap" not in out
    assert "comparing 'his' vs 'cast_2024'" in out
    assert "1 comparison(s) formed; 0 skipped" in out


def test_skip_missing_false_raises_instead_of_skipping():
    declared = {"his": HIS, "cast_2025": _TIME_DISJOINT}
    with (
        _fan_recorded(declared) as formed,
        pytest.raises(ValueError, match="do not overlap in time"),
    ):
        comparison.compare(
            reference="cast_2025",
            test="his",
            variables=[TEMPERATURE],
            skip_missing=False,
        )
    assert formed == []


def test_a_climatology_is_exempt_from_the_time_skip(capsys):
    clim = {**_TIME_DISJOINT, "climatology": True}
    declared = {"his": HIS, "clim": clim}
    with _fan_recorded(declared) as formed:
        comparison.compare(reference="clim", test="his", variables=[TEMPERATURE])
    assert formed == [("his", "clim")]
    assert "no declared overlap" not in capsys.readouterr().out


def test_a_space_disjoint_climatology_is_still_skipped(capsys):
    clim = {**_SPACE_DISJOINT, "climatology": True}
    declared = {"his": HIS, "clim": clim}
    with _fan_recorded(declared) as formed:
        comparison.compare(reference="clim", test="his", variables=[TEMPERATURE])
    assert formed == []
    out = capsys.readouterr().out
    assert "skipped 'his' vs 'clim': no declared overlap in space" in out


def test_a_profile_ref_with_no_viable_test_is_never_read():
    profile_meta = {
        **_TIME_DISJOINT,
        "featureType": "profile",
        "axes": {"Z": "DEPTH"},
        "standard_names": {"TEMP": TEMPERATURE},
    }
    declared = {"his": HIS, "cast_2025": profile_meta}
    with _fan_recorded(declared, forbid_reads=True) as formed:
        comparison.compare(reference="cast_2025", test="his", variables=[TEMPERATURE])
    assert formed == []
