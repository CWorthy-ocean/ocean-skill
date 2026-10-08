"""``compare()`` says each repeated warning once, not once per pair.

A notebook ``compare()`` over many references, tests and variables raised the same
warning once per pair -- thousands of lines, nearly all one message with a different
reference name or number in it. :class:`~ocean_skill.comparison._PairLog` records what
the pair loop warns and re-emits each distinct warning once, noting how many pairs it
hit; this checks the grouping rule itself, the wiring through ``compare()``, and that
nothing is swallowed when the loop ends on an error.

Also here: the "catalog metadata may be stale" warning must not fire for a position
that is inside the reference's own declared extent
(:meth:`~ocean_skill.comparison.Comparison._within_declared_extent`).

Mirrors ``tests/test_compare_overlap_skip.py``'s fan-shape mocking:
``catalog.resolve`` is stubbed and ``Comparison.align`` raises the warnings a real
pair would, instead of reading anything.
"""

from __future__ import annotations

import warnings
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest import mock

import pytest

from ocean_skill import _stacklevel, comparison
from ocean_skill.comparison import _summarize_warnings, _warning_template
from tests.test_series import (
    STATION,
    _coarse_grid,
    _model_comparison,
    _point_station,
)

TEMPERATURE = "sea_water_potential_temperature"

_META = {
    "variables": [TEMPERATURE],
    "geospatial_lon_min": -145.0,
    "geospatial_lon_max": -140.0,
    "geospatial_lat_min": 57.0,
    "geospatial_lat_max": 58.0,
    "time_coverage_start": "2024-06-01",
    "time_coverage_end": "2024-06-02",
}


# -- the grouping rule ---------------------------------------------------------------


def test_numbers_and_pair_names_are_masked_in_the_template():
    text = "'sta_7' wobbles ~0.0022° (244 m) on 2024-08-01 across his visits"
    template = _warning_template(text, "sta_7", "his")
    assert template == "'<reference>' wobbles ~N° (N m) on N-N-N across <test> visits"
    # A whole-word match: a model named `his` is not found inside "this".
    assert "this" in _warning_template("this one", "x", "his")


def test_two_pairs_with_the_same_templated_warning_collapse_to_one():
    records = [
        (UserWarning, "'a' is 0.2 km off", ("a", "his", "T")),
        (UserWarning, "'b' is 0.7 km off", ("b", "his", "T")),
    ]
    [(category, message, _)] = _summarize_warnings(records)
    assert category is UserWarning
    assert message.startswith("'a' is 0.2 km off")  # the first occurrence, in full
    assert "×2 pairs: a, b" in message


def test_different_warnings_are_both_kept_in_first_seen_order():
    records = [
        (UserWarning, "first thing", ("a", "his", "T")),
        (UserWarning, "second thing", ("a", "his", "T")),
        (UserWarning, "first thing", ("b", "his", "T")),
    ]
    out = _summarize_warnings(records)
    assert [str(m).split(" [")[0] for _, m, _ in out] == ["first thing", "second thing"]
    assert "×2" in out[0][1]
    assert out[1][1] == "second thing"  # seen once: untouched


def test_the_category_is_part_of_the_group():
    records = [
        (UserWarning, "same text", ("a", "his", "T")),
        (RuntimeWarning, "same text", ("b", "his", "T")),
    ]
    assert [c for c, _, _ in _summarize_warnings(records)] == [
        UserWarning,
        RuntimeWarning,
    ]


def test_one_pair_repeating_itself_says_x3_not_pairs():
    records = [(UserWarning, "again", ("a", "his", "T"))] * 3
    [(_, message, _)] = _summarize_warnings(records)
    assert message == "again [×3]"


def test_one_pairs_different_texts_of_a_template_stay_apart():
    """The numbers are the substance when one pair says several of them."""
    records = [
        (UserWarning, "left out 'a' (20 m layer)", ("a", "his", "T")),
        (UserWarning, "left out 'a' (40 m layer)", ("a", "his", "T")),
        (UserWarning, "left out 'b' (20 m layer)", ("b", "his", "T")),
        (UserWarning, "left out 'b' (40 m layer)", ("b", "his", "T")),
        (UserWarning, "left out 'b' (40 m layer)", ("b", "his", "T")),
    ]
    out = [str(m) for _, m, _ in _summarize_warnings(records)]
    # the first pair's 1st and 2nd texts, each merged with the other pair's
    assert len(out) == 2
    assert out[0].startswith("left out 'a' (20 m layer)")
    assert out[0].endswith("[×2 pairs: a, b]")
    assert out[1].startswith("left out 'a' (40 m layer)")
    assert out[1].endswith("[×2 pairs: a, b]")


def test_many_references_are_listed_up_to_five_then_counted():
    records = [
        (UserWarning, f"'r{i}' drifted by {i} m", (f"r{i}", "his", "T"))
        for i in range(24)
    ]
    [(_, message, _)] = _summarize_warnings(records)
    assert message.endswith("[×24 pairs: r0, r1, r2, r3, r4, … (+19 more)]")


# -- through compare() ---------------------------------------------------------------


@contextmanager
def _warns_per_pair(declared: dict[str, dict], *, fail_on: str | None = None):
    """Stub the catalog; have each pair's ``align`` raise a warning naming itself."""

    def _align(self, refresh=False):
        warnings.warn(
            f"{self.reference_name!r} drifts {len(self.reference_name) * 0.37:.2f} km",
            UserWarning,
            stacklevel=_stacklevel.find(),
        )
        warnings.warn("an unrelated notice", UserWarning, stacklevel=_stacklevel.find())
        if self.reference_name == fail_on:
            raise RuntimeError("boom")

    with ExitStack() as stack:
        stack.enter_context(
            mock.patch(
                "ocean_skill.catalog.resolve",
                lambda n: SimpleNamespace(metadata=declared[n]),
            )
        )
        stack.enter_context(mock.patch.object(comparison.Comparison, "align", _align))
        yield


def test_compare_emits_one_summary_per_repeated_warning():
    declared = {"his": _META, "cast_1": _META, "cast_22": _META}
    with _warns_per_pair(declared), warnings.catch_warnings(record=True) as log:
        warnings.simplefilter("always")
        comparison.compare(
            reference=["cast_1", "cast_22"], test="his", variables=[TEMPERATURE]
        )
    texts = [str(w.message) for w in log]
    assert len([t for t in texts if "drifts" in t]) == 1
    [drift] = [t for t in texts if "drifts" in t]
    assert "×2 pairs: cast_1, cast_22" in drift
    assert len([t for t in texts if t.startswith("an unrelated notice")]) == 1
    # Blamed on this test file, not on ocean-skill's internals.
    assert all(w.filename == __file__ for w in log)


def test_a_library_deprecation_stays_hidden_under_the_callers_filters():
    """Re-raised from where it came from, so filters decide as they always did."""
    library = {"__name__": "fakelib"}
    exec(  # noqa: S102 -- a function whose code lives in a fake library file
        compile(
            "import warnings\n"
            "def old_api():\n"
            "    warnings.warn('old_api is deprecated', DeprecationWarning)\n",
            "/nowhere/fakelib.py",
            "exec",
        ),
        library,
    )

    def _align(self, refresh=False):
        library["old_api"]()
        warnings.warn("kept", UserWarning, stacklevel=_stacklevel.find())

    declared = {"his": _META, "cast_1": _META, "cast_2": _META}
    with (
        mock.patch(
            "ocean_skill.catalog.resolve",
            lambda n: SimpleNamespace(metadata=declared[n]),
        ),
        mock.patch.object(comparison.Comparison, "align", _align),
        warnings.catch_warnings(record=True) as log,
    ):
        warnings.simplefilter("always")
        # What Python's defaults do to a DeprecationWarning raised outside __main__.
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        comparison.compare(
            reference=["cast_1", "cast_2"], test="his", variables=[TEMPERATURE]
        )
    texts = [str(w.message) for w in log]
    assert not any("old_api" in t for t in texts)
    assert any(t.startswith("kept") and "×2 pairs" in t for t in texts)


def test_an_error_ending_the_loop_still_re_emits_then_propagates():
    declared = {"his": _META, "cast_1": _META, "cast_2": _META}
    with (
        _warns_per_pair(declared, fail_on="cast_2"),
        warnings.catch_warnings(record=True) as log,
    ):
        warnings.simplefilter("always")
        with pytest.raises(RuntimeError, match="boom"):
            comparison.compare(
                reference=["cast_1", "cast_2"],
                test="his",
                variables=[TEMPERATURE],
                skip_missing=False,
            )
    assert any("drifts" in str(w.message) and "×2" in str(w.message) for w in log)


def test_no_overlap_skips_are_said_in_one_line(capsys):
    his = {
        **_META,
        "time_coverage_start": "2024-02-01",
        "time_coverage_end": "2024-11-29",
    }
    late = {
        **_META,
        "time_coverage_start": "2025-06-01",
        "time_coverage_end": "2025-06-02",
    }
    declared = {"his": his, **{f"cast_{i}": late for i in range(3)}}
    with _warns_per_pair(declared):
        comparison.compare(
            reference=list(declared)[1:], test="his", variables=[TEMPERATURE]
        )
    out = capsys.readouterr().out
    assert out.count("no declared overlap") == 1
    assert "skipped 3 pair(s) with no declared overlap in time:" in out
    assert "'his' vs 'cast_0'" in out
    assert "0 comparison(s) formed; 3 skipped" in out


# -- "catalog metadata may be stale" -------------------------------------------------

# A model fine enough that its cell (~0.2 km) is smaller than the station's wobble.
_FINE_LON = [-144.30 + 0.002 * i for i in range(51)]
_FINE_LAT = [49.93 + 0.002 * i for i in range(51)]


def _stale_comparison(monkeypatch, *, half_width: float):
    """Build a comparison whose station sits ~1.1 km south of its box's centre."""
    clon, clat = STATION[0], STATION[1] + 0.01
    box = {
        "featureType": "timeSeries",
        "geospatial_lon_min": clon - half_width,
        "geospatial_lon_max": clon + half_width,
        "geospatial_lat_min": clat - half_width,
        "geospatial_lat_max": clat + half_width,
    }
    t = _coarse_grid(_FINE_LON, _FINE_LAT)
    c = _model_comparison(
        monkeypatch,
        test=t,
        reference=_point_station(),
        metadata={"run_baseline": box},
    )
    reads = []
    monkeypatch.setattr(
        c, "_prepare_lane", lambda *a, **kw: reads.append(kw) or (None, None)
    )
    return c, t, (clon, clat), reads


def _verify(c, t, centre):
    r = _point_station()
    with warnings.catch_warnings(record=True) as log:
        warnings.simplefilter("always")
        c._verify_point_window(
            t,
            r,
            (centre[0], centre[1], centre[0], centre[1]),
            use_cache=False,
            refresh=False,
            drop_keys=(),
            keep=(),
            derived_window=None,
            point_window_cells=1,
        )
    return [str(w.message) for w in log]


def test_a_position_inside_the_declared_box_is_not_called_stale(monkeypatch):
    c, t, centre, reads = _stale_comparison(monkeypatch, half_width=0.02)
    assert not any("may be stale" in m for m in _verify(c, t, centre))
    assert len(reads) == 1  # the test lane is still re-read around the actual position


def test_a_position_outside_the_declared_box_still_warns(monkeypatch):
    c, t, centre, reads = _stale_comparison(monkeypatch, half_width=0.0001)
    assert any("may be stale" in m for m in _verify(c, t, centre))
    assert len(reads) == 1


def test_the_box_check_wraps_the_antimeridian(monkeypatch):
    c, *_ = _stale_comparison(monkeypatch, half_width=0.02)
    monkeypatch.setattr(
        comparison, "_domain_of", lambda s: (179.99, 10.0, 180.01, 10.1)
    )
    assert c._within_declared_extent((-179.995, 10.05), 0.2)
    assert c._within_declared_extent((180.001, 10.05), 0.2)
    assert not c._within_declared_extent((170.0, 10.05), 0.2)
