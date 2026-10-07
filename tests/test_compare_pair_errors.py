"""One failing pair does not abort the others in ``compare(skip_missing=True)``.

``compare`` used to catch only ``(KeyError, NoValidData)`` per pair, so any other error
-- an unreadable model file (``OSError``), a shape the lanes could not be matched
on (``ValueError``) -- aborted the whole call and threw away every comparison already
formed. Under ``skip_missing=True`` each pair now fails on its own: the error is printed
with its type, counted, and summarised in one warning at the end; and when *nothing*
formed and a pair failed that way, the first error is re-raised, so a systematic bug is
not hidden behind an empty set. ``KeyError``/``NoValidData`` skips (a missing variable,
a masked cell) are the expected kind and behave exactly as before: an empty set, no
warning, no raise. ``skip_missing=False`` still raises at the first failure.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import pandas as pd
import pytest

import ocean_skill as osk
from ocean_skill import catalog, comparison
from ocean_skill.comparison import compare
from tests._tidal_roms import tidal_roms

POINT = (200.01, 50.01)  # the middle cell of the fixture's 3 x 3 grid
SPEC = {
    "test": "level",
    "reference": "obs",
    "standard_name": "sea_water_temperature",
}
SUMMARY = "failed with an error other than"


def _mooring(column="obs (1)"):
    """Return a ``timeSeries`` obs at one point, 1 m down, at the fixture's 4 steps."""
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2024-07-01", periods=4, freq="h"),
            "lon": POINT[0],
            "lat": POINT[1],
            column: 0.0,
        }
    )
    return frame, {"featureType": "timeSeries", "nominal_depth_m": 1.0}


def _cast():
    """Return a one-level CTD cast at the fixture's middle cell."""
    frame = pd.DataFrame(
        {
            "time": pd.Timestamp("2024-07-01"),
            "lon": POINT[0],
            "lat": POINT[1],
            "depth (m)": [1.0, 3.0],
            "obs (m)": [0.0, 0.0],
        }
    )
    return frame, {"featureType": "profile"}


def _setup(monkeypatch, references, bad=(), exc=OSError):
    """Stub the catalog and reader; reading a name in ``bad`` raises ``exc``.

    Returns the list of names read, in order, so a test can say what was *not* read.
    """
    ds, meta = tidal_roms()
    # nanosecond stamps, like every obs frame (see test_depth_convention_compare)
    ds = ds.assign_coords(time=ds["time"].values.astype("datetime64[ns]"))
    sources = {"his": (ds, meta), **references}
    reads: list[str] = []

    def read(name, **kw):
        reads.append(name)
        if name in bad:
            raise exc(f"cannot open {name}")
        return sources[name][0]

    monkeypatch.setattr(osk, "read", read)
    monkeypatch.setattr("ocean_skill.sources.read", read)
    monkeypatch.setattr(
        catalog, "resolve", lambda name: SimpleNamespace(metadata=sources[name][1])
    )
    monkeypatch.setattr(comparison, "_domain_of", lambda name: None)
    monkeypatch.setattr(comparison, "_outline_of", lambda name, convention=None: None)
    return reads


def _moorings(*names, column="obs (1)"):
    return {name: _mooring(column) for name in names}


def _run(names, **kw):
    kw.setdefault("variables", [SPEC])
    kw.setdefault("depths", [1.0])
    return compare(reference=list(names), test="his", cache=False, **kw)


def _summary(caught):
    return [str(w.message) for w in caught if SUMMARY in str(w.message)]


def test_one_unreadable_reference_leaves_the_others_and_warns_once(monkeypatch, capsys):
    names = ("m1", "m2", "m3")
    _setup(monkeypatch, _moorings(*names), bad={"m2"})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = _run(names)
    assert [c.reference_name for c in out] == ["m1", "m3"]
    printed = capsys.readouterr().out
    assert "skipped" in printed
    assert "OSError: cannot open m2" in printed
    assert "2 comparison(s) formed; 1 skipped" in printed
    (message,) = _summary(caught)
    assert "'m2'" in message
    assert "OSError" in message
    assert "cannot open m2" in message


def test_every_pair_failing_re_raises_the_first_error(monkeypatch):
    names = ("m1", "m2", "m3")
    _setup(monkeypatch, _moorings(*names), bad=set(names))
    with pytest.raises(OSError, match="cannot open m1"):
        _run(names)


def test_skip_missing_false_raises_at_the_first_failure(monkeypatch):
    names = ("m1", "m2", "m3")
    reads = _setup(monkeypatch, _moorings(*names), bad={"m2"})
    with pytest.raises(OSError, match="cannot open m2"):
        _run(names, skip_missing=False)
    assert "m3" not in reads


def test_a_missing_variable_is_still_a_quiet_skip(monkeypatch, capsys):
    names = ("m1", "m2")
    _setup(monkeypatch, _moorings(*names, column="salinity (1)"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = _run(names)
    assert len(out) == 0
    assert not _summary(caught)
    printed = capsys.readouterr().out
    assert "skipped" in printed
    assert "OSError" not in printed


def test_a_bad_value_error_is_unexpected_at_the_align_site(monkeypatch, capsys):
    names = ("m1", "m2")
    _setup(monkeypatch, _moorings(*names), bad={"m1"}, exc=ValueError)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = _run(names)
    assert [c.reference_name for c in out] == ["m2"]
    assert "ValueError: cannot open m1" in capsys.readouterr().out
    assert len(_summary(caught)) == 1


def test_an_unreadable_profile_reference_is_skipped_where_its_levels_are_read(
    monkeypatch, capsys
):
    """The depth plan opens each profile reference for its own levels, before align."""
    refs = {"cast_ok": _cast(), "cast_bad": _cast()}
    _setup(monkeypatch, refs, bad={"cast_bad"})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = compare(
            reference=["cast_bad", "cast_ok"],
            test="his",
            variables=[{**SPEC, "test": "height"}],
            depth_method="interp",
            cache=False,
        )
    assert [c.reference_name for c in out] == ["cast_ok"]
    assert "OSError: cannot open cast_bad" in capsys.readouterr().out
    (message,) = _summary(caught)
    assert "cast_bad" in message


def test_an_unexpected_error_while_enumerating_time_bins_re_raises(monkeypatch):
    _setup(monkeypatch, _moorings("m1"))

    def boom(source, freq, window):
        raise OSError(f"cannot open {source}")

    monkeypatch.setattr(comparison, "_time_bins", boom)
    with pytest.raises(OSError, match="cannot open his"):
        _run(["m1"], times={"resample": "1MS", "reduce": "mean"})


def test_a_value_error_from_the_time_bins_stays_a_quiet_skip(monkeypatch, capsys):
    """The documented skip for a source with no time axis is unchanged."""
    _setup(monkeypatch, _moorings("m1"))

    def boom(source, freq, window):
        raise ValueError(f"{source!r} has no time axis")

    monkeypatch.setattr(comparison, "_time_bins", boom)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = _run(["m1"], times={"resample": "1MS", "reduce": "mean"})
    assert len(out) == 0
    assert not _summary(caught)
    assert "skipped 'his': 'his' has no time axis" in capsys.readouterr().out
