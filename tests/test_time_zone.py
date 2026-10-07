"""Tests for :mod:`ocean_skill.time_zone` -- what a source's naive timestamps mean.

A timestamp with no zone or offset used to be taken as UTC, silently. A source can now
declare ``time_zone`` (an IANA zone, daylight saving and all) or ``utc_offset_h`` (a
fixed offset, local clock minus UTC in the ISO-8601 sense: Alaska daylight time -8,
Alaska standard time -9, India +5.5), and :func:`to_utc` /
:func:`localize_naive_datetime64` are the one place those become UTC. These tests pin
the vocabulary (:func:`canonicalize_time_zone`, :func:`tzinfo_of`,
:func:`time_zone_label`), the sign convention in both directions, the conversion's
handling of everything real data throws at it (mixed zones, blanks, ISO variations, the
hours a daylight-saving change makes ambiguous or impossible), and the warnings that
make each of those losses audible.
"""

from __future__ import annotations

import ast
import contextlib
import datetime as dt
import sys
import warnings
import zoneinfo
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ocean_skill import time_zone as tz

pytestmark = pytest.mark.skipif(
    "America/Anchorage" not in zoneinfo.available_timezones(),
    reason="no system tz database",
)

AKDT = {"time_zone": "America/Anchorage"}  # UTC-8 in July, UTC-9 in January

# -- helpers ------------------------------------------------------------------------


@contextlib.contextmanager
def _record():
    """Record every warning raised inside the block."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield caught


def _messages(caught, containing=""):
    return [str(w.message) for w in caught if containing in str(w.message)]


def _check_utc(result, *expected):
    """Assert ``result`` is a tz-aware UTC Series of ``expected`` (``None`` is NaT)."""
    assert isinstance(result, pd.Series)
    assert isinstance(result.dtype, pd.DatetimeTZDtype), result.dtype
    assert str(result.dtype.tz) == "UTC"
    got = [None if pd.isna(v) else v for v in result]
    want = [None if e is None else pd.Timestamp(e, tz="UTC") for e in expected]
    assert got == want


def _error(call, *args, **kwargs) -> ValueError:
    """Return the ``ValueError`` ``call`` raises, which must say where it is from."""
    with pytest.raises(ValueError) as info:
        call(*args, **kwargs)
    assert str(info.value).startswith("time_zone: ")
    return info.value


#: One clock record through the night the Anchorage clocks go back (2024-11-03, 02:00
#: AKDT becomes 01:00 AKST): 01:00-01:59 happens twice, first as AKDT (UTC-8), then as
#: AKST.
FALL_BACK = [
    "2024-11-03 00:00",
    "2024-11-03 00:30",
    "2024-11-03 01:00",
    "2024-11-03 01:30",
    "2024-11-03 01:00",
    "2024-11-03 01:30",
    "2024-11-03 02:00",
    "2024-11-03 02:30",
]
FALL_BACK_UTC = [
    "2024-11-03 08:00",
    "2024-11-03 08:30",
    "2024-11-03 09:00",
    "2024-11-03 09:30",
    "2024-11-03 10:00",
    "2024-11-03 10:30",
    "2024-11-03 11:00",
    "2024-11-03 11:30",
]


# -- the module stands alone --------------------------------------------------------


def test_the_module_imports_nothing_from_the_rest_of_the_package():
    """It must move into a catalog package unchanged: stdlib, numpy and pandas only."""
    tree = ast.parse(Path(tz.__file__).read_text())
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "relative import"
            roots.add(node.module.split(".")[0])
    assert {r for r in roots if r not in sys.stdlib_module_names} == {"numpy", "pandas"}


# -- canonicalize_time_zone ---------------------------------------------------------


def test_nothing_declared_is_an_empty_dict():
    assert tz.canonicalize_time_zone() == {}
    assert tz.canonicalize_time_zone(None, None) == {}
    assert tz.canonicalize_time_zone(time_zone="", utc_offset_h=None) == {}
    assert tz.canonicalize_time_zone(time_zone="   ") == {}


def test_an_iana_zone_is_kept():
    assert tz.canonicalize_time_zone("America/Anchorage") == {
        "time_zone": "America/Anchorage"
    }
    assert tz.canonicalize_time_zone(time_zone="Asia/Kolkata") == {
        "time_zone": "Asia/Kolkata"
    }


@pytest.mark.parametrize(
    "spelling",
    [
        "america/anchorage",
        "AMERICA/ANCHORAGE",
        "America/anchorage",
        " America/Anchorage ",
    ],
)
def test_the_spelling_of_a_zone_is_canonicalised_case_insensitively(spelling):
    assert tz.canonicalize_time_zone(spelling) == {"time_zone": "America/Anchorage"}


@pytest.mark.parametrize(
    "spelling",
    [
        "UTC",
        "utc",
        "Etc/UTC",
        "etc/utc",
        "GMT",
        "gmt",
        "Z",
        "z",
        "Zulu",
        "Etc/GMT",
        "UCT",
    ],
)
def test_the_utc_aliases_all_become_utc(spelling):
    assert tz.canonicalize_time_zone(spelling) == {"time_zone": "UTC"}


@pytest.mark.parametrize(
    "given, hours",
    [
        (-9, -9.0),
        (-8, -8.0),
        (5.5, 5.5),
        (5.75, 5.75),
        (12.75, 12.75),
        (-3.5, -3.5),
        (0, 0.0),
        (14, 14.0),
        (-14, -14.0),
        ("-9", -9.0),
        ("5.5", 5.5),
        (np.float64(-9), -9.0),
        (np.int64(9), 9.0),
        (-8 + 1e-13, -8.0),  # float fuzz is not a different offset
    ],
)
def test_utc_offsets_are_floats_of_hours_in_quarter_hour_steps(given, hours):
    out = tz.canonicalize_time_zone(utc_offset_h=given)
    assert out == {"utc_offset_h": hours}
    assert type(out["utc_offset_h"]) is float


@pytest.mark.parametrize("bad", [0.3, 5.6, 1 / 3, -9.1])
def test_an_offset_that_is_not_a_quarter_hour_multiple_is_an_error(bad):
    err = _error(tz.canonicalize_time_zone, utc_offset_h=bad)
    assert "multiple of 0.25" in str(err)
    assert repr(bad) in str(err)


@pytest.mark.parametrize("bad", [20, -20, 14.25, -14.25, 100])
def test_an_offset_outside_14_hours_is_an_error(bad):
    err = _error(tz.canonicalize_time_zone, utc_offset_h=bad)
    assert "14" in str(err)
    assert "utc_offset_h" in str(err)


@pytest.mark.parametrize(
    "bad", [float("nan"), float("inf"), "abc", "", True, False, [5], {"h": 5}]
)
def test_an_offset_that_is_not_a_number_is_an_error(bad):
    err = _error(tz.canonicalize_time_zone, utc_offset_h=bad)
    assert "utc_offset_h" in str(err)


def test_the_offset_error_states_the_sign_convention():
    assert "local clock minus UTC" in str(
        _error(tz.canonicalize_time_zone, utc_offset_h=20)
    )


def test_a_zone_and_an_offset_are_alternatives():
    err = _error(tz.canonicalize_time_zone, "America/Anchorage", -9)
    assert "alternatives" in str(err)
    assert "time_zone" in str(err) and "utc_offset_h" in str(err)
    _error(tz.canonicalize_time_zone, time_zone="UTC", utc_offset_h=0)
    # a blank zone is not a declaration, so it is no conflict
    assert tz.canonicalize_time_zone("", -9) == {"utc_offset_h": -9.0}


def test_an_unknown_zone_gets_a_did_you_mean():
    err = _error(tz.canonicalize_time_zone, "America/Anchrage")
    assert "'America/Anchrage'" in str(err)
    assert "did you mean 'America/Anchorage'" in str(err)


def test_a_zone_given_by_its_city_alone_is_suggested_in_full():
    assert "did you mean 'America/Anchorage'" in str(
        _error(tz.canonicalize_time_zone, "Anchorage")
    )


def test_an_abbreviation_is_refused_with_an_explanation():
    err = _error(tz.canonicalize_time_zone, "AKST")
    assert "ambiguous" in str(err)
    assert "utc_offset_h" in str(err)


@pytest.mark.parametrize(
    "spelling, hours", [("UTC-9", "-9"), ("UTC+5:30", "5.5"), ("GMT-09:00", "-9")]
)
def test_an_offset_written_as_a_zone_name_is_pointed_at_utc_offset_h(spelling, hours):
    err = _error(tz.canonicalize_time_zone, spelling)
    assert f"utc_offset_h: {hours}" in str(err)


def test_a_zone_that_is_not_a_string_is_an_error():
    err = _error(tz.canonicalize_time_zone, 5)
    assert "IANA" in str(err)
    _error(tz.canonicalize_time_zone, ["UTC"])


def test_canonicalising_is_idempotent():
    for given in ("america/anchorage", "Z"):
        once = tz.canonicalize_time_zone(given)
        assert tz.canonicalize_time_zone(**once) == once
    once = tz.canonicalize_time_zone(utc_offset_h="-9")
    assert tz.canonicalize_time_zone(**once) == once


# -- tzinfo_of / time_zone_label ----------------------------------------------------


@pytest.mark.parametrize("meta", [None, {}, {"time_zone": None}, {"title": "no zone"}])
def test_an_undeclared_source_has_no_tzinfo_and_no_label(meta):
    assert tz.tzinfo_of(meta) is None
    assert tz.time_zone_label(meta) is None


@pytest.mark.parametrize(
    "meta",
    [
        {"time_zone": "UTC"},
        {"time_zone": "Etc/UTC"},
        {"time_zone": "z"},
        {"utc_offset_h": 0},
    ],
)
def test_utc_is_the_same_as_undeclared(meta):
    assert tz.tzinfo_of(meta) is None
    assert tz.time_zone_label(meta) is None


def test_an_iana_zone_gives_its_zoneinfo_and_its_name():
    meta = {"time_zone": "america/anchorage"}
    zone = tz.tzinfo_of(meta)
    assert isinstance(zone, zoneinfo.ZoneInfo)
    assert zone.key == "America/Anchorage"
    assert tz.time_zone_label(meta) == "America/Anchorage"


@pytest.mark.parametrize(
    "hours, label",
    [
        (-9, "UTC-09:00"),
        (-8, "UTC-08:00"),
        (5.5, "UTC+05:30"),
        (5.75, "UTC+05:45"),
        (9, "UTC+09:00"),
        (-3.5, "UTC-03:30"),
        (12, "UTC+12:00"),
        (14, "UTC+14:00"),
    ],
)
def test_a_fixed_offset_is_labelled_the_iso_way(hours, label):
    meta = {"utc_offset_h": hours}
    assert tz.time_zone_label(meta) == label
    zone = tz.tzinfo_of(meta)
    assert isinstance(zone, dt.tzinfo)
    assert zone.utcoffset(None) == dt.timedelta(hours=hours)


def test_the_declaration_is_validated_not_trusted():
    _error(tz.tzinfo_of, {"time_zone": "Nowhere/Land"})
    _error(tz.time_zone_label, {"utc_offset_h": 0.3})
    _error(tz.tzinfo_of, {"time_zone": "UTC", "utc_offset_h": -9})


# -- to_utc: the sign convention ----------------------------------------------------


@pytest.mark.parametrize(
    "offset, july_noon_in_utc",
    [
        (-8, "2024-07-01 20:00"),  # Alaska daylight time: UTC = clock + 8 h
        (-9, "2024-07-01 21:00"),  # Alaska standard time, kept all year, also in July
        (5.5, "2024-07-01 06:30"),  # India: UTC = clock - 5.5 h
        (9, "2024-07-01 03:00"),  # Japan
        (0, "2024-07-01 12:00"),
        (-3.5, "2024-07-01 15:30"),
        (13, "2024-06-30 23:00"),  # crosses midnight backwards
    ],
)
def test_utc_offset_h_is_local_clock_minus_utc(offset, july_noon_in_utc):
    """``UTC = stamp - utc_offset_h``, in both directions of the sign."""
    out = tz.to_utc(pd.Series(["2024-07-01 12:00"]), {"utc_offset_h": offset})
    _check_utc(out, july_noon_in_utc)


def test_a_fixed_offset_applies_all_year_round():
    out = tz.to_utc(
        pd.Series(["2024-07-01 12:00", "2024-01-15 12:00"]), {"utc_offset_h": -9}
    )
    _check_utc(out, "2024-07-01 21:00", "2024-01-15 21:00")


def test_an_iana_zone_follows_daylight_saving():
    out = tz.to_utc(pd.Series(["2024-07-01 12:00", "2024-01-15 12:00"]), AKDT)
    _check_utc(out, "2024-07-01 20:00", "2024-01-15 21:00")  # AKDT -8, AKST -9


def test_an_iana_zone_equals_its_standard_offset_in_winter_only():
    stamps = pd.Series(["2024-07-01 12:00", "2024-01-15 12:00"])
    zone = tz.to_utc(stamps, AKDT)
    standard = tz.to_utc(stamps, {"utc_offset_h": -9})
    assert zone[1] == standard[1]
    assert zone[0] != standard[0]


@pytest.mark.parametrize(
    "meta",
    [None, {}, {"time_zone": "UTC"}, {"utc_offset_h": 0}, {"time_zone": "Etc/UTC"}],
)
def test_naive_stamps_with_nothing_declared_are_utc_as_they_always_were(meta):
    out = tz.to_utc(pd.Series(["2024-07-01 12:00", "2024-01-15 12:00"]), meta)
    _check_utc(out, "2024-07-01 12:00", "2024-01-15 12:00")


def test_a_midnight_crossing_changes_the_date():
    out = tz.to_utc(pd.Series(["2024-07-01 20:00"]), AKDT)
    _check_utc(out, "2024-07-02 04:00")


# -- to_utc: stamps that carry their own offset -------------------------------------


def test_aware_strings_are_converted_not_shifted_by_the_declared_zone():
    stamps = pd.Series(["2024-07-01T12:00:00Z", "2024-07-01T12:00:00-08:00"])
    with _record():
        out = tz.to_utc(stamps, AKDT)
    _check_utc(out, "2024-07-01 12:00", "2024-07-01 20:00")


def test_aware_strings_with_nothing_declared_are_simply_converted():
    stamps = pd.Series(["2024-07-01T12:00:00Z", "2024-07-01T12:00:00+05:30"])
    with _record() as caught:
        out = tz.to_utc(stamps)
    _check_utc(out, "2024-07-01 12:00", "2024-07-01 06:30")
    assert not caught


def test_a_declared_zone_that_disagrees_with_the_data_warns_once_and_changes_nothing():
    stamps = pd.Series(["2024-07-01T12:00:00Z", "2024-01-15T12:00:00Z"] * 50)
    with _record() as caught:
        out = tz.to_utc(stamps, AKDT, subject="station X")
    assert out.iloc[0] == pd.Timestamp("2024-07-01 12:00", tz="UTC")
    assert out.iloc[1] == pd.Timestamp("2024-01-15 12:00", tz="UTC")
    assert len(caught) == 1  # once per call, however many rows
    assert issubclass(caught[0].category, UserWarning)
    message = str(caught[0].message)
    assert message.startswith("station X: declares time_zone America/Anchorage")
    assert "carry their own UTC offset(s) (UTC+00:00)" in message
    assert "using the timestamps' own offsets" in message
    assert "only applies to naive (offset-free) timestamps" in message


def test_the_conflict_warning_names_a_fixed_offset_the_iso_way():
    with _record() as caught:
        tz.to_utc(pd.Series(["2024-07-01T12:00:00Z"]), {"utc_offset_h": -9})
    assert "declares time_zone UTC-09:00" in _messages(caught)[0]


def test_a_stamp_whose_offset_matches_the_zone_at_that_instant_is_no_conflict():
    # AKDT in July, AKST in January: the offsets the zone itself would have given
    stamps = pd.Series(["2024-07-01T12:00:00-08:00", "2024-01-15T12:00:00-09:00"])
    with _record() as caught:
        out = tz.to_utc(stamps, AKDT)
    assert not caught
    _check_utc(out, "2024-07-01 20:00", "2024-01-15 21:00")


def test_the_same_offset_in_the_wrong_season_is_a_conflict():
    stamps = pd.Series(["2024-01-15T12:00:00-08:00"])  # Anchorage is UTC-9 in January
    with _record() as caught:
        out = tz.to_utc(stamps, AKDT)
    _check_utc(out, "2024-01-15 20:00")  # still taken as written
    assert "(UTC-08:00)" in _messages(caught, "carry their own")[0]


def test_a_fixed_offset_matching_the_data_is_no_conflict():
    with _record() as caught:
        out = tz.to_utc(pd.Series(["2024-07-01T12:00:00-09:00"]), {"utc_offset_h": -9})
    assert not caught
    _check_utc(out, "2024-07-01 21:00")


def test_every_disagreeing_offset_is_listed():
    stamps = pd.Series(
        [
            "2024-07-01T12:00:00Z",
            "2024-07-01T12:00:00+05:30",
            "2024-07-01T12:00:00-08:00",  # agrees
        ]
    )
    with _record() as caught:
        out = tz.to_utc(stamps, AKDT)
    _check_utc(out, "2024-07-01 12:00", "2024-07-01 06:30", "2024-07-01 20:00")
    message = _messages(caught, "carry their own")
    assert len(message) == 1
    assert "UTC+00:00" in message[0] and "UTC+05:30" in message[0]
    assert "UTC-08:00" not in message[0]


def test_declaring_utc_never_warns_about_offsets():
    stamps = pd.Series(["2024-07-01T12:00:00-09:00", "2024-07-01T12:00:00Z"])
    with _record() as caught:
        out = tz.to_utc(stamps, {"time_zone": "UTC"})
    assert not caught
    _check_utc(out, "2024-07-01 21:00", "2024-07-01 12:00")


def test_tz_aware_datetime64_is_converted_and_checked_like_aware_strings():
    aware = pd.Series(
        pd.to_datetime(["2024-07-01 12:00", "2024-01-15 12:00"])
    ).dt.tz_localize("UTC")
    with _record() as caught:
        out = tz.to_utc(aware, AKDT, subject="s")
    _check_utc(out, "2024-07-01 12:00", "2024-01-15 12:00")
    assert len(_messages(caught, "declares time_zone")) == 1
    with _record() as caught:
        tz.to_utc(aware)
    assert not caught


def test_a_tz_aware_series_in_the_declared_zone_is_no_conflict():
    local = pd.Series(
        pd.to_datetime(["2024-07-01 12:00", "2024-01-15 12:00"])
    ).dt.tz_localize("America/Anchorage")
    with _record() as caught:
        out = tz.to_utc(local, AKDT)
    assert not caught
    _check_utc(out, "2024-07-01 20:00", "2024-01-15 21:00")


def test_tz_aware_in_a_foreign_zone_is_converted_to_utc():
    kolkata = pd.Series(pd.to_datetime(["2024-07-01 12:00"])).dt.tz_localize(
        "Asia/Kolkata"
    )
    _check_utc(tz.to_utc(kolkata), "2024-07-01 06:30")


def test_mixed_aware_and_naive_python_datetimes():
    stamps = pd.Series(
        [
            dt.datetime(2024, 7, 1, 12, 0),
            dt.datetime(2024, 7, 1, 12, 0, tzinfo=dt.UTC),
            dt.datetime(2024, 1, 15, 12, 0),
        ],
        dtype=object,
    )
    with _record() as caught:
        out = tz.to_utc(stamps, AKDT)
    _check_utc(out, "2024-07-01 20:00", "2024-07-01 12:00", "2024-01-15 21:00")
    assert len(_messages(caught, "carry their own")) == 1


# -- to_utc: mixed columns ----------------------------------------------------------


def test_mixed_aware_and_naive_strings_each_get_their_own_rule():
    stamps = pd.Series(
        [
            "2024-07-01T12:00:00Z",  # aware: as it stands
            "2024-07-01 12:00:00",  # naive: AKDT
            "2024-01-15 12:00:00",  # naive: AKST
            "2024-07-01T12:00:00-08:00",  # aware, and consistent with the zone
        ]
    )
    with _record() as caught:
        out = tz.to_utc(stamps, AKDT)
    _check_utc(
        out,
        "2024-07-01 12:00",
        "2024-07-01 20:00",
        "2024-01-15 21:00",
        "2024-07-01 20:00",
    )
    assert len(_messages(caught, "carry their own")) == 1  # the Z one disagrees


def test_a_naive_stamp_first_does_not_cost_the_aware_ones():
    stamps = pd.Series(["2024-07-01 12:00:00", "2024-07-01T12:00:00Z"])
    with _record() as caught:
        out = tz.to_utc(stamps, AKDT)
    _check_utc(out, "2024-07-01 20:00", "2024-07-01 12:00")
    assert len(_messages(caught, "carry their own")) == 1  # the Z one disagrees
    # with nothing declared the naive one is UTC, the aware one is itself
    _check_utc(tz.to_utc(stamps), "2024-07-01 12:00", "2024-07-01 12:00")


def test_stamps_with_different_offsets_in_one_column_are_all_read():
    stamps = pd.Series(
        [
            "2024-07-01T12:00:00-08:00",
            "2024-01-15T12:00:00-09:00",
            "2024-07-01T12:00:00+05:30",
        ]
    )
    _check_utc(
        tz.to_utc(stamps), "2024-07-01 20:00", "2024-01-15 21:00", "2024-07-01 06:30"
    )


def test_mixed_offsets_with_naive_stamps_and_blanks():
    stamps = pd.Series(
        [
            "2024-07-01T12:00:00-08:00",
            "2024-07-01T12:00:00+05:30",
            "2024-07-01 12:00:00",
            None,
            "",
            "garbage",
        ]
    )
    with _record() as caught:
        out = tz.to_utc(stamps, {"utc_offset_h": -9})
    _check_utc(
        out,
        "2024-07-01 20:00",
        "2024-07-01 06:30",
        "2024-07-01 21:00",
        None,
        None,
        None,
    )
    # both aware stamps disagree with the declared UTC-09:00, and are listed once
    (message,) = _messages(caught, "carry their own")
    assert "UTC-08:00" in message and "UTC+05:30" in message


def test_iso_variations_in_one_column_are_not_lost():
    """Pandas reads the first element's format and drops the rest; these are rescued."""
    stamps = pd.Series(
        [
            "2024-07-01T12:00:00Z",
            "2024-07-01T12:00:00.500Z",
            "2024-07-01T12:00:00.123456Z",
        ]
    )
    out = tz.to_utc(stamps)
    _check_utc(
        out,
        "2024-07-01 12:00:00",
        "2024-07-01 12:00:00.5",
        "2024-07-01 12:00:00.123456",
    )


def test_a_date_alone_among_date_times_is_read_as_midnight():
    stamps = pd.Series(["2024-07-01 12:00:00", "2024-07-02", "2024-07-03 06:30:00"])
    _check_utc(
        tz.to_utc(stamps, AKDT),
        "2024-07-01 20:00",
        "2024-07-02 08:00",
        "2024-07-03 14:30",
    )


def test_junk_among_good_stamps_is_nat_and_does_not_disturb_the_rest():
    stamps = pd.Series(
        ["2024-07-01 12:00:00", "N/A", "2024-07-01 13:00:00", "--", "TBD"]
    )
    _check_utc(
        tz.to_utc(stamps, AKDT),
        "2024-07-01 20:00",
        None,
        "2024-07-01 21:00",
        None,
        None,
    )


def test_a_time_of_day_among_date_times_is_nat_not_todays_date():
    stamps = pd.Series(["2024-07-01 12:00:00", "12:00", "2024-07-01 13:00:00"])
    _check_utc(tz.to_utc(stamps), "2024-07-01 12:00", None, "2024-07-01 13:00")


def test_a_non_iso_format_is_read_when_the_whole_column_uses_it():
    stamps = pd.Series(["07/01/2024 12:00", "01/15/2024 12:00"])
    _check_utc(tz.to_utc(stamps, AKDT), "2024-07-01 20:00", "2024-01-15 21:00")


# -- to_utc: blanks, indexes, dtypes ------------------------------------------------


def test_missing_and_blank_cells_are_nat():
    stamps = pd.Series(["2024-07-01 12:00", None, np.nan, "", "   ", pd.NaT])
    _check_utc(
        tz.to_utc(stamps, AKDT), "2024-07-01 20:00", None, None, None, None, None
    )


@pytest.mark.parametrize(
    "values",
    [
        [None, np.nan, ""],
        [None],
        [""],
        [np.nan, np.nan],
        [pd.NaT, pd.NaT],
    ],
)
def test_a_column_of_nothing_is_all_nat_but_still_tz_aware_utc(values):
    out = tz.to_utc(pd.Series(values), AKDT)
    _check_utc(out, *([None] * len(values)))


def test_an_empty_input_is_an_empty_tz_aware_utc_series():
    for empty in (
        pd.Series([], dtype=object),
        [],
        np.array([], dtype="datetime64[ns]"),
    ):
        out = tz.to_utc(empty, AKDT)
        assert len(out) == 0
        _check_utc(out)


def test_an_unparseable_cell_is_nat_never_an_error():
    stamps = pd.Series(["not a time", "2024-13-45", "2024-07-01 12:00"])
    _check_utc(tz.to_utc(stamps, AKDT), None, None, "2024-07-01 20:00")


def test_the_series_index_and_name_are_kept():
    stamps = pd.Series(
        ["2024-07-01 12:00", "bad", "2024-01-15 12:00"],
        index=pd.Index(["a", "b", "c"], name="station"),
        name="time",
    )
    out = tz.to_utc(stamps, AKDT)
    assert out.index.equals(stamps.index)
    assert out.index.name == "station"
    assert out.name == "time"
    _check_utc(out, "2024-07-01 20:00", None, "2024-01-15 21:00")


def test_a_non_unique_or_unsorted_index_does_not_confuse_the_rest():
    stamps = pd.Series(
        ["2024-07-01 12:00", "junk", "2024-07-01T13:00:00Z", "2024-07-01 14:00"],
        index=[5, 5, 1, 1],
    )
    with _record():  # the Z stamp disagrees with the declared zone
        out = tz.to_utc(stamps, AKDT)
    assert out.index.tolist() == [5, 5, 1, 1]
    _check_utc(out, "2024-07-01 20:00", None, "2024-07-01 13:00", "2024-07-01 22:00")


def test_a_filtered_series_with_a_gappy_index_keeps_its_labels():
    stamps = pd.Series(["x", "2024-07-01 12:00", "y", "2024-07-01 13:00"])
    kept = stamps[stamps.str.startswith("2024")]
    out = tz.to_utc(kept, AKDT)
    assert out.index.tolist() == [1, 3]
    _check_utc(out, "2024-07-01 20:00", "2024-07-01 21:00")


@pytest.mark.parametrize(
    "make",
    [
        lambda s: list(s),
        lambda s: tuple(s),
        lambda s: np.array(list(s), dtype=object),
        lambda s: np.array(list(s)),
        lambda s: pd.Index(list(s)),
    ],
    ids=["list", "tuple", "object-array", "str-array", "index"],
)
def test_a_plain_sequence_gets_a_series_with_a_range_index(make):
    out = tz.to_utc(make(["2024-07-01 12:00", None, "2024-01-15 12:00"]), AKDT)
    assert isinstance(out, pd.Series)
    assert isinstance(out.index, pd.RangeIndex)
    _check_utc(out, "2024-07-01 20:00", None, "2024-01-15 21:00")


def test_a_lone_stamp_is_a_one_element_series():
    _check_utc(tz.to_utc("2024-07-01 12:00", AKDT), "2024-07-01 20:00")
    _check_utc(tz.to_utc(pd.Timestamp("2024-07-01 12:00"), AKDT), "2024-07-01 20:00")
    _check_utc(tz.to_utc(None, AKDT), None)


@pytest.mark.parametrize("dtype", [object, "string", "category"])
def test_string_like_dtypes_all_parse(dtype):
    stamps = pd.Series(["2024-07-01 12:00", None, "2024-01-15 12:00"], dtype=dtype)
    _check_utc(tz.to_utc(stamps, AKDT), "2024-07-01 20:00", None, "2024-01-15 21:00")


@pytest.mark.parametrize(
    "unit", ["s", "ms", "us", "ns"], ids=lambda u: f"datetime64[{u}]"
)
def test_naive_datetime64_in_every_resolution_is_localised(unit):
    naive = pd.Series(
        np.array(
            ["2024-07-01T12:00", "2024-01-15T12:00", "NaT"], dtype=f"datetime64[{unit}]"
        )
    )
    _check_utc(tz.to_utc(naive, AKDT), "2024-07-01 20:00", "2024-01-15 21:00", None)


def test_naive_datetime64_arrays_and_indexes_are_localised_too():
    values = ["2024-07-01 12:00", "2024-01-15 12:00"]
    array = np.array(values, dtype="datetime64[ns]")
    _check_utc(tz.to_utc(array, AKDT), "2024-07-01 20:00", "2024-01-15 21:00")
    _check_utc(
        tz.to_utc(pd.DatetimeIndex(values), AKDT),
        "2024-07-01 20:00",
        "2024-01-15 21:00",
    )
    _check_utc(
        tz.to_utc(pd.Series(array), {"utc_offset_h": -9}),
        "2024-07-01 21:00",
        "2024-01-15 21:00",
    )


def test_python_datetimes_dates_and_timestamps_are_accepted():
    out = tz.to_utc(
        [
            dt.datetime(2024, 7, 1, 12),
            dt.date(2024, 7, 2),
            pd.Timestamp("2024-07-03 12:00"),
        ],
        AKDT,
    )
    _check_utc(out, "2024-07-01 20:00", "2024-07-02 08:00", "2024-07-03 20:00")


def test_numbers_are_read_as_pandas_always_read_them_nanoseconds_since_the_epoch():
    out = tz.to_utc(pd.Series([1_700_000_000_000_000_000]))
    _check_utc(out, "2023-11-14 22:13:20")


@pytest.mark.parametrize(
    "stamps",
    [
        pd.Series(["2024-07-01 12:00"]),
        pd.Series(["2024-07-01T12:00:00Z"]),
        pd.Series(pd.to_datetime(["2024-07-01 12:00"])),
        pd.Series(pd.to_datetime(["2024-07-01 12:00"])).dt.tz_localize("Asia/Tokyo"),
        pd.Series(["2024-07-01T12:00:00Z", "2024-07-01 12:00:00"]),
        pd.Series(["2024-07-01T12:00:00-08:00", "2024-07-01T12:00:00+05:30"]),
        pd.Series([None, None], dtype=object),
        ["2024-07-01 12:00"],
    ],
)
def test_the_result_is_always_a_tz_aware_utc_series(stamps):
    with (
        _record()
    ):  # aware inputs that disagree with the declared zone warn; not the point
        out = tz.to_utc(stamps, AKDT)
    assert isinstance(out, pd.Series)
    assert isinstance(out.dtype, pd.DatetimeTZDtype)
    assert str(out.dtype.tz) == "UTC"
    assert len(out) == len(stamps)


def test_the_input_is_never_mutated():
    stamps = pd.Series(["2024-07-01 12:00", "junk"], index=["a", "b"], name="t")
    before = stamps.copy()
    tz.to_utc(stamps, AKDT)
    pd.testing.assert_series_equal(stamps, before)
    array = np.array(["2024-07-01T12:00"], dtype="datetime64[ns]")
    tz.to_utc(array, AKDT)
    assert (
        array.tolist()
        == np.array(["2024-07-01T12:00"], dtype="datetime64[ns]").tolist()
    )


def test_one_dimensional_input_only():
    err = _error(tz.to_utc, np.array([["2024-07-01 12:00"]]), AKDT)
    assert "one-dimensional" in str(err)


def test_an_invalid_declaration_raises_instead_of_being_ignored():
    _error(tz.to_utc, ["2024-07-01 12:00"], {"time_zone": "Nowhere/Land"})
    _error(tz.to_utc, ["2024-07-01 12:00"], {"utc_offset_h": 0.3})
    _error(tz.to_utc, ["2024-07-01 12:00"], {"time_zone": "UTC", "utc_offset_h": 0})


def test_a_spelled_differently_declaration_is_read_like_the_canonical_one():
    stamps = pd.Series(["2024-07-01 12:00"])
    assert tz.to_utc(stamps, {"time_zone": "america/anchorage"}).equals(
        tz.to_utc(stamps, AKDT)
    )


# -- to_utc: daylight-saving transitions --------------------------------------------


def test_a_chronological_record_through_the_repeated_hour_is_read_correctly():
    with _record() as caught:
        out = tz.to_utc(pd.Series(FALL_BACK), AKDT)
    assert not caught
    _check_utc(out, *FALL_BACK_UTC)
    assert out.is_monotonic_increasing  # the clock went back; time did not


def test_the_repeated_hour_is_inferred_for_datetime64_input_too():
    naive = pd.Series(pd.to_datetime(FALL_BACK))
    with _record() as caught:
        out = tz.to_utc(naive, AKDT)
    assert not caught
    _check_utc(out, *FALL_BACK_UTC)


def test_a_record_with_several_years_of_transitions_is_inferred_for_each():
    second_year = [s.replace("2024-11-03", "2023-11-05") for s in FALL_BACK]
    out = tz.to_utc(pd.Series(second_year + FALL_BACK), AKDT)
    assert out.notna().all()
    assert out.is_monotonic_increasing


def test_a_lone_ambiguous_reading_is_nat_with_a_warning():
    """01:30 happened twice; one reading cannot say which pass it was."""
    stamps = pd.Series(["2024-11-03 00:00", "2024-11-03 01:30", "2024-11-03 02:30"])
    with _record() as caught:
        out = tz.to_utc(stamps, AKDT, subject="station X")
    _check_utc(out, "2024-11-03 08:00", None, "2024-11-03 11:30")
    assert len(caught) == 1
    assert issubclass(caught[0].category, UserWarning)
    message = str(caught[0].message)
    assert message.startswith("station X: 1 of 3 timestamp(s)")
    assert "America/Anchorage" in message
    assert "repeated" in message
    assert "set to NaT" in message
    assert "utc_offset_h" in message  # how to read a standard-time record instead


def test_a_reversed_record_cannot_be_inferred_and_loses_the_ambiguous_hour():
    with _record() as caught:
        out = tz.to_utc(pd.Series(FALL_BACK[::-1]), AKDT)
    assert out.isna().sum() == 4  # 01:00 and 01:30, both passes
    assert "4 of 8" in _messages(caught)[0]
    # the readings outside the repeated hour are still right
    assert out.iloc[-1] == pd.Timestamp("2024-11-03 08:00", tz="UTC")


def test_a_reading_in_the_skipped_hour_is_nat_with_a_warning():
    """02:30 never happened on 2024-03-10: the clocks jumped from 02:00 to 03:00."""
    stamps = pd.Series(["2024-03-10 01:30", "2024-03-10 02:30", "2024-03-10 03:30"])
    with _record() as caught:
        out = tz.to_utc(stamps, AKDT)
    _check_utc(out, "2024-03-10 10:30", None, "2024-03-10 11:30")
    assert len(caught) == 1
    message = str(caught[0].message)
    assert "1 of 3 timestamp(s)" in message
    assert "America/Anchorage" in message
    assert "skipped" in message


def test_both_losses_in_one_call_make_one_warning_with_both_counts():
    stamps = pd.Series(
        ["2024-03-10 02:30", "2024-03-10 02:45", "2024-11-03 01:30", "2024-07-01 12:00"]
    )
    with _record() as caught:
        out = tz.to_utc(stamps, AKDT)
    assert out.isna().sum() == 3
    messages = _messages(caught)
    assert len(messages) == 1
    assert "3 of 4" in messages[0]
    assert "2 fall in the hour the clocks skipped" in messages[0]
    assert "1 fall in the hour the clocks repeated" in messages[0]


def test_a_fixed_offset_has_no_transitions_so_it_never_loses_anything():
    stamps = pd.Series(["2024-03-10 02:30", "2024-11-03 01:30", *FALL_BACK])
    with _record() as caught:
        out = tz.to_utc(stamps, {"utc_offset_h": -9})
    assert not caught
    assert out.notna().all()


def test_utc_has_no_transitions_either():
    with _record() as caught:
        out = tz.to_utc(pd.Series(["2024-03-10 02:30", "2024-11-03 01:30"]), None)
    assert not caught
    assert out.notna().all()


def test_the_dst_warning_points_at_the_caller():
    with _record() as caught:
        tz.to_utc(pd.Series(["2024-03-10 02:30"]), AKDT)
    assert Path(caught[0].filename).name == Path(__file__).name


def test_the_conflict_warning_points_at_the_caller():
    with _record() as caught:
        tz.to_utc(pd.Series(["2024-07-01T12:00:00Z"]), AKDT)
    assert Path(caught[0].filename).name == Path(__file__).name


def test_a_dst_loss_and_an_offset_conflict_are_two_separate_warnings():
    stamps = pd.Series(["2024-03-10 02:30", "2024-07-01T12:00:00Z", "2024-07-01 12:00"])
    with _record() as caught:
        out = tz.to_utc(stamps, AKDT)
    _check_utc(out, None, "2024-07-01 12:00", "2024-07-01 20:00")
    assert len(_messages(caught, "cannot be placed")) == 1
    assert len(_messages(caught, "carry their own")) == 1
    assert len(caught) == 2


def test_stamps_just_outside_the_transition_windows_are_untouched():
    stamps = pd.Series(
        [
            "2024-03-10 01:59",  # last minute of AKST
            "2024-03-10 03:00",  # first minute of AKDT
            "2024-11-03 00:59",  # AKDT, before the repeat
            "2024-11-03 02:00",  # AKST, after the repeat
        ]
    )
    with _record() as caught:
        out = tz.to_utc(stamps, AKDT)
    assert not caught
    _check_utc(
        out,
        "2024-03-10 10:59",
        "2024-03-10 11:00",
        "2024-11-03 08:59",
        "2024-11-03 11:00",
    )


# -- localize_naive_datetime64 ------------------------------------------------------


def test_a_naive_coordinate_is_read_as_local_and_returned_as_naive_utc():
    values = np.array(["2024-07-01T12:00", "2024-01-15T12:00"], dtype="datetime64[ns]")
    out = tz.localize_naive_datetime64(values, AKDT)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.dtype("datetime64[ns]")
    assert (
        out.tolist()
        == np.array(
            ["2024-07-01T20:00", "2024-01-15T21:00"], dtype="datetime64[ns]"
        ).tolist()
    )


@pytest.mark.parametrize(
    "offset, expected",
    [
        (-8, "2024-07-01T20:00"),
        (-9, "2024-07-01T21:00"),
        (5.5, "2024-07-01T06:30"),
        (9, "2024-07-01T03:00"),
    ],
)
def test_the_offset_sign_is_the_same_for_a_datetime64_coordinate(offset, expected):
    out = tz.localize_naive_datetime64(
        np.array(["2024-07-01T12:00"], dtype="datetime64[ns]"), {"utc_offset_h": offset}
    )
    assert out.tolist() == np.array([expected], dtype="datetime64[ns]").tolist()


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_the_output_is_nanoseconds_whatever_the_input_resolution(unit):
    values = np.array(["2024-07-01T12:00", "NaT"], dtype=f"datetime64[{unit}]")
    out = tz.localize_naive_datetime64(values, AKDT)
    assert out.dtype == np.dtype("datetime64[ns]")
    assert out[0] == np.datetime64("2024-07-01T20:00", "ns")
    assert np.isnat(out[1])


def test_nothing_declared_or_utc_leaves_the_values_alone():
    values = np.array(["2024-07-01T12:00", "NaT"], dtype="datetime64[ns]")
    for meta in (None, {}, {"time_zone": "UTC"}, {"utc_offset_h": 0}):
        with _record() as caught:
            out = tz.localize_naive_datetime64(values, meta)
        assert not caught
        assert out.dtype == values.dtype
        np.testing.assert_array_equal(out, values)


def test_a_multi_dimensional_coordinate_keeps_its_shape():
    values = np.array(
        [["2024-07-01T12:00", "2024-01-15T12:00"], ["NaT", "2024-07-02T00:00"]],
        dtype="datetime64[ns]",
    )
    out = tz.localize_naive_datetime64(values, AKDT)
    assert out.shape == (2, 2)
    assert out[0, 0] == np.datetime64("2024-07-01T20:00", "ns")
    assert out[0, 1] == np.datetime64("2024-01-15T21:00", "ns")
    assert np.isnat(out[1, 0])
    assert out[1, 1] == np.datetime64("2024-07-02T08:00", "ns")


def test_the_localised_coordinate_agrees_with_to_utc():
    values = pd.date_range("2024-01-01", "2024-12-31", freq="6h")
    mine = tz.localize_naive_datetime64(values.to_numpy(), {"utc_offset_h": -9})
    # a fixed offset has no transitions to disagree about
    theirs = tz.to_utc(pd.Series(values), {"utc_offset_h": -9}).dt.tz_convert(None)
    np.testing.assert_array_equal(mine, theirs.to_numpy())


def test_a_coordinate_with_dst_losses_warns_once_with_the_subject():
    values = np.array(
        ["2024-03-10T02:30", "2024-07-01T12:00", "2024-11-03T01:30"],
        dtype="datetime64[ns]",
    )
    with _record() as caught:
        out = tz.localize_naive_datetime64(values, AKDT, subject="the model")
    assert np.isnat(out[0]) and np.isnat(out[2])
    assert out[1] == np.datetime64("2024-07-01T20:00", "ns")
    assert len(caught) == 1
    message = str(caught[0].message)
    assert message.startswith("the model: 2 of 3 timestamp(s)")
    assert "America/Anchorage" in message
    assert Path(caught[0].filename).name == Path(__file__).name


def test_a_repeated_hour_in_a_chronological_coordinate_is_read_correctly():
    values = pd.to_datetime(FALL_BACK).to_numpy()
    with _record() as caught:
        out = tz.localize_naive_datetime64(values, AKDT)
    assert not caught
    np.testing.assert_array_equal(
        out, pd.to_datetime(FALL_BACK_UTC).to_numpy().astype("datetime64[ns]")
    )


def test_the_input_array_is_not_modified():
    values = np.array(["2024-07-01T12:00"], dtype="datetime64[ns]")
    tz.localize_naive_datetime64(values, AKDT)
    assert values[0] == np.datetime64("2024-07-01T12:00", "ns")


def test_times_that_are_not_datetime64_cannot_be_localised_not_silently_skipped():
    err = _error(tz.localize_naive_datetime64, np.array([1, 2], dtype=object), AKDT)
    assert "datetime64" in str(err)
    assert "America/Anchorage" in str(err)
    _error(tz.localize_naive_datetime64, np.array([1.0, 2.0]), AKDT)
    _error(tz.localize_naive_datetime64, np.array([1.0, 2.0]), {"utc_offset_h": -9})


def test_with_nothing_declared_any_dtype_passes_through_untouched():
    """A caller need not check the dtype first: no zone declared means no work."""
    cftime_like = np.array([object(), object()], dtype=object)
    for meta in (None, {}, {"time_zone": "UTC"}, {"utc_offset_h": 0}):
        assert tz.localize_naive_datetime64(cftime_like, meta) is cftime_like
        floats = np.array([1.0, 2.0])
        np.testing.assert_array_equal(
            tz.localize_naive_datetime64(floats, meta), floats
        )


def test_an_invalid_declaration_raises_here_too():
    values = np.array(["2024-07-01T12:00"], dtype="datetime64[ns]")
    _error(tz.localize_naive_datetime64, values, {"time_zone": "Nowhere/Land"})


# -- has_explicit_zone --------------------------------------------------------------


@pytest.mark.parametrize(
    "units",
    [
        "days since 1950-01-01T00:00:00Z",
        "hours since 2000-01-01 00:00:00 -09:00",
        "seconds since 1970-01-01 00:00:00 UTC",
        "seconds since 1970-01-01 00:00:00 GMT",
        "days since 1950-01-01 00:00:00 +05:30",
        "days since 1950-01-01T00:00:00+0530",
        "days since 2000-01-01T00:00:00-09",
        "days since 2000-01-01 00:00:00.5 -9:00",
        "hours since 2000-01-01 12:30Z",
        "days since 1950-01-01 00:00:00 Z",
        "days since 1950-01-01T00:00:00z",
        "days since 1950-01-01 00:00:00 utc",
        "days_since_1950-01-01T00:00:00Z",
        "days_since_1950-01-01_00:00:00_UTC",
        "Days Since 1950-01-01T00:00:00Z",
        "1950-01-01T00:00:00Z",
    ],
)
def test_a_reference_that_names_a_zone_has_an_explicit_zone(units):
    assert tz.has_explicit_zone(units) is True


@pytest.mark.parametrize(
    "units",
    [
        "days since 1950-01-01",  # the -01 is a day of the month, not an offset
        "days since 1950-1-1",
        "days since 1950-01-01 00:00:00",
        "days since 1950-01-01T00:00:00",
        "days since 1950-01-01 0:0:0",
        "hours since 2000-01-01 12:30",
        "seconds since 1970-01-01 00:00:00.000",
        "days_since_1950-01-01",
        "days since",
        "days",
        "degC",
        "",
    ],
)
def test_a_reference_with_no_zone_has_none(units):
    assert tz.has_explicit_zone(units) is False


@pytest.mark.parametrize("units", [None, 5, ["days since 1950-01-01T00:00:00Z"]])
def test_a_non_string_is_not_a_reference(units):
    assert tz.has_explicit_zone(units) is False
