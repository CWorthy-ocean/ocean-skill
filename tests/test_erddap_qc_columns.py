"""``add_erddap_source(variables=[...], mask_failed_qartod=True)`` must get the flags.

intake-erddap masks a column only when its ``<col>_qc_agg`` companion is in the
downloaded frame, and an explicit ``variables`` list names only what the caller
typed -- so asking for ``["sea_water_temperature"]`` brought no flags back and nothing
was masked (QARTOD FAIL spikes reached the AOOS Cook Inlet figures). The builder now
adds each variable's ``_qc_agg`` to the request. intake-erddap's own match then still
misses (a csvp header is ``"<name> (<units>)"``, the flag column's is bare), so the
masking is ocean-skill's own: the flag columns the probe saw are paired to their data
column by :mod:`ocean_skill.qc`.

Network-free: ``TableDAPReader`` is a stand-in over an on-disk CSV shaped like the
live AOOS ``nerrs_kach3wq`` response, returning only the columns it is asked for.
"""

from __future__ import annotations

import warnings

import pandas as pd
import pytest
from intake.readers import datatypes, readers

from ocean_skill import qc
from ocean_skill.build import add_erddap_source, new_catalog

_CSV = (
    "time (UTC),latitude (degrees_north),longitude (degrees_east),"
    "sea_water_temperature (degree_Celsius),sea_water_temperature_qc_agg\n"
    "2024-02-03T02:00:00Z,59.6,-151.4,-1.4,1\n"
    "2024-02-03T02:15:00Z,59.6,-151.4,-1.8,4\n"
    "2024-02-03T02:30:00Z,59.6,-151.4,-1.5,1\n"
    "2024-02-03T02:45:00Z,59.6,-151.4,-1.6,2\n"
    "2024-02-03T03:00:00Z,59.6,-151.4,-1.7,3\n"
    "2024-02-03T03:15:00Z,59.6,-151.4,-1.9,9\n"
)

#: What AOOS declares for each ``_qc_agg`` flag variable (``/info/<dataset_id>``).
_AOOS_FLAGS = {
    "flag_values": [1, 2, 3, 4, 9],
    "flag_meanings": "PASS NOT_EVALUATED SUSPECT FAIL MISSING",
}

_VARS = ["time", "latitude", "longitude", "sea_water_temperature"]


@pytest.fixture
def build_with(monkeypatch, tmp_path):
    """``build_with(variables=..., mask_failed_qartod=...)`` -> (reader, calls)."""
    csv = tmp_path / "kach.csv"
    csv.write_text(_CSV)
    calls: list[dict] = []
    declared: dict = {}

    def factory(
        server, dataset_id, *, variables=None, mask_failed_qartod=True, **kwargs
    ):
        calls.append({"variables": variables, "mask": mask_failed_qartod})
        # like the server, return only the columns asked for
        header = _CSV.split("\n", 1)[0].split(",")
        keep = [c for c in header if not variables or c.split(" (")[0] in variables]
        reader = readers.PandasCSV(datatypes.CSV(url=str(csv)), usecols=keep)
        reader._get_dataset_metadata = lambda s, d: {"variables": declared}
        return reader

    monkeypatch.setattr("intake_erddap.erddap.TableDAPReader", factory)

    def build(qc={"scheme": "qartod"}, flags_declared=False, **kw):
        if flags_declared:
            declared["sea_water_temperature_qc_agg"] = _AOOS_FLAGS
        reader = add_erddap_source(
            new_catalog(title="t"),
            "kach",
            server="https://erddap.aoos.org/erddap",
            dataset_id="nerrs_kach3wq",
            featureType="timeSeries",
            qc=qc,
            **kw,
        )
        return reader, calls

    return build


def test_an_explicit_variables_list_also_requests_the_qc_agg_columns(build_with):
    _, calls = build_with(variables=_VARS, mask_failed_qartod=True)

    assert calls[0]["variables"] == _VARS + [f"{v}_qc_agg" for v in _VARS]


def test_a_failed_sample_comes_back_nan_through_the_probed_contract(build_with):
    reader, _ = build_with(variables=_VARS, mask_failed_qartod=True)

    out = qc.apply(reader.read(), reader.metadata)

    col = "sea_water_temperature (degree_Celsius)"
    assert pd.isna(out[col].iloc[1])  # the FAIL (4) sample
    assert out[col].iloc[[0, 2]].tolist() == [-1.4, -1.5]  # the rest untouched


def test_the_request_is_left_alone_when_masking_is_off_or_nothing_is_listed(build_with):
    _, calls = build_with(variables=_VARS, mask_failed_qartod=False)
    assert calls[0]["variables"] == _VARS  # the caller opted out of flags

    _, calls = build_with(variables=None, mask_failed_qartod=True)
    assert calls[1]["variables"] is None  # "all of them" already includes the flags


def test_a_flag_the_caller_already_listed_is_not_requested_twice(build_with):
    listed = [*_VARS, "sea_water_temperature_qc_agg"]
    _, calls = build_with(variables=listed, mask_failed_qartod=True)

    sent = calls[0]["variables"]
    assert sent.count("sea_water_temperature_qc_agg") == 1
    assert "sea_water_temperature_qc_agg_qc_agg" not in sent


def _masked(reader):
    """Which of the six samples (flags 1, 4, 1, 2, 3, 9) read back NaN."""
    out = qc.apply(reader.read(), reader.metadata)
    return out["sea_water_temperature (degree_Celsius)"].isna().tolist()


def test_mask_failed_keeps_unevaluated_flag_2_and_masks_3_4_9(build_with):
    """"Mask failed" is intake-erddap's rule -- keep 1 and 2 -- not "keep only 1"."""
    reader, _ = build_with(variables=_VARS, mask_failed_qartod=True, qc=None,
                           flags_declared=True)

    assert reader.metadata["qc"]["keep"] == ["GOOD", "UNKNOWN"]
    assert _masked(reader) == [False, True, False, False, True, True]


def test_the_same_default_applies_under_an_explicit_scheme(build_with):
    reader, _ = build_with(variables=_VARS, mask_failed_qartod=True)

    assert _masked(reader) == [False, True, False, False, True, True]


def test_an_explicit_qc_keep_wins_over_the_mask_failed_default(build_with):
    reader, _ = build_with(
        variables=_VARS,
        mask_failed_qartod=True,
        qc={"scheme": "qartod", "keep": ["GOOD"]},
    )

    assert _masked(reader) == [False, True, False, True, True, True]


def test_without_mask_failed_the_generic_keep_good_stands(build_with):
    reader, _ = build_with(variables=None, mask_failed_qartod=False)

    assert reader.metadata["qc"]["keep"] == ["GOOD"]


def test_a_declared_pass_means_good_and_raises_no_uncovered_flag_warning():
    assert qc._qartod_for_meaning("PASS") == "GOOD"
    # the specific rows ahead of it are not displaced by the new needle
    assert qc._qartod_for_meaning("NOT_EVALUATED") == "UNKNOWN"
    assert qc._qartod_for_meaning("FAIL") == "FAIL"
    assert qc._qartod_for_meaning("bad_data") == "FAIL"
    assert qc._qartod_for_meaning("probably_good_data") == "GOOD"

    df = pd.DataFrame({"t_qc_agg": [1, 2, 3, 4, 9], "t": [1.0, 2.0, 3.0, 4.0, 5.0]})
    with pytest.warns(UserWarning, match="declared flag_meanings"):
        spec = qc.spec_from_declared_flags({"t_qc_agg": _AOOS_FLAGS})
    assert spec["flag_to_qartod"] == {
        1: "GOOD", 2: "UNKNOWN", 3: "SUSPECT", 4: "FAIL", 9: "MISSING",
    }  # fmt: skip

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no "not covered by scheme 'declared'"
        qc.resolve_contract(spec, df)
