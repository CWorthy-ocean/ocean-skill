"""Tests for :mod:`ocean_skill.qc` — provider QC flag recognition and application.

Covers detection/pairing, the named scheme registry and the consensus-adoption
rule, ``apply``'s masking, the probe-time contract resolution in
:mod:`ocean_skill.build`, a read-time end-to-end round trip, and the cache-key
folding in :mod:`ocean_skill.comparison`.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from ocean_skill import qc

# -- detect_flag_columns ----------------------------------------------------------


def test_name_and_value_shaped_flag_columns_are_detected():
    df = pd.DataFrame(
        {
            "TEMP_flag": [2, 3, 4, 9, 2],
            "Salinity_CTD_flag": [2, 2, 2, 2, 2],
            "sea_water_oxygen_qc_agg": [1, 1, 1, 1, 1],
            "Temperature_CTD": np.linspace(10.0, 11.0, 5),
        }
    )
    assert set(qc.detect_flag_columns(df)) == {
        "TEMP_flag",
        "Salinity_CTD_flag",
        "sea_water_oxygen_qc_agg",
    }


def test_a_real_measurement_whose_name_merely_contains_qc_is_rejected_by_value():
    """``Temperature_qc[degree_C]``'s *name* matches; its real degC values don't."""
    df = pd.DataFrame({"Temperature_qc[degree_C]": [10.1, 11.2, 12.3, 13.4]})
    assert qc.detect_flag_columns(df) == []


def test_an_int_valued_column_with_no_flag_like_name_is_rejected_by_name():
    """``station_id`` never reaches the value check at all -- its name alone excludes it."""
    df = pd.DataFrame({"station_id": [1, 2, 3, 4, 5]})
    assert qc.detect_flag_columns(df) == []


def test_letter_coded_and_mixed_flag_values_are_still_detected():
    df = pd.DataFrame({"Salinity_qc": ["1", "1", "A", "9"]})
    assert qc.detect_flag_columns(df) == ["Salinity_qc"]


# -- pair_flags ---------------------------------------------------------------------


def test_exact_base_match_pairs_the_flag_to_its_data_column():
    df = pd.DataFrame(
        {
            "Salinity_CTD_flag": [2, 2],
            "Salinity_CTD": [35.0, 35.1],
            "Oxygen_CTD_flag": [2, 9],
            "Oxygen_CTD": [200.0, 201.0],
        }
    )
    pairs = qc.pair_flags(df, ["Salinity_CTD_flag", "Oxygen_CTD_flag"])
    assert pairs == {
        "Salinity_CTD_flag": "Salinity_CTD",
        "Oxygen_CTD_flag": "Oxygen_CTD",
    }


def test_unique_prefix_match_pairs_temp_flag_to_temperature_ctd():
    df = pd.DataFrame({"TEMP_flag": [2, 3], "Temperature_CTD": [10.0, 11.0]})
    assert qc.pair_flags(df, ["TEMP_flag"]) == {"TEMP_flag": "Temperature_CTD"}


def test_an_ambiguous_prefix_is_left_unpaired_with_a_warning():
    df = pd.DataFrame(
        {
            "TEMP_flag": [2, 3],
            "Temperature_A": [10.0, 11.0],
            "Temperature_B": [12.0, 13.0],
        }
    )
    with pytest.warns(UserWarning, match="could not pair"):
        pairs = qc.pair_flags(df, ["TEMP_flag"])
    assert pairs == {}


def test_pairs_argument_overrides_detection():
    df = pd.DataFrame(
        {
            "TEMP_flag": [2, 3],
            "Temperature_A": [10.0, 11.0],
            "Temperature_B": [12.0, 13.0],
        }
    )
    pairs = qc.pair_flags(df, ["TEMP_flag"], {"TEMP_flag": "Temperature_B"})
    assert pairs == {"TEMP_flag": "Temperature_B"}


# -- SCHEMES / expand_scheme ---------------------------------------------------------


def test_expand_scheme_explicit_keys_override_the_registry_default():
    expanded = qc.expand_scheme(
        {"scheme": "woce_bottle", "flag_to_qartod": {6: "SUSPECT"}}
    )
    assert expanded["flag_to_qartod"][6] == "SUSPECT"  # overridden
    assert expanded["flag_to_qartod"][2] == "GOOD"  # registry default kept
    assert expanded["flag_definitions"][2] == "good"


def test_compatible_schemes_includes_woce_bottle_for_good_and_missing():
    assert "woce_bottle" in qc.compatible_schemes({2, 9})


def test_qartod_itself_is_never_a_compatible_scheme_candidate():
    """Qartod is the canonical *output* scale, not a provider convention to guess."""
    assert "qartod" not in qc.compatible_schemes({1, 2, 3, 4, 9})


# -- the consensus rule ---------------------------------------------------------------


def test_unambiguous_values_are_adopted_with_a_warning_naming_the_assumption():
    adopted, candidates, message = qc._consensus({2, 9})
    assert adopted == {2: "GOOD", 9: "MISSING"}
    assert "woce_bottle" in candidates
    assert "adopted flag mapping" in message
    assert "consensus of" in message
    assert "qc={'scheme'" in message  # names the override


def test_disagreeing_values_are_not_adopted_and_the_warning_names_the_disagreement():
    adopted, candidates, message = qc._consensus({1, 2})
    assert adopted is None
    assert set(candidates) >= {"argo", "woce_ctd"}
    assert "disagree" in message
    assert "argo" in message and "woce_ctd" in message


def test_probe_adopts_consensus_and_records_scheme_consensus():
    df = pd.DataFrame(
        {
            "Time[UTC]": pd.date_range("2024-01-01", periods=4, freq="D").astype(str),
            "Temperature[degC]": [10.0, 10.5, 11.0, 11.5],
            "Temperature_flag": [2, 9, 2, 9],
        }
    )
    with pytest.warns(UserWarning, match="adopted flag mapping"):
        contract = qc.resolve_contract({}, df)
    assert contract["scheme"] == "consensus"
    assert contract["flag_to_qartod"] == {2: "GOOD", 9: "MISSING"}
    assert contract["keep"] == ["GOOD"]


def test_probe_does_not_adopt_a_disagreeing_scheme():
    df = pd.DataFrame(
        {
            "Time[UTC]": pd.date_range("2024-01-01", periods=2, freq="D").astype(str),
            "Temperature[degC]": [10.0, 10.5],
            "Temperature_flag": [1, 2],
        }
    )
    with pytest.warns(UserWarning, match="disagree"):
        contract = qc.resolve_contract({}, df)
    assert "scheme" not in contract
    assert "flag_to_qartod" not in contract
    assert contract["flags"] == {"Temperature_flag": "Temperature[degC]"}


def test_flags_with_no_resolvable_scheme_are_recorded_but_warn_not_applied():
    df = pd.DataFrame({"Weird_flag": ["Z"], "Weird": [1.0]})
    # "Z" is not covered by any registered scheme at all -- compatible_schemes([])
    with pytest.warns(UserWarning, match="not fully covered|NOT applied"):
        contract = qc.resolve_contract({}, df)
    assert contract["flags"] == {"Weird_flag": "Weird"}
    assert "flag_to_qartod" not in contract


def test_a_declared_scheme_warns_about_out_of_scheme_observed_values():
    df = pd.DataFrame(
        {
            "Temperature_flag": [2, 7],  # 7 is not a woce_bottle code
            "Temperature": [10.0, 11.0],
        }
    )
    with pytest.warns(UserWarning, match="not covered by scheme"):
        qc.resolve_contract({"scheme": "woce_bottle"}, df)


# -- spec_from_declared_flags (declared metadata, above consensus) -------------------

#: Station Papa's own OceanSITES attrs, as declared at
#: https://data.pmel.noaa.gov/pmel/erddap/info/papa_hourly_temp/index.csv -- the
#: motivating case: {1, 2, 9} alone is ambiguous under the registry (woce_ctd reads
#: 1 as SUSPECT; argo/seadatanet read it GOOD), but the full declared set identifies
#: the scheme outright.
_PAPA_MEANINGS = (
    "no_qc_performed good_data probably_good_data "
    "bad_data_that_are_potentially_correctable bad_data value_changed nominal_value "
    "interpolated_value missing_value"
)


def test_a_named_conventions_string_is_matched_first_and_does_not_warn():
    declared = {
        "TEMP_QC": {
            "flag_values": [0, 1, 2, 3, 4, 5, 7, 8, 9],
            "flag_meanings": _PAPA_MEANINGS,
            "conventions": "OceanSITES reference table 2",
        }
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # an exact conventions match needs no guess
        spec = qc.spec_from_declared_flags(declared, subject="papa_hourly_temp")
    assert spec["scheme"] == "argo"
    assert "conventions" in spec["scheme_source"]
    # the provider's own wording is kept verbatim, not the registry's paraphrase
    assert spec["flag_definitions"][1] == "good_data"


def test_a_declared_flag_values_set_that_exactly_matches_one_scheme_is_adopted():
    """No conventions string at all -- the value set alone is unique to argo."""
    declared = {
        "TEMP_QC": {
            "flag_values": "0, 1, 2, 3, 4, 5, 7, 8, 9",  # ERDDAP's comma-string form
            "flag_meanings": _PAPA_MEANINGS,
        }
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        spec = qc.spec_from_declared_flags(declared)
    assert spec["scheme"] == "argo"
    assert "flag_values" in spec["scheme_source"]


def test_flag_meanings_are_read_onto_qartod_as_a_last_resort_and_warn():
    """{1, 2, 9} alone matches no registered scheme's key set exactly.

    Neither tier 1 nor tier 2 fires, so the declared wording itself is read,
    loudly.
    """
    declared = {
        "TEMP_QC": {
            "flag_values": [1, 2, 9],
            "flag_meanings": "good_data probably_good_data missing_value",
        }
    }
    with pytest.warns(UserWarning, match="declared flag_meanings"):
        spec = qc.spec_from_declared_flags(declared, subject="some_dataset")
    assert spec["scheme"] == "declared"
    assert spec["flag_to_qartod"] == {1: "GOOD", 2: "GOOD", 9: "MISSING"}
    assert spec["scheme_source"] == "declared: flag_meanings read onto QARTOD"


def test_conflicting_declarations_across_flag_columns_adopt_nothing():
    declared = {
        "TEMP_QC": {
            "flag_values": [0, 1, 2, 3, 4, 5, 7, 8, 9],
            "flag_meanings": _PAPA_MEANINGS,
            "conventions": "OceanSITES reference table 2",
        },
        "OTHER_QC": {
            "flag_values": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, "A"],
            "flag_meanings": "no_qc good probably_good probably_bad bad changed "
            "below_detection in_excess interpolated missing uncertain",
            "conventions": "SeaDataNet L20",
        },
    }
    with pytest.warns(UserWarning, match="different flag conventions"):
        spec = qc.spec_from_declared_flags(declared, subject="mixed_dataset")
    assert spec is None


def test_no_declared_flag_metadata_at_all_returns_none_silently():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert qc.spec_from_declared_flags({}) is None
        assert qc.spec_from_declared_flags({"TEMP_QC": {}}) is None


# -- apply ------------------------------------------------------------------------


def _woce_frame():
    return pd.DataFrame(
        {
            "TEMP_flag": [2, 3, 4, 9, 2],
            "Temperature_CTD": [10.0, 11.0, 12.0, 13.0, 14.0],
        }
    )


def test_keep_good_masks_questionable_bad_and_missing_in_place():
    df = _woce_frame()
    contract = qc.resolve_contract({"scheme": "woce_bottle"}, df)
    out = qc.apply(df, {"qc": contract})
    assert out["Temperature_CTD"].tolist()[0] == pytest.approx(10.0)
    assert out["Temperature_CTD"].isna().tolist() == [False, True, True, True, False]
    # the flag column itself rides through unmasked
    assert out["TEMP_flag"].tolist() == [2, 3, 4, 9, 2]


def test_keep_provider_masks_by_the_raw_provider_value_instead():
    df = _woce_frame()
    contract = qc.resolve_contract({"scheme": "woce_bottle"}, df)
    out = qc.apply(df, {"qc": contract}, policy={"keep_provider": [2, 6]})
    assert out["Temperature_CTD"].isna().tolist() == [False, True, True, True, False]


def test_off_leaves_provider_values_completely_untouched():
    df = _woce_frame()
    contract = qc.resolve_contract({"scheme": "woce_bottle"}, df)
    out = qc.apply(df, {"qc": contract}, policy="off")
    assert out["Temperature_CTD"].tolist() == [10.0, 11.0, 12.0, 13.0, 14.0]


def test_fills_are_masked_even_with_no_flags_at_all():
    df = pd.DataFrame(
        {
            "time": pd.date_range("2020-01-01", periods=3),
            "lat": [27.5, 9999.0, 27.5],
            "Temperature": [10.0, 11.0, 9999.0],
        }
    )
    contract = qc.resolve_contract({"fill_values": [9999.0]}, df)
    out = qc.apply(df, {"qc": contract, "axes": {"T": "time"}})
    assert np.isnan(out["lat"].iloc[1])
    assert np.isnan(out["Temperature"].iloc[2])


def test_the_time_column_is_exempt_from_fill_masking():
    df = pd.DataFrame(
        {
            "time": [9999, 1, 2],  # a pathological but exact numeric collision
            "Temperature": [10.0, 11.0, 12.0],
        }
    )
    contract = qc.resolve_contract({"fill_values": [9999.0]}, df)
    out = qc.apply(df, {"qc": contract, "axes": {"T": "time"}})
    assert out["time"].tolist() == [9999, 1, 2]


def test_no_contract_at_all_is_a_true_no_op():
    df = _woce_frame()
    out = qc.apply(df, {})
    assert out is df  # identity, not just equality -- not even copied


def test_apply_records_the_effective_policy_on_attrs():
    df = _woce_frame()
    contract = qc.resolve_contract({"scheme": "woce_bottle"}, df)
    out = qc.apply(df, {"qc": contract})
    assert out.attrs["qc_applied"]["keep"] == ["GOOD"]


# -- probe (ocean_skill.build) --------------------------------------------------------


def _profile_frame():
    return pd.DataFrame(
        {
            "Time[UTC]": pd.date_range("2024-01-01", periods=5, freq="D").astype(str),
            "Latitude[degrees_north]": np.full(5, 27.5),
            "Longitude[degrees_east]": np.full(5, -96.5),
            "TEMP_flag": [2, 3, 4, 9, 2],
            "Temperature_CTD": np.linspace(10.0, 11.0, 5),
        }
    )


def test_probe_records_the_resolved_pairing_and_excludes_flags_from_variables():
    from ocean_skill import build

    md = build._probe_dataframe(
        _profile_frame(), qc={"scheme": "woce_bottle", "scheme_source": "test"}
    )
    assert md["qc"]["flags"] == {"TEMP_flag": "Temperature_CTD"}
    assert "TEMP_flag" not in md["variables"]
    assert md["variables"] == ["Temperature_CTD"]


def test_a_flag_column_whose_name_looks_like_an_axis_is_never_claimed_as_one():
    from ocean_skill import build

    df = pd.DataFrame(
        {
            "Time[UTC]": pd.date_range("2024-01-01", periods=3, freq="D").astype(str),
            "Pressure[dbar]": [10.0, 20.0, 30.0],
            "Pressure_flag": [2, 2, 2],
            "Temperature[degC]": [10.0, 10.5, 11.0],
        }
    )
    md = build._probe_dataframe(df, qc={"flags": ["Pressure_flag"]})
    assert md["axes"]["Z"] == "Pressure[dbar]"
    assert (md["geospatial_vertical_min"], md["geospatial_vertical_max"]) == (
        10.0,
        30.0,
    )


def test_probe_masks_fill_values_before_computing_extents():
    from ocean_skill import build

    df = pd.DataFrame(
        {
            "Time[UTC]": pd.date_range("2024-01-01", periods=4, freq="D").astype(str),
            "Latitude[degrees_north]": [27.5, 27.5, 27.5, 9999.0],
            "Longitude[degrees_east]": [-96.5, -96.5, -96.5, 9999.0],
            "Temperature[degC]": [10.0, 10.5, 11.0, 11.5],
        }
    )
    md = build._probe_dataframe(df, qc={"fill_values": [9999.0]})
    assert (md["geospatial_lat_min"], md["geospatial_lat_max"]) == (27.5, 27.5)
    assert (md["geospatial_lon_min"], md["geospatial_lon_max"]) == (-96.5, -96.5)
    assert md["featureType"] == "timeSeries"  # not misclassified as a trajectory


def test_a_dataframe_with_no_flag_like_columns_gets_no_qc_key_at_all():
    from ocean_skill import build

    df = pd.DataFrame(
        {
            "Time[UTC]": pd.date_range("2024-01-01", periods=3, freq="D").astype(str),
            "Temperature[degC]": [10.0, 10.5, 11.0],
        }
    )
    md = build._probe_dataframe(df)
    assert "qc" not in md


# -- read time (ocean_skill.sources.read / ocean_skill.build) ------------------------


@pytest.fixture
def flagged_source(tmp_path):
    """A tiny on-disk CSV catalog entry with a resolved woce_bottle contract."""
    import intake
    from intake.readers import datatypes, readers

    from ocean_skill.build import _attach
    from ocean_skill.catalog import SourceRef

    csv = tmp_path / "profile.csv"
    csv.write_text(
        "Time[UTC],Temperature[degC],Temperature_flag\n"
        "2024-01-01T00:00:00Z,10.0,2\n"
        "2024-01-02T00:00:00Z,10.5,3\n"
        "2024-01-03T00:00:00Z,11.0,4\n"
        "2024-01-04T00:00:00Z,11.5,9\n"
        "2024-01-05T00:00:00Z,12.0,2\n"
    )
    reader = readers.PandasCSV(datatypes.CSV(url=str(csv)))
    cat = intake.entry.Catalog()
    _attach(
        cat,
        "ctd_flagged",
        reader,
        probe=True,
        name_map=None,
        metadata={},
        qc={"scheme": "woce_bottle", "scheme_source": "test"},
    )
    path = tmp_path / "flagged.catalog.yaml"
    cat.to_yaml_file(str(path))
    return SourceRef(
        name="ctd_flagged",
        catalog="flagged",
        path=path,
        metadata=dict(cat["ctd_flagged"].metadata),
    )


def test_read_applies_the_contract_by_default(flagged_source):
    import ocean_skill as osk

    df = osk.read(flagged_source)
    masked = df["Temperature"].isna().tolist()
    assert masked == [False, True, True, True, False]


def test_read_with_qc_off_matches_the_raw_csv_exactly(flagged_source):
    import ocean_skill as osk

    df = osk.read(flagged_source, qc="off")
    assert df["Temperature"].tolist() == [10.0, 10.5, 11.0, 11.5, 12.0]


def test_default_and_off_differ_at_exactly_the_rejected_flag_rows(flagged_source):
    import ocean_skill as osk

    default = osk.read(flagged_source)["Temperature"]
    off = osk.read(flagged_source, qc="off")["Temperature"]
    differ = default.isna() & off.notna()
    assert differ.tolist() == [False, True, True, True, False]


# -- cache (ocean_skill.comparison.prepare_source) ------------------------------------


def _discoverable(monkeypatch, ref):
    """Make ``ref`` resolvable by its plain name, the way ``prepare_source``
    (unlike ``osk.read``) requires -- it always calls ``catalog.resolve(source)``,
    which needs a string, never a ``SourceRef`` handed to it directly.
    """
    from ocean_skill import catalog

    monkeypatch.setattr(catalog, "discover", lambda *a, **k: {ref.name: ref})
    return ref.name


def test_different_qc_policies_are_different_prepared_cache_keys(
    monkeypatch, flagged_source
):
    from ocean_skill import cache as _cache

    name = _discoverable(monkeypatch, flagged_source)

    seen: list[dict | None] = []
    original = _cache.key_for_prepared

    def spy(**kwargs):
        seen.append(kwargs["select"].get("_qc"))
        return original(**kwargs)

    monkeypatch.setattr(_cache, "key_for_prepared", spy)

    from ocean_skill.comparison import prepare_source

    prepare_source(name, "Temperature", None, None, use_cache=False, qc=None)
    prepare_source(
        name,
        "Temperature",
        None,
        None,
        use_cache=False,
        qc={"keep": ["GOOD", "SUSPECT"]},
    )
    assert seen[0] is not None  # a contract exists -> folded in even with qc=None
    assert seen[0] != seen[1]


def test_a_source_with_no_qc_contract_keys_exactly_as_before(monkeypatch, tmp_path):
    """No ``qc`` metadata at all -> ``_qc`` never enters the key -- byte-identical."""
    import intake
    from intake.readers import datatypes, readers

    from ocean_skill import cache as _cache
    from ocean_skill.build import _attach
    from ocean_skill.catalog import SourceRef

    csv = tmp_path / "plain.csv"
    csv.write_text("Time[UTC],Temperature[degC]\n2024-01-01T00:00:00Z,10.0\n")
    reader = readers.PandasCSV(datatypes.CSV(url=str(csv)))
    cat = intake.entry.Catalog()
    _attach(cat, "plain", reader, probe=True, name_map=None, metadata={})
    path = tmp_path / "plain.catalog.yaml"
    cat.to_yaml_file(str(path))
    source = SourceRef(
        name="plain", catalog="plain", path=path, metadata=dict(cat["plain"].metadata)
    )
    assert "qc" not in source.metadata
    name = _discoverable(monkeypatch, source)

    seen = []
    original = _cache.key_for_prepared

    def spy(**kwargs):
        seen.append(kwargs["select"])
        return original(**kwargs)

    monkeypatch.setattr(_cache, "key_for_prepared", spy)

    from ocean_skill.comparison import prepare_source

    prepare_source(name, "Temperature", None, None, use_cache=False)
    assert "_qc" not in seen[0]


# -- to_dataset flag attrs -------------------------------------------------------------


def test_to_dataset_carries_flag_attrs_and_encodes_letter_codes():
    from ocean_skill import tabular

    df = pd.DataFrame(
        {
            "Time[UTC]": pd.date_range(
                "2024-01-01", periods=4, freq="D", tz="UTC"
            ).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "Salinity[PSU]": [35.0, 35.1, 35.2, 35.3],
            "Salinity_qc": ["1", "1", "A", "9"],
        }
    )
    contract = qc.resolve_contract({"scheme": "seadatanet"}, df)
    meta = {"qc": contract, "nominal_depth_m": 0.0}
    applied = qc.apply(df, meta)
    ds = tabular.to_dataset(applied, meta)

    flag_var = ds["Salinity_qc"]
    assert "flag_values" in flag_var.attrs
    assert "flag_meanings" in flag_var.attrs
    assert "flag_qartod" in flag_var.attrs
    assert flag_var.attrs["flags_for"] == "Salinity"
    # the letter code "A" is encoded to QARTOD's SUSPECT int (3), not dropped
    assert 3 in flag_var.values.tolist()
    assert not np.isnan(flag_var.values).all()

    assert ds["Salinity"].attrs["ancillary_variables"] == "Salinity_qc"
    assert "qc_policy" in ds["Salinity"].attrs


# -- add_erddap_source: qc= threading and declared-flag auto-resolution --------------
#
# All network-free: intake_erddap.erddap.TableDAPReader is monkeypatched to a factory
# that returns a real PandasCSV reader over an on-disk fixture (the same reader kind
# tests/test_qc.py's own flagged_source fixture builds by hand), with
# _get_dataset_metadata monkeypatched onto that instance to stand in for the
# GET /info/<dataset_id>/index.json request add_erddap_source's own qc= resolution
# makes. Header/value shapes mirror the live PMEL Station Papa tables this was built
# against: a bare "<NAME>_QC" flag column (no units suffix -- a quality flag has none)
# paired to "<NAME> (<units>)", flags drawn from {1, 2, 9} -- exactly the set the
# registry disagrees about (woce_ctd reads 1 as SUSPECT; argo/seadatanet read it GOOD),
# so consensus alone cannot resolve it and these tests are not vacuous.

_PAPA_TEMP_VARIABLE_ATTRS = {
    "TEMP_QC": {
        "flag_values": [0, 1, 2, 3, 4, 5, 7, 8, 9],
        "flag_meanings": _PAPA_MEANINGS,
        "conventions": "OceanSITES reference table 2",
    }
}


def _papa_shaped_csv(tmp_path):
    csv = tmp_path / "papa_temp.csv"
    csv.write_text(
        "time (UTC),latitude (degrees_north),longitude (degrees_east),depth (m),"
        "TEMP (degree_Celsius),TEMP_QC\n"
        "2010-01-15T00:00:00Z,50.1,-144.9,1.0,6.401,1\n"
        "2010-01-15T01:00:00Z,50.1,-144.9,1.0,6.398,1\n"
        "2010-01-15T02:00:00Z,50.1,-144.9,1.0,6.395,2\n"
        "2010-01-15T03:00:00Z,50.1,-144.9,5.0,,9\n"
        "2010-01-15T04:00:00Z,50.1,-144.9,1.0,6.402,1\n"
        "2010-01-15T05:00:00Z,50.1,-144.9,1.0,6.400,2\n"
    )
    return csv


def _papa_two_flag_columns_csv(tmp_path):
    """Two data variables, each with its own QC column.

    For the conflicting-conventions case, which needs both flag columns
    actually present in the frame (detect_flag_columns only sees what the
    table carries).
    """
    csv = tmp_path / "papa_temp_psal.csv"
    csv.write_text(
        "time (UTC),latitude (degrees_north),longitude (degrees_east),depth (m),"
        "TEMP (degree_Celsius),TEMP_QC,PSAL (1e-3),PSAL_QC\n"
        "2010-01-15T00:00:00Z,50.1,-144.9,1.0,6.401,1,32.50,1\n"
        "2010-01-15T01:00:00Z,50.1,-144.9,1.0,6.398,2,32.50,2\n"
        "2010-01-15T02:00:00Z,50.1,-144.9,1.0,6.395,9,32.51,9\n"
    )
    return csv


def _fake_table_dap_reader_factory(csv_path, get_dataset_metadata):
    """Build a stand-in for ``intake_erddap.erddap.TableDAPReader``.

    Same call signature, but backed by an on-disk CSV and a caller-supplied
    ``_get_dataset_metadata`` (a plain callable, not necessarily a working
    ERDDAP client) rather than a live ERDDAP server.
    """
    from intake.readers import datatypes, readers

    def factory(
        server, dataset_id, *, variables=None, mask_failed_qartod=True, **kwargs
    ):
        reader = readers.PandasCSV(datatypes.CSV(url=str(csv_path)))
        reader._get_dataset_metadata = get_dataset_metadata
        return reader

    return factory


def _add_papa_temp(
    monkeypatch,
    tmp_path,
    *,
    variable_attrs=None,
    get_metadata=None,
    csv_factory=_papa_shaped_csv,
    **kw,
):
    """Call add_erddap_source against a Papa-shaped fixture.

    intake_erddap.erddap.TableDAPReader is swapped for the fake factory above.
    Exactly one of ``variable_attrs`` (the normal case: a plain {var: attrs}
    dict) or ``get_metadata`` (a custom callable, e.g. one that raises) may be
    given.
    """
    import intake

    from ocean_skill.build import add_erddap_source

    assert (variable_attrs is None) != (get_metadata is None)
    if get_metadata is None:
        variable_attrs = dict(variable_attrs)
        get_metadata = lambda server, dataset_id: {"variables": variable_attrs}  # noqa: E731

    csv = csv_factory(tmp_path)
    monkeypatch.setattr(
        "intake_erddap.erddap.TableDAPReader",
        _fake_table_dap_reader_factory(csv, get_metadata),
    )
    cat = intake.entry.Catalog()
    reader = add_erddap_source(
        cat,
        "papa_temp",
        server="https://data.pmel.noaa.gov/pmel/erddap",
        dataset_id="papa_hourly_temp",
        featureType="timeSeriesProfile",
        mask_failed_qartod=False,
        **kw,
    )
    return reader.metadata


def _no_warning_mentions(records, *needles) -> bool:
    return not any(
        all(n.casefold() in str(r.message).casefold() for n in needles)
        for r in records
    )


def test_an_explicit_scheme_is_applied_and_skips_the_info_request(
    monkeypatch, tmp_path
):
    def _must_not_be_called(server, dataset_id):
        raise AssertionError(
            "declared-flag lookup must be skipped when qc names a scheme"
        )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        md = _add_papa_temp(
            monkeypatch,
            tmp_path,
            get_metadata=_must_not_be_called,
            qc={"scheme": "argo"},
        )
    assert md["qc"]["scheme"] == "argo"
    assert md["qc"]["flag_to_qartod"][1] == "GOOD"
    assert md["qc"]["flag_to_qartod"][2] == "GOOD"
    assert md["qc"]["flag_to_qartod"][9] == "MISSING"
    assert md["qc"]["keep"] == ["GOOD"]
    assert _no_warning_mentions(caught, "scheme")
    assert _no_warning_mentions(caught, "flag")


def test_declared_papa_attrs_are_auto_resolved_with_no_qc_argument(
    monkeypatch, tmp_path
):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        md = _add_papa_temp(
            monkeypatch, tmp_path, variable_attrs=_PAPA_TEMP_VARIABLE_ATTRS, qc=None
        )
    assert md["qc"]["scheme"] == "argo"
    assert "conventions" in md["qc"]["scheme_source"]
    assert md["qc"]["flag_to_qartod"][1] == "GOOD"
    assert md["qc"]["flag_to_qartod"][9] == "MISSING"
    assert md["qc"]["keep"] == ["GOOD"]
    assert _no_warning_mentions(caught, "scheme")
    assert _no_warning_mentions(caught, "flag")


def test_with_no_declared_attrs_the_old_consensus_fallback_still_applies(
    monkeypatch, tmp_path
):
    """Today's behaviour, unchanged.

    ERDDAP declares nothing usable, the flags are still detected and paired,
    but {1, 2, 9} defeats consensus.
    """
    with pytest.warns(UserWarning, match="disagree"):
        md = _add_papa_temp(monkeypatch, tmp_path, variable_attrs={}, qc=None)
    assert md["qc"]["flags"] == {"TEMP_QC": "TEMP (degree_Celsius)"}
    assert "scheme" not in md["qc"]
    assert "flag_to_qartod" not in md["qc"]


def test_an_explicit_qc_argument_wins_over_declared_attrs(monkeypatch, tmp_path):
    """The declared attrs alone would resolve to argo.

    An explicit, different scheme still overrides them, same as
    expand_scheme's own override rule.
    """
    md = _add_papa_temp(
        monkeypatch,
        tmp_path,
        variable_attrs=_PAPA_TEMP_VARIABLE_ATTRS,
        qc={"scheme": "woce_ctd"},
    )
    assert md["qc"]["scheme"] == "woce_ctd"


def test_a_failed_info_request_warns_and_falls_back_without_failing_the_entry(
    monkeypatch, tmp_path
):
    def _broken(server, dataset_id):
        raise ConnectionError("no route to host")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        md = _add_papa_temp(monkeypatch, tmp_path, get_metadata=_broken, qc=None)
    assert any(
        "could not read erddap" in str(r.message).casefold()
        and "declared flag" in str(r.message).casefold()
        for r in caught
    )
    # the entry itself is still usable, and the flag column still recognized
    assert md["featureType"] == "timeSeriesProfile"
    assert md["qc"]["flags"] == {"TEMP_QC": "TEMP (degree_Celsius)"}
    assert "scheme" not in md["qc"]


def test_flag_columns_declaring_conflicting_conventions_synthesize_nothing(
    monkeypatch, tmp_path
):
    """Two data variables in one table, each with its own QC column.

    Each declares a different convention -- the builder-level counterpart of
    test_conflicting_declarations_across_flag_columns_adopt_nothing above: no
    synthesized scheme is handed to resolve_contract, and the conflict is
    still named in a warning even when reached through add_erddap_source.
    """
    variable_attrs = {
        "TEMP_QC": _PAPA_TEMP_VARIABLE_ATTRS["TEMP_QC"],
        "PSAL_QC": {
            "flag_values": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, "A"],
            "flag_meanings": "no_qc good probably_good probably_bad bad changed "
            "below_detection in_excess interpolated missing uncertain",
            "conventions": "SeaDataNet L20",
        },
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        md = _add_papa_temp(
            monkeypatch,
            tmp_path,
            variable_attrs=variable_attrs,
            qc=None,
            csv_factory=_papa_two_flag_columns_csv,
        )
    assert any(
        "different flag conventions" in str(r.message).casefold() for r in caught
    )
    assert "TEMP_QC" in md["qc"]["flags"]
    assert "PSAL_QC" in md["qc"]["flags"]
