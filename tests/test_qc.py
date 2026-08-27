"""Tests for :mod:`ocean_skill.qc` — provider QC flag recognition and application.

Covers detection/pairing, the named scheme registry and the consensus-adoption
rule, ``apply``'s masking, the probe-time contract resolution in
:mod:`ocean_skill.build`, a read-time end-to-end round trip, and the cache-key
folding in :mod:`ocean_skill.comparison`.
"""

from __future__ import annotations

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
    """qartod is the canonical *output* scale, not a provider convention to guess."""
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
