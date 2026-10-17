"""Source names are configuration; figures name them the way a reader would.

``woa23_nitrate_month12`` heads a panel as ``WOA23 nitrate Dec``; a catalog entry can
name itself with ``metadata: {label: ...}``; a model run or anything else already as
readable as it will get is left as written.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ocean_skill import catalog
from ocean_skill.catalog import source_label
from tests.test_profile_comparison import _profile_comparison


@pytest.mark.parametrize(
    "name, expected",
    [
        ("woa23_nitrate_month12", "WOA23 nitrate Dec"),
        ("woa23_phosphate_month01", "WOA23 phosphate Jan"),
        # one entry holding all twelve months: its name cannot say which a page draws
        ("woa23_temperature_monthly", "WOA23 temperature"),
        ("woa23_oxygen_annual", "WOA23 oxygen annual"),
    ],
)
def test_a_period_climatology_reads_dataset_variable_period(name, expected):
    assert source_label(name) == expected


@pytest.mark.parametrize(
    "name",
    [
        "ccs_4km",  # a model run
        "pacmed_pyesper",
        "a+b",  # a pooled name
        "woa23_nitrate_month13",  # not a month
        "woa23_notavariable_month04",  # the vocabulary does not know the variable
        "chl_gapfree_nrt_daily_wind",  # not an entry the package knows
    ],
)
def test_a_name_with_nothing_to_derive_is_left_as_written(name):
    assert source_label(name) == name


def test_an_entrys_own_label_wins(monkeypatch):
    monkeypatch.setattr(
        catalog,
        "resolve",
        lambda name: SimpleNamespace(metadata={"label": "Holte & Talley"}),
    )
    assert source_label("holte_talley_mld_clim") == "Holte & Talley"
    # even over a rule that would otherwise have spelled it
    assert source_label("woa23_nitrate_month12") == "Holte & Talley"


def test_a_blank_or_non_text_label_is_ignored(monkeypatch):
    monkeypatch.setattr(
        catalog, "resolve", lambda name: SimpleNamespace(metadata={"label": "  "})
    )
    assert source_label("woa23_nitrate_month12") == "WOA23 nitrate Dec"
    monkeypatch.setattr(
        catalog, "resolve", lambda name: SimpleNamespace(metadata={"label": 3})
    )
    assert source_label("ccs_4km") == "ccs_4km"


def test_the_packaged_glodap_entry_carries_a_label():
    assert source_label("glodap") == "GLODAPv2"


@pytest.mark.parametrize(
    "name, expected",
    [
        ("chl_gapfree_my_daily_geo", "Copernicus L4 chlorophyll"),
        ("holte_talley_mld_clim", "Holte & Talley"),
        ("glodap", "GLODAPv2"),
    ],
)
def test_the_standard_observation_entries_have_built_in_labels(
    monkeypatch, name, expected
):
    """A shadowing catalog's same-named entry (no label of its own) is still named."""
    monkeypatch.setattr(
        catalog, "resolve", lambda n: SimpleNamespace(metadata={"units": "m"})
    )
    assert source_label(name) == expected
    monkeypatch.setattr(
        catalog, "resolve", lambda n: (_ for _ in ()).throw(KeyError(n))
    )
    assert source_label(name) == expected  # not even resolvable here


def test_an_entrys_own_label_beats_the_built_in_one(monkeypatch):
    monkeypatch.setattr(
        catalog, "resolve", lambda n: SimpleNamespace(metadata={"label": "H and T"})
    )
    assert source_label("holte_talley_mld_clim") == "H and T"


def test_a_model_side_entry_is_named_by_the_label_its_suite_wrote(monkeypatch):
    monkeypatch.setattr(
        catalog,
        "resolve",
        lambda n: SimpleNamespace(metadata={"label": "initial conditions"}),
    )
    assert source_label("ccs_4km_ic") == "initial conditions"


def test_the_label_is_prose_and_never_changes_a_sources_fingerprint():
    assert "label" in catalog._FINGERPRINT_SKIP_KEYS


def test_a_comparisons_item_titles_its_columns_by_the_readable_names():
    c = _profile_comparison()
    c.reference_name = "woa23_temperature_month04"
    c.test_name = "ccs_4km"
    assert c.as_item()["labels"] == ("ccs_4km", "WOA23 temperature Apr")


def test_the_direct_and_set_plot_paths_use_the_readable_names(monkeypatch):
    from ocean_skill.comparison import ComparisonSet

    c = _profile_comparison()
    c.reference_name = "woa23_temperature_month04"
    c.test_name = "ccs_4km"
    captured = []
    monkeypatch.setattr(
        "ocean_skill.plot.registry.render",
        lambda spec, renderer="matplotlib": captured.append(spec) or spec,
    )
    c.plot()
    ComparisonSet([c]).plot()
    assert [s.options["labels"] for s in captured] == [
        ("ccs_4km", "WOA23 temperature Apr")
    ] * 2


def test_two_different_sources_that_read_alike_keep_their_own_names(monkeypatch):
    from ocean_skill.comparison import ComparisonSet, _source_label_map

    monkeypatch.setattr(
        catalog, "resolve", lambda name: SimpleNamespace(metadata={"label": "WOA"})
    )
    # one label for two sources: both fall back, so a legend row never collapses
    both = _source_label_map(["woa_a", "woa_b"])
    assert both == {"woa_a": "woa_a", "woa_b": "woa_b"}
    # the same source twice is not a collision
    assert _source_label_map(["woa_a", "woa_a"]) == {"woa_a": "WOA"}

    a, b = _profile_comparison(), _profile_comparison()
    a.test_name, a.reference_name = "run", "woa_a"
    b.test_name, b.reference_name = "run", "woa_b"
    items = ComparisonSet([a, b])._items()
    assert [i["labels"][1] for i in items] == ["woa_a", "woa_b"]
