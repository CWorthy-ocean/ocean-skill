"""The metrics table: ``osk.table`` / ``ComparisonSet.table`` and :class:`MetricsTable`.

What these defend: the table is the metric records pivoted (same pooling as
``osk.summary``, same duplicate-cell refusal as the portrait), that the one internal
layout -- merged cells included -- is what every writer renders, and that the
paper-table features (seasonal sub-rows, ``R² (surface / column)`` joined cells,
notes) land where they are asked to.

Offline: hand-built comparisons carrying a metrics record, as in ``test_pooling``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import ocean_skill as osk
from ocean_skill.comparison import ComparisonSet
from ocean_skill.table import MetricsTable

TEMP = "sea_water_temperature"
SALT = "sea_water_salinity"
UEAST = "eastward_sea_water_velocity"
ALK = "sea_water_alkalinity"


class _Fake:
    """A comparison's specification plus a metrics record, and nothing else."""

    def __init__(
        self,
        label,
        *,
        variable=TEMP,
        depth="surface",
        time=None,
        test="roms",
        reference="obs",
        units="degC",
        combined=None,
        notes=None,
        demeaned="raw",
        **metrics,
    ):
        self.variable = variable
        self.standard_name = variable
        self.select = {"depth": depth}
        if time is not None:
            self.select["time"] = time
        self.test_name = test
        self.reference_name = reference
        self.label = label
        self.aggregate = None
        self.method = "conservative_normed"
        self.over = None
        self.time_method = "auto"
        self.tolerance = None
        self.bin_anchor = "auto"
        self.units = units
        self._combined = combined
        self.notes = notes
        corr = metrics.pop("corr", 0.9)
        self._record = {
            "bias": 0.1,
            "rmse": 0.5,
            "crmsd": 0.4,
            "corr": corr,
            "r2": corr**2,
            "sigma_ratio": 1.05,
            "n": 100,
            "variable": variable,
            "depth": depth,
            "time": time,
            "test": test,
            "reference": reference,
            "demeaned": demeaned,
            "detided": "raw",
            **metrics,
        }
        if notes:
            self._record["notes"] = notes

    def metrics(self):
        return self._record

    def as_item(self):
        return {"metrics": self._record, "label": self.label}


def _cs(*pairs):
    """A set from ``(label, _Fake)``-style fakes, each named by its own label."""
    return ComparisonSet(list(pairs), labels=[c.label for c in pairs])


def _target_set():
    """Build the target table's comparisons: seasonal for SST/Salinity only."""
    comps = [
        _Fake("SST", depth="surface", bias=0.10, rmse=0.50, corr=0.95, units="degC"),
        _Fake("SST", depth="column", bias=0.05, rmse=0.40, corr=0.92, units="degC"),
        _Fake("SST", depth="surface", time="MAM", bias=0.21, units="degC"),
        _Fake("SST", depth="surface", time="JJA", bias=0.32, units="degC"),
        _Fake("SST", depth="surface", time="SON", bias=-0.13, units="degC"),
        _Fake("Salinity", variable=SALT, bias=0.02, rmse=0.11, corr=0.8, units="1"),
        _Fake("Salinity", variable=SALT, time="MAM", bias=0.03, units="1"),
        _Fake("Salinity", variable=SALT, time="JJA", bias=0.04, units="1"),
        _Fake(
            "Velocity east", variable=UEAST, bias=0.01, rmse=0.07, corr=0.6, units="m/s"
        ),
        _Fake("Alk", variable=ALK, bias=-2.0, rmse=9.0, corr=0.7, units="umol/kg"),
    ]
    return _cs(*comps)


def _plain_set():
    """The same rows with one whole-record comparison each: nothing to split or join."""
    return _cs(
        *(
            c
            for c in _target_set()
            if c.select.get("depth") == "surface" and "time" not in c.select
        )
    )


def _target_table(**kw):
    return _target_set().table(
        metrics=("bias", "r2", "rmse"),
        split={"bias": "time"},
        join={"r2": "depth", "rmse": "depth"},
        info={"SST": {"Type": "CTD"}},
        notes={"SST": "ice-free months only"},
        **kw,
    )


# --- pooling and the basic pivot ----------------------------------------------------


def test_default_table_is_one_row_per_comparison_with_default_metrics():
    cs = _cs(_Fake("a", depth=0), _Fake("b", depth=5))
    t = cs.table()
    assert isinstance(t, MetricsTable)
    df = t.to_dataframe()
    assert list(df.columns) == ["bias", "rmse", "crmsd", "corr", "sigma_ratio", "n"]
    assert list(df.index) == ["a", "b"]


def test_osk_table_is_callable_and_still_the_module():
    # `import ocean_skill.table` makes the submodule a package attribute; the package
    # keeps it callable so the function and the module can share one name
    assert callable(osk.table)
    assert osk.table.MetricsTable is MetricsTable
    assert "table" in osk.__all__ and "MetricsTable" in osk.__all__


def test_osk_table_pools_sets_and_a_lone_comparison_like_summary():
    a = ComparisonSet(
        [_Fake("x", variable=TEMP, depth=0), _Fake("y", variable=SALT, depth=0)]
    )
    b = ComparisonSet([_Fake("x", variable=TEMP, depth=50)])
    lone = _Fake("z", variable=ALK, depth=0)
    t = osk.table([a, [b], lone], metrics=("bias",))
    labels = list(t.to_dataframe().index)
    assert len(labels) == 4
    assert len(set(labels)) == 4  # pooled names tell the members apart


def test_dict_keys_are_labels_and_the_group_field():
    hind = ComparisonSet(
        [_Fake("t", variable=TEMP, depth=0), _Fake("s", variable=SALT, depth=0)]
    )
    fore = ComparisonSet([_Fake("t", variable=TEMP, depth=0, test="fc", bias=0.9)])
    t = osk.table(
        {"hindcast": hind, "forecast": fore},
        rows="variable",
        cols="group",
        metrics=("bias",),
    )
    df = t.to_dataframe()
    assert set(df.columns) == {("bias", "hindcast"), ("bias", "forecast")}
    assert df[("bias", "forecast")].dropna().iloc[0] == pytest.approx(0.9)
    # without cols the keys name the rows
    assert list(
        osk.table({"hindcast": hind[0], "forecast": fore}, metrics=("bias",))
        .to_dataframe()
        .index
    ) == [
        "hindcast",
        "forecast",
    ]


def test_metric_selection_order_and_unknown_raises():
    cs = _cs(_Fake("a"))
    assert list(cs.table(metrics=("rmse", "bias")).to_dataframe().columns) == [
        "rmse",
        "bias",
    ]
    with pytest.raises(ValueError, match="no metric"):
        cs.table(metrics=("bias", "nonsense"))


def test_cols_pivots_a_field_into_metric_level_columns():
    cs = _cs(
        _Fake("A", test="r1", bias=1.0),
        _Fake("A", test="r2", bias=2.0),
        _Fake("B", variable=SALT, test="r1", bias=3.0),
    )
    t = cs.table(metrics=("bias", "rmse"), cols="test")
    df = t.to_dataframe()
    assert list(df.columns) == [
        ("bias", "r1"),
        ("bias", "r2"),
        ("rmse", "r1"),
        ("rmse", "r2"),
    ]
    assert df.loc["A", ("bias", "r2")] == 2.0
    assert np.isnan(df.loc["B", ("bias", "r2")])
    # two header rows: the metric groups over the levels
    assert len(t._layout.header) == 2
    assert "bias" in str(t).splitlines()[0]


def test_rows_by_a_record_field_uses_pretty_levels():
    cs = _cs(
        _Fake("a", depth={"min": 0, "max": 100}),
        _Fake("b", depth=200),
    )
    df = cs.table(rows="depth", metrics=("bias",)).to_dataframe()
    assert list(df.index) == ["0-100 m", "200 m"]


def test_two_comparisons_in_one_cell_raise_like_the_portrait():
    cs = _cs(_Fake("A", depth=0), _Fake("A", depth=5))
    with pytest.raises(ValueError, match="more than one comparison"):
        cs.table(metrics=("bias",))
    # telling them apart by cols resolves it
    assert cs.table(metrics=("bias",), cols="depth").to_dataframe().shape == (1, 2)


# --- split, join, layouts ------------------------------------------------------------


def test_split_rows_draws_the_level_as_a_sub_column_left_of_the_split_metric():
    t = _target_table()
    head = [c.text for c in t._layout.header[0]]
    # the level column sits immediately left of the first split metric (bias)
    assert head[:3] == ["", "Season", "bias"] or head[1] == "Season"
    assert head[2].startswith("bias")
    body = t._layout.body
    sst = body[:3]
    levels = [row[head.index("Season")].text for row in sst]
    assert levels == ["MAM", "JJA", "SON"]  # season order, not alphabetical
    # the row label spans all three sub-rows
    assert sst[0][0].rowspan == 3 and sst[1][0] is None and sst[2][0] is None


def test_split_level_column_sits_before_the_first_split_metric_not_at_the_far_left():
    cs = _target_set()
    t = cs.table(metrics=("r2", "bias"), split={"bias": "time"}, join={"r2": "depth"})
    heads = [c.path[0] for c in t._layout.cols]
    assert heads.index("Season") == heads.index("bias") - 1
    assert heads.index("Season") > heads.index(
        next(h for h in heads if h.startswith("R²"))
    )


def test_row_without_seasonal_records_gets_one_line_with_the_whole_value():
    t = _target_table()
    df = t.to_dataframe()
    assert df.index.names == ["row", "level"]
    assert list(df.loc["Alk"].index) == [""]
    assert df.loc[("Alk", ""), "bias"] == pytest.approx(-2.0)
    assert df.loc[("SST", "MAM"), "bias"] == pytest.approx(0.21)
    # Salinity only has MAM and JJA: its sub-rows are exactly those
    assert list(df.loc["Salinity"].index) == ["MAM", "JJA"]
    text = str(t).splitlines()
    assert sum(line.startswith("Alk") for line in text) == 1


def test_split_columns_layout_has_metric_level_columns_and_no_row_spans():
    t = _target_set().table(
        metrics=("bias", "rmse"),
        split={"bias": "time"},
        join={"rmse": "depth"},
        layout="columns",
    )
    paths = [c.path for c in t._layout.cols]
    assert ("bias (°C)", "MAM") not in paths  # mixed units: no shared suffix
    assert [p for p in paths if p[0] == "bias"] == [
        ("bias", "MAM"),
        ("bias", "JJA"),
        ("bias", "SON"),
        ("bias", "all"),  # rows with no seasonal records read their whole value here
    ]
    assert all(c is None or c.rowspan == 1 for line in t._layout.body for c in line)
    df = t.to_dataframe()
    assert df.loc["SST", ("bias", "MAM")] == pytest.approx(0.21)
    assert df.loc["Alk", ("bias", "all")] == pytest.approx(-2.0)
    assert np.isnan(df.loc["SST", ("bias", "all")])  # SST has seasons: no "all" value
    assert df.loc["Alk", ("rmse", "surface")] == pytest.approx(9.0)


def test_unsplit_metric_reads_the_whole_record_not_a_season():
    cs = _cs(
        _Fake("A", rmse=0.5),
        _Fake("A", time="MAM", rmse=9.0, bias=1.0),
        _Fake("A", time="JJA", rmse=8.0, bias=2.0),
    )
    df = cs.table(metrics=("rmse", "bias"), split={"bias": "time"}).to_dataframe()
    assert df.loc[("A", "MAM"), "rmse"] == pytest.approx(0.5)  # the whole-record value
    assert np.isnan(
        df.loc[("A", "JJA"), "rmse"]
    )  # first line only, like the merged cell
    assert df.loc[("A", "JJA"), "bias"] == pytest.approx(2.0)


def test_join_puts_levels_in_one_cell_with_a_missing_slot():
    t = _target_table()
    text = t.to_markdown()
    sst = next(
        line for line in text.splitlines() if "ice-free" in line or "SST" in line
    )
    assert "0.90 / 0.85" in sst  # corr 0.95**2, 0.92**2 -> surface / column
    sal = next(line for line in text.splitlines() if "Salinity" in line)
    assert "0.64 / –" in sal  # no column-depth Salinity record
    assert "R² (surface / column)" in text.splitlines()[0]
    assert "RMSE (surface / column)" in text.splitlines()[0]


def test_join_labels_rename_the_levels():
    cs = _target_set()
    t = cs.table(
        metrics=("r2", "bias"),
        split={"bias": "time"},
        join={"r2": "depth"},
        join_labels={"depth": ["top", "whole column"]},
    )
    assert "R² (top / whole column)" in t.to_markdown().splitlines()[0]
    with pytest.raises(ValueError, match="one name per level"):
        cs.table(
            metrics=("r2", "bias"),
            split={"bias": "time"},
            join={"r2": "depth"},
            join_labels={"depth": ["only"]},
        )


def test_split_and_join_misuse_raises():
    cs = _target_set()
    with pytest.raises(ValueError, match="not among"):
        cs.table(metrics=("bias",), split={"rmse": "time"})
    with pytest.raises(ValueError, match="both split= and join="):
        cs.table(metrics=("bias",), split={"bias": "time"}, join={"bias": "depth"})
    with pytest.raises(ValueError, match="no record has"):
        cs.table(metrics=("bias",), split={"bias": "nope"})
    with pytest.raises(ValueError, match="layout="):
        cs.table(layout="diagonal")


def test_duplicate_cell_inside_a_split_raises():
    cs = _cs(
        _Fake("A", time="MAM", depth=0),
        _Fake("A", time="MAM", depth=5),
    )
    with pytest.raises(ValueError, match="more than one comparison"):
        cs.table(metrics=("bias",), split={"bias": "time"})


def test_depth_band_levels_get_labels_in_joins_and_notes():
    cs = _cs(
        _Fake("A", depth={"min": 0, "max": 5}),
        _Fake("A", depth={"min": 5, "max": 100}),
    )
    t = cs.table(metrics=("rmse",), join={"rmse": "depth"})
    assert "RMSE (°C; 0-5 m / 5-100 m)" in t.to_markdown().splitlines()[0]


# --- units, headers, round ----------------------------------------------------------


def test_shared_units_go_in_dimensional_headers_only():
    cs = _cs(_Fake("a"), _Fake("b", depth=5))
    head = cs.table(metrics=("bias", "corr", "n")).to_markdown().splitlines()[0]
    assert "bias (°C)" in head
    assert "corr |" in head + " |" and "corr (" not in head
    assert "n (" not in head


def test_mixed_units_become_a_per_row_units_column():
    t = _plain_set().table(metrics=("bias",))
    md = t.to_markdown().splitlines()
    assert "Units" in md[0] and "bias |" in md[0] + " |"
    assert any("umol/kg" in line and "Alk" in line for line in md)
    assert (
        "Units" not in _plain_set().table(metrics=("bias",), units=False).to_markdown()
    )


def test_headers_override_default_names():
    cs = _cs(_Fake("a"))
    head = (
        cs.table(
            metrics=("rmse", "r2"),
            units=False,
            auto_notes=False,
            headers={"rmse": "root MSE", "label": "Station"},
        )
        .to_markdown()
        .splitlines()[0]
    )
    assert head == "| Station | root MSE | R² |"


def test_round_formats_like_the_portrait_and_false_keeps_digits():
    cs = _cs(_Fake("a", bias=0.123456, corr=0.98765, n=12))
    row = (
        cs.table(metrics=("bias", "corr", "n"), units=False, auto_notes=False)
        .to_markdown()
        .splitlines()[2]
    )
    assert row == "| a | 0.123 | 0.99 | 12 |"
    raw = (
        cs.table(metrics=("bias",), units=False, round=False, auto_notes=False)
        .to_markdown()
        .splitlines()[2]
    )
    assert "0.123456" in raw


# --- notes and info -----------------------------------------------------------------


def test_auto_notes_describe_selection_pooling_and_demeaning():
    cs = _cs(
        _Fake(
            "deep",
            depth={"min": 100, "max": 200},
            combined=("pool", 8),
            demeaned="demeaned",
        ),
        _Fake("avg", depth=0, time="2020-01", combined=("average", 3)),
    )
    md = cs.table(metrics=("bias",), units=False).to_markdown().splitlines()
    assert "100-200 m; sample average (8); demeaned" in md[2]
    assert "0 m; 2020-01; station average (3)" in md[3]


def test_auto_notes_switches_and_overrides():
    cs = _cs(_Fake("a", combined=("pool", 4)), _Fake("b", depth=5))
    assert "Notes" not in cs.table(metrics=("bias",), auto_notes=False).to_markdown()
    custom = cs.table(
        metrics=("bias",), auto_notes=lambda rec: f"v={rec['variable'][:3]}"
    )
    assert "v=sea" in custom.to_markdown()
    over = cs.table(metrics=("bias",), auto_notes={"a": "my text"}).to_markdown()
    assert "my text" in over and "5 m" in over
    with pytest.raises(ValueError, match="auto_notes"):
        cs.table(auto_notes=3)


def test_user_notes_from_the_record_and_overridden_by_table():
    cs = _cs(_Fake("a", notes="from compare"), _Fake("b", depth=5, notes="also"))
    md = cs.table(metrics=("bias",), auto_notes=False).to_markdown().splitlines()
    assert "User notes" in md[0] and "from compare" in md[2]
    md = cs.table(
        metrics=("bias",), auto_notes=False, notes={"a": "mine"}
    ).to_markdown()
    assert "mine" in md and "from compare" not in md and "also" in md
    # no notes anywhere: the column is left out
    assert "User notes" not in _cs(_Fake("a")).table(metrics=("bias",)).to_markdown()


def test_info_columns_follow_the_metrics_and_span_the_row():
    t = _target_table()
    heads = [c.path[0] for c in t._layout.cols]
    assert heads.index("Type") > max(
        i for i, h in enumerate(heads) if h.startswith("RMSE")
    )
    j = heads.index("Type")
    assert t._layout.body[0][j].text == "CTD" and t._layout.body[0][j].rowspan == 3
    assert t._layout.body[3][j].text == ""  # the other rows have none


def test_notes_do_not_enter_the_cache_key_or_identity():
    from ocean_skill.comparison import Comparison, _identity

    kw = {"reference": "obs", "test": "roms", "variable": TEMP, "select": {"depth": 0}}
    a, b = Comparison(**kw), Comparison(**kw)
    b.notes = "ice-free months only"
    assert a.notes is None
    assert a._cache_key == b._cache_key
    assert _identity(a) == _identity(b)


def test_compare_notes_reaches_the_metrics_record(monkeypatch):
    """`compare(notes=)` sets the attribute and the record carries it.

    It is not a Comparison() constructor argument, so it cannot reach identity/keys.
    """
    import inspect

    from ocean_skill import comparison as cmp

    assert "notes" in inspect.signature(cmp.compare).parameters
    assert "notes" not in inspect.signature(cmp.Comparison.__init__).parameters
    src = inspect.getsource(cmp.Comparison.metrics)
    assert '"notes": self.notes' in src


# --- the writers ---------------------------------------------------------------------


def test_html_has_real_rowspans_and_escapes():
    t = _target_table()
    html = t._repr_html_()
    assert html == t.to_html()
    assert '<td rowspan="3"' in html
    assert "<table" in html and "R² (surface / column)" in html
    esc = _cs(_Fake("a<b>")).table(metrics=("bias",), units=False).to_html()
    assert "a&lt;b&gt;" in esc


def test_latex_uses_multirow_booktabs_and_escapes():
    tex = _target_table().to_latex()
    assert r"\toprule" in tex and r"\midrule" in tex and r"\bottomrule" in tex
    assert r"\multirow{3}{*}{SST}" in tex
    assert r"R$^{2}$ (surface / column)" in tex
    assert r"$-$0.13" in tex  # a real minus sign
    assert r"\cmidrule" in tex
    special = (
        _cs(_Fake("50% & #1_x", sigma_ratio=1.2))
        .table(metrics=("sigma_ratio",), units=False)
        .to_latex()
    )
    assert r"50\% \& \#1\_x" in special
    assert r"$\sigma$ ratio" in special


def test_latex_two_level_header_uses_multicolumn():
    cs = _cs(_Fake("A", test="r1"), _Fake("A", test="r2"))
    tex = cs.table(metrics=("bias",), cols="test", units=False).to_latex()
    assert r"\multicolumn{2}{c}{bias}" in tex
    assert r"\cmidrule(lr){2-3}" in tex
    assert r"\multirow{2}{*}{}" in tex


def test_markdown_blanks_spanned_cells_and_is_a_pipe_table():
    md = _target_table().to_markdown().splitlines()
    assert md[1].startswith("| :---")
    sst_lines = [
        line for line in md if "MAM" in line or "JJA" in line or "SON" in line
    ][:3]
    assert (
        "SST" in sst_lines[0]
        and "SST" not in sst_lines[1]
        and "SST" not in sst_lines[2]
    )
    cells = [c.strip() for c in sst_lines[1].strip("|").split("|")]
    assert cells[0] == ""  # the row label is blank on the later sub-rows


def test_plain_text_is_aligned_with_spans_blank():
    text = str(_target_table())
    lines = text.splitlines()
    assert repr(_target_table()) == text
    assert "Season" in lines[0] and "bias" in lines[0]
    # columns are fixed width: "Season" starts at the same offset on every sub-row
    col = lines[0].index("Season")
    assert [
        line[col : col + 3] for line in lines[2:] if line.startswith(("SST", "   "))
    ][:3] == [
        "MAM",
        "JJA",
        "SON",
    ]
    assert sum(line.startswith("SST") for line in lines) == 1


def test_to_dataframe_is_numeric_and_to_tidy_is_long():
    t = _target_table()
    df = t.to_dataframe()
    assert df.loc[("SST", "JJA"), "bias"] == pytest.approx(0.32)
    # joined metrics expand to one column per level
    assert ("r2", "surface") in df.columns and ("r2", "column") in df.columns
    assert df.loc[("SST", "MAM"), ("r2", "surface")] == pytest.approx(0.95**2)
    assert np.isnan(df.loc[("SST", "JJA"), ("r2", "surface")])  # on the first line only
    tidy = t.to_tidy()
    assert {"row", "metric", "value"} <= set(tidy.columns)
    assert len(tidy) > len(df)


@pytest.mark.parametrize("suffix", [".tex", ".md", ".html", ".csv"])
def test_save_dispatches_on_suffix(tmp_path, suffix):
    t = _target_table()
    out = t.save(tmp_path / "sub" / f"tab{suffix}")
    text = out.read_text()
    assert out.exists() and text
    if suffix == ".tex":
        assert r"\multirow" in text
    elif suffix == ".md":
        assert text.startswith("|")
    elif suffix == ".html":
        assert "rowspan" in text
    else:
        csv = pd.read_csv(out)
        assert {"row", "metric", "value"} <= set(csv.columns)


def test_save_kwarg_and_unknown_suffix(tmp_path):
    cs = _plain_set()
    cs.table(metrics=("bias",), save=tmp_path / "t.md")
    assert (tmp_path / "t.md").exists()
    with pytest.raises(ValueError, match="suffix"):
        cs.table(metrics=("bias",), save=tmp_path / "t.docx")
    with pytest.raises(ValueError, match="suffix"):
        cs.table(metrics=("bias",)).save(tmp_path / "t")


def test_target_example_end_to_end():
    t = _target_table()
    md = t.to_markdown()
    lines = md.splitlines()
    # variable rows, a seasonal sub-column, joined R2/RMSE, info and notes columns
    assert "Type" in lines[0] and "User notes" in lines[0]
    assert (
        "| Notes |" not in lines[0]
    )  # nothing to say: depth/season are in the headers
    assert "CTD" in md and "ice-free months only" in md
    body_rows = len(lines) - 2
    assert body_rows == 3 + 2 + 1 + 1  # SST x3 seasons, Salinity x2, velocity, Alk
    assert isinstance(t._repr_html_(), str)


# --- r2 ------------------------------------------------------------------------------


def test_r2_is_corr_squared_and_registered_after_corr():
    from ocean_skill.metrics import REGISTRY, compute

    names = list(REGISTRY)
    assert names.index("r2") == names.index("corr") + 1
    assert REGISTRY["r2"].units == "1"

    import xarray as xr

    t = xr.DataArray(np.linspace(0, 1, 60) + 0.1 * np.sin(np.arange(60)), dims="x")
    r = xr.DataArray(np.linspace(0, 1, 60), dims="x")
    rec = compute(
        xr.Dataset({"test": t, "reference": r}), min_samples=1, weighted=False
    )
    assert rec["r2"] == pytest.approx(rec["corr"] ** 2)
    only = compute(
        xr.Dataset({"test": t, "reference": r}),
        names=["r2"],
        min_samples=1,
        weighted=False,
    )
    assert only["r2"] == pytest.approx(
        rec["corr"] ** 2
    )  # corr is computed to derive it


def test_labels_naming_split_and_join_levels_are_rebuilt_into_one_row_per_variable():
    # real pooled labels name every level ("temperature 0-5 m MAM"); with split= and
    # join= over those fields that would give each season and depth its own row
    comps = list(_target_set())
    labels = [
        " ".join(
            str(x)
            for x in (c.variable, c.select["depth"], c.select.get("time", "all"))
        )
        for c in comps
    ]
    t = ComparisonSet(comps, labels=labels).table(
        metrics=("r2", "rmse", "bias"),
        split={"bias": "time"},
        join={"r2": "depth", "rmse": "depth"},
    )
    rows = list(dict.fromkeys(t.to_dataframe().index.get_level_values(0)))
    assert len(rows) == 4
    assert not any("MAM" in r or "surface" in r for r in rows)


def test_row_units_fall_back_to_any_record_of_the_row():
    comps = [
        _Fake("SST", depth="surface", units=None),
        _Fake("SST", depth="column", units="degC"),
        _Fake("Alk", variable=ALK, units="umol/kg"),
    ]
    t = _cs(*comps).table(metrics=("bias",), join={"bias": "depth"})
    assert "°C" in t.to_markdown().splitlines()[2]
