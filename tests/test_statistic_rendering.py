"""A reduced field -- a variance, a standard deviation -- is drawn as what it is.

``aggregate`` leaves ``attrs["statistic"]`` on its result and rewrites
``attrs["units"]`` to match (``(mg/m^3)^2``, ``delta_degC^2``). Both renderers have to
read that: a spread is not on its variable's pinned display range or log scale (the
variance of chlorophyll is nowhere near 0.01-10), and its units are printed readably on
the colour bar (``(mg/m³)²``, ``°C²``). A plain field -- no ``statistic`` -- is drawn
exactly as before.
"""

from __future__ import annotations

import matplotlib.colors as mcolors
import numpy as np
import pytest
import xarray as xr

from ocean_skill.plot._statistic import statistic_of, units_text
from ocean_skill.plot.matplotlib_renderer import _elide, suptitle_text
from ocean_skill.plot.registry import render
from ocean_skill.plot.spec import PlotSpec

CHLOROPHYLL = "mass_concentration_of_chlorophyll_a_in_sea_water"
TEMPERATURE = "sea_water_temperature"


def _map(units: str, statistic: str | None, standard_name: str, offset: float = 0.0):
    rng = np.random.default_rng(3)
    attrs = {"standard_name": standard_name, "units": units}
    if statistic is not None:
        attrs["statistic"] = statistic
    return xr.DataArray(
        offset + rng.uniform(0.001, 0.5, (6, 8)),
        dims=("lat", "lon"),
        coords={
            "lat": np.linspace(20.0, 30.0, 6),
            "lon": np.linspace(-100.0, -90.0, 8),
        },
        attrs=attrs,
    )


def _facet_spec(field) -> PlotSpec:
    return PlotSpec(
        family="field_facet",
        items=[
            {
                "field": field,
                "facet_dim": None,
                "row_dim": None,
                "units": field.attrs["units"],
                "standard_name": field.attrs["standard_name"],
            }
        ],
    )


def _row_spec(units: str, statistic: str | None, standard_name: str) -> PlotSpec:
    test = _map(units, statistic, standard_name, 0.1)
    ref = _map(units, statistic, standard_name, 0.0)
    diff = (test - ref).assign_attrs(units=units)  # arithmetic drops attrs, as in life
    diff.attrs.pop("statistic", None)
    return PlotSpec(
        family="field_row",
        items=[
            {
                "aligned": {"test": test, "reference": ref, "difference": diff},
                "units": units,
                "standard_name": standard_name,
                "labels": ("model", "obs"),
            }
        ],
    )


def _all_meshes(fig):
    from matplotlib.collections import QuadMesh

    return [c for ax in fig.axes for c in ax.collections if isinstance(c, QuadMesh)]


def _meshes(fig):
    """Return the maps' meshes in axes order, not the colour bars' own solids."""
    bars = {m.colorbar.ax for m in _all_meshes(fig) if m.colorbar is not None}
    return [m for ax in fig.axes if ax not in bars for m in _all_meshes_in(ax)]


def _all_meshes_in(ax):
    from matplotlib.collections import QuadMesh

    return [c for c in ax.collections if isinstance(c, QuadMesh)]


def _bar_labels(fig) -> list[str]:
    bars = [m.colorbar for m in _all_meshes(fig) if m.colorbar is not None]
    return [b.ax.get_xlabel() or b.ax.get_ylabel() for b in bars]


def _hv_mesh_opts(obj) -> dict:
    import holoviews as hv

    hv.extension("bokeh")
    mesh = next(iter(obj.traverse(lambda x: x, [hv.QuadMesh])))
    return hv.Store.lookup_options("bokeh", mesh, "plot").kwargs


# --- the helper -----------------------------------------------------------------------


def test_statistic_of_reads_a_dataarray_a_dataset_an_item_and_a_bare_mapping():
    var = _map("(mg/m^3)^2", "var", CHLOROPHYLL)
    plain = _map("mg/m^3", None, CHLOROPHYLL)
    assert statistic_of(var) == "var"
    assert statistic_of(plain) is None
    assert statistic_of(xr.Dataset({"test": var, "reference": var})) == "var"
    assert statistic_of({"field": var}) == "var"
    assert statistic_of({"aligned": {"reference": var}}) == "var"
    assert statistic_of({"test": var}) == "var"  # field_row hands over `aligned` itself
    assert statistic_of({"field": plain}) is None
    assert statistic_of(None) is None


def test_an_item_key_outranks_the_array_attribute():
    var = _map("(mg/m^3)^2", "var", CHLOROPHYLL)
    assert statistic_of({"field": var, "statistic": "std"}) == "std"


def test_units_text_only_rewrites_a_spread():
    assert units_text("delta_degC^2", "var") == "°C²"
    assert units_text("(mg/m^3)^2", "var") == "(mg/m³)²"
    assert units_text("mg/m^3", "mean") == "mg/m^3"  # a level keeps its spelling
    assert units_text("mg/m^3", None) == "mg/m^3"
    assert units_text(None, "var") == ""


# --- static renderer ------------------------------------------------------------------


def test_static_variance_of_chlorophyll_is_linear_with_readable_units():
    fig = render(_facet_spec(_map("(mg/m^3)^2", "var", CHLOROPHYLL)))
    (mesh,) = _meshes(fig)
    assert not isinstance(mesh.norm, mcolors.LogNorm)
    assert mesh.norm.vmax < 1.0  # its own data, not chlorophyll's pinned 0.01-10
    assert _bar_labels(fig) == ["[(mg/m³)²]"]


def test_static_variance_of_temperature_reads_degC_squared():
    fig = render(_facet_spec(_map("delta_degC^2", "var", TEMPERATURE)))
    assert _bar_labels(fig) == ["[°C²]"]


def test_static_plain_chlorophyll_is_still_log_and_keeps_its_label():
    fig = render(_facet_spec(_map("mg/m^3", None, CHLOROPHYLL)))
    (mesh,) = _meshes(fig)
    assert isinstance(mesh.norm, mcolors.LogNorm)
    assert _bar_labels(fig) == ["[mg/m^3]"]


def test_static_difference_of_variances_is_diverging_linear_and_labelled_alike():
    fig = render(_row_spec("(mg/m^3)^2", "var", CHLOROPHYLL))
    test, _ref, diff = _meshes(fig)
    assert not isinstance(test.norm, mcolors.LogNorm)
    assert not isinstance(diff.norm, mcolors.LogNorm)
    assert diff.norm.vmin == pytest.approx(-diff.norm.vmax)  # symmetric about zero
    # test and reference share one bar; the difference has its own
    assert _bar_labels(fig) == ["[(mg/m³)²]", "difference [(mg/m³)²]"]


def test_static_chlorophyll_row_without_a_statistic_stays_log():
    fig = render(_row_spec("mg/m^3", None, CHLOROPHYLL))
    test, _, diff = _meshes(fig)
    assert isinstance(test.norm, mcolors.LogNorm)
    assert not isinstance(diff.norm, mcolors.LogNorm)


# --- interactive renderer -------------------------------------------------------------


def test_holoviews_variance_of_chlorophyll_is_linear_with_readable_units():
    obj = render(
        _facet_spec(_map("(mg/m^3)^2", "var", CHLOROPHYLL)), renderer="holoviews"
    )
    opts = _hv_mesh_opts(obj)
    assert opts["logz"] is False
    assert opts["clabel"] == "(mg/m³)²"


def test_holoviews_variance_of_temperature_reads_degC_squared():
    obj = render(
        _facet_spec(_map("delta_degC^2", "var", TEMPERATURE)), renderer="holoviews"
    )
    assert _hv_mesh_opts(obj)["clabel"] == "°C²"


def test_holoviews_plain_chlorophyll_is_still_log_and_keeps_its_label():
    obj = render(
        _facet_spec(_map("mg/m^3", None, CHLOROPHYLL)), renderer="holoviews"
    )
    opts = _hv_mesh_opts(obj)
    assert opts["logz"] is True
    assert opts["clabel"] == "mg/m^3"


def test_holoviews_row_of_variances_has_no_log_scale_and_a_readable_difference_label():
    import holoviews as hv

    hv.extension("bokeh")
    obj = render(
        _row_spec("(mg/m^3)^2", "var", CHLOROPHYLL), renderer="holoviews"
    )
    meshes = list(obj.traverse(lambda x: x, [hv.QuadMesh]))
    opts = [hv.Store.lookup_options("bokeh", m, "plot").kwargs for m in meshes]
    assert [o["logz"] for o in opts] == [False, False, False]
    assert [o["clabel"] for o in opts] == [
        "(mg/m³)²",
        "(mg/m³)²",
        "test − reference (mg/m³)²",
    ]


# --- titles that carry a statistic AND its window -------------------------------------

WINDOW = "variance of monthly means over 2012-01-01–2012-12-31"


def test_a_statistic_phrase_keeps_its_window_whole_in_the_suptitle():
    title = suptitle_text(CHLOROPHYLL, ("surface", WINDOW))
    assert title.endswith(f"surface · {WINDOW}")
    assert "…" not in title


def test_a_too_long_statistic_phrase_is_elided_before_the_window_not_after():
    phrase = "variance of seasonal-cycle monthly means of daily maxima"
    title = _elide(f"{phrase} over 2012-01-01–2012-12-31")
    assert title.endswith(" over 2012-01-01–2012-12-31")
    assert "…" in title and phrase not in title


def test_a_long_part_without_a_window_is_elided_as_before():
    long_part = "a very long depth-list label that somehow slipped past collapsing"
    assert _elide(long_part).endswith("…")
    assert len(_elide(long_part)) <= 41


def test_both_renderers_title_a_statistic_and_its_window_alike():
    field = _map("(mg/m^3)^2", "var", CHLOROPHYLL)
    spec = _facet_spec(field)
    spec.items[0]["depth"] = "surface"
    spec.items[0]["time"] = WINDOW
    expected = suptitle_text(CHLOROPHYLL, ("surface", WINDOW))
    assert WINDOW in expected
    assert render(spec)._suptitle.get_text() == expected
    obj = render(spec, renderer="holoviews")
    assert obj.opts.get("plot").kwargs.get("title") == expected
