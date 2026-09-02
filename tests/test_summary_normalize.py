"""Absolute (data-unit) axes for the Target and Taylor diagrams.

``normalize=True`` (the default, unchanged) divides standard deviation, centred RMSD,
and bias by the reference standard deviation, so comparisons in different units share
one diagram. ``normalize=False`` leaves them in native units instead — the point this
file defends is that both renderers agree on where the points land, what the default
guide rings become, what the axes are limited to, and what they're labelled, and that
mixed references or mixed variables warn rather than silently mislabeling the figure.

Taylor already had ``normalize`` (statically only); this also covers the bug that came
with it — ``TaylorDiagram``'s own ``srange`` is multiples of the reference std, so an
un-normalized diagram needs it divided back down, or the radial axis dwarfs the data.
"""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest

from ocean_skill.plot.summary import paired, target, taylor


class _FakeComparison:
    """A comparison's metric record plus a label and units — the diagrams' interface."""

    def __init__(
        self,
        *,
        label,
        corr=0.9,
        std_test,
        std_reference,
        bias,
        crmsd,
        variable="sea_water_temperature",
        units="degC",
        reference="ref1",
    ):
        self.label = label
        self.units = units
        self._record = {
            "corr": corr,
            "std_test": std_test,
            "std_reference": std_reference,
            "bias": bias,
            "crmsd": crmsd,
            "variable": variable,
            "reference": reference,
        }

    def metrics(self):
        return self._record


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _interactive_target(items, **kwargs):
    from ocean_skill.plot.holoviews_renderer import _target

    return _target(items, **kwargs)


def _items(comparisons):
    """Comparisons in the shape ``PlotSpec.items`` carries them (see spec.py)."""
    return [
        {"label": c.label, "metrics": c.metrics(), "units": c.units} for c in comparisons
    ]


def _points(obj):
    import holoviews as hv

    return [e for e in obj if isinstance(e, hv.Points)]


def _guide_rings(obj):
    """``(radius, dash) for each guide ring the interactive target drew."""
    import holoviews as hv

    out = []
    for e in obj:
        if isinstance(e, hv.Path):
            data = e.data[0]
            radius = float(np.hypot(data[:, 0], data[:, 1]).max())
            out.append((radius, e.opts.get("style").kwargs.get("line_dash")))
    return out


# ------------------------------------------------------------------------ positions


def test_target_normalize_true_default_unchanged():
    """The default behaviour — divide by std_reference — is untouched."""
    comparisons = [
        _FakeComparison(label="m1", std_test=2.4, std_reference=2.0, bias=0.6, crmsd=0.8),
    ]
    fig = target(comparisons, labels=None)
    xy = tuple(fig.axes[0].collections[0].get_offsets()[0])
    assert xy == pytest.approx((0.4, 0.3))


def test_target_normalize_false_plots_raw_units():
    comparisons = [
        _FakeComparison(label="m1", std_test=2.4, std_reference=2.0, bias=0.6, crmsd=0.8),
    ]
    fig = target(comparisons, normalize=False, labels=None)
    xy = tuple(fig.axes[0].collections[0].get_offsets()[0])
    assert xy == pytest.approx((0.8, 0.6))


def test_target_normalize_false_signs_by_over_under_dispersion():
    """std_test < std_reference flips the sign of x, same as normalized mode."""
    comparisons = [
        _FakeComparison(label="m1", std_test=1.6, std_reference=2.0, bias=-0.3, crmsd=0.5),
    ]
    fig = target(comparisons, normalize=False, labels=None)
    x, y = fig.axes[0].collections[0].get_offsets()[0]
    assert x == pytest.approx(-0.5)
    assert y == pytest.approx(-0.3)


# ---------------------------------------------------------------------------- rings


def test_target_absolute_default_rings_scale_with_shared_sigma():
    comparisons = [
        _FakeComparison(label="m1", std_test=2.4, std_reference=2.0, bias=0.6, crmsd=0.8),
        _FakeComparison(label="m2", std_test=1.9, std_reference=2.0, bias=-0.3, crmsd=0.5),
    ]
    fig = target(comparisons, normalize=False, labels=None)
    circles = fig.axes[0].patches
    radii = sorted(c.get_radius() for c in circles)
    assert radii == pytest.approx([1.0, 2.0])
    dashed = next(c for c in circles if c.get_radius() == pytest.approx(2.0))
    dotted = next(c for c in circles if c.get_radius() == pytest.approx(1.0))
    assert dashed.get_linestyle() == "--"
    assert dotted.get_linestyle() == ":"


def test_target_absolute_mixed_sigma_skips_default_rings_and_warns():
    comparisons = [
        _FakeComparison(label="m1", std_test=2.4, std_reference=2.0, bias=0.6, crmsd=0.8),
        _FakeComparison(label="m2", std_test=3.8, std_reference=4.0, bias=-0.3, crmsd=0.5),
    ]
    with pytest.warns(UserWarning, match="no default guide rings"):
        fig = target(comparisons, normalize=False, labels=None)
    assert list(fig.axes[0].patches) == []


def test_target_absolute_explicit_circles_are_data_units_even_with_mixed_sigma():
    comparisons = [
        _FakeComparison(label="m1", std_test=2.4, std_reference=2.0, bias=0.6, crmsd=0.8),
        _FakeComparison(label="m2", std_test=3.8, std_reference=4.0, bias=-0.3, crmsd=0.5),
    ]
    fig = target(comparisons, normalize=False, circles=(1.5,), labels=None)
    radii = [c.get_radius() for c in fig.axes[0].patches]
    assert radii == pytest.approx([1.5])


def test_target_normalize_true_explicit_circles_dash_only_the_unit_circle():
    comparisons = [
        _FakeComparison(label="m1", std_test=1.1, std_reference=1.0, bias=0.1, crmsd=0.2),
    ]
    fig = target(comparisons, circles=(0.3, 0.7), labels=None)
    styles = {round(c.get_radius(), 3): c.get_linestyle() for c in fig.axes[0].patches}
    assert styles == {0.3: ":", 0.7: ":"}


# ----------------------------------------------------------------------- variables


def test_target_absolute_mixed_variables_warns():
    comparisons = [
        _FakeComparison(label="m1", std_test=2.4, std_reference=2.0, bias=0.6, crmsd=0.8,
                          variable="sea_water_temperature"),
        _FakeComparison(label="m2", std_test=1.9, std_reference=2.0, bias=-0.3, crmsd=0.5,
                          variable="sea_water_salinity"),
    ]
    with pytest.warns(UserWarning, match="multiple variables"):
        target(comparisons, normalize=False, labels=None)


def test_target_normalize_true_mixed_variables_does_not_warn(recwarn):
    comparisons = [
        _FakeComparison(label="m1", std_test=2.4, std_reference=2.0, bias=0.6, crmsd=0.8,
                          variable="sea_water_temperature"),
        _FakeComparison(label="m2", std_test=1.9, std_reference=2.0, bias=-0.3, crmsd=0.5,
                          variable="sea_water_salinity"),
    ]
    target(comparisons, labels=None)
    assert not [w for w in recwarn.list if "multiple variables" in str(w.message)]


# ---------------------------------------------------------------------------- lim


def test_target_absolute_lim_tracks_the_data_not_the_normalized_floor():
    """A small-magnitude variable shouldn't inherit normalized mode's 1.2 floor."""
    comparisons = [
        _FakeComparison(label="m1", std_test=0.024, std_reference=0.02, bias=0.006,
                          crmsd=0.008),
    ]
    fig = target(comparisons, normalize=False, labels=None)
    xlim = fig.axes[0].get_xlim()
    assert xlim[1] < 0.2


# -------------------------------------------------------------------------- labels


def test_target_absolute_labels_name_units_when_shared():
    comparisons = [
        _FakeComparison(label="m1", std_test=2.4, std_reference=2.0, bias=0.6, crmsd=0.8,
                          units="psu"),
    ]
    fig = target(comparisons, normalize=False, labels=None)
    assert fig.axes[0].get_ylabel() == "bias [psu]"
    assert "psu" in fig.axes[0].get_xlabel()


def test_target_absolute_labels_omit_units_when_missing():
    comparisons = [
        _FakeComparison(label="m1", std_test=2.4, std_reference=2.0, bias=0.6, crmsd=0.8,
                          units=None),
    ]
    fig = target(comparisons, normalize=False, labels=None)
    assert fig.axes[0].get_ylabel() == "bias"


def test_target_absolute_labels_omit_units_when_mixed():
    comparisons = [
        _FakeComparison(label="m1", std_test=2.4, std_reference=2.0, bias=0.6, crmsd=0.8,
                          units="degC"),
        _FakeComparison(label="m2", std_test=1.9, std_reference=2.0, bias=-0.3, crmsd=0.5,
                          units="psu"),
    ]
    fig = target(comparisons, normalize=False, labels=None)
    assert fig.axes[0].get_ylabel() == "bias"


def test_target_absolute_labels_omit_units_for_different_variables_sharing_a_unit():
    """DIC and alkalinity both land on "mmol/m^3" through this project's own unit
    conversion (see units.convert_units) while still being different quantities on
    different natural scales — a shared unit string alone shouldn't be read as license
    to label the axis, or it would read as reassurance the mixed-variable warning is
    actively contradicting."""
    comparisons = [
        _FakeComparison(label="dic", std_test=2.4, std_reference=2.0, bias=0.6, crmsd=0.8,
                          variable="dissolved_inorganic_carbon", units="mmol/m^3"),
        _FakeComparison(label="alk", std_test=1.9, std_reference=2.0, bias=-0.3, crmsd=0.5,
                          variable="alkalinity", units="mmol/m^3"),
    ]
    with pytest.warns(UserWarning, match="multiple variables"):
        fig = target(comparisons, normalize=False, labels=None)
    assert fig.axes[0].get_ylabel() == "bias"


# --------------------------------------------------------------------------- taylor


def test_taylor_normalize_false_srange_is_in_refstd_units():
    """Regression: `srange` is multiples of refstd, so raw `stds` must be divided back."""
    comparisons = [
        _FakeComparison(label="m1", std_test=3.6, std_reference=2.0, bias=0.1, crmsd=0.2),
    ]
    fig = taylor(comparisons, normalize=False, labels=None)
    ax = fig.axes[0]
    # smax = refstd * srange[1] = 2.0 * max(1.6, 1.15 * 3.6 / 2.0) = 2.0 * 2.07
    assert ax.get_xlim()[1] == pytest.approx(2.0 * 1.15 * 1.8, abs=0.1)


def test_taylor_normalize_false_mixed_reference_warns():
    comparisons = [
        _FakeComparison(label="m1", std_test=2.4, std_reference=2.0, bias=0.6, crmsd=0.8),
        _FakeComparison(label="m2", std_test=3.8, std_reference=4.0, bias=-0.3, crmsd=0.5),
    ]
    with pytest.warns(UserWarning, match="reference star, dashed arc"):
        taylor(comparisons, normalize=False, labels=None)


def test_taylor_normalize_false_labels_radial_axis_with_units():
    comparisons = [
        _FakeComparison(label="m1", std_test=2.4, std_reference=2.0, bias=0.6, crmsd=0.8,
                          units="psu"),
    ]
    fig = taylor(comparisons, normalize=False, labels=None)
    assert fig.axes[0].axis["left"].label.get_text() == "Standard deviation [psu]"


def test_paired_normalize_false_does_not_raise():
    """Regression: `paired` forwards kwargs to both panels; static `target` had no
    `normalize` and no `**kwargs`, so this used to raise `TypeError`."""
    comparisons = [
        _FakeComparison(label="m1", std_test=2.4, std_reference=2.0, bias=0.6, crmsd=0.8),
    ]
    paired(comparisons, normalize=False)


# ------------------------------------------------------------------ renderer parity


def test_both_renderers_agree_on_absolute_positions():
    comparisons = [
        _FakeComparison(label="m1", std_test=2.4, std_reference=2.0, bias=0.6, crmsd=0.8),
        _FakeComparison(label="m2", std_test=1.9, std_reference=2.0, bias=-0.3, crmsd=0.5),
    ]
    fig = target(comparisons, normalize=False, labels=None)
    static_xy = sorted(
        tuple(xy) for c in fig.axes[0].collections for xy in c.get_offsets()
    )

    obj = _interactive_target(_items(comparisons), normalize=False)
    interactive_xy = sorted(
        (float(x), float(y))
        for e in _points(obj)
        for x, y in zip(e.data["x"], e.data["y"], strict=True)
    )

    assert static_xy == pytest.approx(interactive_xy)


def test_both_renderers_agree_on_absolute_rings_and_lim():
    comparisons = [
        _FakeComparison(label="m1", std_test=2.4, std_reference=2.0, bias=0.6, crmsd=0.8),
        _FakeComparison(label="m2", std_test=1.9, std_reference=2.0, bias=-0.3, crmsd=0.5),
    ]
    fig = target(comparisons, normalize=False, labels=None)
    static_radii = sorted(c.get_radius() for c in fig.axes[0].patches)
    static_xlim = fig.axes[0].get_xlim()

    obj = _interactive_target(_items(comparisons), normalize=False)
    interactive_radii = sorted(r for r, _dash in _guide_rings(obj))
    interactive_xlim = obj.opts.get().kwargs["xlim"]

    assert static_radii == pytest.approx(interactive_radii)
    assert static_xlim == pytest.approx(interactive_xlim)


def test_both_renderers_agree_on_absolute_labels():
    comparisons = [
        _FakeComparison(label="m1", std_test=2.4, std_reference=2.0, bias=0.6, crmsd=0.8,
                          units="psu"),
    ]
    fig = target(comparisons, normalize=False, labels=None)
    static_ylabel = fig.axes[0].get_ylabel()

    obj = _interactive_target(_items(comparisons), normalize=False)
    interactive_ylabel = obj.opts.get().kwargs["ylabel"]

    assert static_ylabel == interactive_ylabel == "bias [psu]"


def test_both_renderers_skip_default_rings_on_mixed_sigma():
    comparisons = [
        _FakeComparison(label="m1", std_test=2.4, std_reference=2.0, bias=0.6, crmsd=0.8),
        _FakeComparison(label="m2", std_test=3.8, std_reference=4.0, bias=-0.3, crmsd=0.5),
    ]
    with pytest.warns(UserWarning, match="no default guide rings"):
        fig = target(comparisons, normalize=False, labels=None)
    with pytest.warns(UserWarning, match="no default guide rings"):
        obj = _interactive_target(_items(comparisons), normalize=False)

    assert list(fig.axes[0].patches) == []
    assert _guide_rings(obj) == []
