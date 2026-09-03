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
        time=None,
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
            "time": time,
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


# -------------------------------------------------------------------------- robust


def _radial_comparisons(radii, *, prefix="p"):
    """Comparisons whose normalized Target radius is exactly each of ``radii``.

    ``std_test == std_reference`` zeroes the signed crmsd term, so radius reduces to
    ``|bias|`` — the simplest possible dial for testing the limit formula itself.
    """
    return [
        _FakeComparison(label=f"{prefix}{i}", std_test=1.0, std_reference=1.0, bias=r,
                          crmsd=0.0)
        for i, r in enumerate(radii)
    ]


def _taylor_lines(fig):
    """Every Line2D on a Taylor diagram, including its parasite (aux) axes."""
    out = []
    for ax in fig.axes:
        for target_ax in (ax, *getattr(ax, "parasites", [])):
            out += list(target_ax.lines)
    return out


def _taylor_texts(fig) -> list[str]:
    """Every annotation string on a Taylor diagram, including its parasite axes."""
    out = []
    for ax in fig.axes:
        for target_ax in (ax, *getattr(ax, "parasites", [])):
            out += [t.get_text() for t in target_ax.texts]
    return out


def _target_arrows(ax):
    """Every arrow annotation on ``ax`` (an ``Annotation`` with an arrow_patch set)."""
    return [t for t in ax.texts if getattr(t, "arrow_patch", None) is not None]


def test_target_robust_limits_from_quantile_not_max():
    """21 points puts the 0.95 quantile exactly on an order statistic: 20 inliers."""
    comparisons = _radial_comparisons([1.0] * 20 + [100.0])

    fig_default = target(comparisons, labels=None)
    assert fig_default.axes[0].get_xlim()[1] == pytest.approx(115.0)

    with pytest.warns(UserWarning, match="1 of 21 points fall outside"):
        fig_robust = target(comparisons, robust=True, labels=None)
    # max(1.15 * 1.0, ring_floor=1.25, 1.2) == 1.25 — the guide-ring floor, not the
    # data.
    assert fig_robust.axes[0].get_xlim()[1] == pytest.approx(1.25)


def test_target_robust_float_is_the_quantile():
    comparisons = _radial_comparisons([1.0, 2.0, 3.0, 4.0])
    with pytest.warns(UserWarning, match="2 of 4 points fall outside"):
        fig = target(comparisons, robust=0.5, labels=None)
    # np.quantile([1, 2, 3, 4], 0.5) == 2.5
    assert fig.axes[0].get_xlim()[1] == pytest.approx(1.15 * 2.5)


def test_target_robust_warns_how_many_excluded():
    comparisons = _radial_comparisons([1.0] * 20 + [100.0])
    with pytest.warns(UserWarning, match=r"1 of 21 points fall outside"):
        target(comparisons, robust=True, labels=None)


def test_target_robust_no_exclusions_no_warning(recwarn):
    comparisons = _radial_comparisons([1.0] * 5)
    fig_default = target(comparisons, labels=None)
    fig_robust = target(comparisons, robust=True, labels=None)
    assert not [w for w in recwarn.list if "robust" in str(w.message)]
    robust_xlim = fig_robust.axes[0].get_xlim()
    default_xlim = fig_default.axes[0].get_xlim()
    assert robust_xlim == pytest.approx(default_xlim)


def test_target_robust_points_saved_by_the_floor_are_not_reported(recwarn):
    comparisons = _radial_comparisons([0.1] * 19 + [1.2])
    fig = target(comparisons, robust=True, labels=None)
    assert not [w for w in recwarn.list if "robust" in str(w.message)]
    assert fig.axes[0].get_xlim()[1] == pytest.approx(1.25)


def test_target_robust_rejects_out_of_range():
    comparisons = _radial_comparisons([1.0])
    with pytest.raises(ValueError):
        target(comparisons, robust=95, labels=None)
    with pytest.raises(ValueError):
        target(comparisons, robust=0, labels=None)


def test_target_robust_ignores_nan_radii():
    comparisons = _radial_comparisons([1.0] * 5)
    comparisons.append(
        _FakeComparison(label="nanpoint", std_test=1.0, std_reference=1.0,
                          bias=float("nan"), crmsd=0.0)
    )
    fig = target(comparisons, robust=True, labels=None)
    assert fig.axes[0].get_xlim()[1] == pytest.approx(1.25)


def test_target_robust_clips_arrows_at_the_frame():
    comparisons = [
        _FakeComparison(label="t1", std_test=1.0, std_reference=1.0, bias=0.3,
                          crmsd=0.2, time="2010"),
        _FakeComparison(label="t2", std_test=1.0, std_reference=1.0, bias=0.1,
                          crmsd=0.1, time="2020"),
    ]
    # `ax.patch` is a plain Rectangle, and `Artist.set_clip_path` special-cases that:
    # it sets `clip_box` (an equivalent, cheaper clip) rather than `clip_path`, so
    # `clip_box`, not `clip_path`, is the one that actually toggles here.
    fig_robust = target(comparisons, arrows=True, robust=True, labels=None)
    arrows_robust = _target_arrows(fig_robust.axes[0])
    assert arrows_robust, "expected an arrow between the two time steps"
    assert all(a.arrow_patch.get_clip_box() is not None for a in arrows_robust)

    fig_default = target(comparisons, arrows=True, labels=None)
    arrows_default = _target_arrows(fig_default.axes[0])
    assert arrows_default
    assert all(a.arrow_patch.get_clip_box() is None for a in arrows_default)


def test_taylor_robust_srange_from_quantile():
    stds = [3.0] * 20 + [100.0]
    comparisons = [
        _FakeComparison(label=f"m{i}", std_test=s, std_reference=1.0, bias=0.0,
                          crmsd=0.0, corr=0.9)
        for i, s in enumerate(stds)
    ]
    # The floating axes pad their view limits ~1% past `smax`, hence the loose
    # tolerance.
    fig_default = taylor(comparisons, labels=None)
    assert fig_default.axes[0].get_xlim()[1] == pytest.approx(1.15 * 100.0, rel=0.02)

    with pytest.warns(UserWarning, match="1 of 21 points"):
        fig_robust = taylor(comparisons, robust=True, labels=None)
    assert fig_robust.axes[0].get_xlim()[1] == pytest.approx(1.15 * 3.0, abs=0.1)


def test_taylor_robust_warns_and_clips_outliers():
    """A mid-correlation outlier lands inside the axes' bounding rectangle but outside
    the polar wedge — the case the automatic rectangle clip alone would miss.
    """
    comparisons = [
        _FakeComparison(label=f"m{i}", std_test=1.0, std_reference=1.0, bias=0.0,
                          crmsd=0.0, corr=0.9)
        for i in range(20)
    ]
    comparisons.append(
        _FakeComparison(label="outlier", std_test=5.0, std_reference=1.0, bias=0.0,
                          crmsd=0.0, corr=0.7)
    )
    with pytest.warns(UserWarning, match="1 of 21 points"):
        fig = taylor(comparisons, robust=True, labels=None)
    samples = [ln for ln in _taylor_lines(fig) if ln.get_marker() == "o"]
    assert samples, "expected sample points on the diagram"
    assert all(ln.get_clip_path() is not None for ln in samples)


def test_taylor_default_render_untouched():
    comparisons = [
        _FakeComparison(label=f"m{i}", std_test=1.0, std_reference=1.0, bias=0.0,
                          crmsd=0.0, corr=0.9)
        for i in range(3)
    ]
    fig = taylor(comparisons, labels=None)
    samples = [ln for ln in _taylor_lines(fig) if ln.get_marker() == "o"]
    assert samples
    assert all(ln.get_clip_path() is None for ln in samples)


def test_taylor_robust_annotate_skips_excluded_labels():
    comparisons = [
        _FakeComparison(label=f"m{i}", std_test=1.0, std_reference=1.0, bias=0.0,
                          crmsd=0.0, corr=0.9)
        for i in range(20)
    ]
    comparisons.append(
        _FakeComparison(label="outlier", std_test=5.0, std_reference=1.0, bias=0.0,
                          crmsd=0.0, corr=0.7)
    )
    with pytest.warns(UserWarning, match="1 of 21 points"):
        fig = taylor(comparisons, robust=True, labels="annotate")
    texts = _taylor_texts(fig)
    assert "outlier" not in texts
    assert "m0" in texts


def test_target_lim_overrides_everything():
    comparisons = _radial_comparisons([1.0] * 20 + [100.0])
    fig = target(comparisons, lim=3.0, labels=None)
    assert fig.axes[0].get_xlim() == pytest.approx((-3.0, 3.0))


def test_taylor_lim_sets_radial_axis():
    comparisons = [
        _FakeComparison(label=f"m{i}", std_test=1.0, std_reference=1.0, bias=0.0,
                          crmsd=0.0, corr=0.9)
        for i in range(3)
    ]
    fig = taylor(comparisons, lim=2.5, labels=None)
    assert fig.axes[0].get_xlim()[1] == pytest.approx(2.5, abs=0.1)


def test_robust_and_lim_raise():
    comparisons = _radial_comparisons([1.0])
    with pytest.raises(ValueError):
        target(comparisons, robust=True, lim=3.0, labels=None)
    taylor_comparisons = [
        _FakeComparison(label="m1", std_test=1.0, std_reference=1.0, bias=0.0,
                          crmsd=0.0, corr=0.9)
    ]
    with pytest.raises(ValueError):
        taylor(taylor_comparisons, robust=True, lim=3.0, labels=None)


def test_both_renderers_agree_on_robust_lim():
    comparisons = _radial_comparisons([1.0] * 20 + [100.0])
    with pytest.warns(UserWarning, match="1 of 21 points"):
        fig = target(comparisons, labels=None, robust=True)
    static_xlim = fig.axes[0].get_xlim()

    with pytest.warns(UserWarning, match="1 of 21 points"):
        obj = _interactive_target(_items(comparisons), robust=True)
    interactive_xlim = obj.opts.get().kwargs["xlim"]

    assert static_xlim == pytest.approx(interactive_xlim)


def test_paired_robust_does_not_raise():
    comparisons = _radial_comparisons([1.0] * 20 + [100.0])
    with pytest.warns(UserWarning, match="robust"):
        paired(comparisons, robust=True)


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
    actively contradicting.
    """
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
    `normalize` and no `**kwargs`, so this used to raise `TypeError`.
    """
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
