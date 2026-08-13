"""The metric colour policy: one answer, asked by both renderers.

Pure-function tests — no figures — because the point of `metric_colors` is that the
static and interactive renderers cannot disagree about a panel's colours or range, and
that is settled before either of them draws anything.
"""

from __future__ import annotations

import numpy as np
import pytest

from ocean_skill.colormaps import METRIC_LIMIT_GROUPS, metric_colors

CHL = "mass_concentration_of_chlorophyll_a_in_sea_water"
NITRATE = "mole_concentration_of_nitrate_in_sea_water"


def test_correlation_limits_are_fixed_and_ignore_the_data():
    """Correlation has an absolute scale; percentiles would paint 0.9 like 1.0."""
    tight = metric_colors("corr", np.full((4, 4), 0.95))
    wide = metric_colors("corr", np.linspace(-1, 1, 16))
    assert tight.clim() == (-1.0, 1.0)
    assert wide.clim() == (-1.0, 1.0)


@pytest.mark.parametrize("values", [np.linspace(-3, 1, 20), np.linspace(0.5, 4, 20)])
def test_bias_is_always_symmetric_about_zero(values):
    """White has to sit at no-error even when every cell is biased the same way."""
    colors = metric_colors("bias", values, standard_name=NITRATE)
    assert colors.vmin == pytest.approx(-colors.vmax)
    assert colors.vmax > 0


def test_sigma_ratio_is_symmetric_about_one_and_never_goes_negative():
    colors = metric_colors("sigma_ratio", np.linspace(0.2, 5.0, 40))
    assert (colors.vmin + colors.vmax) / 2 == pytest.approx(1.0)
    assert colors.vmin >= 0.0


@pytest.mark.parametrize("metric", ["rmse", "mae", "crmsd", "std_test", "n"])
def test_magnitudes_pin_zero(metric):
    """Zero must be the same colour in every figure, whatever the data's own floor."""
    colors = metric_colors(metric, np.linspace(3.0, 9.0, 20))
    assert colors.vmin == 0.0


def test_a_count_uses_its_real_maximum_where_a_magnitude_uses_a_percentile():
    counts = np.array([1, 2, 3, 4, 100])
    assert metric_colors("n", counts).vmax == 100.0
    # the same values as an error magnitude: the outlier must not flatten the rest
    assert metric_colors("rmse", counts.astype(float)).vmax < 100.0


def test_a_signed_metric_never_gets_the_variables_log_range():
    """`_RANGES` pins chlorophyll to LogNorm(0.01, 10) — the *field's* display range.

    A LogNorm on a bias map is meaningless (it has negatives) and pinning an rmse panel
    to 0.01-10 pins it to the wrong quantity entirely.
    """
    bias = metric_colors("bias", np.linspace(-2, 2, 20), standard_name=CHL)
    rmse = metric_colors("rmse", np.linspace(0, 30, 20), standard_name=CHL)
    assert not bias.log and not rmse.log
    assert bias.vmin < 0
    assert rmse.vmax > 10  # not clipped to the field's own display maximum


def test_the_field_itself_does_keep_the_variables_log_range():
    mean = metric_colors("mean_test", np.linspace(0.05, 8.0, 20), standard_name=CHL)
    assert mean.log
    assert (mean.vmin, mean.vmax) == (0.01, 10.0)


def test_a_norm_and_a_clim_describe_the_same_limits():
    import matplotlib.colors as mcolors

    colors = metric_colors("bias", np.linspace(-1, 1, 9), standard_name=NITRATE)
    norm = colors.norm()
    assert isinstance(norm, mcolors.Normalize)
    assert (norm.vmin, norm.vmax) == colors.clim()

    log = metric_colors("mean_test", np.linspace(0.05, 8.0, 9), standard_name=CHL)
    assert isinstance(log.norm(), mcolors.LogNorm)


@pytest.mark.parametrize("group", METRIC_LIMIT_GROUPS)
def test_a_limit_group_shares_one_scale_when_given_pooled_values(group):
    """Comparing the model's variability with the observations' needs one bar."""
    pooled = np.concatenate([np.linspace(1, 3, 10), np.linspace(2, 8, 10)])
    limits = {
        metric_colors(name, pooled, standard_name=NITRATE).clim() for name in group
    }
    assert len(limits) == 1


def test_an_unregistered_metric_follows_the_sign_of_its_own_data():
    """So a metric added by metrics.register() draws sensibly before it has a row."""
    signed = metric_colors("willmott_signed", np.linspace(-2, 2, 20))
    assert signed.vmin == pytest.approx(-signed.vmax)
    positive = metric_colors("willmott", np.linspace(0.2, 0.95, 20))
    assert positive.vmin == 0.0


@pytest.mark.parametrize("metric", ["bias", "rmse", "sigma_ratio", "n"])
@pytest.mark.parametrize(
    "values", [None, np.full(6, np.nan), np.zeros(6)], ids=["none", "all-nan", "flat"]
)
def test_no_metric_ever_returns_a_degenerate_or_nonfinite_range(metric, values):
    """An all-NaN panel is normal (a fully masked domain) and must still draw."""
    colors = metric_colors(metric, values)
    assert np.isfinite([colors.vmin, colors.vmax]).all()
    assert colors.vmax > colors.vmin
