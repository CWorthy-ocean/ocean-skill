"""compare(variables=...) given one spec on its own, not wrapped in a list.

A lone pair-spec dict used to be iterated as its keys -- compare() then looked for
variables literally named "test" and "reference", skipped every pair, and only
failed later at ``.plot()`` with "no comparisons to plot". A bare string was walked
character by character the same way. Both are now a one-element list.
"""

from __future__ import annotations

from unittest import mock

import pytest

from ocean_skill import comparison

TEMP = "sea_water_potential_temperature"
SALT = "sea_water_practical_salinity"
MLD_SIGMA_THETA = "ocean_mixed_layer_thickness_defined_by_sigma_theta"

DECLARED = {
    "model": {"variables": [TEMP, SALT]},
    "obs": {"variables": [TEMP, SALT, MLD_SIGMA_THETA]},
}


def _formed_variables(**kwargs):
    """Run compare() against mocked catalog entries, recording each variable formed."""
    formed = []
    with (
        mock.patch(
            "ocean_skill.catalog.resolve", lambda n: mock.Mock(metadata=DECLARED[n])
        ),
        mock.patch.object(
            comparison.Comparison,
            "align",
            lambda self, refresh=False: formed.append(self.variable),
        ),
    ):
        comparison.compare(reference="obs", test="model", **kwargs)
    return formed


PAIR = {
    "test": {"calculate": "mld", "method": "density_threshold"},
    "reference": MLD_SIGMA_THETA,
}


@pytest.mark.parametrize(
    "spec",
    [
        PAIR,
        "temperature",
        {"calculate": "mld", "method": "density_threshold"},
    ],
    ids=["pair-spec", "string", "calculate"],
)
def test_a_lone_spec_is_the_same_as_a_one_element_list(spec):
    lone = _formed_variables(variables=spec)
    assert len(lone) == 1
    assert lone == _formed_variables(variables=[spec])


def test_a_lone_pair_spec_is_not_iterated_as_its_keys(capsys):
    assert _formed_variables(variables=PAIR) == [PAIR]
    assert "not available" not in capsys.readouterr().out


def test_a_lone_string_resolves_to_its_standard_name():
    assert _formed_variables(variables="temperature") == [TEMP]


def test_a_lone_pair_spec_survives_the_aggregate_fan():
    """A lone spec is wrapped before the aggregate= fan, not after.

    An aggregate= list re-enters compare() once per spec with the original
    arguments, so the fan would otherwise iterate the bare dict's keys.
    """
    formed = _formed_variables(
        variables=PAIR,
        aggregate=[
            {"time": "mean"},
            {"time": [{"groupby": "month", "reduce": "mean"}, "var"]},
        ],
    )
    assert formed == [PAIR, PAIR]
