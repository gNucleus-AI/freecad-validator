"""Regression coverage for decimal node-budget rounding boundaries."""

import pytest

from freecad_validator.fem.mesh_budget import default_node_cap


@pytest.mark.parametrize(
    "reference_nodes,ratio,expected",
    [
        (49999, 1.1, 55000),
        (50000, 1.1, 55000),
        (50001, 1.1, 56000),
        (50000, 1.0999999999999999, 55000),
        (50000, 1.1000000000000003, 56000),
        (126125, 1.3, 164000),
    ],
)
def test_default_cap_uses_decimal_ratio(reference_nodes, ratio, expected):
    assert default_node_cap(reference_nodes, ratio) == expected


@pytest.mark.parametrize("ratio", [float("nan"), float("inf"), 1.0, 0.0, -1.0])
def test_invalid_ratio_is_rejected(ratio):
    with pytest.raises(ValueError, match="finite and greater than 1"):
        default_node_cap(50000, ratio)
