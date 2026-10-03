"""Node-budget policy and default task ceiling."""

import math
from decimal import ROUND_CEILING, Decimal

MESH_BUDGET_ZERO_RATIO = 1.3
CEILING_ROUNDING = 1000


def default_node_cap(reference_nodes, ratio=MESH_BUDGET_ZERO_RATIO):
    """Derive a ceiling from reference nodes, rounded up to a whole thousand."""
    if type(reference_nodes) is not int or reference_nodes <= 0:
        raise ValueError("reference source-node baseline must be a positive integer")
    if not math.isfinite(ratio) or ratio <= 1:
        raise ValueError("mesh-budget ceiling ratio must be finite and greater than 1")
    # Preserve the configured decimal multiplier at exact rounding boundaries.
    blocks = Decimal(reference_nodes) * Decimal(str(ratio)) / CEILING_ROUNDING
    return int(blocks.to_integral_value(rounding=ROUND_CEILING)) * CEILING_ROUNDING


def node_budget_score(candidate_nodes, reference_nodes, cap):
    """Return (score, status); invalid references are evaluator errors.

    The cap is an allowed budget, inclusive: every positive candidate count
    within it receives full credit. Accuracy and mesh validity are checked
    separately; reference size does not impose an additional passing threshold.
    """
    if type(cap) is not int or cap <= 0:
        raise ValueError("max_node_count must be a positive integer")
    if type(reference_nodes) is not int or not 0 < reference_nodes <= cap:
        raise ValueError(
            "reference source-node baseline must be positive and within the task ceiling"
        )
    if type(candidate_nodes) is not int or candidate_nodes <= 0 or candidate_nodes > cap:
        return 0.0, "invalid"
    return 100.0, "full"
