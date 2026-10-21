"""Node-budget policy and default task ceiling."""

import json
import math
import re
from decimal import ROUND_CEILING, Decimal
from pathlib import Path

from freecad_validator.fem.errors import EvaluationError

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


def load_case_mesh_budget(reference_path):
    """Load only the trusted reference's sibling JSON, never candidate files.

    The local_max entry maps one preprocessing BodyN identifier to its node cap.
    Global node-budget configuration continues to use the existing task ceiling.
    """
    path = Path(reference_path).resolve().parent / "mesh_budget.json"
    if not path.exists():
        return None
    try:
        budget = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_keys)
    except (OSError, ValueError) as exc:
        raise EvaluationError(f"Cannot load case mesh_budget.json: {exc}") from exc
    if not isinstance(budget, dict) or set(budget) != {"local_max"}:
        raise EvaluationError("mesh_budget.json must contain a local_max object")
    local = budget["local_max"]
    if not isinstance(local, dict) or len(local) != 1:
        raise EvaluationError("local_max must specify exactly one BodyN node cap")
    for body, entry in local.items():
        count = entry
        if isinstance(entry, dict):
            if set(entry) != {"max_node_count", "geometry"}:
                raise EvaluationError("Local target requires max_node_count and geometry")
            count = entry["max_node_count"]
            validate_target_geometry(entry["geometry"])
        if not re.fullmatch(r"Body[1-9][0-9]*", body) or type(count) is not int or count <= 0:
            raise EvaluationError("local_max requires a BodyN key and a positive integer node cap")
    return budget


def validate_target_geometry(geometry):
    """Validate a world-coordinate reference locator, not candidate tolerances."""
    if not isinstance(geometry, dict) or set(geometry) != {"volume_mm3", "center_mm", "bounds_mm"}:
        raise EvaluationError("Target geometry requires volume_mm3, center_mm and bounds_mm")
    volume = geometry["volume_mm3"]
    if type(volume) not in (int, float) or not math.isfinite(volume) or volume <= 0:
        raise EvaluationError("Target geometry volume must be positive and finite")
    for key, length in (("center_mm", 3), ("bounds_mm", 6)):
        values = geometry[key]
        if (
            not isinstance(values, list)
            or len(values) != length
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
        ):
            raise EvaluationError(f"Target geometry {key} requires {length} finite coordinates")
    bounds = geometry["bounds_mm"]
    if any(not bounds[i] < bounds[i + 3] for i in range(3)):
        raise EvaluationError("Target geometry bounds must have positive extents")
    if any(not bounds[i] <= geometry["center_mm"][i] <= bounds[i + 3] for i in range(3)):
        raise EvaluationError("Target geometry center must lie within bounds")


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate mesh budget key: {key}")
        result[key] = value
    return result


def score_local_node_budget(budget, reference_counts, candidate_counts):
    """Award full local credit through the cap, then log2 decay to zero at twice it."""
    body, entry = next(iter(budget["local_max"].items()))
    cap = entry["max_node_count"] if isinstance(entry, dict) else entry
    reference_count = reference_counts.get(body)
    if type(reference_count) is not int or not 0 < reference_count <= cap:
        raise EvaluationError("Reference local body must have positive node count within local_max")
    if any(type(n) is not int or n < 0 for n in candidate_counts.values()):
        raise EvaluationError("Candidate body node counts must be nonnegative integers")
    count = candidate_counts.get(body, 0)
    passed = 0 < count <= cap
    score = 0.0
    if passed:
        score = 100.0
    elif cap < count < 2 * cap:
        score = 100.0 * (1.0 - math.log2(count / cap))
    return {
        "score": score,
        "body": body,
        "max_node_count": cap,
        "reference_node_count": reference_count,
        "candidate_node_count": count,
        "ratio_to_cap": count / cap,
        "status": (
            "missing_body"
            if body not in candidate_counts
            else "within_budget"
            if passed
            else "outside_budget"
        ),
    }
