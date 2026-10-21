"""Apply mesh-budget credit to a FEM report."""

import math
from copy import deepcopy

from freecad_validator.fem.errors import EvaluationError
from freecad_validator.fem.schema import grade_from_score


def apply_local_mesh_budget(report, evaluation):
    """Replace mesh-budget category credit before the preprocessing multiplier.

    Global node-cap validity gates remain in force. Regional failures reduce
    this category only, using its existing weight.
    """
    if "preprocessing" in report.subscores_details:
        raise EvaluationError("Local mesh budget must be applied before preprocessing")
    if not isinstance(evaluation, dict):
        raise EvaluationError("Missing case local mesh-budget evaluation")
    score = evaluation.get("score")
    if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 100:
        raise EvaluationError("Invalid case local mesh-budget score")
    combined = deepcopy(report)
    detail = combined.subscores_details["mesh_budget"]
    if "local_max" in detail:
        raise EvaluationError("Local mesh budget has already been applied")
    global_score = detail["raw_score"]
    detail["global"] = {"score": global_score}
    raw = min(global_score, score)
    old_points = detail["weighted_points"]
    points = round(raw * detail["weight"], 3)
    detail.update(raw_score=round(raw, 2), weighted_points=points, local_max=evaluation)
    combined.subscores["mesh_budget"] = round(raw, 2)
    if not combined.gates_triggered:
        combined.overall_score = max(0.0, combined.overall_score - old_points + points)
    combined.grade = grade_from_score(combined.overall_score)
    combined.pass_fail_flags["mesh_within_budget"] = raw == 100
    combined.evidence.append(f"case local_max mesh budget: {score:.6f}/100")
    if score < 100:
        combined.engineering_feedback.append(
            "The configured body loses local mesh-budget credit; see its node count, "
            "budget and status. Valid local counts above the budget lose credit "
            "logarithmically, reaching zero at twice the budget."
        )
    return combined
