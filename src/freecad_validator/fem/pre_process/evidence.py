"""Shared adapter for the read-only preprocessing worker's result contract."""

from freecad_validator.fem.errors import ExtractionError
from freecad_validator.fem.setup_similarity import _number


def preprocessing_score(geometry):
    """Return G credit; None denotes a completed comparison with no applicable edits."""
    status = geometry.get("status")
    if status in ("candidate_invalid", "missing_clean_bodies"):
        return 0.0
    if status == "evaluation_error":
        raise ExtractionError("Preprocessing evaluation failed: " + str(geometry.get("error")))
    if "score" not in geometry or "reference_changed_body_count" not in geometry:
        raise ExtractionError("Preprocessing worker returned incomplete geometry evidence")
    score = geometry["score"]
    if score is None:
        if (
            geometry["reference_changed_body_count"] != 0
            or geometry.get("extra_changed_body_count") != 0
        ):
            raise ExtractionError("Preprocessing worker omitted required geometry credit")
    elif not _number(score) or not 0 <= score <= 1:
        raise ExtractionError("Preprocessing score must be finite and in [0, 1]")
    return score
