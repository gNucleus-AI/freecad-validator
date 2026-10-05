"""Errors that prevent a reliable preprocessing geometry evaluation."""


class EvaluationError(RuntimeError):
    """Geometry inputs or the evaluation runtime could not be evaluated reliably."""


class CandidateGeometryError(EvaluationError):
    """The submitted document has no usable valid analysis geometry."""


class BodyCorrespondenceError(EvaluationError):
    """Valid input geometry could not be reliably mapped to original bodies."""

    def __init__(self, message: str, input_role: str):
        self.input_role = input_role
        super().__init__(f"{input_role} body correspondence: {message}")
