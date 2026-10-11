"""Distinguish invalid submissions from failures to evaluate geometry."""

from freecad_validator.fem.errors import EvaluationError


class CandidateGeometryError(EvaluationError):
    """The submitted document has no usable valid analysis geometry."""


class MissingCleanBodiesError(EvaluationError):
    """Required independent clean bodies are not saved; preprocessing credit is zero."""

    def __init__(self, message: str, input_role: str):
        self.input_role = input_role
        super().__init__(f"{input_role} clean bodies: {message}")


class BodyCorrespondenceError(EvaluationError):
    """Valid input geometry could not be reliably mapped to original bodies."""

    def __init__(self, message: str, input_role: str):
        self.input_role = input_role
        super().__init__(f"{input_role} body correspondence: {message}")
