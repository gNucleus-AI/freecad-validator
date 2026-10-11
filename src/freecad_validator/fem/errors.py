"""Errors distinguished from measured candidate failures."""


class ExtractionError(RuntimeError):
    """An adapter could not produce a valid extraction payload."""


class EvaluationError(RuntimeError):
    """Reference data or evaluator infrastructure could not be evaluated."""
