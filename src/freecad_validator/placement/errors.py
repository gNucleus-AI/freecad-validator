"""Stable exceptions for placement-validation operation boundaries."""


class PlacementValidationError(RuntimeError):
    """Base exception for placement-validation operations."""


class InvalidFCStdError(PlacementValidationError):
    """An input file cannot provide valid geometry for the requested operation."""


class AlignmentError(PlacementValidationError):
    """A valid FCStd pair cannot be aligned under the configured constraints."""


class MetricComputationError(PlacementValidationError):
    """A metric cannot be computed from an otherwise valid aligned context."""
