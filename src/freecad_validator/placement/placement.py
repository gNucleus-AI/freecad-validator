"""Public localized-placement metric API contract."""

from __future__ import annotations

from freecad_validator.placement.alignment import AlignedFCStdPair
from freecad_validator.placement.models import PlacementConfig, PlacementResult


def calculate_placement(
    aligned: AlignedFCStdPair,
    config: PlacementConfig | None = None,
) -> PlacementResult:
    """Calculate only localized percentile deviation and its mapped score.

    This operation must not realign the models, run FreeCAD Booleans, calculate
    surface coverage, or aggregate any other score. Unexpected computation
    failures raise ``MetricComputationError``.
    """
    raise NotImplementedError("calculate_placement implementation has not been contributed")
