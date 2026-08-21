"""Public tolerance-based surface-coverage API contract."""

from __future__ import annotations

from freecad_validator.placement.alignment import AlignedFCStdPair
from freecad_validator.placement.models import SurfaceCoverageConfig, SurfaceCoverageResult


def calculate_surface_coverage(
    aligned: AlignedFCStdPair,
    config: SurfaceCoverageConfig | None = None,
) -> SurfaceCoverageResult:
    """Calculate bidirectional surface coverage after the shared alignment.

    The score is the average of both directional coverage fractions under the
    resolved distance tolerance. This operation must not realign the models or
    claim to calculate a set-theoretic surface-area intersection-over-union.
    Unexpected computation failures raise ``MetricComputationError``.
    """
    raise NotImplementedError("calculate_surface_coverage implementation has not been contributed")
