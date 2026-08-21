"""Implementation-neutral placement-validation API contract.

The function bodies are contribution points. The public names, operation
boundaries, serializable models, and failure semantics are defined here first
so alignment and metric implementations can be authored independently.
"""

from freecad_validator.placement.alignment import AlignedFCStdPair, align_fcstd
from freecad_validator.placement.errors import (
    AlignmentError,
    InvalidFCStdError,
    MetricComputationError,
    PlacementValidationError,
)
from freecad_validator.placement.models import (
    AlignmentConfig,
    AlignmentResult,
    AxisAlignedBoundingBox,
    PlacementConfig,
    PlacementResult,
    PlacementValidationResult,
    SurfaceCoverageConfig,
    SurfaceCoverageResult,
    VolumetricIoUResult,
)
from freecad_validator.placement.placement import calculate_placement
from freecad_validator.placement.surface import calculate_surface_coverage
from freecad_validator.placement.volume import calculate_volumetric_iou

__all__ = [
    "AlignedFCStdPair",
    "AlignmentConfig",
    "AlignmentError",
    "AlignmentResult",
    "AxisAlignedBoundingBox",
    "InvalidFCStdError",
    "MetricComputationError",
    "PlacementConfig",
    "PlacementResult",
    "PlacementValidationError",
    "PlacementValidationResult",
    "SurfaceCoverageConfig",
    "SurfaceCoverageResult",
    "VolumetricIoUResult",
    "align_fcstd",
    "calculate_placement",
    "calculate_surface_coverage",
    "calculate_volumetric_iou",
]
