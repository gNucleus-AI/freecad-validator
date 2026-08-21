"""Serializable models for the placement-validation API contract.

These models intentionally describe outputs, not implementation state. In
particular, FreeCAD shapes, sampled points, meshes, and alignment-engine
objects must remain private to an implementation of :class:`AlignedFCStdPair`.
"""

from __future__ import annotations

import math
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
NonNegativeFloat = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
PositiveFloat = Annotated[float, Field(gt=0.0, allow_inf_nan=False)]
UnitFloat = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
Vector3 = tuple[FiniteFloat, FiniteFloat, FiniteFloat]
RotationMatrix3 = tuple[Vector3, Vector3, Vector3]


class APIModel(BaseModel):
    """Common behavior for immutable, forward-compatible public records."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class AlignmentConfig(APIModel):
    """Controls FCStd sampling and rigid SE(3) alignment only."""

    sample_count: int = Field(default=6000, ge=128, strict=True)
    tessellation_mm: PositiveFloat = 0.2
    random_seed: int = Field(default=1729, ge=0, strict=True)
    trim: FiniteFloat = Field(default=0.98, gt=0.0, le=1.0)
    refine_top_k: int = Field(default=6, ge=1, strict=True)
    max_refine_iterations: int = Field(default=40, ge=1, strict=True)


class PlacementConfig(APIModel):
    """Controls the percentile-deviation placement metric only."""

    percentile: FiniteFloat = Field(default=98.0, ge=95.0, le=99.0)
    matched_deviation_mm: NonNegativeFloat = 0.25
    zero_score_deviation_mm: NonNegativeFloat = 5.0

    @model_validator(mode="after")
    def validate_score_thresholds(self) -> PlacementConfig:
        if self.zero_score_deviation_mm <= self.matched_deviation_mm:
            raise ValueError("zero_score_deviation_mm must exceed matched_deviation_mm")
        return self


class SurfaceCoverageConfig(APIModel):
    """Controls tolerance-based bidirectional surface coverage."""

    tolerance_relative_to_reference_bbox: FiniteFloat = Field(
        default=0.01,
        gt=0.0,
    )


class AlignmentResult(APIModel):
    """Serializable rigid-alignment diagnostics.

    The transform maps candidate coordinates into the reference coordinate
    system as ``rotation @ candidate + translation_mm``. Scaling, reflection,
    and shearing are outside this contract.
    """

    rotation: RotationMatrix3
    translation_mm: Vector3
    rmse_mm: NonNegativeFloat
    normalized_rmse: NonNegativeFloat
    identity_rmse_mm: NonNegativeFloat | None = None
    details: dict[str, JsonValue] = Field(default_factory=dict)


class PlacementResult(APIModel):
    """Localized deviation score computed after the shared alignment."""

    score: UnitFloat
    percentile: FiniteFloat = Field(ge=95.0, le=99.0)
    surface_deviation_mm: NonNegativeFloat
    reference_to_candidate_deviation_mm: NonNegativeFloat
    candidate_to_reference_deviation_mm: NonNegativeFloat
    details: dict[str, JsonValue] = Field(default_factory=dict)


class AxisAlignedBoundingBox(APIModel):
    """Axis-aligned bounds in the aligned reference coordinate system."""

    minimum_mm: Vector3
    maximum_mm: Vector3

    @model_validator(mode="after")
    def validate_bounds(self) -> AxisAlignedBoundingBox:
        if any(low > high for low, high in zip(self.minimum_mm, self.maximum_mm, strict=True)):
            raise ValueError("minimum_mm must not exceed maximum_mm on any axis")
        return self


class VolumetricIoUResult(APIModel):
    """Solid-volume Jaccard score and mismatch diagnostics.

    Boolean failure is data, not an exception: implementations return
    ``available=False``, ``iou=None``, and a non-empty ``error``. Invalid input
    and alignment failures remain exceptions at the operation boundary.
    """

    available: bool = Field(strict=True)
    iou: UnitFloat | None
    reference_volume_mm3: NonNegativeFloat | None = None
    candidate_volume_mm3: NonNegativeFloat | None = None
    intersection_volume_mm3: NonNegativeFloat | None = None
    union_volume_mm3: NonNegativeFloat | None = None
    symmetric_difference_mm3: NonNegativeFloat | None = None
    symmetric_difference_bbox: AxisAlignedBoundingBox | None = None
    error: str | None = None
    details: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_availability(self) -> VolumetricIoUResult:
        if self.available:
            required = (
                self.iou,
                self.reference_volume_mm3,
                self.candidate_volume_mm3,
                self.intersection_volume_mm3,
                self.union_volume_mm3,
                self.symmetric_difference_mm3,
            )
            if any(value is None for value in required):
                raise ValueError("available volumetric IoU requires all score and volume fields")
            if self.error is not None:
                raise ValueError("available volumetric IoU must not contain an error")
        else:
            if self.iou is not None:
                raise ValueError("unavailable volumetric IoU must have iou=None")
            if self.error is None or not self.error.strip():
                raise ValueError("unavailable volumetric IoU requires an error")
        return self


class SurfaceCoverageResult(APIModel):
    """Tolerance-based symmetric surface-coverage score.

    ``score`` is the equal-direction average of ``reference_coverage`` and
    ``candidate_coverage``. It is not a set-theoretic surface-area IoU.
    """

    score: UnitFloat
    reference_coverage: UnitFloat
    candidate_coverage: UnitFloat
    tolerance_mm: PositiveFloat
    details: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_equal_direction_average(self) -> SurfaceCoverageResult:
        expected = 0.5 * (self.reference_coverage + self.candidate_coverage)
        if not math.isclose(self.score, expected, rel_tol=1.0e-12, abs_tol=1.0e-12):
            raise ValueError("score must equal the average of both directional coverages")
        return self


class PlacementValidationResult(APIModel):
    """All independently reported outputs from one aligned FCStd pair."""

    alignment: AlignmentResult
    placement: PlacementResult
    volumetric_iou: VolumetricIoUResult
    surface_coverage: SurfaceCoverageResult
