"""FreeCAD-free contract tests for placement validation."""

from __future__ import annotations

import pytest

import freecad_validator.placement as placement_api

EXPECTED_EXPORTS = {
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
}


def test_public_exports() -> None:
    assert set(placement_api.__all__) == EXPECTED_EXPORTS


def test_result_models_serialize_to_json_values() -> None:
    alignment = placement_api.AlignmentResult(
        rotation=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        translation_mm=(0.0, 0.0, 0.0),
        rmse_mm=0.1,
        normalized_rmse=0.01,
        identity_rmse_mm=1.0,
        details={"backend": "contributor-defined"},
    )
    placement = placement_api.PlacementResult(
        score=0.8,
        percentile=98.0,
        surface_deviation_mm=1.2,
        reference_to_candidate_deviation_mm=1.0,
        candidate_to_reference_deviation_mm=1.2,
    )
    volume = placement_api.VolumetricIoUResult(
        available=False,
        iou=None,
        error="example Boolean failure",
    )
    surface = placement_api.SurfaceCoverageResult(
        score=0.9,
        reference_coverage=0.85,
        candidate_coverage=0.95,
        tolerance_mm=0.25,
    )

    result = placement_api.PlacementValidationResult(
        alignment=alignment,
        placement=placement,
        volumetric_iou=volume,
        surface_coverage=surface,
    ).model_dump(mode="json")

    assert result["placement"]["score"] == 0.8
    assert result["volumetric_iou"]["iou"] is None


def test_models_reject_contract_violations() -> None:
    with pytest.raises(ValueError, match="zero_score_deviation_mm must exceed"):
        placement_api.PlacementConfig(
            matched_deviation_mm=1.0,
            zero_score_deviation_mm=1.0,
        )

    with pytest.raises(ValueError):
        placement_api.PlacementResult(
            score=1.01,
            percentile=98.0,
            surface_deviation_mm=0.0,
            reference_to_candidate_deviation_mm=0.0,
            candidate_to_reference_deviation_mm=0.0,
        )

    with pytest.raises(ValueError, match="minimum_mm must not exceed"):
        placement_api.AxisAlignedBoundingBox(
            minimum_mm=(1.0, 0.0, 0.0),
            maximum_mm=(0.0, 1.0, 1.0),
        )


def test_operations_are_explicitly_unimplemented() -> None:
    with pytest.raises(NotImplementedError):
        placement_api.align_fcstd("reference.FCStd", "candidate.FCStd")

    aligned_context = object()
    for operation in (
        placement_api.calculate_placement,
        placement_api.calculate_surface_coverage,
        placement_api.calculate_volumetric_iou,
    ):
        with pytest.raises(NotImplementedError):
            operation(aligned_context)
