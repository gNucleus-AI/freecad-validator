"""Public volumetric-IoU API contract."""

from __future__ import annotations

from freecad_validator.placement.alignment import AlignedFCStdPair
from freecad_validator.placement.models import VolumetricIoUResult


def calculate_volumetric_iou(aligned: AlignedFCStdPair) -> VolumetricIoUResult:
    """Calculate solid-volume Jaccard IoU after the shared alignment.

    This operation must not realign the models or aggregate another score.
    Normal FreeCAD Boolean failure is represented by
    ``VolumetricIoUResult(available=False, ...)`` so batch callers retain the
    other metrics. Invalid contexts and contract violations may raise
    ``MetricComputationError``.
    """
    raise NotImplementedError("calculate_volumetric_iou implementation has not been contributed")
