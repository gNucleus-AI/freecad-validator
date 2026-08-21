"""Public rigid-alignment API contract for FCStd pairs."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from freecad_validator.placement.models import AlignmentConfig, AlignmentResult


@runtime_checkable
class AlignedFCStdPair(Protocol):
    """Opaque reusable context returned by :func:`align_fcstd`.

    Implementations may cache FreeCAD shapes, samples, meshes, and native
    alignment objects, but those objects are deliberately not public fields.
    Metric functions must accept the context returned by the same
    implementation of ``align_fcstd``.
    """

    @property
    def reference_path(self) -> Path:
        """Resolved reference FCStd path."""
        ...

    @property
    def candidate_path(self) -> Path:
        """Resolved candidate FCStd path."""
        ...

    @property
    def alignment_result(self) -> AlignmentResult:
        """Serializable transform and alignment diagnostics."""
        ...


def align_fcstd(
    reference_fcstd: str | Path,
    candidate_fcstd: str | Path,
    config: AlignmentConfig | None = None,
) -> AlignedFCStdPair:
    """Load an FCStd pair and rigidly align candidate onto reference.

    Contract:
      * apply rotation and translation only; never scale, shear, or reflect;
      * be deterministic for the same files and configuration;
      * return an opaque context reusable by every metric call;
      * raise ``InvalidFCStdError`` for invalid inputs and ``AlignmentError``
        when a valid pair cannot be aligned.

    The algorithm is intentionally unspecified so independent authors can
    contribute implementations without changing the public API.
    """
    raise NotImplementedError("align_fcstd implementation has not been contributed")
