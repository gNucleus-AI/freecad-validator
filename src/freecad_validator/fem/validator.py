"""Configured object-oriented entry point for FEM validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from freecad_validator.fem.schema import (
    DISP_TOL,
    GROSS_TOL,
    MESH_BUDGET_ZERO_RATIO,
    STRESS_TOL,
    ScoringReport,
)
from freecad_validator.fem.step_interface import (
    DEFAULT_SUBPROCESS_TIMEOUT_SECONDS,
    score_step_fcstd,
    score_trusted_payloads,
)


@dataclass(frozen=True, slots=True)
class FEMValidator:
    """Reusable configuration for deterministic FEM scoring.

    The FCStd path runs FreeCAD and CalculiX subprocesses. Callers validating
    untrusted candidate files should execute it in an isolated container.
    """

    freecad_cmd: str | None = None
    extract_dir: str | None = None
    displacement_tolerance: float = DISP_TOL
    stress_tolerance: float = STRESS_TOL
    gross_error_tolerance: float = GROSS_TOL
    mesh_budget_zero_ratio: float = MESH_BUDGET_ZERO_RATIO
    require_preprocessing: bool = False
    require_boolean: bool = False
    timeout_seconds: float = DEFAULT_SUBPROCESS_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        tolerances = {
            "displacement_tolerance": self.displacement_tolerance,
            "stress_tolerance": self.stress_tolerance,
            "gross_error_tolerance": self.gross_error_tolerance,
        }
        for name, value in tolerances.items():
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.mesh_budget_zero_ratio <= 1.0:
            raise ValueError("mesh_budget_zero_ratio must be greater than 1.0")
        if self.timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be positive")

    def validate(
        self,
        *,
        step_path: str,
        reference_fcstd: str,
        candidate_fcstd: str,
    ) -> ScoringReport:
        """Extract, replay, and score one candidate FCStd."""
        return score_step_fcstd(
            step_path,
            reference_fcstd,
            candidate_fcstd,
            freecad_cmd=self.freecad_cmd,
            extract_dir=self.extract_dir,
            disp_tol=self.displacement_tolerance,
            stress_tol=self.stress_tolerance,
            gross_tol=self.gross_error_tolerance,
            mesh_budget_zero_ratio=self.mesh_budget_zero_ratio,
            require_preprocessing=self.require_preprocessing,
            require_boolean=self.require_boolean,
            timeout_seconds=self.timeout_seconds,
        )

    def validate_trusted_payloads(
        self,
        *,
        target_geometry: dict[str, Any],
        reference_submission: dict[str, Any],
        candidate_submission: dict[str, Any],
        geometry_source: str = "STEP",
    ) -> ScoringReport:
        """Score trusted, validator-generated payloads without FreeCAD.

        Never pass candidate-controlled JSON to this low-level API because it
        trusts replay-verification fields produced by the FCStd adapter.
        """
        return score_trusted_payloads(
            target_geometry,
            reference_submission,
            candidate_submission,
            disp_tol=self.displacement_tolerance,
            stress_tol=self.stress_tolerance,
            gross_tol=self.gross_error_tolerance,
            mesh_budget_zero_ratio=self.mesh_budget_zero_ratio,
            geometry_source=geometry_source,
        )
