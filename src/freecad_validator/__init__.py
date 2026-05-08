"""``gnucleus-freecad-validator`` — heuristic geometry-similarity +
spec-consistency scoring for FreeCAD parts.

Public API::

    from freecad_validator import Validator, ValidationResult

    validator = Validator()
    result = validator.validate(
        candidate_fcstd="path/to/my_model.FCStd",
        reference_fcstd="path/to/ground_truth.FCStd",
        spec_json="path/to/spec.json",
    )
    print(result.combined, result.geometry_similarity, result.cad_spec_consistency)
"""
from __future__ import annotations

from freecad_validator.validator import HeuristicValidator as Validator
from freecad_validator.validator import ValidationResult


__all__ = ["Validator", "ValidationResult"]
__version__ = "0.1.0.dev0"
