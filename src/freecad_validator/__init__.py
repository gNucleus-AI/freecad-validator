"""Deterministic validation for FreeCAD CAD and FEM artifacts.

The original geometry/spec API remains available at package level::

    from freecad_validator import Validator, ValidationResult

    validator = Validator()
    result = validator.validate(
        candidate_fcstd="path/to/my_model.FCStd",
        reference_fcstd="path/to/ground_truth.FCStd",
        spec_json="path/to/spec.json",
    )
    print(result.combined, result.geometry_similarity, result.cad_spec_consistency)

Placement protocols live under :mod:`freecad_validator.placement`, and solved
FreeCAD/CalculiX FEM validation lives under :mod:`freecad_validator.fem`.
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from freecad_validator.comparators.geometry import GeometryTolerances
from freecad_validator.consistency.checker import SpecTolerances
from freecad_validator.validator import CombineMethod, ValidationResult
from freecad_validator.validator import HeuristicValidator as Validator

__all__ = [
    "CombineMethod",
    "GeometryTolerances",
    "SpecTolerances",
    "Validator",
    "ValidationResult",
]

try:
    # Track the version declared in pyproject.toml so we don't drift.
    __version__ = _pkg_version("gnucleus-freecad-validator")
except PackageNotFoundError:
    # Running from a source checkout without an installed dist.
    __version__ = "0.0.0+unknown"
