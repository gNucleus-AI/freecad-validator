"""Compare candidate-stored FEM fields with a trusted solver replay."""

from __future__ import annotations

import math
from typing import Any

REPLAY_REL_TOL = 0.02
REPLAY_HARD_FAIL_REL_TOL = 0.10
REPLAY_SCALAR_FIELDS = (
    "DisplacementLengths",
    "vonMises",
    "MaxShear",
    "Temperature",
    "EigenmodeFrequencies",
)
STATIC_REQUIRED_REPLAY_FIELDS = (
    "DisplacementLengths",
    "DisplacementVectors",
    "vonMises",
)


def select_scored_results(
    stored: dict[str, Any],
    replayed: dict[str, Any],
    verification: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    if verification.get("status") == "accepted_mismatch":
        return stored, "stored"
    return replayed, "replayed"


def field_error(stored: list[Any], replayed: list[Any]) -> dict[str, Any]:
    if len(stored) != len(replayed):
        return {
            "length_match": False,
            "stored_count": len(stored),
            "replayed_count": len(replayed),
        }

    if stored and isinstance(stored[0], tuple):
        stored_flat = [component for vector in stored for component in vector]
        replayed_flat = [component for vector in replayed for component in vector]
        stored_peak = max(
            math.sqrt(sum(component * component for component in vector)) for vector in stored
        )
        replayed_peak = max(
            math.sqrt(sum(component * component for component in vector)) for vector in replayed
        )
    else:
        stored_flat = stored
        replayed_flat = replayed
        stored_peak = max(abs(value) for value in stored)
        replayed_peak = max(abs(value) for value in replayed)

    if not all(math.isfinite(value) for value in stored_flat + replayed_flat):
        return {"length_match": True, "finite": False}

    squared_error = sum(
        (stored_value - replayed_value) ** 2
        for stored_value, replayed_value in zip(stored_flat, replayed_flat, strict=True)
    )
    replayed_energy = sum(value * value for value in replayed_flat)
    count = max(1, len(replayed_flat))
    rms_error = math.sqrt(squared_error / count)
    replayed_rms = math.sqrt(replayed_energy / count)
    normalized_rms_error = rms_error / max(replayed_rms, 1e-12)
    peak_relative_error = abs(stored_peak - replayed_peak) / max(replayed_peak, 1e-12)
    return {
        "length_match": True,
        "finite": True,
        "normalized_rms_error": normalized_rms_error,
        "peak_relative_error": peak_relative_error,
        "stored_peak": stored_peak,
        "replayed_peak": replayed_peak,
    }


def compare_result_snapshots(
    stored: dict[str, Any],
    replayed: dict[str, Any],
    analysis_type: str,
) -> dict[str, Any]:
    failures = []
    warnings = []
    comparisons = {}
    if stored["node_numbers"] != replayed["node_numbers"]:
        failures.append("replayed result node ordering does not match the stored result")

    stored_fields = stored["fields"]
    replayed_fields = replayed["fields"]
    required = (
        STATIC_REQUIRED_REPLAY_FIELDS if analysis_type in ("static", "thermal_mechanical") else ()
    )
    for name in required:
        if name not in stored_fields:
            failures.append(f"stored result is missing required field {name}")
        if name not in replayed_fields:
            failures.append(f"replayed result is missing required field {name}")

    for name in sorted(set(stored_fields) | set(replayed_fields)):
        if name not in stored_fields or name not in replayed_fields:
            failures.append(f"field set differs after replay: {name}")
            continue
        comparison = field_error(stored_fields[name], replayed_fields[name])
        comparisons[name] = comparison
        if not comparison.get("length_match") or not comparison.get("finite", True):
            failures.append(f"field {name} is structurally invalid after replay")
            continue
        if comparison["normalized_rms_error"] > REPLAY_HARD_FAIL_REL_TOL:
            failures.append(
                f"field {name} normalized RMS error "
                f"{comparison['normalized_rms_error']:.3%} exceeds "
                f"{REPLAY_HARD_FAIL_REL_TOL:.0%} hard-fail tolerance"
            )
        elif comparison["normalized_rms_error"] > REPLAY_REL_TOL:
            warnings.append(
                f"field {name} normalized RMS error "
                f"{comparison['normalized_rms_error']:.3%} exceeds "
                f"{REPLAY_REL_TOL:.0%} verification tolerance"
            )
        if comparison["peak_relative_error"] > REPLAY_HARD_FAIL_REL_TOL:
            failures.append(
                f"field {name} peak error {comparison['peak_relative_error']:.3%} "
                f"exceeds {REPLAY_HARD_FAIL_REL_TOL:.0%} hard-fail tolerance"
            )
        elif comparison["peak_relative_error"] > REPLAY_REL_TOL:
            warnings.append(
                f"field {name} peak error {comparison['peak_relative_error']:.3%} "
                f"exceeds {REPLAY_REL_TOL:.0%} verification tolerance"
            )

    status = "mismatch" if failures else ("accepted_mismatch" if warnings else "verified")
    return {
        "passed": not failures,
        "status": status,
        "relative_tolerance": REPLAY_REL_TOL,
        "hard_fail_relative_tolerance": REPLAY_HARD_FAIL_REL_TOL,
        "failures": failures,
        "warnings": warnings,
        "field_comparisons": comparisons,
    }
