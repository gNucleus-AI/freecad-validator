"""Validate a solved FEM candidate against an engineer-generated reference.

Inputs (all required):
* ``step_path``            - the source 3D geometry that was to be analysed (STEP).
* ``fcstd_path_reference`` - the solved reference FCStd.
* ``fcstd_path_candidate`` - the solved candidate FCStd to validate.

What it evaluates on the candidate:
* geometry fidelity     - did the candidate analyse the STEP solid (volume match)?
* accuracy vs reference - candidate result quantities vs the reference values;
* mesh budget           - candidate element count vs the reference baseline;
* problem-setup match   - analysis type, material, restraint, and load;
* physical validity, mesh quality, numerical reliability of the candidate.

Returns a ``ScoringReport`` (the standard type): overall 0-100, per-category
sub-scores, pass/fail flags, detected failure modes, numerical comparison table
(candidate vs reference), feedback, confidence and reproducibility status.

A raw FCStd carries no convergence study or written report, so those categories
are weighted out. Accuracy-vs-reference is the largest single weight; a gross miss on
the critical quantity caps the score (a failed reproduction); a geometry mismatch
or a physically impossible result hard-caps the score regardless.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from freecad_validator._freecad_loader import resolve_freecad_command
from freecad_validator.fem.schema import (
    DISP_TOL,
    GROSS_TOL,
    MESH_BUDGET_ZERO_RATIO,
    STRESS_TOL,
    CaseDefinition,
    FailureMode,
    ScoringReport,
    Submission,
)
from freecad_validator.fem.scorer import score_result

HERE = Path(__file__).resolve().parent
FCSTD_ADAPTER = str(HERE / "adapters" / "fcstd.py")
STEP_ADAPTER = str(HERE / "adapters" / "step.py")


class ExtractionError(RuntimeError):
    """An adapter could not produce a valid extraction payload."""


class RuntimeEnvironmentError(ExtractionError):
    """The validator runtime is missing or misconfigured."""


DEFAULT_SUBPROCESS_TIMEOUT_SECONDS = 900.0
DIAGNOSTIC_TAIL_CHARACTERS = 16_000


# Accuracy-vs-reference is the largest single weight; mesh_budget (candidate element
# count vs the reference's) is a deliberate 25% efficiency factor. A raw FCStd carries
# no report/convergence study, so those stay at 0.
PREPROCESSING_VOLUME_REL_TOL = 1e-3
PREPROCESSING_SURFACE_AREA_REL_TOL = 1e-2
BOOLEAN_TOPOLOGY_FIELDS = (
    "num_solids",
    "num_compsolids",
    "shape_types",
    "num_faces",
    "num_edges",
)
BOOLEAN_CONTAINER_FIELDS = ("num_compsolids", "shape_types")
BOOLEAN_REGION_INTEGER_FIELDS = ("num_faces", "num_edges", "num_shells")
BOOLEAN_REGION_FLOAT_FIELDS = ("volume_mm3", "surface_area_mm2")
BOOLEAN_TOPOLOGY_REL_TOL = 1e-5


def _relative_difference(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), 1e-9)


def _preprocessing_gate_report(
    input_geometry: dict[str, Any], candidate_geometry: dict[str, Any]
) -> ScoringReport | None:
    """Return an immediate zero when candidate geometry matches the input STEP."""
    keys = ("volume_mm3", "surface_area_mm2")
    if not all(float(input_geometry.get(key, 0.0) or 0.0) > 0.0 for key in keys):
        return None
    if not all(float(candidate_geometry.get(key, 0.0) or 0.0) > 0.0 for key in keys):
        return None

    volume_diff = _relative_difference(
        float(input_geometry["volume_mm3"]),
        float(candidate_geometry["volume_mm3"]),
    )
    area_diff = _relative_difference(
        float(input_geometry["surface_area_mm2"]),
        float(candidate_geometry["surface_area_mm2"]),
    )
    if (
        volume_diff > PREPROCESSING_VOLUME_REL_TOL
        or area_diff > PREPROCESSING_SURFACE_AREA_REL_TOL
    ):
        return None

    message = (
        "Candidate volume and surface area match the input STEP; "
        "required geometry preprocessing was not performed."
    )
    failure = {
        "code": FailureMode.PREPROCESSING_NOT_PERFORMED,
        "severity": "critical",
        "category": "problem_setup",
        "evidence": message,
    }
    return ScoringReport(
        case_id="step+reference+candidate",
        overall_score=0.0,
        grade="invalid",
        subscores={"problem_setup": 0.0},
        pass_fail_flags={"setup_correct": False},
        failure_modes_detected=[failure],
        gates_triggered=[{"reason": FailureMode.PREPROCESSING_NOT_PERFORMED}],
        engineering_feedback=[f"[CRITICAL/problem_setup] {message}"],
        suggested_fixes=[
            "Apply the required geometry preprocessing before creating and solving the FEM model."
        ],
        confidence=1.0,
        reproducibility_status="not_evaluated",
        evidence=[
            "preprocessing_gate: input STEP and candidate geometry match",
            f"preprocessing_volume_rel_diff: {volume_diff:.6g}",
            f"preprocessing_surface_area_rel_diff: {area_diff:.6g}",
        ],
    )


def _boolean_topology_signature(
    geometry: dict[str, Any], source: str
) -> dict[str, Any]:
    missing = [field for field in BOOLEAN_TOPOLOGY_FIELDS if field not in geometry]
    if "regions" not in geometry:
        missing.append("regions")
    if missing:
        raise ExtractionError(
            f"{source} geometry is missing Boolean topology fields: "
            + ", ".join(missing)
        )
    if not isinstance(geometry["regions"], list):
        raise ExtractionError(f"{source} geometry Boolean field 'regions' is not a list")
    return {
        **{field: geometry[field] for field in BOOLEAN_TOPOLOGY_FIELDS},
        "regions": geometry["regions"],
    }


def _boolean_regions_match(expected: Any, actual: Any) -> bool:
    if not isinstance(expected, dict) or not isinstance(actual, dict):
        return False
    if any(
        expected.get(field) != actual.get(field)
        for field in BOOLEAN_REGION_INTEGER_FIELDS
    ):
        return False
    for field in BOOLEAN_REGION_FLOAT_FIELDS:
        expected_value = expected.get(field)
        actual_value = actual.get(field)
        if not isinstance(expected_value, (int, float)) or not isinstance(
            actual_value, (int, float)
        ):
            return False
        if (
            _relative_difference(float(expected_value), float(actual_value))
            > BOOLEAN_TOPOLOGY_REL_TOL
        ):
            return False
    return True


def _boolean_topology_mismatches(
    expected: dict[str, Any], actual: dict[str, Any]
) -> list[str]:
    mismatches = []
    for field in BOOLEAN_TOPOLOGY_FIELDS:
        if expected.get(field) != actual.get(field):
            mismatches.append(
                f"{field}: expected {expected.get(field)!r}, got {actual.get(field)!r}"
            )

    expected_regions = expected.get("regions")
    actual_regions = actual.get("regions")
    if not isinstance(expected_regions, list) or not isinstance(actual_regions, list):
        mismatches.append("regions: missing or invalid region list")
        return mismatches
    if len(expected_regions) != len(actual_regions):
        mismatches.append(
            f"regions: expected {len(expected_regions)}, got {len(actual_regions)}"
        )
        return mismatches

    unmatched_actual = list(actual_regions)
    for index, expected_region in enumerate(expected_regions):
        matching_index = next(
            (
                actual_index
                for actual_index, actual_region in enumerate(unmatched_actual)
                if _boolean_regions_match(expected_region, actual_region)
            ),
            None,
        )
        if matching_index is None:
            mismatches.append(f"regions[{index}]: no matching candidate region")
        else:
            unmatched_actual.pop(matching_index)
    return mismatches


def _boolean_minimum_reference_mismatches(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> list[str]:
    """Compare only the topology requirements needed to prove Boolean output."""
    mismatches = []
    reference_regions = reference["regions"]
    candidate_regions = candidate["regions"]
    if len(reference_regions) != len(candidate_regions):
        mismatches.append(
            f"regions: expected {len(reference_regions)}, got {len(candidate_regions)}"
        )
    for field in BOOLEAN_CONTAINER_FIELDS:
        if reference[field] != candidate[field]:
            mismatches.append(
                f"{field}: expected {reference[field]!r}, got {candidate[field]!r}"
            )
    return mismatches


def _boolean_failure_report(
    failure_mode: str, message: str, evidence: list[str]
) -> ScoringReport:
    return ScoringReport(
        case_id="step+reference+candidate",
        overall_score=0.0,
        grade="invalid",
        subscores={"problem_setup": 0.0},
        pass_fail_flags={"setup_correct": False},
        failure_modes_detected=[{
            "code": failure_mode,
            "severity": "critical",
            "category": "problem_setup",
            "evidence": message,
        }],
        gates_triggered=[{"reason": failure_mode}],
        engineering_feedback=[f"[CRITICAL/problem_setup] {message}"],
        suggested_fixes=[
            "Apply Boolean Fragments and preserve the required region/container topology."
        ],
        confidence=1.0,
        reproducibility_status="not_evaluated",
        evidence=evidence,
    )


def _boolean_gate_report(
    source_geometry: dict[str, Any],
    reference_geometry: dict[str, Any],
    candidate_geometry: dict[str, Any],
) -> ScoringReport | None:
    source_signature = _boolean_topology_signature(source_geometry, "source STEP")
    reference_signature = _boolean_topology_signature(
        reference_geometry, "reference FCStd"
    )
    if not _boolean_topology_mismatches(source_signature, reference_signature):
        raise ExtractionError(
            "need_boolean is true, but source STEP and reference FCStd Boolean "
            "topologies are indistinguishable"
        )
    try:
        candidate_signature = _boolean_topology_signature(
            candidate_geometry, "candidate FCStd"
        )
    except ExtractionError as exc:
        return _boolean_failure_report(
            FailureMode.GEOMETRY_MISMATCH,
            "Candidate geometry could not satisfy the required Boolean topology.",
            [f"boolean_candidate_geometry_error: {exc}"],
        )

    if not _boolean_topology_mismatches(source_signature, candidate_signature):
        return _boolean_failure_report(
            FailureMode.BOOLEAN_NOT_PERFORMED,
            "Candidate topology still matches the raw STEP; required Boolean "
            "Fragments were not performed.",
            ["boolean_gate: candidate topology matches source STEP"],
        )

    reference_mismatches = _boolean_minimum_reference_mismatches(
        reference_signature, candidate_signature
    )
    if reference_mismatches:
        evidence = ["boolean_gate: candidate topology differs from reference FCStd"]
        evidence.extend(reference_mismatches[:20])
        return _boolean_failure_report(
            FailureMode.GEOMETRY_MISMATCH,
            "Candidate Boolean region/container topology differs from the reference.",
            evidence,
        )
    return None


STEP_WEIGHTS = {
    "accuracy_vs_reference": 0.35,
    "mesh_budget": 0.25,            # candidate elements vs reference baseline (0 at >=130%)
    "problem_setup": 0.15,          # incl. geometry fidelity + analysis/material/BC/load match
    "physical_validity": 0.15,
    "numerical_reliability": 0.05,
    "mesh_quality": 0.05,
    "engineering_reporting": 0.0,
}


def _reference_quantities(reference_sub: dict[str, Any], disp_tol: float, stress_tol: float) -> dict[str, Any]:
    res = reference_sub.get("results", {})
    q: dict[str, Any] = {}
    if "max_displacement_mm" in res:
        q["max_displacement_mm"] = {"value": res["max_displacement_mm"], "unit": "mm",
                                    "tol_rel": disp_tol, "critical": True}
    if "max_von_mises_MPa" in res:
        q["max_von_mises_MPa"] = {"value": res["max_von_mises_MPa"], "unit": "MPa",
                                  "tol_rel": stress_tol, "critical": False}
    if "max_shear_MPa" in res:
        q["max_shear_MPa"] = {"value": res["max_shear_MPa"], "unit": "MPa",
                              "tol_rel": stress_tol, "critical": False}
    freqs = res.get("natural_frequencies_Hz")
    if freqs:
        q["first_natural_frequency_Hz"] = {"value": freqs[0], "unit": "Hz",
                                           "tol_rel": disp_tol, "critical": True}
    return q


def build_case(target_geom: dict[str, Any], reference_sub: dict[str, Any],
               disp_tol: float = DISP_TOL, stress_tol: float = STRESS_TOL,
               gross_tol: float = GROSS_TOL,
               mesh_budget_zero_ratio: float = MESH_BUDGET_ZERO_RATIO,
               case_id: str = "step+reference+candidate") -> CaseDefinition:
    """Build from target geometry and an engineer-generated reference.

    The reference element count is the mesh-budget baseline.
    """
    geometry = {k: target_geom[k] for k in ("volume_mm3", "bbox_mm", "characteristic_length_mm")
                if k in target_geom}
    mesh_exp: dict[str, Any] = {"expect_convergence": False}
    baseline = (reference_sub.get("mesh") or {}).get("num_elements")
    if baseline:
        mesh_exp["baseline_num_elements"] = baseline
        mesh_exp["budget_zero_ratio"] = mesh_budget_zero_ratio
    return CaseDefinition(
        case_id=case_id, title="geometry + reference + candidate evaluation",
        category="fem", subtype="",
        analysis_type=reference_sub.get("analysis_type", "static"),
        units_expected={"length": "mm", "force": "N", "stress": "MPa"},
        geometry=geometry, material=dict(reference_sub.get("material", {})),
        expected_bcs=reference_sub.get("boundary_conditions") or [{"type": "fixed"}],
        expected_loads=reference_sub.get("loads") or [],
        reference={"method": "numerical", "source": "engineer-generated solved FCStd",
                   "quantities": _reference_quantities(reference_sub, disp_tol, stress_tol),
                   "gross_tol": gross_tol},
        mesh_expectations=mesh_exp,
        weights_override=dict(STEP_WEIGHTS), rubric={"requires_safety_factor": False})


def score_trusted_payloads(target_geom: dict[str, Any], reference_sub: dict[str, Any],
                          candidate_sub: dict[str, Any], disp_tol: float = DISP_TOL,
                          stress_tol: float = STRESS_TOL, gross_tol: float = GROSS_TOL,
                          mesh_budget_zero_ratio: float = MESH_BUDGET_ZERO_RATIO,
                          geometry_source: str = "STEP") -> ScoringReport:
    """Score validator-generated extraction payloads without FreeCAD.

    This is a trusted low-level API. All three dictionaries must come directly
    from validator-controlled adapters or equivalent protected code. Never pass
    candidate-controlled JSON here; replay-verification fields are trusted.
    """
    case = build_case(target_geom, reference_sub, disp_tol, stress_tol, gross_tol,
                      mesh_budget_zero_ratio)
    sub = Submission.from_dict({**candidate_sub, "case_id": case.case_id})
    report = score_result(case, sub)

    report.evidence.insert(
        0,
        f"evaluation_mode: reference-based ({geometry_source} target + engineer reference + candidate)",
    )
    extraction_failure = (candidate_sub.get("meta") or {}).get("candidate_extraction_failure")
    if extraction_failure:
        report.evidence.insert(1, f"candidate_extraction_failure: {extraction_failure}")
    sv, fv = target_geom.get("volume_mm3"), sub.geometry.get("volume_mm3")
    if sv and fv:
        report.evidence.insert(1, f"geometry_fidelity: target_volume={sv:.0f} mm^3, "
                                  f"candidate_volume={fv:.0f} mm^3")
    base_n = (reference_sub.get("mesh") or {}).get("num_elements")
    cand_n = (candidate_sub.get("mesh") or {}).get("num_elements")
    if base_n and cand_n:
        report.evidence.insert(2, f"mesh_budget: candidate={cand_n} elements vs reference baseline="
                                  f"{base_n} ({cand_n/base_n*100:.0f}%; 0 at "
                                  f"{mesh_budget_zero_ratio*100:.0f}%)")
    return report


def _terminate_process_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _diagnostic_tail(log_path: str) -> str:
    try:
        with open(log_path, "rb") as log_file:
            log_file.seek(0, os.SEEK_END)
            size = log_file.tell()
            log_file.seek(max(0, size - DIAGNOSTIC_TAIL_CHARACTERS))
            return log_file.read().decode("utf-8", errors="replace").strip()
    except OSError:
        return ""


def _run_adapter(
    freecad_cmd: str,
    adapter: str,
    out_path: str,
    adapter_args: list[str],
    timeout_seconds: float,
) -> dict[str, Any]:
    if not os.path.exists(freecad_cmd):
        raise ExtractionError(
            f"FreeCAD not found at {freecad_cmd!r}; set FREECAD_CMD to your FreeCAD 1.1 freecadcmd.")
    if timeout_seconds <= 0.0:
        raise ValueError("timeout_seconds must be positive")
    log_path = f"{out_path}.log"
    try:
        if os.path.exists(out_path):
            os.remove(out_path)
        command = [freecad_cmd, adapter, *adapter_args, out_path]
        popen_options: dict[str, Any] = {}
        if os.name == "nt":
            popen_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_options["start_new_session"] = True
        with open(log_path, "w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                command,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                **popen_options,
            )
            try:
                return_code = process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired as exc:
                _terminate_process_tree(process)
                raise ExtractionError(
                    f"adapter timed out after {timeout_seconds:g} seconds"
                ) from exc
        if return_code != 0 or not os.path.exists(out_path):
            diagnostic = _diagnostic_tail(log_path)
            detail = diagnostic or "adapter produced no diagnostic output"
            raise ExtractionError(
                f"adapter failed with exit {return_code}: {detail}")
        with open(out_path, encoding="utf-8") as fh:
            payload = json.load(fh)
    except ExtractionError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExtractionError(f"adapter extraction failed: {exc}") from exc
    if not isinstance(payload, dict):
        raise ExtractionError("adapter JSON is not an object")
    return payload


def _extract(
    freecad_cmd: str,
    adapter: str,
    in_path: str,
    out_path: str,
    extra_args: list[str] | None = None,
    timeout_seconds: float = DEFAULT_SUBPROCESS_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    adapter_args = [os.path.abspath(in_path), *(extra_args or [])]
    try:
        return _run_adapter(
            freecad_cmd,
            adapter,
            out_path,
            adapter_args,
            timeout_seconds,
        )
    except ExtractionError as exc:
        raise ExtractionError(f"extraction failed for {in_path}: {exc}") from exc


def _runtime_preflight(
    freecad_cmd: str,
    extract_dir: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    out_path = os.path.join(extract_dir, "fem_runtime.json")
    try:
        payload = _run_adapter(
            freecad_cmd,
            FCSTD_ADAPTER,
            out_path,
            ["runtime-info"],
            min(timeout_seconds, 30.0),
        )
    except (ExtractionError, ValueError) as exc:
        raise RuntimeEnvironmentError(f"FEM runtime preflight failed: {exc}") from exc
    runtime = payload.get("runtime")
    if not isinstance(runtime, dict) or not runtime.get("calculix"):
        raise RuntimeEnvironmentError(
            "FEM runtime preflight produced no CalculiX version"
        )
    return runtime


def _failed_candidate(reason: str) -> dict[str, Any]:
    return {
        "solver": {"converged": False},
        "results": {},
        "artifacts": {},
        "meta": {"candidate_extraction_failure": reason},
    }


def _score_step_fcstd(
    step_path: str,
    fcstd_path_reference: str,
    fcstd_path_candidate: str,
    freecad_cmd: str,
    extract_dir: str,
    disp_tol: float,
    stress_tol: float,
    gross_tol: float,
    mesh_budget_zero_ratio: float,
    require_preprocessing: bool,
    require_boolean: bool,
    timeout_seconds: float,
    runtime_provenance: dict[str, Any],
) -> ScoringReport:
    fc = freecad_cmd
    os.makedirs(extract_dir, exist_ok=True)
    tag = os.path.splitext(os.path.basename(fcstd_path_candidate))[0]
    step_geom = _extract(
        fc, STEP_ADAPTER, step_path, os.path.join(extract_dir, f"{tag}_step.json"),
        timeout_seconds=timeout_seconds,
    )
    reference_geometry_payload = None
    if require_boolean:
        reference_geometry_payload = _extract(
            fc, FCSTD_ADAPTER, fcstd_path_reference,
            os.path.join(extract_dir, f"{tag}_boolean_reference.json"),
            extra_args=["geometry-only"],
            timeout_seconds=timeout_seconds,
        )
    candidate_geometry_payload = None
    if require_preprocessing or require_boolean:
        candidate_geometry_payload = _extract(
            fc, FCSTD_ADAPTER, fcstd_path_candidate,
            os.path.join(extract_dir, f"{tag}_required_geometry_candidate.json"),
            extra_args=["geometry-only"],
            timeout_seconds=timeout_seconds,
        )
    if require_preprocessing:
        if candidate_geometry_payload is None:
            raise ExtractionError("candidate geometry extraction was not performed")
        gate_report = _preprocessing_gate_report(
            step_geom,
            candidate_geometry_payload.get("geometry") or {},
        )
        if gate_report is not None:
            gate_report.runtime_provenance = dict(runtime_provenance)
            return gate_report
    if require_boolean:
        if reference_geometry_payload is None or candidate_geometry_payload is None:
            raise ExtractionError("Boolean geometry extraction was not performed")
        gate_report = _boolean_gate_report(
            step_geom,
            reference_geometry_payload.get("geometry") or {},
            candidate_geometry_payload.get("geometry") or {},
        )
        if gate_report is not None:
            gate_report.runtime_provenance = dict(runtime_provenance)
            return gate_report

    reference = _extract(
        fc,
        FCSTD_ADAPTER,
        fcstd_path_reference,
        os.path.join(extract_dir, f"{tag}_reference.json"),
        timeout_seconds=timeout_seconds,
    )
    if reference.get("no_result"):
        reason = reference.get("extraction_error") or "no FEM result object found"
        raise ExtractionError(f"reference FCStd contains no loaded FEM result: {reason}")
    try:
        candidate = _extract(fc, FCSTD_ADAPTER, fcstd_path_candidate,
                             os.path.join(extract_dir, f"{tag}_candidate.json"),
                             extra_args=["verify-solve"],
                             timeout_seconds=timeout_seconds)
    except ExtractionError as exc:
        candidate = _failed_candidate(str(exc))
    if candidate.get("no_result"):
        reason = candidate.get("extraction_error") or "no FEM result object found"
        candidate = _failed_candidate(reason)
    target_geometry = step_geom
    geometry_source = "STEP"
    if require_preprocessing:
        target_geometry = reference.get("geometry") or {}
        if not target_geometry:
            raise ExtractionError(
                "reference FCStd contains no geometry for preprocessing validation"
            )
        geometry_source = "reference FCStd geometry"
    report = score_trusted_payloads(
        target_geometry,
        reference,
        candidate,
        disp_tol,
        stress_tol,
        gross_tol,
        mesh_budget_zero_ratio,
        geometry_source,
    )
    report.runtime_provenance = dict(runtime_provenance)
    return report


def score_step_fcstd(
    step_path: str,
    fcstd_path_reference: str,
    fcstd_path_candidate: str,
    freecad_cmd: str | None = None,
    extract_dir: str | None = None,
    disp_tol: float = DISP_TOL,
    stress_tol: float = STRESS_TOL,
    gross_tol: float = GROSS_TOL,
    mesh_budget_zero_ratio: float = MESH_BUDGET_ZERO_RATIO,
    require_preprocessing: bool = False,
    require_boolean: bool = False,
    timeout_seconds: float = DEFAULT_SUBPROCESS_TIMEOUT_SECONDS,
) -> ScoringReport:
    """Extract and score one candidate against an engineer reference.

    FreeCAD and CalculiX are required and execute in subprocesses. Run this API
    in an isolated container when the candidate FCStd is untrusted. A missing or
    broken runtime raises :class:`RuntimeEnvironmentError` before scoring.
    """
    if timeout_seconds <= 0.0:
        raise ValueError("timeout_seconds must be positive")
    fc = resolve_freecad_command(freecad_cmd)
    if extract_dir is not None:
        os.makedirs(extract_dir, exist_ok=True)
        runtime_provenance = _runtime_preflight(fc, extract_dir, timeout_seconds)
        return _score_step_fcstd(
            step_path,
            fcstd_path_reference,
            fcstd_path_candidate,
            fc,
            extract_dir,
            disp_tol,
            stress_tol,
            gross_tol,
            mesh_budget_zero_ratio,
            require_preprocessing,
            require_boolean,
            timeout_seconds,
            runtime_provenance,
        )
    with tempfile.TemporaryDirectory(prefix="freecad-validator-fem-") as temp_dir:
        runtime_provenance = _runtime_preflight(fc, temp_dir, timeout_seconds)
        return _score_step_fcstd(
            step_path,
            fcstd_path_reference,
            fcstd_path_candidate,
            fc,
            temp_dir,
            disp_tol,
            stress_tol,
            gross_tol,
            mesh_budget_zero_ratio,
            require_preprocessing,
            require_boolean,
            timeout_seconds,
            runtime_provenance,
        )
