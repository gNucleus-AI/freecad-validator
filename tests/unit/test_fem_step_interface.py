"""Tests for the STEP + reference + candidate FEM validator."""

import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import pytest

from freecad_validator.fem import (
    FailureMode,
    RuntimeEnvironmentError,
    score_step_fcstd,
    score_trusted_payloads,
)
from freecad_validator.fem.step_interface import (
    FCSTD_ADAPTER,
    ExtractionError,
    _extract,
    _preprocessing_gate_report,
    _run_adapter,
)
from freecad_validator.fem.topology_compare import topology_difference, topology_mismatches

STEP = {"volume_mm3": 1.0e6, "characteristic_length_mm": 387.0, "bbox_mm": [120, 360, 60]}


@pytest.fixture(autouse=True)
def resolve_fake_freecad_command(monkeypatch):
    """Keep extraction tests independent of a host FreeCAD installation."""
    monkeypatch.setattr(
        "freecad_validator.fem.step_interface.resolve_freecad_command",
        lambda explicit=None: explicit or "/fake/freecadcmd",
    )
    monkeypatch.setattr(
        "freecad_validator.fem.step_interface._runtime_preflight",
        lambda *_args, **_kwargs: {
            "freecad": "1.1.0",
            "occt": "7.8.1",
            "python": "3.11.9",
            "calculix": "2.23",
        },
    )


LABEL = dict(
    analysis_type="static",
    material={"E_MPa": 200000.0, "nu": 0.3},
    boundary_conditions=[{"type": "fixed", "location": "Body/Face10"}],
    loads=[{"type": "force", "magnitude_N": 500.0, "location": "Body/Face14"}],
    results={"max_displacement_mm": 3.0e-4, "max_von_mises_MPa": 0.35, "max_shear_MPa": 0.19},
    geometry={"volume_mm3": 1.0e6, "characteristic_length_mm": 387.0},
    mesh={"num_nodes": 20000, "num_elements": 12000},  # mesh-budget baseline
)


def _cand(**over):
    base = dict(
        analysis_type="static",
        units={"length": "mm", "force": "N", "stress": "MPa"},
        material={"E_MPa": 200000.0, "nu": 0.3},
        boundary_conditions=[{"type": "fixed"}],
        loads=[{"type": "force", "magnitude_N": 500.0}],
        mesh={"num_nodes": 20000, "num_elements": 12000},
        solver={"converged": True, "applied_load_N": 500.0, "reaction_force_N": [0, 0, 500.1]},
        results={"max_displacement_mm": 3.05e-4, "max_von_mises_MPa": 0.37, "max_shear_MPa": 0.196},
        geometry={"volume_mm3": 1.005e6, "characteristic_length_mm": 387.0},
        artifacts={"result_file": "c.FCStd", "input_deck": "c.FCStd"},
    )
    base.update(over)
    return base


def _topology_geometry(
    region_volumes,
    *,
    num_compsolids=0,
    shape_types=None,
):
    regions = [
        {
            "volume_mm3": volume,
            "surface_area_mm2": 1000.0 / len(region_volumes),
            "num_faces": 6 + index,
            "num_edges": 12 + index,
            "num_shells": 1,
        }
        for index, volume in enumerate(region_volumes)
    ]
    return {
        "volume_mm3": sum(region_volumes),
        "surface_area_mm2": 1000.0,
        "characteristic_length_mm": 387.0,
        "bbox_mm": [120.0, 360.0, 60.0],
        "num_solids": len(regions),
        "num_compsolids": num_compsolids,
        "shape_types": shape_types or ["Compound"],
        "num_faces": sum(region["num_faces"] for region in regions),
        "num_edges": sum(region["num_edges"] for region in regions),
        "regions": regions,
    }


def test_candidate_matches_label_scores_high():
    rep = score_trusted_payloads(STEP, LABEL, _cand())
    assert rep.overall_score >= 90
    assert rep.grade == "excellent"
    crit = next(c for c in rep.numerical_comparisons if c["critical"])
    assert crit["within_tol"]


def test_trusted_replay_without_reaction_evidence_has_full_numerical_score():
    candidate = _cand(
        solver={"converged": True, "replay_verified": False, "replay_accepted": True},
    )
    rep = score_trusted_payloads(STEP, LABEL, candidate)
    assert rep.subscores["numerical_reliability"] == 100.0
    assert not any(
        finding["code"] == "NO_REACTION_CHECK"
        for finding in rep.subscores_details["numerical_reliability"]["findings"]
    )


def test_opposing_load_magnitudes_are_not_summed_for_reaction_balance():
    candidate = _cand(
        loads=[
            {"type": "force", "magnitude_N": 500.0, "direction": [1, 0, 0]},
            {"type": "force", "magnitude_N": 500.0, "direction": [-1, 0, 0]},
        ],
        solver={"converged": True, "reaction_force_N": [0, 0, 0]},
    )
    rep = score_trusted_payloads(STEP, LABEL, candidate)
    findings = rep.subscores_details["numerical_reliability"]["findings"]
    assert not any(finding["code"] == FailureMode.REACTION_IMBALANCE for finding in findings)
    assert any(finding["code"] == "NO_REACTION_CHECK" for finding in findings)


def test_accuracy_table_present_with_reference():
    rep = score_trusted_payloads(STEP, LABEL, _cand())
    assert rep.numerical_comparisons  # Candidate-vs-reference comparisons exist.
    assert any("reference-based" in e for e in rep.evidence)


def test_gross_miss_vs_label_gates_to_zero():
    rep = score_trusted_payloads(
        STEP,
        LABEL,
        _cand(
            results={
                "max_displacement_mm": 6.0e-4,
                "max_von_mises_MPa": 0.70,
                "max_shear_MPa": 0.38,
            }
        ),
    )
    assert rep.overall_score == 0
    assert rep.grade == "invalid"
    assert any(f["code"] == FailureMode.ACCURACY_GROSS_ERROR for f in rep.failure_modes_detected)
    assert rep.gates_triggered


def test_omitting_critical_quantity_gates_to_zero():
    """Omitting the critical quantity must NOT out-score reporting it wrong.

    Regression for the omit-displacement reward hack: a fabricated result that
    drops max_displacement_mm (stress only) once scored 64.5 because the gross-miss
    gate skipped a None rel_error. A missing critical quantity now gates to 0, the
    same as a gross error - so omitting it can never beat getting it wrong.
    """
    rep = score_trusted_payloads(
        STEP, LABEL, _cand(results={"max_von_mises_MPa": 0.35, "max_shear_MPa": 0.19})
    )  # no displacement
    assert rep.overall_score == 0
    assert rep.grade == "invalid"
    assert any(f["code"] == FailureMode.ACCURACY_GROSS_ERROR for f in rep.failure_modes_detected)
    assert rep.gates_triggered


def test_gross_wrong_load_magnitude_gates_to_zero():
    """A grossly wrong applied load is a different problem and must gate, even if
    the reported results happen to sit near the label (compensating-error blind spot)."""
    rep = score_trusted_payloads(
        STEP, LABEL, _cand(loads=[{"type": "force", "magnitude_N": 5000.0}])
    )  # 10x the label's 500 N
    assert rep.overall_score == 0
    assert rep.grade == "invalid"
    assert any(f["code"] == FailureMode.WRONG_LOAD for f in rep.failure_modes_detected)
    assert rep.gates_triggered


def test_mild_load_mismatch_penalized_not_gated():
    """An 8% load error dents problem_setup but does not gate."""
    rep = score_trusted_payloads(
        STEP, LABEL, _cand(loads=[{"type": "force", "magnitude_N": 540.0}])
    )  # +8% -> major, no gate
    assert not rep.gates_triggered
    assert rep.subscores["problem_setup"] < 100
    assert any(f["code"] == FailureMode.WRONG_LOAD for f in rep.failure_modes_detected)


def test_reversed_force_direction_gates_to_zero():
    """Same magnitude but the force is reversed (180 deg) -> identical result magnitudes,
    so accuracy is blind and only the load check catches it -> gate."""
    rep = score_trusted_payloads(
        STEP,
        {**LABEL, "loads": [{"type": "force", "magnitude_N": 500.0, "direction": [0, 0, 1]}]},
        _cand(loads=[{"type": "force", "magnitude_N": 500.0, "direction": [0, 0, -1]}]),
    )
    assert rep.overall_score == 0
    assert any(f["code"] == FailureMode.WRONG_LOAD for f in rep.failure_modes_detected)
    assert rep.gates_triggered


def test_misaligned_force_direction_gates_to_zero():
    """Past the tight alignment bound (>= 15 deg) the load points the wrong way -> gate."""
    rep = score_trusted_payloads(
        STEP,
        {**LABEL, "loads": [{"type": "force", "magnitude_N": 500.0, "direction": [0, 0, 1]}]},
        _cand(loads=[{"type": "force", "magnitude_N": 500.0, "direction": [1, 0, 0]}]),
    )  # 90 deg
    assert rep.overall_score == 0
    assert rep.gates_triggered
    assert any(f["code"] == FailureMode.WRONG_LOAD for f in rep.failure_modes_detected)


def test_small_direction_offset_penalized_not_gated():
    """A small (6-15 deg) direction offset is a wrong load -> penalty, but not yet a gate."""
    rep = score_trusted_payloads(
        STEP,
        {**LABEL, "loads": [{"type": "force", "magnitude_N": 500.0, "direction": [0, 0, 1]}]},
        _cand(loads=[{"type": "force", "magnitude_N": 500.0, "direction": [0.1736, 0, 0.9848]}]),
    )  # ~10 deg off
    assert not rep.gates_triggered
    assert 0 < rep.subscores["problem_setup"] < 100
    assert any(f["code"] == FailureMode.WRONG_LOAD for f in rep.failure_modes_detected)


def test_matching_load_magnitude_and_direction_scores_high():
    """Right magnitude and direction -> no load finding, still excellent."""
    rep = score_trusted_payloads(
        STEP,
        {**LABEL, "loads": [{"type": "force", "magnitude_N": 500.0, "direction": [0, 0, 1]}]},
        _cand(loads=[{"type": "force", "magnitude_N": 500.0, "direction": [0, 0, 1]}]),
    )
    assert rep.overall_score >= 90
    assert not any(f["code"] == FailureMode.WRONG_LOAD for f in rep.failure_modes_detected)


def _fl(mag, centroid, direction=(0.0, 0.0, 1.0)):
    """A force load at a resolved face (centroid + direction), as the adapter emits."""
    return {
        "type": "force",
        "magnitude_N": mag,
        "centroid": list(centroid),
        "direction": list(direction),
    }


def test_per_face_matched_loads_score_high():
    """Same loads at the same faces -> matched per face -> no load finding."""
    loads = [_fl(1000, (10, 0, 0)), _fl(500, (-10, 0, 0))]
    rep = score_trusted_payloads(
        STEP, {**LABEL, "loads": loads}, _cand(loads=[_fl(1000, (10, 0, 0)), _fl(500, (-10, 0, 0))])
    )
    assert rep.overall_score >= 90
    assert not any(f["code"] == FailureMode.WRONG_LOAD for f in rep.failure_modes_detected)


def test_compensating_swap_caught_though_totals_match():
    """Loads swapped between two faces: the net-aggregate total is identical (1500 N), so
    the old aggregate missed it - per-face matching catches the 1000<->500 swap and gates."""
    label_loads = [_fl(1000, (10, 0, 0)), _fl(500, (-10, 0, 0))]
    cand_loads = [_fl(500, (10, 0, 0)), _fl(1000, (-10, 0, 0))]
    rep = score_trusted_payloads(STEP, {**LABEL, "loads": label_loads}, _cand(loads=cand_loads))
    assert rep.overall_score == 0
    assert any(f["code"] == FailureMode.WRONG_LOAD for f in rep.failure_modes_detected)
    assert rep.gates_triggered


def test_load_on_wrong_face_flagged_missing_and_spurious():
    """A load on a different face than the label's -> unmatched: a missing load at the
    label's face and a spurious one at the candidate's (neither gates on its own)."""
    rep = score_trusted_payloads(
        STEP, {**LABEL, "loads": [_fl(500, (0, 0, 0))]}, _cand(loads=[_fl(500, (300, 0, 0))])
    )  # > match_tol apart
    assert any(f["code"] == FailureMode.WRONG_LOAD for f in rep.failure_modes_detected)
    assert any(
        "applies none there" in fb or "not present in the label" in fb
        for fb in rep.engineering_feedback
    )


def test_geometry_mismatch_vs_step_gates_to_zero():
    rep = score_trusted_payloads(
        STEP, LABEL, _cand(geometry={"volume_mm3": 5.0e6, "characteristic_length_mm": 660.0})
    )
    assert rep.overall_score == 0
    assert any(f["code"] == FailureMode.GEOMETRY_MISMATCH for f in rep.failure_modes_detected)


def test_wrong_material_vs_label_flagged():
    rep = score_trusted_payloads(STEP, LABEL, _cand(material={"E_MPa": 70000.0, "nu": 0.33}))
    assert any(f["code"] == FailureMode.WRONG_MATERIAL for f in rep.failure_modes_detected)


def test_missing_restraint_caps_score():
    rep = score_trusted_payloads(STEP, LABEL, _cand(boundary_conditions=[]))
    assert any(
        f["code"] == FailureMode.MISSING_BOUNDARY_CONDITION for f in rep.failure_modes_detected
    )


def test_physically_impossible_gates_to_zero():
    rep = score_trusted_payloads(
        STEP,
        LABEL,
        _cand(
            results={"max_displacement_mm": 800.0, "max_von_mises_MPa": 0.35, "max_shear_MPa": 0.19}
        ),
    )
    assert rep.overall_score == 0
    assert any(f["code"] == FailureMode.PHYSICALLY_IMPOSSIBLE for f in rep.failure_modes_detected)


def test_candidate_extraction_failure_gates_to_zero():
    candidate = {
        "solver": {"converged": False},
        "results": {},
        "artifacts": {},
        "meta": {"candidate_extraction_failure": "no FEM result object found"},
    }
    rep = score_trusted_payloads(STEP, LABEL, candidate)
    assert rep.overall_score == 0
    assert rep.grade == "invalid"
    assert any(f["code"] == FailureMode.MISSING_RESULTS for f in rep.failure_modes_detected)
    assert any("candidate_extraction_failure" in evidence for evidence in rep.evidence)


def test_failed_solver_replay_gates_to_zero_even_with_claimed_solver_evidence():
    candidate = _cand(
        meta={
            "solver_replay": {
                "passed": False,
                "status": "mismatch",
                "failures": ["DisplacementLengths peak error 90% exceeds 2%"],
            },
        }
    )
    rep = score_trusted_payloads(STEP, LABEL, candidate)

    assert rep.overall_score == 0
    assert rep.grade == "invalid"
    assert any(
        failure["code"] == FailureMode.UNVERIFIED_SOLVER_OUTPUT
        for failure in rep.failure_modes_detected
    )
    assert rep.pass_fail_flags["not_hallucinated"] is False


def test_verified_replay_is_reproducibility_evidence_without_fake_input_deck():
    candidate = _cand(
        solver={"converged": True, "replay_verified": True},
        artifacts={"result_file": "candidate.FCStd"},
        meta={"solver_replay": {"passed": True, "status": "verified"}},
    )
    rep = score_trusted_payloads(STEP, LABEL, candidate)

    assert rep.reproducibility_status == "reproducible"
    assert not any(
        failure["code"] == FailureMode.NON_REPRODUCIBLE for failure in rep.failure_modes_detected
    )


def test_accepted_replay_is_reproducibility_evidence_without_fake_input_deck():
    candidate = _cand(
        solver={"converged": True, "replay_verified": False, "replay_accepted": True},
        artifacts={"result_file": "candidate.FCStd"},
        meta={"solver_replay": {"passed": True, "status": "accepted_mismatch"}},
    )
    rep = score_trusted_payloads(STEP, LABEL, candidate)

    assert rep.reproducibility_status == "reproducible"
    assert not any(
        failure["code"] == FailureMode.NON_REPRODUCIBLE for failure in rep.failure_modes_detected
    )


def test_candidate_fcstd_extraction_requires_trusted_solver_replay():
    with TemporaryDirectory() as extract_dir:
        with patch(
            "freecad_validator.fem.step_interface._extract", side_effect=[STEP, LABEL, _cand()]
        ) as extract:
            score_step_fcstd(
                "source.step",
                "reference.FCStd",
                "candidate.FCStd",
                freecad_cmd="/fake/freecadcmd",
                extract_dir=extract_dir,
            )

    assert extract.call_args_list[0].kwargs.get("extra_args") is None
    assert extract.call_args_list[1].kwargs.get("extra_args") is None
    assert extract.call_args_list[2].kwargs["extra_args"] == ["verify-solve"]


def test_required_boolean_raw_step_topology_gates_before_full_scoring():
    source_geometry = _topology_geometry([1.0e6])
    reference_geometry = _topology_geometry(
        [0.4e6, 0.6e6],
        num_compsolids=1,
        shape_types=["CompSolid"],
    )
    with TemporaryDirectory() as extract_dir:
        with patch(
            "freecad_validator.fem.step_interface._extract",
            side_effect=[
                source_geometry,
                {"geometry": reference_geometry},
                {"geometry": source_geometry},
            ],
        ) as extract:
            rep = score_step_fcstd(
                "source.step",
                "reference.FCStd",
                "candidate.FCStd",
                freecad_cmd="/fake/freecadcmd",
                extract_dir=extract_dir,
                require_boolean=True,
            )

    assert rep.overall_score == 0
    assert rep.gates_triggered == [{"reason": FailureMode.BOOLEAN_NOT_PERFORMED}]
    assert len(extract.call_args_list) == 3
    assert extract.call_args_list[1].kwargs["extra_args"] == ["geometry-only"]
    assert extract.call_args_list[2].kwargs["extra_args"] == ["geometry-only"]


def test_required_boolean_minimum_topology_allows_per_region_differences():
    source_geometry = _topology_geometry([1.0e6])
    reference_geometry = _topology_geometry(
        [0.4e6, 0.6e6],
        num_compsolids=1,
        shape_types=["CompSolid"],
    )
    candidate_gate_geometry = _topology_geometry(
        [0.25e6, 0.75e6],
        num_compsolids=1,
        shape_types=["CompSolid"],
    )
    candidate_gate_geometry["regions"][0].update(
        {
            "surface_area_mm2": 125.0,
            "num_faces": 50,
            "num_edges": 75,
        }
    )
    candidate_gate_geometry["regions"][1].update(
        {
            "surface_area_mm2": 875.0,
            "num_faces": 60,
            "num_edges": 90,
        }
    )
    candidate_gate_geometry["num_faces"] = 110
    candidate_gate_geometry["num_edges"] = 165
    label = {**LABEL, "geometry": reference_geometry}
    candidate = _cand(geometry=reference_geometry)
    with TemporaryDirectory() as extract_dir:
        with patch(
            "freecad_validator.fem.step_interface._extract",
            side_effect=[
                source_geometry,
                {"geometry": reference_geometry},
                {"geometry": candidate_gate_geometry},
                label,
                candidate,
            ],
        ) as extract:
            rep = score_step_fcstd(
                "source.step",
                "reference.FCStd",
                "candidate.FCStd",
                freecad_cmd="/fake/freecadcmd",
                extract_dir=extract_dir,
                require_boolean=True,
            )

    assert source_geometry["volume_mm3"] == reference_geometry["volume_mm3"]
    assert source_geometry["surface_area_mm2"] == reference_geometry["surface_area_mm2"]
    assert rep.overall_score > 0
    assert rep.gates_triggered == []
    assert len(extract.call_args_list) == 5
    assert extract.call_args_list[-1].kwargs["extra_args"] == ["verify-solve"]


def test_required_boolean_wrong_topology_gates_as_geometry_mismatch():
    source_geometry = _topology_geometry([1.0e6])
    reference_geometry = _topology_geometry(
        [0.4e6, 0.6e6],
        num_compsolids=1,
        shape_types=["CompSolid"],
    )
    wrong_geometry = _topology_geometry([0.2e6, 0.3e6, 0.5e6])
    with TemporaryDirectory() as extract_dir:
        with patch(
            "freecad_validator.fem.step_interface._extract",
            side_effect=[
                source_geometry,
                {"geometry": reference_geometry},
                {"geometry": wrong_geometry},
            ],
        ) as extract:
            rep = score_step_fcstd(
                "source.step",
                "reference.FCStd",
                "candidate.FCStd",
                freecad_cmd="/fake/freecadcmd",
                extract_dir=extract_dir,
                require_boolean=True,
            )

    assert rep.overall_score == 0
    assert rep.gates_triggered == [{"reason": FailureMode.GEOMETRY_MISMATCH}]
    assert len(extract.call_args_list) == 3


def test_required_boolean_wrong_container_gates_as_geometry_mismatch():
    source_geometry = _topology_geometry([0.4e6, 0.6e6])
    reference_geometry = _topology_geometry(
        [0.4e6, 0.6e6],
        num_compsolids=1,
        shape_types=["CompSolid"],
    )
    wrong_container_geometry = _topology_geometry([0.3e6, 0.7e6])
    with TemporaryDirectory() as extract_dir:
        with patch(
            "freecad_validator.fem.step_interface._extract",
            side_effect=[
                source_geometry,
                {"geometry": reference_geometry},
                {"geometry": wrong_container_geometry},
            ],
        ) as extract:
            rep = score_step_fcstd(
                "source.step",
                "reference.FCStd",
                "candidate.FCStd",
                freecad_cmd="/fake/freecadcmd",
                extract_dir=extract_dir,
                require_boolean=True,
            )

    assert rep.overall_score == 0
    assert rep.gates_triggered == [{"reason": FailureMode.GEOMETRY_MISMATCH}]
    assert "num_compsolids" in rep.evidence[1]
    assert "shape_types" in rep.evidence[2]
    assert len(extract.call_args_list) == 3


def test_required_boolean_rejects_indistinguishable_reference_topology():
    geometry = _topology_geometry([1.0e6])
    with TemporaryDirectory() as extract_dir:
        with patch(
            "freecad_validator.fem.step_interface._extract",
            side_effect=[
                geometry,
                {"geometry": geometry},
                {"geometry": geometry},
            ],
        ):
            with pytest.raises(ExtractionError, match="indistinguishable"):
                score_step_fcstd(
                    "source.step",
                    "reference.FCStd",
                    "candidate.FCStd",
                    freecad_cmd="/fake/freecadcmd",
                    extract_dir=extract_dir,
                    require_boolean=True,
                )


def test_required_preprocessing_matching_label_geometry_gates_before_other_scoring():
    input_geometry = _topology_geometry([1.0e6])
    matching_geometry = {"geometry": _topology_geometry([1.0e6])}
    with TemporaryDirectory() as extract_dir:
        with patch(
            "freecad_validator.fem.step_interface._extract",
            side_effect=[input_geometry, matching_geometry],
        ) as extract:
            rep = score_step_fcstd(
                "source.step",
                "reference.FCStd",
                "candidate.FCStd",
                freecad_cmd="/fake/freecadcmd",
                extract_dir=extract_dir,
                require_preprocessing=True,
            )

    assert rep.overall_score == 0
    assert rep.grade == "invalid"
    assert rep.gates_triggered == [{"reason": FailureMode.PREPROCESSING_NOT_PERFORMED}]
    assert len(extract.call_args_list) == 2
    assert extract.call_args_list[0].kwargs.get("extra_args") is None
    assert extract.call_args_list[1].kwargs["extra_args"] == ["geometry-only"]


def test_required_preprocessing_changed_geometry_continues_normal_scoring():
    input_geometry = {"volume_mm3": 1.0e6, "surface_area_mm2": 2.5e5}
    candidate_geometry = {"geometry": {"volume_mm3": 0.75e6, "surface_area_mm2": 2.0e5}}
    preprocessing_label = {
        **LABEL,
        "geometry": {"volume_mm3": 0.75e6, "characteristic_length_mm": 387.0},
    }
    candidate = _cand(geometry={"volume_mm3": 0.75e6, "characteristic_length_mm": 387.0})
    with TemporaryDirectory() as extract_dir:
        with patch(
            "freecad_validator.fem.step_interface._extract",
            side_effect=[input_geometry, candidate_geometry, preprocessing_label, candidate],
        ) as extract:
            rep = score_step_fcstd(
                "source.step",
                "reference.FCStd",
                "candidate.FCStd",
                freecad_cmd="/fake/freecadcmd",
                extract_dir=extract_dir,
                require_preprocessing=True,
            )

    assert rep.overall_score > 0
    assert not any(
        failure["code"] == FailureMode.GEOMETRY_MISMATCH for failure in rep.failure_modes_detected
    )
    assert any("reference FCStd geometry target" in item for item in rep.evidence)
    assert len(extract.call_args_list) == 4
    assert extract.call_args_list[0].kwargs.get("extra_args") is None
    assert extract.call_args_list[1].kwargs["extra_args"] == ["geometry-only"]
    assert extract.call_args_list[-1].kwargs["extra_args"] == ["verify-solve"]


def test_corrupt_candidate_is_scored_zero_after_trusted_inputs_extract():
    with TemporaryDirectory() as extract_dir:
        with patch(
            "freecad_validator.fem.step_interface._extract",
            side_effect=[STEP, LABEL, ExtractionError("candidate FCStd is corrupt")],
        ) as extract:
            rep = score_step_fcstd(
                "source.step",
                "reference.FCStd",
                "candidate.FCStd",
                freecad_cmd="/fake/freecadcmd",
                extract_dir=extract_dir,
            )

    assert rep.overall_score == 0
    assert rep.grade == "invalid"
    assert [call.args[2] for call in extract.call_args_list] == [
        "source.step",
        "reference.FCStd",
        "candidate.FCStd",
    ]
    assert any("candidate FCStd is corrupt" in evidence for evidence in rep.evidence)


def test_candidate_no_result_payload_is_scored_zero():
    with TemporaryDirectory() as extract_dir:
        with patch(
            "freecad_validator.fem.step_interface._extract",
            side_effect=[
                STEP,
                LABEL,
                {"no_result": True, "extraction_error": "no FEM result object found"},
            ],
        ):
            rep = score_step_fcstd(
                "source.step",
                "reference.FCStd",
                "candidate.FCStd",
                freecad_cmd="/fake/freecadcmd",
                extract_dir=extract_dir,
            )

    assert rep.overall_score == 0
    assert rep.grade == "invalid"
    assert any("no FEM result object found" in evidence for evidence in rep.evidence)


def test_reference_extraction_failure_remains_infrastructure_error():
    with TemporaryDirectory() as extract_dir:
        with patch(
            "freecad_validator.fem.step_interface._extract",
            side_effect=[STEP, ExtractionError("reference FCStd is corrupt")],
        ):
            with pytest.raises(ExtractionError, match="reference FCStd is corrupt"):
                score_step_fcstd(
                    "source.step",
                    "reference.FCStd",
                    "candidate.FCStd",
                    freecad_cmd="/fake/freecadcmd",
                    extract_dir=extract_dir,
                )


def test_reference_no_result_payload_remains_infrastructure_error():
    with TemporaryDirectory() as extract_dir:
        with patch(
            "freecad_validator.fem.step_interface._extract",
            side_effect=[
                STEP,
                {"no_result": True, "extraction_error": "no FEM result object found"},
            ],
        ):
            with pytest.raises(
                ExtractionError, match="reference FCStd contains no loaded FEM result"
            ):
                score_step_fcstd(
                    "source.step",
                    "reference.FCStd",
                    "candidate.FCStd",
                    freecad_cmd="/fake/freecadcmd",
                    extract_dir=extract_dir,
                )


def test_extract_does_not_reuse_stale_json():
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        freecad_cmd = temp_path / "freecadcmd"
        freecad_cmd.write_text("", encoding="utf-8")
        out_path = temp_path / "candidate.json"
        out_path.write_text('{"results": {"stale": true}}', encoding="utf-8")
        process = Mock(pid=1234)
        process.wait.return_value = 1

        with patch("freecad_validator.fem.step_interface.subprocess.Popen", return_value=process):
            with pytest.raises(ExtractionError, match="adapter failed with exit 1"):
                _extract(str(freecad_cmd), "adapter.py", "candidate.FCStd", str(out_path))

        assert not out_path.exists()


def test_adapter_timeout_terminates_process_tree(tmp_path):
    freecad_cmd = tmp_path / "freecadcmd"
    freecad_cmd.write_text("", encoding="utf-8")
    process = Mock(pid=1234)
    process.wait.side_effect = [subprocess.TimeoutExpired("adapter", 0.01), 0]
    process.poll.return_value = None

    with (
        patch("freecad_validator.fem.step_interface.subprocess.Popen", return_value=process),
        patch("freecad_validator.fem.step_interface._terminate_process_tree") as terminate,
        pytest.raises(ExtractionError, match="timed out after 0.01 seconds"),
    ):
        _run_adapter(
            str(freecad_cmd),
            "adapter.py",
            str(tmp_path / "out.json"),
            [],
            0.01,
        )
    terminate.assert_called_once_with(process)


def test_adapter_workers_isolate_and_clean_temp_directories(tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import json, os, pathlib, sys\n"
        "root = pathlib.Path(os.environ['TMPDIR'])\n"
        "assert root.is_dir()\n"
        "assert os.environ['TEMP'] == os.environ['TMP'] == str(root)\n"
        "(root / 'vtk_extract_datadir').mkdir()\n"
        "pathlib.Path(sys.argv[-1]).write_text(json.dumps({'temp': str(root)}))\n",
        encoding="utf-8",
    )
    original_tmp = os.environ.get("TMPDIR")
    payloads = [
        _run_adapter(sys.executable, str(worker), str(tmp_path / f"out{i}.json"), [], 10)
        for i in range(2)
    ]
    assert payloads[0]["temp"] != payloads[1]["temp"]
    assert all(not Path(payload["temp"]).exists() for payload in payloads)
    assert os.environ.get("TMPDIR") == original_tmp


def test_runtime_failure_is_not_scored_as_candidate_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "freecad_validator.fem.step_interface._runtime_preflight",
        Mock(side_effect=RuntimeEnvironmentError("CalculiX unavailable")),
    )
    with pytest.raises(RuntimeEnvironmentError, match="CalculiX unavailable"):
        score_step_fcstd(
            "source.step",
            "reference.FCStd",
            "candidate.FCStd",
            freecad_cmd="/fake/freecadcmd",
            extract_dir=str(tmp_path),
        )


def test_report_includes_runtime_provenance():
    with TemporaryDirectory() as extract_dir:
        with patch(
            "freecad_validator.fem.step_interface._extract",
            side_effect=[STEP, LABEL, _cand()],
        ):
            report = score_step_fcstd(
                "source.step",
                "reference.FCStd",
                "candidate.FCStd",
                freecad_cmd="/fake/freecadcmd",
                extract_dir=extract_dir,
            )
    assert report.runtime_provenance["freecad"] == "1.1.0"
    assert report.runtime_provenance["calculix"] == "2.23"


def test_adapter_import_does_not_initialize_package(tmp_path):
    """The installed adapter can run under another interpreter without package init."""
    (tmp_path / "FreeCAD.py").write_text(
        "class Units:\n    pass\n",
        encoding="utf-8",
    )
    femtools = tmp_path / "femtools"
    femtools.mkdir()
    (femtools / "ccxtools.py").write_text(
        "class FemToolsCcx:\n    pass\n",
        encoding="utf-8",
    )
    env = {"PYTHONPATH": str(tmp_path), "PATH": ""}
    completed = subprocess.run(
        [sys.executable, FCSTD_ADAPTER, "import-check"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == ("[fcstd_adapter] pure replay module loaded")


# --- mesh-budget category (candidate #elements vs label baseline) ---------- #
def test_mesh_budget_active_with_label():
    rep = score_trusted_payloads(STEP, LABEL, _cand())  # 12000 == baseline 12000
    assert rep.subscores_details["mesh_budget"]["weight"] == 0.25
    assert rep.subscores["mesh_budget"] == 100.0  # at/below baseline -> full


def test_mesh_budget_fewer_elements_not_penalised():
    rep = score_trusted_payloads(STEP, LABEL, _cand(mesh={"num_elements": 6000}))
    assert rep.subscores["mesh_budget"] == 100.0


def test_mesh_budget_partial_when_over():
    # 12000 * 1.15 = 13800 -> halfway to the 130% ceiling -> ~50
    rep = score_trusted_payloads(STEP, LABEL, _cand(mesh={"num_elements": 13800}))
    assert 45 <= rep.subscores["mesh_budget"] <= 55
    assert rep.pass_fail_flags["mesh_within_budget"] is True


def test_mesh_budget_zero_at_130_percent():
    rep = score_trusted_payloads(STEP, LABEL, _cand(mesh={"num_elements": 15600}))  # 130%
    assert rep.subscores["mesh_budget"] == 0.0
    assert any(f["code"] == "MESH_BUDGET_EXCEEDED" for f in rep.failure_modes_detected)
    assert rep.pass_fail_flags["mesh_within_budget"] is False


def test_mesh_budget_costs_25_points_when_zero():
    """An otherwise-perfect candidate that blows the mesh budget loses ~25 pts."""
    full = score_trusted_payloads(STEP, LABEL, _cand()).overall_score
    over = score_trusted_payloads(STEP, LABEL, _cand(mesh={"num_elements": 16000})).overall_score
    assert full - over >= 20


def test_mesh_budget_underresolved_floor_not_spoofable():
    """A trivially coarse mesh cannot spoof full mesh-budget credit.

    Regression for the mesh-budget spoof: an agent could solve on a fine mesh, then
    attach a 1-element mesh to the result so the budget reads 'fewer elements ->
    full credit'. A mesh below the floor (5% of baseline = 600) now scores 0, so the
    25% efficiency share cannot be farmed with a degenerate mesh."""
    rep = score_trusted_payloads(STEP, LABEL, _cand(mesh={"num_elements": 1}))
    assert rep.subscores["mesh_budget"] == 0.0
    assert any(f["code"] == "MESH_BUDGET_UNDERRESOLVED" for f in rep.failure_modes_detected)
    assert rep.pass_fail_flags["mesh_within_budget"] is False
    # and a legitimately efficient mesh just above the floor still earns full credit
    ok = score_trusted_payloads(STEP, LABEL, _cand(mesh={"num_elements": 6000}))
    assert ok.subscores["mesh_budget"] == 100.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok:", fn.__name__)
    print(f"{len(fns)} step-interface tests passed")


def test_material_counts_are_compared_independently_of_first_material():
    steel = {"E_MPa": 200000.0, "nu": 0.3, "rho_kg_m3": 7900.0, "body_count": 3}
    aluminum = {"E_MPa": 70000.0, "nu": 0.35, "rho_kg_m3": 2700.0, "body_count": 2}
    reference = {**LABEL, "materials": [steel, aluminum]}
    candidate = _cand(material=aluminum, materials=[aluminum, steel])
    report = score_trusted_payloads(STEP, reference, candidate)
    assert report.subscores["problem_setup"] == 100
    assert not report.gates_triggered

    candidate["materials"] = [{**aluminum, "body_count": 3}, {**steel, "body_count": 2}]
    report = score_trusted_payloads(STEP, reference, candidate)
    assert report.overall_score == 0
    assert any(f["code"] == FailureMode.WRONG_MATERIAL for f in report.failure_modes_detected)


@pytest.mark.parametrize(
    "property_name,invalid_value",
    [
        ("E_MPa", 0.0),
        ("rho_kg_m3", -1.0),
        ("nu", 0.8),
        ("E_MPa", float("inf")),
        ("rho_kg_m3", float("nan")),
        ("nu", float("nan")),
    ],
)
def test_every_material_card_must_be_physically_admissible(property_name, invalid_value):
    steel = {"E_MPa": 200000.0, "nu": 0.3, "rho_kg_m3": 7900.0, "body_count": 1}
    aluminum = {"E_MPa": 70000.0, "nu": 0.35, "rho_kg_m3": 2700.0, "body_count": 1}
    reference = {**LABEL, "materials": [steel, aluminum]}
    invalid_aluminum = {**aluminum, property_name: invalid_value}
    report = score_trusted_payloads(
        STEP,
        reference,
        _cand(material=steel, materials=[steel, invalid_aluminum]),
    )
    assert report.overall_score == 0
    assert any(
        finding["category"] == "physical_validity"
        and finding["code"] == FailureMode.PHYSICALLY_IMPOSSIBLE
        and "Material 2" in finding["evidence"]
        for finding in report.failure_modes_detected
    )


def test_missing_material_counts_cannot_bypass_new_reference_check():
    report = score_trusted_payloads(
        STEP, {**LABEL, "materials": [{**LABEL["material"], "body_count": 1}]}, _cand()
    )
    assert report.overall_score == 0
    assert any(f["code"] == FailureMode.WRONG_MATERIAL for f in report.failure_modes_detected)


def test_tolerated_material_reordering_does_not_zero_a_correct_submission():
    materials = [{"E_MPa": 200000.0, "body_count": 1}, {"E_MPa": 210000.0, "body_count": 2}]
    candidate = [dict(materials[0]), {**materials[1], "E_MPa": 190000.0}]
    report = score_trusted_payloads(
        STEP, {**LABEL, "materials": materials}, _cand(materials=candidate)
    )
    assert report.overall_score >= 90
    assert not report.gates_triggered


@pytest.mark.parametrize(
    "changes",
    [
        {"E_MPa": 190000.0},
        {"rho_kg_m3": 7800.0},
        {"nu": 0.31},
    ],
)
def test_partial_material_variation_scores_like_uniform_variation(changes):
    material = {**LABEL["material"], "rho_kg_m3": 7900.0, "body_count": 1}
    reference = {**LABEL, "materials": [material, material]}
    baseline = score_trusted_payloads(STEP, reference, _cand(materials=[material, material]))
    for cards in ([material, {**material, **changes}], [{**material, **changes}] * 2):
        report = score_trusted_payloads(STEP, reference, _cand(materials=cards))
        assert report.overall_score == baseline.overall_score
        assert report.subscores["problem_setup"] == 100
        assert not report.gates_triggered


def test_preprocessing_topology_changes_with_equal_volume_and_area_are_allowed():
    source = _topology_geometry([0.4e6, 0.6e6])
    partitioned_edge = deepcopy(source)
    partitioned_edge["regions"][0]["num_edges"] += 2
    partitioned_edge["num_edges"] += 2
    split_solid = _topology_geometry([0.2e6, 0.2e6, 0.6e6])
    changed_regions = _topology_geometry([0.45e6, 0.55e6])
    for candidate in (partitioned_edge, split_solid, changed_regions):
        assert candidate["volume_mm3"] == source["volume_mm3"]
        assert candidate["surface_area_mm2"] == source["surface_area_mm2"]
        assert _preprocessing_gate_report(source, candidate) is None


def test_preprocessing_region_order_and_float_tolerance():
    source = _topology_geometry([0.4e6, 0.6e6])
    for relative_change, unchanged in ((1e-6, True), (2e-5, False)):
        candidate = deepcopy(source)
        candidate["regions"].reverse()
        for geometry in [candidate, *candidate["regions"]]:
            for field in ("volume_mm3", "surface_area_mm2"):
                geometry[field] *= 1 + relative_change
        report = _preprocessing_gate_report(source, candidate)
        if unchanged:
            assert report.overall_score == 0
            assert report.gates_triggered == [{"reason": FailureMode.PREPROCESSING_NOT_PERFORMED}]
        else:
            assert report is None


def test_preprocessing_region_matching_can_reassign_an_ambiguous_pair():
    source = _topology_geometry([100.0009, 99.9991])
    candidate = _topology_geometry([100.0, 100.0018])
    for geometry in (source, candidate):
        for region in geometry["regions"]:
            region.update(surface_area_mm2=500.0, num_faces=6, num_edges=12)

    # Source region 0 fits either candidate region, while source region 1 only
    # fits candidate region 0. Greedy pairing used to consume region 0 first.
    assert topology_difference(source, candidate) < 1.0
    assert topology_mismatches(source, candidate) == []
    report = _preprocessing_gate_report(source, candidate)
    assert report.overall_score == 0
    assert report.gates_triggered == [{"reason": FailureMode.PREPROCESSING_NOT_PERFORMED}]


def test_preprocessing_uses_maximum_normalized_difference():
    source = _topology_geometry([1.0e6])
    candidate = deepcopy(source)
    for geometry in (candidate, candidate["regions"][0]):
        geometry["volume_mm3"] *= 1 + 0.6e-5
        geometry["surface_area_mm2"] *= 1 + 0.8e-5

    # Several differences inside tolerance must not add up to a change.
    difference = topology_difference(source, candidate)
    assert difference == pytest.approx(0.8, rel=1e-4)
    report = _preprocessing_gate_report(source, candidate)
    assert report.overall_score == 0
    recorded = next(
        item for item in report.evidence if item.startswith("preprocessing_max_normalized_diff:")
    )
    assert float(recorded.split(":")[1]) == pytest.approx(difference, rel=1e-5)


def test_preprocessing_single_count_change_exceeds_normalized_threshold():
    source = _topology_geometry([1.0e6])
    source["num_edges"] = source["regions"][0]["num_edges"] = 100000
    candidate = deepcopy(source)
    candidate["num_edges"] += 1
    candidate["regions"][0]["num_edges"] += 1
    assert topology_difference(source, candidate) == 2.0
    assert _preprocessing_gate_report(source, candidate) is None


def test_preprocessing_missing_topology_cannot_establish_an_unchanged_match():
    for missing_side in (0, 1):
        geometries = [_topology_geometry([1.0e6]), _topology_geometry([1.0e6])]
        del geometries[missing_side]["regions"]
        with pytest.raises(ExtractionError, match="missing Boolean topology fields: regions"):
            _preprocessing_gate_report(*geometries)


def test_required_preprocessing_face_partition_continues_to_solver_verification():
    source = _topology_geometry([1.0e6])
    prepared = deepcopy(source)
    prepared["num_faces"] += 1
    prepared["num_edges"] += 2
    prepared["regions"][0]["num_faces"] += 1
    prepared["regions"][0]["num_edges"] += 2
    label = {**LABEL, "geometry": prepared}
    candidate = _cand(geometry=prepared)
    with TemporaryDirectory() as extract_dir:
        with patch(
            "freecad_validator.fem.step_interface._extract",
            side_effect=[source, {"geometry": prepared}, label, candidate],
        ) as extract:
            report = score_step_fcstd(
                "source.step",
                "reference.FCStd",
                "candidate.FCStd",
                freecad_cmd="/fake/freecadcmd",
                extract_dir=extract_dir,
                require_preprocessing=True,
            )

    assert report.overall_score > 0
    assert report.gates_triggered == []
    assert len(extract.call_args_list) == 4
    assert extract.call_args_list[-1].kwargs["extra_args"] == ["verify-solve"]
    assert any("reference FCStd geometry target" in item for item in report.evidence)
