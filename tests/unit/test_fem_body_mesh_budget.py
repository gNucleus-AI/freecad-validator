"""Case-local budgets reuse body identity and preserve the existing score gates."""

import json
from copy import deepcopy
from unittest.mock import patch

import pytest
from test_fem_step_interface import LABEL, STEP, _cand, resolve_fake_freecad_command  # noqa: F401

from freecad_validator.fem.errors import EvaluationError
from freecad_validator.fem.mesh.budget import load_case_mesh_budget, score_local_node_budget
from freecad_validator.fem.mesh.integration import apply_local_mesh_budget
from freecad_validator.fem.step_interface import (
    MESH_BUDGET_ADAPTER,
    PREPROCESSING_ADAPTER,
    score_step_fcstd,
    score_trusted_payloads,
)


@pytest.mark.parametrize("count,expected", [(1, 100), (49, 100), (100, 100), (101, 0), (0, 0)])
def test_inclusive_local_cap(count, expected):
    result = score_local_node_budget({"local_max": {"Body1": 100}}, {"Body1": 40}, {"Body1": count})
    assert result["score"] == expected
    assert result["ratio_to_cap"] == count / 100


def test_only_configured_body_is_scored():
    result = score_local_node_budget(
        {"local_max": {"Body1": 100}},
        {"Body1": 40, "Body2": 50},
        {"Body1": 100, "Body2": 10000, "ExtraBody1": 30000},
    )
    assert result["score"] == 100
    result = score_local_node_budget({"local_max": {"Body1": 100}}, {"Body1": 40}, {"Body2": 50})
    assert result["score"] == 0
    assert result["status"] == "missing_body"


@pytest.mark.parametrize("reference", [{}, {"Body1": 0}, {"Body1": 101}])
def test_invalid_reference_local_cap_raises(reference):
    with pytest.raises(EvaluationError, match="Reference local body"):
        score_local_node_budget({"local_max": {"Body1": 100}}, reference, {"Body1": 100})


@pytest.mark.parametrize(
    "payload",
    [
        "{}",
        "[]",
        '{"Body0": 10}',
        '{"Body1": true}',
        '{"Body1": 1.2}',
        '{"Body1": 0}',
        '{"Body1": -1}',
        '{"Body1": 10, "Body1": 20}',
        '{"local_max": {}}',
        '{"local_max": {"Body1": 100, "Body2": 200}}',
        '{"local_max": {"Body0": 100}}',
        '{"local_max": {"Body1": true}}',
        '{"local_max": {"Body1": 0}}',
        '{"local_max": {"Body1": 1.5}}',
        '{"local_max": {"Body1": 100, "Body1": 200}}',
        '{"local_max": {"Body1": 100}, "other": 3}',
        "not json",
    ],
)
def test_invalid_case_budget_raises(tmp_path, payload):
    (tmp_path / "mesh_budget.json").write_text(payload)
    with pytest.raises(EvaluationError):
        load_case_mesh_budget(tmp_path / "reference.FCStd")


def test_trusted_reference_directory_only(tmp_path):
    trusted = tmp_path / "case"
    candidate = tmp_path / "submission"
    trusted.mkdir()
    candidate.mkdir()
    (candidate / "mesh_budget.json").write_text('{"Body1": 999999}')
    assert load_case_mesh_budget(trusted / "reference.FCStd") is None
    (trusted / "mesh_budget.json").write_text('{"local_max": {"Body1": 123}}')
    assert load_case_mesh_budget(trusted / "reference.FCStd") == {"local_max": {"Body1": 123}}


def test_geometry_locator_is_loaded_and_cap_is_scored(tmp_path):
    budget = {
        "local_max": {
            "Body1": {
                "max_node_count": 100,
                "geometry": {
                    "volume_mm3": 1,
                    "center_mm": [0.5, 0.5, 0.5],
                    "bounds_mm": [0, 0, 0, 1, 1, 1],
                },
            }
        }
    }
    (tmp_path / "mesh_budget.json").write_text(json.dumps(budget))
    assert load_case_mesh_budget(tmp_path / "reference.FCStd") == budget
    assert score_local_node_budget(budget, {"Body1": 40}, {"Body1": 101})["score"] == 0


@pytest.mark.parametrize(
    "key,value",
    [
        ("volume_mm3", -1),
        ("volume_mm3", True),
        ("volume_mm3", float("nan")),
        ("center_mm", [0, 0]),
        ("center_mm", [0, 0, float("inf")]),
        ("center_mm", [0, 0, 2]),
        ("bounds_mm", [0, 0, 0, 0, 1, 1]),
    ],
)
def test_invalid_geometry_locator_is_rejected(tmp_path, key, value):
    geometry = {"volume_mm3": 1, "center_mm": [0.5, 0.5, 0.5], "bounds_mm": [0, 0, 0, 1, 1, 1]}
    geometry[key] = value
    budget = {"local_max": {"Body1": {"max_node_count": 100, "geometry": geometry}}}
    (tmp_path / "mesh_budget.json").write_text(json.dumps(budget))
    with pytest.raises(EvaluationError):
        load_case_mesh_budget(tmp_path / "reference.FCStd")


def test_category_weight_and_global_gate_are_preserved():
    original = score_trusted_payloads(STEP, LABEL, _cand())
    snapshot = deepcopy(original.to_dict())
    evaluation = score_local_node_budget(
        {"local_max": {"Body1": 100}}, {"Body1": 40}, {"Body1": 101}
    )
    result = apply_local_mesh_budget(original, evaluation)
    assert result.overall_score == pytest.approx(original.overall_score - 25)
    assert result.subscores["mesh_budget"] == 0
    assert result.pass_fail_flags["mesh_within_budget"] is False
    assert original.to_dict() == snapshot
    with pytest.raises(EvaluationError, match="already"):
        apply_local_mesh_budget(result, evaluation)
    gated = score_trusted_payloads(STEP, LABEL, _cand(mesh={"num_nodes": 50000}))
    assert gated.gates_triggered
    assert apply_local_mesh_budget(gated, {"score": 100}).overall_score == 0


@pytest.mark.parametrize("preprocessing", [False, True])
def test_case_file_triggers_existing_worker_and_applies_budget_before_multiplier(
    tmp_path, preprocessing
):
    (tmp_path / "mesh_budget.json").write_text('{"local_max": {"Body1": 100}}')
    evaluation = {
        "score": 0.8,
        "mesh_budget": {"score": 50},
        "reference_changed_body_count": 5,
        "extra_changed_body_count": 0,
    }
    with patch(
        "freecad_validator.fem.step_interface._extract",
        side_effect=[STEP, LABEL, _cand(), evaluation],
    ) as extract:
        result = score_step_fcstd(
            "raw.step",
            str(tmp_path / "reference.FCStd"),
            "answer.FCStd",
            extract_dir=str(tmp_path / "extracted"),
            require_preprocessing=preprocessing,
        )
    baseline = score_trusted_payloads(LABEL["geometry"] if preprocessing else STEP, LABEL, _cand())
    assert result.overall_score == pytest.approx(
        (baseline.overall_score - 12.5) * (0.8 if preprocessing else 1)
    )
    assert extract.call_count == 4
    assert extract.call_args.args[1] == (
        PREPROCESSING_ADAPTER if preprocessing else MESH_BUDGET_ADAPTER
    )
    assert ("preprocessing" in result.subscores_details) == preprocessing


def test_missing_budget_worker_data_cannot_award_credit(tmp_path):
    (tmp_path / "mesh_budget.json").write_text('{"local_max": {"Body1": 100}}')
    with patch(
        "freecad_validator.fem.step_interface._extract",
        side_effect=[STEP, LABEL, _cand(), {"score": 1}],
    ):
        with pytest.raises(EvaluationError, match="Missing case"):
            score_step_fcstd(
                "raw.step",
                str(tmp_path / "reference.FCStd"),
                "answer.FCStd",
                extract_dir=str(tmp_path),
            )


def test_invalid_candidate_clean_bodies_lose_budget_credit(tmp_path):
    (tmp_path / "mesh_budget.json").write_text(json.dumps({"local_max": {"Body1": 100}}))
    evaluation = {"status": "missing_clean_bodies", "score": 0, "input_role": "candidate"}
    with patch(
        "freecad_validator.fem.step_interface._extract",
        side_effect=[STEP, LABEL, _cand(), evaluation],
    ):
        result = score_step_fcstd(
            "raw.step", str(tmp_path / "reference.FCStd"), "answer.FCStd", extract_dir=str(tmp_path)
        )
    assert result.subscores["mesh_budget"] == 0
