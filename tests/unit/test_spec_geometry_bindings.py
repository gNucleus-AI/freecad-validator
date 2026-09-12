"""V2 binding routing, all-parameter penalties and target-independent matching."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from freecad_validator.consistency.checker import ConsistencyChecker
from freecad_validator.consistency.geometry_bindings import (
    GeometryBinding,
    GeometryBindingError,
    GeometryBindingSpec,
    align_datum,
    apply_bindings,
    evaluate_binding,
    parse_bindings,
)
from freecad_validator.consistency.report import ConsistencyReport, ParamFinding, compute_summary
from freecad_validator.measurement.schema import FeatureTreeEntry, MeasurementBank
from freecad_validator.measurement.spatial import (
    SpatialBank,
    SpatialDatum,
    SpatialFeature,
    SpatialLocation,
    SpatialMeasurementError,
)
from freecad_validator.scorers.spec_consistency import HeuristicSpecConsistencyScorer
from freecad_validator.scorers.spec_consistency_v2 import HeuristicSpecConsistencyScorerV2
from freecad_validator.spec.parser import parse_spec
from freecad_validator.validator import HeuristicValidator


def feature(x, diameter):
    return SpatialFeature(
        kind="cylinder",
        position=(x, 0, 5),
        direction=(0, 0, 1),
        convex=False,
        scale=10,
        values={"radius": diameter / 2, "diameter": diameter, "axial_extent": 10},
        bounds_min=(x - 5, -5, 0),
        bounds_max=(x + 5, 5, 10),
    )


def bank():
    return SpatialBank(
        datum=SpatialDatum(
            center=(0, 0, 0), diagonal=100, frames=[((1, 0, 0), (0, 1, 0), (0, 0, 1))]
        ),
        features=[feature(-20, 10), feature(20, 8)],
    )


def binding():
    return GeometryBinding(
        mode="geometry",
        reason="The two mounting holes, each at its own location",
        quantity="diameter",
        witnesses=[
            dict(kind="cylinder", position=(x, 0, 5), direction=(0, 0, 1), convex=False, scale=10)
            for x in [-20, 20]
        ],
    )


def evaluate(b, target=10):
    return evaluate_binding(
        binding(), target, b, np.eye(3), np.zeros(3), tol_scalar=0.01, tol_pos=0.01
    )


def test_equal_value_on_wrong_hole_cannot_rescue_corresponding_hole():
    b = bank()
    status, values, evidence, _ = evaluate(b)
    assert status == "inconsistent" and values == [10, 8]
    assert [r["candidate"]["position"] for r in evidence] == [(-20, 0, 5), (20, 0, 5)]
    # Changing the expected number cannot change which physical holes match.
    assert evaluate(b, 8)[2] == evidence


def test_missing_instance_cannot_reuse_another_hole():
    b = bank()
    b.features.pop()
    status, values, evidence, _ = evaluate(b)
    assert status == "not_found" and values is None
    assert evidence[1]["candidate"] is None


def test_correct_size_at_wrong_location_fails():
    b = bank()
    b.features[1] = feature(22, 10)
    assert evaluate(b)[0] == "not_found"


def test_geometry_pass_overrides_old_failure_and_geometry_failure_overrides_old_pass():
    b = bank()
    raw = {"key_parameters": "diameter = 10 mm\ntransition_length = 4 mm"}
    structured = parse_spec(raw)
    cfg = GeometryBindingSpec(
        version=1,
        datum=b.datum,
        parameters={
            "diameter": binding(),
            "transition_length": {"mode": "legacy", "reason": "Not independently observable"},
        },
    )
    report = ConsistencyReport(spec_name="fixture", fcstd_path="candidate")
    report.consistent = [ParamFinding(param="diameter", spec_value=10, measured_value=10)]
    report.not_found = [ParamFinding(param="transition_length", spec_value=4)]
    apply_bindings(report, cfg, b, structured, tol_scalar=0.01, tol_pos=0.01)
    summary = compute_summary(report)
    assert (summary.total_params, summary.inconsistent, summary.not_found) == (2, 1, 1)
    assert report.not_found[0].param == "transition_length"
    b.features[1] = feature(20, 10)
    apply_bindings(report, cfg, b, structured, tol_scalar=0.01, tol_pos=0.01)
    summary = compute_summary(report)
    assert (summary.total_params, summary.consistent, summary.not_found) == (2, 1, 1)


def test_v2_all_thirteen_parameters_count_including_legacy_not_found(tmp_path, monkeypatch):
    path = tmp_path / "spec.json"
    candidate = tmp_path / "answer.FCStd"
    path.write_text("{}")
    candidate.write_bytes(b"fixture")
    report = ConsistencyReport(spec_name="fixture", fcstd_path=str(candidate))
    report.consistent = [ParamFinding(param=f"p{i}", spec_value=i) for i in range(11)]
    report.inconsistent = [ParamFinding(param="bound_bad", spec_value=10)]
    report.not_found = [ParamFinding(param="legacy_missing", spec_value=40)]
    report.summary = compute_summary(report)
    scorer = HeuristicSpecConsistencyScorerV2()
    monkeypatch.setattr(scorer._checker, "check", lambda *a: report)
    result = scorer.score(str(path), str(candidate))
    assert result.score == 0.8 and result.details["total_params"] == 13
    assert result.details["failures"] == 2 and result.details["failure_denominator"] == 10


def test_v1_ignores_v2_binding_metadata_and_does_not_extract_spatial(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "freecad_validator.consistency.checker.extract_bank",
        lambda path: calls.append(path) or MeasurementBank(solid_count=1),
    )
    report = ConsistencyChecker().check(
        {"key_parameters": "length = 10 mm", "geometry_bindings": {"broken": True}}, "candidate"
    )
    assert calls == ["candidate"] and report.summary.total_params == 1
    assert not report.binding_details
    assert isinstance(
        HeuristicValidator(scorer_version="v1")._spec_scorer, HeuristicSpecConsistencyScorer
    )
    assert not isinstance(
        HeuristicValidator(scorer_version="v1")._spec_scorer, HeuristicSpecConsistencyScorerV2
    )
    assert isinstance(
        HeuristicValidator(scorer_version="v2")._spec_scorer, HeuristicSpecConsistencyScorerV2
    )


def test_v2_invalid_binding_does_not_silently_fall_back():
    raw = {
        "key_parameters": "length = 10 mm",
        "geometry_bindings": {"version": 1, "parameters": {}},
    }
    with pytest.raises(GeometryBindingError, match="every parsed parameter"):
        parse_bindings(raw, parse_spec(raw))
    with pytest.raises(ValueError, match="selection filter"):
        data = copy.deepcopy(binding().model_dump())
        data["witnesses"][0]["radius"] = 5
        GeometryBinding.model_validate(data)


def test_repeated_moments_do_not_make_circular_pattern_pose_arbitrary():
    points = np.array([(2 * np.cos(a), 2 * np.sin(a), 0) for a in np.arange(20) * np.pi / 10])
    angle = 0.37
    turn = np.array(
        ((np.cos(angle), -np.sin(angle), 0), (np.sin(angle), np.cos(angle), 0), (0, 0, 1))
    )
    offset = np.array((12, -17, 4))

    def datum(positions, center, frame):
        return SpatialDatum(
            center=tuple(center),
            diagonal=6,
            frames=[tuple(map(tuple, frame))],
            landmarks=[
                SpatialLocation(
                    kind="cylinder", position=tuple(p), direction=(0, 0, 1), convex=True, scale=1
                )
                for p in positions
            ],
        )

    reference = datum(points, (0, 0, 0), np.eye(3))
    candidate = datum(points @ turn.T + offset, offset, np.eye(3))
    rotation, translation, _ = align_datum(reference, candidate)
    moved = (points @ turn.T + offset) @ rotation.T + translation
    distances = np.linalg.norm(points[:, None] - moved[None, :], axis=2)
    assert distances.min(axis=1).max() < 1e-8


def test_invalid_shape_fails_parameters_but_backend_failure_propagates(monkeypatch):
    raw = {
        "key_parameters": "diameter = 10 mm",
        "geometry_bindings": GeometryBindingSpec(
            version=1, datum=bank().datum, parameters={"diameter": binding()}
        ).model_dump(),
    }
    checker = ConsistencyChecker(use_geometry_bindings=True)

    def invalid(*args, **kwargs):
        return MeasurementBank(solid_count=1, spatial_unavailable_reason="Invalid native shape")

    monkeypatch.setattr("freecad_validator.consistency.checker.extract_bank", invalid)
    report = checker.check(raw, "candidate")
    assert report.summary.not_found == report.summary.total_params == 1

    def unavailable(*args, **kwargs):
        raise SpatialMeasurementError("Native measurement failed")

    monkeypatch.setattr("freecad_validator.consistency.checker.extract_bank", unavailable)
    with pytest.raises(SpatialMeasurementError, match="Native measurement"):
        checker.check(raw, "candidate")


@pytest.mark.parametrize("solid_count", [1, 2])
def test_unavailable_spatial_geometry_preserves_every_legacy_check(monkeypatch, solid_count):
    raw = {
        "key_parameters": "diameter = 10 mm\n"
        + "\n".join(f"legacy{i}_length = {5 if i == 13 else 4} mm" for i in range(14)),
        "geometry_bindings": GeometryBindingSpec(
            version=1,
            datum=bank().datum,
            parameters={
                "diameter": binding(),
                **{
                    f"legacy{i}_length": {"mode": "legacy", "reason": "Legacy owner"}
                    for i in range(14)
                },
            },
        ).model_dump(),
    }
    measured = MeasurementBank(
        solid_count=solid_count,
        spatial_unavailable_reason="Not a valid single solid",
        feature_tree=[
            FeatureTreeEntry(
                name="Pad", type_id="PartDesign::Pad", label="Pad", properties={"Length": 4}
            )
        ],
    )
    monkeypatch.setattr(
        "freecad_validator.consistency.checker.extract_bank", lambda *a, **kw: measured
    )
    result = ConsistencyChecker(use_geometry_bindings=True).check(raw, "candidate")
    assert result.summary.total_params == 15
    assert result.summary.consistent == 13
    assert [f.param for f in result.inconsistent] == ["legacy13_length"]
    assert [f.param for f in result.not_found] == ["diameter"]
    assert result.binding_details["parameters"]["diameter"]["status"] == "not_found"


def test_missing_spatial_bank_without_input_reason_is_an_error():
    cfg = GeometryBindingSpec(version=1, datum=bank().datum, parameters={"diameter": binding()})
    with pytest.raises(GeometryBindingError, match="spatial measurement bank"):
        apply_bindings(
            ConsistencyReport(spec_name="fixture", fcstd_path="candidate"),
            cfg,
            None,
            parse_spec({"key_parameters": "diameter = 10 mm"}),
            tol_scalar=0.01,
            tol_pos=0.01,
        )


def test_valid_candidate_without_supported_landmarks_fails_only_bound_parameter():
    measured = bank()
    datum = measured.datum.model_copy(deep=True)
    datum.landmarks = [
        SpatialLocation(kind="plane", position=(0, 0, 0), direction=(1, 0, 0), scale=10)
    ]
    cfg = GeometryBindingSpec(
        version=1,
        datum=datum,
        parameters={
            "diameter": binding(),
            "transition_length": {"mode": "legacy", "reason": "History"},
        },
    )
    report = ConsistencyReport(spec_name="fixture", fcstd_path="sphere")
    report.consistent = [ParamFinding(param="transition_length", spec_value=4, measured_value=4)]
    apply_bindings(
        report,
        cfg,
        measured,
        parse_spec({"key_parameters": "diameter = 10 mm\ntransition_length = 4 mm"}),
        tol_scalar=0.01,
        tol_pos=0.01,
    )
    assert [f.param for f in report.consistent] == ["transition_length"]
    assert [f.param for f in report.not_found] == ["diameter"]
