"""Count-only material checks: no candidate/reference body correspondence."""

import pytest

from freecad_validator.fem.material_counts import grouped_material_counts
from freecad_validator.fem.schema import CaseDefinition, Submission
from freecad_validator.fem.setup_compare import compare_material_body_counts

STEEL = {
    "name": "Steel-Generic",
    "E_MPa": 200000.0,
    "rho_kg_m3": 7900.0,
    "nu": 0.3,
    "body_count": 3,
}
ALUMINUM = {
    "name": "Aluminum-Generic",
    "E_MPa": 70000.0,
    "rho_kg_m3": 2700.0,
    "nu": 0.35,
    "body_count": 2,
}


def test_order_names_and_split_material_objects_do_not_affect_counts():
    candidate = [
        dict(ALUMINUM),
        {**STEEL, "name": "first", "body_count": 1},
        {**STEEL, "name": "second", "body_count": 2},
    ]
    assert compare_material_body_counts([STEEL, ALUMINUM], candidate) == []
    assert len(grouped_material_counts(candidate)) == 2


def test_matching_counts_do_not_require_body_ids_or_geometry():
    # Count matching does not establish correspondence between individual bodies.
    reference = [STEEL, ALUMINUM]
    candidate = [dict(reference[1]), dict(reference[0])]
    assert compare_material_body_counts(reference, candidate) == []


def test_equal_total_count_with_wrong_material_distribution_fails():
    findings = compare_material_body_counts(
        [STEEL, ALUMINUM], [{**STEEL, "body_count": 2}, {**ALUMINUM, "body_count": 3}]
    )
    assert len(findings) == 2
    assert all(f.code == "WRONG_MATERIAL" and f.severity == "critical" for f in findings)
    assert "expected 3" in findings[1].message


@pytest.mark.parametrize(
    "candidate", [[], [STEEL], [STEEL, ALUMINUM, {**STEEL, "E_MPa": 300000.0}]]
)
def test_missing_or_extra_material_card_fails(candidate):
    assert compare_material_body_counts([STEEL, ALUMINUM], candidate)


@pytest.mark.parametrize("count", [None, -1, 1.5, True, "3"])
def test_missing_or_invalid_candidate_count_fails(count):
    findings = compare_material_body_counts([STEEL], [{**STEEL, "body_count": count}])
    assert findings[0].code == "WRONG_MATERIAL"
    assert findings[0].severity == "critical"


def test_invalid_reference_count_is_an_evaluation_error():
    with pytest.raises(ValueError, match="body_count"):
        compare_material_body_counts([{**STEEL, "body_count": -1}], [STEEL])


def test_existing_modulus_tolerance_is_preserved():
    assert compare_material_body_counts([STEEL], [{**STEEL, "E_MPa": 240000.0}]) == []
    assert compare_material_body_counts([STEEL], [{**STEEL, "E_MPa": 241000.0}])


@pytest.mark.parametrize("reverse_reference", [False, True])
@pytest.mark.parametrize("reverse_candidate", [False, True])
def test_tolerated_modulus_change_can_reverse_material_order(reverse_reference, reverse_candidate):
    reference = [{**STEEL, "body_count": 1}, {**STEEL, "E_MPa": 210000.0, "body_count": 2}]
    candidate = [dict(reference[0]), {**reference[1], "E_MPa": 190000.0}]
    if reverse_reference:
        reference.reverse()
    if reverse_candidate:
        candidate.reverse()
    assert compare_material_body_counts(reference, candidate) == []


def test_matching_does_not_reuse_a_candidate_for_two_reference_cards():
    reference = [{**STEEL, "body_count": 1}, {**STEEL, "E_MPa": 210000.0, "body_count": 1}]
    candidate = [dict(reference[0]), {**reference[1], "E_MPa": 300000.0}]
    assert any(f.severity == "critical" for f in compare_material_body_counts(reference, candidate))


def test_ambiguous_missing_modulus_match_can_move_to_another_card():
    reference = [{**ALUMINUM, "body_count": 1}, {**STEEL, "body_count": 1}]
    candidate = [dict(reference[0]), {**reference[1], "E_MPa": None}]
    findings = compare_material_body_counts(reference, candidate)
    assert len(findings) == 1
    assert findings[0].code == "MATERIAL_NOT_STATED"
    assert findings[0].severity == "minor"


def test_no_new_density_or_poisson_tolerance_is_introduced():
    assert compare_material_body_counts([STEEL], [{**STEEL, "rho_kg_m3": 7800.0, "nu": 0.31}]) == []


@pytest.mark.parametrize(
    "changes",
    [
        {"E_MPa": 190000.0},
        {"rho_kg_m3": 7800.0},
        {"nu": 0.31},
    ],
)
@pytest.mark.parametrize("merged_candidate", [False, True])
def test_tolerated_card_splits_and_merges_preserve_counts(changes, merged_candidate):
    merged = [{**STEEL, "body_count": 2}]
    split = [{**STEEL, "body_count": 1}, {**STEEL, **changes, "body_count": 1}]
    reference, candidate = (split, merged) if merged_candidate else (merged, split)
    assert compare_material_body_counts(reference, candidate) == []


def test_split_material_counts_still_reject_wrong_distribution():
    candidate = [
        {**STEEL, "body_count": 1},
        {**STEEL, "E_MPa": 190000.0, "body_count": 1},
        {**ALUMINUM, "body_count": 3},
    ]
    assert any(
        f.severity == "critical" for f in compare_material_body_counts([STEEL, ALUMINUM], candidate)
    )


def test_compatible_card_capacity_can_be_shared_but_not_reused():
    reference = [{"E_MPa": 100.0, "body_count": 2}, {"E_MPa": 120.0, "body_count": 1}]
    candidate = [{"E_MPa": 90.0, "body_count": 1}, {"E_MPa": 110.0, "body_count": 2}]
    assert compare_material_body_counts(reference, candidate) == []
    candidate = [{"E_MPa": 90.0, "body_count": 2}, {"E_MPa": 300.0, "body_count": 1}]
    assert any(f.severity == "critical" for f in compare_material_body_counts(reference, candidate))


def test_modulus_compatibility_is_not_transitive():
    reference = [{"E_MPa": 100.0, "body_count": 3}]
    candidate = [{"E_MPa": E, "body_count": 1} for E in (100.0, 120.0, 144.0)]
    assert any(f.severity == "critical" for f in compare_material_body_counts(reference, candidate))


def test_split_unknown_modulus_retains_one_minor_penalty_per_card():
    reference = [{**STEEL, "body_count": 2}, {**ALUMINUM, "body_count": 1}]
    findings = compare_material_body_counts(reference, [{"body_count": 3}])
    assert len(findings) == 1
    assert findings[0].code == "MATERIAL_NOT_STATED"
    assert findings[0].severity == "minor"


def test_material_counts_survive_schema_roundtrip():
    case = CaseDefinition("materials", "Materials", materials=[STEEL, ALUMINUM])
    submission = Submission("materials", materials=[ALUMINUM, STEEL])
    assert CaseDefinition.from_dict(case.to_dict()).materials == case.materials
    assert Submission.from_dict(submission.to_dict()).materials == submission.materials
