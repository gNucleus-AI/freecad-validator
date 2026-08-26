"""The full validate() path against real FreeCAD geometry."""

from __future__ import annotations

import pytest
from _geometry import make_plain_part_box, write_spec

from freecad_validator import Validator

pytestmark = pytest.mark.needs_freecad


def _validate(candidate, reference, spec, **kw):
    return Validator(**kw).validate(
        candidate_fcstd=str(candidate),
        reference_fcstd=str(reference),
        spec_json=str(spec),
    )


def test_identical_geometry_saturates(box_10x5x3, box_spec):
    """A part scored against itself must saturate geometry similarity.
    This is the broad regression canary for the whole geometry path."""
    result = _validate(box_10x5x3, box_10x5x3, box_spec)
    assert result.geometry_similarity == pytest.approx(1.0)


def test_different_geometry_scores_lower(box_10x5x3, box_20x5x3, box_spec):
    """Doubling one dimension must be visible to the geometry score."""
    result = _validate(box_20x5x3, box_10x5x3, box_spec)
    assert result.geometry_similarity < 1.0


def test_scores_are_in_range(box_10x5x3, box_20x5x3, box_spec):
    """All three published numbers stay inside [0, 1]."""
    result = _validate(box_20x5x3, box_10x5x3, box_spec)
    for value in (result.geometry_similarity, result.cad_spec_consistency, result.combined):
        assert 0.0 <= value <= 1.0


def test_combined_is_harmonic_mean_of_the_two_axes(box_10x5x3, box_20x5x3, box_spec):
    """Default combiner is the harmonic mean of the two sub-scores."""
    r = _validate(box_20x5x3, box_10x5x3, box_spec)
    g, s = r.geometry_similarity, r.cad_spec_consistency
    expected = 2.0 * g * s / (g + s) if g > 0 and s > 0 else 0.0
    assert r.combined == pytest.approx(expected)


def test_min_combiner_pins_to_weaker_axis(box_10x5x3, box_20x5x3, box_spec):
    r = _validate(box_20x5x3, box_10x5x3, box_spec, combine_method="min")
    assert r.combined == pytest.approx(min(r.geometry_similarity, r.cad_spec_consistency))


def test_document_without_partdesign_body_is_rejected(tmp_path, box_10x5x3, box_spec):
    """A bare Part::Box is not a PartDesign feature tree. The validator
    zero-gates it and says so — the contract is easy to trip over, since
    the document opens fine and has a perfectly good solid in it."""
    plain = make_plain_part_box(tmp_path / "plain.FCStd", 10, 5, 3)
    result = _validate(plain, box_10x5x3, box_spec)
    assert result.geometry_similarity == 0.0
    assert "PartDesign::Body" in result.geometry_similarity_reason


def test_zero_on_one_axis_gates_combined(tmp_path, box_10x5x3, box_spec):
    """A zeroed axis must drag `combined` to 0 rather than be averaged away."""
    plain = make_plain_part_box(tmp_path / "plain2.FCStd", 10, 5, 3)
    result = _validate(plain, box_10x5x3, box_spec)
    assert result.combined == 0.0


def test_reasons_are_populated(box_10x5x3, box_spec):
    """Both reason strings are human-readable output, not empty."""
    result = _validate(box_10x5x3, box_10x5x3, box_spec)
    assert result.geometry_similarity_reason.strip()
    assert result.cad_spec_consistency_reason.strip()


def test_cylinder_round_trips(cylinder_r4_h12, box_spec):
    """Curved geometry saturates against itself too (the box case alone
    would not exercise the conic-surface paths)."""
    result = _validate(cylinder_r4_h12, cylinder_r4_h12, box_spec)
    assert result.geometry_similarity == pytest.approx(1.0)


def test_box_vs_cylinder_is_not_identical(box_10x5x3, cylinder_r4_h12, box_spec):
    """Different surface types must not read as the same part."""
    result = _validate(cylinder_r4_h12, box_10x5x3, box_spec)
    assert result.geometry_similarity < 1.0


def test_matching_spec_scores_fully_consistent(box_10x5x3, box_spec):
    """The spec pass must AGREE with geometry it actually describes.
    Asserting only a range would let a scorer returning a constant pass."""
    result = _validate(box_10x5x3, box_10x5x3, box_spec)
    assert result.cad_spec_consistency == pytest.approx(1.0)


def test_wrong_spec_scores_inconsistent(tmp_path, box_10x5x3):
    """Every parameter contradicted by the geometry must be marked
    inconsistent, not quietly ignored."""
    wrong = write_spec(tmp_path / "wrong.json", length="999 mm", width="888 mm", height="777 mm")
    result = _validate(box_10x5x3, box_10x5x3, wrong)
    assert result.cad_spec_consistency == pytest.approx(0.0)


def test_spec_score_discriminates_between_right_and_wrong(tmp_path, box_10x5x3, box_spec):
    """The two directions must be ordered — the property a constant
    return value could never satisfy."""
    wrong = write_spec(tmp_path / "wrong2.json", length="999 mm", height="777 mm")
    right = _validate(box_10x5x3, box_10x5x3, box_spec).cad_spec_consistency
    bad = _validate(box_10x5x3, box_10x5x3, wrong).cad_spec_consistency
    assert right > bad


def test_partially_wrong_spec_scores_between(tmp_path, box_10x5x3):
    """Two of three parameters right must land strictly between the
    all-right and all-wrong cases, so the score is graded not binary."""
    partial = write_spec(tmp_path / "partial.json", length="10 mm", width="5 mm", height="777 mm")
    result = _validate(box_10x5x3, box_10x5x3, partial)
    assert 0.0 < result.cad_spec_consistency < 1.0


def test_spec_zero_gates_combined_even_with_perfect_geometry(tmp_path, box_10x5x3):
    """Identical geometry scores 1.0, but a fully wrong spec must still
    drag `combined` to 0 — neither axis can rescue the other."""
    wrong = write_spec(tmp_path / "wrong3.json", length="999 mm", width="888 mm", height="777 mm")
    result = _validate(box_10x5x3, box_10x5x3, wrong)
    assert result.geometry_similarity == pytest.approx(1.0)
    assert result.combined == 0.0
