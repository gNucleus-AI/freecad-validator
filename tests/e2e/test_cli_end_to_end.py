"""The `freecad-validator` CLI over real geometry.

The CLI is the documented entry point in the README, so its argument
wiring and output contract are worth exercising against real files
rather than mocks.
"""

from __future__ import annotations

import json

import pytest

from freecad_validator.cli.main import main

pytestmark = pytest.mark.needs_freecad


def _run(capsys, *argv) -> str:
    code = main(list(argv))
    assert code == 0, f"CLI exited {code}"
    return capsys.readouterr().out


def test_validate_prints_the_three_scores(capsys, box_10x5x3, box_spec):
    out = _run(capsys, "validate", str(box_10x5x3), str(box_10x5x3), str(box_spec))
    assert "geometry_similarity" in out
    assert "cad_spec_consistency" in out
    assert "combined" in out


def test_validate_json_is_machine_readable(capsys, box_10x5x3, box_spec):
    """--json is what downstream tooling consumes, so it must parse and
    carry the documented keys."""
    out = _run(capsys, "validate", str(box_10x5x3), str(box_10x5x3), str(box_spec), "--json")
    payload = json.loads(out)
    assert payload["geometry_similarity"] == pytest.approx(1.0)
    assert 0.0 <= payload["cad_spec_consistency"] <= 1.0
    assert "failure_budget=10" in payload["cad_spec_consistency_reason"]
    assert 0.0 <= payload["combined"] <= 1.0


def test_validate_json_reports_a_difference(capsys, box_10x5x3, box_20x5x3, box_spec):
    out = _run(capsys, "validate", str(box_20x5x3), str(box_10x5x3), str(box_spec), "--json")
    assert json.loads(out)["geometry_similarity"] < 1.0


def test_combine_method_flag_is_applied(capsys, box_10x5x3, box_20x5x3, box_spec):
    out = _run(
        capsys,
        "validate",
        str(box_20x5x3),
        str(box_10x5x3),
        str(box_spec),
        "--json",
        "--combine-method",
        "min",
    )
    payload = json.loads(out)
    assert payload["combined"] == pytest.approx(
        min(payload["geometry_similarity"], payload["cad_spec_consistency"])
    )


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_tolerance_flag_is_accepted(capsys, box_10x5x3, box_20x5x3, box_spec, version):
    """Loosening the volume tolerance must not crash the arg wiring."""
    out = _run(
        capsys,
        "validate",
        str(box_20x5x3),
        str(box_10x5x3),
        str(box_spec),
        "--json",
        "--scorer",
        version,
        "--volume-far-rel-tol",
        "5.0",
    )
    assert 0.0 <= json.loads(out)["geometry_similarity"] <= 1.0


def test_v2_bbox_rejection_threshold_override_is_applied(capsys, box_10x5x3, box_20x5x3, box_spec):
    args = ["validate", str(box_20x5x3), str(box_10x5x3), str(box_spec), "--json"]
    default = json.loads(_run(capsys, *args))
    relaxed = json.loads(_run(capsys, *args, "--bbox-far-rel-tol", "0.6"))

    assert "gated geometry to 0.0" in default["geometry_similarity_reason"]
    assert "gate passed" in relaxed["geometry_similarity_reason"]


def test_v2_cli_accepts_bbox_gate_below_one_percent(capsys, box_10x5x3, box_20x5x3, box_spec):
    def score(candidate):
        return json.loads(
            _run(
                capsys,
                "validate",
                str(candidate),
                str(box_10x5x3),
                str(box_spec),
                "--bbox-far-rel-tol",
                "0.005",
                "--json",
            )
        )

    identical = score(box_10x5x3)
    different = score(box_20x5x3)
    assert identical["geometry_similarity"] == pytest.approx(1.0)
    assert "gate passed" in identical["geometry_similarity_reason"]
    assert different["geometry_similarity"] == 0.0
    assert "gated geometry to 0.0" in different["geometry_similarity_reason"]
