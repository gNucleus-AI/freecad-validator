"""CLI scoring options must apply to the selected geometry scorer."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

import freecad_validator.cli.main as cli_module
from freecad_validator import GeometryTolerances, ValidationResult, Validator
from freecad_validator.cli.main import main as cli_main
from freecad_validator.scorers.arguments import (
    _TOLERANCE_VERSIONS,
    add_tolerance_arguments,
    tolerances_from_args,
)
from freecad_validator.scorers.geometry import (
    main as v1_main,
)
from freecad_validator.scorers.geometry_v2 import main as v2_main
from freecad_validator.validator import main as validator_main

UNSUPPORTED_OPTIONS = [
    ("v1", "--surface-types-matched-rel-tol"),
    ("v1", "--surface-types-far-rel-tol"),
    ("v1", "--principal-moments-matched-rel-tol"),
    ("v1", "--principal-moments-far-rel-tol"),
    ("v2", "--bbox-matched-rel-tol"),
    ("v2", "--surface-types-exact-tol"),
    ("v2", "--surface-types-zero-score"),
]


@pytest.mark.parametrize(("version", "flag"), UNSUPPORTED_OPTIONS)
@pytest.mark.parametrize("command", ["validate", "batch", "validator_module"])
def test_joint_cli_rejects_options_for_another_version(capsys, tmp_path, version, flag, command):
    if command == "batch":
        entrypoint = cli_main
        args = ["batch", "--sample-data-dir", str(tmp_path)]
    else:
        entrypoint = validator_main if command == "validator_module" else cli_main
        args = [] if command == "validator_module" else ["validate"]
        args += ["candidate.FCStd", "reference.FCStd", "spec.json"]
    args += [flag, "0.02", "--scorer", version]

    with pytest.raises(SystemExit) as exc:
        entrypoint(args)

    assert exc.value.code == 2
    error = capsys.readouterr().err
    assert f"not supported by scorer {version}" in error
    assert flag in error
    assert "Traceback" not in error
    assert list(tmp_path.iterdir()) == []


def test_default_v2_rejects_v1_option_even_when_value_equals_default(capsys):
    with pytest.raises(SystemExit) as exc:
        cli_main(
            [
                "validate",
                "candidate.FCStd",
                "reference.FCStd",
                "spec.json",
                "--bbox-matched-rel-tol",
                "0.01",
            ]
        )
    assert exc.value.code == 2
    error = capsys.readouterr().err
    assert "not supported by scorer v2" in error
    assert "--bbox-matched-rel-tol (v1 only)" in error


@pytest.mark.parametrize(("version", "flag"), UNSUPPORTED_OPTIONS)
def test_standalone_scorer_does_not_register_other_version_options(capsys, version, flag):
    entrypoint = v1_main if version == "v1" else v2_main
    with pytest.raises(SystemExit) as exc:
        entrypoint(["reference.FCStd", "candidate.FCStd", flag, "0.02"])
    assert exc.value.code == 2
    error = capsys.readouterr().err
    assert "unrecognized arguments" in error
    assert flag in error


@pytest.mark.parametrize(("version", "entrypoint"), [("v1", v1_main), ("v2", v2_main)])
def test_standalone_help_shows_only_supported_geometry_options(capsys, version, entrypoint):
    with pytest.raises(SystemExit) as exc:
        entrypoint(["--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--volume-far-rel-tol" in help_text
    assert "--bbox-far-rel-tol" in help_text
    assert "diagnostic-only" not in help_text
    for scorer, flag in UNSUPPORTED_OPTIONS:
        if scorer == version:
            assert flag not in help_text
        else:
            assert flag in help_text


def test_joint_help_groups_geometry_options_by_version(capsys):
    with pytest.raises(SystemExit) as exc:
        cli_main(["validate", "--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "geometry tolerances (v1, v2):" in help_text
    assert "geometry tolerances (v1):" in help_text
    assert "geometry tolerances (v2):" in help_text
    assert "diagnostic-only" not in help_text


@pytest.mark.parametrize(
    ("version", "flag", "field"),
    [
        ("v1", "--bbox-matched-rel-tol", "bbox_matched_rel_tol"),
        ("v1", "--surface-types-exact-tol", "surface_types_exact_tol"),
        ("v2", "--surface-types-matched-rel-tol", "surface_types_matched_rel_tol"),
        ("v2", "--principal-moments-matched-rel-tol", "principal_moments_matched_rel_tol"),
        ("v2", "--bbox-far-rel-tol", "bbox_far_rel_tol"),
    ],
)
def test_supported_options_preserve_explicit_values(version, flag, field):
    parser = argparse.ArgumentParser()
    parser.add_argument("--scorer", choices=["v1", "v2"], default="v2")
    add_tolerance_arguments(parser)
    args = parser.parse_args([flag, "0.02", "--scorer", version])
    tolerances = tolerances_from_args(args, scorer_version=args.scorer)
    assert getattr(tolerances, field) == 0.02


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_omitted_options_keep_scorer_defaults(version):
    parser = argparse.ArgumentParser()
    add_tolerance_arguments(parser)
    assert tolerances_from_args(parser.parse_args([]), scorer_version=version) is None


@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.parametrize("command", ["validate", "batch", "validator_module", "scorer_module"])
@pytest.mark.parametrize("matched", ["0.01", "0.02"], ids=["equal", "inverted"])
def test_cli_rejects_unordered_thresholds_before_reading_files(
    capsys, tmp_path, version, command, matched
):
    if command == "batch":
        entrypoint = cli_main
        args = ["batch", "--sample-data-dir", str(tmp_path), "--scorer", version]
    elif command == "scorer_module":
        entrypoint = v1_main if version == "v1" else v2_main
        args = ["reference.FCStd", "candidate.FCStd"]
    else:
        entrypoint = validator_main if command == "validator_module" else cli_main
        args = [] if command == "validator_module" else ["validate"]
        args += ["candidate.FCStd", "reference.FCStd", "spec.json", "--scorer", version]
    args += ["--volume-matched-rel-tol", matched, "--volume-far-rel-tol", "0.01"]

    with pytest.raises(SystemExit) as exc:
        entrypoint(args)

    assert exc.value.code == 2
    error = capsys.readouterr().err
    assert "volume_matched_rel_tol" in error
    assert "volume_far_rel_tol" in error
    assert "must be less than" in error
    assert "Traceback" not in error
    assert "errors.pydantic.dev" not in error
    assert "input_value" not in error
    assert "input_type" not in error
    if command in ("validate", "batch"):
        assert f"usage: freecad-validator {command} " in error
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("threshold", "expected_matched"),
    [(0.005, 0.0005), (0.01, 0.001), (0.1, 0.01), (0.6, 0.01)],
)
def test_v2_bbox_gate_can_be_tightened_without_a_matched_cli_option(threshold, expected_matched):
    parser = argparse.ArgumentParser()
    add_tolerance_arguments(parser, scorer_version="v2")
    args = parser.parse_args(["--bbox-far-rel-tol", str(threshold)])

    tolerances = tolerances_from_args(args, scorer_version="v2")

    assert tolerances.bbox_far_rel_tol == threshold
    assert tolerances.bbox_matched_rel_tol == pytest.approx(expected_matched)
    assert tolerances.bbox_matched_rel_tol < tolerances.bbox_far_rel_tol
    api_tolerances = GeometryTolerances.for_scorer("v2", bbox_far_rel_tol=threshold)
    assert api_tolerances == tolerances
    # The version-aware configuration is accepted directly by the public API.
    validator = Validator(scorer_version="v2", geom_tolerances=api_tolerances)
    assert validator._geometry_scorer._geom.tolerances == tolerances


def test_v1_bbox_override_still_requires_explicitly_ordered_thresholds():
    parser = argparse.ArgumentParser()
    add_tolerance_arguments(parser, scorer_version="v1")
    args = parser.parse_args(["--bbox-far-rel-tol", "0.005"])
    with pytest.raises(ValueError, match="bbox_matched_rel_tol.*must be less than"):
        tolerances_from_args(args, scorer_version="v1")

    args = parser.parse_args(["--bbox-matched-rel-tol", "0.0005", "--bbox-far-rel-tol", "0.005"])
    tolerances = tolerances_from_args(args, scorer_version="v1")
    assert tolerances.bbox_matched_rel_tol == 0.0005
    assert tolerances.bbox_far_rel_tol == 0.005


def test_cli_tolerance_fields_match_model():
    assert set(_TOLERANCE_VERSIONS) == set(GeometryTolerances.model_fields)


@pytest.mark.parametrize("command", ["validate", "batch"])
def test_handlers_work_with_a_freshly_parsed_namespace(tmp_path, monkeypatch, command):
    result = ValidationResult(
        geometry_similarity=1.0,
        cad_spec_consistency=1.0,
        combined=1.0,
        geometry_similarity_reason="matched",
        cad_spec_consistency_reason="matched",
    )
    captured = []

    def validator(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(combine_method="harmonic", validate=lambda **_paths: result)

    monkeypatch.setattr(cli_module, "Validator", validator)
    parser = argparse.ArgumentParser()
    if command == "validate":
        cli_module._add_validate_args(parser)
        args = ["candidate.FCStd", "reference.FCStd", "spec.json"]
    else:
        case = tmp_path / "data" / "case"
        case.mkdir(parents=True)
        for filename in ("candidate.FCStd", "reference.FCStd", "spec.json"):
            (case / filename).touch()
        cli_module._add_batch_args(parser)
        args = ["--sample-data-dir", str(tmp_path)]
    parsed = parser.parse_args([*args, "--scorer", "v2", "--bbox-far-rel-tol", "0.005"])
    assert not hasattr(parsed, "geom_tolerances")

    assert getattr(cli_module, f"_run_{command}")(parsed) == 0

    assert len(captured) == 1
    assert captured[0]["scorer_version"] == "v2"
    assert captured[0]["geom_tolerances"] == GeometryTolerances.for_scorer(
        "v2", bbox_far_rel_tol=0.005
    )


def test_scoring_value_error_is_not_reported_as_invalid_cli_options(monkeypatch, capsys):
    def fail(**_kwargs):
        raise ValueError("invalid BREP")

    monkeypatch.setattr(cli_module, "Validator", fail)
    with pytest.raises(ValueError, match="invalid BREP"):
        cli_main(["validate", "candidate.FCStd", "reference.FCStd", "spec.json"])
    assert "usage:" not in capsys.readouterr().err


@pytest.mark.parametrize("entrypoint,args", [(cli_main, ["validate"]), (validator_main, [])])
def test_joint_scorer_help_uses_shared_defaults(monkeypatch, capsys, entrypoint, args):
    import freecad_validator.validator as validator_module

    monkeypatch.setattr(validator_module, "DEFAULT_SCORER_VERSION", "v1")
    monkeypatch.setattr(validator_module, "DEFAULT_V2_FAILURE_BUDGET", 7)
    with pytest.raises(SystemExit) as exc:
        entrypoint([*args, "--help"])
    assert exc.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert "default: v1" in help_text
    assert "spec failure budget 7" in help_text
    assert "independent OCCT bbox gate" in help_text


@pytest.mark.parametrize("value", ["0", "nan", "inf"])
def test_field_validation_errors_name_the_option_without_pydantic_payload(capsys, value):
    with pytest.raises(SystemExit) as exc:
        cli_main(["validate", "c.FCStd", "r.FCStd", "s.json", "--volume-far-rel-tol", value])
    assert exc.value.code == 2
    error = capsys.readouterr().err
    assert "volume_far_rel_tol:" in error
    assert "errors.pydantic.dev" not in error
    assert "input_value" not in error
