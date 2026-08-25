"""Opt-in end-to-end FEM replay using a generated, non-sensitive fixture."""

import shutil
import subprocess
from pathlib import Path

import pytest

from freecad_validator._freecad_loader import resolve_freecad_command
from freecad_validator.fem import FEMValidator


@pytest.mark.needs_freecad
@pytest.mark.needs_calculix
def test_generated_cantilever_replays_with_calculix(tmp_path):
    freecad_cmd = resolve_freecad_command()
    step_path = tmp_path / "cantilever.step"
    reference_path = tmp_path / "reference.FCStd"
    candidate_path = tmp_path / "candidate.FCStd"
    generator = Path(__file__).with_name("generate_fem_fixture.py")

    subprocess.run(
        [freecad_cmd, str(generator), str(step_path), str(reference_path)],
        check=True,
        timeout=180,
    )
    shutil.copy2(reference_path, candidate_path)

    report = FEMValidator(freecad_cmd=freecad_cmd, timeout_seconds=180).validate(
        step_path=str(step_path),
        reference_fcstd=str(reference_path),
        candidate_fcstd=str(candidate_path),
    )

    assert report.overall_score >= 85.0
    assert report.gates_triggered == []
    assert report.runtime_provenance["freecad"].startswith("1.1")
    assert report.runtime_provenance["calculix"]
