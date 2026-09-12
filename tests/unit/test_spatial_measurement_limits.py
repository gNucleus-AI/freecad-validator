"""Incomplete measurements are distinguishable from missing candidate geometry."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from freecad_validator.consistency.geometry_bindings import (
    GeometryBinding,
    GeometryBindingSpec,
    apply_bindings,
    evaluate_binding,
)
from freecad_validator.consistency.report import ConsistencyReport
from freecad_validator.measurement import spatial
from freecad_validator.spec.parser import parse_spec


@pytest.mark.parametrize(
    "vector", [(0, 0, 0), (float("nan"), 0, 1), (float("inf"), 0, 1), (1, 2), None, ("bad", 0, 1)]
)
def test_invalid_direction_has_a_typed_diagnostic(vector):
    with pytest.raises(spatial.SpatialMeasurementError, match="finite, nonzero 3D vector"):
        spatial._direction(vector)


@pytest.mark.parametrize("magnitude", [1e-300, 1.0, 1e300])
def test_finite_direction_normalizes_without_overflow_or_underflow(magnitude):
    assert spatial._direction((-3 * magnitude, -4 * magnitude, 0)) == pytest.approx((0.6, 0.8, 0))


def test_wall_pair_timeout_terminates_and_cleans_up_process(tmp_path, monkeypatch):
    pid_file = tmp_path / "worker.pid"
    monkeypatch.setattr(spatial, "_PLANE_PAIR_TIMEOUT_SECONDS", 1.0)
    monkeypatch.setattr(
        spatial, "import_freecad", lambda: SimpleNamespace(__file__=str(tmp_path / "FreeCAD.so"))
    )
    original_run = subprocess.run
    directories = []

    def blocked_worker(command, **kwargs):
        directories.append(Path(command[1]).parent)
        # Exercise the real process deadline while simulating an unresponsive native call.
        return original_run(
            [
                sys.executable,
                "-c",
                "import os, time\nfrom pathlib import Path\n"
                f"Path({str(pid_file)!r}).write_text(str(os.getpid()))\ntime.sleep(30)\n",
            ],
            **kwargs,
        )

    monkeypatch.setattr(spatial.subprocess, "run", blocked_worker)
    started = time.monotonic()
    with pytest.raises(spatial._PlanePairTimeout, match="exceeded 1 seconds"):
        spatial._run_plane_pairs(SimpleNamespace(exportBrepToString=lambda: "fixture"))
    assert time.monotonic() - started < 5
    assert all(not directory.exists() for directory in directories)
    if os.name == "posix":
        with pytest.raises(ProcessLookupError):
            os.kill(int(pid_file.read_text()), 0)


@pytest.mark.parametrize("embedded", [False, True])
def test_worker_failure_is_not_a_partial_measurement(tmp_path, monkeypatch, embedded):
    monkeypatch.setattr(
        spatial,
        "import_freecad",
        lambda: (
            SimpleNamespace()
            if embedded
            else SimpleNamespace(__file__=str(tmp_path / "FreeCAD.so"))
        ),
    )

    def failed_worker(command, **kwargs):
        (Path(command[1]).parent / "result.json").write_text(
            json.dumps({"error": "Native wall intersection failed"})
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(spatial.subprocess, "run", failed_worker)
    with pytest.raises(spatial.SpatialMeasurementError, match="Native wall intersection failed"):
        spatial._run_plane_pairs(SimpleNamespace(exportBrepToString=lambda: "fixture"))


@pytest.mark.parametrize("reason", ["disabled", "timed out"])
def test_unavailable_wall_pairs_cannot_be_scored_as_missing_features(reason):
    bank = spatial.SpatialBank(
        datum=spatial.SpatialDatum(
            center=(0, 0, 0), diagonal=10, frames=[((1, 0, 0), (0, 1, 0), (0, 0, 1))]
        ),
        limitations=[f"{spatial.PLANE_PAIR_UNAVAILABLE}: {reason}"],
    )
    binding = GeometryBinding(
        mode="geometry",
        reason="Corresponding walls",
        quantity="separation",
        witnesses=[
            dict(
                kind="plane_pair", position=(0, 0, 0), direction=(1, 0, 0), scale=10, region="void"
            )
        ],
    )
    with pytest.raises(spatial.SpatialMeasurementError, match=reason):
        evaluate_binding(binding, 4, bank, np.eye(3), np.zeros(3), tol_scalar=0.01, tol_pos=0.01)
    datum = bank.datum.model_copy(deep=True)
    datum.landmarks.append(
        spatial.SpatialLocation(kind="plane", position=(0, 0, 0), direction=(1, 0, 0), scale=10)
    )
    # Missing landmarks must not turn an incomplete extraction into not_found either.
    with pytest.raises(spatial.SpatialMeasurementError, match=reason):
        apply_bindings(
            ConsistencyReport(spec_name="fixture", fcstd_path="candidate"),
            GeometryBindingSpec(version=1, datum=datum, parameters={"slot_width": binding}),
            bank,
            parse_spec({"key_parameters": "slot_width = 4 mm"}),
            tol_scalar=0.01,
            tol_pos=0.01,
        )


def test_unavailable_wall_pairs_do_not_disable_completed_measurements():
    feature = spatial.SpatialFeature(
        kind="cylinder",
        position=(0, 0, 5),
        direction=(0, 0, 1),
        scale=10,
        convex=False,
        bounds_min=(-5, -5, 0),
        bounds_max=(5, 5, 10),
        values={"radius": 5, "diameter": 10},
    )
    bank = spatial.SpatialBank(
        datum=spatial.SpatialDatum(
            center=(0, 0, 5), diagonal=10, frames=[((1, 0, 0), (0, 1, 0), (0, 0, 1))]
        ),
        features=[feature],
        limitations=[f"{spatial.PLANE_PAIR_UNAVAILABLE}: disabled"],
    )
    binding = GeometryBinding(
        mode="geometry",
        reason="Bore",
        quantity="diameter",
        witnesses=[spatial.location(feature).model_dump()],
    )
    result = evaluate_binding(
        binding, 10, bank, np.eye(3), np.zeros(3), tol_scalar=0.01, tol_pos=0.01
    )
    assert result[0] == "consistent" and result[1] == [10]
