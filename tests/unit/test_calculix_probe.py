"""The needs_calculix probe must skip for a missing solver and fail for anything else.

The probe decides whether the FEM end-to-end test runs. If it treats every
preflight failure as "no CalculiX", a broken FCStd adapter reports as a skipped
test on every host — the suite would go green while the only test that executes
the adapter never runs. These tests pin the distinction.

FreeCAD-free: both the command lookup and the preflight are replaced, so this
runs in CI.
"""

from __future__ import annotations

import pytest
from _calculix import calculix_skip_reason

from freecad_validator import _freecad_loader
from freecad_validator.fem import step_interface

# Shape of the real chain: _runtime_preflight relays the adapter's subprocess
# log tail, which ends in the traceback that stopped it.
CCX_MISSING = (
    "FEM runtime preflight failed: adapter failed with exit 1: "
    "Traceback (most recent call last):\n"
    '  File "fcstd.py", line 96, in runtime_info\n'
    "RuntimeError: CalculiX executable was not found; install ccx or "
    "configure its path in FreeCAD FEM preferences"
)
CCX_UNUSABLE = (
    "FEM runtime preflight failed: adapter failed with exit 1: "
    "Traceback (most recent call last):\n"
    "RuntimeError: CalculiX executable failed its version preflight"
)
ADAPTER_BROKEN = (
    "FEM runtime preflight failed: adapter failed with exit 1: "
    "Traceback (most recent call last):\n"
    '  File "fcstd.py", line 41\n'
    "SyntaxError: invalid syntax"
)


def _preflight_raising(message: str):
    def _preflight(freecad_cmd, extract_dir, timeout_seconds):
        raise step_interface.RuntimeEnvironmentError(message)

    return _preflight


@pytest.fixture
def freecad_cmd_found(monkeypatch):
    monkeypatch.setattr(
        _freecad_loader, "resolve_freecad_command", lambda *args, **kwargs: "/bin/freecadcmd"
    )


def test_usable_solver_reports_no_reason(monkeypatch, freecad_cmd_found):
    monkeypatch.setattr(
        step_interface, "_runtime_preflight", lambda *args, **kwargs: {"calculix": "2.23"}
    )
    assert calculix_skip_reason() is None


@pytest.mark.parametrize("message", [CCX_MISSING, CCX_UNUSABLE])
def test_absent_solver_reports_a_skip_reason(monkeypatch, freecad_cmd_found, message):
    monkeypatch.setattr(step_interface, "_runtime_preflight", _preflight_raising(message))
    assert calculix_skip_reason() == "CalculiX is not usable on this host"


def test_missing_version_in_payload_reports_a_skip_reason(monkeypatch, freecad_cmd_found):
    # _runtime_preflight's own wording when the adapter ran but returned no version.
    monkeypatch.setattr(
        step_interface,
        "_runtime_preflight",
        _preflight_raising("FEM runtime preflight produced no CalculiX version"),
    )
    assert calculix_skip_reason() == "CalculiX is not usable on this host"


def test_broken_adapter_propagates(monkeypatch, freecad_cmd_found):
    """The regression this file exists for: a defect must not read as a skip."""
    monkeypatch.setattr(step_interface, "_runtime_preflight", _preflight_raising(ADAPTER_BROKEN))
    with pytest.raises(step_interface.RuntimeEnvironmentError, match="SyntaxError"):
        calculix_skip_reason()


def test_adapter_timeout_propagates(monkeypatch, freecad_cmd_found):
    monkeypatch.setattr(
        step_interface,
        "_runtime_preflight",
        _preflight_raising("FEM runtime preflight failed: adapter timed out after 30 seconds"),
    )
    with pytest.raises(step_interface.RuntimeEnvironmentError, match="timed out"):
        calculix_skip_reason()


def test_absent_freecad_command_reports_a_skip_reason(monkeypatch):
    def _raise(*args, **kwargs):
        raise FileNotFoundError("FreeCAD's command-line executable could not be located.")

    monkeypatch.setattr(_freecad_loader, "resolve_freecad_command", _raise)
    reason = calculix_skip_reason()
    assert reason is not None
    assert "command-line executable is unavailable" in reason
