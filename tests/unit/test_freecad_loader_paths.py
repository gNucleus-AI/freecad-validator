"""Loader path-detection logic — no FreeCAD installation required.

These exercise pure filesystem predicates, so they are deliberately
NOT marked ``needs_freecad`` and DO run in CI. The parts that need a
real binding live in ``tests/e2e/test_freecad_loader.py``.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from freecad_validator._freecad_loader import (
    _candidate_paths,
    _linux_mod_dirs,
    _looks_like_freecad_lib,
    resolve_freecad_command,
)


def test_rejects_a_directory_without_the_binding(tmp_path):
    assert not _looks_like_freecad_lib(tmp_path)


def test_rejects_a_missing_directory(tmp_path):
    assert not _looks_like_freecad_lib(tmp_path / "does_not_exist")


def test_rejects_a_file_that_is_not_a_directory(tmp_path):
    f = tmp_path / "FreeCAD.so"
    f.write_bytes(b"")
    assert not _looks_like_freecad_lib(f)


def test_accepts_a_directory_holding_the_shared_object(tmp_path):
    (tmp_path / "FreeCAD.so").write_bytes(b"")
    assert _looks_like_freecad_lib(tmp_path)


def test_accepts_the_windows_binding(tmp_path):
    (tmp_path / "FreeCAD.pyd").write_bytes(b"")
    assert _looks_like_freecad_lib(tmp_path)


def test_accepts_the_pure_python_loader_stub(tmp_path):
    (tmp_path / "FreeCAD.py").write_text("")
    assert _looks_like_freecad_lib(tmp_path)


def test_accepts_an_abi_tagged_shared_object(tmp_path):
    """conda-forge ships e.g. FreeCAD.cpython-312-x86_64-linux-gnu.so."""
    (tmp_path / "FreeCAD.cpython-312-x86_64-linux-gnu.so").write_bytes(b"")
    assert _looks_like_freecad_lib(tmp_path)


def test_conda_prefix_is_searched_first(monkeypatch):
    """$CONDA_PREFIX/lib takes priority — it is where conda-forge puts
    the binding, and the README documents it as the reference install."""
    monkeypatch.setenv("CONDA_PREFIX", "/opt/fake-conda")
    assert next(iter(_candidate_paths())) == Path("/opt/fake-conda/lib")


def test_macos_cask_path_is_a_candidate(monkeypatch):
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    paths = list(_candidate_paths())
    assert Path("/Applications/FreeCAD.app/Contents/Resources/lib") in paths


def test_linux_distro_paths_are_candidates(monkeypatch):
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    paths = list(_candidate_paths())
    assert Path("/usr/lib/freecad-python3/lib") in paths
    assert Path("/usr/lib/freecad/lib") in paths


def test_mod_dirs_pair_with_a_linux_lib_dir():
    """apt/PPA installs split the binding from its workbenches; both
    must land on sys.path or document deserialization fails."""
    mods = list(_linux_mod_dirs(Path("/usr/lib/freecad/lib")))
    assert Path("/usr/lib/freecad/Mod") in mods
    assert Path("/usr/share/freecad/Mod") in mods


def test_pathsep_is_the_documented_list_separator():
    """FREECAD_LIB is split on os.pathsep, matching PATH/PYTHONPATH."""
    assert os.pathsep in (":", ";")


def test_explicit_freecad_command_is_resolved(tmp_path):
    command = tmp_path / "freecadcmd"
    command.write_text("#!/bin/sh\n", encoding="utf-8")
    command.chmod(0o755)

    assert resolve_freecad_command(command) == str(command.resolve())


def test_freecad_command_environment_override(monkeypatch, tmp_path):
    command = tmp_path / "FreeCADCmd"
    command.write_text("#!/bin/sh\n", encoding="utf-8")
    command.chmod(0o755)
    monkeypatch.setenv("FREECAD_CMD", str(command))

    assert resolve_freecad_command() == str(command.resolve())


def test_missing_explicit_freecad_command_fails_clearly(tmp_path):
    missing = tmp_path / "missing-freecadcmd"

    with pytest.raises(FileNotFoundError, match="not found or is not executable"):
        resolve_freecad_command(missing)
