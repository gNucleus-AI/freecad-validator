"""The FreeCAD binding resolver, against a real installation.

``_freecad_loader`` is the package's promise that ``pip install`` plus a
system FreeCAD is enough — no PYTHONPATH wrangling. That can only be
checked against a real install, so it lives here.

The FREECAD_LIB tests run in a CLEAN SUBPROCESS. In-process they would
be meaningless: ``import_freecad`` returns at its ``import FreeCAD``
fast path as soon as anything has imported FreeCAD (conftest does, at
collection time), so the override branch would never execute and a
broken one would still pass.

Pure path-detection logic has no FreeCAD dependency and lives in
``tests/unit/test_freecad_loader_paths.py`` so CI runs it.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from freecad_validator._freecad_loader import import_freecad

pytestmark = pytest.mark.needs_freecad

_RESOLVE = (
    "from freecad_validator._freecad_loader import import_freecad;"
    "print(import_freecad().__file__)"
)


def _child(code: str, **env_overrides: str) -> subprocess.CompletedProcess:
    """Run `code` in a fresh interpreter with a controlled environment."""
    env = {k: v for k, v in os.environ.items() if k != "FREECAD_LIB"}
    env.update(env_overrides)
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300,
    )


@pytest.fixture(scope="module")
def real_lib_dir() -> Path:
    """The directory the loader actually resolves on this host."""
    return Path(import_freecad().__file__).parent


@pytest.fixture(scope="module")
def builtin_resolution_works() -> bool:
    """True when a clean interpreter resolves FreeCAD with NO FREECAD_LIB
    set. A custom install that legitimately REQUIRES the documented
    override is a supported configuration, so tests asserting built-in
    discovery must skip there rather than fail."""
    return _child(_RESOLVE).returncode == 0


@pytest.fixture(scope="module")
def bare_import_works() -> bool:
    """True when a clean interpreter can ``import FreeCAD`` unaided (the
    conda layout). There the fast path wins and FREECAD_LIB is never
    consulted, so override tests cannot observe anything."""
    return _child("import FreeCAD").returncode == 0


def test_import_freecad_returns_a_usable_module():
    fc = import_freecad()
    assert hasattr(fc, "Version")
    assert getattr(fc, "__file__", None)


def test_version_is_the_pinned_release():
    """The README pins FreeCAD 1.1.0, so the suite must be running on it.
    A digits-only check would pass on 0.21.2 or 1.2.0 and quietly
    validate the package against a version it does not claim to support.

    Version() yields STRINGS, not ints — the README's verification
    snippet and any caller comparing them depend on that shape.
    """
    version = import_freecad().Version()
    assert len(version) >= 3
    assert version[:3] == ["1", "1", "0"], (
        f"expected FreeCAD 1.1.0 (README prerequisite), got {'.'.join(version[:3])}"
    )


def test_repeated_calls_return_the_same_module():
    assert import_freecad() is import_freecad()


def test_can_open_and_close_a_document(box_10x5x3):
    """The resolved module is wired up enough to deserialize a document —
    on split installs that needs the Mod/workbench dirs too, not just
    the directory holding FreeCAD.so."""
    fc = import_freecad()
    doc = fc.openDocument(str(box_10x5x3))
    try:
        bodies = [o for o in doc.Objects if o.TypeId == "PartDesign::Body"]
        assert len(bodies) == 1
        assert bodies[0].Shape.Volume == pytest.approx(150.0)
    finally:
        fc.closeDocument(doc.Name)


def test_resolves_in_a_clean_interpreter(real_lib_dir, builtin_resolution_works):
    """Baseline: a fresh process with no FREECAD_LIB still finds FreeCAD
    via the built-in candidates. Skipped on installs that require the
    override — those are supported, just not auto-discoverable."""
    if not builtin_resolution_works:
        pytest.skip("FreeCAD is not on a built-in candidate path; FREECAD_LIB is required here")
    proc = _child(_RESOLVE)
    assert proc.returncode == 0, proc.stderr
    assert Path(proc.stdout.strip()).parent == real_lib_dir


def test_freecad_lib_override_is_used(tmp_path, real_lib_dir, bare_import_works):
    """Point FREECAD_LIB at a SYMLINK to the real lib, in a directory the
    loader would never search on its own. Resolving to a path under that
    symlink proves the override branch ran — passing the real directory
    would be indistinguishable from the built-in candidate."""
    if bare_import_works:
        pytest.skip("bare `import FreeCAD` succeeds here; the override branch is unreachable")
    link = tmp_path / "freecad_lib_link"
    link.symlink_to(real_lib_dir, target_is_directory=True)
    proc = _child(_RESOLVE, FREECAD_LIB=str(link))
    assert proc.returncode == 0, proc.stderr
    assert Path(proc.stdout.strip()).parent == link, (
        f"resolved {proc.stdout.strip()!r}; the override was not used"
    )


def test_freecad_lib_accepts_a_path_separated_list(tmp_path, real_lib_dir, bare_import_works):
    """Linux installs pass lib + Mod dirs in one variable, PATH-style;
    a junk entry alongside the real one must not break resolution."""
    if bare_import_works:
        pytest.skip("bare `import FreeCAD` succeeds here; the override branch is unreachable")
    link = tmp_path / "freecad_lib_list_link"
    link.symlink_to(real_lib_dir, target_is_directory=True)
    value = os.pathsep.join(["/nonexistent/freecad", str(link)])
    proc = _child(_RESOLVE, FREECAD_LIB=value)
    assert proc.returncode == 0, proc.stderr
    assert Path(proc.stdout.strip()).parent == link


def test_bogus_freecad_lib_falls_back_to_platform_defaults(real_lib_dir, builtin_resolution_works):
    """A wrong override must not defeat a working default — the loader
    falls through to its built-in candidates. Only meaningful where a
    built-in candidate exists to fall back TO."""
    if not builtin_resolution_works:
        pytest.skip("no built-in candidate on this host; there is nothing to fall back to")
    proc = _child(_RESOLVE, FREECAD_LIB="/nonexistent/path/to/freecad")
    assert proc.returncode == 0, proc.stderr
    assert Path(proc.stdout.strip()).parent == real_lib_dir


def test_import_error_names_the_override_when_nothing_resolves(tmp_path, bare_import_works):
    """With every candidate hidden, the failure must be an ImportError
    whose message tells the user about FREECAD_LIB."""
    if bare_import_works:
        pytest.skip("bare `import FreeCAD` succeeds here; resolution cannot be forced to fail")
    code = (
        "import freecad_validator._freecad_loader as L;"
        "L._candidate_paths = lambda: iter(());"
        "\ntry:\n"
        "    L.import_freecad()\n"
        "    print('RESOLVED')\n"
        "except ImportError as e:\n"
        "    print('FREECAD_LIB' in str(e))\n"
    )
    proc = _child(code, FREECAD_LIB=str(tmp_path))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "True", proc.stdout
