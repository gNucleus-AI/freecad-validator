"""Resolve the FreeCAD Python binding without manual ``PYTHONPATH`` setup.

FreeCAD ships its Python module outside the standard ``site-packages``
tree — macOS Homebrew puts it under
``/Applications/FreeCAD.app/Contents/Resources/lib``, apt at
``/usr/lib/freecad-python3/lib``, etc. This helper:

  1. Tries ``import FreeCAD`` directly (fast path — works under
     conda-forge, or after the user has already exported
     ``PYTHONPATH``).
  2. On failure, walks a list of well-known install paths plus the
     ``FREECAD_LIB`` env override, prepends the first directory whose
     contents look like a FreeCAD lib to ``sys.path``, and retries.
  3. If still failing, raises :class:`ImportError` with
     platform-specific install + override guidance.

Set ``FREECAD_LIB`` explicitly when FreeCAD lives in a non-standard
location — that path is tried before any built-in candidate.
"""
from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path
from typing import Iterable


def _candidate_paths() -> Iterable[Path]:
    """Yield directories that *might* contain FreeCAD's Python binding,
    in priority order. The env-var override comes first; conda
    second; then platform-default install locations."""
    # Explicit user override wins. ``FREECAD_LIB`` may be a single
    # directory or an ``os.pathsep``-separated list (``:`` on Unix,
    # ``;`` on Windows) — same convention as ``PATH`` / ``PYTHONPATH``.
    env = os.environ.get("FREECAD_LIB")
    if env:
        for part in env.split(os.pathsep):
            part = part.strip()
            if part:
                yield Path(part)

    # conda-forge::freecad puts the binding directly under $CONDA_PREFIX/lib.
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        yield Path(conda_prefix) / "lib"

    # macOS — Homebrew cask
    yield Path("/Applications/FreeCAD.app/Contents/Resources/lib")
    # macOS — Homebrew bottle (both Apple Silicon and Intel prefixes)
    for cellar in (Path("/opt/homebrew/Cellar/freecad"),
                   Path("/usr/local/Cellar/freecad")):
        if cellar.is_dir():
            for v in sorted(cellar.glob("*/lib"), reverse=True):
                yield v

    # Linux distro packages
    for p in (
        "/usr/lib/freecad-python3/lib",
        "/usr/lib/freecad/lib",
        "/usr/lib64/freecad/lib",
        "/usr/local/lib/freecad/lib",
    ):
        yield Path(p)


def _looks_like_freecad_lib(d: Path) -> bool:
    if not d.is_dir():
        return False
    # The binding is one of FreeCAD.so (Linux/macOS) or FreeCAD.pyd
    # (Windows); pure-python FreeCAD.py is the loader stub.
    for pat in ("FreeCAD.so", "FreeCAD*.so", "FreeCAD.pyd", "FreeCAD.py"):
        if next(d.glob(pat), None) is not None:
            return True
    return False


def _install_hint() -> str:
    return textwrap.dedent("""\
        FreeCAD's Python module could not be imported.

        Install FreeCAD on your system:
          macOS  : brew install --cask freecad
          Ubuntu : sudo apt-get install freecad
          Fedora : sudo dnf install freecad
          conda  : conda install -c conda-forge freecad

        If FreeCAD lives in a non-standard location, set FREECAD_LIB
        to the directory that contains FreeCAD.so / FreeCAD.py:
          export FREECAD_LIB=/path/to/freecad/lib
        """)


def import_freecad():
    """Import and return the FreeCAD Python module.

    Auto-detects common install paths on macOS / Linux / conda before
    giving up. Subsequent calls reuse the cached module via Python's
    normal import machinery — no measurable cost beyond the first
    successful resolution.

    Raises
    ------
    ImportError
        FreeCAD couldn't be located. The error message includes
        platform-specific install hints and the ``FREECAD_LIB`` env
        override.
    """
    try:
        import FreeCAD  # type: ignore
        return FreeCAD
    except ImportError:
        pass

    tried: list[Path] = []
    for path in _candidate_paths():
        tried.append(path)
        if not _looks_like_freecad_lib(path):
            continue
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
        try:
            import FreeCAD  # type: ignore
            return FreeCAD
        except ImportError:
            # Bad / partial install at this path — keep looking.
            continue

    msg = _install_hint() + "\nPaths searched:\n  " + "\n  ".join(str(p) for p in tried)
    raise ImportError(msg)
