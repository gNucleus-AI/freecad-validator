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
import shutil
import sys
import textwrap
from collections.abc import Iterable
from pathlib import Path


def _resolve_executable(candidate: str | os.PathLike[str]) -> str | None:
    """Return an absolute executable path for a path or command name."""
    value = os.fspath(candidate)
    resolved = shutil.which(value)
    if resolved:
        return str(Path(resolved).resolve())
    path = Path(value).expanduser()
    if path.is_file() and os.access(path, os.X_OK):
        return str(path.resolve())
    return None


def resolve_freecad_command(
    explicit: str | os.PathLike[str] | None = None,
) -> str:
    """Locate the FreeCAD 1.1 command-line executable.

    ``explicit`` wins, followed by ``FREECAD_CMD``, the active conda
    environment, ``PATH``, and common macOS/Windows application locations.
    The returned value is always an absolute executable path.
    """
    requested = explicit or os.environ.get("FREECAD_CMD")
    if requested:
        resolved = _resolve_executable(requested)
        if resolved:
            return resolved
        raise FileNotFoundError(
            f"FreeCAD command {os.fspath(requested)!r} was not found or is not executable"
        )

    candidates: list[str | os.PathLike[str]] = []
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.extend(
            Path(conda_prefix) / "bin" / name for name in ("freecadcmd", "FreeCADCmd")
        )
    candidates.extend(("freecadcmd", "FreeCADCmd"))
    candidates.extend(
        (
            "/Applications/FreeCAD.app/Contents/Resources/bin/freecadcmd",
            "/Applications/FreeCAD.app/Contents/MacOS/FreeCADCmd",
        )
    )
    program_files = os.environ.get("ProgramFiles")
    if program_files:
        candidates.append(Path(program_files) / "FreeCAD 1.1" / "bin" / "FreeCADCmd.exe")

    for candidate in candidates:
        resolved = _resolve_executable(candidate)
        if resolved:
            return resolved
    raise FileNotFoundError(
        "FreeCAD's command-line executable could not be located. Install FreeCAD 1.1 "
        "or set FREECAD_CMD to freecadcmd/FreeCADCmd."
    )


def _candidate_paths() -> Iterable[Path]:
    """Yield directories that *might* contain FreeCAD's Python binding,
    in priority order: conda first, then platform-default install
    locations. The ``FREECAD_LIB`` env override is handled separately
    in :func:`import_freecad` so that all of its entries are added to
    ``sys.path`` even when they are workbench/Mod directories rather
    than the binding lib itself."""
    # conda-forge::freecad puts the binding directly under $CONDA_PREFIX/lib.
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        yield Path(conda_prefix) / "lib"

    # macOS — Homebrew cask
    yield Path("/Applications/FreeCAD.app/Contents/Resources/lib")
    # macOS — Homebrew bottle (both Apple Silicon and Intel prefixes)
    for cellar in (Path("/opt/homebrew/Cellar/freecad"), Path("/usr/local/Cellar/freecad")):
        if cellar.is_dir():
            yield from sorted(cellar.glob("*/lib"), reverse=True)

    # Linux distro packages
    for p in (
        "/usr/lib/freecad-python3/lib",
        "/usr/lib/freecad/lib",
        "/usr/lib64/freecad/lib",
        "/usr/local/lib/freecad/lib",
    ):
        yield Path(p)


def _linux_mod_dirs(lib_dir: Path) -> Iterable[Path]:
    """Yield workbench/Mod directories that pair with a Linux ``lib``
    candidate. FreeCAD on apt/PPA installs splits the binding (under
    ``/usr/lib/freecad*/lib``) from its Python workbenches (under
    ``/usr/lib/freecad*/Mod`` and ``/usr/share/freecad*/Mod``); the
    workbenches need to be on ``sys.path`` for ``FreeCAD.open()`` to
    deserialize ``Part``/``Sketcher``/``PartDesign`` objects. The
    caller filters by :meth:`Path.is_dir`."""
    # Mod sits next to ``lib`` under the package root, e.g.
    # /usr/lib/freecad/Mod (often a symlink into /usr/share/freecad/Mod).
    yield lib_dir.parent / "Mod"
    # Canonical workbench tree on apt/PPA, in case the symlink above
    # is absent (custom builds, relocated installs).
    yield Path("/usr/share/freecad/Mod")


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

    # Explicit user override. ``FREECAD_LIB`` may be a single directory
    # or an ``os.pathsep``-separated list (``:`` on Unix, ``;`` on
    # Windows) — same convention as ``PATH`` / ``PYTHONPATH``. We add
    # *every* listed directory to ``sys.path`` (even ones that don't
    # contain ``FreeCAD.so``) because Linux distro installs require
    # the lib *and* Mod dirs to be importable, e.g.
    # ``FREECAD_LIB=/usr/lib/freecad/lib:/usr/lib/freecad/lib/Mod:/usr/share/freecad/Mod``.
    env = os.environ.get("FREECAD_LIB")
    if env:
        for part in env.split(os.pathsep):
            part = part.strip()
            if not part:
                continue
            p = Path(part)
            tried.append(p)
            if p.is_dir() and str(p) not in sys.path:
                sys.path.insert(0, str(p))
        try:
            import FreeCAD  # type: ignore

            return FreeCAD
        except ImportError:
            # User-supplied paths didn't resolve; fall through to the
            # built-in candidates.
            pass

    for path in _candidate_paths():
        tried.append(path)
        if not _looks_like_freecad_lib(path):
            continue
        # Add the lib dir + any sibling Mod dirs so FreeCAD's own
        # workbench imports succeed when it deserializes a document.
        for d in (path, *_linux_mod_dirs(path)):
            if d.is_dir() and str(d) not in sys.path:
                sys.path.insert(0, str(d))
        try:
            import FreeCAD  # type: ignore

            return FreeCAD
        except ImportError:
            # Bad / partial install at this path — keep looking.
            continue

    msg = _install_hint() + "\nPaths searched:\n  " + "\n  ".join(str(p) for p in tried)
    raise ImportError(msg)
