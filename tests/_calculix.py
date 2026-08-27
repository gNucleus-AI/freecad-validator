"""Whether the ``needs_calculix`` tests can run on this host.

Lives beside conftest rather than inside it so tests can import it by an
unambiguous name — ``conftest`` resolves to whichever conftest.py pytest
imported first, which in a full-tree run is ``tests/e2e/conftest.py``. Same
pattern as ``tests/e2e/_geometry.py``.
"""

from __future__ import annotations

# The only two preflight failures that mean "this host has no CalculiX":
# `runtime_info(require_calculix=True)` in the FCStd adapter raises the first
# when discovery finds no ccx and the second when the binary will not answer
# `-v`, and `_runtime_preflight` adds the third if the payload comes back
# without a version. The adapter's traceback reaches us because `_run_adapter`
# puts the subprocess log tail into its ExtractionError.
#
# Every other preflight failure — a broken adapter, a timeout, unreadable JSON
# — is a real defect and must reach the test run. Muffling those into a skip
# would silently drop the one test that exercises the FEM adapter end to end,
# which is the failure this skip machinery exists to prevent, inverted.
NO_CALCULIX_SIGNS = (
    "CalculiX executable was not found",
    "CalculiX executable failed its version preflight",
    "produced no CalculiX version",
)


def calculix_skip_reason() -> str | None:
    """Why ``needs_calculix`` tests cannot run here, or None if they can.

    Delegates to the package's own preflight instead of re-deriving ccx
    discovery. That check runs inside FreeCAD's embedded Python, which is the
    only interpreter that sees a bundled solver (the macOS .app ships ``ccx``
    beside ``freecadcmd``, invisible to ``shutil.which`` out here) and
    FreeCAD's ``ccxBinaryPath`` preference, and it verifies the binary answers
    ``-v`` — so a present-but-broken ccx reports unusable.

    Reaching into `_runtime_preflight` couples this to a private helper and to
    the message text it relays; that is deliberate. Classifying by message is
    the only way to tell "no solver installed" from "the adapter is broken",
    and the failure mode is safe: an unrecognized message propagates.
    """
    # Imported inside the function, not at module scope: conftest imports this
    # module before it installs the FreeCAD stub, and freecad_validator's
    # package import chain reaches `import FreeCAD`.
    import tempfile

    from freecad_validator._freecad_loader import resolve_freecad_command
    from freecad_validator.fem.step_interface import (
        RuntimeEnvironmentError,
        _runtime_preflight,
    )

    try:
        freecad_cmd = resolve_freecad_command()
    except FileNotFoundError as exc:
        # No freecadcmd is an environment gap of the same kind as no binding.
        return f"FreeCAD's command-line executable is unavailable: {exc}"

    try:
        with tempfile.TemporaryDirectory() as extract_dir:
            _runtime_preflight(freecad_cmd, extract_dir, 30.0)
    except RuntimeEnvironmentError as exc:
        if any(sign in str(exc) for sign in NO_CALCULIX_SIGNS):
            return "CalculiX is not usable on this host"
        raise
    return None
