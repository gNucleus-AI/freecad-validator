"""Pytest configuration.

The ``needs_freecad`` marker tags tests that depend on a real FreeCAD
installation — the end-to-end suite under ``tests/e2e/``. CI runs
``pytest -m "not needs_freecad"`` and never loads FreeCAD; run them
locally against a real install with

    pytest -m needs_freecad

On a host without FreeCAD they skip rather than fail.

``needs_calculix`` tags the subset that also runs a real CalculiX
solve. It skips the same way: a host can carry FreeCAD without a
usable ``ccx``, and those tests would otherwise fail the local
pre-release run for a missing solver rather than a real defect.

Additionally: a lightweight stub for the ``FreeCAD`` module is
installed into ``sys.modules`` *before* test collection when the real
module isn't available. This lets the import-only smoke tests
(``tests/unit/test_imports.py``) verify the package's pure-Python
surface even on hosts without FreeCAD — e.g. GitHub Actions CI.
"""

from __future__ import annotations

import sys
import types

import pytest
from _calculix import calculix_skip_reason


def _real_freecad_importable() -> bool:
    """True iff a real FreeCAD module loads (not our stub).

    Resolves through the package's own loader rather than a bare
    ``import FreeCAD``. A bare import ignores ``FREECAD_LIB`` and the
    platform install paths, so on macOS (.dmg) and apt installs it
    reports "no FreeCAD" even when FreeCAD is installed and the
    library itself loads it fine — which would silently skip every
    ``needs_freecad`` test and run the rest against the stub below.
    """
    try:
        from freecad_validator._freecad_loader import import_freecad

        freecad = import_freecad()
    except Exception:
        return False
    return getattr(freecad, "__file__", None) is not None


# Install a no-op stub at collection time so module-level
# ``import FreeCAD`` in package code resolves. Tests that actually
# need FreeCAD's runtime behavior must be marked ``needs_freecad``
# and are skipped below when the real module isn't available.
if not _real_freecad_importable():
    sys.modules.setdefault("FreeCAD", types.ModuleType("FreeCAD"))


@pytest.fixture(scope="session")
def freecad_available() -> bool:
    return _real_freecad_importable()


def pytest_collection_modifyitems(config, items):
    if not _real_freecad_importable():
        # No binding means no solver either, so both markers go.
        skip_marker = pytest.mark.skip(reason="FreeCAD is not importable on this host")
        for item in items:
            if {"needs_freecad", "needs_calculix"} & item.keywords.keys():
                item.add_marker(skip_marker)
        return

    # The probe costs a FreeCAD subprocess, so only pay for it when a test
    # in this run actually needs the solver.
    calculix_items = [item for item in items if "needs_calculix" in item.keywords]
    if not calculix_items:
        return
    reason = calculix_skip_reason()
    if reason is None:
        return
    skip_marker = pytest.mark.skip(reason=reason)
    for item in calculix_items:
        item.add_marker(skip_marker)
