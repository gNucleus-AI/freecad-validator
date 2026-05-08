"""Pytest configuration.

The ``needs_freecad`` marker tags tests that depend on a real FreeCAD
installation. CI runners typically don't have FreeCAD, so those
tests are skipped unless ``-m needs_freecad`` is explicitly passed.

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


def _real_freecad_importable() -> bool:
    """True iff a real FreeCAD module loads (not our stub)."""
    try:
        import FreeCAD  # type: ignore
    except Exception:
        return False
    return getattr(FreeCAD, "__file__", None) is not None


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
    if _real_freecad_importable():
        return
    skip_marker = pytest.mark.skip(reason="FreeCAD is not importable on this host")
    for item in items:
        if "needs_freecad" in item.keywords:
            item.add_marker(skip_marker)
