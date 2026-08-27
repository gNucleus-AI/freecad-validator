"""Shared fixtures for the end-to-end suite.

Every fixture BUILDS its geometry with FreeCAD at test time and writes
it under pytest's ``tmp_path``. Nothing is committed: a stored .FCStd
would pin the suite to one FreeCAD/OCCT build, and the same FreeCAD
version ships different kernels on different channels (the official
1.1.0 binaries carry OCCT 7.8.1, conda-forge's 1.1.0 carries 7.9.3).

The whole package is marked ``needs_freecad`` — CI deselects it.
Builders live in ``_geometry.py`` so test modules can import them.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _geometry import make_box, make_cylinder, write_spec

pytestmark = pytest.mark.needs_freecad


@pytest.fixture(scope="session")
def box_10x5x3(tmp_path_factory) -> Path:
    return make_box(tmp_path_factory.mktemp("box") / "box_10x5x3.FCStd", 10, 5, 3)


@pytest.fixture(scope="session")
def box_20x5x3(tmp_path_factory) -> Path:
    return make_box(tmp_path_factory.mktemp("box2") / "box_20x5x3.FCStd", 20, 5, 3)


@pytest.fixture(scope="session")
def cylinder_r4_h12(tmp_path_factory) -> Path:
    return make_cylinder(tmp_path_factory.mktemp("cyl") / "cyl_r4_h12.FCStd", 4, 12)


@pytest.fixture(scope="session")
def box_spec(tmp_path_factory) -> Path:
    return write_spec(
        tmp_path_factory.mktemp("spec") / "box_spec.json",
        length="10 mm",
        width="5 mm",
        height="3 mm",
    )
