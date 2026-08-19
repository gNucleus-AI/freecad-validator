"""Geometry builders for the end-to-end suite.

Kept out of ``conftest.py`` so test modules can import them directly —
pytest fixtures cannot be imported, and conftest is not an importable
module. Every builder writes a document the caller owns and closes it.
"""
from __future__ import annotations

import json
from pathlib import Path


def freecad():
    """The real FreeCAD module, resolved through the package loader."""
    from freecad_validator._freecad_loader import import_freecad

    return import_freecad()


def _save(doc, path: Path) -> Path:
    doc.recompute()
    doc.saveAs(str(path))
    return path


def make_box(path: Path, length: float, width: float, height: float) -> Path:
    """A single-solid PartDesign box.

    The validator requires exactly one solid inside a PartDesign feature
    tree, so this builds Body + AdditiveBox rather than a bare
    ``Part::Box`` (which scores 0 with a "no PartDesign::Body found"
    reason — see test_validate_end_to_end.py).
    """
    fc = freecad()
    doc = fc.newDocument(path.stem)
    try:
        body = doc.addObject("PartDesign::Body", "Body")
        box = body.newObject("PartDesign::AdditiveBox", "Box")
        box.Length, box.Width, box.Height = length, width, height
        return _save(doc, path)
    finally:
        fc.closeDocument(doc.Name)


def make_cylinder(path: Path, radius: float, height: float) -> Path:
    """A single-solid PartDesign cylinder (exercises conic-surface paths)."""
    fc = freecad()
    doc = fc.newDocument(path.stem)
    try:
        body = doc.addObject("PartDesign::Body", "Body")
        cyl = body.newObject("PartDesign::AdditiveCylinder", "Cylinder")
        cyl.Radius, cyl.Height = radius, height
        return _save(doc, path)
    finally:
        fc.closeDocument(doc.Name)


def make_plain_part_box(path: Path, length: float, width: float, height: float) -> Path:
    """A ``Part::Box`` with NO PartDesign Body — the shape the validator
    is documented to reject."""
    fc = freecad()
    doc = fc.newDocument(path.stem)
    try:
        box = doc.addObject("Part::Box", "Box")
        box.Length, box.Width, box.Height = length, width, height
        return _save(doc, path)
    finally:
        fc.closeDocument(doc.Name)


def write_spec(path: Path, **key_parameters: str) -> Path:
    """A minimal spec dict in the shape ``parse_spec`` expects."""
    lines = "\n".join(f"- {k} = {v}" for k, v in key_parameters.items())
    path.write_text(json.dumps({
        "name": "test part",
        "description": "A part built by the end-to-end fixtures.",
        "key_parameters": lines,
    }))
    return path
