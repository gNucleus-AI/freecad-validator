"""Existing geometry fixture writer, kept outside the read-only grader."""

from pathlib import Path

import FreeCAD


def export_geometry(shape, path: Path) -> None:
    """Write one detached feature for unambiguous geometry extraction."""
    doc = FreeCAD.newDocument("PreProcessGeometry")
    preferences = FreeCAD.ParamGet("User parameter:BaseApp/Preferences/Document")
    previous_binary_brep = preferences.GetBool("SaveBinaryBrep", False)
    try:
        obj = doc.addObject("Part::Feature", "Geometry")
        obj.Shape = shape
        doc.recompute()
        # OCCT's legacy ASCII BRep writer can lose valid partitioned surfaces.
        # Scratch geometry must survive reopening before it can be scored.
        preferences.SetBool("SaveBinaryBrep", True)
        doc.saveAs(str(path))
    finally:
        preferences.SetBool("SaveBinaryBrep", previous_binary_brep)
        FreeCAD.closeDocument(doc.Name)
