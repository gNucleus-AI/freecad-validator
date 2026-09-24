"""Exercise the standalone adapter's material reader with real FreeCAD solids."""

import json
import runpy
import sys
from pathlib import Path

import FreeCAD
import ObjectsFem
import Part


def check_counts(extract):
    doc = FreeCAD.newDocument("MaterialCounts")
    try:
        body = doc.addObject("Part::Feature", "Body")
        body.Shape = Part.makeCompound(
            [Part.makeBox(1, 2, 3, FreeCAD.Vector(x, 0, 0)) for x in (0, 4, 8)]
        )
        body.Placement.Base = FreeCAD.Vector(10, 20, 30)
        steel = ObjectsFem.makeMaterialSolid(doc, "Steel")
        steel.Material = {
            "Name": "Steel",
            "YoungsModulus": "200 GPa",
            "PoissonRatio": "0.3",
            "Density": "7900 kg/m^3",
        }
        steel.References = [(body, ("Solid1", "Solid2", "Solid1"))]
        aluminum = ObjectsFem.makeMaterialSolid(doc, "Aluminum")
        aluminum.Material = {
            "Name": "Aluminum",
            "YoungsModulus": "70 GPa",
            "PoissonRatio": "0.35",
            "Density": "2700 kg/m^3",
        }
        aluminum.References = [(body, ("Solid3",))]
        doc.recompute()
        reference = extract([steel, aluminum], 3)
        assert {m["E_MPa"]: m["body_count"] for m in reference} == {200000.0: 2, 70000.0: 1}
        assert extract([aluminum, steel], 3) == reference

        steel.References = []
        assert extract([steel, aluminum], 3) == reference
        same_default = ObjectsFem.makeMaterialSolid(doc, "SameDefaultSteel")
        same_default.Material = steel.Material
        same_default.References = []
        assert extract([same_default, steel, aluminum], 3) == reference

        second = ObjectsFem.makeMaterialSolid(doc, "MoreSteel")
        second.Material = {**steel.Material, "Name": "Renamed", "YoungsModulus": "200000 MPa"}
        second.References = [(body, ("Solid2",))]
        steel.References = [(body, ("Solid1",))]
        split = extract([second, aluminum, steel], 3)
        assert {m["E_MPa"]: m["body_count"] for m in split} == {200000.0: 2, 70000.0: 1}

        for references in (
            [(body, ("",)), (body, ("Solid1", "Solid2"))],
            [(body, ())],
            [],
            [(body, ("Solid1",))],
        ):
            steel.References = references
            assert extract([steel], 3)[0]["body_count"] == 3

        # Count-only matching intentionally does not establish body correspondence.
        steel.References = [(body, ("Solid2", "Solid3"))]
        aluminum.References = [(body, ("Solid1",))]
        assert extract([steel, aluminum], 3) == reference
        steel.References = [(body, ("Solid3",))]
        aluminum.References = [(body, ("Solid1", "Solid2"))]
        swapped = extract([steel, aluminum], 3)
        assert {m["E_MPa"]: m["body_count"] for m in swapped} == {200000.0: 1, 70000.0: 2}

        steel.References = []
        aluminum.References = []
        try:
            extract([steel, aluminum], 3)
        except ValueError as exc:
            assert "Multiple materials" in str(exc)
        else:
            raise AssertionError("Conflicting default materials were accepted")
        return {"passed": True, "reference_counts": reference}
    finally:
        FreeCAD.closeDocument(doc.Name)


def main():
    output = Path(next(arg for arg in sys.argv if arg.endswith(".json")))
    adapter = next(arg for arg in sys.argv if arg.endswith("/fcstd.py"))
    # The embedded interpreter must not import the installed host package.
    sys.argv = [adapter, "import-check"]
    namespace = runpy.run_path(adapter)
    result = check_counts(namespace["extract_material_body_counts"])
    output.write_text(json.dumps(result), encoding="utf-8")


main()
