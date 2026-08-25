# Read a STEP solid and emit its geometry (volume, bbox, characteristic length).
# Run under FreeCAD:  freecadcmd step.py <input.step> <output.json>
import json
import os
import sys

import Part  # noqa: F401  (FreeCAD provides this; only available inside freecadcmd)

args = [a for a in sys.argv if a.lower().endswith((".step", ".stp", ".json"))]
step = next(a for a in args if a.lower().endswith((".step", ".stp")))
outs = [a for a in args if a.lower().endswith(".json")]
out = outs[0] if outs else os.path.splitext(step)[0] + ".geom.json"

shape = Part.Shape()
shape.read(step)
bb = shape.BoundBox
solids = list(shape.Solids)
regions = sorted(
    (
        {
            "volume_mm3": solid.Volume,
            "surface_area_mm2": solid.Area,
            "num_faces": len(solid.Faces),
            "num_edges": len(solid.Edges),
            "num_shells": len(solid.Shells),
        }
        for solid in solids
    ),
    key=lambda region: (
        region["volume_mm3"],
        region["surface_area_mm2"],
        region["num_faces"],
        region["num_edges"],
    ),
)
geom = {
    "source_step": os.path.basename(step),
    "volume_mm3": shape.Volume,
    "surface_area_mm2": shape.Area,
    "bbox_mm": [bb.XLength, bb.YLength, bb.ZLength],
    "characteristic_length_mm": bb.DiagonalLength,
    "faces": len(shape.Faces),
    "solids": len(shape.Solids),
    "num_solids": len(solids),
    "num_compsolids": len(shape.CompSolids),
    "shape_types": [str(shape.ShapeType)],
    "num_faces": sum(region["num_faces"] for region in regions),
    "num_edges": sum(region["num_edges"] for region in regions),
    "regions": regions,
}
with open(out, "w", encoding="utf-8") as fh:
    json.dump(geom, fh, indent=2)
print(f"[step_geom] wrote {out}  volume={shape.Volume:.0f} mm^3  diag={bb.DiagonalLength:.1f} mm")
