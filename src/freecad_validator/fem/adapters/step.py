# Read a STEP solid and emit its geometry (volume, bbox, characteristic length).
# Run under FreeCAD:  freecadcmd step.py <input.step> <output.json>
import importlib.util
import json
import os
import sys
from pathlib import Path

import Part  # noqa: F401  (FreeCAD provides this; only available inside freecadcmd)

geometry_path = Path(__file__).resolve().parents[1] / "geometry.py"
geometry_spec = importlib.util.spec_from_file_location("_freecad_validator_geometry", geometry_path)
if geometry_spec is None or geometry_spec.loader is None:
    raise ImportError(f"cannot load geometry module from {geometry_path}")
geometry = importlib.util.module_from_spec(geometry_spec)
geometry_spec.loader.exec_module(geometry)

args = [a for a in sys.argv if a.lower().endswith((".step", ".stp", ".json"))]
step = next(a for a in args if a.lower().endswith((".step", ".stp")))
outs = [a for a in args if a.lower().endswith(".json")]
out = outs[0] if outs else os.path.splitext(step)[0] + ".geom.json"

shape = Part.Shape()
shape.read(step)
geom = {
    **geometry.geometry_facts([shape]),
    "source_step": os.path.basename(step),
    "faces": len(shape.Faces),
    "solids": len(shape.Solids),
}
with open(out, "w", encoding="utf-8") as fh:
    json.dump(geom, fh, indent=2)
print(
    f"[step_geom] wrote {out}  volume={shape.Volume:.0f} mm^3  diag={geom['characteristic_length_mm']:.1f} mm"
)
