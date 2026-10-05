"""Generate a small solved cantilever fixture with FreeCAD's public FEM example."""

import os
import sys

import FreeCAD
import Part
from femexamples.ccx_cantilever_faceload import setup
from femtools.ccxtools import FemToolsCcx


def main():
    step_path = next(arg for arg in sys.argv if arg.lower().endswith(".step"))
    fcstd_path = next(arg for arg in sys.argv if arg.lower().endswith(".fcstd"))
    working_dir = os.path.dirname(os.path.abspath(fcstd_path))

    doc = setup(test_mode=True)
    if "raw-hole" in sys.argv:
        box = doc.Box.Shape
        bounds = box.BoundBox
        hole = Part.makeCylinder(
            0.2 * min(bounds.XLength, bounds.YLength),
            bounds.ZLength,
            FreeCAD.Vector(bounds.Center.x, bounds.Center.y, bounds.ZMin),
        )
        box.cut(hole).exportStep(step_path)
    else:
        Part.export([doc.Box], step_path)
    solver = doc.CalculiXCcxTools
    solver.WorkingDir = working_dir
    fem = FemToolsCcx(doc.Analysis, solver)
    fem.update_objects()
    fem.setup_working_dir(working_dir, create=True)
    fem.setup_ccx()
    prerequisites = fem.check_prerequisites()
    if prerequisites:
        raise RuntimeError(f"CalculiX prerequisites failed: {prerequisites}")
    fem.purge_results()
    fem.write_inp_file()
    return_code = fem.ccx_run()
    if return_code not in (None, 0):
        raise RuntimeError(f"CalculiX failed with exit code {return_code}")
    fem.load_results()
    doc.recompute()
    doc.saveAs(fcstd_path)


main()
