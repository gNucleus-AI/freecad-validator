"""Verify saved fields by solving the saved mesh; never recompute CAD or remesh."""

import ctypes
import os
import sys
import tempfile

from feminout.importCcxFrdResults import read_frd_result
from femresult.resulttools import calculate_principal_stress_std, calculate_von_mises
from femtools.ccxtools import FemToolsCcx

from freecad_validator.fem.replay_compare import compare_result_snapshots


def preload_openmp_runtime():
    resources_dir = os.path.dirname(os.path.dirname(sys.executable))
    libomp = os.path.join(resources_dir, "lib", "libomp.dylib")
    if not os.path.exists(libomp):
        return
    try:
        ctypes.CDLL(libomp, mode=ctypes.RTLD_GLOBAL)
    except OSError:
        return


def _replay(analysis, solver, stored):
    preload_openmp_runtime()
    with tempfile.TemporaryDirectory(prefix="fem-replay-") as directory:
        fea = FemToolsCcx(analysis, solver)
        fea.update_objects()
        fea.setup_working_dir(directory, create=True)
        fea.setup_ccx()
        if not os.path.isfile(fea.ccx_binary):
            raise FileNotFoundError("CalculiX executable is unavailable")
        error = fea.check_prerequisites()
        if error:
            raise ValueError(f"Saved analysis cannot be solved: {error}")
        # The writer reads the existing nodes/elements and resolves constraint
        # sets. Do not purge results, load result objects, or recompute the CAD.
        fea.write_inp_file()
        if fea.ccx_run() not in (None, 0):
            raise ValueError("CalculiX verification solve failed")
        data = read_frd_result(os.path.splitext(fea.inp_file_name)[0] + ".frd")
        states = [state for state in data["Results"] if state.get("disp") and state.get("stress")]
        if not states:
            raise ValueError("CalculiX produced no static displacement/stress fields")
        state = states[-1]
        nodes = stored["node_numbers"]
        if set(state["disp"]) != set(nodes) or set(state["stress"]) != set(nodes):
            raise ValueError("Verification solve returned a different node set")
        vectors = [state["disp"][node] for node in nodes]
        tensors = [state["stress"][node] for node in nodes]
        fields = {
            "DisplacementVectors": [(v.x, v.y, v.z) for v in vectors],
            "DisplacementLengths": [v.Length for v in vectors],
            "vonMises": [calculate_von_mises(tensor) for tensor in tensors],
        }
        if "MaxShear" in stored["fields"]:
            fields["MaxShear"] = [calculate_principal_stress_std(tensor)[3] for tensor in tensors]
        replayed = {"node_numbers": nodes, "fields": fields}
        return compare_result_snapshots(stored, replayed, "static"), replayed
