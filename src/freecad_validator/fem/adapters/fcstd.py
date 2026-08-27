"""Extract a FreeCAD FEM result document (.FCStd) into the scorer's JSON schema.

MUST be run under a FEM-enabled FreeCAD 1.1.0 interpreter compatible with the
files being compared. For example:

    freecadcmd fcstd.py <input.FCStd> <output.json>

It reads the saved result object (the same object FreeCAD_FEM_Workflow.py writes
its summary from) plus the material, constraints, solver and mesh, and writes a
dict that maps onto ``freecad_validator.fem.schema``. The JSON is
written to a FILE, so FreeCAD's banner/warning noise on stdout never corrupts it.

Notes
-----
* freecadcmd does not set __name__ == "__main__" and passes script args through
  sys.argv, so the entry point is called unconditionally at the bottom and args
  are discovered by extension (.FCStd / .json) rather than by position.
* An .FCStd carries geometry, material, BCs/loads, mesh and nodal result fields,
  but those arrays alone do not prove a solver ran. Candidate extraction uses
  ``verify-solve`` to rerun CalculiX from the saved analysis and compare the
  replayed fields before setting solver/artifact evidence. Reference extraction
  stays read-only and never vouches for candidate convergence.
"""

import ctypes
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def preload_openmp_runtime():
    resources_dir = os.path.dirname(os.path.dirname(sys.executable))
    libomp = os.path.join(resources_dir, "lib", "libomp.dylib")
    if not os.path.exists(libomp):
        return
    try:
        ctypes.CDLL(libomp, mode=ctypes.RTLD_GLOBAL)
    except OSError:
        return


preload_openmp_runtime()

import FreeCAD  # noqa: E402
from femtools.ccxtools import FemToolsCcx  # noqa: E402
from FreeCAD import Units  # noqa: E402

# FreeCAD may embed a different Python minor version from the interpreter that
# installed this wheel. Loading this pure-Python module by file path avoids
# importing the package root and any host-interpreter extension modules.
replay_path = Path(__file__).resolve().parents[1] / "replay_compare.py"
replay_spec = importlib.util.spec_from_file_location(
    "_freecad_validator_replay_compare", replay_path
)
if replay_spec is None or replay_spec.loader is None:
    raise ImportError(f"cannot load replay comparison module from {replay_path}")
replay_compare = importlib.util.module_from_spec(replay_spec)
replay_spec.loader.exec_module(replay_compare)

REPLAY_REL_TOL = replay_compare.REPLAY_REL_TOL
REPLAY_SCALAR_FIELDS = replay_compare.REPLAY_SCALAR_FIELDS
compare_result_snapshots = replay_compare.compare_result_snapshots
select_scored_results = replay_compare.select_scored_results


def runtime_info(require_calculix=False):
    """Return runtime versions and optionally verify the configured solver."""
    import Part

    freecad_version = ".".join(str(value) for value in FreeCAD.Version()[:3])
    info = {
        "freecad": freecad_version,
        "freecad_build": list(FreeCAD.Version()),
        "occt": str(Part.OCC_VERSION),
        "python": sys.version.split()[0],
    }
    if not require_calculix:
        return info

    configured = FreeCAD.ParamGet("User parameter:BaseApp/Preferences/Mod/Fem/Ccx").GetString(
        "ccxBinaryPath", ""
    )
    ccx = shutil.which(configured or "ccx")
    if ccx is None:
        sibling = Path(sys.executable).resolve().with_name("ccx")
        ccx = str(sibling) if sibling.is_file() and os.access(sibling, os.X_OK) else None
    if ccx is None:
        raise RuntimeError(
            "CalculiX executable was not found; install ccx or configure its path "
            "in FreeCAD FEM preferences"
        )
    completed = subprocess.run(
        [ccx, "-v"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    version_output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    version_match = re.search(r"Version\s+([0-9]+(?:\.[0-9]+)+)", version_output)
    # Some CalculiX builds return a non-zero status after printing their version.
    # A successfully parsed version is the portable availability check.
    if version_match is None:
        raise RuntimeError("CalculiX executable failed its version preflight")
    info["calculix"] = version_match.group(1)
    info["calculix_executable"] = Path(ccx).name
    return info


def qty(value, unit):
    """Convert a FreeCAD quantity/string/number to a float in `unit`."""
    try:
        return float(Units.Quantity(value).getValueAs(unit))
    except (ValueError, TypeError):
        return float(value)


def _is_result_object(obj):
    """A loaded mechanical FEM result (has nodal fields). Single source of truth
    for what counts as a 'result', shared by discovery, the pre-replay purge and
    replay-output selection so they can never disagree (a disagreement would be a
    hole: a result invisible to the purge but visible to selection)."""
    return obj.TypeId == "Fem::FemResultObjectPython" and getattr(obj, "NodeNumbers", None)


def find_result(doc):
    for o in doc.Objects:
        if _is_result_object(o):
            return o
    return None


def extract_result_values(res):
    results = {}
    disp = list(getattr(res, "DisplacementLengths", []) or [])
    vm = list(getattr(res, "vonMises", []) or [])
    shear = list(getattr(res, "MaxShear", []) or [])
    temp = list(getattr(res, "Temperature", []) or [])
    if disp:
        results["max_displacement_mm"] = max(disp)
        results["mean_displacement_mm"] = sum(disp) / len(disp)
    if vm:
        results["max_von_mises_MPa"] = max(vm)
        results["mean_von_mises_MPa"] = sum(vm) / len(vm)
    if shear:
        results["max_shear_MPa"] = max(shear)
    if temp:
        results["max_temperature_C"] = max(temp)
    freqs = list(getattr(res, "EigenmodeFrequencies", []) or [])
    if freqs:
        results["natural_frequencies_Hz"] = freqs
    return results


def snapshot_result_fields(res):
    fields = {}
    for name in REPLAY_SCALAR_FIELDS:
        values = list(getattr(res, name, []) or [])
        if values:
            fields[name] = [float(value) for value in values]

    vectors = list(getattr(res, "DisplacementVectors", []) or [])
    if vectors:
        fields["DisplacementVectors"] = [
            (float(value.x), float(value.y), float(value.z)) for value in vectors
        ]

    return {
        "node_numbers": [int(node) for node in list(getattr(res, "NodeNumbers", []) or [])],
        "fields": fields,
    }


def find_replay_context(doc, result, solved_node_count):
    analyses = [obj for obj in doc.Objects if obj.TypeId == "Fem::FemAnalysis"]
    matching_analyses = []
    for analysis in analyses:
        group = list(getattr(analysis, "Group", []) or [])
        if result in group or getattr(result, "Mesh", None) in group:
            matching_analyses.append(analysis)
    if len(matching_analyses) != 1:
        raise RuntimeError(
            "stored FEM result is not owned by exactly one analysis "
            f"(found {len(matching_analyses)})"
        )

    analysis = matching_analyses[0]
    group = list(getattr(analysis, "Group", []) or [])
    solvers = [obj for obj in group if "Solver" in obj.TypeId or "Ccx" in obj.TypeId]
    if len(solvers) != 1:
        raise RuntimeError(
            f"analysis must contain exactly one solver for replay (found {len(solvers)})"
        )

    source_meshes = []
    for obj in group:
        fem_mesh = getattr(obj, "FemMesh", None)
        node_count = getattr(fem_mesh, "NodeCount", 0) if fem_mesh is not None else 0
        if "FemMeshShape" in obj.TypeId and node_count == solved_node_count:
            source_meshes.append(obj)
    if len(source_meshes) != 1:
        raise RuntimeError(
            "analysis must contain exactly one source mesh matching the stored result "
            f"({solved_node_count} nodes; found {len(source_meshes)})"
        )
    return analysis, solvers[0], source_meshes[0]


def verify_solver_replay(doc, stored_result, analysis_type, output_path):
    stored_snapshot = snapshot_result_fields(stored_result)
    work_parent = os.path.dirname(os.path.abspath(output_path))
    work_dir = tempfile.mkdtemp(prefix="fem-replay-", dir=work_parent)
    os.chmod(work_dir, 0o700)
    try:
        analysis, solver, source_mesh = find_replay_context(
            doc, stored_result, len(stored_snapshot["node_numbers"])
        )
        solver.WorkingDir = work_dir
        fea = FemToolsCcx(analysis, solver)
        fea.update_objects()
        fea.setup_working_dir(work_dir, create=True)
        fea.setup_ccx()
        prerequisite_error = fea.check_prerequisites()
        if prerequisite_error:
            raise RuntimeError(f"CalculiX replay prerequisites failed: {prerequisite_error}")

        # Clear every existing result before re-solving. FreeCAD's helper removes
        # analysis-group members, while documents can also contain detached result
        # objects. Removing both makes the freshly loaded replay result unambiguous.
        fea.purge_results()
        for stale in [obj for obj in doc.Objects if _is_result_object(obj)]:
            doc.removeObject(stale.Name)
        doc.recompute()
        fea.write_inp_file()
        return_code = fea.ccx_run()
        if return_code not in (None, 0):
            raise RuntimeError(f"CalculiX replay failed with exit code {return_code}")
        fea.load_results()
        doc.recompute()

        replay_results = [obj for obj in doc.Objects if _is_result_object(obj)]
        if not replay_results:
            raise RuntimeError("CalculiX replay produced no loaded FEM result")
        replayed_result = max(replay_results, key=lambda obj: len(obj.NodeNumbers))
        replayed_snapshot = snapshot_result_fields(replayed_result)
        verification = compare_result_snapshots(stored_snapshot, replayed_snapshot, analysis_type)
        verification.update(
            {
                "analysis": analysis.Name,
                "solver": solver.Name,
                "mesh": source_mesh.Name,
                "node_count": len(replayed_snapshot["node_numbers"]),
            }
        )
        return verification, extract_result_values(replayed_result)
    except Exception as exc:
        return {
            "passed": False,
            "status": "solve_failed",
            "relative_tolerance": REPLAY_REL_TOL,
            "failures": [str(exc)],
            "field_comparisons": {},
        }, None
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def extract_material(doc):
    for o in doc.Objects:
        if "Material" in o.TypeId and getattr(o, "Material", None):
            m = dict(o.Material)
            out = {"name": m.get("Name", "")}
            if m.get("YoungsModulus"):
                out["E_MPa"] = qty(m["YoungsModulus"], "MPa")
            if m.get("PoissonRatio"):
                out["nu"] = float(m["PoissonRatio"])
            if m.get("Density"):
                out["rho_kg_m3"] = qty(m["Density"], "kg/m^3")
            return out
    return {}


# CalculiX/ccxtools AnalysisType string -> scorer vocabulary
_ANALYSIS = {
    "static": "static",
    "frequency": "modal",
    "thermomech": "thermal_mechanical",
    "buckling": "buckling",
    "check": "static",
}


def extract_solver(doc):
    for o in doc.Objects:
        if "Solver" in o.TypeId or "Ccx" in o.TypeId:
            at = getattr(o, "AnalysisType", "static")
            return _ANALYSIS.get(str(at), "static"), getattr(o, "GeometricalNonlinearity", "linear")
    return "static", "linear"


def refs_str(o):
    try:
        return "; ".join(f"{r[0].Name}/{','.join(r[1])}" for r in o.References)
    except Exception:
        return ""


def force_direction(o):
    """Unit direction the force actually points, or None.

    Lets the scorer check WHICH WAY the load points, not just its magnitude.
    DirectionVector is already the FINAL applied direction: the CalculiX writer
    (Mod/Fem/femsolver/calculix/write_constraint_force.py) forms the nodal load
    straight from DirectionVector and never references `Reversed`, so we must NOT
    negate it here. A load written as (DirectionVector, Reversed=True) is
    physically identical to the same DirectionVector with Reversed=False; the
    solved displacement field confirms CalculiX ignores `Reversed`."""
    dv = getattr(o, "DirectionVector", None)
    if dv is None or getattr(dv, "Length", 0) < 1e-9:
        return None
    u = FreeCAD.Vector(dv)
    u.normalize()
    return [u.x, u.y, u.z]


def refs_centroid(o):
    """Area/length-weighted centroid (mm) of the loaded sub-elements, or None.

    Both reference and candidate are built on the same source solid, so this point is
    comparable even when their internal face/edge indices differ - it lets the
    scorer check WHERE the load is applied without relying on reference names."""
    pts, wts = [], []
    try:
        for obj, subs in o.References:
            shp = getattr(obj, "Shape", None)
            if shp is None:
                continue
            for sub in subs or [None]:
                el = shp.getElement(sub) if sub else shp
                c = el.CenterOfMass
                w = getattr(el, "Area", None) or getattr(el, "Length", None) or 1.0
                pts.append((c.x, c.y, c.z))
                wts.append(float(w))
    except Exception:
        return None
    if not pts:
        return None
    tw = sum(wts) or 1.0
    return [sum(p[i] * w for p, w in zip(pts, wts, strict=True)) / tw for i in range(3)]


def extract_bcs_loads(doc):
    bcs, loads = [], []
    for o in doc.Objects:
        t = o.TypeId
        if t in ("Fem::ConstraintFixed", "Fem::ConstraintDisplacement"):
            bcs.append({"type": "fixed", "location": refs_str(o)})
        elif t in ("Fem::ConstraintBearing",):
            bcs.append({"type": "support", "location": refs_str(o)})
        elif t == "Fem::ConstraintForce":
            loads.append(
                {
                    "type": "force",
                    "magnitude_N": qty(o.Force, "N"),
                    "reversed": bool(getattr(o, "Reversed", False)),
                    "direction": force_direction(o),
                    "centroid": refs_centroid(o),
                    "location": refs_str(o),
                }
            )
        elif t == "Fem::ConstraintPressure":
            loads.append(
                {
                    "type": "pressure",
                    "magnitude_Pa": qty(o.Pressure, "Pa"),
                    "magnitude_N": 0,
                    "reversed": bool(getattr(o, "Reversed", False)),
                    "centroid": refs_centroid(o),
                    "location": refs_str(o),
                }
            )
        elif t == "Fem::ConstraintSelfWeight":
            loads.append({"type": "self_weight", "magnitude_N": 0})
    return bcs, loads


def _topo(x):
    """A Part TopoShape from either a TopoShape or a document object, else None."""
    if x is None:
        return None
    if getattr(x, "Solids", None) is not None:  # already a TopoShape (has .Solids)
        return x
    return getattr(x, "Shape", None)  # a document object -> its shape


def analysed_solids(doc):
    """The solid shape(s) that make up the analysed geometry, as a list.

    The geometry-fidelity check compares the analysed volume to the STEP's TOTAL
    volume, so for a multi-body part we must return EVERY analysed solid (their
    volumes are summed by the caller), not just one - while never counting leftover
    imported sub-parts or a fusion's consumed inputs (that would double-count).
    Priority:
      1) the geometry the FEM mesh was generated on (mesh object's Part/Shape link)
         - exactly what was meshed; a compound reports all its solids, so this is
           correct for one solid, a fusion, or a multi-solid compound;
      2) the DISTINCT bodies referenced by the analysis constraints (BCs/loads) -
         robust for a disjoint multi-body assembly and immune to leftover sub-parts,
         because only the bodies actually constrained/loaded are counted;
      3) the single largest-volume solid object - last resort (single-body only).
    """
    # 1) meshed geometry (one object, possibly a compound of several solids)
    for o in doc.Objects:
        if "FemMeshShape" in o.TypeId or "FemMeshGmsh" in o.TypeId:
            shp = _topo(getattr(o, "Part", None)) or _topo(getattr(o, "Shape", None))
            if shp is not None and getattr(shp, "Solids", None):
                return [shp]
    # 2) the distinct bodies the analysis constraints reference
    refd = {}
    for o in doc.Objects:
        if not o.TypeId.startswith("Fem::Constraint"):
            continue
        try:
            refs = o.References or []
        except Exception:
            refs = []
        for parent, _subs in refs:
            shp = getattr(parent, "Shape", None)
            if shp is not None and getattr(shp, "Solids", None):
                refd[getattr(parent, "Name", id(parent))] = shp
    if refd:
        return list(refd.values())
    # 3) the single largest-volume solid object
    best, best_vol = None, 0.0
    for o in doc.Objects:
        shp = getattr(o, "Shape", None)
        if shp is not None and getattr(shp, "Solids", None) and shp.Volume > best_vol:
            best, best_vol = shp, shp.Volume
    return [best] if best is not None else []


def extract_geometry(doc):
    shapes = analysed_solids(doc)
    if not shapes:
        return {}
    solids = [solid for shape in shapes for solid in shape.Solids]
    volume = sum(s.Volume for s in shapes)
    surface_area = sum(s.Area for s in shapes)
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
    bbs = [s.BoundBox for s in shapes]
    xs = [b.XMin for b in bbs] + [b.XMax for b in bbs]
    ys = [b.YMin for b in bbs] + [b.YMax for b in bbs]
    zs = [b.ZMin for b in bbs] + [b.ZMax for b in bbs]
    dx, dy, dz = max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)
    return {
        "characteristic_length_mm": (dx * dx + dy * dy + dz * dz) ** 0.5,
        "bbox_mm": [dx, dy, dz],
        "volume_mm3": volume,
        "surface_area_mm2": surface_area,
        "num_solids": len(solids),
        "num_compsolids": sum(len(shape.CompSolids) for shape in shapes),
        "shape_types": sorted({str(shape.ShapeType) for shape in shapes}),
        "num_faces": sum(region["num_faces"] for region in regions),
        "num_edges": sum(region["num_edges"] for region in regions),
        "regions": regions,
    }


def assert_safe_fcstd(path):
    """Reject an FCStd whose zip embeds a path-traversal member (zip-slip).

    Included files may otherwise be extracted outside the intended directory
    when FreeCAD opens the document. Validate every member before opening it:
    absolute paths, drive letters, and ``..`` components that escape the archive
    root are rejected. Legitimate FCStd members use relative paths."""
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
    except zipfile.BadZipFile:
        return  # not a zip (unlikely for .FCStd); openDocument will handle it
    for name in names:
        norm = name.replace("\\", "/")
        if norm.startswith("/") or (len(norm) > 1 and norm[1] == ":"):
            raise SystemExit(
                f"[fcstd_adapter] SECURITY: absolute path in FCStd zip member: {name!r}"
            )
        parts = [p for p in norm.split("/") if p not in ("", ".")]
        depth = 0
        for p in parts:
            depth += -1 if p == ".." else 1
            if depth < 0:
                raise SystemExit(
                    f"[fcstd_adapter] SECURITY: path traversal in FCStd zip member: {name!r}"
                )


def main():
    args = sys.argv
    import_check = "import-check" in args
    verify_solve = "verify-solve" in args
    geometry_only = "geometry-only" in args
    runtime_only = "runtime-info" in args
    fcstd = next((a for a in args if a.lower().endswith(".fcstd")), None)
    outs = [a for a in args if a.lower().endswith(".json")]
    if import_check:
        print("[fcstd_adapter] pure replay module loaded")
        return
    if runtime_only:
        if not outs:
            raise SystemExit("runtime-info requires an output JSON path")
        with open(outs[0], "w", encoding="utf-8") as fh:
            json.dump({"runtime": runtime_info(require_calculix=True)}, fh, indent=2)
        print(f"[fcstd_adapter] wrote {outs[0]}  runtime_info=true")
        return
    if not fcstd:
        raise SystemExit("usage: freecadcmd fcstd_adapter.py <input.FCStd> <output.json>")
    out = outs[0] if outs else os.path.splitext(fcstd)[0] + ".scorer.json"

    assert_safe_fcstd(fcstd)
    doc = FreeCAD.openDocument(fcstd)
    if geometry_only:
        out_dict = {
            "source_fcstd": os.path.basename(fcstd),
            "geometry": extract_geometry(doc),
            "runtime": runtime_info(),
        }
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(out_dict, fh, indent=2)
        FreeCAD.closeDocument(doc.Name)
        print(f"[fcstd_adapter] wrote {out}  geometry_only=true")
        return
    res = find_result(doc)
    if res is None:
        failure_reason = f"no FEM result object found in {fcstd}"
        out_dict = {
            "source_fcstd": os.path.basename(fcstd),
            "no_result": True,
            "extraction_error": failure_reason,
            "solver": {"converged": False},
            "results": {},
            "artifacts": {},
            "runtime": runtime_info(),
        }
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(out_dict, fh, indent=2)
        FreeCAD.closeDocument(doc.Name)
        print(f"[fcstd_adapter] wrote {out}  no_result=true")
        return

    analysis, nonlinearity = extract_solver(doc)
    bcs, loads = extract_bcs_loads(doc)
    material = extract_material(doc)
    geometry = extract_geometry(doc)

    results = extract_result_values(res)
    disp = list(getattr(res, "DisplacementLengths", []) or [])

    # Mesh stats for the mesh-budget category. CRITICAL: tie the reported mesh to
    # the SOLVE, not to whatever mesh is attached to the result. A genuine result
    # carries exactly one displacement value per mesh node, so the mesh actually
    # solved on is the one whose NodeCount == len(DisplacementLengths). Reading
    # Reading res.Mesh alone can select a mesh that differs from the one used for
    # the stored result, so choose the mesh that
    # matches the displacement field (searching every mesh in the document, not
    # just res.Mesh) and treat the result as incoherent when, for a displacement
    # analysis, no mesh matches it.
    n_solved = len(disp)
    solved_fm = None
    # search every mesh in the document, plus the result's own attached mesh (so we
    # do not depend on res.Mesh also being enumerated in doc.Objects)
    mesh_candidates = list(doc.Objects)
    if getattr(res, "Mesh", None) is not None:
        mesh_candidates.append(res.Mesh)
    for o in mesh_candidates:
        cand_fm = getattr(o, "FemMesh", None)
        nc = getattr(cand_fm, "NodeCount", 0) if cand_fm is not None else 0
        if not nc:
            continue
        if n_solved:
            if nc == n_solved:
                solved_fm = cand_fm
                break
        else:
            solved_fm = cand_fm  # no displacement field to match against (e.g. modal)
            break
    mesh = {}
    if solved_fm is not None:
        mesh = {
            "num_nodes": solved_fm.NodeCount,
            "num_elements": solved_fm.VolumeCount,
            "element_type": "tet10",
        }

    # Cheap coherence prechecks before the authoritative solver replay. Passing
    # these checks is necessary but is NOT evidence that CalculiX ran:
    #  (a) a static/thermomech result MUST carry a displacement field - it is the
    #      PRIMARY solution variable, and vonMises/MaxShear are DERIVED from it; a
    #      result with derived stress but no displacement was hand-filled;
    #  (b) the displacement field MUST correspond to a real mesh in the document
    #      (NodeCount == len(disp)) - otherwise the mesh was swapped after solving
    #      after solving and incorrectly receive mesh-budget credit.
    static_no_disp = analysis in ("static", "thermal_mechanical") and not disp
    mesh_matches_solve = (n_solved == 0) or (solved_fm is not None)
    coherent = (not static_no_disp) and mesh_matches_solve

    out_dict = {
        "source_fcstd": os.path.basename(fcstd),
        "analysis_type": analysis,
        "geometrical_nonlinearity": nonlinearity,
        "units": {"length": "mm", "force": "N", "stress": "MPa"},
        "material": material,
        "geometry": geometry,
        "boundary_conditions": bcs,
        "loads": loads,
        "mesh": mesh,
        "results": results,
        "runtime": runtime_info(require_calculix=verify_solve),
    }
    verification = None
    replayed_results = None
    if verify_solve and coherent:
        verification, replayed_results = verify_solver_replay(doc, res, analysis, out)
    elif verify_solve:
        verification = {
            "passed": False,
            "status": "precheck_failed",
            "relative_tolerance": REPLAY_REL_TOL,
            "failures": ["stored result failed displacement/mesh coherence checks"],
            "field_comparisons": {},
        }

    replay_accepted = bool(verification and verification.get("passed"))
    if replay_accepted:
        # Only the trusted validator-side replay can establish convergence and
        # reproducibility. Within the strict tolerance, score trusted replayed
        # values. For an accepted 2%-10% mismatch, retain the stored values so a
        # deterministic FreeCAD remapping difference does not replace the result
        # that was actually saved by the original solve.
        out_dict["results"], result_source = select_scored_results(
            out_dict["results"], replayed_results, verification
        )
        replay_verified = verification.get("status") == "verified"
        out_dict["solver"] = {
            "name": "CalculiX",
            "converged": True,
            "replay_verified": replay_verified,
            "replay_accepted": True,
            "replay_status": verification.get("status"),
            "result_source": result_source,
        }
        out_dict["artifacts"] = {
            "result_file": os.path.basename(fcstd),
        }
    else:
        # A loaded result object and matching array lengths are structural checks,
        # not proof that CalculiX ran. Never vouch for unverified stored arrays.
        out_dict["solver"] = {
            "name": "CalculiX",
            "converged": False,
            "replay_verified": False,
            "replay_accepted": False,
        }
        out_dict["artifacts"] = {}
        if verification is not None:
            out_dict["meta"] = {"solver_replay": verification}

    if not coherent:
        if static_no_disp:
            out_dict["incoherent_result"] = (
                f"{analysis} result reports {sorted(results)} but the displacement field is "
                "empty - the primary solution variable is missing, so this was not solved"
            )
        else:
            out_dict["incoherent_result"] = (
                f"no mesh in the document matches the {n_solved}-value displacement field "
                "(NodeCount != len(DisplacementLengths)) - the reported mesh was not the one "
                "solved on (mesh/result mismatch)"
            )
    elif verification is not None and not replay_accepted:
        out_dict["incoherent_result"] = (
            "stored FEM fields were not reproduced by a trusted validator-side CalculiX replay"
        )
    if verification is not None and replay_accepted:
        out_dict["meta"] = {"solver_replay": verification}
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(out_dict, fh, indent=2)
    FreeCAD.closeDocument(doc.Name)
    print(f"[fcstd_adapter] wrote {out}  results={out_dict['results']}")


main()
