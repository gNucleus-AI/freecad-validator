"""Extract and verify saved static analyses under FreeCAD's Python."""

import ctypes
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType


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

# Locate the installed package without loading host-interpreter extensions.
package_root = Path(__file__).resolve().parents[2]
for name, directory in (
    ("freecad_validator", package_root),
    ("freecad_validator.fem", package_root / "fem"),
):
    if name not in sys.modules:
        package = ModuleType(name)
        package.__path__ = [str(directory)]
        sys.modules[name] = package

import FreeCAD  # noqa: E402

from freecad_validator.fem.fcstd_archive import assert_safe_fcstd  # noqa: E402
from freecad_validator.fem.freecad_io import (  # noqa: E402
    extract_material_body_counts as extract_material_body_counts,
)
from freecad_validator.fem.replay_compare import select_scored_results  # noqa: E402
from freecad_validator.fem.static_extraction import (  # noqa: E402
    UnsupportedSetupError,
    _extract_document,
    _result_values,
)
from freecad_validator.fem.static_replay import _replay  # noqa: E402


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


def main():
    args = sys.argv[1:]
    if "import-check" in args:
        print("[fcstd_adapter] pure replay module loaded")
        return
    outputs = [value for value in args if value.lower().endswith(".json")]
    if not outputs:
        raise SystemExit("An output JSON path is required")
    output = outputs[-1]
    if "runtime-info" in args:
        payload = {"runtime": runtime_info(require_calculix=True)}
    else:
        source = next(value for value in args if value.lower().endswith(".fcstd"))
        verify = "verify-solve" in args
        doc = None
        try:
            assert_safe_fcstd(source, require_document=True)
            doc = FreeCAD.openDocument(source)
            payload, snapshot, analysis, solver = _extract_document(doc)
            if verify:
                verification, replayed = _replay(analysis, solver, snapshot)
                passed = verification["passed"]
                payload["meta"] = {"solver_replay": verification}
                payload["solver"] = {
                    "converged": passed,
                    "replay_accepted": passed,
                    "replay_verified": verification["status"] == "verified",
                }
                payload["artifacts"] = {"result_file": "saved.FCStd"} if passed else {}
                if passed:
                    payload["results"], source_name = select_scored_results(
                        payload["results"], _result_values(replayed), verification
                    )
                    payload["solver"]["result_source"] = source_name
        except (ValueError, RuntimeError) as exc:
            if not verify or isinstance(exc, UnsupportedSetupError):
                raise
            payload = {
                "solver": {"converged": False},
                "results": {},
                "meta": {"candidate_extraction_failure": str(exc)},
            }
        finally:
            if doc is not None:
                FreeCAD.closeDocument(doc.Name)
    with open(output, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)


main()
