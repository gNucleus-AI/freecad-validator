"""Batch point-in-solid queries with reusable OCCT classifiers.

The native backend runs in a child process so its memory is released after the
queries. BREP transfer preserves the evaluated geometry without rebuilding it.
"""

import json
import logging
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _python_interpreter() -> str:
    """Find a Python that can load the worker's OCP dependencies, once per process."""
    names = (f"python{sys.version_info.major}.{sys.version_info.minor}", "python3", "python")
    candidates = []
    if Path(sys.executable).name.startswith("python"):
        candidates.append(sys.executable)
    candidates.extend(str(Path(sys.prefix) / "bin" / name) for name in names)
    candidates.extend(path for name in names if (path := shutil.which(name)))
    worker = Path(__file__).with_name("_occt_solid_occupancy_worker.py")
    diagnostics = []
    checked = set()
    for candidate in candidates:
        path = Path(candidate)
        if not path.is_file() or path.absolute() in checked:
            continue
        checked.add(path.absolute())
        # Load the actual imports without running classification or reading geometry.
        try:
            probe = subprocess.run(
                [
                    str(path),
                    "-P",
                    "-c",
                    "import runpy, sys; runpy.run_path(sys.argv[1])",
                    str(worker),
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=15,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            diagnostics.append(f"{path}: {exc}")
            continue
        if probe.returncode == 0:
            logging.info("OCCT solid classification Python: %s", path)
            return str(path)
        diagnostics.append(f"{path}: {probe.stderr.strip()[-500:]}")
    detail = (
        "; ".join(diagnostics) or "No Python executable found in the current environment or PATH"
    )
    raise RuntimeError(f"No Python interpreter with OCP for solid classification. {detail}")


def solid_occupancies(
    breps: list[str],
    points: list[tuple[float, float, float]],
    tolerance: float,
) -> list[list[bool]]:
    """Return one occupancy list per solid; boundary points count as inside."""
    worker = Path(__file__).with_name("_occt_solid_occupancy_worker.py")
    proc = subprocess.run(
        [_python_interpreter(), "-P", str(worker)],
        input=json.dumps({"breps": breps, "points": points, "tolerance": tolerance}),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"OCCT solid classification failed ({proc.returncode}): {proc.stderr.strip()[-1000:]}"
        )
    return json.loads(proc.stdout)
