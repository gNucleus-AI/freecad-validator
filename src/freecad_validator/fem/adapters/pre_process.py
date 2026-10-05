"""Run the installed geometry worker under FreeCAD's embedded Python.

The wheel may have been installed by a different Python version. Register only
its package locations, without importing host-interpreter dependencies from the
public package initializers. Numeric dependencies must be installed in FreeCAD's
Python environment.
"""

import sys
from pathlib import Path
from types import ModuleType

package_root = Path(__file__).resolve().parents[2]
for name, directory in (
    ("freecad_validator", package_root),
    ("freecad_validator.fem", package_root / "fem"),
):
    if name not in sys.modules:
        package = ModuleType(name)
        package.__path__ = [str(directory)]
        sys.modules[name] = package

from freecad_validator.fem.pre_process.score import main  # noqa: E402

raw, reference, candidate, output = sys.argv[-4:]
main(
    [
        "--assembly",
        "--raw",
        raw,
        "--reference",
        reference,
        "--candidate",
        candidate,
        "--out",
        output,
    ]
)
