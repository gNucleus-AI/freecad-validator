"""Run the shared body-matching worker for a case without preprocessing edits."""

import sys
from pathlib import Path
from types import ModuleType

# As in the preprocessing adapter, use the installed package's sources with
# FreeCAD's interpreter and its own numeric dependencies.
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
    ],
    require_preprocessing=False,
)
