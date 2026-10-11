"""The packaged geometry runtime uses only public modules and declared dependencies."""

import ast
import sys
from pathlib import Path

from freecad_validator.fem import step_interface


def test_geometry_runtime_dependency_boundary():
    directory = Path(step_interface.__file__).parent / "pre_process"
    allowed = sys.stdlib_module_names | {
        "FreeCAD",
        "Part",
        "numpy",
        "scipy",
        "OCP",
        "freecad_validator",
    }
    for path in directory.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            assert all(module.split(".")[0] in allowed for module in modules), path
            assert all(
                not module.startswith("freecad_validator.")
                or module.startswith("freecad_validator.fem.")
                for module in modules
            ), path
