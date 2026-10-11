"""Exercise an installed FCStd adapter without importing its host package."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    adapter = Path(sys.argv[1]).resolve()
    if not adapter.is_file():
        raise FileNotFoundError(adapter)

    with tempfile.TemporaryDirectory() as temp_dir:
        stubs = Path(temp_dir)
        (stubs / "FreeCAD.py").write_text(
            "class Units:\n    pass\n",
            encoding="utf-8",
        )
        femtools = stubs / "femtools"
        femtools.mkdir()
        (femtools / "ccxtools.py").write_text(
            "class FemToolsCcx:\n    pass\n",
            encoding="utf-8",
        )
        for package, module, functions in (
            ("femmesh", "meshtools", ["sub_shape_at_global_placement"]),
            ("feminout", "importCcxFrdResults", ["read_frd_result"]),
            ("femresult", "resulttools", ["calculate_principal_stress_std", "calculate_von_mises"]),
        ):
            directory = stubs / package
            directory.mkdir()
            (directory / (module + ".py")).write_text(
                "\n".join(f"def {name}(*args): pass" for name in functions),
                encoding="utf-8",
            )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(stubs)
        completed = subprocess.run(
            [sys.executable, str(adapter), "import-check"],
            check=False,
            capture_output=True,
            text=True,
            env=env,
            timeout=10,
        )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr)
    if "pure replay module loaded" not in completed.stdout:
        raise RuntimeError(f"unexpected adapter output: {completed.stdout!r}")


if __name__ == "__main__":
    main()
