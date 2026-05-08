"""Score one (candidate, reference, spec) triple.

Usage::

    python examples/01_single_case.py CANDIDATE.FCStd REFERENCE.FCStd SPEC.json

Requires FreeCAD on PATH (so ``import FreeCAD`` works) and
``gnucleus-freecad-validator`` installed.
"""
from __future__ import annotations

import sys

from freecad_validator import Validator


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(__doc__)
        return 2
    _, candidate, reference, spec = argv

    validator = Validator()
    result = validator.validate(
        candidate_fcstd=candidate,
        reference_fcstd=reference,
        spec_json=spec,
    )
    # The overall verdict is `result.combined` — harmonic mean of the
    # two sub-scores. Print all three so the breakdown is visible.
    print(f"score (combined)     : {result.combined:.4f}")
    print(f"  geometry_similarity  : {result.geometry_similarity:.4f}")
    print(f"  cad_spec_consistency : {result.cad_spec_consistency:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
