"""V2 mixed geometry-bound and legacy Spec checks, with equal failure penalties."""

from __future__ import annotations

from freecad_validator.consistency.checker import ConsistencyChecker, SpecTolerances
from freecad_validator.scorers.spec_consistency import HeuristicSpecConsistencyScorer


class HeuristicSpecConsistencyScorerV2(HeuristicSpecConsistencyScorer):
    """Honor trusted geometry_bindings; specs without bindings retain old checks."""

    name = "heuristic_spec_consistency_v2"

    def __init__(
        self, tolerances: SpecTolerances | None = None, *, failure_budget: int | None = 10
    ):
        super().__init__(tolerances, failure_budget=failure_budget)
        self._checker = ConsistencyChecker(tolerances=tolerances, use_geometry_bindings=True)
