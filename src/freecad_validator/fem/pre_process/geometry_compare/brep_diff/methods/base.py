"""Common interface for BREP diff methods.

A method recovers an A->B correspondence (and, for the mesh method, a localized
add/remove signal) and turns it into a `DiffResult` change set via the shared
`classify_from_matches`. `BrepDiffMethod.diff` wraps `run` with wall-clock timing
and propagates failures as evaluation errors. A failed computation must never
be interpreted as a successful empty change set by geometry scoring.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import replace

from freecad_validator.fem.errors import EvaluationError
from freecad_validator.fem.pre_process.geometry_compare.brep_diff.geometry_ops import (
    classify_from_matches,
)
from freecad_validator.fem.pre_process.geometry_compare.brep_diff.models import (
    BrepDocument,
    DiffConfig,
    DiffResult,
    MatchRecord,
)


class BrepDiffMethod(ABC):
    name: str = ""

    def __init__(self, config: DiffConfig | None = None) -> None:
        self.config = config or DiffConfig(method=self.name)

    @abstractmethod
    def run(self, reference: BrepDocument, candidate: BrepDocument) -> DiffResult:
        """Compute the classified change set for one (A, B) pair."""

    def diff(self, reference: BrepDocument, candidate: BrepDocument) -> DiffResult:
        start = time.perf_counter()
        try:
            result = self.run(reference, candidate)
        except Exception as exc:
            raise EvaluationError(
                f"Geometry diff {self.name} failed: {type(exc).__name__}: {exc}"
            ) from exc
        return replace(result, runtime_s=time.perf_counter() - start)

    # -- helpers for matcher-based subclasses -------------------------------- #
    def classify(
        self,
        reference: BrepDocument,
        candidate: BrepDocument,
        matches: tuple[MatchRecord, ...],
        warnings: tuple[str, ...] = (),
        overlap_override: bool = False,
    ) -> DiffResult:
        return classify_from_matches(
            reference, candidate, matches, self.config, self.name, warnings, overlap_override
        )
