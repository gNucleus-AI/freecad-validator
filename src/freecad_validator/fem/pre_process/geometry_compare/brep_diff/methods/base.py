"""Common interface for BREP diff methods.

A method recovers an A->B correspondence (and, for the mesh method, a localized
add/remove signal) and turns it into a `DiffResult` change set via the shared
`classify_from_matches`. `BrepDiffMethod.diff` wraps `run` with wall-clock timing
and never raises: a failed method returns an empty-but-valid result so the batch
runner can record the failure instead of aborting.
"""

from __future__ import annotations

import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import asdict, replace

from freecad_validator.fem.pre_process.geometry_compare.brep_diff.geometry_ops import (
    classify_from_matches,
)
from freecad_validator.fem.pre_process.geometry_compare.brep_diff.models import (
    BrepDocument,
    DiffConfig,
    DiffResult,
    MatchRecord,
    make_empty_summary,
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
        except Exception:
            elapsed = time.perf_counter() - start
            return self._failed_result(reference, candidate, traceback.format_exc(), elapsed)
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

    def _failed_result(
        self,
        reference: BrepDocument,
        candidate: BrepDocument,
        message: str,
        elapsed: float,
    ) -> DiffResult:
        return DiffResult(
            method=self.name,
            config=asdict(self.config),
            reference=reference.path,
            candidate=candidate.path,
            freecad_version=reference.freecad_version or candidate.freecad_version,
            occt_version=reference.occt_version or candidate.occt_version,
            summary=make_empty_summary(),
            removed=(),
            added=(),
            modified=(),
            unchanged=(),
            matches=(),
            warnings=(f"method failed: {message.strip().splitlines()[-1]}",),
            runtime_s=elapsed,
        )
