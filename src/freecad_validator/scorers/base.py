"""Base class for FCStd-candidate scorers.

A scorer produces a single 0..1 score for a candidate `.FCStd` against
some reference. The *reference* is whatever ground truth the scorer
uses — another `.FCStd` (for `HeuristicGeometryScorer`) or a spec
JSON (for `HeuristicSpecConsistencyScorer`). Subclasses document the
expected reference type.

Return type is the same `ComparisonResult` used by the comparator layer:
no need for a separate ScoreResult when the shape is identical.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from freecad_validator.comparators.base import ComparisonResult


class FCStdBaseScorer(ABC):
    """Turn a (reference, candidate-FCStd) pair into a 0..1 score + reason."""

    #: Short stable identifier used in logs and error messages.
    name: str = ""

    @abstractmethod
    def score(self, reference: str, candidate: str) -> ComparisonResult:
        """Score how well `candidate` matches `reference`.

        Concrete scorers document what the two paths are expected to be
        (typically both .FCStd, or `reference` = spec .json + `candidate`
        = .FCStd). `ComparisonResult.score` must be in [0, 1] (0 =
        completely different, 1 = identical). `details` carries
        implementation-specific breakdown (per-aspect subscores, bucket
        counts, etc.) for callers that want to inspect the reasoning.
        """
