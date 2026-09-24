"""Static material and restraint comparisons shared without changing their policy."""

from collections import deque

from freecad_validator.fem.material_counts import grouped_material_counts
from freecad_validator.fem.schema import Finding

RESTRAINT_TYPES = {
    "fixed",
    "clamped",
    "encastre",
    "pinned",
    "restraint",
    "displacement",
    "fixed_support",
    "support",
}
PARTIAL_RESTRAINT_TYPES = {"symmetry", "symmetric", "roller", "frictionless", "contact"}


def compare_youngs_modulus(reference, candidate):
    expected = next(
        (
            float(reference[key])
            for key in ("E_MPa", "youngs_modulus_MPa")
            if isinstance(reference.get(key), (int, float))
        ),
        None,
    )
    actual = next(
        (
            float(candidate[key])
            for key in ("E_MPa", "youngs_modulus_MPa")
            if isinstance(candidate.get(key), (int, float))
        ),
        None,
    )
    if expected and actual:
        error = abs(actual - expected) / abs(expected)
        if error > 0.20:
            return [
                Finding(
                    "problem_setup",
                    "WRONG_MATERIAL",
                    "critical",
                    f"Young's modulus {actual:g} MPa differs from the specified "
                    f"{expected:g} MPa by {error * 100:.0f}% - wrong material.",
                    penalty=80,
                )
            ]
    elif expected and not actual:
        return [
            Finding(
                "problem_setup",
                "MATERIAL_NOT_STATED",
                "minor",
                "Young's modulus not reported.",
                penalty=8,
            )
        ]
    return []


def _match_material_counts(expected, actual, compatible):
    """Match count capacities, allowing cards to split without reusing bodies."""
    offset = 1 + len(expected)
    sink = offset + len(actual)
    residual = [[0] * (sink + 1) for _ in range(sink + 1)]
    total = sum(card["body_count"] for card in expected)
    for i, card in enumerate(expected):
        residual[0][i + 1] = card["body_count"]
        for j in compatible[i]:
            residual[i + 1][offset + j] = min(card["body_count"], actual[j]["body_count"])
    for j, card in enumerate(actual):
        residual[offset + j][sink] = card["body_count"]

    matched = 0
    while matched < total:
        parents = {0: None}
        queue = deque([0])
        while queue and sink not in parents:
            node = queue.popleft()
            for neighbor, capacity in enumerate(residual[node]):
                if capacity > 0 and neighbor not in parents:
                    parents[neighbor] = node
                    queue.append(neighbor)
        if sink not in parents:
            return None
        amount, node = total - matched, sink
        while node != 0:
            parent = parents[node]
            amount = min(amount, residual[parent][node])
            node = parent
        node = sink
        while node != 0:
            parent = parents[node]
            residual[parent][node] -= amount
            residual[node][parent] += amount
            node = parent
        matched += amount

    return [
        (i, j)
        for i in range(len(expected))
        for j in compatible[i]
        if residual[offset + j][i + 1] > 0
    ]


def compare_material_body_counts(reference, candidate):
    """Match all solid counts under the existing E tolerance, regardless of card splits."""
    expected = grouped_material_counts(reference)
    if not expected:
        raise ValueError("Reference has no material body counts")
    try:
        actual = grouped_material_counts(candidate)
    except (TypeError, ValueError, AttributeError) as exc:
        return [
            Finding(
                "problem_setup",
                "WRONG_MATERIAL",
                "critical",
                f"Invalid candidate material body counts: {exc}",
                penalty=80,
            )
        ]
    if sum(card["body_count"] for card in actual) != sum(card["body_count"] for card in expected):
        return [
            Finding(
                "problem_setup",
                "WRONG_MATERIAL",
                "critical",
                "The total number of material-assigned bodies differs.",
                evidence=f"expected={expected}; candidate={actual}",
                penalty=80,
            )
        ]
    pair_findings = [[compare_youngs_modulus(want, got) for got in actual] for want in expected]
    # Prefer a complete match with stated moduli; retain the existing minor
    # missing-modulus penalty only when no such complete match exists.
    for allow_minor in (False, True):
        compatible = [
            [
                j
                for j, got in enumerate(actual)
                if not any(f.severity == "critical" or not allow_minor for f in pair_findings[i][j])
            ]
            for i in range(len(expected))
        ]
        pairs = _match_material_counts(expected, actual, compatible)
        if pairs is not None:
            # A missing-modulus card can supply several reference cards; penalize it once.
            missing_cards = {j: pair_findings[i][j] for i, j in pairs if pair_findings[i][j]}
            return [finding for findings in missing_cards.values() for finding in findings]

    # No complete count match exists. Keep the per-card diagnostics for the
    # usual unambiguous cases (wrong modulus or wrong count).
    findings = []
    for index, (want, got) in enumerate(zip(expected, actual, strict=False), 1):
        findings.extend(compare_youngs_modulus(want, got))
        if want["body_count"] != got["body_count"]:
            name = want.get("name") or f"material {index}"
            findings.append(
                Finding(
                    "problem_setup",
                    "WRONG_MATERIAL",
                    "critical",
                    f"{name} is assigned to {got['body_count']} bodies; expected {want['body_count']}.",
                    evidence=f"material_card={index}; expected={want}; candidate={got}",
                    penalty=80,
                )
            )
    if not any(f.severity == "critical" for f in findings):
        findings.append(
            Finding(
                "problem_setup",
                "WRONG_MATERIAL",
                "critical",
                "Material body counts cannot all be matched within the Young's-modulus tolerance.",
                evidence=f"expected={expected}; candidate={actual}",
                penalty=80,
            )
        )
    return findings


def compare_restraints(expected, candidate, needs_restraint):
    full = any(b.get("type", "").lower() in RESTRAINT_TYPES for b in candidate)
    partial = any(b.get("type", "").lower() in PARTIAL_RESTRAINT_TYPES for b in candidate)
    case_full = any(b.get("type", "").lower() in RESTRAINT_TYPES for b in expected)
    if needs_restraint and expected:
        if case_full and not full:
            if partial:
                return [
                    Finding(
                        "problem_setup",
                        "MISSING_BOUNDARY_CONDITION",
                        "major",
                        "Only partial/symmetry restraints found where a full restraint "
                        "is required; rigid-body modes may not be removed.",
                        penalty=30,
                    )
                ]
            return [
                Finding(
                    "problem_setup",
                    "MISSING_BOUNDARY_CONDITION",
                    "critical",
                    "No restraint/fixed boundary condition - the model is "
                    "under-constrained (rigid-body motion).",
                    penalty=80,
                )
            ]
        if not case_full and not (full or partial):
            return [
                Finding(
                    "problem_setup",
                    "MISSING_BOUNDARY_CONDITION",
                    "critical",
                    "No restraint at all - even a symmetry model needs its symmetry "
                    "planes constrained to remove rigid-body motion.",
                    penalty=80,
                )
            ]
    if expected and len(candidate) < len(expected):
        return [
            Finding(
                "problem_setup",
                "BC_COUNT_LOW",
                "minor",
                f"Fewer boundary conditions ({len(candidate)}) than expected ({len(expected)}).",
                penalty=10,
            )
        ]
    return []
