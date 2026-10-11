"""Presence requirements, separate from continuous setup agreement."""

from collections import Counter

from freecad_validator.fem.schema import Finding


def required_setup_findings(reference, candidate):
    findings = []
    for key, label in (("loads", "load"), ("boundary_conditions", "restraint")):
        expected = Counter(item.get("type") for item in reference.get(key, []))
        actual = Counter(item.get("type") for item in candidate.get(key, []))
        # Restraint formulations can differ legitimately. Missing all required
        # restraints is invalid; exact surface/type agreement is scored by S.
        if key == "boundary_conditions":
            if expected and not actual:
                findings.append(
                    Finding(
                        "problem_setup",
                        "REQUIRED_SETUP_MISSING",
                        "major",
                        "Required restraints are absent from the solved analysis.",
                    )
                )
            continue
        for kind in expected:
            if not actual[kind]:
                findings.append(
                    Finding(
                        "problem_setup",
                        "REQUIRED_SETUP_MISSING",
                        "major",
                        f"Required {kind} {label} is absent from the solved analysis.",
                    )
                )
    return findings
