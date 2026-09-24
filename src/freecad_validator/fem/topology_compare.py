"""Topology summary comparisons for preprocessing and Boolean validation."""

import math

TOPOLOGY_FIELDS = ("num_solids", "num_compsolids", "shape_types", "num_faces", "num_edges")
CONTAINER_FIELDS = ("num_compsolids", "shape_types")
REGION_INTEGER_FIELDS = ("num_faces", "num_edges", "num_shells")
REGION_FLOAT_FIELDS = ("volume_mm3", "surface_area_mm2")
TOPOLOGY_REL_TOL = 1e-5
# Half a count keeps integer comparisons exact after tolerance normalization.
TOPOLOGY_COUNT_ABS_TOL = 0.5


def _region_difference(expected, actual):
    """Maximum tolerance-normalized difference; values <= 1 match."""
    if not isinstance(expected, dict) or not isinstance(actual, dict):
        return math.inf
    maximum = 0.0
    for field in REGION_INTEGER_FIELDS + REGION_FLOAT_FIELDS:
        expected_value, actual_value = expected.get(field), actual.get(field)
        if not isinstance(expected_value, (int, float)) or not isinstance(
            actual_value, (int, float)
        ):
            return math.inf
        if not math.isfinite(expected_value) or not math.isfinite(actual_value):
            return math.inf
        difference = abs(float(expected_value) - float(actual_value))
        if field in REGION_INTEGER_FIELDS:
            difference /= TOPOLOGY_COUNT_ABS_TOL
        else:
            difference /= max(abs(float(expected_value)), abs(float(actual_value)), 1e-9)
            difference /= TOPOLOGY_REL_TOL
        maximum = max(maximum, difference)
    return maximum


def _assign_region(index, differences, threshold, owners, visited):
    """Find an augmenting path for a one-to-one region assignment."""
    for actual, difference in enumerate(differences[index]):
        if difference > threshold or actual in visited:
            continue
        visited.add(actual)
        if actual not in owners or _assign_region(
            owners[actual], differences, threshold, owners, visited
        ):
            owners[actual] = index
            return True
    return False


def _regions_match_at_threshold(differences, threshold):
    owners = {}
    return all(
        _assign_region(index, differences, threshold, owners, set())
        for index in range(len(differences))
    )


def _region_bottleneck_difference(expected_regions, actual_regions):
    """Smallest possible maximum difference across a one-to-one assignment."""
    if not expected_regions:
        return 0.0
    differences = [
        [_region_difference(expected, actual) for actual in actual_regions]
        for expected in expected_regions
    ]
    thresholds = sorted(
        {difference for row in differences for difference in row if math.isfinite(difference)}
    )
    if not thresholds or not _regions_match_at_threshold(differences, thresholds[-1]):
        return math.inf
    lower, upper = 0, len(thresholds) - 1
    while lower < upper:
        middle = (lower + upper) // 2
        if _regions_match_at_threshold(differences, thresholds[middle]):
            upper = middle
        else:
            lower = middle + 1
    return thresholds[lower]


def topology_difference(expected, actual):
    """Minimum maximum normalized difference across one-to-one region pairings."""
    differences = [
        abs(expected[field] - actual[field]) / TOPOLOGY_COUNT_ABS_TOL
        for field in TOPOLOGY_FIELDS
        if field != "shape_types"
    ]
    # A categorical mismatch must also exceed the normalized threshold of 1.
    differences.append(0.0 if expected["shape_types"] == actual["shape_types"] else 2.0)
    expected_regions, actual_regions = expected["regions"], actual["regions"]
    differences.append(abs(len(expected_regions) - len(actual_regions)) / TOPOLOGY_COUNT_ABS_TOL)
    if len(expected_regions) != len(actual_regions):
        return max(differences)
    differences.append(_region_bottleneck_difference(expected_regions, actual_regions))
    return max(differences)


def topology_mismatches(expected, actual):
    """Full topology summary comparison for detecting unchanged source geometry."""
    mismatches = []
    for field in TOPOLOGY_FIELDS:
        if expected.get(field) != actual.get(field):
            mismatches.append(
                f"{field}: expected {expected.get(field)!r}, got {actual.get(field)!r}"
            )
    expected_regions, actual_regions = expected.get("regions"), actual.get("regions")
    if not isinstance(expected_regions, list) or not isinstance(actual_regions, list):
        mismatches.append("regions: missing or invalid region list")
        return mismatches
    if len(expected_regions) != len(actual_regions):
        mismatches.append(f"regions: expected {len(expected_regions)}, got {len(actual_regions)}")
        return mismatches
    difference = _region_bottleneck_difference(expected_regions, actual_regions)
    if difference > 1.0:
        mismatches.append(
            "regions: no one-to-one match within tolerance "
            f"(minimum normalized difference {difference:g})"
        )
    return mismatches


def container_mismatches(reference, candidate):
    """Match region count and container to the reference, allowing face subdivisions."""
    mismatches = []
    if len(reference["regions"]) != len(candidate["regions"]):
        mismatches.append(
            f"regions: expected {len(reference['regions'])}, got {len(candidate['regions'])}"
        )
    for field in CONTAINER_FIELDS:
        if reference[field] != candidate[field]:
            mismatches.append(f"{field}: expected {reference[field]!r}, got {candidate[field]!r}")
    return mismatches
