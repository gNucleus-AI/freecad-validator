"""Continuous setup agreement. All similarities are in [0, 1].

This initial policy uses equal setup-category weights and no configurable knobs.
Reference geometry descriptors must be extracted by the trusted adapter.
"""

import math

from freecad_validator.fem.validators import _norm_analysis


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _similarity(expected, actual):
    if not _number(expected) or not _number(actual):
        return 0.0
    if expected == actual:
        return 1.0
    return max(0.0, 1.0 - abs(expected - actual) / max(abs(expected), abs(actual)))


def _position(expected, actual, scale):
    if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
        return 0.0
    if not all(_number(x) for x in [*expected, *actual]):
        return 0.0
    return max(0.0, 1.0 - math.dist(expected, actual) / scale)


def _direction(expected, actual):
    if any(
        not isinstance(vector, (list, tuple)) or len(vector) != 3 for vector in (expected, actual)
    ):
        return 0.0
    if not all(_number(x) for x in [*expected, *actual]):
        return 0.0
    norm = math.hypot(*expected) * math.hypot(*actual)
    if norm == 0:
        return float(expected == actual)
    cosine = max(-1.0, min(1.0, sum(a * b for a, b in zip(expected, actual, strict=False)) / norm))
    return 1.0 - math.acos(cosine) / math.pi


def _assignment(matrix):
    """Maximum-weight one-to-one assignment, including unmatched zero-credit items.

    Hungarian algorithm, O(n^3), so ambiguous nearby surfaces cannot steal a
    better pairing. Padding penalizes both missing and extra setup objects.
    """
    rows = len(matrix)
    cols = len(matrix[0]) if rows else 0
    size = max(rows, cols)
    if not size:
        return 1.0
    costs = [
        [1.0 - (matrix[i][j] if i < rows and j < cols else 0.0) for j in range(size)]
        for i in range(size)
    ]
    u, v, p, way = ([0.0] * (size + 1), [0.0] * (size + 1), [0] * (size + 1), [0] * (size + 1))
    for row in range(1, size + 1):
        p[0] = row
        minimum, used = [math.inf] * (size + 1), [False] * (size + 1)
        column = 0
        while True:
            used[column] = True
            current, delta, next_column = p[column], math.inf, 0
            for j in range(1, size + 1):
                if used[j]:
                    continue
                reduced = costs[current - 1][j - 1] - u[current] - v[j]
                if reduced < minimum[j]:
                    minimum[j], way[j] = reduced, column
                if minimum[j] < delta:
                    delta, next_column = minimum[j], j
            for j in range(size + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minimum[j] -= delta
            column = next_column
            if p[column] == 0:
                break
        while column:
            previous = way[column]
            p[column] = p[previous]
            column = previous
    return max(0.0, min(1.0, 1.0 - sum(costs[p[j] - 1][j - 1] for j in range(1, size + 1)) / size))


def _collection(expected, actual, compare, scale):
    if not expected and not actual:
        return 1.0
    if not expected or not actual:
        return 0.0
    return _assignment([[compare(left, right, scale) for right in actual] for left in expected])


def _geometry(expected, actual, scale):
    values = []
    for key in ("volume_mm3", "surface_area_mm2", "area_mm2"):
        if key in expected:
            values.append(_similarity(expected[key], actual.get(key)))
    for key in ("centroid", "bbox_min", "bbox_max"):
        if key in expected:
            values.append(_position(expected[key], actual.get(key), scale))
    if "bbox_mm" in expected:
        values.append(_position(expected["bbox_mm"], actual.get("bbox_mm"), scale))
    if "normal" in expected:
        values.append(_direction(expected["normal"], actual.get("normal")))
    # Physical measurements also serve restraint/contact surfaces.
    return math.prod(values) if values else 0.0


def _geometry_setup(target, reference, candidate, scale):
    """Physical agreement times reference-FCStd topology agreement.

    Trusted extraction unwraps single-child Compounds before recording types.
    Counts and container types are proxies, not a mesh-connectivity check.
    """
    components = {}
    for key in ("num_solids", "num_compsolids"):
        expected, actual = reference.get(key), candidate.get(key)
        if type(expected) is not int or expected < (1 if key == "num_solids" else 0):
            raise ValueError(f"Reference geometry requires valid {key}")
        components[key] = 0.0
        if type(actual) is int and actual >= 0:
            components[key] = (
                min(expected, actual) / max(expected, actual) if expected or actual else 1.0
            )
    expected_types = reference.get("shape_types")
    if not isinstance(expected_types, list) or not expected_types:
        raise ValueError("Reference geometry requires normalized shape_types")
    components["container_types"] = float(expected_types == candidate.get("shape_types"))
    physical = _geometry(target, candidate, scale)
    topology = sum(components.values()) / len(components)
    return {
        "score": physical * topology,
        "physical": physical,
        "topology": topology,
        "topology_components": components,
    }


def _parameters(expected, actual):
    values = []
    for key in sorted(set(expected) | set(actual)):
        if key not in expected or key not in actual:
            values.append(0.0)
        elif _number(expected[key]):
            values.append(_similarity(expected[key], actual[key]))
        elif isinstance(expected[key], (list, tuple)):
            other = actual[key]
            values.append(
                sum(_similarity(a, b) for a, b in zip(expected[key], other, strict=False))
                / len(other)
                if isinstance(other, (list, tuple)) and other and len(other) == len(expected[key])
                else 0.0
            )
        else:
            values.append(float(expected[key] == actual[key]))
    return sum(values) / len(values) if values else 1.0


def _material(expected, actual, scale):
    keys = ("E_MPa", "nu", "rho_kg_m3", "body_count")
    values = [_similarity(expected[k], actual.get(k)) for k in keys if k in expected]
    return sum(values) / len(values) if values else 0.0


def _condition(expected, actual, scale):
    if expected.get("type") != actual.get("type"):
        return 0.0
    values = []
    for key in ("magnitude_N", "magnitude_Pa"):
        if key in expected:
            values.append(_similarity(expected[key], actual.get(key)))
    if "direction" in expected:
        values.append(_direction(expected["direction"], actual.get("direction")))
    if "centroid" in expected:
        values.append(_position(expected["centroid"], actual.get("centroid"), scale))
    if "surfaces" in expected or "surfaces" in actual:
        values.append(
            _collection(expected.get("surfaces", []), actual.get("surfaces", []), _geometry, scale)
        )
    if "parameters" in expected or "parameters" in actual:
        values.append(_parameters(expected.get("parameters", {}), actual.get("parameters", {})))
    if "reversed" in expected:
        values.append(float(expected["reversed"] == actual.get("reversed")))
    return math.prod(values) if values else 1.0


def _contact(expected, actual, scale):
    if expected.get("type") != actual.get("type"):
        return 0.0
    # Master/slave roles matter to the discretized formulation; names do not.
    surfaces = [
        _collection(expected.get(side, []), actual.get(side, []), _geometry, scale)
        if expected.get(side) and actual.get(side)
        else 0.0
        for side in ("master", "slave")
    ]
    return math.prod(surfaces) * _parameters(
        expected.get("parameters", {}), actual.get("parameters", {})
    )


def _setup(target, label, candidate):
    scale = target.get("characteristic_length_mm")
    if not _number(scale) or scale <= 0:
        raise ValueError("Target characteristic_length_mm must be finite and positive")
    expected_analysis = _norm_analysis(label.get("analysis_type", ""))
    actual_analysis = _norm_analysis(candidate.get("analysis_type", ""))
    geometry = _geometry_setup(
        target, label.get("geometry", {}), candidate.get("geometry", {}), scale
    )
    details = {
        "analysis_type": float(
            bool(candidate.get("analysis_type")) and expected_analysis == actual_analysis
        ),
        "materials": _collection(
            label.get("materials") or [label.get("material", {})],
            candidate.get("materials") or [candidate.get("material", {})],
            _material,
            scale,
        ),
        "restraints": _collection(
            label.get("boundary_conditions", []),
            candidate.get("boundary_conditions", []),
            _condition,
            scale,
        ),
        "loads": _collection(label.get("loads", []), candidate.get("loads", []), _condition, scale),
        "geometry": geometry["score"],
    }
    if label.get("contacts") or candidate.get("contacts"):
        details["contacts"] = _collection(
            label.get("contacts", []), candidate.get("contacts", []), _contact, scale
        )
    return sum(details.values()) / len(details), details, geometry
