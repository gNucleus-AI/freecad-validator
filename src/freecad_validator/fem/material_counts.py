"""Material-card grouping for count-only solid assignment checks."""

import math

MATERIAL_PROPERTIES = ("E_MPa", "rho_kg_m3", "nu")


def material_signature(material):
    """Identify a normalized parameter card, independently of its object/name."""
    values = []
    for prop in MATERIAL_PROPERTIES:
        value = material.get(prop)
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
            raise ValueError(f"Invalid material property {prop}: {value!r}")
        values.append(float(value) if value is not None else None)
    return tuple(values)


def grouped_material_counts(materials):
    """Aggregate duplicate cards and return a deterministic presentation order."""
    grouped = {}
    for material in materials:
        signature = material_signature(material)
        count = material.get("body_count")
        if type(count) is not int or count < 0:
            raise ValueError(f"Invalid material body_count: {count!r}")
        if not count:
            continue
        if signature not in grouped:
            grouped[signature] = {**material, "body_count": 0}
        grouped[signature]["body_count"] += count
    return [
        grouped[key]
        for key in sorted(
            grouped,
            key=lambda values: tuple(-math.inf if value is None else value for value in values),
        )
    ]
