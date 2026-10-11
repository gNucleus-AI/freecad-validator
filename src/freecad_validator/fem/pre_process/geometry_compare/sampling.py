"""Shared deterministic spatial probes and read-only point-in-solid queries."""

import FreeCAD

from freecad_validator.fem.pre_process.occt_solid_occupancy import (
    solid_occupancies,
)


def _van_der_corput(index: int, base: int) -> float:
    value = 0.0
    denominator = 1.0
    while index:
        index, remainder = divmod(index, base)
        denominator *= base
        value += remainder / denominator
    return value


def _halton_points(count, bbox):
    mins, maxs = bbox
    sizes = tuple(maxs[axis] - mins[axis] for axis in range(3))
    points = []
    for index in range(1, count + 1):
        unit = (_van_der_corput(index, 2), _van_der_corput(index, 3), _van_der_corput(index, 5))
        points.append(tuple(mins[axis] + sizes[axis] * unit[axis] for axis in range(3)))
    return points


def _inside(shape, point, tolerance):
    # OCCT may derive BoundBox from cached tessellation, slightly inside a curved
    # surface. Do not reject a point using that box before the solid classifier.
    return bool(shape.isInside(FreeCAD.Vector(*point), tolerance, True))


def _inside_many(shape, points, tolerance):
    """Batch large native-solid queries; keep small and virtual queries in-process.

    Both paths include boundary points and use the same tolerance. The cutoff
    only avoids child-process startup for short batches; it changes no probes.
    """
    if len(points) < 4096 or not hasattr(shape, "exportBrepToString"):
        return [_inside(shape, point, tolerance) for point in points]
    return solid_occupancies([shape.exportBrepToString()], points, tolerance)[0]


def point_in_bounds(point, box, tolerance):
    """Cull queries only with a precomputed analytical (not triangulation) box."""
    return all(
        low - tolerance <= value <= high + tolerance
        for value, low, high in zip(
            point,
            (box.XMin, box.YMin, box.ZMin),
            (box.XMax, box.YMax, box.ZMax),
            strict=False,
        )
    )


def _occupied_probes(shape, initial_count):
    """Reuse the same probes for correspondence; never cache input-query results globally."""
    box = (
        shape.optimalBoundingBox(False, False)
        if hasattr(shape, "optimalBoundingBox")
        else shape.BoundBox
    )
    bounds = ((box.XMin, box.YMin, box.ZMin), (box.XMax, box.YMax, box.ZMax))
    count = initial_count
    occupied = []
    previous = 0
    while True:
        points = _halton_points(count, bounds)
        batch = points[previous:]
        occupied.extend(
            point
            for point, inside in zip(batch, _inside_many(shape, batch, 0.0), strict=True)
            if inside
        )
        if len(occupied) >= 64:
            return occupied, count
        if count == 65536:
            raise ValueError("Insufficient occupied probes for body correspondence")
        previous, count = count, min(count * 4, 65536)
