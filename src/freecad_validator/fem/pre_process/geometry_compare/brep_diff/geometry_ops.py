"""Shared, FreeCAD-free primitives for BREP diff methods.

Everything here operates on the frozen `BrepDocument` / `SubshapeRecord`
descriptors. The two responsibilities are (1) a similarity / tolerance model
used by every matcher and (2) the shared `classify_from_matches` that turns an
A->B correspondence into a `DiffResult` change set. Method-specific matchers
live under `methods/`.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from freecad_validator.fem.pre_process.geometry_compare.brep_diff.models import (
    SUBSHAPE_KINDS,
    BrepDocument,
    DiffConfig,
    DiffResult,
    MatchRecord,
    SubshapeRecord,
    make_empty_summary,
)

EPSILON = 1e-12


# --------------------------------------------------------------------------- #
# Tolerance model
# --------------------------------------------------------------------------- #
def document_scale(reference: BrepDocument, candidate: BrepDocument) -> float:
    """Length scale that normalises every tolerance and deviation — the REFERENCE
    (target/base) bbox diagonal only, floored at 1.0.

    Deliberately NOT `max(reference, candidate)`: the candidate is the untrusted,
    agent-produced solid, and every downstream tolerance (`sample_tolerance`,
    `linear_tolerance`, the equivalence `vol_eps`) scales with this value. Keying it off
    the candidate's own bounding box would let a candidate widen its own pass tolerance by
    inflating its bbox (e.g. a far-off stray feature) — a reward-hack vector. Scaling by the
    fixed reference removes that lever while leaving legitimate same-size edits unchanged.
    `candidate` is kept in the signature for call-site symmetry and is intentionally unused.
    """
    del candidate
    return max(reference.bbox_diagonal(), 1.0)


def linear_tolerance(
    config: DiffConfig, scale: float, a_tol: float = 0.0, b_tol: float = 0.0
) -> float:
    return max(
        config.linear_tolerance,
        config.relative_tolerance * scale,
        4.0 * max(a_tol, b_tol, 0.0),
    )


def sample_tolerance(
    config: DiffConfig, scale: float, a_tol: float = 0.0, b_tol: float = 0.0
) -> float:
    return max(
        linear_tolerance(config, scale, a_tol, b_tol),
        0.35 * config.tessellation_deflection,
    )


# --------------------------------------------------------------------------- #
# Vector helpers
# --------------------------------------------------------------------------- #
def norm3(values: tuple[float, float, float] | list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.linalg.norm(arr))


def distance3(a_values: tuple[float, float, float], b_values: tuple[float, float, float]) -> float:
    return norm3(np.asarray(a_values, dtype=np.float64) - np.asarray(b_values, dtype=np.float64))


def angle_between(
    a_values: tuple[float, float, float], b_values: tuple[float, float, float]
) -> float:
    a_vec = np.asarray(a_values, dtype=np.float64)
    b_vec = np.asarray(b_values, dtype=np.float64)
    a_norm = float(np.linalg.norm(a_vec))
    b_norm = float(np.linalg.norm(b_vec))
    if a_norm <= EPSILON or b_norm <= EPSILON:
        return 0.0
    dot = float(np.clip(np.dot(a_vec, b_vec) / (a_norm * b_norm), -1.0, 1.0))
    # Treat antiparallel normals as equal (face orientation can flip across a diff).
    return min(math.acos(dot), math.acos(float(np.clip(-dot, -1.0, 1.0))))


def relative_abs_delta(a_value: float, b_value: float) -> float:
    return abs(a_value - b_value) / max(abs(a_value), abs(b_value), 1.0)


def vector_delta(a_values: tuple[float, ...], b_values: tuple[float, ...], scale: float) -> float:
    return (
        norm3(np.asarray(a_values, dtype=np.float64) - np.asarray(b_values, dtype=np.float64))
        / scale
    )


def sample_distance_stats(
    a_samples: tuple[tuple[float, float, float], ...],
    b_samples: tuple[tuple[float, float, float], ...],
) -> dict[str, float]:
    if not a_samples or not b_samples:
        return {"sample_mean": 0.0, "sample_p95": 0.0, "sample_max": 0.0}
    a_array = np.asarray(a_samples, dtype=np.float64)
    b_array = np.asarray(b_samples, dtype=np.float64)
    a_to_b = cKDTree(b_array).query(a_array, k=1)[0]
    b_to_a = cKDTree(a_array).query(b_array, k=1)[0]
    combined = np.concatenate([a_to_b, b_to_a])
    return {
        "sample_mean": float(np.mean(combined)),
        "sample_p95": float(np.percentile(combined, 95)),
        "sample_max": float(np.max(combined)),
    }


# --------------------------------------------------------------------------- #
# Seam-invariant surface distance (area-uniform mesh sampling + point-to-triangle)
# --------------------------------------------------------------------------- #
# The per-face UV-grid samples on `SubshapeRecord.samples` land on different physical
# points when a face's seam/parameterisation differs, so a point-to-point nearest-neighbour
# distance over them penalises two representations of the *same* surface (the seam artifact).
# These primitives instead sample points AREA-UNIFORMLY over a watertight tessellation and
# measure each point's distance to the OTHER solid's triangulated surface (not to its sampled
# points), so the statistic depends only on the occupied surface, not on how it was
# parameterised or how densely each face was gridded. No ICP alignment is applied.
def closest_point_on_triangles(
    p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> np.ndarray:
    """Closest point on each triangle (a, b, c) to point p; all inputs broadcast to (..., 3).

    Ericson, *Real-Time Collision Detection* §5.1.5 — the region-based barycentric
    solution, fully vectorised with `np.where` over the seven Voronoi regions."""
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = np.sum(ab * ap, axis=-1)
    d2 = np.sum(ac * ap, axis=-1)
    bp = p - b
    d3 = np.sum(ab * bp, axis=-1)
    d4 = np.sum(ac * bp, axis=-1)
    cp = p - c
    d5 = np.sum(ab * cp, axis=-1)
    d6 = np.sum(ac * cp, axis=-1)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    denom = np.where(np.abs(va + vb + vc) < EPSILON, 1.0, va + vb + vc)
    v = vb / denom
    w = vc / denom
    result = a + v[..., None] * ab + w[..., None] * ac  # face-interior projection
    # Edge regions (clamped 1-D projections), then vertex regions win last.
    denom_ab = np.where(np.abs(d1 - d3) < EPSILON, 1.0, d1 - d3)
    result = np.where(
        ((vc <= 0) & (d1 >= 0) & (d3 <= 0))[..., None],
        a + np.clip(d1 / denom_ab, 0.0, 1.0)[..., None] * ab,
        result,
    )
    denom_ac = np.where(np.abs(d2 - d6) < EPSILON, 1.0, d2 - d6)
    result = np.where(
        ((vb <= 0) & (d2 >= 0) & (d6 <= 0))[..., None],
        a + np.clip(d2 / denom_ac, 0.0, 1.0)[..., None] * ac,
        result,
    )
    denom_bc = np.where(np.abs((d4 - d3) + (d5 - d6)) < EPSILON, 1.0, (d4 - d3) + (d5 - d6))
    result = np.where(
        ((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0))[..., None],
        b + np.clip((d4 - d3) / denom_bc, 0.0, 1.0)[..., None] * (c - b),
        result,
    )
    result = np.where(((d1 <= 0) & (d2 <= 0))[..., None], a, result)
    result = np.where(((d3 >= 0) & (d4 <= d3))[..., None], b, result)
    result = np.where(((d6 >= 0) & (d5 <= d6))[..., None], c, result)
    return result


def area_uniform_samples(verts: np.ndarray, tris: np.ndarray, count: int, seed: int) -> np.ndarray:
    """`count` points sampled uniformly by area over the triangle mesh (verts, tris).

    Deterministic for a fixed `seed` (a `default_rng` with no wall-clock entropy), so a
    re-score reproduces the same cloud. Returns an (N, 3) float64 array,
    empty when the mesh has no positive area."""
    if count <= 0 or len(tris) == 0:
        return np.empty((0, 3), dtype=np.float64)
    v = verts[tris].astype(np.float64)  # (T, 3, 3)
    areas = 0.5 * np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1)
    total = float(areas.sum())
    if total <= 0.0:
        return np.empty((0, 3), dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(v), size=count, p=areas / total)
    r1 = np.sqrt(rng.random(count))
    r2 = rng.random(count)
    w0 = (1.0 - r1)[:, None]
    w1 = (r1 * (1.0 - r2))[:, None]
    w2 = (r1 * r2)[:, None]
    return w0 * v[idx, 0] + w1 * v[idx, 1] + w2 * v[idx, 2]


# Point-to-triangle work is done in (points x triangles) blocks; this caps how many
# (point, triangle) pairs one block may hold so the float64 temporaries stay bounded
# (~72 bytes/pair before intermediates). Purely a memory/throughput knob — results are
# identical for any value.
PAIR_BUDGET = 2_000_000


def point_to_mesh_distances(
    points: np.ndarray, verts: np.ndarray, tris: np.ndarray, k: int = 16
) -> np.ndarray:
    """Exact nearest-surface distances, evaluating all triangles in bounded blocks.
    Centroid-based pruning can miss large nearby facets on uneven tessellations.
    """
    del k
    if len(points) == 0:
        return np.zeros(0, dtype=np.float64)
    if len(tris) == 0:
        return np.full(len(points), np.inf, dtype=np.float64)
    tri_set = verts[tris].astype(np.float64)  # (T, 3, 3)
    out = np.full(len(points), np.inf, dtype=np.float64)
    per_chunk = max(1, PAIR_BUDGET // len(tri_set))
    for start in range(0, len(points), per_chunk):
        chunk = points[start : start + per_chunk]
        p = chunk[:, None, :]
        closest = closest_point_on_triangles(
            p, tri_set[None, :, 0, :], tri_set[None, :, 1, :], tri_set[None, :, 2, :]
        )
        out[start : start + per_chunk] = np.linalg.norm(p - closest, axis=-1).min(axis=1)
    return out


def surface_distance_stats(
    verts_a: np.ndarray,
    tris_a: np.ndarray,
    verts_b: np.ndarray,
    tris_b: np.ndarray,
    count: int,
    seed: int,
) -> dict[str, float] | None:
    """Symmetric point-cloud-to-surface distance stats between two tessellated solids.

    Area-uniformly samples each solid, measures every sample's distance to the *other*
    solid's surface, and pools both directions — the seam-invariant analogue of
    `sample_distance_stats`. Returns None if either mesh yields no samples (caller falls
    back to the legacy per-face sampling)."""
    pa = area_uniform_samples(verts_a, tris_a, count, seed)
    pb = area_uniform_samples(verts_b, tris_b, count, seed)
    if len(pa) == 0 or len(pb) == 0:
        return None
    combined = np.concatenate(
        [
            point_to_mesh_distances(pa, verts_b, tris_b),
            point_to_mesh_distances(pb, verts_a, tris_a),
        ]
    )
    return {
        "sample_mean": float(np.mean(combined)),
        "sample_p95": float(np.percentile(combined, 95)),
        "sample_max": float(np.max(combined)),
    }


# --------------------------------------------------------------------------- #
# Pairwise similarity
# --------------------------------------------------------------------------- #
def geometry_deltas(
    a_entity: SubshapeRecord,
    b_entity: SubshapeRecord,
    scale: float,
    include_samples: bool,
) -> dict[str, float]:
    deltas = {
        "center": distance3(a_entity.center, b_entity.center),
        "center_norm": distance3(a_entity.center, b_entity.center) / scale,
        "measure_rel": relative_abs_delta(a_entity.measure, b_entity.measure),
        "bbox_size_norm": vector_delta(a_entity.bbox_size(), b_entity.bbox_size(), scale),
        "bbox_min_norm": vector_delta(a_entity.bbox_min, b_entity.bbox_min, scale),
        "orientation_angle": angle_between(a_entity.orientation, b_entity.orientation),
        "type_mismatch": 0.0 if a_entity.geometry_type == b_entity.geometry_type else 1.0,
    }
    scalar_keys = sorted(set(a_entity.scalars) | set(b_entity.scalars))
    if scalar_keys:
        deltas["scalar_rel"] = sum(
            relative_abs_delta(a_entity.scalars.get(key, 0.0), b_entity.scalars.get(key, 0.0))
            for key in scalar_keys
        ) / len(scalar_keys)
    else:
        deltas["scalar_rel"] = 0.0
    if include_samples:
        deltas.update(sample_distance_stats(a_entity.samples, b_entity.samples))
        deltas["sample_p95_norm"] = deltas["sample_p95"] / scale
    else:
        deltas["sample_mean"] = 0.0
        deltas["sample_p95"] = 0.0
        deltas["sample_max"] = 0.0
        deltas["sample_p95_norm"] = 0.0
    return deltas


def topology_degree_delta(a_entity: SubshapeRecord, b_entity: SubshapeRecord) -> float:
    total = 0.0
    for kind in SUBSHAPE_KINDS:
        total += abs(a_entity.degree(kind) - b_entity.degree(kind)) / max(
            a_entity.degree(kind), b_entity.degree(kind), 1
        )
    return total / len(SUBSHAPE_KINDS)


def geometry_similarity(
    a_entity: SubshapeRecord,
    b_entity: SubshapeRecord,
    scale: float,
    include_samples: bool = False,
    topology_weight: float = 0.0,
    topology_score: float = 0.0,
) -> tuple[float, dict[str, float]]:
    deltas = geometry_deltas(a_entity, b_entity, scale, include_samples)
    intrinsic_cost = (
        0.22 * min(deltas["measure_rel"], 1.0)
        + 0.20 * min(deltas["bbox_size_norm"], 1.0)
        + 0.20 * min(deltas["center_norm"], 1.0)
        + 0.12 * min(deltas["orientation_angle"] / math.pi, 1.0)
        + 0.10 * min(deltas["scalar_rel"], 1.0)
        + 0.08 * deltas["type_mismatch"]
        + 0.08 * topology_degree_delta(a_entity, b_entity)
    )
    if include_samples:
        intrinsic_cost = 0.75 * intrinsic_cost + 0.25 * min(deltas["sample_p95_norm"], 1.0)
    intrinsic_score = max(0.0, 1.0 - intrinsic_cost)
    if topology_weight > 0.0:
        score = (1.0 - topology_weight) * intrinsic_score + topology_weight * topology_score
    else:
        score = intrinsic_score
    return max(0.0, min(1.0, score)), deltas


# --------------------------------------------------------------------------- #
# Hashing / dedupe / lookup
# --------------------------------------------------------------------------- #


def dedupe_matches(matches: list[MatchRecord] | tuple[MatchRecord, ...]) -> tuple[MatchRecord, ...]:
    """Keep a 1-to-1 matching greedily by descending score (highest-confidence
    correspondence wins when several propose the same A-id or B-id)."""
    by_kind_a: set[tuple[str, str]] = set()
    by_kind_b: set[tuple[str, str]] = set()
    kept: list[MatchRecord] = []
    for match in sorted(matches, key=lambda item: item.score, reverse=True):
        a_key = (match.kind, match.a_id)
        b_key = (match.kind, match.b_id)
        if a_key in by_kind_a or b_key in by_kind_b:
            continue
        by_kind_a.add(a_key)
        by_kind_b.add(b_key)
        kept.append(match)
    return tuple(sorted(kept, key=lambda item: (item.kind, item.a_id, item.b_id)))


def match_lookup(matches: tuple[MatchRecord, ...]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {kind: {} for kind in SUBSHAPE_KINDS}
    for match in matches:
        result[match.kind][match.a_id] = match.b_id
    return result


def reverse_match_lookup(matches: tuple[MatchRecord, ...]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {kind: {} for kind in SUBSHAPE_KINDS}
    for match in matches:
        result[match.kind][match.b_id] = match.a_id
    return result


def all_samples(
    document: BrepDocument, kind: str | None = None
) -> tuple[tuple[float, float, float], ...]:
    samples: list[tuple[float, float, float]] = []
    kinds = SUBSHAPE_KINDS if kind is None else (kind,)
    for entity_kind in kinds:
        for entity in document.entities(entity_kind):
            samples.extend(entity.samples)
    return tuple(samples)


def _samples_tree(document: BrepDocument) -> cKDTree | None:
    samples = all_samples(document)
    if not samples:
        return None
    return cKDTree(np.asarray(samples, dtype=np.float64))


def entity_boundary_change_fraction(
    entity: SubshapeRecord,
    other_tree: cKDTree | None,
    tolerance: float,
) -> float:
    """Fraction of an entity's samples that lie OFF the other document's surface
    (a trim change). `other_tree` is a KDTree of the other document's full sample
    cloud, built once by the caller — rebuilding it per entity is O(n) per pair and
    dominates runtime on the gear."""
    if not entity.samples:
        return 0.0
    if other_tree is None:
        return 1.0
    distances = other_tree.query(np.asarray(entity.samples, dtype=np.float64), k=1)[0]
    return float(np.count_nonzero(distances > tolerance) / len(distances))


# --------------------------------------------------------------------------- #
# Classification (matches -> DiffResult)
# --------------------------------------------------------------------------- #
# Relative tolerance on EXACT extracted attributes (area, length, radius, ...). These
# are read from OCCT and are method-independent: a truly-unchanged subshape reproduces
# them to ~1e-12, so a tight bound ignores numerical noise but detects small edits.
MEASURE_REL_TOL = 1e-5


def is_modified_pair(
    a_entity: SubshapeRecord,
    b_entity: SubshapeRecord,
    match: MatchRecord,
    config: DiffConfig,
    scale: float,
) -> bool:
    """Classify a matched pair as modified vs unchanged from GEOMETRY, not from the
    method's matching confidence. Exact extracted attributes (type, measure, scalars,
    orientation) use tight tolerances; positional signals (centroid, sampled Hausdorff)
    use a registration/tessellation-tolerant bound so an ICP-aligned or sampled match is
    not spuriously flagged modified by sub-tessellation noise."""
    sample_tol = sample_tolerance(config, scale, a_entity.tolerance, b_entity.tolerance)
    # exact, method-independent attributes
    if a_entity.geometry_type != b_entity.geometry_type:
        return True
    if relative_abs_delta(a_entity.measure, b_entity.measure) > MEASURE_REL_TOL:
        return True
    for key in set(a_entity.scalars) | set(b_entity.scalars):
        if (
            relative_abs_delta(a_entity.scalars.get(key, 0.0), b_entity.scalars.get(key, 0.0))
            > MEASURE_REL_TOL
        ):
            return True
    if match.deltas.get("orientation_angle", 0.0) > config.angular_tolerance_rad:
        return True
    # positional signals — tolerant of registration / tessellation noise
    if match.deltas.get("center", 0.0) > sample_tol:
        return True
    if match.deltas.get("sample_p95", 0.0) > sample_tol:
        return True
    if (
        max(
            match.deltas.get("a_boundary_change_fraction", 0.0),
            match.deltas.get("b_boundary_change_fraction", 0.0),
        )
        > 0.25
    ):
        return True
    return False


def classify_from_matches(
    reference: BrepDocument,
    candidate: BrepDocument,
    matches: tuple[MatchRecord, ...],
    config: DiffConfig,
    method: str,
    warnings: tuple[str, ...] = (),
    overlap_override: bool = False,
) -> DiffResult:
    scale = document_scale(reference, candidate)
    matches = dedupe_matches(matches)
    summary = make_empty_summary()
    removed: list[dict[str, Any]] = []
    added: list[dict[str, Any]] = []
    modified: list[dict[str, Any]] = []
    unchanged: list[dict[str, Any]] = []
    forward = match_lookup(matches)
    reverse = reverse_match_lookup(matches)
    match_by_key = {match.key(): match for match in matches}
    reference_tree = _samples_tree(reference) if overlap_override else None
    candidate_tree = _samples_tree(candidate) if overlap_override else None

    for kind in SUBSHAPE_KINDS:
        a_map = reference.entity_map(kind)
        b_map = candidate.entity_map(kind)
        for a_id, a_entity in a_map.items():
            if a_id in forward[kind]:
                continue
            removed.append(
                {
                    "kind": kind,
                    "a": a_entity.public_summary(),
                    "geometry_type": a_entity.geometry_type,
                    "reason": "no matched candidate subshape",
                }
            )
            summary["removed"][kind] += 1
        for b_id, b_entity in b_map.items():
            if b_id in reverse[kind]:
                continue
            added.append(
                {
                    "kind": kind,
                    "b": b_entity.public_summary(),
                    "geometry_type": b_entity.geometry_type,
                    "reason": "no matched reference subshape",
                }
            )
            summary["added"][kind] += 1
        for a_id, b_id in forward[kind].items():
            a_entity = a_map[a_id]
            b_entity = b_map[b_id]
            match = match_by_key[(kind, a_id, b_id)]
            changed = is_modified_pair(a_entity, b_entity, match, config, scale)
            deltas = dict(match.deltas)
            if overlap_override:
                tol = sample_tolerance(config, scale, a_entity.tolerance, b_entity.tolerance)
                a_fraction = entity_boundary_change_fraction(a_entity, candidate_tree, tol)
                b_fraction = entity_boundary_change_fraction(b_entity, reference_tree, tol)
                changed = changed or max(a_fraction, b_fraction) > 0.25
                deltas["a_boundary_change_fraction"] = a_fraction
                deltas["b_boundary_change_fraction"] = b_fraction
            row = {
                "kind": kind,
                "a": a_entity.public_summary(),
                "b": b_entity.public_summary(),
                "score": match.score,
                "reason": match.reason,
                "deltas": deltas,
            }
            if changed:
                modified.append(row)
                summary["modified"][kind] += 1
            else:
                unchanged.append(row)
                summary["unchanged"][kind] += 1

    match_rows = [
        {
            "kind": match.kind,
            "a_id": match.a_id,
            "b_id": match.b_id,
            "score": match.score,
            "reason": match.reason,
            "deltas": dict(match.deltas),
        }
        for match in matches
    ]
    return DiffResult(
        method=method,
        config=asdict(config),
        reference=reference.path,
        candidate=candidate.path,
        freecad_version=reference.freecad_version or candidate.freecad_version,
        occt_version=reference.occt_version or candidate.occt_version,
        summary=summary,
        removed=tuple(removed),
        added=tuple(added),
        modified=tuple(modified),
        unchanged=tuple(unchanged),
        matches=tuple(match_rows),
        warnings=warnings,
    )


# Above this many candidate pairs, the dense Hungarian (O(n^3)) and the full Python
# cost loop are too slow (the gear has 1233 edges), so switch to centroid-pruned
# greedy matching. Below it, keep the exact dense assignment.
DENSE_ASSIGNMENT_CAP = 40_000
PRUNE_CANDIDATES = 24


def _finalize_matches(
    pairs: list[tuple[int, int, float, float]],
    a_entities: list[SubshapeRecord],
    b_entities: list[SubshapeRecord],
    scale: float,
    kind: str,
    reason: str,
    support_fn,
) -> list[MatchRecord]:
    matches: list[MatchRecord] = []
    for i, j, score, support in pairs:
        _, deltas = geometry_similarity(a_entities[i], b_entities[j], scale, include_samples=True)
        if support_fn is not None:
            deltas["topology_support"] = support
        matches.append(
            MatchRecord(
                kind=kind,
                a_id=a_entities[i].id,
                b_id=b_entities[j].id,
                score=score,
                reason=reason,
                deltas=deltas,
            )
        )
    return matches


def _dense_assignment(a_entities, b_entities, scale, min_score, topology_weight, support_fn):
    cost = np.ones((len(a_entities), len(b_entities)), dtype=np.float64)
    supports: dict[tuple[int, int], float] = {}
    for i, a in enumerate(a_entities):
        for j, b in enumerate(b_entities):
            support = support_fn(a, b) if support_fn is not None else 0.0
            score, _ = geometry_similarity(
                a,
                b,
                scale,
                include_samples=False,
                topology_weight=topology_weight,
                topology_score=support,
            )
            cost[i, j] = 1.0 - score
            supports[(i, j)] = support
    rows, cols = linear_sum_assignment(cost)
    pairs = []
    for i, j in zip(rows, cols, strict=False):
        score = 1.0 - float(cost[i, j])
        if score >= min_score:
            pairs.append((int(i), int(j), score, supports[(i, j)]))
    return pairs


def _greedy_pruned_assignment(
    a_entities, b_entities, scale, min_score, topology_weight, support_fn
):
    a_centroids = np.asarray([e.center for e in a_entities], dtype=np.float64)
    b_centroids = np.asarray([e.center for e in b_entities], dtype=np.float64)
    k = min(len(b_entities), PRUNE_CANDIDATES)
    _, neighbors = cKDTree(b_centroids).query(a_centroids, k=k)
    if neighbors.ndim == 1:
        neighbors = neighbors[:, None]
    scored: list[tuple[float, int, int, float]] = []
    for i, a in enumerate(a_entities):
        for j in neighbors[i]:
            j = int(j)
            support = support_fn(a, b_entities[j]) if support_fn is not None else 0.0
            score, _ = geometry_similarity(
                a,
                b_entities[j],
                scale,
                include_samples=False,
                topology_weight=topology_weight,
                topology_score=support,
            )
            if score >= min_score:
                scored.append((score, i, j, support))
    scored.sort(reverse=True)
    used_a: set[int] = set()
    used_b: set[int] = set()
    pairs = []
    for score, i, j, support in scored:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        pairs.append((i, j, score, support))
    return pairs


def optimal_assignment(
    a_entities: list[SubshapeRecord],
    b_entities: list[SubshapeRecord],
    scale: float,
    kind: str,
    reason: str,
    min_score: float,
    topology_weight: float = 0.0,
    support_fn=None,
) -> list[MatchRecord]:
    """1-to-1 matching between two entity lists.

    The matching cost uses only the cheap analytic descriptor (no sampled-Hausdorff);
    the expensive sampled deltas needed for classification are recomputed only for the
    accepted pairs. Small problems use an exact dense Hungarian assignment; large ones
    (the gear) use centroid-pruned greedy matching so runtime stays bounded.
    `support_fn(a, b) -> float` optionally injects a topology-vote score.
    """
    if not a_entities or not b_entities:
        return []
    if len(a_entities) * len(b_entities) <= DENSE_ASSIGNMENT_CAP:
        pairs = _dense_assignment(
            a_entities, b_entities, scale, min_score, topology_weight, support_fn
        )
    else:
        pairs = _greedy_pruned_assignment(
            a_entities, b_entities, scale, min_score, topology_weight, support_fn
        )
    return _finalize_matches(pairs, a_entities, b_entities, scale, kind, reason, support_fn)
