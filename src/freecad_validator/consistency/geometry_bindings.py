"""V2 parameter checks anchored to finite final-solid geometry.

A binding is trusted task data, chosen before seeing candidates. Registration
uses a common geometric datum; witness matching uses type, material side,
position and direction, never the requested parameter value. A missing witness
cannot be rescued by the legacy scalar pool. All parameters remain in scoring.
"""

from __future__ import annotations

import itertools
import math
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.optimize import linear_sum_assignment

from freecad_validator.consistency.report import ConsistencyReport, ParamFinding
from freecad_validator.measurement.spatial import SpatialBank, SpatialDatum, SpatialLocation
from freecad_validator.spec.parser import StructuredSpec


class GeometryBindingError(ValueError):
    """Invalid trusted binding configuration or unavailable spatial measurement."""


class LegacyBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["legacy"]
    reason: str = Field(min_length=1)


class Witness(SpatialLocation):
    region: Literal["material", "void", "mixed"] | None = None
    radius: float | None = Field(default=None, gt=0)


class GeometryBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    mode: Literal["geometry"]
    reason: str = Field(min_length=1)
    quantity: Literal[
        "radius", "diameter", "axial_extent", "length", "separation", "center_distance"
    ]
    witnesses: list[Witness] = Field(min_length=1, max_length=512)
    require_stock_envelope: bool = False

    @model_validator(mode="after")
    def check_quantity(self):
        if self.require_stock_envelope and self.quantity not in ("radius", "diameter"):
            raise ValueError("stock envelope requires a cylinder radius or diameter binding")
        if self.quantity in ("radius", "diameter") and any(
            w.radius is not None for w in self.witnesses
        ):
            raise ValueError("A radius/diameter target cannot also be a witness selection filter")
        allowed = {
            "radius": "cylinder",
            "diameter": "cylinder",
            "axial_extent": "cylinder",
            "length": "line",
            "separation": "plane_pair",
        }
        if self.quantity == "center_distance":
            if len(self.witnesses) != 2 or any(w.kind != "cylinder" for w in self.witnesses):
                raise ValueError("center_distance needs exactly two cylinder witnesses")
        elif any(w.kind != allowed[self.quantity] for w in self.witnesses):
            raise ValueError("witness kind does not support the requested quantity")
        return self


ParameterBinding = Annotated[LegacyBinding | GeometryBinding, Field(discriminator="mode")]


class GeometryBindingSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    version: Literal[1]
    datum: SpatialDatum | None = None
    parameters: dict[str, ParameterBinding]

    @model_validator(mode="after")
    def require_datum(self):
        if any(b.mode == "geometry" for b in self.parameters.values()) and self.datum is None:
            raise ValueError("geometry bindings require a reference datum")
        return self


def parse_bindings(raw: dict, structured: StructuredSpec) -> GeometryBindingSpec | None:
    if "geometry_bindings" not in raw:
        return None
    try:
        result = GeometryBindingSpec.model_validate(raw["geometry_bindings"])
    except ValueError as exc:
        raise GeometryBindingError(f"Invalid geometry_bindings: {exc}") from exc
    keys = (
        set(structured.scalars)
        | set(structured.counts)
        | set(structured.vectors)
        | set(structured.strings)
    )
    if set(result.parameters) != keys:
        raise GeometryBindingError(
            "geometry_bindings must annotate every parsed parameter exactly once; "
            f"missing={sorted(keys - set(result.parameters))}, extra={sorted(set(result.parameters) - keys)}"
        )
    return result


def _rotations():
    result = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            matrix = np.zeros((3, 3))
            for i, j in enumerate(permutation):
                matrix[i, j] = signs[i]
            if np.linalg.det(matrix) > 0:
                result.append(matrix)
    return result


_ROTATIONS = _rotations()


def _radial_frames(datum: SpatialDatum):
    """Fix rotation about a repeated principal axis using off-axis geometry.

    An inertia frame has arbitrary in-plane orientation when two moments are
    equal. Cylinder-center directions provide reference-independent pose seeds
    even for such gears and circular patterns. No parameter targets are used.
    """
    result = []
    for landmark in datum.landmarks:
        if landmark.kind != "cylinder":
            continue
        axis = np.asarray(landmark.direction)
        radial = np.asarray(landmark.position) - np.asarray(datum.center)
        radial -= axis * np.dot(radial, axis)
        if np.linalg.norm(radial) < datum.diagonal * 1e-6:
            continue
        radial /= np.linalg.norm(radial)
        result.append(
            (_type_key(landmark), np.column_stack((radial, np.cross(axis, radial), axis)))
        )
        if len(result) == 4:
            break
    return result


def _type_key(feature):
    return feature.kind, feature.convex


def align_datum(
    reference: SpatialDatum, candidate: SpatialDatum
) -> tuple[np.ndarray, np.ndarray, float]:
    """Resolve a common rigid pose from analytic frames and unlabelled landmarks.

    No parameter target or pass/fail result participates. Finite frame search
    handles axis signs/permutations. Translation is refined from landmark
    correspondences; witness dimensions are never used to choose a match.
    """
    if reference == candidate:
        return np.eye(3), np.zeros(3), 0.0
    rp = np.array([p.position for p in reference.landmarks])
    cp = np.array([p.position for p in candidate.landmarks])
    if not len(rp) or not len(cp):
        raise GeometryBindingError("No spatial landmarks available for the binding datum")
    rn = np.array([p.direction for p in reference.landmarks])
    cn = np.array([p.direction for p in candidate.landmarks])
    compatible = np.array(
        [[_type_key(r) == _type_key(c) for c in candidate.landmarks] for r in reference.landmarks]
    )
    diag = reference.diagonal
    best = None
    seen = set()
    rotations = [
        np.asarray(rf) @ permutation @ np.asarray(cf).T
        for rf, cf, permutation in itertools.product(reference.frames, candidate.frames, _ROTATIONS)
    ]
    for (rt, rf), (ct, cf) in itertools.product(
        _radial_frames(reference), _radial_frames(candidate)
    ):
        if rt == ct:
            for signs in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)):
                rotations.append(rf @ np.diag(signs) @ cf.T)
    for rotation in rotations:
        key = tuple(np.round(rotation.flatten(), 7))
        if key in seen:
            continue
        seen.add(key)
        translation = np.asarray(reference.center) - rotation @ np.asarray(candidate.center)
        moved_n = cn @ rotation.T
        angular = 1 - np.clip(np.abs(rn @ moved_n.T), 0, 1)
        for _ in range(4):
            moved = cp @ rotation.T + translation
            distances = np.linalg.norm(rp[:, None, :] - moved[None, :, :], axis=2)
            cost = distances + angular * diag * 0.1
            cost[~compatible] = diag * 10
            indices = cost.argmin(axis=1)
            nearest = cost[np.arange(len(rp)), indices]
            valid = nearest < diag * 0.15
            if not np.any(valid):
                break
            delta = np.median(rp[valid] - moved[indices[valid]], axis=0)
            translation += delta
            if np.linalg.norm(delta) < 1e-8:
                break
        moved = cp @ rotation.T + translation
        distances = np.linalg.norm(rp[:, None, :] - moved[None, :, :], axis=2)
        cost = distances + angular * diag * 0.1
        cost[~compatible] = diag * 10
        # Both sides matter; clipping only keeps a missing local feature from
        # pulling the datum away from the rest of the body.
        fit = float(
            (
                np.minimum(cost.min(axis=1), diag * 0.1).mean()
                + np.minimum(cost.min(axis=0), diag * 0.1).mean()
            )
            / 2
        )
        if best is None or fit < best[0] - 1e-10:
            best = fit, rotation, translation
    return best[1], best[2], best[0]


def evaluate_binding(
    binding: GeometryBinding,
    expected: float,
    bank: SpatialBank,
    rotation: np.ndarray,
    translation: np.ndarray,
    *,
    tol_scalar: float,
    tol_pos: float,
):
    candidates = [f for f in bank.features if f.kind == binding.witnesses[0].kind]
    if binding.quantity == "separation" and any(
        "plane_pair extraction unavailable" in x for x in bank.limitations
    ):
        raise GeometryBindingError("; ".join(bank.limitations))
    cost = np.full((len(binding.witnesses), max(len(candidates), len(binding.witnesses))), 1e9)
    for i, witness in enumerate(binding.witnesses):
        for j, feature in enumerate(candidates):
            if witness.convex != feature.convex or witness.region != feature.region:
                continue
            if witness.radius is not None:
                actual_radius = feature.values.get("radius")
                if (
                    actual_radius is None
                    or abs(actual_radius - witness.radius) / max(actual_radius, witness.radius)
                    > tol_scalar
                ):
                    continue
            angular = abs(float(np.dot(witness.direction, rotation @ np.array(feature.direction))))
            if angular < math.cos(math.radians(2)):
                continue
            distance = float(
                np.linalg.norm(
                    np.array(witness.position)
                    - (rotation @ np.array(feature.position) + translation)
                )
            )
            tolerance = max(1e-6, tol_pos * witness.scale)
            if distance > tolerance:
                continue
            order_quantity = {"cylinder": "radius", "plane_pair": "separation"}.get(feature.kind)
            if order_quantity:
                tolerance = max(1e-6, tol_pos * witness.scale)
                peers = [
                    f.values[order_quantity]
                    for f in candidates
                    if f.convex == feature.convex
                    and f.region == feature.region
                    and math.dist(f.position, feature.position) <= tolerance
                    and abs(np.dot(f.direction, feature.direction)) > 1 - 1e-7
                    and f.values[order_quantity] > feature.values[order_quantity] + 1e-6
                ]
                peers.sort()
                order = sum(i == 0 or v - peers[i - 1] > 1e-6 for i, v in enumerate(peers))
                if witness.coincident_order != order:
                    continue
            cost[i, j] = distance / tolerance
    rows, columns = linear_sum_assignment(cost)
    matched = {
        int(i): candidates[j] for i, j in zip(rows, columns, strict=True) if cost[i, j] < 1e8
    }
    evidence = [
        {
            "witness": i,
            "reference_location": w.model_dump(),
            "candidate": matched[i].model_dump() if i in matched else None,
        }
        for i, w in enumerate(binding.witnesses)
    ]
    if len(matched) != len(binding.witnesses):
        return (
            "not_found",
            None,
            evidence,
            "Missing corresponding final-geometry witness; legacy values cannot replace it",
        )
    if binding.quantity == "center_distance":
        values = [math.dist(matched[0].position, matched[1].position)]
    else:
        values = [matched[i].values[binding.quantity] for i in range(len(binding.witnesses))]
    ok = all(abs(v - expected) / max(abs(v), abs(expected), 1e-9) <= tol_scalar for v in values)
    if binding.require_stock_envelope:
        for feature in matched.values():
            if "outside_cylinder_volume" not in feature.values:
                return (
                    "not_found",
                    values,
                    evidence,
                    "The bound long cylindrical stock envelope is absent",
                )
            if feature.values["outside_cylinder_volume"] > max(
                1e-6, feature.values["body_volume"] * 1e-8
            ):
                return (
                    "inconsistent",
                    values,
                    evidence,
                    "Material extends outside the bound cylindrical stock envelope",
                )
    return (
        ("consistent" if ok else "inconsistent"),
        values,
        evidence,
        "Compared every bound final-geometry instance",
    )


def apply_bindings(
    report: ConsistencyReport,
    bindings: GeometryBindingSpec,
    bank: SpatialBank | None,
    structured: StructuredSpec,
    *,
    unavailable_reason: str | None = None,
    tol_scalar: float,
    tol_pos: float,
):
    geometric = {k: b for k, b in bindings.parameters.items() if b.mode == "geometry"}
    expected = {
        **structured.scalars,
        **structured.counts,
        **structured.vectors,
        **structured.strings,
    }
    details = {
        "parameters": {},
        "geometry_parameters": len(geometric),
        "legacy_parameters": len(bindings.parameters) - len(geometric),
    }
    if geometric and bank is not None and not bank.datum.landmarks and bindings.datum.landmarks:
        # A valid but featureless candidate (for example a sphere) is missing
        # the datum's physical supports; this is not a backend failure.
        bank = None
        unavailable_reason = "Candidate has no supported spatial landmarks for the binding datum"
    if geometric and bank is None and unavailable_reason is None:
        raise GeometryBindingError("V2 geometry bindings require a spatial measurement bank")
    if geometric and bank is not None:
        rotation, translation, fit = align_datum(bindings.datum, bank.datum)
        details["alignment"] = {
            "rotation": rotation.tolist(),
            "translation": translation.tolist(),
            "landmark_fit_mm": fit,
        }
    for key, binding in bindings.parameters.items():
        if binding.mode == "legacy":
            details["parameters"][key] = {"mode": "legacy", "reason": binding.reason}
            continue
        if isinstance(expected[key], (str, tuple)):
            raise GeometryBindingError(f"Non-scalar geometry binding for {key}")
        if bank is None:
            bucket, values, evidence, reason = "not_found", None, [], unavailable_reason
        else:
            bucket, values, evidence, reason = evaluate_binding(
                binding,
                float(expected[key]),
                bank,
                rotation,
                translation,
                tol_scalar=tol_scalar,
                tol_pos=tol_pos,
            )
        # Apply last, including to old 'consistent' findings. No downstream
        # legacy refinement is allowed to overwrite a bound-geometry failure.
        for name in ("consistent", "inconsistent", "not_found"):
            setattr(report, name, [f for f in getattr(report, name) if f.param != key])
        finding = ParamFinding(
            param=key,
            spec_value=expected[key],
            measured_value=values,
            unit="mm",
            feature="final_geometry:" + binding.quantity,
            reason=reason,
        )
        getattr(report, bucket).append(finding)
        details["parameters"][key] = {
            "mode": "geometry",
            "status": bucket,
            "quantity": binding.quantity,
            "reason": binding.reason,
            "evidence": evidence,
        }
    report.binding_details = details
