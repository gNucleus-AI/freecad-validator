"""Native V2 geometry witnesses survive poses and expose locally wrong holes."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from freecad_validator._freecad_loader import import_freecad
from freecad_validator.consistency.geometry_bindings import (
    GeometryBinding,
    align_datum,
    evaluate_binding,
)
from freecad_validator.measurement import spatial
from freecad_validator.measurement.spatial import extract_spatial, location

pytestmark = pytest.mark.needs_freecad


def make_shape(Part, App, wrong=False):
    s = Part.makeBox(60, 35, 10)
    for x, y, radius in [(9, 8, 3), (24, 23, 3), (48, 11, 4)]:
        if wrong and x == 24:
            radius = 2
        s = s.cut(Part.makeCylinder(radius, 10, App.Vector(x, y, 0)))
    return s.Solids[0]


def test_native_positions_repeated_holes_and_rigid_pose():
    App = import_freecad()
    Part = importlib.import_module("Part")
    shape = make_shape(Part, App)
    reference = extract_spatial(shape)
    holes = [f for f in reference.features if f.kind == "cylinder" and f.values["diameter"] == 6]
    assert len(holes) == 2 and holes[0].position != holes[1].position
    cfg = GeometryBinding(
        mode="geometry",
        quantity="diameter",
        reason="Two specified mounting holes",
        witnesses=[location(f).model_dump() for f in holes],
    )
    for wrong in [False, True]:
        candidate = make_shape(Part, App, wrong)
        candidate.rotate(App.Vector(), App.Vector(1, 2, 3), 37)
        candidate.translate(App.Vector(200, -91, 12))
        measured = extract_spatial(candidate)
        rotation, translation, fit = align_datum(reference.datum, measured.datum)
        status, values, _, _ = evaluate_binding(
            cfg, 6, measured, rotation, translation, tol_scalar=0.01, tol_pos=0.01
        )
        assert status == ("inconsistent" if wrong else "consistent"), (status, values, fit)
        assert sorted(values) == ([4, 6] if wrong else [6, 6])
    reordered = Part.makeSolid(Part.makeShell(list(reversed(shape.Faces))))
    measured = extract_spatial(reordered)
    rotation, translation, _ = align_datum(reference.datum, measured.datum)
    assert (
        evaluate_binding(cfg, 6, measured, rotation, translation, tol_scalar=0.01, tol_pos=0.01)[0]
        == "consistent"
    )


def test_plane_pair_reports_actual_material_and_void_support():
    App = import_freecad()
    Part = importlib.import_module("Part")
    shape = Part.makeBox(30, 20, 10).cut(Part.makeBox(10, 20, 7, App.Vector(10, 0, 3)))
    bank = extract_spatial(shape)
    pairs = [f for f in bank.features if f.kind == "plane_pair"]
    assert any(f.region == "material" and abs(f.values["separation"] - 3) < 1e-6 for f in pairs)
    assert any(f.region == "void" and abs(f.values["separation"] - 10) < 1e-6 for f in pairs)


@pytest.mark.parametrize("region", ["material", "void"])
@pytest.mark.parametrize("minority_fraction", [0, 5e-7, 2e-6, 2e-5])
@pytest.mark.parametrize("moved", [False, True])
def test_wall_region_cutoff_preserves_small_physical_features(region, minority_fraction, moved):
    App = import_freecad()
    Part = importlib.import_module("Part")
    shape = Part.makeBox(30, 20, 10)
    if region == "material":
        prism = shape.copy()
        separation = 30
    else:
        prism = Part.makeBox(10, 20, 7, App.Vector(10, 0, 3))
        shape = shape.cut(prism)
        separation = 10
    if minority_fraction:
        # An off-center cavity or floor-attached rib misses the centroid probes.
        # Its real volume tests the boolean classification on both sides of the
        # cutoff; a 1e-4 relaxation would incorrectly retain the mixed supports.
        thickness = minority_fraction * prism.Volume / 5
        base = App.Vector(2, 2, 2) if region == "material" else App.Vector(12, 2, 3)
        defect = Part.makeBox(1, thickness, 5, base)
        shape = shape.cut(defect) if region == "material" else shape.fuse(defect)
    direction = App.Vector(1, 0, 0)
    if moved:
        rotation = App.Rotation(App.Vector(1, 2, 3), 37)
        direction = rotation.multVec(direction)
        for solid in (shape, prism):
            solid.rotate(App.Vector(), App.Vector(1, 2, 3), 37)
            solid.translate(App.Vector(123, -97, 41))
    assert shape.isValid() and len(shape.Solids) == 1
    occupied = shape.common(prism).Volume / prism.Volume
    measured_fraction = 1 - occupied if region == "material" else occupied
    assert measured_fraction == pytest.approx(minority_fraction, abs=1e-10)
    bank = extract_spatial(shape)
    assert not bank.limitations
    supports = [
        f
        for f in bank.features
        if f.kind == "plane_pair"
        and abs(f.values["separation"] - separation) < 1e-6
        and abs(np.dot(f.direction, tuple(direction))) > 1 - 1e-7
        and np.linalg.norm(np.asarray(f.position) - tuple(prism.CenterOfMass)) < 1e-6
    ]
    if minority_fraction < 1e-6:
        assert len(supports) == 1 and supports[0].region == region
    else:
        assert not supports


@pytest.mark.parametrize("reason", ["disabled", "timeout"])
def test_unfinished_native_wall_pairs_are_explicitly_unavailable(monkeypatch, reason):
    import_freecad()
    Part = importlib.import_module("Part")
    calls = []

    def timeout(shape):
        calls.append(shape)
        raise spatial._PlanePairTimeout("Wall-pair measurement exceeded its deadline")

    monkeypatch.setattr(spatial, "_run_plane_pairs", timeout)
    bank = extract_spatial(Part.makeBox(10, 5, 3), include_plane_pairs=reason != "disabled")
    assert bool(calls) == (reason == "timeout")
    assert any(message.startswith(spatial.PLANE_PAIR_UNAVAILABLE) for message in bank.limitations)
    assert not any(f.kind == "plane_pair" for f in bank.features)
    assert any(f.kind == "plane" for f in bank.features)
    cfg = GeometryBinding(
        mode="geometry",
        quantity="separation",
        reason="Box thickness",
        witnesses=[
            dict(
                kind="plane_pair",
                position=(5, 2.5, 1.5),
                direction=(0, 0, 1),
                scale=10,
                region="material",
            )
        ],
    )
    with pytest.raises(spatial.SpatialMeasurementError, match=spatial.PLANE_PAIR_UNAVAILABLE):
        evaluate_binding(cfg, 3, bank, np.eye(3), np.zeros(3), tol_scalar=0.01, tol_pos=0.01)


def test_wall_pair_process_preserves_rotated_finite_supports():
    App = import_freecad()
    Part = importlib.import_module("Part")
    shape = Part.makeBox(30, 20, 10).cut(Part.makeBox(10, 20, 7, App.Vector(10, 0, 3)))
    reference = extract_spatial(shape)
    candidate = shape.copy()
    candidate.rotate(App.Vector(), App.Vector(1, 2, 3), 37)
    candidate.translate(App.Vector(123, -97, 41))
    measured = extract_spatial(candidate)
    rotation, translation, _ = align_datum(reference.datum, measured.datum)
    for region, separation in [("material", 3), ("void", 10)]:
        supports = [
            f
            for f in reference.features
            if f.kind == "plane_pair"
            and f.region == region
            and abs(f.values["separation"] - separation) < 1e-6
        ]
        assert supports
        cfg = GeometryBinding(
            mode="geometry",
            quantity="separation",
            reason="Corresponding finite walls",
            witnesses=[dict(**location(f).model_dump(), region=region) for f in supports],
        )
        status, values, _, _ = evaluate_binding(
            cfg, separation, measured, rotation, translation, tol_scalar=0.01, tol_pos=0.01
        )
        assert status == "consistent" and values == pytest.approx([separation] * len(supports))


def test_disconnected_plane_overlap_keeps_separate_finite_supports():
    App = import_freecad()
    Part = importlib.import_module("Part")
    shape = Part.makeBox(40, 30, 10)
    shape = shape.cut(Part.makeBox(20, 25, 3, App.Vector(10, 0, 7)))
    shape = shape.cut(Part.makeBox(20, 25, 3, App.Vector(10, 5, 0)))
    bank = extract_spatial(shape)
    supports = [
        f
        for f in bank.features
        if f.kind == "plane_pair"
        and f.region == "material"
        and abs(f.values["separation"] - 10) < 1e-6
        and abs(f.direction[2]) > 0.999
    ]
    assert len(supports) == 2
    assert sorted(round(f.position[0], 6) for f in supports) == [5, 35]


def test_concentric_arcs_use_radial_order_without_target_size_matching():
    App = import_freecad()
    Part = importlib.import_module("Part")

    def lobed(radius):
        central = Part.makeCylinder(8, 6)
        lobes = Part.makeCylinder(radius, 6).common(Part.makeBox(24, 6, 6, App.Vector(-12, -3, 0)))
        return central.fuse(lobes).removeSplitter()

    reference = extract_spatial(lobed(10))
    arcs = [f for f in reference.features if f.kind == "cylinder" and f.convex]
    assert sorted((f.values["radius"], f.coincident_order) for f in arcs) == [(8, 1), (10, 0)]
    cfg = GeometryBinding(
        mode="geometry",
        quantity="diameter",
        reason="Outer lobes",
        witnesses=[location(next(f for f in arcs if f.coincident_order == 0)).model_dump()],
    )
    for radius in [10, 9, 8]:
        bank = extract_spatial(lobed(radius))
        rotation, translation, _ = align_datum(reference.datum, bank.datum)
        result = evaluate_binding(
            cfg, 20, bank, rotation, translation, tol_scalar=0.01, tol_pos=0.01
        )
        assert result[0] == ("consistent" if radius == 10 else "inconsistent"), result
        assert result[1] == [radius * 2]


def test_degenerate_sphere_edges_do_not_prevent_spatial_extraction():
    import_freecad()
    Part = importlib.import_module("Part")
    assert extract_spatial(Part.makeSphere(10)).datum.diagonal > 0


@pytest.mark.parametrize("offset", [0.15, 0.18])
@pytest.mark.parametrize("tol_pos", [0.005, 0.01, 0.02])
def test_cylindrical_order_survives_position_tolerance_changes(offset, tol_pos):
    App = import_freecad()
    Part = importlib.import_module("Part")
    central = Part.makeCylinder(8, 6)
    lobes = Part.makeCylinder(10, 6, App.Vector(offset, 0, 0)).common(
        Part.makeBox(24, 6, 6, App.Vector(-12, -3, 0))
    )
    shape = central.fuse(lobes).removeSplitter()
    assert shape.isValid() and len(shape.Solids) == 1
    reference = extract_spatial(shape)
    arcs = [f for f in reference.features if f.kind == "cylinder"]
    assert sorted(f.values["radius"] for f in arcs) == [8, 10]
    for moved in (False, True):
        candidate = shape.copy()
        if moved:
            candidate.rotate(App.Vector(), App.Vector(1, 2, 3), 37)
            candidate.translate(App.Vector(123, -97, 41))
        measured = extract_spatial(candidate)
        rotation, translation, _ = align_datum(reference.datum, measured.datum)
        for arc in arcs:
            cfg = GeometryBinding(
                mode="geometry",
                quantity="radius",
                reason="The corresponding cylindrical arc",
                witnesses=[location(arc).model_dump()],
            )
            status, values, _, _ = evaluate_binding(
                cfg,
                arc.values["radius"],
                measured,
                rotation,
                translation,
                tol_scalar=0.01,
                tol_pos=tol_pos,
            )
            assert status == "consistent" and values == [arc.values["radius"]]


def evaluate_native_witness(reference, candidate, predicate, quantity, target):
    ref = extract_spatial(reference)
    feature = next(f for f in ref.features if predicate(f))
    cfg = GeometryBinding(
        mode="geometry",
        quantity=quantity,
        reason="Fixture physical feature",
        witnesses=[dict(**location(feature).model_dump(), region=feature.region)],
    )
    bank = extract_spatial(candidate)
    return evaluate_binding(
        cfg, target, bank, np.eye(3), np.zeros(3), tol_scalar=0.01, tol_pos=0.01
    ), bank


def test_disconnected_blind_bores_cannot_measure_as_one_through_bore():
    App = import_freecad()
    Part = importlib.import_module("Part")
    body = Part.makeBox(30, 25, 20)
    reference = body.cut(Part.makeCylinder(3, 20, App.Vector(10, 9, 0)))
    candidate = body.cut(Part.makeCylinder(3, 2, App.Vector(10, 9, 0)))
    candidate = candidate.cut(Part.makeCylinder(3, 2, App.Vector(10, 9, 18)))
    assert candidate.isInside(App.Vector(10, 9, 10), 1e-6, True)
    result, bank = evaluate_native_witness(
        reference,
        candidate,
        lambda f: f.kind == "cylinder" and not f.convex,
        "axial_extent",
        20,
    )
    assert result[0] == "not_found"
    assert [f.values["axial_extent"] for f in bank.features if f.kind == "cylinder"] == [2, 2]


def test_near_coincident_arcs_keep_order_within_position_tolerance():
    App = import_freecad()
    Part = importlib.import_module("Part")

    def lobed(offset):
        central = Part.makeCylinder(8, 6, App.Vector(0, 0, -3))
        lobes = Part.makeCylinder(10, 6, App.Vector(0, 0, -3 + offset)).common(
            Part.makeBox(24, 6, 6, App.Vector(-12, -3, -3 + offset))
        )
        return central.fuse(lobes)

    result, _ = evaluate_native_witness(
        lobed(0),
        lobed(0.001),
        lambda f: f.kind == "cylinder" and abs(f.values["radius"] - 8) < 1e-7,
        "radius",
        8,
    )
    assert result[0] == "consistent" and result[1] == [8]


def test_coplanar_partitions_do_not_disable_wall_measurement():
    App = import_freecad()
    Part = importlib.import_module("Part")
    plain = Part.makeBox(100, 100, 10)
    blocks = [
        Part.makeBox(10, 10, 10, App.Vector(i * 10, j * 10, 0))
        for i in range(10)
        for j in range(10)
    ]
    split = blocks[0].multiFuse(blocks[1:])
    assert split.isValid() and len(split.Solids) == 1 and len(split.Faces) > 180
    assert plain.cut(split).Volume + split.cut(plain).Volume < 1e-7
    result, bank = evaluate_native_witness(
        plain,
        split,
        lambda f: f.kind == "plane_pair" and f.values["separation"] == 10,
        "separation",
        10,
    )
    assert not bank.limitations
    assert result[0] == "consistent" and result[1] == [10]


def test_actual_rotation_preserves_separate_coaxial_cylinder_intervals():
    App = import_freecad()
    Part = importlib.import_module("Part")
    shape = Part.makeCylinder(10, 2)
    shape = shape.fuse(Part.makeCylinder(8, 20, App.Vector(0, 0, 2)))
    shape = shape.fuse(Part.makeCylinder(10, 3, App.Vector(0, 0, 22)))
    # The off-axis blind hole fixes the otherwise arbitrary roll about the shaft.
    shape = shape.cut(Part.makeCylinder(1, 2, App.Vector(3, 4, 23)))
    ref = extract_spatial(shape)
    cylinders = [f for f in ref.features if f.kind == "cylinder"]
    assert sorted(
        f.values["axial_extent"] for f in cylinders if f.convex and f.values["radius"] == 10
    ) == [2, 3]
    for axis, angle, offset in [((1, 2, 3), 37, (123, -97, 41)), ((2, -1, 4), 113, (-97, 9, 214))]:
        candidate = shape.copy()
        candidate.rotate(App.Vector(), App.Vector(*axis), angle)
        candidate.translate(App.Vector(*offset))
        bank = extract_spatial(candidate)
        rotation, translation, fit = align_datum(ref.datum, bank.datum)
        for feature in cylinders:
            cfg = GeometryBinding(
                mode="geometry",
                quantity="axial_extent",
                reason="Each continuous wall",
                witnesses=[location(feature).model_dump()],
            )
            result = evaluate_binding(
                cfg,
                feature.values["axial_extent"],
                bank,
                rotation,
                translation,
                tol_scalar=0.01,
                tol_pos=0.01,
            )
            assert result[0] == "consistent", (fit, feature, result)
