"""Watertight tessellation and solid Boolean conversion."""

from __future__ import annotations

from typing import Any

import manifold3d
import numpy as np

from freecad_validator.fem.pre_process.geometry_compare.brep_diff.models import DiffConfig
from freecad_validator.fem.pre_process.geometry_compare.scorers import scorer_base


def tessellate_shape(shape: Any, deflection: float) -> tuple[np.ndarray, np.ndarray]:
    """Watertight triangulation of an already-loaded OCCT shape."""
    raw_verts, raw_tris = shape.tessellate(deflection)
    verts = np.asarray([[v.x, v.y, v.z] for v in raw_verts], dtype=np.float32)
    tris = np.asarray(raw_tris, dtype=np.uint32)
    return verts, tris


def tessellate_solid(
    path: str, deflection: float, config: DiffConfig | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate the cached detached shape used by the geometric comparison."""
    shape = scorer_base.load_shape(path, config)
    if shape is None:
        raise ValueError(f"no scorable solid to tessellate for {path}")
    return tessellate_shape(shape, deflection)


def to_manifold(verts: np.ndarray, tris: np.ndarray):
    mesh = manifold3d.Mesh(vert_properties=verts, tri_verts=tris)
    return manifold3d.Manifold(mesh)
