"""Read STEP solids and extract geometry descriptors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import FreeCAD  # noqa: F401 — initialize the core before Part
import Part

from freecad_validator.fem.pre_process.geometry_compare.brep_diff.models import DiffConfig
from freecad_validator.fem.pre_process.geometry_compare.loaders.loaded_part import (
    LoadedPart,
)
from freecad_validator.fem.pre_process.spatial import SpatialShape, spatial_document


def representative_shape(shape: Any) -> Any:
    """Read the occupied union through point queries without constructing a solid."""
    solids = list(getattr(shape, "Solids", []) or [])
    if len(solids) == 1:
        return solids[0]
    if len(solids) > 1:
        return SpatialShape(solids)
    return shape


def open_step_part(step_path: str | Path, config: DiffConfig) -> LoadedPart:
    """STEP path -> `LoadedPart` (BrepDocument + representative OCCT Shape).

    Retain the shape produced by `Part.Shape().read()` for read-only queries;
    no deep copy is needed.
    """
    raw = Part.Shape()
    raw.read(str(step_path))
    shape = representative_shape(raw)
    document = spatial_document(shape, str(Path(step_path).resolve()), config)
    return LoadedPart(document=document, shape=shape)
