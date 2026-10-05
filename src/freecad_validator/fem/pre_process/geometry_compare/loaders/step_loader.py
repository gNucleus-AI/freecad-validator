"""Read STEP solids and extract geometry descriptors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import FreeCAD  # noqa: F401 — initialize the core before Part
import Part

from freecad_validator.fem.pre_process.geometry_compare.brep_diff import extractor
from freecad_validator.fem.pre_process.geometry_compare.brep_diff.models import DiffConfig
from freecad_validator.fem.pre_process.geometry_compare.loaders.loaded_part import (
    LoadedPart,
)


def representative_shape(shape: Any) -> Any:
    """Form the solid union of the STEP input for occupied-material comparison."""
    solids = list(getattr(shape, "Solids", []) or [])
    if len(solids) == 1:
        return solids[0]
    if len(solids) > 1:
        try:
            return solids[0].fuse(solids[1:])
        except Exception:
            return shape
    return shape


def open_step_part(step_path: str | Path, config: DiffConfig) -> LoadedPart:
    """STEP path -> `LoadedPart` (BrepDocument + representative OCCT Shape).

    The shape a `Part.Shape().read()` produces is already detached (it is not
    owned by any open document), so no `.copy()` is needed for it to outlive this
    call, unlike the FCStd backend.
    """
    raw = Part.Shape()
    raw.read(str(step_path))
    shape = representative_shape(raw)
    freecad_version, occt_version = extractor.runtime_versions()
    document = extractor.document_from_shape(
        shape,
        path=str(Path(step_path).resolve()),
        object_name=Path(step_path).stem,
        freecad_version=freecad_version,
        occt_version=occt_version,
        gate_reason=None,
        config=config,
    )
    return LoadedPart(document=document, shape=shape)
