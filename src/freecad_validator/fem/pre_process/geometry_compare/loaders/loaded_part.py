"""Geometry descriptors paired with a detached OCCT shape."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from freecad_validator.fem.pre_process.geometry_compare.brep_diff.models import BrepDocument


@dataclass(frozen=True)
class LoadedPart:
    """One CAD part loaded into the shared scoring representation."""

    document: BrepDocument
    shape: Any
