"""Load geometry descriptors from FCStd or STEP inputs."""

from __future__ import annotations

from pathlib import Path

from freecad_validator.fem.pre_process.geometry_compare.brep_diff.extractor import (
    open_fcstd_part,
)
from freecad_validator.fem.pre_process.geometry_compare.brep_diff.models import DiffConfig
from freecad_validator.fem.pre_process.geometry_compare.loaders.loaded_part import (
    LoadedPart,
)
from freecad_validator.fem.pre_process.geometry_compare.loaders.step_loader import (
    open_step_part,
)

# file suffix (lowercased) -> backend that turns the path into a LoadedPart
_BACKENDS = {
    ".fcstd": open_fcstd_part,
    ".step": open_step_part,
    ".stp": open_step_part,
}
SUPPORTED_SUFFIXES = tuple(_BACKENDS)


def load_part(path: str | Path, config: DiffConfig) -> LoadedPart:
    """CAD file -> `LoadedPart`, dispatched by file suffix.

    Raises `ValueError` for an unsupported suffix. Missing-file / empty-solid
    handling is the caller's job (see `scorer_base.load_doc`), mirroring the old
    extractor contract.
    """
    suffix = Path(path).suffix.lower()
    backend = _BACKENDS.get(suffix)
    if backend is None:
        raise ValueError(
            f"unsupported CAD format '{suffix}' for {path}; "
            f"supported: {', '.join(SUPPORTED_SUFFIXES)}"
        )
    return backend(Path(path), config)
