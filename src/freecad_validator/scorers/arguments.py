"""Geometry CLI options shared by both scorer versions and the validator."""

from __future__ import annotations

import argparse

from pydantic import ValidationError

from freecad_validator.comparators.geometry import GeometryTolerances

_TOLERANCE_VERSIONS = {
    "volume_matched_rel_tol": ("v1", "v2"),
    "volume_far_rel_tol": ("v1", "v2"),
    "area_matched_rel_tol": ("v1", "v2"),
    "area_far_rel_tol": ("v1", "v2"),
    "bbox_far_rel_tol": ("v1", "v2"),
    "bbox_matched_rel_tol": ("v1",),
    "surface_types_exact_tol": ("v1",),
    "surface_types_zero_score": ("v1",),
    "surface_types_matched_rel_tol": ("v2",),
    "surface_types_far_rel_tol": ("v2",),
    "principal_moments_matched_rel_tol": ("v2",),
    "principal_moments_far_rel_tol": ("v2",),
}


class GeometryArgumentError(ValueError):
    """Invalid geometry options, distinct from errors while scoring a file."""


def add_tolerance_arguments(
    parser: argparse.ArgumentParser, *, scorer_version: str | None = None
) -> None:
    """Register options grouped by version, retaining explicit overrides.

    Fixed-version CLIs expose only their own options. The joint CLI exposes
    both versions and rejects inapplicable overrides after parsing --scorer.
    """
    if scorer_version not in (None, "v1", "v2"):
        raise ValueError(f"unknown scorer version: {scorer_version!r}")
    defaults = GeometryTolerances()
    groups = {}
    for field_name, versions in _TOLERANCE_VERSIONS.items():
        if scorer_version is not None and scorer_version not in versions:
            continue
        label = f"geometry tolerances ({', '.join(versions)})"
        if label not in groups:
            groups[label] = parser.add_argument_group(label)
        groups[label].add_argument(
            f"--{field_name.replace('_', '-')}",
            type=float,
            default=None,
            help=f"override {field_name} (default: {getattr(defaults, field_name)})",
        )


def tolerances_from_args(
    args: argparse.Namespace, *, scorer_version: str
) -> GeometryTolerances | None:
    """Validate CLI applicability, then use the shared Python configuration API."""
    if scorer_version not in ("v1", "v2"):
        raise GeometryArgumentError(f"unknown scorer version: {scorer_version!r}")
    overrides = {
        name: getattr(args, name)
        for name in GeometryTolerances.model_fields
        if getattr(args, name, None) is not None
    }
    unsupported = [
        f"--{name.replace('_', '-')} ({', '.join(_TOLERANCE_VERSIONS[name])} only)"
        for name in overrides
        if scorer_version not in _TOLERANCE_VERSIONS[name]
    ]
    if unsupported:
        raise GeometryArgumentError(
            f"geometry options not supported by scorer {scorer_version}: {', '.join(unsupported)}"
        )
    if not overrides:
        return None
    try:
        return GeometryTolerances.for_scorer(scorer_version, **overrides)
    except ValidationError as exc:
        messages = []
        for error in exc.errors(include_url=False, include_context=False, include_input=False):
            field = ".".join(str(part) for part in error["loc"])
            message = error["msg"].removeprefix("Value error, ")
            messages.append(f"{field}: {message}" if field else message)
        raise GeometryArgumentError("; ".join(messages)) from exc
