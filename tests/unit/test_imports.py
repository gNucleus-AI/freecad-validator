"""FreeCAD-free import smoke tests.

These run in CI without FreeCAD installed and fail loudly on any
broken import in the public surface of the package.
"""
from __future__ import annotations


def test_top_level_exports():
    import freecad_validator

    assert hasattr(freecad_validator, "Validator")
    assert hasattr(freecad_validator, "ValidationResult")
    assert "Validator" in freecad_validator.__all__
    assert "ValidationResult" in freecad_validator.__all__
    assert isinstance(freecad_validator.__version__, str)


def test_subpackages_import():
    # Verify every subpackage's pure-Python surface imports cleanly.
    import freecad_validator.cli.main  # noqa: F401
    import freecad_validator.consistency.checker  # noqa: F401
    import freecad_validator.consistency.checks  # noqa: F401
    import freecad_validator.consistency.compare  # noqa: F401
    import freecad_validator.consistency.report  # noqa: F401
    import freecad_validator.scorers.base  # noqa: F401
    import freecad_validator.spec.parser  # noqa: F401


def test_categories_are_registered_class_per_file():
    """Each `categories/<name>.py` exposes a `*Category` subclass."""
    from freecad_validator.consistency.categories import (
        base,
        box,
        flange_plate,
        gear,
        hex,
        key,
        keyway,
        pin,
        pulley,
        spline,
        spring,
        washer,
    )

    for mod in (box, flange_plate, gear, hex, key, keyway, pin,
                pulley, spline, spring, washer):
        cat_subclasses = [
            getattr(mod, name) for name in dir(mod)
            if name.endswith("Category") and isinstance(getattr(mod, name), type)
            and issubclass(getattr(mod, name), base.Category)
        ]
        assert cat_subclasses, f"{mod.__name__} has no Category subclass"
