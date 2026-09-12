"""An unavailable native backend must not silently disable the size gate."""

import builtins

import pytest

from freecad_validator.comparators.occt_bbox import oriented_bbox_dimensions


@pytest.mark.parametrize("error_type", [ImportError, OSError])
def test_missing_ocp_explains_required_extra(monkeypatch, error_type):
    real_import = builtins.__import__

    def without_ocp(name, *args, **kwargs):
        if name.startswith("OCP"):
            raise error_type("OCP unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_ocp)
    with pytest.raises(RuntimeError, match=r"gnucleus-freecad-validator\[v2\]"):
        oriented_bbox_dimensions("")
