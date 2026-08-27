"""Unit tests for the numeric helpers (scorer/metrics.py)."""

from freecad_validator.fem import metrics


def test_relative_error():
    assert metrics.relative_error(110, 100) == 0.1
    assert metrics.relative_error(90, 100) == 0.1
    assert metrics.relative_error(5, 0) == 5.0  # safe ref==0 fallback


def test_tolerance_band_score_boundaries():
    assert metrics.tolerance_band_score(0.05, 0.10) == 100.0  # inside tol
    assert metrics.tolerance_band_score(0.10, 0.10) == 100.0  # at tol
    assert metrics.tolerance_band_score(0.40, 0.10, fade=4) == 0.0  # at fade*tol
    mid = metrics.tolerance_band_score(0.25, 0.10, fade=4)
    assert 0.0 < mid < 100.0


def test_within_tolerance():
    ok, err = metrics.within_tolerance(104, 100, 0.05)
    assert ok and abs(err - 0.04) < 1e-9
    ok, err = metrics.within_tolerance(120, 100, 0.05)
    assert not ok


def test_mesh_quality_good_vs_bad():
    good, _ = metrics.mesh_quality_score(
        {"min_jacobian": 0.8, "max_aspect_ratio": 2.5, "max_skewness": 0.4}
    )
    bad, notes = metrics.mesh_quality_score(
        {
            "min_jacobian": 0.2,
            "max_aspect_ratio": 25.0,
            "max_skewness": 0.98,
            "pct_elements_below_jac": 8.0,
        }
    )
    assert good > 95
    assert bad < 40
    assert notes  # human-readable issues reported


def test_convergence_converged():
    study = [
        {"n_elements": 4000, "value": 0.96},
        {"n_elements": 32000, "value": 0.99},
        {"n_elements": 256000, "value": 1.00},
    ]
    out = metrics.convergence_from_study(study)
    assert out["converged"] is True
    assert out["n_grids"] == 3
    assert out["gci_fine"] is not None
    assert out["apparent_order"] is not None


def test_convergence_diverging_singularity():
    study = [
        {"n_elements": 4000, "value": 100.0},
        {"n_elements": 32000, "value": 160.0},
        {"n_elements": 256000, "value": 250.0},
    ]
    out = metrics.convergence_from_study(study)
    assert out["converged"] is False
    assert out["diverging"] is True


def test_values_agree_and_discrepancy():
    assert metrics.values_agree([100.0, 100.5, 99.8]) is True
    assert metrics.values_agree([100.0, 135.0]) is False
    assert metrics.max_pairwise_discrepancy([100.0, 135.0]) > 0.2


def test_stress_unit_consistency():
    assert metrics.stress_unit_consistent({"force": "N", "length": "mm", "stress": "MPa"}) is True
    assert metrics.stress_unit_consistent({"force": "N", "length": "mm", "stress": "Pa"}) is False
    assert metrics.stress_unit_consistent({"force": "N"}) is None


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok:", fn.__name__)
    print(f"{len(fns)} metric tests passed")
