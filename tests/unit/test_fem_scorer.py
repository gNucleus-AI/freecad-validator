"""Behavioral tests for :mod:`freecad_validator.fem.scorer`."""


from freecad_validator.fem import CaseDefinition, FailureMode, Submission, score_result


def _case():
    return CaseDefinition(
        case_id="T01", title="Cantilever tip load", subtype="cantilever", analysis_type="static",
        units_expected={"length": "mm", "force": "N", "stress": "MPa"},
        geometry={"characteristic_length_mm": 1000.0},
        material={"E_MPa": 210000.0, "nu": 0.30, "yield_MPa": 250.0},
        expected_bcs=[{"type": "fixed"}],
        expected_loads=[{"type": "force", "magnitude_N": 1000.0}],
        reference={"method": "analytical", "source": "Roark",
                   "quantities": {"max_displacement_mm": {"value": 3.05, "unit": "mm", "tol_rel": 0.10, "critical": True},
                                  "max_von_mises_MPa": {"value": 48.0, "unit": "MPa", "tol_rel": 0.15}}},
        mesh_expectations={"expect_convergence": True})


def _excellent():
    return Submission(
        case_id="T01", analysis_type="static",
        units={"length": "mm", "force": "N", "stress": "MPa"},
        material={"E_MPa": 210000.0, "nu": 0.30},
        boundary_conditions=[{"type": "fixed", "location": "root"}],
        loads=[{"type": "force", "magnitude_N": 1000.0, "direction": [0, 0, -1]}],
        mesh={"element_type": "C3D10", "num_elements": 60000, "local_refinement": True,
              "quality": {"min_jacobian": 0.7, "max_aspect_ratio": 3.0, "max_skewness": 0.45,
                          "pct_elements_below_jac": 0.0},
              "convergence_study": [{"n_elements": 4000, "value": 2.95},
                                    {"n_elements": 32000, "value": 3.02},
                                    {"n_elements": 256000, "value": 3.05}]},
        solver={"name": "CalculiX", "converged": True, "residual": 1e-9,
                "applied_load_N": 1000.0, "reaction_force_N": [0, 0, 1000.2]},
        results={"max_displacement_mm": 3.05, "max_von_mises_MPa": 48.0, "safety_factor": 5.2},
        reported_values={"max_displacement_mm": {"text": 3.05, "table": 3.05, "plot": 3.05}},
        report={"assumptions": ["linear elastic"], "interpretation": "Tip deflects 3.05 mm; peak stress at the root as expected for a cantilever under an end load.",
                "limitations": ["sharp built-in edge singularity ignored"], "convergence_claim": True,
                "mesh_justification": "second-order tets refined at the support; <1% change on refinement",
                "safety_factor_discussion": "SF=5.2 vs yield"},
        artifacts={"input_deck": "job.inp", "result_file": "job.frd", "script": "run.py"})


def test_excellent_scores_high():
    rep = score_result(_case(), _excellent())
    assert rep.overall_score >= 90
    assert rep.grade in ("excellent", "good")
    assert isinstance(rep.subscores["problem_setup"], float)
    assert "raw_score" in rep.subscores_details["problem_setup"]
    assert rep.pass_fail_flags["physically_valid"]
    assert rep.pass_fail_flags["reproducible"]
    assert rep.pass_fail_flags["not_hallucinated"]
    assert not rep.failure_modes_detected


def test_physically_impossible_gates_to_zero():
    sub = _excellent()
    sub.results["max_displacement_mm"] = 1800.0  # > characteristic length
    rep = score_result(_case(), sub)
    assert rep.overall_score == 0
    assert rep.grade == "invalid"
    assert any(f["code"] == FailureMode.PHYSICALLY_IMPOSSIBLE for f in rep.failure_modes_detected)
    assert not rep.pass_fail_flags["physically_valid"]
    assert rep.gates_triggered


def test_negative_von_mises_impossible():
    sub = _excellent()
    sub.results["max_von_mises_MPa"] = -10.0
    rep = score_result(_case(), sub)
    assert any(f["code"] == FailureMode.PHYSICALLY_IMPOSSIBLE for f in rep.failure_modes_detected)


def test_missing_boundary_condition():
    sub = _excellent()
    sub.boundary_conditions = []
    rep = score_result(_case(), sub)
    assert any(f["code"] == FailureMode.MISSING_BOUNDARY_CONDITION for f in rep.failure_modes_detected)
    assert not rep.pass_fail_flags["setup_correct"]


def test_hallucinated_solver_output():
    sub = _excellent()
    sub.solver = {}
    sub.artifacts = {}
    rep = score_result(_case(), sub)
    codes = {f["code"] for f in rep.failure_modes_detected}
    assert FailureMode.HALLUCINATED_SOLVER_OUTPUT in codes or FailureMode.NON_REPRODUCIBLE in codes
    assert rep.overall_score == 0


def test_hallucinated_convergence():
    sub = _excellent()
    sub.mesh["convergence_study"] = []          # no study
    sub.report["convergence_claim"] = True       # but claims convergence
    rep = score_result(_case(), sub)
    assert any(f["code"] == FailureMode.HALLUCINATED_CONVERGENCE for f in rep.failure_modes_detected)


def test_internal_inconsistency():
    sub = _excellent()
    sub.reported_values = {"max_displacement_mm": {"text": 3.05, "table": 4.5, "plot": 3.05}}
    rep = score_result(_case(), sub)
    assert any(f["code"] == FailureMode.INTERNAL_INCONSISTENCY for f in rep.failure_modes_detected)


def test_reaction_imbalance():
    sub = _excellent()
    sub.solver["reaction_force_N"] = [0, 0, 1400.0]  # 40% off applied 1000
    rep = score_result(_case(), sub)
    assert any(f["code"] == FailureMode.REACTION_IMBALANCE for f in rep.failure_modes_detected)


def test_accuracy_gross_error():
    sub = _excellent()
    sub.results["max_displacement_mm"] = 30.5   # 10x reference
    rep = score_result(_case(), sub)
    acc = rep.subscores["accuracy_vs_reference"]
    assert acc < 50


def test_wrong_material_flagged():
    sub = _excellent()
    sub.material["E_MPa"] = 70000.0  # aluminium instead of steel
    rep = score_result(_case(), sub)
    assert any(f["code"] == FailureMode.WRONG_MATERIAL for f in rep.failure_modes_detected)


def test_determinism():
    case, sub = _case(), _excellent()
    a = score_result(case, sub).overall_score
    b = score_result(case, sub).overall_score
    assert a == b


def test_accuracy_weight_redistributed_when_no_reference():
    case = _case()
    case.reference = {"method": "none"}  # no quantities, no ranges
    rep = score_result(case, _excellent())
    # accuracy weight should be redistributed: its weight shown as 0
    assert rep.subscores_details["accuracy_vs_reference"]["weight"] == 0.0
    total_w = sum(rep.subscores_details[c]["weight"] for c in rep.subscores_details)
    assert abs(total_w - 1.0) < 1e-6


def test_polished_but_wrong_cannot_pass():
    """A perfect write-up cannot rescue a physically impossible result."""
    sub = _excellent()  # full report
    sub.results["max_displacement_mm"] = 5000.0
    rep = score_result(_case(), sub)
    assert rep.overall_score == 0
    assert rep.subscores["engineering_reporting"] >= 80  # report still good
    assert rep.grade == "invalid"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok:", fn.__name__)
    print(f"{len(fns)} scorer tests passed")
