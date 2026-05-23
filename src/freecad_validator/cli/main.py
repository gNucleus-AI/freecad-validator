"""``freecad-validator`` CLI.

Four subcommands::

    freecad-validator validate CANDIDATE.FCStd REFERENCE.FCStd SPEC.json
    freecad-validator batch    --sample-data-dir <path>
    freecad-validator join     --candidate-dir <path> --reference-dir <path> --output-dir <path>
    freecad-validator render   INPUT.FCStd OUTPUT.png

The CLI is a thin wrapper over the public Python API
(``freecad_validator.Validator``); every subcommand can be reproduced
in a few lines of Python.

``render`` requires the optional ``render`` extra (``pip install
gnucleus-freecad-validator[render]``) which pulls in PyVista / VTK.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import sys
from collections.abc import Iterable
from pathlib import Path

from freecad_validator import Validator
from freecad_validator.scorers.geometry import (
    add_tolerance_arguments,
    tolerances_from_args,
)
from freecad_validator.scorers.spec_consistency import (
    add_spec_tolerance_arguments,
    spec_tolerances_from_args,
)
from freecad_validator.validator import COMBINE_METHODS, DEFAULT_COMBINE_METHOD


def _add_combine_method_argument(p: argparse.ArgumentParser) -> None:
    """Shared `--combine-method` flag for `validate` and `batch`."""
    p.add_argument(
        "--combine-method", choices=COMBINE_METHODS, default=DEFAULT_COMBINE_METHOD,
        help="how to aggregate geometry_similarity and cad_spec_consistency into "
             f"`combined` (default: {DEFAULT_COMBINE_METHOD}). 'harmonic' = "
             "2gs/(g+s); 'min' = min(g, s) — strictest, pins to the weakest axis.",
    )

# ---------------------------------------------------------------------------
# `validate` — score one (candidate, reference, spec) triple
# ---------------------------------------------------------------------------


def _add_validate_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("candidate_fcstd", help="path to the candidate .FCStd")
    p.add_argument("reference_fcstd", help="path to the reference .FCStd")
    p.add_argument("spec_json", help="path to the spec .json")
    add_tolerance_arguments(p)
    add_spec_tolerance_arguments(p)
    _add_combine_method_argument(p)
    p.add_argument("--json", dest="emit_json", action="store_true",
                   help="emit the result as JSON on stdout")


def _run_validate(args: argparse.Namespace) -> int:
    validator = Validator(
        geom_tolerances=tolerances_from_args(args),
        spec_tolerances=spec_tolerances_from_args(args),
        combine_method=args.combine_method,
    )
    result = validator.validate(
        candidate_fcstd=args.candidate_fcstd,
        reference_fcstd=args.reference_fcstd,
        spec_json=args.spec_json,
    )
    if args.emit_json:
        print(json.dumps(result.model_dump(), indent=2))
    else:
        print(f"geometry_similarity        : {result.geometry_similarity:.6f}")
        print(f"cad_spec_consistency       : {result.cad_spec_consistency:.6f}")
        print(f"combined ({validator.combine_method:<8})       : {result.combined:.6f}")
        print(f"geometry_similarity_reason : {result.geometry_similarity_reason}")
        print(f"spec_consistency_reason    : {result.cad_spec_consistency_reason}")
    return 0


# ---------------------------------------------------------------------------
# `batch` — score every case under a sample-data directory
# ---------------------------------------------------------------------------


def _add_batch_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--sample-data-dir", type=Path, required=True,
                   help="root containing data/<case>/{candidate,reference}.FCStd + spec.json")
    p.add_argument("--output-csv", type=Path, default=None,
                   help="per-case results CSV "
                        "(default: <sample-data-dir>/validation_results.csv)")
    p.add_argument("--output-summary", type=Path, default=None,
                   help="aggregate summary JSON "
                        "(default: <sample-data-dir>/validation_summary.json)")
    add_tolerance_arguments(p)
    add_spec_tolerance_arguments(p)
    _add_combine_method_argument(p)


def _stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 4),
        "median": round(statistics.median(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def _find_spec_json(case_dir: Path) -> Path | None:
    """Locate the spec JSON inside a case directory.

    Search order:
      1. ``spec.json`` — preferred default.
      2. ``<dir-name>.json`` — fallback for datasets that name the
         spec after the case directory.
      3. The single ``*.json`` file in the directory if there's
         exactly one (excluding the validator's own output files).

    Returns ``None`` if nothing matches.
    """
    spec = case_dir / "spec.json"
    if spec.is_file():
        return spec
    legacy = case_dir / f"{case_dir.name}.json"
    if legacy.is_file():
        return legacy
    candidates = [
        p for p in case_dir.glob("*.json")
        if p.name not in ("validation_results.json", "validation_summary.json")
    ]
    if len(candidates) == 1:
        return candidates[0]
    return None


def _validate_one(case_dir: Path, validator: Validator) -> tuple[str, dict]:
    cid = case_dir.name
    candidate = case_dir / "candidate.FCStd"
    reference = case_dir / "reference.FCStd"
    spec_json = _find_spec_json(case_dir)
    if not candidate.is_file():
        return cid, {"error": f"missing candidate: {candidate}"}
    if not reference.is_file():
        return cid, {"error": f"missing reference: {reference}"}
    if spec_json is None:
        return cid, {"error": f"missing spec JSON in {case_dir} "
                              f"(looked for spec.json, {cid}.json, or a single *.json)"}
    try:
        result = validator.validate(
            candidate_fcstd=str(candidate),
            reference_fcstd=str(reference),
            spec_json=str(spec_json),
        )
    except Exception as exc:
        return cid, {"error": f"{type(exc).__name__}: {exc}"}
    return cid, {
        "geometry_similarity": result.geometry_similarity,
        "cad_spec_consistency": result.cad_spec_consistency,
        "combined": result.combined,
        "geometry_similarity_reason": result.geometry_similarity_reason,
        "cad_spec_consistency_reason": result.cad_spec_consistency_reason,
    }


def _iter_cases(data_dir: Path) -> Iterable[Path]:
    yield from sorted(p for p in data_dir.iterdir() if p.is_dir())


def _run_batch(args: argparse.Namespace) -> int:
    data_dir = args.sample_data_dir / "data"
    if not data_dir.is_dir():
        print(f"error: {data_dir} is not a directory")
        return 2
    out_csv = args.output_csv or args.sample_data_dir / "validation_results.csv"
    out_json = args.output_summary or args.sample_data_dir / "validation_summary.json"

    validator = Validator(
        geom_tolerances=tolerances_from_args(args),
        spec_tolerances=spec_tolerances_from_args(args),
        combine_method=args.combine_method,
    )
    rows = []
    geom_scores: list[float] = []
    spec_scores: list[float] = []
    combined: list[float] = []
    n_err = 0

    cases = list(_iter_cases(data_dir))
    print(f"validating {len(cases)} case(s) under {data_dir}")
    for i, cd in enumerate(cases, 1):
        cid, result = _validate_one(cd, validator)
        if "error" in result:
            n_err += 1
            print(f"[{i:3d}/{len(cases)}] {cid}  ERROR  {result['error']}")
            rows.append({
                "case_id": cid, "geometry_similarity": "",
                "cad_spec_consistency": "", "combined": "",
                "geometry_similarity_reason": "",
                "cad_spec_consistency_reason": "",
                "error": result["error"],
            })
            continue
        geom_scores.append(result["geometry_similarity"])
        spec_scores.append(result["cad_spec_consistency"])
        combined.append(result["combined"])
        print(
            f"[{i:3d}/{len(cases)}] {cid}  "
            f"geom={result['geometry_similarity']:.3f}  "
            f"spec={result['cad_spec_consistency']:.3f}  "
            f"combined={result['combined']:.3f}"
        )
        rows.append({
            "case_id": cid,
            "geometry_similarity": f"{result['geometry_similarity']:.4f}",
            "cad_spec_consistency": f"{result['cad_spec_consistency']:.4f}",
            "combined": f"{result['combined']:.4f}",
            "geometry_similarity_reason": result["geometry_similarity_reason"],
            "cad_spec_consistency_reason": result["cad_spec_consistency_reason"],
            "error": "",
        })

    fields = [
        "case_id", "geometry_similarity", "cad_spec_consistency", "combined",
        "geometry_similarity_reason", "cad_spec_consistency_reason", "error",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    summary = {
        "cases_total": len(cases),
        "cases_validated": len(combined),
        "cases_errored": n_err,
        "geometry_similarity": _stats(geom_scores),
        "cad_spec_consistency": _stats(spec_scores),
        "combined": _stats(combined),
    }
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print()
    print(f"validated: {summary['cases_validated']}/{summary['cases_total']}  errors: {n_err}")
    print(f"geometry_similarity : {summary['geometry_similarity']}")
    print(f"cad_spec_consistency: {summary['cad_spec_consistency']}")
    print(f"combined            : {summary['combined']}")
    print(f"per-case CSV: {out_csv}")
    print(f"summary JSON: {out_json}")
    return 0


# ---------------------------------------------------------------------------
# `join` — pair candidate + reference trees into a sample-data directory
# ---------------------------------------------------------------------------


def _add_join_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--candidate-dir", type=Path, required=True)
    p.add_argument("--reference-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--force", action="store_true",
                   help="rmtree --output-dir first instead of merging into it")


def _candidate_fcstd(case_dir: Path) -> Path | None:
    """Find the candidate's FCStd. Defaults to ``answer.FCStd``;
    falls back to any ``*.FCStd`` so hand-curated dirs work too."""
    p = case_dir / "answer.FCStd"
    if p.is_file():
        return p
    for p in case_dir.glob("*.FCStd"):
        if p.is_file():
            return p
    return None


def _reference_fcstd(case_dir: Path) -> Path | None:
    """Find the reference's FCStd. Defaults to ``<id>_parameterized.FCStd``;
    falls back to any ``*.FCStd``."""
    cid = case_dir.name
    p = case_dir / f"{cid}_parameterized.FCStd"
    if p.is_file():
        return p
    for p in case_dir.glob("*.FCStd"):
        if p.is_file():
            return p
    return None


def _run_join(args: argparse.Namespace) -> int:
    cand_data = args.candidate_dir / "data"
    ref_data = args.reference_dir / "data"
    if not cand_data.is_dir():
        print(f"error: {cand_data} is not a directory")
        return 2
    if not ref_data.is_dir():
        print(f"error: {ref_data} is not a directory")
        return 2

    if args.output_dir.exists() and args.force:
        shutil.rmtree(args.output_dir)
    out_data = args.output_dir / "data"
    out_data.mkdir(parents=True, exist_ok=True)

    cand_ids = {p.name for p in cand_data.iterdir() if p.is_dir()}
    ref_ids = {p.name for p in ref_data.iterdir() if p.is_dir()}
    shared = cand_ids & ref_ids

    matched: list[str] = []
    skipped_no_cand: list[str] = []
    skipped_no_ref: list[str] = []
    for cid in sorted(shared):
        cand_case = cand_data / cid
        ref_case = ref_data / cid
        cand_fcstd = _candidate_fcstd(cand_case)
        ref_fcstd = _reference_fcstd(ref_case)
        if cand_fcstd is None:
            skipped_no_cand.append(cid)
            continue
        if ref_fcstd is None:
            skipped_no_ref.append(cid)
            continue
        dst = out_data / cid
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cand_fcstd, dst / "candidate.FCStd")
        shutil.copy2(ref_fcstd, dst / "reference.FCStd")
        spec = ref_case / f"{cid}.json"
        if spec.is_file():
            shutil.copy2(spec, dst / spec.name)
        param_check = ref_case / "param_check.py"
        if param_check.is_file():
            shutil.copy2(param_check, dst / "param_check.py")
        matched.append(cid)

    print(f"matched              : {len(matched)}")
    print(f"candidate-only       : {len(cand_ids - ref_ids)}")
    print(f"reference-only       : {len(ref_ids - cand_ids)}")
    print(f"skipped no candidate : {len(skipped_no_cand)}")
    print(f"skipped no reference : {len(skipped_no_ref)}")
    print(f"output: {args.output_dir}")
    return 0


# ---------------------------------------------------------------------------
# `render` — rasterize one FCStd to a PNG (requires the `render` extra)
# ---------------------------------------------------------------------------


def _add_render_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("input_fcstd", help="path to the input .FCStd")
    p.add_argument("output_png", help="path to write the PNG (parent dirs are created)")
    p.add_argument("--finer-mesh", action="store_true",
                   help="scale tessellation tolerances to the part's bbox diagonal "
                        "for higher-fidelity output (can 10×+ vertex/face counts; "
                        "default tessellates at fixed mesh=0.1mm / edge=0.05mm)")


def _run_render(args: argparse.Namespace) -> int:
    # Local import so users without the `render` extra don't pay the
    # PyVista/VTK import cost for `validate` / `batch` / `join`.
    from freecad_validator.render.render_freecad import render_freecad_file
    ok = render_freecad_file(
        args.input_fcstd, args.output_png, finer_mesh=args.finer_mesh,
    )
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


_SUBCOMMANDS = (
    ("validate", "score one (candidate, reference, spec) triple",
     _add_validate_args, _run_validate),
    ("batch", "score every case under a sample-data directory",
     _add_batch_args, _run_batch),
    ("join", "build a sample-data directory from separate trees",
     _add_join_args, _run_join),
    ("render", "rasterize a .FCStd to a PNG (requires `[render]` extra)",
     _add_render_args, _run_render),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="freecad-validator",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text, add_args, run_fn in _SUBCOMMANDS:
        p = sub.add_parser(name, help=help_text)
        add_args(p)
        p.set_defaults(func=run_fn)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
