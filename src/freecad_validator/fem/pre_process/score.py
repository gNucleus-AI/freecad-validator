"""Run with FreeCAD's Python; emit a preprocessing geometry score JSON."""

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

from freecad_validator.fem.pre_process.assembly import score_assembly
from freecad_validator.fem.pre_process.errors import BodyCorrespondenceError, CandidateGeometryError
from freecad_validator.fem.pre_process.geometry_compare.brep_diff.models import DiffConfig
from freecad_validator.fem.pre_process.integration import apply_preprocessing_score
from freecad_validator.fem.pre_process.scorer import PreProcessScorer
from freecad_validator.fem.schema import ScoringReport


def main(arguments: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--reference", help="Reference clean STEP or FCStd")
    parser.add_argument("--candidate", help="Answer clean STEP or FCStd")
    parser.add_argument("--raw-object", help="FCStd object Name to select")
    parser.add_argument("--reference-object", help="FCStd object Name to select")
    parser.add_argument("--candidate-object", help="FCStd object Name to select")
    parser.add_argument("--region-samples", type=int, default=8192)
    parser.add_argument(
        "--assembly",
        action="store_true",
        help="Automatically match original bodies using raw STEP and solved FCStd files",
    )
    parser.add_argument("--out", type=Path, help="Also write JSON to this path")
    parser.add_argument(
        "--fem-report",
        type=Path,
        help="Existing verified static FEM report to multiply by task geometry credit",
    )
    args = parser.parse_args(arguments)
    if args.fem_report and not args.assembly:
        parser.error("--fem-report requires --assembly")
    if args.assembly and (not args.reference or not args.candidate):
        parser.error("--assembly requires raw STEP, reference FCStd and candidate FCStd")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    scorer = PreProcessScorer(DiffConfig(region_sample_count=args.region_samples))
    if args.assembly:
        try:
            report = score_assembly(args.raw, args.reference, args.candidate, scorer)
        except CandidateGeometryError as exc:
            report = {"status": "candidate_invalid", "score": 0.0, "error": str(exc)}
        except BodyCorrespondenceError as exc:
            report = {
                "status": "evaluation_error",
                "error": str(exc),
                "error_type": "body_correspondence",
                "input_role": exc.input_role,
            }
        except (RuntimeError, ValueError, OSError) as exc:
            report = {"status": "evaluation_error", "error": str(exc)}
    else:
        if args.reference is None or args.candidate is None:
            parser.error("Single-body mode requires --reference and --candidate")
        result = scorer.score_detailed(
            args.raw,
            args.reference,
            args.candidate,
            raw_object=args.raw_object,
            reference_object=args.reference_object,
            candidate_object=args.candidate_object,
        )
        report = asdict(result) if result is not None else {"status": "skipped", "score": None}
    if args.fem_report and report.get("status") != "evaluation_error":
        fem_report = ScoringReport(**json.loads(args.fem_report.read_text(encoding="utf-8")))
        combined = apply_preprocessing_score(fem_report, report["score"])
        report = {
            "geometry": report,
            "fem_report": combined.to_dict(),
            "overall_score": combined.overall_score,
            "reward": round(combined.overall_score / 100, 6),
        }
    payload = json.dumps(report, indent=2, allow_nan=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
    print(payload, flush=True)
    if report.get("status") == "evaluation_error":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
