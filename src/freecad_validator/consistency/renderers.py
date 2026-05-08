"""Report renderers — text (terminal), markdown (LLM-friendly), JSON."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from freecad_validator.consistency.report import ConsistencyReport


def _fmt(v: Any) -> str:
    if isinstance(v, tuple):
        return "(" + ", ".join(f"{x:.3f}" for x in v) + ")"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def render_text(report: ConsistencyReport) -> str:
    """Human-readable rendering."""
    lines: list[str] = []
    lines.append(f"Spec:  {report.spec_name}   FCStd: {report.fcstd_path}")
    if report.error:
        lines.append(f"ERROR: {report.error}")
    if report.summary is not None:
        s = report.summary
        lines.append(
            f"Total params: {s.total_params}   "
            f"consistent={s.consistent}  inconsistent={s.inconsistent}  "
            f"not_found={s.not_found}  unexpected_features={s.unexpected_features}"
        )
        lines.append(
            f"Consistency rate: {s.consistency_rate:.1%}   "
            f"Measurable rate: {s.measurable_rate:.1%}"
        )
    lines.append("")
    lines.append(f"Consistent ({len(report.consistent)}):")
    for f in report.consistent:
        lines.append(
            f"  {f.param:32s} = {_fmt(f.spec_value):>10s} {f.unit:5s}"
            f"  (CAD: {_fmt(f.measured_value)}, on {f.feature})"
        )
    lines.append("")
    lines.append(f"Inconsistent ({len(report.inconsistent)}):")
    for f in report.inconsistent:
        rd = f"{f.rel_diff:.3f}" if f.rel_diff is not None else "—"
        lines.append(
            f"  {f.param:32s} = {_fmt(f.spec_value):>10s} {f.unit:5s}"
            f"  (CAD: {_fmt(f.measured_value)}, on {f.feature}, rel_diff={rd})"
            + (f"  [{f.reason}]" if f.reason else "")
        )
    lines.append("")
    lines.append(f"Not found ({len(report.not_found)}):")
    for f in report.not_found:
        lines.append(
            f"  {f.param:32s} = {_fmt(f.spec_value):>10s} {f.unit:5s}"
            + (f"  [{f.reason}]" if f.reason else "")
        )
    if report.unexpected_features:
        lines.append("")
        lines.append(f"Unexpected features ({len(report.unexpected_features)}):")
        for feat in report.unexpected_features:
            lines.append(f"  {feat}")
    return "\n".join(lines)


def render_markdown(report: ConsistencyReport) -> str:
    """Markdown rendering — the format LLMs handle best in prompts."""
    lines: list[str] = []
    lines.append(f"# Consistency report: `{report.spec_name}`")
    lines.append("")
    lines.append(f"- **FCStd**: `{report.fcstd_path}`")
    if report.summary is not None:
        s = report.summary
        lines.append(f"- **Total params**: {s.total_params}")
        lines.append(
            f"- **Summary**: consistent={s.consistent}, "
            f"inconsistent={s.inconsistent}, "
            f"not_found={s.not_found}, "
            f"unexpected_features={s.unexpected_features}"
        )
        lines.append(
            f"- **Consistency rate**: {s.consistency_rate:.1%} "
            f"(measurable rate: {s.measurable_rate:.1%})"
        )
    if report.error:
        lines.append(f"- **Error**: {report.error}")
    lines.append("")

    lines.append(f"## Consistent ({len(report.consistent)})")
    lines.append("")
    if report.consistent:
        lines.append("| Param | Spec value | Unit | CAD value | Feature |")
        lines.append("|---|---|---|---|---|")
        for f in report.consistent:
            lines.append(
                f"| `{f.param}` | {_fmt(f.spec_value)} | {f.unit} |"
                f" {_fmt(f.measured_value)} | `{f.feature or ''}` |"
            )
    else:
        lines.append("_(none)_")
    lines.append("")

    lines.append(f"## Inconsistent ({len(report.inconsistent)})")
    lines.append("")
    if report.inconsistent:
        lines.append("| Param | Spec value | Unit | CAD value | Feature | rel_diff | Reason |")
        lines.append("|---|---|---|---|---|---|---|")
        for f in report.inconsistent:
            rd = f"{f.rel_diff:.3f}" if f.rel_diff is not None else "—"
            lines.append(
                f"| `{f.param}` | {_fmt(f.spec_value)} | {f.unit} |"
                f" {_fmt(f.measured_value)} | `{f.feature or ''}` |"
                f" {rd} | {f.reason or ''} |"
            )
    else:
        lines.append("_(none)_")
    lines.append("")

    lines.append(f"## Not found ({len(report.not_found)})")
    lines.append("")
    if report.not_found:
        lines.append("| Param | Spec value | Unit | Reason |")
        lines.append("|---|---|---|---|")
        for f in report.not_found:
            lines.append(
                f"| `{f.param}` | {_fmt(f.spec_value)} | {f.unit} |"
                f" {f.reason or ''} |"
            )
    else:
        lines.append("_(none)_")
    lines.append("")

    if report.unexpected_features:
        lines.append(f"## Unexpected features ({len(report.unexpected_features)})")
        lines.append("")
        for feat in report.unexpected_features:
            lines.append(f"- `{feat}`")
        lines.append("")
    return "\n".join(lines)


def render_json(report: ConsistencyReport) -> str:
    return report.model_dump_json(indent=2)


RENDERERS: dict[str, Callable[[ConsistencyReport], str]] = {
    "text":     render_text,
    "markdown": render_markdown,
    "json":     render_json,
}

EXT_FOR_FORMAT: dict[str, str] = {"text": "txt", "markdown": "md", "json": "json"}
