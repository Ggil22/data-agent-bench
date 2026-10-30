"""Aggregate episode records into a Markdown report and a CSV summary."""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

from data_agent_bench.bench.failures import FLAGS, OUTCOMES


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval; well-behaved for small n and rates near 0 or 1."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def summarize(records: list[dict]) -> list[dict]:
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_model[r["model"]].append(r)
    rows = []
    for model, recs in by_model.items():
        n = len(recs)
        k = sum(r["correct"] for r in recs)
        lo, hi = wilson_interval(k, n)
        rows.append(
            {
                "model": model,
                "episodes": n,
                "success_rate": k / n,
                "ci_low": lo,
                "ci_high": hi,
                "mean_steps": sum(r["n_steps"] for r in recs) / n,
                "mean_latency_s": sum(r["latency_s"] for r in recs) / n,
                "prompt_tokens": sum(r["prompt_tokens"] for r in recs),
                "completion_tokens": sum(r["completion_tokens"] for r in recs),
            }
        )
    return sorted(rows, key=lambda row: row["success_rate"], reverse=True)


def render_markdown(records: list[dict], title: str) -> str:
    summary = summarize(records)
    models = [row["model"] for row in summary]
    lines = [f"# {title}", ""]

    lines += [
        "## Overall",
        "",
        "| Model | Episodes | Success | 95% CI | Mean steps | Mean latency (s) | Tokens in / out |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in summary:
        lines.append(
            f"| {row['model']} | {row['episodes']} | {row['success_rate']:.1%} "
            f"| {row['ci_low']:.1%} - {row['ci_high']:.1%} | {row['mean_steps']:.1f} "
            f"| {row['mean_latency_s']:.1f} "
            f"| {row['prompt_tokens']:,} / {row['completion_tokens']:,} |"
        )

    lines += ["", "## Success rate by category", "", _header(["Category", *models])]
    for category in sorted({r["category"] for r in records}):
        cells = []
        for m in models:
            recs = [r for r in records if r["model"] == m and r["category"] == category]
            k = sum(r["correct"] for r in recs)
            cells.append(f"{k}/{len(recs)}" if recs else "-")
        lines.append(f"| {category} | " + " | ".join(cells) + " |")

    lines += ["", "## Outcomes (failure taxonomy)", "", _header(["Outcome", *models])]
    counts = {m: Counter(r["outcome"] for r in records if r["model"] == m) for m in models}
    for outcome in OUTCOMES:
        if any(counts[m][outcome] for m in models):
            lines.append(
                f"| {outcome} | " + " | ".join(str(counts[m][outcome]) for m in models) + " |"
            )

    lines += ["", "## Behaviour flags (share of episodes)", "", _header(["Flag", *models])]
    for flag in FLAGS:
        cells = []
        for m in models:
            recs = [r for r in records if r["model"] == m]
            cells.append(f"{sum(flag in r['flags'] for r in recs) / len(recs):.0%}")
        lines.append(f"| {flag} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Failed episodes",
        "",
        "| Model | Task | Outcome | Prediction | Truth |",
        "|---|---|---|---|---|",
    ]
    for r in sorted(records, key=lambda r: (r["model"], r["task_id"])):
        if not r["correct"]:
            pred = str(r["prediction"]).replace("|", "\\|")[:60]
            lines.append(
                f"| {r['model']} | {r['task_id']} | {r['outcome']} | {pred} | {r['truth']} |"
            )
    return "\n".join(lines) + "\n"


def write_report(records: list[dict], out_dir: Path, title: str) -> Path:
    report_path = out_dir / "report.md"
    report_path.write_text(render_markdown(records, title), encoding="utf-8")
    summary = summarize(records)
    if summary:
        with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
    return report_path


def _header(columns: list[str]) -> str:
    return "| " + " | ".join(columns) + " |\n|" + "---|" * len(columns)
