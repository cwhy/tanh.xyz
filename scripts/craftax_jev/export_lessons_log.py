#!/usr/bin/env python3
"""Write the memory log for a lessons run: every admission decision, in order.

Usage:
    python scripts/craftax_jev/export_lessons_log.py runs/lessons-seed0 out.md

Shows what the model was asked to judge, what it called it, whether memory
already covered it, and what survived — so the store can be read as a history
of decisions rather than a final list.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

MARK = {"reinforce": "+", "avoid": "-", "neither": "."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("out_path", type=Path)
    parser.add_argument("--wrap", type=int, default=150, help="truncate transitions to N chars")
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in (args.run_dir / "decisions.jsonl").read_text().splitlines()
        if line.strip()
    ]
    summary = json.loads((args.run_dir / "summary.json").read_text())
    judged = [r for r in rows if r.get("lesson_label")]
    candidates = [r for r in judged if r["lesson_label"] != "neither"]
    stored = [r for r in rows if r.get("lesson_new")]
    coverage = [r["lesson_coverage"] for r in judged if r["lesson_coverage"] is not None]

    out: list[str] = []
    out.append(f"# Memory log — {args.run_dir.name}\n")
    out.append(
        f"{summary['steps']} steps · {len(summary['achievements'])} achievements "
        f"({', '.join(summary['achievements']) or 'none'}) · return {summary['total_reward']} "
        f"· seed {summary['seed']} · {summary['model']}\n"
    )
    out.append(
        f"\n**Admission:** {len(judged)} transitions judged · "
        f"{len(candidates)} labelled reinforce or avoid · "
        f"{len(candidates) - len(stored)} rejected as already covered · "
        f"**{len(stored)} kept**\n"
    )
    if coverage:
        out.append(
            f"\n**Coverage score:** min {min(coverage):.2f} · median "
            f"{statistics.median(coverage):.2f} · max {max(coverage):.2f} · "
            f"{100 * sum(1 for c in coverage if c >= 0.5) / len(coverage):.0f}% at or above the 0.5 bar\n"
        )

    out.append("\n## Every admission decision\n")
    out.append("\n`+` reinforce · `-` avoid · **KEPT** stored · *covered* rejected\n")
    out.append("\n| step | call | coverage | outcome | transition |\n|---|---|---|---|---|\n")
    for row in candidates:
        text = row["state"]["last_transition"] or ""
        if len(text) > args.wrap:
            text = text[: args.wrap] + "…"
        outcome = "**KEPT**" if row["lesson_new"] else "*covered*"
        out.append(
            f"| {row['step']} | {MARK[row['lesson_label']]} {row['lesson_label']} | "
            f"{row['lesson_coverage']:.2f} | {outcome} | `{text}` |\n"
        )

    final = rows[-1]["state"].get("lessons", {"reinforce": [], "avoid": []})
    for label in ("reinforce", "avoid"):
        out.append(f"\n## Final store — {label} ({len(final[label])})\n\n")
        for entry in final[label]:
            out.append(f"- `{entry}`\n")

    out.append("\n## Store growth\n\n")
    marks = [(r["step"], r["lessons_size"]) for r in stored]
    out.append(", ".join(f"step {step} → {size}" for step, size in marks) + "\n")

    args.out_path.write_text("".join(out))
    print(f"wrote {args.out_path} ({args.out_path.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
