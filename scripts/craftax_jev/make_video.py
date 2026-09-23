#!/usr/bin/env python3
"""Render the decision log into a replay video: game frame beside Jev's answers.

Usage:
    python scripts/craftax_jev/make_video.py runs/run-01 --fps 6

Writes composite PNGs next to the run and encodes rollout.mp4 with ffmpeg.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

PAPER = "#fdfbf7"
INK = "#2d2d2d"
SOFT = "#69645f"
RED = "#ff4d4d"
BLUE = "#2d5da1"
BLUE_SOFT = "#dce8f8"
YELLOW = "#fff2a8"

TOP_ACTIONS = 8
MONO = ["DejaVu Sans Mono"]


def draw_actions(ax: Any, record: dict[str, Any]) -> None:
    answer = record["answers"]["action"]
    ranked = sorted(answer["probabilities"].items(), key=lambda kv: kv[1], reverse=True)
    ranked = ranked[:TOP_ACTIONS][::-1]
    names = [name.lower().replace("_", " ") for name, _ in ranked]
    values = [value for _, value in ranked]
    colors = [RED if name == answer["choice"] else BLUE_SOFT for name, _ in ranked]

    bars = ax.barh(range(len(values)), values, color=colors, edgecolor=INK, linewidth=0.6)
    ax.set_yticks(range(len(values)))
    ax.set_yticklabels(names, fontsize=11, color=INK, fontfamily=MONO)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0", ".5", "1"], fontsize=9, color=SOFT, fontfamily=MONO)
    ax.tick_params(length=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(SOFT)
    ax.set_facecolor(PAPER)
    for bar, value in zip(bars, values):
        ax.text(
            min(value + 0.02, 0.92),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            fontsize=9,
            color=SOFT,
            fontfamily=MONO,
        )
    ax.set_title(
        f"choice: action   ·   confidence {answer['confidence']:.2f}",
        fontsize=11,
        color=INK,
        loc="left",
        fontfamily=MONO,
        pad=8,
    )


def draw_signals(ax: Any, record: dict[str, Any]) -> None:
    answers = record["answers"]
    rows = [
        ("priority", answers["priority"]["choice"], answers["priority"]["confidence"]),
        (
            "danger",
            f"{answers['danger']['score']:.2f} / 2",
            answers["danger"]["confidence"],
        ),
    ]
    nouls = [
        ("facing useful", answers["facing_useful"]["noul"]),
        ("can craft now", answers["can_craft_now"]["noul"]),
        ("should explore", answers["should_explore"]["noul"]),
    ]

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(
        "speculative questions, same call",
        fontsize=11,
        color=INK,
        loc="left",
        fontfamily=MONO,
        pad=8,
    )

    for index, (label, value, confidence) in enumerate(rows):
        y = 0.88 - index * 0.2
        ax.text(0, y, label, fontsize=10, color=SOFT, fontfamily=MONO, va="center")
        ax.text(0.34, y, str(value), fontsize=11, color=INK, fontfamily=MONO, va="center")
        ax.text(
            0.74,
            y,
            f"conf {confidence:.2f}",
            fontsize=9,
            color=SOFT,
            fontfamily=MONO,
            va="center",
        )

    for index, (label, value) in enumerate(nouls):
        y = 0.42 - index * 0.16
        ax.text(0, y, label, fontsize=10, color=SOFT, fontfamily=MONO, va="center")
        ax.add_patch(
            plt.Rectangle((0.34, y - 0.035), 0.4, 0.07, facecolor=BLUE_SOFT, edgecolor=INK, linewidth=0.6)
        )
        ax.add_patch(
            plt.Rectangle((0.34, y - 0.035), 0.4 * value, 0.07, facecolor=BLUE, edgecolor="none")
        )
        ax.text(
            0.78, y, f"{value:.2f}", fontsize=9, color=SOFT, fontfamily=MONO, va="center"
        )


def compose(record: dict[str, Any], frame_path: Path, out_path: Path, unlocked: list[str]) -> None:
    vitals = record["state"]["vitals"]
    world = record["state"]["world"]

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=PAPER)

    ax_game = fig.add_axes([0.03, 0.13, 0.40, 0.72])
    ax_game.imshow(plt.imread(frame_path))
    ax_game.axis("off")

    draw_actions(fig.add_axes([0.60, 0.46, 0.34, 0.39]), record)
    draw_signals(fig.add_axes([0.50, 0.13, 0.46, 0.25]), record)

    fig.text(
        0.03,
        0.93,
        f"craftax-classic  ·  step {record['step']:>4}  ·  jev picks: {record['action'].lower()}",
        fontsize=15,
        color=INK,
        fontfamily=MONO,
    )
    fig.text(
        0.03,
        0.89,
        f"hp {vitals['health']}  food {vitals['food']}  drink {vitals['drink']}  "
        f"energy {vitals['energy']}  ·  {world['time_of_day']}",
        fontsize=11,
        color=SOFT,
        fontfamily=MONO,
    )

    achievement_text = ", ".join(name.replace("_", " ") for name in unlocked) or "none yet"
    fig.text(
        0.03,
        0.055,
        f"achievements ({len(unlocked)}/22): {achievement_text}",
        fontsize=10,
        color=INK if unlocked else SOFT,
        fontfamily=MONO,
        wrap=True,
    )
    if record["new_achievements"]:
        fig.text(
            0.03,
            0.015,
            "unlocked: " + ", ".join(n.replace("_", " ") for n in record["new_achievements"]),
            fontsize=11,
            color=RED,
            fontfamily=MONO,
            bbox=dict(facecolor=YELLOW, edgecolor="none", pad=3),
        )

    fig.savefig(out_path, facecolor=PAPER)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0, help="only render N steps")
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in (args.run_dir / "decisions.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if args.limit:
        records = records[: args.limit]

    composites = args.run_dir / "composites"
    composites.mkdir(exist_ok=True)

    unlocked: list[str] = []
    for record in records:
        unlocked = unlocked + [
            name for name in record["new_achievements"] if name not in unlocked
        ]
        compose(
            record,
            args.run_dir / "frames" / f"{record['step']:05d}.png",
            composites / f"{record['step']:05d}.png",
            unlocked,
        )
        if record["step"] % 50 == 0:
            print(f"composed step {record['step']}", flush=True)

    out_path = args.run_dir / "rollout.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-framerate", str(args.fps),
            "-i", str(composites / "%05d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-crf", "23",
            str(out_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
