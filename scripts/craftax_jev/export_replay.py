#!/usr/bin/env python3
"""Turn decision logs into a scrubbable replay: one sprite atlas, one JSON per run.

Usage:
    python scripts/craftax_jev/export_replay.py public/data/demos/craftax-jev \
        naive=runs/run-01 suppress=runs/run-04-suppress pure=runs/pure-seed0

The atlas is built from Craftax's own 16x16 textures (MIT licensed, see the
Craftax repository). The viewer in the research note draws the 7x9 view from
tile indices, so a 200-step replay is about 80 kB instead of a 1.6 MB video.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from craftax.craftax_classic.constants import Action
from PIL import Image

ASSETS = Path(__file__).parent / ".venv/lib/python3.12/site-packages/craftax/craftax_classic/assets"

# Tile order is the atlas order; the JSON stores indices into this list.
TILES: list[tuple[str, str]] = [
    ("invalid", "debug_tile.png"),
    ("out_of_bounds", "debug_tile.png"),
    ("grass", "grass.png"),
    ("water", "water.png"),
    ("stone", "stone.png"),
    ("tree", "tree.png"),
    ("wood", "wood.png"),
    ("path", "path.png"),
    ("coal", "coal.png"),
    ("iron", "iron.png"),
    ("diamond", "diamond.png"),
    ("crafting_table", "table.png"),
    ("furnace", "furnace.png"),
    ("sand", "sand.png"),
    ("lava", "lava.png"),
    ("plant", "plant_on_grass.png"),
    ("ripe_plant", "ripe_plant_on_grass.png"),
    ("zombie", "zombie.png"),
    ("cow", "cow.png"),
    ("skeleton", "skeleton.png"),
    ("arrow", "arrow-up.png"),
    ("you", "player-down.png"),
    ("you_left", "player-left.png"),
    ("you_right", "player-right.png"),
    ("you_up", "player-up.png"),
    ("you_down", "player-down.png"),
    ("you_sleep", "player-sleep.png"),
]

TILE_INDEX = {name: index for index, (name, _) in enumerate(TILES)}
# Sprites with transparency need grass painted under them.
OVERLAY = {"zombie", "cow", "skeleton", "arrow", "you", "you_left", "you_right", "you_up", "you_down", "you_sleep"}
ACTION_ORDER = [action.name for action in Action]
SIZE = 16


def build_atlas(out_path: Path) -> None:
    atlas = Image.new("RGBA", (SIZE * len(TILES), SIZE))
    for index, (_, filename) in enumerate(TILES):
        tile = Image.open(ASSETS / filename).convert("RGBA").resize((SIZE, SIZE), Image.NEAREST)
        atlas.paste(tile, (index * SIZE, 0))
    atlas.save(out_path)
    print(f"atlas: {out_path} ({out_path.stat().st_size / 1024:.0f} kB, {len(TILES)} tiles)")


def facing_tile_name(token: str, direction: str) -> str:
    return f"you_{direction}" if token == "you" and direction in ("left", "right", "up", "down") else token


def export_run(run_dir: Path, out_path: Path, label: str) -> None:
    rows = [
        json.loads(line)
        for line in (run_dir / "decisions.jsonl").read_text().splitlines()
        if line.strip()
    ]
    summary = json.loads((run_dir / "summary.json").read_text())

    steps = []
    unlocked: list[str] = []
    last_memory: list[str] | None = None
    for row in rows:
        state, answers = row["state"], row["answers"]
        direction = state["facing"]["direction"]
        grid = []
        for line in state["view"]["grid"]:
            for token in line.split():
                name = facing_tile_name(token, direction)
                grid.append(TILE_INDEX.get(name, TILE_INDEX["invalid"]))
        unlocked = unlocked + [a for a in row["new_achievements"] if a not in unlocked]
        vitals = state["vitals"]
        steps.append(
            {
                "g": grid,
                "d": direction,
                "v": [vitals["health"], vitals["food"], vitals["drink"], vitals["energy"]],
                "inv": state["inventory"],
                "a": ACTION_ORDER.index(row["action"]),
                "p": [
                    round(1000 * answers["action"]["probabilities"].get(name, 0.0))
                    for name in ACTION_ORDER
                ],
                "c": round(answers["action"]["confidence"], 3),
                "pr": [answers["priority"]["choice"], round(answers["priority"]["confidence"], 2)],
                "dg": [round(answers["danger"]["score"], 2), round(answers["danger"]["confidence"], 2)],
                "n": [
                    round(answers["facing_useful"]["noul"], 3),
                    round(answers["can_craft_now"]["noul"], 3),
                    round(answers["should_explore"]["noul"], 3),
                ],
                "e": row.get("effect", ""),
                "b": row.get("blocked_options", []),
                "ach": list(unlocked),
                "new": row["new_achievements"],
                "light": state["world"]["light_level"],
            }
        )
        kept = (state.get("memory") or {}).get("kept")
        if kept is not None and kept != last_memory:
            # The list only changes when something is written, so checkpoints are exact.
            steps[-1]["mem"] = kept
            last_memory = kept
        if row.get("memory_new"):
            steps[-1]["mn"] = row["memory_new"]

    payload = {
        "meta": {
            "label": label,
            "run": run_dir.name,
            "tiles": [name for name, _ in TILES],
            "overlay": sorted(OVERLAY),
            "actions": ACTION_ORDER,
            "cols": 9,
            "rows": 7,
            "tile_size": SIZE,
            "summary": summary,
            "has_memory": any("mem" in step for step in steps),
        },
        "steps": steps,
    }
    out_path.write_text(json.dumps(payload, separators=(",", ":")))
    print(
        f"{label}: {out_path.name} — {len(steps)} steps, "
        f"{out_path.stat().st_size / 1024:.0f} kB"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("runs", nargs="+", help="label=path/to/run pairs")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    build_atlas(args.out_dir / "tiles.png")
    for pair in args.runs:
        label, _, path = pair.partition("=")
        export_run(Path(path), args.out_dir / f"replay-{label}.json", label)


if __name__ == "__main__":
    main()
