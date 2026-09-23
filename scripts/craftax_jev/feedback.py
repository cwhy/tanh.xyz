#!/usr/bin/env python3
"""Exact effect of one Craftax transition, and a tally of what has failed here.

The point is to separate what the action did from what the world did on its own.
Hunger ticking down, a zombie stepping sideways and the light level falling are
not effects of pressing a button, so none of them count here.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import numpy as np
from craftax.craftax_classic.constants import Action

INVENTORY_FIELDS = (
    "wood",
    "stone",
    "coal",
    "iron",
    "diamond",
    "sapling",
    "wood_pickaxe",
    "stone_pickaxe",
    "iron_pickaxe",
    "wood_sword",
    "stone_sword",
    "iron_sword",
)
MOB_FIELDS = ("zombies", "cows", "skeletons")
DIRECTION_NAMES = {0: "none", 1: "left", 2: "right", 3: "up", 4: "down"}


def _inventory_delta(prev: Any, nxt: Any) -> dict[str, int]:
    delta = {}
    for name in INVENTORY_FIELDS:
        change = int(getattr(nxt.inventory, name)) - int(getattr(prev.inventory, name))
        if change:
            delta[name] = change
    return delta


def _live_mob_health(state: Any) -> int:
    return sum(
        int(np.sum(np.asarray(getattr(state, field).health) * np.asarray(getattr(state, field).mask)))
        for field in MOB_FIELDS
    )


def describe_effect_full(prev: Any, nxt: Any) -> str:
    """Every action-caused change, not just the first one found."""
    parts: list[str] = []
    delta = _inventory_delta(prev, nxt)
    if delta:
        parts.extend(f"{name} {change:+d}".replace("_", " ") for name, change in delta.items())
    gained = int(np.asarray(nxt.achievements).sum()) - int(np.asarray(prev.achievements).sum())
    if gained > 0:
        parts.append("unlocked an achievement")
    if not np.array_equal(np.asarray(prev.map), np.asarray(nxt.map)):
        parts.append("changed a block")
    if _live_mob_health(nxt) < _live_mob_health(prev):
        parts.append("hit a mob")
    if bool(prev.is_sleeping) != bool(nxt.is_sleeping):
        parts.append("fell asleep" if bool(nxt.is_sleeping) else "woke up")
    if not np.array_equal(np.asarray(prev.player_position), np.asarray(nxt.player_position)):
        parts.append("moved")
    if int(prev.player_direction) != int(nxt.player_direction):
        parts.append(f"turned to face {DIRECTION_NAMES[int(nxt.player_direction)]}")
    return ", ".join(parts) if parts else "nothing happened"


def describe_effect(prev: Any, nxt: Any) -> str:
    """One short phrase for what the action actually did."""
    delta = _inventory_delta(prev, nxt)
    if delta:
        return ", ".join(
            f"{name} {change:+d}".replace("_", " ") for name, change in delta.items()
        )

    prev_achievements = np.asarray(prev.achievements)
    if int(np.asarray(nxt.achievements).sum()) > int(prev_achievements.sum()):
        return "unlocked an achievement"

    if not np.array_equal(np.asarray(prev.map), np.asarray(nxt.map)):
        return "changed a block"

    if _live_mob_health(nxt) < _live_mob_health(prev):
        return "hit a mob"

    if bool(prev.is_sleeping) != bool(nxt.is_sleeping):
        return "fell asleep" if bool(nxt.is_sleeping) else "woke up"

    if not np.array_equal(
        np.asarray(prev.player_position), np.asarray(nxt.player_position)
    ):
        return "moved"

    if int(prev.player_direction) != int(nxt.player_direction):
        return f"only turned to face {DIRECTION_NAMES[int(nxt.player_direction)]}"

    return "nothing happened"


def situation_key(state: Any, facing_tile: str) -> tuple:
    """Same tile, same facing, same thing in front — so a failure still applies."""
    row, col = np.asarray(state.player_position)
    return (int(row), int(col), int(state.player_direction), facing_tile)


class FailureTally:
    """Counts actions that provably did nothing in this exact situation."""

    def __init__(self) -> None:
        self._counts: dict[tuple, dict[str, int]] = {}

    def record(
        self, key: tuple, action_name: str, effect: str, null_effect: str = "nothing happened"
    ) -> None:
        bucket = self._counts.setdefault(key, {})
        if effect == null_effect:
            bucket[action_name] = bucket.get(action_name, 0) + 1
        else:
            bucket.pop(action_name, None)

    def failures(self, key: tuple) -> dict[str, int]:
        return dict(self._counts.get(key, {}))

    def blocked(self, key: tuple, threshold: int) -> set[str]:
        return {
            name
            for name, count in self._counts.get(key, {}).items()
            if count >= threshold
        }


ALL_ACTIONS = [action.name for action in Action]


class Memory:
    """A long-term store the model curates itself.

    Every tick, one Noul asks whether the transition that just happened is worth
    remembering. Code does the bookkeeping and nothing else: identical entries
    merge into a count, and when the store is full the lowest-scored entry goes.
    """

    def __init__(self, capacity: int = 24, threshold: float = 0.5) -> None:
        self.capacity = capacity
        self.threshold = threshold
        self._entries: dict[str, dict[str, Any]] = {}

    def consider(self, text: str, score: float, step: int) -> bool:
        if score < self.threshold:
            return False
        entry = self._entries.get(text)
        if entry is None:
            self._entries[text] = {"score": score, "count": 1, "last_step": step}
        else:
            entry["score"] = max(entry["score"], score)
            entry["count"] += 1
            entry["last_step"] = step
        if len(self._entries) > self.capacity:
            weakest = min(self._entries.items(), key=lambda kv: (kv[1]["score"], kv[1]["last_step"]))
            del self._entries[weakest[0]]
        return True

    def entries(self) -> list[str]:
        ordered = sorted(
            self._entries.items(),
            key=lambda kv: (kv[1]["score"], kv[1]["last_step"]),
            reverse=True,
        )
        return [
            text + (f" (happened {entry['count']} times)" if entry["count"] > 1 else "")
            for text, entry in ordered
        ]

    def __len__(self) -> int:
        return len(self._entries)


# --- prior-free versions of the two leaks -----------------------------------
#
# describe_effect() names Craftax's own fields and decides which of them count
# as caused by the action. situation_key() uses the tile you face, which is the
# game's rule about what an action acts on. The two below replace that judgment
# with a diff of the observation the model itself receives: code reports what
# visibly changed and the model decides what it means.

MAX_LISTED_CELLS = 4

# What an action that achieved nothing reports. The wording matters: the old
# "nothing you can see changed" reads as though the world were inert, when what
# happened is that the action was refused. This says so, without naming any rule
# of this particular game.
NO_EFFECT = (
    "no effect - the action was not possible here, so something it needs was "
    "missing or you were in the wrong spot. the game is working; work out what."
)


def _grid_cells(observation: dict[str, Any]) -> list[list[str]]:
    return [line.split() for line in observation["view"]["grid"]]


def observation_diff(
    prev_obs: dict[str, Any], next_obs: dict[str, Any], max_cells: int = MAX_LISTED_CELLS
) -> str:
    """What changed in the observation, with no opinion about why.

    `max_cells = 0` lists every changed cell instead of summarising them, so the
    transition is the complete diff rather than a digest of it.
    """
    parts: list[str] = []

    changed_cells = []
    for row, (before, after) in enumerate(zip(_grid_cells(prev_obs), _grid_cells(next_obs))):
        for col, (was, now) in enumerate(zip(before, after)):
            if was != now:
                changed_cells.append(f"view r{row}c{col} {was}->{now}")
    if max_cells == 0 or len(changed_cells) <= max_cells:
        parts.extend(changed_cells)
    else:
        parts.append(f"view changed in {len(changed_cells)} places")

    if prev_obs["facing"]["direction"] != next_obs["facing"]["direction"]:
        parts.append(
            f"facing {prev_obs['facing']['direction']}->{next_obs['facing']['direction']}"
        )

    for field, was in prev_obs["vitals"].items():
        now = next_obs["vitals"].get(field)
        if now != was:
            parts.append(f"{field} {was}->{now}")

    for item in sorted(set(prev_obs["inventory"]) | set(next_obs["inventory"])):
        was = prev_obs["inventory"].get(item, 0)
        now = next_obs["inventory"].get(item, 0)
        if was != now:
            parts.append(f"{item} {was}->{now}")

    gained = set(next_obs["achievements_unlocked"]) - set(prev_obs["achievements_unlocked"])
    for name in sorted(gained):
        parts.append(f"achievement {name}")

    return ", ".join(parts) if parts else NO_EFFECT


def view_key(observation: dict[str, Any]) -> tuple:
    """Same visible surroundings, same heading — no notion of a tile in front."""
    return (observation["facing"]["direction"], tuple(observation["view"]["grid"]))


def collapse_repeats(lines: list[str]) -> list[str]:
    """Run-length collapse so a stuck stretch costs one line, not eighty."""
    collapsed: list[str] = []
    for line in lines:
        if collapsed and collapsed[-1][0] == line:
            collapsed[-1][1] += 1
        else:
            collapsed.append([line, 1])
    return [text if count == 1 else f"{text}  (x{count} in a row)" for text, count in collapsed]


def _world_cells(observation: dict[str, Any]) -> dict[tuple[int, int], str]:
    """The view keyed by world coordinate rather than by position on screen."""
    origin = observation["view"]["origin"]
    cells: dict[tuple[int, int], str] = {}
    for row, line in enumerate(observation["view"]["grid"]):
        for col, token in enumerate(line.split()):
            cells[(origin["x"] + col, origin["y"] + row)] = token
    return cells


def world_diff(prev_obs: dict[str, Any], next_obs: dict[str, Any]) -> str:
    """Every change in world coordinates: what changed, what came into view, what left.

    A tile that stays visible and stays the same produces nothing, so walking no
    longer rewrites the whole view — it reports the edge that appeared and the
    edge that fell away.
    """
    before, after = _world_cells(prev_obs), _world_cells(next_obs)
    parts: list[str] = []

    for spot in sorted(set(before) & set(after)):
        if before[spot] != after[spot] and "you" not in (before[spot], after[spot]):
            parts.append(f"({spot[0]},{spot[1]}) {before[spot]}->{after[spot]}")

    appeared = sorted(set(after) - set(before))
    if appeared:
        parts.append(
            "came into view: "
            + ", ".join(f"({x},{y}) {after[(x, y)]}" for x, y in appeared)
        )

    gone = sorted(set(before) - set(after))
    if gone:
        parts.append(
            "left view: " + ", ".join(f"({x},{y}) {before[(x, y)]}" for x, y in gone)
        )

    if prev_obs["facing"]["direction"] != next_obs["facing"]["direction"]:
        parts.append(
            f"facing {prev_obs['facing']['direction']}->{next_obs['facing']['direction']}"
        )
    if prev_obs.get("position") != next_obs.get("position"):
        here = next_obs["position"]
        parts.append(f"you are now at ({here['x']},{here['y']})")

    for field, was in prev_obs["vitals"].items():
        now = next_obs["vitals"].get(field)
        if now != was:
            parts.append(f"{field} {was}->{now}")

    for item in sorted(set(prev_obs["inventory"]) | set(next_obs["inventory"])):
        was, now = prev_obs["inventory"].get(item, 0), next_obs["inventory"].get(item, 0)
        if was != now:
            parts.append(f"{item} {was}->{now}")

    for name in sorted(
        set(next_obs["achievements_unlocked"]) - set(prev_obs["achievements_unlocked"])
    ):
        parts.append(f"achievement {name}")

    return ", ".join(parts) if parts else NO_EFFECT


class Lessons:
    """Memory as a set of distinct lessons, admitted by the model.

    Two questions decide everything: whether the transition is something to
    reinforce or avoid, and whether memory already covers it. There is no merge
    rule, no count, and no capacity — a lesson is either new or it is not, and
    the text stays exactly as the transition was recorded.
    """

    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []

    def consider(self, text: str, label: str, covered: bool, step: int) -> bool:
        if label == "neither" or covered:
            return False
        self.entries.append({"text": text, "label": label, "step": step})
        return True

    def save(self, path: "pathlib.Path") -> None:
        path.write_text(json.dumps(self.entries, indent=1))

    def load(self, path: "pathlib.Path") -> int:
        """Carry what earlier lives learned into this one."""
        if not path.exists():
            return 0
        self.entries = json.loads(path.read_text())
        return len(self.entries)

    def by_label(self, label: str) -> list[str]:
        return [entry["text"] for entry in self.entries if entry["label"] == label]

    def __len__(self) -> int:
        return len(self.entries)


# --- compact world-frame: types indexed once, entries carry coordinates --------

TOKEN_TYPES = [
    "invalid", "out_of_bounds", "grass", "water", "stone", "tree", "wood", "path",
    "coal", "iron", "diamond", "crafting_table", "furnace", "sand", "lava", "plant",
    "ripe_plant", "zombie", "cow", "skeleton", "arrow", "you",
]
TOKEN_INDEX = {name: index for index, name in enumerate(TOKEN_TYPES)}


def _group_by_type(cells: dict[tuple[int, int], str], spots: list[tuple[int, int]]) -> str:
    """`3@(28,28),(29,28) 4@(30,28)` — the type once, then its coordinates."""
    grouped: dict[int, list[tuple[int, int]]] = {}
    for spot in spots:
        grouped.setdefault(TOKEN_INDEX.get(cells[spot], 0), []).append(spot)
    return " ".join(
        f"{kind}@" + ",".join(f"({x},{y})" for x, y in sorted(coords))
        for kind, coords in sorted(grouped.items())
    )


def world_diff_compact(prev_obs: dict[str, Any], next_obs: dict[str, Any]) -> str:
    """The world-frame diff with type names replaced by indices into TOKEN_TYPES."""
    before, after = _world_cells(prev_obs), _world_cells(next_obs)
    parts: list[str] = []

    for spot in sorted(set(before) & set(after)):
        was, now = before[spot], after[spot]
        if was != now and "you" not in (was, now):
            parts.append(
                f"({spot[0]},{spot[1]}) {TOKEN_INDEX.get(was, 0)}->{TOKEN_INDEX.get(now, 0)}"
            )

    appeared = sorted(set(after) - set(before))
    if appeared:
        parts.append("in: " + _group_by_type(after, appeared))

    gone = sorted(set(before) - set(after))
    if gone:
        parts.append("out: " + _group_by_type(before, gone))

    if prev_obs["facing"]["direction"] != next_obs["facing"]["direction"]:
        parts.append(f"facing {prev_obs['facing']['direction']}->{next_obs['facing']['direction']}")
    if prev_obs.get("position") != next_obs.get("position"):
        here = next_obs["position"]
        parts.append(f"you at ({here['x']},{here['y']})")

    for field, was in prev_obs["vitals"].items():
        now = next_obs["vitals"].get(field)
        if now != was:
            parts.append(f"{field} {was}->{now}")

    for item in sorted(set(prev_obs["inventory"]) | set(next_obs["inventory"])):
        was, now = prev_obs["inventory"].get(item, 0), next_obs["inventory"].get(item, 0)
        if was != now:
            parts.append(f"{item} {was}->{now}")

    for name in sorted(
        set(next_obs["achievements_unlocked"]) - set(prev_obs["achievements_unlocked"])
    ):
        parts.append(f"achievement {name}")

    return ", ".join(parts) if parts else NO_EFFECT


def world_diff_general(prev_obs: dict[str, Any], next_obs: dict[str, Any]) -> str:
    """The same transition with nothing place-specific left in it.

    Coordinates are dropped and so is the view scrolling, because neither can
    recur: a lesson worth carrying is about what a kind of tile does when you
    act on it, not about the tile at (33,31). Repeated identical changes collapse
    to one, so chopping two trees in one step still reads as one lesson.
    """
    before, after = _world_cells(prev_obs), _world_cells(next_obs)
    parts: list[str] = []

    changes: list[str] = []
    for spot in sorted(set(before) & set(after)):
        was, now = before[spot], after[spot]
        if was != now and "you" not in (was, now):
            change = f"{was}->{now}"
            if change not in changes:
                changes.append(change)
    parts.extend(changes)

    if prev_obs["facing"]["direction"] != next_obs["facing"]["direction"]:
        parts.append("turned")

    for field, was in prev_obs["vitals"].items():
        now = next_obs["vitals"].get(field)
        if now != was and isinstance(now, int) and isinstance(was, int):
            parts.append(f"{field} {'up' if now > was else 'down'}")
        elif now != was:
            parts.append(f"{field} {was}->{now}")

    for item in sorted(set(prev_obs["inventory"]) | set(next_obs["inventory"])):
        was, now = prev_obs["inventory"].get(item, 0), next_obs["inventory"].get(item, 0)
        if was != now:
            parts.append(f"{item} {was}->{now}")

    for name in sorted(
        set(next_obs["achievements_unlocked"]) - set(prev_obs["achievements_unlocked"])
    ):
        parts.append(f"achievement {name}")

    return ", ".join(parts) if parts else NO_EFFECT
