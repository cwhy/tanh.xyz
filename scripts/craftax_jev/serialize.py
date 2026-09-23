#!/usr/bin/env python3
"""Turn a Craftax-Classic EnvState into the JSON state a System One model reads.

Jev accepts text only (string, object, or array), so the symbolic environment
state has to be flattened into named fields: a 7x9 view of named blocks, the
tile the player faces, vitals, inventory, and what is reachable nearby.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from feedback import TOKEN_TYPES
from craftax.craftax_classic.constants import (
    CLOSE_BLOCKS,
    OBS_DIM,
    Achievement,
    BlockType,
)

BLOCK_NAMES = {block.value: block.name.lower() for block in BlockType}
DIRECTION_NAMES = {0: "none", 1: "left", 2: "right", 3: "up", 4: "down"}
DIRECTION_OFFSETS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}

# Blocks and mobs worth reporting a bearing for.
POINTS_OF_INTEREST = (
    "tree",
    "water",
    "stone",
    "coal",
    "iron",
    "diamond",
    "crafting_table",
    "furnace",
    "sand",
    "lava",
    "ripe_plant",
    "cow",
    "zombie",
    "skeleton",
)

MOB_FIELDS = (("zombie", "zombies"), ("cow", "cows"), ("skeleton", "skeletons"))


def _bearing(d_row: int, d_col: int) -> str:
    parts = []
    if d_row < 0:
        parts.append(f"{-d_row} up")
    elif d_row > 0:
        parts.append(f"{d_row} down")
    if d_col < 0:
        parts.append(f"{-d_col} left")
    elif d_col > 0:
        parts.append(f"{d_col} right")
    return ", ".join(parts) if parts else "here"


def local_view(state: Any) -> np.ndarray:
    """The 7x9 window of block ids centred on the player, as Craftax renders it."""
    height, width = OBS_DIM
    pad = max(height, width) + 2
    grid = np.pad(
        np.asarray(state.map), pad, constant_values=BlockType.OUT_OF_BOUNDS.value
    )
    row, col = np.asarray(state.player_position)
    top, left = row - height // 2 + pad, col - width // 2 + pad
    return grid[top : top + height, left : left + width]


def view_tokens(state: Any) -> list[list[str]]:
    """The local view as block names, with visible mobs drawn over their tile."""
    height, width = OBS_DIM
    tokens = [[BLOCK_NAMES[int(v)] for v in row] for row in local_view(state)]
    player = np.asarray(state.player_position)
    for name, field in MOB_FIELDS:
        mobs = getattr(state, field)
        for index, alive in enumerate(np.asarray(mobs.mask)):
            if not alive:
                continue
            row, col = np.asarray(mobs.position[index]) - player
            row, col = int(row) + height // 2, int(col) + width // 2
            if 0 <= row < height and 0 <= col < width:
                tokens[row][col] = name
    tokens[height // 2][width // 2] = "you"
    return tokens


def nearest_features(tokens: list[list[str]]) -> dict[str, str]:
    """Closest instance of each point of interest, as a bearing from the player."""
    height, width = OBS_DIM
    best: dict[str, tuple[int, str]] = {}
    for row_index, row in enumerate(tokens):
        for col_index, token in enumerate(row):
            if token not in POINTS_OF_INTEREST:
                continue
            d_row, d_col = row_index - height // 2, col_index - width // 2
            distance = abs(d_row) + abs(d_col)
            if token not in best or distance < best[token][0]:
                best[token] = (distance, _bearing(d_row, d_col))
    return {name: f"{where} ({distance} tiles)" for name, (distance, where) in best.items()}


def is_near(state: Any, block: BlockType) -> bool:
    """Craftax counts a block as usable when it sits in the 8 tiles around you."""
    grid = np.asarray(state.map)
    row, col = np.asarray(state.player_position)
    for d_row, d_col in np.asarray(CLOSE_BLOCKS):
        probe_row, probe_col = int(row + d_row), int(col + d_col)
        if 0 <= probe_row < grid.shape[0] and 0 <= probe_col < grid.shape[1]:
            if grid[probe_row, probe_col] == block.value:
                return True
    return False


def facing_block(state: Any) -> str:
    grid = np.asarray(state.map)
    row, col = np.asarray(state.player_position)
    d_row, d_col = DIRECTION_OFFSETS.get(int(state.player_direction), (0, 0))
    probe_row, probe_col = int(row + d_row), int(col + d_col)
    if not (0 <= probe_row < grid.shape[0] and 0 <= probe_col < grid.shape[1]):
        return "out_of_bounds"
    tokens = view_tokens(state)
    height, width = OBS_DIM
    return tokens[height // 2 + d_row][width // 2 + d_col]


def time_of_day(light_level: float) -> str:
    if light_level < 0.25:
        return "night"
    if light_level < 0.7:
        return "dusk or dawn"
    return "day"


def serialize(
    state: Any,
    level: str = "engineered",
    position: bool = False,
    world_frame: bool = False,
) -> dict[str, Any]:
    """The state object sent as `state` in one System One request.

    `engineered` adds what I know about Craftax: bearings to the things I decided
    are interesting, the reach rule for workstations, the tile in front resolved
    for you, and the list of achievements left to get. `raw` drops all of it and
    sends the observation: the grid, which way you face, your body, your bag.
    """
    tokens = view_tokens(state)
    inventory = {
        name: int(getattr(state.inventory, name))
        for name in (
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
    }
    unlocked = [
        achievement.name.lower()
        for achievement in Achievement
        if bool(np.asarray(state.achievements)[achievement.value])
    ]
    light = float(state.light_level)
    if level == "raw":
        spot = np.asarray(state.player_position)
        height, width = OBS_DIM
        view: dict[str, Any] = {
            "note": "7 rows by 9 columns, you are at the centre; rows run top to bottom",
            "grid": [" ".join(row) for row in tokens],
        }
        if world_frame:
            view["tile_types"] = {
                str(index): name for index, name in enumerate(TOKEN_TYPES)
            }
            view["origin"] = {
                "x": int(spot[1]) - width // 2,
                "y": int(spot[0]) - height // 2,
            }
            view["note"] += (
                "; origin is the world coordinate of the top-left cell, x grows "
                "right and y grows down"
            )
        return {
            "view": view,
            "facing": {"direction": DIRECTION_NAMES[int(state.player_direction)]},
            **(
                {
                    "position": {
                        "x": int(spot[1]),
                        "y": int(spot[0]),
                        "note": "where you are on a 48 by 48 world; x grows right, y grows down",
                    }
                }
                if position or world_frame
                else {}
            ),
            "vitals": {
                "health": int(state.player_health),
                "food": int(state.player_food),
                "drink": int(state.player_drink),
                "energy": int(state.player_energy),
                "max_for_each": 9,
                "sleeping": bool(state.is_sleeping),
            },
            "inventory": {name: count for name, count in inventory.items() if count},
            "world": {"light_level": round(light, 2), "timestep": int(state.timestep)},
            "achievements_unlocked": unlocked,
        }

    return {
        "view": {
            "note": "7 rows by 9 columns, you are at the centre; rows run top to bottom",
            "grid": [" ".join(row) for row in tokens],
        },
        "facing": {
            "direction": DIRECTION_NAMES[int(state.player_direction)],
            "tile_in_front": facing_block(state),
        },
        "nearest": nearest_features(tokens),
        "vitals": {
            "health": int(state.player_health),
            "food": int(state.player_food),
            "drink": int(state.player_drink),
            "energy": int(state.player_energy),
            "max_for_each": 9,
            "sleeping": bool(state.is_sleeping),
        },
        "inventory": {name: count for name, count in inventory.items() if count},
        "workstations_in_reach": {
            "crafting_table": is_near(state, BlockType.CRAFTING_TABLE),
            "furnace": is_near(state, BlockType.FURNACE),
        },
        "world": {
            "time_of_day": time_of_day(light),
            "light_level": round(light, 2),
            "timestep": int(state.timestep),
        },
        "achievements_unlocked": unlocked,
        "achievements_remaining": [
            achievement.name.lower()
            for achievement in Achievement
            if achievement.name.lower() not in unlocked
        ],
    }
