#!/usr/bin/env python3
"""Memory with no merge rule: full states in, and the model picks what applies.

Every admitted memory keeps the state it was formed in, whole. Nothing is keyed,
hashed or deduplicated. At decision time a Choice offers one line per memory and
asks which past situation resembles the present one; only the chosen memory is
expanded back into the state, so the prompt carries one full snapshot rather
than twenty.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

from typesafe_sdk import Choice

RECALL_INSTRUCTIONS = (
    "Which of these past situations is most like the one you are in now, so that "
    "what happened there tells you something about what to do here?"
)


def snapshot(state: dict[str, Any]) -> dict[str, Any]:
    """The parts of a state worth keeping whole."""
    return {
        "view": state["view"]["grid"],
        "facing": state["facing"]["direction"],
        "vitals": state["vitals"],
        "inventory": state["inventory"],
    }


class Recall:
    def __init__(self, capacity: int = 20) -> None:
        self.capacity = capacity
        self.entries: list[dict[str, Any]] = []

    def add(self, state: dict[str, Any], action: str, effect: str, label: str, step: int) -> None:
        self.entries.append(
            {
                "label": label,
                "action": action,
                "effect": effect,
                "step": step,
                "state": snapshot(state),
            }
        )
        if len(self.entries) > self.capacity:
            self.entries.pop(0)

    def summary(self, entry: dict[str, Any]) -> str:
        """The one line the model chooses between."""
        bag = ", ".join(f"{k} {v}" for k, v in entry["state"]["inventory"].items()) or "empty"
        return (
            f"[{entry['label']}] facing {entry['state']['facing']}, bag {bag}: "
            f"{entry['action'].lower()} -> {entry['effect'][:70]}"
        )

    def question(self) -> Choice | None:
        if not self.entries:
            return None
        criteria: dict[str, None] = {self.summary(e): None for e in self.entries}
        criteria["none of them"] = None
        return Choice(instructions=RECALL_INSTRUCTIONS, criteria=criteria)

    def find(self, chosen: str) -> dict[str, Any] | None:
        for entry in self.entries:
            if self.summary(entry) == chosen:
                return entry
        return None

    def expand(self, entry: dict[str, Any]) -> dict[str, Any]:
        """The full memory, put back into the state for the decision."""
        return {
            "note": (
                "the past situation you judged most like this one, in full. what you "
                "did there and what happened"
            ),
            "then": entry["state"],
            "you_did": entry["action"].lower(),
            "it_led_to": entry["effect"],
            "you_called_it": entry["label"],
        }

    def save(self, path: pathlib.Path) -> None:
        path.write_text(json.dumps(self.entries, indent=1))

    def load(self, path: pathlib.Path) -> int:
        if not path.exists():
            return 0
        self.entries = json.loads(path.read_text())
        return len(self.entries)

    def __len__(self) -> int:
        return len(self.entries)
