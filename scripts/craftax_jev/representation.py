#!/usr/bin/env python3
"""A representation the model builds for itself, instead of one I hand it.

Two running estimates, re-asked every turn against the previous value:

  object accessibility  - for each kind of thing in view, can it be acted on soon
  action accessibility  - for each action, would it do anything if taken now

Both are carried back into the state, so each turn updates a belief rather than
starting over. On top of them sits a preference table keyed by the accessibility
signature: state-action pairs the model scores good or bad, added when a new
combination shows up and re-scored every step.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

from typesafe_sdk import Noul

OBJECT_PREFIX = "obj__"
ACTION_PREFIX = "act__"
PREF_PREFIX = "pref__"
ACCESSIBLE_AT = 0.5


class Accessibility:
    """Running estimates of what is reachable and what is possible."""

    def __init__(self) -> None:
        self.objects: dict[str, float] = {}
        self.actions: dict[str, float] = {}

    def questions(self, visible: list[str], actions: list[str]) -> dict[str, Noul]:
        asked: dict[str, Noul] = {}
        for name in visible:
            previous = self.objects.get(name)
            hint = (
                f" Your last estimate for this was {previous:.2f}; update it."
                if previous is not None
                else ""
            )
            asked[OBJECT_PREFIX + name] = Noul(
                instructions=(
                    f"You can act on a {name.replace('_', ' ')} within the next step "
                    f"or two, without anything getting in the way.{hint}"
                )
            )
        for name in actions:
            previous = self.actions.get(name)
            hint = (
                f" Your last estimate for this was {previous:.2f}; update it."
                if previous is not None
                else ""
            )
            asked[ACTION_PREFIX + name] = Noul(
                instructions=(
                    f"The action {name} would actually do something if you took it "
                    f"right now, rather than being refused.{hint}"
                )
            )
        return asked

    def update(self, answers: dict[str, Any]) -> None:
        for key, answer in answers.items():
            if key.startswith(OBJECT_PREFIX):
                self.objects[key[len(OBJECT_PREFIX) :]] = answer["noul"]
            elif key.startswith(ACTION_PREFIX):
                self.actions[key[len(ACTION_PREFIX) :]] = answer["noul"]

    def signature(self) -> str:
        """The compact state description the preference table is keyed on."""
        reachable = sorted(
            name for name, value in self.objects.items() if value >= ACCESSIBLE_AT
        )
        return "+".join(reachable) if reachable else "nothing reachable"

    def render(self) -> dict[str, Any]:
        return {
            "note": (
                "your own running estimates, updated each turn: how reachable each "
                "kind of thing is, and whether each action would do anything now"
            ),
            "objects": {k: round(v, 2) for k, v in sorted(self.objects.items())},
            "actions": {
                k: round(v, 2)
                for k, v in sorted(self.actions.items(), key=lambda kv: -kv[1])
            },
        }


class PreferenceTable:
    """State-action pairs the model scores, where the state is the signature."""

    def __init__(self, capacity: int = 14) -> None:
        self.capacity = capacity
        self.pairs: dict[tuple[str, str], dict[str, Any]] = {}

    def note(self, signature: str, action: str, step: int) -> bool:
        """Layer one: track a pair the first time it turns up."""
        key = (signature, action)
        if key in self.pairs:
            self.pairs[key]["seen"] += 1
            self.pairs[key]["last_step"] = step
            return False
        if len(self.pairs) >= self.capacity:
            stalest = min(self.pairs.items(), key=lambda kv: kv[1]["last_step"])
            del self.pairs[stalest[0]]
        self.pairs[key] = {"good": None, "seen": 1, "last_step": step}
        return True

    def questions(self) -> dict[str, Noul]:
        """Layer two: re-score every tracked pair, every step."""
        asked: dict[str, Noul] = {}
        for index, ((signature, action), entry) in enumerate(self.pairs.items()):
            previous = entry["good"]
            hint = (
                f" Your last estimate was {previous:.2f}; update it."
                if previous is not None
                else ""
            )
            asked[f"{PREF_PREFIX}{index}"] = Noul(
                instructions=(
                    f"When what you can reach is [{signature}], taking the action "
                    f"{action} turns out well.{hint}"
                )
            )
        return asked

    def update(self, answers: dict[str, Any]) -> None:
        keys = list(self.pairs)
        for name, answer in answers.items():
            if not name.startswith(PREF_PREFIX):
                continue
            index = int(name[len(PREF_PREFIX) :])
            if index < len(keys):
                self.pairs[keys[index]]["good"] = answer["noul"]

    def render(self) -> dict[str, Any]:
        scored = [
            {"when": signature, "do": action, "good": round(entry["good"], 2)}
            for (signature, action), entry in self.pairs.items()
            if entry["good"] is not None
        ]
        scored.sort(key=lambda row: -row["good"])
        return {
            "note": "how well each action has looked in each kind of situation",
            "pairs": scored,
        }

    def save(self, path: pathlib.Path) -> None:
        path.write_text(
            json.dumps(
                [
                    {"signature": s, "action": a, **entry}
                    for (s, a), entry in self.pairs.items()
                ],
                indent=1,
            )
        )

    def load(self, path: pathlib.Path) -> int:
        if not path.exists():
            return 0
        for row in json.loads(path.read_text()):
            self.pairs[(row["signature"], row["action"])] = {
                "good": row["good"],
                "seen": row["seen"],
                "last_step": row["last_step"],
            }
        return len(self.pairs)

    def __len__(self) -> int:
        return len(self.pairs)
