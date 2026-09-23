#!/usr/bin/env python3
"""The question set sent with every Craftax state.

One Choice picks the action: it is the policy. The Score and the Nouls are
speculative, fanned out in the same request because extra questions cost no
extra latency, and they are what the replay video visualises alongside the
chosen action.
"""

from __future__ import annotations

from craftax.craftax_classic.constants import Action
from typesafe_sdk import Choice, Noul, Score

# Keyed by the exact Action name so the answer maps back without a lookup table.
ACTION_CRITERIA: dict[str, str] = {
    "NOOP": "Do nothing this tick.",
    "LEFT": "Step one tile left. If that tile is blocked you only turn to face left.",
    "RIGHT": "Step one tile right. If that tile is blocked you only turn to face right.",
    "UP": "Step one tile up. If that tile is blocked you only turn to face up.",
    "DOWN": "Step one tile down. If that tile is blocked you only turn to face down.",
    "DO": (
        "Use the tile you are facing: chop a tree for wood, mine stone or coal "
        "(needs a wood pickaxe), mine iron (needs a stone pickaxe), mine diamond "
        "(needs an iron pickaxe), drink from water, eat a cow or a ripe plant, "
        "collect a sapling from grass, or hit an adjacent zombie or skeleton."
    ),
    "SLEEP": (
        "Sleep until morning. Restores energy, but you cannot act and mobs can "
        "hit you while you sleep."
    ),
    "PLACE_STONE": "Place a stone block in front of you. Needs 1 stone.",
    "PLACE_TABLE": "Place a crafting table in front of you. Needs 2 wood.",
    "PLACE_FURNACE": "Place a furnace in front of you. Needs 1 stone.",
    "PLACE_PLANT": "Plant a sapling on grass in front of you. Needs 1 sapling.",
    "MAKE_WOOD_PICKAXE": "Craft a wood pickaxe. Needs 1 wood and a crafting table in reach.",
    "MAKE_STONE_PICKAXE": "Craft a stone pickaxe. Needs 1 wood, 1 stone and a crafting table in reach.",
    "MAKE_IRON_PICKAXE": (
        "Craft an iron pickaxe. Needs 1 wood, 1 stone, 1 coal, 1 iron, and both a "
        "crafting table and a furnace in reach."
    ),
    "MAKE_WOOD_SWORD": "Craft a wood sword. Needs 1 wood and a crafting table in reach.",
    "MAKE_STONE_SWORD": "Craft a stone sword. Needs 1 wood, 1 stone and a crafting table in reach.",
    "MAKE_IRON_SWORD": (
        "Craft an iron sword. Needs 1 wood, 1 coal, 1 iron, and both a crafting "
        "table and a furnace in reach."
    ),
}

ACTION_ORDER = [action.name for action in Action]
assert set(ACTION_ORDER) == set(ACTION_CRITERIA), "action list drifted from Craftax"

ACTION_INSTRUCTIONS = (
    "You are the player in a survival crafting game. Pick the single action "
    "to take this tick. Stay alive, keep food, drink and energy up, and work "
    "towards the achievements still listed as remaining. An action whose "
    "requirements are not met does nothing, so only pick one you can do now."
)


# What each action does mechanically, with no recipe, requirement or advice in it.
MINIMAL_ACTION_CRITERIA: dict[str, str] = {
    "NOOP": "Do nothing.",
    "LEFT": "Move or face left.",
    "RIGHT": "Move or face right.",
    "UP": "Move or face up.",
    "DOWN": "Move or face down.",
    "DO": "Interact with the tile you are facing.",
    "SLEEP": "Sleep.",
    "PLACE_STONE": "Place stone in front of you.",
    "PLACE_TABLE": "Place a crafting table in front of you.",
    "PLACE_FURNACE": "Place a furnace in front of you.",
    "PLACE_PLANT": "Plant a sapling in front of you.",
    "MAKE_WOOD_PICKAXE": "Craft a wood pickaxe.",
    "MAKE_STONE_PICKAXE": "Craft a stone pickaxe.",
    "MAKE_IRON_PICKAXE": "Craft an iron pickaxe.",
    "MAKE_WOOD_SWORD": "Craft a wood sword.",
    "MAKE_STONE_SWORD": "Craft a stone sword.",
    "MAKE_IRON_SWORD": "Craft an iron sword.",
}

MINIMAL_INSTRUCTIONS = (
    "You are the player in this game. Pick the single action to take this tick."
)


def action_question(
    allowed: list[str] | None = None, style: str = "full"
) -> Choice:
    """The policy question.

    `full` hands over the tech tree and the survival advice. `minimal` describes
    only what each button does. `names` gives the option names and nothing else.
    """
    names = allowed or ACTION_ORDER
    if style == "names":
        return Choice(
            instructions=MINIMAL_INSTRUCTIONS,
            criteria={name: None for name in names},
        )
    if style == "minimal":
        return Choice(
            instructions=MINIMAL_INSTRUCTIONS,
            criteria={name: MINIMAL_ACTION_CRITERIA[name] for name in names},
        )
    return Choice(
        instructions=ACTION_INSTRUCTIONS,
        criteria={name: ACTION_CRITERIA[name] for name in names},
    )


QUESTIONS = {
    "action": action_question(),
    "priority": Choice(
        instructions="What does the player most need to deal with right now?",
        criteria={
            "danger": "A zombie or skeleton is close enough to do damage.",
            "drink": "Drink is low and water should be found or used.",
            "food": "Food is low and a cow or ripe plant should be eaten.",
            "energy": "Energy is low and the player should sleep.",
            "gather": "Nothing is urgent, so collect wood, stone or ore.",
            "craft": "Materials are in hand and something can be crafted or placed.",
        },
    ),
    "danger": Score(
        instructions="How dangerous is the player's immediate situation?",
        criteria=[
            "Nothing hostile in view",
            "A hostile mob is in view but not adjacent",
            "A hostile mob is adjacent and can hit the player now",
        ],
    ),
    "facing_useful": Noul(
        instructions=(
            "The tile directly in front of the player can be usefully harvested, "
            "used or attacked right now with the DO action."
        ),
    ),
    "can_craft_now": Noul(
        instructions=(
            "The player is in reach of a crafting table and holds the materials for "
            "at least one item they have not made yet."
        ),
    ),
    "worth_remembering": Noul(
        instructions=(
            "Judge the single transition in the state field `last_transition`. It is "
            "worth writing into long-term memory: it teaches something about how this "
            "world works that will still be useful hundreds of steps from now. "
            "Ordinary movement that taught nothing is not worth remembering."
        ),
    ),
    "lesson": Choice(
        instructions=(
            "Look at the single transition in the state field `last_transition`. "
            "Is it a lesson about how to act in this world?"
        ),
        criteria={
            "reinforce": "It worked. In a situation like this, do that again.",
            "avoid": "It was wasted or harmful. In a situation like this, do not do that.",
            "novel": (
                "Neither good nor bad, but rare or surprising: something you had not "
                "seen this world do before, and worth remembering for that alone."
            ),
            "neither": "It is ordinary and carries no lesson either way.",
        },
    ),
    "already_known": Noul(
        instructions=(
            "The lesson in `last_transition` is already covered by something in the "
            "`lessons` lists: acting on what is already written there would lead to "
            "the same behaviour. Judge the lesson, not the exact wording — the same "
            "lesson in a different place or with different coordinates is still covered."
        ),
    ),
    "should_explore": Noul(
        instructions=(
            "Nothing useful is in the current view, so the player should travel to "
            "new ground rather than act here."
        ),
    ),
}


FOCUS_INSTRUCTIONS = (
    "Before you act, pick the one thing in view worth paying attention to this "
    "tick. Exactly one. Everything else is ignored until the next tick."
)


def focus_question(visible: list[str]) -> Choice:
    """A Choice built from whatever is actually in view this tick, plus nothing."""
    criteria = {name: f"Pay attention to the {name.replace('_', ' ')}." for name in visible}
    criteria["nothing"] = "Nothing in view is worth attending to."
    return Choice(instructions=FOCUS_INSTRUCTIONS, criteria=criteria)


def focus_cell_question(cells: list[str]) -> Choice:
    """The same question over specific cells: every visible tile, by coordinate."""
    criteria: dict[str, None] = {label: None for label in cells}
    criteria["nothing"] = None
    return Choice(instructions=FOCUS_INSTRUCTIONS, criteria=criteria)


OWN_EXPLORATION_INSTRUCTIONS = (
    "You are the player in this game. Pick the single action to take this tick. "
    "The option you rank highest is the one that will be taken: nothing else "
    "chooses for you, and nothing will take a random action on your behalf. So "
    "exploring is your own job. If finding out what an action does is worth more "
    "to you right now than the best move you already know, rank that action "
    "highest and it will happen."
)


def own_exploration_question(allowed: list[str] | None = None) -> Choice:
    """The policy question, with the model told that it owns exploration."""
    names = allowed or ACTION_ORDER
    return Choice(
        instructions=OWN_EXPLORATION_INSTRUCTIONS,
        criteria={name: MINIMAL_ACTION_CRITERIA[name] for name in names},
    )


AVOID_PREFIX = "avoid__"


def avoid_questions(actions: list[str] | None = None) -> dict[str, Noul]:
    """One veto question per action, asked alongside the choice of what to do."""
    return {
        AVOID_PREFIX
        + name: Noul(
            instructions=(
                f"Taking the action {name} right now would be a mistake you would "
                "regret."
            )
        )
        for name in (actions or ACTION_ORDER)
    }
