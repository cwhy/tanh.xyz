#!/usr/bin/env python3
"""Play Craftax-Classic with one TypeSafe System One call per environment step.

Usage:
    python scripts/craftax_jev/agent.py --steps 400 --out runs/run-01

Naive on purpose: no macro-actions, no planner, no action masking. Every tick
serialises the environment state, asks Jev one Choice over the 17 actions plus
five speculative questions, and presses whatever comes back. Writes one PNG per
step and a JSONL decision log for the replay video.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import jax
import jax.numpy as jnp
import numpy as np
from craftax.craftax_classic.constants import (
    BLOCK_PIXEL_SIZE_HUMAN,
    Achievement,
    Action,
)
from craftax.craftax_classic.renderer import make_craftax_pixel_renderer
from craftax.craftax_env import make_craftax_env_from_name
from typesafe_sdk import RetryPolicy, TypeSafeClient

from feedback import (
    FailureTally,
    Lessons,
    Memory,
    describe_effect,
    MAX_LISTED_CELLS,
    NO_EFFECT,
    collapse_repeats,
    describe_effect_full,
    world_diff,
    world_diff_compact,
    world_diff_general,
    observation_diff,
    situation_key,
    view_key,
)
from recall import Recall
from representation import Accessibility, PreferenceTable
from questions import (
    ACTION_ORDER,
    QUESTIONS,
    action_question,
    focus_cell_question,
    focus_question,
    own_exploration_question,
    avoid_questions,
)
from serialize import facing_block, serialize

REPO_ROOT = Path(__file__).resolve().parents[2]



def read_api_key() -> str:
    """The key lives in the gitignored .env at the repo root as JEV_API_KEY."""
    env_path = REPO_ROOT / ".env"
    for line in env_path.read_text().splitlines():
        name, _, value = line.partition("=")
        if name.strip() == "JEV_API_KEY":
            return value.strip().strip("'\"")
    raise SystemExit(f"JEV_API_KEY not found in {env_path}")


def effect_signature(state_json: dict[str, Any]) -> tuple:
    """The parts of the state an action is supposed to change."""
    return (
        tuple(sorted(state_json["inventory"].items())),
        tuple(state_json["vitals"].values()),
        state_json["facing"]["direction"],
        state_json["facing"]["tile_in_front"],
        tuple(state_json["nearest"].items()),
        len(state_json["achievements_unlocked"]),
    )


def answers_to_json(response: Any) -> dict[str, Any]:
    record: dict[str, Any] = {}
    for name, answer in response.answers.items():
        if hasattr(answer, "choice"):
            record[name] = {
                "type": "choice",
                "choice": answer.choice,
                "confidence": answer.confidence,
                "probabilities": dict(answer.probabilities),
            }
        elif hasattr(answer, "score"):
            record[name] = {
                "type": "score",
                "score": answer.score,
                "confidence": answer.confidence,
                "legend": dict(answer.legend) if answer.legend else None,
                "probabilities": dict(answer.probabilities),
            }
        else:
            record[name] = {"type": "noul", "noul": answer.noul}
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0, help="the world: fixed across episodes")
    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="repeat index on the same world; varies dynamics and action sampling",
    )
    parser.add_argument("--model", default="jev-latest")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "runs/latest")
    parser.add_argument(
        "--state-level",
        choices=("engineered", "raw"),
        default="engineered",
        help="raw drops the bearings, reach flags and remaining-achievement list",
    )
    parser.add_argument(
        "--criteria",
        choices=("full", "minimal", "names"),
        default="full",
        help="how much the action options explain: full tech tree, mechanics only, or nothing",
    )
    parser.add_argument(
        "--feedback",
        type=int,
        default=0,
        help="show the last N transitions and what each one actually did",
    )
    parser.add_argument(
        "--veto",
        action="store_true",
        help="ask an avoid probability per action and reject sampled actions above the bar",
    )
    parser.add_argument("--veto-threshold", type=float, default=0.7)
    parser.add_argument(
        "--veto-novelty",
        type=float,
        default=0.25,
        help="how much the bar is relaxed for an action never tried this episode",
    )
    parser.add_argument("--veto-tries", type=int, default=5)
    parser.add_argument(
        "--own-exploration",
        action="store_true",
        help="tell the model its top choice is executed, so exploring is its job, and use argmax",
    )
    parser.add_argument(
        "--recall",
        action="store_true",
        help="keep whole states in memory and let the model pick which one applies",
    )
    parser.add_argument(
        "--recall-capacity",
        type=int,
        default=20,
        help="how many full-state memories to hold",
    )
    parser.add_argument(
        "--represent",
        action="store_true",
        help="build accessibility estimates and a state-action preference table, instead of lessons",
    )
    parser.add_argument(
        "--pref-capacity",
        type=int,
        default=14,
        help="how many state-action pairs the preference table tracks at once",
    )
    parser.add_argument(
        "--lesson-coords",
        action="store_true",
        help="keep coordinates in stored lessons (the old behaviour; they then never merge)",
    )
    parser.add_argument(
        "--carry",
        type=Path,
        default=None,
        help="file holding lessons from earlier lives: loaded at start, written at the end",
    )
    parser.add_argument(
        "--untried",
        action="store_true",
        help="tell the agent which actions it has never tried this episode",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="sample the action from the returned distribution instead of taking the top one",
    )
    parser.add_argument(
        "--sample-below",
        type=float,
        default=None,
        help="with --sample, only sample when action confidence is under this; else argmax",
    )
    parser.add_argument(
        "--focus",
        action="store_true",
        help="ask Jev what to attend to in its own call, before the decision call",
    )
    parser.add_argument(
        "--focus-mode",
        choices=("type", "cell"),
        default="type",
        help="focus on a kind of thing, or on one specific tile by coordinate",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="index tile types once in the state, and use the indices in transitions",
    )
    parser.add_argument(
        "--lessons",
        action="store_true",
        help=(
            "memory as model-admitted lessons: a Choice labels each transition "
            "reinforce/avoid/neither, a Noul rejects ones already covered"
        ),
    )
    parser.add_argument(
        "--world-frame",
        action="store_true",
        help="label tiles by world coordinate, so changes are what appeared, changed or left",
    )
    parser.add_argument(
        "--full-diff",
        action="store_true",
        help="report every change in a transition, not a digest of it",
    )
    parser.add_argument(
        "--full-history",
        action="store_true",
        help="record every state change since the episode began, not a sliding window",
    )
    parser.add_argument(
        "--pure",
        action="store_true",
        help=(
            "no game priors in the feedback: effects are a diff of the observation "
            "and situations are keyed on the visible surroundings, not the facing tile"
        ),
    )
    parser.add_argument(
        "--position",
        action="store_true",
        help="give the raw state a sense of place: absolute x, y on the world grid",
    )
    parser.add_argument(
        "--memory",
        type=int,
        default=0,
        help="keep up to N transitions that the model itself marks worth remembering",
    )
    parser.add_argument(
        "--memory-threshold",
        type=float,
        default=0.5,
        help="noul score above which a transition enters memory",
    )
    parser.add_argument(
        "--suppress",
        type=int,
        default=0,
        help="drop an action from the options after it does nothing N times here",
    )
    parser.add_argument(
        "--history",
        type=int,
        default=0,
        help="include the last N actions and whether they changed anything (0 = naive)",
    )
    args = parser.parse_args()

    frames_dir = args.out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "decisions.jsonl"

    env = make_craftax_env_from_name("Craftax-Classic-Symbolic-v1", auto_reset=False)
    env_params = env.default_params
    render = jax.jit(make_craftax_pixel_renderer(BLOCK_PIXEL_SIZE_HUMAN))

    # The world comes from --seed alone, so repeats start from the same map; the
    # rolling stream carries --episode, so mob behaviour and sampling differ.
    _, reset_rng = jax.random.split(jax.random.PRNGKey(args.seed))
    rng = jax.random.PRNGKey(args.seed * 1000 + args.episode)
    _, state = env.reset(reset_rng, env_params)

    client = TypeSafeClient(
        api_key=read_api_key(),
        model=args.model,
        retry=RetryPolicy(max_retries=4),
        timeout=30.0,
    )

    base_questions = {
        **QUESTIONS,
        "action": (
            own_exploration_question()
            if args.own_exploration
            else action_question(style=args.criteria)
        ),
    }
    if args.veto:
        base_questions = {**base_questions, **avoid_questions()}
    if not args.memory:
        base_questions.pop("worth_remembering", None)
    if not args.lessons:
        base_questions.pop("lesson", None)
        base_questions.pop("already_known", None)
    recall = Recall(capacity=args.recall_capacity)
    access = Accessibility()
    prefs = PreferenceTable(capacity=args.pref_capacity)
    lessons = Lessons()
    carried = 0
    if args.carry:
        if args.recall:
            carried = recall.load(args.carry)
        elif args.represent:
            carried = prefs.load(args.carry)
        else:
            carried = lessons.load(args.carry)
    if carried:
        print(f"carried {carried} lessons from earlier lives", flush=True)
    sampler = np.random.default_rng(args.seed * 1000 + args.episode)
    ever_tried: set[str] = set()
    focus_now: str | None = None
    pending_state = None
    pending_action = None
    pending_effect = ""
    memory = Memory(capacity=args.memory, threshold=args.memory_threshold)
    pending_memory: str | None = None
    history: deque[str] = deque(maxlen=max(args.history, args.feedback, 1))
    full_history: list[str] = []
    tally = FailureTally()
    unlocked: set[str] = set()
    total_reward = 0.0
    input_tokens = 0
    started = time.monotonic()

    with log_path.open("w") as log:
        for step in range(args.steps):
            state_json = serialize(state, level=args.state_level, position=args.position, world_frame=args.world_frame)
            if args.history:
                state_json["recent_actions"] = {
                    "note": "most recent last; use this to notice when you are stuck",
                    "actions": list(history),
                }
            facing_now = facing_block(state)
            questions = base_questions
            blocked: list[str] = []

            key = view_key(state_json) if args.pure else situation_key(state, facing_now)
            failures = tally.failures(key)

            if args.full_history:
                state_json["history"] = {
                    "note": (
                        "every change since you woke up, oldest first. a line that "
                        "changed nothing was a wasted action"
                    ),
                    "transitions": collapse_repeats(full_history),
                }

            if args.feedback:
                state_json["feedback"] = {
                    "note": (
                        "what your last actions actually did. an action reported as "
                        "having no effect was refused, not ignored: it needs something "
                        "you do not have, or a different position"
                    ),
                    "recent": list(history),
                    "wasted_from_this_exact_spot": failures,
                }

            if args.recall:
                state_json["last_transition"] = pending_memory or "nothing yet"

            if args.lessons:
                state_json["last_transition"] = pending_memory or "nothing yet"
                state_json["lessons"] = {
                    "note": (
                        "what you have learnt about acting in this world, including "
                        "in earlier lives here. avoid means it wasted a turn or hurt "
                        "you in that situation"
                    ),
                    "reinforce": lessons.by_label("reinforce"),
                    "avoid": lessons.by_label("avoid"),
                    "novel": lessons.by_label("novel"),
                }

            if args.memory:
                state_json["last_transition"] = pending_memory or "nothing yet"
                state_json["memory"] = {
                    "note": "what you decided earlier was worth remembering in this world",
                    "kept": memory.entries(),
                }

            if args.represent:
                state_json["accessibility"] = access.render()
                state_json["preferences"] = prefs.render()

            if args.untried:
                state_json["actions_never_tried"] = {
                    "note": (
                        "you have not once tried these this episode, so you do not "
                        "know what they do"
                    ),
                    "actions": [name for name in ACTION_ORDER if name not in ever_tried],
                }

            if args.suppress:
                blocked = sorted(tally.blocked(key, args.suppress))
                allowed = [name for name in ACTION_ORDER if name not in blocked]
                if len(allowed) >= 2:
                    questions = {
                        **base_questions,
                        "action": action_question(allowed, style=args.criteria),
                    }
                else:
                    blocked = []

            recalled = None
            if args.recall:
                ask = recall.question()
                if ask is not None:
                    picked = client.system_one(state_json, {"recall": ask})
                    choice = picked.answers["recall"].choice
                    input_tokens += picked.usage.input_tokens
                    entry = recall.find(choice)
                    if entry is not None:
                        recalled = choice
                        state_json["recalled"] = recall.expand(entry)

            focus_answer = None
            focus_latency = 0.0
            if args.focus:
                if args.focus_mode == "cell":
                    origin = state_json["view"].get("origin", {"x": 0, "y": 0})
                    cells = []
                    for row, line in enumerate(state_json["view"]["grid"]):
                        for col, token in enumerate(line.split()):
                            if token in ("you", "out_of_bounds", "invalid"):
                                continue
                            cells.append(
                                f"({origin['x'] + col},{origin['y'] + row}) {token}"
                            )
                    focus_ask = focus_cell_question(cells)
                else:
                    visible = sorted(
                        {
                            token
                            for line in state_json["view"]["grid"]
                            for token in line.split()
                            if token not in ("you", "out_of_bounds", "invalid")
                        }
                    )
                    focus_ask = focus_question(visible)
                focus_started = time.monotonic()
                focus_response = client.system_one(state_json, {"focus": focus_ask})
                focus_latency = time.monotonic() - focus_started
                focus_answer = focus_response.answers["focus"]
                focus_now = focus_answer.choice
                input_tokens += focus_response.usage.input_tokens
                state_json["focus"] = {
                    "note": "the one thing you chose to attend to this tick",
                    "on": focus_now,
                    "confidence": round(focus_answer.confidence, 3),
                }

            if args.represent:
                visible = sorted(
                    {
                        token
                        for line in state_json["view"]["grid"]
                        for token in line.split()
                        if token not in ("you", "out_of_bounds", "invalid")
                    }
                )
                questions = {
                    **questions,
                    **access.questions(visible, ACTION_ORDER),
                    **prefs.questions(),
                }

            call_started = time.monotonic()
            response = client.system_one(state_json, questions)
            latency = time.monotonic() - call_started

            answers = answers_to_json(response)
            if args.represent:
                access.update(answers)
                prefs.update(answers)
            action_name = answers["action"]["choice"]
            avoid = {
                key[len("avoid__") :]: value["noul"]
                for key, value in answers.items()
                if key.startswith("avoid__")
            }
            rejected: list[str] = []

            def vetoed(name: str) -> bool:
                """Known-bad blocks; never-tried gets a looser bar."""
                bar = args.veto_threshold + (
                    args.veto_novelty if name not in ever_tried else 0.0
                )
                return avoid.get(name, 0.0) >= bar

            sampled = False
            if args.sample:
                confidence = answers["action"]["confidence"]
                if args.sample_below is None or confidence < args.sample_below:
                    names = list(answers["action"]["probabilities"])
                    weights = np.array(
                        [answers["action"]["probabilities"][n] for n in names], dtype=float
                    )
                    if weights.sum() > 0:
                        weights = weights / weights.sum()
                        drawn = names[int(sampler.choice(len(names), p=weights))]
                        if args.veto:
                            for _ in range(args.veto_tries):
                                if not vetoed(drawn):
                                    break
                                rejected.append(drawn)
                                drawn = names[int(sampler.choice(len(names), p=weights))]
                            else:
                                drawn = min(
                                    names,
                                    key=lambda n: avoid.get(n, 0.0)
                                    - (args.veto_novelty if n not in ever_tried else 0.0),
                                )
                        sampled = drawn != action_name
                        action_name = drawn
            ever_tried.add(action_name)
            if args.represent:
                prefs.note(access.signature(), action_name, step)

            if args.recall and pending_state is not None:
                label = answers["lesson"]["choice"]
                if label != "neither":
                    recall.add(pending_state, pending_action, pending_effect, label, step)

            lesson_new = None
            lesson_label = None
            coverage = None
            if args.lessons and pending_memory is not None:
                lesson_label = answers["lesson"]["choice"]
                coverage = answers["already_known"]["noul"]
                if lessons.consider(
                    pending_memory, lesson_label, coverage >= 0.5, step
                ):
                    lesson_new = pending_memory

            memory_new = None
            if args.memory and pending_memory is not None:
                score = answers["worth_remembering"]["noul"]
                if memory.consider(pending_memory, score, step):
                    memory_new = pending_memory
            action = Action[action_name].value
            input_tokens += response.usage.input_tokens

            iio.imwrite(
                frames_dir / f"{step:05d}.png",
                np.asarray(render(state), dtype=np.uint8),
            )

            rng, step_rng = jax.random.split(rng)
            _, next_state, reward, done, _ = env.step(
                step_rng, state, jnp.int32(action), env_params
            )
            total_reward += float(reward)

            if args.world_frame:
                differ = world_diff_compact if args.compact else world_diff
                effect = differ(
                    state_json,
                    serialize(
                        next_state,
                        level=args.state_level,
                        position=args.position,
                        world_frame=True,
                    ),
                )
            elif args.pure:
                effect = observation_diff(
                    state_json,
                    serialize(next_state, level=args.state_level, position=args.position, world_frame=args.world_frame),
                    max_cells=0 if args.full_diff else MAX_LISTED_CELLS,
                )
            elif args.full_diff:
                effect = describe_effect_full(state, next_state)
            else:
                effect = describe_effect(state, next_state)
            tally.record(key, action_name, effect, null_effect=NO_EFFECT if (args.pure or args.world_frame) else "nothing happened")
            if args.world_frame and not args.lesson_coords:
                next_obs = serialize(
                    next_state,
                    level=args.state_level,
                    position=args.position,
                    world_frame=True,
                )
                pending_memory = (
                    f"{action_name.lower()} -> {world_diff_general(state_json, next_obs)}"
                )
            elif args.pure or args.world_frame:
                pending_memory = f"{action_name.lower()} -> {effect}"
            else:
                where = ""
                if args.position:
                    spot = np.asarray(state.player_position)
                    where = f"at ({int(spot[1])},{int(spot[0])}) "
                pending_memory = f"{where}facing {facing_now}: {action_name.lower()} -> {effect}"

            line = f"{action_name.lower()} -> {effect}"
            if args.focus:
                line = f"[focus {focus_now}] {line}"
            pending_state, pending_action, pending_effect = state_json, action_name, effect
            full_history.append(line)
            if args.feedback:
                history.append(line)
            elif args.history:
                changed = effect_signature(state_json) != effect_signature(
                    serialize(next_state, level=args.state_level, position=args.position, world_frame=args.world_frame)
                )
                history.append(
                    f"{action_name.lower()}" + ("" if changed else " (nothing changed)")
                )


            now_unlocked = {
                achievement.name.lower()
                for achievement in Achievement
                if bool(np.asarray(next_state.achievements)[achievement.value])
            }
            new_unlocks = sorted(now_unlocked - unlocked)
            unlocked = now_unlocked

            log.write(
                json.dumps(
                    {
                        "step": step,
                        "state": state_json,
                        "answers": answers,
                        "action": action_name,
                        "sampled_away_from_top": sampled,
                        "vetoed": rejected,
                        "avoid_of_taken": round(avoid.get(action_name, 0.0), 3) if avoid else None,
                        "reward": float(reward),
                        "done": bool(done),
                        "new_achievements": new_unlocks,
                        "effect": effect,
                        "blocked_options": blocked,
                        "memory_new": memory_new,
                        "memory_size": len(memory),
                        "lesson_new": lesson_new,
                        "lesson_label": lesson_label,
                        "lesson_coverage": coverage,
                        "lessons_size": len(lessons),
                        "signature": access.signature() if args.represent else None,
                        "pref_pairs": len(prefs) if args.represent else 0,
                        "recalled": recalled,
                        "recall_size": len(recall),
                        "questions_asked": len(questions),
                        "focus": focus_now if args.focus else None,
                        "focus_confidence": (
                            round(focus_answer.confidence, 3) if focus_answer else None
                        ),
                        "focus_probabilities": (
                            dict(focus_answer.probabilities) if focus_answer else None
                        ),
                        "focus_latency_s": round(focus_latency, 3),
                        "latency_s": round(latency, 3),
                        "input_tokens": response.usage.input_tokens,
                        "model": response.model,
                    }
                )
                + "\n"
            )
            log.flush()

            marker = f"  +{', '.join(new_unlocks)}" if new_unlocks else ""
            print(
                f"{step:4d}  {action_name:<19} "
                f"conf {answers['action']['confidence']:.2f}  "
                f"hp {state_json['vitals']['health']}  "
                f"-> {effect:<24}{marker}",
                flush=True,
            )

            state = next_state
            if bool(done):
                print(f"episode ended at step {step}")
                break

    elapsed = time.monotonic() - started
    summary = {
        "steps": step + 1,
        "achievements": sorted(unlocked),
        "total_reward": round(total_reward, 3),
        "wall_clock_s": round(elapsed, 1),
        "input_tokens": input_tokens,
        "cost_usd": round(input_tokens * 42e-9, 4),
        "feedback": args.feedback,
        "suppress": args.suppress,
        "memory": args.memory,
        "position": args.position,
        "pure": args.pure,
        "full_history": args.full_history,
        "full_diff": args.full_diff,
        "world_frame": args.world_frame,
        "compact": args.compact,
        "focus": args.focus,
        "focus_mode": args.focus_mode,
        "untried": args.untried,
        "sample": args.sample,
        "sample_below": args.sample_below,
        "own_exploration": args.own_exploration,
        "veto": args.veto,
        "veto_threshold": args.veto_threshold,
        "veto_novelty": args.veto_novelty,
        "memory_kept": len(memory),
        "lessons": args.lessons,
        "lessons_kept": len(lessons),
        "lessons_carried_in": carried,
        "lesson_coords": args.lesson_coords,
        "represent": args.represent,
        "pref_pairs": len(prefs),
        "state_level": args.state_level,
        "criteria": args.criteria,
        "model": args.model,
        "seed": args.seed,
        "episode": args.episode,
    }
    if args.carry:
        (recall if args.recall else prefs if args.represent else lessons).save(args.carry)
        print(f"saved {len(lessons)} lessons to {args.carry}", flush=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
