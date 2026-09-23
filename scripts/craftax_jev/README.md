# Craftax × TypeSafe (Jev)

A demo agent that plays [Craftax-Classic](https://github.com/MichaelTMatthews/Craftax)
with one [TypeSafe System One](https://docs.typesafe.ai/introduction) call per
environment step. Jev never generates text or an action string: a `Choice`
question over the 17 Craftax actions *is* the policy, and five speculative
questions ride along in the same request for the replay view.

Written up in [`/research-notes/craftax-jev.html`](../../public/research-notes/craftax-jev.html).

## Layout

| File | Role |
| --- | --- |
| `serialize.py` | `EnvState` → the JSON object sent as `state` (7×9 named-block view, facing tile, vitals, inventory, bearings, achievements) |
| `questions.py` | The six questions: one action `Choice`, one priority `Choice`, a danger `Score`, three `Noul`s |
| `agent.py` | The loop: serialize, ask, press the answer, log JSONL + one PNG per step |
| `feedback.py` | Exact effect of one transition, and the tally of actions that provably did nothing here |
| `make_video.py` | Composites each step into game frame + answer panel and encodes `rollout.mp4` |
| `export_replay.py` | Builds the sprite atlas and the compact per-run JSON the note's replay viewer scrubs through |

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python craftax typesafe-sdk imageio matplotlib
```

`agent.py` reads `JEV_API_KEY` from the gitignored `.env` at the repo root.

## Run

```bash
.venv/bin/python agent.py --steps 600 --seed 0 --out runs/run-01
.venv/bin/python make_video.py runs/run-01 --fps 6
```

Flags:

- `--feedback N` puts the last N transitions in the state as `do -> nothing happened`,
  plus a tally of what has already been wasted from this exact tile and facing.
  The effect is computed from the action alone — hunger decay, nightfall and mob
  movement do not count as the action having worked.
- `--suppress N` drops an action from the `Choice` options after it has done
  nothing N times in the same situation, cleared when the block in front changes.
  Still raw action control; code just stops offering a provably inert option.
- `--state-level raw` drops the engineered context: no bearings, no reach flags,
  no pre-resolved tile in front, no remaining-achievement list.
- `--criteria minimal` describes only what each button does, with no recipes or
  requirements; `--criteria names` gives the option names alone.
- `--memory N` keeps up to N transitions that the model itself marks worth
  remembering, via one extra `Noul` in the same call. Identical entries merge
  into a count; the lowest-scored entry is evicted when full.
- `--full-diff` records every change in a transition rather than a digest: all
  changed cells, vitals, items and achievements. Also fixes the engineered
  path, which reported only the first change it found.
- `--full-history` records every state change since the episode began instead of
  a sliding window, with consecutive repeats collapsed. Measured as a loss.
- `--pure` removes the two game priors from the feedback: the effect becomes a
  diff of the observation (`view r2c4 tree->grass, wood 0->1`) instead of a
  curated phrase, and situations are keyed on the visible surroundings rather
  than the tile you face. Costs more than half the score; see the note.
- `--position` gives the raw state absolute x, y. Measured as a loss: it makes
  every memory entry unique, so nothing generalises.
- `--history N` is the earlier, broken version of `--feedback` kept for the record:
  it compares whole states, so it calls an action successful when a zombie moves.

The default of 0 for all three is the naive setting.

Measured over seeds 0, 1 and 2 with a 400-step cap (mean achievements):

| Setting | Mean |
| --- | --- |
| engineered, nothing else | 3.0 |
| engineered `--feedback 8` | 2.7 |
| engineered `--feedback 8 --suppress 3` | 7.7 |
| `--state-level raw --criteria minimal` | 1.0 |
| raw + `--feedback 16` | 4.0 |
| raw + `--feedback 16 --suppress 3` | 4.7 |
| raw + `--feedback 16 --memory 24` | 2.7 |
| raw + `--feedback 16 --memory 24 --position` | 3.0 (5 seeds) |
| raw + `--feedback 16 --pure` | 1.8 (5 seeds) |
| raw + `--feedback 16 --memory 24 --pure` | 2.6 (5 seeds) |
| raw + `--full-history --pure` | 1.2 (5 seeds) |
| raw + `--feedback 16 --memory 24 --pure --full-diff` | 2.2 (5 seeds) |

The last row is the prior-free number: no Craftax rule reaches the model in the
state, the options, or the feedback vocabulary.

Export replays for the note with:

```bash
.venv/bin/python export_replay.py ../../public/data/demos/craftax-jev \
  naive=runs/run-01 suppress=runs/run-04-suppress pure=runs/pure-hist-seed0
```

Runs are gitignored; each one writes `decisions.jsonl`, `summary.json`,
`frames/`, `composites/` and `rollout.mp4`.
