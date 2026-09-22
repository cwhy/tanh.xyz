# Jev prompt manipulation investigation

The expanded report is `public/research-notes/jev-prompt-manipulation.html`.
Canonical evidence is under `public/data/demos/jev-prompt-manipulation/`.
See `reproduce.txt` for pinned runtimes and commands.

The comparison covers Jev 1.13.0, SemIf (Qwen3.5-4B BF16), and Winnow-12B Q8.
All remaining comparator inference runs on remote CUDA GPUs at the user's request.
Do not launch model inference on the laptop. Jev uses its hosted API; its key
stays in the local environment and is never copied to the server or public data.

`advanced.py` runs development and held-out phases. `refine.py` runs the
feedback-informed analyst round. Each backend has 132 initial/adaptive calls
and 48 refinement calls. `select.py` freezes the union of best scope/policy and
direct-label attacks before held-out calls. Output files cannot be overwritten.
`render_advanced.py` validates complete evidence and renders the expanded note.

The old report and 160-call pilot page are archived locally under
`private/research-notes/jev-2026-09-22-before-story/` (Git-ignored and outside
the published site). The pilot data remains available as evidence.
Its runner is `run.py`; its historical renderer is `render.py` and must not be
used to replace the expanded report. The pilot found no flips in its narrow
suite, which did not establish general robustness.

`local-exploration/` preserves preliminary SemIf MLX results and the interrupted
Winnow Metal run. They are excluded from canonical comparison counts. Connection
and startup failures are also kept separately; failed calls are not successes.

These are synthetic, non-graphic classification tests with fake `.invalid`
addresses. The adaptations are research-informed, not full reproductions of
PAIR, GCG, or the original large-budget Best-of-N experiments. Eight held-out
items and repeated calls cannot establish population-level safety rates.

## Follow-up chapters

`render_followups.py` renders the larger campaign and Laya/typed extension from
archived scores in the `comprehensive/` and `laya-typed/` data subdirectories.
The original 1,332-call comparison remains separate. Together the three campaigns
contain 14,495 defender calls, excluding the 160-call pilot and attacker calls.
The follow-up source audits reproduced 8,063 and 5,100 calls respectively.
