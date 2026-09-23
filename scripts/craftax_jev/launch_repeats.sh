#!/bin/bash
cd "$(dirname "$0")"
COMMON="--steps 300 --state-level raw --criteria minimal --feedback 16 --pure --world-frame --compact --lessons"
for s in 0 1 2 3 4; do
  for e in 0 1 2 3 4; do
    while [ "$(pgrep -fc 'agent.py')" -ge 6 ]; do sleep 5; done
    .venv/bin/python agent.py $COMMON --seed $s --episode $e --out runs/rep-base-s${s}e${e} > /dev/null 2>&1 &
    sleep 1
    while [ "$(pgrep -fc 'agent.py')" -ge 6 ]; do sleep 5; done
    .venv/bin/python agent.py $COMMON --untried --sample --seed $s --episode $e --out runs/rep-try-s${s}e${e} > /dev/null 2>&1 &
    sleep 1
  done
done
wait
echo ALL_RUNS_DONE
