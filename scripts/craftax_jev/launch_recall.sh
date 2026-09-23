#!/bin/bash
cd "$(dirname "$0")"
COMMON="--steps 300 --state-level raw --criteria minimal --feedback 16 --pure --world-frame --compact --lessons --recall --untried --sample"
run_world () {
  local s=$1
  rm -f runs/carryr-s${s}.json
  for e in 0 1 2 3 4; do
    .venv/bin/python agent.py $COMMON --seed $s --episode $e \
      --carry runs/carryr-s${s}.json --out runs/rec-s${s}e${e} > runs/rec-s${s}e${e}.log 2>&1
  done
}
for s in 0 1 2 3 4; do run_world $s & done
wait
echo ALL_RECALL_DONE
