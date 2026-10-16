#!/usr/bin/env bash
# Robustness of a trained system to the block's physics: the same z -> policy -> waypoints, executed open loop with a
# changed block mass and friction (none is a level of the grasp search in handsoff/sim/grip_robust.py), on the held-out
# sim clips and, if scripts/run_test.sh has been run for these weights, the real test set. CPU, about 1 min per
# condition per seed. Usage: bash scripts/robustness.sh [WEIGHTS] [ROBOT]
#   defaults results/final (after scripts/train.sh), hand; ROBOT=arm: the hand on the UR5e. TEST: run_test.sh output.
source "$(dirname -- "$0")/common.sh"
export WEIGHTS=${1:-results/final} ROBOT=${2:-hand} TEST=${TEST:-results/test}
chain() {
  for C in "0.12 1.0" "0.06 1.0" "0.04 0.8" "0.12 0.6" "0.09 0.5" "0.30 0.7"; do      # kg, friction (nominal first)
    read -r M F <<< "$C"
    for D in "$WEIGHTS"/seed*; do
      S=$(basename "$D"); O=results/robustness/$(basename "$WEIGHTS")_$ROBOT/m${M}_f${F}/$S
      A=(--sim "$WEIGHTS/scene.json" --plans "$WEIGHTS/durations.npz" --policy "$D/policy/policy.pt" --robot "$ROBOT"
         --block-mass "$M" --friction "$F")
      echo "START $(date -Is) $S mass $M friction $F"
      spider -m handsoff.policy.policy eval "${A[@]}" --z "$D/z.npz" --out "$O/sim" --sets sim
      if [[ -f $TEST/$S/z.npz ]]; then
        spider -m handsoff.policy.policy eval "${A[@]}" --z "$TEST/$S/z.npz" --out "$O/test" --sets real
      fi
    done
  done
  echo "DONE $(date -Is) results/robustness"
}
detach results/logs/robustness.log
