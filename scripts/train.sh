#!/usr/bin/env bash
# Train and evaluate the final system, 3 seeds (about 15 min on one GB10; the head takes seconds on the GPU, the policy
# about 3 min on 8 CPU cores): teacher plans -> intent head (sim only) -> z of every clip -> policy -> physics tests on
# held-out sim starts, the composition queries and the real development takes (skipped if REAL is absent; the head and
# policy never train on real clips). Also writes results/final/scene.json and durations.npz (for scripts/run_test.sh).
# Usage: bash scripts/train.sh [SIM] [REAL]   (defaults data/sim/intent_tasks_train_v3, data/real/demos3_approved_rgb_v3)
source "$(dirname -- "$0")/common.sh"
export SIM=${1:-data/sim/intent_tasks_train_v3} REAL=${2:-data/real/demos3_approved_rgb_v3}
chain() {
  R=(); if [[ -d $REAL ]]; then R=(--real "$REAL"); else echo "no real development set at $REAL: sim tests only"; fi
  echo "START $(date -Is) rules"; spider -m handsoff.rules --sim "$SIM" "${R[@]}"
  echo "START $(date -Is) plans"; spider -m handsoff.policy.policy plans --sim "$SIM" --plans results/final/plans.npz
  for seed in 0 1 2; do
    O=results/final/seed$seed
    echo "START $(date -Is) head seed $seed"; spider -m handsoff.model.train "$SIM" --out $O/head --seed $seed > /dev/null
    echo "START $(date -Is) embed seed $seed"; spider -m handsoff.policy.policy embed --sim "$SIM" --head $O/head "${R[@]}" --z $O/z.npz
    echo "START $(date -Is) policy seed $seed"; spider -m handsoff.policy.policy train --z $O/z.npz --plans results/final/plans.npz --out $O/policy
    echo "START $(date -Is) eval seed $seed"; spider -m handsoff.policy.policy eval --sim "$SIM" --z $O/z.npz --plans results/final/plans.npz --out $O/policy
  done
  echo "START $(date -Is) arithmetic examples"; spider -m handsoff.model.arithmetic results/final/seed0/head --sim "$SIM"
  echo "DONE $(date -Is) results/final"
}
detach results/logs/train.log
