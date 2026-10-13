#!/usr/bin/env bash
# Train the intent heads (about 1.5 h on one GB10): choose the epoch count per stage by leave-one-take-out over the
# training takes (R19), then the final heads for the three stages x seeds 0-2, then the task-arithmetic examples.
# Usage: bash scripts/train.sh [SIM] [REAL]   (defaults data/sim/intent_tasks_train_v3, data/real/demos3_approved_rgb_v3)
source "$(dirname -- "$0")/common.sh"
export SIM=${1:-data/sim/intent_tasks_train_v3} REAL=${2:-data/real/demos3_approved_rgb_v3}
chain() {
  echo "START $(date -Is) rules"; spider -m handsoff.rules --sim "$SIM" --real "$REAL"
  echo "START $(date -Is) epoch choice"; spider -m handsoff.model.choose_epochs --sim "$SIM" --out results/epoch_choice
  for stage in sim sim_real sim_real_cosmos; do
    epochs=$(spider -c "import json; print(json.load(open('results/epoch_choice/summary.json'))['stages']['$stage']['chosen_epochs'])")
    for seed in 0 1 2; do
      echo "START $(date -Is) final $stage seed $seed ($epochs epochs)"
      spider -m handsoff.model.train "$SIM" --out results/final/${stage}_seed$seed --stage $stage --real-data "$REAL" \
        --epochs "$epochs" --seed $seed > /dev/null
    done
  done
  echo "START $(date -Is) arithmetic"; spider -m handsoff.model.arithmetic results/final/sim_real_cosmos_seed0 --sim "$SIM"
  echo "DONE $(date -Is) results/final"
}
detach results/logs/train.log
