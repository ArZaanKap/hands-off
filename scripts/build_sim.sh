#!/usr/bin/env bash
# Build the sim training set (about 2.5 h on one GB10): MuJoCo physics -> phone cameras -> flat render and block
# tracks -> photoreal ovrtx render and crop -> data checks -> Cosmos-Embed1 features. Stages resume.
# Usage: bash scripts/build_sim.sh [DATA]      (default data/sim/intent_tasks_train_v3; log results/logs/build_sim.log)
source "$(dirname -- "$0")/common.sh"
export D=${1:-data/sim/intent_tasks_train_v3}
chain() {
  for stage in physics window cameras render tracks; do
    echo "START $(date -Is) $stage"; MUJOCO_GL=egl spider -m handsoff.sim.dataset --stage $stage --out "$D"
  done
  for stage in export render crop; do
    echo "START $(date -Is) photoreal $stage"
    if [[ $stage == render ]]; then render -m handsoff.sim.photoreal $stage --data "$D"; else spider -m handsoff.sim.photoreal $stage --data "$D"; fi
  done
  echo "START $(date -Is) checks"; spider -m handsoff.checks --sim "$D"
  echo "START $(date -Is) embed"; spider -m handsoff.model.embed "$D" --batch 4 --save-every 4
  echo "DONE $(date -Is) $D"
}
detach results/logs/build_sim.log
