#!/usr/bin/env bash
# New recordings -> the robot: inputs exactly as for the development takes -> each trained head -> the policy ->
# the robot at the fixed start. No sim dataset needed. Before it: `prepare sync` (CPU), then watch every
# *_both_phones.mp4 and confirm RAW/labels.json. Step by step for new recordings: docs/own_recordings.md.
# Usage: bash scripts/run_test.sh RAW [OUT] [REVIEW] [DATA]
#   e.g. bash scripts/run_test.sh data/raw/my_takes    (defaults results/test, results/test_inputs, data/real/test_takes)
# WEIGHTS: the trained system, default weights/ (shipped, seed 0); after scripts/train.sh: WEIGHTS=results/final (3 seeds).
source "$(dirname -- "$0")/common.sh"
export RAW=$1 OUT=${2:-results/test} REVIEW=${3:-results/test_inputs} DATA=${4:-data/real/test_takes} WEIGHTS=${WEIGHTS:-weights}
chain() {
  echo "START $(date -Is) inputs"; mask -m handsoff.real.prepare inputs --raw "$RAW" --review "$REVIEW" --out "$DATA"
  echo "START $(date -Is) embed features"; spider -m handsoff.model.embed "$DATA" --batch 4 --save-every 4
  echo "START $(date -Is) checks"; spider -m handsoff.checks --real "$DATA" --labels "$RAW/labels.json"
  for F in "$WEIGHTS"/seed*; do
    S=$(basename "$F")
    echo "START $(date -Is) z $S"; spider -m handsoff.policy.policy embed --real-only --head "$F/head" --real "$DATA" --z "$OUT/$S/z.npz"
    echo "START $(date -Is) robot $S"; spider -m handsoff.policy.policy eval --sim "$WEIGHTS/scene.json" --z "$OUT/$S/z.npz" \
      --plans "$WEIGHTS/durations.npz" --policy "$F/policy/policy.pt" --out "$OUT/$S" --sets real
  done
  echo "DONE $(date -Is) $OUT/seed*/report.json"
}
detach "$OUT/chain.log"
