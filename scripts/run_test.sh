#!/usr/bin/env bash
# The single test on new recordings, run once (R08): inputs exactly as for the training takes -> the 9 final heads ->
# the robot. Before it: `prepare sync` (CPU), the user watches every *_both_phones.mp4 and confirms RAW/labels.json
# (R09), and the test take ids are in the test fold of config/real_split.json (R08).
# Usage: bash scripts/run_test.sh RAW [OUT] [REVIEW] [DATA]
#   e.g. bash scripts/run_test.sh data/raw/test_takes    (defaults results/test, results/test_inputs, data/real/test_takes)
source "$(dirname -- "$0")/common.sh"
export RAW=$1 OUT=${2:-results/test} REVIEW=${3:-results/test_inputs} DATA=${4:-data/real/test_takes}
chain() {
  SIM=data/sim/intent_tasks_train_v3
  echo "START $(date -Is) rules (code)"; spider -m handsoff.rules
  echo "START $(date -Is) inputs"; mask -m handsoff.real.prepare inputs --raw "$RAW" --review "$REVIEW" --out "$DATA"
  echo "START $(date -Is) embed"; spider -m handsoff.model.embed "$DATA" --batch 4 --save-every 4
  echo "START $(date -Is) rules"; spider -m handsoff.rules --sim $SIM --real "$DATA" --labels "$RAW/labels.json"
  echo "START $(date -Is) execute"; spider -m handsoff.model.evaluate execute --sim $SIM --queries "$DATA" --out "$OUT" \
    --runs results/final/*_seed* --purpose "Single test on new recordings (never trained on, never used to pick a model)"
  echo "START $(date -Is) render"; MUJOCO_GL=egl spider -m handsoff.model.evaluate render --sim $SIM --queries "$DATA" --out "$OUT"
  echo "START $(date -Is) report"; spider -m handsoff.model.evaluate report --sim $SIM --queries "$DATA" --out "$OUT"
  echo "DONE $(date -Is) $OUT/summary.md"
}
detach "$OUT/chain.log"
