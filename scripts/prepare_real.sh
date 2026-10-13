#!/usr/bin/env bash
# The user's phone takes -> model inputs, after `prepare sync` and the user's label confirmation (see README):
# SAM3 masks, crops and block tracks (+ approved Cosmos variants), Cosmos-Embed1 features, user-rule checks.
# Usage: bash scripts/prepare_real.sh RAW REVIEW OUT LABELS TAG [COSMOS]
#   training takes: bash scripts/prepare_real.sh data/raw/train_takes results/train_inputs data/real/demos3_approved_rgb_v3 \
#                        config/labels_train.json demos3 results/cosmos_demos3_v1
source "$(dirname -- "$0")/common.sh"
export RAW=$1 REVIEW=$2 OUT=$3 LABELS=$4 TAG=$5 COSMOS=${6:-}
chain() {
  echo "START $(date -Is) inputs"
  mask -m handsoff.real.prepare inputs --raw "$RAW" --review "$REVIEW" --out "$OUT" --labels "$LABELS" --tag "$TAG" ${COSMOS:+--cosmos "$COSMOS"}
  echo "START $(date -Is) embed"; spider -m handsoff.model.embed "$OUT" --batch 4 --save-every 4
  echo "DONE $(date -Is) $OUT"
}
detach "results/logs/prepare_$(basename "$OUT").log"
