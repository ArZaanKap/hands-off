#!/usr/bin/env bash
# Generate the Cosmos-Transfer2.5 appearance variants for every spec in COSMOS/specs (from `real.cosmos specs`), with
# the distilled edge model (4 steps); skips specs whose output exists. Needs the Cosmos-Transfer2.5 repo in
# external/cosmos-transfer2.5 (`uv sync --extra=cu130`; the checkpoints are gated on Hugging Face) and ~75 GB of memory.
# Every output then needs a full-video human review (`real.cosmos review`) before any use.
# Used only for the control experiment in docs/tried_and_dropped.md; the final system uses no generated video.
# Usage: bash scripts/run_cosmos.sh COSMOS
source "$(dirname -- "$0")/common.sh"
export COSMOS=$(realpath "$1")
chain() {
  todo=(); for f in "$COSMOS"/specs/*.json; do [[ -s "$COSMOS/out/$(basename "$f" .json).mp4" ]] || todo+=("$f"); done
  echo "START $(date -Is) ${#todo[@]} specs"; [[ ${#todo[@]} -gt 0 ]] || exit 0
  cd external/cosmos-transfer2.5
  timeout -k 60 $((600 * ${#todo[@]})) env COSMOS_EXPERIMENTAL_CHECKPOINTS=1 nice -n 10 .venv/bin/python examples/inference.py \
    -i "${todo[@]}" -o "$COSMOS/out" --model=edge/distilled --disable-guardrails
  echo "DONE $(date -Is) $COSMOS/out"
}
detach results/logs/cosmos.log
