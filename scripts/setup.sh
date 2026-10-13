#!/usr/bin/env bash
# One-time setup (tested on an aarch64 DGX Spark, CUDA 13, Python 3.12, uv). Three venvs because their pins conflict:
#   .venv-spider  simulation (MuJoCo), Cosmos-Embed1 (needs transformers 4.51.3), training, evaluation
#   .venv-mask    SAM3 (needs transformers 5.x); request access to facebook/sam3 on Hugging Face and log in first
#   .venv-render  ovrtx, NVIDIA's RTX renderer, for the photoreal sim images
# Plus the Inspire hand model (SPIDER repo, pinned commit) and the HDRI backgrounds. Cosmos-Transfer2.5 (only needed to
# generate new appearance variants) is set up separately: see scripts/run_cosmos.sh.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
cuda_torch() {   # no aarch64 cu130 torchvision wheel: the PyPI build, without letting it replace the CUDA torch
  uv pip install --python "$1/bin/python" 'torch==2.9.0+cu130' --index-url https://download.pytorch.org/whl/cu130
  uv pip install --python "$1/bin/python" --no-deps 'torchvision==0.24.0'
}
for pair in spider:sim mask:mask render:render; do
  venv=.venv-${pair%%:*}; [[ -x $venv/bin/python ]] || uv venv "$venv" --python 3.12
  [[ $venv == .venv-render ]] || cuda_torch "$venv"
  uv pip install --python "$venv/bin/python" -r "env/requirements-${pair##*:}.txt"
done
[[ -d external/spider ]] || { git clone https://github.com/facebookresearch/spider external/spider &&
  git -C external/spider checkout 44717007de41cbef7565dff7ff9f4453557a2d3d; }
.venv-spider/bin/python -m handsoff.sim.ovrtx fetch   # HDRI backgrounds -> data/assets/hdri
