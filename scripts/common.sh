# Shared by the run scripts: one runner per venv (8 CPU threads, low priority) and a detached runner.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
spider() { OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 taskset -c 0-7 nice -n 10 .venv-spider/bin/python "$@"; }
mask() { OMP_NUM_THREADS=8 taskset -c 0-7 nice -n 10 .venv-mask/bin/python "$@"; }
render() { OMP_NUM_THREADS=8 taskset -c 0-7 nice -n 10 .venv-render/bin/python "$@"; }
# detach LOG: run the function `chain` in the background (survives logout, not a reboot); every step must succeed.
detach() {
  mkdir -p "$(dirname "$1")"
  export -f spider mask render chain
  nohup setsid bash -c 'set -euo pipefail; chain' > "$1" 2>&1 < /dev/null &
  echo "$!" > "${1%.log}.pid"; echo "Detached PID $!; log $1"
}
