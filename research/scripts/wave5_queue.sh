#!/bin/zsh
# Wave-5 stage-C queue.  Waits for the feature caches, then runs the block arms
# one at a time so the machine is never oversubscribed.
set -u
cd "$(dirname "$0")/../.."
ROOT="$PWD"
PY="/path/to/workspace/structural-break/.venv/bin/python"
export SBR_ROOT="$ROOT"

wait_for() {
  local f="$1"
  while [ ! -s "$f" ]; do sleep 10; done
  # the driver writes the .npy last; give it a moment to close
  sleep 5
}

echo "waiting for feature caches..."
wait_for "$ROOT/cache/features/m12_rdep.npy"
wait_for "$ROOT/cache/features/m10_persist.npy"
echo "caches ready $(date)"

for exp in RT-740 RT-750 RT-760; do
  echo "=== $exp $(date) ==="
  "$PY" -u "$ROOT/research/scripts/wave5_blocks.py" "$exp"
done
echo "QUEUE COMPLETE $(date)"
