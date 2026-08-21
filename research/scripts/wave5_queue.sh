#!/bin/zsh
# Wave-5 stage-C queue.  Waits for the feature caches, then runs the block arms
# one at a time so the machine is never oversubscribed.
set -u
ROOT="/path/to/workspace/structural-break-wave5"
PY="/path/to/workspace/structural-break/.venv/bin/python"
export SBR_ROOT="$ROOT"
cd "$ROOT"

# The driver preallocates the .npy with open_memmap(mode="w+"), so the file is
# FULL SIZE from the first second and testing for its existence proves nothing.
# Wait for the driver's own completion line instead.  This is not hypothetical:
# an earlier version of this script started RT-740 twice on a cache that was
# mostly zeros, which would have produced a real-looking and entirely fake
# number had either run been allowed to finish.
wait_build() {
  local log="$1"
  while ! grep -q '^total ' "$log" 2>/dev/null; do sleep 10; done
  sleep 3
  echo "  build complete: $log"
}

echo "waiting for feature caches..."
wait_build "$ROOT/logs/build_m12.log"
wait_build "$ROOT/logs/build_m10.log"
echo "caches ready $(date)"

for exp in RT-740 RT-750 RT-760; do
  echo "=== $exp $(date) ==="
  "$PY" -u "$ROOT/research/scripts/wave5_blocks.py" "$exp"
done
echo "QUEUE COMPLETE $(date)"
