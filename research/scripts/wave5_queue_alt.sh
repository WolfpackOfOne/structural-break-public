#!/bin/zsh
set -u
ROOT="/path/to/workspace/structural-break-wave5"
PY="/path/to/workspace/structural-break/.venv/bin/python"
export SBR_ROOT="$ROOT"
cd "$ROOT"
for p in alt1 alt2; do
  echo "=== $p $(date) ==="
  "$PY" -u "$ROOT/research/scripts/wave5_alt_blocks.py" "$p"
done
echo "ALT COMPLETE $(date)"
