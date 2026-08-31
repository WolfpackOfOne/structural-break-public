#!/bin/zsh
set -u
ROOT="/path/to/workspace/structural-break-wave6"
PY="/path/to/workspace/structural-break/.venv/bin/python"
export SBR_ROOT="$ROOT"; cd "$ROOT"
# wait for the driver's own completion line -- NOT for the file to exist; the
# oracle builder preallocates with open_memmap exactly as sbr's driver does
while ! grep -q '^total ' "$ROOT/logs/w6_oracle_build.log" 2>/dev/null; do sleep 10; done
sleep 5
echo "starting RT-900 $(date)"
"$PY" -u "$ROOT/research/scripts/wave6_e2_tau.py"
echo "starting W6-E2 eval $(date)"
"$PY" -u "$ROOT/research/scripts/wave6_e2_eval.py"
echo "E2 COMPLETE $(date)"
