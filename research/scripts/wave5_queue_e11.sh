#!/bin/zsh
set -u
ROOT="/path/to/workspace/structural-break-wave5"
PY="/path/to/workspace/structural-break/.venv/bin/python"
export SBR_ROOT="$ROOT"
cd "$ROOT"
# start once stage D has finished, keeping two trainers resident at most
while ! grep -q 'STAGE D COMPLETE' "$ROOT/logs/w5staged.log" 2>/dev/null; do sleep 30; done
sleep 10
echo "starting W5-E11 architecture rebuild $(date)"
"$PY" -u "$ROOT/research/scripts/wave5_e11_arch.py"
echo "E11 COMPLETE $(date)"
