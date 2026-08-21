#!/bin/zsh
set -u
ROOT="/path/to/workspace/structural-break-wave5"
PY="/path/to/workspace/structural-break/.venv/bin/python"
export SBR_ROOT="$ROOT"
cd "$ROOT"
# hold off until the m10 feature build has released its workers and memory
while ! grep -q '^total ' "$ROOT/logs/build_m10.log" 2>/dev/null; do sleep 10; done
sleep 10
echo "starting W5-E3 arms $(date)"
"$PY" -u "$ROOT/research/scripts/wave5_e3_hardneg.py" RT-710 RT-711 RT-712
echo "E3 COMPLETE $(date)"
