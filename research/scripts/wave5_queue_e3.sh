#!/bin/zsh
set -u
ROOT="/path/to/workspace/structural-break-wave5"
PY="/path/to/workspace/structural-break/.venv/bin/python"
export SBR_ROOT="$ROOT"
cd "$ROOT"
# W5-E3 runs last: it answers a pre-registered question but cannot change the
# LB-002 decision, and W5-E11 can.  Two trainers maximum -- three saturated this
# machine's 10 GB swap earlier and cost roughly 5x in wall time.
while ! grep -q 'E11 COMPLETE' "$ROOT/logs/w5e11.log" 2>/dev/null; do sleep 60; done
sleep 10
echo "starting W5-E3 arms $(date)"
"$PY" -u "$ROOT/research/scripts/wave5_e3_hardneg.py" RT-710 RT-711 RT-712
echo "E3 COMPLETE $(date)"
