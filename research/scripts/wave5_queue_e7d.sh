#!/bin/zsh
set -u
ROOT="/path/to/workspace/structural-break-wave5"
PY="/path/to/workspace/structural-break/.venv/bin/python"
export SBR_ROOT="$ROOT"
cd "$ROOT"
# Keep at most three trainers resident: 16 GB total and a CHAMP arm over 576
# columns needs ~4.6 GB for its training and validation matrices alone.
while ! grep -q 'W5-E9 done\|^\[RT-701\]' "$ROOT/logs/w5e9.log" 2>/dev/null; do sleep 20; done
sleep 10
echo "starting RT-731 $(date)"
"$PY" -u "$ROOT/research/scripts/wave5_e7_focus.py" d
echo "RT-731 COMPLETE $(date)"
