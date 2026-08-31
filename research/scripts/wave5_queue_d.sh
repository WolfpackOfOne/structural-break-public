#!/bin/zsh
set -u
ROOT="/path/to/workspace/structural-break-wave5"
PY="/path/to/workspace/structural-break/.venv/bin/python"
export SBR_ROOT="$ROOT"
cd "$ROOT"
# RT-731 is already running from the W5-E7 stage-D launch; only RT-751 is queued
# here.  Wait for BOTH the block queue and RT-731 so at most two trainers are
# resident -- three saturated this machine's swap earlier and cost ~5x.
while ! grep -q 'QUEUE COMPLETE' "$ROOT/logs/w5queue.log" 2>/dev/null; do sleep 30; done
while ! grep -q 'W5-E7 done' "$ROOT/logs/w5e7d.log" 2>/dev/null; do sleep 30; done
sleep 10
echo "starting stage D $(date)"
"$PY" -u "$ROOT/research/scripts/wave5_staged.py" RT-751
echo "STAGE D COMPLETE $(date)"
