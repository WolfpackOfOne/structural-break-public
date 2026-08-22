#!/bin/zsh
set -u
ROOT="/path/to/workspace/structural-break-wave5"
PY="/path/to/workspace/structural-break/.venv/bin/python"
export SBR_ROOT="$ROOT"
cd "$ROOT"
# TWO trainers maximum.  With three resident the 16 GB machine saturated its
# 10 GB swap (9.9 GB used, 26M swapouts) and every arm slowed by ~5x: the
# feature memmaps are 10 GB and each trainer's page-cache working set is
# several GB of them, so concurrency past two costs more than it buys.
while ! grep -q 'QUEUE COMPLETE' "$ROOT/logs/w5queue.log" 2>/dev/null; do sleep 30; done
sleep 15
echo "starting W5-E3 arms $(date)"
"$PY" -u "$ROOT/research/scripts/wave5_e3_hardneg.py" RT-710 RT-711 RT-712
echo "E3 COMPLETE $(date)"
