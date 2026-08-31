#!/bin/bash
# Launch a wave-3 job detached, with the machine held awake for its whole life.
#
#   research/scripts/bg.sh <logname> <script.py> [args...]
#
# caffeinate -ims blocks idle sleep, disk idle sleep and system sleep; -w ties
# the assertion to the job's pid so it is released the moment the job exits.
set -euo pipefail
WT="/path/to/workspace/structural-break-claude-wave3"
VENV="/path/to/workspace/structural-break/.venv/bin/python"
LOG="$WT/logs/$1.log"; shift
mkdir -p "$WT/logs"
cd "$WT"
nohup env PYTHONPATH="$WT/src:$WT/research/scripts" SBR_ROOT="$WT" \
  SBR_STORE="$WT/cache/store" SBR_FEATURES="$WT/cache/features" \
  "$VENV" "$@" > "$LOG" 2>&1 &
PID=$!
nohup caffeinate -ims -w "$PID" > /dev/null 2>&1 &
echo "pid $PID  log $LOG  (machine held awake until it exits)"
