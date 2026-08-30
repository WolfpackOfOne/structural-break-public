#!/bin/bash
# Runs after LOMO. Order = descending scientific value.
cd /home/claude/sb
export PYTHONPATH=/home/claude/sb/src:/home/claude/sb/research/scripts
while ! grep -q "LOMO DONE" logs/battery.log 2>/dev/null; do sleep 60; done

# (1) the control that makes LOMO interpretable at all
python3 research/scripts/wave2_lomo_control.py >> logs/battery2.log 2>&1
echo "LOMO CONTROL DONE" >> logs/battery2.log

# (2) the two surprising removals, re-tested at the CHAMPION protocol
python3 research/scripts/wave2_lomo_confirm.py m01_seq m00_core >> logs/battery2.log 2>&1
echo "LOMO CONFIRM DONE" >> logs/battery2.log

# (3) the negative control that was OOM-killed
python3 research/scripts/wave2_nc_randomblock.py >> logs/battery2.log 2>&1
echo "NC RANDOMBLOCK DONE" >> logs/battery2.log
