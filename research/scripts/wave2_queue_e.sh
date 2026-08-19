#!/bin/bash
# Single serial queue for everything after LOMO. 2 cores: never run two at once.
cd /home/claude/sb
export PYTHONPATH=/home/claude/sb/src:/home/claude/sb/research/scripts
while ! grep -q "LOMO DONE" logs/battery.log 2>/dev/null; do sleep 60; done

# (1) the control WITHOUT which leave-one-module-out is uninterpretable
python3 research/scripts/wave2_lomo_control.py   >> logs/battery2.log 2>&1
echo "LOMO CONTROL DONE" >> logs/battery2.log

# (2) the two surprising removals, re-tested at the CHAMPION protocol
python3 research/scripts/wave2_lomo_confirm.py m01_seq m00_core >> logs/battery2.log 2>&1
echo "LOMO CONFIRM DONE" >> logs/battery2.log

# (3) fold-assignment stability
python3 research/scripts/wave2_battery.py altf   >> logs/battery2.log 2>&1
echo "ALTF DONE" >> logs/battery2.log

# (4) seed stability
python3 research/scripts/wave2_battery.py seeds  >> logs/battery2.log 2>&1
echo "SEEDS DONE" >> logs/battery2.log

# (5) the negative control that was OOM-killed
python3 research/scripts/wave2_nc_randomblock.py >> logs/battery2.log 2>&1
echo "NC RANDOMBLOCK DONE" >> logs/battery2.log

# (6) cross-fitted feature-count reduction (gain-ranked; see the caveat in
#     STATE_OF_RESEARCH_V2 -- gain is a poor proxy for metric value here)
python3 research/scripts/wave2_battery.py fcount >> logs/battery2.log 2>&1
echo "BATTERY DONE" >> logs/battery2.log
