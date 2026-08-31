#!/bin/bash
# Serial compute queue, 2 cores. Order = descending scientific value per hour.
cd /home/claude/sb
export PYTHONPATH=/home/claude/sb/src:/home/claude/sb/research/scripts

# (1) Submission C: the best legally deployable system (7 boosters, one engine)
python3 research/scripts/wave2_train_ensemble.py >> logs/ens_train.log 2>&1
echo "ENS TRAIN DONE" >> logs/battery.log

# (2) negative controls -- cheap, and they gate everything else
python3 research/scripts/wave2_battery.py nc     >> logs/battery.log 2>&1
echo "NC DONE" >> logs/battery.log

# (3) leave-one-module-out at the battery protocol
python3 research/scripts/wave2_battery.py lomo   >> logs/battery.log 2>&1
echo "LOMO DONE" >> logs/battery.log

# (4) fold-assignment stability
python3 research/scripts/wave2_battery.py altf   >> logs/battery.log 2>&1
echo "ALTF DONE" >> logs/battery.log

# (5) seed stability
python3 research/scripts/wave2_battery.py seeds  >> logs/battery.log 2>&1
echo "SEEDS DONE" >> logs/battery.log

# (6) cross-fitted feature-count reduction
python3 research/scripts/wave2_battery.py fcount >> logs/battery.log 2>&1
echo "BATTERY DONE" >> logs/battery.log
