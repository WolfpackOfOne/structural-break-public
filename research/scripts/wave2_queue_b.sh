#!/bin/bash
# Runs after queue A. 2 cores: strictly serial.
cd /home/claude/sb
export PYTHONPATH=/home/claude/sb/src:/home/claude/sb/research/scripts
while ! grep -q "QUEUE-A DONE" logs/streams.log 2>/dev/null; do sleep 60; done
python3 research/scripts/wave2_streams.py RT-120R >> logs/streams.log 2>&1
python3 research/scripts/wave2_streams.py RT-125R >> logs/streams.log 2>&1
echo "QUEUE-B DONE" >> logs/streams.log
python3 research/scripts/wave2_deployable_ensemble.py > logs/deployable.log 2>&1
echo "ENSEMBLE DONE" >> logs/streams.log
