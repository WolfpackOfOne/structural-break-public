#!/bin/bash
# Serial compute queue.  2 cores: never run two of these at once.
set -x
cd /home/claude/sb
export PYTHONPATH=/home/claude/sb/src:/home/claude/sb/research/scripts
python3 research/scripts/wave2_streams.py RT-120R  >> logs/streams.log 2>&1
python3 research/scripts/wave2_streams.py RT-121R  >> logs/streams.log 2>&1
python3 research/scripts/wave2_streams.py RT-124R  >> logs/streams.log 2>&1
python3 research/scripts/wave2_streams.py RT-122R  >> logs/streams.log 2>&1
python3 research/scripts/wave2_streams.py RT-125R  >> logs/streams.log 2>&1
python3 research/scripts/wave2_streams.py RT-123R  >> logs/streams.log 2>&1
echo "QUEUE-A DONE" >> logs/streams.log
