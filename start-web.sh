#!/bin/bash
# Start the Goswami Whisper Flask web interface
set -e

cd "$(dirname "$0")"

# Kill any existing instance
bash kill-web.sh

# Start
. venv/bin/activate
nohup python main.py > /tmp/goswami-whisper.log 2>&1 &
echo "Started Goswami Whisper (PID: $!)"
