#!/bin/bash
# Kill the Goswami Whisper Flask web interface
set -e

PIDS=$(pgrep -f 'python main.py' || true)

if [ -z "$PIDS" ]; then
    echo "Goswami Whisper is not running."
    exit 0
fi

echo "Killing Goswami Whisper (PID(s): $PIDS)..."
kill -9 $PIDS
echo "Done."
