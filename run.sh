#!/bin/bash

ISAAC_PYTHON="/home/miislab-server10/isaac-sim-5.1.0/python.sh"

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MAIN_SCRIPT="$PROJECT_DIR/main.py"

echo "[INFO] Launching Controller..."
echo "[INFO] Using Isaac Sim Python: $ISAAC_PYTHON"

$ISAAC_PYTHON "$MAIN_SCRIPT" "$@"

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "[ERROR] Simulation exited with error code $EXIT_CODE"
else
    echo "[INFO] Simulation finished."
fi