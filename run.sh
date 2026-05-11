#!/bin/bash

ISAAC_PYTHON="/home/miislab-server8/isaac-sim-5.1.0/python.sh"
ISAAC_ROS2_HUMBLE="/home/miislab-server8/isaac-sim-5.1.0/exts/isaacsim.ros2.bridge/humble"

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MAIN_SCRIPT="$PROJECT_DIR/main.py"

# Use Isaac Sim's internal ROS2 Humble libraries (Python 3.11 compatible).
export ROS_DISTRO=humble
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export LD_LIBRARY_PATH="$ISAAC_ROS2_HUMBLE/lib:$LD_LIBRARY_PATH"

# Prepend Isaac Sim's internal rclpy (Python 3.11) so it shadows the system
# rclpy (Python 3.10) which cannot load its C extension under Python 3.11.
# System paths are kept for pure-Python packages like message_filters/sensor_msgs.
export PYTHONPATH="$ISAAC_ROS2_HUMBLE/rclpy:$PYTHONPATH"

echo "[INFO] Launching Controller..."
echo "[INFO] Using Isaac Sim Python: $ISAAC_PYTHON"

$ISAAC_PYTHON "$MAIN_SCRIPT" "$@"

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "[ERROR] Simulation exited with error code $EXIT_CODE"
else
    echo "[INFO] Simulation finished."
fi