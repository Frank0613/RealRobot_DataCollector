"""
Robot-specific configuration for the Jetcobot arm.

To use a different arm, copy this file to a new folder (e.g. robot_franka/)
alongside that arm's URDF / USD / Lula descriptor YAML, fill in the values
below, and switch the import line in main.py.
"""
import os
import numpy as np


# ---------- Asset paths ----------
ROBOT_DIR = os.path.dirname(os.path.abspath(__file__))
USD_PATH = os.path.join(ROBOT_DIR, "jetcobot.usd")
URDF_PATH = os.path.join(ROBOT_DIR, "jetcobot.urdf")
DESCRIPTOR_PATH = os.path.join(ROBOT_DIR, "jetcobot_description.yaml")

# ---------- Articulation ----------
# Display name (used for the Isaac Robot wrapper and default prim path)
ROBOT_NAME = "jetcobot"

# Arm joints — must match the URDF and the `cspace` order in the descriptor YAML
ARM_JOINT_NAMES = ["Joint_1", "Joint_2", "Joint_3", "Joint_4", "Joint_5", "Joint_6"]

# Initial / reset pose for the arm joints (same length and order as ARM_JOINT_NAMES)
ARM_HOME_Q = np.array([0.0, -0.3, 0.5, 0.0, 0.5, 0.0])

# Link name in the URDF that the IK target refers to (the "tool0" / EE link)
EE_FRAME_NAME = "gripper_base"

# ---------- Gripper (set GRIPPER_JOINT_NAME = None if the arm has no gripper) ----------
GRIPPER_JOINT_NAME = "gripper_controller"
GRIPPER_OPEN_POS = -0.7         # joint angle when fully open
GRIPPER_CLOSED_POS = 0.15       # joint angle when fully closed
GRIPPER_SPEED = 0.05            # rad per simulation step

# ---------- PD gains ----------
KPS_ARM = 1e5
KDS_ARM = 1e3
KPS_GRIPPER = 1e4
KDS_GRIPPER = 1e2

# ---------- Real robot ROS2 topics (data collection) ----------
# These are subscribed alongside the sim and saved into the same HDF5 demo.
# ApproximateTimeSynchronizer pairs the three topics by header.stamp.
REAL_JOINT_STATE_TOPIC = "/arm_joint_states"
REAL_RGB_TOPIC         = "/k4a/rgb/image_raw/compressed"
REAL_RGB_COMPRESSED    = True   # True → sensor_msgs/CompressedImage, False → sensor_msgs/Image
REAL_DEPTH_TOPIC       = "/k4a/depth_to_rgb/image_raw"
REAL_SYNC_SLOP         = 0.05   # seconds — max time skew between the 3 topics
REAL_SYNC_QUEUE_SIZE   = 10