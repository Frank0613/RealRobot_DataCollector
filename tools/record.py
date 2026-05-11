import os
import h5py
import numpy as np
from datetime import datetime


class DataCollector:
    """
    Minimal demo recorder. One HDF5 file, multiple demos appended under /data.

    Per frame it stores (sim side):
      - arm_joint_pos  (n_arm_joints,) float32
      - arm_joint_vel  (n_arm_joints,) float32
      - gripper_pos    scalar float32   (only if the robot has a gripper)

    Per frame it stores (real side, only when a `ros_sub` is provided AND
    a synced triple has been cached — frames without synced data are skipped):
      - real_joint_pos  (n_real_joints,) float32
      - real_joint_vel  (n_real_joints,) float32
      - real_rgb        (H, W, 3) uint8
      - real_depth      (H, W) uint16 or float32
      - real_stamp      scalar float64 (rgb header.stamp in seconds)

    Per demo it stores attrs:
      - num_samples, robot_name, timestamp, arm_joint_names
      - real_joint_names  (only when ros_sub provided synced data)
    """

    def __init__(self, save_dir="datasets", filename="dataset.hdf5"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.filepath = os.path.join(self.save_dir, filename)

        self.recording = False
        self.frames = []

        # Initialize the file if it doesn't already have a /data group
        with h5py.File(self.filepath, "a") as f:
            if "data" not in f:
                root = f.create_group("data")
                root.attrs["total"] = 0
                root.attrs["created"] = datetime.now().isoformat()

    # ---------- recording lifecycle ----------

    def start(self):
        if self.recording:
            return
        self.recording = True
        self.frames = []
        print("[DataCollector] Start recording")

    def discard(self):
        """Drop the in-progress demo without saving."""
        if not self.recording and not self.frames:
            return
        n = len(self.frames)
        self.recording = False
        self.frames = []
        print(f"[DataCollector] Discarded {n} frames")

    # ---------- per-step capture ----------

    def collect_frame(self, controller, ros_sub=None):
        if not self.recording:
            return
        if controller._arm_idx is None:
            return

        qp = controller.robot.get_joint_positions()
        qv = controller.robot.get_joint_velocities()
        if qp is None or qv is None:
            return

        # Strict-sync rule: if a real-robot subscriber is attached but no
        # synchronized triple has arrived yet, skip this frame entirely so
        # every saved frame contains both sim and real data.
        real_data = None
        if ros_sub is not None:
            real_data = ros_sub.get_latest()
            if real_data is None:
                return

        arm_idx = controller._arm_idx
        frame = {
            "arm_joint_pos": np.asarray(qp[arm_idx], dtype=np.float32),
            "arm_joint_vel": np.asarray(qv[arm_idx], dtype=np.float32),
        }
        if controller._gripper_idx is not None:
            frame["gripper_pos"] = np.float32(qp[controller._gripper_idx])

        if real_data is not None:
            frame.update(real_data)

        self.frames.append(frame)

    # ---------- save ----------

    def save(self, controller, ros_sub=None):
        """Save the current demo to HDF5 and reset state. Returns True on success."""
        if not self.frames:
            print("[DataCollector] Nothing to save")
            self.recording = False
            return False

        cfg = controller.cfg
        with h5py.File(self.filepath, "a") as f:
            root = f["data"]
            demo_id = int(root.attrs["total"])
            demo_name = f"demo_{demo_id}"
            demo = root.create_group(demo_name)

            demo.attrs["num_samples"] = len(self.frames)
            demo.attrs["robot_name"] = cfg.ROBOT_NAME
            demo.attrs["timestamp"] = datetime.now().isoformat()
            demo.attrs["arm_joint_names"] = list(cfg.ARM_JOINT_NAMES)
            if ros_sub is not None and ros_sub.real_joint_names:
                demo.attrs["real_joint_names"] = ros_sub.real_joint_names

            # Stack each key across frames into one dataset; compress images.
            keys = self.frames[0].keys()
            for key in keys:
                arr = np.stack([fr[key] for fr in self.frames])
                if arr.ndim >= 3:  # image streams (T,H,W) or (T,H,W,C)
                    demo.create_dataset(
                        key, data=arr,
                        compression="gzip", compression_opts=4, chunks=True,
                    )
                else:
                    demo.create_dataset(key, data=arr)

            root.attrs["total"] = demo_id + 1

        print(f"[DataCollector] Saved {demo_name} ({len(self.frames)} frames) → {self.filepath}")
        self.recording = False
        self.frames = []
        return True