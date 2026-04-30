import os
import h5py
import numpy as np
from omni.isaac.core.utils.types import ArticulationAction


def _decode(val):
    """h5py attrs may come back as bytes — coerce to str."""
    if isinstance(val, bytes):
        return val.decode()
    return val


def replay_dataset(world, controller, dataset_path):
    """
    Replay every demo in `dataset_path` once, in order.

    Aborts (with a warning) if any demo was recorded on a robot whose
    `robot_name` differs from the current controller's robot.
    """
    if not os.path.exists(dataset_path):
        print(f"[Replay] File not found: {dataset_path}")
        return

    cfg = controller.cfg

    with h5py.File(dataset_path, "r") as f:
        if "data" not in f:
            print(f"[Replay] No 'data' group in {dataset_path}")
            return

        data = f["data"]
        demo_names = sorted(
            list(data.keys()),
            key=lambda s: int(s.split("_")[-1]) if s.startswith("demo_") else 0,
        )
        if not demo_names:
            print("[Replay] No demos found in dataset")
            return

        # ---- Robot mismatch check (every demo) ----
        for name in demo_names:
            rec = _decode(data[name].attrs.get("robot_name", ""))
            if rec and rec != cfg.ROBOT_NAME:
                print(
                    f"[Replay] WARNING: '{name}' was recorded on robot '{rec}' "
                    f"but the current robot is '{cfg.ROBOT_NAME}'. Aborting replay."
                )
                return

        print(f"[Replay] Replaying {len(demo_names)} demo(s) on '{cfg.ROBOT_NAME}'")

        for name in demo_names:
            demo = data[name]
            arm_pos = np.asarray(demo["arm_joint_pos"])  # (T, n_arm)
            gripper_pos = (
                np.asarray(demo["gripper_pos"]) if "gripper_pos" in demo else None
            )
            T = arm_pos.shape[0]
            print(f"[Replay] {name}  ({T} frames)")

            # Reset back to home before each demo
            world.reset()
            controller.initialize_handles()

            arm_idx = controller._arm_idx
            gripper_idx = controller._gripper_idx
            num_dof = controller.robot.num_dof

            for t in range(T):
                current = controller.robot.get_joint_positions()
                action = current.copy() if current is not None else np.zeros(num_dof)
                for i, idx in enumerate(arm_idx):
                    action[idx] = arm_pos[t, i]
                if gripper_pos is not None and gripper_idx is not None:
                    action[gripper_idx] = float(gripper_pos[t])

                controller.robot.apply_action(
                    ArticulationAction(joint_positions=action)
                )
                world.step(render=True)

        print("[Replay] Done")