import numpy as np
from omni.isaac.core import World
from omni.isaac.core.robots import Robot
from omni.isaac.core.objects import VisualSphere
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.motion_generation import LulaKinematicsSolver
from scipy.spatial.transform import Rotation as R


# Visual marker showing where the IK target is (independent of any robot)
EE_MARKER_PATH = "/World/EETargetMarker"
EE_MARKER_RADIUS = 0.008


class RobotIKController:
    """
    Generic IK controller for any arm described by:
      - a URDF file
      - a Lula robot-description YAML
      - a USD file
      - a per-robot config object (see robot/cfg.py for the expected schema)

    All robot-specific values live in `cfg`; this class is robot-agnostic.
    """

    def __init__(
        self,
        world: World,
        cfg,
        prim_path: str = None,
        position: np.ndarray = None,
    ):
        self.world = world
        self.cfg = cfg
        self.prim_path = prim_path or f"/World/{cfg.ROBOT_NAME}"
        position = np.zeros(3) if position is None else np.asarray(position)

        if not is_prim_path_valid(self.prim_path):
            add_reference_to_stage(usd_path=cfg.USD_PATH, prim_path=self.prim_path)

        self.robot = self.world.scene.add(
            Robot(prim_path=self.prim_path, name=cfg.ROBOT_NAME, position=position)
        )

        self.kinematics_solver = LulaKinematicsSolver(
            robot_description_path=cfg.DESCRIPTOR_PATH,
            urdf_path=cfg.URDF_PATH,
        )

        # Resolved in initialize_handles() (after the articulation is live)
        self._arm_idx = None
        self._gripper_idx = None

        # IK target state (in robot base_link frame)
        self.target_pos = None
        self.target_rot = None
        self.gripper_state = 0
        self.current_gripper_pos = None

        # Red ball marker showing the IK target
        self.ee_marker = None

    @property
    def has_gripper(self) -> bool:
        return getattr(self.cfg, "GRIPPER_JOINT_NAME", None) is not None

    # ---------- setup ----------

    def initialize_handles(self):
        cfg = self.cfg
        print(f"[RobotIKController] Initializing handles for '{cfg.ROBOT_NAME}'...")

        # Reset transient state
        self.gripper_state = 0
        self.current_gripper_pos = cfg.GRIPPER_OPEN_POS if self.has_gripper else None

        # Marker (created once)
        if self.ee_marker is None and not is_prim_path_valid(EE_MARKER_PATH):
            self.ee_marker = VisualSphere(
                prim_path=EE_MARKER_PATH,
                name="ee_target_marker",
                radius=EE_MARKER_RADIUS,
                color=np.array([1.0, 0.0, 0.0]),
            )

        # Resolve joint indices
        dof_names = self.robot.dof_names
        print(f"[ArmIKController] DOF names: {dof_names}")
        self._arm_idx = [dof_names.index(n) for n in cfg.ARM_JOINT_NAMES]

        if self.has_gripper:
            if cfg.GRIPPER_JOINT_NAME in dof_names:
                self._gripper_idx = dof_names.index(cfg.GRIPPER_JOINT_NAME)
            else:
                print(
                    f"[ArmIKController] Gripper joint '{cfg.GRIPPER_JOINT_NAME}' "
                    f"not in dof_names; gripper control disabled"
                )
                self._gripper_idx = None
        else:
            self._gripper_idx = None

        # Set home pose
        num_dof = self.robot.num_dof
        home_q = np.zeros(num_dof)
        for i, idx in enumerate(self._arm_idx):
            home_q[idx] = cfg.ARM_HOME_Q[i]
        if self._gripper_idx is not None:
            home_q[self._gripper_idx] = cfg.GRIPPER_OPEN_POS
        self.robot.set_joint_positions(home_q)

        # PD gains
        kps = np.full(num_dof, cfg.KPS_ARM)
        kds = np.full(num_dof, cfg.KDS_ARM)
        if self._gripper_idx is not None:
            kps[self._gripper_idx] = cfg.KPS_GRIPPER
            kds[self._gripper_idx] = cfg.KDS_GRIPPER
        self.robot.get_articulation_controller().set_gains(kps=kps, kds=kds)

        # Initial IK target = forward kinematics of home (always reachable)
        fk_pos, fk_rot_mat = self.kinematics_solver.compute_forward_kinematics(
            cfg.EE_FRAME_NAME, np.asarray(cfg.ARM_HOME_Q)
        )
        self.target_pos = np.array(fk_pos, dtype=float)
        q_xyzw = R.from_matrix(np.array(fk_rot_mat)).as_quat()
        self.target_rot = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
        print(f"[ArmIKController] Initial target_pos={self.target_pos}")
        print(f"[ArmIKController] Initial target_rot(wxyz)={self.target_rot}")

        self._sync_marker_pose()

    # ---------- frame helpers ----------

    def _local_to_world(self, local_pos, local_quat_wxyz):
        base_pos, base_rot_wxyz = self.robot.get_world_pose()
        r_base = R.from_quat([
            base_rot_wxyz[1], base_rot_wxyz[2],
            base_rot_wxyz[3], base_rot_wxyz[0],
        ])
        world_pos = base_pos + r_base.apply(local_pos)
        r_local = R.from_quat([
            local_quat_wxyz[1], local_quat_wxyz[2],
            local_quat_wxyz[3], local_quat_wxyz[0],
        ])
        q = (r_base * r_local).as_quat()
        return world_pos, np.array([q[3], q[0], q[1], q[2]])

    def _sync_marker_pose(self):
        if self.ee_marker is None or self.target_pos is None:
            return
        world_pos, world_rot = self._local_to_world(self.target_pos, self.target_rot)
        self.ee_marker.set_world_pose(position=world_pos, orientation=world_rot)

    # ---------- gripper ----------

    def _update_gripper(self, gripper_cmd):
        cfg = self.cfg
        self.gripper_state = float(gripper_cmd)
        goal = cfg.GRIPPER_CLOSED_POS if self.gripper_state == 1 else cfg.GRIPPER_OPEN_POS
        if self.current_gripper_pos < goal:
            self.current_gripper_pos = min(self.current_gripper_pos + cfg.GRIPPER_SPEED, goal)
        elif self.current_gripper_pos > goal:
            self.current_gripper_pos = max(self.current_gripper_pos - cfg.GRIPPER_SPEED, goal)
        return self.current_gripper_pos

    # ---------- main control step ----------

    def apply_control(self, delta_pos, gripper_cmd, delta_rot=None):
        current_joints = self.robot.get_joint_positions()
        if current_joints is None or self._arm_idx is None:
            return

        if np.linalg.norm(delta_pos) > 0:
            self.target_pos = self.target_pos + delta_pos

        if delta_rot is not None and np.linalg.norm(delta_rot) > 0:
            r_current = R.from_quat([
                self.target_rot[1], self.target_rot[2],
                self.target_rot[3], self.target_rot[0],
            ])
            r_delta = R.from_euler('xyz', delta_rot)
            q = (r_delta * r_current).as_quat()
            self.target_rot = np.array([q[3], q[0], q[1], q[2]])

        # Move the marker first — even if IK fails the user sees their target
        self._sync_marker_pose()

        warm_start = current_joints[self._arm_idx]
        ik_results, success = self.kinematics_solver.compute_inverse_kinematics(
            frame_name=self.cfg.EE_FRAME_NAME,
            target_position=self.target_pos,
            target_orientation=self.target_rot,
            warm_start=warm_start,
        )

        if not success:
            print(f"[ArmIKController] IK failed | target_pos={self.target_pos}")
            return

        action = current_joints.copy()
        for i, idx in enumerate(self._arm_idx):
            action[idx] = ik_results[i]
        if self._gripper_idx is not None:
            action[self._gripper_idx] = self._update_gripper(gripper_cmd)

        self.robot.apply_action(ArticulationAction(joint_positions=action))
