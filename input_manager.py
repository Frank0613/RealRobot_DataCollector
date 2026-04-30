import carb
import omni.appwindow
import numpy as np


# Tune these directly
MOVE_SPEED = 0.0005     # meters per step on WASDQE
ROTATE_SPEED = 0.003    # radians per step on Z/X T/G C/V


class InputManager:
    """
    Keyboard teleop:
      WASDQE         translate target (x/y/z)
      Z/X T/G C/V    rotate target around x/y/z
      K              toggle gripper
      R              reset
    """

    def __init__(self):
        self._input = carb.input.acquire_input_interface()
        self._app_window = omni.appwindow.get_default_app_window()
        self._keyboard = self._app_window.get_keyboard()

        self.gripper_is_open = True
        self.prev_k = False
        self.prev_r = False

    def reset(self):
        """Reset all latched input state."""
        self.gripper_is_open = True
        self.prev_k = False
        self.prev_r = False

    def _rising(self, key, prev_attr):
        cur = bool(self._input.get_keyboard_value(self._keyboard, key))
        rising = cur and not getattr(self, prev_attr)
        setattr(self, prev_attr, cur)
        return cur, rising

    def get_command(self):
        """
        Returns: delta_pos(3,), delta_rot(3,), gripper_cmd, reset_cmd, is_any_action
        gripper_cmd: 0 = open, 1 = closed
        """
        K = carb.input.KeyboardInput

        delta = np.zeros(3)
        delta_rot = np.zeros(3)

        move_map = {
            K.W: (0,  MOVE_SPEED),
            K.S: (0, -MOVE_SPEED),
            K.A: (1,  MOVE_SPEED),
            K.D: (1, -MOVE_SPEED),
            K.Q: (2,  MOVE_SPEED),
            K.E: (2, -MOVE_SPEED),
        }
        any_move = False
        for key, (axis, val) in move_map.items():
            if self._input.get_keyboard_value(self._keyboard, key):
                delta[axis] += val
                any_move = True

        rot_map = {
            K.Z: (0,  ROTATE_SPEED),
            K.X: (0, -ROTATE_SPEED),
            K.T: (1,  ROTATE_SPEED),
            K.G: (1, -ROTATE_SPEED),
            K.C: (2,  ROTATE_SPEED),
            K.V: (2, -ROTATE_SPEED),
        }
        any_rot = False
        for key, (axis, val) in rot_map.items():
            if self._input.get_keyboard_value(self._keyboard, key):
                delta_rot[axis] += val
                any_rot = True

        # Gripper toggle (rising edge)
        k_cur, k_rise = self._rising(K.K, "prev_k")
        if k_rise:
            self.gripper_is_open = not self.gripper_is_open
        gripper_cmd = 0 if self.gripper_is_open else 1

        # Reset (rising edge)
        _, reset_cmd = self._rising(K.R, "prev_r")

        is_any_action = any_move or any_rot or k_cur
        return delta, delta_rot, gripper_cmd, reset_cmd, is_any_action