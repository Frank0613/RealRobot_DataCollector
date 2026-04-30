import os
import sys


def main():
    print("Starting Isaac Sim...")
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": False})

    from omni.isaac.core import World
    import omni.usd
    from pxr import UsdLux, Sdf, Gf

    from input_manager import InputManager
    from robot_controller import RobotIKController

    # ---- pick which arm to use here ----
    # To switch to a different arm, swap this import for another robot package
    # (e.g. `from robot_franka import cfg as robot_cfg`).
    from robot import cfg as robot_cfg

    def add_lights():
        stage = omni.usd.get_context().get_stage()
        # Soft ambient fill
        dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
        dome.CreateIntensityAttr(800.0)
        dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
        # Directional sun
        distant = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/DistantLight"))
        distant.CreateIntensityAttr(2500.0)
        distant.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
        distant.AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 0.0, 0.0))

    # Sanity-check asset files
    for p in (robot_cfg.USD_PATH, robot_cfg.URDF_PATH, robot_cfg.DESCRIPTOR_PATH):
        if not os.path.exists(p):
            print(f"[ERROR] Missing file: {p}")
            simulation_app.close()
            sys.exit(1)

    # World + default Isaac ground plane (blue grid) + lights
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    add_lights()

    controller = RobotIKController(world=world, cfg=robot_cfg)
    input_mgr = InputManager()

    world.reset()
    controller.initialize_handles()

    print("==========================================")
    print(f" {robot_cfg.ROBOT_NAME} IK Teleop")
    print(" Move    : WASDQE")
    print(" Rotate  : Z/X  T/G  C/V")
    print(" Gripper : K (toggle)")
    print(" Reset   : R")
    print("==========================================")

    needs_reset = False
    while simulation_app.is_running():
        if world.is_playing():
            if needs_reset:
                world.reset()
                controller.initialize_handles()
                input_mgr.reset()
                needs_reset = False

            delta_pos, delta_rot, gripper_cmd, reset_cmd, _ = input_mgr.get_command()
            if reset_cmd:
                needs_reset = True

            controller.apply_control(delta_pos, gripper_cmd, delta_rot)
            world.step(render=True)
        else:
            simulation_app.update()

    simulation_app.close()


if __name__ == "__main__":
    main()