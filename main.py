import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Robot IK teleop / dataset tools")
    parser.add_argument(
        "--readfile",
        type=str,
        help="Print HDF5 structure of datasets/<NAME>.hdf5 and exit",
    )
    parser.add_argument(
        "--replay",
        type=str,
        help="Replay every demo in datasets/<NAME>.hdf5 once, then close",
    )
    args, _ = parser.parse_known_args()

    # --readfile: do not boot Isaac Sim, just inspect the file
    if args.readfile:
        from tools.hdf5_reader import print_structure_by_path
        path = os.path.join("datasets", f"{args.readfile}.hdf5")
        print_structure_by_path(path)
        return

    print("Starting Isaac Sim...")
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": False})

    # Enable ROS2 Bridge with internal humble libraries.
    # Setting must be applied before the extension loads.
    import carb
    carb.settings.get_settings().set(
        "/exts/isaacsim.ros2.bridge/ros_distro", "humble"
    )
    from isaacsim.core.utils.extensions import enable_extension
    enable_extension("isaacsim.ros2.bridge")
    simulation_app.update()

    from omni.isaac.core import World
    import omni.usd
    from pxr import UsdLux, Sdf, Gf

    from input_manager import InputManager
    from robot_controller import RobotIKController
    from tools.record import DataCollector

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

    world.reset()
    controller.initialize_handles()

    # --replay: play the dataset once and exit
    if args.replay:
        from tools.replay import replay_dataset
        path = os.path.join("datasets", f"{args.replay}.hdf5")
        replay_dataset(world, controller, path)
        simulation_app.close()
        return

    input_mgr = InputManager()
    data_collector = DataCollector(
        save_dir="datasets",
        filename=f"{robot_cfg.ROBOT_NAME}.hdf5",
    )

    print("==========================================")
    print(f" {robot_cfg.ROBOT_NAME} IK Teleop")
    print(" Move    : WASDQE")
    print(" Rotate  : Z/X  T/G  C/V")
    print(" Gripper : K (toggle)")
    print(" Save    : B (save demo + reset)")
    print(" Reset   : R (discards demo if recording)")
    print("==========================================")

    needs_reset = False
    while simulation_app.is_running():
        if world.is_playing():
            if needs_reset:
                world.reset()
                controller.initialize_handles()
                input_mgr.reset()
                needs_reset = False
                print("[Env] Reset")

            (delta_pos, delta_rot, gripper_cmd,
             reset_cmd, save_cmd, is_any_action) = input_mgr.get_command()

            # Auto-start recording on the first user action
            if not data_collector.recording and is_any_action:
                data_collector.start()

            # B: save current demo, then reset
            if save_cmd and data_collector.recording:
                data_collector.save(controller)
                needs_reset = True

            # R: reset; if mid-recording, discard the unsaved demo
            if reset_cmd:
                if data_collector.recording:
                    data_collector.discard()
                needs_reset = True

            controller.apply_control(delta_pos, gripper_cmd, delta_rot)
            data_collector.collect_frame(controller)
            world.step(render=True)
        else:
            simulation_app.update()

    simulation_app.close()


if __name__ == "__main__":
    main()