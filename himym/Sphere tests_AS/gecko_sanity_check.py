"""
Gecko motion sanity viewer on SphericalWorld
--------------------------------------------
- Builds the spherical world
- Spawns the gecko (single-attach, bbox-aware via SphericalWorld)
- Warm-settles onto the sphere with radial gravity
- Drives all actuators with phase-shifted sinusoids
- Shows live viewer (does NOT auto-close; press ESC)
- Prints basic displacements and contacts while running
"""

import os
import numpy as np
import mujoco
from mujoco import viewer

from ariel.simulation.environments.spherical_world import SphericalWorld
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko


def main():
    # -------------------------------------------------
    # Build world + robot
    # -------------------------------------------------
    # os.environ["MUJOCO_GL"] = "angle"  # uncomment on Windows if you see a black screen
    world = SphericalWorld(radius=5.0, radial_gravity=True)

    g = gecko()
    gecko_spec = g.spec if hasattr(g, "spec") else g

    world.spawn(gecko_spec)

    model = world.spec.compile()
    world.attach_model(model)

    # Physics parameters (stable)
    model.opt.timestep = 0.004
    model.opt.iterations = 80
    data = mujoco.MjData(model)

    # Let contacts form physically
    world.warm_start_settle(data, steps=800)
    print(f"[Settle] Contacts after warm start: {data.ncon}")

    # -------------------------------------------------
    # Introspect the model
    # -------------------------------------------------
    print("\n===== MODEL INSPECTION =====")
    print(f"nq (joint positions): {model.nq}")
    print(f"nv (joint velocities): {model.nv}")
    print(f"nu (actuators): {model.nu}")
    print(f"nbody: {model.nbody}")

    print("\nActuator details:")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        gear = np.array(model.actuator_gear[i], dtype=float)
        rng = np.array(model.actuator_ctrlrange[i], dtype=float)
        print(f"  [{i}] name={name} gear={np.round(gear,3).tolist()} ctrlrange={np.round(rng,3).tolist()}")

    print("\nRoot body name:", mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, 0))
    print("Gravity (MuJoCo opt):", model.opt.gravity)
    print("============================\n")

    # -------------------------------------------------
    # Controller: phase-shifted sinusoids
    # -------------------------------------------------
    N_ACT = model.nu
    freq = 1.2
    amp = 1.2
    phase_offsets = np.linspace(0.0, np.pi, max(1, N_ACT), endpoint=False)

    start_root = np.copy(data.xipos[0])
    start_com = np.copy(data.subtree_com[0])
    last_print_t = -1.0

    # -------------------------------------------------
    # Launch viewer (manual open-loop run)
    # -------------------------------------------------
    print("Launching viewer — Gecko should visibly wiggle. Press ESC to quit.\n")
    v = viewer.launch_passive(model, data)
    v.cam.lookat[:] = [0.0, 0.0, 0.0]
    v.cam.distance = world.radius * 3.0
    v.cam.elevation = -20
    v.cam.azimuth = 120

    while v.is_running():
        t = data.time
        ctrl = amp * np.sin(2.0 * np.pi * freq * t + phase_offsets)

        # Clip to actuator range if available
        lo = model.actuator_ctrlrange[:N_ACT, 0]
        hi = model.actuator_ctrlrange[:N_ACT, 1]
        valid_mask = lo < hi
        if np.any(valid_mask):
            ctrl = np.clip(ctrl, lo, hi)
        data.ctrl[:] = ctrl

        world.step(data)

        if t - last_print_t >= 0.25:
            root_disp = float(np.linalg.norm(data.xipos[0][:2] - start_root[:2]))
            com_disp = float(np.linalg.norm(data.subtree_com[0][:2] - start_com[:2]))
            print(f"t={t:6.2f}s | contacts={data.ncon:2d} | root_dxy={root_disp:.4f} | com_dxy={com_disp:.4f}")
            last_print_t = t

        v.sync()

    v.close()  # ensure proper cleanup

    final_root_disp = float(np.linalg.norm(data.xipos[0][:2] - start_root[:2]))
    final_com_disp = float(np.linalg.norm(data.subtree_com[0][:2] - start_com[:2]))
    print("\nSimulation ended.")
    print(f"Final root XY displacement: {final_root_disp:.4f} m")
    print(f"Final CoM  XY displacement: {final_com_disp:.4f} m")


if __name__ == "__main__":
    main()
