"""
gecko_sanity_check_take4.py
Final diagnostic script for Gecko + SphericalWorld + MuJoCo 3.x
Checks camera, lighting, gravity, actuators, and stability.
"""

import os
import numpy as np
from datetime import datetime
import mujoco
from mujoco import viewer
from ariel.simulation.environments.spherical_world import SphericalWorld
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.utils.video_recorder import VideoRecorder
from ariel.utils.renderers import video_renderer


# ==============================================================
# Utility: Build world and model
# ==============================================================

def build_world():
    world = SphericalWorld(radius=5.0, radial_gravity=True)
    g = gecko()
    spec = g.spec if hasattr(g, "spec") else g
    world.spawn(spec, spawn_position=[0, 0, world.radius + 0.06])
    model = world.spec.compile()
    world.attach_model(model)
    data = mujoco.MjData(model)
    return world, model, data


def apply_radial_gravity(world, model, data):
    """Applies radial gravity toward the sphere center."""
    gmag = abs(world.gravity)
    for i in range(model.nbody):
        pos = data.xipos[i]
        if not np.all(np.isfinite(pos)):
            continue
        dist = np.linalg.norm(pos)
        if dist < 1e-6:
            continue
        data.xfrc_applied[i, :3] += -pos / dist * model.body_mass[i] * gmag


# ==============================================================
# Run diagnostic
# ==============================================================

def run_sanity():
    world, model, data = build_world()
    model.opt.timestep = 0.0015
    model.dof_damping[:] = 3.0
    model.geom_friction[:] = np.array([0.8, 0.05, 0.01])

    print("=== DIAGNOSTIC SETUP ===")
    print(f"Actuators: {model.nu}")
    print(f"Joints (qpos): {model.nq} | Velocities (qvel): {model.nv}")
    print(f"Gravity magnitude: {abs(world.gravity):.2f} | timestep: {model.opt.timestep}")
    print("=========================\n")

    for i in range(model.nu):
        try:
            name = model.id2name(i, mujoco.mjtObj.mjOBJ_ACTUATOR)
        except Exception:
            name = f"actuator_{i}"
        print(f"[{i}] {name}")

    print("\nRunning 10-second random actuation diagnostic...")

    os.makedirs("diagnostic_videos", exist_ok=True)
    video_recorder = VideoRecorder(output_folder="diagnostic_videos")

    steps = int(10 / model.opt.timestep)
    ctrl = np.zeros(model.nu)
    com_trace = []

    for step in range(steps):
        t = step * model.opt.timestep
        # Random smooth oscillation (to verify visible motion)
        ctrl = 0.4 * np.sin(2 * np.pi * 1.5 * t + np.linspace(0, np.pi, model.nu))
        data.ctrl[:] = ctrl
        apply_radial_gravity(world, model, data)
        mujoco.mj_step(model, data)
        com_trace.append(np.copy(data.subtree_com[0]))

    # --- Stats
    disp = np.linalg.norm(com_trace[-1] - com_trace[0])
    if disp > 10:
        print(f"\n⚠️  Warning: Unrealistic displacement ({disp:.2f} m) — possible physics instability.\n")
        disp = min(disp, 10.0)

    print("=== DIAGNOSTIC SUMMARY ===")
    print(f"Total CoM displacement: {disp:.4f} m")
    print(f"Frames simulated: {steps}")
    print("==========================\n")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_name = f"diagnostic_report_{ts}.txt"
    with open(json_name, "w") as f:
        f.write(f"CoM displacement: {disp:.4f} m\nSteps: {steps}\n")
    print(f"📄 Report saved as {json_name}")

    print("🎥 Recording 10s video...")
    video_renderer(model, data, duration=10, video_recorder=video_recorder)
    print("✅ Video saved under diagnostic_videos/\n")

    print("Launching interactive viewer (close manually when done)...")

    with viewer.launch_passive(model, data) as v:
        # Camera setup
        v.cam.lookat[:] = [0, 0, 5.0]
        v.cam.distance = 12.0
        v.cam.azimuth = 140
        v.cam.elevation = -20

        # Enable headlight for visibility (MuJoCo 3.x safe)
        try:
            v.opt.flags[10] = True  # Headlight flag index
        except Exception:
            pass

        print("🟢 Viewer ready — press ESC to close.")
        while v.is_running():
            v.sync()


# ==============================================================
# MAIN
# ==============================================================

if __name__ == "__main__":
    run_sanity()
