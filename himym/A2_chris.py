"""
Gecko Sanity Diagnostic (ARIEL + MuJoCo)
-----------------------------------------
Checks whether actuators, gravity, and contacts are functioning properly
for the Gecko robot on a spherical world.
Records a video and prints motion diagnostics.
"""

import mujoco
from mujoco import viewer
import numpy as np
import json, os
from datetime import datetime

# --- ARIEL imports ---
from ariel.simulation.environments.spherical_world import SphericalWorld
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.utils.video_recorder import VideoRecorder
from ariel.utils.renderers import video_renderer


# =====================================================
# SETUP
# =====================================================

world = SphericalWorld(radius=5.0, radial_gravity=True)
g = gecko()
gecko_spec = g.spec if hasattr(g, "spec") else g
world.spawn(gecko_spec, spawn_position=[0, 0, world.radius + 0.05])

model = world.spec.compile()
world.attach_model(model)
data = mujoco.MjData(model)

# --- Physics tuning ---
model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICIT
model.opt.solver = mujoco.mjtSolver.mjSOL_CG
model.opt.timestep = 0.002
model.opt.iterations = 80
model.opt.ls_iterations = 50
model.opt.tolerance = 1e-10
model.dof_damping[:] = 0.2
model.geom_friction[:] = np.array([1.0, 0.05, 0.01])
model.actuator_ctrlrange[:] = np.array([[-1.0, 1.0]] * model.nu)

print("\n=== DIAGNOSTIC SETUP ===")
print(f"Actuators: {model.nu}")
print(f"Joints (qpos): {model.nq} | Velocities (qvel): {model.nv}")
print(f"Gravity magnitude: {abs(world.gravity):.2f} | timestep: {model.opt.timestep}")
print("=========================\n")


# =====================================================
# UTILITY FUNCTIONS
# =====================================================

def apply_radial_gravity():
    """Apply gravity toward sphere center (safe version)."""
    gmag = abs(world.gravity)
    for i in range(model.nbody):
        pos = data.xipos[i]
        if not np.all(np.isfinite(pos)):
            continue
        dist = np.linalg.norm(pos)
        if dist < 1e-6:
            continue
        direction = -pos / dist
        data.xfrc_applied[i, :3] += direction * model.body_mass[i] * gmag


# =====================================================
# SIMULATION LOOP
# =====================================================

STEPS = 3000
dt = model.opt.timestep

energy_trace, com_trace = [], []
contact_count = 0

print("Running diagnostic simulation (no recording yet)...")

for step in range(STEPS):
    t = step * dt
    # Apply sinusoidal torque pattern to all actuators
    data.ctrl[:] = np.sin(t * 6.0 + np.arange(model.nu)) * 0.7

    apply_radial_gravity()
    mujoco.mj_step(model, data)

    # Log motion every few frames
    if step % 50 == 0:
        com = np.copy(data.subtree_com[0])
        com_trace.append(com.tolist())
        total_energy = np.sum(np.square(data.qvel)) + np.sum(np.abs(data.qpos))
        energy_trace.append(float(total_energy))
        contact_count += len(data.contact)

# =====================================================
# POST-SIM DIAGNOSTICS
# =====================================================

end_com = np.array(com_trace[-1])
start_com = np.array(com_trace[0])
disp = np.linalg.norm(end_com - start_com)
vel_var = np.var(data.qvel)
acc_var = np.var(data.qacc)
energy_change = energy_trace[-1] - energy_trace[0]
avg_contacts = contact_count / max(1, len(com_trace))

print("\n=== DIAGNOSTIC SUMMARY ===")
print(f"Total CoM displacement: {disp:.4f} m")
print(f"Velocity variance:      {vel_var:.6f}")
print(f"Acceleration variance:  {acc_var:.6f}")
print(f"Energy change Δ:        {energy_change:.4f}")
print(f"Avg contact count:      {avg_contacts:.2f}")
print("==========================\n")


# =====================================================
# SAVE JSON REPORT
# =====================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
report = {
    "displacement_m": float(disp),
    "velocity_variance": float(vel_var),
    "acceleration_variance": float(acc_var),
    "energy_delta": float(energy_change),
    "avg_contacts": float(avg_contacts),
    "gravity": float(world.gravity),
    "n_actuators": int(model.nu),
    "timestep": float(model.opt.timestep),
}

json_path = f"gecko_diagnostics_{timestamp}.json"
with open(json_path, "w") as f:
    json.dump(report, f, indent=2)

print(f"📄 JSON report saved: {json_path}")


# =====================================================
# VIDEO RECORDING (USING ARIEL RENDERER)
# =====================================================

print("🎥 Recording 30-second diagnostic video...")

os.makedirs("diagnostic_videos", exist_ok=True)
video_recorder = VideoRecorder(output_folder="diagnostic_videos")
video_renderer(
    model,
    data,
    duration=30,
    video_recorder=video_recorder,
)

print("✅ Video saved under diagnostic_videos/ (auto-named)\n")


# =====================================================
# OPTIONAL VIEWER
# =====================================================

print("Launching interactive viewer (close manually when done)...")
with viewer.launch_passive(model, data) as v:
    v.cam.lookat[:] = [0, 0, 0]
    v.cam.distance = world.radius * 3.0
    v.cam.elevation = -20
    v.cam.azimuth = 120
    while v.is_running():
        v.sync()
