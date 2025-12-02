"""
Gecko Sanity Diagnostic (ARIEL + Robogen-Lite)
----------------------------------------------
Tests all hinge actuators by driving them sinusoidally.
Prints actuator list, movement metrics, and records a video.
"""

import mujoco
from mujoco import viewer
import numpy as np
import os, json
from datetime import datetime

from ariel.simulation.environments.spherical_world import SphericalWorld
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.utils.video_recorder import VideoRecorder
from ariel.utils.renderers import video_renderer


# =====================================================
# WORLD + ROBOT SETUP
# =====================================================

world = SphericalWorld(radius=5.0, radial_gravity=True)
gecko_obj = gecko()
spec = gecko_obj.spec if hasattr(gecko_obj, "spec") else gecko_obj
world.spawn(spec, spawn_position=[0, 0, world.radius + 0.05])

model = world.spec.compile()
world.attach_model(model)
data = mujoco.MjData(model)

model.opt.timestep = 0.002
model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICIT
model.dof_damping[:] = 0.2
model.geom_friction[:] = np.array([1.0, 0.05, 0.01])
model.opt.gravity[:] = [0, 0, -9.81]

print("\n=== DIAGNOSTIC SETUP ===")
print(f"Actuators: {model.nu}")
print(f"Joints (qpos): {model.nq} | Velocities (qvel): {model.nv}")
print(f"Gravity magnitude: 9.81 | timestep: {model.opt.timestep}")
print("=========================\n")

# =====================================================
# ACTUATOR NAMES FROM SPEC
# =====================================================

act_names = []
if hasattr(spec, "actuators"):
    for a in spec.actuators:
        act_names.append(a.name if hasattr(a, "name") else "unnamed_actuator")
else:
    act_names = [f"actuator_{i}" for i in range(model.nu)]

print("=== ACTUATOR LIST ===")
for i, n in enumerate(act_names):
    print(f"[{i}] {n}")
print("=====================\n")


# =====================================================
# SAFE RADIAL GRAVITY
# =====================================================

def apply_radial_gravity():
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
# SINUSOIDAL TEST DRIVE
# =====================================================

STEPS = 3000
AMP = 0.8
FREQ = 2.5
dt = model.opt.timestep

energy_trace, com_trace = [], []
contact_count = 0

print("Running diagnostic simulation...")

for step in range(STEPS):
    t = step * dt
    data.ctrl[:] = AMP * np.sin(2 * np.pi * FREQ * t + np.arange(model.nu))
    apply_radial_gravity()
    mujoco.mj_step(model, data)

    if step % 50 == 0:
        com_trace.append(np.copy(data.subtree_com[0]))
        energy_trace.append(np.sum(data.qvel ** 2))
        contact_count += len(data.contact)

end_com = np.array(com_trace[-1])
start_com = np.array(com_trace[0])
disp = np.linalg.norm(end_com - start_com)
vel_var = np.var(data.qvel)
energy_delta = energy_trace[-1] - energy_trace[0]
avg_contacts = contact_count / max(1, len(com_trace))

print("\n=== DIAGNOSTIC SUMMARY ===")
print(f"Total CoM displacement: {disp:.4f} m")
print(f"Velocity variance:      {vel_var:.6f}")
print(f"Energy change Δ:        {energy_delta:.4f}")
print(f"Avg contact count:      {avg_contacts:.2f}")
print("==========================\n")


# =====================================================
# SAVE JSON REPORT
# =====================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
report = {
    "displacement_m": float(disp),
    "velocity_variance": float(vel_var),
    "energy_delta": float(energy_delta),
    "avg_contacts": float(avg_contacts),
    "n_actuators": int(model.nu),
    "timestep": float(model.opt.timestep),
}
json_path = f"gecko_diagnostics_{timestamp}.json"
with open(json_path, "w") as f:
    json.dump(report, f, indent=2)
print(f"📄 JSON report saved: {json_path}")


# =====================================================
# RECORD VIDEO (30 s)
# =====================================================

os.makedirs("diagnostic_videos", exist_ok=True)
print("🎥 Recording 30-second diagnostic video...")

video_recorder = VideoRecorder(output_folder="diagnostic_videos")
video_renderer(model, data, duration=30, video_recorder=video_recorder)
print("✅ Video saved under diagnostic_videos/\n")


# =====================================================
# VIEWER
# =====================================================

print("Launching interactive viewer (close manually)...")
with viewer.launch_passive(model, data) as v:
    v.cam.lookat[:] = [0, 0, 0]
    v.cam.distance = world.radius * 3.0
    v.cam.elevation = -20
    v.cam.azimuth = 120
    while v.is_running():
        v.sync()
