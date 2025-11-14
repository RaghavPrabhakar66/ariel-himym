"""
Gecko Neural Controller Prototype (Stage 1)
-------------------------------------------
A single Gecko controlled by a random neural network.

Goal:
- Verify that a feed-forward neural controller produces physical motion.
- No evolution yet — just one randomly wired Gecko "brain".

Inputs  : sin(time), cos(time), joint angles (qpos), joint velocities (qvel), noise
Outputs : actuator target positions (−1.5 .. 1.5 rad)
"""

import mujoco
from mujoco import viewer
import numpy as np
import os

# --- Local imports ---
from ariel.simulation.environments.spherical_world import SphericalWorld
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko

# =====================================================
# WORLD & MODEL SETUP
# =====================================================

# os.environ["MUJOCO_GL"] = "egl"   # uncomment if running headless
world = SphericalWorld(radius=5.0, radial_gravity=True)

g = gecko()
gecko_spec = g.spec if hasattr(g, "spec") else g
world.spawn(gecko_spec, spawn_position=[0, 0, world.radius + 0.05])

model = world.spec.compile()
world.attach_model(model)
model.opt.timestep = 0.005
model.opt.iterations = 50
dt = model.opt.timestep
N_ACT = model.nu

# slightly higher traction
model.geom_friction[:] = np.array([1.5, 0.1, 0.01])

# =====================================================
# HELPERS
# =====================================================

def safe_apply_radial_gravity(world, data):
    """Apply radial gravity safely to every body."""
    gmag = abs(world.gravity)
    for i in range(world.model.nbody):
        pos = np.nan_to_num(data.xipos[i], nan=0.0)
        dist = np.linalg.norm(pos)
        if dist < 1e-6 or not np.isfinite(dist):
            continue
        direction = -pos / dist
        data.xfrc_applied[i, :3] += direction * world.model.body_mass[i] * gmag


# =====================================================
# SIMPLE NEURAL CONTROLLER
# =====================================================

class NeuralController:
    """2-layer tanh MLP controller: inputs→hidden→outputs."""
    def __init__(self, in_size, hid_size, out_size):
        self.in_size = in_size
        self.hid_size = hid_size
        self.out_size = out_size

        # Gaussian init; small variance to keep it stable
        self.W1 = np.random.randn(in_size, hid_size) * 0.4
        self.b1 = np.random.randn(hid_size) * 0.1
        self.W2 = np.random.randn(hid_size, out_size) * 0.4
        self.b2 = np.random.randn(out_size) * 0.1

    def forward(self, x):
        """Forward pass (expects shape [in_size,])."""
        h = np.tanh(x @ self.W1 + self.b1)
        y = np.tanh(h @ self.W2 + self.b2)
        return y  # range [-1, 1]


# =====================================================
# CONTROLLER INITIALIZATION
# =====================================================

# Correct input dimensionality:
#  sin(t), cos(t) → 2
#  qpos → nq
#  qvel → nv
#  noise → 1
obs_size = 2 + model.nq + model.nv + 1
hid_size = 12
ctrl_size = N_ACT

controller = NeuralController(obs_size, hid_size, ctrl_size)

# =====================================================
# SIMULATION
# =====================================================

data = mujoco.MjData(model)
steps = 4000  # ≈20 s

start_com = np.copy(data.subtree_com[0])

print("\nLaunching Gecko with random neural controller…")
print("Close viewer manually when finished observing.\n")

with viewer.launch_passive(model, data) as v:
    v.cam.lookat[:] = [0, 0, 0]
    v.cam.distance = world.radius * 3.0
    v.cam.elevation = -20
    v.cam.azimuth = 120

    while v.is_running():
        if data.time >= steps * dt:
            # Stop stepping but keep viewer open
            v.sync()
            continue

        t = data.time
        # Observation vector
        obs = np.concatenate([
            [np.sin(t), np.cos(t)],
            np.copy(data.qpos),
            np.copy(data.qvel),
            [np.random.uniform(-1, 1)],
        ])

        # Neural output → actuator targets
        ctrl_out = controller.forward(obs)
        ctrl_scaled = 1.5 * ctrl_out
        data.ctrl[:] = np.clip(ctrl_scaled[:N_ACT], -1.5, 1.5)

        # Physics step
        safe_apply_radial_gravity(world, data)
        mujoco.mj_step(model, data)
        v.sync()

end_com = np.copy(data.subtree_com[0])
disp = np.linalg.norm(end_com[:2] - start_com[:2])
print(f"Final CoM displacement: {disp:.4f} m")
