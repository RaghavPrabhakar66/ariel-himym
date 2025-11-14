"""
Test script for SphericalWorld + Gecko robot in ARIEL.
Opens an interactive MuJoCo viewer to visualize the spherical world.
Press ESC to quit the viewer.
"""

import mujoco
from mujoco import viewer
import numpy as np
import os

# --- Local imports ---
from ariel.simulation.environments.spherical_world import SphericalWorld
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko

# --- Optional EGL/ANGLE setup for headless/Windows ---
# os.environ["MUJOCO_GL"] = "egl"   # for Linux headless
# os.environ["MUJOCO_GL"] = "angle" # on Windows if you get black screen

# -----------------------------
# Build world + robot
# -----------------------------
light_settings = {
    "pos": [0, 0, 15],
    "dir": [0, 0, -1],
    "diffuse": [2.0, 2.0, 2.0],
    "specular": [1.0, 1.0, 1.0],
    "ambient": [0.3, 0.3, 0.3],
    "cutoff": 100,
    "exponent": 3,
}

# Build world
world = SphericalWorld(
    radius=5.0,
    rgba=(0.6, 0.8, 1.0, 1.0),
    light_settings=light_settings
)

# --- Secondary fill light (underneath the planet) ---
world.spec.worldbody.add_light(
    name="fill_light",
    pos=[0, 0, -15],
    dir=[0, 0, 1],
    diffuse=[1.5, 1.5, 1.5],
    specular=[0.4, 0.4, 0.4],
    cutoff=100,
    exponent=4,
    castshadow=False,
)

# Instantiate Gecko
gecko_obj = gecko()
gecko_spec = gecko_obj.spec if hasattr(gecko_obj, "spec") else gecko_obj

# Spawn Gecko slightly above the sphere
world.spawn(gecko_spec, spawn_position=[0, 0, world.radius + 0.05])

# Compile the world to a MuJoCo model
model = world.spec.compile()
world.model = model
data = mujoco.MjData(model)

print("Opening viewer... (press ESC to quit)")

# -----------------------------
# Launch interactive viewer
# -----------------------------
with viewer.launch_passive(model, data) as v:
    # Move the camera outside the sphere
    v.cam.lookat[:] = [0, 0, 0]
    v.cam.distance = world.radius * 3.0   # pull camera back
    v.cam.elevation = -20
    v.cam.azimuth = 120

    # Run simulation until you close viewer (ESC)
    while v.is_running():
        world.apply_radial_gravity(data)
        mujoco.mj_step(model, data)
        v.sync()
