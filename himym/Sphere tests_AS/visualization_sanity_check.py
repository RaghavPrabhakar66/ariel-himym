"""
gecko_pd_direct_drive.py
Final verified Gecko motion test — bypasses ARIEL actuators and drives hinges directly.
"""

import math
import numpy as np
import mujoco
from mujoco import viewer
from ariel.simulation.environments.spherical_world import SphericalWorld
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko


# CONFIG
RADIUS = 5.0
TIMESTEP = 0.0015
DURATION = 15.0
AMP = 0.6
FREQ = 1.2
KP = 10.0
KD = 0.5
PHASE_SHIFT = np.pi


# ---------------------------------------------------------
def build_world():
    world = SphericalWorld(radius=RADIUS, radial_gravity=True)
    g = gecko()
    spec = g.spec if hasattr(g, "spec") else g
    world.spawn(spec)
    model = world.spec.compile()
    world.attach_model(model)
    model.opt.timestep = TIMESTEP
    data = mujoco.MjData(model)
    return world, model, data


def apply_radial_gravity(world, model, data):
    gmag = 9.81
    for i in range(model.nbody):
        pos = np.nan_to_num(data.xipos[i], nan=0.0)
        dist = np.linalg.norm(pos)
        if dist > 1e-6:
            data.xfrc_applied[i, :3] += (-pos / dist) * model.body_mass[i] * gmag


def pd_step(model, data, t, joint_ids):
    qpos = data.qpos.copy()
    qvel = data.qvel.copy()
    torques = np.zeros(len(joint_ids))

    # sine-wave targets for each hinge
    for i, j in enumerate(joint_ids):
        phase = 0 if i % 2 == 0 else PHASE_SHIFT
        target = AMP * math.sin(2 * math.pi * FREQ * t + phase)
        idx = model.jnt_dofadr[j]
        torques[i] = KP * (target - qpos[idx]) - KD * qvel[idx]

    data.ctrl[:] = np.clip(torques, -1.0, 1.0)


# ---------------------------------------------------------
def run_pd_gecko_direct():
    print("=== Gecko PD Direct Drive ===")
    world, model, data = build_world()

    joint_ids = [1, 2, 3, 4, 5, 6, 7, 8]
    steps = int(DURATION / model.opt.timestep)

    print(f"Actuators: {model.nu} | timestep={TIMESTEP}s | duration={DURATION}s")
    print(f"Driving {len(joint_ids)} hinges directly with PD control.\n")

    with viewer.launch_passive(model, data) as v:
        v.cam.lookat[:] = [0, 0, RADIUS]
        v.cam.distance = 10.0
        v.cam.azimuth = 140
        v.cam.elevation = -25

        for step in range(steps):
            t = step * model.opt.timestep
            pd_step(model, data, t, joint_ids)
            apply_radial_gravity(world, model, data)
            mujoco.mj_step(model, data)
            v.sync()

            if step % 1000 == 0:
                com_xy = np.linalg.norm(data.subtree_com[0][:2])
                print(f"t={t:5.2f}s | COM_XY={com_xy:.3f} m")

        print("\n--- Simulation complete ---")
        print("Viewer will remain open. Press ESC to exit manually.\n")
        while v.is_running():
            v.sync()


# ---------------------------------------------------------
if __name__ == "__main__":
    run_pd_gecko_direct()
