# sanity_check_gecko.py

import time
import numpy as np
import mujoco
from mujoco import viewer

from ariel.simulation.environments import SimpleFlatWorld
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.utils.runners import simple_runner


def map_to_ctrlrange(model, u):
    u = np.clip(u, -1.0, 1.0)
    low = model.actuator_ctrlrange[:, 0]
    high = model.actuator_ctrlrange[:, 1]
    return low + (u + 1.0) * 0.5 * (high - low)


def build_gecko_world():
    world = SimpleFlatWorld()
    g = gecko()
    world.spawn(g.spec, spawn_position=[0, 0, 0])
    model = world.spec.compile()
    data = mujoco.MjData(model)
    return model, data


def print_actuator_mapping(model):
    print("\n=== Actuator → Joint Mapping ===")
    for i in range(model.nu):
        jid = model.actuator_trnid[i, 0]
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        rng = model.actuator_ctrlrange[i]
        print(f"act[{i}] -> joint[{jname}], range={rng}")
    print("===============================\n")


def actuator_smoke_test(model, data, seconds=2.0):
    print(">>> Running MAX-RANGE actuator test (robot SHOULD MOVE)...")

    low = model.actuator_ctrlrange[:, 0]
    high = model.actuator_ctrlrange[:, 1]

    # push joints toward high limit
    data.ctrl[:] = high

    simple_runner(model=model, data=data, duration=seconds)

    mujoco.mj_resetData(model, data)
    print("✔ Smoke test finished.\n")


def sine_controller(model, data):
    t = data.time
    n = model.nu
    phases = np.linspace(0, np.pi, n, endpoint=False)
    u = np.sin(2.0 * t + phases)  # [-1,1]
    data.ctrl[:] = map_to_ctrlrange(model, u)


def run_sine_and_view(model, data, seconds=4.0):
    print(">>> Running simple sine gait for 4 seconds...")
    mujoco.set_mjcb_control(lambda m, d: sine_controller(m, d))

    simple_runner(model=model, data=data, duration=seconds)
    mujoco.mj_resetData(model, data)

    print(">>> Launching viewer (be sure to press SPACE to start sim)")
    viewer.launch(model, data)

    mujoco.set_mjcb_control(None)
    print("✔ Viewer closed.\n")


def main():
    model, data = build_gecko_world()

    print_actuator_mapping(model)

    actuator_smoke_test(model, data, seconds=2.0)

    run_sine_and_view(model, data, seconds=4.0)

    print("✅ Sanity check complete.")


if __name__ == "__main__":
    main()
