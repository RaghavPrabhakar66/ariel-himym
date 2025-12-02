import mujoco
import numpy as np

from ariel.simulation.environments.spherical_world import SphericalWorld
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko


# ============================================================
#  UTILS
# ============================================================

def print_actuators(model: mujoco.MjModel) -> None:
    print("\n--- ACTUATORS ---")
    from mujoco import mjtObj
    for i in range(model.nu):
        raw = mujoco.mj_id2name(model, mjtObj.mjOBJ_ACTUATOR, i)
        name = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        print(f"  {i}: {name}")
    print("-----------------\n")


def get_core_geom_id(model: mujoco.MjModel) -> int:
    """
    Try to find the gecko's core geom by name.
    Falls back gracefully if not found.
    """
    from mujoco import mjtObj
    # This name exists in your earlier printout
    gid = mujoco.mj_name2id(model, mjtObj.mjOBJ_GEOM, "robot_0core")
    if gid == -1:
        # fallback: first non-planet geom
        for i in range(model.ngeom):
            raw = mujoco.mj_id2name(model, mjtObj.mjOBJ_GEOM, i)
            name = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            if "planet" not in name:
                return i
        # worst case: 0
        return 0
    return gid


def build_simple_gecko_gait(model: mujoco.MjModel):
    """
    Same gait as before, but separated here for clarity.
    """

    from mujoco import mjtObj

    nu = model.nu
    groups = {
        "fl_leg": [],
        "fr_leg": [],
        "bl_leg": [],
        "br_leg": [],
        "fl_flip": [],
        "fr_flip": [],
        "other": []
    }

    for i in range(nu):
        raw = mujoco.mj_id2name(model, mjtObj.mjOBJ_ACTUATOR, i)
        name = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        lname = name.lower()

        if "fl_leg" in lname:
            groups["fl_leg"].append(i)
        elif "fr_leg" in lname:
            groups["fr_leg"].append(i)
        elif "bl_leg" in lname:
            groups["bl_leg"].append(i)
        elif "br_leg" in lname:
            groups["br_leg"].append(i)
        elif "fl_flipper" in lname:
            groups["fl_flip"].append(i)
        elif "fr_flipper" in lname:
            groups["fr_flip"].append(i)
        else:
            groups["other"].append(i)

    print("Actuator grouping for gait:")
    for k, v in groups.items():
        print(f"  {k}: {v}")

    leg_amp = 0.7
    flip_amp = 0.4
    omega = 2.0
    omega_flip = 3.0

    def controller(model: mujoco.MjModel, data: mujoco.MjData):
        t = data.time
        data.ctrl[:] = 0.0

        # Diagonal pair 1
        for idx in groups["fl_leg"]:
            data.ctrl[idx] = leg_amp * np.sin(omega * t)
        for idx in groups["br_leg"]:
            data.ctrl[idx] = leg_amp * np.sin(omega * t)

        # Diagonal pair 2 (π phase shift)
        for idx in groups["fr_leg"]:
            data.ctrl[idx] = leg_amp * np.sin(omega * t + np.pi)
        for idx in groups["bl_leg"]:
            data.ctrl[idx] = leg_amp * np.sin(omega * t + np.pi)

        # Flippers
        for idx in groups["fl_flip"]:
            data.ctrl[idx] = flip_amp * np.sin(omega_flip * t)
        for idx in groups["fr_flip"]:
            data.ctrl[idx] = flip_amp * np.sin(omega_flip * t + np.pi)

    return controller


# ============================================================
#  MAIN WALK DEMO WITH FOLLOW-CAM
# ============================================================

def run_gecko_walk_on_sphere(sim_time: float = 20.0):
    print("\n========== G E C K O   W A L K   O N   S P H E R E ==========")

    # 1) World with strong gravity + high friction
    world = SphericalWorld(
        radius=10.0,
        friction=(20.0, 8.0, 2.0),
        gravity=40.0,
        radial_gravity=True,
    )

    # 2) Gecko spec
    g = gecko()
    robot_spec = g.spec if hasattr(g, "spec") else g

    print("[INFO] Spawning gecko...")
    world.spawn(robot_spec, prefix_id=0)

    # 3) Compile & attach
    print("[INFO] Compiling model...")
    model = world.spec.compile()
    data = mujoco.MjData(model)
    world.attach_model(model)

    print(f"[INFO] model.nq={model.nq}, nv={model.nv}, nu={model.nu}")
    if model.nu == 0:
        raise RuntimeError("Model has nu==0, no actuators.")

    print_actuators(model)

    # 4) Let the gecko settle physically on the sphere
    print("[INFO] Settling onto sphere...")
    world.warm_start_settle(data, steps=1000)

    # 5) Find core geom for tracking + camera
    core_gid = get_core_geom_id(model)
    print(f"[INFO] Using geom {core_gid} as core tracker.")

    def core_pos():
        return np.copy(data.geom_xpos[core_gid])

    start_pos = core_pos()
    start_dir = start_pos / np.linalg.norm(start_pos)
    radius = np.linalg.norm(start_pos)
    print(f"[INFO] Start core pos: {start_pos}, |r|={radius:.4f}")

    # 6) Gait controller
    controller = build_simple_gecko_gait(model)

    # 7) Viewer with follow-camera
    from mujoco import viewer
    print("[INFO] Launching viewer — press ESC to close when done.")

    dt = model.opt.timestep
    total_steps = int(sim_time / dt)

    with viewer.launch_passive(model, data) as v:
        step = 0
        while v.is_running() and step < total_steps:
            # Apply gait
            controller(model, data)

            # Physics step
            world.step(data)

            # Follow the gecko core
            cpos = core_pos()
            v.cam.lookat[:] = cpos
            v.cam.distance = world.radius * 1.5  # closer
            v.cam.elevation = -20
            v.cam.azimuth = 120

            if step % 200 == 0:
                print(f"  t={data.time:5.2f}  core={cpos}")

            v.sync()
            step += 1

    # 8) Displacement on sphere based on core position
    end_pos = core_pos()
    end_dir = end_pos / np.linalg.norm(end_pos)

    dot = float(np.clip(np.dot(start_dir, end_dir), -1.0, 1.0))
    angle = np.arccos(dot)
    arc_length = radius * angle

    print("\n========== W A L K   R E S U L T S ==========")
    print(f"Start core: {start_pos}")
    print(f"End   core: {end_pos}")
    print(f"Angle between start/end dirs: {np.degrees(angle):.4f} deg")
    print(f"Approx. geodesic distance along sphere: {arc_length:.4f} m")
    print("==============================================")


if __name__ == "__main__":
    run_gecko_walk_on_sphere(sim_time=20.0)
