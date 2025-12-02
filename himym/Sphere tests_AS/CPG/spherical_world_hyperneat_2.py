import mujoco
import numpy as np
from mujoco import viewer

from ariel.simulation.environments.spherical_world import SphericalWorld
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko


# --------------------------------------------------------------
#  CORE GEOM AUTO-DETECTION (largest robot geom)
# --------------------------------------------------------------
def auto_select_core_geom(model: mujoco.MjModel) -> int:
    best_gid = -1
    best_score = -1.0

    for gid in range(model.ngeom):
        rawname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        name = rawname.decode() if isinstance(rawname, bytes) else rawname

        if not name:
            continue
        if "planet" in name:
            continue
        if not name.startswith("robot_0"):
            continue

        s = model.geom_size[gid]
        score = float(s[0] + s[1] + s[2])
        if score > best_score:
            best_score = score
            best_gid = gid

    print(f"[INFO] auto-selected core geom id={best_gid}")
    return best_gid


# --------------------------------------------------------------
#  GENTLE, RAMPED CPG CONTROLLER
# --------------------------------------------------------------
def make_cpg_controller(model: mujoco.MjModel):
    """
    Gecko trot gait, but with:
      - smaller amplitudes,
      - smooth ramp-up over first few seconds,
      - hard clipping to actuator_ctrlrange.
    """

    # Actuator indices (based on your earlier dump)
    NECK = 0
    SPINE = 1
    FL_LEG = 2
    FL_FLIP = 3
    FR_LEG = 4
    FR_FLIP = 5
    BL_LEG = 6
    BR_LEG = 7

    # Pull control ranges from model (usually [-1,1], but we don't assume)
    ctrl_low = model.actuator_ctrlrange[:, 0].copy()
    ctrl_high = model.actuator_ctrlrange[:, 1].copy()

    # Gait parameters (gentler than before)
    base_freq = 0.7          # Hz-ish
    leg_amp = 0.3            # was 0.6
    flip_amp = 0.18          # was 0.35
    spine_amp = 0.07         # smaller spine motion
    phase_offset = np.pi     # diagonal trot

    ramp_duration = 3.0      # seconds to go from 0 → full amplitude

    def ctrl(data: mujoco.MjData):
        t = data.time

        # Smooth ramp factor in [0,1]
        ramp = min(1.0, max(0.0, t / ramp_duration))

        phase = 2.0 * np.pi * base_freq * t

        # Legs: diagonal pairs
        fl = leg_amp * np.sin(phase)
        br = leg_amp * np.sin(phase)
        fr = leg_amp * np.sin(phase + phase_offset)
        bl = leg_amp * np.sin(phase + phase_offset)

        # Flippers slightly delayed w.r.t. their leg
        flf = flip_amp * np.sin(phase + 0.3)
        frf = flip_amp * np.sin(phase + phase_offset + 0.3)

        # Spine gentle wave
        spine = spine_amp * np.sin(phase * 0.5)

        # Neck mostly still (you can modulate later)
        neck = 0.0

        # Apply ramp
        u = np.zeros(model.nu)
        u[NECK] = neck
        u[SPINE] = spine
        u[FL_LEG] = fl
        u[FL_FLIP] = flf
        u[FR_LEG] = fr
        u[FR_FLIP] = frf
        u[BL_LEG] = bl
        u[BR_LEG] = br

        u *= ramp

        # Clip against actuator control ranges
        for j in range(model.nu):
            lo, hi = ctrl_low[j], ctrl_high[j]
            data.ctrl[j] = np.clip(u[j], lo, hi)

    return ctrl


# --------------------------------------------------------------
#  SAFE GEODESIC ON SPHERE
# --------------------------------------------------------------
def geodesic_distance_sphere(p1, p2, R: float):
    n1, n2 = np.linalg.norm(p1), np.linalg.norm(p2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0, 0.0

    u, v = p1 / n1, p2 / n2
    dot = np.clip(np.dot(u, v), -1.0, 1.0)
    angle = np.arccos(dot)
    return R * angle, np.degrees(angle)


# --------------------------------------------------------------
#  MAIN: WALK ON SPHERE
# --------------------------------------------------------------
def run_gecko_on_sphere(sim_time: float = 30.0):
    print("\n========== G E C K O   W A L K   O N   S P H E R E ==========")

    # 1) World + robot
    world = SphericalWorld(radius=10.0, radial_gravity=True)

    robot = gecko().spec
    print("[INFO] Spawning gecko...")
    world.spawn(robot, prefix_id=0)

    print("[INFO] Compiling model...")
    model = world.spec.compile()
    data = mujoco.MjData(model)
    world.attach_model(model)

    print(f"[INFO] model.nq={model.nq}, nv={model.nv}, nu={model.nu}")

    # 2) Core geom for tracking
    core_gid = auto_select_core_geom(model)

    def core_pos():
        return data.geom_xpos[core_gid].copy()

    # 3) Let robot settle *without control*
    print("[INFO] Settling (no actuation)...")
    for _ in range(1000):
        world.step(data)

    start = core_pos()
    print(f"[INFO] Start core pos: {start} (|r|={np.linalg.norm(start):.4f})")

    # 4) Build controller
    controller = make_cpg_controller(model)

    # 5) Viewer loop with follow camera
    print("[INFO] Launching viewer — press ESC to close")

    with viewer.launch_passive(model, data) as v:
        v.cam.distance = world.radius * 2.5
        v.cam.elevation = -25
        v.cam.azimuth = 90

        steps = int(sim_time / model.opt.timestep)

        for step in range(steps):
            if not v.is_running():
                break

            # Control + physics
            controller(data)
            world.step(data)

            # Camera follows core geom
            c = core_pos()
            v.cam.lookat[:] = c
            v.cam.distance = max(
                world.radius * 1.6,
                np.linalg.norm(c) + world.radius * 0.4
            )

            v.sync()

            # Log once per second
            if step % int(1.0 / model.opt.timestep) == 0:
                print(f"  t={data.time:5.2f}  core={c}")

    # 6) Results
    end = core_pos()
    dist, angle = geodesic_distance_sphere(start, end, world.radius)

    print("\n========== W A L K   R E S U L T S ==========")
    print(f" Start core: {start}")
    print(f" End   core: {end}")
    print(f" Angle moved: {angle:.2f} deg")
    print(f" Geodesic dist: {dist:.3f} m")
    print("===============================================")


# --------------------------------------------------------------
#  ENTRY POINT
# --------------------------------------------------------------
if __name__ == "__main__":
    run_gecko_on_sphere(sim_time=30.0)
