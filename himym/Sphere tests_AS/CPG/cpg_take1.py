import mujoco
import numpy as np

from ariel.simulation.environments.spherical_world import SphericalWorld
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko


# ============================================================
#  VIEWER: Replays EXACT same movement using SAME CPG params
# ============================================================
def replay_gecko_motion(model, world, amp, freq, phase, runtime=15.0):
    """
    Replay the exact CPG motion indefinitely until the user presses ESC.
    """
    from mujoco import viewer

    data = mujoco.MjData(model)
    dt = model.opt.timestep

    print("\n=== VIEWER OPEN — Press ESC to close manually ===")

    with viewer.launch_passive(model, data) as v:
        # Camera setup
        v.cam.lookat[:] = [0, 0, world.radius]
        v.cam.distance = world.radius * 3.0
        v.cam.azimuth = 135
        v.cam.elevation = -25

        while v.is_running():     # ⬅⬅⬅ Viewer stays alive FOREVER
            t = data.time

            # EXACT CPG CONTROL
            for j in range(model.nu):
                ctrl = amp[j] * np.sin(freq[j] * t + phase[j])
                data.ctrl[j] = np.clip(ctrl, -1.5, 1.5)

            # Physics step
            world.step(data)

            # Update display
            v.sync()

    print("[VIEW] Viewer closed by user.")

# ============================================================
#  MAIN: Run CPG demo + print logs + record parameters
# ============================================================
def run_single_gecko_cpg_on_sphere(sim_time: float = 10.0):
    # 1) Build spherical world and spawn ONE gecko
    world = SphericalWorld(radius=10.0, radial_gravity=True)

    g = gecko()
    robot_spec = g.spec if hasattr(g, "spec") else g
    world.spawn(robot_spec, prefix_id=0)

    # 2) Compile model and data
    model = world.spec.compile()
    data = mujoco.MjData(model)
    world.attach_model(model)

    print(f"[INFO] model.nq={model.nq}, nv={model.nv}, nu={model.nu}")

    if model.nu == 0:
        raise RuntimeError("Model has nu == 0 (no actuators). Can't run CPG.")

    # 3) CPG PARAMETERS — KEEPING YOUR EXACT STRUCTURE
    num_joints = model.nu
    amp = np.random.uniform(0.3, 0.7, size=num_joints)
    freq = np.random.uniform(0.5, 2.0, size=num_joints)
    phase = np.random.uniform(0.0, 2*np.pi, size=num_joints)

    print(f"[INFO] Using {num_joints} CPG-controlled joints")

    # 4) Simulation loop
    dt = model.opt.timestep
    steps = int(sim_time / dt)
    print(f"[INFO] Running for {sim_time}s → {steps} steps (dt={dt})")

    start_com = np.copy(data.subtree_com[0])

    for step in range(steps):
        t = data.time

        # === EXACT SAME CPG LOGIC ===
        for j in range(num_joints):
            control_value = amp[j] * np.sin(freq[j] * t + phase[j])
            data.ctrl[j] = np.clip(control_value, -1.5, 1.5)

        world.step(data)

        if step % 200 == 0:
            com = np.copy(data.subtree_com[0])
            print(f"t={t:5.2f}  subtree_com[0]={com}")

    end_com = np.copy(data.subtree_com[0])
    disp = np.linalg.norm(end_com[:2] - start_com[:2])
    print(f"[INFO] Approx XY displacement over {sim_time}s: {disp:.3f} m")

    # RETURN CPG parameters so we can replay the motion EXACTLY
    return model, world, amp, freq, phase


# ============================================================
#  MAIN ENTRY — RUN + REPLAY
# ============================================================
if __name__ == "__main__":
    model, world, amp, freq, phase = run_single_gecko_cpg_on_sphere(sim_time=10.0)

    # Now open a viewer to replay EXACT movement
    replay_gecko_motion(model, world, amp, freq, phase, runtime=15.0)
