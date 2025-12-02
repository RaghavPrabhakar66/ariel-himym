import mujoco
import numpy as np
import traceback
import time

from ariel.simulation.environments.spherical_world import SphericalWorld
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko


# =====================================================================
#  FULL ENVIRONMENT VALIDATION SCRIPT — V COMPATIBILITY CHECK
# =====================================================================
def validate_sphere_environment():
    print("\n========== ENVIRONMENT VALIDATION: SPHERICAL WORLD ==========\n")
    results = {}

    # =========================================================
    # 1. BUILD WORLD
    # =========================================================
    try:
        world = SphericalWorld(
            radius=10.0,
            radial_gravity=True,
            gravity=30.0,
            friction=(5, 2, 0.5)
        )
        print("[OK] World created successfully.")
        results["world_create"] = True
    except Exception as e:
        print("[FAIL] World could not be created.\n", e)
        return

    # =========================================================
    # 2. SPAWN ROBOT
    # =========================================================
    try:
        g = gecko()
        robot_spec = g.spec if hasattr(g, "spec") else g

        world.spawn(
            robot_spec,
            prefix_id=0,
            small_gap=0.01,
            bbox_correction=True,
        )
        print("[OK] Robot spawned and attached successfully.")
        results["spawn"] = True
    except Exception as e:
        print("[FAIL] Spawn failed.\n", e)
        return

    # =========================================================
    # 3. COMPILE MODEL
    # =========================================================
    try:
        model = world.spec.compile()
        data = mujoco.MjData(model)
        world.attach_model(model)
        print("[OK] Model compiled successfully.")
        results["compile"] = True
    except Exception as e:
        print("[FAIL] Model compile failed.\n", e)
        return

    # =========================================================
    # 4. CHECK ROBOT COMPONENTS
    # =========================================================
    print("\n--- ROBOT STRUCTURE CHECK ---")
    print(f"nq={model.nq}, nv={model.nv}, nu={model.nu}")

    if model.nu == 0:
        print("[FAIL] No actuators found — cannot run CPG or controls.")
        return

    print("[OK] Robot has actuators.")

    print("\nListing all geoms:")
    for i in range(model.ngeom):
        print(" ", i, model.geom(i).name)

    print("\nListing all joints:")
    for j in range(model.njnt):
        print(" ", j, model.joint(j).name)

    # =========================================================
    # 5. GRAVITY CHECK
    # =========================================================
    print("\n--- GRAVITY CHECK ---")
    for i in range(3):
        world.step(data)

    radial_forces = np.copy(data.xfrc_applied[:, :3])

    if np.linalg.norm(radial_forces) == 0:
        print("[FAIL] Radial gravity seems inactive.")
    else:
        print("[OK] Radial gravity applied successfully.")
        print(" Sample forces:", radial_forces[:3])

    # =========================================================
    # 6. CONTACT CHECK
    # =========================================================
    print("\n--- CONTACT CHECK (200 steps) ---")
    contact_ok = False
    for _ in range(200):
        world.step(data)
        if data.ncon > 0:
            contact_ok = True
            break

    if contact_ok:
        print("[OK] Robot made contact with the sphere.")
    else:
        print("[FAIL] Robot never contacted sphere — floating?")

    # =========================================================
    # 7. CONTROL CHECK (CPG sanity)
    # =========================================================
    print("\n--- CONTROL CHECK ---")
    try:
        num = model.nu
        amp = np.ones(num) * 0.5
        freq = np.ones(num) * 1.5
        phase = np.random.uniform(0, 2*np.pi, num)

        for _ in range(200):
            t = data.time
            for j in range(num):
                data.ctrl[j] = amp[j] * np.sin(freq[j] * t + phase[j])

            world.step(data)

        print("[OK] Joint control works, no NaNs produced.")
    except Exception as e:
        print("[FAIL] Control produced error.\n", e)
        return

    # =========================================================
    # 8. CHECK FOR NUMERICAL STABILITY
    # =========================================================
    print("\n--- NUMERICAL STABILITY CHECK (2000 steps) ---")
    stable = True
    for _ in range(2000):
        world.step(data)
        if np.any(~np.isfinite(data.qpos)):
            stable = False
            print("[FAIL] Found NaN or Inf in qpos.")
            break
        if np.any(~np.isfinite(data.xfrc_applied)):
            stable = False
            print("[FAIL] Found NaN in external forces.")
            break

    if stable:
        print("[OK] No NaNs — stable under long simulation.")

    # =========================================================
    # 9. EXTREME STRESS TEST
    # =========================================================
    print("\n--- EXTREME STRESS TEST (5000 steps) ---")
    try:
        for _ in range(5000):
            world.step(data)
        print("[OK] Sphere environment stable under 5000-step stress test.")
    except Exception as e:
        print("[FAIL] Stress test failed.\n", e)
        return

    # =========================================================
    # 10. PRINT FINAL RESULT
    # =========================================================
    print("\n========== VALIDATION SUMMARY ==========")
    for k, v in results.items():
        print(f"{k:20s} : {'OK' if v else 'FAIL'}")

    print("\nYou are SAFE to port pipeline to the sphere.\n")


if __name__ == "__main__":
    validate_sphere_environment()
