import mujoco
import numpy as np

from ariel.simulation.environments.spherical_world import SphericalWorld
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko


def pretty_name(model, obj_type, idx):
    """Safe name getter (handles bytes / None)."""
    raw = mujoco.mj_id2name(model, obj_type, idx)
    if raw is None:
        return ""
    return raw.decode() if isinstance(raw, bytes) else raw


def main():
    print("\n========== S P H E R E   +   G E C K O   S A N I T Y ==========\n")

    # ----------------------------------------------------------
    # 1) Build world + spawn gecko
    # ----------------------------------------------------------
    R = 10.0
    print(f"[INFO] Creating SphericalWorld with radius={R} ...")
    world = SphericalWorld(radius=R, radial_gravity=True)

    g = gecko()
    robot_spec = g.spec

    print("[INFO] Spawning gecko (bbox_correction=True) ...")
    world.spawn(robot_spec, prefix_id=0)  # default bbox_correction=True

    print("[INFO] Compiling combined model ...")
    model = world.spec.compile()
    data = mujoco.MjData(model)
    world.attach_model(model)

    mujoco.mj_forward(model, data)

    print(f"[OK] model.nq={model.nq}, nv={model.nv}, nu={model.nu}, ngeom={model.ngeom}")
    print("----------------------------------------------------------")

    # ----------------------------------------------------------
    # 2) Identify planet geom and its radius
    # ----------------------------------------------------------
    try:
        planet_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "planet")
    except TypeError:
        planet_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, b"planet")

    if planet_gid < 0:
        print("[FATAL] Could not find geom named 'planet' in model.")
        return

    planet_r = float(model.geom_size[planet_gid][0])
    planet_pos = data.geom_xpos[planet_gid].copy()
    print(f"[INFO] Planet geom id={planet_gid}")
    print(f"       planet radius (geom_size[0]) = {planet_r:.4f}")
    print(f"       planet position (geom_xpos)  = {planet_pos}\n")

    # ----------------------------------------------------------
    # 3) List ALL geoms with radial distances
    # ----------------------------------------------------------
    robot_radii = []
    robot_gids = []

    print("--- GEOM RADII FROM ORIGIN ---")
    for gid in range(model.ngeom):
        name = pretty_name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        pos = data.geom_xpos[gid].copy()
        r = float(np.linalg.norm(pos))

        kind = "OTHER"
        if gid == planet_gid:
            kind = "PLANET"
        elif name.startswith("robot_0"):
            kind = "ROBOT"
            robot_radii.append(r)
            robot_gids.append(gid)

        print(f"  gid={gid:2d}  kind={kind:6s}  name='{name:20s}'  r={r:8.4f}  pos={pos}")

    if not robot_radii:
        print("\n[FATAL] Found no geoms whose name starts with 'robot_0'.")
        return

    min_r0 = float(np.min(robot_radii))
    max_r0 = float(np.max(robot_radii))

    print("\n--- INITIAL ROBOT RADIAL RANGE ---")
    print(f"  planet radius R        = {planet_r:.4f}")
    print(f"  min robot geom radius  = {min_r0:.4f}")
    print(f"  max robot geom radius  = {max_r0:.4f}")
    print(f"  min_r - R              = {min_r0 - planet_r:+.4f}  ( >0 = outside, <0 = inside )")
    print("----------------------------------------------------------")

    # ----------------------------------------------------------
    # 4) Short settle with radial gravity only, track contacts
    # ----------------------------------------------------------
    steps_settle = 2000
    print(f"[INFO] Running {steps_settle} settle steps with radial gravity ...")

    min_r_over_time = min_r0
    max_r_over_time = max_r0
    max_contacts = 0

    for step in range(steps_settle):
        world.step(data)

        # Track min/max robot radii over time
        rs = []
        for gid in robot_gids:
            rs.append(np.linalg.norm(data.geom_xpos[gid]))
        rs = np.asarray(rs)

        min_r_t = float(np.min(rs))
        max_r_t = float(np.max(rs))

        min_r_over_time = min(min_r_over_time, min_r_t)
        max_r_over_time = max(max_r_over_time, max_r_t)

        max_contacts = max(max_contacts, data.ncon)

        # Print a few snapshots
        if step % 200 == 0:
            print(
                f"  step={step:4d}  "
                f"min_r={min_r_t:8.4f}  max_r={max_r_t:8.4f}  "
                f"min_r-R={min_r_t - planet_r:+.4f}  ncon={data.ncon}"
            )

    print("\n--- POST-SETTLE ROBOT RADIAL RANGE ---")
    print(f"  planet radius R              = {planet_r:.4f}")
    print(f"  min robot radius over time   = {min_r_over_time:.4f}")
    print(f"  max robot radius over time   = {max_r_over_time:.4f}")
    print(f"  min_r_over_time - R          = {min_r_over_time - planet_r:+.4f}")
    print(f"  max_r_over_time - R          = {max_r_over_time - planet_r:+.4f}")
    print(f"  max number of contacts (ncon) seen = {max_contacts}")
    print("----------------------------------------------------------")

    # ----------------------------------------------------------
    # 5) Interpret result in plain language
    # ----------------------------------------------------------
    eps = 1e-2  # 1 cm tolerance

    if min_r_over_time > planet_r + eps:
        print("[RESULT] All robot geoms remain CLEARLY OUTSIDE the sphere surface.")
    elif abs(min_r_over_time - planet_r) <= eps:
        print("[RESULT] Robot is essentially ON the sphere surface (within ~1 cm).")
    else:
        print("[RESULT] Some robot geoms appear INSIDE the sphere (min_r < R).")
        print("         This would explain camera weirdness or lack of contact.")
        print("         In that case, we should adjust spawn z / bbox logic in SphericalWorld.")

    if max_contacts > 0:
        print("[RESULT] Contacts between robot and planet WERE detected.")
    else:
        print("[RESULT] No contacts detected; either robot is inside, or slightly floating above the surface.")

    print("\n========== S A N I T Y   C H E C K   D O N E ==========\n")


if __name__ == "__main__":
    main()
