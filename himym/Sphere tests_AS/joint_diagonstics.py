"""
gecko_joint_diagnostics.py
Inspect Gecko's MuJoCo model: joints, actuators, and DOF mappings.

Safely prints full info about all joints, actuators, and their qpos ranges.
"""

import mujoco
import numpy as np
from ariel.simulation.environments.spherical_world import SphericalWorld
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko


def joint_dof_count(joint_type: int) -> int:
    """Return number of DOFs for a given joint type."""
    # 0=free, 1=ball, 2=slide, 3=hinge
    if joint_type == 0:
        return 6
    elif joint_type == 1:
        return 3
    else:
        return 1


def safe_name(model, obj_type, idx):
    """Return safe object name."""
    try:
        n = mujoco.mj_id2name(model, obj_type, idx)
        return n if n is not None else f"<unnamed_{idx}>"
    except Exception:
        return f"<invalid_{idx}>"


def run_diagnostics():
    world = SphericalWorld(radius=5.0, radial_gravity=True)
    g = gecko()
    spec = g.spec if hasattr(g, "spec") else g
    world.spawn(spec)

    model = world.spec.compile()
    world.attach_model(model)
    data = mujoco.MjData(model)

    print("=== Gecko Joint & Actuator Diagnostics ===\n")
    print(f"nq={model.nq}, nv={model.nv}, nu={model.nu}, njnt={model.njnt}\n")

    # ---- JOINTS ----
    print("---- Joints ----")
    for j in range(model.njnt):
        name = safe_name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        jtype = model.jnt_type[j]
        jadr = model.jnt_dofadr[j]
        ndof = joint_dof_count(jtype)
        rng = model.jnt_range[j]
        jtype_str = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}.get(jtype, "?")
        print(f"[{j:2d}] {name:30s} type={jtype_str:<6s} dofadr={jadr:<2d} ndof={ndof} range={rng}")

    # ---- ACTUATORS ----
    print("\n---- Actuators ----")
    for i in range(model.nu):
        aname = safe_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        trnid = model.actuator_trnid[i]
        j = int(trnid[0])
        jname = safe_name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        ctrlrange = model.actuator_ctrlrange[i]
        gear = model.actuator_gear[i]
        bias = model.actuator_biastype[i]
        print(f"[{i:2d}] {aname:25s} → joint[{j:2d}] {jname:25s} biasType={bias:<2d} ctrl={ctrlrange} gear={gear}")

    # ---- DOF MAP ----
    print("\n---- DOF Address Mapping (qpos/qvel indices) ----")
    for j in range(model.njnt):
        dofadr = model.jnt_dofadr[j]
        jtype = model.jnt_type[j]
        ndof = joint_dof_count(jtype)
        if ndof > 0:
            name = safe_name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            print(f"joint[{j:2d}] '{name}' -> qpos[{dofadr}:{dofadr+ndof}]")

    print("\nDiagnostics complete.\n")


if __name__ == "__main__":
    run_diagnostics()
