import copy
import time

import mujoco
import numpy as np
from mujoco import viewer

# ARIEL imports
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld

# HyperNEAT imports (your local files: hyperneat.py, genetic_operators.py)
from hyperneat import CPPN, SubstrateNetwork, create_substrate_for_gecko
from genetic_operators import create_initial_hyperneat_genome


# =====================================================
# CONTROLLER: HyperNEAT + CPG
# =====================================================
def build_controller(model: mujoco.MjModel, num_joints: int, genotype):
    """
    Build a HyperNEAT + CPG controller for a single gecko.
    Inputs to substrate:
      - joint_angles (num_joints)
      - CPG oscillators (4)
      - directional (2)  -> here fixed to [1,0] to say "go +X"
      - bias (1)
    """
    cppn = CPPN(genotype)

    input_coords, hidden_coords, output_coords = create_substrate_for_gecko(
        num_joints=num_joints,
        use_hidden_layer=True,
        hidden_layer_size=num_joints,
    )

    substrate = SubstrateNetwork(
        input_coords=input_coords,
        hidden_coords=hidden_coords,
        output_coords=output_coords,
        cppn=cppn,
        weight_threshold=0.2,
    )

    CONTROL_MIN, CONTROL_MAX = -1.0, 1.0

    def controller(m: mujoco.MjModel, d: mujoco.MjData):
        # Skip 7-DOF free joint, then take joint angles
        joint_angles = d.qpos[7:7 + num_joints].copy()

        # Simple CPG oscillators
        t = d.time
        osc1 = np.sin(t * 1.0)
        osc2 = np.cos(t * 1.0)
        osc3 = np.sin(t * 2.0)
        osc4 = np.cos(t * 2.0)
        cpg = np.array([osc1, osc2, osc3, osc4])

        # Directional input: "go +X"
        directional = np.array([1.0, 0.0])

        # Bias
        bias = np.array([1.0])

        inp = np.concatenate([joint_angles, cpg, directional, bias])
        motor_out = substrate.activate(inp)

        d.ctrl[:] = np.clip(motor_out, CONTROL_MIN, CONTROL_MAX)

    return controller


# =====================================================
# FITNESS EVALUATION
# =====================================================
def evaluate_genotype(
    model: mujoco.MjModel,
    core_geom_id: int,
    num_joints: int,
    genotype,
    sim_time: float = 5.0,
):
    """
    Evaluate one genotype:
      - start at current pose
      - run for sim_time seconds with HyperNEAT+CPG controller
      - fitness = forward distance along +X, only if positive
    Returns: (fitness, total_dist, dx, dy, start_pos, end_pos, sim_time)
    """
    data = mujoco.MjData(model)
    controller = build_controller(model, num_joints, genotype)
    mujoco.set_mjcb_control(controller)

    mujoco.mj_forward(model, data)
    start = data.geom_xpos[core_geom_id].copy()

    n_steps = int(sim_time / model.opt.timestep)
    for _ in range(n_steps):
        mujoco.mj_step(model, data)

    end = data.geom_xpos[core_geom_id].copy()
    dx = float(end[0] - start[0])
    dy = float(end[1] - start[1])
    dist = float(np.sqrt(dx * dx + dy * dy))

    # Reward **only** forward movement along +X, penalize sideways a bit
    raw_forward = dx - 0.2 * abs(dy)
    fitness = max(0.0, raw_forward)

    return fitness, dist, dx, dy, start, end, sim_time


# =====================================================
# SIMPLE GA
# =====================================================
def mutate_genotype(parent):
    """
    Very simple mutation: jitter HyperNEAT connection weights.
    Assumes genotype['connections'] is a list of objects with .weight
    (this matches your friend's HyperNEAT implementation).
    """
    child = copy.deepcopy(parent)
    if "connections" in child:
        for conn in child["connections"]:
            # 70% chance to perturb each weight
            if np.random.rand() < 0.7:
                conn.weight += np.random.normal(0.0, 0.5)
    return child


def evolve(
    model: mujoco.MjModel,
    core_geom_id: int,
    num_joints: int,
    generations: int = 20,
    pop_size: int = 30,
):
    """
    Tiny GA over HyperNEAT genotypes:
      - initialize random pop
      - evaluate all
      - keep top K, mutate to refill
    """
    population = [
        create_initial_hyperneat_genome(
            num_inputs=4,        # CPPN inputs (x1,y1,x2,y2)
            num_outputs=1,       # CPPN output (weight)
            activation="sine",   # like V's config
        )
        for _ in range(pop_size)
    ]

    best_geno = None
    best_fit = -1e9

    print("Robot has", num_joints, "controllable joints\n")

    for g in range(1, generations + 1):
        print(f"=== Generation {g}/{generations} ===")
        scored = []

        for geno in population:
            fit, dist, dx, dy, _, _, _ = evaluate_genotype(
                model, core_geom_id, num_joints, geno
            )
            scored.append((fit, geno))

        scored.sort(reverse=True, key=lambda x: x[0])
        gen_best = scored[0][0]
        print(f"  Best fitness: {gen_best:.4f}")

        if gen_best > best_fit:
            best_fit = gen_best
            best_geno = copy.deepcopy(scored[0][1])

        # select top K parents
        num_parents = min(5, len(scored))
        parents = [geno for (fit, geno) in scored[:num_parents]]

        # rebuild population
        new_pop = parents.copy()
        while len(new_pop) < pop_size:
            p = parents[np.random.randint(len(parents))]
            new_pop.append(mutate_genotype(p))

        population = new_pop

    print(f"\nFinal best fitness: {best_fit:.4f}")
    return best_geno


# =====================================================
# DEMO BEST INDIVIDUAL
# =====================================================
def show_best(
    model: mujoco.MjModel,
    core_geom_id: int,
    num_joints: int,
    best_genotype,
):
    """
    Reuse SAME model, fresh data, open MuJoCo viewer and let it run
    until you close the window.
    """
    data = mujoco.MjData(model)
    controller = build_controller(model, num_joints, best_genotype)
    mujoco.set_mjcb_control(controller)

    mujoco.mj_forward(model, data)
    start = data.geom_xpos[core_geom_id].copy()

    print("\n[INFO] Launching viewer... (close window to finish)")
    with viewer.launch_passive(model, data) as v:
        while v.is_running():
            mujoco.mj_step(model, data)
            v.sync()
            time.sleep(0.001)

    end = data.geom_xpos[core_geom_id].copy()
    dx = float(end[0] - start[0])
    dy = float(end[1] - start[1])
    dist = float(np.sqrt(dx * dx + dy * dy))
    print(f"[INFO] Δx (forward): {dx:.4f} m, Δy (side): {dy:.4f} m, total: {dist:.4f} m")


# =====================================================
# MAIN
# =====================================================
def main():
    print("\n========== EVOLVING A SINGLE GECKO ==========\n")

    # ---- Build world and robot ONCE ----
    world = SimpleFlatWorld()  # your version: only floor_size argument
    robot = gecko()

    # IMPORTANT: no bounding-box correction, no extra compile inside spawn
    world.spawn(
        robot.spec,
        spawn_position=[0, 0, 0.5],
        correct_for_bounding_box=False,
    )

    # Compile full MJCF once (world + robot)
    model = world.spec.compile()

    # Identify actuators / joints and core geom
    num_joints = model.nu
    core_geom_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "robot-0core",
    )

    if core_geom_id < 0:
        raise RuntimeError("Could not find geom 'robot-0core' in model!")

    print(f"[INFO] Gecko actuators (num_joints): {num_joints}\n")

    # ---- Evolution ----
    best = evolve(
        model=model,
        core_geom_id=core_geom_id,
        num_joints=num_joints,
        generations=20,
        pop_size=30,
    )

    # ---- Demo ----
    print("\n========== SHOWING BEST ==========\n")
    show_best(model, core_geom_id, num_joints, best)


if __name__ == "__main__":
    main()
