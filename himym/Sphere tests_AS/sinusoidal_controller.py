"""
Evolving Gecko locomotion on a spherical world (directional locomotion)
-----------------------------------------------------------------------
- Uses position-controlled actuators (±1.5 radians)
- Fitness = forward displacement along +X direction
- Higher surface friction for traction
- Viewer opens once after evolution and stays open until closed manually
"""

import mujoco
from mujoco import viewer
import numpy as np
import random
from deap import base, creator, tools

# --- Local imports ---
from ariel.simulation.environments.spherical_world import SphericalWorld
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko


# =====================================================
# SETUP
# =====================================================
world = SphericalWorld(radius=5.0, radial_gravity=True)
g = gecko()
gecko_spec = g.spec if hasattr(g, "spec") else g
world.spawn(gecko_spec, spawn_position=[0, 0, world.radius + 0.05])

model = world.spec.compile()
world.attach_model(model)
model.opt.timestep = 0.005
model.opt.iterations = 50
dt = model.opt.timestep
N_ACT = model.nu

# Increase friction for better traction
model.geom_friction[:] = np.array([1.2, 0.1, 0.01])

# =====================================================
# SAFE RADIAL GRAVITY
# =====================================================
def safe_apply_radial_gravity(world, data):
    gmag = abs(world.gravity)
    for i in range(world.model.nbody):
        pos = np.nan_to_num(data.xipos[i], nan=0.0)
        dist = np.linalg.norm(pos)
        if dist < 1e-6 or not np.isfinite(dist):
            continue
        direction = -pos / dist
        data.xfrc_applied[i, :3] += direction * world.model.body_mass[i] * gmag


# =====================================================
# FITNESS FUNCTION
# =====================================================
def evaluate(ind):
    """Simulate gecko with given controller params and return directional fitness."""
    freq = np.clip(0.5 + 1.5 * abs(ind[0]), 0.5, 2.0)
    amp = np.clip(0.4 + 0.6 * abs(ind[1]), 0.4, 1.0)

    # Distinct phase offsets for asymmetry
    base_phase = np.linspace(0, 2*np.pi, N_ACT, endpoint=False)
    phase_offsets = base_phase + np.pi * np.array(ind[2:2 + N_ACT])

    data = mujoco.MjData(model)
    start_com = np.copy(data.subtree_com[0])
    steps = 1500
    target_dir = np.array([1.0, 0.0])  # +X direction

    try:
        for i in range(steps):
            t = i * dt
            ctrl = 1.5 * amp * np.sin(2 * np.pi * freq * t + phase_offsets)
            data.ctrl[:] = np.clip(ctrl, -1.5, 1.5)
            safe_apply_radial_gravity(world, data)
            mujoco.mj_step(model, data)
            if not np.all(np.isfinite(data.qpos)):
                raise FloatingPointError

        end_com = np.copy(data.subtree_com[0])
        disp_vec = end_com[:2] - start_com[:2]
        total_disp = np.linalg.norm(disp_vec)
        forward_disp = np.dot(disp_vec, target_dir)
        fitness = max(0.0, forward_disp)  # ignore backward motion

        # Return both for logging, but only forward_disp for evolution
        return (fitness, total_disp)

    except FloatingPointError:
        return (0.0, 0.0)


# =====================================================
# EVOLUTIONARY ALGORITHM (DEAP)
# =====================================================
POP_SIZE = 8
GENS = 20
PARAMS = 2 + N_ACT  # freq, amp, phases

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("param", random.uniform, -1.0, 1.0)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.param, n=PARAMS)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxBlend, alpha=0.3)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.3, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)


def evolve():
    pop = toolbox.population(n=POP_SIZE)
    best_fit = -1.0
    best_ind = None

    for gen in range(GENS):
        print(f"\n=== Generation {gen+1}/{GENS} ===")

        for ind in pop:
            fwd, total = evaluate(ind)
            ind.fitness.values = (fwd,)
            ind.total_disp = total

        offspring = tools.selBest(pop, k=2) + toolbox.select(pop, len(pop) - 2)
        offspring = list(map(toolbox.clone, offspring))

        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.7:
                toolbox.mate(c1, c2)
                del c1.fitness.values, c2.fitness.values

        for m in offspring:
            if random.random() < 0.3:
                toolbox.mutate(m)
                del m.fitness.values

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid:
            fwd, total = evaluate(ind)
            ind.fitness.values = (fwd,)
            ind.total_disp = total

        pop = tools.selBest(pop + offspring, k=POP_SIZE)
        best = max(pop, key=lambda i: i.fitness.values[0])
        best_fit = best.fitness.values[0]
        best_total = best.total_disp

        print(f"Best forward disp: {best_fit:.4f} | Total disp: {best_total:.4f}")

    print("\n=== Evolution complete ===")
    print("Best individual parameters:", np.round(best, 3))
    print("Final fitness:", best_fit)
    return best


# =====================================================
# VISUALIZE BEST INDIVIDUAL
# =====================================================
def visualize_best(ind):
    freq = np.clip(0.5 + 1.5 * abs(ind[0]), 0.5, 2.0)
    amp = np.clip(0.4 + 0.6 * abs(ind[1]), 0.4, 1.0)
    base_phase = np.linspace(0, 2*np.pi, N_ACT, endpoint=False)
    phase_offsets = base_phase + np.pi * np.array(ind[2:2 + N_ACT])
    data = mujoco.MjData(model)
    steps = 3000  # longer playback for visual clarity

    print("\nOpening viewer — close manually when finished observing.\n")

    with viewer.launch_passive(model, data) as v:
        v.cam.lookat[:] = [0, 0, 0]
        v.cam.distance = world.radius * 3.0
        v.cam.elevation = -20
        v.cam.azimuth = 120

        while v.is_running():
            if data.time < steps * dt:
                t = data.time
                ctrl = 1.5 * amp * np.sin(2 * np.pi * freq * t + phase_offsets)
                data.ctrl[:] = np.clip(ctrl, -1.5, 1.5)
                safe_apply_radial_gravity(world, data)
                mujoco.mj_step(model, data)
            v.sync()


# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    best = evolve()
    visualize_best(best)
