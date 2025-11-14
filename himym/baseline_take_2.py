from __future__ import annotations
import os
import json
import time
import logging
import random
from datetime import datetime
from functools import partial
from typing import List, Tuple

import numpy as np
from deap import base, creator, tools
import multiprocessing

# MuJoCo imports (assumes mujoco >= 2.3 Python bindings)
import mujoco
from mujoco import viewer

# Local project imports (your original layout)
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.simulation.environments import (
    SimpleFlatWorld,
    TiltedFlatWorld,
    RuggedTerrainWorld,
)
from ariel.utils.runners import simple_runner

# -----------------------------
# Configuration / Hyperparams
# -----------------------------
SEED = 42
NUM_GENERATIONS = 15
POP_SIZE = 60
MATE_CHANCE = 0.75
MUTATE_CHANCE = 0.35
ELITES = 2
INITIAL_MUT_SIGMA = 0.3
MAX_MUT_SIGMA = 3.0
STAGNATION_WINDOW = 100
MAX_STAGNATION = 10
HIDDEN_SIZE = 12
DEBUG = False
DEBUG_DURATION = 3
DEFAULT_DURATION = 20
EVAL_TRIALS = 3  # number of randomized trials per individual to get robust cost estimate
N_PROC = max(1, multiprocessing.cpu_count() - 1)

# Cost weights (these scale different penalty terms to make the optimization stable)
W_LATERAL = 2.0
W_STRAIGHT = 8.0
W_ENERGY = 0.01
W_FLIP = 100.0
W_TIME = 0.0  # optional time penalty for short-living episodes

OUTPUT_FOLDER = "./results"
FILENAME_PREFIX = "gecko_minimization"

# -----------------------------
# Utilities
# -----------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def weights_from_list_to_matrix(individual: List[float], input_size: int, hidden_size: int, output_size: int):
    index = 0
    input_size_bias = input_size + 1
    W1_size = input_size_bias * hidden_size
    W2_size = hidden_size * hidden_size
    W3_size = hidden_size * output_size

    W1 = np.array(individual[index: index + W1_size]).reshape((input_size_bias, hidden_size))
    index += W1_size
    W2 = np.array(individual[index: index + W2_size]).reshape((hidden_size, hidden_size))
    index += W2_size
    W3 = np.array(individual[index: index + W3_size]).reshape((hidden_size, output_size))

    return {"W1": W1, "W2": W2, "W3": W3}


# -----------------------------
# Controller factory
# -----------------------------

def make_controller(weights: dict, actuator_limit=None):
    """Return a controller closure that uses the provided weights and respects actuator_limit.

    The controller expects model.data and writes to data.ctrl.
    The network architecture: inputs = qpos + bias -> tanh -> tanh -> tanh (scaled to actuator ranges)
    """

    def controller(model, data):
        try:
            inputs = np.append(data.qpos, 1.0)
            l1 = np.tanh(np.dot(inputs, weights['W1']))
            l2 = np.tanh(np.dot(l1, weights['W2']))
            raw = np.tanh(np.dot(l2, weights['W3']))  # in (-1,1)

            # Map to actuator ranges
            if actuator_limit is None:
                scaled = raw * (np.pi / 2)
                clip_min, clip_max = -np.pi / 2, np.pi / 2
            else:
                arr = np.array(actuator_limit)
                if arr.shape == ():
                    scaled = raw * float(arr)
                    clip_min, clip_max = -float(arr), float(arr)
                else:
                    # arr expected to be per-actuator max absolute value
                    scaled = raw * arr
                    clip_min, clip_max = -np.abs(arr), np.abs(arr)

            # assign into data.ctrl safely
            try:
                data.ctrl[:] = np.clip(scaled, clip_min, clip_max)
            except Exception:
                # some mujoco bindings may require assignment per index
                for i in range(min(len(data.ctrl), len(scaled))):
                    data.ctrl[i] = float(np.clip(scaled[i], clip_min if hasattr(clip_min, '__iter__') else clip_min, clip_max if hasattr(clip_max, '__iter__') else clip_max))
        except Exception as e:
            # if controller fails, preserve zeros (safer than crashing the sim)
            try:
                data.ctrl[:] = 0.0
            except Exception:
                pass

    return controller


# -----------------------------
# Cost / fitness evaluation (minimize cost)
# -----------------------------

def evaluate_individual_cost(individual: List[float], duration: float = 20.0, trials: int = EVAL_TRIALS, seed_base: int = 0) -> Tuple[float]:
    """Return a single scalar cost to be minimized. We run multiple short episodes and average the cost.

    Cost formula (to minimize):
      cost = -forward_progress + W_LATERAL * lateral_drift + W_STRAIGHT * straightness_penalty + W_ENERGY * energy + W_FLIP * flip_penalty

    - forward_progress is displacement along +Y axis (we *negate* it so that larger forward -> smaller cost)
    - lateral_drift = absolute displacement along X
    - straightness_penalty approximates how much the path deviates from a straight line
    - energy is mean squared control signal
    - flip_penalty is a large penalty if robot flips or falls (low core z)

    Returns a 1-tuple (cost,) compatible with DEAP.
    """

    costs = []

    # small helper to run one episode in a given world
    def single_run(world_ctor, seed_offset):
        # Build the world and spawn gecko
        mujoco.set_mjcb_control(None)
        world = world_ctor()
        gecko_core = gecko()
        world.spawn(gecko_core.spec, spawn_position=[0, 0, 0])
        model = world.spec.compile()
        data = mujoco.MjData(model)

        # bind geoms for tracking core positions (fallback to body if geom search fails)
        try:
            geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
            to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]
        except Exception:
            to_track = []

        # compute sizes
        input_size = len(data.qpos)
        output_size = model.nu

        # map genome to weights
        weights = weights_from_list_to_matrix(individual, input_size, HIDDEN_SIZE, output_size)

        # actuator limit discovery
        actuator_limit = None
        try:
            if hasattr(model, 'actuator_ctrlrange'):
                arr = np.array(model.actuator_ctrlrange)
                # ctrlrange shape (nu,2) -> we want max abs for each actuator
                if arr.ndim == 2 and arr.shape[1] == 2:
                    actuator_limit = np.max(np.abs(arr), axis=1)
                else:
                    actuator_limit = np.max(np.abs(arr))
        except Exception:
            actuator_limit = None

        controller = make_controller(weights, actuator_limit=actuator_limit)
        mujoco.set_mjcb_control(lambda m, d: controller(m, d))

        # run simulation and collect history
        HISTORY = []
        CTRL_HISTORY = []

        # define how many steps to run using model timestep
        try:
            timestep = float(model.opt.timestep) if hasattr(model, 'opt') and hasattr(model.opt, 'timestep') else 0.002
        except Exception:
            timestep = 0.002
        n_steps = max(1, int(np.ceil(duration / timestep)))

        for step in range(n_steps):
            # step the sim
            try:
                mujoco.mj_step(model, data)
            except Exception:
                # some mujoco versions use mj_step1/mj_step2
                try:
                    mujoco.mj_step1(model, data)
                    mujoco.mj_step2(model, data)
                except Exception:
                    break

            # record ctrl
            try:
                CTRL_HISTORY.append(np.array(data.ctrl).copy())
            except Exception:
                pass

            # record position using bound geom if available
            if len(to_track) > 0:
                try:
                    HISTORY.append(to_track[0].xpos.copy())
                except Exception:
                    pass
            else:
                # fallback: attempt to read base pos from qpos (first three entries often x,y,z)
                try:
                    if len(data.qpos) >= 3:
                        HISTORY.append(np.array([float(data.qpos[0]), float(data.qpos[1]), float(data.qpos[2])]))
                except Exception:
                    pass

        mujoco.set_mjcb_control(None)

        if len(HISTORY) < 2:
            # failed to get data; assign a large cost
            return float('inf')

        positions = np.array(HISTORY)

        # Basic descriptors
        forward = float(positions[-1, 1] - positions[0, 1])
        lateral = float(abs(positions[-1, 0] - positions[0, 0]))

        # path length vs straight-line to compute straightness
        deltas = np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1)
        path_length = float(np.sum(deltas))
        straight_line = max(1e-6, float(np.linalg.norm(positions[-1, :2] - positions[0, :2])))
        # fraction of extra distance taken (0 -> perfectly straight forward)
        straightness_penalty = max(0.0, (path_length - straight_line) / (path_length + 1e-6))

        # energy (mean squared control signals recorded)
        try:
            if len(CTRL_HISTORY) > 0:
                ctrl_arr = np.vstack(CTRL_HISTORY)
                energy = float(np.mean(np.square(ctrl_arr)))
            else:
                energy = 0.0
        except Exception:
            energy = 0.0

        # flip / torso down penalty: if core's z drops below threshold at any point -> penalty
        z_values = positions[:, 2] if positions.shape[1] > 2 else np.zeros(len(positions))
        flip_penalty = float(np.any(z_values < 0.05)) * 1.0  # boolean to 0/1

        # Compose cost: negative forward (so bigger forward reduces cost)
        cost = -forward + W_LATERAL * lateral + W_STRAIGHT * straightness_penalty + W_ENERGY * energy + W_FLIP * flip_penalty

        # If robot went backwards strongly, add more penalty
        if forward < -0.01:
            cost += 1.0

        return float(cost)

    # Choose trials: mix of worlds to encourage straight forward robustness
    world_constructors = [SimpleFlatWorld] #[SimpleFlatWorld, TiltedFlatWorld, RuggedTerrainWorld]
    selected_worlds = [world_constructors[i % len(world_constructors)] for i in range(trials)]

    for t_idx, world_ctor in enumerate(selected_worlds):
        seed_offset = seed_base + t_idx
        try:
            c = single_run(world_ctor, seed_offset)
            costs.append(c)
        except Exception:
            costs.append(float('inf'))

    # average (ignore inf if any infinite -> return inf)
    if any([np.isinf(c) for c in costs]):
        return (float('inf'),)

    mean_cost = float(np.mean(costs))
    return (mean_cost,)


# -----------------------------
# JSON saving
# -----------------------------

def save_results_json(output_folder: str, filename_prefix: str, fitness_history: List[Tuple[float, float, float]], best_individual, input_size: int, hidden_size: int, output_size: int, final_population: list):
    os.makedirs(output_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.json"
    out_path = os.path.join(output_folder, filename)

    generations = {}
    for i, (best, mean, worst) in enumerate(fitness_history):
        generations[f"gen{i}"] = {"best": float(best), "mean": float(mean), "worst": float(worst)}

    metadata = {
        "total_generations": len(fitness_history),
    }

    best_genome = [float(x) for x in best_individual]
    weights = weights_from_list_to_matrix(best_genome, input_size, hidden_size, output_size)
    weights_lists = {k: v.tolist() for k, v in weights.items()}

    out_data = {
        "generations": generations,
        "metadata": metadata,
        "weights": weights_lists,
    }

    with open(out_path, 'w') as f:
        json.dump(out_data, f, indent=2)

    logging.info(f"Saved JSON to {out_path}")
    return out_path


# -----------------------------
# GA main
# -----------------------------

def main():
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

    random.seed(SEED)
    np.random.seed(SEED)

    # small dummy world to compute sizes
    mujoco.set_mjcb_control(None)
    world = SimpleFlatWorld()
    gecko_core = gecko()
    world.spawn(gecko_core.spec, spawn_position=[0, 0, 0])
    model = world.spec.compile()
    data = mujoco.MjData(model)

    input_size = len(data.qpos)
    input_size_bias = input_size + 1
    output_size = model.nu

    genome_length = (input_size_bias * HIDDEN_SIZE) + (HIDDEN_SIZE * HIDDEN_SIZE) + (HIDDEN_SIZE * output_size)
    logging.info(f"Genome length computed = {genome_length}")

    # DEAP setup for minimization
    try:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    except Exception:
        pass

    try:
        creator.create("Individual", list, fitness=creator.FitnessMin)
    except Exception:
        pass

    toolbox = base.Toolbox()
    toolbox.register("attr_param", lambda: random.uniform(-1.0, 1.0))
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_param, n=genome_length)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # evaluate function for DEAP (wrap with desired duration and trials)
    eval_func = partial(evaluate_individual_cost, duration=(DEBUG_DURATION if DEBUG else DEFAULT_DURATION), trials=EVAL_TRIALS)
    toolbox.register("evaluate", eval_func)

    toolbox.register("mate", tools.cxUniform, indpb=0.5)

    # changing_sigma wrapper
    MUT_SIGMA = INITIAL_MUT_SIGMA

    def changing_sigma_wrapper(individual):
        nonlocal MUT_SIGMA
        out = tools.mutGaussian(individual, mu=0.0, sigma=MUT_SIGMA, indpb=0.1)
        return out

    toolbox.register("mutate", changing_sigma_wrapper)
    toolbox.register("select", tools.selTournament, tournsize=6)

    # parallel map
    pool = multiprocessing.Pool(processes=N_PROC)
    toolbox.register("map", pool.map)

    pop = toolbox.population(n=POP_SIZE)

    logging.info("Evaluating initial population...")
    invalid = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = list(toolbox.map(toolbox.evaluate, invalid))
    for ind, fit in zip(invalid, fitnesses):
        ind.fitness.values = fit

    hof = tools.HallOfFame(1)
    fitness_history = []

    best_so_far = float('inf')
    no_improve = 0

    for gen in range(NUM_GENERATIONS):
        # adaptive mutation sigma
        if no_improve >= STAGNATION_WINDOW and MUT_SIGMA < MAX_MUT_SIGMA:
            MUT_SIGMA *= 1.3
            logging.info(f"increasing sigma to {MUT_SIGMA}")
        else:
            MUT_SIGMA = max(0.02, MUT_SIGMA * 0.995)

        # Select parents and create offspring
        parents = toolbox.select(pop, POP_SIZE - ELITES)
        parents = list(map(toolbox.clone, parents))
        random.shuffle(parents)

        children = []
        for p1, p2 in zip(parents[::2], parents[1::2]):
            c1, c2 = p1, p2
            if random.random() < MATE_CHANCE:
                toolbox.mate(c1, c2)
                if hasattr(c1, 'fitness'):
                    del c1.fitness.values
                if hasattr(c2, 'fitness'):
                    del c2.fitness.values
            if random.random() < MUTATE_CHANCE:
                toolbox.mutate(c1)
                toolbox.mutate(c2)
                if hasattr(c1, 'fitness'):
                    del c1.fitness.values
                if hasattr(c2, 'fitness'):
                    del c2.fitness.values
            children.extend([c1, c2])

        # Evaluate invalid children in parallel
        invalid_children = [ind for ind in children if not hasattr(ind, 'fitness') or not ind.fitness.valid]
        if invalid_children:
            fits = list(toolbox.map(toolbox.evaluate, invalid_children))
            for ind, fit in zip(invalid_children, fits):
                ind.fitness.values = fit

        # form new population with elitism
        elites = tools.selBest(pop, ELITES)
        total = pop + children
        pop = toolbox.select(total, POP_SIZE - ELITES) + elites

        # logging
        best = tools.selBest(pop, 1)[0]
        hof.update(pop)

        # compute mean & worst
        fits = [ind.fitness.values[0] for ind in pop]
        mean_fit = float(np.mean(fits))
        worst_fit = float(np.max(fits))
        best_fit = float(np.min(fits))
        fitness_history.append((best_fit, mean_fit, worst_fit))

        logging.info(f"Gen {gen}: best={best_fit:.6f} mean={mean_fit:.6f} worst={worst_fit:.6f}")

        # stagnation tracking (we're minimizing)
        if best_fit < best_so_far - 1e-8:
            best_so_far = best_fit
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= MAX_STAGNATION:
            logging.info(f"No improvement for {MAX_STAGNATION} generations. Stopping at gen {gen}.")
            break

    # finalize
    best_ind = hof[0]

    out_path = save_results_json(
        output_folder=OUTPUT_FOLDER,
        filename_prefix=FILENAME_PREFIX,
        fitness_history=fitness_history,
        best_individual=best_ind,
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        output_size=output_size,
        final_population=pop,
    )

    # visualize best
    try:
        HISTORY_BEST = []
        weights = weights_from_list_to_matrix(best_ind, input_size, HIDDEN_SIZE, output_size)

        # spawn a fresh world and run viewer
        mujoco.set_mjcb_control(None)
        world = SimpleFlatWorld()
        gecko_core = gecko()
        world.spawn(gecko_core.spec, spawn_position=[0, 0, 0])
        model = world.spec.compile()
        data = mujoco.MjData(model)
        geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
        to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]

        # actuator limits
        actuator_limit = None
        try:
            if hasattr(model, 'actuator_ctrlrange'):
                arr = np.array(model.actuator_ctrlrange)
                if arr.ndim == 2 and arr.shape[1] == 2:
                    actuator_limit = np.max(np.abs(arr), axis=1)
                else:
                    actuator_limit = np.max(np.abs(arr))
        except Exception:
            actuator_limit = None

        def best_controller(model, data):
            # mimic make_controller but record history
            try:
                inputs = np.append(data.qpos, 1.0)
                l1 = np.tanh(np.dot(inputs, weights['W1']))
                l2 = np.tanh(np.dot(l1, weights['W2']))
                raw = np.tanh(np.dot(l2, weights['W3']))

                if actuator_limit is None:
                    scaled = raw * (np.pi / 2)
                    clip_min, clip_max = -np.pi / 2, np.pi / 2
                else:
                    arr = np.array(actuator_limit)
                    if arr.shape == ():
                        scaled = raw * float(arr)
                        clip_min, clip_max = -float(arr), float(arr)
                    else:
                        scaled = raw * arr
                        clip_min, clip_max = -np.abs(arr), np.abs(arr)

                data.ctrl[:] = np.clip(scaled, clip_min, clip_max)
            except Exception:
                try:
                    data.ctrl[:] = 0.0
                except Exception:
                    pass

            # record
            try:
                if len(to_track) > 0:
                    HISTORY_BEST.append(to_track[0].xpos.copy())
            except Exception:
                pass

        mujoco.set_mjcb_control(lambda m, d: best_controller(m, d))
        viewer.launch(model, data)

        # after closing viewer, save history + simple plot data to JSON
        try:
            if len(HISTORY_BEST) > 1:
                hist_arr = np.array(HISTORY_BEST).tolist()
                hist_out = os.path.join(OUTPUT_FOLDER, f"history_best_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
                with open(hist_out, 'w') as f:
                    json.dump(hist_arr, f)
                logging.info(f"Saved best run history to {hist_out}")
        except Exception:
            pass

    except Exception as e:
        logging.exception("Failed to visualize best individual")

    pool.close()
    pool.join()


if __name__ == '__main__':
    main()
