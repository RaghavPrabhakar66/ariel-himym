from pyexpat import model
import numpy as np
import mujoco
from mujoco import viewer
import matplotlib.pyplot as plt
import time, copy, random
from deap import base, creator, tools
from functools import partial
import logging

# Local libraries
from ariel.utils.renderers import video_renderer
from ariel.utils.video_recorder import VideoRecorder
from ariel.simulation.environments import SimpleFlatWorld, TiltedFlatWorld, BoxyRugged, RuggedTerrainWorld

# import prebuilt robot phenotypes
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.utils.runners import simple_runner
from ariel.utils.renderers import single_frame_renderer

import json
import os
from datetime import datetime

# -----------------------------
# Utility functions
# -----------------------------

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def weights_from_list_to_matrix(individual, input_size, hidden_size, output_size):
    """
    Convert a flat genome list into the weight matrices W1, W2, W3.
    W1 shape: (input_size + 1, hidden_size)
    W2 shape: (hidden_size, hidden_size)
    W3 shape: (hidden_size, output_size)
    """
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
# Controller
# -----------------------------

def persistent_controller(to_track, weights, history_controller, actuator_limit=None):
    """Return a controller closure that uses the provided weights.

    actuator_limit: a scalar or array-like of shape (model.nu,) that gives safe limits.
    If None, the controller will clip outputs to +/- pi/2 as a reasonable default.
    """

    def controller(model, data):
        # inputs: qpos plus bias
        inputs = np.append(data.qpos, 1.0)

        # forward pass
        layer1 = np.tanh(np.dot(inputs, weights['W1']))
        layer2 = np.tanh(np.dot(layer1, weights['W2']))
        outputs = np.sin(np.dot(layer2, weights['W3']))  # in [-1,1]

        # scale outputs to actuator ranges
        if actuator_limit is None:
            scaled = outputs * (np.pi / 2)
            clip_min, clip_max = -np.pi / 2, np.pi / 2
        else:
            # if actuator_limit is a scalar assume symmetric
            try:
                arr = np.array(actuator_limit)
                if arr.shape == ():
                    scaled = outputs * float(arr)
                    clip_min, clip_max = -float(arr), float(arr)
                else:
                    scaled = outputs * arr
                    clip_min, clip_max = np.min(-arr), np.max(arr)
            except Exception:
                scaled = outputs * (np.pi / 2)
                clip_min, clip_max = -np.pi / 2, np.pi / 2

        # assign safely to data.ctrl view
        try:
            data.ctrl[:] = np.clip(scaled, clip_min, clip_max)
        except Exception:
            # fallback assignment
            data.ctrl = np.clip(scaled, clip_min, clip_max)

        # Save core position to history if available
        if len(to_track) > 0:
            try:
                # ensure copy so later modifications don't mutate history
                history_controller.append(to_track[0].xpos.copy())
            except Exception:
                # if binding fails silently continue
                pass

    return controller


# -----------------------------
# Visualization helpers
# -----------------------------

def plot(best_history, NUM_GENERATION):
    best_history = np.array(best_history)
    generation = list(range(len(best_history)))
    plt.figure()
    plt.plot(generation, best_history)
    plt.xlabel("generation")
    plt.ylabel("best_fitness")
    plt.tight_layout()
    plt.show()


def show_qpos_history(history: list):
    if len(history) == 0:
        print("No history to plot")
        return

    pos_data = np.array(history)
    plt.figure(figsize=(8, 6))
    plt.plot(pos_data[:, 0], pos_data[:, 1], label='Path')
    plt.plot(pos_data[0, 0], pos_data[0, 1], 'go', label='Start')
    plt.plot(pos_data[-1, 0], pos_data[-1, 1], 'ro', label='End')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Robot Path in XY Plane')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

    max_range = max(0.3, np.abs(pos_data).max())
    plt.xlim(-max_range, max_range)
    plt.ylim(-max_range, max_range)
    plt.tight_layout()
    plt.show()


# -----------------------------
# JSON saving
# -----------------------------

def save_results_json(output_folder, filename_prefix,
                      fitness_history, best_individual,
                      input_size, hidden_size, output_size,
                      final_population=None):

    os.makedirs(output_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.json"
    out_path = os.path.join(output_folder, filename)

    generations_obj = {f"gen{i}": float(v) for i, v in enumerate(fitness_history)}

    total_generations = len(fitness_history)
    final_best = float(fitness_history[-1]) if len(fitness_history) > 0 else 0.0

    if final_population is not None and len(final_population) > 0:
        pop_vals = [float(ind.fitness.values[0]) for ind in final_population]
        final_best = float(max(pop_vals))
        final_mean = float(np.mean(pop_vals))
        final_worst = float(min(pop_vals))
    else:
        final_mean = final_best
        final_worst = final_best

    metadata = {
        "total_generations": int(total_generations),
        "final_best": final_best,
        "final_mean": final_mean,
        "final_worst": final_worst
    }

    best_genome = [float(x) for x in best_individual]
    weights = weights_from_list_to_matrix(best_genome, input_size, hidden_size, output_size)
    weights_lists = {k: v.tolist() for k, v in weights.items()}

    out_data = {
        "generations": generations_obj,
        "metadata": metadata,
        "weights": weights_lists
    }

    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)

    print(f"Saved JSON to {out_path}")
    return out_path


# -----------------------------
# Fitness evaluation
# -----------------------------

def fitness_eval_ind(ind, duration = 20):
    mujoco.set_mjcb_control(None)
    world = SimpleFlatWorld()#BoxyRugged()#TiltedFlatWorld() #SimpleFlatWorld()
    gecko_core = gecko()
    world.spawn(gecko_core.spec, spawn_position=[0, 0, 0])
    model = world.spec.compile()
    data = mujoco.MjData(model)
    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]


    input_size = len(data.qpos)
    hidden_size = 12
    output_size = model.nu


    weights = weights_from_list_to_matrix(ind, input_size, hidden_size, output_size)
    HISTORY = []
    controller = persistent_controller(to_track,weights, HISTORY)
    mujoco.set_mjcb_control(lambda m,d: controller(m, d))


    simple_runner(model=model,data=data,duration=duration)
    mujoco.set_mjcb_control(None)
    if len(HISTORY) < 2:
        return (0.0,)
    positions = np.array(HISTORY)


    dy = positions[-1,1] - positions[0,1] # forward progress
    dx = abs(positions[-1,0] - positions[0,0]) # sideways drift
    penalty_coeff = 1


    # ✅ reward forward movement, penalize sideways
    fitness = max(0, dy - (dx * penalty_coeff))


    return (fitness,)


# -----------------------------
# DEAP setup (safe)
# -----------------------------
try:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
except Exception:
    # already created in this session
    pass

try:
    creator.create("Individual", list, fitness=creator.FitnessMax)
except Exception:
    pass


# -----------------------------
# Main GA loop
# -----------------------------

def main():
    # Logging
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

    # RNG seeds for reproducibility
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)

    # GA hyperparameters
    NUM_GENERATIONS = 25
    POP_SIZE = 50
    MATE_CHANCE = 0.7
    MUTATE_CHANCE = 0.3
    ELITES = 2
    MUT_SIGMA = 0.3
    MAX_MUT_SIGMA = 3
    NO_IMPROVE = 0
    STAGNATION_WINDOW = 100
    BEST_FOR_NOW = -np.inf
    MAX_STAGNATION = 7

    # Debug flags
    DEBUG = False
    DEBUG_DURATION = 2
    DEFAULT_DURATION = 20
    eval_duration = DEBUG_DURATION if DEBUG else DEFAULT_DURATION

    # Create a small dummy world to compute sizes (safe pre-check)
    mujoco.set_mjcb_control(None)
    world = SimpleFlatWorld()
    gecko_core = gecko()
    world.spawn(gecko_core.spec, spawn_position=[0, 0, 0])
    model = world.spec.compile()
    data = mujoco.MjData(model)

    input_size = len(data.qpos)
    input_size_bias = input_size + 1

    hidden_size = 12
    output_size = model.nu

    genome_length = (input_size_bias * hidden_size) + (hidden_size * hidden_size) + (hidden_size * output_size)
    logging.info(f"Genome length computed = {genome_length}")

    # DEAP toolbox
    toolbox = base.Toolbox()
    toolbox.register("attr_param", lambda: random.uniform(-1, 1))
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_param, n=genome_length)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # register fitness with a duration parameter closure
    toolbox.register("evaluate_fitness", lambda ind: fitness_eval_ind(ind, duration=eval_duration))

    toolbox.register("mate", tools.cxUniform, indpb=0.5)

    # changing_sigma as wrapper mutate function
    def changing_sigma_wrapper(individual):
        nonlocal MUT_SIGMA
        return tools.mutGaussian(individual, mu=0.0, sigma=MUT_SIGMA, indpb=0.1)

    toolbox.register("mutate", changing_sigma_wrapper)
    toolbox.register("select", tools.selTournament, tournsize=6)

    # initialize population
    pop = toolbox.population(n=POP_SIZE)

    # initial evaluation
    logging.info("Evaluating initial population...")
    for i, ind in enumerate(pop):
        ind.fitness.values = toolbox.evaluate_fitness(ind)
        logging.debug(f"Init individual {i} fitness = {ind.fitness.values[0]}")

    best_individual_ever = tools.HallOfFame(1)
    fitness_history = []

    for gen in range(NUM_GENERATIONS):
        best = tools.selBest(pop, 1)[0]

        if best.fitness.values[0] <= BEST_FOR_NOW + 1e-8:
            NO_IMPROVE += 1
        else:
            NO_IMPROVE = 0
            BEST_FOR_NOW = best.fitness.values[0]

        if NO_IMPROVE >= STAGNATION_WINDOW and MUT_SIGMA < MAX_MUT_SIGMA:
            MUT_SIGMA *= 1.3
            logging.info(f"increasing sigma to get out of local minimum to {MUT_SIGMA}")
        else:
            MUT_SIGMA = max(0.05, MUT_SIGMA * 0.995)

        if NO_IMPROVE >= MAX_STAGNATION:
            logging.info(f"No improvement for {MAX_STAGNATION} generations. Stopping at generation {gen}.")
            break

        # selection and variation
        parents = toolbox.select(pop, POP_SIZE - ELITES)
        parents = list(map(copy.deepcopy, parents))
        random.shuffle(parents)

        new_children = []
        for parent1, parent2 in zip(parents[::2], parents[1::2]):
            child1, child2 = parent1, parent2
            if random.random() < MATE_CHANCE:
                toolbox.mate(child1, child2)
                if hasattr(child1, 'fitness'):
                    del child1.fitness.values
                if hasattr(child2, 'fitness'):
                    del child2.fitness.values
            if random.random() < MUTATE_CHANCE:
                toolbox.mutate(child1)
                toolbox.mutate(child2)
                if hasattr(child1, 'fitness'):
                    del child1.fitness.values
                if hasattr(child2, 'fitness'):
                    del child2.fitness.values
            new_children.extend([child1, child2])

        # evaluate invalid offspring
        invalid = [ind for ind in new_children if not hasattr(ind, 'fitness') or not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate_fitness(ind)

        elites = tools.selBest(pop, ELITES)
        total_population = pop + new_children
        survivors = toolbox.select(total_population, POP_SIZE - ELITES)
        pop[:] = survivors + elites

        best = tools.selBest(pop, 1)[0]
        fitness_history.append(best.fitness.values[0])
        best_individual_ever.update(pop)

        logging.info(f"generation {gen}: best fitness = {best.fitness.values[0]:.6f}")

    # end GA
    best_ind = best_individual_ever[0]

    out_path = save_results_json(
        output_folder="./results",
        filename_prefix="fitness_statistics",
        fitness_history=fitness_history,
        best_individual=best_ind,
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        final_population=pop
    )

    # visualize best agent
    HISTORY_BEST = []
    weights = weights_from_list_to_matrix(best_ind, input_size, hidden_size, output_size)

    # Spawn a fresh world for visualization
    mujoco.set_mjcb_control(None)
    world = SimpleFlatWorld()
    gecko_core = gecko()
    world.spawn(gecko_core.spec, spawn_position=[0, 0, 0])
    model = world.spec.compile()
    data = mujoco.MjData(model)
    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]

    actuator_limit = None
    try:
        if hasattr(model, 'actuator_ctrlrange'):
            arr = np.array(model.actuator_ctrlrange)
            actuator_limit = np.max(np.abs(arr), axis=1)
            if np.allclose(actuator_limit, actuator_limit[0]):
                actuator_limit = float(actuator_limit[0])
    except Exception:
        actuator_limit = None

    global_controller = persistent_controller(to_track, weights, HISTORY_BEST, actuator_limit=actuator_limit)
    mujoco.set_mjcb_control(lambda m, d: global_controller(m, d))

    # Launch viewer to inspect behaviour
    viewer.launch(model, data)

    # After closing viewer, plot and save results
    show_qpos_history(HISTORY_BEST)
    plot(fitness_history, len(fitness_history))


if __name__ == "__main__":
    main()
