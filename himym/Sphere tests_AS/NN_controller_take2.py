"""
NN_controller_take5.py
Evolve a neural controller that makes the Gecko move on a spherical world.
- Stable physics + safety checks
- Expressive inputs (qpos, qvel, CoM, time, noise)
- Directional fitness (forward CoM-x)
- Smooth servo control with clipping
- Records 30s video of the best individual and opens viewer
"""

import os, random, json
from datetime import datetime
import numpy as np
import mujoco
from mujoco import viewer
from deap import base, creator, tools

from ariel.simulation.environments.spherical_world import SphericalWorld
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.utils.video_recorder import VideoRecorder
from ariel.utils.renderers import video_renderer


# -----------------------------
# Reproducibility (tweak if you like)
# -----------------------------
random.seed(42)
np.random.seed(42)


# -----------------------------
# Neural controller (1 hidden layer, tanh)
# -----------------------------
class NeuralController:
    def __init__(self, input_size, hidden_size, output_size, weights):
        self.i = input_size
        self.h = hidden_size
        self.o = output_size
        self.W1, self.b1, self.W2, self.b2 = self._decode(weights)

    def _decode(self, genome):
        i, h, o = self.i, self.h, self.o
        idx = 0
        W1 = np.array(genome[idx: idx + i*h]).reshape(i, h); idx += i*h
        b1 = np.array(genome[idx: idx + h]); idx += h
        W2 = np.array(genome[idx: idx + h*o]).reshape(h, o); idx += h*o
        b2 = np.array(genome[idx: idx + o])
        return W1, b1, W2, b2

    def forward(self, x):
        h = np.tanh(x @ self.W1 + self.b1)
        y = np.tanh(h @ self.W2 + self.b2)  # in [-1, 1]
        return y


# -----------------------------
# Build world + robot (helper)
# -----------------------------
def build_world_and_robot(spawn_gap=0.05):
    world = SphericalWorld(radius=5.0, radial_gravity=True)
    g = gecko()
    spec = g.spec if hasattr(g, "spec") else g
    world.spawn(spec, spawn_position=[0, 0, world.radius + spawn_gap])
    model = world.spec.compile()
    world.attach_model(model)
    data = mujoco.MjData(model)
    return world, model, data


# -----------------------------
# Safe radial gravity (+instability detect)
# -----------------------------
def apply_radial_gravity(world, model, data):
    gmag = abs(world.gravity)
    for i in range(model.nbody):
        pos = data.xipos[i]
        if not np.all(np.isfinite(pos)):
            return False
        dist = np.linalg.norm(pos)
        if dist < 1e-6:
            continue
        data.xfrc_applied[i, :3] += -pos / dist * model.body_mass[i] * gmag
    return True


# -----------------------------
# Evaluation function
# -----------------------------
def evaluate_genome(genome):
    # Build a fresh world every evaluation (keeps states independent)
    world, model, data = build_world_and_robot(spawn_gap=0.06)

    # Physics stabilization
    model.opt.timestep = 0.0015
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICIT
    model.dof_damping[:] = 3.0
    model.geom_friction[:] = np.array([0.8, 0.04, 0.01])  # slide, spin, roll

    # Reset all dynamic data to a clean slate
    mujoco.mj_resetData(model, data)

    # Controller + sim params
    steps       = 2000
    ctrl_limit  = 1.0        # radians (servo targets)
    n_act       = model.nu
    obs_dim     = model.nq + model.nv + 3 + 1 + 3  # qpos, qvel, CoM(3), time(1), noise(3)
    controller  = NeuralController(obs_dim, hidden_size=24, output_size=n_act, weights=genome)

    # Smooth control buffer (simple low-pass)
    smooth_ctrl = np.zeros(n_act, dtype=np.float64)

    # Metrics
    start_com   = np.copy(data.subtree_com[0])
    energy_trace = []

    try:
        for step in range(steps):
            # Compose observation
            time_signal = np.sin(2 * np.pi * 0.6 * (step / steps))  # 0.6 Hz-ish over episode
            noise       = np.random.uniform(-0.03, 0.03, size=3)

            obs = np.concatenate([
                np.clip(data.qpos, -1.0, 1.0),
                np.clip(data.qvel, -6.0, 6.0),
                np.clip(data.subtree_com[0], -6.0, 6.0),
                [time_signal],
                noise
            ], dtype=np.float64)

            # Forward pass
            act = controller.forward(obs)
            if not np.all(np.isfinite(act)):
                return (0.0,)

            # Smooth + clip (servo targets)
            target = np.clip(act, -ctrl_limit, ctrl_limit)
            smooth_ctrl = 0.9 * smooth_ctrl + 0.1 * target
            data.ctrl[:] = smooth_ctrl

            # Gravity + step
            if not apply_radial_gravity(world, model, data):
                return (0.0,)

            mujoco.mj_step(model, data)

            # Early abort on instability
            if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.xipos)):
                return (0.0,)

            # Energy proxy
            energy_trace.append(float(np.dot(data.qvel, data.qvel)))
    except Exception:
        return (0.0,)

    end_com = np.copy(data.subtree_com[0])
    forward_disp = float(end_com[0] - start_com[0])  # move along +X
    energy_var   = float(np.var(energy_trace)) if energy_trace else 0.0

    # Directional fitness + mild energy penalty
    fitness = forward_disp - 0.08 * np.sqrt(max(energy_var, 0.0))
    if not np.isfinite(fitness):
        fitness = 0.0
    return (fitness,)


# -----------------------------
# Evolutionary algorithm
# -----------------------------
def evolve():
    # Determine genome length from controller topology
    input_size  = 15 + 14 + 3 + 1 + 3  # nq(15) + nv(14) + com(3) + time(1) + noise(3)
    hidden_size = 24
    n_act       = 8

    genome_len = input_size*hidden_size + hidden_size + hidden_size*n_act + n_act

    # DEAP setup (guard against re-creation)
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, -0.5, 0.5)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=genome_len)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_genome)
    toolbox.register("mate", tools.cxBlend, alpha=0.4)
    toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.25, indpb=0.15)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop_size = 16
    gens     = 5
    pop      = toolbox.population(n=pop_size)

    best_fit = -np.inf
    best_ind = None
    history  = []

    for g in range(gens):
        # Evaluate current pop
        fits = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fits):
            ind.fitness.values = fit

        # Make offspring (clone → cx → mut)
        offspring = [toolbox.clone(ind) for ind in pop]

        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.6:
                toolbox.mate(c1, c2)
                if hasattr(c1.fitness, "values"): del c1.fitness.values
                if hasattr(c2.fitness, "values"): del c2.fitness.values

        for m in offspring:
            if random.random() < 0.35:
                toolbox.mutate(m)
                if hasattr(m.fitness, "values"): del m.fitness.values

        pop[:] = offspring

        # Re-evaluate only invalid
        invalid = [ind for ind in pop if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)

        # Stats
        vals = [i.fitness.values[0] for i in pop]
        mean_fit = float(np.mean(vals)) if vals else 0.0
        history.append(mean_fit)

        best_now = tools.selBest(pop, 1)[0]
        if best_now.fitness.values[0] > best_fit:
            best_fit = best_now.fitness.values[0]
            best_ind = best_now

        print(f"=== Generation {g+1}/{gens} ===")
        print(f"  Best fitness: {best_fit:.4f}")
        print(f"  Mean fitness: {mean_fit:.4f}")
        print("-------------------------------")

    print("\n=== Evolution Complete ===")
    print(f"Best final fitness: {best_fit:.4f}")

    # Save history (optional)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"evolution_history_{ts}.json", "w") as f:
        json.dump({"mean_fitness": history, "best_fitness": best_fit}, f, indent=2)

    return best_ind, history


# -----------------------------
# Replay + video of best
# -----------------------------
def visualize(best_genome):
    print("\nReplaying best individual (recording 30 s)...")
    world, model, data = build_world_and_robot(spawn_gap=0.06)

    # Physics settings (same as eval)
    model.opt.timestep   = 0.0015
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICIT
    model.dof_damping[:] = 3.0
    model.geom_friction[:] = np.array([0.8, 0.04, 0.01])
    mujoco.mj_resetData(model, data)

    # Controller
    ctrl_limit  = 1.0
    input_size  = 15 + 14 + 3 + 1 + 3
    controller  = NeuralController(input_size, 24, model.nu, best_genome)
    smooth_ctrl = np.zeros(model.nu, dtype=np.float64)

    # Step some time (so final pose isn’t just the start)
    steps = int(10.0 / model.opt.timestep)
    for step in range(steps):
        time_signal = np.sin(2 * np.pi * 0.6 * (step / steps))
        noise       = np.random.uniform(-0.03, 0.03, size=3)
        obs = np.concatenate([
            np.clip(data.qpos, -1.0, 1.0),
            np.clip(data.qvel, -6.0, 6.0),
            np.clip(data.subtree_com[0], -6.0, 6.0),
            [time_signal],
            noise
        ], dtype=np.float64)

        act = controller.forward(obs)
        target = np.clip(act, -ctrl_limit, ctrl_limit)
        smooth_ctrl = 0.9 * smooth_ctrl + 0.1 * target
        data.ctrl[:] = smooth_ctrl

        apply_radial_gravity(world, model, data)
        mujoco.mj_step(model, data)

    # Record 30s from current state
    os.makedirs("evolved_videos", exist_ok=True)
    video_rec = VideoRecorder(output_folder="evolved_videos")
    video_renderer(model, data, duration=30, video_recorder=video_rec)
    print("✅ Video saved under evolved_videos/")

    # Launch viewer
    print("Launching viewer (close manually)...")
    with viewer.launch_passive(model, data) as v:
        v.cam.lookat[:] = [0, 0, 0]
        v.cam.distance = world.radius * 3.0
        v.cam.elevation = -20
        v.cam.azimuth = 120
        while v.is_running():
            v.sync()


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    best, hist = evolve()
    visualize(best)
