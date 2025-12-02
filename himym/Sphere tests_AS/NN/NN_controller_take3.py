"""
gecko_evolution_train.py
------------------------

End-to-end neuroevolution for the Gecko on SphericalWorld.

Pipeline:
1. Diagnostics: joints, actuators, DOF mapping.
2. PD + CPG baseline that drives the joints.
3. Neural controller that modulates PD targets for each actuator.
4. DEAP evolution with displacement-based fitness.
5. Visualization of best individual in a live MuJoCo viewer, with
   a correctly-timed video (VIS_TIME seconds, at VIDEO_FPS).

Relies on:
- SphericalWorld (your current working implementation)
- gecko() phenotype
- torques via data.ctrl[:] (gaintype=0, biastype=1).
"""

import os
import json
import random
from datetime import datetime

import numpy as np
import mujoco
from mujoco import viewer
from deap import base, creator, tools

from ariel.simulation.environments.spherical_world import SphericalWorld
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.utils.video_recorder import VideoRecorder


# =========================================================
# GLOBAL CONFIG
# =========================================================

RADIUS = 5.0
TIMESTEP = 0.0015
DAMPING = 3.0
WORLD_FRICTION = np.array([0.8, 0.05, 0.01])

EVAL_TIME = 10.0       # seconds per fitness evaluation
VIS_TIME = 20.0        # seconds for visualization

KP = 10.0              # PD proportional gain (soft)
KD = 0.5               # PD derivative gain
BASE_AMP = 0.35        # baseline CPG amplitude (radians)
BASE_FREQ = 1.0        # baseline CPG frequency (Hz)
NN_DELTA_SCALE = 0.4   # max NN delta in radians
TAU_LIMIT = 0.6        # torque clamp for data.ctrl

POP_SIZE = 60
N_GENS = 80
HIDDEN = 24            # hidden units in neural controller                              

# Video config
VIDEO_WIDTH = 720
VIDEO_HEIGHT = 720
VIDEO_FPS = 30


# =========================================================
# UTILS: SAFE NAMING & DOF COUNT
# =========================================================

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


# =========================================================
# BUILD WORLD + DIAGNOSTICS
# =========================================================

def build_world():
    """Create SphericalWorld + Gecko, compile and warm-settle."""
    world = SphericalWorld(radius=RADIUS, radial_gravity=True)
    g = gecko()
    spec = g.spec if hasattr(g, "spec") else g
    world.spawn(spec)  # your working spawn()

    model = world.spec.compile()
    world.attach_model(model)
    model.opt.timestep = TIMESTEP
    model.dof_damping[:] = DAMPING
    model.geom_friction[:] = WORLD_FRICTION

    data = mujoco.MjData(model)

    # Warm settle (no control, just gravity) so contacts are stable
    for _ in range(200):
        world.apply_radial_gravity(data)
        mujoco.mj_step(model, data)

    return world, model, data


def print_diagnostics():
    """Print joint, actuator, and DOF mapping info (once)."""
    world, model, data = build_world()
    print("=== Gecko Joint & Actuator Diagnostics (Evolution Script) ===\n")
    print(f"nq={model.nq}, nv={model.nv}, nu={model.nu}, njnt={model.njnt}\n")

    # Joints
    print("---- Joints ----")
    for j in range(model.njnt):
        name = safe_name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        jtype = model.jnt_type[j]
        jadr = model.jnt_dofadr[j]
        ndof = joint_dof_count(jtype)
        rng = model.jnt_range[j]
        jtype_str = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}.get(jtype, "?")
        print(f"[{j:2d}] {name:28s} type={jtype_str:<6s} dofadr={jadr:<2d} ndof={ndof} range={rng}")

    # Actuators
    print("\n---- Actuators ----")
    for i in range(model.nu):
        aname = safe_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        trnid = model.actuator_trnid[i]
        j = int(trnid[0])
        jname = safe_name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        gt = model.actuator_gaintype[i]
        bt = model.actuator_biastype[i]
        cr = model.actuator_ctrlrange[i]
        gear = model.actuator_gear[i]
        print(
            f"[{i:2d}] {aname:25s} → joint[{j:2d}] {jname:25s} "
            f"gaintype={gt:<2d} biastype={bt:<2d} ctrl={cr} gear={gear}"
        )

    # DOF map
    print("\n---- DOF Address Mapping (qpos indices) ----")
    for j in range(model.njnt):
        dofadr = model.jnt_dofadr[j]
        jtype = model.jnt_type[j]
        ndof = joint_dof_count(jtype)
        if ndof > 0:
            name = safe_name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            print(f"joint[{j:2d}] '{name}' -> qpos[{dofadr}:{dofadr+ndof}]")

    print("\nDiagnostics complete.\n")


def actuator_joint_maps(model):
    """
    Build:
    - act2joint: list of len nu, mapping actuator i -> joint index j
    - joint2dof: dict j -> qpos index (first DOF)
    """
    act2joint = []
    for i in range(model.nu):
        trnid = model.actuator_trnid[i]
        j = int(trnid[0])
        act2joint.append(j)

    joint2dof = {}
    for j in range(model.njnt):
        jadr = model.jnt_dofadr[j]
        joint2dof[j] = jadr

    return act2joint, joint2dof


# =========================================================
# PD + BASELINE CPG
# =========================================================

def cpg_baseline_angle(t: float, joint_index: int) -> float:
    """
    Baseline CPG target for a given joint (in radians).

    Joint indices from diagnostics:
      0: free root
      1: neck
      2: spine
      3: bl_leg
      4: br_leg
      5: fl_leg
      6: fl_flipper
      7: fr_leg
      8: fr_flipper
    """
    w = 2.0 * np.pi * BASE_FREQ
    A = BASE_AMP

    # Neck
    if joint_index == 1:
        return 0.3 * A * np.sin(w * t)

    # Spine
    if joint_index == 2:
        return 0.4 * A * np.sin(w * t + np.pi / 2.0)

    # Back legs: 3 (BL), 4 (BR)
    if joint_index in (3, 4):
        phase = 0.0 if joint_index == 3 else np.pi
        return A * np.sin(w * t + phase)

    # Front legs: 5 (FL), 7 (FR) – opposite phase to backs
    if joint_index in (5, 7):
        phase = np.pi if joint_index == 5 else 0.0
        return A * np.sin(w * t + phase)

    # Flippers: 6 (FL), 8 (FR) – higher frequency tap
    if joint_index in (6, 8):
        phase = 0.0 if joint_index == 6 else np.pi
        return 0.6 * A * np.sin(2.0 * w * t + phase)

    # Root or unknown
    return 0.0


def pd_torques_from_targets(model, data, t, act2joint, joint2dof, nn_delta=None):
    """
    Compute torques for each actuator from PD targets.

    - act2joint: actuator -> joint index
    - joint2dof: joint -> qpos index
    - nn_delta: optional np.ndarray of shape (nu,), in [-1,1].
                We scale to [-NN_DELTA_SCALE, NN_DELTA_SCALE] radians.
    """
    qpos = data.qpos.copy()
    qvel = data.qvel.copy()

    nu = model.nu
    torques = np.zeros(nu, dtype=np.float32)

    if nn_delta is None:
        nn_delta = np.zeros(nu, dtype=np.float32)

    nn_delta = np.clip(nn_delta, -1.0, 1.0) * NN_DELTA_SCALE

    for i in range(nu):
        j = act2joint[i]
        if j <= 0:
            # ignore root/invalid
            continue
        dof_idx = joint2dof[j]

        base_target = cpg_baseline_angle(t, j)
        target = base_target + float(nn_delta[i])

        angle = qpos[dof_idx]
        vel = qvel[dof_idx]
        tau = KP * (target - angle) - KD * vel
        torques[i] = np.clip(tau, -TAU_LIMIT, TAU_LIMIT)

    return torques


# =========================================================
# NEURAL CONTROLLER
# =========================================================

class NeuralController:
    """Simple 2-layer MLP with tanh activations."""

    def __init__(self, inp, hid, out, weights):
        self.i = inp
        self.h = hid
        self.o = out
        self.W1, self.b1, self.W2, self.b2 = self._decode(weights)

    def _decode(self, genome):
        i, h, o = self.i, self.h, self.o
        idx = 0
        W1 = np.array(genome[idx: idx + i * h], dtype=np.float32).reshape(i, h)
        idx += i * h
        b1 = np.array(genome[idx: idx + h], dtype=np.float32)
        idx += h
        W2 = np.array(genome[idx: idx + h * o], dtype=np.float32).reshape(h, o)
        idx += h * o
        b2 = np.array(genome[idx: idx + o], dtype=np.float32)
        return W1, b1, W2, b2

    def forward(self, x):
        h = np.tanh(x @ self.W1 + self.b1)
        y = np.tanh(h @ self.W2 + self.b2)
        return y


# =========================================================
# FITNESS EVALUATION
# =========================================================

def evaluate_genome(genome):
    """
    Evaluate a genome by running PD + NN for EVAL_TIME s and measuring
    horizontal CoM displacement.
    """
    world, model, data = build_world()
    mujoco.mj_resetData(model, data)

    act2joint, joint2dof = actuator_joint_maps(model)

    # Observation: qpos, qvel, CoM(3), time_sin, time_cos
    obs_dim = model.nq + model.nv + 3 + 2
    controller = NeuralController(obs_dim, HIDDEN, model.nu, genome)

    steps = int(EVAL_TIME / model.opt.timestep)
    start_com = np.copy(data.subtree_com[0])
    nan_hits = 0

    for step in range(steps):
        t = step * model.opt.timestep

        time_sin = np.sin(2.0 * np.pi * 0.5 * t)
        time_cos = np.cos(2.0 * np.pi * 0.5 * t)

        obs = np.concatenate([
            np.clip(data.qpos, -2.0, 2.0),
            np.clip(data.qvel, -10.0, 10.0),
            np.clip(data.subtree_com[0], -RADIUS, RADIUS),
            np.array([time_sin, time_cos], dtype=np.float32)
        ]).astype(np.float32)

        nn_out = controller.forward(obs)  # [-1,1]
        torques = pd_torques_from_targets(
            model, data, t, act2joint, joint2dof, nn_delta=nn_out
        )

        data.ctrl[:] = torques
        world.apply_radial_gravity(data)
        mujoco.mj_step(model, data)

        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            nan_hits += 1
            if nan_hits > 3:
                return (0.0,)

    end_com = np.copy(data.subtree_com[0])
    disp_vec = end_com[:2] - start_com[:2]
    disp_xy = float(np.linalg.norm(disp_vec))

    if end_com[2] < 0.0:
        disp_xy *= 0.1

    if not np.isfinite(disp_xy):
        disp_xy = 0.0

    return (disp_xy,)


# =========================================================
# EVOLUTION
# =========================================================

def setup_deap(genome_len):
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, -0.5, 0.5)
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attr_float,
        n=genome_len,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_genome)
    toolbox.register("mate", tools.cxBlend, alpha=0.4)
    toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.25, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("clone", lambda ind: creator.Individual(ind))
    return toolbox


def evolve():
    # Determine genome length from obs + NN shape
    world, model, data = build_world()
    obs_dim = model.nq + model.nv + 3 + 2
    n_act = model.nu

    genome_len = obs_dim * HIDDEN + HIDDEN + HIDDEN * n_act + n_act

    toolbox = setup_deap(genome_len)
    pop = toolbox.population(n=POP_SIZE)

    best_fit = -1e9
    best_ind = None
    mean_hist = []
    best_hist = []

    for gen in range(N_GENS):
        # Evaluate (or re-evaluate) individuals
        for ind in pop:
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind)

        fits = [ind.fitness.values[0] for ind in pop]
        mean_fit = float(np.mean(fits)) if fits else 0.0
        best_now = max(pop, key=lambda ind: ind.fitness.values[0])

        if best_now.fitness.values[0] > best_fit:
            best_fit = best_now.fitness.values[0]
            best_ind = toolbox.clone(best_now)

        mean_hist.append(mean_fit)
        best_hist.append(best_fit)

        print(f"=== Gen {gen+1}/{N_GENS} ===")
        print(f"  Best so far: {best_fit:.4f} | Mean: {mean_fit:.4f}")
        print("-------------------------")

        # Selection
        offspring = toolbox.select(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        # Crossover
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.6:
                toolbox.mate(c1, c2)
                del c1.fitness.values, c2.fitness.values

        # Mutation
        for m in offspring:
            if random.random() < 0.3:
                toolbox.mutate(m)
                del m.fitness.values

        pop[:] = offspring

    print("\n=== Evolution complete ===")
    print(f"Best final fitness: {best_fit:.4f}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    hist_path = f"evolution_history_{ts}.json"
    with open(hist_path, "w") as f:
        json.dump(
            {"mean_fitness": mean_hist, "best_fitness": best_hist},
            f,
            indent=2,
        )
    print(f"History saved to {hist_path}")

    return best_ind


# =========================================================
# VISUALIZATION OF BEST INDIVIDUAL + VIDEO
# =========================================================

def visualize(best):
    """Replay best individual with live viewer and correctly-timed video."""
    print("\n🎬 Replaying best individual (with video)...")
    world, model, data = build_world()
    mujoco.mj_resetData(model, data)

    act2joint, joint2dof = actuator_joint_maps(model)
    obs_dim = model.nq + model.nv + 3 + 2
    controller = NeuralController(obs_dim, HIDDEN, model.nu, best)

    steps = int(VIS_TIME / model.opt.timestep)
    start_com = np.copy(data.subtree_com[0])

    # --- Set up video recording ---
    os.makedirs("evolution_videos", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"gecko_best_{ts}"
    print(f"Video will be saved under evolution_videos/ with base name '{base_name}'")

    # Offscreen renderer (new API: render() takes no data argument)
    renderer = mujoco.Renderer(model, VIDEO_WIDTH, VIDEO_HEIGHT)

    # Video recorder using ariel's implementation
    video_rec = VideoRecorder(
        file_name=base_name,
        output_folder="evolution_videos",
        width=VIDEO_WIDTH,
        height=VIDEO_HEIGHT,
        fps=VIDEO_FPS,
    )

    # Compute how many sim steps correspond to one video frame so that
    # VIS_TIME seconds → ~VIS_TIME * VIDEO_FPS frames.
    dt = model.opt.timestep
    frame_skip = max(1, int(round(1.0 / (VIDEO_FPS * dt))))

    with viewer.launch_passive(model, data) as v:
        v.cam.lookat[:] = [0.0, 0.0, RADIUS]
        v.cam.distance = 10.0
        v.cam.azimuth = 140
        v.cam.elevation = -25

        print(f"Running for {VIS_TIME:.1f} s...")
        for step in range(steps):
            t = step * model.opt.timestep

            time_sin = np.sin(2.0 * np.pi * 0.5 * t)
            time_cos = np.cos(2.0 * np.pi * 0.5 * t)

            obs = np.concatenate([
                np.clip(data.qpos, -2.0, 2.0),
                np.clip(data.qvel, -10.0, 10.0),
                np.clip(data.subtree_com[0], -RADIUS, RADIUS),
                np.array([time_sin, time_cos], dtype=np.float32)
            ]).astype(np.float32)

            nn_out = controller.forward(obs)
            torques = pd_torques_from_targets(
                model, data, t, act2joint, joint2dof, nn_delta=nn_out
            )

            data.ctrl[:] = torques
            world.apply_radial_gravity(data)
            mujoco.mj_step(model, data)

            # Record at VIDEO_FPS real-time
            if step % frame_skip == 0:
                renderer.update_scene(data)
                rgb = renderer.render()
                video_rec.write(rgb)

            v.sync()

        # End-of-run displacement
        end_com = np.copy(data.subtree_com[0])
        disp_vec = end_com[:2] - start_com[:2]
        disp_xy = float(np.linalg.norm(disp_vec))
        print(f"\nSurface-ish (XY) displacement over {VIS_TIME:.1f}s: {disp_xy:.3f} m")

    # Finalize video
    video_rec.release()
    print("✅ Video recording finished.")
    print("Viewer will remain open. Press ESC to close.")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    # 1) Show diagnostics ONCE so we know everything is wired correctly
    print_diagnostics()

    # 2) Evolve controllers
    best_individual = evolve()

    # 3) Visualize the best + record video
    visualize(best_individual)
