# Gecko Evolution - Forward Gait, No Rotation (FIXED)
# ===============================================
import numpy as np
import mujoco
from mujoco import viewer
import random, copy
from deap import base, creator, tools

from ariel.simulation.environments import SimpleFlatWorld
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.utils.runners import simple_runner

# ---------------- Parameters --------------------
HIDDEN = 12
DURATION = 20.0         # longer so movement is measurable
GAIT_FREQ = 3.0

SIDEWAYS_PENALTY   = 2.0
YAW_RATE_PENALTY   = 1.2
TURN_IMBAL_PENALTY = 0.6

# ---------------- Builders -----------------------
def build_robot():
    world = SimpleFlatWorld()
    r = gecko()
    world.spawn(r.spec, spawn_position=[0, 0, 0])
    model = world.spec.compile()
    data  = mujoco.MjData(model)

    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    tracked = [data.bind(g) for g in geoms if "core" in g.name]
    if not tracked:
        tracked = [data.bind(geoms[0])]
    return world, model, data, tracked

# ---------------- Weights ------------------------
def weights_from_list(ind, inp, hid, out):
    ind = np.asarray(ind, dtype=np.float32)
    i = 0
    W1 = ind[i:i+(inp+3)*hid].reshape((inp+3, hid)); i += (inp+3)*hid
    W2 = ind[i:i+hid*hid].reshape((hid, hid));        i += hid*hid
    W3 = ind[i:i+hid*out].reshape((hid, out))
    return {"W1": W1, "W2": W2, "W3": W3}

def map_to_ctrl(model, u):
    u  = np.clip(u, -1, 1)
    lo = model.actuator_ctrlrange[:,0]
    hi = model.actuator_ctrlrange[:,1]
    return lo + (u+1)*0.5*(hi-lo)

# ---------------- Rollout (fresh sim per eval) ----
def rollout(W):
    # fresh world/model/data each time -> no state leakage
    _, model, data, tracked = build_robot()
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    pos_hist, fwd_hist, u_hist = [], [], []

    def ctrl(model, data):
        phase = data.time * GAIT_FREQ
        x  = np.concatenate([data.qpos, data.qvel, [1.0, np.sin(phase), np.sin(phase+np.pi/2)]])
        h1 = np.tanh(x @ W["W1"])
        h2 = np.tanh(h1 @ W["W2"])
        net = np.tanh(h2 @ W["W3"])

        data.ctrl[:] = map_to_ctrl(model, net)

        pos_hist.append(tracked[0].xpos.copy())
        R = tracked[0].xmat.reshape(3,3)
        fwd_hist.append(R[:,1].copy())   # body +Y = forward for this morphology
        u_hist.append(net.copy())

    mujoco.set_mjcb_control(lambda m,d: ctrl(m,d))
    try:
        simple_runner(model, data, duration=DURATION)
    finally:
        mujoco.set_mjcb_control(None)

    return np.array(pos_hist), np.array(fwd_hist), np.array(u_hist)

# ---------------- Fitness -------------------------
def _yaw_from_xy(v):
    xy = np.array(v[:2])
    n = np.linalg.norm(xy)
    if n < 1e-8: return 0.0
    xy /= n
    return float(np.arctan2(xy[1], xy[0]))

def fitness(pos_hist, fwd_hist, u_hist):
    if len(pos_hist) < 3:
        return 0.0

    # displacement in XY
    disp = pos_hist[-1] - pos_hist[0]
    disp[2] = 0.0
    disp_xy = disp[:2]

    # average normalized forward (body +Y) over the rollout
    dirs = np.array(fwd_hist, dtype=np.float64)
    for i in range(len(dirs)):
        n = np.linalg.norm(dirs[i][:2])
        if n > 1e-8:
            dirs[i][:2] /= n
        else:
            dirs[i] = np.array([0,1,0], dtype=np.float64)
    fwd_avg = dirs.mean(axis=0)
    n = np.linalg.norm(fwd_avg[:2])
    if n < 1e-8:
        fwd_avg = np.array([0,1,0], dtype=np.float64)
    else:
        fwd_avg[:2] /= n

    # rotation-proof forward progress: projection of displacement along fwd_avg
    forward = float(np.dot(disp_xy, fwd_avg[:2]))

    # allow learning even if early controllers step slightly "backwards" by noise
    forward = max(0.0, forward)

    # sideways drift = residual after removing forward component
    sideways_vec = disp_xy - forward * fwd_avg[:2]
    sideways = float(np.linalg.norm(sideways_vec))

    # integrated yaw-rate penalty
    yaws  = np.array([_yaw_from_xy(v) for v in dirs])
    dyaws = np.arctan2(np.sin(np.diff(yaws)), np.cos(np.diff(yaws)))
    yaw_cost = float(np.sum(np.abs(dyaws))) * YAW_RATE_PENALTY

    # left–right imbalance penalty (actuator indices: 2,3,6 vs 4,5,7)
    if u_hist.ndim == 2 and u_hist.shape[1] >= 8:
        u_left  = u_hist[:, [2,3,6]].mean(axis=1)
        u_right = u_hist[:, [4,5,7]].mean(axis=1)
        turn_cost = float(np.mean(np.abs(u_left - u_right))) * TURN_IMBAL_PENALTY
    else:
        turn_cost = 0.0

    return max(0.0, forward - SIDEWAYS_PENALTY*sideways - yaw_cost - turn_cost)

# ---------------- Evolution ------------------------
# Build one robot only to size the genome (don’t reuse for evals)
_, model0, data0, _ = build_robot()
inp = len(data0.qpos) + len(data0.qvel)
nu  = model0.nu
GENOME = (inp+3)*HIDDEN + HIDDEN*HIDDEN + HIDDEN*nu
print("[INFO] genome length =", GENOME)

if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)

tb = base.Toolbox()
tb.register("gene", lambda: random.uniform(-1,1))
tb.register("individual", tools.initRepeat, creator.Individual, tb.gene, GENOME)
tb.register("population", tools.initRepeat, list, tb.individual)

def eval_ind(ind):
    W = weights_from_list(ind, inp, HIDDEN, nu)
    pos, fwd, u = rollout(W)
    return (fitness(pos, fwd, u),)

tb.register("evaluate", eval_ind)
tb.register("mate", tools.cxUniform, indpb=0.5)
tb.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.35, indpb=0.08)
tb.register("select", tools.selTournament, tournsize=5)

def evolve():
    pop = tb.population(n=40)
    for ind in pop:
        ind.fitness.values = tb.evaluate(ind)

    for gen in range(30):
        offspring = list(map(copy.deepcopy, tb.select(pop, len(pop))))
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.7:
                tb.mate(c1, c2); del c1.fitness.values, c2.fitness.values
            if random.random() < 0.3:
                tb.mutate(c1); del c1.fitness.values
                tb.mutate(c2); del c2.fitness.values

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = tb.evaluate(ind)

        pop = tools.selBest(pop + offspring, 40)
        best = max(pop, key=lambda x: x.fitness.values[0])
        print(f"[GEN {gen:02d}] best={best.fitness.values[0]:.3f}")
    return max(pop, key=lambda x: x.fitness.values[0])

# ---------------- Viewer ---------------------------
def view(best):
    # fresh viewer world
    _, model, data, tracked = build_robot()
    W = weights_from_list(best, inp, HIDDEN, nu)

    def ctrl(model, data):
        phase = data.time * GAIT_FREQ
        x  = np.concatenate([data.qpos, data.qvel, [1.0, np.sin(phase), np.sin(phase+np.pi/2)]])
        h1 = np.tanh(x @ W["W1"])
        h2 = np.tanh(h1 @ W["W2"])
        net = np.tanh(h2 @ W["W3"])

        # plant both front flippers (slightly stronger on right to fix floating)
        if net.shape[0] >= 6:
            net[3] -= 0.35   # FL flipper
            net[5] -= 0.42   # FR flipper

        data.ctrl[:] = map_to_ctrl(model, net)

    mujoco.set_mjcb_control(lambda m,d: ctrl(m,d))
    print("[VIEWER] Press SPACE to start gait.")
    viewer.launch(model, data)
    mujoco.set_mjcb_control(None)

if __name__ == "__main__":
    random.seed(42); np.random.seed(42)
    best = evolve()
    view(best)