import os, random, copy
import numpy as np
import mujoco
from mujoco import viewer
from deap import base, creator, tools

from ariel.simulation.environments import SimpleFlatWorld
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.utils.runners import simple_runner


# ------------------- SETTINGS -------------------
NUM_GENERATIONS = 50
POP_SIZE = 60
HIDDEN = 12
DURATION = 25.0
GAIT_FREQ = 2.8
MUT_SIGMA = 0.5
MUTATE_CHANCE = 0.4
MATE_CHANCE = 0.7
ELITES = 1
USE_VIEWER = True
# ------------------------------------------------


def build_robot():
    world = SimpleFlatWorld()
    r = gecko()
    world.spawn(r.spec, [0, 0, 0])
    model = world.spec.compile()
    data = mujoco.MjData(model)

    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    tracked = [data.bind(g) for g in geoms if "core" in g.name]
    if not tracked:
        tracked = [data.bind(geoms[0])]
    return model, data, tracked, len(data.qpos), len(data.qvel), model.nu


def weights_from_list(ind, inp, hid, out):
    i = 0
    W1 = np.array(ind[i:i + inp * hid]).reshape(inp, hid); i += inp * hid
    W2 = np.array(ind[i:i + hid * hid]).reshape(hid, hid); i += hid * hid
    W3 = np.array(ind[i:i + hid * out]).reshape(hid, out)
    return {"W1": W1, "W2": W2, "W3": W3}


def map_to_ctrl(model, u):
    u = np.clip(u, -1, 1)
    low = model.actuator_ctrlrange[:, 0]
    high = model.actuator_ctrlrange[:, 1]
    return low + (u + 1) * 0.5 * (high - low)


def make_controller(W, tracked, hist):
    def ctrl(model, data):
        phase = data.time * GAIT_FREQ
        phase1 = np.sin(phase)
        phase2 = np.sin(phase + np.pi/2)

        x = np.concatenate([data.qpos, data.qvel, [1.0, phase1, phase2]])
        h1 = np.tanh(x @ W["W1"])
        h2 = np.tanh(h1 @ W["W2"])
        out = np.tanh(h2 @ W["W3"])

        data.ctrl[:] = map_to_ctrl(model, out)
        hist.append(tracked[0].xpos.copy())
    return ctrl


# ---------- BLENDED CONTROLLER + FLIPPER GROUND CONTACT ----------
def make_controller_blend(W, tracked, hist, sign=1.0):
    def ctrl(model, data):
        phase = data.time * GAIT_FREQ
        n = model.nu
        base = np.sin(phase + np.linspace(0, np.pi, n, endpoint=False))

        x = np.concatenate([data.qpos, data.qvel, [1.0,
              np.sin(phase), np.sin(phase + np.pi/2)]])
        h1 = np.tanh(x @ W["W1"])
        h2 = np.tanh(h1 @ W["W2"])
        net = np.tanh(h2 @ W["W3"])

        flipper_bias = np.zeros_like(net)
        flipper_bias[:2] = -0.35  # lower front flippers for real ground contact

        u = sign * (0.7 * net + 0.3 * base + flipper_bias)
        data.ctrl[:] = map_to_ctrl(model, u)
        hist.append(tracked[0].xpos.copy())
    return ctrl
# -----------------------------------------------------------------


def rollout(model, data, tracked, W):
    hist=[]
    ctrl = make_controller(W, tracked, hist)
    mujoco.set_mjcb_control(lambda m,d: ctrl(m,d))
    try:
        simple_runner(model, data, duration=DURATION)
    finally:
        mujoco.set_mjcb_control(None)
    return np.array(hist)


# ------------------ AUTO-FORWARD FITNESS ------------------
def fitness(history):
    if len(history) < 2:
        return 0.0
    d = history[-1] - history[0]
    d[2] = 0.0

    ax = np.argmax(np.abs(d[:2]))
    forward = d[ax]
    if forward <= 0:
        return 0.0

    sideways = np.linalg.norm(d[:2]) - abs(forward)
    return max(0, forward - 2.0 * sideways)   # STRONGER STRAIGHTENING
# -----------------------------------------------------------


def probe_sign(W):
    model, data, tracked, *_ = build_robot()
    hist=[]
    ctrl = make_controller(W, tracked, hist)
    mujoco.set_mjcb_control(lambda m,d: ctrl(m,d))
    try:
        simple_runner(model, data, duration=3.0)
    finally:
        mujoco.set_mjcb_control(None)

    if len(hist) < 2:
        return 1.0

    d = hist[-1] - hist[0]
    d[2] = 0.0
    ax = np.argmax(np.abs(d[:2]))
    forward = d[ax]
    return 1.0 if forward >= 0 else -1.0


def main():
    random.seed(42); np.random.seed(42)
    model, data, tracked, qpos, qvel, nu = build_robot()
    inp = qpos + qvel + 3
    genome = inp*HIDDEN + HIDDEN*HIDDEN + HIDDEN*nu
    print("[INFO] genome length =", genome)

    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    tb = base.Toolbox()
    tb.register("gene", lambda: random.uniform(-1,1))
    tb.register("individual", tools.initRepeat, creator.Individual, tb.gene, n=genome)
    tb.register("population", tools.initRepeat, list, tb.individual)
    tb.register("evaluate", lambda ind: (fitness(rollout(model, data, tracked,
                    weights_from_list(ind, inp, HIDDEN, nu))),))
    tb.register("mate", tools.cxUniform, indpb=0.5)

    def mut(ind):
        tools.mutGaussian(ind, 0, MUT_SIGMA, 0.1)
        return ind

    tb.register("mutate", mut)
    tb.register("select", tools.selTournament, tournsize=6)

    pop = tb.population(n=POP_SIZE)
    for p in pop: p.fitness.values = tb.evaluate(p)

    hof = tools.HallOfFame(1)
    best_prev = -1
    no_imp = 0

    for g in range(NUM_GENERATIONS):
        best = tools.selBest(pop,1)[0]
        if best.fitness.values[0] <= best_prev + 1e-8:
            no_imp += 1
        else:
            best_prev = best.fitness.values[0]; no_imp = 0

        print(f"[GEN {g:02d}] best={best.fitness.values[0]:.3f}")
        if no_imp >= 6:
            break

        parents = list(map(copy.deepcopy, tb.select(pop, POP_SIZE-ELITES)))
        random.shuffle(parents)
        kids=[]
        for p1,p2 in zip(parents[::2], parents[1::2]):
            if random.random() < MATE_CHANCE:
                tb.mate(p1,p2); del p1.fitness.values,p2.fitness.values
            if random.random() < MUTATE_CHANCE:
                tb.mutate(p1); tb.mutate(p2)
                del p1.fitness.values,p2.fitness.values
            kids += [p1,p2]

        invalid = [k for k in kids if not k.fitness.valid]
        for k in invalid:
            k.fitness.values = tb.evaluate(k)

        elites = tools.selBest(pop, ELITES)
        pop = tb.select(pop+kids, POP_SIZE-ELITES) + elites
        hof.update(pop)

    best = hof[0]
    W = weights_from_list(best, inp, HIDDEN, nu)
    sign = probe_sign(W)

    model, data, tracked, *_ = build_robot()
    hist=[]
    ctrl = make_controller_blend(W, tracked, hist, sign=sign)
    mujoco.set_mjcb_control(lambda m,d: ctrl(m,d))

    print("[VIEWER] Press SPACE to start gait.")
    viewer.launch(model, data)
    mujoco.set_mjcb_control(None)
    print("[DONE]")


if __name__ == "__main__":
    main()