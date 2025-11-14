import mujoco
from ariel.simulation.environments import SimpleFlatWorld
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko

world = SimpleFlatWorld()
robot = gecko()
world.spawn(robot.spec, [0,0,0])
model = world.spec.compile()
data = mujoco.MjData(model)

# Apply forward kinematics once so orientation is valid
mujoco.mj_forward(model, data)

geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
tracked = [data.bind(g) for g in geoms if "core" in g.name]

print("Core rotation matrix (world frame):")
print(tracked[0].xmat.reshape(3,3))
