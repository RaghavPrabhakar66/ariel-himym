import mujoco
from ariel.simulation.environments import SimpleFlatWorld
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko

# Build world + robot
world = SimpleFlatWorld()
g = gecko()
world.spawn(g.spec, spawn_position=[0, 0, 0])
model = world.spec.compile()

print("Actuator ctrlrange:")
print(model.actuator_ctrlrange)