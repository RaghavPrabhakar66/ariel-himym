from ariel.simulation.environments.spherical_world import SphericalWorld
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
import mujoco

world = SphericalWorld(radius=5.0, radial_gravity=True)
g = gecko()
spec = g.spec if hasattr(g, "spec") else g
world.spawn(spec)

model = world.spec.compile()
world.attach_model(model)

print("Actuators:", model.nu)
for i in range(model.nu):
    print(f"[{i}] type={model.actuator_gaintype[i]}  biastype={model.actuator_biastype[i]}  dynprm={model.actuator_dynprm[i]}")
    print("   trnid:", model.actuator_trnid[i])
    print()
