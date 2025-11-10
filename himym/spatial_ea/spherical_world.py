import mujoco
from mujoco import viewer
import numpy as np

# Simulation parameters
SPHERE_RADIUS = 100.0       # Planet radius
NUM_SPHERES = 500             # Number of orbiting spheres
SIM_TIMESTEP = 0.002
CENTRAL_G = 9.81


def spherical_world_xml(radius: float, num_spheres: int) -> str:
    """Return MJCF XML string with a central planet and multiple free spheres."""
    xml = f"""
<mujoco model="spherical_world">
  <compiler angle="degree" coordinate="local"/>
  <option timestep="{SIM_TIMESTEP}" gravity="0 0 0"/>

  <worldbody>
    <!-- Central planet -->
    <geom name="planet" type="sphere" pos="0 0 0" size="{radius}" rgba="0.2 0.6 0.2 1"
          contype="1" conaffinity="1"/>
"""
    # Randomly place spheres around the planet
    for i in range(num_spheres):
        # Random spherical coordinates
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi)
        r = radius + 5.0  # distance from center, never zero

        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)

        xml += f"""
    <body name="sphere_{i}" pos="{x} {y} {z}">
      <freejoint/>
      <geom name="marker_{i}" type="sphere" size="0.3" rgba="1 0 0 1" density="1000"
            contype="1" conaffinity="1"/>
    </body>
"""
    xml += "</worldbody>\n</mujoco>"
    return xml


def random_perpendicular_vector(v):
    """Return a random unit vector perpendicular to v."""
    rand_vec = np.random.rand(3) - 0.5
    perp_vec = rand_vec - np.dot(rand_vec, v) * v
    perp_vec /= np.linalg.norm(perp_vec)
    return perp_vec


def apply_central_gravity(model, data, g=9.81, center=(0, 0, 0)):
    """Apply central gravity toward the planet center."""
    center = np.array(center)
    data.xfrc_applied[:] = 0.0
    for body_id in range(model.nbody):
        if body_id == 0:
            continue
        mass = model.body_mass[body_id]
        if mass <= 0:
            continue
        pos = data.xpos[body_id]
        dir_vec = center - pos
        dist = np.linalg.norm(dir_vec)
        if dist < 1e-6:
            continue
        dir_unit = dir_vec / dist
        data.xfrc_applied[body_id, 0:3] = mass * g * dir_unit


def main():
    xml = spherical_world_xml(SPHERE_RADIUS, NUM_SPHERES)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    # Set initial tangential velocities
    for i in range(NUM_SPHERES):
        start_idx = 6 * i
        pos_vec = data.xpos[i + 1]  # body_id 0 is planet
        dist = np.linalg.norm(pos_vec)
        if dist < 1e-3:  # avoid divide by zero
            dist = 1e-3
        radius_unit = pos_vec / dist

        tangential_vec = random_perpendicular_vector(radius_unit)
        orbital_speed = np.sqrt(CENTRAL_G * dist)
        velocity = tangential_vec * orbital_speed

        data.qvel[start_idx + 3:start_idx + 6] = velocity


    print("Launching viewer... Press ESC to exit.")

    # Launch interactive viewer
    with viewer.launch_passive(model, data) as v:
        while v.is_running():
            apply_central_gravity(model, data, CENTRAL_G)
            mujoco.mj_step(model, data)
            v.sync()


if __name__ == "__main__":
    main()
