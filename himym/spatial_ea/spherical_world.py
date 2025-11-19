import mujoco
from mujoco import viewer
import numpy as np

# Simulation parameters
SPHERE_RADIUS = 100.0       # Planet radius
NUM_SPHERES = 500             # Number of orbiting spheres
SIM_TIMESTEP = 0.02
CENTRAL_G = 9.81
SPEED_FACTOR = 200.0


def spherical_world_xml(radius: float, num_spheres: int) -> str:
    """Return MJCF XML string with a central planet and multiple free spheres."""
    xml = f"""
<mujoco model="spherical_world">
  <compiler angle="degree" coordinate="local"/>
  <option timestep="{SIM_TIMESTEP}" gravity="0 0 0"/>

  <asset>
    <!-- Create multiple textures for a colorful planet -->
    <texture name="skybox" type="skybox" builtin="gradient" 
             rgb1="0.3 0.5 0.7" rgb2="0.1 0.1 0.2" 
             width="512" height="512"/>
    
    <texture name="planet_tex" type="cube" builtin="flat" 
             rgb1="0.2 0.7 0.3" rgb2="0.7 0.3 0.2" 
             width="256" height="256" 
             mark="cross" markrgb="0.9 0.9 0.1"/>
    
    <!-- Create a material that uses the texture -->
    <material name="planet_mat" texture="planet_tex" texrepeat="4 4"/>
  </asset>

  <worldbody>
    <!-- Central planet with chess texture -->
    <geom name="planet" type="sphere" pos="0 0 0" size="{radius}" 
          material="planet_mat"
          contype="1" conaffinity="1"/>
"""
    # Randomly place spheres around the planet
    for i in range(num_spheres):
        # Avoid exact poles to prevent degenerate positions
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0.1, np.pi - 0.1)
        r = radius + np.random.uniform(5.0, 10.0)  # safe distance above surface

        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)

        xml += f"""
    <body name="sphere_{i}" pos="{x:.3f} {y:.3f} {z:.3f}">
      <freejoint/>
      <geom name="marker_{i}" type="sphere" size="0.5" rgba="1 0 0 1" density="1000"
            contype="1" conaffinity="1"/>
    </body>
"""
    xml += "</worldbody>\n</mujoco>"
    return xml


def perpendicular_unit_vector(v):
    """
    Return a unit vector perpendicular to v.
    Guaranteed non-zero for any input.
    """
    v = np.array(v)
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        return np.array([1.0, 0.0, 0.0])
    v /= norm

    # Pick a vector not parallel to v
    if abs(v[0]) < 0.9:
        temp = np.array([1, 0, 0])
    else:
        temp = np.array([0, 1, 0])

    perp = np.cross(v, temp)
    return perp / np.linalg.norm(perp)


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

    # Set initial tangential velocities for stable orbits
    for i in range(NUM_SPHERES):
        start_idx = 6 * i
        pos_vec = data.xpos[i + 1]  # body_id 0 is planet
        dist = np.linalg.norm(pos_vec)
        if dist < 1e-3:
            dist = 1e-3  # safety

        radius_unit = pos_vec / dist
        tangential_vec = perpendicular_unit_vector(radius_unit)

        # Circular orbit velocity
        orbital_speed = np.sqrt(CENTRAL_G * dist) * SPEED_FACTOR
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