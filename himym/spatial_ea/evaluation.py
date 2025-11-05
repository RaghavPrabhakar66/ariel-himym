"""
Fitness evaluation functions for evolutionary algorithm.
"""

import mujoco
import numpy as np
from spatial_individual import SpatialIndividual
from simulation_utils import apply_sinusoidal_control
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld


def evaluate_population(
    population: list[SpatialIndividual],
    world_size: list[float],
    simulation_time: float,
    control_clip_min: float,
    control_clip_max: float
) -> list[float]:
    """
    Evaluate each individual in isolation and assign fitness based on distance traveled.
    
    Args:
        population: List of individuals to evaluate
        world_size: Size of simulation world [width, height]
        simulation_time: Duration of evaluation simulation
        control_clip_min: Minimum control value
        control_clip_max: Maximum control value
        
    Returns:
        List of fitness values for each individual
    """
    print(f"  Testing each robot in isolated environment...")
    
    # Create single isolated environment for reuse
    isolated_world = SimpleFlatWorld(world_size)
    isolated_robot = gecko()
    isolated_world.spawn(
        isolated_robot.spec, 
        spawn_position=[0, 0, 0.5],
        correct_for_bounding_box=False
    )
    isolated_model = isolated_world.spec.compile()
    
    # Get core geom name
    all_geoms = isolated_world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    core_geom_id = None
    for geom in all_geoms:
        if geom.name == "robot-0core":
            # Find the geom ID in the compiled model
            core_geom_id = mujoco.mj_name2id(
                isolated_model, mujoco.mjtObj.mjOBJ_GEOM, "robot-0core"
            )
            break
    
    fitness_values = []
    
    # Evaluate each robot individually using the same environment
    for i, individual in enumerate(population):
        # Create fresh data for this evaluation
        isolated_data = mujoco.MjData(isolated_model)
        
        # Record start position
        mujoco.mj_forward(isolated_model, isolated_data)
        start_position = isolated_data.geom_xpos[core_geom_id].copy()
        
        # Controller for single robot
        def single_robot_controller(model: mujoco.MjModel, data: mujoco.MjData) -> None:
            for j in range(min(model.nu, len(individual.genotype) // 3)):
                apply_sinusoidal_control(
                    genotype=individual.genotype,
                    joint_index=j,
                    ctrl_index=j,  # Single robot, so ctrl_index = joint_index
                    model=model,
                    data=data,
                    control_clip_min=control_clip_min,
                    control_clip_max=control_clip_max
                )
        
        # Set controller and run simulation
        mujoco.set_mjcb_control(single_robot_controller)
        sim_steps = int(simulation_time / isolated_model.opt.timestep)
        for _ in range(sim_steps):
            mujoco.mj_step(isolated_model, isolated_data)
        
        # Record end position and calculate fitness
        end_position = isolated_data.geom_xpos[core_geom_id].copy()
        distance = np.linalg.norm(end_position - start_position)
        
        individual.fitness = distance
        individual.start_position = start_position
        individual.end_position = end_position
        fitness_values.append(distance)
        
        if (i + 1) % 5 == 0:
            print(f"    Evaluated {i + 1}/{len(population)} robots")
    
    print(f"  Fitness evaluation complete!")
    return fitness_values
