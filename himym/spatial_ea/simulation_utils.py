"""
Simulation utilities for evolutionary algorithm.

This module provides functions for spawning robots, creating controllers,
and managing multi-robot simulations.
"""

import mujoco
import numpy as np
from typing import Any
from spatial_individual import SpatialIndividual
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld


def generate_spawn_positions(
    population_size: int,
    spawn_x_range: tuple[float, float],
    spawn_y_range: tuple[float, float],
    spawn_z: float,
    min_spawn_distance: float,
    max_attempts: int = 1000
) -> list[np.ndarray]:
    """
    Generate non-overlapping spawn positions for robots.
    
    Args:
        population_size: Number of positions to generate
        spawn_x_range: (min, max) for x coordinates
        spawn_y_range: (min, max) for y coordinates
        spawn_z: Z coordinate for all positions
        min_spawn_distance: Minimum distance between positions
        max_attempts: Maximum attempts to find valid position
        
    Returns:
        List of spawn positions as numpy arrays
    """
    positions = []
    
    for i in range(population_size):
        attempts = 0
        position_found = False
        
        while not position_found and attempts < max_attempts:
            # Generate random position
            x = np.random.uniform(spawn_x_range[0], spawn_x_range[1])
            y = np.random.uniform(spawn_y_range[0], spawn_y_range[1])
            z = spawn_z
            new_pos = np.array([x, y, z])
            
            # Check distance to all existing positions
            valid = True
            for existing_pos in positions:
                distance = np.linalg.norm(new_pos[:2] - existing_pos[:2])  # Only check x,y
                if distance < min_spawn_distance:
                    valid = False
                    break
            
            if valid:
                positions.append(new_pos)
                position_found = True
            else:
                attempts += 1
        
        if not position_found:
            # If we couldn't find a non-overlapping position, use grid placement
            print(f"  Warning: Could not find non-overlapping position for robot {i}, using grid fallback")
            grid_size = int(np.ceil(np.sqrt(population_size)))
            grid_x = (i % grid_size) * min_spawn_distance + spawn_x_range[0]
            grid_y = (i // grid_size) * min_spawn_distance + spawn_y_range[0]
            positions.append(np.array([grid_x, grid_y, spawn_z]))
    
    return positions


def spawn_population_in_world(
    population: list[SpatialIndividual],
    positions: list[np.ndarray],
    world_size: list[float]
) -> tuple[SimpleFlatWorld, mujoco.MjModel, mujoco.MjData, list[Any]]:
    """
    Spawn a population of robots in a simulation world.
    
    Args:
        population: List of individuals to spawn
        positions: List of spawn positions
        world_size: Size of world [width, height]
        
    Returns:
        Tuple of (world, model, data, robots)
    """
    mujoco.set_mjcb_control(None)
    world = SimpleFlatWorld(world_size)
    robots = []
    
    for i, individual in enumerate(population):
        robot = gecko()
        robots.append(robot)
        pos = positions[i]
        individual.spawn_position = np.array(pos)
        individual.robot_index = i
        world.spawn(robot.spec, spawn_position=pos, prefix_id=i)
    
    # Compile world
    model = world.spec.compile()
    data = mujoco.MjData(model)
    
    mujoco.mj_forward(model, data)  # Forward simulate to initialize positions
    
    return world, model, data, robots


def get_tracked_geoms(
    world: SimpleFlatWorld,
    data: mujoco.MjData,
    population_size: int
) -> list[Any]:
    """
    Get tracked geoms for all robots in the population.
    
    Args:
        world: Simulation world
        data: MuJoCo data
        population_size: Number of robots to track
        
    Returns:
        List of bound geom objects
    """
    all_geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    tracked_geoms = []
    
    for i in range(population_size):
        # Match exactly "robot-{i}core" - the main core of robot i
        core_name = f"robot-{i}core"
        for geom in all_geoms:
            if geom.name == core_name:
                tracked_geoms.append(data.bind(geom))
                break
    
    return tracked_geoms


def apply_sinusoidal_control(
    genotype: list[float],
    joint_index: int,
    ctrl_index: int,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    control_clip_min: float,
    control_clip_max: float,
    bias: float = 0.0
) -> None:
    """
    Apply sinusoidal control to a single joint based on genotype.
    
    This is the core controller logic used by all controller variants.
    
    Args:
        genotype: Individual's genotype containing [amp, freq, phase, ...] for each joint
        joint_index: Which joint to control (0-indexed)
        ctrl_index: Index in the control array (data.ctrl)
        model: MuJoCo model
        data: MuJoCo data
        control_clip_min: Minimum control value
        control_clip_max: Maximum control value
        bias: Additional bias to add to control (default 0.0)
    """
    if ctrl_index < model.nu and joint_index * 3 + 2 < len(genotype):
        amplitude = genotype[joint_index * 3]
        frequency = genotype[joint_index * 3 + 1]
        phase = genotype[joint_index * 3 + 2]
        
        control_value = amplitude * np.sin(frequency * data.time + phase)
        control_value += bias
        
        data.ctrl[ctrl_index] = np.clip(
            control_value,
            control_clip_min,
            control_clip_max
        )


def create_sinusoidal_controller(
    population: list[SpatialIndividual],
    num_joints: int,
    control_clip_min: float,
    control_clip_max: float,
    num_spawned_robots: int
):
    """
    Create a controller that applies sinusoidal joint controls.
    
    Args:
        population: List of individuals
        num_joints: Number of joints per robot
        control_clip_min: Minimum control value
        control_clip_max: Maximum control value
        num_spawned_robots: Number of actually spawned robots
        
    Returns:
        Controller function
    """
    def controller(model: mujoco.MjModel, data: mujoco.MjData) -> None:
        for robot_idx in range(min(num_spawned_robots, len(population))):
            individual = population[robot_idx]
            
            for j in range(num_joints):
                ctrl_idx = robot_idx * num_joints + j
                apply_sinusoidal_control(
                    genotype=individual.genotype,
                    joint_index=j,
                    ctrl_index=ctrl_idx,
                    model=model,
                    data=data,
                    control_clip_min=control_clip_min,
                    control_clip_max=control_clip_max
                )
    
    return controller


def create_mating_controller(
    population: list[SpatialIndividual],
    tracked_geoms: list[Any],
    num_joints: int,
    control_clip_min: float,
    control_clip_max: float,
    fitness_values: list[float],
    world_size: list[float] | None = None,
    use_periodic_boundaries: bool = False
):
    """
    Create a controller that biases movement towards attractive neighbors.
    
    Args:
        population: List of individuals
        tracked_geoms: List of tracked geom objects
        num_joints: Number of joints per robot
        control_clip_min: Minimum control value
        control_clip_max: Maximum control value
        fitness_values: Fitness scores for calculating attractiveness
        world_size: World dimensions [width, height, z] for periodic boundaries
        use_periodic_boundaries: Whether to use periodic distance calculations
        
    Returns:
        Controller function
    """
    # Calculate fitness-based attractiveness
    max_fitness = max(fitness_values) if max(fitness_values) > 0 else 1.0
    attractiveness = [f / max_fitness for f in fitness_values]
    num_spawned_robots = len(tracked_geoms)
    
    def controller(model: mujoco.MjModel, data: mujoco.MjData) -> None:
        for robot_idx in range(min(num_spawned_robots, len(population))):
            individual = population[robot_idx]
            current_pos = tracked_geoms[robot_idx].xpos.copy()
            
            # Find most attractive neighbor
            best_neighbor_idx = None
            best_score = -1
            min_dist = float('inf')
            
            for other_idx in range(min(num_spawned_robots, len(population))):
                if other_idx == robot_idx:
                    continue
                
                other_pos = tracked_geoms[other_idx].xpos.copy()
                
                # Calculate distance (periodic or direct)
                if use_periodic_boundaries and world_size is not None:
                    from periodic_boundary_utils import periodic_distance
                    distance = periodic_distance(
                        current_pos, other_pos, (world_size[0], world_size[1])
                    )
                else:
                    distance = np.linalg.norm(current_pos - other_pos)
                
                # Score = attractiveness / distance (prefer close + fit partners)
                if distance > 0.1:  # Avoid division by zero
                    score = attractiveness[other_idx] / distance
                    if score > best_score:
                        best_score = score
                        best_neighbor_idx = other_idx
                        min_dist = distance
            
            # Apply control with bias towards attractive neighbor
            for j in range(num_joints):
                ctrl_idx = robot_idx * num_joints + j
                
                # Calculate directional bias if found attractive neighbor
                bias = 0.0
                if best_neighbor_idx is not None and min_dist > 0.5:
                    neighbor_pos = tracked_geoms[best_neighbor_idx].xpos.copy()
                    
                    # Calculate direction (periodic or direct)
                    if use_periodic_boundaries and world_size is not None:
                        from periodic_boundary_utils import periodic_displacement
                        direction_2d = periodic_displacement(
                            current_pos, neighbor_pos, (world_size[0], world_size[1])
                        )
                    else:
                        direction = neighbor_pos - current_pos
                        direction_2d = np.array([direction[0], direction[1]])
                    
                    # Simple directional bias scaled by neighbor's attractiveness
                    bias = 0.2 * attractiveness[best_neighbor_idx] * np.sign(direction_2d[j % 2])
                
                # Apply sinusoidal control with bias
                apply_sinusoidal_control(
                    genotype=individual.genotype,
                    joint_index=j,
                    ctrl_index=ctrl_idx,
                    model=model,
                    data=data,
                    control_clip_min=control_clip_min,
                    control_clip_max=control_clip_max,
                    bias=bias
                )
    
    return controller


def track_trajectories(
    tracked_geoms: list[Any],
    sample_interval: int
) -> list[list[np.ndarray]]:
    """
    Initialize trajectory tracking for robots.
    
    Args:
        tracked_geoms: List of geoms to track
        sample_interval: Interval for sampling positions
        
    Returns:
        List of trajectories (each initially containing starting position)
    """
    trajectories = [[] for _ in range(len(tracked_geoms))]
    
    # Record initial positions
    for i in range(len(tracked_geoms)):
        pos = tracked_geoms[i].xpos.copy()
        trajectories[i].append(pos[:2])  # Store x, y only
    
    return trajectories


def update_trajectories(
    trajectories: list[list[np.ndarray]],
    tracked_geoms: list[Any],
    step: int,
    sample_interval: int
) -> None:
    """
    Update trajectories with current positions if at sample interval.
    
    Args:
        trajectories: List of trajectories to update
        tracked_geoms: List of geoms being tracked
        step: Current simulation step
        sample_interval: Interval for sampling
    """
    if step % sample_interval == 0:
        for i in range(len(tracked_geoms)):
            pos = tracked_geoms[i].xpos.copy()
            trajectories[i].append(pos[:2])
