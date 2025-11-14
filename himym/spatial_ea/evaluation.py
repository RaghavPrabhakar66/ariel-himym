"""
Fitness evaluation functions for evolutionary algorithm.
"""

import mujoco
import numpy as np
from spatial_individual import SpatialIndividual
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
from hyperneat import CPPN, SubstrateNetwork, create_substrate_for_gecko

def evaluate_population(
    population: list[SpatialIndividual],
    world_size: list[float],
    simulation_time: float,
    control_clip_min: float,
    control_clip_max: float,
    use_directional_fitness: bool = False,
    target_distance_min: float = 5.0,
    target_distance_max: float = 10.0,
    progress_weight: float = 0.5
) -> list[float]:
    """
    Evaluate each individual in isolation and assign fitness based on distance traveled.
    Uses HyperNEAT controller.
    
    Args:
        population: List of individuals to evaluate
        world_size: Size of simulation world [width, height]
        simulation_time: Duration of evaluation simulation
        control_clip_min: Minimum control value
        control_clip_max: Maximum control value
        use_directional_fitness: Use directional fitness (movement toward target)
        target_distance_min: Minimum distance to target (meters)
        target_distance_max: Maximum distance to target (meters)
        progress_weight: Weight for progress toward target component
        
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
    num_joints = isolated_model.nu
    
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
        
        # Set random orientation for the robot to avoid directional bias
        # Find the freejoint for robot-0
        joint_id = mujoco.mj_name2id(isolated_model, mujoco.mjtObj.mjOBJ_JOINT, "robot-0")
        robot_yaw = 0.0  # Track the robot's orientation for target placement
        if joint_id >= 0:
            qpos_addr = isolated_model.jnt_qposadr[joint_id]
            # Generate random yaw angle
            robot_yaw = np.random.uniform(0, 2 * np.pi)
            # Set quaternion for rotation around z-axis
            isolated_data.qpos[qpos_addr + 3] = np.cos(robot_yaw / 2)  # qw
            isolated_data.qpos[qpos_addr + 4] = 0.0                     # qx
            isolated_data.qpos[qpos_addr + 5] = 0.0                     # qy
            isolated_data.qpos[qpos_addr + 6] = np.sin(robot_yaw / 2)  # qz
        
        # Record start position
        mujoco.mj_forward(isolated_model, isolated_data)
        start_position = isolated_data.geom_xpos[core_geom_id].copy()
        
        # Generate random target position if using directional fitness
        if use_directional_fitness:
            target_distance = np.random.uniform(target_distance_min, target_distance_max)
            # Generate target angle RELATIVE to robot's facing direction
            # This ensures all robots have the same task difficulty
            # Random angle relative to forward direction (-π to π for full range)
            relative_angle = np.random.uniform(-np.pi, np.pi)
            # Absolute angle in world frame = robot's yaw + relative angle
            target_angle = robot_yaw + relative_angle
            
            target_position = start_position + np.array([
                target_distance * np.cos(target_angle),
                target_distance * np.sin(target_angle),
                0.0  # Same height
            ])
            
            # Store target for potential visualization
            individual.target_position = target_position.copy()
        else:
            target_position = None
        
        # Create HyperNEAT controller
        cppn = CPPN(individual.genotype)
        input_coords, hidden_coords, output_coords = create_substrate_for_gecko(
            num_joints=num_joints,
            use_hidden_layer=True,
            hidden_layer_size=num_joints
        )
        substrate = SubstrateNetwork(
            input_coords=input_coords,
            hidden_coords=hidden_coords,
            output_coords=output_coords,
            cppn=cppn,
            weight_threshold=0.2
        )
        
        def single_robot_controller(model: mujoco.MjModel, data: mujoco.MjData) -> None:
            # Get joint angles (skip free joint - 7 DOF)
            joint_angles = data.qpos[7:7+num_joints].copy()
            
            # Add CPG-style oscillator inputs (multiple frequencies for rich patterns)
            osc1 = np.sin(data.time * 1.0)  # 1 Hz oscillator
            osc2 = np.cos(data.time * 1.0)  # 1 Hz (90° phase shift)
            osc3 = np.sin(data.time * 2.0)  # 2 Hz oscillator
            osc4 = np.cos(data.time * 2.0)  # 2 Hz (90° phase shift)
            
            cpg_inputs = np.array([osc1, osc2, osc3, osc4])
            bias_input = np.array([1.0])
            
            # Calculate directional input to target (if directional fitness enabled)
            if use_directional_fitness and target_position is not None:
                current_position = data.geom_xpos[core_geom_id][:2]  # x, y only
                target_vector = target_position[:2] - current_position
                distance = np.linalg.norm(target_vector)
                if distance > 0.01:  # Avoid division by zero
                    directional_inputs = target_vector / distance  # Normalized direction
                else:
                    directional_inputs = np.array([0.0, 0.0])  # At target
            else:
                directional_inputs = np.array([0.0, 0.0])  # No target info
            
            # Combine inputs: joint angles + CPG oscillators + directional + bias
            sensor_inputs = np.concatenate([joint_angles, cpg_inputs, directional_inputs, bias_input])
            
            # Get motor outputs from substrate
            motor_outputs = substrate.activate(sensor_inputs)
            
            # Apply to actuators
            data.ctrl[:] = np.clip(
                motor_outputs,
                control_clip_min,
                control_clip_max
            )
        
        # Set controller and run simulation
        mujoco.set_mjcb_control(single_robot_controller)
        sim_steps = int(simulation_time / isolated_model.opt.timestep)
        for _ in range(sim_steps):
            mujoco.mj_step(isolated_model, isolated_data)
        
        # Record end position and calculate fitness
        end_position = isolated_data.geom_xpos[core_geom_id].copy()
        
        # Calculate fitness based on mode
        if use_directional_fitness and target_position is not None:
            # Directional fitness: reward progress toward target
            start_to_target = np.linalg.norm(target_position - start_position)
            end_to_target = np.linalg.norm(target_position - end_position)
            progress_toward_target = start_to_target - end_to_target  # Positive if moved closer
            
            # Total distance traveled (always positive, rewards movement)
            total_distance = np.linalg.norm(end_position - start_position)
            
            # IMPROVED FITNESS: Base fitness on movement, with directional bonus
            # This prevents negative fitness from discouraging exploration
            # Formula: fitness = distance * (1 + directional_bonus)
            # Where directional_bonus = progress_weight * (progress / distance) if distance > 0
            
            if total_distance > 0.01:  # Avoid division by zero
                # Calculate directional bonus (ranges from -1 to +1)
                direction_quality = progress_toward_target / total_distance
                # Apply progress weight to directional component
                directional_bonus = progress_weight * direction_quality
                # Combined fitness: base movement + directional multiplier
                fitness = total_distance * (1.0 + directional_bonus)
            else:
                # No movement - minimal fitness
                fitness = 0.0
            
            individual.fitness = fitness
            individual.start_position = start_position
            individual.end_position = end_position
            individual.progress_toward_target = progress_toward_target
            individual.total_distance = total_distance
        else:
            # Simple distance-only fitness
            distance = np.linalg.norm(end_position - start_position)
            individual.fitness = distance
            individual.start_position = start_position
            individual.end_position = end_position
            individual.progress_toward_target = 0.0
            individual.total_distance = distance
        
        fitness_values.append(individual.fitness)
        
        if (i + 1) % 5 == 0:
            print(f"    Evaluated {i + 1}/{len(population)} robots")
    
    print(f"  FITNESS EVALUATION COMPLETE")
    return fitness_values
