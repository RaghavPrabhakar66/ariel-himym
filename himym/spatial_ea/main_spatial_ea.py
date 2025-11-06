import mujoco
import numpy as np

# Import configuration
from ea_config import config

# Import local modules
from spatial_individual import SpatialIndividual
from genetic_operators import (
    create_initial_genotype,
    crossover_one_point,
    mutate_gaussian,
    clone_individual
)
from selection import apply_selection
from evaluation import evaluate_population
from visualization import plot_fitness_evolution, save_mating_trajectories
from simulation_utils import (
    generate_spawn_positions,
    spawn_population_in_world,
    get_tracked_geoms,
    create_sinusoidal_controller,
    create_mating_controller,
    track_trajectories,
    update_trajectories
)

# Import robot and environment
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
from ariel.utils.renderers import video_renderer
from ariel.utils.video_recorder import VideoRecorder


class SpatialEA:
    """
    Spatial Evolutionary Algorithm for robot movement.
    
    This EA uses physical proximity in simulation to determine mating pairs,
    where robots move towards fitter neighbors during a mating phase.
    """
    
    def __init__(
        self, 
        population_size: int | None = None, 
        num_generations: int | None = None, 
        num_joints: int | None = None
    ):
        """
        Initialize the Spatial EA.
        
        Args:
            population_size: Initial population size
            num_generations: Number of generations to evolve
            num_joints: Number of controllable joints per robot
        """
        self.population_size = population_size or config.population_size
        self.num_generations = num_generations or config.num_generations
        self.num_joints = num_joints
        
        self.population: list[SpatialIndividual] = []
        self.generation = 0
        self.fitness_history: list[float] = []
        self.best_individual_history: list[SpatialIndividual] = []
        
        # World and simulation
        self.world = None
        self.model = None
        self.data = None
        self.robots = []
        self.tracked_geoms = []
        
        # Store current positions for cross-generation persistence
        self.current_positions: list[np.ndarray] = []
        
        # Counter for assigning unique IDs to individuals
        self.next_unique_id = 0
    
    def create_individual(self) -> SpatialIndividual:
        """Create a new individual with random genotype."""
        individual = SpatialIndividual(
            unique_id=self.next_unique_id, 
            generation=self.generation
        )
        self.next_unique_id += 1
        
        individual.genotype = create_initial_genotype(
            num_joints=self.num_joints,
            amplitude_range=(config.amplitude_init_min, config.amplitude_init_max),
            frequency_range=(config.frequency_init_min, config.frequency_init_max),
            phase_range=(config.phase_min, config.phase_max)
        )
        
        return individual
    
    def initialize_population(self) -> None:
        """Initialize population with random individuals."""
        print(f"Initializing population of {self.population_size} individuals")
        self.population = [self.create_individual() for _ in range(self.population_size)]
    
    def spawn_population(self) -> None:
        """Spawn population in simulation world."""
        print(f"Spawning {self.population_size} robots in simulation space")
        
        # Sanity check: verify positions match population size
        if len(self.current_positions) > 0 and len(self.current_positions) != self.population_size:
            print(f"  WARNING: Position count ({len(self.current_positions)}) != population size ({self.population_size})")
            print(f"  Regenerating all positions...")
            self.current_positions = []
        
        # Determine spawn positions
        if len(self.current_positions) == self.population_size:
            print(f"  Using positions from previous generation")
            positions = [pos.copy() for pos in self.current_positions]
        else:
            print(f"  Generating new random spawn positions")
            positions = generate_spawn_positions(
                population_size=self.population_size,
                spawn_x_range=(config.spawn_x_min, config.spawn_x_max),
                spawn_y_range=(config.spawn_y_min, config.spawn_y_max),
                spawn_z=config.spawn_z,
                min_spawn_distance=config.min_spawn_distance
            )
        
        # Spawn robots
        self.world, self.model, self.data, self.robots = spawn_population_in_world(
            population=self.population,
            positions=positions,
            world_size=config.world_size
        )
        
        # Track robot geoms
        self.tracked_geoms = get_tracked_geoms(
            world=self.world,
            data=self.data,
            population_size=self.population_size
        )
        
        print(f"Spawned {len(self.population)} robots successfully")
        print(f"Tracking {len(self.tracked_geoms)} core geoms")
    
    def evaluate_population_fitness(self) -> list[float]:
        """Evaluate fitness for all individuals."""
        print(f"  Evaluating generation {self.generation + 1}")
        
        fitness_values = evaluate_population(
            population=self.population,
            world_size=config.world_size,
            simulation_time=config.simulation_time,
            control_clip_min=config.control_clip_min,
            control_clip_max=config.control_clip_max
        )
        
        return fitness_values
    
    def mating_movement_phase(
        self, 
        duration: float = 60.0, 
        save_trajectories: bool = True,
        record_video: bool = False
    ) -> None:
        """
        Run mating movement phase where robots move towards attractive neighbors.
        
        Args:
            duration: Duration of movement phase
            save_trajectories: Whether to save trajectory visualization
            record_video: Whether to record video
        """
        print(f"  MATING MOVEMENT PHASE ({duration}s)")
        
        fitness_values = [ind.fitness for ind in self.population]
        num_spawned = len(self.tracked_geoms)
        
        # Initialize trajectory tracking
        sample_interval = max(1, int(duration / self.model.opt.timestep) // 100)
        print(f"  Tracking trajectories for {num_spawned} spawned robots")
        
        trajectories = track_trajectories(self.tracked_geoms, sample_interval)
        
        # Create mating controller
        controller = create_mating_controller(
            population=self.population,
            tracked_geoms=self.tracked_geoms,
            num_joints=self.num_joints,
            control_clip_min=config.control_clip_min,
            control_clip_max=config.control_clip_max,
            fitness_values=fitness_values,
            world_size=config.world_size,
            use_periodic_boundaries=config.use_periodic_boundaries
        )
        
        # Set up video recording if requested
        video_recorder = None
        renderer = None
        if record_video:
            video_name = f"generation_{self.generation + 1:03d}_mating_movement"
            video_recorder = VideoRecorder(
                file_name=video_name,
                output_folder=config.video_folder
            )
            
            scene_option = mujoco.MjvOption()
            scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
            renderer = mujoco.Renderer(
                self.model,
                width=video_recorder.width,
                height=video_recorder.height
            )
            
            steps_per_frame = max(
                1, 
                int(self.model.opt.timestep * duration * video_recorder.fps / duration)
            )
            print(f"  Recording video: {video_name}")
            print(f"  Steps per frame: {steps_per_frame}")
        
        # Run mating movement simulation
        mujoco.set_mjcb_control(controller)
        sim_steps = int(duration / self.model.opt.timestep)
        
        for step in range(sim_steps):
            mujoco.mj_step(self.model, self.data)
            
            # PERIODIC BOUNDARY FOR MOVEMENT
            if config.use_periodic_boundaries:
                from periodic_boundary_utils import apply_periodic_boundaries_to_simulation
                apply_periodic_boundaries_to_simulation(
                    self.tracked_geoms, 
                    (config.world_size[0], config.world_size[1])
                )
            
            # Sample positions periodically
            update_trajectories(trajectories, self.tracked_geoms, step, sample_interval)
            
            print(f"    Mating step {step + 1}/{sim_steps}", end='\r')
            
            # Record video frame if enabled
            if record_video and renderer is not None and video_recorder is not None:
                if step % steps_per_frame == 0:
                    renderer.update_scene(self.data, scene_option=scene_option)
                    video_recorder.write(frame=renderer.render())
        
        # Record final positions
        for i in range(num_spawned):
            pos = self.tracked_geoms[i].xpos.copy()
            trajectories[i].append(pos[:2])
        
        print(f"  MATING MOVEMENT COMPLETE")
        
        # Clean up video recording
        if record_video and video_recorder is not None:
            video_recorder.release()
            print(f"  Video saved: {video_recorder.frame_count} frames")
            if renderer is not None:
                renderer.close()
        
        # Update positions for next generation
        self.current_positions = []
        for i in range(num_spawned):
            pos = self.tracked_geoms[i].xpos.copy()
            self.current_positions.append(pos)
        
        print(f"  Updated positions for next generation")
        print(f"    Tracked {len(self.current_positions)} positions")
        
        # Save trajectory visualization
        if save_trajectories:
            save_path = f"{config.figures_folder}/mating_trajectories_gen_{self.generation + 1:03d}.png"
            save_mating_trajectories(
                trajectories=trajectories,
                population=self.population,
                fitness_values=fitness_values,
                generation=self.generation + 1,
                population_size=self.population_size,
                simulation_time=config.simulation_time,
                world_size=config.world_size,
                robot_size=config.robot_size,
                save_path=save_path,
                use_periodic_boundaries=config.use_periodic_boundaries
            )
    
    def create_next_generation(self, record_video: bool = False) -> None:
        """
        Create next generation through movement-based pairing and reproduction.
        
        Args:
            record_video: Whether to record video of mating movement
        """
        print(f"  Creating next generation with movement-based selection...")
        
        # Allow robots to move towards partners
        self.mating_movement_phase(
            duration=config.simulation_time, 
            save_trajectories=True,
            record_video=record_video
        )
        
        new_population: list[SpatialIndividual] = []
        new_positions: list[np.ndarray] = []
        
        # Find pairs based on proximity
        pairs = []
        pair_positions = []
        paired_indices = set()
        
        # Sort population by fitness (descending) to prioritize high-fitness individuals
        fitness_ranking = sorted(
            enumerate(self.population), 
            key=lambda x: x[1].fitness, 
            reverse=True
        )
        
        for idx, _ in fitness_ranking:
            if idx in paired_indices:
                continue  # Already paired
            
            current_pos = self.tracked_geoms[idx].xpos.copy()
            
            # Find highest fitness partner within pairing radius
            best_partner_idx = None
            best_partner_fitness = -1
            
            for other_idx, other_ind in enumerate(self.population):
                if other_idx == idx or other_idx in paired_indices:
                    continue
                
                other_pos = self.tracked_geoms[other_idx].xpos.copy()
                
                # Calculate distance using periodic boundaries if enabled
                if config.use_periodic_boundaries:
                    from periodic_boundary_utils import periodic_distance
                    distance = periodic_distance(
                        current_pos, other_pos, (config.world_size[0], config.world_size[1])
                    )
                else:
                    distance = np.linalg.norm(current_pos - other_pos)
                
                # Check if within pairing radius and has higher fitness than current best
                if distance <= config.pairing_radius and other_ind.fitness > best_partner_fitness:
                    best_partner_fitness = other_ind.fitness
                    best_partner_idx = other_idx
            
            # If found a partner within radius, create pair
            if best_partner_idx is not None:
                pairs.append((idx, best_partner_idx))
                paired_indices.add(idx)
                paired_indices.add(best_partner_idx)
                
                # Calculate offspring positions
                parent1_pos = self.current_positions[idx]
                parent2_pos = self.current_positions[best_partner_idx]
                
                # Random positions on circle edge around each parent
                angle1 = np.random.uniform(0, 2 * np.pi)
                child1_offset = np.array([
                    config.offspring_radius * np.cos(angle1),
                    config.offspring_radius * np.sin(angle1),
                    0.0
                ])
                
                angle2 = np.random.uniform(0, 2 * np.pi)
                child2_offset = np.array([
                    config.offspring_radius * np.cos(angle2),
                    config.offspring_radius * np.sin(angle2),
                    0.0
                ])
                
                # Apply offspring positions with periodic wrapping if enabled
                if config.use_periodic_boundaries:
                    from periodic_boundary_utils import wrap_offspring_position
                    child1_pos = wrap_offspring_position(
                        parent1_pos, child1_offset, (config.world_size[0], config.world_size[1])
                    )
                    child2_pos = wrap_offspring_position(
                        parent2_pos, child2_offset, (config.world_size[0], config.world_size[1])
                    )
                else:
                    # Non-periodic: just add offset (may go outside bounds)
                    child1_pos = parent1_pos + child1_offset
                    child2_pos = parent2_pos + child2_offset
                
                pair_positions.append((child1_pos, child2_pos))
        
        print(f"  Created {len(pairs)} pairs from {self.population_size} robots")
        print(f"  Unpaired robots: {self.population_size - len(paired_indices)}")
        
        # Create offspring from pairs
        for pair_idx, (parent1_idx, parent2_idx) in enumerate(pairs):
            parent1 = self.population[parent1_idx]
            parent2 = self.population[parent2_idx]
            
            # Crossover
            if np.random.random() < config.crossover_rate:
                child1, child2, self.next_unique_id = crossover_one_point(
                    parent1, parent2, self.next_unique_id, self.generation + 1
                )
            else:
                # No crossover - clone parents
                child1, self.next_unique_id = clone_individual(
                    parent1, self.next_unique_id, self.generation + 1
                )
                child2, self.next_unique_id = clone_individual(
                    parent2, self.next_unique_id, self.generation + 1
                )
            
            # Mutation
            child1, self.next_unique_id = mutate_gaussian(
                child1,
                mutation_rate=config.mutation_rate,
                mutation_strength=config.mutation_strength,
                next_unique_id=self.next_unique_id,
                amplitude_range=(config.amplitude_min, config.amplitude_max),
                frequency_range=(config.frequency_min, config.frequency_max),
                phase_max=config.phase_max
            )
            
            child2, self.next_unique_id = mutate_gaussian(
                child2,
                mutation_rate=config.mutation_rate,
                mutation_strength=config.mutation_strength,
                next_unique_id=self.next_unique_id,
                amplitude_range=(config.amplitude_min, config.amplitude_max),
                frequency_range=(config.frequency_min, config.frequency_max),
                phase_max=config.phase_max
            )
            
            new_population.append(child1)
            new_positions.append(pair_positions[pair_idx][0])
            
            new_population.append(child2)
            new_positions.append(pair_positions[pair_idx][1])
        
        # Extend population with offspring
        self.population.extend(new_population)
        self.current_positions.extend(new_positions)
        
        # Verify consistency
        if len(self.population) != len(self.current_positions):
            print(f"  ERROR: Population size ({len(self.population)}) != "
                  f"Position count ({len(self.current_positions)})")
            raise RuntimeError("Population and position arrays out of sync!")
        
        # Update population size before selection
        old_size = self.population_size
        self.population_size = len(self.population)
        
        print(f"  Population extended from {old_size} to {self.population_size} individuals")
        print(f"  Added {len(new_population)} offspring")
        print(f"  Position tracking updated: {len(self.current_positions)} positions")
        print(f"  Paired individuals: {len(paired_indices)} out of {old_size}")
        
        # Apply selection to manage population size
        self.population, self.current_positions, self.population_size = apply_selection(
            population=self.population,
            current_positions=self.current_positions,
            method=config.selection_method,
            target_size=config.max_population_size,
            current_generation=self.generation
        )
    
    def run_evolution(self, record_generation_videos: bool = False) -> SpatialIndividual:
        """
        Run the evolutionary algorithm.
        
        Args:
            record_generation_videos: Whether to record video for each generation
            
        Returns:
            Best individual found
        """
        print("=" * 60)
        print("SPATIAL EVOLUTIONARY ALGORITHM")
        print("=" * 60)
        print(f"Population size: {self.population_size}")
        print(f"Generations: {self.num_generations}")
        print(f"Robot joints: {self.num_joints}")
        if record_generation_videos:
            print(f"Video recording: ENABLED (one video per generation)")
        print("=" * 60)
        
        # Initialize
        self.initialize_population()
        
        # Evolution loop
        for gen in range(self.num_generations):
            self.generation = gen
            print(f"\n{'='*60}")
            print(f"Generation {gen + 1}/{self.num_generations}")
            print(f"{'='*60}")
            
            # Spawn population in simulation
            self.spawn_population()
            
            # Evaluate fitness
            fitness_values = self.evaluate_population_fitness()
            
            # Track statistics
            best_fitness = max(fitness_values)
            avg_fitness = np.mean(fitness_values)
            self.fitness_history.append(best_fitness)
            
            best_individual = max(self.population, key=lambda ind: ind.fitness)
            self.best_individual_history.append(best_individual)
            
            if config.print_generation_stats:
                print(f"  Best fitness: {best_fitness:.4f}")
                print(f"  Average fitness: {avg_fitness:.4f}")
                print(f"  Worst fitness: {min(fitness_values):.4f}")
            
            # Create next generation (except for last generation)
            if gen < self.num_generations - 1:
                self.create_next_generation(record_video=record_generation_videos)
            else:
                # Run mating movement to capture final positions
                self.mating_movement_phase(
                    duration=config.simulation_time, 
                    save_trajectories=True,
                    record_video=record_generation_videos
                )
        
        print(f"\n{'='*60}")
        print("EVOLUTION COMPLETE")
        print(f"{'='*60}")
        
        return self.get_best_individual()
    
    def get_best_individual(self) -> SpatialIndividual:
        """Get the best individual from current population."""
        return max(self.population, key=lambda ind: ind.fitness)
    
    def demonstrate_best(self) -> None:
        """Demonstrate the best individual in isolation."""
        print(f"\n{'='*60}")
        print("DEMONSTRATING BEST INDIVIDUAL")
        print(f"{'='*60}")
        
        best = self.get_best_individual()
        print(f"Best fitness: {best.fitness:.4f}")
        
        if config.print_final_genotype:
            print("\nBest genotype (amplitude, frequency, phase per joint):")
            genotype = best.genotype
            for j in range(self.num_joints):
                if j * 3 + 2 < len(genotype):
                    amp, freq, phase = genotype[j*3], genotype[j*3+1], genotype[j*3+2]
                    print(f"  Joint {j}: amp={amp:.3f}, freq={freq:.3f}, phase={phase:.3f}")
        
        # Create single robot demo
        mujoco.set_mjcb_control(None)
        demo_world = SimpleFlatWorld(config.world_size)
        demo_robot = gecko()
        demo_world.spawn(demo_robot.spec, spawn_position=[0, 0, 0])
        demo_model = demo_world.spec.compile()
        demo_data = mujoco.MjData(demo_model)
        
        # Diagnostic info
        print(f"\nDiagnostics:")
        print(f"  Evolution num_joints (self.num_joints): {self.num_joints}")
        print(f"  Demo model actuators (demo_model.nu): {demo_model.nu}")
        print(f"  Best genotype length: {len(best.genotype)}")
        print(f"  Expected genotype length: {self.num_joints * 3}")
        
        # Controller for single robot
        def demo_controller(model: mujoco.MjModel, data: mujoco.MjData) -> None:
            genotype = best.genotype
            for j in range(min(demo_model.nu, len(genotype) // 3)):
                if j * 3 + 2 < len(genotype):
                    amplitude = genotype[j * 3]
                    frequency = genotype[j * 3 + 1]
                    phase = genotype[j * 3 + 2]
                    control_value = amplitude * np.sin(frequency * data.time + phase)
                    data.ctrl[j] = np.clip(
                        control_value,
                        config.control_clip_min,
                        config.control_clip_max
                    )
        
        mujoco.set_mjcb_control(demo_controller)
        
        # Record video
        video_recorder = VideoRecorder(output_folder=config.video_folder)
        print("Recording best individual demonstration...")
        
        video_renderer(
            demo_model,
            demo_data,
            duration=config.final_demo_time,
            video_recorder=video_recorder,
        )
        
        print("Demonstration complete!")
    
    def demonstrate_final_population(self) -> None:
        """Demonstrate the final evolved population."""
        print(f"\n{'='*60}")
        print("DEMONSTRATING FINAL POPULATION")
        print(f"{'='*60}")
        print(f"Recording {self.population_size} robots...")
        
        # Spawn final population
        self.spawn_population()
        
        # Set controller
        controller = create_sinusoidal_controller(
            population=self.population,
            num_joints=self.num_joints,
            control_clip_min=config.control_clip_min,
            control_clip_max=config.control_clip_max,
            num_spawned_robots=len(self.tracked_geoms)
        )
        mujoco.set_mjcb_control(controller)
        
        # Record video
        video_recorder = VideoRecorder(output_folder=config.video_folder)
        
        video_renderer(
            self.model,
            self.data,
            duration=config.multi_robot_demo_time,
            video_recorder=video_recorder,
        )
        
        print("Final population demonstration complete!")
    
    def plot_fitness_evolution(self) -> None:
        """Plot fitness over generations."""
        save_path = f"{config.figures_folder}/spatial_ea_fitness_evolution.png"
        plot_fitness_evolution(self.fitness_history, save_path)


def main():
    """Main entry point for the spatial EA."""
    temp_world = SimpleFlatWorld(config.world_size)
    temp_robot = gecko()
    temp_world.spawn(temp_robot.spec, spawn_position=[0, 0, 0])
    temp_model = temp_world.spec.compile()
    num_joints = temp_model.nu
    
    print(f"Robot has {num_joints} controllable joints")
    
    spatial_ea = SpatialEA(
        population_size=config.population_size,
        num_generations=config.num_generations,
        num_joints=num_joints
    )
    
    # Run evolution with optional video recording for each generation
    # Set to True to record video for each generation's mating movement phase
    spatial_ea.run_evolution(record_generation_videos=False)
    
    # Demonstrate results
    spatial_ea.demonstrate_best()
    spatial_ea.demonstrate_final_population()
    
    # Plot results
    spatial_ea.plot_fitness_evolution()
    
    print(f"\n{'='*60}")
    print("ALL TASKS COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__": 
    main()
