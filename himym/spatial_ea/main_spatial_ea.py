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
from parent_selection import find_pairs, calculate_offspring_positions
from evolution_data_collector import EvolutionDataCollector
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
from periodic_boundary_utils import apply_periodic_boundaries_to_simulation


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
        
        # Data collection
        self.data_collector = EvolutionDataCollector(config=config)
        
        # World and simulation
        self.world = None
        self.model = None
        self.data = None
        self.robots = []
        self.tracked_geoms = []
        
        # Store current positions for cross-generation persistence
        self.current_positions: list[np.ndarray] = []
        
        # Store current orientations (yaw angles in radians)
        self.current_orientations: list[float] = []
        
        # Counter for assigning unique IDs to individuals
        self.next_unique_id = 0
        
        # Mating zone management
        self.current_zone_centers: list[tuple[float, float]] = []
        self._initialize_mating_zones()
    
    def _initialize_mating_zones(self) -> None:
        """Initialize mating zone positions."""
        from parent_selection import generate_random_zone_centers
        
        # Initialize zones if needed for pairing or movement bias
        if config.pairing_method != "mating_zone" and config.movement_bias != "nearest_zone":
            return
        
        if config.num_mating_zones == 1:
            # Use configured center for single zone
            self.current_zone_centers = [tuple(config.mating_zone_center)]
        else:
            # Generate random positions for multiple zones
            self.current_zone_centers = generate_random_zone_centers(
                num_zones=config.num_mating_zones,
                world_size=(config.world_size[0], config.world_size[1]),
                zone_radius=config.mating_zone_radius,
                min_zone_distance=config.min_zone_distance
            )
        
        print(f"  Initialized {len(self.current_zone_centers)} mating zone(s)")
        if config.pairing_method == "mating_zone":
            print(f"  Zones used for: pairing")
        if config.movement_bias == "nearest_zone":
            print(f"  Zones used for: movement bias")
        if config.dynamic_mating_zones:
            print(f"  Zones will change every {config.zone_change_interval} generations")
    
    def _update_mating_zones(self) -> None:
        """Update mating zone positions if dynamic zones are enabled."""
        from parent_selection import generate_random_zone_centers
        
        # Only update zones if they're being used
        if config.pairing_method != "mating_zone" and config.movement_bias != "nearest_zone":
            return
        
        if not config.dynamic_mating_zones:
            return
        
        # Check if it's time to change zones
        if self.generation > 0 and self.generation % config.zone_change_interval == 0:
            print(f"\n  UPDATING MATING ZONES (Generation {self.generation})")
            self.current_zone_centers = generate_random_zone_centers(
                num_zones=config.num_mating_zones,
                world_size=(config.world_size[0], config.world_size[1]),
                zone_radius=config.mating_zone_radius,
                min_zone_distance=config.min_zone_distance
            )
            
            print(f"  New zone centers: {self.current_zone_centers}")
    
    def create_individual(self) -> SpatialIndividual:
        """Create a new individual with random genotype."""
        individual = SpatialIndividual(
            unique_id=self.next_unique_id, 
            generation=self.generation,
            initial_energy=config.initial_energy
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
            self.current_orientations = []
        
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
        
        # Determine spawn orientations
        if len(self.current_orientations) == self.population_size:
            print(f"  Using orientations from previous generation")
            orientations = self.current_orientations.copy()
        else:
            print(f"  Generating new random spawn orientations")
            # Random yaw angles between 0 and 2*pi
            orientations = [np.random.uniform(0, 2 * np.pi) for _ in range(self.population_size)]
        
        # Spawn robots
        self.world, self.model, self.data, self.robots = spawn_population_in_world(
            population=self.population,
            positions=positions,
            world_size=config.world_size,
            orientations=orientations
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
        save_trajectories: bool = True
    ) -> None:
        """
        Run mating movement phase where robots move towards attractive neighbors.
        
        Args:
            duration: Duration of movement phase
            save_trajectories: Whether to save trajectory visualization
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
            movement_bias=config.movement_bias,
            world_size=config.world_size,
            use_periodic_boundaries=config.use_periodic_boundaries,
            mating_zone_centers=self.current_zone_centers if self.current_zone_centers else None,
            mating_zone_radius=config.mating_zone_radius
        )
        
        # Set up video recording if requested
        video_recorder = None
        renderer = None
        if config.record_generation_videos:
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
                
                apply_periodic_boundaries_to_simulation(
                    self.tracked_geoms, 
                    (config.world_size[0], config.world_size[1])
                )
            
            # Sample positions periodically
            update_trajectories(trajectories, self.tracked_geoms, step, sample_interval)
            
            print(f"    Mating step {step + 1}/{sim_steps}", end='\r')
            
            # Record video frame if enabled
            if config.record_generation_videos and renderer is not None and video_recorder is not None:
                if step % steps_per_frame == 0:
                    renderer.update_scene(self.data, scene_option=scene_option)
                    video_recorder.write(frame=renderer.render())
        
        # Record final positions
        for i in range(num_spawned):
            pos = self.tracked_geoms[i].xpos.copy()
            trajectories[i].append(pos[:2])
        
        print(f"  MATING MOVEMENT COMPLETE")
        
        # Clean up video recording
        if config.record_generation_videos and video_recorder is not None:
            video_recorder.release()
            print(f"  Video saved: {video_recorder.frame_count} frames")
            if renderer is not None:
                renderer.close()
        
        # Update positions for next generation
        self.current_positions = []
        self.current_orientations = []
        for i in range(num_spawned):
            pos = self.tracked_geoms[i].xpos.copy()
            self.current_positions.append(pos)
            
            # Extract orientation (yaw) from quaternion
            joint_name = f"robot-{i}"
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id >= 0:
                qpos_addr = self.model.jnt_qposadr[joint_id]
                # Get quaternion from qpos (indices 3-6)
                qw = self.data.qpos[qpos_addr + 3]
                qx = self.data.qpos[qpos_addr + 4]
                qy = self.data.qpos[qpos_addr + 5]
                qz = self.data.qpos[qpos_addr + 6]
                
                # Convert quaternion to yaw angle (rotation around z-axis)
                # yaw = atan2(2*(qw*qz + qx*qy), 1 - 2*(qy^2 + qz^2))
                yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
                self.current_orientations.append(yaw)
            else:
                # Fallback if joint not found
                self.current_orientations.append(0.0)
        
        print(f"  Updated positions for next generation")
        print(f"    Tracked {len(self.current_positions)} positions")
        print(f"    Tracked {len(self.current_orientations)} orientations")
        
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
                use_periodic_boundaries=config.use_periodic_boundaries,
                mating_zone_centers=self.current_zone_centers if config.pairing_method == "mating_zone" else None,
                mating_zone_radius=config.mating_zone_radius if config.pairing_method == "mating_zone" else None,
                pairing_method=config.pairing_method
            )
    
    def create_next_generation(self) -> None:
        """
        Create next generation through movement-based pairing and reproduction.
        """
        print(f"  Creating next generation with movement-based selection...")
        
        # Deplete energy for all individuals (time-based passive depletion)
        if config.enable_energy:
            for ind in self.population:
                ind.energy -= config.energy_depletion_rate
            
            energy_values = [ind.energy for ind in self.population]
            print(f"  Energy after depletion: min={min(energy_values):.1f}, "
                  f"max={max(energy_values):.1f}, avg={np.mean(energy_values):.1f}")
        
        # Allow robots to move towards partners
        self.mating_movement_phase(
            duration=config.simulation_time, 
            save_trajectories=True
        )
        
        new_population: list[SpatialIndividual] = []
        new_positions: list[np.ndarray] = []
        new_orientations: list[float] = []
        
        # Find pairs based on selected pairing method
        pairs, paired_indices = find_pairs(
            population=self.population,
            tracked_geoms=self.tracked_geoms,
            method=config.pairing_method,
            pairing_radius=config.pairing_radius,
            world_size=(config.world_size[0], config.world_size[1]),
            use_periodic_boundaries=config.use_periodic_boundaries,
            mating_zone_centers=self.current_zone_centers if self.current_zone_centers else [tuple(config.mating_zone_center)],
            mating_zone_radius=config.mating_zone_radius
        )
        
        print(f"  Created {len(pairs)} pairs from {self.population_size} robots")
        print(f"  Pairing method: {config.pairing_method}")
        print(f"  Unpaired robots: {self.population_size - len(paired_indices)}")
        
        # Record mating statistics
        self.data_collector.record_mating_stats(
            num_pairs=len(pairs),
            num_unpaired=self.population_size - len(paired_indices),
            population_size=self.population_size
        )
        
        # Calculate offspring positions for all pairs
        pair_positions = calculate_offspring_positions(
            pairs=pairs,
            current_positions=self.current_positions,
            offspring_radius=config.offspring_radius,
            world_size=(config.world_size[0], config.world_size[1]),
            use_periodic_boundaries=config.use_periodic_boundaries
        )
        
        # Create offspring from pairs
        for pair_idx, (parent1_idx, parent2_idx) in enumerate(pairs):
            parent1 = self.population[parent1_idx]
            parent2 = self.population[parent2_idx]
            
            # Apply mating energy effects to parents
            if config.enable_energy:
                if config.mating_energy_effect == "restore":
                    # Mating restores energy to initial value
                    parent1.energy = config.initial_energy
                    parent2.energy = config.initial_energy
                elif config.mating_energy_effect == "cost":
                    # Mating costs energy
                    parent1.energy -= config.mating_energy_amount
                    parent2.energy -= config.mating_energy_amount
                # If "none", no energy effect from mating
            
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
            new_orientations.append(np.random.uniform(0, 2 * np.pi))
            
            new_population.append(child2)
            new_positions.append(pair_positions[pair_idx][1])
            new_orientations.append(np.random.uniform(0, 2 * np.pi))
        
        # Extend population with offspring
        self.population.extend(new_population)
        self.current_positions.extend(new_positions)
        self.current_orientations.extend(new_orientations)
        
        # Verify consistency
        if len(self.population) != len(self.current_positions):
            print(f"  ERROR: Population size ({len(self.population)}) != "
                  f"Position count ({len(self.current_positions)})")
            raise RuntimeError("Population and position arrays out of sync!")
        
        if len(self.population) != len(self.current_orientations):
            print(f"  ERROR: Population size ({len(self.population)}) != "
                  f"Orientation count ({len(self.current_orientations)})")
            raise RuntimeError("Population and orientation arrays out of sync!")
        
        # Update population size before selection
        old_size = self.population_size
        self.population_size = len(self.population)
        
        print(f"  Population extended from {old_size} to {self.population_size} individuals")
        print(f"  Added {len(new_population)} offspring")
        print(f"  Position tracking updated: {len(self.current_positions)} positions")
        print(f"  Paired individuals: {len(paired_indices)} out of {old_size}")
        
        # Record reproduction statistics
        self.data_collector.record_reproduction(
            num_offspring=len(new_population),
            population_before=old_size
        )
        
        # Report mating energy effects
        if config.enable_energy:
            energy_values = [ind.energy for ind in self.population]
            print(f"  Energy after mating: min={min(energy_values):.1f}, "
                  f"max={max(energy_values):.1f}, avg={np.mean(energy_values):.1f}")
            if config.mating_energy_effect == "restore":
                print(f"  Mating effect: Energy restored to {config.initial_energy} for {len(pairs)} mating pairs")
            elif config.mating_energy_effect == "cost":
                print(f"  Mating effect: Energy cost of {config.mating_energy_amount} for {len(pairs)} mating pairs")
            
            # Record energy stats after mating
            self.data_collector.record_energy_stats(self.population, "after_mating")
        
        # Show fitness statistics before selection
        fitness_values = [ind.fitness for ind in self.population]
        print(f"  Pre-selection fitness range: {min(fitness_values):.4f} to {max(fitness_values):.4f}")
        print(f"  Pre-selection fitness variation: {max(fitness_values) - min(fitness_values):.4f}")
        
        # Record population size before selection
        population_before_selection = len(self.population)
        
        # Apply selection to manage population size
        self.population, self.current_positions, self.population_size, self.current_orientations = apply_selection(
            population=self.population,
            current_positions=self.current_positions,
            method=config.selection_method,
            target_size=config.max_population_size,
            current_generation=self.generation,
            current_orientations=self.current_orientations,
            paired_indices=paired_indices,
            max_age=config.max_age
        )
        
        # Record selection statistics
        self.data_collector.record_selection(
            population_before=population_before_selection,
            population_after=len(self.population)
        )
    
    def run_evolution(self) -> SpatialIndividual:
        """
        Run the evolutionary algorithm.
            
        Returns:
            Best individual found
        """
        print("=" * 60)
        print("SPATIAL EVOLUTIONARY ALGORITHM")
        print("=" * 60)
        print(f"Population size: {self.population_size}")
        print(f"Generations: {self.num_generations}")
        print(f"Robot joints: {self.num_joints}")
        if config.record_generation_videos:
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
            
            # Record generation start
            self.data_collector.record_generation_start(gen, len(self.population))
            
            # Update mating zones if dynamic
            self._update_mating_zones()
            
            # Spawn population in simulation
            self.spawn_population()
            
            # Evaluate fitness
            fitness_values = self.evaluate_population_fitness()
            
            # Record fitness statistics
            self.data_collector.record_fitness_stats(self.population, gen)
            
            # Record age statistics
            self.data_collector.record_age_stats(self.population, gen)
            
            # Record genotype diversity
            self.data_collector.record_genotype_diversity(self.population)
            
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
                self.create_next_generation()
            else:
                # Run mating movement to capture final positions
                self.mating_movement_phase(
                    duration=config.simulation_time, 
                    save_trajectories=True
                )
        
        print(f"\n{'='*60}")
        print("EVOLUTION COMPLETE")
        print(f"{'='*60}")
        
        # Save collected data
        print(f"\nSaving evolution data...")
        summary = self.data_collector.get_summary_stats()
        print(f"\nEvolution Summary:")
        print(f"  Total generations: {summary['total_generations']}")
        print(f"  Population: {summary['population']['initial']} → {summary['population']['final']}")
        print(f"  Best fitness ever: {summary['fitness']['best_ever']:.4f}")
        print(f"  Total births: {summary['total_births']}")
        print(f"  Total deaths: {summary['total_deaths']}")
        print(f"  Avg mating success: {summary['avg_mating_success_rate']:.1f}%")
        
        self.data_collector.save_to_csv(config.results_folder)
        self.data_collector.save_to_npz(config.results_folder)
        self.data_collector.plot_evolution_statistics(config.figures_folder)
        
        # Save final population controllers
        self.save_final_controllers()
        
        return self.get_best_individual()
    
    def save_final_controllers(self) -> None:
        """
        Save the final population's controllers (genotypes) to files.
        
        Saves in multiple formats:
        - JSON: Human-readable with individual metadata
        - NPZ: NumPy format for fast loading
        - TXT: Simple readable format for best individual
        """
        from datetime import datetime
        from pathlib import Path
        import json
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_folder = Path(config.results_folder)
        results_folder.mkdir(parents=True, exist_ok=True)
        
        # 1. Save all controllers as JSON with metadata
        json_path = results_folder / f"final_controllers_{timestamp}.json"
        controllers_data = {
            'timestamp': timestamp,
            'generation': self.generation,
            'num_joints': self.num_joints,
            'population_size': len(self.population),
            'config': {
                'selection_method': config.selection_method,
                'enable_energy': config.enable_energy,
                'mating_energy_effect': config.mating_energy_effect if config.enable_energy else None,
            },
            'controllers': []
        }
        
        for ind in self.population:
            controller = {
                'unique_id': ind.unique_id,
                'generation_born': ind.generation,
                'age': self.generation - ind.generation,
                'fitness': ind.fitness,
                'energy': ind.energy if config.enable_energy else None,
                'genotype': ind.genotype,
                'parent_ids': ind.parent_ids,
                'position': ind.spawn_position.tolist() if ind.spawn_position is not None else None,
            }
            controllers_data['controllers'].append(controller)
        
        # Sort by fitness (best first)
        controllers_data['controllers'].sort(key=lambda x: x['fitness'], reverse=True)
        
        with open(json_path, 'w') as f:
            json.dump(controllers_data, f, indent=2)
        print(f"  Final controllers saved to JSON: {json_path}")
        
        # 2. Save all genotypes as NPZ (fast loading for analysis)
        npz_path = results_folder / f"final_genotypes_{timestamp}.npz"
        genotype_array = np.array([ind.genotype for ind in self.population])
        fitness_array = np.array([ind.fitness for ind in self.population])
        energy_array = np.array([ind.energy for ind in self.population]) if config.enable_energy else None
        age_array = np.array([self.generation - ind.generation for ind in self.population])
        id_array = np.array([ind.unique_id for ind in self.population])
        
        npz_data = {
            'genotypes': genotype_array,
            'fitness': fitness_array,
            'ages': age_array,
            'ids': id_array,
            'num_joints': self.num_joints,
            'generation': self.generation,
        }
        if energy_array is not None:
            npz_data['energy'] = energy_array
        
        np.savez(npz_path, **npz_data)
        print(f"  Final genotypes saved to NPZ: {npz_path}")
        
        # 3. Save best controller in human-readable format
        best = self.get_best_individual()
        txt_path = results_folder / f"best_controller_{timestamp}.txt"
        
        with open(txt_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("BEST CONTROLLER\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Individual ID: {best.unique_id}\n")
            f.write(f"Born in Generation: {best.generation}\n")
            f.write(f"Age: {self.generation - best.generation} generations\n")
            f.write(f"Fitness: {best.fitness:.6f}\n")
            if config.enable_energy:
                f.write(f"Energy: {best.energy:.2f}\n")
            f.write(f"Parent IDs: {best.parent_ids}\n")
            f.write(f"\nGenotype ({len(best.genotype)} values for {self.num_joints} joints):\n")
            f.write("-" * 60 + "\n")
            
            # Format genotype as joint parameters
            for j in range(self.num_joints):
                if j * 3 + 2 < len(best.genotype):
                    amp = best.genotype[j * 3]
                    freq = best.genotype[j * 3 + 1]
                    phase = best.genotype[j * 3 + 2]
                    f.write(f"Joint {j}:\n")
                    f.write(f"  Amplitude: {amp:.6f}\n")
                    f.write(f"  Frequency: {freq:.6f}\n")
                    f.write(f"  Phase:     {phase:.6f}\n")
            
            f.write("\n" + "-" * 60 + "\n")
            f.write("Raw genotype array:\n")
            f.write(str(best.genotype) + "\n")
        
        print(f"  Best controller saved to TXT: {txt_path}")
        
        # Summary
        print(f"\n  Saved {len(self.population)} controllers from final population")
        print(f"  Best fitness: {best.fitness:.6f} (ID: {best.unique_id})")
    
    @staticmethod
    def load_controllers_from_json(json_path: str) -> dict:
        """
        Load controllers from a saved JSON file.
        
        Args:
            json_path: Path to the JSON file
            
        Returns:
            Dictionary containing all controller data
        """
        import json
        with open(json_path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def load_genotypes_from_npz(npz_path: str) -> dict:
        """
        Load genotypes from a saved NPZ file.
        
        Args:
            npz_path: Path to the NPZ file
            
        Returns:
            Dictionary with arrays: genotypes, fitness, ages, ids
        """
        data = np.load(npz_path)
        return {
            'genotypes': data['genotypes'],
            'fitness': data['fitness'],
            'ages': data['ages'],
            'ids': data['ids'],
            'num_joints': int(data['num_joints']),
            'generation': int(data['generation']),
            'energy': data['energy'] if 'energy' in data else None,
        }
    
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
    
    # Run evolution - video recording controlled by ea_config.yaml
    spatial_ea.run_evolution()
    
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
