#!/usr/bin/env python3
"""
Script to run experiments comparing different EA configurations.

Usage:
    python run_experiments.py --experiment baseline
    python run_experiments.py --experiment all
    python run_experiments.py --custom
"""

import argparse
from experiment_runner import ExperimentRunner, ExperimentConfig


def define_experiments() -> dict[str, ExperimentConfig]:
    """Define all available experiments."""
    
    experiments = {}
    
    # ===== Baseline Experiments =====
    # DONE
    experiments['baseline_random_fitnessBased'] = ExperimentConfig(
        experiment_name="baseline_random_fitnessBased",
        num_runs=10,
        
        # Population parameters (from ea_config.yaml)
        population_size=30,
        num_generations=50,
        max_population_limit=100,
        min_population_limit=1,
        stop_on_limits=True,
        
        # Selection parameters
        pairing_method="random",
        movement_bias="none",
        selection_method="fitness_based",
        target_population_size=30,
        pairing_radius=100.0,
        offspring_radius=2.0,
        max_age=10,
        
        # Mutation/Crossover
        mutation_rate=0.8,
        mutation_strength=0.5,
        add_connection_rate=0.05,
        add_node_rate=0.03,
        crossover_rate=0.9,
        
        # Simulation
        simulation_time=30.0,
        use_periodic_boundaries=True,
        
        # Incubation
        incubation_enabled=False,
        
        # Output
        save_snapshots=True,
        save_trajectories=True,
        save_individual_runs=True,
    )

    experiments['baseline_random_parents_die'] = ExperimentConfig(
        experiment_name="baseline_random_parents_die",
        num_runs=5,
        
        # Population parameters (from ea_config.yaml)
        population_size=30,
        num_generations=50,
        max_population_limit=100,
        min_population_limit=1,
        stop_on_limits=True,
        
        # Selection parameters
        pairing_method="random",
        movement_bias="none",
        selection_method="parents_die",
        target_population_size=30,
        pairing_radius=100.0,
        offspring_radius=2.0,
        max_age=10,
        
        # Mutation/Crossover
        mutation_rate=0.8,
        mutation_strength=0.5,
        add_connection_rate=0.05,
        add_node_rate=0.03,
        crossover_rate=0.9,
        
        # Simulation
        simulation_time=60.0,
        use_periodic_boundaries=True,
        
        # Incubation
        incubation_enabled=False,
        
        # Output
        save_snapshots=False,
        save_trajectories=False,
        save_individual_runs=False,
    )

    experiments['baseline_random_age_based'] = ExperimentConfig(
        experiment_name="baseline_random_age_based",
        num_runs=5,
        
        # Population parameters (from ea_config.yaml)
        population_size=30,
        num_generations=50,
        max_population_limit=100,
        min_population_limit=1,
        stop_on_limits=True,
        
        # Selection parameters
        pairing_method="random",
        movement_bias="none",
        selection_method="age_based",
        target_population_size=30,
        pairing_radius=100.0,
        offspring_radius=2.0,
        
        # Mutation/Crossover
        mutation_rate=0.8,
        mutation_strength=0.5,
        add_connection_rate=0.05,
        add_node_rate=0.03,
        crossover_rate=0.9,
        
        # Simulation
        simulation_time=60.0,
        use_periodic_boundaries=True,
        
        # Incubation
        incubation_enabled=False,
        
        # Output
        save_snapshots=False,
        save_trajectories=False,
        save_individual_runs=False,
    )
    # DONE
    experiments['baseline_proximity_fitnessBased'] = ExperimentConfig(
        experiment_name="baseline_proximity_fitnessBased",
        num_runs=10,
        
        # Population parameters
        population_size=30,
        num_generations=50,
        max_population_limit=100,
        min_population_limit=1,
        stop_on_limits=True,
        
        # Selection parameters
        pairing_method="proximity_pairing",
        movement_bias="none",
        selection_method="fitness_based",
        target_population_size=30,
        pairing_radius=100.0,
        offspring_radius=2.0,
        
        # Mutation/Crossover
        mutation_rate=0.8,
        mutation_strength=0.5,
        add_connection_rate=0.05,
        add_node_rate=0.03,
        crossover_rate=0.9,
        
        # Simulation
        simulation_time=30.0,
        use_periodic_boundaries=True,
        
        # Incubation
        incubation_enabled=False,
        
        # Output
        save_snapshots=True,
        save_trajectories=True,
        save_individual_runs=True,
    )




    # ===== Nearest Neighbor Movement Bias Experiments =====
    experiments['nearest_neighbor_fitBased'] = ExperimentConfig(
        experiment_name="nearest_neighbor_fitBased",
        num_runs=10,
        
        # Incubation
        incubation_enabled=True,
        incubation_num_generations=20,
        
        # Population parameters 
        population_size=30,
        num_generations=50,
        max_population_limit=100,
        min_population_limit=1,
        stop_on_limits=True,
        
        # Selection parameters
        pairing_method="proximity_pairing",
        movement_bias="nearest_neighbor",
        selection_method="fitness_based",
        target_population_size=30,
        pairing_radius=1.5,
        offspring_radius=2.0,
        
        # Mutation/Crossover (shared with incubation)
        mutation_rate=0.8,
        mutation_strength=0.5,
        add_connection_rate=0.05,
        add_node_rate=0.03,
        crossover_rate=0.9,
        
        # Simulation
        simulation_time=30.0,
        use_periodic_boundaries=True,
        
        # Output
        save_snapshots=True,
        save_trajectories=True,
        save_individual_runs=True,
    )
    # DONE
    experiments['nearest_neighbor_energyCost'] = ExperimentConfig(
        experiment_name="nearest_neighbor_energyCost",
        num_runs=30,
        
        # Incubation
        incubation_enabled=False,
        
        # Population parameters 
        population_size=10,
        num_generations=50,
        max_population_limit=100,
        min_population_limit=1,
        stop_on_limits=True,
        
        # Selection parameters
        pairing_method="proximity_pairing",
        movement_bias="nearest_neighbor",
        selection_method="energy_based",
        pairing_radius=1.5,
        offspring_radius=3.0,

        enable_energy=True,
        initial_energy=100.0,
        energy_depletion_rate=10.0,
        mating_energy_effect="cost",
        mating_energy_amount=35.0,
        
        # Mutation/Crossover (shared with incubation)
        mutation_rate=0.8,
        mutation_strength=0.5,
        add_connection_rate=0.05,
        add_node_rate=0.03,
        crossover_rate=0.9,
        
        # Simulation
        simulation_time=30.0,
        use_periodic_boundaries=True,
        
        # Output
        save_snapshots=True,
        save_trajectories=True,
        save_individual_runs=True,
    )

    experiments['nearest_neighbor_probAge'] = ExperimentConfig(
        experiment_name="nearest_neighbor_probAge",
        num_runs=30,
        
        # Incubation
        incubation_enabled=False,
        
        # Population parameters 
        population_size=10,
        num_generations=50,
        max_population_limit=100,
        min_population_limit=1,
        stop_on_limits=True,
        
        # Selection parameters
        pairing_method="proximity_pairing",
        movement_bias="nearest_neighbor",
        selection_method="probabilistic_age",
        target_population_size=30,
        pairing_radius=1.5,
        offspring_radius=3.0,
        max_age=10,
        
        # Mutation/Crossover (shared with incubation)
        mutation_rate=0.8,
        mutation_strength=0.5,
        add_connection_rate=0.05,
        add_node_rate=0.03,
        crossover_rate=0.9,
        
        # Simulation
        simulation_time=30.0,
        use_periodic_boundaries=True,
        
        # Output
        save_snapshots=True,
        save_trajectories=True,
        save_individual_runs=True,
    )




    # ===== Mating Zone Movement Bias Experiments =====
    
    # Static zones - never move
    experiments['static_matingZone_assignedMating_fitBased'] = ExperimentConfig(
        experiment_name="static_matingZone_assignedMating_fitBased",
        num_runs=10,
        
        # Incubation
        incubation_enabled=False,
        incubation_num_generations=20,
        
        # Population parameters 
        population_size=20,
        num_generations=50,
        max_population_limit=100,
        min_population_limit=1,
        stop_on_limits=True,
        
        # Selection parameters
        pairing_method="mating_zone",
        movement_bias="assigned_zone",
        selection_method="fitness_based",
        target_population_size=30,
        pairing_radius=10.0,
        offspring_radius=2.0,

        mating_zone_radius=2.5,
        num_mating_zones=4,
        zone_relocation_strategy="static",  # Zones never move
        min_zone_distance=2.0,
        
        # Mutation/Crossover
        mutation_rate=0.8,
        mutation_strength=0.5,
        add_connection_rate=0.05,
        add_node_rate=0.03,
        crossover_rate=0.9,
        
        # Simulation
        simulation_time=60.0,
        use_periodic_boundaries=True,
        
        # Output
        save_snapshots=True,
        save_trajectories=True,
        save_individual_runs=True,
    )
    
    # Generation interval zones - move every N generations
    # DONE
    experiments['interval_matingZone_assignedMating_fitBased'] = ExperimentConfig(
        experiment_name="interval_matingZone_assignedMating_fitBased",
        num_runs=10,
        
        # Incubation
        incubation_enabled=False,
        incubation_num_generations=20,
        
        # Population parameters 
        population_size=20,
        num_generations=50,
        max_population_limit=100,
        min_population_limit=1,
        stop_on_limits=True,
        
        # Selection parameters
        pairing_method="mating_zone",
        movement_bias="assigned_zone",
        selection_method="fitness_based",
        target_population_size=30,
        pairing_radius=10.0,
        offspring_radius=3.0,

        mating_zone_radius=2.0,
        num_mating_zones=6,
        zone_relocation_strategy="generation_interval",  # Move every N generations
        zone_change_interval=10,
        min_zone_distance=2.0,
        
        # Mutation/Crossover
        mutation_rate=0.8,
        mutation_strength=0.5,
        add_connection_rate=0.05,
        add_node_rate=0.03,
        crossover_rate=0.9,
        
        # Simulation
        simulation_time=60.0,
        use_periodic_boundaries=True,
        
        # Output
        save_snapshots=True,
        save_trajectories=True,
        save_individual_runs=True,
    )
    
    # Event-driven zones - move after matings occur (NEW!)
    experiments['eventDriven_matingZone_assignedMating_fitBased'] = ExperimentConfig(
        experiment_name="eventDriven_matingZone_assignedMating_fitBased",
        num_runs=10,
        
        # Incubation
        incubation_enabled=True,
        incubation_num_generations=20,
        
        # Population parameters 
        population_size=30,
        num_generations=100,
        stop_on_limits=True,
        
        # Selection parameters
        pairing_method="mating_zone",
        movement_bias="assigned_zone",
        selection_method="fitness_based",
        target_population_size=30,
        pairing_radius=10.0,
        offspring_radius=3.0,

        mating_zone_radius=2.0,
        num_mating_zones=15,
        zone_relocation_strategy="event_driven",
        min_zone_distance=2.0,
        
        # Mutation/Crossover
        mutation_rate=0.8,
        mutation_strength=0.5,
        add_connection_rate=0.05,
        add_node_rate=0.03,
        crossover_rate=0.9,
        
        # Simulation
        simulation_time=30.0,
        use_periodic_boundaries=True,
        
        # Output
        save_snapshots=True,
        save_trajectories=True,
        save_individual_runs=True,
    )
    
    # ===== Selection Method Experiments =====
    
    experiments['fitness_selection'] = ExperimentConfig(
        experiment_name="fitness_based_selection",
        num_runs=5,
        selection_method="fitness_based",
        target_population_size=30,
        pairing_method="random",
        movement_bias="none",
        population_size=10,
        num_generations=50,
        save_snapshots=False,
        save_trajectories=False,
        save_individual_runs=False,
    )
    
    experiments['energy_selection'] = ExperimentConfig(
        experiment_name="energy_based_selection",
        num_runs=5,
        selection_method="energy_based",
        enable_energy=True,
        initial_energy=100.0,
        energy_depletion_rate=15.0,
        mating_energy_effect="restore",
        pairing_method="random",
        movement_bias="none",
        population_size=10,
        num_generations=50,
        save_snapshots=False,
        save_trajectories=False,
        save_individual_runs=False,
    )
    
    # ===== Population Control Experiments =====
    
    experiments['no_control'] = ExperimentConfig(
        experiment_name="no_population_control",
        num_runs=10,
        selection_method="parents_die",
        pairing_method="random",
        movement_bias="none",
        population_size=10,
        num_generations=50,
        save_snapshots=False,
        save_trajectories=False,
        save_individual_runs=False,
    )
    
    experiments['fitness_control'] = ExperimentConfig(
        experiment_name="fitness_control_target_30",
        num_runs=10,
        selection_method="fitness_based",
        target_population_size=30,
        pairing_method="random",
        movement_bias="none",
        population_size=10,
        num_generations=50,
        save_snapshots=False,
        save_trajectories=False,
        save_individual_runs=False,
    )
    
    # ===== Mutation Rate Experiments =====
    
    experiments['low_mutation'] = ExperimentConfig(
        experiment_name="low_mutation_rate",
        num_runs=5,
        mutation_rate=0.1,
        mutation_strength=0.3,
        pairing_method="random",
        movement_bias="none",
        population_size=10,
        num_generations=50,
        save_snapshots=False,
        save_trajectories=False,
        save_individual_runs=False,
    )
    
    experiments['high_mutation'] = ExperimentConfig(
        experiment_name="high_mutation_rate",
        num_runs=5,
        mutation_rate=0.9,
        mutation_strength=0.5,
        pairing_method="random",
        movement_bias="none",
        population_size=10,
        num_generations=50,
        save_snapshots=False,
        save_trajectories=False,
        save_individual_runs=False,
    )
    
    return experiments


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run evolutionary algorithm experiments"
    )
    parser.add_argument(
        '--experiment',
        type=str,
        default='baseline_random',
        help='Experiment name to run (or "all" for all experiments)'
    )
    parser.add_argument(
        '--runs',
        type=int,
        default=None,
        help='Override number of runs per experiment'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='__experiments__',
        help='Base output directory'
    )
    parser.add_argument(
        '--compare',
        nargs='+',
        help='List of experiments to compare (after running)'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available experiments'
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Run trials in parallel using multiprocessing'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=None,
        help='Number of parallel workers (default: CPU count - 1)'
    )
    parser.add_argument(
        '--grid-file',
        type=str,
        default=None,
        help='YAML file specifying a parameter grid for grid search (keys are ExperimentConfig attrs)'
    )
    parser.add_argument(
        '--grid-base-experiment',
        type=str,
        default=None,
        help='Name of the experiment defined in this script to use as base for the grid search'
    )
    parser.add_argument(
        '--grid-sequential',
        action='store_true',
        help='Run each grid point sequentially (still can run trials in parallel inside each experiment)'
    )
    
    args = parser.parse_args()
    
    # Define experiments
    experiments = define_experiments()
    
    # List experiments if requested
    if args.list:
        print("\nAvailable experiments:")
        print("=" * 60)
        for name, config in experiments.items():
            print(f"  {name:25s} - {config.experiment_name}")
        print("=" * 60)
        print("\nUse --experiment <name> to run a specific experiment")
        print("Use --experiment all to run all experiments")
        return
    
    # Create runner
    runner = ExperimentRunner(base_output_dir=args.output_dir)
    
    # Determine which experiments to run
    if args.experiment == 'all':
        experiments_to_run = list(experiments.values())
    else:
        # If user requested a grid search, handle separately
        if args.grid_file is not None:
            # grid_base_experiment must be provided
            if args.grid_base_experiment is None:
                print("Error: --grid-base-experiment must be provided when using --grid-file")
                return

            if args.grid_base_experiment not in experiments:
                print(f"Error: Unknown base experiment '{args.grid_base_experiment}'")
                print(f"Available: {', '.join(experiments.keys())}")
                return

            # Load grid YAML
            import yaml
            with open(args.grid_file, 'r') as f:
                grid = yaml.safe_load(f)

            base_exp = experiments[args.grid_base_experiment]
            df = runner.run_grid_search(
                base_experiment=base_exp,
                param_grid=grid,
                num_workers=args.num_workers,
                parallel_runs=not args.grid_sequential,
                verbose=True
            )

            print(f"Grid search finished. Summary:\n{df}")
            return
        if args.experiment not in experiments:
            print(f"Error: Unknown experiment '{args.experiment}'")
            print(f"Available: {', '.join(experiments.keys())}")
            print("Use --list to see all experiments")
            return
        experiments_to_run = [experiments[args.experiment]]
    
    # Override num_runs if specified
    if args.runs is not None:
        for exp in experiments_to_run:
            exp.num_runs = args.runs
    
    # Run experiments
    print(f"\nRunning {len(experiments_to_run)} experiment(s)...")
    if args.parallel:
        print(f"Parallel execution enabled with {args.num_workers or 'auto'} workers")
    
    completed_experiments = []
    for exp_config in experiments_to_run:
        try:
            # Choose parallel or sequential execution
            if args.parallel:
                results = runner.run_experiment_parallel(
                    exp_config, 
                    num_workers=args.num_workers,
                    verbose=True
                )
            else:
                results = runner.run_experiment(exp_config, verbose=True)
            
            if results:
                completed_experiments.append(exp_config.experiment_name)
        except KeyboardInterrupt:
            print("\n\nInterrupted by user!")
            break
        except Exception as e:
            print(f"\n\nError in experiment {exp_config.experiment_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create comparison if multiple experiments completed
    if len(completed_experiments) > 1:
        print(f"\n\nCreating comparison of {len(completed_experiments)} experiments...")
        try:
            runner.compare_experiments(
                completed_experiments,
                output_path=runner.base_output_dir / "experiment_comparison.png"
            )
        except Exception as e:
            print(f"Error creating comparison: {e}")
    
    # Create custom comparison if requested
    if args.compare:
        print(f"\n\nCreating custom comparison...")
        try:
            runner.compare_experiments(
                args.compare,
                output_path=runner.base_output_dir / "custom_comparison.png"
            )
        except Exception as e:
            print(f"Error creating custom comparison: {e}")
    
    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE!")
    print(f"Results saved to: {runner.base_output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
