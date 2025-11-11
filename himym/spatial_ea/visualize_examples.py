"""
Example script demonstrating how to use the experiment visualizer.

This shows various ways to analyze and visualize experiment data.
"""

from pathlib import Path
from visualize_experiment import ExperimentVisualizer


def example_1_basic_usage():
    """Example 1: Basic visualization of most recent experiment."""
    print("=" * 70)
    print("EXAMPLE 1: Basic Usage - Visualize Most Recent Experiment")
    print("=" * 70)
    
    # Find most recent experiment
    results_dir = Path(__file__).parent.parent.parent / '__results__'
    csv_files = list(results_dir.glob('evolution_data_*.csv'))
    
    if not csv_files:
        print("No experiment data found. Run main_spatial_ea.py first.")
        return
    
    latest = max(csv_files, key=lambda p: p.stat().st_mtime)
    print(f"\nAnalyzing: {latest.name}\n")
    
    # Create visualizer
    viz = ExperimentVisualizer(latest)
    
    # Generate complete report
    viz.generate_report()
    
    print("\n✓ Complete analysis report generated!\n")


def example_2_custom_plots():
    """Example 2: Create individual custom plots."""
    print("=" * 70)
    print("EXAMPLE 2: Custom Plots - Individual Visualizations")
    print("=" * 70)
    
    results_dir = Path(__file__).parent.parent.parent / '__results__'
    csv_files = list(results_dir.glob('evolution_data_*.csv'))
    
    if not csv_files:
        print("No experiment data found.")
        return
    
    latest = max(csv_files, key=lambda p: p.stat().st_mtime)
    print(f"\nAnalyzing: {latest.name}\n")
    
    viz = ExperimentVisualizer(latest)
    
    # Create custom output directory
    custom_dir = results_dir / 'custom_analysis'
    custom_dir.mkdir(exist_ok=True)
    
    # Generate individual plots
    print("Creating comprehensive overview...")
    viz.plot_all(save_path=custom_dir / 'overview.png')
    
    if viz.controllers:
        print("Creating parameter distribution plot...")
        viz.plot_controller_parameters(save_path=custom_dir / 'params.png')
        
        print("Creating parameter heatmap...")
        viz.plot_parameter_evolution_heatmap(save_path=custom_dir / 'heatmap.png')
    
    print(f"\n✓ Custom plots saved to: {custom_dir}\n")


def example_3_data_exploration():
    """Example 3: Explore data programmatically."""
    print("=" * 70)
    print("EXAMPLE 3: Data Exploration - Access and Analyze Data")
    print("=" * 70)
    
    results_dir = Path(__file__).parent.parent.parent / '__results__'
    csv_files = list(results_dir.glob('evolution_data_*.csv'))
    
    if not csv_files:
        print("No experiment data found.")
        return
    
    latest = max(csv_files, key=lambda p: p.stat().st_mtime)
    print(f"\nAnalyzing: {latest.name}\n")
    
    viz = ExperimentVisualizer(latest)
    
    # Access DataFrame
    if viz.df is not None:
        print("Available metrics in CSV:")
        print(f"  {list(viz.df.columns)}\n")
        
        print("Data shape:")
        print(f"  Generations: {len(viz.df)}")
        print(f"  Metrics: {len(viz.df.columns)}\n")
        
        print("Fitness evolution:")
        print(f"  Initial best: {viz.df['fitness_best'].iloc[0]:.6f}")
        print(f"  Final best: {viz.df['fitness_best'].iloc[-1]:.6f}")
        print(f"  Best ever: {viz.df['fitness_best'].max():.6f} (gen {viz.df['fitness_best'].idxmax()})")
        improvement = viz.df['fitness_best'].iloc[-1] - viz.df['fitness_best'].iloc[0]
        print(f"  Improvement: {improvement:+.6f}\n")
        
        # Check for convergence
        window = max(1, len(viz.df) // 5)
        recent_fitness = viz.df['fitness_best'].iloc[-window:]
        fitness_change = recent_fitness.max() - recent_fitness.min()
        print(f"Convergence check (last {window} generations):")
        print(f"  Fitness change: {fitness_change:.6f}")
        if fitness_change < 0.001:
            print("  → Evolution appears to have converged! ✓")
        else:
            print("  → Still evolving...")
    
    # Access controller data
    if viz.controllers:
        print("\nFinal population:")
        controllers = viz.controllers['controllers']
        print(f"  Population size: {len(controllers)}")
        
        fitness_values = [c['fitness'] for c in controllers]
        print(f"  Best fitness: {max(fitness_values):.6f}")
        print(f"  Avg fitness: {sum(fitness_values)/len(fitness_values):.6f}")
        print(f"  Worst fitness: {min(fitness_values):.6f}")
        
        if 'energy' in controllers[0]:
            energies = [c['energy'] for c in controllers]
            print(f"\n  Energy range: {min(energies):.1f} - {max(energies):.1f}")
        
        ages = [c['age'] for c in controllers]
        print(f"  Age range: {min(ages)} - {max(ages)} generations")
    
    print("\n✓ Data exploration complete!\n")


def example_4_compare_experiments():
    """Example 4: Compare multiple experiments."""
    print("=" * 70)
    print("EXAMPLE 4: Comparison - Analyze Multiple Experiments")
    print("=" * 70)
    
    results_dir = Path(__file__).parent.parent.parent / '__results__'
    csv_files = sorted(results_dir.glob('evolution_data_*.csv'), 
                      key=lambda p: p.stat().st_mtime)
    
    if len(csv_files) < 2:
        print("Need at least 2 experiments to compare.")
        print(f"Found {len(csv_files)} experiment(s).")
        return
    
    print(f"\nFound {len(csv_files)} experiments. Comparing...\n")
    
    comparison = []
    for csv_file in csv_files[-5:]:  # Last 5 experiments
        viz = ExperimentVisualizer(csv_file)
        
        if viz.df is not None:
            exp_name = csv_file.stem.replace('evolution_data_', '')
            best_fitness = viz.df['fitness_best'].max()
            final_fitness = viz.df['fitness_best'].iloc[-1]
            generations = len(viz.df)
            
            if viz.controllers:
                selection = viz.controllers.get('config', {}).get('selection_method', 'N/A')
            else:
                selection = 'N/A'
            
            comparison.append({
                'name': exp_name,
                'generations': generations,
                'best_fitness': best_fitness,
                'final_fitness': final_fitness,
                'selection': selection
            })
    
    # Display comparison table
    print("Experiment Comparison:")
    print("-" * 90)
    print(f"{'Timestamp':<20} {'Gens':>5} {'Best Fit':>12} {'Final Fit':>12} {'Selection':<20}")
    print("-" * 90)
    
    for exp in comparison:
        print(f"{exp['name']:<20} {exp['generations']:>5} "
              f"{exp['best_fitness']:>12.6f} {exp['final_fitness']:>12.6f} "
              f"{exp['selection']:<20}")
    
    print("-" * 90)
    
    # Find best experiment
    best_exp = max(comparison, key=lambda x: x['best_fitness'])
    print(f"\nBest experiment: {best_exp['name']}")
    print(f"  Best fitness: {best_exp['best_fitness']:.6f}")
    print(f"  Selection method: {best_exp['selection']}")
    
    print("\n✓ Comparison complete!\n")


def example_5_energy_analysis():
    """Example 5: Analyze energy system dynamics."""
    print("=" * 70)
    print("EXAMPLE 5: Energy Analysis - Energy System Dynamics")
    print("=" * 70)
    
    results_dir = Path(__file__).parent.parent.parent / '__results__'
    csv_files = list(results_dir.glob('evolution_data_*.csv'))
    
    if not csv_files:
        print("No experiment data found.")
        return
    
    latest = max(csv_files, key=lambda p: p.stat().st_mtime)
    print(f"\nAnalyzing: {latest.name}\n")
    
    viz = ExperimentVisualizer(latest)
    
    if viz.df is None:
        print("No CSV data loaded.")
        return
    
    # Check if energy system is enabled
    if 'energy_avg' not in viz.df.columns:
        print("Energy system not enabled in this experiment.")
        return
    
    print("Energy system statistics:")
    print("-" * 70)
    
    # Initial and final energy
    print(f"Initial average energy: {viz.df['energy_avg'].iloc[0]:.2f}")
    print(f"Final average energy: {viz.df['energy_avg'].iloc[-1]:.2f}")
    
    energy_lost = viz.df['energy_avg'].iloc[0] - viz.df['energy_avg'].iloc[-1]
    print(f"Energy lost: {energy_lost:.2f}\n")
    
    # Depletion analysis
    if 'energy_depleted_count' in viz.df.columns:
        total_depleted = viz.df['energy_depleted_count'].sum()
        print(f"Total energy depletions: {total_depleted:.0f}")
        
        if total_depleted > 0:
            first_depletion = viz.df[viz.df['energy_depleted_count'] > 0]['generation'].iloc[0]
            print(f"First depletion: Generation {first_depletion}")
            max_depleted = viz.df['energy_depleted_count'].max()
            print(f"Max depleted in one generation: {max_depleted:.0f}\n")
        else:
            print("No energy depletions occurred.\n")
    
    # Energy configuration
    if viz.controllers:
        config = viz.controllers.get('config', {})
        if 'enable_energy' in config:
            print("Configuration:")
            print(f"  Energy enabled: {config.get('enable_energy', 'N/A')}")
            print(f"  Mating effect: {config.get('mating_energy_effect', 'N/A')}")
    
    print("\n✓ Energy analysis complete!\n")


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "EXPERIMENT VISUALIZER EXAMPLES" + " " * 23 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    examples = [
        ("1", "Basic Usage", example_1_basic_usage),
        ("2", "Custom Plots", example_2_custom_plots),
        ("3", "Data Exploration", example_3_data_exploration),
        ("4", "Compare Experiments", example_4_compare_experiments),
        ("5", "Energy Analysis", example_5_energy_analysis),
    ]
    
    print("Available examples:")
    for num, name, _ in examples:
        print(f"  {num}. {name}")
    print("  a. Run all examples")
    print()
    
    choice = input("Select example to run (1-5, a, or Enter for all): ").strip().lower()
    print()
    
    if choice == '' or choice == 'a':
        # Run all examples
        for num, name, func in examples:
            func()
            print()
    elif choice in ['1', '2', '3', '4', '5']:
        # Run specific example
        for num, name, func in examples:
            if choice == num:
                func()
                break
    else:
        print("Invalid choice.")


if __name__ == '__main__':
    main()
