"""
Data collection system for evolutionary algorithm statistics.

This module tracks all relevant statistics across generations for analysis
and visualization: population size, fitness, energy, age, mating success, etc.
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from spatial_individual import SpatialIndividual


class EvolutionDataCollector:
    """
    Collects and stores evolution statistics across generations.
    
    Tracks:
    - Population dynamics (size, births, deaths)
    - Fitness statistics (best, average, worst, std)
    - Energy statistics (min, max, average, std) if energy enabled
    - Age distribution (min, max, average, std)
    - Mating statistics (pairs formed, unpaired individuals, success rate)
    - Generation-specific events
    """
    
    def __init__(self, config: Any = None):
        """
        Initialize the data collector.
        
        Args:
            config: Configuration object (optional, for metadata)
        """
        self.config = config
        self.start_time = datetime.now()
        
        # Generation-level data storage (lists indexed by generation)
        self.generations: list[int] = []
        
        # Population statistics
        self.population_size: list[int] = []
        self.births: list[int] = []  # Offspring created
        self.deaths: list[int] = []  # Individuals removed by selection
        
        # Fitness statistics
        self.fitness_best: list[float] = []
        self.fitness_avg: list[float] = []
        self.fitness_worst: list[float] = []
        self.fitness_std: list[float] = []
        
        # Energy statistics (if enabled)
        self.energy_min: list[float] = []
        self.energy_max: list[float] = []
        self.energy_avg: list[float] = []
        self.energy_std: list[float] = []
        self.energy_depleted_count: list[int] = []  # Individuals with energy <= 0
        
        # Age statistics
        self.age_min: list[int] = []
        self.age_max: list[int] = []
        self.age_avg: list[float] = []
        self.age_std: list[float] = []
        
        # Mating statistics
        self.mating_pairs: list[int] = []
        self.unpaired_individuals: list[int] = []
        self.mating_success_rate: list[float] = []  # pairs / (population / 2)
        
        # Diversity metrics
        self.genotype_diversity: list[float] = []  # Std of all genotype values
        
        # Event tracking
        self.events: dict[int, list[str]] = {}  # generation -> list of event descriptions
        
        # Early stopping information
        self.stopped_early: bool = False
        self.stop_reason: str = ""
        self.planned_generations: int = 0
    
    def record_generation_start(self, generation: int, population_size: int) -> None:
        """Record the start of a generation."""
        self.generations.append(generation)
        self.population_size.append(population_size)
        self.events[generation] = []
    
    def record_extinct_generation(self, generation: int) -> None:
        """
        Record a generation with 0 population (extinction).
        Adds the generation with placeholder values for all stats.
        
        Args:
            generation: Generation number
        """
        self.generations.append(generation)
        self.population_size.append(0)
        self.events[generation] = []
        
        # Add placeholder values for fitness stats (use 0.0)
        self.fitness_best.append(0.0)
        self.fitness_avg.append(0.0)
        self.fitness_worst.append(0.0)
        self.fitness_std.append(0.0)
        
        # Add placeholder values for age stats
        self.age_min.append(0)
        self.age_max.append(0)
        self.age_avg.append(0.0)
        self.age_std.append(0.0)
        
        # Add placeholder for genotype diversity
        self.genotype_diversity.append(0.0)
        
        # Don't add to births/deaths/mating as those are offset by 1 generation
    
    def record_fitness_stats(
        self, 
        population: list[SpatialIndividual],
        generation: int
    ) -> None:
        """
        Record fitness statistics for the current population.
        
        Args:
            population: Current population
            generation: Current generation number
        """
        fitness_values = [ind.fitness for ind in population]
        
        self.fitness_best.append(max(fitness_values))
        self.fitness_avg.append(np.mean(fitness_values))
        self.fitness_worst.append(min(fitness_values))
        self.fitness_std.append(np.std(fitness_values))
    
    def record_energy_stats(
        self, 
        population: list[SpatialIndividual],
        stage: str = ""
    ) -> None:
        """
        Record energy statistics for the current population.
        
        Args:
            population: Current population
            stage: Stage description (e.g., "after_depletion", "after_mating")
        """
        energy_values = [ind.energy for ind in population]
        
        self.energy_min.append(min(energy_values))
        self.energy_max.append(max(energy_values))
        self.energy_avg.append(np.mean(energy_values))
        self.energy_std.append(np.std(energy_values))
        self.energy_depleted_count.append(sum(1 for e in energy_values if e <= 0))
    
    def record_age_stats(
        self, 
        population: list[SpatialIndividual],
        current_generation: int
    ) -> None:
        """
        Record age statistics for the current population.
        
        Args:
            population: Current population
            current_generation: Current generation number
        """
        ages = [current_generation - ind.generation for ind in population]
        
        self.age_min.append(min(ages) if ages else 0)
        self.age_max.append(max(ages) if ages else 0)
        self.age_avg.append(np.mean(ages) if ages else 0.0)
        self.age_std.append(np.std(ages) if ages else 0.0)
    
    def record_mating_stats(
        self, 
        num_pairs: int,
        num_unpaired: int,
        population_size: int
    ) -> None:
        """
        Record mating statistics.
        
        Args:
            num_pairs: Number of successful mating pairs
            num_unpaired: Number of unpaired individuals
            population_size: Total population size before mating
        """
        self.mating_pairs.append(num_pairs)
        self.unpaired_individuals.append(num_unpaired)
        
        # Calculate mating success rate (what % of potential pairs formed)
        max_possible_pairs = population_size // 2
        success_rate = (num_pairs / max_possible_pairs * 100) if max_possible_pairs > 0 else 0.0
        self.mating_success_rate.append(success_rate)
    
    def record_reproduction(self, num_offspring: int, population_before: int) -> None:
        """
        Record reproduction statistics.
        
        Args:
            num_offspring: Number of offspring created
            population_before: Population size before adding offspring
        """
        self.births.append(num_offspring)
    
    def record_selection(self, population_before: int, population_after: int) -> None:
        """
        Record selection statistics.
        
        Args:
            population_before: Population size before selection
            population_after: Population size after selection
        """
        deaths = population_before - population_after
        self.deaths.append(deaths)
    
    def record_genotype_diversity(self, population: list[SpatialIndividual]) -> None:
        """
        Record genotype diversity metric.
        
        Args:
            population: Current population
        """
        if not population:
            self.genotype_diversity.append(0.0)
            return
        
        # HyperNEAT genome - measure diversity of connection weights
        sample_genotype = population[0].genotype
        
        if isinstance(sample_genotype, dict):
            all_weights = []
            for ind in population:
                if 'connections' in ind.genotype:
                    all_weights.extend([c.weight for c in ind.genotype['connections'] if c.enabled])
            diversity = np.std(all_weights) if all_weights else 0.0
        else:
            # Fallback for unexpected genotype format
            diversity = 0.0
        
        self.genotype_diversity.append(diversity)
    
    def add_event(self, generation: int, event_description: str) -> None:
        """
        Add an event note for a specific generation.
        
        Args:
            generation: Generation number
            event_description: Description of the event
        """
        if generation not in self.events:
            self.events[generation] = []
        self.events[generation].append(event_description)
    
    def record_early_stop(self, generation: int, reason: str, planned_generations: int) -> None:
        """
        Record that evolution stopped early.
        
        Args:
            generation: Generation at which evolution stopped
            reason: Reason for early stopping
            planned_generations: Originally planned number of generations
        """
        self.stopped_early = True
        self.stop_reason = reason
        self.planned_generations = planned_generations
        self.add_event(generation, f"EARLY STOP: {reason}")
    
    def get_summary_stats(self) -> dict[str, Any]:
        """
        Get summary statistics across all generations.
        
        Returns:
            Dictionary of summary statistics
        """
        summary = {
            'total_generations': len(self.generations),
            'planned_generations': self.planned_generations if self.stopped_early else len(self.generations),
            'stopped_early': self.stopped_early,
            'stop_reason': self.stop_reason if self.stopped_early else "Completed all generations",
            'run_duration': str(datetime.now() - self.start_time),
            'population': {
                'initial': self.population_size[0] if self.population_size else 0,
                'final': self.population_size[-1] if self.population_size else 0,
                'max': max(self.population_size) if self.population_size else 0,
                'min': min(self.population_size) if self.population_size else 0,
                'avg': np.mean(self.population_size) if self.population_size else 0.0,
            },
            'fitness': {
                'best_ever': max(self.fitness_best) if self.fitness_best else 0.0,
                'final_best': self.fitness_best[-1] if self.fitness_best else 0.0,
                'avg_improvement': (
                    self.fitness_avg[-1] - self.fitness_avg[0]
                    if len(self.fitness_avg) > 1 else 0.0
                ),
            },
            'total_births': sum(self.births) if self.births else 0,
            'total_deaths': sum(self.deaths) if self.deaths else 0,
            'avg_mating_success_rate': (
                np.mean(self.mating_success_rate) if self.mating_success_rate else 0.0
            ),
        }
        
        # Add energy stats if available
        if self.energy_avg:
            summary['energy'] = {
                'final_avg': self.energy_avg[-1],
                'final_min': self.energy_min[-1],
                'total_depleted': sum(self.energy_depleted_count),
            }
        
        return summary
    
    def save_to_csv(self, output_folder: str = "./__results__") -> str:
        """
        Save all collected data to CSV files.
        
        Args:
            output_folder: Folder to save CSV files
            
        Returns:
            Path to the main CSV file
        """
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = Path(output_folder) / f"evolution_data_{timestamp}.csv"
        
        # Prepare data rows
        with open(csv_path, 'w', newline='') as f:
            # Determine all columns
            fieldnames = [
                'generation', 'population_size', 'births', 'deaths',
                'fitness_best', 'fitness_avg', 'fitness_worst', 'fitness_std',
                'age_min', 'age_max', 'age_avg', 'age_std',
                'mating_pairs', 'unpaired', 'mating_success_rate',
                'genotype_diversity'
            ]
            
            # Add energy columns if data exists
            if self.energy_avg:
                fieldnames.extend([
                    'energy_min', 'energy_max', 'energy_avg', 'energy_std',
                    'energy_depleted_count'
                ])
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            # Write data rows
            for i, gen in enumerate(self.generations):
                row = {
                    'generation': gen,
                    'population_size': self.population_size[i] if i < len(self.population_size) else '',
                    'births': self.births[i] if i < len(self.births) else '',
                    'deaths': self.deaths[i] if i < len(self.deaths) else '',
                    'fitness_best': self.fitness_best[i] if i < len(self.fitness_best) else '',
                    'fitness_avg': self.fitness_avg[i] if i < len(self.fitness_avg) else '',
                    'fitness_worst': self.fitness_worst[i] if i < len(self.fitness_worst) else '',
                    'fitness_std': self.fitness_std[i] if i < len(self.fitness_std) else '',
                    'age_min': self.age_min[i] if i < len(self.age_min) else '',
                    'age_max': self.age_max[i] if i < len(self.age_max) else '',
                    'age_avg': self.age_avg[i] if i < len(self.age_avg) else '',
                    'age_std': self.age_std[i] if i < len(self.age_std) else '',
                    'mating_pairs': self.mating_pairs[i] if i < len(self.mating_pairs) else '',
                    'unpaired': self.unpaired_individuals[i] if i < len(self.unpaired_individuals) else '',
                    'mating_success_rate': self.mating_success_rate[i] if i < len(self.mating_success_rate) else '',
                    'genotype_diversity': self.genotype_diversity[i] if i < len(self.genotype_diversity) else '',
                }
                
                # Add energy data if available
                if self.energy_avg and i < len(self.energy_avg):
                    row.update({
                        'energy_min': self.energy_min[i],
                        'energy_max': self.energy_max[i],
                        'energy_avg': self.energy_avg[i],
                        'energy_std': self.energy_std[i],
                        'energy_depleted_count': self.energy_depleted_count[i],
                    })
                
                writer.writerow(row)
        
        print(f"\n  Data saved to CSV: {csv_path}")
        return str(csv_path)
    
    def save_to_npz(self, output_folder: str = "./__results__") -> str:
        """
        Save all collected data to NumPy NPZ file.
        
        Args:
            output_folder: Folder to save NPZ file
            
        Returns:
            Path to the NPZ file
        """
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        npz_path = Path(output_folder) / f"evolution_data_{timestamp}.npz"
        
        # Prepare data dictionary
        data_dict = {
            'generations': np.array(self.generations),
            'population_size': np.array(self.population_size),
            'births': np.array(self.births),
            'deaths': np.array(self.deaths),
            'fitness_best': np.array(self.fitness_best),
            'fitness_avg': np.array(self.fitness_avg),
            'fitness_worst': np.array(self.fitness_worst),
            'fitness_std': np.array(self.fitness_std),
            'age_min': np.array(self.age_min),
            'age_max': np.array(self.age_max),
            'age_avg': np.array(self.age_avg),
            'age_std': np.array(self.age_std),
            'mating_pairs': np.array(self.mating_pairs),
            'unpaired': np.array(self.unpaired_individuals),
            'mating_success_rate': np.array(self.mating_success_rate),
            'genotype_diversity': np.array(self.genotype_diversity),
        }
        
        # Add energy data if available
        if self.energy_avg:
            data_dict.update({
                'energy_min': np.array(self.energy_min),
                'energy_max': np.array(self.energy_max),
                'energy_avg': np.array(self.energy_avg),
                'energy_std': np.array(self.energy_std),
                'energy_depleted_count': np.array(self.energy_depleted_count),
            })
        
        # Save summary as metadata
        summary = self.get_summary_stats()
        data_dict['summary_json'] = json.dumps(summary)
        
        np.savez(npz_path, **data_dict)
        
        print(f"  Data saved to NPZ: {npz_path}")
        return str(npz_path)
    
    def plot_evolution_statistics(self, output_folder: str = "./__figures__") -> str:
        """
        Generate comprehensive visualization of evolution statistics.
        
        Args:
            output_folder: Folder to save the plot
            
        Returns:
            Path to the saved figure
        """
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig_path = Path(output_folder) / f"evolution_statistics_{timestamp}.png"
        
        # Determine number of subplots based on available data
        num_plots = 5 if self.energy_avg else 4
        
        fig, axes = plt.subplots(num_plots, 1, figsize=(12, 3 * num_plots))
        fig.suptitle('Evolution Statistics Over Time', fontsize=16, fontweight='bold')
        
        generations = np.array(self.generations)
        
        # 1. Population dynamics
        ax = axes[0]
        ax.plot(generations, self.population_size, 'b-', linewidth=2, label='Population Size')
        if self.births:
            # Births/deaths start from generation 1 (after first reproduction)
            # Only plot births/deaths data that corresponds to completed generations
            num_birth_data = min(len(self.births), len(generations) - 1)
            if num_birth_data > 0:
                birth_gens = generations[1:num_birth_data+1]
                ax.plot(birth_gens, self.births[:num_birth_data], 'g--', label='Births', alpha=0.7)
        if self.deaths:
            num_death_data = min(len(self.deaths), len(generations) - 1)
            if num_death_data > 0:
                death_gens = generations[1:num_death_data+1]
                ax.plot(death_gens, self.deaths[:num_death_data], 'r--', label='Deaths', alpha=0.7)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Count')
        ax.set_title('Population Dynamics')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Fitness evolution
        ax = axes[1]
        ax.plot(generations, self.fitness_best, 'g-', linewidth=2, label='Best')
        ax.plot(generations, self.fitness_avg, 'b-', linewidth=2, label='Average')
        ax.plot(generations, self.fitness_worst, 'r-', linewidth=1, label='Worst', alpha=0.7)
        ax.fill_between(
            generations,
            np.array(self.fitness_avg) - np.array(self.fitness_std),
            np.array(self.fitness_avg) + np.array(self.fitness_std),
            alpha=0.2, color='blue', label='±1 Std'
        )
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness')
        ax.set_title('Fitness Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Age distribution
        ax = axes[2]
        ax.plot(generations, self.age_avg, 'purple', linewidth=2, label='Average Age')
        ax.plot(generations, self.age_max, 'r--', label='Max Age', alpha=0.7)
        ax.plot(generations, self.age_min, 'g--', label='Min Age', alpha=0.7)
        ax.fill_between(
            generations,
            np.array(self.age_avg) - np.array(self.age_std),
            np.array(self.age_avg) + np.array(self.age_std),
            alpha=0.2, color='purple', label='±1 Std'
        )
        ax.set_xlabel('Generation')
        ax.set_ylabel('Age (generations)')
        ax.set_title('Age Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Mating success
        ax = axes[3]
        ax2 = ax.twinx()
        
        # Use only data points where mating stats exist
        mating_gens = np.array(self.generations[:len(self.mating_pairs)])
        
        line1 = ax.plot(mating_gens, self.mating_pairs, 'b-', linewidth=2, label='Pairs Formed')
        line2 = ax.plot(mating_gens, self.unpaired_individuals, 'r--', label='Unpaired', alpha=0.7)
        line3 = ax2.plot(mating_gens, self.mating_success_rate, 'g-', linewidth=2, label='Success Rate (%)', alpha=0.8)
        
        ax.set_xlabel('Generation')
        ax.set_ylabel('Count', color='b')
        ax2.set_ylabel('Success Rate (%)', color='g')
        ax.set_title('Mating Statistics')
        
        # Combine legends
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='best')
        ax.grid(True, alpha=0.3)
        
        # 5. Energy dynamics (if available)
        if self.energy_avg and len(self.energy_avg) > 0:
            ax = axes[4]
            # Create x-axis with same length as energy data
            # Energy is only recorded during reproduction (not on last generation)
            energy_step = len(self.generations) / len(self.energy_avg) if len(self.energy_avg) > 0 else 1
            energy_x = np.arange(len(self.energy_avg)) * energy_step
            
            ax.plot(energy_x, self.energy_avg, 'orange', linewidth=2, label='Average Energy')
            ax.plot(energy_x, self.energy_max, 'g--', label='Max Energy', alpha=0.7)
            ax.plot(energy_x, self.energy_min, 'r--', label='Min Energy', alpha=0.7)
            ax.fill_between(
                energy_x,
                np.array(self.energy_avg) - np.array(self.energy_std),
                np.array(self.energy_avg) + np.array(self.energy_std),
                alpha=0.2, color='orange', label='±1 Std'
            )
            
            # Plot depleted count on secondary axis
            ax2 = ax.twinx()
            ax2.plot(energy_x, self.energy_depleted_count, 'r:', linewidth=2, label='Depleted Count', alpha=0.6)
            ax2.set_ylabel('Depleted Count', color='r')
            
            ax.set_xlabel('Generation')
            ax.set_ylabel('Energy Level')
            ax.set_title('Energy Dynamics (Recorded After Mating)')
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
        
        # Add early stopping indicator if applicable
        if self.stopped_early:
            # Add vertical line at stopping point on all subplots
            last_gen = generations[-1]
            for ax_idx in range(len(axes)):
                ax = axes[ax_idx]
                ax.axvline(x=last_gen, color='red', linestyle=':', linewidth=2, alpha=0.5)
                
            # Add title annotation
            title_text = f'Evolution Statistics Over Time\n STOPPED EARLY: {self.stop_reason}'
            fig.suptitle(title_text, fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Evolution statistics plot saved: {fig_path}")
        return str(fig_path)
