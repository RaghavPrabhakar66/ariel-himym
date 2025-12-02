"""
Comprehensive visualization tools for spatial EA experiment data.

This module provides various visualization functions for analyzing evolution runs,
including time series plots, distributions, correlations, and controller analysis.
"""

import json
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec


class ExperimentVisualizer:
    """
    Visualizes all data from a spatial EA experiment run.
    
    Supports loading from CSV, NPZ, and JSON formats and creates
    comprehensive visualizations for analysis.
    """
    
    def __init__(self, data_path: Union[str, Path]):
        """
        Initialize visualizer with experiment data.
        
        Args:
            data_path: Path to CSV, NPZ, or base name (without extension)
                      Will automatically try to load related files.
        """
        self.data_path = Path(data_path)
        self.base_name = self._extract_base_name()
        self.results_dir = self.data_path.parent
        
        # Data containers
        self.df: Optional[pd.DataFrame] = None
        self.npz_data: Optional[dict] = None
        self.controllers: Optional[dict] = None
        
        # Load available data
        self._load_data()
        
    def _extract_base_name(self) -> str:
        """Extract base filename without extension."""
        stem = self.data_path.stem
        # Handle evolution_data_TIMESTAMP or final_controllers_TIMESTAMP
        if stem.startswith('evolution_data_'):
            return stem
        elif stem.startswith('final_controllers_'):
            # Extract timestamp and reconstruct evolution_data name
            timestamp = stem.replace('final_controllers_', '')
            return f'evolution_data_{timestamp}'
        elif stem.startswith('final_genotypes_'):
            timestamp = stem.replace('final_genotypes_', '')
            return f'evolution_data_{timestamp}'
        else:
            return stem
    
    def _load_data(self) -> None:
        """Load all available data files (CSV, NPZ, JSON)."""
        # Try to load CSV
        csv_path = self.results_dir / f'{self.base_name}.csv'
        if csv_path.exists():
            print(f"Loading CSV: {csv_path}")
            self.df = pd.read_csv(csv_path)
            print(f"  Loaded {len(self.df)} generations with {len(self.df.columns)} metrics")
        
        # Try to load NPZ
        npz_path = self.results_dir / f'{self.base_name}.npz'
        if npz_path.exists():
            print(f"Loading NPZ: {npz_path}")
            self.npz_data = dict(np.load(npz_path, allow_pickle=True))
            print(f"  Loaded {len(self.npz_data)} arrays")
        
        # Try to load controller JSON - search for files with similar timestamp
        # Extract timestamp from base_name (evolution_data_TIMESTAMP)
        timestamp = self.base_name.replace('evolution_data_', '')
        
        # Try exact match first
        controller_path = self.results_dir / f'final_controllers_{timestamp}.json'
        if not controller_path.exists():
            # Search for files with similar timestamp (within a few seconds)
            import glob
            pattern = str(self.results_dir / f'final_controllers_{timestamp[:13]}*.json')
            matches = glob.glob(pattern)
            if matches:
                controller_path = Path(matches[0])
        
        if controller_path.exists():
            print(f"Loading controllers: {controller_path}")
            with open(controller_path, 'r') as f:
                self.controllers = json.load(f)
            print(f"  Loaded {len(self.controllers['controllers'])} controllers")
    
    def plot_all(self, save_path: Optional[Union[str, Path]] = None) -> None:
        """
        Create comprehensive multi-panel visualization of all data.
        
        Args:
            save_path: Optional path to save figure. If None, displays interactively.
        """
        if self.df is None:
            print("No CSV data loaded. Cannot create plots.")
            return
        
        # Create large figure with many subplots
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Row 1: Population dynamics
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_population_dynamics(ax1)
        
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_births_deaths(ax2)
        
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_mating_success(ax3)
        
        # Row 2: Fitness evolution
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_fitness_evolution(ax4)
        
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_fitness_distribution(ax5)
        
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_genotype_diversity(ax6)
        
        # Row 3: Energy and Age
        ax7 = fig.add_subplot(gs[2, 0])
        self._plot_energy_evolution(ax7)
        
        ax8 = fig.add_subplot(gs[2, 1])
        self._plot_age_distribution(ax8)
        
        ax9 = fig.add_subplot(gs[2, 2])
        self._plot_energy_depletion(ax9)
        
        # Row 4: Advanced analysis
        ax10 = fig.add_subplot(gs[3, 0])
        self._plot_fitness_vs_age(ax10)
        
        ax11 = fig.add_subplot(gs[3, 1])
        self._plot_fitness_vs_energy(ax11)
        
        ax12 = fig.add_subplot(gs[3, 2])
        self._plot_summary_statistics(ax12)
        
        plt.suptitle(f'Experiment Analysis: {self.base_name}', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Comprehensive visualization saved: {save_path}")
        else:
            plt.show()
    
    def _plot_population_dynamics(self, ax) -> None:
        """Plot population size over generations."""
        if 'population_size' not in self.df.columns:
            ax.text(0.5, 0.5, 'No population data', ha='center', va='center')
            return
        
        generations = self.df['generation']
        ax.plot(generations, self.df['population_size'], 'b-', linewidth=2, label='Population')
        ax.fill_between(generations, 0, self.df['population_size'], alpha=0.3)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Population Size')
        ax.set_title('Population Dynamics')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _plot_births_deaths(self, ax) -> None:
        """Plot births and deaths over time."""
        if 'births' not in self.df.columns or 'deaths' not in self.df.columns:
            ax.text(0.5, 0.5, 'No birth/death data', ha='center', va='center')
            return
        
        generations = self.df['generation']
        ax.plot(generations, self.df['births'], 'g-', linewidth=2, label='Births', marker='o')
        ax.plot(generations, self.df['deaths'], 'r-', linewidth=2, label='Deaths', marker='x')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Count')
        ax.set_title('Births and Deaths per Generation')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _plot_mating_success(self, ax) -> None:
        """Plot mating success rate and pairs formed."""
        if 'mating_pairs' not in self.df.columns:
            ax.text(0.5, 0.5, 'No mating data', ha='center', va='center')
            return
        
        generations = self.df['generation']
        ax2 = ax.twinx()
        
        # Pairs on left axis
        ax.bar(generations, self.df['mating_pairs'], alpha=0.6, color='purple', label='Mating Pairs')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Mating Pairs', color='purple')
        ax.tick_params(axis='y', labelcolor='purple')
        
        # Success rate on right axis
        if 'mating_success_rate' in self.df.columns:
            ax2.plot(generations, self.df['mating_success_rate'] * 100, 
                    'orange', linewidth=2, marker='o', label='Success Rate')
            ax2.set_ylabel('Success Rate (%)', color='orange')
            ax2.tick_params(axis='y', labelcolor='orange')
            ax2.set_ylim([0, 105])
        
        ax.set_title('Mating Success')
        ax.grid(True, alpha=0.3)
    
    def _plot_fitness_evolution(self, ax) -> None:
        """Plot fitness statistics over time."""
        if 'fitness_best' not in self.df.columns:
            ax.text(0.5, 0.5, 'No fitness data', ha='center', va='center')
            return
        
        generations = self.df['generation']
        
        # Plot best, average, and worst fitness
        ax.plot(generations, self.df['fitness_best'], 'g-', linewidth=2.5, 
               label='Best', marker='o', markersize=4)
        ax.plot(generations, self.df['fitness_avg'], 'b-', linewidth=2, 
               label='Average', alpha=0.8)
        ax.plot(generations, self.df['fitness_worst'], 'r-', linewidth=1.5, 
               label='Worst', alpha=0.6)
        
        # Confidence band (mean ± std)
        if 'fitness_std' in self.df.columns:
            ax.fill_between(generations, 
                           self.df['fitness_avg'] - self.df['fitness_std'],
                           self.df['fitness_avg'] + self.df['fitness_std'],
                           alpha=0.2, color='blue')
        
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness')
        ax.set_title('Fitness Evolution')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _plot_fitness_distribution(self, ax) -> None:
        """Plot fitness distribution evolution."""
        if 'fitness_avg' not in self.df.columns or 'fitness_std' not in self.df.columns:
            ax.text(0.5, 0.5, 'No fitness statistics', ha='center', va='center')
            return
        
        # Plot coefficient of variation over time
        cv = (self.df['fitness_std'] / self.df['fitness_avg']) * 100
        generations = self.df['generation']
        
        ax.plot(generations, cv, 'purple', linewidth=2, marker='o')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Coefficient of Variation (%)')
        ax.set_title('Fitness Diversity (CV)')
        ax.grid(True, alpha=0.3)
        
        # Add text annotation for trend
        if len(cv) > 1:
            trend = "increasing" if cv.iloc[-1] > cv.iloc[0] else "decreasing"
            ax.text(0.95, 0.95, f'Trend: {trend}', 
                   transform=ax.transAxes, ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def _plot_genotype_diversity(self, ax) -> None:
        """Plot genotype diversity over time."""
        if 'genotype_diversity' not in self.df.columns:
            ax.text(0.5, 0.5, 'No diversity data', ha='center', va='center')
            return
        
        generations = self.df['generation']
        diversity = self.df['genotype_diversity']
        
        ax.plot(generations, diversity, 'teal', linewidth=2.5, marker='s', markersize=4)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Genotype Std Dev')
        ax.set_title('Genotype Diversity')
        ax.grid(True, alpha=0.3)
        
        # Highlight max diversity
        max_idx = diversity.idxmax()
        ax.axhline(diversity[max_idx], color='red', linestyle='--', alpha=0.5)
        ax.text(generations.iloc[-1], diversity[max_idx], 
               f'Max: {diversity[max_idx]:.3f}', 
               ha='right', va='bottom')
    
    def _plot_energy_evolution(self, ax) -> None:
        """Plot energy statistics over time."""
        if 'energy_avg' not in self.df.columns:
            ax.text(0.5, 0.5, 'Energy system not enabled', ha='center', va='center')
            return
        
        generations = self.df['generation']
        
        ax.plot(generations, self.df['energy_max'], 'g--', linewidth=1.5, 
               label='Max', alpha=0.7)
        ax.plot(generations, self.df['energy_avg'], 'b-', linewidth=2.5, 
               label='Average', marker='o', markersize=4)
        ax.plot(generations, self.df['energy_min'], 'r--', linewidth=1.5, 
               label='Min', alpha=0.7)
        
        # Shade energy range
        ax.fill_between(generations, self.df['energy_min'], self.df['energy_max'],
                       alpha=0.2, color='blue')
        
        ax.set_xlabel('Generation')
        ax.set_ylabel('Energy')
        ax.set_title('Energy Evolution')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(bottom=0)
    
    def _plot_age_distribution(self, ax) -> None:
        """Plot age statistics over time."""
        if 'age_avg' not in self.df.columns:
            ax.text(0.5, 0.5, 'No age data', ha='center', va='center')
            return
        
        generations = self.df['generation']
        
        ax.plot(generations, self.df['age_max'], 'purple', linewidth=2, 
               label='Oldest', marker='^', markersize=5)
        ax.plot(generations, self.df['age_avg'], 'blue', linewidth=2.5, 
               label='Average', marker='o', markersize=4)
        ax.plot(generations, self.df['age_min'], 'green', linewidth=2, 
               label='Youngest', marker='v', markersize=5)
        
        ax.set_xlabel('Generation')
        ax.set_ylabel('Age (generations)')
        ax.set_title('Age Distribution')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _plot_energy_depletion(self, ax) -> None:
        """Plot number of energy-depleted individuals."""
        if 'energy_depleted_count' not in self.df.columns:
            ax.text(0.5, 0.5, 'No energy depletion data', ha='center', va='center')
            return
        
        generations = self.df['generation']
        depleted = self.df['energy_depleted_count']
        
        ax.bar(generations, depleted, color='red', alpha=0.7, edgecolor='darkred')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Depleted Individuals')
        ax.set_title('Energy-Depleted Population')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add total count
        total_depleted = depleted.sum()
        ax.text(0.95, 0.95, f'Total depleted: {total_depleted:.0f}', 
               transform=ax.transAxes, ha='right', va='top',
               bbox=dict(boxstyle='round', facecolor='salmon', alpha=0.5))
    
    def _plot_fitness_vs_age(self, ax) -> None:
        """Plot correlation between fitness and age (final generation)."""
        if self.controllers is None:
            ax.text(0.5, 0.5, 'No controller data', ha='center', va='center')
            return
        
        controllers = self.controllers['controllers']
        ages = [c['age'] for c in controllers]
        fitness = [c['fitness'] for c in controllers]
        
        scatter = ax.scatter(ages, fitness, c=fitness, cmap='viridis', 
                           s=100, alpha=0.7, edgecolors='black', linewidth=1)
        ax.set_xlabel('Age (generations)')
        ax.set_ylabel('Fitness')
        ax.set_title('Fitness vs Age (Final Population)')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Fitness', rotation=270, labelpad=15)
        
        # Calculate correlation if multiple data points
        if len(ages) > 2:
            corr = np.corrcoef(ages, fitness)[0, 1]
            ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                   transform=ax.transAxes, ha='left', va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def _plot_fitness_vs_energy(self, ax) -> None:
        """Plot correlation between fitness and energy (final generation)."""
        if self.controllers is None:
            ax.text(0.5, 0.5, 'No controller data', ha='center', va='center')
            return
        
        controllers = self.controllers['controllers']
        
        # Check if energy data exists
        if 'energy' not in controllers[0]:
            ax.text(0.5, 0.5, 'Energy not tracked in controllers', ha='center', va='center')
            return
        
        energy = [c['energy'] for c in controllers]
        fitness = [c['fitness'] for c in controllers]
        
        scatter = ax.scatter(energy, fitness, c=fitness, cmap='plasma', 
                           s=100, alpha=0.7, edgecolors='black', linewidth=1)
        ax.set_xlabel('Energy')
        ax.set_ylabel('Fitness')
        ax.set_title('Fitness vs Energy (Final Population)')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Fitness', rotation=270, labelpad=15)
        
        # Calculate correlation if multiple unique values
        if len(set(energy)) > 1:
            corr = np.corrcoef(energy, fitness)[0, 1]
            ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                   transform=ax.transAxes, ha='left', va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def _plot_summary_statistics(self, ax) -> None:
        """Display summary statistics as text."""
        ax.axis('off')
        
        # Gather statistics
        stats = []
        
        if self.df is not None:
            total_gens = len(self.df)
            stats.append(f"Total Generations: {total_gens}")
            
            if 'population_size' in self.df.columns:
                pop_start = self.df['population_size'].iloc[0]
                pop_end = self.df['population_size'].iloc[-1]
                stats.append(f"Population: {pop_start} → {pop_end}")
            
            if 'fitness_best' in self.df.columns:
                best_ever = self.df['fitness_best'].max()
                best_gen = self.df['fitness_best'].idxmax()
                stats.append(f"Best Fitness: {best_ever:.6f} (gen {best_gen})")
                
                fitness_improvement = (self.df['fitness_best'].iloc[-1] - 
                                      self.df['fitness_best'].iloc[0])
                stats.append(f"Fitness Δ: {fitness_improvement:+.6f}")
            
            if 'births' in self.df.columns and 'deaths' in self.df.columns:
                total_births = self.df['births'].sum()
                total_deaths = self.df['deaths'].sum()
                stats.append(f"Total Births: {total_births:.0f}")
                stats.append(f"Total Deaths: {total_deaths:.0f}")
            
            if 'mating_success_rate' in self.df.columns:
                avg_mating = self.df['mating_success_rate'].mean() * 100
                stats.append(f"Avg Mating Success: {avg_mating:.1f}%")
            
            if 'energy_depleted_count' in self.df.columns:
                total_depleted = self.df['energy_depleted_count'].sum()
                stats.append(f"Energy Depleted: {total_depleted:.0f}")
            
            if 'age_max' in self.df.columns:
                oldest_ever = self.df['age_max'].max()
                stats.append(f"Oldest Individual: {oldest_ever:.0f} gens")
        
        if self.controllers is not None:
            final_pop = len(self.controllers['controllers'])
            stats.append(f"\nFinal Population: {final_pop}")
            
            selection = self.controllers.get('config', {}).get('selection_method', 'N/A')
            stats.append(f"Selection: {selection}")
        
        # Display as formatted text
        stats_text = '\n'.join(stats)
        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax.set_title('Summary Statistics', fontweight='bold')
    
    def plot_controller_parameters(self, save_path: Optional[Union[str, Path]] = None) -> None:
        """
        Visualize controller parameters (genotype distributions).
        
        Args:
            save_path: Optional path to save figure.
        """
        if self.controllers is None:
            print("No controller data loaded.")
            return
        
        controllers = self.controllers['controllers']
        num_joints = self.controllers['num_joints']
        
        # Extract all genotypes
        genotypes = np.array([c['genotype'] for c in controllers])
        
        # Create figure with subplots for each joint
        n_rows = (num_joints + 2) // 3  # 3 columns
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4*n_rows))
        axes = axes.flatten() if num_joints > 1 else [axes]
        
        for joint in range(num_joints):
            ax = axes[joint]
            amp_idx = joint * 3
            freq_idx = joint * 3 + 1
            phase_idx = joint * 3 + 2
            
            amplitudes = genotypes[:, amp_idx]
            frequencies = genotypes[:, freq_idx]
            phases = genotypes[:, phase_idx]
            
            # Create grouped bar chart
            x = np.arange(3)
            width = 0.8
            
            means = [amplitudes.mean(), frequencies.mean(), phases.mean()]
            stds = [amplitudes.std(), frequencies.std(), phases.std()]
            
            bars = ax.bar(x, means, width, yerr=stds, capsize=5, 
                         color=['red', 'green', 'blue'], alpha=0.7,
                         edgecolor='black', linewidth=1.5)
            
            ax.set_ylabel('Value')
            ax.set_title(f'Joint {joint} Parameters')
            ax.set_xticks(x)
            ax.set_xticklabels(['Amplitude', 'Frequency', 'Phase'])
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, mean, std in zip(bars, means, stds):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + std,
                       f'{mean:.2f}±{std:.2f}',
                       ha='center', va='bottom', fontsize=8)
        
        # Remove empty subplots
        for idx in range(num_joints, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.suptitle(f'Controller Parameters Distribution (n={len(controllers)})', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Controller parameters plot saved: {save_path}")
        else:
            plt.show()
    
    def plot_parameter_evolution_heatmap(self, save_path: Optional[Union[str, Path]] = None) -> None:
        """
        Create heatmap of parameter values across all joints and individuals.
        
        Args:
            save_path: Optional path to save figure.
        """
        if self.controllers is None:
            print("No controller data loaded.")
            return
        
        controllers = self.controllers['controllers']
        genotypes = np.array([c['genotype'] for c in controllers])
        fitness = np.array([c['fitness'] for c in controllers])
        
        # Sort by fitness for visualization
        sort_idx = np.argsort(fitness)[::-1]
        genotypes_sorted = genotypes[sort_idx]
        fitness_sorted = fitness[sort_idx]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Heatmap of genotype values
        im1 = ax1.imshow(genotypes_sorted, aspect='auto', cmap='viridis', interpolation='nearest')
        ax1.set_xlabel('Parameter Index')
        ax1.set_ylabel('Individual (sorted by fitness)')
        ax1.set_title('Genotype Parameters Heatmap')
        plt.colorbar(im1, ax=ax1, label='Parameter Value')
        
        # Add fitness values as secondary y-axis labels
        ax1_right = ax1.twinx()
        ax1_right.set_ylim(ax1.get_ylim())
        ax1_right.set_yticks(range(len(fitness_sorted)))
        ax1_right.set_yticklabels([f'{f:.4f}' for f in fitness_sorted], fontsize=8)
        ax1_right.set_ylabel('Fitness', rotation=270, labelpad=20)
        
        # Heatmap of parameter correlations
        corr_matrix = np.corrcoef(genotypes.T)
        im2 = ax2.imshow(corr_matrix, aspect='auto', cmap='coolwarm', 
                        vmin=-1, vmax=1, interpolation='nearest')
        ax2.set_xlabel('Parameter Index')
        ax2.set_ylabel('Parameter Index')
        ax2.set_title('Parameter Correlation Matrix')
        plt.colorbar(im2, ax=ax2, label='Correlation')
        
        plt.suptitle('Parameter Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Parameter heatmap saved: {save_path}")
        else:
            plt.show()
    
    def generate_individual_plots(self, output_dir: Path) -> None:
        """
        Generate separate plot files for each metric.
        
        Args:
            output_dir: Directory to save individual plots.
        """
        if self.df is None:
            print("No CSV data loaded. Cannot create individual plots.")
            return
        
        individual_dir = output_dir / 'individual'
        individual_dir.mkdir(exist_ok=True)
        
        # List of plots to generate
        plots = [
            ('population_dynamics', self._plot_population_dynamics, 'Population Dynamics'),
            ('births_deaths', self._plot_births_deaths, 'Births and Deaths'),
            ('mating_success', self._plot_mating_success, 'Mating Success'),
            ('fitness_evolution', self._plot_fitness_evolution, 'Fitness Evolution'),
            ('fitness_distribution', self._plot_fitness_distribution, 'Fitness Distribution'),
            ('genotype_diversity', self._plot_genotype_diversity, 'Genotype Diversity'),
            ('energy_evolution', self._plot_energy_evolution, 'Energy Evolution'),
            ('age_distribution', self._plot_age_distribution, 'Age Distribution'),
            ('energy_depletion', self._plot_energy_depletion, 'Energy Depletion'),
            ('fitness_vs_age', self._plot_fitness_vs_age, 'Fitness vs Age'),
            ('fitness_vs_energy', self._plot_fitness_vs_energy, 'Fitness vs Energy'),
        ]
        
        for filename, plot_func, title in plots:
            fig, ax = plt.subplots(figsize=(8, 6))
            plot_func(ax)
            plt.suptitle(f'{title} - {self.base_name}', fontsize=12, fontweight='bold')
            plt.tight_layout()
            
            save_path = individual_dir / f'{self.base_name}_{filename}.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"  Created {len(plots)} individual plots in: {individual_dir.name}/")
    
    def generate_minimal_plots(self, output_dir: Path) -> None:
        """
        Generate minimal publication-ready plots suitable for double-column format.
        
        These plots are designed to be clear and readable in a two-column academic paper
        with minimal visual clutter and appropriate font sizes.
        
        Args:
            output_dir: Directory to save minimal plots.
        """
        if self.df is None:
            print("No CSV data loaded. Cannot create minimal plots.")
            return
        
        minimal_dir = output_dir / 'minimal'
        minimal_dir.mkdir(exist_ok=True)
        
        # Set publication style
        plt.rcParams.update({
            'font.size': 9,
            'axes.labelsize': 10,
            'axes.titlesize': 11,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 8,
            'figure.titlesize': 11,
            'lines.linewidth': 1.5,
            'lines.markersize': 4,
            'axes.linewidth': 0.8,
            'grid.linewidth': 0.5,
        })
        
        # Figure size for double-column: ~3.5 inches wide (half of 7" column width)
        fig_width = 3.5
        fig_height = 2.5
        
        # 1. Fitness Evolution (most important)
        self._create_minimal_fitness_plot(minimal_dir, fig_width, fig_height)
        
        # 2. Population Dynamics
        self._create_minimal_population_plot(minimal_dir, fig_width, fig_height)
        
        # 3. Age and Energy Combined
        self._create_minimal_age_energy_plot(minimal_dir, fig_width, fig_height)
        
        # 4. Mating Success
        if 'mating_pairs' in self.df.columns:
            self._create_minimal_mating_plot(minimal_dir, fig_width, fig_height)
        
        # 5. Diversity Metrics
        self._create_minimal_diversity_plot(minimal_dir, fig_width, fig_height)
        
        # 6. Final Population Analysis (if controllers available)
        if self.controllers:
            self._create_minimal_final_population_plot(minimal_dir, fig_width, fig_height)
        
        # Reset to default style
        plt.rcParams.update(plt.rcParamsDefault)
        
        print(f"  Created minimal publication plots in: {minimal_dir.name}/")
    
    def _create_minimal_fitness_plot(self, output_dir: Path, width: float, height: float) -> None:
        """Create minimal fitness evolution plot."""
        fig, ax = plt.subplots(figsize=(width, height))
        
        if 'fitness_best' in self.df.columns:
            generations = self.df['generation']
            ax.plot(generations, self.df['fitness_best'], 'g-', label='Best', linewidth=2)
            ax.plot(generations, self.df['fitness_avg'], 'b-', label='Mean', linewidth=1.5, alpha=0.8)
            
            if 'fitness_std' in self.df.columns:
                ax.fill_between(generations,
                               self.df['fitness_avg'] - self.df['fitness_std'],
                               self.df['fitness_avg'] + self.df['fitness_std'],
                               alpha=0.2, color='blue')
            
            ax.set_xlabel('Generation')
            ax.set_ylabel('Fitness')
            ax.legend(loc='best', frameon=False)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{self.base_name}_minimal_fitness.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_minimal_population_plot(self, output_dir: Path, width: float, height: float) -> None:
        """Create minimal population dynamics plot."""
        fig, ax = plt.subplots(figsize=(width, height))
        
        if 'population_size' in self.df.columns:
            generations = self.df['generation']
            ax.plot(generations, self.df['population_size'], 'b-', linewidth=2)
            ax.fill_between(generations, 0, self.df['population_size'], alpha=0.3)
            
            ax.set_xlabel('Generation')
            ax.set_ylabel('Population Size')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{self.base_name}_minimal_population.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_minimal_age_energy_plot(self, output_dir: Path, width: float, height: float) -> None:
        """Create minimal combined age and energy plot."""
        has_age = 'age_avg' in self.df.columns
        has_energy = 'energy_avg' in self.df.columns
        
        if not has_age and not has_energy:
            return
        
        generations = self.df['generation']
        
        if has_age and has_energy:
            # Two subplots vertically stacked
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(width, height * 1.5), 
                                           sharex=True)
            
            # Age plot
            ax1.plot(generations, self.df['age_avg'], 'purple', linewidth=2, label='Mean')
            ax1.fill_between(generations, self.df['age_min'], self.df['age_max'], 
                            alpha=0.3, color='purple')
            ax1.set_ylabel('Age (gen)')
            ax1.grid(True, alpha=0.3, linestyle='--')
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.legend(loc='best', frameon=False)
            
            # Energy plot
            ax2.plot(generations, self.df['energy_avg'], 'orange', linewidth=2, label='Mean')
            ax2.fill_between(generations, self.df['energy_min'], self.df['energy_max'],
                            alpha=0.3, color='orange')
            ax2.set_xlabel('Generation')
            ax2.set_ylabel('Energy')
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.legend(loc='best', frameon=False)
            
            plt.tight_layout()
            plt.savefig(output_dir / f'{self.base_name}_minimal_age_energy.png',
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        elif has_age:
            # Age only
            fig, ax = plt.subplots(figsize=(width, height))
            ax.plot(generations, self.df['age_avg'], 'purple', linewidth=2)
            ax.fill_between(generations, self.df['age_min'], self.df['age_max'],
                           alpha=0.3, color='purple')
            ax.set_xlabel('Generation')
            ax.set_ylabel('Age (generations)')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(output_dir / f'{self.base_name}_minimal_age.png',
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        elif has_energy:
            # Energy only
            fig, ax = plt.subplots(figsize=(width, height))
            ax.plot(generations, self.df['energy_avg'], 'orange', linewidth=2)
            ax.fill_between(generations, self.df['energy_min'], self.df['energy_max'],
                           alpha=0.3, color='orange')
            ax.set_xlabel('Generation')
            ax.set_ylabel('Energy')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(output_dir / f'{self.base_name}_minimal_energy.png',
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_minimal_mating_plot(self, output_dir: Path, width: float, height: float) -> None:
        """Create minimal mating success plot."""
        fig, ax = plt.subplots(figsize=(width, height))
        
        generations = self.df['generation']
        ax.bar(generations, self.df['mating_pairs'], alpha=0.7, color='purple', 
              edgecolor='darkviolet', linewidth=0.5)
        
        ax.set_xlabel('Generation')
        ax.set_ylabel('Mating Pairs')
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{self.base_name}_minimal_mating.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_minimal_diversity_plot(self, output_dir: Path, width: float, height: float) -> None:
        """Create minimal genotype diversity plot."""
        if 'genotype_diversity' not in self.df.columns:
            return
        
        fig, ax = plt.subplots(figsize=(width, height))
        
        generations = self.df['generation']
        ax.plot(generations, self.df['genotype_diversity'], 'teal', 
               linewidth=2, marker='o', markersize=3)
        
        ax.set_xlabel('Generation')
        ax.set_ylabel('Genotype Diversity (SD)')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{self.base_name}_minimal_diversity.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_minimal_final_population_plot(self, output_dir: Path, 
                                             width: float, height: float) -> None:
        """Create minimal final population scatter plot."""
        controllers = self.controllers['controllers']
        ages = [c['age'] for c in controllers]
        fitness = [c['fitness'] for c in controllers]
        
        fig, ax = plt.subplots(figsize=(width, height))
        
        scatter = ax.scatter(ages, fitness, c=fitness, cmap='viridis',
                           s=30, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel('Age (generations)')
        ax.set_ylabel('Fitness')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add minimal colorbar
        cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label('Fitness', rotation=270, labelpad=12, fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{self.base_name}_minimal_final_pop.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, output_dir: Optional[Union[str, Path]] = None, 
                       individual_plots: bool = False,
                       minimal_plots: bool = False) -> None:
        """
        Generate a complete analysis report with all visualizations.
        
        Args:
            output_dir: Directory to save all plots. Defaults to results directory.
            individual_plots: If True, create separate files for each metric plot.
            minimal_plots: If True, create publication-ready minimal plots (double-column format).
        """
        if output_dir is None:
            output_dir = self.results_dir / 'analysis'
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"\nGenerating comprehensive analysis report in: {output_dir}")
        print("=" * 70)
        
        # Main comprehensive plot
        main_plot = output_dir / f'{self.base_name}_comprehensive.png'
        print(f"Creating main visualization...")
        self.plot_all(save_path=main_plot)
        
        # Individual plots (if requested)
        if individual_plots:
            print(f"Creating individual metric plots...")
            self.generate_individual_plots(output_dir)
        
        # Minimal publication plots (if requested)
        if minimal_plots:
            print(f"Creating minimal publication plots...")
            self.generate_minimal_plots(output_dir)
        
        # Controller parameters (if available)
        if self.controllers is not None:
            params_plot = output_dir / f'{self.base_name}_parameters.png'
            print(f"Creating parameter distribution plot...")
            self.plot_controller_parameters(save_path=params_plot)
            
            heatmap_plot = output_dir / f'{self.base_name}_heatmap.png'
            print(f"Creating parameter heatmap...")
            self.plot_parameter_evolution_heatmap(save_path=heatmap_plot)
        
        # Save summary statistics to text file
        summary_file = output_dir / f'{self.base_name}_summary.txt'
        print(f"Writing summary statistics...")
        self._write_summary_file(summary_file)
        
        print("=" * 70)
        print(f"Report generation complete! Files saved in: {output_dir}")
        print(f"  - Main plot: {main_plot.name}")
        if individual_plots:
            print(f"  - Individual plots: 12 separate files")
        if minimal_plots:
            print(f"  - Minimal plots: 6 publication-ready files")
        if self.controllers is not None:
            print(f"  - Parameters: {params_plot.name}")
            print(f"  - Heatmap: {heatmap_plot.name}")
        print(f"  - Summary: {summary_file.name}")
    
    def _write_summary_file(self, filepath: Path) -> None:
        """Write summary statistics to text file."""
        with open(filepath, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write(f"EXPERIMENT ANALYSIS SUMMARY: {self.base_name}\n")
            f.write("=" * 70 + "\n\n")
            
            if self.df is not None:
                f.write("EVOLUTION STATISTICS\n")
                f.write("-" * 70 + "\n")
                f.write(f"Total Generations: {len(self.df)}\n")
                
                if 'population_size' in self.df.columns:
                    f.write(f"Population (start): {self.df['population_size'].iloc[0]}\n")
                    f.write(f"Population (end): {self.df['population_size'].iloc[-1]}\n")
                
                if 'fitness_best' in self.df.columns:
                    f.write(f"\nFitness:\n")
                    f.write(f"  Best ever: {self.df['fitness_best'].max():.6f} (gen {self.df['fitness_best'].idxmax()})\n")
                    f.write(f"  Final best: {self.df['fitness_best'].iloc[-1]:.6f}\n")
                    f.write(f"  Final average: {self.df['fitness_avg'].iloc[-1]:.6f}\n")
                    f.write(f"  Final worst: {self.df['fitness_worst'].iloc[-1]:.6f}\n")
                    improvement = self.df['fitness_best'].iloc[-1] - self.df['fitness_best'].iloc[0]
                    f.write(f"  Improvement: {improvement:+.6f}\n")
                
                if 'births' in self.df.columns and 'deaths' in self.df.columns:
                    f.write(f"\nPopulation Dynamics:\n")
                    f.write(f"  Total births: {self.df['births'].sum():.0f}\n")
                    f.write(f"  Total deaths: {self.df['deaths'].sum():.0f}\n")
                
                if 'mating_pairs' in self.df.columns:
                    f.write(f"\nMating Statistics:\n")
                    f.write(f"  Total mating pairs: {self.df['mating_pairs'].sum():.0f}\n")
                    f.write(f"  Average success rate: {self.df['mating_success_rate'].mean() * 100:.1f}%\n")
                
                if 'age_max' in self.df.columns:
                    f.write(f"\nAge Statistics:\n")
                    f.write(f"  Oldest individual: {self.df['age_max'].max():.0f} generations\n")
                    f.write(f"  Final average age: {self.df['age_avg'].iloc[-1]:.2f} generations\n")
                
                if 'energy_depleted_count' in self.df.columns:
                    f.write(f"\nEnergy Statistics:\n")
                    f.write(f"  Total energy depletions: {self.df['energy_depleted_count'].sum():.0f}\n")
                    f.write(f"  Final average energy: {self.df['energy_avg'].iloc[-1]:.2f}\n")
            
            if self.controllers is not None:
                f.write("\n" + "=" * 70 + "\n")
                f.write("FINAL POPULATION ANALYSIS\n")
                f.write("-" * 70 + "\n")
                
                controllers = self.controllers['controllers']
                fitness_values = [c['fitness'] for c in controllers]
                
                f.write(f"Population size: {len(controllers)}\n")
                f.write(f"Best fitness: {max(fitness_values):.6f}\n")
                f.write(f"Average fitness: {np.mean(fitness_values):.6f}\n")
                f.write(f"Worst fitness: {min(fitness_values):.6f}\n")
                f.write(f"Fitness std: {np.std(fitness_values):.6f}\n")
                
                if 'energy' in controllers[0]:
                    energies = [c['energy'] for c in controllers]
                    f.write(f"\nEnergy distribution:\n")
                    f.write(f"  Min: {min(energies):.2f}\n")
                    f.write(f"  Max: {max(energies):.2f}\n")
                    f.write(f"  Average: {np.mean(energies):.2f}\n")
                
                ages = [c['age'] for c in controllers]
                f.write(f"\nAge distribution:\n")
                f.write(f"  Youngest: {min(ages)} generations\n")
                f.write(f"  Oldest: {max(ages)} generations\n")
                f.write(f"  Average: {np.mean(ages):.2f} generations\n")
                
                config = self.controllers.get('config', {})
                f.write(f"\nConfiguration:\n")
                for key, value in config.items():
                    f.write(f"  {key}: {value}\n")
            
            f.write("\n" + "=" * 70 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 70 + "\n")


def main():
    """Example usage: visualize the most recent experiment."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Visualize spatial EA experiment data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize most recent experiment (comprehensive plot only)
  python visualize_experiment.py
  
  # Generate all plot types
  python visualize_experiment.py --individual --minimal
  
  # Visualize specific experiment
  python visualize_experiment.py --file __results__/evolution_data_20251109_230140.csv --individual
        """
    )
    
    parser.add_argument('--file', '-f', type=str, 
                       help='Path to specific CSV, NPZ, or JSON file (default: most recent)')
    parser.add_argument('--individual', '-i', action='store_true',
                       help='Generate individual plots for each metric')
    parser.add_argument('--minimal', '-m', action='store_true',
                       help='Generate minimal publication-ready plots')
    parser.add_argument('--all', '-a', action='store_true',
                       help='Generate all plot types (comprehensive + individual + minimal)')
    parser.add_argument('--output', '-o', type=str,
                       help='Output directory (default: __results__/analysis/)')
    
    args = parser.parse_args()
    
    # Determine which file to visualize
    if args.file:
        data_file = Path(args.file)
        if not data_file.exists():
            print(f"Error: File not found: {data_file}")
            return 1
    else:
        # Find the most recent evolution data file
        results_dir = Path(__file__).parent.parent.parent / '__results__'
        csv_files = list(results_dir.glob('evolution_data_*.csv'))
        
        if not csv_files:
            print("No experiment data found in __results__/")
            print("Run an evolution experiment first using main_spatial_ea.py")
            return 1
        
        data_file = max(csv_files, key=lambda p: p.stat().st_mtime)
    
    print(f"Visualizing: {data_file.name}")
    print()
    
    # Create visualizer
    viz = ExperimentVisualizer(data_file)
    
    # Determine plot types
    if args.all:
        individual = True
        minimal = True
    else:
        individual = args.individual
        minimal = args.minimal
    
    # Output directory
    output_dir = Path(args.output) if args.output else None
    
    # Generate report
    viz.generate_report(output_dir=output_dir, 
                       individual_plots=individual,
                       minimal_plots=minimal)
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main() or 0)
