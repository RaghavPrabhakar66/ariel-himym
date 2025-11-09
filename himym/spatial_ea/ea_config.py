"""
Configuration loader for Evolutionary Algorithm parameters.

This module loads configuration from ea_config.yaml and provides
easy access to all EA parameters.
"""

import yaml
from pathlib import Path
from typing import Any


class EAConfig:
    """Configuration class for Evolutionary Algorithm parameters."""
    
    def __init__(self, config_path: str | None = None):
        """
        Initialize configuration from YAML file.
        
        Args:
            config_path: Path to the configuration YAML file.
                        If None, looks for ea_config.yaml in the same directory.
        """
        if config_path is None:
            # Look for config file in the same directory as this module
            config_path = str(Path(__file__).parent / "ea_config.yaml")
        
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)
    
    # Population Parameters
    @property
    def population_size(self) -> int:
        return self._config['population']['size']
    
    @property
    def num_generations(self) -> int:
        return self._config['population']['num_generations']
    
    # Selection Parameters
    @property
    def pairing_radius(self) -> float:
        return self._config['selection'].get('pairing_radius', 2.0)
    
    @property
    def offspring_radius(self) -> float:
        return self._config['selection'].get('offspring_radius', 0.3)
    
    @property
    def max_population_size(self) -> int:
        return self._config['selection'].get('max_population_size', 20)
    
    @property
    def selection_method(self) -> str:
        return self._config['selection'].get('selection_method', 'parents_die')
    
    @property
    def pairing_method(self) -> str:
        return self._config['selection'].get('pairing_method', 'proximity_fitness')
    
    @property
    def mating_zone_center(self) -> list[float]:
        """Center coordinates (x, y) of the mating zone."""
        return self._config['selection'].get('mating_zone_center', [0.0, 0.0])
    
    @property
    def mating_zone_radius(self) -> float:
        """Radius of the mating zone circle."""
        return self._config['selection'].get('mating_zone_radius', 3.0)
    
    @property
    def num_mating_zones(self) -> int:
        """Number of randomly placed mating zones."""
        return self._config['selection'].get('num_mating_zones', 1)
    
    @property
    def dynamic_mating_zones(self) -> bool:
        """Whether mating zones change position over time."""
        return self._config['selection'].get('dynamic_mating_zones', False)
    
    @property
    def zone_change_interval(self) -> int:
        """Generations between zone position changes (for dynamic zones)."""
        return self._config['selection'].get('zone_change_interval', 5)
    
    @property
    def min_zone_distance(self) -> float:
        """Minimum distance between zone centers (in zone radii units)."""
        return self._config['selection'].get('min_zone_distance', 2.0)
    
    @property
    def movement_bias(self) -> str:
        """Movement bias during mating phase: 'nearest_neighbor', 'nearest_zone', or 'none'."""
        return self._config['selection'].get('movement_bias', 'nearest_neighbor')
    
    @property
    def max_age(self) -> int:
        """Maximum age for probabilistic_age selection (death probability = age/max_age)."""
        return self._config['selection'].get('max_age', 10)
    
    # Energy-Based Selection Parameters
    @property
    def enable_energy(self) -> bool:
        """Enable energy-based survival mechanics."""
        return self._config['selection'].get('enable_energy', False)
    
    @property
    def initial_energy(self) -> float:
        """Starting energy for each individual."""
        return self._config['selection'].get('initial_energy', 100.0)
    
    @property
    def energy_depletion_rate(self) -> float:
        """Energy lost per generation (time-based passive depletion)."""
        return self._config['selection'].get('energy_depletion_rate', 1.0)
    
    @property
    def mating_energy_effect(self) -> str:
        """Effect of mating on energy: 'restore' (full reset), 'cost' (energy penalty), 'none' (no effect)."""
        return self._config['selection'].get('mating_energy_effect', 'restore')
    
    @property
    def mating_energy_amount(self) -> float:
        """Energy restored (if 'restore') or cost deducted (if 'cost') when mating occurs."""
        return self._config['selection'].get('mating_energy_amount', 50.0)
    
    # Crossover Parameters
    @property
    def crossover_rate(self) -> float:
        return self._config['crossover']['rate']
    
    # Mutation Parameters
    @property
    def mutation_rate(self) -> float:
        return self._config['mutation']['rate']
    
    @property
    def mutation_strength(self) -> float:
        return self._config['mutation']['strength']
    
    # Genotype Parameters - Amplitude
    @property
    def amplitude_min(self) -> float:
        return self._config['genotype']['amplitude']['min']
    
    @property
    def amplitude_max(self) -> float:
        return self._config['genotype']['amplitude']['max']
    
    @property
    def amplitude_init_min(self) -> float:
        return self._config['genotype']['amplitude']['init_min']
    
    @property
    def amplitude_init_max(self) -> float:
        return self._config['genotype']['amplitude']['init_max']
    
    # Genotype Parameters - Frequency
    @property
    def frequency_min(self) -> float:
        return self._config['genotype']['frequency']['min']
    
    @property
    def frequency_max(self) -> float:
        return self._config['genotype']['frequency']['max']
    
    @property
    def frequency_init_min(self) -> float:
        return self._config['genotype']['frequency']['init_min']
    
    @property
    def frequency_init_max(self) -> float:
        return self._config['genotype']['frequency']['init_max']
    
    # Genotype Parameters - Phase
    @property
    def phase_min(self) -> float:
        return self._config['genotype']['phase']['min']
    
    @property
    def phase_max(self) -> float:
        return self._config['genotype']['phase']['max']
    
    # Simulation Parameters
    @property
    def simulation_time(self) -> float:
        return self._config['simulation']['time']
    
    @property
    def final_demo_time(self) -> float:
        return self._config['simulation']['final_demo_time']
    
    @property
    def multi_robot_demo_time(self) -> float:
        return self._config['simulation']['multi_robot_demo_time']
    
    @property
    def control_clip_min(self) -> float:
        return self._config['simulation']['control_clip_min']
    
    @property
    def control_clip_max(self) -> float:
        return self._config['simulation']['control_clip_max']
    
    @property
    def use_periodic_boundaries(self) -> bool:
        return self._config['simulation'].get('use_periodic_boundaries', False)
    
    # Multi-Robot Parameters
    @property
    def world_size(self) -> list[float]:
        return self._config['multi_robot']['world_size']
    
    @property
    def spawn_x_min(self) -> float:
        return self._config['multi_robot']['spawn_area']['x_min']
    
    @property
    def spawn_x_max(self) -> float:
        return self._config['multi_robot']['spawn_area']['x_max']
    
    @property
    def spawn_y_min(self) -> float:
        return self._config['multi_robot']['spawn_area']['y_min']
    
    @property
    def spawn_y_max(self) -> float:
        return self._config['multi_robot']['spawn_area']['y_max']
    
    @property
    def spawn_z(self) -> float:
        return self._config['multi_robot']['spawn_area']['z']
    
    @property
    def min_spawn_distance(self) -> float:
        return self._config['multi_robot'].get('min_spawn_distance', 0.6)
    
    @property
    def robot_size(self) -> float:
        """Approximate robot size/diameter for visualization purposes (meters)."""
        return self._config['multi_robot'].get('robot_size', 0.4)
    
    # Output Paths
    @property
    def video_folder(self) -> str:
        return self._config['output']['video_folder']
    
    @property
    def figures_folder(self) -> str:
        return self._config['output']['figures_folder']
    
    @property
    def results_folder(self) -> str:
        return self._config['output']['results_folder']
    
    # Logging
    @property
    def print_generation_stats(self) -> bool:
        return self._config['logging']['print_generation_stats']
    
    @property
    def print_final_genotype(self) -> bool:
        return self._config['logging']['print_final_genotype']
    
    # Video Recording
    @property
    def record_generation_videos(self) -> bool:
        return self._config['video'].get('record_generation_videos', False)
    
    def get_raw_config(self) -> dict[str, Any]:
        """Return the raw configuration dictionary."""
        return self._config
    
    def __repr__(self) -> str:
        return f"EAConfig(population_size={self.population_size}, num_generations={self.num_generations})"


# Create a default configuration instance
config = EAConfig()
