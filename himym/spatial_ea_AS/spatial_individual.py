"""
Spatial Individual class for evolutionary algorithm.

This module defines the SpatialIndividual class which represents
a single robot individual in the spatial evolutionary algorithm.
"""

import numpy as np


class SpatialIndividual:
    """
    Represents a single individual (robot) in the spatial evolutionary algorithm.
    
    Attributes:
        unique_id: Unique identifier for this individual
        generation: Generation number when this individual was created
        genotype: List of gene values (amplitude, frequency, phase for each joint)
        fitness: Fitness score (typically distance traveled)
        start_position: Starting position in evaluation
        end_position: Ending position in evaluation
        spawn_position: Position where robot was spawned in simulation
        robot_index: Index of this robot in the population
        parent_ids: List of unique IDs of parent individuals
        energy: Current energy level (for energy-based selection)
    """
    
    def __init__(
        self, 
        unique_id: int | None = None, 
        generation: int = 0,
        initial_energy: float = 100.0
    ):
        """
        Initialize a new individual.
        
        Args:
            unique_id: Unique identifier for this individual
            generation: Generation number when created
            initial_energy: Starting energy level
        """
        self.unique_id = unique_id
        self.generation = generation
        self.genotype: list[float] = []
        self.fitness: float = 0.0
        self.evaluated: bool = False  # Track whether fitness has been evaluated
        self.start_position: np.ndarray | None = None
        self.end_position: np.ndarray | None = None
        self.spawn_position: np.ndarray | None = None
        self.robot_index: int | None = None
        self.parent_ids: list[int] = []
        self.energy: float = initial_energy
    
    def __repr__(self) -> str:
        """String representation of the individual."""
        eval_status = "true" if self.evaluated else "false"
        return (f"SpatialIndividual(id={self.unique_id}, gen={self.generation}, "
                f"fitness={self.fitness:.4f}, energy={self.energy:.1f}, eval={eval_status})")
    
    def copy(self) -> 'SpatialIndividual':
        """
        Create a copy of this individual with a new unique ID.
        
        Note: This does not copy the unique_id - caller must set a new one.
        
        Returns:
            A new SpatialIndividual with copied attributes
        """
        new_individual = SpatialIndividual(generation=self.generation, initial_energy=self.energy)
        new_individual.genotype = self.genotype.copy()
        new_individual.fitness = self.fitness
        new_individual.evaluated = False  # New individual needs evaluation if genotype changes
        new_individual.parent_ids = [self.unique_id]
        
        if self.start_position is not None:
            new_individual.start_position = self.start_position.copy()
        if self.end_position is not None:
            new_individual.end_position = self.end_position.copy()
        if self.spawn_position is not None:
            new_individual.spawn_position = self.spawn_position.copy()
            
        return new_individual
