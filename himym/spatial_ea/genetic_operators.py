"""
Genetic operators for evolutionary algorithm.

This module provides crossover and mutation functions for evolving
robot control parameters.
"""

import numpy as np
from spatial_individual import SpatialIndividual


def create_initial_genotype(
    num_joints: int,
    amplitude_range: tuple[float, float],
    frequency_range: tuple[float, float],
    phase_range: tuple[float, float]
) -> list[float]:
    """
    Create a random initial genotype for a robot.
    
    Args:
        num_joints: Number of robot joints to control
        amplitude_range: (min, max) for amplitude values
        frequency_range: (min, max) for frequency values
        phase_range: (min, max) for phase values
        
    Returns:
        List of genotype values [amp1, freq1, phase1, amp2, freq2, phase2, ...]
    """
    genotype = []
    for _ in range(num_joints):
        amplitude = np.random.uniform(amplitude_range[0], amplitude_range[1])
        frequency = np.random.uniform(frequency_range[0], frequency_range[1])
        phase = np.random.uniform(phase_range[0], phase_range[1])
        genotype.extend([amplitude, frequency, phase])
    
    return genotype


def crossover_one_point(
    parent1: SpatialIndividual,
    parent2: SpatialIndividual,
    next_unique_id: int,
    generation: int
) -> tuple[SpatialIndividual, SpatialIndividual, int]:
    """
    Perform one-point crossover between two parents.
    
    Args:
        parent1: First parent individual
        parent2: Second parent individual
        next_unique_id: Next available unique ID for offspring
        generation: Generation number for offspring
        
    Returns:
        Tuple of (child1, child2, updated_next_unique_id)
    """
    child1 = SpatialIndividual(unique_id=next_unique_id, generation=generation)
    next_unique_id += 1
    child1.parent_ids = [parent1.unique_id, parent2.unique_id]
    
    child2 = SpatialIndividual(unique_id=next_unique_id, generation=generation)
    next_unique_id += 1
    child2.parent_ids = [parent1.unique_id, parent2.unique_id]
    
    crossover_point = np.random.randint(1, len(parent1.genotype))
    
    child1.genotype = (
        parent1.genotype[:crossover_point] + 
        parent2.genotype[crossover_point:]
    )
    child2.genotype = (
        parent2.genotype[:crossover_point] + 
        parent1.genotype[crossover_point:]
    )
    
    return child1, child2, next_unique_id


def mutate_gaussian(
    individual: SpatialIndividual,
    mutation_rate: float,
    mutation_strength: float,
    next_unique_id: int,
    amplitude_range: tuple[float, float],
    frequency_range: tuple[float, float],
    phase_max: float
) -> tuple[SpatialIndividual, int]:
    """
    Apply Gaussian mutation to an individual's genotype.
    
    Args:
        individual: Individual to mutate
        mutation_rate: Probability of mutating each gene
        mutation_strength: Standard deviation of Gaussian noise
        next_unique_id: Next available unique ID
        amplitude_range: (min, max) for clamping amplitude values
        frequency_range: (min, max) for clamping frequency values
        phase_max: Maximum phase value (wraps around)
        
    Returns:
        Tuple of (mutated_individual, updated_next_unique_id)
    """
    mutated = SpatialIndividual(unique_id=next_unique_id, generation=individual.generation)
    next_unique_id += 1
    mutated.parent_ids = [individual.unique_id]
    mutated.genotype = individual.genotype.copy()
    
    for i in range(len(mutated.genotype)):
        if np.random.random() < mutation_rate:
            # Add Gaussian noise
            mutated.genotype[i] += np.random.normal(0, mutation_strength)
            
            # Clamp values to reasonable ranges
            param_type = i % 3
            if param_type == 0:  # amplitude
                mutated.genotype[i] = np.clip(
                    mutated.genotype[i], 
                    amplitude_range[0], 
                    amplitude_range[1]
                )
            elif param_type == 1:  # frequency
                mutated.genotype[i] = np.clip(
                    mutated.genotype[i], 
                    frequency_range[0], 
                    frequency_range[1]
                )
            else:  # phase
                mutated.genotype[i] = mutated.genotype[i] % phase_max
    
    return mutated, next_unique_id


def clone_individual(
    individual: SpatialIndividual,
    next_unique_id: int,
    generation: int
) -> tuple[SpatialIndividual, int]:
    """
    Clone an individual with a new unique ID.
    
    Args:
        individual: Individual to clone
        next_unique_id: Next available unique ID
        generation: Generation number for clone
        
    Returns:
        Tuple of (cloned_individual, updated_next_unique_id)
    """
    clone = SpatialIndividual(unique_id=next_unique_id, generation=generation)
    next_unique_id += 1
    clone.genotype = individual.genotype.copy()
    clone.parent_ids = [individual.unique_id]
    
    return clone, next_unique_id
