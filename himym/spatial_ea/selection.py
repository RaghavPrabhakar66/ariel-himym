"""
Selection strategies for evolutionary algorithm.

This module provides different selection methods for managing
population size and determining which individuals survive.
"""

import numpy as np
from spatial_individual import SpatialIndividual


def apply_selection(
    population: list[SpatialIndividual],
    current_positions: list[np.ndarray],
    method: str,
    target_size: int,
    current_generation: int
) -> tuple[list[SpatialIndividual], list[np.ndarray], int]:
    """
    Apply selection to reduce population to target size.
    
    Args:
        population: Current population
        current_positions: Current positions corresponding to population
        method: Selection method ('parents_die', 'fitness_based', 'age_based')
        target_size: Target population size
        current_generation: Current generation number
        
    Returns:
        Tuple of (selected_population, selected_positions, new_population_size)
    """
    initial_size = len(population)
    
    # If population is already at or below target, no selection needed
    if initial_size <= target_size:
        print(f"  Selection: Population size ({initial_size}) <= target ({target_size}), no selection needed")
        return population, current_positions, initial_size
    
    print(f"  Applying {method} selection: {initial_size} → {target_size}")
    
    # Create list of indices to keep
    indices_to_keep = []
    
    if method == "parents_die":
        indices_to_keep = _selection_parents_die(
            population, target_size, current_generation
        )
    elif method == "fitness_based":
        indices_to_keep = _selection_fitness_based(population, target_size)
    elif method == "age_based":
        indices_to_keep = _selection_age_based(population, target_size)
    else:
        print(f"    Warning: Unknown selection method '{method}', using fitness_based")
        indices_to_keep = _selection_fitness_based(population, target_size)
    
    # Sort indices to maintain order
    indices_to_keep.sort()
    
    # Create new population and positions
    new_population = [population[i] for i in indices_to_keep]
    new_positions = [current_positions[i] for i in indices_to_keep]
    new_size = len(new_population)

    print(f"  SELECTION COMPLETE: {initial_size} -> {new_size}")

    if method == "parents_die":
        parent_count = sum(1 for ind in new_population if ind.generation == current_generation)
        offspring_count = sum(1 for ind in new_population if ind.generation > current_generation)
        print(f"    Survivors: {parent_count} parents, {offspring_count} offspring")
    
    return new_population, new_positions, new_size


def _selection_parents_die(
    population: list[SpatialIndividual],
    target_size: int,
    current_generation: int
) -> list[int]:
    """
    Selection where parents die and only offspring survive.
    
    If not enough offspring, keeps best parents to fill target size.
    
    Args:
        population: Current population
        target_size: Target population size
        current_generation: Current generation number
        
    Returns:
        List of indices to keep
    """
    offspring_indices = []
    parent_indices = []
    
    for i, ind in enumerate(population):
        if ind.generation > current_generation:
            offspring_indices.append(i)
        else:
            parent_indices.append(i)
    
    # If we have enough offspring, keep only offspring
    if len(offspring_indices) >= target_size:
        print(f"    Enough offspring ({len(offspring_indices)}), removing all parents")
        # If too many offspring, apply fitness-based selection among them
        if len(offspring_indices) > target_size:
            offspring_fitness = [(i, population[i].fitness) for i in offspring_indices]
            offspring_fitness.sort(key=lambda x: x[1], reverse=True)
            return [i for i, _ in offspring_fitness[:target_size]]
        else:
            return offspring_indices
    else:
        # Not enough offspring, keep all offspring plus best parents
        print(f"    Only {len(offspring_indices)} offspring, keeping best parents too")
        indices_to_keep = offspring_indices.copy()
        needed_parents = target_size - len(offspring_indices)
        
        # Select best parents by fitness
        parent_fitness = [(i, population[i].fitness) for i in parent_indices]
        parent_fitness.sort(key=lambda x: x[1], reverse=True)
        indices_to_keep.extend([i for i, _ in parent_fitness[:needed_parents]])
        
        return indices_to_keep


def _selection_fitness_based(
    population: list[SpatialIndividual],
    target_size: int
) -> list[int]:
    """
    Selection based purely on fitness (keep best individuals).
    
    Args:
        population: Current population
        target_size: Target population size
        
    Returns:
        List of indices to keep
    """
    fitness_ranking = [(i, ind.fitness) for i, ind in enumerate(population)]
    fitness_ranking.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in fitness_ranking[:target_size]]


def _selection_age_based(
    population: list[SpatialIndividual],
    target_size: int
) -> list[int]:
    """
    Selection based on age (keep youngest individuals).
    
    Args:
        population: Current population
        target_size: Target population size
        
    Returns:
        List of indices to keep
    """
    age_ranking = [(i, ind.generation) for i, ind in enumerate(population)]
    age_ranking.sort(key=lambda x: x[1], reverse=True)  # Higher generation = younger
    return [i for i, _ in age_ranking[:target_size]]
