"""
Selection strategies for evolutionary algorithm.

This module provides different selection methods for managing
population size and determining which individuals survive.
"""

import numpy as np
import random
from spatial_individual import SpatialIndividual


def apply_selection(
    population: list[SpatialIndividual],
    current_positions: list[np.ndarray],
    method: str,
    target_size: int,
    current_generation: int,
    current_orientations: list[float] | None = None,
    paired_indices: set[int] | None = None,
    max_age: int = 10
) -> tuple[list[SpatialIndividual], list[np.ndarray], int, list[float]]:
    """
    Apply selection to reduce population to target size.
    
    Args:
        population: Current population
        current_positions: Current positions corresponding to population
        method: Selection method ('parents_die', 'fitness_based', 'age_based')
        target_size: Target population size
        current_generation: Current generation number
        current_orientations: Current orientations (yaw angles) corresponding to population
        paired_indices: Set of indices that mated this generation (for parents_die method)
        max_age: Maximum age for probabilistic_age selection (death probability = age/max_age)
        
    Returns:
        Tuple of (selected_population, selected_positions, new_population_size, selected_orientations)
    """
    initial_size = len(population)
    
    # Probabilistic age selection and energy-based selection ignore target size
    if method == "probabilistic_age":
        print(f"  Applying {method} selection: {initial_size} → natural dynamics (no target)")
        indices_to_keep = _selection_probabilistic_age(
            population, target_size, current_generation, max_age
        )
    elif method == "energy_based":
        print(f"  Applying {method} selection: {initial_size} → natural dynamics (no target)")
        indices_to_keep = _selection_energy_based(population, target_size)
    else:
        # Other methods enforce target size
        # If population is below target, no selection needed
        if initial_size < target_size:
            print(f"  Selection: Population size ({initial_size}) < target ({target_size}), no selection needed")
            orientations = current_orientations if current_orientations else []
            return population, current_positions, initial_size, orientations
        
        # If population equals or exceeds target, apply selection
        print(f"  Applying {method} selection: {initial_size} → {target_size}")
        
        # Create list of indices to keep
        if method == "parents_die":
            indices_to_keep = _selection_parents_die(
                population, target_size, current_generation, paired_indices
            )
        elif method == "fitness_based":
            indices_to_keep = _selection_fitness_based(population, target_size)
        elif method == "age_based":
            indices_to_keep = _selection_age_based(population, target_size)
        else:
            print(f"    Warning: Unknown selection method '{method}', using parents_die")
            indices_to_keep = _selection_parents_die(
                population, target_size, current_generation, paired_indices
            )
    
    # Sort indices to maintain order
    indices_to_keep.sort()
    
    # Create new population, positions, and orientations
    new_population = [population[i] for i in indices_to_keep]
    new_positions = [current_positions[i] for i in indices_to_keep]
    new_orientations = [current_orientations[i] for i in indices_to_keep] if current_orientations else []
    new_size = len(new_population)

    print(f"  SELECTION COMPLETE: {initial_size} -> {new_size}")

    if method == "parents_die":
        current_gen_count = sum(1 for ind in new_population if ind.generation == current_generation)
        offspring_count = sum(1 for ind in new_population if ind.generation > current_generation)
        older_gen_count = sum(1 for ind in new_population if ind.generation < current_generation)
        print(f"    Final survivors: {offspring_count} offspring, {current_gen_count} current gen, {older_gen_count} older gen")
    elif method == "fitness_based":
        # Show fitness range to help diagnose if all fitnesses are the same
        fitnesses = [ind.fitness for ind in new_population]
        print(f"    Fitness range: {min(fitnesses):.4f} to {max(fitnesses):.4f} (range: {max(fitnesses) - min(fitnesses):.4f})")
    elif method == "probabilistic_age":
        # Show age distribution of survivors
        ages = [current_generation - ind.generation for ind in new_population]
        avg_age = sum(ages) / len(ages) if ages else 0
        print(f"    Survivor age range: {min(ages)} to {max(ages)} generations (avg: {avg_age:.1f})")
        # Show generation breakdown
        gen_counts = {}
        for ind in new_population:
            gen_counts[ind.generation] = gen_counts.get(ind.generation, 0) + 1
        print(f"    Generation distribution: {dict(sorted(gen_counts.items()))}")
    elif method == "energy_based":
        # Show energy distribution of survivors
        if new_population:
            energies = [ind.energy for ind in new_population]
            avg_energy = sum(energies) / len(energies)
            print(f"    Survivor energy: min={min(energies):.1f}, max={max(energies):.1f}, avg={avg_energy:.1f}")
        else:
            print(f"    No survivors (population extinct)")
    
    return new_population, new_positions, new_size, new_orientations


def _selection_parents_die(
    population: list[SpatialIndividual],
    target_size: int,
    current_generation: int,
    paired_indices: set[int] | None = None
) -> list[int]:
    """
    Selection where parents that mated this generation die and offspring survive.
    
    Only individuals from the CURRENT generation that ACTUALLY MATED are eliminated.
    Individuals that didn't mate (even if from current generation) are preserved.
    Individuals from PREVIOUS generations are preserved.
    
    If not enough offspring, keeps best non-parent individuals to fill target size.
    
    Args:
        population: Current population
        target_size: Target population size
        current_generation: Current generation number
        paired_indices: Set of indices that mated (if None, treats all current gen as parents)
        
    Returns:
        List of indices to keep
    """
    offspring_indices = []  # New offspring (generation > current)
    parents_indices = []  # Parents that mated this generation (should die)
    survivors_indices = []  # All other individuals (keep them)
    
    for i, ind in enumerate(population):
        if ind.generation > current_generation:
            # These are the newly created offspring
            offspring_indices.append(i)
        elif paired_indices is not None and i in paired_indices and ind.generation == current_generation:
            # These are individuals from current generation that mated (should die)
            parents_indices.append(i)
        else:
            # These are either:
            # - individuals from previous generations, OR
            # - individuals from current generation that didn't mate
            # Keep them all
            survivors_indices.append(i)
    
    print(f"    Population breakdown: {len(offspring_indices)} offspring, "
          f"{len(parents_indices)} parents that mated, {len(survivors_indices)} other survivors")
    
    # Start with all offspring and survivors
    indices_to_keep = offspring_indices + survivors_indices
    
    # If we don't have enough individuals, we need to keep some parents
    if len(indices_to_keep) < target_size:
        needed = target_size - len(indices_to_keep)
        print(f"    Not enough offspring+survivors ({len(indices_to_keep)}), keeping {needed} parents")
        
        # Select best parents by fitness
        parents_fitness = [(i, population[i].fitness) for i in parents_indices]
        parents_fitness.sort(key=lambda x: x[1], reverse=True)
        indices_to_keep.extend([i for i, _ in parents_fitness[:needed]])
    
    # If we have too many individuals, apply fitness-based selection
    elif len(indices_to_keep) > target_size:
        print(f"    Too many individuals ({len(indices_to_keep)}), applying fitness selection")
        # Prioritize offspring, but if needed, remove based on fitness
        fitness_ranking = [(i, population[i].fitness) for i in indices_to_keep]
        fitness_ranking.sort(key=lambda x: x[1], reverse=True)
        indices_to_keep = [i for i, _ in fitness_ranking[:target_size]]
    else:
        print(f"    Removing all {len(parents_indices)} parents that mated")
    
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


def _selection_probabilistic_age(
    population: list[SpatialIndividual],
    target_size: int,
    current_generation: int,
    max_age: int = 10
) -> list[int]:
    """
    Selection with age-dependent death probability (no population size enforcement).
    
    Each individual has a probability of death that increases with age.
    Population size will fluctuate naturally based on probabilistic outcomes.

    Args:
        population: Current population
        target_size: Target population size (ignored in this method)
        current_generation: Current generation number
        max_age: Maximum age (death probability = age/max_age, capped at 1.0)
        
    Returns:
        List of indices to keep (whoever survives the probabilistic culling)
    """
    survivors = []
    deaths = []
    
    for i, ind in enumerate(population):
        age = current_generation - ind.generation
        # Death probability increases linearly with age, capped at 1.0
        p_death = min(1.0, age / max_age) if max_age > 0 else 0.0
        
        # Roll the dice!
        if random.random() > p_death:
            survivors.append(i)
        else:
            deaths.append(i)
    
    print(f"    Probabilistic age-based death (max_age={max_age}): {len(deaths)} died, {len(survivors)} survived naturally")
    print(f"    Population change: {len(population)} → {len(survivors)} (no size enforcement)")
    
    return survivors


def _selection_energy_based(
    population: list[SpatialIndividual],
    target_size: int
) -> list[int]:
    """
    Selection based on energy levels (individuals with energy <= 0 die).
    
    This selection method removes any individual whose energy has been depleted.
    Population size will fluctuate naturally based on energy dynamics.
    
    Args:
        population: Current population
        target_size: Target population size (ignored in this method)
        
    Returns:
        List of indices to keep (individuals with energy > 0)
    """
    survivors = []
    deaths = []
    
    for i, ind in enumerate(population):
        if ind.energy > 0:
            survivors.append(i)
        else:
            deaths.append(i)
    
    print(f"    Energy-based death: {len(deaths)} died (energy depleted), {len(survivors)} survived")
    print(f"    Population change: {len(population)} → {len(survivors)} (no size enforcement)")
    
    if survivors:
        energy_values = [population[i].energy for i in survivors]
        print(f"    Survivor energy: min={min(energy_values):.1f}, max={max(energy_values):.1f}, avg={np.mean(energy_values):.1f}")
    
    return survivors
