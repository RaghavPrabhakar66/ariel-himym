"""Selection strategies for evolutionary algorithm."""

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
    """Apply selection to reduce population to target size."""
    initial_size = len(population)
    
    if method == "probabilistic_age":
        print(f"  Applying {method} selection: {initial_size} → natural dynamics (no target)")
        indices_to_keep = _selection_probabilistic_age(
            population, target_size, current_generation, max_age
        )
    elif method == "energy_based":
        print(f"  Applying {method} selection: {initial_size} → natural dynamics (no target)")
        indices_to_keep = _selection_energy_based(population, target_size)
    else:
        if initial_size < target_size:
            print(f"  Selection: Population size ({initial_size}) < target ({target_size}), no selection needed")
            orientations = current_orientations if current_orientations else []
            return population, current_positions, initial_size, orientations
        
        print(f"  Applying {method} selection: {initial_size} → {target_size}")
        
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
    
    indices_to_keep.sort()
    
    if len(population) > 0 and hasattr(population[0], 'robot_index'):
        old_robot_indices = [population[i].robot_index for i in indices_to_keep]
        print(f"    Keeping population indices: {indices_to_keep[:10]}{'...' if len(indices_to_keep) > 10 else ''}")
        print(f"    Their robot_index values: {old_robot_indices[:10]}{'...' if len(old_robot_indices) > 10 else ''}")
    
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
        fitnesses = [ind.fitness for ind in new_population]
        if fitnesses:
            print(f"    Fitness range: {min(fitnesses):.4f} to {max(fitnesses):.4f} (range: {max(fitnesses) - min(fitnesses):.4f})")
        else:
            print(f"    No survivors (population extinct)")
    elif method == "probabilistic_age":
        ages = [current_generation - ind.generation for ind in new_population]
        if ages:
            avg_age = sum(ages) / len(ages)
            print(f"    Survivor age range: {min(ages)} to {max(ages)} generations (avg: {avg_age:.1f})")
            gen_counts = {}
            for ind in new_population:
                gen_counts[ind.generation] = gen_counts.get(ind.generation, 0) + 1
            print(f"    Generation distribution: {dict(sorted(gen_counts.items()))}")
        else:
            print(f"    No survivors (population extinct)")
    elif method == "energy_based":
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
    """Selection where parents that mated this generation die and offspring survive."""
    offspring_indices = []
    parents_indices = []
    survivors_indices = []
    
    for i, ind in enumerate(population):
        if ind.generation > current_generation:
            offspring_indices.append(i)
        elif paired_indices is not None and i in paired_indices and ind.generation == current_generation:
            parents_indices.append(i)
        else:
            survivors_indices.append(i)
    
    print(f"    Population breakdown: {len(offspring_indices)} offspring, "
          f"{len(parents_indices)} parents that mated, {len(survivors_indices)} other survivors")
    
    indices_to_keep = offspring_indices + survivors_indices
    
    if len(indices_to_keep) < target_size:
        needed = target_size - len(indices_to_keep)
        print(f"    Not enough offspring+survivors ({len(indices_to_keep)}), keeping {needed} parents")
        
        parents_fitness = [(i, population[i].fitness) for i in parents_indices]
        parents_fitness.sort(key=lambda x: x[1], reverse=True)
        indices_to_keep.extend([i for i, _ in parents_fitness[:needed]])
    
    elif len(indices_to_keep) > target_size:
        print(f"    Too many individuals ({len(indices_to_keep)}), applying fitness selection")
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
    """Selection based purely on fitness (keep best individuals)."""
    fitness_ranking = [(i, ind.fitness) for i, ind in enumerate(population)]
    fitness_ranking.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in fitness_ranking[:target_size]]


def _selection_age_based(
    population: list[SpatialIndividual],
    target_size: int
) -> list[int]:
    """Selection based on age (keep youngest individuals)."""
    age_ranking = [(i, ind.generation) for i, ind in enumerate(population)]
    age_ranking.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in age_ranking[:target_size]]


def _selection_probabilistic_age(
    population: list[SpatialIndividual],
    target_size: int,
    current_generation: int,
    max_age: int = 10
) -> list[int]:
    """Selection with age-dependent death probability (no population size enforcement)."""
    survivors = []
    deaths = []
    
    for i, ind in enumerate(population):
        age = current_generation - ind.generation
        p_death = min(1.0, age / max_age) if max_age > 0 else 0.0
        
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
    """Selection based on energy levels (individuals with energy <= 0 die)."""
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
