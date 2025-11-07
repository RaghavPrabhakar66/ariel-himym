"""
Parent selection (pairing/mating) strategies for spatial evolutionary algorithm.

This module provides different strategies for selecting parent pairs
for reproduction based on spatial proximity, fitness, or other criteria.
"""

import numpy as np
from typing import Any

try:
    from spatial_individual import SpatialIndividual
    from periodic_boundary_utils import periodic_distance
except ImportError:
    # Support both direct execution and package imports
    from himym.spatial_ea.spatial_individual import SpatialIndividual
    from himym.spatial_ea.periodic_boundary_utils import periodic_distance


def find_pairs(
    population: list[SpatialIndividual],
    tracked_geoms: list,
    method: str,
    pairing_radius: float,
    world_size: tuple[float, float],
    use_periodic_boundaries: bool = False,
    **kwargs
) -> tuple[list[tuple[int, int]], set[int]]:
    if method == "proximity_fitness":
        return _proximity_fitness_pairing(
            population, tracked_geoms, pairing_radius, 
            world_size, use_periodic_boundaries
        )
    elif method == "random":
        return _random_pairing(population)
    elif method == "fitness_proportional":
        return _fitness_proportional_pairing(population)
    elif method == "tournament":
        tournament_size = kwargs.get('tournament_size', 3)
        return _tournament_pairing(population, tournament_size)
    else:
        print(f"  Warning: Unknown pairing method '{method}', using proximity_fitness")
        return _proximity_fitness_pairing(
            population, tracked_geoms, pairing_radius,
            world_size, use_periodic_boundaries
        )


def _proximity_fitness_pairing(
    population: list[SpatialIndividual],
    tracked_geoms: list,
    pairing_radius: float,
    world_size: tuple[float, float],
    use_periodic_boundaries: bool
) -> tuple[list[tuple[int, int]], set[int]]:
    pairs = []
    paired_indices = set()
    
    # Sort population by fitness (descending) to prioritize high-fitness individuals
    fitness_ranking = sorted(
        enumerate(population), 
        key=lambda x: x[1].fitness, 
        reverse=True
    )
    
    for idx, _ in fitness_ranking:
        if idx in paired_indices:
            continue  # Already paired
        
        current_pos = tracked_geoms[idx].xpos.copy()
        
        # Find highest fitness partner within pairing radius
        best_partner_idx = None
        best_partner_fitness = -1
        
        for other_idx, other_ind in enumerate(population):
            if other_idx == idx or other_idx in paired_indices:
                continue
            
            other_pos = tracked_geoms[other_idx].xpos.copy()
            
            # Calculate distance using periodic boundaries if enabled
            if use_periodic_boundaries:
                distance = periodic_distance(current_pos, other_pos, world_size)
            else:
                distance = np.linalg.norm(current_pos - other_pos)
            
            # Check if within pairing radius and has higher fitness than current best
            if distance <= pairing_radius and other_ind.fitness > best_partner_fitness:
                best_partner_fitness = other_ind.fitness
                best_partner_idx = other_idx
        
        # If found a partner within radius, create pair
        if best_partner_idx is not None:
            pairs.append((idx, best_partner_idx))
            paired_indices.add(idx)
            paired_indices.add(best_partner_idx)
    
    return pairs, paired_indices


def _random_pairing(
    population: list[SpatialIndividual]
) -> tuple[list[tuple[int, int]], set[int]]:
    """
    Randomly pair individuals from the population.
    
    This method ignores spatial information and fitness, creating random pairs.
    Useful as a baseline or for testing non-spatial evolution.
    
    Args:
        population: Current population of individuals
        
    Returns:
        Tuple of (pairs, paired_indices)
    """
    pairs = []
    paired_indices = set()
    
    # Create shuffled list of indices
    indices = list(range(len(population)))
    np.random.shuffle(indices)
    
    # Pair consecutive individuals in shuffled list
    for i in range(0, len(indices) - 1, 2):
        idx1 = indices[i]
        idx2 = indices[i + 1]
        pairs.append((idx1, idx2))
        paired_indices.add(idx1)
        paired_indices.add(idx2)
    
    return pairs, paired_indices


def _fitness_proportional_pairing(
    population: list[SpatialIndividual]
) -> tuple[list[tuple[int, int]], set[int]]:
    """
    Pair individuals using fitness-proportional selection (roulette wheel).
    
    Each individual is selected with probability proportional to their fitness.
    This allows high-fitness individuals to mate multiple times while still
    giving low-fitness individuals a chance.
    
    Args:
        population: Current population of individuals
        
    Returns:
        Tuple of (pairs, paired_indices)
    """
    pairs = []
    paired_indices = set()
    
    # Get fitness values and ensure they're positive
    fitness_values = np.array([ind.fitness for ind in population])
    
    # Handle negative fitness by shifting
    min_fitness = np.min(fitness_values)
    if min_fitness < 0:
        fitness_values = fitness_values - min_fitness + 1e-6
    
    # Add small epsilon to avoid zero probabilities
    fitness_values = fitness_values + 1e-6
    
    # Calculate selection probabilities
    total_fitness = np.sum(fitness_values)
    probabilities = fitness_values / total_fitness
    
    # Number of pairs to create (up to half the population)
    num_pairs = len(population) // 2
    
    # Select pairs with replacement (an individual can be selected multiple times)
    for _ in range(num_pairs):
        # Select two parents
        parent_indices = np.random.choice(
            len(population), 
            size=2, 
            replace=False,  # Don't select the same individual twice for a single pair
            p=probabilities
        )
        
        idx1, idx2 = parent_indices
        pairs.append((int(idx1), int(idx2)))
        paired_indices.add(int(idx1))
        paired_indices.add(int(idx2))
    
    return pairs, paired_indices


def _tournament_pairing(
    population: list[SpatialIndividual],
    tournament_size: int = 3
) -> tuple[list[tuple[int, int]], set[int]]:
    """
    Pair individuals using tournament selection.
    
    For each parent needed, run a tournament where K random individuals compete
    and the fittest wins. This provides a good balance between selection pressure
    and diversity.
    
    Args:
        population: Current population of individuals
        tournament_size: Number of individuals in each tournament
        
    Returns:
        Tuple of (pairs, paired_indices)
    """
    pairs = []
    paired_indices = set()
    
    def run_tournament() -> int:
        """Run a single tournament and return winner's index."""
        tournament_indices = np.random.choice(
            len(population), 
            size=min(tournament_size, len(population)), 
            replace=False
        )
        
        # Find fittest in tournament
        best_idx = tournament_indices[0]
        best_fitness = population[best_idx].fitness
        
        for idx in tournament_indices[1:]:
            if population[idx].fitness > best_fitness:
                best_fitness = population[idx].fitness
                best_idx = idx
        
        return int(best_idx)
    
    # Create pairs
    num_pairs = len(population) // 2
    
    for _ in range(num_pairs):
        parent1_idx = run_tournament()
        parent2_idx = run_tournament()
        
        # Ensure parents are different (run new tournament if same)
        while parent1_idx == parent2_idx:
            parent2_idx = run_tournament()
        
        pairs.append((parent1_idx, parent2_idx))
        paired_indices.add(parent1_idx)
        paired_indices.add(parent2_idx)
    
    return pairs, paired_indices


def calculate_offspring_positions(
    pairs: list[tuple[int, int]],
    current_positions: list[np.ndarray],
    offspring_radius: float,
    world_size: tuple[float, float],
    use_periodic_boundaries: bool = False
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Calculate spawn positions for offspring from parent pairs.
    
    Each offspring is placed at a random angle around its parent,
    at a fixed radius (offspring_radius).
    
    Args:
        pairs: List of (parent1_idx, parent2_idx) tuples
        current_positions: List of current positions for all individuals
        offspring_radius: Radius around parent where offspring spawn
        world_size: World dimensions (width, height) for boundary wrapping
        use_periodic_boundaries: Whether to wrap positions using periodic boundaries
        
    Returns:
        List of (child1_pos, child2_pos) tuples corresponding to pairs
    """
    pair_positions = []
    
    for parent1_idx, parent2_idx in pairs:
        parent1_pos = current_positions[parent1_idx]
        parent2_pos = current_positions[parent2_idx]
        
        # Random positions on circle edge around each parent
        angle1 = np.random.uniform(0, 2 * np.pi)
        child1_offset = np.array([
            offspring_radius * np.cos(angle1),
            offspring_radius * np.sin(angle1),
            0.0
        ])
        
        angle2 = np.random.uniform(0, 2 * np.pi)
        child2_offset = np.array([
            offspring_radius * np.cos(angle2),
            offspring_radius * np.sin(angle2),
            0.0
        ])
        
        # Apply offspring positions with periodic wrapping if enabled
        if use_periodic_boundaries:
            try:
                from periodic_boundary_utils import wrap_offspring_position
            except ImportError:
                from himym.spatial_ea.periodic_boundary_utils import wrap_offspring_position
            
            child1_pos = wrap_offspring_position(
                parent1_pos, child1_offset, world_size
            )
            child2_pos = wrap_offspring_position(
                parent2_pos, child2_offset, world_size
            )
        else:
            # Non-periodic: just add offset (may go outside bounds)
            child1_pos = parent1_pos + child1_offset
            child2_pos = parent2_pos + child2_offset
        
        pair_positions.append((child1_pos, child2_pos))
    
    return pair_positions
