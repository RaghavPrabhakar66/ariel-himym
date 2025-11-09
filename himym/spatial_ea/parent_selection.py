import numpy as np
from typing import Any

try:
    from spatial_individual import SpatialIndividual
    from periodic_boundary_utils import periodic_distance
except ImportError:
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
    if method == "proximity_pairing":
        return _proximity_pairing(
            population, tracked_geoms, pairing_radius, 
            world_size, use_periodic_boundaries
        )
    elif method == "random":
        return _random_pairing(population)
    elif method == "mating_zone":
        # Support both single and multiple zones
        mating_zone_centers = kwargs.get('mating_zone_centers', None)
        if mating_zone_centers is None:
            # Fall back to single zone for backward compatibility
            mating_zone_center = kwargs.get('mating_zone_center', (0.0, 0.0))
            mating_zone_centers = [mating_zone_center]
        
        mating_zone_radius = kwargs.get('mating_zone_radius', 3.0)
        return _mating_zone_pairing(
            population, tracked_geoms, pairing_radius,
            world_size, use_periodic_boundaries,
            mating_zone_centers, mating_zone_radius
        )
    else:
        print(f"  Warning: Unknown pairing method '{method}', using proximity_pairing")
        return _proximity_pairing(
            population, tracked_geoms, pairing_radius,
            world_size, use_periodic_boundaries
        )


def _proximity_pairing(
    population: list[SpatialIndividual],
    tracked_geoms: list,
    pairing_radius: float,
    world_size: tuple[float, float],
    use_periodic_boundaries: bool
) -> tuple[list[tuple[int, int]], set[int]]:
    pairs = []
    paired_indices = set()
    
    # Iterate through population in order 
    for idx in range(len(population)):
        current_pos = tracked_geoms[idx].xpos.copy()
        
        # Find nearest neighbor within pairing radius
        nearest_partner_idx = None
        nearest_distance = float('inf')
        
        for other_idx in range(len(population)):
            if other_idx == idx:
                continue
            
            other_pos = tracked_geoms[other_idx].xpos.copy()
            
            # Calculate distance using periodic boundaries if enabled
            if use_periodic_boundaries:
                distance = periodic_distance(current_pos, other_pos, world_size)
            else:
                distance = np.linalg.norm(current_pos - other_pos)
            
            # Check if within pairing radius and is closer than current nearest
            if distance <= pairing_radius and distance < nearest_distance:
                nearest_distance = distance
                nearest_partner_idx = other_idx
        
        # If found a partner within radius, create pair
        if nearest_partner_idx is not None:
            pairs.append((idx, nearest_partner_idx))
            paired_indices.add(idx)
            paired_indices.add(nearest_partner_idx)
    
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

def _mating_zone_pairing(
    population: list[SpatialIndividual],
    tracked_geoms: list,
    pairing_radius: float,
    world_size: tuple[float, float],
    use_periodic_boundaries: bool,
    mating_zone_centers: list[tuple[float, float]],
    mating_zone_radius: float
) -> tuple[list[tuple[int, int]], set[int]]:
    """
    Pair individuals based on nearest neighbor, but only if they are within a mating zone.
    
    Supports multiple mating zones. Robots can mate if they are in ANY of the zones.
    
    Args:
        population: Current population of individuals
        tracked_geoms: List of MuJoCo geom objects for tracking positions
        pairing_radius: Maximum distance for pairing within the zone
        world_size: World dimensions (width, height)
        use_periodic_boundaries: Whether to use periodic boundary conditions
        mating_zone_centers: List of (x, y) coordinates of mating zone centers
        mating_zone_radius: Radius of the mating zones (same for all zones)
        
    Returns:
        Tuple of (pairs, paired_indices)
    """
    pairs = []
    paired_indices = set()
    
    # First, identify which robots are in ANY mating zone
    robots_in_zone = []
    for idx in range(len(population)):
        robot_pos = tracked_geoms[idx].xpos.copy()
        
        # Check if robot is in any of the mating zones
        in_any_zone = False
        for zone_center in mating_zone_centers:
            zone_center_3d = np.array([zone_center[0], zone_center[1], 0.0])
            
            # Calculate distance to zone center
            if use_periodic_boundaries:
                distance_to_center = periodic_distance(robot_pos, zone_center_3d, world_size)
            else:
                distance_to_center = np.linalg.norm(robot_pos - zone_center_3d)
            
            # Check if robot is within this mating zone
            if distance_to_center <= mating_zone_radius:
                in_any_zone = True
                break
        
        if in_any_zone:
            robots_in_zone.append(idx)
    
    # Now pair robots that are in the zone using nearest neighbor
    for idx in robots_in_zone:
        current_pos = tracked_geoms[idx].xpos.copy()
        
        # Find nearest neighbor within pairing radius (among robots in zone)
        nearest_partner_idx = None
        nearest_distance = float('inf')
        
        for other_idx in robots_in_zone:
            if other_idx == idx:
                continue
            
            other_pos = tracked_geoms[other_idx].xpos.copy()
            
            # Calculate distance using periodic boundaries if enabled
            if use_periodic_boundaries:
                distance = periodic_distance(current_pos, other_pos, world_size)
            else:
                distance = np.linalg.norm(current_pos - other_pos)
            
            # Check if within pairing radius and is closer than current nearest
            if distance <= pairing_radius and distance < nearest_distance:
                nearest_distance = distance
                nearest_partner_idx = other_idx
        
        # If found a partner within radius, create pair
        if nearest_partner_idx is not None:
            pairs.append((idx, nearest_partner_idx))
            paired_indices.add(idx)
            paired_indices.add(nearest_partner_idx)
    
    return pairs, paired_indices

def generate_random_zone_centers(
    num_zones: int,
    world_size: tuple[float, float],
    zone_radius: float,
    min_zone_distance: float,
    margin: float = 1.0
) -> list[tuple[float, float]]:
    """
    Generate random positions for multiple mating zones.
    
    Ensures zones don't overlap too much and stay within world boundaries.
    
    Args:
        num_zones: Number of zones to generate
        world_size: World dimensions (width, height)
        zone_radius: Radius of each zone
        min_zone_distance: Minimum distance between zone centers (in zone radii units)
        margin: Margin from world edges (in meters)
        
    Returns:
        List of (x, y) coordinates for zone centers
    """
    if num_zones <= 0:
        return []
    
    zone_centers = []
    min_distance = zone_radius * min_zone_distance
    
    # Available area for zone centers
    x_min = margin + zone_radius
    x_max = world_size[0] - margin - zone_radius
    y_min = margin + zone_radius
    y_max = world_size[1] - margin - zone_radius
    
    max_attempts = 1000 * num_zones  # Prevent infinite loops
    attempts = 0
    
    while len(zone_centers) < num_zones and attempts < max_attempts:
        attempts += 1
        
        # Generate random position
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        new_center = (x, y)
        
        # Check distance to existing zones
        too_close = False
        for existing_center in zone_centers:
            distance = np.sqrt((x - existing_center[0])**2 + (y - existing_center[1])**2)
            if distance < min_distance:
                too_close = True
                break
        
        if not too_close:
            zone_centers.append(new_center)
    
    if len(zone_centers) < num_zones:
        print(f"  Warning: Could only place {len(zone_centers)} of {num_zones} zones")
        print(f"  Consider reducing num_mating_zones or min_zone_distance")
    
    return zone_centers

def get_mating_zone_info(config: Any) -> dict[str, Any]:
    """
    Get information about the mating zone for visualization or robot behavior.
    
    Args:
        config: EAConfig object containing mating zone parameters
        
    Returns:
        Dictionary with mating zone information:
            - 'center': (x, y) tuple of zone center
            - 'radius': float radius of zone
            - 'enabled': bool indicating if mating_zone method is active
    """
    return {
        'center': tuple(config.mating_zone_center),
        'radius': config.mating_zone_radius,
        'enabled': config.pairing_method == 'mating_zone'
    }

def is_in_mating_zone(
    position: np.ndarray,
    mating_zone_centers: list[tuple[float, float]],
    mating_zone_radius: float,
    world_size: tuple[float, float],
    use_periodic_boundaries: bool = False
) -> bool:
    """
    Check if a robot position is within any mating zone.
    
    Args:
        position: 3D position array [x, y, z] of the robot
        mating_zone_centers: List of (x, y) coordinates of zone centers
        mating_zone_radius: Radius of the mating zones
        world_size: World dimensions (width, height)
        use_periodic_boundaries: Whether to use periodic boundary conditions
        
    Returns:
        True if position is within any mating zone, False otherwise
    """
    for zone_center in mating_zone_centers:
        zone_center_3d = np.array([zone_center[0], zone_center[1], 0.0])
        
        if use_periodic_boundaries:
            distance = periodic_distance(position, zone_center_3d, world_size)
        else:
            distance = np.linalg.norm(position - zone_center_3d)
        
        if distance <= mating_zone_radius:
            return True
    
    return False

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
