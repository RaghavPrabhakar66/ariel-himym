"""
Utilities for implementing periodic boundary conditions in spatial EA.

This module provides functions for:
1. Wrapping positions to stay within periodic boundaries
2. Computing distances with periodic topology (toroidal distance)
3. Finding nearest neighbors across periodic boundaries
"""

import numpy as np
from typing import Any


def wrap_position(
    position: np.ndarray,
    world_size: tuple[float, float]
) -> np.ndarray:
    """
    Wrap a position to stay within periodic boundaries.
    
    Args:
        position: Position array [x, y, z]
        world_size: World dimensions [width, height]
        
    Returns:
        Wrapped position array
    """
    wrapped_pos = position.copy()
    wrapped_pos[0] = position[0] % world_size[0]
    wrapped_pos[1] = position[1] % world_size[1]
    # Z coordinate is not wrapped
    return wrapped_pos


def periodic_distance(
    pos1: np.ndarray,
    pos2: np.ndarray,
    world_size: tuple[float, float]
) -> float:
    """
    Calculate the minimum distance between two points with periodic boundaries.
    
    This implements toroidal distance on a 2D surface.
    
    Args:
        pos1: First position [x, y, z] or [x, y]
        pos2: Second position [x, y, z] or [x, y]
        world_size: World dimensions [width, height]
        
    Returns:
        Minimum distance considering periodic wrapping
        
    Example:
        In a world of size [10, 10]:
        - Direct distance from (1, 1) to (9, 9) is ~11.3
        - Periodic distance is ~2.8 (wrapping around both axes)
    """
    # Extract x, y coordinates
    p1 = pos1[:2]
    p2 = pos2[:2]
    
    # Calculate difference for each dimension
    dx = abs(p2[0] - p1[0])
    dy = abs(p2[1] - p1[1])
    
    # Apply periodic boundary: take minimum of direct or wrapped distance
    dx = min(dx, world_size[0] - dx)
    dy = min(dy, world_size[1] - dy)
    
    return np.sqrt(dx**2 + dy**2)


def periodic_displacement(
    pos1: np.ndarray,
    pos2: np.ndarray,
    world_size: tuple[float, float]
) -> np.ndarray:
    """
    Calculate the displacement vector from pos1 to pos2 with periodic boundaries.
    
    Returns the shortest displacement vector considering wrapping.
    
    Args:
        pos1: Starting position [x, y, z] or [x, y]
        pos2: Target position [x, y, z] or [x, y]
        world_size: World dimensions [width, height]
        
    Returns:
        Displacement vector [dx, dy] (shortest path)
        
    Example:
        In a world of size [10, 10]:
        From (1, 1) to (9, 9):
        - Direct displacement: [8, 8]
        - Periodic displacement: [-2, -2] (shorter to wrap around)
    """
    # Extract x, y coordinates
    p1 = pos1[:2]
    p2 = pos2[:2]
    
    # Calculate raw displacement
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    
    # Apply periodic wrapping to get shortest path
    if abs(dx) > world_size[0] / 2:
        dx = dx - np.sign(dx) * world_size[0]
    if abs(dy) > world_size[1] / 2:
        dy = dy - np.sign(dy) * world_size[1]
    
    return np.array([dx, dy])


def find_nearest_periodic(
    current_pos: np.ndarray,
    other_positions: list[np.ndarray],
    world_size: tuple[float, float],
    exclude_indices: set[int] | None = None
) -> tuple[int | None, float]:
    """
    Find the nearest neighbor considering periodic boundaries.
    
    Args:
        current_pos: Current position [x, y, z]
        other_positions: List of other positions
        world_size: World dimensions [width, height]
        exclude_indices: Set of indices to exclude from search
        
    Returns:
        Tuple of (nearest_index, distance) or (None, inf) if no valid neighbors
    """
    if exclude_indices is None:
        exclude_indices = set()
    
    nearest_idx = None
    nearest_dist = float('inf')
    
    for i, other_pos in enumerate(other_positions):
        if i in exclude_indices:
            continue
        
        dist = periodic_distance(current_pos, other_pos, world_size)
        if dist < nearest_dist:
            nearest_dist = dist
            nearest_idx = i
    
    return nearest_idx, nearest_dist


def check_periodic_spawn_overlap(
    new_pos: np.ndarray,
    existing_positions: list[np.ndarray],
    world_size: tuple[float, float],
    min_distance: float
) -> bool:
    """
    Check if a new spawn position overlaps with existing ones (periodic version).
    
    Args:
        new_pos: New position to check [x, y, z]
        existing_positions: List of existing positions
        world_size: World dimensions [width, height]
        min_distance: Minimum allowed distance
        
    Returns:
        True if position is valid (no overlap), False otherwise
    """
    for existing_pos in existing_positions:
        dist = periodic_distance(new_pos, existing_pos, world_size)
        if dist < min_distance:
            return False
    return True


def wrap_offspring_position(
    parent_pos: np.ndarray,
    offset: np.ndarray,
    world_size: tuple[float, float]
) -> np.ndarray:
    """
    Calculate offspring position with periodic wrapping.
    
    Args:
        parent_pos: Parent position [x, y, z]
        offset: Offset from parent [dx, dy, dz]
        world_size: World dimensions [width, height]
        
    Returns:
        Wrapped offspring position
    """
    child_pos = parent_pos + offset
    return wrap_position(child_pos, world_size)


def visualize_periodic_trajectory(
    trajectory: list[np.ndarray],
    world_size: tuple[float, float],
    wrap_threshold: float = 0.5
) -> list[list[np.ndarray]]:
    """
    Split trajectory into segments when wrapping occurs.
    
    This helps visualize trajectories properly - when a robot wraps around
    the boundary, we don't want to draw a line across the entire world.
    
    Args:
        trajectory: List of positions [x, y]
        world_size: World dimensions [width, height]
        wrap_threshold: Fraction of world size to consider as a wrap
        
    Returns:
        List of trajectory segments (each segment is a list of positions)
        
    Example:
        If robot moves from (24, 5) to (1, 5) in a 25x25 world,
        this creates two segments: [(24, 5)] and [(1, 5)]
        so we don't draw a line from x=24 to x=1.
    """
    if len(trajectory) < 2:
        return [trajectory]
    
    segments = []
    current_segment = [trajectory[0]]
    
    for i in range(1, len(trajectory)):
        prev_pos = trajectory[i - 1]
        curr_pos = trajectory[i]
        
        # Check if there was a wrap in x or y
        dx = abs(curr_pos[0] - prev_pos[0])
        dy = abs(curr_pos[1] - prev_pos[1])
        
        wrapped_x = dx > world_size[0] * wrap_threshold
        wrapped_y = dy > world_size[1] * wrap_threshold
        
        if wrapped_x or wrapped_y:
            # Wrap detected - start new segment
            segments.append(current_segment)
            current_segment = [curr_pos]
        else:
            # No wrap - continue current segment
            current_segment.append(curr_pos)
    
    # Add final segment
    if current_segment:
        segments.append(current_segment)
    
    return segments


def apply_periodic_boundaries_to_simulation(
    tracked_geoms: list[Any],
    world_size: tuple[float, float]
) -> None:
    """
    Apply periodic boundary wrapping to all robots in simulation.
    
    This should be called after each simulation step to teleport robots
    that have moved outside the boundaries.
    
    Args:
        tracked_geoms: List of tracked geom objects with .xpos attributes
        world_size: World dimensions [width, height]
    """
    for geom in tracked_geoms:
        pos = geom.xpos
        wrapped_x = pos[0] % world_size[0]
        wrapped_y = pos[1] % world_size[1]
        
        # Only modify if wrapping actually occurred
        if abs(wrapped_x - pos[0]) > 0.001 or abs(wrapped_y - pos[1]) > 0.001:
            pos[0] = wrapped_x
            pos[1] = wrapped_y
