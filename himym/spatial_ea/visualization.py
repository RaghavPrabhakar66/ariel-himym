"""
Visualization functions for evolutionary algorithm.

This module provides plotting and trajectory visualization
for robot populations and fitness evolution.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from spatial_individual import SpatialIndividual


def plot_fitness_evolution(
    fitness_history: list[float],
    save_path: str
) -> None:
    """
    Plot fitness over generations.
    
    Args:
        fitness_history: List of best fitness values per generation
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, len(fitness_history) + 1), 
        fitness_history, 
        'b-o', 
        linewidth=2, 
        markersize=6
    )
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness (Distance Traveled)")
    plt.title("Spatial EA: Evolution of Robot Movement")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path)
    print(f"Fitness plot saved to {save_path}")
    plt.close()


def save_mating_trajectories(
    trajectories: list[list[np.ndarray]],
    population: list[SpatialIndividual],
    fitness_values: list[float],
    generation: int,
    population_size: int,
    simulation_time: float,
    world_size: list[float],
    robot_size: float,
    save_path: str,
    use_periodic_boundaries: bool = False,
    mating_zone_centers: list[tuple[float, float]] | None = None,
    mating_zone_radius: float | None = None,
    pairing_method: str | None = None
) -> None:
    """
    Save visualization of robot trajectories during mating movement.
    
    Args:
        trajectories: List of trajectories (each is list of 2D positions)
        population: Population of individuals
        fitness_values: Fitness scores for each individual
        generation: Current generation number
        population_size: Size of population
        simulation_time: Duration of simulation
        world_size: Size of world [width, height]
        robot_size: Approximate robot diameter for visualization
        save_path: Path to save the figure
        use_periodic_boundaries: Whether to handle periodic boundary wrapping in visualization
        mating_zone_centers: Optional list of (x, y) centers of mating zones for visualization
        mating_zone_radius: Optional radius of mating zones for visualization
        pairing_method: Optional pairing method name for display
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Set axis limits first (needed for marker size calculation)
    ax.set_xlim(-0.2, world_size[0] + 0.2)
    ax.set_ylim(-0.2, world_size[1] + 0.2)
    ax.set_aspect('equal')
    
    # Calculate marker size to match robot size
    marker_size = _calculate_marker_size(robot_size, ax, fig)
    
    # Draw world boundaries
    world_rect = patches.Rectangle(
        (0, 0), 
        world_size[0], 
        world_size[1],
        linewidth=2, 
        edgecolor='black', 
        facecolor='lightgray',
        alpha=0.1
    )
    ax.add_patch(world_rect)
    
    # Draw mating zones if provided
    if mating_zone_centers is not None and mating_zone_radius is not None:
        for i, zone_center in enumerate(mating_zone_centers):
            mating_zone_circle = patches.Circle(
                zone_center,
                mating_zone_radius,
                linewidth=3,
                edgecolor='red',
                facecolor='lightcoral',
                alpha=0.2,
                linestyle='--',
                label='Mating Zone' if i == 0 else None  # Only label first zone
            )
            ax.add_patch(mating_zone_circle)
            
            # Mark the center of each mating zone
            ax.plot(
                zone_center[0],
                zone_center[1],
                marker='*',
                color='red',
                markersize=15,
                markeredgecolor='darkred',
                markeredgewidth=2,
                zorder=5,
                label='Zone Center' if i == 0 else None  # Only label first zone
            )
    
    # Get unique IDs from population
    unique_ids = [ind.unique_id for ind in population if ind.unique_id is not None]
    
    # Create color mapping based on unique_id
    id_colors = [plt.cm.tab20(uid % 20 / 20) for uid in unique_ids]
    
    # Fitness range for context
    max_fitness = max(fitness_values) if max(fitness_values) > 0 else 1.0
    min_fitness = min(fitness_values)
    
    for i, trajectory in enumerate(trajectories):
        trajectory = np.array(trajectory)
        
        color = id_colors[i]
        
        # Plot trajectory line with periodic boundary handling
        if use_periodic_boundaries:
            from periodic_boundary_utils import visualize_periodic_trajectory
            # Get trajectory segments that handle wrapping
            segments = visualize_periodic_trajectory(
                trajectory, 
                world_size[:2],  # Only x, y dimensions
                wrap_threshold=0.5
            )
            # Plot each segment separately to avoid lines across the world
            for segment in segments:
                segment_array = np.array(segment)
                ax.plot(
                    segment_array[:, 0], 
                    segment_array[:, 1], 
                    color=color, 
                    alpha=0.6, 
                    linewidth=2, 
                    zorder=1
                )
        else:
            # Standard non-periodic plotting
            ax.plot(
                trajectory[:, 0], 
                trajectory[:, 1], 
                color=color, 
                alpha=0.6, 
                linewidth=2, 
                zorder=1
            )
        
        # Mark start position (circle) - size matches robot
        ax.plot(
            trajectory[0, 0], 
            trajectory[0, 1], 
            'o', 
            color=color, 
            markersize=marker_size, 
            markeredgecolor='black', 
            markeredgewidth=1.5, 
            alpha=0.8, 
            zorder=2
        )
        
        # Mark end position (square) - size matches robot
        ax.plot(
            trajectory[-1, 0], 
            trajectory[-1, 1], 
            's', 
            color=color, 
            markersize=marker_size,
            markeredgecolor='black', 
            markeredgewidth=2,
            alpha=0.9, 
            zorder=3
        )
        
        # Add unique_id at end position
        ax.text(
            trajectory[-1, 0], 
            trajectory[-1, 1], 
            str(unique_ids[i]),
            ha='center', 
            va='center', 
            fontsize=7, 
            fontweight='bold', 
            color='white', 
            zorder=4
        )
        
        # Add fitness as small annotation near start position
        ax.text(
            trajectory[0, 0] + 0.15, 
            trajectory[0, 1] + 0.15, 
            f'{fitness_values[i]:.2f}',
            ha='left', 
            va='bottom', 
            fontsize=6, 
            color='black', 
            alpha=0.7, 
            zorder=4
        )
    
    # Plot settings
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_xlabel('X Position (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y Position (m)', fontsize=12, fontweight='bold')
    
    title = f'Mating Movement Trajectories - Generation {generation}\n'
    title += f'Population: {population_size} | Duration: {simulation_time}s\n'
    title += f'Fitness Range: {min_fitness:.3f} - {max_fitness:.3f}\n'
    if pairing_method:
        title += f'Pairing Method: {pairing_method}\n'
    if mating_zone_centers is not None and mating_zone_radius is not None:
        if len(mating_zone_centers) == 1:
            title += f'Mating Zone: Center ({mating_zone_centers[0][0]:.1f}, {mating_zone_centers[0][1]:.1f}), Radius {mating_zone_radius:.1f}m\n'
        else:
            title += f'Mating Zones: {len(mating_zone_centers)} zones, Radius {mating_zone_radius:.1f}m each\n'
    if use_periodic_boundaries:
        title += f'Periodic Boundaries: ENABLED (trajectories split at wraps)\n'
    title += f'Colors: Unique Individual IDs | Numbers: Individual IDs | Small text: Fitness'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add legend with matching marker sizes
    legend_elements = [
        Line2D(
            [0], [0], 
            marker='o', 
            color='w', 
            markerfacecolor='gray', 
            markersize=marker_size,
            markeredgecolor='black', 
            markeredgewidth=1.5,
            label='Start Position (robot size)'
        ),
        Line2D(
            [0], [0], 
            marker='s', 
            color='w',
            markerfacecolor='gray', 
            markersize=marker_size,
            markeredgecolor='black', 
            markeredgewidth=2,
            label='End Position (robot size, with ID)'
        ),
        Line2D(
            [0], [0], 
            color='gray', 
            linewidth=2,
            label='Trajectory (colored by individual)'
        ),
    ]
    
    # Add mating zones to legend if they exist
    if mating_zone_centers is not None and mating_zone_radius is not None:
        legend_elements.extend([
            patches.Patch(
                facecolor='lightcoral',
                edgecolor='red',
                linewidth=3,
                linestyle='--',
                alpha=0.2,
                label=f'Mating Zone{"s" if len(mating_zone_centers) > 1 else ""} (only robots inside can mate)'
            ),
            Line2D(
                [0], [0],
                marker='*',
                color='w',
                markerfacecolor='red',
                markersize=15,
                markeredgecolor='darkred',
                markeredgewidth=2,
                linestyle='',
                label='Mating Zone Center'
            )
        ])
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved mating trajectories to {save_path}")
    plt.close()


def _calculate_marker_size(
    robot_size_meters: float, 
    ax: Axes, 
    fig: Figure
) -> float:
    """
    Calculate matplotlib marker size to match physical robot size.
    
    Args:
        robot_size_meters: Robot diameter in meters (data units)
        ax: Matplotlib axes
        fig: Matplotlib figure
        
    Returns:
        Marker size in points
    """
    # Get axis bounds in data coordinates
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # Get figure size in inches
    fig_width, fig_height = fig.get_size_inches()
    
    # Calculate data units per inch
    data_width = xlim[1] - xlim[0]
    data_height = ylim[1] - ylim[0]
    data_per_inch_x = data_width / fig_width
    data_per_inch_y = data_height / fig_height
    
    # Use average to account for aspect ratio
    data_per_inch = (data_per_inch_x + data_per_inch_y) / 2
    
    # Convert robot size from data units (meters) to inches
    robot_size_inches = robot_size_meters / data_per_inch
    
    # Convert inches to points (1 inch = 72 points)
    # For circular markers, markersize is the diameter in points
    marker_size_points = robot_size_inches * 72
    
    return marker_size_points


def plot_mating_zone(
    world_size: list[float],
    mating_zone_centers: list[tuple[float, float]],
    mating_zone_radius: float,
    save_path: str,
    robot_positions: list[tuple[float, float]] | None = None,
    use_periodic_boundaries: bool = False
) -> None:
    """
    Create a standalone visualization of the mating zones.
    
    Useful for understanding zone placement and coverage. Supports multiple zones.
    
    Args:
        world_size: Size of world [width, height]
        mating_zone_centers: List of (x, y) centers of mating zones
        mating_zone_radius: Radius of mating zones
        save_path: Path to save the figure
        robot_positions: Optional list of (x, y) robot positions to show
        use_periodic_boundaries: Whether periodic boundaries are enabled
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Set axis limits
    ax.set_xlim(-0.2, world_size[0] + 0.2)
    ax.set_ylim(-0.2, world_size[1] + 0.2)
    ax.set_aspect('equal')
    
    # Draw world boundaries
    world_rect = patches.Rectangle(
        (0, 0), 
        world_size[0], 
        world_size[1],
        linewidth=2, 
        edgecolor='black', 
        facecolor='lightgray',
        alpha=0.1
    )
    ax.add_patch(world_rect)
    
    # Draw mating zones
    for i, zone_center in enumerate(mating_zone_centers):
        mating_zone_circle = patches.Circle(
            zone_center,
            mating_zone_radius,
            linewidth=3,
            edgecolor='red',
            facecolor='lightcoral',
            alpha=0.3,
            linestyle='--',
            label='Mating Zone' if i == 0 else None
        )
        ax.add_patch(mating_zone_circle)
        
        # Mark the center of each mating zone
        ax.plot(
            zone_center[0],
            zone_center[1],
            marker='*',
            color='red',
            markersize=20,
            markeredgecolor='darkred',
            markeredgewidth=2,
            zorder=5,
            label='Zone Center' if i == 0 else None
        )
    
    # Plot robot positions if provided
    if robot_positions:
        for i, (x, y) in enumerate(robot_positions):
            # Check if robot is in any zone
            in_zone = False
            for zone_center in mating_zone_centers:
                distance = np.sqrt((x - zone_center[0])**2 + (y - zone_center[1])**2)
                if distance <= mating_zone_radius:
                    in_zone = True
                    break
            
            color = 'green' if in_zone else 'gray'
            marker = 'o' if in_zone else 'x'
            label = 'In Zone' if i == 0 and in_zone else ('Outside Zone' if i == 0 and not in_zone else None)
            
            ax.plot(
                x, y,
                marker=marker,
                color=color,
                markersize=10,
                markeredgecolor='black',
                markeredgewidth=1,
                label=label
            )
            
            # Add robot index
            ax.text(x, y + 0.3, str(i), ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Calculate zone statistics
    zone_area = np.pi * mating_zone_radius**2
    world_area = world_size[0] * world_size[1]
    
    # Plot settings
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_xlabel('X Position (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y Position (m)', fontsize=12, fontweight='bold')
    
    title = 'Mating Zone Configuration\n'
    if len(mating_zone_centers) == 1:
        title += f'Zone Center: ({mating_zone_centers[0][0]:.1f}, {mating_zone_centers[0][1]:.1f})\n'
    else:
        title += f'Number of Zones: {len(mating_zone_centers)}\n'
    title += f'Zone Radius: {mating_zone_radius:.1f}m | Zone Area: {zone_area:.2f} m² each\n'
    title += f'Total Zone Area: {zone_area * len(mating_zone_centers):.2f} m²\n'
    title += f'World Size: {world_size[0]:.1f}m × {world_size[1]:.1f}m | World Area: {world_area:.2f} m²\n'
    coverage_percent_total = (zone_area * len(mating_zone_centers) / world_area) * 100
    title += f'Total Zone Coverage: {coverage_percent_total:.1f}% of world\n'
    if use_periodic_boundaries:
        title += 'Periodic Boundaries: ENABLED'
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add legend
    legend_elements = [
        patches.Patch(
            facecolor='lightcoral',
            edgecolor='red',
            linewidth=3,
            linestyle='--',
            alpha=0.3,
            label='Mating Zone'
        ),
        Line2D(
            [0], [0],
            marker='*',
            color='w',
            markerfacecolor='red',
            markersize=20,
            markeredgecolor='darkred',
            markeredgewidth=2,
            linestyle='',
            label='Zone Center'
        )
    ]
    
    if robot_positions:
        legend_elements.extend([
            Line2D(
                [0], [0],
                marker='o',
                color='w',
                markerfacecolor='green',
                markersize=10,
                markeredgecolor='black',
                linestyle='',
                label='Robot In Zone (can mate)'
            ),
            Line2D(
                [0], [0],
                marker='x',
                color='gray',
                markersize=10,
                markeredgecolor='black',
                linestyle='',
                label='Robot Outside Zone (cannot mate)'
            )
        ])
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Mating zone visualization saved to {save_path}")
    plt.close()
