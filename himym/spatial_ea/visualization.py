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
    save_path: str
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
    
    # Get unique IDs from population
    unique_ids = [ind.unique_id for ind in population if ind.unique_id is not None]
    
    # Create color mapping based on unique_id
    id_colors = [plt.cm.tab20(uid % 20 / 20) for uid in unique_ids]
    
    # Calculate attractiveness scores
    max_fitness = max(fitness_values) if max(fitness_values) > 0 else 1.0
    attractiveness = [f / max_fitness for f in fitness_values]
    
    # Fitness range for context
    min_fitness = min(fitness_values)
    
    for i, trajectory in enumerate(trajectories):
        trajectory = np.array(trajectory)
        
        color = id_colors[i]
        
        # Plot trajectory line
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
