#!/usr/bin/env python3
"""
Visual demonstration of periodic boundaries.

Creates a simple matplotlib visualization showing how periodic
boundaries work and why they matter for spatial EA.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def demo_periodic_concept():
    """Visualize the concept of periodic boundaries."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    world_size = 25
    
    # Robot positions
    robot_a = np.array([2, 12])
    robot_b = np.array([23, 12])
    
    # LEFT PLOT: Non-periodic (hard boundaries)
    ax = axes[0]
    ax.set_xlim(-1, world_size + 1)
    ax.set_ylim(-1, world_size + 1)
    ax.set_aspect('equal')
    ax.set_title('Non-Periodic (Standard)\nDistance = 21.0m', fontsize=14, fontweight='bold')
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.grid(True, alpha=0.3)
    
    # World boundary
    world_rect = patches.Rectangle(
        (0, 0), world_size, world_size,
        linewidth=3, edgecolor='black', facecolor='lightgray', alpha=0.2
    )
    ax.add_patch(world_rect)
    
    # Robots
    ax.scatter(*robot_a, s=500, c='blue', marker='o', edgecolor='darkblue', linewidth=2, zorder=10, label='Robot A')
    ax.scatter(*robot_b, s=500, c='red', marker='o', edgecolor='darkred', linewidth=2, zorder=10, label='Robot B')
    
    # Direct distance line
    ax.plot([robot_a[0], robot_b[0]], [robot_a[1], robot_b[1]], 
            'g--', linewidth=2, alpha=0.7, label='Direct distance: 21.0m')
    
    # Annotations
    ax.annotate('A', robot_a, fontsize=16, fontweight='bold', 
                ha='center', va='center', color='white')
    ax.annotate('B', robot_b, fontsize=16, fontweight='bold',
                ha='center', va='center', color='white')
    
    # Distance annotation
    mid_point = (robot_a + robot_b) / 2
    ax.annotate('21.0m', mid_point, fontsize=12, ha='center',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax.legend(loc='upper right')
    
    # RIGHT PLOT: Periodic (toroidal topology)
    ax = axes[1]
    ax.set_xlim(-1, world_size + 1)
    ax.set_ylim(-1, world_size + 1)
    ax.set_aspect('equal')
    ax.set_title('Periodic (Toroidal)\nDistance = 4.0m', fontsize=14, fontweight='bold')
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.grid(True, alpha=0.3)
    
    # World boundary (with wrap indicators)
    world_rect = patches.Rectangle(
        (0, 0), world_size, world_size,
        linewidth=3, edgecolor='purple', facecolor='lightgray', alpha=0.2,
        linestyle='--'
    )
    ax.add_patch(world_rect)
    
    # Robots
    ax.scatter(*robot_a, s=500, c='blue', marker='o', edgecolor='darkblue', linewidth=2, zorder=10, label='Robot A')
    ax.scatter(*robot_b, s=500, c='red', marker='o', edgecolor='darkred', linewidth=2, zorder=10, label='Robot B')
    
    # Periodic distance (wrapping around)
    # Show path wrapping left
    wrap_path_x = [robot_a[0], -1, -1, world_size + 1, world_size + 1, robot_b[0]]
    wrap_path_y = [robot_a[1], robot_a[1], robot_b[1], robot_b[1], robot_b[1], robot_b[1]]
    
    ax.plot([robot_a[0], 0], [robot_a[1], robot_a[1]], 
            'g-', linewidth=3, alpha=0.7)
    ax.plot([world_size, robot_b[0]], [robot_b[1], robot_b[1]], 
            'g-', linewidth=3, alpha=0.7, label='Wrapped distance: 4.0m')
    
    # Ghost robots (showing wrap-around)
    ax.scatter(robot_b[0] - world_size, robot_b[1], s=300, c='red', 
               marker='o', alpha=0.3, edgecolor='darkred', linewidth=1, zorder=5)
    ax.scatter(robot_a[0] + world_size, robot_a[1], s=300, c='blue',
               marker='o', alpha=0.3, edgecolor='darkblue', linewidth=1, zorder=5)
    
    # Annotations
    ax.annotate('A', robot_a, fontsize=16, fontweight='bold',
                ha='center', va='center', color='white')
    ax.annotate('B', robot_b, fontsize=16, fontweight='bold',
                ha='center', va='center', color='white')
    
    # Distance annotation
    ax.annotate('4.0m', (1, robot_a[1] + 1), fontsize=12, ha='center',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Wrap indicators
    ax.annotate('← Wraps', (world_size/2, -0.5), fontsize=10, ha='center', color='purple')
    ax.annotate('Wraps →', (world_size/2, world_size + 0.5), fontsize=10, ha='center', color='purple')
    
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('__figures__/periodic_boundaries_concept.png', dpi=150, bbox_inches='tight')
    print("Saved: __figures__/periodic_boundaries_concept.png")
    plt.show()


def demo_pairing_radius():
    """Show how pairing radius works with periodic boundaries."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    world_size = 25
    robot_pos = np.array([12.5, 12.5])
    
    # LEFT: Small radius (5m)
    ax = axes[0]
    ax.set_xlim(0, world_size)
    ax.set_ylim(0, world_size)
    ax.set_aspect('equal')
    ax.set_title('Pairing Radius = 5.0m (Good)\nLocal mating only', fontsize=14, fontweight='bold')
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.grid(True, alpha=0.3)
    
    # World
    world_rect = patches.Rectangle(
        (0, 0), world_size, world_size,
        linewidth=3, edgecolor='purple', facecolor='lightgray', alpha=0.2, linestyle='--'
    )
    ax.add_patch(world_rect)
    
    # Robot
    ax.scatter(*robot_pos, s=500, c='blue', marker='o', edgecolor='darkblue', linewidth=2, zorder=10)
    
    # Pairing radius circle
    circle = patches.Circle(robot_pos, 5.0, fill=False, edgecolor='green', linewidth=2, linestyle='--')
    ax.add_patch(circle)
    ax.fill_between([robot_pos[0]-5, robot_pos[0]+5], robot_pos[1]-5, robot_pos[1]+5,
                     alpha=0.2, color='green')
    
    # Sample other robots
    other_robots = np.array([[15, 14], [10, 10], [14, 9], [8, 15], [20, 20]])
    for i, pos in enumerate(other_robots):
        dist = np.linalg.norm(pos - robot_pos)
        if dist <= 5.0:
            ax.scatter(*pos, s=200, c='green', marker='s', alpha=0.7, label='Can pair' if i == 0 else '')
        else:
            ax.scatter(*pos, s=200, c='gray', marker='s', alpha=0.5, label='Too far' if i == 0 else '')
    
    ax.annotate('5m radius', robot_pos + [3.5, 3.5], fontsize=10,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    ax.legend()
    
    # RIGHT: Large radius (25m) - BAD!
    ax = axes[1]
    ax.set_xlim(0, world_size)
    ax.set_ylim(0, world_size)
    ax.set_aspect('equal')
    ax.set_title('Pairing Radius = 25.0m (BAD!)\nAll robots can pair', fontsize=14, fontweight='bold', color='red')
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.grid(True, alpha=0.3)
    
    # World
    world_rect = patches.Rectangle(
        (0, 0), world_size, world_size,
        linewidth=3, edgecolor='purple', facecolor='lightgray', alpha=0.2, linestyle='--'
    )
    ax.add_patch(world_rect)
    
    # Robot
    ax.scatter(*robot_pos, s=500, c='blue', marker='o', edgecolor='darkblue', linewidth=2, zorder=10)
    
    # Pairing radius - covers entire world!
    ax.fill_between([0, world_size], 0, world_size, alpha=0.3, color='red')
    
    # All other robots can pair
    for pos in other_robots:
        ax.scatter(*pos, s=200, c='green', marker='s', alpha=0.7)
    
    # Add some robots in corners
    corner_robots = np.array([[1, 1], [1, 24], [24, 1], [24, 24]])
    for pos in corner_robots:
        ax.scatter(*pos, s=200, c='green', marker='s', alpha=0.7)
    
    ax.annotate('MAX DIST = 12.5m\n(with periodic boundaries)', 
                (12.5, 2), fontsize=11, ha='center',
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
    
    ax.annotate('All robots\nwithin range!', (20, 20), fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('__figures__/periodic_boundaries_pairing_radius.png', dpi=150, bbox_inches='tight')
    print("Saved: __figures__/periodic_boundaries_pairing_radius.png")
    plt.show()


def demo_wrapping_visualization():
    """Show how trajectory wrapping works."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    world_size = 25
    
    ax.set_xlim(-1, world_size + 1)
    ax.set_ylim(-1, world_size + 1)
    ax.set_aspect('equal')
    ax.set_title('Robot Trajectory with Periodic Wrapping', fontsize=14, fontweight='bold')
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.grid(True, alpha=0.3)
    
    # World
    world_rect = patches.Rectangle(
        (0, 0), world_size, world_size,
        linewidth=3, edgecolor='purple', facecolor='lightgray', alpha=0.2, linestyle='--'
    )
    ax.add_patch(world_rect)
    
    # Trajectory that wraps
    trajectory = np.array([
        [10, 10],
        [15, 12],
        [20, 13],
        [23, 14],
        [24.5, 14.5],  # About to wrap
        [1.0, 15],     # Wrapped!
        [3, 16],
        [5, 18],
        [7, 22],
        [8, 24],       # About to wrap vertically
        [9, 1],        # Wrapped vertically!
        [12, 3],
        [15, 5],
    ])
    
    # Plot trajectory segments (split at wraps)
    colors = ['blue', 'green', 'red']
    segments = [
        trajectory[0:5],   # Before first wrap
        trajectory[5:9],   # After first wrap
        trajectory[9:],    # After second wrap
    ]
    
    for i, segment in enumerate(segments):
        ax.plot(segment[:, 0], segment[:, 1], 
                linewidth=3, color=colors[i], alpha=0.7,
                label=f'Segment {i+1}')
        
        # Mark start and end of each segment
        ax.scatter(segment[0, 0], segment[0, 1], s=200, c=colors[i], marker='o', zorder=10)
        ax.scatter(segment[-1, 0], segment[-1, 1], s=200, c=colors[i], marker='s', zorder=10)
    
    # Annotate wraps
    ax.annotate('Wrap X →', (24, 14.2), fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', lw=2))
    
    ax.annotate('Wrap Y →', (8.5, 24), fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', lw=2))
    
    # Start and end markers
    ax.scatter(*trajectory[0], s=500, c='green', marker='*', edgecolor='darkgreen', 
               linewidth=2, zorder=15, label='Start')
    ax.scatter(*trajectory[-1], s=500, c='red', marker='X', edgecolor='darkred',
               linewidth=2, zorder=15, label='End')
    
    ax.legend(loc='lower left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('__figures__/periodic_boundaries_trajectory.png', dpi=150, bbox_inches='tight')
    print("Saved: __figures__/periodic_boundaries_trajectory.png")
    plt.show()


if __name__ == '__main__':
    import os
    os.makedirs('__figures__', exist_ok=True)
    
    print("Creating visualizations...")
    print("\n1. Periodic boundaries concept...")
    demo_periodic_concept()
    
    print("\n2. Pairing radius demonstration...")
    demo_pairing_radius()
    
    print("\n3. Trajectory wrapping...")
    demo_wrapping_visualization()
    
    print("\n✓ All visualizations created!")
