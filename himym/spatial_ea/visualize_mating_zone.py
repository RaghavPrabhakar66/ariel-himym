#!/usr/bin/env python3
"""
Script to visualize the mating zone configuration.

This creates a standalone visualization showing the mating zone
and optionally some example robot positions.
"""

import os
from visualization import plot_mating_zone
from parent_selection import generate_random_zone_centers
from ea_config import EAConfig

def main():
    """Create mating zone visualization."""
    
    # Load configuration
    config = EAConfig()
    
    # Create figures folder if it doesn't exist
    os.makedirs(config.figures_folder, exist_ok=True)
    
    print("=" * 60)
    print("MATING ZONE VISUALIZATION")
    print("=" * 60)
    print(f"Pairing method: {config.pairing_method}")
    print(f"Number of zones: {config.num_mating_zones}")
    print(f"Mating zone radius: {config.mating_zone_radius}")
    print(f"Dynamic zones: {config.dynamic_mating_zones}")
    if config.dynamic_mating_zones:
        print(f"Zone change interval: {config.zone_change_interval} generations")
    print(f"World size: {config.world_size}")
    print()
    
    # Generate zone centers
    if config.num_mating_zones == 1:
        zone_centers = [tuple(config.mating_zone_center)]
    else:
        zone_centers = generate_random_zone_centers(
            num_zones=config.num_mating_zones,
            world_size=(config.world_size[0], config.world_size[1]),
            zone_radius=config.mating_zone_radius,
            min_zone_distance=config.min_zone_distance
        )
    
    print(f"Zone centers: {zone_centers}")
    print()
    
    # Example robot positions (optional)
    # Some inside the zones, some outside
    example_positions = []
    if len(zone_centers) > 0:
        # Add positions near first zone
        zone_center = zone_centers[0]
        example_positions = [
            # In zone
            zone_center,
            (zone_center[0] + 1.0, zone_center[1]),
            (zone_center[0], zone_center[1] + 1.5),
            (zone_center[0] - 0.5, zone_center[1] - 0.5),
            # Outside zone
            (zone_center[0] + 5.0, zone_center[1]),
            (1.0, 1.0),
            (config.world_size[0] - 1.0, config.world_size[1] - 1.0),
            (zone_center[0], zone_center[1] + 4.0),
        ]
    
    # Create visualization with example robots
    save_path = f"{config.figures_folder}/mating_zone_configuration.png"
    plot_mating_zone(
        world_size=config.world_size,
        mating_zone_centers=zone_centers,
        mating_zone_radius=config.mating_zone_radius,
        save_path=save_path,
        robot_positions=example_positions,
        use_periodic_boundaries=config.use_periodic_boundaries
    )
    
    # Create visualization without robots (clean view)
    save_path_clean = f"{config.figures_folder}/mating_zone_configuration_clean.png"
    plot_mating_zone(
        world_size=config.world_size,
        mating_zone_centers=zone_centers,
        mating_zone_radius=config.mating_zone_radius,
        save_path=save_path_clean,
        robot_positions=None,
        use_periodic_boundaries=config.use_periodic_boundaries
    )
    
    print("\n" + "=" * 60)
    print("Visualizations created successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
