"""
Genetic operators for evolutionary algorithm.

This module provides crossover and mutation functions for evolving
robot control parameters (HyperNEAT neural network genotypes).
"""

import numpy as np
import copy
from hyperneat import CPPNNode, CPPNConnection, ACTIVATION_FUNCTIONS
from spatial_individual import SpatialIndividual
from ea_config import config
from typing import Any
import copy


def clone_individual(
    individual: SpatialIndividual,
    next_unique_id: int,
    generation: int
) -> tuple[SpatialIndividual, int]:
    """
    Clone an individual with a new unique ID.
    
    Args:
        individual: Individual to clone
        next_unique_id: Next available unique ID
        generation: Generation number for clone
        
    Returns:
        Tuple of (cloned_individual, updated_next_unique_id)
    """
    clone = SpatialIndividual(
        unique_id=next_unique_id, 
        generation=generation,
        initial_energy=config.initial_energy
    )
    next_unique_id += 1
    clone.genotype = individual.genotype.copy()
    clone.parent_ids = [individual.unique_id]
    
    return clone, next_unique_id


# =============================================================================
# HyperNEAT Genetic Operators
# =============================================================================

def create_initial_hyperneat_genome(
    num_inputs: int = 4,
    num_outputs: int = 1,
    activation: str = 'sine'
) -> dict[str, Any]:
    """
    Create initial HyperNEAT CPPN genome with high diversity.
    
    Creates varied initial networks with:
    - Wide weight distribution (std=2.0 instead of 0.5)
    - Random activation functions for output
    - Probabilistic hidden nodes (30% chance)
    - Extra random connections (20% chance each)
    
    Args:
        num_inputs: Number of CPPN inputs (spatial coordinates)
        num_outputs: Number of CPPN outputs (weight value)
        activation: Activation function for output node (can be randomized)
        
    Returns:
        CPPN genome dictionary
    """
    nodes = []
    connections = []
    
    # Input nodes (spatial coordinates: x1, y1, x2, y2)
    for i in range(num_inputs):
        nodes.append(CPPNNode(node_id=i, activation='linear', layer=0))
    
    # Randomize output activation for diversity (if activation is 'sine')
    if activation == 'sine':
        # Choose from activations that work well for CPPNs
        output_activation = np.random.choice([
            'sine', 'tanh', 'gaussian', 'sigmoid', 'linear', 'relu', 'abs'
        ])
    else:
        output_activation = activation
    
    # Decide if we should add hidden nodes (50% chance - increased from 30%)
    add_hidden = np.random.random() < 0.5
    
    if add_hidden:
        # Add 1-2 hidden nodes
        num_hidden = np.random.randint(1, 3)
        hidden_ids = []
        
        for h in range(num_hidden):
            hidden_id = num_inputs + h
            hidden_activation = np.random.choice(list(ACTIVATION_FUNCTIONS.keys()))
            nodes.append(CPPNNode(
                node_id=hidden_id,
                activation=hidden_activation,
                layer=1
            ))
            hidden_ids.append(hidden_id)
        
        # Output node in layer 2
        output_id = num_inputs + num_hidden
        nodes.append(CPPNNode(
            node_id=output_id,
            activation=output_activation,
            layer=2
        ))
        
        # Connect inputs to hidden nodes with WIDE weight range
        for i in range(num_inputs):
            for hidden_id in hidden_ids:
                # 70% chance to connect each input to each hidden (increased from 60%)
                if np.random.random() < 0.7:
                    weight = np.random.randn() * 3.0  # Even wider distribution (was 2.0)
                    connections.append(CPPNConnection(
                        from_node=i,
                        to_node=hidden_id,
                        weight=weight,
                        enabled=True
                    ))
        
        # Connect hidden to output
        for hidden_id in hidden_ids:
            weight = np.random.randn() * 3.0
            connections.append(CPPNConnection(
                from_node=hidden_id,
                to_node=output_id,
                weight=weight,
                enabled=True
            ))
        
        # Also allow some direct input-to-output connections (40% chance each)
        for i in range(num_inputs):
            if np.random.random() < 0.4:
                weight = np.random.randn() * 3.0
                connections.append(CPPNConnection(
                    from_node=i,
                    to_node=output_id,
                    weight=weight,
                    enabled=True
                ))
    else:
        # Minimal structure: direct input to output
        output_id = num_inputs
        nodes.append(CPPNNode(
            node_id=output_id,
            activation=output_activation,
            layer=1
        ))
        
        # Create connections from each input to output with VERY WIDE weight range
        for i in range(num_inputs):
            # Use very wide distribution for maximum diversity
            weight = np.random.randn() * 3.0  # Increased from 2.0
            connections.append(CPPNConnection(
                from_node=i,
                to_node=output_id,
                weight=weight,
                enabled=True
            ))
        
        # Add extra random connections (30% chance for each, increased from 20%)
        for i in range(num_inputs):
            if np.random.random() < 0.3 and len([c for c in connections if c.from_node == i]) < 2:
                weight = np.random.randn() * 3.0
                connections.append(CPPNConnection(
                    from_node=i,
                    to_node=output_id,
                    weight=weight,
                    enabled=True
                ))
    
    return {'nodes': nodes, 'connections': connections}


def crossover_hyperneat(
    parent1: SpatialIndividual,
    parent2: SpatialIndividual,
    next_unique_id: int,
    generation: int
) -> tuple[SpatialIndividual, SpatialIndividual, int]:
    """
    Perform crossover between two HyperNEAT individuals.
    
    Uses NEAT-style crossover: matching genes are randomly selected,
    disjoint/excess genes come from fitter parent.
    
    Args:
        parent1: First parent individual
        parent2: Second parent individual
        next_unique_id: Next available unique ID for offspring
        generation: Generation number for offspring
        
    Returns:
        Tuple of (child1, child2, updated_next_unique_id)
    """
    child1 = SpatialIndividual(
        unique_id=next_unique_id, 
        generation=generation,
        initial_energy=config.initial_energy
    )
    next_unique_id += 1
    child1.parent_ids = [parent1.unique_id, parent2.unique_id]
    
    child2 = SpatialIndividual(
        unique_id=next_unique_id, 
        generation=generation,
        initial_energy=config.initial_energy
    )
    next_unique_id += 1
    child2.parent_ids = [parent1.unique_id, parent2.unique_id]
    
    # Deep copy parent genomes
    genome1 = copy.deepcopy(parent1.genotype)
    genome2 = copy.deepcopy(parent2.genotype)
    
    # Create child genomes by combining parent connections
    child1_genome = copy.deepcopy(genome1)
    child2_genome = copy.deepcopy(genome2)
    
    # Crossover connections
    # Build connection maps
    conn_map1 = {(c.from_node, c.to_node): c for c in genome1['connections']}
    conn_map2 = {(c.from_node, c.to_node): c for c in genome2['connections']}
    
    # For matching connections, randomly choose from either parent
    all_conn_keys = set(conn_map1.keys()) | set(conn_map2.keys())
    
    child1_connections = []
    child2_connections = []
    
    for key in all_conn_keys:
        if key in conn_map1 and key in conn_map2:
            # Matching gene - randomly choose
            if np.random.random() < 0.5:
                child1_connections.append(copy.deepcopy(conn_map1[key]))
                child2_connections.append(copy.deepcopy(conn_map2[key]))
            else:
                child1_connections.append(copy.deepcopy(conn_map2[key]))
                child2_connections.append(copy.deepcopy(conn_map1[key]))
        elif key in conn_map1:
            # Disjoint - include from parent1
            child1_connections.append(copy.deepcopy(conn_map1[key]))
            child2_connections.append(copy.deepcopy(conn_map1[key]))
        else:
            # Disjoint - include from parent2
            child1_connections.append(copy.deepcopy(conn_map2[key]))
            child2_connections.append(copy.deepcopy(conn_map2[key]))
    
    child1_genome['connections'] = child1_connections
    child2_genome['connections'] = child2_connections
    
    child1.genotype = child1_genome
    child2.genotype = child2_genome
    
    return child1, child2, next_unique_id


def mutate_hyperneat(
    individual: SpatialIndividual,
    next_unique_id: int,
    weight_mutation_rate: float = 0.8,
    weight_mutation_power: float = 0.5,
    add_connection_rate: float = 0.05,
    add_node_rate: float = 0.03
) -> tuple[SpatialIndividual, int]:
    """
    Apply mutation to HyperNEAT CPPN genome.
    
    Mutations:
    - Weight mutation: Perturb connection weights
    - Add connection: Create new connection between nodes
    - Add node: Split existing connection with new node
    
    Args:
        individual: Individual to mutate
        next_unique_id: Next available unique ID
        weight_mutation_rate: Probability of mutating each weight
        weight_mutation_power: Standard deviation of weight perturbation
        add_connection_rate: Probability of adding new connection
        add_node_rate: Probability of adding new node
        
    Returns:
        Tuple of (mutated_individual, updated_next_unique_id)
    """    
    mutated = SpatialIndividual(
        unique_id=next_unique_id, 
        generation=individual.generation,
        initial_energy=config.initial_energy
    )
    next_unique_id += 1
    mutated.parent_ids = [individual.unique_id]
    
    # Deep copy genome
    genome = copy.deepcopy(individual.genotype)
    mutated.genotype = genome
    
    # 1. Weight mutation
    for conn in genome['connections']:
        if np.random.random() < weight_mutation_rate:
            if np.random.random() < 0.9:
                # Perturb weight
                conn.weight += np.random.normal(0, weight_mutation_power)
                conn.weight = np.clip(conn.weight, -3.0, 3.0)
            else:
                # Assign new random weight
                conn.weight = np.random.randn() * 2.0
    
    # 2. Add connection mutation
    if np.random.random() < add_connection_rate:
        nodes = genome['nodes']
        # Try to add connection between two random nodes
        if len(nodes) >= 2:
            # Get nodes from different layers
            input_output_nodes = [n for n in nodes if n.layer in [0, max(n.layer for n in nodes)]]
            if len(input_output_nodes) >= 2:
                from_node = np.random.choice([n for n in nodes if n.layer < max(n.layer for n in nodes)])
                to_node = np.random.choice([n for n in nodes if n.layer > from_node.layer])
                
                # Check if connection already exists
                existing = any(
                    c.from_node == from_node.node_id and c.to_node == to_node.node_id
                    for c in genome['connections']
                )
                
                if not existing:
                    new_conn = CPPNConnection(
                        from_node=from_node.node_id,
                        to_node=to_node.node_id,
                        weight=np.random.randn() * 0.5,
                        enabled=True
                    )
                    genome['connections'].append(new_conn)
    
    # 3. Add node mutation
    if np.random.random() < add_node_rate and len(genome['connections']) > 0:
        # Choose random enabled connection to split
        enabled_conns = [c for c in genome['connections'] if c.enabled]
        if enabled_conns:
            conn_to_split = np.random.choice(enabled_conns)
            
            # Verify nodes exist before splitting
            from_node = next((n for n in genome['nodes'] if n.node_id == conn_to_split.from_node), None)
            to_node = next((n for n in genome['nodes'] if n.node_id == conn_to_split.to_node), None)
            
            if from_node is None or to_node is None:
                # Skip this mutation - connection references missing nodes
                # This can happen due to crossover or data corruption
                print(f"Warning: Connection {conn_to_split.from_node}->{conn_to_split.to_node} references missing nodes, skipping add_node mutation")
            else:
                # Disable old connection
                conn_to_split.enabled = False
                
                # Create new node
                new_node_id = max(n.node_id for n in genome['nodes']) + 1
                new_layer = from_node.layer + 1
                
                # Choose random activation
                activation = np.random.choice(list(ACTIVATION_FUNCTIONS.keys()))
                new_node = CPPNNode(
                    node_id=new_node_id,
                    activation=activation,
                    layer=new_layer
                )
                genome['nodes'].append(new_node)
                
                # Create two new connections
                conn1 = CPPNConnection(
                    from_node=from_node.node_id,
                    to_node=new_node_id,
                    weight=1.0,
                    enabled=True
                )
                conn2 = CPPNConnection(
                    from_node=new_node_id,
                    to_node=to_node.node_id,
                    weight=conn_to_split.weight,
                    enabled=True
                )
                genome['connections'].extend([conn1, conn2])
    
    return mutated, next_unique_id
