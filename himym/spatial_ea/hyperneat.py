"""
HyperNEAT implementation for evolutionary robotics.

This module implements Compositional Pattern Producing Networks (CPPNs) and
the substrate-based ANN generation for robot control.
"""

import numpy as np
from typing import Callable, Any
from dataclasses import dataclass


# Activation functions for CPPN
def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation function."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def tanh_activation(x: np.ndarray) -> np.ndarray:
    """Tanh activation function."""
    return np.tanh(x)


def sine(x: np.ndarray) -> np.ndarray:
    """Sine activation function."""
    return np.sin(x)


def gaussian(x: np.ndarray) -> np.ndarray:
    """Gaussian activation function."""
    return np.exp(-x**2)


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation function."""
    return np.maximum(0, x)


def linear(x: np.ndarray) -> np.ndarray:
    """Linear activation function."""
    return x


def abs_activation(x: np.ndarray) -> np.ndarray:
    """Absolute value activation function."""
    return np.abs(x)


# Mapping of activation function names to functions
ACTIVATION_FUNCTIONS: dict[str, Callable] = {
    'sigmoid': sigmoid,
    'tanh': tanh_activation,
    'sine': sine,
    'gaussian': gaussian,
    'relu': relu,
    'linear': linear,
    'abs': abs_activation,
}


@dataclass
class CPPNNode:
    """Node in the CPPN network."""
    node_id: int
    activation: str  # Name of activation function
    layer: int  # 0 = input, 1+ = hidden, final = output


@dataclass
class CPPNConnection:
    """Connection in the CPPN network."""
    from_node: int
    to_node: int
    weight: float
    enabled: bool = True


class CPPN:
    """
    Compositional Pattern Producing Network.
    
    Takes spatial coordinates as input and outputs connection weights
    for the substrate ANN.
    """
    
    def __init__(self, genome: dict[str, Any]):
        """
        Initialize CPPN from genome.
        
        Args:
            genome: Dictionary containing 'nodes' and 'connections'
        """
        self.nodes: list[CPPNNode] = genome['nodes']
        self.connections: list[CPPNConnection] = genome['connections']
        
        # Build network structure
        self.num_inputs = sum(1 for n in self.nodes if n.layer == 0)
        self.num_outputs = sum(1 for n in self.nodes if n.layer == max(n.layer for n in self.nodes))
        
        # Create activation lookup
        self.node_activations = {
            node.node_id: ACTIVATION_FUNCTIONS[node.activation]
            for node in self.nodes
        }
        
        # Build connectivity matrix for efficient evaluation
        self._build_network_structure()
    
    def _build_network_structure(self) -> None:
        """Build network structure for efficient feed-forward evaluation."""
        # Sort nodes by layer
        self.nodes_by_layer: dict[int, list[CPPNNode]] = {}
        for node in self.nodes:
            if node.layer not in self.nodes_by_layer:
                self.nodes_by_layer[node.layer] = []
            self.nodes_by_layer[node.layer].append(node)
        
        # Build connection lookup
        self.incoming_connections: dict[int, list[CPPNConnection]] = {}
        for conn in self.connections:
            if conn.enabled:
                if conn.to_node not in self.incoming_connections:
                    self.incoming_connections[conn.to_node] = []
                self.incoming_connections[conn.to_node].append(conn)
    
    def activate(self, inputs: np.ndarray) -> np.ndarray:
        """
        Activate CPPN with input coordinates.
        
        Args:
            inputs: Input values (typically spatial coordinates)
            
        Returns:
            Output values (connection weight)
        """
        # Initialize node values
        node_values: dict[int, float] = {}
        
        # Set input values
        input_nodes = [n for n in self.nodes if n.layer == 0]
        for i, node in enumerate(input_nodes):
            node_values[node.node_id] = inputs[i] if i < len(inputs) else 0.0
        
        # Process each layer in order
        max_layer = max(n.layer for n in self.nodes)
        for layer in range(1, max_layer + 1):
            if layer not in self.nodes_by_layer:
                continue
            
            for node in self.nodes_by_layer[layer]:
                # Sum weighted inputs
                weighted_sum = 0.0
                if node.node_id in self.incoming_connections:
                    for conn in self.incoming_connections[node.node_id]:
                        if conn.from_node in node_values:
                            weighted_sum += conn.weight * node_values[conn.from_node]
                
                # Apply activation function
                activation_fn = self.node_activations[node.node_id]
                node_values[node.node_id] = activation_fn(np.array([weighted_sum]))[0]
        
        # Extract output values
        output_nodes = [n for n in self.nodes if n.layer == max_layer]
        outputs = np.array([node_values.get(n.node_id, 0.0) for n in output_nodes])
        
        return outputs


class SubstrateNetwork:
    """
    Substrate-based Artificial Neural Network.
    
    The substrate defines the spatial layout of neurons. HyperNEAT uses
    the CPPN to generate connection weights based on spatial relationships.
    """
    
    def __init__(
        self,
        input_coords: list[tuple[float, ...]],
        hidden_coords: list[tuple[float, ...]] | None,
        output_coords: list[tuple[float, ...]],
        cppn: CPPN,
        weight_threshold: float = 0.2
    ):
        """
        Initialize substrate network.
        
        Args:
            input_coords: Spatial coordinates of input neurons
            hidden_coords: Spatial coordinates of hidden neurons (optional)
            output_coords: Spatial coordinates of output neurons
            cppn: CPPN to generate weights
            weight_threshold: Minimum absolute weight to create connection
        """
        self.input_coords = input_coords
        self.hidden_coords = hidden_coords if hidden_coords else []
        self.output_coords = output_coords
        self.cppn = cppn
        self.weight_threshold = weight_threshold
        
        # Generate network weights from CPPN
        self._generate_weights()
    
    def _generate_weights(self) -> None:
        """Generate ANN connection weights using CPPN."""
        num_inputs = len(self.input_coords)
        num_hidden = len(self.hidden_coords)
        num_outputs = len(self.output_coords)
        
        # Generate weights for input -> hidden (or input -> output if no hidden)
        if num_hidden > 0:
            self.weights_input_hidden = np.zeros((num_inputs, num_hidden))
            for i, in_coord in enumerate(self.input_coords):
                for h, hid_coord in enumerate(self.hidden_coords):
                    # Query CPPN with source and target coordinates
                    cppn_input = np.array(list(in_coord) + list(hid_coord))
                    weight = self.cppn.activate(cppn_input)[0]
                    
                    # Apply threshold
                    if abs(weight) >= self.weight_threshold:
                        self.weights_input_hidden[i, h] = weight
            
            # Generate weights for hidden -> output
            self.weights_hidden_output = np.zeros((num_hidden, num_outputs))
            for h, hid_coord in enumerate(self.hidden_coords):
                for o, out_coord in enumerate(self.output_coords):
                    cppn_input = np.array(list(hid_coord) + list(out_coord))
                    weight = self.cppn.activate(cppn_input)[0]
                    
                    if abs(weight) >= self.weight_threshold:
                        self.weights_hidden_output[h, o] = weight
        else:
            # Direct input -> output connections
            self.weights_input_output = np.zeros((num_inputs, num_outputs))
            for i, in_coord in enumerate(self.input_coords):
                for o, out_coord in enumerate(self.output_coords):
                    cppn_input = np.array(list(in_coord) + list(out_coord))
                    weight = self.cppn.activate(cppn_input)[0]
                    
                    if abs(weight) >= self.weight_threshold:
                        self.weights_input_output[i, o] = weight
    
    def activate(self, inputs: np.ndarray) -> np.ndarray:
        """
        Activate substrate network.
        
        Args:
            inputs: Input sensor values
            
        Returns:
            Output actuator values
        """
        inputs = np.array(inputs)
        
        if len(self.hidden_coords) > 0:
            # Feed-forward through hidden layer
            hidden = np.tanh(np.dot(inputs, self.weights_input_hidden))
            # Use linear output with small scaling to prevent saturation
            outputs = 0.5 * np.dot(hidden, self.weights_hidden_output)
        else:
            # Direct input to output with linear activation
            outputs = 0.5 * np.dot(inputs, self.weights_input_output)
        
        return outputs


def create_minimal_cppn_genome(
    num_inputs: int = 4,
    num_outputs: int = 1,
    activation: str = 'sine'
) -> dict[str, Any]:
    """
    Create a minimal CPPN genome.
    
    Args:
        num_inputs: Number of input nodes (typically 4: x1, y1, x2, y2)
        num_outputs: Number of output nodes (typically 1: weight)
        activation: Activation function for output node
        
    Returns:
        CPPN genome dictionary
    """
    nodes = []
    connections = []
    
    # Input nodes (spatial coordinates)
    for i in range(num_inputs):
        nodes.append(CPPNNode(node_id=i, activation='linear', layer=0))
    
    # Output node
    output_id = num_inputs
    nodes.append(CPPNNode(node_id=output_id, activation=activation, layer=1))
    
    # Create connections from each input to output with random weights
    for i in range(num_inputs):
        weight = np.random.randn() * 0.5
        connections.append(CPPNConnection(
            from_node=i,
            to_node=output_id,
            weight=weight,
            enabled=True
        ))
    
    return {'nodes': nodes, 'connections': connections}


def create_substrate_for_gecko(
    num_joints: int,
    use_hidden_layer: bool = True,
    hidden_layer_size: int | None = None,
    num_cpg_oscillators: int = 4,
    num_directional_inputs: int = 2
) -> tuple[list[tuple[float, ...]], list[tuple[float, ...]] | None, list[tuple[float, ...]]]:
    """
    Create substrate configuration for gecko robot.
    
    The substrate defines spatial positions for:
    - Input neurons: Sensors (joint angles, CPG oscillators, directional inputs, bias)
    - Hidden neurons: Optional hidden layer  
    - Output neurons: Joint actuators
    
    Args:
        num_joints: Number of controllable joints
        use_hidden_layer: Whether to include hidden layer
        hidden_layer_size: Size of hidden layer (default: num_joints)
        num_cpg_oscillators: Number of CPG oscillator inputs (default: 4)
        num_directional_inputs: Number of directional input neurons (default: 2 for x,y direction)
        
    Returns:
        Tuple of (input_coords, hidden_coords, output_coords)
    """
    # Input layer: joint sensors + CPG oscillators + directional inputs + bias
    # Arranged in a line from -1 to 1
    num_inputs = num_joints + num_cpg_oscillators + num_directional_inputs + 1  # joints + CPG + direction + bias
    input_coords = []
    for i in range(num_inputs):
        x = -1.0 + (2.0 * i / (num_inputs - 1)) if num_inputs > 1 else 0.0
        input_coords.append((x, -0.5))  # y = -0.5 for input layer
    
    # Hidden layer (optional)
    hidden_coords = None
    if use_hidden_layer:
        hidden_size = hidden_layer_size if hidden_layer_size else num_joints
        hidden_coords = []
        for i in range(hidden_size):
            x = -1.0 + (2.0 * i / (hidden_size - 1)) if hidden_size > 1 else 0.0
            hidden_coords.append((x, 0.0))  # y = 0.0 for hidden layer
    
    # Output layer: joint actuators
    # Arranged in a line from -1 to 1
    output_coords = []
    for i in range(num_joints):
        x = -1.0 + (2.0 * i / (num_joints - 1)) if num_joints > 1 else 0.0
        output_coords.append((x, 0.5))  # y = 0.5 for output layer
    
    return input_coords, hidden_coords, output_coords
