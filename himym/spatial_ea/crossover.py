import numpy as np
from typing import Tuple, List

def single_point_crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Single-point crossover: Split at random point and swap segments.
    
    Args:
        parent1: First parent array
        parent2: Second parent array
    
    Returns:
        Tuple of two offspring arrays
    """
    length = len(parent1)
    point = np.random.randint(1, length)
    
    offspring1 = np.concatenate([parent1[:point], parent2[point:]])
    offspring2 = np.concatenate([parent2[:point], parent1[point:]])
    
    return offspring1, offspring2


def two_point_crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Two-point crossover: Swap the segment between two random points.
    
    Args:
        parent1: First parent array
        parent2: Second parent array
    
    Returns:
        Tuple of two offspring arrays
    """
    length = len(parent1)
    point1, point2 = sorted(np.random.choice(range(1, length), size=2, replace=False))
    
    offspring1 = np.concatenate([parent1[:point1], parent2[point1:point2], parent1[point2:]])
    offspring2 = np.concatenate([parent2[:point1], parent1[point1:point2], parent2[point2:]])
    
    return offspring1, offspring2


def uniform_crossover(parent1: np.ndarray, parent2: np.ndarray, prob: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Uniform crossover: Each gene independently chosen from either parent.
    
    Args:
        parent1: First parent array
        parent2: Second parent array
        prob: Probability of selecting from parent1
    
    Returns:
        Tuple of two offspring arrays
    """
    mask = np.random.random(len(parent1)) < prob
    
    offspring1 = np.where(mask, parent1, parent2)
    offspring2 = np.where(mask, parent2, parent1)
    
    return offspring1, offspring2


def blend_crossover(parent1: np.ndarray, parent2: np.ndarray, alpha: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Blend crossover (BLX-α): Offspring values randomly selected from extended range.
    
    Args:
        parent1: First parent array
        parent2: Second parent array
        alpha: Extension factor (0.5 is common)
    
    Returns:
        Tuple of two offspring arrays
    """
    min_vals = np.minimum(parent1, parent2)
    max_vals = np.maximum(parent1, parent2)
    range_vals = max_vals - min_vals
    
    lower = min_vals - alpha * range_vals
    upper = max_vals + alpha * range_vals
    
    offspring1 = np.random.uniform(lower, upper)
    offspring2 = np.random.uniform(lower, upper)
    
    return offspring1, offspring2


def simulated_binary_crossover(parent1: np.ndarray, parent2: np.ndarray, eta: float = 20.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulated Binary Crossover (SBX): Mimics single-point crossover for binary strings.
    
    Args:
        parent1: First parent array
        parent2: Second parent array
        eta: Distribution index (larger = more similar to parents)
    
    Returns:
        Tuple of two offspring arrays
    """
    offspring1 = np.zeros_like(parent1)
    offspring2 = np.zeros_like(parent2)
    
    for i in range(len(parent1)):
        if np.random.random() < 0.5:
            if abs(parent1[i] - parent2[i]) > 1e-14:
                u = np.random.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1.0 / (eta + 1))
                else:
                    beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                
                offspring1[i] = 0.5 * ((parent1[i] + parent2[i]) - beta * abs(parent1[i] - parent2[i]))
                offspring2[i] = 0.5 * ((parent1[i] + parent2[i]) + beta * abs(parent1[i] - parent2[i]))
            else:
                offspring1[i] = parent1[i]
                offspring2[i] = parent2[i]
        else:
            offspring1[i] = parent1[i]
            offspring2[i] = parent2[i]
    
    return offspring1, offspring2


def arithmetic_crossover(parent1: np.ndarray, parent2: np.ndarray, alpha: float = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Arithmetic crossover: Linear combination of parents.
    
    Args:
        parent1: First parent array
        parent2: Second parent array
        alpha: Blending coefficient (if None, random for each gene)
    
    Returns:
        Tuple of two offspring arrays
    """
    if alpha is None:
        alpha = np.random.random(len(parent1))
    
    offspring1 = alpha * parent1 + (1 - alpha) * parent2
    offspring2 = (1 - alpha) * parent1 + alpha * parent2
    
    return offspring1, offspring2


# Example usage
if __name__ == "__main__":
    # Create sample parents
    p1 = np.array([1.5, 2.3, 3.7, 4.2, 5.9])
    p2 = np.array([6.1, 7.8, 8.4, 9.0, 10.5])
    
    print("Parent 1:", p1)
    print("Parent 2:", p2)
    print("\n" + "="*60 + "\n")
    
    # Test all crossover operators
    print("Single-Point Crossover:")
    o1, o2 = single_point_crossover(p1, p2)
    print(f"Offspring 1: {o1}")
    print(f"Offspring 2: {o2}\n")
    
    print("Two-Point Crossover:")
    o1, o2 = two_point_crossover(p1, p2)
    print(f"Offspring 1: {o1}")
    print(f"Offspring 2: {o2}\n")
    
    print("Uniform Crossover:")
    o1, o2 = uniform_crossover(p1, p2)
    print(f"Offspring 1: {o1}")
    print(f"Offspring 2: {o2}\n")
    
    print("Blend Crossover (BLX-0.5):")
    o1, o2 = blend_crossover(p1, p2, alpha=0.5)
    print(f"Offspring 1: {o1}")
    print(f"Offspring 2: {o2}\n")
    
    print("Simulated Binary Crossover (SBX):")
    o1, o2 = simulated_binary_crossover(p1, p2, eta=20.0)
    print(f"Offspring 1: {o1}")
    print(f"Offspring 2: {o2}\n")
    
    print("Arithmetic Crossover (α=0.3):")
    o1, o2 = arithmetic_crossover(p1, p2, alpha=0.3)
    print(f"Offspring 1: {o1}")
    print(f"Offspring 2: {o2}\n")