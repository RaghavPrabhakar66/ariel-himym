import numpy as np
from typing import Optional

def uniform_mutation(individual: np.ndarray, bounds: tuple, mutation_rate: float = 0.1) -> np.ndarray:
    """
    Uniform mutation: Replace genes with random values from uniform distribution.
    
    Args:
        individual: Individual to mutate
        bounds: (min, max) values for genes
        mutation_rate: Probability of mutating each gene
    
    Returns:
        Mutated individual
    """
    mutant = individual.copy()
    mask = np.random.random(len(individual)) < mutation_rate
    mutant[mask] = np.random.uniform(bounds[0], bounds[1], np.sum(mask))
    return mutant


def gaussian_mutation(individual: np.ndarray, mutation_rate: float = 0.1, 
                     sigma: float = 0.1, bounds: Optional[tuple] = None) -> np.ndarray:
    """
    Gaussian mutation: Add random value from normal distribution to genes.
    
    Args:
        individual: Individual to mutate
        mutation_rate: Probability of mutating each gene
        sigma: Standard deviation of Gaussian distribution
        bounds: Optional (min, max) to clip values
    
    Returns:
        Mutated individual
    """
    mutant = individual.copy()
    mask = np.random.random(len(individual)) < mutation_rate
    mutant[mask] += np.random.normal(0, sigma, np.sum(mask))
    
    if bounds is not None:
        mutant = np.clip(mutant, bounds[0], bounds[1])
    
    return mutant


def polynomial_mutation(individual: np.ndarray, bounds: tuple, 
                       mutation_rate: float = 0.1, eta: float = 20.0) -> np.ndarray:
    """
    Polynomial mutation: Similar to SBX crossover, creates small perturbations.
    
    Args:
        individual: Individual to mutate
        bounds: (min, max) values for genes
        mutation_rate: Probability of mutating each gene
        eta: Distribution index (larger = smaller perturbations)
    
    Returns:
        Mutated individual
    """
    mutant = individual.copy()
    
    for i in range(len(individual)):
        if np.random.random() < mutation_rate:
            u = np.random.random()
            
            if u < 0.5:
                delta = (2 * u) ** (1.0 / (eta + 1)) - 1
            else:
                delta = 1 - (2 * (1 - u)) ** (1.0 / (eta + 1))
            
            mutant[i] += delta * (bounds[1] - bounds[0])
            mutant[i] = np.clip(mutant[i], bounds[0], bounds[1])
    
    return mutant


def creep_mutation(individual: np.ndarray, bounds: tuple, 
                   mutation_rate: float = 0.1, creep_rate: float = 0.05) -> np.ndarray:
    """
    Creep mutation: Small incremental changes relative to current value.
    
    Args:
        individual: Individual to mutate
        bounds: (min, max) values for genes
        mutation_rate: Probability of mutating each gene
        creep_rate: Maximum relative change (e.g., 0.05 = ±5%)
    
    Returns:
        Mutated individual
    """
    mutant = individual.copy()
    mask = np.random.random(len(individual)) < mutation_rate
    
    # Random perturbation within ±creep_rate of current value
    perturbation = np.random.uniform(-creep_rate, creep_rate, np.sum(mask))
    mutant[mask] *= (1 + perturbation)
    mutant = np.clip(mutant, bounds[0], bounds[1])
    
    return mutant


def non_uniform_mutation(individual: np.ndarray, bounds: tuple, 
                        generation: int, max_generations: int,
                        mutation_rate: float = 0.1, b: float = 5.0) -> np.ndarray:
    """
    Non-uniform mutation: Decreases perturbation magnitude as generations progress.
    
    Args:
        individual: Individual to mutate
        bounds: (min, max) values for genes
        generation: Current generation number
        max_generations: Total number of generations
        mutation_rate: Probability of mutating each gene
        b: Shape parameter (controls decay rate)
    
    Returns:
        Mutated individual
    """
    mutant = individual.copy()
    
    for i in range(len(individual)):
        if np.random.random() < mutation_rate:
            r = np.random.random()
            t = (1 - generation / max_generations) ** b
            
            if np.random.random() < 0.5:
                delta = (bounds[1] - mutant[i]) * (1 - r ** t)
            else:
                delta = -(mutant[i] - bounds[0]) * (1 - r ** t)
            
            mutant[i] += delta
    
    return mutant


def boundary_mutation(individual: np.ndarray, bounds: tuple, 
                     mutation_rate: float = 0.1) -> np.ndarray:
    """
    Boundary mutation: Replace gene with either min or max boundary value.
    
    Args:
        individual: Individual to mutate
        bounds: (min, max) values for genes
        mutation_rate: Probability of mutating each gene
    
    Returns:
        Mutated individual
    """
    mutant = individual.copy()
    mask = np.random.random(len(individual)) < mutation_rate
    
    # Randomly choose upper or lower bound
    boundary_values = np.where(np.random.random(np.sum(mask)) < 0.5, 
                               bounds[0], bounds[1])
    mutant[mask] = boundary_values
    
    return mutant


def adaptive_gaussian_mutation(individual: np.ndarray, fitness_improvement: bool,
                              mutation_rate: float = 0.1, sigma: float = 0.1,
                              bounds: Optional[tuple] = None,
                              increase_factor: float = 1.2,
                              decrease_factor: float = 0.8) -> tuple:
    """
    Adaptive Gaussian mutation: Adjusts sigma based on fitness improvement.
    
    Args:
        individual: Individual to mutate
        fitness_improvement: Whether last mutation improved fitness
        mutation_rate: Probability of mutating each gene
        sigma: Current standard deviation
        bounds: Optional (min, max) to clip values
        increase_factor: Factor to increase sigma if no improvement
        decrease_factor: Factor to decrease sigma if improvement
    
    Returns:
        Tuple of (mutated individual, new sigma)
    """
    # Adjust sigma based on success
    new_sigma = sigma * decrease_factor if fitness_improvement else sigma * increase_factor
    new_sigma = np.clip(new_sigma, 0.001, 1.0)  # Keep sigma in reasonable range
    
    mutant = gaussian_mutation(individual, mutation_rate, new_sigma, bounds)
    
    return mutant, new_sigma


def cauchy_mutation(individual: np.ndarray, mutation_rate: float = 0.1,
                   scale: float = 0.1, bounds: Optional[tuple] = None) -> np.ndarray:
    """
    Cauchy mutation: Uses Cauchy distribution (heavier tails than Gaussian).
    Better for escaping local optima.
    
    Args:
        individual: Individual to mutate
        mutation_rate: Probability of mutating each gene
        scale: Scale parameter of Cauchy distribution
        bounds: Optional (min, max) to clip values
    
    Returns:
        Mutated individual
    """
    mutant = individual.copy()
    mask = np.random.random(len(individual)) < mutation_rate
    
    # Cauchy distribution has heavier tails than Gaussian
    cauchy_samples = np.random.standard_cauchy(np.sum(mask)) * scale
    mutant[mask] += cauchy_samples
    
    if bounds is not None:
        mutant = np.clip(mutant, bounds[0], bounds[1])
    
    return mutant


# Example usage
if __name__ == "__main__":
    # Create sample individual
    individual = np.array([2.5, 5.0, 7.5, 3.2, 8.9])
    bounds = (0.0, 10.0)
    
    print("Original Individual:", individual)
    print("Bounds:", bounds)
    print("\n" + "="*60 + "\n")
    
    # Test all mutation operators
    print("Uniform Mutation:")
    mutant = uniform_mutation(individual, bounds, mutation_rate=0.4)
    print(f"Mutated: {mutant}\n")
    
    print("Gaussian Mutation:")
    mutant = gaussian_mutation(individual, mutation_rate=0.4, sigma=0.5, bounds=bounds)
    print(f"Mutated: {mutant}\n")
    
    print("Polynomial Mutation:")
    mutant = polynomial_mutation(individual, bounds, mutation_rate=0.4, eta=20.0)
    print(f"Mutated: {mutant}\n")
    
    print("Creep Mutation:")
    mutant = creep_mutation(individual, bounds, mutation_rate=0.4, creep_rate=0.1)
    print(f"Mutated: {mutant}\n")
    
    print("Non-Uniform Mutation (generation 50/100):")
    mutant = non_uniform_mutation(individual, bounds, generation=50, 
                                  max_generations=100, mutation_rate=0.4)
    print(f"Mutated: {mutant}\n")
    
    print("Boundary Mutation:")
    mutant = boundary_mutation(individual, bounds, mutation_rate=0.4)
    print(f"Mutated: {mutant}\n")
    
    print("Adaptive Gaussian Mutation (fitness improved):")
    mutant, new_sigma = adaptive_gaussian_mutation(individual, fitness_improvement=True,
                                                   mutation_rate=0.4, sigma=0.3, bounds=bounds)
    print(f"Mutated: {mutant}")
    print(f"New sigma: {new_sigma}\n")
    
    print("Cauchy Mutation:")
    mutant = cauchy_mutation(individual, mutation_rate=0.4, scale=0.3, bounds=bounds)
    print(f"Mutated: {mutant}\n")
    
    # Demonstrate mutation rate effect
    print("="*60)
    print("\nMutation Rate Comparison (Gaussian, σ=0.5):")
    for rate in [0.1, 0.3, 0.5, 1.0]:
        mutant = gaussian_mutation(individual, mutation_rate=rate, sigma=0.5, bounds=bounds)
        print(f"Rate {rate}: {mutant}")