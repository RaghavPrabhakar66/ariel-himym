# Controller Saving and Loading System

## Overview
The spatial EA now automatically saves all final controllers (genotypes) from each evolution run in multiple formats for easy analysis, reuse, and comparison.

## Saved File Formats

### 1. JSON Format (Human-Readable + Metadata)
**Filename**: `final_controllers_TIMESTAMP.json`

**Contents**:
- Run metadata (timestamp, generation, configuration)
- Complete population data sorted by fitness (best first)
- Per-individual data:
  - Unique ID
  - Generation born
  - Age
  - Fitness
  - Energy (if enabled)
  - Full genotype array
  - Parent IDs
  - Spawn position

**Best for**: 
- Human inspection
- Sharing with collaborators
- Importing into analysis tools
- Understanding evolutionary lineage

**Example**:
```json
{
  "timestamp": "20251109_230141",
  "generation": 5,
  "num_joints": 8,
  "population_size": 10,
  "config": {
    "selection_method": "energy_based",
    "enable_energy": true,
    "mating_energy_effect": "restore"
  },
  "controllers": [
    {
      "unique_id": 5,
      "generation_born": 0,
      "age": 5,
      "fitness": 0.401024,
      "energy": 50.0,
      "genotype": [0.831532, 3.188238, ...],
      "parent_ids": [],
      "position": [12.5, 15.3, 0.05]
    },
    ...
  ]
}
```

### 2. NPZ Format (Fast NumPy Arrays)
**Filename**: `final_genotypes_TIMESTAMP.npz`

**Contents**:
- `genotypes`: 2D array (population_size × genotype_length)
- `fitness`: 1D array of fitness values
- `ages`: 1D array of individual ages
- `ids`: 1D array of unique IDs
- `energy`: 1D array of energy levels (if enabled)
- `num_joints`: Integer metadata
- `generation`: Integer metadata

**Best for**:
- Fast loading in Python/NumPy
- Numerical analysis
- Machine learning pipelines
- Statistical comparisons

**Example Usage**:
```python
data = np.load('final_genotypes_20251109_230141.npz')
genotypes = data['genotypes']  # (10, 24) array
fitness = data['fitness']       # (10,) array
```

### 3. TXT Format (Best Controller Only)
**Filename**: `best_controller_TIMESTAMP.txt`

**Contents**:
- Best individual metadata (ID, age, fitness, energy)
- Formatted genotype by joint:
  - Amplitude
  - Frequency  
  - Phase
- Raw genotype array

**Best for**:
- Quick inspection
- Documentation
- Testing single best controller

**Example**:
```
============================================================
BEST CONTROLLER
============================================================

Individual ID: 5
Born in Generation: 0
Age: 5 generations
Fitness: 0.401024
Energy: 50.00
Parent IDs: []

Genotype (24 values for 8 joints):
------------------------------------------------------------
Joint 0:
  Amplitude: 0.831532
  Frequency: 3.188238
  Phase:     0.880633
...
```

## Automatic Saving

Controllers are **automatically saved** at the end of each evolution run:

```python
spatial_ea = SpatialEA(...)
spatial_ea.run_evolution()  # Controllers saved automatically
```

**Console Output**:
```
Saving evolution data...
  Data saved to CSV: __results__/evolution_data_TIMESTAMP.csv
  Data saved to NPZ: __results__/evolution_data_TIMESTAMP.npz
  Evolution statistics plot saved: __figures__/evolution_statistics_TIMESTAMP.png
  Final controllers saved to JSON: __results__/final_controllers_TIMESTAMP.json
  Final genotypes saved to NPZ: __results__/final_genotypes_TIMESTAMP.npz
  Best controller saved to TXT: __results__/best_controller_TIMESTAMP.txt

  Saved 10 controllers from final population
  Best fitness: 0.401024 (ID: 5)
```

## Loading Saved Controllers

### Method 1: Load from JSON
```python
from main_spatial_ea import SpatialEA

# Load all controllers with metadata
data = SpatialEA.load_controllers_from_json('__results__/final_controllers_TIMESTAMP.json')

# Access data
timestamp = data['timestamp']
num_joints = data['num_joints']
config = data['config']
controllers = data['controllers']  # Sorted by fitness (best first)

# Get best controller
best = controllers[0]
best_genotype = best['genotype']
best_fitness = best['fitness']
best_energy = best['energy']
```

### Method 2: Load from NPZ
```python
from main_spatial_ea import SpatialEA
import numpy as np

# Load as numpy arrays
data = SpatialEA.load_genotypes_from_npz('__results__/final_genotypes_TIMESTAMP.npz')

# Access arrays
genotypes = data['genotypes']  # Shape: (population_size, genotype_length)
fitness = data['fitness']       # Shape: (population_size,)
ages = data['ages']             # Shape: (population_size,)
ids = data['ids']               # Shape: (population_size,)
energy = data['energy']         # Shape: (population_size,) or None

# Get best controller
best_idx = fitness.argmax()
best_genotype = genotypes[best_idx]
best_fitness = fitness[best_idx]
```

### Method 3: Read TXT File
```python
# Simple text reading
with open('__results__/best_controller_TIMESTAMP.txt', 'r') as f:
    content = f.read()
    print(content)
```

## Use Cases

### 1. Continue Evolution from Saved Population
```python
# Load previous population
data = SpatialEA.load_controllers_from_json('previous_run.json')

# Create individuals from saved genotypes
population = []
for controller in data['controllers']:
    ind = SpatialIndividual(
        unique_id=next_id,
        generation=current_gen,
        initial_energy=config.initial_energy
    )
    ind.genotype = controller['genotype']
    ind.fitness = controller['fitness']
    population.append(ind)
    next_id += 1

# Continue evolution with loaded population
```

### 2. Test Specific Controllers
```python
# Load best controller
data = SpatialEA.load_controllers_from_json('run.json')
best_genotype = data['controllers'][0]['genotype']

# Test in simulation
# (Use existing evaluation code from main_spatial_ea.py)
```

### 3. Compare Multiple Runs
```python
import glob

results = []
for json_file in glob.glob('__results__/final_controllers_*.json'):
    data = SpatialEA.load_controllers_from_json(json_file)
    results.append({
        'timestamp': data['timestamp'],
        'best_fitness': data['controllers'][0]['fitness'],
        'avg_fitness': np.mean([c['fitness'] for c in data['controllers']]),
        'selection_method': data['config']['selection_method'],
    })

# Analyze across runs
import pandas as pd
df = pd.DataFrame(results)
print(df)
```

### 4. Analyze Parameter Distributions
```python
# Load genotypes
data = SpatialEA.load_genotypes_from_npz('run.npz')
genotypes = data['genotypes']
num_joints = data['num_joints']

# Analyze per-joint parameters
for joint in range(num_joints):
    amp_idx = joint * 3
    freq_idx = joint * 3 + 1
    phase_idx = joint * 3 + 2
    
    print(f"Joint {joint}:")
    print(f"  Amplitude: {genotypes[:, amp_idx].mean():.3f} ± {genotypes[:, amp_idx].std():.3f}")
    print(f"  Frequency: {genotypes[:, freq_idx].mean():.3f} ± {genotypes[:, freq_idx].std():.3f}")
    print(f"  Phase: {genotypes[:, phase_idx].mean():.3f} ± {genotypes[:, phase_idx].std():.3f}")
```

### 5. Extract Evolutionary Lineages
```python
# Load controllers
data = SpatialEA.load_controllers_from_json('run.json')

# Build lineage tree
lineages = {}
for controller in data['controllers']:
    ind_id = controller['unique_id']
    parent_ids = controller['parent_ids']
    lineages[ind_id] = {
        'parents': parent_ids,
        'fitness': controller['fitness'],
        'age': controller['age']
    }

# Trace best individual's lineage
best_id = data['controllers'][0]['unique_id']
print(f"Best individual {best_id} lineage:")
# (Trace through parent_ids to reconstruct ancestry)
```

## Example Script

Run `load_controller_example.py` to see all loading methods in action:

```bash
cd himym/spatial_ea
python load_controller_example.py
```

**Output includes**:
- JSON loading demonstration
- NPZ array loading
- TXT file display
- Genotype diversity analysis
- Fitness statistics

## File Organization

All controller files are saved in the `__results__/` folder with timestamps:

```
__results__/
├── evolution_data_20251109_230141.csv
├── evolution_data_20251109_230141.npz
├── final_controllers_20251109_230141.json
├── final_genotypes_20251109_230141.npz
└── best_controller_20251109_230141.txt
```

Timestamps ensure no overwriting between runs, making it easy to:
- Compare multiple experiments
- Track evolution over many runs
- Archive results systematically

## Summary

**Automatic saving**: ✅ Every evolution run saves controllers  
**Multiple formats**: ✅ JSON, NPZ, TXT for different use cases  
**Easy loading**: ✅ Built-in static methods for loading  
**Metadata included**: ✅ Run configuration and individual details  
**Example code**: ✅ Complete loading examples provided  
**Timestamp naming**: ✅ No file overwrites, easy organization
