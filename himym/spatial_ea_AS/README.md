# Spatial EA Documentation

Welcome to the Spatial EA documentation! This project implements a spatial evolutionary algorithm where robots evolve through physical proximity-based mating.

## Quick Navigation

### 📚 Core Usage Guides

1. **[Spatial EA Usage Guide](SPATIAL_EA_USAGE.md)**
   - Running the core evolutionary algorithm
   - Configuration options and parameters
   - Pairing methods and selection strategies
   - Energy systems and incubation periods
   - Output files and data formats

2. **[Visualization Usage Guide](VISUALIZATION_USAGE.md)**
   - Comprehensive 12-panel analysis plots
   - Individual metric visualizations
   - Publication-ready minimal plots
   - Programmatic data access
   - Custom analysis examples

3. **[Experiment Runner Usage Guide](EXPERIMENT_RUNNER_USAGE.md)**
   - Running batch experiments
   - Parallel execution for speed
   - Statistical aggregation across runs
   - Experiment comparison tools
   - Parameter sweeps and reproducibility

## Getting Started

### Installation

This project uses [uv](https://docs.astral.sh/uv/):

```bash
# Install dependencies
uv venv
uv sync

# Run a basic example
cd himym/spatial_ea
uv run main_spatial_ea.py
```

### Quick Start Workflow

1. **Run a single evolution**:
   ```bash
   cd himym/spatial_ea
   uv run main_spatial_ea.py
   ```

2. **Visualize results**:
   ```bash
   python visualize_experiment.py
   ```

3. **Run systematic experiments**:
   ```bash
   python run_experiments.py --experiment baseline_random_fitnessBased --parallel
   ```

## Project Structure

```
himym/spatial_ea/
├── main_spatial_ea.py              # Core EA implementation
├── ea_config.yaml                  # Main configuration file
├── run_experiments.py              # Batch experiment runner
├── experiment_runner.py            # Experiment management
├── visualize_experiment.py         # Visualization tools
├── visualization.py                # Plotting functions
├── evolution_data_collector.py     # Data collection
└── ...

__results__/                        # Evolution outputs
├── evolution_data_*.csv            # Time series data
├── evolution_data_*.npz            # NumPy format
├── final_controllers_*.json        # Final population
└── analysis/                       # Visualizations

__experiments__/                    # Batch experiment outputs
└── experiment_name_timestamp/
    ├── aggregated_statistics.csv   # Mean/std across runs
    ├── aggregated_results.png      # Visualization
    └── run_000/, run_001/, ...     # Individual runs

__figures__/                        # Trajectory plots
__videos__/                         # Simulation recordings
```

## Documentation Overview

### For First-Time Users

Start with **[Spatial EA Usage Guide](SPATIAL_EA_USAGE.md)**:
- Understand core concepts
- Learn configuration options
- Run your first evolution
- Understand output files

Then explore **[Visualization Usage Guide](VISUALIZATION_USAGE.md)**:
- Generate comprehensive analysis plots
- Access evolution data programmatically
- Create custom visualizations

### For Researchers

Use **[Experiment Runner Usage Guide](EXPERIMENT_RUNNER_USAGE.md)**:
- Run systematic comparisons
- Aggregate statistics across multiple runs
- Perform parameter sweeps
- Generate publication figures

### For Developers

Check the source code documentation:
- `main_spatial_ea.py` - Core EA logic
- `experiment_runner.py` - Batch processing
- `visualize_experiment.py` - Plotting tools
- `ea_config.yaml` - Configuration reference

## Key Features

### Spatial Evolutionary Algorithm
- Physical proximity-based mating
- Multiple pairing methods (random, proximity, zones)
- Movement biases (nearest neighbor, assigned zones)
- Selection strategies (fitness, age, energy-based)
- Incubation periods for bootstrapping

### Experiment Management
- Parallel execution for speed (with proper random seeds!)
- Statistical aggregation across runs
- Early stopping handling (extinction/explosion)
- Comprehensive output structure
- Reproducible experiments

### Visualization & Analysis
- 12-panel comprehensive overviews
- Individual metric plots
- Publication-ready minimal plots
- Parameter distribution analysis
- Correlation heatmaps
- Programmatic data access

## Common Workflows

### 1. Basic Experiment
```bash
# Configure
nano himym/spatial_ea/ea_config.yaml

# Run
cd himym/spatial_ea
uv run main_spatial_ea.py

# Visualize
python visualize_experiment.py
```

### 2. Comparative Study
```bash
# Run multiple configurations
python run_experiments.py --experiment baseline_random_fitnessBased --parallel
python run_experiments.py --experiment baseline_proximity_fitnessBased --parallel

# Compare
python run_experiments.py --compare baseline_random_fitnessBased baseline_proximity_fitnessBased
```

### 3. Parameter Sweep
```python
from experiment_runner import ExperimentRunner, ExperimentConfig

runner = ExperimentRunner()

for mutation_rate in [0.1, 0.3, 0.5, 0.7, 0.9]:
    config = ExperimentConfig(
        experiment_name=f"mutation_{mutation_rate:.1f}",
        num_runs=10,
        mutation_rate=mutation_rate,
        population_size=30,
        num_generations=50
    )
    runner.run_experiment_parallel(config, num_workers=4)

# Compare all
runner.compare_experiments(
    [f"mutation_{r:.1f}" for r in [0.1, 0.3, 0.5, 0.7, 0.9]]
)
```

## Output Files Reference

### Evolution Outputs (`__results__/`)
- `evolution_data_TIMESTAMP.csv` - All generation statistics
- `evolution_data_TIMESTAMP.npz` - NumPy arrays
- `final_controllers_TIMESTAMP.json` - Final population with metadata
- `final_genotypes_TIMESTAMP.npz` - Genotype arrays
- `best_controller_TIMESTAMP.txt` - Best individual (readable)
- `evolution_statistics_TIMESTAMP.png` - Basic 5-panel plot

### Experiment Outputs (`__experiments__/`)
- `aggregated_statistics.csv` - Mean/std across runs
- `aggregated_results.png` - 4-panel visualization
- `summary.json` - High-level statistics
- `run_XXX/` - Individual run directories

### Visualization Outputs (`__results__/analysis/`)
- `{exp}_comprehensive.png` - 12-panel overview
- `{exp}_parameters.png` - Controller distributions
- `{exp}_heatmap.png` - Parameter correlations
- `{exp}_summary.txt` - Text statistics
- `individual/` - Separate metric plots (optional)
- `minimal/` - Publication plots (optional)

## Available Metrics

**Population**: size, births, deaths  
**Fitness**: best, average, worst, std  
**Mating**: pairs, unpaired, success rate  
**Diversity**: genotype diversity  
**Age**: min, max, avg, std (if applicable)  
**Energy**: min, max, avg, std, depleted count (if enabled)

## Troubleshooting

### Population Goes Extinct
- Increase initial population
- Enable incubation period  
- Reduce energy depletion
- Use less aggressive selection

### Slow Performance
- Use parallel execution: `--parallel`
- Disable videos/snapshots
- Reduce simulation time
- Smaller populations

### Identical Results Across Runs
- **Fixed!** Recent update ensures unique seeds
- Don't set `random_seed` unless needed
- Verify `fitness_best_std > 0` in output

See individual guides for detailed troubleshooting.

## Tips and Best Practices

1. **Start simple** - Begin with default config
2. **Visualize everything** - Always check plots
3. **Use parallel execution** - Dramatically faster
4. **Save configurations** - Document your experiments
5. **Check aggregated stats** - Before individual runs
6. **Enable logging** - Keep snapshots during development
7. **Run multiple trials** - Statistics need variance
8. **Compare systematically** - Use experiment runner

## Recent Updates

### Parallel Random Seed Fix (Nov 2025)
Fixed issue where parallel runs produced identical results. Each run now gets a unique time-based seed, ensuring proper variance in aggregated statistics.

See `PARALLEL_RANDOM_SEED_FIX.md` for details.

## Contributing

When adding features:
1. Update relevant usage guide
2. Add examples to example scripts
3. Update this README if structure changes
4. Test with both single and parallel execution

## Support

For issues or questions:
1. Check the relevant usage guide
2. Review example scripts
3. Check error messages in console
4. Verify configuration syntax
5. Review output files for diagnostics

## License

This project is part of the ARIEL framework. See LICENSE for details.

---

**Quick Links:**
- [Spatial EA Usage](SPATIAL_EA_USAGE.md) - Core algorithm
- [Visualization Usage](VISUALIZATION_USAGE.md) - Analysis tools  
- [Experiment Runner Usage](EXPERIMENT_RUNNER_USAGE.md) - Batch experiments
- [Main Project README](../README.md) - ARIEL framework

**Last Updated**: November 2025
