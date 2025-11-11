# Experiment Visualization System - Quick Reference

## What You Get

A complete visualization and analysis toolkit for your spatial EA experiments with:

- **12-panel comprehensive overview** showing all evolution metrics
- **Controller parameter analysis** with distributions and correlations  
- **Automatic data discovery** from CSV, NPZ, and JSON files
- **Summary statistics** in human-readable text format
- **Programmatic data access** for custom analysis

## Quick Start

### 1. Visualize Latest Experiment

```bash
cd himym/spatial_ea
python visualize_experiment.py
```

**Output files** (saved to `__results__/analysis/`):
- `{experiment}_comprehensive.png` - 12-panel overview (391 KB)
- `{experiment}_parameters.png` - Controller parameters (155 KB)
- `{experiment}_heatmap.png` - Parameter correlations (96 KB)
- `{experiment}_summary.txt` - Text statistics (1.5 KB)

### 2. Run Interactive Examples

```bash
cd himym/spatial_ea
python visualize_examples.py
```

Choose from:
1. **Basic Usage** - Generate complete report
2. **Custom Plots** - Individual visualizations
3. **Data Exploration** - Programmatic data access
4. **Compare Experiments** - Multi-run comparison
5. **Energy Analysis** - Energy system dynamics

### 3. Use in Your Scripts

```python
from visualize_experiment import ExperimentVisualizer

# Load any data file (CSV, NPZ, or JSON)
viz = ExperimentVisualizer('__results__/evolution_data_TIMESTAMP.csv')

# Generate all visualizations
viz.generate_report()

# Or create individual plots
viz.plot_all(save_path='overview.png')
viz.plot_controller_parameters(save_path='params.png')
viz.plot_parameter_evolution_heatmap(save_path='heatmap.png')

# Access data directly
df = viz.df  # Pandas DataFrame
controllers = viz.controllers  # Final population dict
```

## What's Visualized

### Comprehensive Overview (12 panels)

**Population Dynamics:**
- Population size over time
- Births and deaths per generation  
- Mating success rates and pairs formed

**Fitness Evolution:**
- Best/average/worst fitness with confidence bands
- Fitness diversity (coefficient of variation)
- Genotype diversity (standard deviation)

**Energy and Age:**
- Energy levels (min/max/avg) over time
- Age distribution (youngest/oldest/avg)
- Energy-depleted individuals count

**Advanced Analysis:**
- Fitness vs Age scatter plot
- Fitness vs Energy scatter plot  
- Summary statistics panel

### Controller Parameters

For each joint, shows:
- Amplitude, frequency, phase distributions
- Mean ± standard deviation bars
- Population-wide parameter statistics

### Parameter Heatmap

Two visualizations:
- **Genotype heatmap**: All parameter values sorted by fitness
- **Correlation matrix**: Which parameters co-evolve

## Data Access

All experiment data is available programmatically:

```python
viz = ExperimentVisualizer('path/to/data.csv')

# Time series data (Pandas DataFrame)
generations = viz.df['generation']
fitness_best = viz.df['fitness_best']
energy_avg = viz.df['energy_avg']
mating_pairs = viz.df['mating_pairs']

# Final population (from JSON)
best_controller = viz.controllers['controllers'][0]
best_genotype = best_controller['genotype']
best_fitness = best_controller['fitness']

# Numpy arrays (from NPZ)
all_data = viz.npz_data
fitness_array = all_data['fitness_best']
```

## Common Use Cases

### Analyze Evolution Progress

```python
viz = ExperimentVisualizer('results.csv')

# Check improvement
initial = viz.df['fitness_best'].iloc[0]
final = viz.df['fitness_best'].iloc[-1]
improvement = final - initial

# Detect convergence
window = len(viz.df) // 5
recent_change = viz.df['fitness_best'].iloc[-window:].max() - \
                viz.df['fitness_best'].iloc[-window:].min()

if recent_change < 0.001:
    print("Converged!")
```

### Compare Selection Methods

```python
results = {}
for method in ['fitness_based', 'age_based', 'energy_based']:
    viz = ExperimentVisualizer(f'results_{method}.csv')
    results[method] = viz.df['fitness_best'].max()

best_method = max(results, key=results.get)
print(f"Best selection method: {best_method}")
```

### Extract Best Controllers

```python
viz = ExperimentVisualizer('results.csv')

# Get top 5 controllers
top_5 = viz.controllers['controllers'][:5]

for i, controller in enumerate(top_5, 1):
    print(f"Rank {i}: Fitness {controller['fitness']:.6f}")
    print(f"  Age: {controller['age']} gens")
    print(f"  Energy: {controller['energy']:.2f}")
```

### Batch Process Multiple Runs

```python
from pathlib import Path

results_dir = Path('__results__')
for csv_file in results_dir.glob('evolution_data_*.csv'):
    viz = ExperimentVisualizer(csv_file)
    
    # Custom analysis per experiment
    analysis_dir = results_dir / f"analysis_{csv_file.stem}"
    viz.generate_report(output_dir=analysis_dir)
    
    print(f"Processed: {csv_file.name}")
```

## Files Created by Main EA

When you run `main_spatial_ea.py`, it automatically saves:

### Evolution Statistics
- `evolution_data_TIMESTAMP.csv` - Time series data (all generations)
- `evolution_data_TIMESTAMP.npz` - NumPy arrays (fast loading)
- `evolution_statistics_TIMESTAMP.png` - Basic 5-panel plot

### Final Controllers  
- `final_controllers_TIMESTAMP.json` - All individuals with metadata
- `final_genotypes_TIMESTAMP.npz` - Genotype arrays only
- `best_controller_TIMESTAMP.txt` - Best individual readable format

### Analysis Visualizations (when you run visualizer)
- `{experiment}_comprehensive.png` - 12-panel detailed analysis
- `{experiment}_parameters.png` - Joint parameter distributions
- `{experiment}_heatmap.png` - Parameter correlations
- `{experiment}_summary.txt` - Statistics summary

## Available Metrics

From `evolution_data_*.csv`:

**Population:**
- `generation`, `population_size`, `births`, `deaths`

**Fitness:**
- `fitness_best`, `fitness_avg`, `fitness_worst`, `fitness_std`

**Age:**
- `age_min`, `age_max`, `age_avg`, `age_std`

**Mating:**
- `mating_pairs`, `unpaired`, `mating_success_rate`

**Diversity:**
- `genotype_diversity`

**Energy** (if enabled):
- `energy_min`, `energy_max`, `energy_avg`, `energy_std`
- `energy_depleted_count`

## Tips

1. **Always visualize after experiments** - Add to your workflow:
   ```python
   # At end of experiment script
   from visualize_experiment import ExperimentVisualizer
   viz = ExperimentVisualizer('path/to/results.csv')
   viz.generate_report()
   ```

2. **Use examples as templates** - Copy code from `visualize_examples.py`

3. **Check summary.txt first** - Quick overview before viewing plots

4. **Compare across runs** - Use Example 4 to identify best configurations

5. **Interactive exploration** - Don't specify save_path to display plots:
   ```python
   viz.plot_all(save_path=None)  # Opens interactive window
   ```

6. **Custom analysis** - Access `viz.df` directly for pandas operations

## Troubleshooting

**Q: No plots generated?**
- Check that CSV files exist in `__results__/`
- Run `main_spatial_ea.py` first to generate data

**Q: Missing controller plots?**
- Ensure controller JSON files exist
- They should be created automatically with matching timestamp

**Q: NaN values in energy plots?**
- Energy system may be disabled (`enable_energy: false` in config)
- This is expected - only available data is plotted

**Q: All values the same?**
- Short runs or no reproduction may show flat lines
- Run for more generations to see dynamics

## Documentation

- **Full guide**: `/home/ariel-himym/VISUALIZATION_GUIDE.md`
- **Controller saving**: `/home/ariel-himym/CONTROLLER_SAVING_GUIDE.md`
- **Examples**: `himym/spatial_ea/visualize_examples.py`
- **Module**: `himym/spatial_ea/visualize_experiment.py`

## Summary

Three ways to visualize:

1. **Command line**: `python visualize_experiment.py` (auto-finds latest)
2. **Interactive**: `python visualize_examples.py` (guided examples)  
3. **Programmatic**: Import `ExperimentVisualizer` in your code

All methods produce the same high-quality visualizations and analysis!

---

**Created by**: Experiment Visualization System
**Version**: 1.0
**Last Updated**: November 2025
