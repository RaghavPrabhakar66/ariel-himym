# Enhanced Visualization Features - Documentation

## Overview

The visualization system now supports three levels of output detail:

1. **Comprehensive** (default) - Single 12-panel overview plot
2. **Individual** - Separate file for each metric (11 plots)
3. **Minimal** - Publication-ready plots for double-column papers (6 plots)

## Quick Reference

### Command Line Usage

```bash
# Basic: Comprehensive plot only (default)
python visualize_experiment.py

# Individual plots: One file per metric
python visualize_experiment.py --individual

# Minimal plots: Publication-ready format
python visualize_experiment.py --minimal

# All plot types: Comprehensive + Individual + Minimal
python visualize_experiment.py --all

# Specific file with all plots
python visualize_experiment.py --file evolution_data_20251109_230140.csv --all

# Custom output directory
python visualize_experiment.py --all --output /path/to/output/
```

### Programmatic Usage

```python
from visualize_experiment import ExperimentVisualizer

viz = ExperimentVisualizer('path/to/data.csv')

# Generate comprehensive only (default)
viz.generate_report()

# Generate with individual plots
viz.generate_report(individual_plots=True)

# Generate with minimal plots
viz.generate_report(minimal_plots=True)

# Generate all types
viz.generate_report(individual_plots=True, minimal_plots=True)

# Custom output directory
viz.generate_report(output_dir='/custom/path/', individual_plots=True, minimal_plots=True)
```

## Plot Types Explained

### 1. Comprehensive Plot (Always Generated)

**File**: `{experiment}_comprehensive.png`  
**Size**: ~391 KB  
**Format**: 4×3 grid (12 panels)  
**Purpose**: Complete overview of all metrics in one image

**Panels**:
- Row 1: Population dynamics, births/deaths, mating success
- Row 2: Fitness evolution, fitness distribution, genotype diversity
- Row 3: Energy evolution, age distribution, energy depletion
- Row 4: Fitness vs age, fitness vs energy, summary statistics

**Best for**: Initial analysis, presentations, comprehensive documentation

### 2. Individual Plots (Optional: `--individual`)

**Directory**: `analysis/individual/`  
**Count**: 11 separate files  
**Size**: ~40-70 KB each  
**Format**: Single metric per file (8×6 inches)

**Files Generated**:
1. `{experiment}_population_dynamics.png` - Population size over time
2. `{experiment}_births_deaths.png` - Births and deaths per generation
3. `{experiment}_mating_success.png` - Mating pairs and success rate
4. `{experiment}_fitness_evolution.png` - Best/avg/worst fitness
5. `{experiment}_fitness_distribution.png` - Fitness diversity (CV)
6. `{experiment}_genotype_diversity.png` - Genotype diversity
7. `{experiment}_energy_evolution.png` - Energy levels
8. `{experiment}_age_distribution.png` - Age statistics
9. `{experiment}_energy_depletion.png` - Depleted individuals
10. `{experiment}_fitness_vs_age.png` - Correlation scatter
11. `{experiment}_fitness_vs_energy.png` - Correlation scatter

**Best for**: 
- Detailed analysis of specific metrics
- Including individual charts in documents
- Sharing specific results

### 3. Minimal Publication Plots (Optional: `--minimal`)

**Directory**: `analysis/minimal/`  
**Count**: 6 files  
**Size**: ~30-80 KB each  
**Format**: Double-column paper format (3.5×2.5 inches)  
**DPI**: 300 (publication quality)

**Design Features**:
- Clean, minimal styling
- Appropriate font sizes (8-11pt)
- No chartjunk
- Suitable for academic papers
- High-resolution (300 DPI)
- Optimized for grayscale printing

**Files Generated**:
1. `{experiment}_minimal_fitness.png`
   - Best and mean fitness with confidence bands
   - Most important metric
   
2. `{experiment}_minimal_population.png`
   - Population size over time
   - Simple line plot
   
3. `{experiment}_minimal_age_energy.png`
   - Two stacked subplots (if both available)
   - Age distribution (top) + Energy levels (bottom)
   - Shared x-axis for comparison
   
4. `{experiment}_minimal_mating.png`
   - Mating pairs per generation
   - Bar chart
   
5. `{experiment}_minimal_diversity.png`
   - Genotype diversity over time
   - Line plot with markers
   
6. `{experiment}_minimal_final_pop.png`
   - Fitness vs Age scatter plot
   - Final population analysis
   - Colored by fitness value

**Best for**:
- Academic papers (double-column format)
- Conference submissions
- Journal articles
- Publication-quality figures

## Detailed Specifications

### Minimal Plot Design Specifications

**Figure Dimensions**:
- Width: 3.5 inches (fits double-column papers)
- Height: 2.5 inches (standard aspect ratio)
- Combined plots: 3.5×3.75 inches

**Typography**:
- Font size: 9pt (body), 10pt (labels), 11pt (title)
- Tick labels: 8pt
- Legend: 8pt
- Font family: Matplotlib default (DejaVu Sans)

**Visual Style**:
- Line width: 1.5pt
- Marker size: 4pt (when used)
- Axes line width: 0.8pt
- Grid line width: 0.5pt
- Grid style: Dashed, 30% opacity
- Removed elements: Top spine, right spine

**Colors**:
- Fitness best: Green
- Fitness mean: Blue
- Population: Blue with fill
- Age: Purple
- Energy: Orange
- Mating: Purple/Dark violet
- Diversity: Teal

**Resolution**:
- DPI: 300 (publication standard)
- Format: PNG with tight bounding box

## Usage Examples

### Example 1: Generate All Plot Types

```bash
cd himym/spatial_ea
python visualize_experiment.py --all
```

**Output**:
```
__results__/analysis/
├── evolution_data_TIMESTAMP_comprehensive.png  (391 KB)
├── evolution_data_TIMESTAMP_parameters.png     (155 KB)
├── evolution_data_TIMESTAMP_heatmap.png        (96 KB)
├── evolution_data_TIMESTAMP_summary.txt        (1.5 KB)
├── individual/
│   ├── evolution_data_TIMESTAMP_population_dynamics.png
│   ├── evolution_data_TIMESTAMP_births_deaths.png
│   ├── evolution_data_TIMESTAMP_fitness_evolution.png
│   ├── ... (11 files total)
└── minimal/
    ├── evolution_data_TIMESTAMP_minimal_fitness.png
    ├── evolution_data_TIMESTAMP_minimal_population.png
    ├── evolution_data_TIMESTAMP_minimal_age_energy.png
    ├── ... (6 files total)

Total: 22 plot files
```

### Example 2: Publication Workflow

```python
from visualize_experiment import ExperimentVisualizer

# Load your experiment
viz = ExperimentVisualizer('results/experiment_run_042.csv')

# Generate publication plots only
viz.generate_report(
    output_dir='paper_figures/',
    individual_plots=False,
    minimal_plots=True
)

# Now use the files in paper_figures/minimal/ in your LaTeX document
```

**LaTeX Usage**:
```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\columnwidth]{paper_figures/minimal/experiment_run_042_minimal_fitness.png}
    \caption{Fitness evolution over 50 generations showing convergence to optimal solution.}
    \label{fig:fitness}
\end{figure}
```

### Example 3: Compare Multiple Experiments

```python
from visualize_experiment import ExperimentVisualizer
from pathlib import Path

experiments = [
    'evolution_data_run1.csv',
    'evolution_data_run2.csv', 
    'evolution_data_run3.csv'
]

for exp in experiments:
    viz = ExperimentVisualizer(exp)
    
    # Create separate directories for each
    output_dir = Path(f'comparison/{exp.stem}')
    
    viz.generate_report(
        output_dir=output_dir,
        individual_plots=True,
        minimal_plots=True
    )
    
print("All experiments visualized!")
```

### Example 4: Automated Pipeline

```bash
#!/bin/bash
# run_and_visualize.sh

# Run experiment
python main_spatial_ea.py --config my_config.yaml

# Automatically generate all visualizations
python visualize_experiment.py --all

# Copy minimal plots to paper directory
cp __results__/analysis/minimal/*.png ~/paper/figures/

echo "Experiment complete and figures ready for paper!"
```

## Best Practices

### For Analysis and Exploration

Use **individual plots** when you need:
- Detailed examination of specific metrics
- To include single metrics in reports
- To compare the same metric across multiple runs
- High-resolution versions of specific plots

```bash
python visualize_experiment.py --individual
```

### For Publications

Use **minimal plots** when you need:
- Figures for journal articles
- Conference paper submissions
- Professional presentations
- Print-quality graphics

```bash
python visualize_experiment.py --minimal
```

**Publication Checklist**:
✅ 300 DPI resolution  
✅ Proper figure sizing (3.5" width for double-column)  
✅ Readable fonts (8-11pt)  
✅ Minimal visual clutter  
✅ High contrast for grayscale printing  
✅ Clear axis labels and legends  

### For Presentations

Use the **comprehensive plot** for:
- Quick overview slides
- Initial data exploration
- Comparing many metrics at once
- Documentation

```bash
python visualize_experiment.py  # Default: comprehensive only
```

### For Complete Documentation

Generate **all plot types** to have every option available:

```bash
python visualize_experiment.py --all
```

## File Size Comparison

| Plot Type | Files | Total Size | Avg per File |
|-----------|-------|------------|--------------|
| Comprehensive | 1 | ~391 KB | 391 KB |
| Individual | 11 | ~600 KB | 55 KB |
| Minimal | 6 | ~340 KB | 57 KB |
| Parameters | 1 | ~155 KB | 155 KB |
| Heatmap | 1 | ~96 KB | 96 KB |
| **Total (all)** | **20** | **~1.6 MB** | **80 KB** |

## Customization

### Modify Minimal Plot Styling

Edit the `generate_minimal_plots()` method in `visualize_experiment.py`:

```python
plt.rcParams.update({
    'font.size': 9,           # Change base font size
    'axes.labelsize': 10,     # Change axis label size
    'figure.titlesize': 11,   # Change title size
    'lines.linewidth': 1.5,   # Change line thickness
    # ... more parameters
})
```

### Adjust Figure Dimensions

```python
# For single-column papers (wider)
fig_width = 7.0  # Full text width
fig_height = 4.0

# For presentations (larger)
fig_width = 5.0
fig_height = 3.5

# Pass to individual plot methods
self._create_minimal_fitness_plot(output_dir, fig_width, fig_height)
```

### Change Color Scheme

Modify color choices in individual plot methods:

```python
# In _create_minimal_fitness_plot()
ax.plot(generations, self.df['fitness_best'], 
       color='#2E7D32',  # Dark green
       label='Best', linewidth=2)
```

## Integration with Existing Workflow

The enhanced visualization system integrates seamlessly:

### Before (Old Workflow)

```python
# Run experiment
spatial_ea.run_evolution()

# Manually check __figures__/ directory
# Limited to basic 5-panel plot
```

### After (New Workflow)

```python
# Run experiment
spatial_ea.run_evolution()

# Generate comprehensive analysis
from visualize_experiment import ExperimentVisualizer
viz = ExperimentVisualizer('latest_results.csv')
viz.generate_report(individual_plots=True, minimal_plots=True)

# Use plots immediately:
# - Individual plots for detailed analysis
# - Minimal plots ready for paper submission
# - Comprehensive plot for presentations
```

## Troubleshooting

**Q: Plots look too small/large**

A: Adjust figure dimensions in the code:
```python
viz.generate_report()
# Then edit minimal plot dimensions in visualize_experiment.py
```

**Q: Fonts not readable in publication**

A: Minimal plots use 300 DPI and appropriate font sizes. If still needed:
```python
# Increase DPI
plt.savefig(path, dpi=600)  # Even higher resolution
```

**Q: Need different aspect ratio**

A: Modify height in `_create_minimal_*_plot()` methods:
```python
fig_height = 3.0  # Taller
# or
fig_height = 2.0  # Shorter
```

**Q: Want different colors for publication**

A: Edit color choices in minimal plot methods to match journal requirements

## Summary

| Feature | Command | Output Files | Best For |
|---------|---------|--------------|----------|
| Comprehensive | `python visualize_experiment.py` | 1 plot | Quick overview |
| Individual | `--individual` or `-i` | 11 plots | Detailed analysis |
| Minimal | `--minimal` or `-m` | 6 plots | Publications |
| All | `--all` or `-a` | 18 plots | Complete documentation |

The visualization system now provides publication-ready plots optimized for double-column academic papers while maintaining all existing functionality for analysis and exploration!
