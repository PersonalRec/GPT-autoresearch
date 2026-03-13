---
skill: analyze-optimization
description: Analyze and visualize optimization results with plots and reports
tags: [optimization, analysis, visualization, plots]
---

# Optimization Analysis

Comprehensive analysis and visualization of optimization results from any algorithm.

## Installation

```bash
# Required
pip install optuna pandas numpy

# For plots (highly recommended)
pip install matplotlib
```

## Quick Start

```python
from optuna_algorithms import OptimizationAnalyzer

# Load results
analyzer = OptimizationAnalyzer('bayesian_output')

# Print summary
analyzer.print_summary()

# Generate all plots
analyzer.plot_all(output_dir='plots', show=False)

# Export results
analyzer.export_best_config('best_config.json')
analyzer.to_markdown_report('report.md')
```

## Command Line Usage

```bash
# Analyze single run
python .claude/skills/analyze-optimization.py bayesian_output

# Compare multiple runs
python .claude/skills/analyze-optimization.py --compare bayesian_output cmaes_output nsga_output

# Custom plot directory
python .claude/skills/analyze-optimization.py bayesian_output --plot_dir my_plots

# Skip exports
python .claude/skills/analyze-optimization.py bayesian_output --no-export

# Get help
python .claude/skills/analyze-optimization.py --help
```

## Analysis Features

### 1. Summary Statistics

```python
analyzer.print_summary()
```

Prints:
- Trial counts (total, completed, failed, success rate)
- Best value and parameters
- Time statistics (total, per trial)
- Value distribution (min, max, mean, median, std)
- Parameter exploration ranges

### 2. Convergence Plot

```python
analyzer.plot_convergence('convergence.png')
```

Shows:
- Best value found over trials
- When optimization converged
- Whether more trials would help

### 3. Parameter Importance

```python
analyzer.plot_parameter_importances('importance.png')
```

Shows:
- Which parameters correlate with objective
- Which parameters matter most
- Guides search space refinement

### 4. Parameter Distributions

```python
analyzer.plot_parameter_distributions('distributions.png')
```

Shows:
- Explored parameter ranges
- Best parameter values (red line)
- Parameter preferences

## Complete Analysis

```python
from optuna_algorithms import OptimizationAnalyzer

# Analyze
analyzer = OptimizationAnalyzer('optimization_output')

# Summary to console
analyzer.print_summary()

# All plots
analyzer.plot_all(output_dir='analysis_plots', show=False)

# Export best configuration
analyzer.export_best_config('best_config.json')

# Generate markdown report
analyzer.to_markdown_report('optimization_report.md')
```

## Comparing Multiple Runs

```python
from optuna_algorithms.analysis import compare_optimizations

summary_df = compare_optimizations(
    result_dirs=['bayesian_output', 'cmaes_output', 'hybrid_output'],
    output_dir='comparison_plots',
)

print(summary_df)
```

Generates:
- Convergence comparison plot (all algorithms)
- Summary table (best values, times, success rates)
- Saved to `comparison_plots/`

### Command Line Comparison

```bash
python .claude/skills/analyze-optimization.py \
  --compare \
  bayesian_output \
  cmaes_output \
  nsga_output \
  --output_dir comparison
```

## Export Formats

### Best Config (JSON)

```python
analyzer.export_best_config('best_config.json')
```

```json
{
  "best_parameters": {
    "learning_rate": 0.0045,
    "batch_size": 128
  },
  "best_value": 0.123456,
  "algorithm": "Bayesian-TPE",
  "metadata": {
    "n_trials": 100,
    "success_rate": 0.95
  }
}
```

### Markdown Report

```python
analyzer.to_markdown_report('report.md')
```

Generates complete markdown report with:
- Summary statistics
- Best parameters
- Algorithm configuration
- Trial history

### CSV Data

Automatically saved by optimizers:
```
optimization_output/all_trials.csv
```

Load with pandas:
```python
import pandas as pd
df = pd.read_csv('optimization_output/all_trials.csv')
```

## Use Cases

### 1. Post-Optimization Analysis

```python
# After running optimization
analyzer = OptimizationAnalyzer('bayesian_output')
analyzer.print_summary()
analyzer.plot_all('plots')
```

**Answer:** Did it work? What were the results?

### 2. Algorithm Comparison

```python
compare_optimizations(
    ['bayesian_output', 'cmaes_output'],
    output_dir='comparison'
)
```

**Answer:** Which algorithm performed best?

### 3. Parameter Tuning

```python
analyzer.plot_parameter_importances('importance.png')
```

**Answer:** Which parameters should I focus on?

### 4. Convergence Check

```python
analyzer.plot_convergence('convergence.png')
```

**Answer:** Did I use enough trials?

### 5. Result Sharing

```python
analyzer.export_best_config('best_config.json')
analyzer.to_markdown_report('report.md')
```

**Answer:** Share results with team

## Output Directory Structure

After optimization:
```
optimization_output/
├── optimization_result.json    # Complete results
├── all_trials.csv              # All trial data
├── trials.jsonl                # Real-time log
├── best_config.json            # Best parameters (after analysis)
├── optimization_report.md      # Markdown report (after analysis)
└── plots/                      # Plots (after analysis)
    ├── convergence.png
    ├── parameter_importance.png
    └── parameter_distributions.png
```

## Requirements

- **pandas, numpy**: Required (data handling)
- **matplotlib**: Highly recommended (plotting)

Without matplotlib, analysis still works but plots are skipped.

## References

- API: `optuna_algorithms/README.md`
- Script: `analyze-optimization.py` (executable)
- Example: `examples/` directory
