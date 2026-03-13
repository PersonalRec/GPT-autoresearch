---
skill: optimize-nsga
description: Multi-objective optimization with NSGA-II/III for Pareto frontiers
tags: [optimization, multi-objective, nsga, pareto]
---

# NSGA Multi-Objective Optimization

NSGA-II/III for optimizing multiple competing objectives and finding Pareto frontiers.

## Installation

```bash
# Required
pip install optuna pandas numpy

# Optional (for plots)
pip install matplotlib
```

## When to Use

✅ **Use NSGA when:**
- Multiple competing objectives (accuracy vs speed, cost vs performance)
- Need entire Pareto frontier (not single solution)
- Trade-off analysis required
- 2-10 objectives

❌ **Don't use when:**
- Single objective → Use Bayesian or CMA-ES
- Very small budget (<100 trials)

## Quick Start

```python
from optuna_algorithms import optimize_nsga

def multi_objective(trial, params):
    """Return tuple of objective values"""
    accuracy = train_model(params)
    latency = measure_latency(params)

    # Return both values (to minimize)
    return accuracy, latency

result = optimize_nsga(
    multi_objective,
    search_space={'lr': (0.001, 0.1), 'size': (64, 512)},
    n_objectives=2,
    directions=['maximize', 'minimize'],  # accuracy up, latency down
    n_trials=200,
)

# Access Pareto frontier
pareto = result.metadata['pareto_front']
for solution in pareto:
    print(f"Objectives: {solution['values']}")
    print(f"Parameters: {solution['params']}")
```

## Command Line Usage

```bash
# Bi-objective optimization
python .claude/skills/optimize-nsga.py --n_objectives 2 --n_trials 200

# Many-objective (use NSGA-III)
python .claude/skills/optimize-nsga.py --n_objectives 5 --variant nsga3 --n_trials 500

# Custom directions
python .claude/skills/optimize-nsga.py --n_objectives 2 --directions maximize minimize

# Get help
python .claude/skills/optimize-nsga.py --help
```

## Configuration

### Variants

- **NSGA-II**: Best for 2-3 objectives (crowding distance)
- **NSGA-III**: Best for 4+ objectives (reference points)
- **'auto'** (default): Automatically selects based on n_objectives

### Key Parameters

- **n_objectives**: Number of objectives (2-10)
- **directions**: List of 'minimize' or 'maximize' for each objective
- **variant**: 'nsga2', 'nsga3', or 'auto'
- **population_size**: Auto or manual (~50 for 2 obj, ~100 for 4+ obj)

## Understanding Results

⚠️ **Important:** Multi-objective returns a **Pareto frontier**, not a single "best" solution.

```python
# All Pareto solutions are equally "optimal"
pareto_front = result.metadata['pareto_front']

print(f"Pareto frontier: {len(pareto_front)} solutions")

# Choose solution based on your preference
for i, solution in enumerate(pareto_front):
    print(f"\nSolution {i}:")
    print(f"  Objectives: {solution['values']}")
    print(f"  Parameters: {solution['params']}")

# Example: Find best compromise
best_compromise = min(pareto_front,
                     key=lambda x: sum(x['values']))
```

## Example: Accuracy vs Latency

```python
from optuna_algorithms import optimize_nsga
import optuna

def accuracy_vs_latency(trial, params):
    """Bi-objective: maximize accuracy, minimize latency"""

    # Train model
    model = train_model(
        learning_rate=params['learning_rate'],
        model_size=params['model_size'],
    )

    # Measure both objectives
    accuracy = model.evaluate()  # Higher is better
    latency_ms = model.measure_latency()  # Lower is better

    return accuracy, latency_ms

result = optimize_nsga(
    accuracy_vs_latency,
    search_space={
        'learning_rate': (0.0001, 0.01),
        'model_size': (64, 512),
    },
    n_objectives=2,
    directions=['maximize', 'minimize'],
    n_trials=200,
    variant='nsga2',
)

# Analyze Pareto frontier
pareto = result.metadata['pareto_front']
print(f"Found {len(pareto)} Pareto-optimal solutions")

# Filter by constraint (e.g., latency < 100ms)
fast_models = [s for s in pareto if s['values'][1] < 100]
print(f"Fast models (<100ms): {len(fast_models)}")
```

## Output

Results saved to `nsga_output/`:
```
nsga_output/
├── optimization_result.json    # Includes Pareto frontier
├── all_trials.csv
└── metadata['pareto_front']    # All Pareto-optimal solutions
```

## Performance Tips

1. **Population size**: ~50×(n_objectives-1) as rule of thumb
2. **Trials**: Need ≥20×population_size for convergence
3. **Variant**: Use NSGA-III for 4+ objectives
4. **Normalization**: Scale objectives to similar ranges

## References

- Config: `optuna_algorithms/configs/nsga_config.yaml`
- API: `optuna_algorithms/README.md`
- Script: `optimize-nsga.py` (executable)
- [Optuna NSGA-II Docs](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.NSGAIISampler.html)
