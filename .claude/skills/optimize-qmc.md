---
skill: optimize-qmc
description: Quasi-Monte Carlo exploration for systematic sampling and DoE
tags: [optimization, qmc, exploration, sobol, doe]
---

# QMC (Quasi-Monte Carlo) Exploration

Systematic exploration using low-discrepancy sequences (Sobol/Halton) for initial exploration and design of experiments.

## Installation

```bash
# Required
pip install optuna pandas numpy

# Optional (for plots)
pip install matplotlib
```

## When to Use

✅ **Use QMC when:**
- Initial exploration (first 50-100 trials)
- Design of Experiments (DoE)
- Warm-starting adaptive algorithms
- Need uniform coverage of search space
- Massively parallel workloads (perfect parallelization)

❌ **Don't use when:**
- As only optimizer (always combine with adaptive method)
- Very large budgets alone (use hybrid approach)

💡 **Best practice:** Use QMC for exploration, then switch to Bayesian/CMA-ES

## Quick Start

```python
from optuna_algorithms import qmc_then_adaptive

# QMC → Bayesian hybrid (recommended)
result = qmc_then_adaptive(
    objective_func=my_objective,
    search_space={'x': (-10, 10), 'y': (-10, 10)},
    n_qmc_trials=50,           # Exploration
    n_adaptive_trials=150,     # Exploitation
    adaptive_method='bayesian',
)

print(f"Best: {result.best_params} = {result.best_value:.6f}")
```

## Standalone QMC

```python
from optuna_algorithms import optimize_qmc

result = optimize_qmc(
    my_objective,
    search_space={'x': (-10, 10), 'y': (-10, 10)},
    n_trials=50,
    qmc_type='sobol',  # or 'halton'
)
```

## Command Line Usage

```bash
# QMC exploration only
python .claude/skills/optimize-qmc.py --n_trials 50

# QMC → Bayesian hybrid (recommended)
python .claude/skills/optimize-qmc.py --n_trials 50 --then bayesian --n_adaptive 150

# QMC → CMA-ES (for continuous problems)
python .claude/skills/optimize-qmc.py --n_trials 100 --then cmaes --n_adaptive 400

# Use Halton for low-dimensional
python .claude/skills/optimize-qmc.py --qmc_type halton --n_trials 50

# Get help
python .claude/skills/optimize-qmc.py --help
```

## Configuration

### QMC Types

- **'sobol'** (default): Best for d > 6 dimensions
- **'halton'**: Best for d ≤ 6 dimensions

Auto-selected based on dimensionality if not specified.

### Hybrid Strategies

**Pattern 1: QMC → Bayesian**
```python
from optuna_algorithms import qmc_then_adaptive

result = qmc_then_adaptive(
    objective_func=obj,
    search_space=space,
    n_qmc_trials=50,
    n_adaptive_trials=150,
    adaptive_method='bayesian',
)
```

**Pattern 2: Sequential (manual)**
```python
# Phase 1: QMC exploration
qmc_result = optimize_qmc(obj, space, n_trials=50)

# Phase 2: Bayesian exploitation
from optuna_algorithms import optimize_bayesian
final_result = optimize_bayesian(
    obj, space,
    n_trials=150,
    n_startup_trials=0,  # Already explored
)
```

## Why QMC?

### Key Benefits

- **O(1/N) convergence** vs O(1/√N) for random sampling
- **Perfect parallelization**: No coordination needed between trials
- **Deterministic**: Reproducible results with same seed
- **Zero overhead**: No model fitting
- **Systematic**: Better space coverage than random

### Comparison

```
Random sampling:  ●   ●  ●     ●●  ●   ● (clustered)
QMC (Sobol):     ● ● ● ● ● ● ● ● ● ● (uniform)
```

## Performance Tips

1. **Use as exploration**: QMC for first 20-30% of trials
2. **Then switch**: To Bayesian/CMA-ES for exploitation
3. **Parallelization**: QMC is embarrassingly parallel
4. **Sequence choice**: Sobol for most cases, Halton for d≤6

## Example: Complex Landscape

```python
from optuna_algorithms import qmc_then_adaptive

def rastrigin(trial, params):
    """Highly multimodal - needs good exploration"""
    import math
    A = 10
    n = len(params)
    return A*n + sum(v**2 - A*math.cos(2*math.pi*v)
                     for v in params.values())

# QMC ensures we explore the entire landscape
result = qmc_then_adaptive(
    rastrigin,
    search_space={
        'x1': (-5.12, 5.12),
        'x2': (-5.12, 5.12),
        'x3': (-5.12, 5.12),
    },
    n_qmc_trials=50,       # Systematic exploration
    n_adaptive_trials=150,  # Focused exploitation
    adaptive_method='bayesian',
)
```

## Output

Results saved to `qmc_output/` (or `qmc_adaptive_output/` for hybrid):
```
qmc_output/
├── optimization_result.json
├── all_trials.csv
└── plots/
```

## References

- Config: `optuna_algorithms/configs/qmc_config.yaml`
- API: `optuna_algorithms/README.md`
- Script: `optimize-qmc.py` (executable)
- [Optuna QMC Docs](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.QMCSampler.html)
