---
skill: optimize-cmaes
description: CMA-ES for continuous optimization with parameter correlations
tags: [optimization, cmaes, evolution-strategy, continuous]
---

# CMA-ES Optimization

Covariance Matrix Adaptation Evolution Strategy - the gold standard for continuous optimization.

## Installation

```bash
# Required
pip install optuna pandas numpy

# Optional (for plots)
pip install matplotlib
```

## When to Use

✅ **Use CMA-ES when:**
- **ALL** parameters are continuous (float only)
- Parameter correlations/interactions expected
- Noisy objective functions
- Large trial budget (500-10000 trials)
- Robotics, control, continuous policy search

❌ **Don't use when:**
- Any discrete/categorical parameters → Use Bayesian or Hybrid
- Small budget (<200 trials) → Use Bayesian
- Multi-objective → Use NSGA

⚠️ **Important:** CMA-ES ONLY works with continuous parameters!

## Quick Start

```python
from optuna_algorithms import optimize_cmaes

def objective(trial, params):
    # Rosenbrock function (has strong correlation)
    x, y = params['x'], params['y']
    return (1 - x)**2 + 100*(y - x**2)**2

result = optimize_cmaes(
    objective,
    search_space={
        'x': (-2.0, 2.0),
        'y': (-1.0, 3.0),
    },
    n_trials=500,
    restart_strategy='ipop',
)

print(f"Best: {result.best_params} = {result.best_value:.6f}")
```

## Command Line Usage

```bash
# Standard CMA-ES
python .claude/skills/optimize-cmaes.py --n_trials 500

# With restart strategy for multimodal problems
python .claude/skills/optimize-cmaes.py --restart ipop --n_trials 1000

# BIPOP variant
python .claude/skills/optimize-cmaes.py --restart bipop --n_trials 2000

# Get help
python .claude/skills/optimize-cmaes.py --help
```

## Configuration

### Restart Strategies

- **None** (default): Standard CMA-ES
- **'ipop'**: Increases population on restart (for multimodal problems)
- **'bipop'**: Alternates large/small populations (complex landscapes)

### Key Parameters

- **restart_strategy**: Helps escape local minima
- **use_separable**: Set `True` for high-dim if parameters independent
- **population_size**: Auto-computed as `4 + 3*ln(n_params)`

## Performance Tips

1. **Trial budget**: Need ≥50×sqrt(n_params) trials minimum
2. **Restart strategy**: Use 'ipop' if stuck in local minima
3. **Normalize**: Scale parameters to similar ranges if possible
4. **Separable CMA**: Only if parameters truly independent (rare)

## Validation

CMA-ES will error if you have non-continuous parameters:

```python
# ❌ This will fail - has integer parameter
search_space = {
    'x': (0.0, 1.0),      # OK - continuous
    'batch_size': (16, 256),  # ❌ FAIL - integer
}

# ✅ This works - all continuous
search_space = {
    'x': (0.0, 1.0),
    'y': (-5.0, 5.0),
    'z': (0.0, 10.0),
}
```

## Output

Results saved to `cmaes_output/`:
```
cmaes_output/
├── optimization_result.json
├── all_trials.csv
├── best_config.json
└── plots/
```

## References

- Config: `optuna_algorithms/configs/cmaes_config.yaml`
- API: `optuna_algorithms/README.md`
- Script: `optimize-cmaes.py` (executable)
- [Optuna CMA-ES Docs](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.CmaEsSampler.html)
