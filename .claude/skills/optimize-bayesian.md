---
skill: optimize-bayesian
description: Bayesian optimization (TPE/GP) for sample-efficient hyperparameter tuning
tags: [optimization, bayesian, tpe, hyperparameter-tuning]
---

# Bayesian Optimization

Sample-efficient hyperparameter tuning using Tree-structured Parzen Estimator (TPE) or Gaussian Process (GP).

## Installation

```bash
# Required
pip install optuna pandas numpy

# Optional (for plots)
pip install matplotlib

# Optional (for GP variant)
pip install torch botorch
```

## When to Use

✅ **Use Bayesian when:**
- General ML hyperparameter tuning (default choice)
- Mixed continuous/discrete/categorical parameters
- Limited trial budget (100-1000 trials)
- Expensive objective evaluations
- High-dimensional spaces (100+ parameters)

❌ **Don't use when:**
- Pure continuous with correlations → Use CMA-ES
- Multiple competing objectives → Use NSGA
- Very small budget (<20 trials) → Use QMC first

## Quick Start

```python
from optuna_algorithms import optimize_bayesian

def objective(trial, params):
    # Your evaluation code
    lr = params['learning_rate']
    result = train_model(lr=lr)
    return result['loss']

result = optimize_bayesian(
    objective,
    search_space={'learning_rate': (0.001, 0.1)},
    n_trials=50,
)

print(f"Best: {result.best_params} = {result.best_value:.6f}")
```

## ML Training Integration

```python
from optuna_algorithms import optimize_bayesian
from train_wrapper import run_training_with_params
import optuna

def objective(trial, params):
    result = run_training_with_params(
        **params,
        output_dir=f"trials/trial_{trial.number}",
        trial_id=trial.number,
    )

    if result['status'] != 'success':
        raise optuna.TrialPruned()

    return result['val_bpb']

result = optimize_bayesian(
    objective,
    search_space={
        'embedding_lr': (0.1, 1.0),
        'matrix_lr': (0.01, 0.15),
        'weight_decay': (0.0, 0.5),
    },
    n_trials=50,
    multivariate=True,
)
```

## Command Line Usage

Use the standalone script:

```bash
# Demo optimization
python .claude/skills/optimize-bayesian.py --n_trials 50

# With ML training integration
python .claude/skills/optimize-bayesian.py --ml --n_trials 20 --multivariate

# Use GP variant
python .claude/skills/optimize-bayesian.py --variant gp --n_trials 100

# Get help
python .claude/skills/optimize-bayesian.py --help
```

## Configuration

### Key Parameters

- **variant**: `'tpe'` (default, fast) or `'gp'` (better for <500 trials)
- **n_startup_trials**: Random trials before modeling (default: 10)
- **multivariate**: Enable parameter interaction modeling (default: False)
- **n_ei_candidates**: Candidates for Expected Improvement (default: 24)

### Search Space

```python
search_space = {
    # Continuous (float)
    'learning_rate': (0.0001, 0.1),    # Auto log-scale for *_lr

    # Integer
    'batch_size': (16, 256),

    # Categorical
    'optimizer': ['adam', 'sgd', 'adamw'],
}
```

## Performance Tips

1. **Startup trials**: Set to ~sqrt(n_trials) for balanced exploration
2. **Multivariate**: Enable if parameters interact (e.g., lr × batch_size)
3. **Log scale**: Automatic for parameters ending in `_lr`
4. **Parallel**: TPE supports parallel evaluation
5. **Pruning**: Use `optuna.TrialPruned()` to skip failed trials

## Output

Results saved to `bayesian_output/`:
```
bayesian_output/
├── optimization_result.json    # Complete results
├── all_trials.csv              # Trial data
├── trials.jsonl                # Real-time log
├── best_config.json            # Best parameters
└── plots/                      # Analysis plots
    ├── convergence.png
    ├── parameter_importance.png
    └── parameter_distributions.png
```

## Analysis

```python
from optuna_algorithms import OptimizationAnalyzer

analyzer = OptimizationAnalyzer('bayesian_output')
analyzer.print_summary()
analyzer.plot_all('plots', show=False)
analyzer.export_best_config('best.json')
```

## Examples

See `examples/ml_tuning.py` for complete ML integration example.

## References

- Config: `optuna_algorithms/configs/bayesian_config.yaml`
- API: `optuna_algorithms/README.md`
- Script: `optimize-bayesian.py` (executable)
- [Optuna TPE Docs](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html)
