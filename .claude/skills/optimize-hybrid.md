---
skill: optimize-hybrid
description: Hybrid optimization with auto-strategy selection for mixed variables
tags: [optimization, hybrid, catcma, mixed-variables, nas]
---

# Hybrid Optimization

Adaptive strategies for mixed variables, neural architecture search, and auto-algorithm selection.

## Installation

```bash
# Required
pip install optuna pandas numpy

# Optional (for plots)
pip install matplotlib

# Optional (for CatCMA - best mixed-variable optimizer)
pip install optunahub
```

## When to Use

✅ **Use Hybrid when:**
- Mixed continuous + integer + categorical parameters
- Neural architecture search (NAS)
- AutoML with diverse hyperparameter types
- **Unsure which algorithm to use** (strategy='auto')
- Complex heterogeneous search spaces

❌ **Don't use when:**
- Pure continuous → Use CMA-ES
- Pure discrete → Use Bayesian
- Single parameter type → Use specialized algorithm

## Quick Start

```python
from optuna_algorithms import optimize_hybrid

result = optimize_hybrid(
    my_objective,
    search_space={
        'learning_rate': (0.0001, 0.1),  # continuous
        'n_layers': (2, 10),              # integer
        'activation': ['relu', 'gelu'],   # categorical
    },
    n_trials=200,
    strategy='auto',  # Automatically selects best approach
)

print(f"Best: {result.best_params} = {result.best_value:.6f}")
```

## Command Line Usage

```bash
# Auto-select strategy (recommended)
python .claude/skills/optimize-hybrid.py --strategy auto --n_trials 200

# Sequential multi-phase (QMC → TPE → CMA-ES)
python .claude/skills/optimize-hybrid.py --strategy sequential --n_trials 500

# Use CatCMA for mixed variables
python .claude/skills/optimize-hybrid.py --strategy catcma --n_trials 300

# Get help
python .claude/skills/optimize-hybrid.py --help
```

## Strategies

### 'auto' (Recommended)

Analyzes your search space and automatically selects:

```python
strategy='auto'

# Decision logic:
# - All continuous → CMA-ES
# - Mixed types + optunahub available → CatCMA
# - Otherwise → TPE (handles all types)
```

### 'catcma'

Best-in-class for mixed variables (2025 research):

```python
strategy='catcma'

# Requires: pip install optunahub
# Handles continuous, integer, categorical jointly
# 1.4x faster than TPE on mixed-variable benchmarks
```

### 'sequential'

Multi-phase optimization:

```python
strategy='sequential'

# Phase 1: QMC (20%) - Exploration
# Phase 2: TPE (50%) - Exploitation
# Phase 3: CMA-ES (30%) - Refinement (if continuous)
```

### 'tpe_only' / 'cmaes_only'

Fallback to specific algorithms:

```python
strategy='tpe_only'    # Use TPE regardless of space
strategy='cmaes_only'  # Use CMA-ES (continuous only)
```

## Example: Neural Architecture Search

```python
from optuna_algorithms import optimize_hybrid

def nas_objective(trial, params):
    """NAS with mixed hyperparameters"""

    # Build model architecture
    model = build_model(
        n_layers=params['n_layers'],        # integer
        layer_type=params['layer_type'],    # categorical
        hidden_size=params['hidden_size'],  # integer
        dropout=params['dropout'],          # continuous
        learning_rate=params['learning_rate'], # continuous
        optimizer=params['optimizer'],      # categorical
    )

    # Train and evaluate
    accuracy = train_and_eval(model)

    return -accuracy  # Minimize (negate for maximization)

result = optimize_hybrid(
    nas_objective,
    search_space={
        # Architecture parameters (discrete)
        'n_layers': (2, 10),
        'layer_type': ['conv', 'dense', 'attention'],
        'hidden_size': (64, 512),

        # Training parameters (mixed)
        'dropout': (0.0, 0.5),
        'learning_rate': (0.0001, 0.01),
        'optimizer': ['adam', 'sgd', 'adamw'],
    },
    n_trials=200,
    strategy='auto',  # Will use CatCMA if available
)

print(f"Best architecture: {result.best_params}")
print(f"Best accuracy: {-result.best_value:.4f}")
```

## Auto-Selection Logic

The system analyzes your search space:

```python
# Space analysis
n_continuous = count_float_params(search_space)
n_integer = count_int_params(search_space)
n_categorical = count_categorical_params(search_space)

if all continuous:
    → Use CMA-ES (best for continuous)

elif mixed and optunahub_available:
    → Use CatCMA (best for mixed)

else:
    → Use TPE (handles everything)
```

## Sequential Strategy Details

```python
from optuna_algorithms import HybridOptimizer

optimizer = HybridOptimizer(
    objective,
    search_space,
    strategy='sequential'
)

result = optimizer.optimize_sequential(
    n_trials=500,
    phase_splits=[0.2, 0.5, 0.3],  # QMC, TPE, CMA-ES
)

# Output shows which phase found the best result
```

## Performance Tips

1. **Start with 'auto'**: Let the system choose
2. **Install optunahub**: Enables CatCMA for mixed variables
3. **Sequential for large budgets**: Best for 500+ trials
4. **TPE is solid fallback**: Works reasonably for all types

## Output

Results saved to `hybrid_output/`:
```
hybrid_output/
├── optimization_result.json
├── all_trials.csv
├── plots/
└── (phase1_qmc/, phase2_tpe/, etc. if sequential)
```

## References

- Config: `optuna_algorithms/configs/hybrid_config.yaml`
- API: `optuna_algorithms/README.md`
- Script: `optimize-hybrid.py` (executable)
- [CatCMA Paper (GECCO 2025)](https://dl.acm.org/doi/10.1145/3712256.3726471)
