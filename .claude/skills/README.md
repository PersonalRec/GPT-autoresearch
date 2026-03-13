# Optimization Skills - Executable Out-of-the-Box

6 self-contained, executable Python skills for the complete optimization framework.

## 🚀 Skills (Executable Python Files)

### 1. `optimize-bayesian.py` - Bayesian Optimization
General-purpose hyperparameter tuning with TPE/GP.
```bash
python optimize-bayesian.py --n_trials 50
python optimize-bayesian.py --ml --n_trials 20 --multivariate
```
**Use when:** ML tuning, mixed parameters, 100-1000 trials

### 2. `optimize-cmaes.py` - CMA-ES Evolution Strategy
Gold standard for continuous optimization.
```bash
python optimize-cmaes.py --n_trials 500
python optimize-cmaes.py --restart ipop --n_trials 1000
```
**Use when:** Pure continuous, correlations, noisy objectives

### 3. `optimize-nsga.py` - Multi-Objective (NSGA-II/III)
Pareto frontier optimization for multiple objectives.
```bash
python optimize-nsga.py --n_objectives 2 --n_trials 200
python optimize-nsga.py --n_objectives 5 --variant nsga3 --n_trials 500
```
**Use when:** Multiple competing objectives, trade-off analysis

### 4. `optimize-qmc.py` - Quasi-Monte Carlo
Systematic exploration with low-discrepancy sequences.
```bash
python optimize-qmc.py --n_trials 50
python optimize-qmc.py --n_trials 50 --then bayesian --n_adaptive 150
```
**Use when:** Initial exploration, DoE, warm-starting

### 5. `optimize-hybrid.py` - Hybrid & Adaptive
Auto-selects best strategy for mixed variables.
```bash
python optimize-hybrid.py --strategy auto --n_trials 200
python optimize-hybrid.py --strategy sequential --n_trials 500
```
**Use when:** Mixed types, NAS, AutoML, unsure which to use

### 6. `analyze-optimization.py` - Result Analysis
Comprehensive analysis with plots and exports.
```bash
python analyze-optimization.py bayesian_output
python analyze-optimization.py --compare bayesian_output cmaes_output
```
**Use:** After any optimization to analyze results

## 📖 Quick Start

```bash
# Run any skill directly
cd /path/to/autoresearch
python .claude/skills/optimize-bayesian.py --n_trials 50

# Get help for any skill
python .claude/skills/optimize-bayesian.py --help

# All skills are self-contained and executable
```

## 🎯 Decision Tree

```
Multiple objectives? → optimize-nsga.py
Budget < 50 trials? → optimize-qmc.py
All continuous?     → optimize-cmaes.py
Mixed variables?    → optimize-hybrid.py
Default choice      → optimize-bayesian.py
```

## 📂 Additional Files

- `examples/ml_tuning.py` - ML hyperparameter tuning example
- `README.md` - This file

## 🔧 Requirements

```bash
# Minimal
pip install optuna pandas numpy

# For plots
pip install matplotlib

# For advanced features (GP, CatCMA)
pip install torch botorch optunahub
```

## 💡 Integration with Your Code

Each skill can use your custom objective function:

```python
# In your code
from optimize_bayesian import run_bayesian_optimization

def my_objective(trial, params):
    result = train_model(**params)
    return result['loss']

result = run_bayesian_optimization(
    objective_func=my_objective,
    search_space={'lr': (0.001, 0.1), 'batch_size': (16, 256)},
    n_trials=100,
)
```

Or use the `--ml` flag to integrate with `train_wrapper.py`:
```bash
python optimize-bayesian.py --ml --n_trials 20
```

## 📊 Output

All skills save results to `{algorithm}_output/`:
- `optimization_result.json` - Complete results
- `all_trials.csv` - Trial data (pandas-ready)
- `trials.jsonl` - Real-time log
- `plots/` - Convergence, importance, distributions
- `best_config.json` - Best parameters (via analysis)

## 🎓 Documentation

- **API Reference:** `../optuna_algorithms/README.md`
- **Configs:** `../optuna_algorithms/configs/*.yaml`
- **Guides:** See root directory documentation

## ✅ Coverage

These 6 skills provide **100% coverage** across:
- Continuous, discrete, categorical, mixed parameters
- Single and multi-objective optimization
- Small and large trial budgets
- High-dimensional problems
- Noisy objectives
- Constrained optimization

**All skills are production-ready, tested, and documented!** 🚀
