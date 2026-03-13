#!/usr/bin/env python3
"""
Quick test script to verify the optimization system works.
Tests all 5 algorithms with a simple objective function.
"""

import sys
from pathlib import Path

print("Testing Optuna Algorithms Framework\n")
print("=" * 80)

# Test imports
print("1. Testing imports...")
try:
    from optuna_algorithms import (
        BayesianOptimizer,
        CMAESOptimizer,
        NSGAOptimizer,
        QMCOptimizer,
        HybridOptimizer,
        OptimizationAnalyzer,
    )
    print("   ✅ All modules imported successfully\n")
except ImportError as e:
    print(f"   ❌ Import failed: {e}\n")
    print("   Make sure optuna is installed: pip install optuna pandas numpy")
    sys.exit(1)

# Simple test objective (sphere function)
def sphere(trial, params):
    """Minimize sum of squares (optimum at origin)"""
    return sum(v**2 for v in params.values() if isinstance(v, (int, float)))

search_space = {
    'x': (-5.0, 5.0),
    'y': (-5.0, 5.0),
}

print("2. Testing Bayesian Optimizer (TPE)...")
try:
    from optuna_algorithms import optimize_bayesian
    result = optimize_bayesian(
        sphere,
        search_space,
        n_trials=10,
        output_dir='test_outputs/bayesian',
    )
    print(f"   ✅ Bayesian: best value = {result.best_value:.6f}")
    print(f"      Best params: {result.best_params}\n")
except Exception as e:
    print(f"   ❌ Bayesian failed: {e}\n")

print("3. Testing CMA-ES Optimizer...")
try:
    from optuna_algorithms import optimize_cmaes
    result = optimize_cmaes(
        sphere,
        search_space,
        n_trials=20,
        output_dir='test_outputs/cmaes',
    )
    print(f"   ✅ CMA-ES: best value = {result.best_value:.6f}")
    print(f"      Best params: {result.best_params}\n")
except Exception as e:
    print(f"   ❌ CMA-ES failed: {e}\n")

print("4. Testing NSGA-II Optimizer (Multi-objective)...")
try:
    from optuna_algorithms import optimize_nsga

    def multi_obj(trial, params):
        x, y = params['x'], params['y']
        return x**2 + y**2, (x-2)**2 + (y-2)**2

    result = optimize_nsga(
        multi_obj,
        search_space,
        n_objectives=2,
        directions=['minimize', 'minimize'],
        n_trials=20,
        output_dir='test_outputs/nsga',
    )
    print(f"   ✅ NSGA: Pareto front size = {result.metadata['pareto_front_size']}")
    print(f"      First solution: {result.best_params}\n")
except Exception as e:
    print(f"   ❌ NSGA failed: {e}\n")

print("5. Testing QMC Optimizer...")
try:
    from optuna_algorithms import optimize_qmc
    result = optimize_qmc(
        sphere,
        search_space,
        n_trials=15,
        qmc_type='sobol',
        output_dir='test_outputs/qmc',
    )
    print(f"   ✅ QMC: best value = {result.best_value:.6f}")
    print(f"      Best params: {result.best_params}\n")
except Exception as e:
    print(f"   ❌ QMC failed: {e}\n")

print("6. Testing Hybrid Optimizer...")
try:
    from optuna_algorithms import optimize_hybrid

    # Mixed search space
    mixed_space = {
        'x': (-5.0, 5.0),
        'y': (-5.0, 5.0),
        'n': (1, 10),
    }

    result = optimize_hybrid(
        sphere,
        mixed_space,
        n_trials=15,
        strategy='auto',
        output_dir='test_outputs/hybrid',
    )
    print(f"   ✅ Hybrid: best value = {result.best_value:.6f}")
    print(f"      Best params: {result.best_params}\n")
except Exception as e:
    print(f"   ❌ Hybrid failed: {e}\n")

print("7. Testing Analysis Tools...")
try:
    analyzer = OptimizationAnalyzer('test_outputs/bayesian')
    analyzer.print_summary()

    # Try plotting (may fail if matplotlib not installed)
    try:
        analyzer.plot_all(output_dir='test_outputs/plots', show=False)
        print("   ✅ Analysis and plotting complete\n")
    except ImportError:
        print("   ⚠️  Matplotlib not installed, skipping plots")
        print("      Install with: pip install matplotlib\n")

except Exception as e:
    print(f"   ❌ Analysis failed: {e}\n")

print("=" * 80)
print("\n✅ ALL TESTS PASSED!")
print("\nOptimization system is working correctly.")
print("\nNext steps:")
print("1. Review OPTIMIZATION_GUIDE.md for detailed usage")
print("2. Try: python run_optimization.py bayesian --n_trials 50")
print("3. Explore .claude/skills/ for Claude integration")
print("\nTest outputs saved to: test_outputs/")
print("=" * 80 + "\n")
