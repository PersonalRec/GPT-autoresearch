#!/usr/bin/env python3
"""
Unified optimization runner for all 5 algorithms.

Usage:
    python run_optimization.py bayesian --n_trials 50
    python run_optimization.py cmaes --n_trials 200
    python run_optimization.py nsga --n_objectives 2 --n_trials 100
    python run_optimization.py qmc --n_trials 50
    python run_optimization.py hybrid --strategy auto --n_trials 150

For ML training:
    python run_optimization.py bayesian --n_trials 20 --use-train-wrapper
"""

import argparse
import sys
from pathlib import Path

# Import optimizers
from optuna_algorithms import (
    BayesianOptimizer,
    CMAESOptimizer,
    NSGAOptimizer,
    QMCOptimizer,
    HybridOptimizer,
    OptimizationAnalyzer,
)

# Import training wrapper if available
try:
    from train_wrapper import run_training_with_params
    TRAIN_WRAPPER_AVAILABLE = True
except ImportError:
    TRAIN_WRAPPER_AVAILABLE = False
    print("⚠️  train_wrapper.py not available. Using demo objective function.")


def create_ml_objective():
    """Create objective function for ML training"""
    import optuna

    def objective(trial, params):
        # Run training with these parameters
        result = run_training_with_params(
            **params,
            output_dir=Path("temp_trial") / f"trial_{trial.number}",
            trial_id=trial.number,
        )

        # Prune failed trials
        if result['status'] != 'success':
            print(f"❌ Trial {trial.number} failed: {result.get('error', 'Unknown')}")
            raise optuna.TrialPruned()

        return result['val_bpb']

    return objective


def create_demo_objective():
    """Create demo objective function (Rosenbrock)"""
    def objective(trial, params):
        # Rosenbrock function: (1-x)^2 + 100(y-x^2)^2
        # Global minimum at (1, 1) with value 0
        if 'x' in params and 'y' in params:
            x = params['x']
            y = params['y']
            return (1 - x)**2 + 100 * (y - x**2)**2

        # Generic: sum of squares
        return sum(v**2 for v in params.values() if isinstance(v, (int, float)))

    return objective


def get_ml_search_space():
    """Search space for ML training"""
    return {
        'embedding_lr': (0.1, 1.0),
        'matrix_lr': (0.01, 0.15),
        'weight_decay': (0.0, 0.5),
        'warmdown_ratio': (0.0, 0.5),
        'depth': (2, 12),
        'device_batch_size': (2, 8),
    }


def get_demo_search_space():
    """Demo search space"""
    return {
        'x': (-5.0, 5.0),
        'y': (-5.0, 5.0),
    }


def run_bayesian(args):
    """Run Bayesian optimization"""
    print(f"\n{'='*80}")
    print("BAYESIAN OPTIMIZATION (TPE)")
    print(f"{'='*80}\n")

    if args.use_train_wrapper and TRAIN_WRAPPER_AVAILABLE:
        objective = create_ml_objective()
        search_space = get_ml_search_space()
    else:
        objective = create_demo_objective()
        search_space = get_demo_search_space()

    optimizer = BayesianOptimizer(
        objective_func=objective,
        search_space=search_space,
        direction='minimize',
        output_dir=args.output_dir or 'bayesian_output',
        variant=args.variant,
        n_startup_trials=args.n_startup,
        multivariate=args.multivariate,
        seed=args.seed,
    )

    result = optimizer.optimize(n_trials=args.n_trials)

    # Analyze
    if args.analyze:
        analyzer = OptimizationAnalyzer(optimizer.output_dir)
        analyzer.print_summary()
        analyzer.plot_all(output_dir=optimizer.output_dir / 'plots', show=False)
        analyzer.export_best_config(optimizer.output_dir / 'best_config.json')

    return result


def run_cmaes(args):
    """Run CMA-ES optimization"""
    print(f"\n{'='*80}")
    print("CMA-ES OPTIMIZATION")
    print(f"{'='*80}\n")

    if args.use_train_wrapper:
        print("⚠️  Warning: CMA-ES works best with continuous parameters only.")
        print("   Consider using --hybrid for ML training with mixed variables.\n")

    objective = create_demo_objective()
    search_space = get_demo_search_space()

    optimizer = CMAESOptimizer(
        objective_func=objective,
        search_space=search_space,
        direction='minimize',
        output_dir=args.output_dir or 'cmaes_output',
        restart_strategy=args.restart_strategy,
        seed=args.seed,
    )

    result = optimizer.optimize(n_trials=args.n_trials)

    if args.analyze:
        analyzer = OptimizationAnalyzer(optimizer.output_dir)
        analyzer.print_summary()
        analyzer.plot_all(output_dir=optimizer.output_dir / 'plots', show=False)

    return result


def run_nsga(args):
    """Run NSGA multi-objective optimization"""
    print(f"\n{'='*80}")
    print(f"NSGA-{args.variant.upper() if args.variant != 'auto' else 'AUTO'} MULTI-OBJECTIVE OPTIMIZATION")
    print(f"{'='*80}\n")

    def multi_obj(trial, params):
        # Demo: two competing objectives
        x = params['x']
        y = params['y']

        obj1 = x**2 + y**2  # Distance from origin
        obj2 = (x-2)**2 + (y-2)**2  # Distance from (2,2)

        return obj1, obj2

    optimizer = NSGAOptimizer(
        objective_func=multi_obj,
        search_space=get_demo_search_space(),
        n_objectives=args.n_objectives,
        directions=['minimize'] * args.n_objectives,
        output_dir=args.output_dir or 'nsga_output',
        variant=args.variant,
        seed=args.seed,
    )

    result = optimizer.optimize(n_trials=args.n_trials)

    print(f"\n📊 Pareto Front: {result.metadata['pareto_front_size']} solutions")
    print("\nTop 5 Pareto solutions:")
    for i, point in enumerate(result.metadata['pareto_front'][:5], 1):
        print(f"{i}. Objectives: {point['values']}, Params: {point['params']}")

    if args.analyze:
        analyzer = OptimizationAnalyzer(optimizer.output_dir)
        analyzer.print_summary()

    return result


def run_qmc(args):
    """Run QMC exploration"""
    print(f"\n{'='*80}")
    print("QMC EXPLORATION")
    print(f"{'='*80}\n")

    if args.use_train_wrapper and TRAIN_WRAPPER_AVAILABLE:
        objective = create_ml_objective()
        search_space = get_ml_search_space()
    else:
        objective = create_demo_objective()
        search_space = get_demo_search_space()

    optimizer = QMCOptimizer(
        objective_func=objective,
        search_space=search_space,
        direction='minimize',
        output_dir=args.output_dir or 'qmc_output',
        qmc_type=args.qmc_type,
        seed=args.seed,
    )

    result = optimizer.optimize(n_trials=args.n_trials)

    if args.analyze:
        analyzer = OptimizationAnalyzer(optimizer.output_dir)
        analyzer.print_summary()

    return result


def run_hybrid(args):
    """Run hybrid optimization"""
    print(f"\n{'='*80}")
    print(f"HYBRID OPTIMIZATION (strategy={args.strategy})")
    print(f"{'='*80}\n")

    if args.use_train_wrapper and TRAIN_WRAPPER_AVAILABLE:
        objective = create_ml_objective()
        search_space = get_ml_search_space()
    else:
        objective = create_demo_objective()
        # Demo with mixed variables
        search_space = {
            'x': (-5.0, 5.0),      # continuous
            'y': (-5.0, 5.0),      # continuous
            'n': (1, 10),          # integer
            'mode': ['fast', 'accurate'],  # categorical
        }

    optimizer = HybridOptimizer(
        objective_func=objective,
        search_space=search_space,
        direction='minimize',
        output_dir=args.output_dir or 'hybrid_output',
        strategy=args.strategy,
        seed=args.seed,
    )

    if args.strategy == 'sequential':
        result = optimizer.optimize_sequential(n_trials=args.n_trials)
    else:
        result = optimizer.optimize(n_trials=args.n_trials)

    if args.analyze:
        analyzer = OptimizationAnalyzer(optimizer.output_dir)
        analyzer.print_summary()
        analyzer.plot_all(output_dir=optimizer.output_dir / 'plots', show=False)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Unified optimization runner for all 5 algorithms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest='algorithm', help='Optimization algorithm')

    # Bayesian
    parser_bayesian = subparsers.add_parser('bayesian', help='Bayesian optimization (TPE/GP)')
    parser_bayesian.add_argument('--n_trials', type=int, default=50)
    parser_bayesian.add_argument('--variant', choices=['tpe', 'gp'], default='tpe')
    parser_bayesian.add_argument('--n_startup', type=int, default=10)
    parser_bayesian.add_argument('--multivariate', action='store_true')
    parser_bayesian.add_argument('--output_dir', type=str)
    parser_bayesian.add_argument('--use-train-wrapper', action='store_true')
    parser_bayesian.add_argument('--analyze', action='store_true', default=True)
    parser_bayesian.add_argument('--seed', type=int, default=42)

    # CMA-ES
    parser_cmaes = subparsers.add_parser('cmaes', help='CMA-ES optimization')
    parser_cmaes.add_argument('--n_trials', type=int, default=200)
    parser_cmaes.add_argument('--restart_strategy', choices=[None, 'ipop', 'bipop'], default=None)
    parser_cmaes.add_argument('--output_dir', type=str)
    parser_cmaes.add_argument('--use-train-wrapper', action='store_true')
    parser_cmaes.add_argument('--analyze', action='store_true', default=True)
    parser_cmaes.add_argument('--seed', type=int, default=42)

    # NSGA
    parser_nsga = subparsers.add_parser('nsga', help='Multi-objective optimization')
    parser_nsga.add_argument('--n_trials', type=int, default=100)
    parser_nsga.add_argument('--n_objectives', type=int, default=2)
    parser_nsga.add_argument('--variant', choices=['nsga2', 'nsga3', 'auto'], default='auto')
    parser_nsga.add_argument('--output_dir', type=str)
    parser_nsga.add_argument('--analyze', action='store_true', default=True)
    parser_nsga.add_argument('--seed', type=int, default=42)

    # QMC
    parser_qmc = subparsers.add_parser('qmc', help='QMC exploration')
    parser_qmc.add_argument('--n_trials', type=int, default=50)
    parser_qmc.add_argument('--qmc_type', choices=['sobol', 'halton'], default='sobol')
    parser_qmc.add_argument('--output_dir', type=str)
    parser_qmc.add_argument('--use-train-wrapper', action='store_true')
    parser_qmc.add_argument('--analyze', action='store_true', default=True)
    parser_qmc.add_argument('--seed', type=int, default=42)

    # Hybrid
    parser_hybrid = subparsers.add_parser('hybrid', help='Hybrid optimization')
    parser_hybrid.add_argument('--n_trials', type=int, default=150)
    parser_hybrid.add_argument('--strategy', choices=['auto', 'catcma', 'sequential', 'tpe_only', 'cmaes_only'], default='auto')
    parser_hybrid.add_argument('--output_dir', type=str)
    parser_hybrid.add_argument('--use-train-wrapper', action='store_true')
    parser_hybrid.add_argument('--analyze', action='store_true', default=True)
    parser_hybrid.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    if not args.algorithm:
        parser.print_help()
        sys.exit(1)

    # Run selected algorithm
    if args.algorithm == 'bayesian':
        result = run_bayesian(args)
    elif args.algorithm == 'cmaes':
        result = run_cmaes(args)
    elif args.algorithm == 'nsga':
        result = run_nsga(args)
    elif args.algorithm == 'qmc':
        result = run_qmc(args)
    elif args.algorithm == 'hybrid':
        result = run_hybrid(args)

    print(f"\n{'='*80}")
    print("✅ OPTIMIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"Best value: {result.best_value:.6f}")
    print(f"Best params: {result.best_params}")
    print(f"Output saved to: {result.metadata.get('search_space', 'N/A')}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
