"""
Example: Bayesian optimization for ML hyperparameter tuning
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from optuna_algorithms import optimize_bayesian
import optuna

# This would import your actual training function
# from train_wrapper import run_training_with_params


def mock_objective(trial, params):
    """
    Mock objective function for demonstration.
    Replace with your actual training code.
    """
    # Simulate training with these hyperparameters
    lr = params['learning_rate']
    wd = params['weight_decay']
    bs = params['batch_size']

    # Mock loss function (replace with actual training)
    # Lower is better
    mock_loss = (lr - 0.01)**2 + (wd - 0.1)**2 + (bs - 128)**2 / 10000

    return mock_loss


if __name__ == "__main__":
    print("Running Bayesian Optimization Example\n")

    result = optimize_bayesian(
        objective_func=mock_objective,
        search_space={
            'learning_rate': (0.0001, 0.1),
            'weight_decay': (0.0, 0.5),
            'batch_size': (16, 256),
        },
        n_trials=30,
        variant='tpe',
        n_startup_trials=5,
        multivariate=True,
        output_dir='example_bayesian_output',
    )

    print(f"\n{'='*80}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"Best value: {result.best_value:.6f}")
    print(f"Best parameters:")
    for k, v in result.best_params.items():
        print(f"  {k}: {v}")
    print(f"{'='*80}\n")

    # Analyze results
    from optuna_algorithms import OptimizationAnalyzer

    analyzer = OptimizationAnalyzer('example_bayesian_output')
    analyzer.print_summary()
    analyzer.plot_all(output_dir='example_plots', show=False)
    analyzer.export_best_config('best_config.json')

    print("\n✅ Results saved to: example_bayesian_output/")
    print("✅ Plots saved to: example_plots/")
    print("✅ Best config saved to: best_config.json")
