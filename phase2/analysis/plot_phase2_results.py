#!/usr/bin/env python3
"""Generate Phase 2 visualization plots."""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set clean style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'

# Build complete timeline of all 74 trials
all_trials = []

# Phase 1: Manual exploration (trials 1-16)
with open('../results.tsv', 'r') as f:
    lines = f.readlines()[1:]  # Skip header
    manual_trials = []
    for line in lines:
        if line.strip():
            parts = line.strip().split('\t')
            # Handle lines with or without commit hash
            if len(parts) >= 5:
                # Has commit: commit, val_bpb, memory_gb, status, description
                val_bpb = float(parts[1])
                desc = parts[4]
            elif len(parts) >= 4:
                # No commit (empty field): val_bpb, memory_gb, status, description
                val_bpb = float(parts[0])
                desc = parts[3]
            else:
                continue

            manual_trials.append({
                'val_bpb': val_bpb,
                'description': desc
            })

# Add first 16 manual trials
all_trials.extend(manual_trials[:16])

# Phase 2: Bayesian unfocused (trials 17-26)
bayesian_unfocused = []
for exp_dir in sorted(Path('../experiments').glob('bayesian_run_2026*')):
    results_file = exp_dir / 'results.jsonl'
    if results_file.exists():
        with open(results_file) as f:
            for line in f:
                data = json.loads(line)
                bayesian_unfocused.append({
                    'val_bpb': data['results']['val_bpb'],
                    'description': 'Bayesian unfocused'
                })
all_trials.extend(bayesian_unfocused)

# Phase 3: Genetic unfocused (trials 27-36)
genetic_unfocused = []
for exp_dir in sorted(Path('../experiments').glob('genetic_run_2026*')):
    results_file = exp_dir / 'results.jsonl'
    if results_file.exists():
        with open(results_file) as f:
            for line in f:
                data = json.loads(line)
                genetic_unfocused.append({
                    'val_bpb': data['results']['val_bpb'],
                    'description': 'Genetic unfocused'
                })
all_trials.extend(genetic_unfocused)

# Phase 4: Bayesian focused (trials 37-51)
bayesian_focused = []
for exp_dir in sorted(Path('../experiments').glob('bayesian_focused_*')):
    results_file = exp_dir / 'results.jsonl'
    if results_file.exists():
        with open(results_file) as f:
            for line in f:
                data = json.loads(line)
                bayesian_focused.append({
                    'val_bpb': data['results']['val_bpb'],
                    'description': 'Bayesian focused'
                })
all_trials.extend(bayesian_focused)

# Phase 5: Genetic focused (trials 52-66)
genetic_focused = []
for exp_dir in sorted(Path('../experiments').glob('genetic_focused_*')):
    results_file = exp_dir / 'results.jsonl'
    if results_file.exists():
        with open(results_file) as f:
            for line in f:
                data = json.loads(line)
                genetic_focused.append({
                    'val_bpb': data['results']['val_bpb'],
                    'description': 'Genetic focused'
                })
all_trials.extend(genetic_focused)

# Phase 6: Manual refinement (trials 67-74)
all_trials.extend(manual_trials[16:])

# Calculate running best and track improvements
running_best = []
best_val = float('inf')
improvements = []

for i, trial in enumerate(all_trials, 1):
    val = trial['val_bpb']
    if val < best_val:
        best_val = val
        desc = trial['description']

        # Clean up descriptions for better readability
        if desc.startswith('baseline'):
            desc = 'baseline'
        elif 'MATRIX_LR' in desc:
            # Extract just the key change
            if 'increase' in desc:
                desc = desc.replace('increase ', '+')
            if 'MATRIX_LR' in desc:
                desc = desc.replace('MATRIX_LR', 'matrix_lr')
        elif 'WARMDOWN' in desc:
            if 'reduce' in desc:
                desc = desc.replace('reduce ', '')
            desc = desc.replace('WARMDOWN_RATIO', 'warmdown')
        elif 'EMBEDDING_LR' in desc:
            if 'increase' in desc or 'reduce' in desc:
                desc = desc.replace('increase ', '+').replace('reduce ', '')
            desc = desc.replace('EMBEDDING_LR', 'emb_lr')
        elif 'WEIGHT_DECAY' in desc:
            if 'reduce' in desc:
                desc = desc.replace('reduce ', '')
            desc = desc.replace('WEIGHT_DECAY', 'weight_decay')

        improvements.append({
            'trial': i,
            'val_bpb': val,
            'description': desc
        })
    running_best.append(best_val)

# Create simple progress plot
fig, ax = plt.subplots(figsize=(12, 6))

# Plot all trials as dots
trial_nums = list(range(1, len(all_trials) + 1))
all_vals = [t['val_bpb'] for t in all_trials]

# Separate kept vs discarded/failed trials
kept_trials = []
kept_vals = []
failed_trials = []
failed_vals = []

for i, trial in enumerate(all_trials, 1):
    val = trial['val_bpb']
    desc = trial['description']
    # Mark as failed if val is very high (crashed) or if it's a discarded manual trial
    if val > 10 or 'discard' in desc:
        failed_trials.append(i)
        failed_vals.append(val if val < 10 else 1.52)  # Cap display for crashes
    else:
        kept_trials.append(i)
        kept_vals.append(val)

# Plot dots
ax.scatter(kept_trials, kept_vals, s=30, color='#2d3436', alpha=0.6, zorder=3, label='Trials')
ax.scatter(failed_trials, failed_vals, s=30, color='#b2bec3', alpha=0.4, zorder=2, label='Failed/discarded')

# Plot running best line
ax.plot(trial_nums, running_best, 'g-', linewidth=2.5, alpha=0.8, zorder=4, label='Best')

# Annotate key improvements (every other one to avoid clutter)
annotate_indices = [0, 1, 3, 5, 6, 7, 8, 9, 10]  # Select key improvements

# Also annotate best Genetic result (even if not an improvement)
genetic_best_trial = None
genetic_best_val = float('inf')
for i, trial in enumerate(all_trials[51:66], 52):  # Genetic focused range
    if trial['val_bpb'] < genetic_best_val and trial['val_bpb'] < 10:
        genetic_best_val = trial['val_bpb']
        genetic_best_trial = i

if genetic_best_trial:
    ax.annotate(f'Genetic best\n({genetic_best_val:.3f})',
               xy=(genetic_best_trial, genetic_best_val),
               xytext=(genetic_best_trial, genetic_best_val - 0.003),
               fontsize=7,
               ha='center',
               color='#6c5ce7',
               bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='#6c5ce7', alpha=0.8, linewidth=0.8),
               arrowprops=dict(arrowstyle='-', lw=0.8, color='#6c5ce7', alpha=0.6))

for idx in annotate_indices:
    if idx < len(improvements):
        imp = improvements[idx]
        desc = imp['description']

        # Clean up descriptions
        if 'baseline' in desc.lower():
            desc = 'baseline'
        elif 'matrix_lr' in desc.lower():
            desc = desc.replace('increase ', '+').replace('reduce ', '').replace('MATRIX_LR', 'matrix_lr')
        elif 'warmdown' in desc.lower():
            desc = desc.replace('reduce ', '').replace('WARMDOWN_RATIO', 'warmdown')
        elif 'emb_lr' in desc.lower() or 'embedding' in desc.lower():
            desc = desc.replace('increase ', '+').replace('reduce ', '').replace('EMBEDDING_LR', 'emb_lr')
        elif 'weight_decay' in desc.lower():
            desc = desc.replace('reduce ', '').replace('WEIGHT_DECAY', 'wd')
        elif 'Bayesian focused' in desc:
            desc = 'Bayesian opt'
        elif 'Genetic focused' in desc:
            desc = 'Genetic opt'

        if len(desc) > 35:
            desc = desc[:32] + '...'

        # Alternate positioning
        y_offset = 0.010 if idx % 2 == 0 else -0.014
        ha = 'center'

        if imp['trial'] < 8:
            ha = 'left'
        elif imp['trial'] > 68:
            ha = 'right'

        ax.annotate(desc,
                   xy=(imp['trial'], imp['val_bpb']),
                   xytext=(imp['trial'], imp['val_bpb'] + y_offset),
                   fontsize=7.5,
                   ha=ha,
                   bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='#636e72', alpha=0.9, linewidth=0.8),
                   arrowprops=dict(arrowstyle='-', lw=0.8, color='#636e72', alpha=0.6))

# Add baseline reference
baseline = 1.451763
ax.axhline(baseline, color='red', linestyle='--', alpha=0.4, linewidth=1)
ax.text(len(all_trials) * 0.98, baseline + 0.002, 'Baseline',
        ha='right', va='bottom', fontsize=9, color='red')

# Styling
ax.set_xlabel('Trial', fontsize=11)
ax.set_ylabel('val_bpb (lower is better)', fontsize=11)
ax.set_title('Phase 2 Progress: 74 trials, 5 methods', fontsize=12, pad=10)
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.2, linewidth=0.5)
ax.set_ylim(min(running_best) - 0.005, baseline + 0.015)

plt.tight_layout()
plt.savefig('phase2_progress.png', dpi=150, bbox_inches='tight')
print("Saved phase2_progress.png")

# Print summary
print(f"\nTotal trials: {len(all_trials)}")
print(f"Total improvements: {len(improvements)}")
print(f"Best val_bpb: {min(running_best):.6f}")
print(f"Baseline: {baseline:.6f}")
print(f"Improvement: {baseline - min(running_best):.6f} ({(baseline - min(running_best))/baseline*100:.2f}%)\n")

# Build dataframe for parameter exploration
params_data = []

# Load focused optimization results with parameters
for exp_dir in sorted(Path('../experiments').glob('bayesian_focused_*')):
    results_file = exp_dir / 'results.jsonl'
    if results_file.exists():
        with open(results_file) as f:
            for line in f:
                data = json.loads(line)
                entry = data['params'].copy()
                entry['val_bpb'] = data['results']['val_bpb']
                entry['method'] = 'bayesian_focused'
                params_data.append(entry)

for exp_dir in sorted(Path('../experiments').glob('genetic_focused_*')):
    results_file = exp_dir / 'results.jsonl'
    if results_file.exists():
        with open(results_file) as f:
            for line in f:
                data = json.loads(line)
                entry = data['params'].copy()
                entry['val_bpb'] = data['results']['val_bpb']
                entry['method'] = 'genetic_focused'
                params_data.append(entry)

params_df = pd.DataFrame(params_data)

# Create parameter exploration plot
if len(params_df) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {'bayesian_focused': '#3498db', 'genetic_focused': '#9b59b6'}

    # Plot parameter relationships
    param_pairs = [
        ('matrix_lr', 'val_bpb', 'Matrix LR vs Performance'),
        ('embedding_lr', 'val_bpb', 'Embedding LR vs Performance'),
        ('weight_decay', 'val_bpb', 'Weight Decay vs Performance'),
        ('warmdown_ratio', 'val_bpb', 'Warmdown Ratio vs Performance'),
    ]

    for idx, (param_x, param_y, title) in enumerate(param_pairs):
        ax = axes[idx // 2, idx % 2]

        if param_x in params_df.columns:
            for method in ['bayesian_focused', 'genetic_focused']:
                method_data = params_df[params_df['method'] == method]
                if len(method_data) > 0:
                    ax.scatter(method_data[param_x], method_data[param_y],
                              label=method.replace('_', ' ').title(),
                              alpha=0.6, s=100, color=colors[method])

            ax.set_xlabel(param_x.replace('_', ' ').title(), fontweight='bold')
            ax.set_ylabel('Validation BPB', fontweight='bold')
            ax.set_title(title, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('phase2_parameter_exploration.png', dpi=150, bbox_inches='tight')
    print("Saved phase2_parameter_exploration.png")
