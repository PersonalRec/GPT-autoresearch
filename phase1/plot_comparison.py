#!/usr/bin/env python3
"""
Phase 1 comparison visualization: Agent vs Bayesian vs Genetic
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load data
def load_bayesian():
    data = []
    with open('bayesian/results.jsonl') as f:
        for line in f:
            entry = json.loads(line)
            if entry['results']['status'] == 'success':
                data.append({
                    'trial': entry['trial_number'],
                    'val_bpb': entry['results']['val_bpb'],
                })
    return sorted(data, key=lambda x: x['trial'])

def load_genetic():
    data = []
    with open('genetic/results.jsonl') as f:
        for line in f:
            entry = json.loads(line)
            if entry['results']['status'] == 'success':
                data.append({
                    'trial': entry['trial_number'],
                    'val_bpb': entry['results']['val_bpb'],
                })
    return sorted(data, key=lambda x: x['trial'])

def load_agent():
    # From results.tsv in root
    data = []
    with open('../../results.tsv') as f:
        next(f)  # skip header
        for i, line in enumerate(f, 1):
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                data.append({
                    'trial': i,
                    'val_bpb': float(parts[1]),
                })
    return data

# Load all data
bayesian = load_bayesian()
genetic = load_genetic()
agent = load_agent()

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Trial progression
ax1.plot([d['trial'] for d in agent], [d['val_bpb'] for d in agent],
         'o-', color='#2E86AB', label='Agent', linewidth=2, markersize=8, alpha=0.7)
ax1.plot([d['trial'] for d in bayesian], [d['val_bpb'] for d in bayesian],
         's-', color='#A23B72', label='Bayesian (TPE)', linewidth=2, markersize=8, alpha=0.7)
ax1.plot([d['trial'] for d in genetic], [d['val_bpb'] for d in genetic],
         '^-', color='#F18F01', label='Genetic (CMA-ES)', linewidth=2, markersize=8, alpha=0.7)

ax1.set_xlabel('Trial Number', fontsize=12)
ax1.set_ylabel('Validation BPB (lower is better)', fontsize=12)
ax1.set_title('Phase 1: Optimization Method Comparison', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(1.35, 1.65)

# Plot 2: Distribution (box plot)
data_for_box = [
    [d['val_bpb'] for d in agent],
    [d['val_bpb'] for d in bayesian],
    [d['val_bpb'] for d in genetic],
]
bp = ax2.boxplot(data_for_box, labels=['Agent\n(15 trials)', 'Bayesian\n(6 success)', 'Genetic\n(8 success)'],
                 patch_artist=True, widths=0.6)

# Color the boxes
colors = ['#2E86AB', '#A23B72', '#F18F01']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

# Color the medians
for median in bp['medians']:
    median.set_color('black')
    median.set_linewidth(2)

ax2.set_ylabel('Validation BPB (lower is better)', fontsize=12)
ax2.set_title('Distribution of Results', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# Add best value annotations
best_vals = [min(d['val_bpb'] for d in data) for data in [agent, bayesian, genetic]]
for i, (val, color) in enumerate(zip(best_vals, colors)):
    ax2.text(i+1, val-0.01, f'Best: {val:.3f}',
             ha='center', va='top', fontsize=9, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3))

plt.tight_layout()
plt.savefig('phase1_comparison.png', dpi=300, bbox_inches='tight')
print("Plot saved to phase1_comparison.png")

# Print summary statistics
print("\n" + "="*60)
print("PHASE 1 SUMMARY STATISTICS")
print("="*60)

for name, data, color in [('Agent', agent, '#2E86AB'),
                           ('Bayesian', bayesian, '#A23B72'),
                           ('Genetic', genetic, '#F18F01')]:
    vals = [d['val_bpb'] for d in data]
    print(f"\n{name}:")
    print(f"  Best:   {min(vals):.6f}")
    print(f"  Mean:   {np.mean(vals):.6f}")
    print(f"  Median: {np.median(vals):.6f}")
    print(f"  Std:    {np.std(vals):.6f}")
    print(f"  Trials: {len(vals)}")
