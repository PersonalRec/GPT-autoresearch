# Search Strategy

## Current best: 1.179734 (x0_lambda=0.05 + matrix_lr=0.08 + RoPE 50K, agent2 task 013, 166 steps)
## Solo baseline: 1.094873 (355 steps, no contention)
## Concurrent baseline: 1.258207 (141 steps, 3 agents sharing GPU)

## CRITICAL: Concurrent Comparison
All experiments run with ~120-177 steps due to 3 agents sharing GPU.
Compare to CONCURRENT baseline (1.258), NOT solo baseline (1.095).
Step count varies 120-177 depending on contention — adds noise.

## Phase: exploitation (refining current best)

## Hardware constraints (DO NOT VIOLATE)
- A100 SXM4 40GB, 3 agents sharing 1 GPU concurrently
- DEVICE_BATCH_SIZE = 32 ALWAYS. Never change this.
- TOTAL_BATCH_SIZE = 2**19 ALWAYS. Never change this.
- Max depth: 10 (VRAM constraint)
- BF16 native, no dtype hacks

## Rankings (vs concurrent baseline 1.258) — 18 experiments done
1. x0_lambda 0.05 + matrix_lr 0.08 + RoPE 50K: 1.180 (-0.078) *** BEST ***
2. x0_lambda 0.05 alone: 1.181 (-0.077)
3. best + warmdown 0.3: 1.197 (-0.061)
4. x0_lambda 0.05 + matrix_lr 0.08: 1.201 (-0.057)
5. matrix_lr 0.08 (old baseline): 1.207 (-0.051)
6. warmdown 0.3 (old baseline): 1.208 (-0.050)
7. x0_lambda 0.05 + warmdown 0.3: 1.212 (-0.046)
8. RoPE 50K (old baseline): 1.223 (-0.035)
9. weight_decay 0.05 + x0_lambda: 1.240 (-0.018)
10. embedding 0.8 + unembed 0.008: 1.242 (-0.016)
11. softcap 30 + x0_lambda: 1.242 (-0.016)
12. adam_beta1 0.9 + x0_lambda: 1.252 (-0.006)
13. x0_lambda + RoPE 50K: 1.253 (-0.005)
14. Concurrent baseline: 1.258 (reference)
15. Depth 9/AR 57: 1.259 (neutral)
16. SSSSL window: 1.280 (+0.022)
17. Warmdown 0.7: 1.333 (+0.075)
18. Lower LRs (half): 1.362 (+0.104)

## What works (confirmed)
- x0_lambda 0.05 (strongest single change)
- Triple combo x0+lr+rope (marginal best, may be noise)
- Warmdown 0.3 (helps alone, not on top of best)

## What fails (avoid)
- Warmdown 0.3 ON TOP of current best (1.197, worse)
- Warmdown > 0.5 (catastrophic)
- Lower LRs (terrible)
- SSSSL window, softcap 30, adam beta1 0.9 (no help)

## Key observations
- x0_lambda=0.05 is the one clear winner
- Step count noise (~120-177) makes small differences unreliable
- Untested on current best: weight_decay, FINAL_LR_FRAC, scalar_lr, embedding_lr

## Queued (experiments 018-021)
- 018: Weight decay 0.05 on current best
- 019: FINAL_LR_FRAC 0.1 on current best
- 020: Warmdown 0.2 on current best
- 021: Scalar LR 1.0 + embedding LR 0.8 on current best

## Next strategy
Refine current best with small hyperparameter tweaks on untested dimensions.
If none improve, try creative approaches: resid_lambda tweaks, different init, head_dim changes.
