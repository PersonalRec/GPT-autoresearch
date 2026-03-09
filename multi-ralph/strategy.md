# Search Strategy

## Current best: (pending baseline)

## Phase: setup

## Prior knowledge

### From H100 leaderboard (15 improvements, 0.998 → 0.977 BPB)
- Halving batch (more steps) helps
- More warmdown (0.7) helps
- 5% warmup helps
- Depth 9 / aspect ratio 57 helps
- x0_lambda init 0.05 helps
- Unembedding LR 0.008 helps
- SSSSL window pattern + shorter windows help
- Embedding LR 0.8 helps
- RoPE base frequency 50K→100K→200K helps

### From Nigel RTX 4070 Ti (15 experiments, 1.193 → 1.158 BPB)
- Higher LRs help A LOT on short step budgets (0.04→0.08)
- Warmup HURTS with few steps (opposite of H100)
- Less warmdown better with few steps (opposite of H100)
- Depth > 8 OOMs at 12GB
- Batch 64 worse than 32 (unexpected — same total batch)
- Smaller total batch (2**18) worse — noisier gradients
- SwiGLU being tested (in progress on nigel)

### A100 SXM4 expectations
- Steps per 5min: ~600-950 (faster than nigel 170, close to H100)
- 40-80GB VRAM total, but sharing 3 processes — budget ~20GB each
- BF16 native — no dtype hacks needed
- VRAM constraint: keep depth ≤ 12, batch ≤ 128
- Expect schedule tuning closer to H100 optima (more steps per run)

## What works (confirmed)
(pending experiments)

## What fails (avoid)
(pending experiments)

## Untested
Everything — awaiting baseline.

## Next strategy
1. Run baseline on A100
2. Generate first diverse batch: one each from LR, schedule, architecture
3. After round 1: identify which category gives biggest gains, double down
