# H200 Experiment Plan — Updated with Discussion #43 (2026-03-10)

## Why this update

The previous plan focused on architecture scaling (depth/width) from a 0.9906 baseline. New external evidence from GitHub Discussion #43 shows larger gains came from **optimization/regularization knobs first**, then architecture. This plan reprioritizes accordingly.

## Ground rules

- Keep 5-minute fixed budget protocol unchanged.
- Change one main variable at a time.
- Prefer low-risk, high-signal tweaks before expensive depth/width sweeps.
- Keep known wins unless explicitly testing them.

## Known local baseline (this pod lineage)

- Best seen locally: val_bpb ~0.990610
- Strong local findings: VE on, ReLU^2, HEAD_DIM=128, SSSL, batch 2^18, DEPTH=12

## Strong external priors from #43 (treat as high-value hypotheses)

Most useful, repeatedly positive changes:
1. tiny **VE weight decay** around `0.003`
2. tiny **embedding weight decay** around `0.001`
3. **lm_head weight decay** around `0.01`
4. transformer **init scale** around `0.68x`
5. `FINAL_LR_FRAC = 0.05`
6. warmdown around `0.75`
7. depth/batch regime favoring more steps in 5 min (depth~9 in that run)

Known dead ends from #43 (de-prioritize):
- weight tying
- parallel attn+MLP
- MQA with n_kv_head=1
- larger models that cut step count too hard

## New priority order

### Phase A — Optimization-only ablations on current best local architecture
Start from current best local config (DEPTH=12 etc.) and test these first:

- A0: Re-baseline (no changes)
- A1: add `VE WD = 0.001`
- A2: increase to `VE WD = 0.003` (if A1 helps)
- A3: add `embedding WD = 0.001`
- A4: add `lm_head WD = 0.01`
- A5: set `FINAL_LR_FRAC = 0.05`
- A6: warmdown `0.75`
- A7: init scale multiplier `0.68`

Decision rule:
- keep if improved; discard otherwise
- if two consecutive regressions > +0.0008, revert to last best and continue with next single tweak

### Phase B — Architecture only after Phase A stabilizes
Run prior depth/width experiments **only after** Phase A best config is established:

- B1: DEPTH=14 @ same batch
- B2: DEPTH=12 wider dim (ASPECT_RATIO 75/80)
- B3: DEPTH=16 only if throughput/token efficiency justifies

Interpretation:
- If Phase A gains exceed architecture gains, continue optimizer-first micro-tuning.
- If architecture wins clearly (>0.0015 over Phase A best), branch and retune LR/WD lightly.

## Throughput guardrail

For any larger model test:
- track steps in 5 min and tok/s
- if step collapse is severe and val_bpb degrades >0.002, stop that branch quickly

## Logging discipline

For each run, capture:
- commit
- val_bpb
- peak_vram_mb
- steps completed
- keep/discard + one-line rationale

Continue writing to `results.tsv` exactly as before.

## Immediate next 5 runs (if no blocker)

1. re-baseline current best config
2. + VE WD 0.001
3. VE WD 0.003
4. + embedding WD 0.001
5. + lm_head WD 0.01

Then evaluate trend before touching depth/width again.
