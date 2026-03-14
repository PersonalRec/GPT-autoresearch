# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

AutoAnything — a framework for autonomous optimization via AI agents. Agents propose changes, an evaluator scores them against a black-box metric, and only improvements are kept. Currently configured for the GPT pretraining use case (optimizing val_bpb). Based on [karpathy/nanochat](https://github.com/karpathy/nanochat).

## Repository Structure

```
autoanything/
├── problem.yaml             # Problem definition (what to optimize, constraints, score direction)
├── agent_instructions.md    # Protocol for agents (how to participate)
├── leaderboard.md           # Auto-updated scoreboard (exported by evaluator)
├── state/
│   └── train.py             # MUTABLE — the only file agents modify
├── context/
│   └── prepare.py           # READ-ONLY — constants, data loading, evaluation
└── evaluator/               # GITIGNORED — private scoring code + history DB
    ├── score.sh              # Runs training, extracts metrics as JSON
    ├── evaluate.py           # Serial evaluation loop (poll, score, merge/discard)
    ├── server.py             # Webhook-driven web evaluator (PR-based workflow)
    └── history.db            # SQLite evaluation history (created on first run)
```

## Commands

```bash
uv sync                                    # install dependencies
uv run context/prepare.py                  # one-time: download data shards + train BPE tokenizer
uv run context/prepare.py --num-shards 8   # download fewer shards for testing
uv run state/train.py                      # run a single training experiment (5 min)
uv run state/train.py > run.log 2>&1       # run with output capture
grep "^val_bpb:\|^peak_vram_mb:" run.log   # extract key metrics

# Evaluator (run on the scoring machine, not by agents)
python evaluator/evaluate.py               # start the serial evaluation loop (polls for branches)
python evaluator/evaluate.py --baseline-only  # just establish the baseline score
python evaluator/evaluate.py --push        # push leaderboard updates to origin
python evaluator/server.py                 # start the webhook-driven web evaluator
python evaluator/server.py --push          # web evaluator with auto-push
```

## Architecture

Two files matter for agents:

- **`context/prepare.py`** (READ-ONLY) — Constants (`MAX_SEQ_LEN=2048`, `TIME_BUDGET=300`, `EVAL_TOKENS`), data download, BPE tokenizer training, `Tokenizer` class, `make_dataloader()`, and `evaluate_bpb()`. Data cached in `~/.cache/autoresearch/`. Do not modify.
- **`state/train.py`** (AGENT-EDITABLE) — The only file agents modify. Contains GPT model (`GPTConfig`, `CausalSelfAttention`, `MLP`, `Block`, `GPT`), `MuonAdamW` optimizer (Muon for 2D matrix params, AdamW for everything else), hyperparameters section, and training loop. Uses Flash Attention 3 via `kernels` package.

Key model details in `state/train.py`:
- Model dim derived from depth: `model_dim = DEPTH * ASPECT_RATIO` (rounded to HEAD_DIM multiple)
- Activation: `relu().square()` (ReGLU variant)
- Value Embeddings (ResFormer) on alternating layers with learned gating
- Residual lambdas (`resid_lambdas`, `x0_lambdas`) for per-layer residual stream mixing
- Sliding window attention pattern (`SSSL` = 3 short + 1 long, last layer always long)
- Logit soft-capping at 15

## Agent Protocol (from agent_instructions.md)

1. Pull latest master, create branch: `proposals/<name>/<description>`
2. Read `problem.yaml`, `context/`, and `leaderboard.md` for context
3. Modify ONLY `state/train.py`
4. Commit with a clear message explaining the approach
5. Push the branch — the evaluator picks it up, scores it, merges if improved

## Evaluator Design

- **Serial evaluation**: one proposal at a time, no race conditions
- **Blind scoring**: agents never see `evaluator/` (gitignored)
- **SQLite history**: all evaluations recorded in `evaluator/history.db`
- **Auto-leaderboard**: `leaderboard.md` updated after each evaluation

## Key Constraints

- Training always runs exactly 5 minutes (wall clock, excluding first 10 warmup steps)
- Only packages in `pyproject.toml` are available (PyTorch 2.9.1, kernels, numpy, etc.)
- Requires NVIDIA GPU with CUDA (Flash Attention 3; uses Hopper-specific kernel on H100)
- Fast-fail: training aborts if loss is NaN or >100
- Simplicity criterion: prefer simpler code at equal performance
