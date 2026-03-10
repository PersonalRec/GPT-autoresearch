# Test Time RL Discover + Auto Research

![teaser](progress.png)

This repo is a focused fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch) that replaces the ad hoc outer experimentation loop with [TTT-Discover](https://github.com/test-time-training/discover).

The core idea is:

- The **inner loop** is still `autoresearch`: edit `train.py`, run a fixed-budget training job, measure `val_bpb`.
- The **outer loop** is now **test-time RL** from TTT-Discover.
- The outer model proposes full replacements for `train.py`.
- The resulting inner-loop metric improvement becomes the reward used to update the outer model online.

This keeps the original spirit of autoresearch, but makes the search policy itself train during the run.

## Credits

This project is derived from:

- [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
- [Learning to Discover at Test Time](https://arxiv.org/abs/2601.16175)
- [test-time-training/discover](https://github.com/test-time-training/discover)

The RL optimization recipe is intended to stay with upstream `discover`; this repo mainly provides the autoresearch-specific environment, reward, runner, and usage wrapper.

## What This Repo Does

The repo has two layers:

1. **Inner optimization target**
   - `prepare.py` downloads data and trains the tokenizer.
   - `train.py` is the only file the outer model edits.
   - `val_bpb` is the optimization metric. Lower is better.

2. **Outer TTT-Discover loop**
   - `run_ttt_discover.py` launches the test-time RL run.
   - `ttt_autoresearch/` adapts autoresearch to the `discover` environment interface.
   - Each candidate `train.py` is executed in an isolated workspace.
   - Reward is computed from `current_best_val_bpb - candidate_val_bpb`.

## Repository Layout

```text
prepare.py                  Fixed data prep and runtime utilities
train.py                    Inner training program edited by the outer model
program.md                  Human-authored research instructions/context
run_ttt_discover.py         Main TTT-Discover entrypoint
ttt_autoresearch/           Adapter layer for environment, reward, runner, config
configs/                    Ready-to-run YAML config
tests/                      Smoke and unit coverage for the adapter
```

## How The RL Loop Works

At each outer-loop step:

1. TTT-Discover samples a group of candidate `train.py` replacements.
2. Each candidate is evaluated by running a real autoresearch training job.
3. The resulting `val_bpb` is parsed from the run logs.
4. Reward is computed from improvement over the current best state.
5. Upstream `discover` performs the online RL update.
6. If a candidate improves `val_bpb`, it becomes the new best `train.py`.

Important details:

- The **action** is the full replacement contents of `train.py`.
- The **reward** is the inner-loop metric outcome, not the patch text.
- The implementation keeps grouped rollouts for the upstream entropic advantage recipe.
- Inner evaluations are serialized by default with `max_concurrent_evaluations: 1` so multiple full training jobs do not fight over the same GPU.

## Quick Start

**Requirements**

- Linux
- A single NVIDIA GPU
- Python 3.11+
- [uv](https://docs.astral.sh/uv/)

Install and prepare the base autoresearch environment:

```bash
# 1. Install dependencies
uv sync

# 2. Download data and train the tokenizer
uv run prepare.py

# 3. Sanity check the original inner loop
uv run train.py
```

Then launch the outer TTT-Discover loop:

```bash
uv run python run_ttt_discover.py --config configs/ttt_discover_autoresearch.yaml
```

## Default Configuration

The default config lives at [configs/ttt_discover_autoresearch.yaml](configs/ttt_discover_autoresearch.yaml).

Current defaults:

- `model_name: Qwen/Qwen3.5-35B-A3B`
- `samples_per_step: 8`
- `max_steps: 8`
- `temperature: 1.0`
- `max_concurrent_evaluations: 1`

That means the RL loop samples grouped candidates for the upstream TTT recipe, but only one full inner autoresearch training run executes at a time on the local machine.

## Model and Renderer Configuration

The model is configurable, but the prompt/response format must match a supported renderer.

Known-good renderer values:

- `qwen3`
- `qwen3_instruct`
- `gpt_oss_no_sysprompt`
- `gpt_oss_low_reasoning`
- `gpt_oss_medium_reasoning`
- `gpt_oss_high_reasoning`

Examples:

```yaml
model_name: Qwen/Qwen3.5-35B-A3B
renderer_name: qwen3
```

```yaml
model_name: openai/gpt-oss-120b
renderer_name: gpt_oss_high_reasoning
```

If you use an unknown model family, you should set `renderer_name` explicitly. The config now fails fast if it cannot infer a compatible renderer.

## Output Artifacts

Each run writes artifacts under `runs/<timestamp>/`:

- `baseline.json`
  - baseline execution metadata for the original `train.py`
- `resolved_config.json`
  - the fully resolved runtime config
- `history.jsonl`
  - one line per evaluated candidate
- `best/train.py`
  - the current best discovered inner-loop program
- `best/metrics.json`
  - the best run metadata and metric
- `candidates/`
  - isolated workspaces with stdout/stderr and per-candidate files
- `discover_log/`
  - upstream sampler/checkpoint/log state from `ttt-discover`

## Inner Loop Assumptions

This repo intentionally keeps the inner autoresearch setup small:

- `prepare.py` remains fixed.
- `train.py` is the only file the outer model edits.
- Training still uses the original fixed wall-clock budget from autoresearch.
- `val_bpb` remains the optimization target because it is stable across vocabulary and architecture changes.

## Design Choices

### Why only `train.py`?

Because that matches the original autoresearch framing and keeps the action space bounded. It also makes it easier to attribute reward to specific inner-loop changes.

### Why grouped rollouts?

Because upstream `discover` uses grouped rollouts for its entropic advantage estimation and reuse behavior. This repo keeps that outer-loop recipe.

### Why serialize inner evaluations?

Because unlike some upstream `discover` tasks, each rollout here is an actual GPU training job. Running several `train.py` jobs concurrently on one GPU would distort the reward surface and often fail operationally.

## Plain AutoResearch Mode Still Works

This fork does not remove the original autoresearch workflow. You can still use it directly:

```bash
uv run prepare.py
uv run train.py
```

The TTT-Discover path is an additional outer loop, not a replacement for the inner codebase.

## Current Readiness

What is tested locally:

- config loading and override behavior
- reward mapping
- candidate parsing
- environment prompt and state flow
- CLI wiring into upstream `discover`
- serialization of inner evaluations

What is still environment-dependent:

- a true end-to-end production run on the target Linux/CUDA machine
- provider-specific model serving details
- real-world throughput and stability under long TTT sessions

So the repo is structurally ready for the intended setup, but final operational confidence still comes from a real GPU run on the target hardware.

## License

MIT
