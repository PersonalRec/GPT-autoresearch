# Multi-Ralph Loop

Parallel multi-agent extension of [Karpathy's autoresearch](https://github.com/karpathy/autoresearch). Runs N Claude agents simultaneously on N GPUs with a rotating coordinator protocol — no central supervisor needed.

## Design

The core idea: **whichever agent finishes first becomes the coordinator.** It reads all results, reasons about the search space, and generates the next batch of experiments for other agents to pick up.

```
                    ┌─────────────────────────────────────┐
                    │     multi-ralph/ (shared files)     │
                    │                                     │
                    │  best/train.py    ← global best     │
                    │  strategy.md      ← search plan     │
                    │  results.tsv      ← all results     │
                    │  queue/           ← pending tasks    │
                    │  active/          ← running now      │
                    │  done/            ← completed        │
                    └──────────┬──────────────────────────┘
                               │ symlinked into each
            ┌──────────────────┼──────────────────────┐
            ▼                  ▼                       ▼
    worktrees/gpu0/     worktrees/gpu1/  ...   worktrees/gpu7/
    (independent repo)  (independent repo)     (independent repo)
```

### Why rotating coordinator?

| Approach | Throughput | Intelligence | Complexity |
|----------|-----------|-------------|------------|
| Sequential (1 agent) | 12/hr | High | Low |
| Pure parallel (8 independent) | 96/hr | Low | Low |
| Supervisor + workers | 87/hr | High | High |
| **Rotating coordinator** | **87/hr** | **High** | **Medium** |

The rotating coordinator gets near-full throughput (all GPUs busy) with intelligent search direction (coordinator reasons about full history), without the complexity of a dedicated supervisor process.

### Agent lifecycle

```
Agent finishes experiment
    │
    ├── Report result to results.tsv and done/
    │
    ├── Beat global best? → Update best/train.py + strategy.md
    │
    ├── Queue empty?
    │   ├── YES → Become coordinator:
    │   │         Read ALL results → Reason about search space
    │   │         → Generate 4-8 new tasks → Write to queue/
    │   │         → Pick one yourself → Run it
    │   │
    │   └── NO → Pick next task from queue/ → Run it
    │
    └── Loop forever
```

### Timing

```
t=0min    GPU 0 runs baseline + generates 7 initial tasks
t=6min    GPU 1-7 pick up tasks, all 8 GPUs running
t=11min   First GPU finishes, becomes coordinator, generates next batch
t=12min   All GPUs running again
...
Steady state: ~87 experiments/hour, coordinator rotates naturally
```

## Quick start

```bash
# On your GPU box
git clone <repo> ~/autoresearch
cd ~/autoresearch
uv sync
uv run prepare.py

# V100 only: fix dtype (no BF16 support)
./multi-ralph/fix-v100-dtype.sh

# Verify one GPU works
CUDA_VISIBLE_DEVICES=0 uv run train.py

# Launch all agents
./multi-ralph/launch.sh         # 8 GPUs (default)
./multi-ralph/launch.sh 4       # or fewer
```

## Monitoring

```bash
# Sessions
screen -ls                             # list all
screen -r ralph-gpu0                   # attach (Ctrl+A D detach)

# Live status
cat multi-ralph/results.tsv            # all experiment results
cat multi-ralph/strategy.md            # current search strategy
ls multi-ralph/queue/                  # pending experiments
ls multi-ralph/active/                 # currently running
ls multi-ralph/done/                   # completed reports
tail -f worktrees/gpu*/run.log         # training output

# Dashboard (auto-refresh)
watch -n 30 'head -5 multi-ralph/strategy.md; echo; cat multi-ralph/results.tsv'
```

## Stopping and cleanup

```bash
# Stop all agents
for i in $(seq 0 7); do screen -S ralph-gpu$i -X quit; done

# Remove worktrees
for i in $(seq 0 7); do git worktree remove --force worktrees/gpu$i; done
```

## File structure

```
multi-ralph/
├── README.md               # this file
├── program-multi.md        # agent protocol (the "research org code")
├── launch.sh               # creates worktrees + launches agents
├── fix-v100-dtype.sh       # patches bfloat16→float16 for V100
├── strategy.md             # living search strategy document
├── results.tsv             # append-only experiment log
├── best/
│   └── train.py            # current global best training script
├── queue/                  # pending experiment assignments
│   └── NNN.md              # experiment spec (title, rationale, changes)
├── active/                 # currently running experiments
│   └── gpu{N}.md           # moved from queue/ when claimed
└── done/                   # completed experiment reports
    └── gpu{N}-NNN.md       # result + insight
```

## Adapting to other hardware

- **A100 (single GPU, multiple agents)**: Default config. `./launch.sh 3` runs 3 agents concurrently sharing one GPU (~12GB each). BF16 native.
- **H100 80GB**: Deeper models (depth 12-20) become viable. Edit `strategy.md` prior knowledge section.
- **Multi-GPU**: Edit `launch.sh` to assign `CUDA_VISIBLE_DEVICES=$AGENT` per agent.
- **Fewer agents**: `./launch.sh 2` works fine. Coordinator still rotates.

## Origin

Built on [autoresearch](https://github.com/karpathy/autoresearch) by @karpathy. The "ralph loop" pattern adds persistent memory and intelligent search to the original experiment loop. Multi-ralph extends it to parallel agents with a rotating coordinator protocol.
