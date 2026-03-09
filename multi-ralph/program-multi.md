# Multi-Ralph: Rotating Coordinator Protocol

N agents sharing a single GPU, one shared brain. No central supervisor — whichever agent finishes first becomes the coordinator. All agents train concurrently on the same GPU (~12GB VRAM each).

## Architecture

```
worktrees/gpu{N}/          # each agent's independent repo copy
  ├── train.py             # edit freely
  ├── run.log              # training output
  └── multi-ralph -> ...   # symlink to shared dir

multi-ralph/               # shared coordination (all agents read/write)
  ├── best/train.py        # current global best
  ├── results.tsv          # append-only, all agents
  ├── strategy.md          # search strategy (updated by coordinator)
  ├── queue/               # pre-generated experiment assignments
  │   ├── 001.md           # unclaimed experiment
  │   └── 002.md           # unclaimed experiment
  ├── active/              # what's running now
  │   └── gpu3.md          # GPU 3's current experiment
  └── done/                # completed experiment reports
      └── gpu3-001.md      # GPU 3's result for task 001
```

## Agent Lifecycle

You are **GPU {N}**. On every iteration:

### 1. Check for assignment

```bash
ls multi-ralph/queue/*.md 2>/dev/null
```

**If a queued task exists:** pick the lowest-numbered one, move it to `active/gpu{N}.md`, and run it.

**If queue is empty:** you become the coordinator (see below).

### 2. Run the experiment

```bash
cp multi-ralph/best/train.py train.py    # always start from global best
# apply the change described in your assignment
# commit
uv run train.py > run.log 2>&1
grep "^val_bpb:\|^peak_vram_mb:" run.log
```

### 3. Report results

Write a result file:
```bash
# multi-ralph/done/gpu{N}-{task_number}.md
# Contains: experiment description, val_bpb, whether it improved, key insight
```

Append to `results.tsv`:
```
<commit>	<val_bpb>	<memory_gb>	<status>	gpu{N}: <description>
```

### 4. Update global best (if improved)

If your val_bpb beats the current best in `strategy.md`:
```bash
cp train.py multi-ralph/best/train.py
```
Update `strategy.md` with new best val_bpb.

If NOT improved:
```bash
cp multi-ralph/best/train.py train.py
```

### 5. Clean up

```bash
rm multi-ralph/active/gpu{N}.md
```

### 6. Become coordinator (if queue is empty)

This is the key step. When no queued tasks exist, you take 30 seconds to think and plan.

## The Coordinator Role

When you become coordinator, you do THREE things:

### A. Read everything

```
multi-ralph/strategy.md          # current search strategy + best result
multi-ralph/results.tsv          # full history from ALL agents
multi-ralph/done/*.md            # detailed reports with insights
ls multi-ralph/active/           # what's still running
```

### B. Reason about the search

Think about:
- What directions have been explored? What's untouched?
- What patterns emerge? (e.g., "all LR increases helped", "architecture changes crash")
- What's the most promising next direction given ALL evidence?
- Are we in exploration mode (try diverse things) or exploitation mode (refine what works)?
- What combinations of individual wins haven't been tested together?

### C. Generate the next batch

Write 2-4 experiment files to `multi-ralph/queue/`:

```markdown
# multi-ralph/queue/015.md

## Experiment: RoPE base frequency 50000

### Rationale
Three LR experiments all improved. Architecture is stabilized.
RoPE changes helped on H100 leaderboard (3 successive increases kept).
High confidence this transfers.

### Changes to train.py
In `_precompute_rotary_embeddings`, change `base=10000` to `base=50000`.

### Expected outcome
~0.001-0.003 improvement based on H100 results.
```

**Numbering:** look at existing files in `queue/` and `done/` to find the next number.

**Task design principles:**
- First 24 experiments (3 rounds): **explore broadly** — one experiment per category (LR, schedule, architecture, RoPE, window, optimizer)
- After that: **exploit** — combine winners, push promising directions further
- Always include 1-2 "wild card" experiments per batch — creative ideas not on any standard list
- If a direction had 3+ failures, stop exploring it
- If a direction had 2+ successes, assign 2 experiments to push it further

### D. Update strategy.md

```markdown
# Search Strategy

## Current best: 0.9852 (commit abc1234, gpu3, experiment 12)

## Phase: exploitation (switched from exploration at experiment 24)

## What works (high confidence)
- Higher LRs (0.04→0.08 confirmed by gpu0, gpu2, gpu4)
- RoPE 50K (gpu1, gpu5 both found improvement)

## What fails (avoid)
- Depth > 9 (OOM on V100 — gpu3, gpu6, gpu7 all crashed)
- Warmup (hurts on short schedules — gpu0, gpu2)

## Untested
- Weight decay reduction
- Softcap changes
- SwiGLU activation

## Next strategy
Combine LR 0.08 + RoPE 50K as new baseline.
Then test schedule changes on top of that stack.
```

## Experiment File Format

### Queue file (multi-ralph/queue/NNN.md)
```markdown
## Experiment: <title>

### Rationale
Why this experiment, given what we know.

### Changes to train.py
Specific code changes to make.

### Expected outcome
What we expect and why.
```

### Active file (multi-ralph/active/gpu{N}.md)
Copy of the queue file, renamed. Just move it:
```bash
mv multi-ralph/queue/015.md multi-ralph/active/gpu3.md
```

### Done file (multi-ralph/done/gpu{N}-NNN.md)
```markdown
## Experiment: <title>

### Result
- val_bpb: 0.9912
- peak_vram_mb: 14200
- status: keep/discard/crash

### Insight
What we learned. One sentence.
```

## Startup Protocol

The very first agent to run does initial setup:

1. Run baseline: `uv run train.py > run.log 2>&1`
2. Record baseline in `results.tsv`
3. Write initial `strategy.md` with baseline val_bpb
4. Generate first batch of 8 experiments to `queue/` (one per GPU, diverse categories)

Other agents wait for queue files to appear (check every 10 seconds):
```bash
while [ -z "$(ls multi-ralph/queue/*.md 2>/dev/null)" ]; do sleep 10; done
```

## Timing

```
Round 1 (t=0-6min):
  GPU 0: runs baseline + generates first 8 tasks (6 min)
  GPU 1-7: waiting for queue

Round 2 (t=6-11min):
  GPU 0-7: all running experiments from queue (5 min each)

Round 3 (t=11-17min):
  First GPU to finish becomes coordinator
  Reads 1-8 results, generates next batch
  Other GPUs pick up tasks as they finish

Steady state (~t=20min+):
  All 8 GPUs running continuously
  Coordinator role rotates to whoever finishes first when queue is empty
  ~8 experiments per 5.5 minutes = ~87/hour
```

## Conflict Handling

**Queue contention:** Two GPUs grab the same queue file. Unlikely (filesystem move is atomic on Linux). If it happens, one gets an error — just pick the next file.

**Simultaneous coordinators:** Two GPUs find empty queue at the same time. Both generate tasks. That's fine — more tasks in the queue just means more options. Slightly redundant but not harmful.

**Best train.py race:** Two GPUs both beat the current best. The last `cp` wins. The "loser" will re-test their change on the next round (since they sync from best/ at the start). If it still helps on top of the new best, it'll be kept.

## A100 Single-GPU Sharing

All agents share CUDA_VISIBLE_DEVICES=0. Each training run uses ~7-13GB VRAM. With 3 concurrent agents = ~21-39GB total, fits comfortably in A100 40/80GB. BF16 native — no dtype hacks needed.

**VRAM budget:** Keep each experiment under ~20GB. No depth > 12 or batch > 128.

## NEVER STOP

Run experiments forever. When coordinator, generate good tasks. When worker, execute faithfully. Never ask the human. If confused, re-read strategy.md and results.tsv — the answer is in the data.
