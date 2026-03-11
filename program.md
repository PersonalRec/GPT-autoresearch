# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with the extended header row (see Logging Results). The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, etc).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_bpb` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val_bpb.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_bpb gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_bpb improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_bpb improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

Note that the script is configured to always stop after 5 minutes, so depending on the computing platform of this computer the numbers might look different. You can extract the key metrics from the log file:

```
grep "^val_bpb:\|^peak_vram_mb:\|^mfu_percent:\|^total_tokens_M:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 8 columns:

```
commit	val_bpb	memory_gb	mfu_pct	tokens_M	status	pareto	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. MFU percentage (e.g. 39.8) — use 0.0 for crashes
5. total tokens in millions (e.g. 499.6) — use 0.0 for crashes
6. status: `keep`, `discard`, or `crash`
7. pareto: `yes` if this run is Pareto-optimal (see Multi-Objective Tracking below), `no` otherwise
8. short text description of what this experiment tried

Example:

```
commit	val_bpb	memory_gb	mfu_pct	tokens_M	status	pareto	description
a1b2c3d	0.997900	44.0	39.8	499.6	keep	yes	baseline
b2c3d4e	0.993200	44.2	39.5	499.6	keep	yes	increase LR to 0.04
c3d4e5f	1.005000	44.0	41.2	520.3	discard	no	switch to GeLU activation
d4e5f6g	0.000000	0.0	0.0	0.0	crash	no	double model width (OOM)
e5f6g7h	1.000100	38.5	42.1	530.0	keep	yes	depth=6 with wider layers (Pareto: worse bpb but much less memory + higher MFU)
```

## Multi-Objective Pareto Tracking

Track three metrics for every run: **val_bpb** (lower is better), **memory_gb** (lower is better), and **mfu_pct** (higher is better).

A run is **Pareto-optimal** if no other completed run is strictly better on ALL three metrics. In other words, this run is the best known trade-off along at least one dimension.

When logging, mark `pareto: yes` if the run is Pareto-optimal across all completed (non-crash) runs. When a new Pareto-optimal run is recorded, check if any previously marked `pareto: yes` runs are now dominated — if so, mentally note they are no longer on the frontier (but do not modify old rows).

**Why this matters:** A run with 0.001 worse val_bpb but 30% less memory contains a valuable insight. Periodically review the Pareto frontier and consider combining approaches from different Pareto-optimal runs (e.g., take the architecture from the low-memory run and the optimizer settings from the low-bpb run).

## Structured Hypotheses

Before each experiment, formulate and record a hypothesis in your commit message using this format:

```
[HYPOTHESIS] I predict [change X] will [improve/worsen] val_bpb by ~[amount] because [reason].
```

After each run, mentally note whether the hypothesis was confirmed or refuted. Use this to build a running model of the system:

- **Sensitivity map**: Which hyperparameters have the highest impact per unit change?
- **Orthogonality map**: Which successful changes are independent and could be combined?
- **Bottleneck diagnosis**: Is the current limiting factor optimization, architecture, data utilization, or compute efficiency?

Every 10 experiments, review your hypothesis accuracy. If you're consistently wrong about a category of changes (e.g., always overestimating architecture gains), recalibrate. Good calibration compounds — after 20 runs you should be able to predict within 10% which changes will help and by how much.

## Checkpoint & Fork Strategy

Maintain checkpoint tags to enable exploration from multiple starting points:

1. After every 10 experiments, tag the current best commit as `best-N` (e.g., `git tag best-10`, `git tag best-20`).
2. Every 5th experiment, instead of iterating on HEAD, pick a previous `best-N` checkpoint and try a **completely different direction** from there. This prevents getting trapped in a local optimum.
3. If the fork produces a better result than current HEAD, rebase onto the fork. If not, return to HEAD and continue.

This adds population diversity to the search without any extra compute — just git tags and occasional `git checkout`.

## Forced Ablation Protocol

Every 5th experiment MUST be a **simplification**: remove a component, reduce a hyperparameter, delete lines of code, simplify logic. The goal is to find the SIMPLEST code that achieves the current best val_bpb.

Examples of good ablation experiments:
- Remove value embeddings entirely — does performance drop?
- Remove per-layer residual scaling — does it matter?
- Simplify the window pattern from "SSSL" to "L" — any change?
- Reduce DEPTH by 1 — how much does val_bpb degrade?
- Remove cautious weight decay (just use regular) — impact?

If performance is equal or better after removing something, **always keep the simpler version**. Complexity is debt. The best breakthroughs in ML (Transformers, ReLU, dropout) were simplifications.

## Cross-Run Pattern Analysis

Every 20 experiments, pause and analyze `results.tsv` holistically before planning the next batch:

1. **Progress curve**: What's the cumulative improvement from baseline? Is it accelerating or plateauing?
2. **Category analysis**: Group experiments by type (architecture, optimizer, hyperparameter, simplification). Which categories produced the biggest gains? Which consistently failed?
3. **Diminishing returns check**: Are recent improvements getting smaller? If so, it's time for a radical departure — try a fundamentally different approach rather than incremental tuning.
4. **Combination search**: List all pairs of successful changes that haven't been tried together. The next experiments should prioritize these combinations.
5. **Hypothesis calibration**: How accurate were your predictions? Where was your mental model wrong?
6. **Pareto frontier review**: Look at all Pareto-optimal runs. Can you combine insights from the low-memory champion with the low-bpb champion?

Write a brief summary (3-5 sentences) in your commit message prefixed with `[ANALYSIS]` so the human can review your reasoning later.

## Statistical Confirmation

For borderline results — where val_bpb is within **0.5%** of the current best (either better or worse) — run the experiment **a second time** to confirm:

1. First run: record result as `pending_confirm` status internally (log as `discard` in TSV for now).
2. Second run: re-run the exact same code (same commit). Record the second val_bpb.
3. Decision: Use the **average** of both runs. If the average improves on the current best, keep. Otherwise, discard.

This prevents two failure modes:
- **False positive**: A lucky run gets kept and you build on a fluke, compounding the error.
- **False negative**: An unlucky run gets discarded and you miss a genuine improvement.

At the frontier where improvements are small (0.001-0.005 val_bpb), noise dominates. A 10-minute confirmation run (two 5-minute runs) is a small price to pay for confidence.

**Exception**: If a run is clearly better (>1% improvement) or clearly worse (>1% degradation), skip confirmation and decide immediately.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. **Check experiment number**: Is this the 5th experiment? → Forced ablation. Is this the 5th experiment that would fork? → Fork from a checkpoint. Is this the 20th? → Cross-run analysis first.
3. **Formulate hypothesis**: Write your prediction for this experiment before modifying code.
4. Tune `train.py` with an experimental idea by directly hacking the code.
5. git commit (include hypothesis in commit message)
6. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
7. Read out the results: `grep "^val_bpb:\|^peak_vram_mb:\|^mfu_percent:\|^total_tokens_M:" run.log`
8. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
9. **Evaluate against hypothesis**: Were you right? Update your mental model.
10. **Check if Pareto-optimal**: Compare against all previous non-crash runs on val_bpb, memory_gb, and mfu_pct.
11. **Statistical confirmation**: If within 0.5% of best, run again and average.
12. Record the results in the tsv with all 8 columns (NOTE: do not commit the results.tsv file, leave it untracked by git)
13. If val_bpb improved (lower), you "advance" the branch, keeping the git commit
14. If val_bpb is equal or worse BUT the run is Pareto-optimal on another dimension, note it but still advance only if val_bpb improved
15. If val_bpb is equal or worse and not Pareto-interesting, git reset back to where you started
16. **Checkpoint**: Every 10 experiments, tag the current best.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
