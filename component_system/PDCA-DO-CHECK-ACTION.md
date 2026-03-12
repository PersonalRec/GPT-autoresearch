# DCA - Adapt, Check, Action

This document merges the former `PDCA-DO.md`, `PDCA-CHECK.md`, and `PDCA-ACTION.md`
into one execution guide for the merged DCA stage.

## Responsibility
Take the generated plan from P, adapt/fix it in the seed worktree,
run the canonical training entrypoint, evaluate results against baseline, and
promote only when the signal is positive.

## Workspace and paths
Your **current working directory is the seed worktree**. All reads and edits must stay inside this workspace. Use **only paths relative to your cwd**, and treat the copied files under `component_system/` as the canonical context inside the worktree. Do not use or request absolute paths, parent-directory paths, or files outside the workspace; the runner has already set your cwd to the correct worktree.

## Input
- Read the task content embedded in the runner prompt.
- Read current baseline state from `component_system/baseline_branches.json` and `component_system/baseline_metrics.json`.
- Read and edit worktree-local files only.

## Baseline measurement (seed_id __baseline__)
For **baseline measurement** tasks you must **retry until the run completes successfully** and you can report real metrics. Do not report empty metrics and stop.

- If training fails with **CUDA out of memory (OOM)**, the default batch size is tuned for H100. Reduce `device_batch_size` in `component_system/components/trainer.py` (`TrainingSettings`: default `device_batch_size=128`). You may also need to reduce `total_batch_size` so that `total_batch_size % (device_batch_size * sequence_length) == 0` for gradient accumulation. Then rerun the entrypoint until training completes and report the resulting metrics.
- Only trivial execution fixes (e.g. batch size for VRAM) are allowed; do not change model architecture or training logic for baseline.
- **Commit any file changes before reporting.** If you modified files (e.g. reduced batch size), commit those changes on the baseline branch. An uncommitted worktree causes the follow-up merge into the baseline branch to fail.

## Workflow
1. Work in the seed worktree prepared by the system (on the seed branch, one branch per seed).
2. Adapt or fix generated code until it runs cleanly.
3. Run the canonical command (allow at least **600 seconds** so the run is not killed by the execution environment; the first step can take ~150s and training runs for 300s):
   - Preferred (when daemon/root `.venv` is active): `timeout 600 uv run --active component_system/entrypoint.py`
   - Fallback (when no active root `.venv` is available): `timeout 600 uv run component_system/entrypoint.py`
4. If there is a simple bug or OOM, fix (e.g. reduce batch size) and rerun. For baseline measurement, keep retrying until the run succeeds.
5. Commit changes on the seed branch before reporting.
6. Print the DCA summary block for the runner; include the current commit SHA in the JSON so the runner can verify and record it.
7. Let the runner evaluate signal and handle promotion policy.

## Output Format
Print a summary block for the runner. Report metrics in the JSON first; the
runner only falls back to parsing training stdout/stderr when the JSON metrics
are missing:

```text
AUTORESEARCH_DCA_SUMMARY_BEGIN
{"checks":["entrypoint"],"notes":"what you adapted or fixed","completed_at":"YYYY-MM-DD HH:MM:SS","commit_sha":"git sha","metrics":{"val_bpb":1.239972,"training_seconds":300.1,"total_seconds":360.4,"peak_vram_mb":11967.8,"mfu_percent":2.15,"total_tokens_M":140.5,"num_steps":268,"num_params_M":11.5,"depth":4}}
AUTORESEARCH_DCA_SUMMARY_END
```

If you cannot provide final metrics, still print the exact same JSON shape with
`"metrics": {}`.

The runner falls back to extracting metrics from canonical training stdout/stderr:
`val_bpb`, `training_seconds`, `total_seconds`, `peak_vram_mb`, `mfu_percent`,
`total_tokens_M`, `num_steps`, `num_params_M`, and `depth`.

If a DCA run finishes but still reports no metrics, the system does not
immediately mark it failed. Instead, it queues a follow-up DCA recovery task
that inspects the saved stdout/stderr logs and reports the metrics in the same
JSON format. Only if that recovery task still cannot recover metrics is the run
treated as failed.

## Check: Signal Rules

| Condition | Signal |
|-----------|--------|
| `val_bpb` drops >= 0.001 vs baseline | `positive_signal` |
| `val_bpb` rises >= 0.001 vs baseline | `negative_signal` |
| difference < 0.001 | `neutral` |
| no historical baseline `last_val_bpb` | `positive_signal` (first recording) |
| metrics missing or training error | `error` |

The threshold is defined in `component_system/config.py` (`PROMOTION_THRESHOLD`).

## Action: Promotion Rules

Only the DCA (Do-Check-Action) stage may trigger a merge into baseline. The Plan stage must never merge code; the system performs the merge automatically after a successful DCA promotion.

The runner records the DCA `commit_sha` from your summary (or from the current branch HEAD if omitted) for traceability. On positive signal, the workflow merges the seed branch into the baseline. If the merge fails (e.g. conflicts), the system queues a merge-resolution DCA run.

### Promotion flow (`positive_signal` only)
1. The system merges the seed branch into the baseline branch (you do not run merge yourself).
2. The workflow updates `baseline_metrics.json` (and `baseline_branches.json` as needed) with `last_val_bpb`, `promoted_from`, `promoted_idea`, `promoted_at`, `promoted_branch`.
3. Promotion metadata is persisted in seed/run state files.

### Merge failure and conflict resolution
- If the merge into baseline fails (e.g. conflicts), the system queues a **new DCA run** with `merge_resolution: true`.
  - **Normal seed**: In the seed worktree, run `git merge __baseline__` (merge the baseline branch into the seed), resolve conflicts, commit, then print the DCA summary so the system can retry promotion.
  - **Baseline seed (__baseline__)**: The goal is to merge __baseline__ *into* the target branch (e.g. master). Run from the directory that has the target branch checked out (use `git worktree list` to find it), then `git merge __baseline__`. Do *not* run from the __baseline__ worktree and do *not* run `git merge master` there—that would merge master into __baseline__, the wrong direction.

### Non-promotion cases
- `neutral`, `negative_signal`, or `error`: log only, no baseline merge/update.
- Failed run info remains available via queue/state logs.

## Constraints
- Training must use `run_mainline_training` or equivalent for evaluation consistency.
- Evaluation (`val_bpb`) must not be skipped.
- Do not edit `baseline_branches.json` or `baseline_metrics.json` directly; the workflow writes them.
- Only `positive_signal` can trigger promotion.
- Keep `component_system/entrypoint.py` as the canonical runner.
- Rely on git history plus state files for traceability.
