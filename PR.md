## Summary

This PR adds a fork-specific execution workflow for autonomous experiments, replacing session-by-session `program.md` interpretation with a deterministic runner plus agent runbook.

### What’s included

- Add `workflows/run_experiment.py` as the single experiment orchestrator:
  - `start`, `resume`, `status` commands
  - top-level stage controls: `setup`, `baseline`, `loop`
  - loop sub-stage controls: `propose`, `apply`, `commit`, `train`, `triage`, `record`, `decide`
  - resumable checkpointing under `workflows/runs/<run_id>/`
  - run-id policy: `<branch-slug>-rNNN`
- Add `AGENTS.md` runbook with explicit natural-language to command mapping for agent sessions.
- Improve observability:
  - `runner.log` (human-readable timeline)
  - `history.jsonl` (structured events)
  - per-iteration artifacts under `workflows/runs/<run_id>/iterations/<NNNN>/`
  - raw stochastic stage traces (`propose/apply/triage` OpenCode outputs)
- Add setup robustness:
  - auto-run `uv run prepare.py` when cache/tokenizer is missing (default on, opt-out via `--no-auto-prepare`)
  - explicit setup precondition checks before baseline/loop
- Add background training support:
  - training stages start in background by default (`--background-train`)
  - `resume` polls/continues in-flight baseline/train jobs
- Update README:
  - OpenCode quickstart instructions
  - fork-specific explanation of why this workflow layer (`run_experiment.py` + `AGENTS.md`) is better than a program.md-only execution style

## Why

In long-running autonomous sessions, prose-only execution is fragile and inconsistent.
This PR makes runs repeatable, resumable, and inspectable while preserving `program.md` as the policy/objective layer.

## Notes

- `results.tsv` remains untracked and is updated as run output.
- `prepare.py` is never modified; only `train.py` is intended to change during experiments.
