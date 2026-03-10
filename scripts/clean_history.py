#!/usr/bin/env python3
"""Reset local autoresearch history/runtime artifacts.

Actions:
1) Checkout main branch (configurable)
2) Remove all extra git worktrees
3) Delete all local branches except main
4) Clear component_system runtime state/history folders
5) Remove .pytest_cache, __pycache__, and results.tsv
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def run_git(args: list[str], cwd: Path, dry_run: bool = False) -> list[str]:
    cmd = ["git", *args]
    if dry_run:
        print(f"[dry-run] {' '.join(cmd)}")
        return []
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return [line for line in proc.stdout.splitlines() if line.strip()]


def is_broken_worktree_remove_error(error: RuntimeError) -> bool:
    msg = str(error)
    return (
        "worktree remove --force" in msg
        and "validation failed, cannot remove working tree" in msg
        and ".git' does not exist" in msg
    )


def remove_children(path: Path, dry_run: bool = False) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        if dry_run:
            print(f"[dry-run] remove {child}")
            continue
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


def remove_pycache_dirs(repo_root: Path, dry_run: bool = False) -> None:
    for pycache in repo_root.rglob("__pycache__"):
        parts = set(pycache.parts)
        if ".venv" in parts or ".git" in parts:
            continue
        if pycache.is_dir():
            if dry_run:
                print(f"[dry-run] remove {pycache}")
            else:
                shutil.rmtree(pycache, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean local branches/worktrees and runtime history.")
    parser.add_argument("--main-branch", default="master", help="Branch to keep. Default: main")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without changing anything")
    args = parser.parse_args()

    repo_root = Path.cwd().resolve()
    print(f"Repository: {repo_root}")

    print("Verifying git repository...")
    run_git(["rev-parse", "--is-inside-work-tree"], cwd=repo_root, dry_run=args.dry_run)

    print(f"Checking out '{args.main_branch}'...")
    run_git(["checkout", args.main_branch], cwd=repo_root, dry_run=args.dry_run)

    print("Removing extra worktrees...")
    run_git(["worktree", "prune"], cwd=repo_root, dry_run=args.dry_run)
    wt_lines = run_git(["worktree", "list", "--porcelain"], cwd=repo_root, dry_run=args.dry_run)
    worktrees: list[Path] = []
    for line in wt_lines:
        if line.startswith("worktree "):
            worktrees.append(Path(line[len("worktree ") :]).resolve())

    for wt in worktrees:
        if wt != repo_root:
            print(f"  - removing worktree {wt}")
            try:
                run_git(["worktree", "remove", "--force", str(wt)], cwd=repo_root, dry_run=args.dry_run)
            except RuntimeError as error:
                if not is_broken_worktree_remove_error(error):
                    raise
                print(f"    ! stale/broken worktree metadata detected, deleting directory: {wt}")
                if args.dry_run:
                    print(f"[dry-run] remove {wt}")
                else:
                    shutil.rmtree(wt, ignore_errors=True)
    run_git(["worktree", "prune"], cwd=repo_root, dry_run=args.dry_run)

    print(f"Deleting local branches except '{args.main_branch}'...")
    branches = run_git(
        ["for-each-ref", "--format=%(refname:short)", "refs/heads"],
        cwd=repo_root,
        dry_run=args.dry_run,
    )
    for branch in branches:
        if branch != args.main_branch:
            print(f"  - deleting branch {branch}")
            run_git(["branch", "-D", branch], cwd=repo_root, dry_run=args.dry_run)

    print("Clearing component-system runtime/history artifacts...")
    history_root = repo_root / "component_system" / "history"
    for name in ("state", "queue", "worktrees", "logs"):
        remove_children(history_root / name, dry_run=args.dry_run)

    pytest_cache = repo_root / ".pytest_cache"
    if pytest_cache.exists():
        if args.dry_run:
            print(f"[dry-run] remove {pytest_cache}")
        else:
            shutil.rmtree(pytest_cache, ignore_errors=True)

    results_tsv = repo_root / "results.tsv"
    if results_tsv.exists():
        if args.dry_run:
            print(f"[dry-run] remove {results_tsv}")
        else:
            results_tsv.unlink(missing_ok=True)

    print("Removing __pycache__ directories...")
    remove_pycache_dirs(repo_root, dry_run=args.dry_run)

    print("Done.")
    print("Remaining branches:")
    for branch in run_git(["branch", "--format=%(refname:short)"], cwd=repo_root, dry_run=args.dry_run):
        print(f"  {branch}")


if __name__ == "__main__":
    main()
