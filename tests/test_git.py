"""Tests for autoanything.git — git operations.

The git module wraps subprocess calls to git. These tests verify
branch listing, filtering, and merge operations use configurable
values (base branch, proposal pattern) instead of hardcoded constants.
"""

import os
import subprocess

import pytest

from autoanything.git import (
    git,
    get_proposal_branches,
    get_head_commit,
    get_commit_message,
)


@pytest.fixture
def git_repo(tmp_path):
    """A real git repo with a few commits and branches for testing."""
    repo = tmp_path / "repo"
    repo.mkdir()

    def _git(*args):
        return subprocess.run(
            ["git"] + list(args),
            capture_output=True, text=True, cwd=str(repo), check=True,
        )

    _git("init", "-b", "main")
    _git("config", "user.email", "test@test.com")
    _git("config", "user.name", "Test")
    (repo / "file.txt").write_text("hello\n")
    _git("add", "file.txt")
    _git("commit", "-m", "initial commit")

    # Create some proposal branches
    _git("checkout", "-b", "proposals/agent/improve-score")
    (repo / "file.txt").write_text("improved\n")
    _git("add", "file.txt")
    _git("commit", "-m", "improve the score by 10%")
    _git("checkout", "main")

    _git("checkout", "-b", "proposals/bot/try-new-approach")
    (repo / "file.txt").write_text("new approach\n")
    _git("add", "file.txt")
    _git("commit", "-m", "try a completely different strategy")
    _git("checkout", "main")

    # A non-proposal branch
    _git("checkout", "-b", "feature/unrelated")
    _git("checkout", "main")

    return repo


class TestGitHelper:
    """The git() helper runs git commands in a specified directory."""

    def test_git_returns_output(self, git_repo):
        result = git("rev-parse", "HEAD", cwd=str(git_repo))
        assert len(result.stdout.strip()) == 40  # SHA-1 hex

    def test_git_raises_on_failure(self, git_repo):
        with pytest.raises(subprocess.CalledProcessError):
            git("checkout", "nonexistent-branch", cwd=str(git_repo))

    def test_git_check_false_no_raise(self, git_repo):
        result = git("checkout", "nonexistent-branch", cwd=str(git_repo), check=False)
        assert result.returncode != 0


class TestGetProposalBranches:
    """Branch listing respects configurable proposal pattern."""

    def test_lists_local_proposal_branches(self, git_repo):
        branches = get_proposal_branches(
            cwd=str(git_repo), pattern="proposals/*",
        )
        names = [b for b in branches]
        assert any("improve-score" in b for b in names)
        assert any("try-new-approach" in b for b in names)

    def test_excludes_non_proposal_branches(self, git_repo):
        branches = get_proposal_branches(
            cwd=str(git_repo), pattern="proposals/*",
        )
        assert not any("feature/unrelated" in b for b in branches)

    def test_custom_pattern(self, git_repo):
        """A different pattern like 'feature/*' should match different branches."""
        branches = get_proposal_branches(
            cwd=str(git_repo), pattern="feature/*",
        )
        assert any("unrelated" in b for b in branches)


class TestGetHeadCommit:
    """Get the current HEAD SHA."""

    def test_returns_sha(self, git_repo):
        sha = get_head_commit(cwd=str(git_repo))
        assert len(sha) == 40
        assert all(c in "0123456789abcdef" for c in sha)


class TestGetCommitMessage:
    """Get the first line of a commit message."""

    def test_returns_message(self, git_repo):
        sha = get_head_commit(cwd=str(git_repo))
        msg = get_commit_message(sha, cwd=str(git_repo))
        assert msg == "initial commit"
