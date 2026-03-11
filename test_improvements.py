"""
Test suite for the 6 autoresearch 10x improvements.

These tests validate the logic that the agent should follow when executing
the improved program.md. They test the decision-making functions — Pareto
tracking, hypothesis formatting, checkpoint scheduling, ablation enforcement,
cross-run analysis, and statistical confirmation — using simulated experiment
data without requiring a GPU or actual training runs.

Usage:
    python test_improvements.py
    python -m pytest test_improvements.py -v
"""

import csv
import io
import math
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Optional


# ============================================================================
# Core data structures
# ============================================================================

@dataclass
class ExperimentResult:
    """A single experiment result row from results.tsv."""
    commit: str
    val_bpb: float
    memory_gb: float
    mfu_pct: float
    tokens_M: float
    status: str  # keep, discard, crash
    pareto: str  # yes, no
    description: str


@dataclass
class Hypothesis:
    """A structured hypothesis for an experiment."""
    change: str
    direction: str  # improve, worsen
    predicted_delta: float
    reason: str

    def format(self) -> str:
        return (
            f"[HYPOTHESIS] I predict {self.change} will {self.direction} "
            f"val_bpb by ~{self.predicted_delta} because {self.reason}."
        )

    @classmethod
    def parse(cls, text: str) -> Optional["Hypothesis"]:
        pattern = (
            r"\[HYPOTHESIS\] I predict (.+?) will (improve|worsen) "
            r"val_bpb by ~([\d.]+) because (.+)\."
        )
        m = re.search(pattern, text)
        if not m:
            return None
        return cls(
            change=m.group(1),
            direction=m.group(2),
            predicted_delta=float(m.group(3)),
            reason=m.group(4),
        )


# ============================================================================
# 1. Multi-Objective Pareto Tracking
# ============================================================================

def is_dominated(candidate: ExperimentResult, other: ExperimentResult) -> bool:
    """Return True if `other` strictly dominates `candidate` on all 3 objectives.

    Objectives: val_bpb (lower better), memory_gb (lower better), mfu_pct (higher better).
    `other` dominates `candidate` if other is <= on val_bpb, <= on memory, >= on mfu,
    AND strictly better on at least one.
    """
    better_or_equal = (
        other.val_bpb <= candidate.val_bpb
        and other.memory_gb <= candidate.memory_gb
        and other.mfu_pct >= candidate.mfu_pct
    )
    strictly_better = (
        other.val_bpb < candidate.val_bpb
        or other.memory_gb < candidate.memory_gb
        or other.mfu_pct > candidate.mfu_pct
    )
    return better_or_equal and strictly_better


def is_pareto_optimal(candidate: ExperimentResult, all_results: list[ExperimentResult]) -> bool:
    """Return True if candidate is not dominated by any other result."""
    for other in all_results:
        if other is candidate or other.status == "crash":
            continue
        if is_dominated(candidate, other):
            return False
    return True


def compute_pareto_frontier(results: list[ExperimentResult]) -> list[ExperimentResult]:
    """Return the subset of results that are Pareto-optimal."""
    non_crash = [r for r in results if r.status != "crash"]
    return [r for r in non_crash if is_pareto_optimal(r, non_crash)]


def test_pareto_basic_dominance():
    """Test: A run that is worse on ALL metrics is not Pareto-optimal."""
    results = [
        ExperimentResult("a1b2c3d", 0.990, 40.0, 40.0, 500.0, "keep", "yes", "baseline"),
        ExperimentResult("b2c3d4e", 1.010, 45.0, 35.0, 480.0, "keep", "no", "worse on everything"),
    ]
    frontier = compute_pareto_frontier(results)
    assert len(frontier) == 1
    assert frontier[0].commit == "a1b2c3d"
    print("  PASS: basic dominance — dominated run excluded from frontier")


def test_pareto_tradeoff_preserved():
    """Test: Two runs trading off val_bpb vs memory are both Pareto-optimal."""
    results = [
        ExperimentResult("aaa", 0.990, 44.0, 39.0, 500.0, "keep", "yes", "best bpb, high mem"),
        ExperimentResult("bbb", 1.000, 35.0, 39.0, 500.0, "keep", "yes", "worse bpb, low mem"),
    ]
    frontier = compute_pareto_frontier(results)
    assert len(frontier) == 2
    print("  PASS: tradeoff preserved — both runs on frontier")


def test_pareto_crash_excluded():
    """Test: Crashed runs are never Pareto-optimal."""
    results = [
        ExperimentResult("aaa", 0.990, 44.0, 39.0, 500.0, "keep", "yes", "good run"),
        ExperimentResult("bbb", 0.000, 0.0, 0.0, 0.0, "crash", "no", "OOM crash"),
    ]
    frontier = compute_pareto_frontier(results)
    assert len(frontier) == 1
    assert frontier[0].commit == "aaa"
    print("  PASS: crash excluded from frontier")


def test_pareto_three_way_frontier():
    """Test: Three runs each best on one dimension form a 3-point frontier."""
    results = [
        ExperimentResult("aaa", 0.980, 50.0, 35.0, 500.0, "keep", "yes", "best bpb"),
        ExperimentResult("bbb", 1.010, 30.0, 38.0, 500.0, "keep", "yes", "best memory"),
        ExperimentResult("ccc", 1.000, 45.0, 45.0, 500.0, "keep", "yes", "best mfu"),
    ]
    frontier = compute_pareto_frontier(results)
    assert len(frontier) == 3
    print("  PASS: three-way frontier — each run best on one dimension")


def test_pareto_new_run_displaces():
    """Test: A new run that dominates an existing frontier member removes it."""
    results = [
        ExperimentResult("aaa", 0.990, 44.0, 39.0, 500.0, "keep", "yes", "old best"),
        ExperimentResult("bbb", 0.985, 43.0, 40.0, 510.0, "keep", "yes", "new best, better on all"),
    ]
    frontier = compute_pareto_frontier(results)
    assert len(frontier) == 1
    assert frontier[0].commit == "bbb"
    print("  PASS: new run displaces dominated frontier member")


# ============================================================================
# 2. Structured Hypotheses
# ============================================================================

def validate_hypothesis_format(commit_message: str) -> bool:
    """Check that a commit message contains a properly formatted hypothesis."""
    return Hypothesis.parse(commit_message) is not None


def evaluate_hypothesis_accuracy(
    hypothesis: Hypothesis,
    baseline_bpb: float,
    actual_bpb: float,
) -> dict:
    """Evaluate how accurate a hypothesis was. Returns accuracy metrics."""
    actual_delta = baseline_bpb - actual_bpb  # positive = improvement
    predicted_delta = hypothesis.predicted_delta
    if hypothesis.direction == "worsen":
        predicted_delta = -predicted_delta

    direction_correct = (
        (hypothesis.direction == "improve" and actual_delta > 0)
        or (hypothesis.direction == "worsen" and actual_delta < 0)
    )
    magnitude_error = abs(actual_delta - predicted_delta)

    return {
        "direction_correct": direction_correct,
        "predicted_delta": predicted_delta,
        "actual_delta": actual_delta,
        "magnitude_error": magnitude_error,
        "within_10_pct": magnitude_error <= abs(predicted_delta) * 0.10 if predicted_delta != 0 else actual_delta == 0,
    }


def test_hypothesis_valid_format():
    """Test: A well-formed hypothesis parses correctly."""
    msg = "[HYPOTHESIS] I predict increasing DEPTH to 12 will improve val_bpb by ~0.005 because more layers increase capacity."
    h = Hypothesis.parse(msg)
    assert h is not None
    assert h.change == "increasing DEPTH to 12"
    assert h.direction == "improve"
    assert h.predicted_delta == 0.005
    print("  PASS: valid hypothesis format parses correctly")


def test_hypothesis_invalid_format():
    """Test: A commit message without hypothesis format returns None."""
    msg = "Increase depth to 12 layers"
    h = Hypothesis.parse(msg)
    assert h is None
    print("  PASS: invalid format returns None")


def test_hypothesis_accuracy_correct_direction():
    """Test: Direction accuracy detected when improvement matches prediction."""
    h = Hypothesis("increase LR", "improve", 0.003, "better convergence")
    result = evaluate_hypothesis_accuracy(h, baseline_bpb=1.000, actual_bpb=0.995)
    assert result["direction_correct"] is True
    assert abs(result["actual_delta"] - 0.005) < 1e-9
    print("  PASS: correct direction detected")


def test_hypothesis_accuracy_wrong_direction():
    """Test: Direction inaccuracy detected when result opposes prediction."""
    h = Hypothesis("increase LR", "improve", 0.003, "better convergence")
    result = evaluate_hypothesis_accuracy(h, baseline_bpb=1.000, actual_bpb=1.010)
    assert result["direction_correct"] is False
    print("  PASS: wrong direction detected")


def test_hypothesis_roundtrip():
    """Test: Format then parse produces same hypothesis."""
    original = Hypothesis("reduce weight decay to 0.1", "improve", 0.002, "less regularization helps small models")
    formatted = original.format()
    parsed = Hypothesis.parse(formatted)
    assert parsed is not None
    assert parsed.change == original.change
    assert parsed.direction == original.direction
    assert parsed.predicted_delta == original.predicted_delta
    assert parsed.reason == original.reason
    print("  PASS: hypothesis roundtrip (format → parse) preserves fields")


# ============================================================================
# 3. Checkpoint & Fork Strategy
# ============================================================================

@dataclass
class ExperimentTracker:
    """Tracks experiment count and manages checkpoint/fork scheduling."""
    experiment_count: int = 0
    checkpoints: dict = field(default_factory=dict)  # tag_name -> commit
    current_best_bpb: float = float("inf")
    current_best_commit: str = ""

    def record_experiment(self, result: ExperimentResult):
        self.experiment_count += 1
        if result.status != "crash" and result.val_bpb < self.current_best_bpb:
            self.current_best_bpb = result.val_bpb
            self.current_best_commit = result.commit

    def should_checkpoint(self) -> bool:
        """Every 10 experiments, checkpoint the current best."""
        return self.experiment_count > 0 and self.experiment_count % 10 == 0

    def create_checkpoint(self) -> str:
        tag = f"best-{self.experiment_count}"
        self.checkpoints[tag] = self.current_best_commit
        return tag

    def should_fork(self) -> bool:
        """Every 5th experiment, fork from a previous checkpoint."""
        return (
            self.experiment_count > 0
            and self.experiment_count % 5 == 0
            and len(self.checkpoints) > 0
        )

    def get_fork_target(self) -> Optional[str]:
        if not self.checkpoints:
            return None
        tags = sorted(self.checkpoints.keys())
        # Pick the oldest checkpoint not yet forked from (round-robin)
        idx = (self.experiment_count // 5 - 1) % len(tags)
        return tags[idx]

    def is_ablation_turn(self) -> bool:
        """Every 5th experiment must be an ablation/simplification."""
        return self.experiment_count > 0 and self.experiment_count % 5 == 0

    def is_analysis_turn(self) -> bool:
        """Every 20 experiments, do a cross-run analysis."""
        return self.experiment_count > 0 and self.experiment_count % 20 == 0


def test_checkpoint_at_10():
    """Test: Checkpoint is triggered at experiment 10."""
    tracker = ExperimentTracker()
    for i in range(10):
        tracker.record_experiment(ExperimentResult(
            f"c{i:05d}", 1.0 - i * 0.001, 44.0, 39.0, 500.0, "keep", "no", f"exp {i}"
        ))
    assert tracker.should_checkpoint()
    tag = tracker.create_checkpoint()
    assert tag == "best-10"
    assert tracker.checkpoints["best-10"] == "c00009"  # best result
    print("  PASS: checkpoint triggered at experiment 10")


def test_checkpoint_not_at_7():
    """Test: Checkpoint is NOT triggered at experiment 7."""
    tracker = ExperimentTracker()
    for i in range(7):
        tracker.record_experiment(ExperimentResult(
            f"c{i:05d}", 1.0, 44.0, 39.0, 500.0, "keep", "no", f"exp {i}"
        ))
    assert not tracker.should_checkpoint()
    print("  PASS: no checkpoint at experiment 7")


def test_fork_requires_checkpoint():
    """Test: Fork is not triggered if no checkpoints exist yet."""
    tracker = ExperimentTracker()
    for i in range(5):
        tracker.record_experiment(ExperimentResult(
            f"c{i:05d}", 1.0, 44.0, 39.0, 500.0, "keep", "no", f"exp {i}"
        ))
    assert not tracker.should_fork()  # No checkpoints created yet
    print("  PASS: fork requires existing checkpoints")


def test_fork_after_checkpoint():
    """Test: Fork triggers at experiment 15 after checkpoint at 10."""
    tracker = ExperimentTracker()
    for i in range(10):
        tracker.record_experiment(ExperimentResult(
            f"c{i:05d}", 1.0 - i * 0.001, 44.0, 39.0, 500.0, "keep", "no", f"exp {i}"
        ))
    tracker.create_checkpoint()  # best-10
    for i in range(10, 15):
        tracker.record_experiment(ExperimentResult(
            f"c{i:05d}", 0.990, 44.0, 39.0, 500.0, "keep", "no", f"exp {i}"
        ))
    assert tracker.should_fork()
    target = tracker.get_fork_target()
    assert target == "best-10"
    print("  PASS: fork targets checkpoint after experiment 15")


def test_checkpoint_tracks_best():
    """Test: Checkpoint records the best val_bpb commit, not the latest."""
    tracker = ExperimentTracker()
    results = [
        ExperimentResult("aaa", 1.000, 44.0, 39.0, 500.0, "keep", "no", "baseline"),
        ExperimentResult("bbb", 0.990, 44.0, 39.0, 500.0, "keep", "no", "improvement"),
        ExperimentResult("ccc", 1.005, 44.0, 39.0, 500.0, "discard", "no", "regression"),
        ExperimentResult("ddd", 0.985, 44.0, 39.0, 500.0, "keep", "no", "best yet"),
        ExperimentResult("eee", 0.995, 44.0, 39.0, 500.0, "discard", "no", "not best"),
        ExperimentResult("fff", 0.988, 44.0, 39.0, 500.0, "discard", "no", "not best"),
        ExperimentResult("ggg", 0.992, 44.0, 39.0, 500.0, "discard", "no", "not best"),
        ExperimentResult("hhh", 0.991, 44.0, 39.0, 500.0, "discard", "no", "not best"),
        ExperimentResult("iii", 0.993, 44.0, 39.0, 500.0, "discard", "no", "not best"),
        ExperimentResult("jjj", 0.994, 44.0, 39.0, 500.0, "discard", "no", "not best"),
    ]
    for r in results:
        tracker.record_experiment(r)
    assert tracker.current_best_commit == "ddd"
    assert tracker.current_best_bpb == 0.985
    print("  PASS: checkpoint tracks best val_bpb commit, not latest")


# ============================================================================
# 4. Forced Ablation Protocol
# ============================================================================

ABLATION_KEYWORDS = [
    "remove", "delete", "simplify", "reduce", "drop", "disable",
    "eliminate", "strip", "cut", "ablate", "without",
]


def is_ablation_experiment(description: str) -> bool:
    """Check if an experiment description indicates an ablation/simplification."""
    desc_lower = description.lower()
    return any(kw in desc_lower for kw in ABLATION_KEYWORDS)


def count_lines_changed(diff_text: str) -> tuple[int, int]:
    """Count lines added and removed from a unified diff."""
    added = sum(1 for line in diff_text.splitlines() if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in diff_text.splitlines() if line.startswith("-") and not line.startswith("---"))
    return added, removed


def is_net_simplification(diff_text: str) -> bool:
    """Check if a diff results in net fewer lines (simplification)."""
    added, removed = count_lines_changed(diff_text)
    return removed >= added


def test_ablation_schedule():
    """Test: Ablation is required at experiments 5, 10, 15, 20, etc."""
    tracker = ExperimentTracker()
    ablation_turns = []
    for i in range(25):
        tracker.record_experiment(ExperimentResult(
            f"c{i:05d}", 1.0, 44.0, 39.0, 500.0, "keep", "no", f"exp {i}"
        ))
        if tracker.is_ablation_turn():
            ablation_turns.append(tracker.experiment_count)
    assert ablation_turns == [5, 10, 15, 20, 25]
    print("  PASS: ablation required at experiments 5, 10, 15, 20, 25")


def test_ablation_keyword_detection():
    """Test: Ablation descriptions are correctly identified by keywords."""
    assert is_ablation_experiment("Remove value embeddings entirely")
    assert is_ablation_experiment("Simplify window pattern to L")
    assert is_ablation_experiment("reduce depth from 8 to 6")
    assert not is_ablation_experiment("Add residual connections")
    assert not is_ablation_experiment("Increase learning rate")
    print("  PASS: ablation keywords correctly detected")


def test_ablation_net_simplification():
    """Test: A diff that removes more lines than it adds is a simplification."""
    diff = """\
--- a/train.py
+++ b/train.py
@@ -10,8 +10,3 @@
-        self.ve_gate_channels = 32
-        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
-
-    def forward_ve(self, x, ve):
-        gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
-        return v + gate.unsqueeze(-1) * ve
+        pass
"""
    assert is_net_simplification(diff)
    added, removed = count_lines_changed(diff)
    assert removed == 6
    assert added == 1
    print("  PASS: net simplification correctly detected in diff")


def test_ablation_not_simplification():
    """Test: A diff that adds more lines than it removes is NOT a simplification."""
    diff = """\
--- a/train.py
+++ b/train.py
@@ -10,1 +10,5 @@
-        x = self.c_fc(x)
+        x = self.c_fc(x)
+        x = self.gate(x)
+        x = self.norm(x)
+        x = F.dropout(x, p=0.1)
+        x = self.extra_layer(x)
"""
    assert not is_net_simplification(diff)
    print("  PASS: non-simplification correctly rejected")


def test_ablation_equal_val_bpb_keeps_simpler():
    """Test: When val_bpb is equal, simpler code (fewer lines) should be kept."""
    baseline = ExperimentResult("aaa", 0.990, 44.0, 39.0, 500.0, "keep", "yes", "baseline")
    ablation = ExperimentResult("bbb", 0.990, 42.0, 40.0, 500.0, "keep", "yes", "remove VE, same bpb")

    # Equal val_bpb + simpler (is_ablation=True) → should keep
    should_keep = (
        ablation.val_bpb <= baseline.val_bpb
        and is_ablation_experiment(ablation.description)
    )
    assert should_keep
    print("  PASS: equal val_bpb + simplification → keep the simpler version")


# ============================================================================
# 5. Cross-Run Pattern Analysis
# ============================================================================

def categorize_experiment(description: str) -> str:
    """Categorize an experiment by type based on its description.

    Priority: ablation keywords are only used if no domain keyword matches first.
    Architecture/optimizer/data keywords take precedence because 'reduce depth'
    is an architecture change, not a generic ablation.
    """
    desc_lower = description.lower()
    if any(kw in desc_lower for kw in ["lr", "learning rate", "warmup", "warmdown", "decay", "momentum", "beta"]):
        return "optimizer"
    if any(kw in desc_lower for kw in ["depth", "layer", "head", "embed", "dimension", "width", "attention", "mlp", "window"]):
        return "architecture"
    if any(kw in desc_lower for kw in ["batch", "accumulation", "sequence"]):
        return "data"
    if any(kw in desc_lower for kw in ABLATION_KEYWORDS):
        return "ablation"
    return "other"


def analyze_results(results: list[ExperimentResult]) -> dict:
    """Perform cross-run analysis on experiment results."""
    non_crash = [r for r in results if r.status != "crash"]
    if not non_crash:
        return {"total_experiments": 0}

    baseline = non_crash[0]
    best = min(non_crash, key=lambda r: r.val_bpb)

    # Category analysis
    categories = {}
    for r in non_crash[1:]:  # skip baseline
        cat = categorize_experiment(r.description)
        if cat not in categories:
            categories[cat] = {"count": 0, "improvements": 0, "total_delta": 0.0}
        categories[cat]["count"] += 1
        delta = baseline.val_bpb - r.val_bpb
        categories[cat]["total_delta"] += delta
        if r.status == "keep":
            categories[cat]["improvements"] += 1

    # Diminishing returns: compare improvement rate of first half vs second half
    half = len(non_crash) // 2
    if half > 1:
        first_half_best = min(non_crash[:half], key=lambda r: r.val_bpb).val_bpb
        second_half_best = min(non_crash[half:], key=lambda r: r.val_bpb).val_bpb
        first_half_improvement = baseline.val_bpb - first_half_best
        second_half_improvement = first_half_best - second_half_best
        diminishing = second_half_improvement < first_half_improvement * 0.5
    else:
        diminishing = False

    return {
        "total_experiments": len(results),
        "total_improvement": baseline.val_bpb - best.val_bpb,
        "best_commit": best.commit,
        "categories": categories,
        "diminishing_returns": diminishing,
        "crash_rate": sum(1 for r in results if r.status == "crash") / len(results),
        "pareto_frontier_size": len(compute_pareto_frontier(results)),
    }


def test_analysis_category_grouping():
    """Test: Experiments are correctly grouped into categories."""
    results = [
        ExperimentResult("aaa", 1.000, 44.0, 39.0, 500.0, "keep", "yes", "baseline"),
        ExperimentResult("bbb", 0.995, 44.0, 39.0, 500.0, "keep", "no", "increase LR to 0.05"),
        ExperimentResult("ccc", 0.998, 44.0, 39.0, 500.0, "discard", "no", "add extra attention head"),
        ExperimentResult("ddd", 0.990, 44.0, 39.0, 500.0, "keep", "no", "reduce depth to 6"),
    ]
    analysis = analyze_results(results)
    assert "optimizer" in analysis["categories"]
    assert "architecture" in analysis["categories"]
    assert analysis["categories"]["optimizer"]["count"] == 1
    # Both "add extra attention head" and "reduce depth to 6" match architecture keywords
    assert analysis["categories"]["architecture"]["count"] == 2
    print("  PASS: experiments correctly categorized")


def test_analysis_total_improvement():
    """Test: Total improvement is correctly calculated from baseline to best."""
    results = [
        ExperimentResult("aaa", 1.000, 44.0, 39.0, 500.0, "keep", "yes", "baseline"),
        ExperimentResult("bbb", 0.990, 44.0, 39.0, 500.0, "keep", "no", "tweak LR"),
        ExperimentResult("ccc", 0.985, 44.0, 39.0, 500.0, "keep", "no", "tweak depth"),
    ]
    analysis = analyze_results(results)
    assert abs(analysis["total_improvement"] - 0.015) < 1e-9
    assert analysis["best_commit"] == "ccc"
    print("  PASS: total improvement calculated correctly")


def test_analysis_diminishing_returns():
    """Test: Diminishing returns detected when second half improves less than half of first."""
    results = [
        ExperimentResult("a", 1.000, 44.0, 39.0, 500.0, "keep", "yes", "baseline"),
        ExperimentResult("b", 0.980, 44.0, 39.0, 500.0, "keep", "no", "big LR improvement"),
        ExperimentResult("c", 0.970, 44.0, 39.0, 500.0, "keep", "no", "architecture depth change"),
        ExperimentResult("d", 0.968, 44.0, 39.0, 500.0, "keep", "no", "tiny LR tweak"),
        ExperimentResult("e", 0.967, 44.0, 39.0, 500.0, "keep", "no", "marginal depth change"),
    ]
    analysis = analyze_results(results)
    # First half (a,b): improvement = 1.0 - 0.98 = 0.02
    # Second half (c,d,e): improvement = 0.98 - 0.967 = 0.013 — but measured from first_half_best
    # first_half_best = 0.98, second_half_best = 0.967, improvement = 0.013
    # 0.013 < 0.02 * 0.5 = 0.01? No. So not diminishing in this case.
    # Let's test with a clearer case
    results2 = [
        ExperimentResult("a", 1.000, 44.0, 39.0, 500.0, "keep", "yes", "baseline"),
        ExperimentResult("b", 0.950, 44.0, 39.0, 500.0, "keep", "no", "big improvement"),
        ExperimentResult("c", 0.948, 44.0, 39.0, 500.0, "keep", "no", "tiny improvement"),
        ExperimentResult("d", 0.947, 44.0, 39.0, 500.0, "keep", "no", "tiny improvement"),
    ]
    analysis2 = analyze_results(results2)
    assert analysis2["diminishing_returns"] is True
    print("  PASS: diminishing returns correctly detected")


def test_analysis_crash_rate():
    """Test: Crash rate is correctly computed."""
    results = [
        ExperimentResult("a", 1.000, 44.0, 39.0, 500.0, "keep", "yes", "baseline"),
        ExperimentResult("b", 0.000, 0.0, 0.0, 0.0, "crash", "no", "OOM"),
        ExperimentResult("c", 0.990, 44.0, 39.0, 500.0, "keep", "no", "good"),
        ExperimentResult("d", 0.000, 0.0, 0.0, 0.0, "crash", "no", "bug"),
        ExperimentResult("e", 0.985, 44.0, 39.0, 500.0, "keep", "no", "good"),
    ]
    analysis = analyze_results(results)
    assert abs(analysis["crash_rate"] - 0.4) < 1e-9
    print("  PASS: crash rate correctly computed (2/5 = 40%)")


def test_analysis_schedule():
    """Test: Analysis is triggered at experiments 20, 40, 60, etc."""
    tracker = ExperimentTracker()
    analysis_turns = []
    for i in range(45):
        tracker.record_experiment(ExperimentResult(
            f"c{i:05d}", 1.0, 44.0, 39.0, 500.0, "keep", "no", f"exp {i}"
        ))
        if tracker.is_analysis_turn():
            analysis_turns.append(tracker.experiment_count)
    assert analysis_turns == [20, 40]
    print("  PASS: analysis triggered at experiments 20 and 40")


# ============================================================================
# 6. Statistical Confirmation
# ============================================================================

def needs_confirmation(current_best_bpb: float, new_bpb: float, threshold_pct: float = 0.5) -> bool:
    """Determine if a result is borderline and needs a confirmation run.

    Returns True if the new result is within threshold_pct% of current best
    (either better or worse).
    """
    if current_best_bpb == 0:
        return False
    pct_diff = abs(new_bpb - current_best_bpb) / current_best_bpb * 100
    return pct_diff <= threshold_pct


def confirm_result(run1_bpb: float, run2_bpb: float, current_best_bpb: float) -> dict:
    """Evaluate a confirmed (double-run) result.

    Returns decision dict with average bpb and keep/discard recommendation.
    """
    avg_bpb = (run1_bpb + run2_bpb) / 2
    return {
        "run1_bpb": run1_bpb,
        "run2_bpb": run2_bpb,
        "avg_bpb": avg_bpb,
        "variance": abs(run1_bpb - run2_bpb),
        "keep": avg_bpb < current_best_bpb,
    }


def test_confirmation_borderline_better():
    """Test: A result 0.3% better than current best requires confirmation."""
    current_best = 1.000
    new_result = 0.997  # 0.3% better
    assert needs_confirmation(current_best, new_result)
    print("  PASS: borderline better (0.3%) requires confirmation")


def test_confirmation_borderline_worse():
    """Test: A result 0.4% worse than current best requires confirmation."""
    current_best = 1.000
    new_result = 1.004  # 0.4% worse
    assert needs_confirmation(current_best, new_result)
    print("  PASS: borderline worse (0.4%) requires confirmation")


def test_confirmation_clear_improvement():
    """Test: A result 2% better than current best does NOT need confirmation."""
    current_best = 1.000
    new_result = 0.980  # 2% better
    assert not needs_confirmation(current_best, new_result)
    print("  PASS: clear improvement (2%) skips confirmation")


def test_confirmation_clear_regression():
    """Test: A result 3% worse than current best does NOT need confirmation."""
    current_best = 1.000
    new_result = 1.030  # 3% worse
    assert not needs_confirmation(current_best, new_result)
    print("  PASS: clear regression (3%) skips confirmation")


def test_confirmation_average_decides():
    """Test: The average of two runs determines keep/discard, not individual runs."""
    current_best = 1.000

    # Run 1 looks like improvement, Run 2 reveals it was noise
    result = confirm_result(run1_bpb=0.997, run2_bpb=1.005, current_best_bpb=current_best)
    assert result["avg_bpb"] == 1.001
    assert result["keep"] is False  # average is worse than current best
    print("  PASS: average of two runs decides — false positive caught")


# ============================================================================
# TSV I/O
# ============================================================================

TSV_HEADER = "commit\tval_bpb\tmemory_gb\tmfu_pct\ttokens_M\tstatus\tpareto\tdescription"


def write_results_tsv(results: list[ExperimentResult], path: str):
    """Write results to a TSV file."""
    with open(path, "w") as f:
        f.write(TSV_HEADER + "\n")
        for r in results:
            f.write(f"{r.commit}\t{r.val_bpb:.6f}\t{r.memory_gb:.1f}\t{r.mfu_pct:.1f}\t{r.tokens_M:.1f}\t{r.status}\t{r.pareto}\t{r.description}\n")


def read_results_tsv(path: str) -> list[ExperimentResult]:
    """Read results from a TSV file."""
    results = []
    with open(path, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            results.append(ExperimentResult(
                commit=row["commit"],
                val_bpb=float(row["val_bpb"]),
                memory_gb=float(row["memory_gb"]),
                mfu_pct=float(row["mfu_pct"]),
                tokens_M=float(row["tokens_M"]),
                status=row["status"],
                pareto=row["pareto"],
                description=row["description"],
            ))
    return results


def test_tsv_roundtrip():
    """Test: Results survive a write → read roundtrip through TSV."""
    results = [
        ExperimentResult("a1b2c3d", 0.997900, 44.0, 39.8, 499.6, "keep", "yes", "baseline"),
        ExperimentResult("b2c3d4e", 0.993200, 44.2, 39.5, 499.6, "keep", "yes", "increase LR to 0.04"),
        ExperimentResult("c3d4e5f", 0.000000, 0.0, 0.0, 0.0, "crash", "no", "double model width (OOM)"),
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
        path = f.name
    try:
        write_results_tsv(results, path)
        loaded = read_results_tsv(path)
        assert len(loaded) == 3
        assert loaded[0].commit == "a1b2c3d"
        assert abs(loaded[0].val_bpb - 0.997900) < 1e-6
        assert loaded[1].pareto == "yes"
        assert loaded[2].status == "crash"
        print("  PASS: TSV roundtrip preserves all fields")
    finally:
        os.unlink(path)


# ============================================================================
# Runner
# ============================================================================

def run_all_tests():
    test_sections = [
        ("1. Multi-Objective Pareto Tracking", [
            test_pareto_basic_dominance,
            test_pareto_tradeoff_preserved,
            test_pareto_crash_excluded,
            test_pareto_three_way_frontier,
            test_pareto_new_run_displaces,
        ]),
        ("2. Structured Hypotheses", [
            test_hypothesis_valid_format,
            test_hypothesis_invalid_format,
            test_hypothesis_accuracy_correct_direction,
            test_hypothesis_accuracy_wrong_direction,
            test_hypothesis_roundtrip,
        ]),
        ("3. Checkpoint & Fork Strategy", [
            test_checkpoint_at_10,
            test_checkpoint_not_at_7,
            test_fork_requires_checkpoint,
            test_fork_after_checkpoint,
            test_checkpoint_tracks_best,
        ]),
        ("4. Forced Ablation Protocol", [
            test_ablation_schedule,
            test_ablation_keyword_detection,
            test_ablation_net_simplification,
            test_ablation_not_simplification,
            test_ablation_equal_val_bpb_keeps_simpler,
        ]),
        ("5. Cross-Run Pattern Analysis", [
            test_analysis_category_grouping,
            test_analysis_total_improvement,
            test_analysis_diminishing_returns,
            test_analysis_crash_rate,
            test_analysis_schedule,
        ]),
        ("6. Statistical Confirmation", [
            test_confirmation_borderline_better,
            test_confirmation_borderline_worse,
            test_confirmation_clear_improvement,
            test_confirmation_clear_regression,
            test_confirmation_average_decides,
        ]),
        ("Bonus: TSV I/O", [
            test_tsv_roundtrip,
        ]),
    ]

    total = 0
    passed = 0
    failed = 0

    for section_name, tests in test_sections:
        print(f"\n{'='*60}")
        print(f" {section_name}")
        print(f"{'='*60}")
        for test_fn in tests:
            total += 1
            try:
                test_fn()
                passed += 1
            except AssertionError as e:
                failed += 1
                print(f"  FAIL: {test_fn.__name__} — {e}")
            except Exception as e:
                failed += 1
                print(f"  ERROR: {test_fn.__name__} — {type(e).__name__}: {e}")

    print(f"\n{'='*60}")
    print(f" Results: {passed}/{total} passed, {failed} failed")
    print(f"{'='*60}")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
