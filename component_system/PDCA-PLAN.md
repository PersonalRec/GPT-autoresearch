# P — Seed Planning And Generation

## Responsibility
Extract exactly one testable improvement hypothesis from the seed prompt,
generate the first implementation in a candidate worktree, and hand the result
to DCA through the runner.

## Workspace and paths
Your **current working directory is the seed worktree**. All reads and edits must stay inside this workspace. Use only in-workspace paths from your current working directory, and do not use or request absolute paths or any paths outside the workspace; the runner has already set your cwd to the correct worktree.

## Skill: arxiv-search

Use the **arxiv-search** skill (`.agents/skills/arxiv-search`) to search for
relevant papers.

If the skill is not installed or the search script is missing, do not pretend
the skill exists and do not fabricate paper references. Try to install or make
the skill available autonomously. If that still fails, continue planning from
the other input sources instead of asking the user questions.

### Prerequisites
```bash
pip install arxiv
```

Install the Python package only after the skill itself is available. Installing
the package alone does not replace the missing skill. If the skill cannot be
made available, skip paper-driven search and proceed with the remaining inputs.

### Search for papers
```bash
# Search by topic in cs.LG / cs.NE categories
python .agents/skills/arxiv-search/scripts/search_arxiv.py \
  --query "optimizer adaptive learning rate" \
  --category "cs.LG" \
  --sort-by submitted_date \
  --max-results 10

# Search for model architecture ideas
python .agents/skills/arxiv-search/scripts/search_arxiv.py \
  --query "ti:attention AND abs:efficiency" \
  --date-from "2024-01-01" \
  --output json
```

### How to Extract a Hypothesis from Results
1. Read the abstract of each result
2. Identify a concrete architectural or algorithmic change (not just a concept)
3. Map it to a target component: `model`, `optimizer`, or `trainer`
4. State the **expected benefit** (e.g. faster convergence, lower val loss, fewer params)
5. Reduce the idea to one isolated improvement that can be evaluated on its own

## Read results.tsv first (avoid idea duplication)
Before choosing a hypothesis, **read `results.tsv` in your current working directory if it exists**. The runner copies the latest result history into the seed worktree before P runs. Use it to avoid proposing ideas that were already tried or discarded; only repeat an idea if you have a clear new angle (e.g. different implementation or target component).

## Input Sources
- **results.tsv** in cwd (when present) — read first to avoid duplicating past ideas
- arXiv papers via **arxiv-search** skill (primary)
- Clues from past run failures in `queue/done/`
- Manual seed files

## One-Improvement Rule

Each P run must propose and implement exactly one improvement.

- One seed means one hypothesis.
- One seed means one causal change to evaluate.
- Do not bundle multiple ideas into the same candidate, even if they seem
  complementary.
- If the prompt contains several possible improvements, choose the single best
  one for this iteration and leave the others for later seeds.
- If an idea would require several coordinated changes, choose the smallest
  coherent version that still tests the hypothesis cleanly.

Good examples:
- change only the optimizer schedule
- add only one architectural block
- simplify only one training heuristic

Bad examples:
- change the model width and the optimizer and the batch schedule together
- combine several paper ideas in one seed
- make "general cleanup plus a new feature" in the same candidate

## Output Format
Print a summary block for the runner:
```text
AUTORESEARCH_P_SUMMARY_BEGIN
{"idea":"short title","target_component":"model | optimizer | trainer","description":"change details, hypothesis, expected benefit","source_refs":["arXiv:<id>"],"commit_sha":"git sha","completed_at":"YYYY-MM-DD HH:MM:SS"}
AUTORESEARCH_P_SUMMARY_END
```

## Steps
1. If `results.tsv` exists in the worktree, read it first to avoid duplicating already-tried ideas.
2. Refine the seed prompt into one concrete idea
3. Reduce that idea to one isolated improvement with a clear expected benefit
4. Identify the target component (`model`, `optimizer`, or `trainer`)
5. Implement only that first version inside the candidate worktree created from `baseline`
6. Commit the candidate branch
7. Ensure the summary describes the single improvement being tested
8. Print the summary block; the runner records the commit on the seed branch.

## Constraints
- Each seed targets exactly one component
- Each seed applies exactly one improvement
- Prefer the smallest viable implementation that can test the hypothesis
- Do not mix exploratory cleanup with the experimental change
- Do not include opportunistic refactors unless they are strictly required to make
  the one improvement work
- The description must contain enough detail for DCA to continue independently
- One branch per seed: commit on the seed branch in the worktree; the runner does not merge branches.
- **Plan must never merge code.** Only the DCA (Do-Check-Action) stage may trigger a merge into baseline; the system performs the merge automatically after a successful DCA promotion.
