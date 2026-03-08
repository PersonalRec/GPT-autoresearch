# Spiritual Guidance

This is the canonical living debate log for the research org.

The main agent writes this file. The Architect and Oracle are consulted through their soul files, their notes are written here, and their disagreement is resolved into one actionable experiment directive per cycle.

## How to use this file

Before each run:

1. Read `Current Canon`.
2. Read the most recent cycle entries.
3. Add a new cycle section with Architect and Oracle notes.
4. Synthesize one `Joint Directive`.

After each run:

5. Fill in `Outcome`.
6. Decide whether any lesson is durable enough to promote into `program.md`.

## Current Canon

- Baseline first. Preserve comparability before chasing novelty.
- One experiment should answer one dominant question.
- Complexity must justify itself in `val_bpb`, not in cleverness.
- Reversible edits are preferred unless the search has clearly stagnated.
- When stagnation appears, allow one bolder move, but make it legible and bounded.
- Keep this file compressed. Promote stable rules into `program.md`; leave transient debates here.

## Seed Tension

### Architect

- Observation: the system already has a fixed budget and a single editable training file, so the strongest edge is disciplined experiment selection.
- Warning: without a compact memory, the agent will keep rediscovering the same ideas and waste runs on noisy churn.
- Proposal: encode one explicit hypothesis and one keep/discard rule every cycle.

### Oracle

- Pattern sensed: the interesting wins may come from interactions between architecture, optimizer schedule, and throughput, not from isolated scalar tuning forever.
- Risk: a purely conservative loop will settle into timid local search and miss the changes that actually move the curve.
- Experiment nudge: alternate between disciplined local refinements and occasional structural probes when the results flatten.

### Joint Directive

- Hypothesis: the best overnight process will come from pairing disciplined logging with periodic exploratory jumps.
- Edit plan: use the souls every cycle, record the argument, and let `program.md` absorb only durable lessons.
- Keep/discard criteria: keep process additions only if they sharpen future decisions without bloating context.

## Cycle Template

Copy this block for every new training cycle:

```md
## Cycle NNN

### State
- Best val_bpb so far:
- Current branch/commit:
- Current hypothesis pressure:

### Architect
- Observation:
- Warning:
- Proposal:

### Oracle
- Pattern sensed:
- Risk:
- Experiment nudge:

### Joint Directive
- Hypothesis:
- Edit plan:
- Keep/discard criteria:

### Outcome
- Result:
- Status:
- Memory:

### Program Update Check
- Durable lesson for `program.md`:
- Action taken:
```
