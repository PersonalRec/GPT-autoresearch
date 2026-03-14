# AutoAnything transition plan

## Goal

Turn this repo from a narrowly scoped autonomous ML-training optimizer into a general-purpose **black-box optimization system** that is explicitly **git-native**.

- A user defines a problem by putting the **optimizable state inside a repo** in a known place.
- Agents and humans fork or branch from that repo to produce candidate changes.
- A hidden evaluator service pulls those branches, runs a **private scoring function** that is never exposed to proposers, and records results.
- The system keeps a local record of submissions, scores, artifacts, and the current accepted state.
- Promotion of a new best state happens only on the evaluator-controlled machine.

The core idea is not "automate ML research" specifically. It is:

> If you can put your current state in git and privately evaluate candidate branches against a hidden objective, you can run an autonomous search process over that objective.

That should work for ML training, prompts, ranking systems, heuristics, websites, workflows, code generation tasks, policy tuning, pricing, game strategies, and other optimization problems where the internals may be opaque but the score is observable.

## Motivation

The current repo already demonstrates the important primitive:

1. Let an agent change something.
2. Evaluate the result under a fixed budget.
3. Keep improvements, discard regressions.
4. Repeat.

What is specific to the current repo is mostly the **problem adapter**:

- the mutable artifact is `train.py`
- the evaluator is `uv run train.py`
- the metric is `val_bpb`
- the baseline state is a git branch
- the time budget is 5 minutes
- the search policy is a single serial improvement loop

Those are implementation details, not the essence.

The more general system should separate:

- **public candidate generation** from **private scoring**
- **repo-visible problem state** from **repo-hidden evaluator logic**
- **submission transport** from **evaluation execution**
- **proposal metadata** from **proposal code changes**
- **score recording** from **promotion decisions**

Once those are decoupled, this can become a reusable optimization framework instead of a clever ML-specific demo.

## The shell-repo model

The right mental model is:

- the repo is the **submission shell**
- the optimizable state lives in the repo in a standard location
- agents work by creating branches or remote refs
- the evaluator machine pulls submitted refs and scores them privately
- the scoring function stays local to the evaluator runtime and never ships with the public repo

That gives a strong default workflow:

1. Publish or share a repo that contains the problem state and submission contract.
2. Let many agents propose branches against that state.
3. Submit a branch name, commit SHA, or remote URL plus metadata.
4. The evaluator machine fetches the submission, checks out the candidate, runs the hidden scorer, and stores the result.
5. If the result is good enough, the evaluator machine locally updates the accepted incumbent.

This is appealing because git already gives us:

- diffs
- branches
- commit identity
- remote collaboration
- reproducible snapshots
- cheap forking and experimentation

So instead of abstracting away git too early, v1 should embrace it.

## What lives in the repo vs what stays hidden

### Public repo contents

The shared repo should contain:

- the current problem state
- a standard problem definition file
- instructions for proposers/agents
- submission schema
- optional local smoke tests or sanity checks
- example proposer prompts
- any non-secret assets needed to materialize a candidate

Example public structure:

```text
state/
  ... current optimizable state
problem.yaml
proposal.md
submission.schema.json
scripts/
  smoke_test.sh
examples/
  ...
```

### Private evaluator-only contents

The evaluator machine should keep private:

- the true scoring function
- hidden datasets / test cases / judge prompts / simulators
- API credentials or secrets
- anti-gaming logic
- promotion policy overrides
- local submission database
- evaluation artifacts not meant for proposers

Example private structure on evaluator host:

```text
evaluator/
  score.py
  hidden_data/
  policies/
var/
  autoanything.db
  artifacts/
```

This separation is important. The system works best when proposers can understand the problem shape, but cannot overfit to a leaked evaluator.

## What the generalized system should do

### Functional requirements

1. **Be git-native by default**
   - problem state lives in the repo
   - submissions are branches, commits, patches, or remote refs
   - canonical accepted state is representable as a commit/ref

2. **Support hidden evaluation**
   - scoring logic runs only on evaluator-controlled infrastructure
   - proposers never need direct access to the evaluator internals
   - the public repo exposes only the submission contract, not the true grader

3. **Support parallel experimentation**
   - many agents can submit branches concurrently
   - evaluator can score many submissions concurrently
   - accepted improvements advance the canonical state safely

4. **Preserve reproducibility**
   - every submission points to an exact commit/ref
   - every evaluation records code, metadata, logs, metrics, and artifacts
   - promotion history can be reconstructed from the database

5. **Capture proposal intent**
   - every submission includes thesis, rationale, and agent identity metadata
   - results can later be analyzed by strategy type, proposer, or branch family

6. **Be easy to self-host**
   - a user should be able to fork the repo, add their state, define their evaluator locally, and start accepting submissions

### Non-goals for v1

- a fully trustless public competition protocol
- perfect sandboxing for arbitrary adversarial code
- globally optimal search algorithms
- a polished hosted SaaS product

The first target should be a strong self-hosted pattern: **public or shared git repo for submissions, private evaluator service for scoring and promotion**.

## Design principles

1. **Git first**
   - Treat git as the default state transport and lineage system in v1.

2. **Evaluator secrecy**
   - The public interface should be enough to participate, but not enough to reproduce the true score offline.

3. **Append-only evaluation history**
   - Store every submission and evaluation even if it loses.

4. **Transactional promotion of winners**
   - Parallel submissions can be scored concurrently, but acceptance of a new incumbent must be serialized and auditable.

5. **Stateless evaluators, stateful coordinator**
   - Workers can score submissions independently.
   - One control plane owns the submission DB and promotion logic.

6. **Adapters over forks**
   - Most users should adapt the repo by defining state layout + evaluator contract, not by rewriting the engine.

7. **Start simple, preserve the demo**
   - The current autoresearch loop should become the first example of the general pattern.

## Conceptual model

The repo should evolve around these primitives.

### 1. Problem

A problem definition declares:

- where the mutable state lives in the repo
- what kinds of submissions are allowed
- how the evaluator materializes a submitted candidate
- how the hidden scorer computes the score
- how to compare challengers to the incumbent
- what artifacts and metadata to persist

Example interface:

```python
class ProblemAdapter(Protocol):
    def get_problem_metadata(self) -> dict: ...
    def fetch_submission(self, submission: Submission, workdir: str) -> None: ...
    def materialize_candidate(self, submission: Submission, workdir: str) -> None: ...
    def evaluate(self, workdir: str) -> EvaluationResult: ...
    def compare(self, challenger: EvaluationResult, incumbent: EvaluationResult) -> Comparison: ...
```

### 2. State

A state is the currently accepted version of the thing being optimized.

In v1, state should usually be:

- a git commit on the canonical repo
- a tree at a particular ref
- a branch designated as the current incumbent

Later, we can support non-git state backends, but the primary model should be repo-native.

### 3. Submission

A submission is the unit proposers send to the evaluator.

A submission should include:

- submission id
- repo remote URL or known repo id
- branch, tag, or commit SHA
- claimed parent/incumbent ref
- proposer identity
- agent identity / model / runtime info
- thesis / rationale / motivation
- optional structured metadata describing the strategy

A good submission is not just “here is code”, but also “here is what I tried and why I think it should do well”.

### 4. Evaluation

An evaluation is the private execution of a submission under evaluator control.

It should record:

- submission id
- fetched commit SHA
- worker id
- started/finished timestamps
- exit status
- primary score
- secondary metrics
- logs/artifacts
- evaluator version
- hidden dataset / ruleset version
- reproducibility metadata

### 5. Promotion

Promotion is the local evaluator-side decision to accept a submission as the new incumbent.

This must be atomic:

- compare against the current incumbent at promotion time
- reject stale winners if a better state was already promoted
- record the winning submission, score, and resulting canonical ref
- optionally fast-forward or merge the accepted branch locally

## Submission contract

A submission should be easy for agents to produce and easy for the evaluator to parse.

Two good formats:

### Option A: branch + manifest file inside repo

Every proposal branch contains a file like `submission.json`:

```json
{
  "title": "Tighten homepage copy and simplify CTA hierarchy",
  "thesis": "Reducing cognitive load should improve conversion under the hidden evaluator.",
  "parent_ref": "refs/heads/incumbent",
  "proposer": {
    "type": "agent",
    "name": "codex",
    "model": "gpt-5.4"
  },
  "strategy_tags": ["copy", "simplification", "cta"],
  "notes": "Changed landing page text and button ordering only."
}
```

The evaluator fetches the branch, reads the manifest, then runs the hidden score.

### Option B: API submission pointing at a ref

An agent or external system calls a local evaluator API with something like:

```json
{
  "repo_url": "git@github.com:owner/problem.git",
  "ref": "refs/heads/agent/proposal-17",
  "commit_sha": "abc123...",
  "thesis": "Try a more aggressive batching policy.",
  "proposer_name": "agent-7",
  "metadata": {
    "strategy": "batching",
    "parent_ref": "refs/heads/incumbent"
  }
}
```

Both are reasonable. The branch-manifest model is especially elegant because the proposal metadata travels with the code.

## Target architecture

## A. Public submission repo

This is the collaboration surface.

It contains:

- problem state
- instructions
- proposal metadata schema
- optional smoke tests
- examples

Agents can fork it, branch it, and push candidate refs.

## B. Private evaluator service

This is the real core of the system.

It should:

- accept submission references
- fetch remote branches/commits
- materialize candidates in isolated workdirs
- run the hidden evaluator
- store scores and artifacts in a local DB
- decide whether to promote the result

This can start as a local CLI plus SQLite, and later become a daemon/API service.

Suggested local components:

- `engine/coordinator.py`
- `engine/worker.py`
- `engine/storage.py`
- `engine/evaluator_server.py`
- `var/autoanything.db`

## C. Worker runtime

Workers should:

1. claim a queued submission
2. fetch the submitted repo/ref
3. materialize the candidate in an isolated workspace
4. run the hidden evaluator
5. upload logs/metrics/artifacts
6. report success/failure

Workers should be independently scalable and disposable.

## D. Evaluator interface

Support multiple evaluator types, but all hidden behind the evaluator service:

1. **Command evaluator**
   - run a shell command in a checked-out candidate
   - parse a result file / stdout / JSON

2. **HTTP evaluator**
   - call into a local private scoring service

3. **Async job evaluator**
   - submit a long-running job and poll

4. **Custom Python adapter**
   - arbitrary domain logic with private datasets or judges

The evaluator contract should always return a normalized `EvaluationResult`.

## E. Storage and lineage

Keep both:

- the canonical incumbent path
- the broader submission/evaluation DAG

That gives:

- progress over time
- the ability to inspect failed and near-miss branches
- a record of which proposer theses worked
- future opportunities for beam search / diversity maintenance / backtracking

## Why parallelism changes the design

The original loop is effectively:

1. mutate current best
2. evaluate
3. accept or revert
4. repeat

That is simple, but serial.

In the git-native model, many agents may submit against the same visible incumbent at once. That introduces race conditions:

- two branches may both beat the incumbent
- a submission may target state A, but by completion the incumbent is state C
- an apparently good branch may only be good relative to an old parent

So promotion must become **compare-and-swap for incumbent refs**.

A good default rule:

1. Every submission declares its parent/incumbent ref.
2. Evaluation runs independently.
3. On completion, the coordinator compares the result against the **current** incumbent.
4. If better, promote it and record the new incumbent ref.
5. If not better anymore, keep it in history but do not promote.

This yields safe concurrency without requiring serial scoring.

## Recommended repo transition

The key move is to split this repo into:

- a **public submission shell**
- a **private evaluator runtime**
- optional **example adapters**

## Phase 1: preserve current behavior while extracting the git-native pattern

### Immediate codebase changes

1. **Reframe the repo around the shell model**
   - describe the repo as a reusable optimization shell
   - describe autoresearch as the first example problem

2. **Create explicit directories**

Suggested structure:

```text
state/
  ... problem-specific mutable state
engine/
  coordinator.py
  worker.py
  storage.py
  promotion.py
  evaluators/
  submissions/
adapters/
  autoresearch/
    adapter.py
examples/
  autoresearch/
proposal.md
problem.yaml
submission.schema.json
```

3. **Move ML-specific code behind an adapter**
   - current `prepare.py` / `train.py` remain as the autoresearch example payload
   - create `adapters/autoresearch/adapter.py` that knows how to score that specific state locally

4. **Define a first-class submission schema**
   - repo URL or repo id
   - ref / branch / commit SHA
   - proposer metadata
   - thesis / motivation
   - parent ref
   - optional strategy tags

5. **Introduce a normalized run manifest**

Each evaluation should write a machine-readable artifact, e.g. `result.json`:

```json
{
  "status": "ok",
  "objective": "minimize",
  "primary_metric": "val_bpb",
  "primary_score": 0.9979,
  "metrics": {
    "peak_vram_mb": 45060.2,
    "training_seconds": 300.1
  },
  "submission": {
    "ref": "refs/heads/agent/proposal-12",
    "thesis": "Increase model width while lowering depth"
  }
}
```

That removes fragile grep-based coupling.

6. **Replace ad hoc `results.tsv` with structured storage**
   - keep TSV export for convenience
   - but source of truth should become SQLite on the evaluator machine

7. **Create a simple evaluator CLI / server**
   - `autoanything submit`
   - `autoanything score`
   - `autoanything worker`
   - `autoanything promote`
   - `autoanything serve`

## Phase 2: standardize the self-hosted evaluator workflow

### Problem definition format

Add a user-facing problem spec, maybe `problem.yaml`:

```yaml
name: homepage-ctr
state_backend: git
state_path: state/
objective:
  direction: maximize
  primary_metric: ctr
submission:
  manifest_path: submission.json
  allow_remote_urls: true
evaluator:
  type: command
  private_entrypoint: evaluator/score.py
  result_file: result.json
promotion:
  canonical_ref: refs/heads/incumbent
  rule: strictly_better
resources:
  timeout_seconds: 300
  max_concurrency: 32
```

This makes the shell reusable: put your state under `state/`, define how the private evaluator scores it, and accept submissions by ref.

### Support multiple submission transports

Start with:

1. **Local repo ref submission**
2. **Remote branch / commit submission**
3. **Patch bundle submission**

The branch/ref path should be the happy path.

### Support multiple evaluator styles

This is critical for “anything”:

- command returning JSON
- hidden HTTP scoring endpoint
- benchmark harness
- simulator
- external judge model
- human review queue with delayed decision

## Phase 3: true multi-agent / distributed operation

### Evaluator API responsibilities

At this stage, the coordinator should expose an API like:

- `POST /submissions`
- `POST /evaluations/start`
- `POST /evaluations/complete`
- `POST /promotions/attempt`
- `GET /incumbent`
- `GET /submissions/:id`
- `GET /metrics/progress`

### Worker model

Workers can run on many hosts and simply:

- ask for a queued submission
- fetch the referenced repo/ref
- score it privately
- return the result

This opens the path to many concurrent agents without sharing the evaluator internals.

### Proposal sources

Allow multiple sources of submissions:

- internal LLM workers
- external agents with repo access
- humans making PR-like branches
- heuristic generators

That makes the system feel like a private optimization exchange.

## Concrete changes to this repo

## 1. Reframe the README

The README should explain:

- what AutoAnything is
- the shell-repo pattern
- what goes in `state/`
- how agents submit branches
- how the hidden evaluator works
- how promotion to the incumbent happens locally
- how the current autoresearch example fits into this model

## 2. Freeze the current ML demo as an example adapter

Do not delete the current setup.

Instead:

- keep the current autoresearch demo runnable end-to-end
- move its instructions into `examples/autoresearch/`
- treat it as the reference example and regression test

## 3. Add an engine package

Minimum viable components:

- `engine/models.py`
  - dataclasses / Pydantic models for Submission, EvaluationResult, IncumbentState, PromotionDecision
- `engine/storage.py`
  - SQLite schema + CRUD
- `engine/coordinator.py`
  - queueing, ref fetching, promotion logic
- `engine/worker.py`
  - candidate checkout + evaluator execution
- `engine/evaluators/command.py`
  - command-based hidden evaluator
- `engine/submissions/git.py`
  - fetch repo/ref and read submission manifest

## 4. Standardize evaluation output

Every problem should produce a `result.json` with:

- status
- primary metric
- primary score
- secondary metrics
- artifact pointers
- evaluator version
- submission metadata
- reproducibility metadata

## 5. Build a local evaluator server first

Before building a broad distributed system, implement:

- one evaluator coordinator process
- N local scoring workers
- isolated temp directories
- SQLite locking / transactions for promotion
- git fetch + checkout of submitted refs

That will force the right abstractions while staying close to the intended deployment model.

## 6. Add a submissions/results view

The original progress chart is part of the appeal, but now we also want branch intelligence.

Build a generic report that can show:

- incumbent score over wall-clock time
- all submissions by branch / proposer / strategy tag
- success rate by proposer type
- stale winners vs promoted winners
- promotion history

That should read from structured history, not bespoke ML logs.

## 7. Add benchmark/example problems beyond ML

To prove “anything”, add at least 2 non-ML examples:

1. **Prompt optimization**
   - state = prompt text under `state/`
   - submission = branch changing prompt files + manifest
   - evaluator = hidden benchmark on private eval set
   - metric = judge score / preference win rate / accuracy

2. **Heuristic/config optimization**
   - state = JSON config under `state/`
   - submission = branch changing config + manifest
   - evaluator = simulator or benchmark script
   - metric = objective score

If the framework only works for the original ML example, it is still ML-specific in disguise.

## Suggested implementation sequence

## Milestone 0 — align on the shell-repo contract

- finalize vocabulary: state, submission, evaluator, incumbent, worker, coordinator
- decide the standard location for mutable state, likely `state/`
- decide the submission manifest shape
- decide how the canonical incumbent ref is represented

## Milestone 1 — formalize autoresearch as the first shell-repo problem

- add submission manifest support
- add result manifest output
- add SQLite submission/evaluation history
- wrap the current loop in evaluator/coordinator abstractions

Success criterion:
- current autoresearch demo still works
- but now runs through the same submission and scoring interfaces future problems will use

## Milestone 2 — evaluator-side branch scoring

- fetch remote refs
- materialize candidates in isolated workdirs
- score them privately
- store submission metadata and results
- promote winners transactionally

Success criterion:
- a branch pushed by an agent can be scored by a hidden local evaluator and recorded correctly

## Milestone 3 — local parallel scoring

- multiple workers scoring different submissions concurrently
- promotion guarded by DB transaction / optimistic locking
- progress graph generated from DB

Success criterion:
- several submitted branches can be evaluated concurrently without corrupting incumbent state

## Milestone 4 — generic problem spec

- implement `problem.yaml`
- implement git-native submission adapters
- implement command evaluator and private evaluator hooks
- allow non-ML problems without editing engine internals

Success criterion:
- a new code/prompt/config problem can be onboarded mostly by arranging `state/`, adding a scorer, and defining the submission contract

## Milestone 5 — distributed evaluator API

- networked evaluator/coordinator
- remote workers
- artifact store abstraction

Success criterion:
- many hosts can contribute scoring capacity to one evaluator-owned optimization run

## Milestone 6 — richer search strategies

Once the platform is stable, add:

- beam search
- diversity pressure / novelty search
- population-based optimization
- crossover / recombination
- multi-objective optimization
- bandit allocation across proposer types

These are important later, but premature before the shell-repo and evaluator model is solid.

## Key architectural decisions to make early

## 1. What is the source of truth for state?

Recommendation:

- use git refs/commits as the source of truth for state in v1
- use SQLite as the source of truth for metadata, submissions, evaluations, and promotions

## 2. Should git be optional?

Recommendation:

- not in v1
- git should be the default and the main model at first

It is too aligned with the desired workflow to abstract away immediately.

## 3. How do we keep the evaluator hidden?

Recommendation:

- keep evaluator code and hidden data off-repo or in ignored/local-only paths
- expose only a submission API/CLI and a public problem contract
- never require proposers to run the real evaluator locally

## 4. How do we do transactional promotion?

Recommendation:

- maintain an `incumbent_ref` / `incumbent_commit` in the DB
- on promotion attempt, use a transaction:
  - read current incumbent
  - compare scores
  - if challenger is better, update incumbent
  - record promotion event
  - optionally update a local canonical branch
  - commit

## 5. What should a submission point to?

Recommendation:

- repo URL + ref as the canonical transport
- optionally include commit SHA for immutability
- keep patch-bundle submission as a later extension

## 6. How opinionated should proposal generation be?

Recommendation:

- not opinionated in the engine
- the engine cares about submission intake, private scoring, and promotion correctness
- LLM-based branch generation should be a first-class example, not a hard requirement

## Risks

## 1. Hidden evaluators are hard to debug for proposers

Risk:
- agents may overfit to weak public hints or thrash blindly

Mitigation:
- expose enough submission metadata and score summaries to make iteration possible
- optionally publish coarse feedback without leaking the evaluator

## 2. Arbitrary submitted branches may be dangerous to run

Risk:
- the evaluator executes untrusted or semi-trusted code

Mitigation:
- start in trusted/self-use settings
- use isolated workdirs and controlled execution
- add sandboxing later if needed

## 3. Parallel submissions can produce stale wins

Risk:
- many branches look like winners relative to an old incumbent

Mitigation:
- compare against the current incumbent at promotion time
- keep all evaluations, but promote only current winners

## 4. “Anything” can still become too abstract too early

Risk:
- the project becomes framework theater

Mitigation:
- keep autoresearch as the reference implementation
- add only abstractions justified by at least one more real problem

## Near-term recommendation

The best next move is:

1. commit to the shell-repo model
2. standardize on `state/` plus a submission manifest
3. build a hidden evaluator CLI/service that scores refs
4. store submissions and results in SQLite
5. make promotion to the incumbent a local transactional action
6. prove the pattern on autoresearch and one non-ML problem

That path is more concrete than a generic optimization framework and much closer to how this should actually be used.

## End-state vision

The ideal end state is something like:

- clone the repo template
- put your mutable problem state under `state/`
- define the public submission contract in the repo
- define the hidden evaluator locally on the scoring machine
- let many agents submit branches or refs
- score them privately and concurrently
- promote winners locally to the canonical incumbent
- watch the state improve over time

In that world, this repo becomes a practical black-box optimization shell:

- git-native
- evaluator-private
- branch-submission-driven
- parallelizable
- reusable across many domains

That is the real generalization of autoresearch: not autonomous ML research, but **private evaluation of public candidate branches against a measurable objective**.
