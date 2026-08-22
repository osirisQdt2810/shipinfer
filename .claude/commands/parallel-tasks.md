---
description: Decompose a task list, run non-conflicting parts in parallel (coder→reviewer each), then one final reviewer checks the whole for cross-task conflicts/impact.
argument-hint: paste the task list (or a path/description of it)
---

# /parallel-tasks — decompose → parallel coder/review → one final integration review

Input: a task list in `$ARGUMENTS` (or ask the user to paste it / point at a file). It may be one
task or many; they may be independent, share a few files, or together form a whole feature.

The shape is bookended by two **single-owner** roles, with parallel work in the middle:

> **Planner** (one) → **Coder→Reviewer per task** (parallel, isolated) → **Final integration reviewer** (one, over everything)

## The invariant (this is what keeps parallelism safe)
**Two agents must never write the same file at the same time.** The Planner's job is to carve the
work so anything running concurrently is file-disjoint; overlapping/related work is sequenced in one
lane; each parallel lane runs in its own git worktree. Speed is never worth a data race on the tree.

## Phase 1 — Plan (the Planner — one agent that writes no code)
A single planner agent (this repo's `planner`, else `Explore`+`general-purpose`) does NOT write code;
it produces the parallelization plan:
1. **Decompose** the request(s) into concrete, self-contained sub-tasks.
2. **Scout** the repo (Grep/Read) to estimate each sub-task's **file footprint** and blast radius.
3. **Group into lanes**:
   - Footprints overlap, or the tasks are related / one feature → **same lane**, run **sequentially**
     in dependency order.
   - Lanes are mutually file-disjoint → run in **parallel**. Unsure? Same lane (correctness > speed).
4. **Flag the seams** — the shared modules, interfaces/contracts, config, schema/migrations, or
   registries that more than one lane might touch or be impacted by. These are what the final review
   scrutinizes.
5. **Output the plan**: lanes, order within each, max concurrency (default ≤ 5), and the flagged
   seams. Show it to the user and adjust before spawning anything.

Not everything should parallelize: a single tightly-coupled feature is usually one sequential lane —
splitting it fights the coupling. Parallelism pays only when tasks are genuinely independent.

## Phase 2 — Build + per-task review (parallel across lanes, sequential within a lane)
Each parallel lane works in its **own git worktree**. For **every task**, run the loop:
1. **Coder** (`coder`, else `general-purpose`) — implement the task + its tests; validate with *this
   project's own* tooling (build/tests/lint/types). Leave the lane's worktree green.
2. **Reviewer** (`reviewer`, else `general-purpose`) — independent pass over that task's diff:
   correctness (each finding with a concrete failing input→wrong output), architecture (abstraction,
   reuse, cohesion, coupling; respect the repo's seams/conventions), simplification/refactor, tests.
   Verdict: ship / needs-fix.
3. **Fix loop** — if real issues, a coder/`debugger` fixes them in the same worktree, then re-review.
   Bound it (default 2 rounds); if it won't converge, stop and surface it.

Mechanism by scale: **many tasks** → a Workflow (`pipeline(tasks, coder, reviewer, fix)`,
`isolation:'worktree'` on the coder stage, `parallel()` across lanes); **a few** → the Agent tool
(coder subagents in one message run concurrently, one worktree per lane, then reviewers, then fixes).
The Workflow tool needs opt-in — invoking this command is that opt-in.

## Phase 3 — Integrate, then ONE final review over the whole
1. **Merge** each finished lane onto the working branch. Disjoint lanes merge cleanly — **a merge
   conflict means the Planner mis-partitioned**; resolve it and say so.
2. **Final integration reviewer** (one agent, fresh eyes over the COMBINED diff) — checks what
   per-task reviewers structurally could not, because each saw only its own slice:
   - cross-task **conflicts** (two lanes changed the same behavior/contract in incompatible ways),
   - **mutual impact** — an interface/signature/config/schema change in lane A that silently breaks
     lane B (or existing callers),
   - **duplication / divergence** — two lanes added the same helper, or diverging patterns for one
     thing,
   - overall **coherence** and that the seams the Planner flagged still hold.
   It reports real issues with a concrete impact scenario each; a clean pass is stated explicitly.
3. **Full validation** — run the entire test/lint suite on the integrated tree (each lane only
   validated its slice). Route any fallout to `debugger`/`coder`.
4. **Report** per task (what changed, per-task verdict) + the final integration verdict + test result.

## Guardrails
- Never two coders on overlapping files; every parallel coder in its own worktree.
- Reviewers never rubber-stamp — "reviewed, nothing found" is a real result; silence is not.
- Keep fix loops bounded; surface non-convergence instead of looping.
- Commit/push only when the user asks, and branch off the main branch first.
