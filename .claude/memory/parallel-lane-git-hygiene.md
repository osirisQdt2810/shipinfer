---
name: parallel-lane-git-hygiene
description: Parallel coder agents in one checkout must stage only their own paths and must not create branches
metadata:
  type: feedback
---

When several coder agents work in parallel on one repository, either give each a real
`git worktree`, or keep them all on **one shared branch** and tell each agent explicitly:

- stage only its own paths (`git add shipvision/tracking tests/tracking`), and
- never run `git add -A`, `git add .`, `git commit -a`, `git stash`, or `git checkout`.

Do **not** ask each agent to `git checkout -b feat/<its-lane>` in a shared checkout.

**Why:** on 2026-08-23 four lanes were launched into one `shipvision` checkout, each told to
create its own branch. They took turns checking out, so the branch pointer no longer matched
the lane: the reid lane's commit landed on `feat/imgproc`, and `feat/reid` was left pointing
at the base commit. Disjoint *file* footprints were never the problem — the shared **index
and HEAD** were. The live hazard was one lane running `git add -A` and committing another
lane's half-written package.

**How to apply:** disjoint directories make a single accumulating branch perfectly coherent —
it is what a clean merge of the lanes would have produced anyway — so the cheap fix is one
branch plus explicit staging, and split by path into per-lane branches at integration time
if separate PRs are wanted. Reach for `isolation: "worktree"` when lanes must genuinely
share files or run different branches at once; it costs disk and setup per agent.

Related: [[pr-workflow-automerge]], [[two-repo-split]]
