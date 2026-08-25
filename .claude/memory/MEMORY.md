# Project memory

Read at session start. One line per memory; the file it points at holds the detail.
This lives **in the repository** (not in `~/.claude`) so it travels with the code to any
machine the project is developed on.

- [ponytail-principle.md](ponytail-principle.md) — reuse optimised libraries; never hand-roll what torch/TRT already does
- [package-per-extension-point.md](package-per-extension-point.md) — folder + registry, one class per file
- [commit-coauthor-rule.md](commit-coauthor-rule.md) — when a commit gets the Claude co-author trailer
- [git-remotes-ssh.md](git-remotes-ssh.md) — all git remotes use SSH, never HTTPS
- [commit-identity.md](commit-identity.md) — author as osirisQdt2810 via its noreply address; GitHub links by email, not by pusher
- [pr-workflow-automerge.md](pr-workflow-automerge.md) — every PR follows the template and carries `automerge`
- [design-references.md](design-references.md) — Triton and vLLM are the design references
- [workspace-rename.md](workspace-rename.md) — shipproj was renamed to shipinfer; the old path is a symlink
- [reference-repos.md](reference-repos.md) — what each checkout under `references/` is
- [two-repo-split.md](two-repo-split.md) — shipinfer = system, shipvision = algorithms; one-way dependency
- [parallel-lane-git-hygiene.md](parallel-lane-git-hygiene.md) — parallel agents in one checkout: stage own paths only, never create branches
- [run-in-container.md](run-in-container.md) — every test/benchmark/measurement runs in the container; enforced by a PreToolUse hook
- [release-gpu-when-done.md](release-gpu-when-done.md) — free the GPU as soon as the task ends; the box is shared and VRAM is watched
- [operator-request-log.md](operator-request-log.md) — log every operator request verbatim in docs/qa/user.md
- [PR review loop](pr-review-loop.md) — push, work elsewhere, verify a blocking finding before fixing it, re-trigger by label
- [Keep PRs small](keep-prs-small.md) — few commits, few files; PR #3's ~100 commits is the counter-example
- [Finish the work](finish-the-work.md) — a PR opened is not a stopping point; .claude/TASKS.md plus a Stop hook decide when a session may end
