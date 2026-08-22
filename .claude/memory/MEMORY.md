# Project memory

Read at session start. One line per memory; the file it points at holds the detail.
This lives **in the repository** (not in `~/.claude`) so it travels with the code to any
machine the project is developed on.

- [ponytail-principle.md](ponytail-principle.md) — reuse optimised libraries; never hand-roll what torch/TRT already does
- [package-per-extension-point.md](package-per-extension-point.md) — folder + registry, one class per file
- [commit-coauthor-rule.md](commit-coauthor-rule.md) — when a commit gets the Claude co-author trailer
- [git-remotes-ssh.md](git-remotes-ssh.md) — all git remotes use SSH, never HTTPS
- [pr-workflow-automerge.md](pr-workflow-automerge.md) — every PR follows the template and carries `automerge`
- [design-references.md](design-references.md) — Triton and vLLM are the design references
- [workspace-rename.md](workspace-rename.md) — shipproj was renamed to shipinfer; the old path is a symlink
- [reference-repos.md](reference-repos.md) — what each checkout under `references/` is
