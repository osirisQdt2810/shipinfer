---
name: commit-identity
description: Commits must be authored as osirisQdt2810 via its GitHub noreply address, not the work email
metadata:
  type: feedback
---

Commit as `osirisQdt2810 <152402665+osirisQdt2810@users.noreply.github.com>` in every repo of
this project. Set it **repo-local** (`git config user.name/user.email`) rather than passing
`-c user.name=... -c user.email=...` per commit, and set it in submodules too — they carry
their own config.

**Why:** GitHub attributes a commit by its author *email*, not by who pushed it. Committing
as `phucnp <phuc.nguyen@moreh.com.vn>` linked every commit to a different GitHub account,
`phucnguyen-ht`, even though the repositories belong to `osirisQdt2810` — so the contributor
graph named someone else, and the work email was published in every public repo's log. The
`<id>+<login>@users.noreply.github.com` form is always verified on that account and leaks no
address.

**How to apply:** on any new repo for this project, set the two config values before the
first commit. If a repo's history is already mis-attributed, only re-author it when that is
safe — a sole commit with no PR can be `--amend --reset-author` and force-pushed with
`--force-with-lease`; a branch with an open PR must not be rewritten, because it invalidates
the review state. Verify with
`gh api repos/<owner>/<repo>/commits?per_page=1 --jq '.[].author.login'` — a `null` there
means the commit is linked to no account at all.

Related: [[git-remotes-ssh]], [[commit-coauthor-rule]], [[pr-workflow-automerge]]
