---
name: pr-workflow-automerge
description: Work goes on a branch and through a PR that follows the template and carries the automerge label
metadata:
  type: feedback
---

Non-trivial work is branched, pushed, and opened as a PR whose body fills every heading of
`.github/pull_request_template.md` (Context → Content/Changes → Test Plan with real
output). The PR carries the `automerge` label so the pipeline merges it once tests are
green and the Claude review returns APPROVE.

The user added a `CLAUDE_CODE_OAUTH_TOKEN` actions secret to the repository so the review
job runs; the expectation is that a review's findings get fixed and pushed until the PR is
complete, without asking.

**Why:** they want the review-and-fix loop to run to completion autonomously.

**How to apply:** after pushing, create the PR with the filled template and the `automerge`
label, then poll the checks and act on review feedback. The local `gh` CLI is not
authenticated — `gh auth login` is needed once before PRs can be opened from here.
