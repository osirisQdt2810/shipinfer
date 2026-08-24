---
name: pr-review-loop
description: The PR review loop is a loop, not a handoff — push, work elsewhere, check each blocking finding before fixing it, re-trigger by label when the review is wrong
metadata:
  type: feedback
---

The operator's protocol for CI, stated 24 Aug 2026 (`docs/qa/user.md` V32):

1. Push, then **go do other work** — the Claude review takes 10–20 minutes and idle-polling
   it wastes the wait.
2. `APPROVE` + the `automerge` label present → it merges itself. Say nothing, do nothing.
3. `BLOCKING` → **check whether the finding is actually true** before fixing it. Real: fix
   and push (the push re-triggers). Wrong: comment on the PR with the evidence, then
   re-trigger with `gh pr edit N --remove-label automerge && gh pr edit N --add-label automerge`
   — the workflow fires on `labeled`, so a label toggle re-runs the review without an empty
   commit.
4. Loop until merged.

**Why:** left as a handoff, the PR sits waiting for a human who was explicitly told not to be
asked. And a review taken on faith is worse than one argued with — round 6 of PR #3 reported
stale `shipinfer-imgproc` references that `git show HEAD:...` proved absent, and "fixing"
them would have made the next reviewer's map of the code wrong.

**How to apply:** keep `automerge` on for every PR except one editing `.github/workflows/**`,
which cannot pass the review job at all. See [[pr-workflow-automerge]] for the gate's exact
conditions, and [[keep-prs-small]] for why the loop got long in the first place.
