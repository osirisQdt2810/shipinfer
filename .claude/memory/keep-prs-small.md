---
name: keep-prs-small
description: A PR gets few commits and few files — PR #3's ~100 commits and 20k+ lines is the counter-example the operator named
metadata:
  type: feedback
---

Split work by seam and open the next PR rather than letting one grow. If a branch is past
roughly 15 commits, that is the signal.

**Why:** PR #3 reached ~100 commits and over 20 000 changed lines, which is past GitHub's
diff API limit — the reviewer had to check the branch out instead of reading a diff, and six
review rounds followed. The operator called it out directly (`docs/qa/user.md` V33): this one
is allowed to stand, the next must not repeat it. Size is not a cosmetic problem; it is why
the review could not see the change whole, and why fixes to earlier rounds' fixes kept
landing in the same PR.

**How to apply:** one seam per PR — the harness, then the defects it found, then the docs.
Rebasing a long branch into fewer commits helps the diff but not the review scope; the real
fix is opening earlier. See [[pr-review-loop]].
