---
name: commit-coauthor-rule
description: Only large feature commits carry the Claude co-author trailer; small ones do not
metadata:
  type: feedback
---

Add `Co-Authored-By: Claude ...` **only** to commits for large features or large changes.
Small, incidental commits — the kind that "feel like the user could have done them" (a
config tweak, a typo, a one-line fix) — must not carry the trailer.

**Why:** the user wants attribution to reflect real authorship weight, not appear on every
trivial change.

**How to apply:** judge by the size and substance of the diff, not by who typed it. When in
doubt on a small commit, leave the trailer off.
