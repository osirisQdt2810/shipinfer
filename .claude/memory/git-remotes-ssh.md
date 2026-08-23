---
name: git-remotes-ssh
description: All git remotes for this project and its references use SSH, never HTTPS
metadata:
  type: feedback
---

Every remote is `git@github.com:...`, never `https://github.com/...` — including the
`references/` checkouts and any new clone from the `ShipControlPrj` org. The user restated
this even for public repositories.

**Why:** their authentication is SSH-key based; HTTPS remotes prompt or fail.

**How to apply:** when given a GitHub URL, translate it to the SSH form before cloning or
setting a remote. Verify with `git remote -v` after any change.
