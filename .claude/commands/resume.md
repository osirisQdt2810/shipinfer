---
description: Resume work from last session by reading the journal
---

Help me pick up where I left off.

Steps:

1. Read `.claude/memory/MEMORY.md` — the durable facts about how this project is worked on.
2. Read the newest 1–2 entries in `.claude/JOURNAL.md`.
3. `git status`, `git log --oneline -8`, and `git branch --show-current` — what is actually
   on disk versus what the journal claims.
4. If the latest entry names specific files, read them to refresh the in-progress context.
5. `gh pr list --state open` if there might be a PR waiting on a review.

Then give me:

- **Where things stand** — the last thing finished, and whether it landed.
- **What is in flight** — uncommitted changes, an open branch, a PR awaiting review.
- **The obvious next step**, from the journal's "Next" section, with a one-line reason.
- **Anything that looks wrong** — a dirty tree that contradicts the journal, a failing
  check, a branch that never got a PR.

Do not start work. Brief me, then wait.
