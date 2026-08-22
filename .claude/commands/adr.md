---
description: Create a new Architecture Decision Record in DECISIONS.md
---

Add an ADR to `.claude/DECISIONS.md` for the decision I describe.

Steps:

1. Read `.claude/DECISIONS.md` and take the next number.
2. If this decision changes an existing one, mark that ADR **Superseded by ADR-NNN** rather
   than editing it — the record of what we used to believe is the useful part.
3. Append the new ADR using the house format:

```
## ADR-NNN — <the decision, as a statement not a topic>

**Status:** Accepted · <YYYY-MM-DD>

**Context.** What forced a choice. Include the concrete constraint or the measurement —
"a 1080p frame is ~6 MB" beats "frames are large".

**Decision.** What we do, in the imperative. Name the alternative that was rejected and why.

**Consequences.** What this costs, what it makes easy, and what a future reader will find
surprising. An ADR with no downside listed has not been thought through.
```

Keep it short and specific. If the decision is obvious enough that no reasonable engineer
would choose otherwise, it is not an ADR — it is a convention, and belongs in
`.claude/CONVENTIONS.md`.
