---
description: End-of-day journal entry summarizing today's session
---

Write today's entry at the top of `.claude/JOURNAL.md`.

Steps:

1. `git log --oneline --since=midnight` and `git diff --stat HEAD~N` to see what actually
   changed today. Do not write the entry from memory of the conversation.
2. Re-read the previous entry so the new one continues it rather than repeating it.
3. Append at the **top**, under the heading, using the house format:

```
## YYYY-MM-DD — <one line: what changed, not what was discussed>

**What this session did.** Two or three sentences.

**Built.** Bullets, grouped by area. Name the seam, not the file list.

**Evidence.** Real numbers: test counts, bench throughput and p99, per-device share.
Paste what the tool printed; do not paraphrase a measurement.

**Notable bugs found and fixed.** The interesting ones — the ones a future reader would
otherwise rediscover. Include what the wrong behaviour looked like.

**Next.** What the next session should pick up, and why that and not something else.
```

4. If a large feature landed, also append to `.claude/FEATURE_LOG.md`. If a pattern
   changed, check that the ADR was written.

Be honest about what did not work. An entry that records only successes is useless the next
time the same wall is hit.
