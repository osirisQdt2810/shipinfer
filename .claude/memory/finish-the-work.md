---
name: finish-the-work
description: A PR being opened is not a stopping point — the ledger .claude/TASKS.md decides when a session may end, and a Stop hook enforces it
metadata:
  type: feedback
---

The operator's instruction, in capitals (`docs/qa/user.md` V38): *"BẠN PHẢI LÀM CHO XONG,
không được dở chừng và bắt tôi phải kêu bạn tiếp tục"*. It then failed twice more (V37, V39),
both times in the same shape: open a PR, write a summary containing "I'll continue", end the
turn.

**Why:** ending a turn *is* stopping. There is no "and then I keep going" — control returns to
the operator, and they have to prompt again, which is the exact thing they asked not to have to
do. The rule was visible both times, so the failure was not memory. It was that nothing
checked: "is there work left?" got answered by a feeling at the end of a long stretch of tool
calls instead of by a file.

**How to apply:** keep `.claude/TASKS.md` current as you go. `scripts/hooks/unfinished_work.py`
is a `Stop` hook that blocks while any line is `[ ]` or `[~]` and hands the list back. Mark
`[!]` with the question for work that genuinely needs the operator — that does not block. A
milestone (a PR opened, a push, a green suite) is something to *report while continuing*, never
a reason to end the turn. See [[pr-review-loop]].
