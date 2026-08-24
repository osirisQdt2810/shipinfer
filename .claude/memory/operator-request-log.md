# Every operator request is logged verbatim in docs/qa/user.md

Operator, 23 Aug 2026: `rule: ghi lại toàn bộ prompt yêu cầu của tôi vào docs/qa/user.md
nhé`.

**Why:** most of this project's rules arrived as short mid-work `note:` interjections, and a
rule stated once in passing does not survive the next compaction. Worse, the harness does
**not** persist a message sent while a turn is already running as a `user` record, so those
requests exist nowhere afterwards except in an assistant-written paraphrase. Without a log,
the operator has to restate rules they already gave.

**How to apply:**
- Append each new request to Section 1 of `docs/qa/user.md` as it arrives, in the operator's
  own words and original language — not summarised, not translated, not tidied.
- Keep Section 1 (verbatim, from the transcript) and Section 2 (reconstructed, from
  compaction summaries) separate and labelled. Presenting a paraphrase as a quotation would
  put words in the operator's mouth.
- Section 3 is the standing-rules index — the fastest way to reload constraints at the start
  of a session, and the place a new rule gets its row.
- Scaffolding in English per the project docs rule; the quotations stay in the language they
  were written in, because a translated quotation is not a quotation.
