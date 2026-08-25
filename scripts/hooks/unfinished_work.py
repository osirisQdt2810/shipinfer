#!/usr/bin/env python3
"""`Stop` hook: refuse to end the session while `.claude/TASKS.md` still has open work.

WHY THIS EXISTS
---------------
On 24 Aug 2026 the operator wrote, in capitals: *"BẠN PHẢI LÀM CHO XONG, không được dở chừng
và bắt tôi phải kêu bạn tiếp tục"* — finish, do not stop halfway and make me tell you to
continue. It then happened twice more. Both times the shape was identical: a PR was opened, a
summary was written, and the turn ended — with the words "I'll continue" in it.

Ending a turn *is* stopping. There is no "and then I keep going". The failure was not
forgetting the rule; the rule was in front of me. It was that **nothing checked**, so the
question "is there work left?" was answered by a feeling at the end of a long stretch of tool
calls rather than by a file.

So this is deliberately not a reminder. A reminder is what already failed. It returns
`decision: block`, which the harness turns back into another turn, with the open list attached.

WHY IT IS SAFE TO BLOCK
-----------------------
Three escapes, because a hook that can trap a session is worse than the problem:

* Mark a line `[!]` with the question on it. Work genuinely waiting on the operator does not
  block — but the answer has to be the first thing said next.
* `SHIPINFER_ALLOW_STOP=1` for one command, when the operator has said so.
* A consecutive-block cap. If this fires ``MAX_CONSECUTIVE`` times in a row without the ledger
  changing, it stands down and says so: at that point the loop is not making progress and
  another turn will not help, which is itself worth telling the operator.

The counter lives beside the ledger and resets whenever the ledger's content changes, so real
progress always buys a fresh budget.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / ".claude" / "TASKS.md"
STATE = ROOT / ".claude" / ".tasks_state.json"

#: How many times in a row this may block without the ledger changing. Past this the loop is
#: not converging and another turn is not the answer.
MAX_CONSECUTIVE = 12

OPEN_MARKS = ("- [ ]", "- [~]")


def open_items(text: str) -> list[str]:
    """Every open line, minus the ledger's own explanatory table."""
    items: list[str] = []
    in_body = False
    for line in text.splitlines():
        if line.startswith("## Now"):
            in_body = True
            continue
        if not in_body:
            continue
        stripped = line.strip()
        if any(stripped.startswith(mark) for mark in OPEN_MARKS):
            items.append(stripped)
    return items


def allow(reason: str) -> None:
    print(json.dumps({"suppressOutput": True, "systemMessage": reason}))
    sys.exit(0)


def main() -> int:
    # The hook is fed the stop event on stdin; nothing in it is needed, but draining it keeps
    # the harness from seeing a broken pipe. Any failure reading it is irrelevant — this hook
    # decides whether a session may stop, and it must not fail closed on a pipe.
    with contextlib.suppress(Exception):
        sys.stdin.read()

    if os.environ.get("SHIPINFER_ALLOW_STOP") == "1":
        allow("unfinished-work hook: stood down by SHIPINFER_ALLOW_STOP=1")
    if not LEDGER.is_file():
        allow("unfinished-work hook: no .claude/TASKS.md, nothing to check")

    text = LEDGER.read_text(encoding="utf-8")
    items = open_items(text)
    if not items:
        STATE.unlink(missing_ok=True)
        allow("unfinished-work hook: the ledger is clear")

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    state = {}
    if STATE.is_file():
        try:
            state = json.loads(STATE.read_text())
        except Exception:
            state = {}
    # A changed ledger means real progress, so the budget resets. Unchanged means the last turn
    # did not move anything, and the cap is what stops that becoming a spin.
    count = state.get("count", 0) + 1 if state.get("digest") == digest else 1
    STATE.write_text(json.dumps({"digest": digest, "count": count}))

    if count > MAX_CONSECUTIVE:
        allow(
            f"unfinished-work hook: standing down after {MAX_CONSECUTIVE} turns with an "
            f"unchanged ledger — {len(items)} item(s) still open, but the loop is not "
            f"converging and that is worth saying out loud rather than spinning."
        )

    listing = "\n".join(items[:12])
    more = "" if len(items) <= 12 else f"\n  ...and {len(items) - 12} more"
    print(
        json.dumps(
            {
                "decision": "block",
                "reason": (
                    f"{len(items)} item(s) are still open in .claude/TASKS.md, so this is not "
                    f"a stopping point. Opening a PR, pushing, and writing a summary are "
                    f"milestones, not stopping points.\n\n"
                    f"{listing}{more}\n\n"
                    f"Pick the next one and keep going. Mark it `[~]` while you work and `[x]` "
                    f"with the evidence when it is done. If something genuinely needs the "
                    f"operator, mark it `[!]` with the question — that one does not block."
                ),
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
