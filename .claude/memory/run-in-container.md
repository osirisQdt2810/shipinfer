# Every test, benchmark and measurement runs in the container

Stated by the operator repeatedly and with rising emphasis: `note: important ! mọi thứ đều
phải chạy trong docker nhé`, then `rule: mọi test đều chạy trong container`, then
`System cần chạy chính xác trong container + chạy GPUs, chứ không phải là mock/fake test
trên CPU`.

**Why:** a host number is not a production number. Host `nvcc` on this box is 11.5 against
a 12.6 driver, so `sm_89` will not build at all, and host timings do not describe the
deployment. Reporting a host measurement as if it were a container one is the exact
dishonesty the rule prevents.

**How to apply:**
- `deploy/rootless/test.sh [pytest args]` for the suite, `deploy/rootless/prove.sh` for the
  container+GPU attestation, `make shell` for an interactive shell.
- Enforced, not remembered: `scripts/hooks/require_container.py` is a `PreToolUse` hook on
  `Bash` in `.claude/settings.json` and **denies** a host pytest / benchmark / device run.
  The hook exists because this rule was documented first and then quietly broken — the host
  is faster to iterate on, which is a reason, not a permission.
- Per-command override `SHIPINFER_ALLOW_HOST_RUN=1`, only with the operator's agreement, and
  say so in the report. There is deliberately no session-wide switch.
- The offline tier being CPU-only *by design* is not the same as the GPU path being covered.
  See [[release-gpu-when-done]] for what to do when the GPU work finishes.
