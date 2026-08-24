# Open work

The ledger `scripts/hooks/unfinished_work.py` reads. **While any line here is `[ ]`, the
session may not end** — the `Stop` hook refuses and hands the list back.

Written because intention failed twice: on 24 Aug I opened a PR, wrote "I'll continue", and
ended the turn instead. Ending a turn *is* stopping — there is no "and then I keep going" — so
the fix has to be a mechanism rather than a promise, because the promise is the thing that
broke.

## States

| mark | meaning |
|---|---|
| `[ ]` | open. The session may not end. |
| `[x]` | done, with the evidence named on the line. |
| `[~]` | in progress this session. Still blocks stopping. |
| `[!]` | **blocked on the operator**, with the question stated on the line. Does not block stopping — but the answer must be the first thing said. |
| `[-]` | dropped, with the reason. Only the operator drops work. |

## The rule

A stopping point is one of exactly three things:

1. every line is `[x]`, `[!]` or `[-]`;
2. an action needs the operator's confirmation before it is safe to take;
3. the operator interrupted.

**Opening a PR is not a stopping point. Pushing is not a stopping point. Writing a summary is
not a stopping point.** Those are milestones, and a milestone is a thing to report *while*
continuing.

---

## Now

- [ ] **PR #6** — round-2 blocking answered and pushed; waiting on re-review, then merge
- [ ] **PR #8** — round-1 blocking, 4 items: per-frame `cudaMalloc`/`cudaFree` on the dispatch
      path (and the reusable buffer voided by `(void)`), skipped-vs-failed branches
      indistinguishable, reassembly eviction emits no event and no attribution (ADR-005), no
      ADR and no FEATURE_LOG entry for a second data plane
- [ ] **V40** — restructure `csrc/` to mirror `src/`; `.hpp` and `.cpp` side by side, no split
      `include/`
- [ ] **PR #8 should-fix** — `complete()` compares set sizes not inclusion; `sharding.py` lands
      with the launcher that uses it, not here; publish the full counters with the 390.5 so it
      is like-for-like; a CI job that compiles `csrc/`; the nits list
- [ ] **Plane 3 — MOT and MTMC.** Absent entirely. `shipinfer` imports only
      `shipvision.detection.engine_build`. The DAG ends at the embedders and tracklets go
      nowhere. (V27)
- [ ] **`docs/qa/triton.md`** — the analysis is written; none of the eight features in its
      "should take" table is implemented (V26)
- [ ] **RTSP in the benchmark** — tests cover it, the benchmark replays JPEGs, so NVDEC has
      never been exercised by a measurement (R55)
- [ ] **Benchmark tiers algo and kernel** — only the system tier exists (R44)
- [ ] **PR #3 findings 2, 6, 7, 8** — Kafka `produce()` counted as success, `actor.stop(0.0)`
      abandoning every camera thread, `pending_frames` gauge going stale per camera,
      uninterruptible reconnect `time.sleep`
- [ ] **`wheels.sh` does not stage the TensorRT wheel**, so a fresh cache breaks every bench run
- [ ] **`conftest.py` calls `device_count()` during collection**, so an unhealthy driver reddens
      the *offline* tier — which ADR-001 says must need no driver
- [ ] **`shipvision` NV12 work** — 1021 lines uncommitted in the submodule, 26 tests passing,
      14 skipped for want of a native build. Needs its own PR in that repository (ADR-010)
- [ ] **The `std::memcpy` audit** — deferred by the operator until the system is complete (V28)
- [ ] **Re-verify `docs/qa/verification.md`** once the above lands
