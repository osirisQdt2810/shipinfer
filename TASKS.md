# TASKS — what is not done

As of 28 Sep 2026. This lists only the **open** items. The ledger they come from is
[`.claude/TASKS.md`](.claude/TASKS.md) — a test and the `Stop` hook parse it, so it stays the
source of truth; each row below names its ledger ID.

## 1. In progress — mine

**`THREE-X-BASELINE-ON-FOUR-GPUS` — the throughput target, and it is not met.**
The operator's target (V182): **≥ 3× the baseline, ~3 000 img/s**, for the whole pipeline
(decode → … → mtmc track), on the **same four GPUs** the baseline runs on: 3 × 938.6 = **2 816**.

| | img/s on four A5000s | vs baseline |
|---|---|---|
| baseline `sim_pipeline_v2` (fp16, saturated) | 938.6 | 1.0× |
| **full chain today** (tracked, n=8) | **[848.1, 939.2]** | **~1.0×** |
| detect only (n=3) | [1 805.2, 1 873.6] | ~1.95× |
| **target** | **≥ 2 816** | **3.0×** |

The 3× has to come from three places at once: ingest must read ≥ 3 000 img/s (50 RTSP cameras
read ~2 200 today); detection alone must clear 3 000 (it clears ~1 850); and the rest of the
chain must stop costing ~2× detection. Scheduling knobs are already spent — workers, batch
window and detector instances all measured flat at saturation.

Next, in order: stage ablation at saturation → the ingest ceiling on its own → the levers by
size (detector precision, how often each person is embedded — 7.8 embeddings a frame is the
largest single cost — segmenter resolution, instances per model). Any lever that changes what
the pipeline outputs is measured for that change too, and says so.

> V167's "3 000 is met" was a four-GPU reading multiplied by four. V182 pins the target to four
> GPUs, so that claim is withdrawn.

**`MERGED-BRANCHES-ARE-NOT-DELETED` — the merge step deletes the branch itself (V183).**
The repo's `delete_branch_on_merge` setting fires for a merge a person makes, not for one made by
the workflow's `GITHUB_TOKEN`, and that is how every `automerge` PR is merged — 257 of 275 such
branches were still on the remote. They are pruned now; the fix stops them piling up again.

## 2. Needs the operator

| Ledger ID | What is needed |
|---|---|
| `GPU 7 IS INVISIBLE TO CUDA` | CUDA cannot open GPU 7 (bus `D2:00.0`); the driver still lists it. `nvidia-smi --gpu-reset -i 7` (root, nothing holding it) or a reseat. ~12 % of the box idle. |
| `EVENTS-MISSING-STAGES-MEANS-TWO-THINGS` | Yes/no: is `pipeline/deepstream/run.py` deployed anywhere outside this box? If no, `missing_stages` becomes per-frame everywhere, schema v5 kept. |
| `T4` | Pull `nvcr.io/nvidia/deepstream` (~6 GB) so the DeepStream topology can run; and track or drop `mtmc_deepstream.py`, untracked at the root since 25 Aug and cited by `docs/design/topology-deepstream.md`. |
| `SV-LICENSE` | shipvision needs `LICENSES/MIT.txt` — whose name goes on the copyright line? |
| `API-WEDGED-REPORT-FLAKE-IS-NOT-A-TIMEOUT` | A budget call: wait for the next occurrence of the 1-in-25 flake (it now prints its own diagnosis), or spend hours bisecting. Default: wait. |
| `C11` | Deferred by the operator (V28) until the system is declared complete. |
| `TWO-UNMERGED-BRANCHES-WITH-NO-PR` | Keep or delete `feat/device-lanes-v2` (4 commits, 7 Sep) and `refactor/one-logger` (1 commit, 68 files, 28 Aug). Both conflict with `main` and have **no PR**, so deleting them loses the commits for good. The other 258 stale branches are already deleted. |

## 3. In the shipvision repository

| Ledger ID | What |
|---|---|
| `D5` | The one-crossing MTMC matchers are unreachable from shipping code — wire them or delete them, in a shipvision PR. |
| `C13` | shipvision has no native C++ MTMC tracker; V64 asked for one. |
| `C27` | Native NMS downloads its mask into pageable memory (`image_ops.cu:362`). The pinned-staging fix is recovered on `backup/csrc-native-pinned-nms`. |

## 4. Open by design

| Ledger ID | Why it stays open |
|---|---|
| `SHIPVISION-TRACK-LAST-MATCH` | The parity register's own test requires this line to be open until the fix lands; closing it early reddens the suite. Priced and deliberately unbuilt. |
