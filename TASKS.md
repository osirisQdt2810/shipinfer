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
| full chain, 28 Sep, static plans (tracked, ABBA n=2) | [1 061, 1 120] | 1.13–1.19× |
| **full chain, dynamic plans** (tracked, ABBA n=3) | **[1 213.8, 1 299.3]** | **1.29–1.38×** |
| detect only, 28 Sep (offer-bound) | [1 909, 1 922] | ≥ 2.0× |
| **target** | **≥ 2 816** | **3.0×** |

The 28 Sep stage ablation found three things:

- **Every engine plan is static-batch, and every partial batch is padded.** The detector fills
  2.7 of its 8 rows. Padding is about 40 % of GPU time.
- **The mtmc barrier holds a worker about 20 ms a frame.** The bench's camera roster can never
  close an instant `complete`.
- **Even at full batch the chain costs 1.43–1.49 GPU-ms a frame.** So four GPUs top out at
  2 680–2 790 frames/s with no waste at all.

**Step 1 is done: ingest is not the wall.** Detect-only reads and accepts about 3 810/s of
4 000 offered, and the RTSP generator is ruled out, since 2 or 4 servers give the same
numbers. The 3× gap is entirely in the chain after the detector.

**Step 3 is done: dynamic-batch plans.** The full chain tracks [1 213.8, 1 299.3] against
static's [1 067.5, 1 116.9], a gain of +14.8 %. GPU SM use fell from 88–91 % to 62–66 %,
so the GPU is no longer the limit. What binds now is how long each frame holds its worker.

The plan, approved 28 Sep:

1. Measure the ingest ceiling at 3 200/4 000 offered.
2. Give the bench an mtmc roster the fleet can actually complete.
3. Dynamic-batch engine plans.
4. Take the mtmc wait off the worker, only if needed after step 2.
5. Run a frame's object models concurrently.
6. Re-sweep the scheduling knobs.
7. CUDA graphs.
8. The output-changing levers: segmenter work, INT8, embedding cadence.

Each step-8 lever reports its output delta; the operator allowed them on that condition (V186).

> V167's "3 000 is met" was a four-GPU reading multiplied by four. V182 pins the target to four
> GPUs, so that claim is withdrawn.

## 2. Needs the operator

| Ledger ID | What is needed |
|---|---|
| `SIBLING-REPOS-RETIRED` | Run `! gh auth refresh -h github.com -s delete_repo`, so the four superseded repos (`shipinfer-mot`, `-imgproc`, `-reid`, `-mtmc`) can be deleted. The two with history are already bundled in `.artifacts/archive/`. |
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
