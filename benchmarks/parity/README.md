# The ingest parity harness

One scenario, driven over the **real `IngestManager` in both planes**, each writing a
deterministic JSONL trace, both held to one committed golden. This is the acceptance gate
for CLAUDE.md's sync rule: a change to a Python data-plane seam is not finished until the
C++ seam carries it, and until now nothing checked.

```
benchmarks/parity/scenarios/<name>.scn   what each camera's decode path is told to do
benchmarks/parity/golden/<name>.jsonl    the trace, emitted once by the Python plane
benchmarks/tests/test_parity_ingest.py   the Python plane against it, plus the vacuity guard
csrc/tests/test_ingest_parity.cpp        the C++ plane against the same file
```

Run them: `pytest benchmarks/tests/test_parity_ingest.py`, and
`python scripts/build_csrc.py --offline && ./csrc/build/test_ingest_parity` from the
repository root (or set `SHIPINFER_PARITY_GOLDEN` to this directory). Both are offline —
g++ alone, no CUDA, no GStreamer, and **no GPU tier by design**: everything compared here is
scheduling, bookkeeping and error taxonomy, none of which needs a device.

## The record kinds

Every record is `kind`, `camera`, integer `n[]` and string `t[]`, in that key order. **No
floats**: a rounding that differs in the last bit is a gate that flaps. The table is spelled
twice — `FIELDS` in `trace.py` and `kFields()` in `csrc/tests/parity_trace.h` — and
`TestTheFieldTablesAgree` fails if they drift.

| kind | n[] | t[] | emitted |
|---|---|---|---|
| `source_open` | `attempt` | `outcome` | scripted source, actor thread |
| `source_read` | `index` | `outcome` | scripted source, actor thread |
| `source_close` | `index` | — | scripted source, actor thread |
| `frame` | `frame_id` | — | recording sink, on being offered a frame |
| `drop` | — | `reason` | recording sink, on refusing one |
| `retry` | `attempt`, `peek_us` | — | observed when the actor's failure count rises |
| `state` | — | `from`, `to` | observed when the actor's state changes |
| `health` | `frames_read` … `consecutive_failures` | `state`, `last_error` | once per camera, after the stop |
| `stop` | `abandoned` | — | the fleet's `stop()` return |
| `end` | `cameras`, `frames_read`, `frames_published`, `frames_dropped` | — | the fleet's totals |

The first line is the header: `{"schema":1,"scenario":"…","plane":"python|cpp"}`.

## What is compared, and what is not

Per-camera sequences in order; fleet-level records as their own sequence. **Cross-camera
interleaving is never compared** — which camera's thread got there first is scheduler
nondeterminism, and a gate that flaps is a gate somebody turns off.

Elided on purpose: `captured_ns`/`captured_unix_ns` (wall clocks), `fps` (windowed over a
real clock), and the jittered delay itself — `mt19937_64` on one side and `random.Random` on
the other. The backoff is compared through its **un-jittered `peek()` sequence**, read from a
production `ExponentialBackoff` that each recorder builds from the same settings the actor's
is built from and steps in lockstep with the observed failure count. The actor's own instance
is private on both planes and its history is not recoverable anyway — `peek()` answers for the
attempt it is *at* — so a mirror is as close as this gets; what it buys is that a change to
either plane's `peek()` moves that plane's trace. Both planes recomputing `initial · factorⁿ`
from the scenario's own config, which is what the first version did, made the column
**unfailable**: `peek()` was changed to double every reconnect delay and the gate stayed green.
The jitter *bound* (`0.8·d ≤ delay ≤ d`) is asserted inside each plane, at
`tests/ingest/test_timing.py` and `csrc/tests/test_ingest.cpp`.

**`stop.abandoned` is 0 in all three goldens, so abandonment parity is untested here.** The
seam is real — a `stop()` whose join times out detaches the thread, and the two planes are
*known* to differ on what a second `stop()` then answers (`stop_fate_stickiness`) — but the
scenario grammar cannot express it. Abandoning a thread needs an open that blocks until the
driver releases it, i.e. a source that outlives the trace: the detached thread keeps emitting
records after the trace has been captured, which is nondeterministic in this plane and a
use-after-free on the C++ writer, which lives on `run_scenario`'s stack. A flaky parity gate
gets switched off, so the divergence is reproduced in-plane instead
(`test_a_second_stop_still_answers_live_on_the_python_plane`) and the C++ half states the rule
in `csrc/shipinfer/ingest/camera/actor.h`.

Everything is observed from the **actor's own thread**, at the source and sink calls, so the
trace is that thread's program order rather than a poll racing it.

## The golden rule

A golden is emitted **once** and committed:

```bash
python scripts/emit_parity_golden.py --scenario reconnect --emit-golden
```

The entry point sits under `scripts/` rather than inside the package because
`scripts/hooks/require_container.py` denies every `-m` module under the `benchmarks` root
— rightly, for the bench runners, and wrongly for this one, which imports numpy and
`shipinfer.ingest`, touches no device and produces no measurement. Documenting a command the
project's own hook denies teaches the reader to reach for `SHIPINFER_ALLOW_HOST_RUN`, so the
command moved instead; `TestTheDocumentedCommandsRun` holds every command on this page to it.

It refuses to overwrite without `--force`. **Regenerating one to make a plane pass is the
one thing this harness exists to prevent.** A difference is either a bug or an entry in
`known.py` — with a citation on both sides, an OPEN ledger line naming the fix, and a case
that reproduces it. `xfail` is banned: an entry whose divergence has been fixed must fail,
or the register rots into a suppression list — and `test_every_registered_divergence_still_
fires` in the C++ gate fails when an entry stops firing, because a fix at *this* plane's call
site would otherwise just stop printing `KNOWN:` and go unnoticed.

The register is consulted **only across planes**. A fresh Python run against a
Python-emitted golden must be identical, because a difference there is drift within one
plane and no cross-plane decision can excuse it.

**What each half proves is not the same thing.** `TestPythonPlaneMatchesGolden` runs the
Python plane against a file the Python plane emitted: that is determinism and
change-detection — it fails the moment a Python seam moves — and it is **not** evidence of
correctness or of cross-implementation agreement, because the golden has no independent
authority. All of the cross-plane evidence is in the C++ half, `csrc/tests/test_ingest_parity`,
which matches a golden it did not produce.

## Adding a case

1. Write `scenarios/<name>.scn` — the format's normative description is the docstring of
   `scenario.py`. Every enabled camera must end (`read exhaust`, `open
   SourceUnavailableError`, or a `sink closed`), or the loader refuses it.
2. Declare `records_min`: the floor the golden promises, so a vacuous one fails.
3. `python scripts/emit_parity_golden.py --scenario <name> --emit-golden`, read the file,
   commit it.
4. Add the name to `NAMES` in `benchmarks/tests/test_parity_ingest.py` and to the list in
   `csrc/tests/test_ingest_parity.cpp`'s `main`.

## The instrument is written twice

`ScriptedSource` exists in `drive_python.py` and in `csrc/tests/scripted_source.h`, and that
duplication is this harness's own risk. It is contained by emitting `source_open`,
`source_read` and `source_close` for every call: the two scripts drifting apart shows up in
the source-event stream first, where it cannot be mistaken for a divergence in the actor.

## The second seam: the request queue

The same shape, one directory down, and **one design decision inverted**.

```
benchmarks/parity/scenarios/queues/<name>.scn   what the queue is told to do, call by call
benchmarks/parity/golden/<name>.jsonl           the trace, emitted once by the Python plane
benchmarks/tests/test_parity_queues.py          the Python plane against it + the vacuity guard
csrc/tests/test_queue_parity.cpp                the C++ plane against the same file
```

Run them: `pytest benchmarks/tests/test_parity_queues.py`, and
`python scripts/build_csrc.py --offline && ./csrc/build/test_queue_parity`. Emit a golden
with `python scripts/emit_parity_golden.py --kind queue --scenario <name> --emit-golden`.

**Every queue record is a fleet record, and the item's camera travels in its words.** The
ingest gate groups per camera because thread interleaving is nondeterministic; here that is
exactly backwards — *which* camera's item comes out next **is** the invariant. A queue run is
single-threaded with no clock in it, so the whole trace is one ordered sequence and `diff.py`
needs no change to compare it. `kFleetKinds()` is a set on the C++ side for the same reason
the field names are a table: so a test can compare the two spellings.

| kind | n[] | t[] |
|---|---|---|
| `qput` | `rows`, `depth` | `camera`, `status` |
| `qbatch` | `items`, `rows` | — |
| `qserved` | `rows` | `camera` |
| `qdrop` | — | `camera`, `reason` |
| `qstats` | `accepted`, `rejected`, `evicted`, `expired`, `depth`, `capacity` | — |
| `qcam` | `depth`, `rejected`, `evicted`, `expired` | `camera` |

Two things the drivers **refuse** rather than compare, because neither queue promises them:
a `take` from an open empty queue (`get_batch` blocks there by contract, and a scenario that
hangs one plane is not a gate), and more than one drop in a single operation — the order of
several simultaneous drops is each queue's internal order. `close` is the exception: it
returns its drained items in order, so those are emitted from that sequence.

Each scenario carries one invariant and the golden *shows* it:

| scenario | what it holds shut |
|---|---|
| `fair_eviction` | a flood evicts **its own** oldest frames; the quiet camera's survives |
| `reject_is_the_default` | a full queue refuses; backpressure reaches the producer |
| `priority_lanes` | tracking-critical does not queue behind a background batch |
| `expiry_on_take` | an expired request is accepted, then dropped on the way **out** |
