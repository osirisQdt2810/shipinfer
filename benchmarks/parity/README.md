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
the other. The backoff is compared through its **un-jittered `peek()` sequence**, identical
from initial/factor/cap alone; the jitter *bound* (`0.8·d ≤ delay ≤ d`) is asserted inside
each plane, at `tests/ingest/test_timing.py` and `csrc/tests/test_ingest.cpp`.

Everything is observed from the **actor's own thread**, at the source and sink calls, so the
trace is that thread's program order rather than a poll racing it.

## The golden rule

A golden is emitted **once** and committed:

```bash
python -m benchmarks.parity.drive_python --scenario reconnect --emit-golden
```

It refuses to overwrite without `--force`. **Regenerating one to make a plane pass is the
one thing this harness exists to prevent.** A difference is either a bug or an entry in
`known.py` — with a citation on both sides, an OPEN ledger line naming the fix, and a case
that reproduces it. `xfail` is banned: an entry whose divergence has been fixed must fail,
or the register rots into a suppression list.

The register is consulted **only across planes**. A fresh Python run against a
Python-emitted golden must be identical, because a difference there is drift within one
plane and no cross-plane decision can excuse it.

## Adding a case

1. Write `scenarios/<name>.scn` — the format's normative description is the docstring of
   `scenario.py`. Every enabled camera must end (`read exhaust`, `open
   SourceUnavailableError`, or a `sink closed`), or the loader refuses it.
2. Declare `records_min`: the floor the golden promises, so a vacuous one fails.
3. `--emit-golden`, read the file, commit it.
4. Add the name to `NAMES` in `benchmarks/tests/test_parity_ingest.py` and to the list in
   `csrc/tests/test_ingest_parity.cpp`'s `main`.

## The instrument is written twice

`ScriptedSource` exists in `drive_python.py` and in `csrc/tests/scripted_source.h`, and that
duplication is this harness's own risk. It is contained by emitting `source_open`,
`source_read` and `source_close` for every call: the two scripts drifting apart shows up in
the source-event stream first, where it cannot be mistaken for a divergence in the actor.
