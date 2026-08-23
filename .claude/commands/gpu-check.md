---
description: Run the GPU test tier plus a load benchmark, and report the evidence
---

Run the hardware tier and produce the evidence a PR needs. "The offline suite is green" is
not evidence that this server balances load; a per-device breakdown is.

Steps:

1. `shipinfer doctor` — confirm the devices, the accelerator kind and whether the native
   extension is built. If it reports no accelerator, stop and say so; do not fake it.
2. `pytest -q -m gpu` — the real-device tier.
3. `pytest -q -m multigpu` — the balancing tier, if at least two devices are visible.
4. `shipinfer bench person_embedder --cameras 50 --fps 20 --seconds 5 --skew 8`
   — the fairness and balance evidence under 8x camera skew.
5. If `native/` changed: rebuild first with `python scripts/build_native.py --arch <sm>`
   and re-run step 4, comparing against the previous numbers.

Then report:

- device inventory (count, model, free memory)
- test results for both markers, with the real tail output
- the bench table verbatim: throughput, p50/p95/p99, per-device share, per-camera min/median/max
- an explicit judgement on two things:
  - **balance** — is any device below ~8% or above ~18% of the load?
  - **fairness** — is the quietest camera being served in proportion to what it submitted?
- anything you could not run, and why

Do not round numbers in your favour and do not describe a run you did not perform.
