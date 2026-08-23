## Context
<!-- Why is this change needed? Paraphrase the request in your own words. -->
- Related issue:
- Background / motivation:
- Constraints / assumptions:

---

## Content / Changes
-
-

- Refactors:
- New features:
- Removed / deprecated behavior:

---

## Test Plan

### Test Details
<!-- Commands, configs, or steps used to test -->
-

### Test Output / Feature Demonstration
<!-- Paste real output. "Tests pass" is not a test plan. -->
-

---

## Module checklist
<!-- "N/A — <reason>" is a valid answer; deleting a line is not. -->

- **No upward dependency**: this repository does not import the `shipinfer` parent package.
  Modules are standalone libraries; the parent adapts them, never the other way round.
- **Extension points are registries**: a new tracker / matcher / backend is a new file plus
  a `@REGISTRY.register` decorator, not a branch in an `if/elif`.
- **Reuse over reimplementation**: nothing here reimplements what numpy, scipy, torch or
  torchvision already does well. If it does, say which and why the library was insufficient.
- **Numerical claims are measured**: any accuracy or speed claim has numbers in Test Output.
- **Determinism**: a stateful algorithm gives the same output for the same input sequence;
  ties are broken deterministically (a stable sort, not whichever hash ordering won today).
- **Tests**: new behaviour has tests; changed behaviour has updated tests; a test that
  needs no GPU is not marked as needing one.
- **Docs**: `README.md` and `CLAUDE.md` still describe what the code does.
