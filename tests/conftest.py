"""Shared fixtures.

Two rules the whole suite depends on:

1. **The default run needs no GPU.** ``pytest`` selects ``-m "not gpu"``, and everything
   outside that marker must pass on a laptop with no NVIDIA driver. The scheduler's
   fairness and balancing guarantees are the most valuable things here to test, and a test
   that needs sixteen GPUs gets written once and then never run again.
2. **No shared global state between tests.** Each test builds its own server, its own
   metrics registry and its own repository, so a counter from one cannot leak into another.
3. **The offline tier hides the accelerators.** On a host with a driver, "needs no GPU" is
   not the same as "touches no GPU": the server tests build real ``DeviceManager``s, and
   with devices visible each one opened a CUDA context. That made the offline tier depend
   on how much VRAM someone else's job had left — 110 tests failed with
   ``CUDA error: out of memory`` on a shared box while GPU 0 was full. So when no device
   marker is selected, :func:`pytest_configure` blanks ``CUDA_VISIBLE_DEVICES`` and
   ``HIP_VISIBLE_DEVICES`` before anything imports torch, and a host run *is* the CI run.
"""

from __future__ import annotations

import os
import re
import shutil
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from shipinfer.core.metrics import ServerMetrics
from shipinfer.core.request import InferenceRequest, RequestContext, ResponseFuture
from shipinfer.core.settings import ServerSettings
from shipinfer.core.types import Tensor
from shipinfer.scheduling.work import WorkItem

DATA = Path(__file__).parent / "data"


# -- tiers ------------------------------------------------------------------------------

DEVICE_MARKERS = ("gpu", "multigpu")


def device_tier_requested(markexpr: str) -> bool:
    """Can this ``-m`` expression select a test that carries a device marker?

    Exact for the boolean marker expressions pytest accepts, and decided before collection —
    which is the only time it can be decided, because the CUDA runtime reads
    ``CUDA_VISIBLE_DEVICES`` once. Every identifier in the expression other than the device
    markers is a free variable, and the question is whether *some* assignment of them, with
    one device marker true and the others false, satisfies the expression: ``"not slow"``
    selects a fast GPU test, ``"gpu and slow"`` selects a slow one, ``"not gpu"`` still
    selects a ``multigpu``-only test, and the default ``"not gpu and not multigpu"`` selects
    none of them. The free variables are enumerated exhaustively; there are never more than
    a handful in a command line. An expression pytest cannot parse, or one with implausibly
    many identifiers, answers "yes" — a typo must never hide a device from a run that meant
    to use one, and pytest will report the typo itself.
    """
    from itertools import product

    from _pytest.mark.expression import Expression

    if not markexpr.strip():
        return True
    try:
        expression = Expression.compile(markexpr)
    except Exception:
        return True
    free = sorted(
        {token for token in re.findall(r"[^\s()]+", markexpr) if token not in _EXPRESSION_WORDS}
        - set(DEVICE_MARKERS)
    )
    if len(free) > 12:
        return True
    for device in DEVICE_MARKERS:
        for values in product((False, True), repeat=len(free)):
            keywords = dict(zip(free, values, strict=True))
            keywords[device] = True
            if expression.evaluate(lambda name, k=keywords: k.get(name, False)):
                return True
    return False


_EXPRESSION_WORDS = frozenset({"and", "or", "not"})


def pytest_configure(config) -> None:
    """Hide the accelerators from a run that selected no device-tier test.

    Runs before collection, which is before any test module imports torch: the CUDA runtime
    reads ``CUDA_VISIBLE_DEVICES`` once, at its first initialisation, so setting it later
    would be silently too late. An operator who passed ``-m gpu`` is unaffected — that run
    still meets the container gate below.
    """
    if device_tier_requested(config.getoption("markexpr") or ""):
        return
    for variable in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        os.environ[variable] = ""


@pytest.fixture(scope="session")
def tier_predicate():
    """The predicate :func:`pytest_configure` uses, for tests that state its truth table."""
    return device_tier_requested


@pytest.fixture(scope="session")
def probe_device_count():
    """The driver probe :func:`pytest_collection_modifyitems` uses, count and failure together."""
    return _probe_device_count


@pytest.fixture(scope="session")
def device_count_or_zero():
    """The driver probe wrapper :func:`pytest_collection_modifyitems` uses, for tests that pin its two answers."""
    return _device_count_or_zero


# -- markers ------------------------------------------------------------------------------


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items) -> None:
    """Gate the GPU tiers on a container, then skip them when there is no device.

    ``trylast`` is load-bearing. pytest applies ``-m`` deselection in its *own*
    ``pytest_collection_modifyitems``, so running before it means ``items`` still holds
    every GPU test even for a plain offline run — and the gate below then refused the
    offline tier on any host, which broke CI on a plain runner. Running last means the list
    is what will actually execute.

    Two separate concerns, and the order matters.

    **The container gate comes first.** The rule that measurements run in a container was
    enforced by a hook reading the *text* of a shell command, and review showed a deny-list
    over text cannot be made sound: ``( pytest tests/ )``, ``eval "pytest tests/"``,
    ``coverage run -m pytest`` and six other ordinary spellings all walked through it. This
    check is inside the process that would do the work, so no spelling avoids it — and it
    has to run before the skip logic below, which returns early on a host that *has* GPUs
    and would therefore let the whole device tier run there.

    Only the device tiers are gated. The offline tier must keep running anywhere: that is
    ADR-001, it is what CI does on a plain runner, and it is the promise that makes the pure
    layers verifiable without a driver.

    **Then the skip.** ``-m gpu`` on a GPU-less host should report "skipped", not "failed":
    the tests are fine, the hardware is absent, and conflating the two makes a red CI run
    meaningless.
    """
    from shipinfer.runtime import containment
    from shipinfer.runtime.platform import device_count

    device_markers = {"gpu", "multigpu"}
    selected = [i for i in items if device_markers & set(i.keywords)]
    if selected:
        try:
            containment.require_container(f"{len(selected)} GPU-tier test(s)")
        except RuntimeError as exc:
            # `pytest.exit` rather than letting the raise escape: an exception out of a
            # collection hook is reported as INTERNALERROR with a traceback, which reads
            # like a broken suite rather than a refused run. The operator needs the reason,
            # not our stack.
            pytest.exit(str(exc), returncode=pytest.ExitCode.USAGE_ERROR)

    # Only ask the driver when something actually needs it, and survive it refusing to answer.
    #
    # `device_count()` imports torch and touches CUDA. On a box whose driver is unwell that
    # raises — `DeferredCudaCallError` from torch.cuda's deferred `_check_capability`, in my
    # case — during *collection*, which errors every test in the run including the ones that
    # need no device at all. That happened: 28 failures across `tests/server/`, none of them
    # touching a GPU, on a plain `pytest`.
    #
    # ADR-001 says the offline tier must pass on a machine with no driver. A machine with a
    # *broken* driver is the same promise, and this hook was breaking it.
    if not selected:
        return

    count, failure = _probe_device_count(device_count)
    if count >= 2:
        return
    # When the driver refused to answer, say so in the skip reason: an all-skipped `-m gpu` run
    # on a broken box must read "the driver failed", not "no hardware here".
    because = "" if failure is None else f" (asking the driver failed: {failure})"
    no_gpu = pytest.mark.skip(reason="needs a CUDA device" + because)
    no_multi = pytest.mark.skip(reason="needs at least 2 CUDA devices" + because)
    for item in items:
        if "multigpu" in item.keywords and count < 2:
            item.add_marker(no_multi)
        elif "gpu" in item.keywords and count < 1:
            item.add_marker(no_gpu)


def _device_count_or_zero(probe) -> int:
    """:func:`_probe_device_count` without the reason, for callers that only need the number."""
    return _probe_device_count(probe)[0]


def _probe_device_count(probe) -> tuple[int, str | None]:
    """How many devices there are, or zero and the failure if asking was itself a failure.

    A driver that raises rather than reporting zero is, from here, indistinguishable from a
    machine with no driver — and the right response to both is to skip the device tier, not to
    abort collection for the whole suite.

    The `except Exception` is deliberate and the narrow alternative was tried: the failure that
    actually bit was `torch.cuda.DeferredCudaCallError`, which is not exported anywhere
    importable without touching `torch.cuda` — the very thing that is failing. Catching by type
    would mean importing the module whose import is the problem.
    """
    try:
        return probe(), None
    except Exception as exc:
        import warnings

        warnings.warn(
            f"could not ask the driver how many devices there are ({exc!r}); treating this as "
            f"a machine with none, so the device tiers skip and the offline tier still runs",
            stacklevel=2,
        )
        return 0, repr(exc)


# -- repositories ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def demo_repository_path() -> Path:
    """The repository shipped with the project, used as a realistic fixture."""
    return Path(__file__).resolve().parents[1] / "model_repository"


@pytest.fixture()
def tmp_repository(tmp_path: Path) -> Iterator[Path]:
    """A minimal two-model repository, writable by the test."""
    root = tmp_path / "model_repository"
    (root / "echo" / "1").mkdir(parents=True)
    (root / "echo" / "config.yaml").write_text("""
name: echo
platform: mock
max_batch_size: 8
inputs:
  - {name: x, data_type: FP32, dims: [4]}
outputs:
  - {name: y, data_type: FP32, dims: [4]}
instance_groups:
  - {kind: KIND_CPU, count: 2}
dynamic_batching:
  enabled: true
  max_queue_delay_us: 2000
  preferred_batch_sizes: [2, 4, 8]
parameters:
  latency_ms: 0.5
""".lstrip())
    (root / "slow" / "1").mkdir(parents=True)
    (root / "slow" / "config.yaml").write_text("""
name: slow
platform: mock
max_batch_size: 4
inputs:
  - {name: x, data_type: FP32, dims: [2]}
outputs:
  - {name: y, data_type: FP32, dims: [2]}
instance_groups:
  - {kind: KIND_CPU, count: 1}
dynamic_batching:
  enabled: false
parameters:
  latency_ms: 5.0
""".lstrip())
    yield root
    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture()
def settings(tmp_repository: Path) -> ServerSettings:
    return ServerSettings(model_repository=tmp_repository)


@pytest.fixture()
def metrics() -> ServerMetrics:
    return ServerMetrics()


# -- request helpers ---------------------------------------------------------------------


@pytest.fixture()
def make_request():
    """Build an ``InferenceRequest`` with sensible defaults."""

    def _make(
        model: str = "echo",
        *,
        camera: str = "cam0",
        frame: int = 0,
        width: int = 4,
        rows: int = 1,
        **kwargs,
    ) -> InferenceRequest:
        return InferenceRequest(
            model_name=model,
            inputs={"x": Tensor.from_numpy(np.zeros((rows, width), dtype=np.float32))},
            context=RequestContext(camera_id=camera, frame_id=frame),
            **kwargs,
        )

    return _make


@pytest.fixture()
def make_item(make_request):
    """Build a ``WorkItem`` around a request."""

    def _make(**kwargs) -> WorkItem:
        request = make_request(**kwargs)
        return WorkItem(request, ResponseFuture(request))

    return _make
