"""What a resolved plan needs from the model repository, read without a driver.

ADR-014 puts the repository in the control plane and ADR-020 says the chain crosses to the
C++ plane as a resolved plan. This is the rest of that hand-over: the geometry a slot does
not declare, plus the runtime numbers the other plane restated by hand.

Read from `config.yaml` rather than from a loaded engine, because a plan is produced on a
machine that may have no driver at all (`shipinfer plan`, and the parity golden CI emits).
`pool.py` asks the *live* model the same questions and refuses the same way, which stays the
stricter check.

What is deliberately NOT here: `max_batch_size` and the input and output names. The C++ side
reads those off the engine it loaded, and `bench.cpp` states the rule -- "the config is a
claim about the plan; the plan is the fact". Carrying them would give one number two sources.
"""

from __future__ import annotations

from dataclasses import dataclass

from .model_config import InstanceKind, ModelConfig
from .model_repository import ModelRepository

__all__ = ["ModelRuntime", "model_extents", "model_runtimes"]


@dataclass(frozen=True, slots=True)
class ModelRuntime:
    """One model's resolved runtime, as the plan carries it."""

    #: The declared input ``(height, width)``, or ``None`` where the input is dynamic.
    extent: tuple[int, int] | None
    #: Instances PER DEVICE, or ``None`` where this model asks for no device instance at
    #: all. Triton's `instance_group.count` semantics, which is what `InstanceGroup.expand`
    #: divides among the shards sharing a GPU.
    instances: int | None
    #: The batch window in microseconds, and `0` where dynamic batching is off. Per model:
    #: the demo repository states 5000, 8000, 8000 and 3000 for its four.
    queue_delay_us: int
    #: The artefact, relative to the repository root: `<name>/<version>/<engine_file>`.
    artefact: str


def model_runtimes(repository: ModelRepository) -> dict[str, ModelRuntime]:
    """Each model's resolved runtime, keyed by model name."""
    return {name: _runtime_of(repository, name) for name in repository.names()}


def model_extents(repository: ModelRepository) -> dict[str, tuple[int, int]]:
    """Each model's declared ``(height, width)``, where it declares a static one.

    A model whose input is dynamic has no entry, so a chain that needs it is refused by name
    rather than guessed at.
    """
    return {
        name: runtime.extent
        for name, runtime in model_runtimes(repository).items()
        if runtime.extent is not None
    }


def _runtime_of(repository: ModelRepository, name: str) -> ModelRuntime:
    entry = repository.entry(name)
    config = entry.config
    batching = config.dynamic_batching
    engine_file = str(config.parameters.get("engine_file", "model.plan"))
    return ModelRuntime(
        extent=_extent_of(config),
        instances=_device_instances(config),
        queue_delay_us=batching.max_queue_delay_us if batching.enabled else 0,
        artefact=f"{name}/{entry.latest}/{engine_file}",
    )


def _device_instances(config: ModelConfig) -> int | None:
    """How many instances one GPU carries, read as ``InstanceGroup.expand`` reads it.

    The SUM over the groups that place a DEVICE instance: two ``KIND_GPU`` groups each place
    ``count`` on every device they target, while a ``KIND_CPU`` group places its own on the
    host, and adding that would run a GPU instance for a rule that asked for none.
    ``KIND_AUTO`` is a device group wherever a GPU is visible, which the plane reading this
    plan is. ``None`` where nothing device-bound is asked for, so the plan carries no line
    and the consumer refuses by name rather than being handed a zero.
    """
    counts = [
        group.count
        for group in config.instance_groups
        if group.kind is not InstanceKind.CPU
    ]
    if not config.instance_groups:
        return 1  # `instance_groups: []` is the default single instance, as `expand` treats it
    return sum(counts) or None


def _extent_of(config: ModelConfig) -> tuple[int, int] | None:
    """The last two dims of a ``(3, H, W)`` input -- the first input that has one, which is
    the preference ``pool.py`` applies to a single-input model."""
    for declared in config.inputs:
        dims = tuple(declared.dims or ())
        if len(dims) == 3 and all(isinstance(n, int) and n > 0 for n in dims[1:]):
            return (int(dims[1]), int(dims[2]))
    return None
