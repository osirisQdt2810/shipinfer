"""The input extent each model declares — the geometry a resolved plan needs.

Read from `config.yaml` rather than from a loaded engine, because a plan is produced by the
control plane on a machine that may have no driver at all (`shipinfer plan`, and the parity
golden CI emits). `pool.py` asks the *live* model the same question and refuses the same way,
which stays the stricter check.
"""

from __future__ import annotations

from .model_repository import ModelRepository

__all__ = ["model_extents"]


def model_extents(repository: ModelRepository) -> dict[str, tuple[int, int]]:
    """Each model's declared ``(height, width)``, where it declares a static one.

    The last two dims of a ``(3, H, W)`` input — the first input that has one, which is the
    same preference ``pool.py`` applies to a single-input model. A model whose input is
    dynamic has no entry, so a chain that needs it is refused by name rather than guessed at.
    """
    extents: dict[str, tuple[int, int]] = {}
    for name in repository.names():
        for declared in repository.entry(name).config.inputs:
            dims = tuple(declared.dims or ())
            if len(dims) == 3 and all(isinstance(n, int) and n > 0 for n in dims[1:]):
                extents[name] = (int(dims[1]), int(dims[2]))
                break
    return extents
