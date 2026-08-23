"""The placement-policy registry.

Importing this module gives you the registry object but *not* the implementations —
:mod:`shipinfer.scheduling.policies` imports those for their registration side effect.
Keeping the split means a new policy file never has to touch this one.
"""

from __future__ import annotations

from shipinfer.core.registry import Registry
from shipinfer.scheduling.policies.base import PlacementPolicy

__all__ = ["POLICIES", "build_policy"]

POLICIES: Registry[PlacementPolicy] = Registry("placement policy", PlacementPolicy)


def build_policy(name: str, **kwargs: object) -> PlacementPolicy:
    """Instantiate a policy by registered name.

    Args:
        name: e.g. ``"locality_spillover"``. Aliases resolve too.
        **kwargs: forwarded to the policy's constructor. Unknown keyword arguments raise
            from the constructor rather than being silently dropped, so a typo in a config
            file fails at start-up.
    """
    return POLICIES.create(name, **kwargs)
