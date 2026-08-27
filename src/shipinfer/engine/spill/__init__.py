"""The cross-process spill tier: what a shard offers its peers, over a shared ring (ADR-015).

Under the `service` topology a shard keeps serving its own GPU's instances and *also* offers
them to its peers, so a request can spill to a neighbour's device instead of queueing behind
a local one. :mod:`~shipinfer.engine.spill.wire` is the byte format,
:mod:`~shipinfer.engine.spill.remote_instance` is the peer's instance seen as a
``Placeable`` plus the two threads that serve it, and :mod:`~shipinfer.engine.spill.mesh`
is the rings and threads one shard runs to join the tier.

Nothing is re-exported here on purpose: ``remote_instance`` imports ``wire`` through this
package, and the tier is built lazily by :meth:`InferenceServer._join_service_tier` so a
single-process ``serve`` never pays for it.
"""
