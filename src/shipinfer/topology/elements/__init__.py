"""Element implementations, one module per family, imported here so they register.

Importing this package is what puts names in the registries, exactly as
``shipinfer.ingest.sources`` does for video sources. Nothing else imports the modules
directly.

**The import-safety rule every module in here must keep.** A module here may import
``core``, ``topology`` and the standard library at module scope, and **nothing else**. Its
runtime — GStreamer, TensorRT, DeepStream, the model pool — is imported inside ``_do_open``,
so that:

* ``import shipinfer.topology`` stays free of torch, cv2 and the server on any host, which
  is what ``tests/test_architecture.py`` asserts in a subprocess;
* a chain file can be *validated* on a laptop with no accelerator, because
  :meth:`~shipinfer.topology.chain.Topology.from_spec` instantiates every element to read
  its caps;
* a host that lacks the runtime still lists the implementation and fails at ``open()`` with
  a typed error naming the package to install, rather than at load with "unknown element" —
  two different problems with two different fixes (``ingest/registry.py`` argues this at
  length).

The one exception is a module whose *import* is impossible without the runtime, ``pyds``
being the standing example. That one registers with
:meth:`~shipinfer.core.registry.Registry.register_lazy` instead, and its docstring says why.
"""

from __future__ import annotations

from shipinfer.topology.elements import decode, mtmc, output, pool, recognize, track

__all__ = ["decode", "mtmc", "output", "pool", "recognize", "track"]
