"""One conversion for the embedding, shared by everything that emits an event.

The spelling under test is not a style preference. ``tuple(float(v) for v in row)`` is a
per-element Python loop on the emission path: ``person_embedder`` emits 2048 floats and the
documented load is ~15 000 objects/s, so ~30 M ``float()`` calls a second, paid even with the
``null`` sink. It has been written and removed twice already, which is why the helper now
lives in ``core`` where every producer of a
:class:`~shipinfer.core.events.PerceptionEvent` can reach the *same* one.
"""

from __future__ import annotations

import numpy as np

from shipinfer.core.events import as_embedding


class TestTheConversion:
    """What the helper promises: plain floats, flat, and a copy."""

    def test_it_yields_plain_python_floats(self):
        """Not ``np.float32``. A consumer that pickles the record must not meet a subclass."""
        converted = as_embedding(np.arange(4, dtype=np.float32))

        assert converted == (0.0, 1.0, 2.0, 3.0)
        assert all(type(value) is float for value in converted)

    def test_it_agrees_with_the_generator_it_replaces(self):
        """The fast spelling is not an approximation of the slow one — it is the same value.

        ``float(np.float32(x))`` and ``np.float32(x).tolist()`` both widen to the same
        double, so replacing the loop changes no wire byte. This is the assertion that lets
        the loop be deleted wherever it turns up next.
        """
        row = np.linspace(-1.0, 1.0, 33, dtype=np.float32)

        assert as_embedding(row) == tuple(float(value) for value in row)

    def test_it_flattens_a_row_that_arrives_shaped(self):
        assert as_embedding(np.arange(4, dtype=np.float32).reshape(2, 2)) == (
            0.0,
            1.0,
            2.0,
            3.0,
        )

    def test_it_copies_so_the_batch_behind_it_can_be_freed(self):
        """Reassembly drops the crop batch as soon as the records are built; a view would
        keep tens of megabytes alive per in-flight frame (``pipeline/graph/state.py``)."""
        batch = np.zeros((2, 3), dtype=np.float32)
        converted = as_embedding(batch[0])
        batch[0, 0] = 9.0

        assert converted == (0.0, 0.0, 0.0)


class TestThereIsExactlyOneOfIt:
    """ONE helper, not copies that drift.

    Both generations emit the same event: ``pipeline/graph/state.py`` builds records from a
    frame's stage outputs, the DeepStream probe builds them from NvDs metadata, and the
    chain's ``output`` element will build them from a chain item's meta. Identity, not
    equality — two functions that happen to agree today are exactly how the generator came
    back the second time.
    """

    def test_the_graph_emission_path_uses_it(self):
        from shipinfer.pipeline.graph import state

        assert state.as_embedding is as_embedding
        assert state.RECORD_CONVERTERS["embedding"] is as_embedding

    def test_the_deepstream_probe_uses_it(self):
        from shipinfer.pipeline.deepstream import probe

        assert probe.as_embedding is as_embedding
