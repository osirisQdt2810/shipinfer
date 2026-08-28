"""What the runner tells an element, and the two dependencies that arrive as shapes.

:class:`~shipinfer.topology.base.ElementContext` is the inversion the whole pure layer rests
on: an element is *told* its shard, its device, its model pool, its timeouts — and, since
phase C, its metrics registry, its worker count and its image ops. It never imports any of
them, which is what keeps ``import shipinfer.topology`` free of a driver.

Two of the new three cannot even be *named* by this layer, and the tests that matter here are
the ones that keep the stand-ins honest. ``MetricsRegistry`` is fine: it lives in ``core``.
``ImageOps`` is not — it lives in ``runtime``, the accelerator seam — so ``topology`` declares
:class:`~shipinfer.topology.base.ImageOpsLike`, a protocol with the one member the elements
call. A protocol that has silently stopped matching the class it stands for is worse than no
protocol at all: it type-checks, and it fails on the first call. So it is compared against the
real thing here, by signature.

Importing ``shipinfer.runtime.ops.base`` costs numpy and nothing else — the implementations
that need torch live in sibling modules behind the registry — so this file stays in the
offline tier.
"""

from __future__ import annotations

import dataclasses
import inspect
import typing

import pytest

from shipinfer.core.metrics import MetricsRegistry
from shipinfer.runtime.ops.base import ImageOps, LetterboxResult
from shipinfer.topology.base import ElementContext, ImageOpsLike, LetterboxLike


class TestTheNewFieldsDefaultToNotTold:
    @pytest.mark.parametrize("field", ["metrics", "workers", "ops"])
    def test_an_unfilled_context_says_nothing_rather_than_inventing_something(
        self, field: str
    ) -> None:
        """``None`` means "the runner did not say", and every element must handle it.

        Every field is optional so a chain can be loaded, validated and walked with mock
        elements before a runner, an engine or a device exists — which is what the offline
        tier does on every one of these tests. The three added in phase C keep that property.
        """
        assert getattr(ElementContext(), field) is None

    def test_it_is_still_frozen_so_an_element_cannot_choose_its_own_placement(self) -> None:
        context = ElementContext(workers=4)

        with pytest.raises(dataclasses.FrozenInstanceError):
            context.workers = 1  # type: ignore[misc]

    def test_the_metrics_registry_is_the_core_one_and_not_a_stand_in(self) -> None:
        """``MetricsRegistry`` lives in ``core``, which ``topology`` may import, so this one
        is named outright. Only the dependencies that live *above* this layer get a protocol;
        inventing one here would be ceremony over an import that is already legal."""
        registry = MetricsRegistry()

        assert ElementContext(metrics=registry).metrics is registry
        assert typing.get_type_hints(ElementContext)["metrics"] == (MetricsRegistry | None)


class TestTheImageOpsProtocolStillMatchesTheRealOne:
    """The stand-in and the class it stands for, compared rather than trusted.

    ``topology`` may not import ``runtime``, so nothing makes ``ImageOps`` satisfy
    ``ImageOpsLike`` except somebody keeping them in step. A renamed argument or a new
    required parameter would leave the protocol type-checking and the call failing at
    ``_do_open`` on a shard, which is the worst place to find out.
    """

    def test_the_letterbox_signature_is_the_same_one(self) -> None:
        real = inspect.signature(ImageOps.letterbox_batch)
        stand_in = inspect.signature(ImageOpsLike.letterbox_batch)

        assert list(real.parameters) == list(stand_in.parameters)
        assert [p.kind for p in real.parameters.values()] == [
            p.kind for p in stand_in.parameters.values()
        ]
        assert real.parameters["pad_value"].default == (
            stand_in.parameters["pad_value"].default
        )

    def test_a_real_implementation_would_satisfy_the_protocol(self) -> None:
        """Structurally, which is the only way it can be satisfied: nothing subclasses this.

        ``hasattr`` and not ``isinstance``: the protocol is deliberately not
        ``runtime_checkable`` — an ``isinstance`` against it would check the *names* and none
        of the signatures, which is exactly the reassurance that would let the test above
        rot.
        """
        for member in ("letterbox_batch",):
            assert callable(getattr(ImageOps, member, None)), member

    def test_the_letterbox_result_carries_every_member_the_protocol_names(self) -> None:
        """The scales and pads must be *carried*, never recomputed — postprocess has to invert
        exactly the transform preprocess applied — so a member dropped from the stand-in is a
        member an element stops carrying."""
        fields = {field.name for field in dataclasses.fields(LetterboxResult)}

        named = {
            name for name in typing.get_type_hints(LetterboxLike) if not name.startswith("_")
        }
        assert (
            named <= fields
        ), f"the protocol names members the result has not: {named - fields}"
        assert named == {"tensor", "scales", "pads", "extents"}

    def test_the_protocol_stays_narrower_than_the_class_it_stands_for(self) -> None:
        """A protocol member nobody calls is a coupling nobody needs.

        ``crop_batch``, ``nms`` and ``letterbox_to_device`` are real and are deliberately
        absent: the first element that crops adds the member with the test that needs it. This
        pins the decision so that "add it while you are here" is a choice somebody makes on
        purpose.
        """
        stand_in = {name for name in vars(ImageOpsLike) if not name.startswith("_")}

        assert stand_in == {"letterbox_batch"}
