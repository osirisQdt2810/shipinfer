"""The fan-out: one frame becomes N crops, and the cardinality of the DAG changes here.

This is the only stage where a row count changes, which is why it says so
(:attr:`CropStage.expands`) and why :meth:`PipelineGraph.validate` uses that flag to check
every other stage keeps its cardinality. In the previous generation the same transition was
a loop pushing ``PersonFrame`` objects into a shared buffer, and losing track of which frame
a crop belonged to is what let a crowded camera evict a quiet one's work
(``references/bitbucket-subfaceid/docs/flow.md``).

Cropping is also where the pipeline stops moving whole frames around: a 1080p frame is 6 MB
and a person crop is a few KB, so from here on only the small things travel — which is the
same argument ADR-004 makes for keeping a frame on the GPU it was decoded on and spilling
only its crops.

**Two crop sets of the same class is normal, not waste.** The segmenter wants 512x512 and
the embedder wants 256x128; resizing once from the full-resolution frame is both cheaper and
sharper than resizing the 512x512 crop down again, which is exactly why the demo
pipeline emits both, from one pass over the frame.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from shipinfer.core.errors import ConfigurationError
from shipinfer.pipeline.graph.objects import ObjectBatch
from shipinfer.pipeline.graph.stage import Cardinality, PipelineStage
from shipinfer.pipeline.graph.state import DETECTIONS, FRAME_INPUT
from shipinfer.runtime.ops import ImageOps, NormalizeParams

if TYPE_CHECKING:  # pragma: no cover - typing only
    from shipinfer.pipeline.graph.state import FrameState

__all__ = ["CropSpec", "CropStage"]


@dataclass(frozen=True, slots=True)
class CropSpec:
    """One crop set to produce: which class, at what size, under what name."""

    #: The batch name downstream stages consume, e.g. ``"ship_reid_crops"``.
    name: str
    #: The detection label to select — ``"ship"``, ``"person"``.
    class_name: str
    #: ``(height, width)`` of each crop, matching the consuming model's declared input.
    size: tuple[int, int]

    def __post_init__(self) -> None:
        if self.size[0] < 1 or self.size[1] < 1:
            raise ConfigurationError(
                f"crop {self.name!r}: size must be positive, got {self.size}"
            )

    @property
    def row_shape(self) -> tuple[int, int, int]:
        return (3, self.size[0], self.size[1])


class CropStage(PipelineStage):
    """Cut every detection out of the frame, once per configured crop set.

    A crop set with no members is produced with **zero rows** rather than omitted. That is
    what makes conditional execution work without a second mechanism: the name exists, so
    the graph is valid, and it is empty, so no stage requiring it is planned — the ship
    segmenter is simply never called on a frame that holds only people.

    Args:
        ops: the image operations to crop with. ``crop_batch`` is one call for all N boxes,
            never a Python loop around a kernel launch.
        crops: the sets to produce, in order.
    """

    cardinality: ClassVar[Cardinality] = Cardinality.PER_OBJECT
    #: The fan-out. One frame in, a variable number of objects out — the only stage where
    #: that is true, and the reason validation can insist every other stage is uniform.
    expands: ClassVar[bool] = True

    def __init__(
        self,
        name: str,
        *,
        ops: ImageOps,
        crops: Sequence[CropSpec],
        normalize: NormalizeParams | None = None,
    ) -> None:
        if not crops:
            raise ConfigurationError(f"stage {name!r} must declare at least one crop set")
        duplicates = {c.name for c in crops if sum(1 for o in crops if o.name == c.name) > 1}
        if duplicates:
            raise ConfigurationError(
                f"stage {name!r} declares crop set(s) twice: {sorted(duplicates)}"
            )
        super().__init__(
            name,
            # The pixels *and* the boxes; conditional on the boxes only. Declaring the image
            # is what tells the graph this stage is its last reader, and therefore when the
            # 6 MB may be freed — a stage that reads `state.image` without saying so would
            # be handed a released frame.
            consumes=(DETECTIONS, FRAME_INPUT),
            requires=(DETECTIONS,),
            produces=tuple(spec.name for spec in crops),
        )
        self._ops = ops
        self._crops = tuple(crops)
        self._normalize = normalize or NormalizeParams()

    @property
    def crops(self) -> tuple[CropSpec, ...]:
        return self._crops

    def spec(self, name: str) -> CropSpec | None:
        """The crop set produced under ``name``, for the graph's shape validation."""
        return next((spec for spec in self._crops if spec.name == name), None)

    def _do_run(self, state: FrameState) -> int:
        total = 0
        for spec in self._crops:
            boxes, indices = state.detections.boxes_of(spec.class_name)
            if not indices:
                state.attach(ObjectBatch.empty(spec.name, spec.class_name, spec.row_shape))
                continue
            tensor = self._ops.crop_batch(state.image, boxes, spec.size, self._normalize)
            state.attach(
                ObjectBatch(
                    name=spec.name,
                    class_name=spec.class_name,
                    object_indices=indices,
                    data=tensor,
                    boxes=boxes,
                )
            )
            total += len(indices)
        return total
