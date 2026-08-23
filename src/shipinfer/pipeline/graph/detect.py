"""Stage 1: one detector over the whole frame.

Runs on **every** frame from every camera, so at 50 cameras x 20 fps it sees 1000 requests a
second — the highest-volume model in the system and the one whose instance count matters
most. Everything downstream is conditional on what it says.

Two things happen here besides the inference, and both belong here rather than anywhere
else. Pre-processing (letterbox + BGR->RGB + normalise + NHWC->NCHW) goes through
:class:`~shipinfer.runtime.ops.ImageOps` because that is the accelerator seam: the fused
CUDA kernel, the torch path and the numpy reference all satisfy it, so this stage is
identical on a 16-GPU node and on a laptop with no driver (ADR-003, ADR-007). Decoding
consumes the scale and pad that pre-processing *reported*, because recomputing them from the
shapes is where off-by-one box drift comes from.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar

from shipinfer.core.errors import InferenceError
from shipinfer.core.types import Tensor
from shipinfer.pipeline.graph.detections import DecodeParams, decode_detections
from shipinfer.pipeline.graph.stage import Cardinality, ModelStage, Servable
from shipinfer.pipeline.graph.state import DETECTIONS, FRAME_INPUT
from shipinfer.runtime.ops import ImageOps, NormalizeParams

if TYPE_CHECKING:  # pragma: no cover - typing only
    from shipinfer.pipeline.graph.state import FrameState

__all__ = ["DEFAULT_PAD_VALUE", "DetectStage"]

#: The YOLO letterbox grey. Kept as the default because a detector config that omits it was
#: trained with it, and a different bar colour is a silent accuracy loss.
DEFAULT_PAD_VALUE = 114


class DetectStage(ModelStage):
    """Letterbox one frame, run the detector, decode the boxes into frame pixels.

    Args:
        ops: the image operations to pre-process with. Injected rather than resolved here so
            a worker thread bound to one GPU can hold ops bound to the same GPU — an
            ``ImageOps`` instance is per-thread by contract, not shared.
        dst_size: the model's input extent, ``(height, width)``.
        decode: score threshold, per-frame cap and the class-id table.
        boxes_output / count_output: the detector's output tensor names. ``count_output`` is
            optional because not every export reports how many rows of its fixed-size output
            it filled — when it does, trusting it matters: the trailing rows of a padded
            output are undefined, not zero.
    """

    cardinality: ClassVar[Cardinality] = Cardinality.PER_FRAME

    def __init__(
        self,
        name: str,
        model: str,
        *,
        resolve: Callable[[str], Servable],
        ops: ImageOps,
        input_name: str = "images",
        dst_size: tuple[int, int] = (640, 640),
        decode: DecodeParams | None = None,
        normalize: NormalizeParams | None = None,
        boxes_output: str = "boxes",
        count_output: str | None = "num_detections",
        pad_value: int = DEFAULT_PAD_VALUE,
        timeout_s: float = 5.0,
    ) -> None:
        super().__init__(
            name,
            model,
            resolve=resolve,
            input_name=input_name,
            timeout_s=timeout_s,
            consumes=(FRAME_INPUT,),
            requires=(FRAME_INPUT,),
            produces=(DETECTIONS,),
        )
        self._ops = ops
        self._dst_size = (int(dst_size[0]), int(dst_size[1]))
        self._decode = decode or DecodeParams()
        self._normalize = normalize or NormalizeParams()
        self._boxes_output = boxes_output
        self._count_output = count_output
        self._pad_value = int(pad_value)

    @property
    def expected_row_shape(self) -> tuple[int, ...]:
        """``(3, H, W)`` — checked against the detector's declared input at start-up."""
        return (3, self._dst_size[0], self._dst_size[1])

    @property
    def dst_size(self) -> tuple[int, int]:
        return self._dst_size

    def _do_run(self, state: FrameState) -> int:
        letterboxed = self._ops.letterbox_batch(
            [state.image], self._dst_size, self._normalize, pad_value=self._pad_value
        )
        # Stored on the state, not recomputed downstream: the decode must undo exactly the
        # transform that was applied, and these are the numbers that were applied.
        state.scale = float(letterboxed.scales[0])
        state.pad = (float(letterboxed.pads[0][0]), float(letterboxed.pads[0][1]))

        response = self._infer(state, {self.input_name: Tensor.from_numpy(letterboxed.tensor)})
        rows = response.outputs.get(self._boxes_output)
        if rows is None:
            raise InferenceError(
                f"stage {self.name!r}: detector {self.model_name!r} returned no output "
                f"{self._boxes_output!r} (got: {sorted(response.outputs)})"
            )
        count: int | None = None
        if self._count_output is not None:
            reported = response.outputs.get(self._count_output)
            if reported is not None:
                count = int(reported.numpy().reshape(-1)[0])

        state.set_detections(
            decode_detections(
                rows.numpy()[0],
                params=self._decode,
                scale=state.scale,
                pad=state.pad,
                frame_hw=(state.height, state.width),
                count=count,
            )
        )
        return len(state.detections)
