"""What flows out of the ingest plane: a decoded image and the tag that identifies it.

Two files rather than one because the counter is not part of the value. A
:class:`~shipinfer.ingest.frame.frame.Frame` is immutable data that travels; a
:class:`~shipinfer.ingest.frame.tag.FrameCounter` is mutable state that belongs to exactly
one thread and must survive the reconnect that destroys the source.
"""

from shipinfer.ingest.frame.frame import Frame
from shipinfer.ingest.frame.tag import FrameCounter

__all__ = ["Frame", "FrameCounter"]
