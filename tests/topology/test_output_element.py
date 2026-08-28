"""The ``output`` element: what a frame's metadata becomes, and what it refuses to guess.

Green with or without ``3rdparty/shipvision`` — nothing here runs a tracker. The tracks these
tests hand the element are a four-field double, which is honest rather than lazy: the element
reads ``track_id`` and ``state`` off whatever the ``track`` element filed and never asks the
library anything, and a test that needed the real ``Track`` would be testing shipvision.

What is worth pinning here is the **attribution**, because every way of getting it wrong looks
like a healthy chain: a track id on the wrong object, a global id on the right object but from
the wrong track, or every identity field silently ``null`` because a key was missing. Each of
those has a test, and each of them names the key that carries it.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from shipinfer.core.errors import ConfigurationError, ValidationError
from shipinfer.core.events import SCHEMA_VERSION
from shipinfer.core.metrics import MetricsRegistry
from shipinfer.core.request import RequestContext
from shipinfer.topology.base import ChainItem, ElementContext, ElementKind
from shipinfer.topology.caps import Caps
from shipinfer.topology.elements.detections import Detections
from shipinfer.topology.elements.output import JsonLinesOutput, NullOutput, SinkOutput
from shipinfer.topology.registry import create_element, registry_for
from shipinfer.topology.sinks import NullResultSink

pytestmark = [pytest.mark.timeout(60)]

EPOCH_NS = 1_700_000_000_000_000_000


@dataclass
class FakeTrack:
    """What the element reads off a published track. Four fields is the whole contract.

    A double rather than ``shipvision.types.Track`` so this file runs on a checkout with no
    submodule — and, more usefully, so the test says out loud how little the element needs. If
    this ever has to grow a field, the element started depending on the tracker.
    """

    track_id: int
    state: str = "confirmed"


def detections(*labels: str) -> Detections:
    """One box per label, in the shape a ``pool`` detector files."""
    count = len(labels)
    return Detections(
        boxes=np.array([[i, i, i + 10.0, i + 10.0] for i in range(count)], dtype=np.float32),
        scores=np.array([0.9 - 0.1 * i for i in range(count)], dtype=np.float32),
        class_ids=np.array([8 if label == "ship" else 0 for label in labels], dtype=np.int32),
        labels=labels,
    )


def item(camera: str = "cam-a", frame: int = 3, **meta: Any) -> ChainItem:
    return ChainItem(
        context=RequestContext(
            camera_id=camera,
            frame_id=frame,
            captured_ns=1,
            captured_unix_ns=EPOCH_NS,
        ),
        caps=Caps.parse("meta@cpu"),
        meta=dict(meta),
    )


@pytest.fixture()
def element():
    """A ``null`` output, opened, keeping its events so a test can read them back."""
    built = create_element("output", "null", "output", {"keep_last": 8})
    built.open(ElementContext(shard_id=1))
    yield built
    built.close()


def emitted(built: SinkOutput):
    sink = built._sink
    assert isinstance(sink, NullResultSink)
    return sink.events()


class TestTheDeclaredContract:
    """Caps and registration, which the loader reads before anything is opened."""

    def test_it_is_a_sink_on_the_metadata_plane(self) -> None:
        """``produces`` empty is what makes it a sink; ``meta@cpu`` first is the preference.

        ``nv12@gpu`` is deliberately absent. A sink serialises, so it can never read device
        memory, and declaring it would let a chain grow a per-frame device-to-host download
        that nothing in the config file mentions (arch.md §8).
        """
        assert JsonLinesOutput.accepts == ("meta@cpu", "bgr@cpu")
        assert JsonLinesOutput.produces == ()
        assert JsonLinesOutput("output").is_sink

    def test_it_names_no_model_and_needs_no_pool(self) -> None:
        """Serialising is not inference: a chain that runs only this needs no engine."""
        assert JsonLinesOutput.requires_model_name is False
        assert JsonLinesOutput.needs_model is False
        assert JsonLinesOutput.needs_image_ops is False

    @pytest.mark.parametrize(
        ("name", "impl"),
        [
            ("jsonlines", "jsonlines"),
            ("jsonl", "jsonlines"),
            ("file", "jsonlines"),
            ("null", "none"),
            ("none", "none"),
            ("count", "none"),
            ("kafka", "kafka"),
            ("broker", "kafka"),
        ],
    )
    def test_every_sink_is_reachable_by_name(self, name: str, impl: str) -> None:
        """The aliases are the sinks' own, so a chain file reads the same either way.

        ``none`` is canonical and ``null`` is its alias, the other way round from the sink
        registry: YAML reads a bare ``impl: null`` as the null *literal*, so leading with that
        name would put a schema error with no diagnosis in front of the one implementation a
        chain reaches for when it has nowhere to publish yet.
        """
        assert registry_for(ElementKind.OUTPUT).canonical(name) == impl

    def test_a_chain_file_can_write_the_null_sink_without_quoting_it(self) -> None:
        from shipinfer.topology import ChainSpec, Topology

        chain = Topology.from_spec(
            ChainSpec.from_yaml(
                "name: n\nelements:\n" "  decode: {impl: replay}\n" "  output: {impl: none}\n"
            )
        )

        assert isinstance(chain.node("output").element, NullOutput)

    def test_a_bare_yaml_null_is_a_schema_error_and_not_the_null_sink(self) -> None:
        """Which is the whole reason ``none`` is canonical and ``null`` is only its alias.

        ``yaml.safe_load("impl: null")`` returns the *literal* ``None``, not the string, so a
        chain file that leads with the obvious spelling never reaches the registry at all. The
        refusal is the schema's and it names the field; what matters is that it happens at
        load rather than resolving to some default sink, and that ``impl: none`` above is the
        spelling a reader is pointed at.
        """
        import yaml

        from shipinfer.core.errors import ChainSpecError
        from shipinfer.topology import ChainSpec

        assert yaml.safe_load("impl: null") == {"impl": None}
        with pytest.raises(ChainSpecError, match=r"elements\.output\.impl"):
            ChainSpec.from_yaml(
                "name: n\nelements:\n" "  decode: {impl: replay}\n" "  output: {impl: null}\n"
            )

    def test_the_kafka_implementation_is_registered_lazily(self) -> None:
        """Its module is imported when a chain declares it, and not before.

        In a subprocess because the registries are process-wide and a lazy entry stays lazy
        only until the *first* reader resolves it — in this interpreter another test already
        has. The property is about a fresh process anyway, which is where it matters:
        ``confluent_kafka`` is the one heavy dependency ``topology`` names, and a host with no
        librdkafka must still be able to list the implementation and validate a chain that
        publishes to a broker. ``tests/test_architecture.py::TestImportIsCheap`` asserts the
        package-level half; this one names the module that carries it.
        """
        code = (
            "import sys, shipinfer.topology; "
            "from shipinfer.topology.base import ElementKind; "
            "from shipinfer.topology.registry import registry_for; "
            "target = 'shipinfer.topology.elements.output_kafka'; "
            "assert target not in sys.modules, 'imported before anyone asked for it'; "
            "assert 'kafka' in registry_for(ElementKind.OUTPUT).names(); "
            "cls = registry_for(ElementKind.OUTPUT).get('kafka'); "
            "assert cls.sink_name == 'kafka'; "
            "assert target in sys.modules, 'resolving the entry must import it'"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

        assert result.returncode == 0, result.stdout + result.stderr


class TestTheRowsAreTheDetections:
    """One record per detection, in the detector's own order — v1's shape, kept."""

    def test_every_detection_becomes_one_object(self, element) -> None:
        element.process(item(detections=detections("ship", "person", "ship")))

        (event,) = emitted(element)
        assert [o.class_name for o in event.objects] == ["ship", "person", "ship"]
        assert [o.det_id for o in event.objects] == ["cam-a_3_0", "cam-a_3_1", "cam-a_3_2"]

    def test_the_box_is_the_detectors_source_pixels(self, element) -> None:
        element.process(item(detections=detections("ship"), frame_hw=(480, 640)))

        (event,) = emitted(element)
        assert event.objects[0].bbox == (0.0, 0.0, 10.0, 10.0)
        assert (event.img_width, event.img_height) == (640, 480)

    def test_a_frame_the_detector_never_answered_for_still_emits(self, element) -> None:
        """An event with no objects and `detect` in `missing_stages`, never a dropped frame.

        A consumer that stopped seeing a camera would otherwise have to guess whether it went
        quiet or went down, which is the exact ambiguity `missing_stages` exists to remove.
        """
        element.process(item(missing_stages=("detect",)))

        (event,) = emitted(element)
        assert event.objects == ()
        assert event.missing_stages == ("detect",)
        assert event.is_partial

    def test_the_tag_rides_through_untouched(self, element) -> None:
        element.process(item(camera="cam-07", frame=184102, detections=detections("ship")))

        (event,) = emitted(element)
        assert event.key == ("cam-07", 184102)
        assert event.captured_unix_ns == EPOCH_NS


class TestPerObjectValuesLandOnTheirOwnRow:
    """The scatter-back, in both shapes an element may legitimately file."""

    def test_vectors_keyed_by_detection_index_land_on_that_row(self, element) -> None:
        element.process(
            item(
                detections=detections("ship", "person"),
                vectors={1: np.array([0.5, 0.25], dtype=np.float32)},
            )
        )

        (event,) = emitted(element)
        assert event.objects[0].embedding == ()
        assert event.objects[1].embedding == (0.5, 0.25)

    def test_a_row_per_detection_sequence_is_read_in_order(self, element) -> None:
        element.process(
            item(
                detections=detections("ship", "person"),
                vectors=np.array([[1.0], [2.0]], dtype=np.float32),
            )
        )

        (event,) = emitted(element)
        assert [o.embedding for o in event.objects] == [(1.0,), (2.0,)]

    def test_vectors_that_cannot_be_attributed_are_refused(self, element) -> None:
        """A mis-wired chain, not a bad frame: publishing would empty a whole field silently."""
        with pytest.raises(ValidationError, match="vectors"):
            element.process(
                item(
                    detections=detections("ship"),
                    vectors=np.zeros((4, 2), dtype=np.float32),
                )
            )

    def test_an_identity_lands_with_its_similarity(self, element) -> None:
        element.process(item(detections=detections("ship", "ship"), identities={1: (42, 0.87)}))

        (event,) = emitted(element)
        assert (event.objects[0].ship_id, event.objects[0].similarity) == (None, None)
        assert (event.objects[1].ship_id, event.objects[1].similarity) == (42, 0.87)

    def test_an_unknown_identity_is_none_and_never_zero(self, element) -> None:
        """0 is a legitimate gallery id, so it cannot double as "the gallery had no answer"."""
        element.process(item(detections=detections("ship"), identities=[(None, None)]))

        (event,) = emitted(element)
        assert event.objects[0].ship_id is None


class TestTheEmbeddingConversionIsTheSharedOne:
    """One helper for the emission path, in ``core``, called by both generations.

    ``tuple(float(v) for v in vector)`` is a per-element Python loop: 2048 floats per crop at
    the documented ~15 000 crops/s is ~30 M ``float()`` calls a second, paid even with the
    ``null`` sink. It has been written and removed twice — once on ``pipeline/``'s emission
    path, once here — so the fix is not a faster spelling in this module, it is that there is
    only one function and everything that emits an event calls it.
    """

    def test_this_element_converts_through_the_core_helper(self) -> None:
        """Identity, not equality: two functions that agree today are how it came back."""
        from shipinfer.core.events import as_embedding
        from shipinfer.pipeline.graph import state
        from shipinfer.topology.elements import output as output_module

        assert output_module.as_embedding is as_embedding
        assert state.as_embedding is as_embedding

    def test_the_published_vector_is_what_the_helper_makes(self, element) -> None:
        from shipinfer.core.events import as_embedding

        row = np.linspace(-1.0, 1.0, 17, dtype=np.float32)
        element.process(item(detections=detections("ship"), vectors=[row]))

        (event,) = emitted(element)
        assert event.objects[0].embedding == as_embedding(row)
        assert all(type(value) is float for value in event.objects[0].embedding)

    def test_a_row_that_was_never_filed_is_empty_not_null(self, element) -> None:
        """The element's own rule, and the only thing it still decides about a vector."""
        element.process(item(detections=detections("ship")))

        (event,) = emitted(element)
        assert event.objects[0].embedding == ()


class TestTracksLandOnTheObjectTheyBelongTo:
    """``tracks`` + ``track_rows`` + ``global_ids``, read as one answer in step."""

    def tracked(self, element, *, rows, tracks, **meta) -> Any:
        element.process(
            item(
                detections=detections("ship", "person"),
                tracks=tracks,
                track_rows=rows,
                **meta,
            )
        )
        (event,) = emitted(element)
        return event

    def test_a_track_id_lands_on_the_row_the_tracker_matched(self, element) -> None:
        event = self.tracked(element, rows=(1,), tracks=[FakeTrack(9)])

        assert event.objects[0].track_id is None
        assert (event.objects[1].track_id, event.objects[1].track_state) == (9, "confirmed")

    def test_a_global_id_lands_on_the_same_row_as_its_track(self, element) -> None:
        """The fan-out this element exists to get right: two ids, one attribution.

        `global_ids` is aligned with `tracks`, not with detections, so reading it against the
        rows directly would put a fleet identity on whichever object happened to be at the
        same position — a corruption with no symptom until two cameras disagree about which
        ship they are looking at.
        """
        event = self.tracked(element, rows=(1,), tracks=[FakeTrack(9)], global_ids=[31])

        assert event.objects[1].global_id == 31
        assert event.objects[0].global_id is None
        payload = event.as_dict()
        assert payload["body_global_id_vec"] == [31]
        assert payload["ship_global_id_vec"] == [None]

    def test_a_track_matched_to_no_detection_is_dropped_not_forced(self, element) -> None:
        """``-1`` is what the attribution says when nothing overlapped enough to be evidence."""
        event = self.tracked(element, rows=(-1,), tracks=[FakeTrack(9)], global_ids=[31])

        assert [o.track_id for o in event.objects] == [None, None]
        assert [o.global_id for o in event.objects] == [None, None]

    def test_numpy_ids_are_published_as_plain_numbers(self, element) -> None:
        """A numpy int64 serialises through ``json`` as a failure, not as a number."""
        event = self.tracked(
            element,
            rows=(0,),
            tracks=[FakeTrack(np.int64(9))],
            global_ids=[np.int64(31)],
        )

        assert json.loads(event.to_json())["ship_track_id_vec"] == [9]
        assert type(event.objects[0].global_id) is int

    def test_tracks_with_no_attribution_are_refused(self, element) -> None:
        """Publishing anyway means every ``track_id`` null on a chain that *is* tracking."""
        with pytest.raises(ValidationError, match="track_rows"):
            element.process(item(detections=detections("ship"), tracks=[FakeTrack(9)]))

    def test_global_ids_out_of_step_with_the_tracks_are_refused(self, element) -> None:
        with pytest.raises(ValidationError, match="global_ids"):
            element.process(
                item(
                    detections=detections("ship"),
                    tracks=[FakeTrack(9)],
                    track_rows=(0,),
                    global_ids=[31, 32],
                )
            )

    def test_a_frame_with_no_tracks_publishes_null_rather_than_refusing(self, element) -> None:
        """Tracking off, or a frame that lost the ordering race. Both are events, not errors."""
        element.process(item(detections=detections("ship"), missing_stages=("track",)))

        (event,) = emitted(element)
        assert event.objects[0].track_id is None
        assert event.missing_stages == ("track",)


class TestTheEventIsSchemaV4:
    """What lands on the wire, read back from the file a chain would write."""

    def test_one_line_per_frame_carries_the_version_and_the_new_keys(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "events.jsonl"
        built = create_element(
            "output", "jsonlines", "output", {"path": str(path), "flush_every": 0}
        )
        built.open(ElementContext(shard_id=0))
        try:
            built.process(
                item(
                    detections=detections("ship"),
                    frame_hw=(480, 640),
                    tracks=[FakeTrack(9)],
                    track_rows=(0,),
                    global_ids=[31],
                )
            )
        finally:
            built.close()

        (line,) = path.read_text(encoding="utf-8").splitlines()
        payload = json.loads(line)
        assert payload["schema_version"] == SCHEMA_VERSION == 4
        assert payload["ship_track_id_vec"] == [9]
        assert payload["ship_global_id_vec"] == [31]

    def test_closing_the_element_flushes_the_sink(self, tmp_path: Path) -> None:
        """A chain that stopped without this loses buffered events with no error anywhere."""
        path = tmp_path / "events.jsonl"
        built = create_element("output", "jsonlines", "output", {"path": str(path)})
        built.open(ElementContext())
        built.process(item(detections=detections("ship")))

        assert path.read_text(encoding="utf-8") == "", "buffered, as configured"
        built.close()

        assert len(path.read_text(encoding="utf-8").splitlines()) == 1


class TestTheSourceIdIsToldNotGuessed:
    """v1's ``sub_id``: which perception process produced this."""

    def test_it_defaults_to_the_shard_the_runner_named(self) -> None:
        built = create_element("output", "null", "output", {"keep_last": 1})
        built.open(ElementContext(shard_id=3))
        try:
            built.process(item(detections=detections("ship")))
            (event,) = emitted(built)
        finally:
            built.close()

        assert event.source_id == "shard-3"

    def test_a_chain_may_name_it(self) -> None:
        built = create_element(
            "output", "null", "output", {"keep_last": 1, "source_id": "perception-a"}
        )
        built.open(ElementContext(shard_id=3))
        try:
            built.process(item(detections=detections("ship")))
            (event,) = emitted(built)
        finally:
            built.close()

        assert event.source_id == "perception-a"

    def test_an_empty_one_is_refused_at_load(self) -> None:
        with pytest.raises(ConfigurationError, match="source_id"):
            create_element("output", "null", "output", {"source_id": "  "})


class TestTheSinkIsBuiltAtOpen:
    """A bad sink param stops the deploy, not the first frame (CONVENTIONS 2.6)."""

    def test_a_param_the_sink_does_not_accept_is_refused_by_name(self) -> None:
        built = create_element("output", "jsonlines", "output", {"pathh": "/tmp/x.jsonl"})

        with pytest.raises(ConfigurationError, match="jsonlines"):
            built.open(ElementContext())

    def test_a_param_the_sink_rejects_is_refused_at_open(self) -> None:
        built = create_element("output", "jsonlines", "output", {"flush_every": -1})

        with pytest.raises(ConfigurationError):
            built.open(ElementContext())

    def test_processing_before_open_is_a_typed_refusal(self) -> None:
        from shipinfer.core.errors import ServerStateError

        built = create_element("output", "null", "output")

        with pytest.raises(ServerStateError):
            built.process(item(detections=detections("ship")))


class TestAFailingSinkIsCountedNotRaised:
    """A broker outage must not stop the walk, and must not be invisible either."""

    def counters(self, registry: MetricsRegistry, name: str, **labels: str) -> float:
        for handle in registry.collect():
            if handle.name == name:
                return handle.value(**labels)
        return 0.0

    def test_a_published_event_is_counted_for_its_camera(self) -> None:
        registry = MetricsRegistry()
        built = create_element("output", "null", "output")
        built.open(ElementContext(metrics=registry))
        try:
            built.process(item(camera="cam-b", detections=detections("ship")))
        finally:
            built.close()

        assert self.counters(registry, "shipinfer_output_events_total", camera="cam-b") == 1

    def test_a_refused_event_is_counted_and_the_walk_continues(self) -> None:
        """The `bool` from `ResultSink.emit` is the only report a synchronous sink makes.

        A sink that is failing has to be one counter away from an operator; the alternative
        this codebase already paid for once is `frames_emitted` climbing at full rate through
        total publish loss.
        """
        registry = MetricsRegistry()
        built = create_element("output", "null", "output")
        built.open(ElementContext(metrics=registry))
        try:
            built._sink.close()  # every emit now refuses, exactly as a dead broker does
            assert built.process(item(camera="cam-b", detections=detections("ship"))) is None
        finally:
            built.close()

        assert (
            self.counters(registry, "shipinfer_output_events_dropped_total", camera="cam-b")
            == 1
        )

    def test_a_late_refusal_is_charged_to_the_frame_it_belongs_to(self) -> None:
        """An asynchronous transport answers for frame *n* while *n+k* is being produced.

        Folding that into this frame's `bool` would charge one camera's loss to another
        camera's frame, which is what `drain_delivery_failures` exists to prevent — so the
        element drains it and charges each refusal to the camera in the tag it came with.
        """
        registry = MetricsRegistry()
        built = create_element("output", "null", "output")
        built.open(ElementContext(metrics=registry))
        try:
            built._sink.drain_delivery_failures = lambda: (("cam-late", 11),)  # type: ignore[method-assign]
            built.process(item(camera="cam-b", detections=detections("ship")))
        finally:
            built.close()

        assert (
            self.counters(registry, "shipinfer_output_events_dropped_total", camera="cam-late")
            == 1
        )


class TestTheNullImplementationIsTheDefaultShape:
    def test_it_publishes_nowhere_and_still_counts(self) -> None:
        built = NullOutput("output", {"keep_last": 2})
        built.open(ElementContext())
        try:
            built.process(item(detections=detections("ship")))
            built.process(item(frame=4, detections=detections("person")))
            assert len(emitted(built)) == 2
        finally:
            built.close()
