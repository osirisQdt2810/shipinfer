"""The control-plane vocabulary survives the wire, in both directions.

Five dataclasses, one protobuf message each, and two helpers per pair. That is a small enough
surface that a reader might ask why it needs a test file — and the answer is that a field
added to ``shard.proto`` and forgotten in ``to_pb`` is silent: the message still builds, the
default rides across, and the shard reads a zero where the launcher sent a three. So the
assertion here is always ``from_pb(x.to_pb()) == x`` on a value with **nothing at its
default**, which is the only shape of round trip that catches a dropped field.

Two of the cases are the ones the design actually turns on. Fifty cameras in one health
report is the deployment's sizing (arch.md), not an arbitrary large number: it is what a
launcher reads on every probe, and a ``Struct`` that flattened or truncated it would look
fine at two. ``abandoned=3`` is a lifetime signal — while it is non-zero a detached thread
still references the shard's buffers — and a round trip that turned it into 0 would read as
the clean shutdown.
"""

from __future__ import annotations

import pytest

pytest.importorskip("google.protobuf", reason="the grpc extra is not installed")

from shipinfer.core.errors import ConfigurationError
from shipinfer.core.request import Priority
from shipinfer.launch.control import (
    AddCameraResult,
    CameraSpec,
    ShardHealth,
    ShardIdentity,
    ShardState,
    StopResult,
)
from shipinfer.launch.proto import shard_pb2


class TestTheIdentityRoundTrips:
    def test_every_field_survives(self) -> None:
        identity = ShardIdentity(shard_id=7, control_port=50107, pid=4242)

        assert ShardIdentity.from_pb(identity.to_pb()) == identity

    def test_the_bound_port_is_what_travels_not_the_requested_one(self) -> None:
        """A shard told port 0 picks an ephemeral one, and the parent learns it from here."""
        message = ShardIdentity(shard_id=1, control_port=0, pid=9).to_pb()
        message.control_port = 39963

        assert ShardIdentity.from_pb(message).control_port == 39963


class TestACameraRoundTrips:
    def test_every_field_survives(self) -> None:
        camera = CameraSpec(
            camera_id="cam-17",
            url="rtsp://10.0.0.4/live",
            fps=19.5,
            loop=False,
            priority=Priority.HIGH,
        )

        assert CameraSpec.from_pb(camera.to_pb()) == camera

    def test_a_camera_nobody_set_loop_on_still_loops(self) -> None:
        """proto3's default for a bool is false and this field's default is TRUE.

        Without ``optional`` on the wire the two are the same bytes, so a client that never
        set the field — every client written before it existed — would be asking each shard
        to stop its cameras at EOF, which for an RTSP fleet means one pass and silence.
        """
        message = CameraSpec(camera_id="cam-1", url="rtsp://x").to_pb()
        message.ClearField("loop")

        assert not message.HasField("loop")
        assert CameraSpec.from_pb(message).loop is True

    def test_no_loop_survives_the_wire_rather_than_reading_as_unset(self) -> None:
        """The other half: a deliberate ``false`` must not be indistinguishable from silence."""
        message = CameraSpec(camera_id="cam-1", url="clip.mp4", loop=False).to_pb()

        assert message.HasField("loop")
        assert CameraSpec.from_pb(message).loop is False

    @pytest.mark.parametrize("band", list(Priority))
    def test_every_band_survives_the_wire(self, band: Priority) -> None:
        camera = CameraSpec(camera_id="cam-1", url="rtsp://x", priority=band)

        assert CameraSpec.from_pb(camera.to_pb()).priority is band

    def test_the_critical_band_survives_being_zero(self) -> None:
        """``Priority.TRACKING_CRITICAL`` is ``0``, and 0 is proto3's "nothing was set".

        Carried as an int it is indistinguishable from silence, and the two readings are
        opposites: the highest lane on the deployment, or none at all. The wire enum spends
        its zero on ``UNSPECIFIED`` so the band itself is never 0 — asserted here, because a
        later hand-numbered enum that put ``TRACKING_CRITICAL`` back at 0 would pass every
        round-trip test and re-open the bug.
        """
        message = CameraSpec(
            camera_id="cam-hot", url="rtsp://x", priority=Priority.TRACKING_CRITICAL
        ).to_pb()

        assert message.priority != 0
        assert message.priority == shard_pb2.CAMERA_PRIORITY_TRACKING_CRITICAL
        assert CameraSpec.from_pb(message).priority is Priority.TRACKING_CRITICAL

    def test_a_camera_nobody_gave_a_band_arrives_unset_rather_than_normal(self) -> None:
        """``None`` is not a lane: "the shard decides" and "the launcher said normal" differ.

        A launcher that flattened the first into the second would overwrite the band a
        single-process deployment resolves from its own ``ingest.cameras``.
        """
        message = CameraSpec(camera_id="cam-1", url="rtsp://x").to_pb()

        assert message.priority == shard_pb2.CAMERA_PRIORITY_UNSPECIFIED
        assert CameraSpec.from_pb(message).priority is None

    def test_a_band_from_a_newer_shard_is_refused_by_name_rather_than_demoted(self) -> None:
        """proto3 enums are open, so an unknown lane arrives as an integer.

        Mapping it to ``NORMAL`` would place a camera in a lane nobody asked for and say
        nothing; the typed refusal names the value the launcher actually sent.
        """
        message = CameraSpec(camera_id="cam-1", url="rtsp://x").to_pb()
        message.priority = 99

        with pytest.raises(ConfigurationError, match="99"):
            CameraSpec.from_pb(message)

    def test_the_two_vocabularies_name_the_same_bands(self) -> None:
        """``control.py`` maps by NAME, so a band added to one side must exist on the other.

        A number-based map would put a new ``Priority`` member silently into its neighbour's
        lane. This is the assertion that keeps the name-based one honest in both directions.
        """
        wire = {
            value.name.removeprefix("CAMERA_PRIORITY_")
            for value in shard_pb2.CameraPriority.DESCRIPTOR.values
        } - {"UNSPECIFIED"}

        assert wire == {band.name for band in Priority}

    def test_fifty_cameras_survive_as_a_set(self) -> None:
        """The deployment's sizing, not a round number: 50 cameras is the fleet (arch.md)."""
        cameras = [
            CameraSpec(camera_id=f"cam-{i:02d}", url=f"rtsp://10.0.0.{i}/live", fps=20.0 + i)
            for i in range(50)
        ]

        back = [CameraSpec.from_pb(c.to_pb()) for c in cameras]

        assert back == cameras
        assert len({c.camera_id for c in back}) == 50


class TestARefusalRoundTrips:
    def test_a_refusal_carries_its_reason(self) -> None:
        """The reason is the payload. `accepted=False` alone tells a launcher nothing."""
        result = AddCameraResult(accepted=False, reason="camera 'cam-3' is already running")

        assert AddCameraResult.from_pb(result.to_pb()) == result

    def test_an_acceptance_carries_no_reason(self) -> None:
        assert AddCameraResult.from_pb(AddCameraResult(True).to_pb()) == AddCameraResult(True)


class TestTheAbandonmentCountRoundTrips:
    def test_three_abandoned_threads_arrive_as_three(self) -> None:
        """A lifetime signal: while it is non-zero the caller must not unwind buffers."""
        result = StopResult(abandoned=3)

        back = StopResult.from_pb(result.to_pb())

        assert back == result
        assert back.abandoned == 3
        assert not back.clean

    def test_a_clean_stop_is_distinguishable_from_a_failed_one(self) -> None:
        """Both report 0 abandoned; only one of them is clean."""
        clean = StopResult(abandoned=0)
        broken = StopResult(abandoned=0, detail="RuntimeError: the executor would not stop")

        assert StopResult.from_pb(clean.to_pb()).clean
        assert not StopResult.from_pb(broken.to_pb()).clean
        assert StopResult.from_pb(broken.to_pb()).detail == broken.detail


class TestHealthRoundTrips:
    def test_the_three_report_dicts_survive_nested(self) -> None:
        health = ShardHealth(
            state=ShardState.RUNNING,
            engine={
                "live": True,
                "ready": True,
                "status": "degraded",
                "models": {"ready": 3, "total": 4},
            },
            cameras={"cam-1": {"state": "streaming", "fps": 19.5, "frames_dropped": 2}},
            vram_budget={"slabs_mib": 4096, "own_ctx_mib": 243, "peers": 3},
        )

        back = ShardHealth.from_pb(health.to_pb())

        assert back.state == "running"
        assert back.engine == health.engine
        assert back.cameras == health.cameras
        assert back.vram_budget == health.vram_budget

    def test_fifty_cameras_survive_one_report(self) -> None:
        cameras = {
            f"cam-{i:02d}": {"state": "streaming", "fps": 20.0, "frames_read": i * 100}
            for i in range(50)
        }

        back = ShardHealth.from_pb(ShardHealth(state="running", cameras=cameras).to_pb())

        assert back.cameras == cameras
        assert len(back.cameras) == 50

    def test_an_unknown_state_carries_its_reason(self) -> None:
        """An empty health report must not read as a healthy one."""
        health = ShardHealth(state=ShardState.UNKNOWN, detail="RuntimeError: no executor")

        back = ShardHealth.from_pb(health.to_pb())

        assert back.state == "unknown"
        assert back.detail == health.detail
        assert back.engine == {}

    def test_the_state_travels_as_its_wire_spelling(self) -> None:
        """`str(ShardState.READY)` is `ready`, not `ShardState.READY`.

        `str.Enum.__str__` is `Enum`'s, so without the override on `ShardState` the member
        name would go onto the wire and into an operator's dashboard.
        """
        message = ShardHealth(state=ShardState.READY).to_pb()

        assert message.state == "ready"
        assert message.state in {s.value for s in ShardState}


class TestTheMessagesMatchTheProto:
    """The dataclasses and the .proto name the same fields, checked against the descriptor.

    A field added to one side and not the other is the failure this whole file exists for;
    this is the cheap half of it, and it fails with the field's name rather than with a
    value that quietly stayed at its default.
    """

    @pytest.mark.parametrize(
        ("message", "expected"),
        [
            (shard_pb2.ShardIdentity, {"shard_id", "control_port", "pid"}),
            (shard_pb2.CameraSpec, {"camera_id", "url", "fps", "loop", "priority"}),
            (shard_pb2.AddCameraReply, {"accepted", "reason"}),
            (shard_pb2.StopReply, {"abandoned", "detail"}),
            (
                shard_pb2.HealthReply,
                {"state", "engine", "cameras", "vram_budget", "detail"},
            ),
            (shard_pb2.TopologyRequest, {"chain_yaml", "shared_by", "share_rank"}),
        ],
    )
    def test_the_field_set_is_what_control_py_maps(self, message, expected) -> None:
        assert {f.name for f in message.DESCRIPTOR.fields} == expected

    def test_the_service_declares_the_eight_rpcs_arch_md_names(self) -> None:
        service = shard_pb2.DESCRIPTOR.services_by_name["Shard"]

        assert {m.name for m in service.methods} == {
            "Ready",
            "UpdateTopology",
            "AddCamera",
            "RemoveCamera",
            "Health",
            "Stats",
            "Drain",
            "Stop",
        }
