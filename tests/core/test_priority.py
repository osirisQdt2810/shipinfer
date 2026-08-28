"""One band vocabulary, read by the two doors a camera can arrive through.

``Priority.parse`` exists because those doors disagreed. ``POST /streams`` had matched band
*names* since it was written, while ``ingest.cameras`` — the file an operator actually
edits — was a bare ``Priority`` annotation, so pydantic's ``IntEnum`` coercion took ``0``
and refused ``tracking_critical``. Three docstrings in ``src/`` told the operator to write
the name. The rule now lives in ``core`` and both doors call it, so the tests that matter
here are the ones that pin *agreement*: the same spellings accepted, and the same words in
the refusal (the HTTP half of that is
``tests/api/test_streams.py::TestPostStreams::test_both_doors_refuse_an_unknown_band_in_the_same_words``).
"""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from shipinfer.core.request import Priority
from shipinfer.core.settings import ServerSettings
from shipinfer.core.settings.ingest import CameraConfig

#: Every refusal shares one message, so every refusal test can look for one thing.
REFUSAL = "is not a priority band"


class TestPriorityParse:
    """The vocabulary itself: what a band may be written as, and what it may not."""

    @pytest.mark.parametrize("band", list(Priority))
    def test_a_band_is_named_by_its_own_name_in_lower_case(self, band: Priority) -> None:
        """The published spelling — `BAND_NAMES`, and what an operator writes in YAML."""
        assert Priority.parse(band.name.lower()) is band

    @pytest.mark.parametrize("band", list(Priority))
    def test_a_band_is_named_by_its_own_name_in_upper_case(self, band: Priority) -> None:
        """The Python spelling, and a generated gRPC stub's. One lane, one answer."""
        assert Priority.parse(band.name) is band

    def test_the_case_is_not_load_bearing_at_all(self) -> None:
        """Not just the two spellings that exist: the match is case-insensitive, and a
        config file written by a human is where `Tracking_Critical` comes from."""
        assert Priority.parse("Tracking_Critical") is Priority.TRACKING_CRITICAL

    @pytest.mark.parametrize("band", list(Priority))
    def test_every_number_still_resolves(self, band: Priority) -> None:
        """`priority: 0` is in deployed config files; a fix that breaks them is not one."""
        assert Priority.parse(int(band)) is band

    @pytest.mark.parametrize("band", list(Priority))
    def test_a_band_parses_to_itself(self, band: Priority) -> None:
        """Callers hold `Priority` already — `CameraConfig(priority=Priority.HIGH)` is the
        common one — and a validator that ran before them must not be the thing that breaks
        it."""
        assert Priority.parse(band) is band

    @pytest.mark.parametrize("band", list(Priority))
    def test_the_numeric_string_spellings_stay_accepted(self, band: Priority) -> None:
        """`"0"` parsed before this method existed, because pydantic coerces a numeric
        string into an `IntEnum`. Nothing about this fix is a reason to start refusing it:
        an environment variable is a string, and `SHIPINFER_...=2` is how one is written."""
        assert Priority.parse(str(int(band))) is band

    @pytest.mark.parametrize(
        "value",
        ["urgent", "tracking-critical", "", "  ", "high ", 7, -1, None, 2.0, object()],
    )
    def test_a_value_that_is_not_a_band_is_refused(self, value: object) -> None:
        """Including `tracking-critical`: a hyphen is not a member name in any case, and no
        stub, document or Python member spells it that way, so accepting it would invent a
        fifth spelling for a lane that already has three."""
        with pytest.raises(ValueError, match=REFUSAL):
            Priority.parse(value)

    @pytest.mark.parametrize("value", [True, False])
    def test_a_boolean_is_refused_rather_than_read_as_a_number(self, value: bool) -> None:
        """`priority: no` in YAML is `False`, and `False == 0 == TRACKING_CRITICAL`.

        `bool` is an `int` subclass, so the plain numeric path would hand an operator who
        meant "off" the *highest* lane on the deployment — the falsy-zero trap ADR-005 keeps
        paying for, in the one direction that is worst to get wrong.
        """
        with pytest.raises(ValueError, match=REFUSAL):
            Priority.parse(value)

    def test_the_refusal_names_the_value_and_lists_every_band(self) -> None:
        """A refusal an operator can act on without opening the source.

        The bands are listed from the enum, not spelled out, so a fifth lane appears in the
        message without a second edit.
        """
        with pytest.raises(ValueError) as excinfo:
            Priority.parse("urgent")

        message = str(excinfo.value)
        assert "'urgent'" in message
        for band in Priority:
            assert band.name.lower() in message

    def test_the_message_does_not_offer_the_numbers_it_accepts(self) -> None:
        """`POST /streams` refuses the integers outright (ADR-005) while sharing this text,
        so a message advertising `0-3` would be the document-disagrees-with-itself failure
        `BandName` was introduced to end. The numbers are tolerated, not offered."""
        with pytest.raises(ValueError) as excinfo:
            Priority.parse("urgent")

        assert not any(str(int(band)) in str(excinfo.value) for band in Priority)


class TestACameraBandIsWrittenByName:
    """The door this fix opened: `ingest.cameras[].priority` takes the name."""

    @pytest.mark.parametrize("spelling", ["tracking_critical", "TRACKING_CRITICAL"])
    def test_the_configured_name_resolves_to_the_band(self, spelling: str) -> None:
        camera = CameraConfig(camera_id="gate", uri="rtsp://x", priority=spelling)

        assert camera.priority is Priority.TRACKING_CRITICAL

    @pytest.mark.parametrize("band", list(Priority))
    def test_every_band_can_be_configured_by_name(self, band: Priority) -> None:
        camera = CameraConfig(camera_id="gate", uri="rtsp://x", priority=band.name.lower())

        assert camera.priority is band

    def test_the_number_still_parses(self) -> None:
        """`priority: 0` was the only spelling that worked before, so it is the one an
        existing deployment has written down."""
        camera = CameraConfig(camera_id="gate", uri="rtsp://x", priority=0)

        assert camera.priority is Priority.TRACKING_CRITICAL

    def test_an_unconfigured_camera_is_still_normal(self) -> None:
        """A `mode="before"` validator does not run for a field that was not supplied; this
        is what says the default survived it."""
        assert CameraConfig(camera_id="gate", uri="rtsp://x").priority is Priority.NORMAL

    def test_an_unknown_band_is_refused_at_start_up_naming_the_bands(self) -> None:
        """CONVENTIONS 2.6: validate at start-up, not at first frame. A typo'd lane is a
        `ValidationError` while the operator is still looking at the terminal."""
        with pytest.raises(ValidationError, match=REFUSAL):
            CameraConfig(camera_id="gate", uri="rtsp://x", priority="urgent")

    def test_a_yaml_camera_table_carrying_the_name_loads_through_the_settings_tree(
        self,
    ) -> None:
        """The whole point, end to end: what an operator writes, parsed the way the CLI
        parses it (`cli/common.py::build_settings` hands a document to `ServerSettings`).

        Before the validator this document was a start-up failure naming `ingest.cameras.0`,
        which is how a fleet ended up running every camera at `normal` — the name was the
        documented spelling and the only one that could not be loaded.
        """
        document = yaml.safe_load("""
            ingest:
              cameras:
                - camera_id: gate
                  uri: rtsp://gate/stream
                  priority: tracking_critical
                - camera_id: carpark
                  uri: rtsp://carpark/stream
                  priority: BACKGROUND
                - camera_id: quay
                  uri: rtsp://quay/stream
            """)

        settings = ServerSettings(**document)

        assert [camera.priority for camera in settings.ingest.cameras] == [
            Priority.TRACKING_CRITICAL,
            Priority.BACKGROUND,
            Priority.NORMAL,
        ]

    def test_a_yaml_boolean_does_not_become_the_top_lane(self) -> None:
        """`priority: no` is `False` to a YAML parser, and `False` is `0`. An operator
        turning a lane off must not be given the one every other camera queues behind."""
        document = yaml.safe_load("""
            ingest:
              cameras:
                - camera_id: gate
                  uri: rtsp://gate/stream
                  priority: no
            """)

        with pytest.raises(ValidationError, match=REFUSAL):
            ServerSettings(**document)
