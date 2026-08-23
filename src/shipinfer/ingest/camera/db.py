"""Load a camera fleet from a JSON file, including the reference system's format.

There is an existing fleet database in production: ``config/cameradb.json`` in
``references/bitbucket-subfaceid``, a list of records keyed ``cameraID`` / ``videoSource`` /
``codecType`` / ``configFile``, with the GStreamer knobs in a separate ini file. Fifty
cameras' worth of URLs and credentials is not something to retype, so this module translates
that shape into :class:`~shipinfer.core.settings.ingest.CameraConfig` rather than asking an
operator to migrate before they can try the new server.

The native shape is accepted too, so one loader covers both and a migration can happen one
camera at a time.
"""

from __future__ import annotations

import configparser
import json
from pathlib import Path
from typing import Any

from shipinfer.core.errors import ConfigurationError
from shipinfer.core.logging import get_logger
from shipinfer.core.redact import redact, redact_in
from shipinfer.core.settings.ingest import CameraConfig

__all__ = ["load_camera_db", "translate_reference_record"]

_LOG = get_logger("ingest.camera_db")

#: ``codecType`` values that mean "use the NVIDIA hardware decoder". Anything else leaves
#: hwaccel unset, so the fleet-wide default or the environment decides.
_HW_CODEC_MARKERS = ("NVV4L2", "NVDEC", "NVMM", "NVCODEC")
_SW_CODEC_MARKERS = ("AVDEC", "SW", "CPU", "SOFT")

#: ``sourceType`` -> our source name. ``RTSP_SOURCE`` deliberately maps to ``None``: which
#: RTSP backend to use is a deployment decision, not a per-camera one.
_SOURCE_TYPES: dict[str, str | None] = {
    "RTSP_SOURCE": None,
    "VIDEO": "replay",
    "FILE": "replay",
}

#: GstRTSPLowerTrans bit values, as used by ``protocols=`` in the reference ini file.
_TRANS_UDP = 0x01
_TRANS_UDP_MCAST = 0x02
_TRANS_TCP = 0x04


def _hwaccel_from_codec_type(codec_type: str) -> bool | None:
    upper = codec_type.upper()
    if any(marker in upper for marker in _HW_CODEC_MARKERS):
        return True
    if any(marker in upper for marker in _SW_CODEC_MARKERS):
        return False
    return None


def _transport_from_protocols(mask: int) -> str | None:
    """Translate a GStreamer ``protocols`` bitmask into our three-way choice.

    ``protocols=7`` — the value in the reference ini — is UDP|UDP_MCAST|TCP, i.e. "let
    rtspsrc decide", which is ``auto`` here. A mask naming exactly one transport pins it.
    """
    if mask == _TRANS_TCP:
        return "tcp"
    if mask and not mask & _TRANS_TCP and mask & (_TRANS_UDP | _TRANS_UDP_MCAST):
        return "udp"
    if mask:
        return "auto"
    return None


def _read_gst_ini(path: Path) -> dict[str, Any]:
    """Pull ``latency`` and ``protocols`` out of a reference ``gstconfig.ini``.

    Missing or unreadable is not an error: the file only ever carried defaults, and a fleet
    database that points at one that has moved should still load with ours.
    """
    parser = configparser.ConfigParser()
    try:
        if not parser.read(path):
            return {}
    except configparser.Error as exc:
        _LOG.warning("ignoring unreadable gst config %s: %s", path, redact_in(str(exc)))
        return {}

    extracted: dict[str, Any] = {}
    for section in parser.sections():
        if parser.has_option(section, "latency"):
            extracted["latency_ms"] = parser.getint(section, "latency")
        if parser.has_option(section, "protocols"):
            transport = _transport_from_protocols(parser.getint(section, "protocols"))
            if transport is not None:
                extracted["transport"] = transport
    return extracted


def translate_reference_record(record: dict[str, Any], *, base_dir: Path) -> CameraConfig:
    """Turn one ``cameradb.json`` record into a :class:`CameraConfig`.

    Args:
        record: the raw record, with the reference system's key names.
        base_dir: directory of the database file, used to resolve the relative
            ``configFile`` path the reference records carry (``../config/gstconfig.ini``).

    Raises:
        ConfigurationError: the record has no camera id or no video source. Those two are
            the record; everything else has a default.
    """
    camera_id = str(record.get("cameraID") or "").strip()
    uri = str(record.get("videoSource") or "").strip()
    if not camera_id or not uri:
        # The record holds `videoSource` verbatim, so printing it puts the fleet password
        # in a start-up log. The two identifying fields are enough to find the row.
        raise ConfigurationError(
            f"camera record needs both 'cameraID' and 'videoSource': "
            f"cameraID={record.get('cameraID')!r} videoSource={redact(uri)!r}"
        )

    fields: dict[str, Any] = {"camera_id": camera_id, "uri": uri}

    source_type = str(record.get("sourceType") or "RTSP_SOURCE").upper()
    if source_type not in _SOURCE_TYPES:
        raise ConfigurationError(
            f"camera {camera_id!r}: unsupported sourceType {source_type!r}; "
            f"expected one of {sorted(_SOURCE_TYPES)}"
        )
    source = _SOURCE_TYPES[source_type]
    if source is not None:
        fields["source"] = source

    hwaccel = _hwaccel_from_codec_type(str(record.get("codecType") or ""))
    if hwaccel is not None:
        fields["hwaccel"] = hwaccel

    width = record.get("cameraWidth")
    height = record.get("cameraHeight")
    if width and height:
        # The reference pipeline fed these straight into `videoscale`, so keeping them
        # preserves its behaviour: a camera that changes resolution still delivers the
        # declared size instead of silently changing the model's input.
        fields["width"] = int(width)
        fields["height"] = int(height)

    config_file = record.get("configFile")
    if config_file:
        fields.update(_read_gst_ini((base_dir / str(config_file)).resolve()))

    return CameraConfig(**fields)


def load_camera_db(path: str | Path) -> list[CameraConfig]:
    """Read a camera database, in either the reference format or ours.

    Accepted layouts, all of which appear in practice:

    * ``{"contents": [ ... ]}`` — the reference ``cameradb.json``;
    * a bare JSON list of records;
    * ``{"cameras": [ ... ]}`` — the shape that mirrors ``ingest.cameras`` in settings.

    Records are recognised individually: one holding ``cameraID`` is translated from the
    reference shape, anything else is validated as a native :class:`CameraConfig`. That is
    what lets a fleet be migrated one camera at a time instead of all at once.

    Raises:
        ConfigurationError: the file is missing, is not JSON, has an unrecognised top-level
            shape, or declares the same camera twice. All four are start-up failures: a
            perception deployment that comes up with half its fleet is worse than one that
            refuses to start.
    """
    file_path = Path(path).expanduser()
    try:
        raw = json.loads(file_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(f"camera database {file_path} does not exist") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(
            f"camera database {file_path} is not valid JSON: {redact_in(str(exc))}"
        ) from exc

    records = raw.get("contents", raw.get("cameras")) if isinstance(raw, dict) else raw
    if not isinstance(records, list):
        raise ConfigurationError(
            f"camera database {file_path} must hold a list, or a "
            "'contents'/'cameras' key holding one"
        )

    base_dir = file_path.parent
    cameras: list[CameraConfig] = []
    seen: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ConfigurationError(f"{file_path}: entry {index} is not an object")
        try:
            camera = (
                translate_reference_record(record, base_dir=base_dir)
                if "cameraID" in record
                else CameraConfig(**record)
            )
        except ValueError as exc:
            # pydantic's ValidationError is a ValueError; re-raise as the project's type so
            # a caller catches one thing and the message names the offending entry.
            # pydantic embeds the offending input in its message, which for a camera row
            # is the URI. Redacted for the same reason the record above is not printed.
            raise ConfigurationError(
                f"{file_path}: entry {index} is invalid: {redact_in(str(exc))}"
            ) from exc
        if camera.camera_id in seen:
            raise ConfigurationError(
                f"{file_path}: camera {camera.camera_id!r} is declared more than once"
            )
        seen.add(camera.camera_id)
        cameras.append(camera)

    _LOG.info("loaded %d camera(s) from %s", len(cameras), file_path)
    return cameras
