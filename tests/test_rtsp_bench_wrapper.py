"""`scripts/cpp_bench_over_rtsp.sh` splits each half of the fleet over N RTSP servers.

The generator is a suspect for the ingest ceiling: two server processes took 4.2-4.7 cores at
4 000 offered while frames read scattered 2 030-2 760. So each half can run on
`SHIPINFER_RTSP_SERVERS` processes on consecutive ports. The default of one a half must stay
today's layout, since every recorded figure was measured on it.

EXECUTED, not spelled: the wrapper runs under bash with a stub `python` first on PATH. The stub
logs each port probe, each server it is asked to start (with its own pid, which is the `$!` the
wrapper records) and the `host_cpu.py` argv, and prints the URIs `--print-uris` would.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts" / "cpp_bench_over_rtsp.sh"

#: Stands in for `python`, keyed on the script it is asked to run.
PYTHON = r"""#!/bin/bash
case "$1" in
  -c) port=$(printf '%s' "$2" | grep -oE "'127\.0\.0\.1', [0-9]+" | grep -oE '[0-9]+$')
      echo "probe $port" >> "$STUB_LOG"; exit 0 ;;
  */rtsp_serve.py)
      shift; uris=0; n=; port=; data=
      while [ $# -gt 0 ]; do
        case "$1" in
          --print-uris) uris=1; shift ;;
          --streams) n=$2; shift 2 ;;
          --port) port=$2; shift 2 ;;
          --data) data=$2; shift 2 ;;
          *) shift ;;
        esac
      done
      if [ "$uris" = 1 ]; then
        for (( i = 0; i < n; i++ )); do echo "rtsp://127.0.0.1:$port/cam$i"; done
      else
        echo "serve $n $port $data $$" >> "$STUB_LOG"
      fi
      exit 0 ;;
  */host_cpu.py) echo "host_cpu $*" >> "$STUB_LOG"; exit 0 ;;
esac
echo "unexpected $*" >> "$STUB_LOG"; exit 99
"""


def run_wrapper(tmp_path: Path, cameras: int, servers: str | None = None):
    """Run the wrapper; returns (rc, stderr, log lines, URI lines)."""
    stub = tmp_path / "bin" / "python"
    stub.parent.mkdir()
    stub.write_text(PYTHON, encoding="utf-8")
    stub.chmod(0o755)
    log, uris = tmp_path / "stub.log", tmp_path / "uris" / "camera-uris.txt"
    log.touch()
    env = {
        "PATH": f"{stub.parent}:/usr/bin:/bin",
        "STUB_LOG": str(log),
        "SHIPINFER_CAMERA_URIS": str(uris),
        "SHIPINFER_RTSP_PERSON_DATA": "/data/person dir",
        "SHIPINFER_RTSP_SHIP_DATA": "/data/ship",
    }
    if servers is not None:
        env["SHIPINFER_RTSP_SERVERS"] = servers
    argv = ["bash", str(WRAPPER), "--cameras", str(cameras), "--fps", "64.0", "--seconds", "1"]
    done = subprocess.run(argv, capture_output=True, text=True, env=env)
    lines = [u for u in uris.read_text().splitlines() if u] if uris.exists() else []
    return done.returncode, done.stderr, log.read_text().splitlines(), lines


def served(log: list[str]) -> list[tuple[int, int, str]]:
    """(streams, port, data) per server started, in start order."""
    rows = [
        line.split(" ", 1)[1].rsplit(" ", 1)[0] for line in log if line.startswith("serve ")
    ]
    return [(int(n), int(port), data) for n, port, data in (r.split(" ", 2) for r in rows)]


class TestTheLayout:
    # EMPTY is unset, as `${VAR:-1}` reads it: an exported-but-blank knob keeps the default.
    @pytest.mark.parametrize("servers", [None, ""], ids=["unset", "empty"])
    def test_the_default_is_the_layout_every_figure_was_measured_on(
        self, tmp_path: Path, servers: str | None
    ) -> None:
        rc, _, log, uris = run_wrapper(tmp_path, cameras=50, servers=servers)
        assert rc == 0
        assert served(log) == [(25, 8554, "/data/person dir"), (25, 8555, "/data/ship")]
        assert uris[0] == "# written by scripts/cpp_bench_over_rtsp.sh -- do not hand-edit"
        assert uris[2:] == [f"rtsp://127.0.0.1:8554/cam{i}" for i in range(25)] + [
            f"rtsp://127.0.0.1:8555/cam{i}" for i in range(25)
        ]

    @pytest.mark.parametrize(
        "cameras, servers, expected",
        [
            (50, "2", [(13, 8554), (12, 8555), (13, 8556), (12, 8557)]),
            (7, "3", [(1, 8554), (1, 8555), (1, 8556), (2, 8557), (1, 8558), (1, 8559)]),
            # A slice with no cameras starts no server; the ship half still begins at PORT+N.
            (5, "4", [(1, 8554), (1, 8555), (1, 8558), (1, 8559), (1, 8560)]),
        ],
    )
    def test_each_half_splits_over_its_servers(
        self, tmp_path: Path, cameras: int, servers: str, expected: list[tuple[int, int]]
    ) -> None:
        rc, _, log, uris = run_wrapper(tmp_path, cameras=cameras, servers=servers)
        assert rc == 0
        assert [(n, port) for n, port, _ in served(log)] == expected
        assert len(uris) - 2 == cameras == sum(n for n, _ in expected)
        # The person half first, in server order, so the file itself carries the split.
        ports = [int(u.split(":")[2].split("/")[0]) for u in uris[2:]]
        assert ports == sorted(ports)

    @pytest.mark.parametrize("servers", ["0", "two", "-1"])
    def test_a_bad_count_starts_nothing(self, tmp_path: Path, servers: str) -> None:
        rc, stderr, log, _ = run_wrapper(tmp_path, cameras=50, servers=servers)
        assert rc == 2
        assert "SHIPINFER_RTSP_SERVERS must be a positive integer" in stderr
        assert served(log) == []


class TestTheServersAreRunnable:
    def test_each_server_is_up_before_the_next_starts(self, tmp_path: Path) -> None:
        # A half's first server encodes the fixture cache in place; a second started beside it
        # would race that write. So every start is followed by its own port's probe.
        _, _, log, _ = run_wrapper(tmp_path, cameras=8, servers="2")
        events = [line.split(" ")[:3] for line in log if line.startswith(("serve", "probe"))]
        for i, event in enumerate(events):
            if event[0] == "serve":
                assert events[i + 1] == ["probe", event[2]], events

    def test_every_server_is_charged_to_the_generator(self, tmp_path: Path) -> None:
        _, _, log, _ = run_wrapper(tmp_path, cameras=50, servers="3")
        pids = [line.rsplit(" ", 1)[1] for line in log if line.startswith("serve ")]
        (host_cpu,) = [line.split(" ") for line in log if line.startswith("host_cpu ")]
        charged = [host_cpu[i + 1] for i, word in enumerate(host_cpu) if word == "--pid"]
        assert len(pids) == 6 and charged == pids
        assert host_cpu[host_cpu.index("--") + 1 :][:3] == [
            "/work/csrc/build/bench",
            "--camera-uris",
            str(tmp_path / "uris" / "camera-uris.txt"),
        ]
