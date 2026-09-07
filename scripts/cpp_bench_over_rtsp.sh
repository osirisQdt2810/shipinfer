#!/usr/bin/env bash
# Run the C++ bench against LOCAL RTSP servers, from inside the bench container.
#
# WHY THIS RUNS INSIDE THE CONTAINER
#
# The frames have to cross a real socket with a real H.264 payload, and the servers have to be
# reachable from the bench. A second container cannot be: the rootless daemon was installed
# --skip-iptables, so its bridge has no NAT (`deploy/rootless/_container.sh` says so at
# length). Loopback inside one container is the only place both ends can meet, which is why
# `cpp.sh` grew `SHIPINFER_CPP_COMMAND` rather than this script growing its own `docker run`.
#
# WHY TWO SERVERS
#
# `scripts/rtsp_serve.py --data` serves one directory, and the benchmark's fleet is half person
# frames and half ship frames -- the mix decides the crop fan-out and therefore the whole
# downstream load. One server fed with person frames starves the ship branch of the graph and
# the analysis then blames the detector, whatever the truth is. This is the same split
# `benchmarks/harness/shipinfer.py::_rtsp_cameras` makes for the Python plane.
#
# WHY THE URIs COME FROM THE SERVER
#
# `rtsp_serve.py --print-uris` prints them; this script writes them to a file and the bench
# reads it (`csrc/shipinfer/ingest/camera_uris.h`). Formatting `<base>/cam<N>` in either place
# would be guessing a layout the server owns, and the failure mode is a connection refusal
# minutes into a run rather than a mistake at start-up.
set -euo pipefail

# THE COUNT AND THE RATE COME FROM THE BENCH'S OWN ARGV, not from the environment. The first
# version read `SHIPINFER_BENCH_*` and they never crossed `docker run` -- no `-e` passes them --
# so a run asked for 8 cameras at 5 fps while the servers published 50 at 20. Every camera then
# indexed into the person half of the URI list and the ship branch of the graph did NO work
# (`per_device ship_segmenter 0:0 1:0`), which is exactly the failure
# `harness/shipinfer.py::_rtsp_cameras` documents. Parsing the argv the bench is about to
# receive makes the servers and the fleet the same numbers by construction.
CAMERAS=50
FPS=20
argv=("$@")
for i in "${!argv[@]}"; do
  case "${argv[$i]}" in
    --cameras) CAMERAS="${argv[$((i + 1))]}" ;;
    --fps) FPS="${argv[$((i + 1))]}" ;;
  esac
done
# `rtsp_serve.py --fps` is an int (it keys the fixture cache); the bench takes a double.
FPS="${FPS%.*}"
PERSON="${SHIPINFER_RTSP_PERSON_DATA:-/work/benchmarks/baseline/data/person_2K}"
SHIP="${SHIPINFER_RTSP_SHIP_DATA:-/work/benchmarks/baseline/data/ship_2K}"
PORT="${SHIPINFER_RTSP_PORT:-8554}"
echo "rtsp fleet: $CAMERAS camera(s) at ${FPS} fps, from this binary's own argv" >&2
SHIP_PORT=$((PORT + 1))
URIS="${SHIPINFER_CAMERA_URIS:-/work/.artifacts/cpp/camera-uris.txt}"
BINARY="/work/csrc/build/${SHIPINFER_CPP_BINARY:-bench}"

# Half each, and the person half FIRST so the file itself carries the split.
half=$(( CAMERAS / 2 ))
ship_half=$(( CAMERAS - half ))

mkdir -p "$(dirname "$URIS")"

python /work/scripts/rtsp_serve.py --streams "$half" --port "$PORT" --fps "$FPS" \
  --data "$PERSON" &
person_pid=$!
python /work/scripts/rtsp_serve.py --streams "$ship_half" --port "$SHIP_PORT" --fps "$FPS" \
  --data "$SHIP" &
ship_pid=$!

# Killed on EVERY exit, including the bench's own failure: a leaked GLib main loop holds the
# port and the next run's servers fail to bind, which reads as a bench problem.
trap 'kill "$person_pid" "$ship_pid" 2>/dev/null || true; wait 2>/dev/null || true' EXIT

# Polling the port answers the actual question. A fixed sleep is either too short -- every
# camera fails its first connect, backs off, and the run measures the backoff -- or dead time
# in every run forever.
for port in "$PORT" "$SHIP_PORT"; do
  for _ in $(seq 1 120); do
    if python -c "
import socket, sys
s = socket.socket()
s.settimeout(0.25)
sys.exit(0 if s.connect_ex(('127.0.0.1', $port)) == 0 else 1)
" 2>/dev/null; then
      break
    fi
    sleep 0.5
  done
done

{
  echo "# written by scripts/cpp_bench_over_rtsp.sh -- do not hand-edit"
  echo "# $half person stream(s) on :$PORT, then $ship_half ship stream(s) on :$SHIP_PORT"
  python /work/scripts/rtsp_serve.py --print-uris --streams "$half" --port "$PORT" \
    --data "$PERSON"
  python /work/scripts/rtsp_serve.py --print-uris --streams "$ship_half" --port "$SHIP_PORT" \
    --data "$SHIP"
} > "$URIS"

exec "$BINARY" --camera-uris "$URIS" "$@"
