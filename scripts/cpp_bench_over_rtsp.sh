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
# WHY TWO HALVES
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
# SERVERS PER HALF. Two processes carried 2 000 img/s on ~1 core and took 4.2-4.7 cores at
# 4 000, with frames read scattering 2 030-2 760 -- so the generator, a cost no deployment pays,
# is a suspect for the ingest ceiling. N processes a half, on consecutive ports, tests that.
SERVERS="${SHIPINFER_RTSP_SERVERS:-1}"
case "$SERVERS" in
  '' | *[!0-9]* | 0) echo "SHIPINFER_RTSP_SERVERS must be a positive integer" >&2; exit 2 ;;
esac
echo "rtsp fleet: $CAMERAS camera(s) at ${FPS} fps, from this binary's own argv" >&2
URIS="${SHIPINFER_CAMERA_URIS:-/work/.artifacts/cpp/camera-uris.txt}"
BINARY="/work/csrc/build/${SHIPINFER_CPP_BINARY:-bench}"

# Half each, and the person half FIRST so the file itself carries the split.
half=$(( CAMERAS / 2 ))
ship_half=$(( CAMERAS - half ))

mkdir -p "$(dirname "$URIS")"

# Polling the port answers the actual question. A fixed sleep is either too short -- every
# camera fails its first connect, backs off, and the run measures the backoff -- or dead time
# in every run forever.
wait_port() {
  for _ in $(seq 1 120); do
    python -c "
import socket, sys
s = socket.socket()
s.settimeout(0.25)
sys.exit(0 if s.connect_ex(('127.0.0.1', $1)) == 0 else 1)
" 2>/dev/null && return 0
    sleep 0.5
  done
}

# One server per slice of a half, in URI order. Each is up before the next starts: a half's
# first server encodes the fixture's cache IN PLACE (`encode_fixture`), so a second one started
# beside it would race that write or serve a half-written stream.
pids=()
counts=()
ports=()
datas=()
# Covers the SETUP below -- a failed bind or a URI listing that exits under `set -e` would
# otherwise leak a GLib main loop that holds the port, and the next run's servers fail to bind
# in a way that reads as a bench problem. It does not cover the run itself: the last line
# `exec`s, which replaces this shell and its traps, and the container's exit reaps the servers.
trap 'kill "${pids[@]}" 2>/dev/null || true; wait 2>/dev/null || true' EXIT
start_half() {  # count data first-port
  local k n
  for (( k = 0; k < SERVERS; k++ )); do
    n=$(( $1 / SERVERS + (k < $1 % SERVERS ? 1 : 0) ))
    [ "$n" -gt 0 ] || continue
    python /work/scripts/rtsp_serve.py --streams "$n" --port "$(( $3 + k ))" --fps "$FPS" \
      --data "$2" &
    pids+=("$!")
    counts+=("$n")
    ports+=("$(( $3 + k ))")
    datas+=("$2")
    wait_port "$(( $3 + k ))"
  done
}
start_half "$half" "$PERSON" "$PORT"
start_half "$ship_half" "$SHIP" "$(( PORT + SERVERS ))"

{
  echo "# written by scripts/cpp_bench_over_rtsp.sh -- do not hand-edit"
  echo "# $half person stream(s), then $ship_half ship stream(s), over ${#pids[@]} server(s)"
  for i in "${!pids[@]}"; do
    python /work/scripts/rtsp_serve.py --print-uris --streams "${counts[$i]}" \
      --port "${ports[$i]}" --data "${datas[$i]}"
  done
} > "$URIS"

# THE SERVERS ABOVE ARE A COST NO DEPLOYMENT PAYS. They generate this run's load inside the
# bench's own container, so the events figure is depressed by their CPU --
# `NOT-GPU-BOUND-AT-FIVE-GPUS` put that at up to ~17% and could not say how much was ours.
# `host_cpu.py` separates the two, which turns that caveat into a number in every run.
#
# `run_cpp_bench.sh` wraps the replay arm the same way with no `--pid`, so `command_cpu_s`
# from the two arms is directly comparable and their difference is our own decode cost.
charged=()
for pid in "${pids[@]}"; do charged+=(--pid "$pid"); done
exec python /work/scripts/host_cpu.py --threads --threads-interval 0.5 "${charged[@]}" \
  -- "$BINARY" --camera-uris "$URIS" "$@"
