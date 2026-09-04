# The wheels every container here needs before it can run anything.
#
# SOURCED INSIDE THE CONTAINER, by `test.sh` and by `run.sh`, so the two cannot drift. They
# differ in one line -- what they `exec` -- and everything above that line is this file.
#
# Two groups, and the split matters. The single list this replaced fell back to a minimal set
# when ANY package in it was unavailable, so one missing wheel silently dropped fastapi, opencv
# and scipy as well -- and the tests needing them skipped, which looks identical to passing.
# Required fails loudly; optional is installed one at a time so a gap costs exactly that
# package and SAYS SO.
#
# The control plane is REQUIRED, not optional. Without grpcio/protobuf, 119 tests --
# tests/launch, the shard service, core/test_priority -- never COLLECT here, and a superset
# that degrades to a subset on a missing wheel is not a superset. grpcio-tools brings the 7
# tests in test_proto_is_current.py, and is PINNED for a reason that bites exactly here: 1.83
# emits a different `shard_pb2_grpc.py` for the same .proto, so an unpinned protoc turns the
# regeneration guard red on an unmodified tree.

pip install -q --root-user-action=ignore --no-index --find-links=/wheels \
  pydantic pydantic-settings typer pyyaml pytest pytest-timeout pytest-asyncio \
  "grpcio>=1.71.2,<2" "protobuf>=5.29,<6" "grpcio-tools==1.71.2"
for package in fastapi httpx starlette uvicorn anyio opencv-python-headless scipy; do
  pip install -q --root-user-action=ignore --no-index --find-links=/wheels "$package" \
    >/dev/null 2>&1 || echo "NOTE: $package is not in /wheels; tests needing it will skip" >&2
done
python -c "import tensorrt" 2>/dev/null || \
  pip install -q --root-user-action=ignore --no-index --find-links=/wheels tensorrt \
    >/dev/null 2>&1 || true
