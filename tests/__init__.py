"""Marks the test tree as a real package, so ``tests.support`` means THIS checkout's.

Without this file ``tests/`` contributes only a namespace portion, and a regular package of
the same name anywhere on ``sys.path`` wins regardless of order. ``deploy/rootless/test.sh``
puts ``3rdparty/shipvision`` on ``PYTHONPATH`` -- shipvision is a flat layout, so there is no
narrower path -- and shipvision ships a top-level ``tests`` package. Inside the container
every ``tests.support`` import therefore resolved to shipvision's tree and the offline tier
aborted during ``pytest_configure``. Invisible on the host, whose ``PYTHONPATH`` has no
shipvision entry. ``benchmarks/tests/__init__.py`` already exists for the same reason.
"""
