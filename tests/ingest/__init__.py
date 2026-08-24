"""Ingest tests.

A package rather than a loose directory so the shared doubles in ``conftest.py`` —
``ScriptedSource``, ``RecordingSleep``, ``FakeClock`` — can be imported by name. They are
classes, not fixtures: a test needs to subclass and parametrise them, which a fixture makes
awkward, and the alternative (a generically named ``helpers`` module on ``sys.path``) is
worse.
"""
