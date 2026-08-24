"""Offline tests for the perception pipeline.

A package rather than a bare directory because the modules share doubles from ``conftest``
by importing them, not only through fixtures — a stub model is a class, and a fixture that
returns a class is less readable than importing it.
"""
