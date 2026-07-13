"""Canonical Leonardo package namespace.

The current PyMOL loader still imports legacy root-level modules. During the
migration those modules remain the implementation source while this namespace
provides a stable import path for new code.
"""

__version__ = "0.2.0a5"
