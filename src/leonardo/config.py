"""Shared package and resource configuration.

This module contains values that must stay consistent across the PyMOL UI,
CLI tools, tests, and release packaging.
"""

import sys
from pathlib import Path

PLUGIN_VERSION = "0.2.0a14"
PLUGIN_BUILD = "2026-10-20-r1"
SUITE_NAME = "Leonardo"
PROJECT_MASTER_FILE_NAME = f"{SUITE_NAME}.project.json"
PROJECT_STATE_FILE_NAME = "project_state.json"
SUITE_TAGLINE = (
    "Leonardo is an open-source initiative dedicated to empowering education "
    "and accelerating research."
)
SUITE_SUBTITLE = "Molecular Modeling Suite"
SUITE_CONTACT_EMAIL = "juvenal.yosa@gmail.com"
SUITE_AUTHOR_LINE = f"Written by Juvenal Yosa PhD. {SUITE_CONTACT_EMAIL}"


def resource_path(name: str) -> Path:
    """Return a bundled resource path in source, plugin ZIP, or wheel layouts."""
    resource_name = str(name)
    package_dir = Path(__file__).resolve().parent
    candidates = (
        package_dir / resource_name,
        package_dir.parent / resource_name,
        package_dir.parent / "assets" / resource_name,
        package_dir.parent.parent / resource_name,
        package_dir.parent.parent / "assets" / resource_name,
        package_dir.parent / "share" / "leonardo" / resource_name,
        Path(sys.prefix) / "share" / "leonardo" / resource_name,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[1]


def resource_string(name: str) -> str:
    """Return a bundled resource path as a string for Qt/legacy APIs."""
    return str(resource_path(name))
