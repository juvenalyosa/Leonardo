"""Test-wide settings."""
import os

# dialogs created in tests must not start the background CPU/GPU benchmark (child processes)
os.environ.setdefault("LEONARDO_NO_HARDWARE_DETECTION", "1")
