"""Legacy import shim for :mod:`leonardo.task_monitor_runtime`."""
import sys

from leonardo import task_monitor_runtime as _impl

sys.modules[__name__] = _impl
