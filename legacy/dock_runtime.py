"""Legacy import shim for :mod:`leonardo.dock_runtime`."""
import sys

from leonardo import dock_runtime as _impl

sys.modules[__name__] = _impl
