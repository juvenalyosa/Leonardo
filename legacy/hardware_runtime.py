"""Legacy import shim for :mod:`leonardo.hardware_runtime`."""
import sys

from leonardo import hardware_runtime as _impl

sys.modules[__name__] = _impl
