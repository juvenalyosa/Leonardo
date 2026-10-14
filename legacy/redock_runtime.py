"""Legacy import shim for :mod:`leonardo.redock_runtime`."""
import sys

from leonardo import redock_runtime as _impl

sys.modules[__name__] = _impl
