"""Legacy import shim for :mod:`leonardo.membrane_runtime`."""
import sys

from leonardo import membrane_runtime as _impl

sys.modules[__name__] = _impl
