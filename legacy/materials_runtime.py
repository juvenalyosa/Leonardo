"""Legacy import shim for :mod:`leonardo.materials_runtime`."""
import sys

from leonardo import materials_runtime as _impl

sys.modules[__name__] = _impl
