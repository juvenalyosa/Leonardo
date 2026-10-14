"""Legacy import shim for :mod:`leonardo.modeler_runtime`."""
import sys

from leonardo import modeler_runtime as _impl

sys.modules[__name__] = _impl
