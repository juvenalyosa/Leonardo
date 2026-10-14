"""Legacy import shim for :mod:`leonardo.scientific_readiness_runtime`."""
import sys

from leonardo import scientific_readiness_runtime as _impl

sys.modules[__name__] = _impl
