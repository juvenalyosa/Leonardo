"""Legacy import shim for :mod:`leonardo.mc_ensemble_runtime`."""
import sys

from leonardo import mc_ensemble_runtime as _impl

sys.modules[__name__] = _impl
