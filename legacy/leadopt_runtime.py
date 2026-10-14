"""Legacy import shim for :mod:`leonardo.leadopt_runtime`."""
import sys

from leonardo import leadopt_runtime as _impl

sys.modules[__name__] = _impl
