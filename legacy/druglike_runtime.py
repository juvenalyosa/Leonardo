"""Legacy import shim for :mod:`leonardo.druglike_runtime`."""
import sys
from leonardo import druglike_runtime as _impl
sys.modules[__name__] = _impl
