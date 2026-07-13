"""Legacy import shim for :mod:`leonardo.vs_runtime`."""
import sys
from leonardo import vs_runtime as _impl
sys.modules[__name__] = _impl
