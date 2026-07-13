"""Legacy import shim for :mod:`leonardo.psi4_runtime`."""
import sys
from leonardo import psi4_runtime as _impl
sys.modules[__name__] = _impl
