"""Legacy import shim for :mod:`leonardo.abfe_runtime`."""
import sys
from leonardo import abfe_runtime as _impl
sys.modules[__name__] = _impl
