"""Legacy import shim for :mod:`leonardo.structure_fetch_runtime`."""
import sys

from leonardo import structure_fetch_runtime as _impl

sys.modules[__name__] = _impl
