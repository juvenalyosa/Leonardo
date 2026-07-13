"""Legacy import shim for :mod:`leonardo.small_molecule_runtime`."""
import sys
from leonardo import small_molecule_runtime as _impl
sys.modules[__name__] = _impl
