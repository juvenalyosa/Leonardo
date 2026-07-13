"""Legacy import shim for :mod:`leonardo.alphafold_runtime`."""
import sys
from leonardo import alphafold_runtime as _impl
sys.modules[__name__] = _impl
