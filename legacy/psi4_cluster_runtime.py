"""Legacy import shim for :mod:`leonardo.psi4_cluster_runtime`."""
import sys

from leonardo import psi4_cluster_runtime as _impl

sys.modules[__name__] = _impl
