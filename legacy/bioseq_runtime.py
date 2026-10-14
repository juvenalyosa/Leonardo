"""Legacy import shim for :mod:`leonardo.bioseq_runtime`."""
import sys

from leonardo import bioseq_runtime as _impl

sys.modules[__name__] = _impl
