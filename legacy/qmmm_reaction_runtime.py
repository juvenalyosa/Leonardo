"""Legacy import shim for :mod:`leonardo.qmmm_reaction_runtime`."""
import sys

from leonardo import qmmm_reaction_runtime as _impl

sys.modules[__name__] = _impl
