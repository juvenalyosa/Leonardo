"""Legacy import shim for :mod:`leonardo.trajectory_interaction_runtime`."""
import sys
from leonardo import trajectory_interaction_runtime as _impl
sys.modules[__name__] = _impl
