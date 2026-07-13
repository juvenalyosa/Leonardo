"""Legacy import shim for :mod:`leonardo.openmm_dynamics_runtime`."""
import sys
from leonardo import openmm_dynamics_runtime as _impl
sys.modules[__name__] = _impl
