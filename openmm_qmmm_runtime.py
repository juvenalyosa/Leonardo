"""Legacy import shim for :mod:`leonardo.openmm_qmmm_runtime`."""
import sys
from leonardo import openmm_qmmm_runtime as _impl
sys.modules[__name__] = _impl
