"""Legacy import shim for :mod:`leonardo.openmm_qmmm_runtime`."""
from leonardo import openmm_qmmm_runtime as _impl
globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})
