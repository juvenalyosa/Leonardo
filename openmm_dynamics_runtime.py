"""Legacy import shim for :mod:`leonardo.openmm_dynamics_runtime`."""
from leonardo import openmm_dynamics_runtime as _impl
globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})
