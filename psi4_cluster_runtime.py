"""Legacy import shim for :mod:`leonardo.psi4_cluster_runtime`."""
from leonardo import psi4_cluster_runtime as _impl
globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})
