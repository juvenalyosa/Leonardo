"""Legacy import shim for :mod:`leonardo.alphafold_runtime`."""
from leonardo import alphafold_runtime as _impl
globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})
