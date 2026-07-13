"""Legacy import shim for :mod:`leonardo.small_molecule_runtime`."""
from leonardo import small_molecule_runtime as _impl
globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})
