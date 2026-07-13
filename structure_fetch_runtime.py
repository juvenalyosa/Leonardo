"""Legacy import shim for :mod:`leonardo.structure_fetch_runtime`."""
from leonardo import structure_fetch_runtime as _impl
globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})
