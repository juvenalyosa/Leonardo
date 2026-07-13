"""Legacy import shim for :mod:`leonardo.scientific_readiness_runtime`."""
from leonardo import scientific_readiness_runtime as _impl
globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})
